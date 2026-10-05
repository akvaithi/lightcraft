//! LightCraft AI Denoise (layer 2): a learned denoiser for demosaiced, linear camera RGB.
//!
//! - [`model`]: the `.lcdn` model file (operations + f16 weights) and CPU inference.
//! - [`NoiseModel`]: per-channel Poisson–Gaussian noise (`σ²(x) = a·x + b`), from the DNG
//!   `NoiseProfile` tag or estimated from the image ([`NoiseModel::estimate`]).
//! - [`prepare`] / [`finish`]: the network's input and output domains ([`model::Preprocess`]).
//! - [`denoise`]: the whole image in overlapping tiles with feathered seams, on a chain of
//!   [`Backend`]s (e.g. a remote GPU server, this machine's GPU, the CPU): a backend that fails
//!   hands the rest of the image to the next one, and the [`Report`] says why.
//!
//! - [`remote`]: a TCP server running tiles on this machine's backend, and its client backend.
//!
//! The same code serves every backend, so a tile denoised remotely or locally differs only by the
//! backends' float rounding.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod model;
pub mod remote;

pub use model::{Model, Op, Tensor};

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use lightcraft_raster::Rgb32f;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("denoise model: {0}")]
    Model(String),
    #[error("denoise: {0}")]
    Shape(String),
    #[error("denoise backend {backend}: {reason}")]
    Backend { backend: String, reason: String },
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Poisson–Gaussian noise of linear camera RGB (white = 1): variance `a·x + b` per channel.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NoiseModel {
    pub a: [f32; 3],
    pub b: [f32; 3],
}

impl NoiseModel {
    /// From a DNG `NoiseProfile` (pairs `S, O` per plane, or one pair for all planes).
    pub fn from_dng_profile(v: &[f64]) -> Option<NoiseModel> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        match v {
            [s, o] if ok(*s) && ok(*o) => Some(NoiseModel { a: [*s as f32; 3], b: [*o as f32; 3] }),
            [s0, o0, s1, o1, s2, o2, ..] if [s0, o0, s1, o1, s2, o2].iter().all(|x| ok(**x)) => {
                Some(NoiseModel { a: [*s0 as f32, *s1 as f32, *s2 as f32], b: [*o0 as f32, *o1 as f32, *o2 as f32] })
            }
            _ => None,
        }
    }

    /// Noise standard deviation of channel `c` at linear level `x`.
    pub fn sigma(&self, c: usize, x: f32) -> f32 {
        let (a, b) = (self.a.get(c).copied().unwrap_or(0.0), self.b.get(c).copied().unwrap_or(0.0));
        (a * x.max(0.0) + b).max(0.0).sqrt()
    }

    /// Estimate from the image itself: in flat areas the difference between a pixel and the mean
    /// of its 3×3 neighbourhood is mostly noise. Its robust variance is measured per brightness
    /// band and a line `a·x + b` is fitted through the bands, per channel.
    pub fn estimate(img: &Rgb32f) -> NoiseModel {
        const BANDS: usize = 24;
        let (w, h) = (img.width, img.height);
        let mut out = NoiseModel { a: [0.0; 3], b: [1e-8; 3] };
        if w < 8 || h < 8 {
            return out;
        }
        // sample every `step`-th pixel: plenty of statistics, bounded time on big images
        let step = ((w * h) as f64 / 2.0e6).sqrt().ceil().max(1.0) as usize;
        let band_of = |m: f32| -> Option<usize> {
            // log-spaced bands from 2^-12 to 1
            if m <= 0.0 {
                return None;
            }
            let l = (m.log2() + 12.0) / 12.0 * BANDS as f32;
            (l >= 0.0).then(|| (l as usize).min(BANDS - 1))
        };
        for c in 0..3 {
            let mut bands: Vec<Vec<(f32, f32, f32)>> = vec![Vec::new(); BANDS]; // (mean, residual², local spread)
            let at = |x: usize, y: usize| img.data.get(y * w + x).map_or(0.0, |p| p[c]);
            let mut y = 1;
            while y + 1 < h {
                let mut x = 1;
                while x + 1 < w {
                    let mut s = 0.0;
                    let mut mn = f32::MAX;
                    let mut mx = f32::MIN;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let v = at(x + dx - 1, y + dy - 1);
                            s += v;
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                    }
                    let mean = s / 9.0;
                    let r = at(x, y) - mean;
                    if let Some(bi) = band_of(mean)
                        && let Some(b) = bands.get_mut(bi)
                    {
                        b.push((mean, r * r, mx - mn));
                    }
                    x += step;
                }
                y += step;
            }
            // per band: keep the flattest half (least structure), robust variance from the median
            let mut pts: Vec<(f32, f32, f32)> = Vec::new(); // (mean, variance, weight)
            for b in &mut bands {
                if b.len() < 64 {
                    continue;
                }
                b.sort_by(|p, q| p.2.total_cmp(&q.2));
                let half = b.len() / 2;
                let flat = &mut b[..half];
                flat.sort_by(|p, q| p.1.total_cmp(&q.1));
                let med = flat[flat.len() / 2].1;
                // median of a χ²₁-scaled sample ≈ 0.455 · variance; r = x − mean(3×3) has 8/9 σ²
                let var = med / 0.455 * (9.0 / 8.0);
                let m = flat.iter().map(|p| p.0).sum::<f32>() / flat.len() as f32;
                pts.push((m, var, (flat.len() as f32).sqrt()));
            }
            if pts.is_empty() {
                continue;
            }
            // weighted least squares var = a·m + b, both ≥ 0
            let sw: f32 = pts.iter().map(|p| p.2).sum();
            let mx = pts.iter().map(|p| p.2 * p.0).sum::<f32>() / sw;
            let my = pts.iter().map(|p| p.2 * p.1).sum::<f32>() / sw;
            let sxx: f32 = pts.iter().map(|p| p.2 * (p.0 - mx) * (p.0 - mx)).sum();
            let sxy: f32 = pts.iter().map(|p| p.2 * (p.0 - mx) * (p.1 - my)).sum();
            let a = if sxx > 1e-12 { (sxy / sxx).max(0.0) } else { 0.0 };
            let b = (my - a * mx).max(1e-8);
            out.a[c] = a;
            out.b[c] = b;
        }
        out
    }
}

const GAMMA: f32 = 2.2;

/// The network input for the `w × h` region of `img` at (`x0`, `y0`) under [`model::Preprocess::Gamma22Sigma`]:
/// `x^(1/2.2)` per channel, then the noise standard deviation in that domain
/// ([`sigma_plane`]). The region may extend past the image: the edge pixels repeat there.
pub fn prepare(img: &Rgb32f, noise: &NoiseModel, x0: isize, y0: isize, w: usize, h: usize) -> Tensor {
    let mut t = Tensor::zeros(4, h, w);
    let n = w * h;
    for y in 0..h {
        for x in 0..w {
            let sx = (x0 + x as isize).clamp(0, img.width.saturating_sub(1) as isize) as usize;
            let sy = (y0 + y as isize).clamp(0, img.height.saturating_sub(1) as isize) as usize;
            let p = img.data.get(sy * img.width + sx).copied().unwrap_or([0.0; 3]);
            let i = y * w + x;
            for c in 0..3 {
                if let Some(v) = t.data.get_mut(c * n + i) {
                    *v = p[c].max(0.0).powf(1.0 / GAMMA);
                }
            }
        }
    }
    let sigma = sigma_plane(t.data.get(..3 * n).unwrap_or(&[]), w, h, noise);
    if let Some(dst) = t.data.get_mut(3 * n..4 * n) {
        dst.copy_from_slice(&sigma);
    }
    t
}

/// The noise channel of [`prepare`] from a tile's three gamma planes (`3 × h × w`): the noise
/// standard deviation, in the gamma domain, at the level of green's 3×3 mean (green is the
/// densest and least noisy channel of a Bayer sensor). A remote server rebuilds the channel with
/// this instead of receiving it.
pub fn sigma_plane(y: &[f32], w: usize, h: usize, noise: &NoiseModel) -> Vec<f32> {
    let n = w * h;
    let a = (noise.a[0] + noise.a[1] + noise.a[2]) / 3.0;
    let b = (noise.b[0] + noise.b[1] + noise.b[2]) / 3.0;
    let green = |x: usize, yy: usize| y.get(n + yy * w + x).map_or(0.0, |v| v.max(0.0).powf(GAMMA));
    let mut out = vec![0f32; n];
    for yy in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            let mut k = 0.0;
            for y2 in yy.saturating_sub(1)..(yy + 2).min(h) {
                for x2 in x.saturating_sub(1)..(x + 2).min(w) {
                    s += green(x2, y2);
                    k += 1.0;
                }
            }
            let m = (if k > 0.0 { s / k } else { 0.0 }).max(1e-4);
            let sd = (a * m + b).max(0.0).sqrt();
            // d(x^(1/γ))/dx = x^(1/γ − 1) / γ
            if let Some(o) = out.get_mut(yy * w + x) {
                *o = (sd * m.powf(1.0 / GAMMA - 1.0) / GAMMA).clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// The denoised linear value from the prepared input `y` (gamma domain) and the network's residual.
#[inline]
pub fn finish(y: f32, residual: f32) -> f32 {
    (y + residual).max(0.0).powf(GAMMA)
}

/// Where tiles run: the CPU, a GPU, a remote server…
pub trait Backend: Sync {
    /// Short description for reports ("CPU", "GPU: Apple M3 Max", "remote 100.x.y.z:7990 (RTX 4090)").
    fn name(&self) -> String;
    /// Run the model on one prepared tile.
    fn run(&self, model: &Model, x: &Tensor) -> Result<Tensor>;
    /// Tiles worth keeping in flight at once (a remote server hides network latency with more).
    fn concurrency(&self) -> usize {
        1
    }
    /// Only for what the other backends leave (the CPU: slow, and it competes with tile
    /// preparation). Other backends share the work at the same time.
    fn last_resort(&self) -> bool {
        false
    }
}

/// [`model::run_cpu`] as a backend.
pub struct Cpu;

impl Backend for Cpu {
    fn name(&self) -> String {
        "CPU".into()
    }
    fn run(&self, model: &Model, x: &Tensor) -> Result<Tensor> {
        model::run_cpu(model, x)
    }
    fn last_resort(&self) -> bool {
        true
    }
}

/// How [`denoise`] went.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Number of tiles.
    pub tiles: usize,
    /// The backends that ran tiles.
    pub backends: Vec<String>,
    /// How many tiles each of them ran.
    pub tiles_by_backend: Vec<(String, usize)>,
    /// Why earlier backends were given up ("remote …: connection refused").
    pub fallbacks: Vec<String>,
    pub noise: Option<NoiseModel>,
}

/// Tile layout: core size and overlap on each side, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub core: usize,
    pub overlap: usize,
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling { core: 256, overlap: 24 }
    }
}

/// Denoise `img` (linear camera RGB, white = 1) with `model`, blending `amount` (0..1) of the
/// result over the input. The `backends` share the tiles: every one that isn't a last resort
/// pulls tiles from a common queue at the same time (a remote GPU and this machine's GPU add up);
/// one that fails hands its tile back and drops out, and last-resort backends (the CPU) finish
/// whatever is left. `progress(fraction)` returns false to cancel.
pub fn denoise(
    img: &Rgb32f,
    noise: &NoiseModel,
    model: &Model,
    backends: &[&dyn Backend],
    amount: f32,
    tiling: Tiling,
    progress: &(dyn Fn(f32) -> bool + Sync),
) -> Result<(Rgb32f, Report)> {
    let (w, h) = (img.width, img.height);
    let m = model.header.multiple.max(1);
    if model.header.cin != 4 || model.header.cout != 3 {
        return Err(Error::Model(format!("expected a 4 → 3 channel model, got {} → {}", model.header.cin, model.header.cout)));
    }
    if w == 0 || h == 0 {
        return Err(Error::Shape("empty image".into()));
    }
    if backends.is_empty() {
        return Err(Error::Backend { backend: "none".into(), reason: "no backend to run on".into() });
    }
    let core = tiling.core.max(m).next_multiple_of(m);
    let ov = tiling.overlap;
    // tile cores
    let mut cores = Vec::new();
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            cores.push((x, y, core.min(w - x), core.min(h - y)));
            x += core;
        }
        y += core;
    }
    let n_tiles = cores.len();
    let amount = amount.clamp(0.0, 1.0);
    // the extended region of each tile: `ov` beyond its core on every side (past the image edge
    // too, so the network never meets a hard border inside the picture), padded up to the model's
    // multiple
    let region = |i: usize| -> Option<(isize, isize, usize, usize)> {
        cores.get(i).map(|&(cx, cy, cw, ch)| {
            (cx as isize - ov as isize, cy as isize - ov as isize, (cw + 2 * ov).next_multiple_of(m), (ch + 2 * ov).next_multiple_of(m))
        })
    };
    struct Shared {
        acc: Vec<[f32; 3]>,
        wsum: Vec<f32>,
        done: usize,
        per_backend: Vec<usize>,
    }
    let shared = Mutex::new(Shared { acc: vec![[0f32; 3]; w * h], wsum: vec![0f32; w * h], done: 0, per_backend: vec![0; backends.len()] });
    let queue: Mutex<VecDeque<usize>> = Mutex::new((0..n_tiles).collect());
    let dead: Vec<AtomicBool> = backends.iter().map(|_| AtomicBool::new(false)).collect();
    let fallbacks: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let cancelled = AtomicBool::new(false);

    let run_tile = |b: &dyn Backend, (x0, y0, pw, ph): (isize, isize, usize, usize)| -> Result<Tensor> {
        let t = prepare(img, noise, x0, y0, pw, ph);
        let r = b.run(model, &t)?;
        if (r.c, r.h, r.w) != (3, ph, pw) || r.data.iter().any(|v| !v.is_finite()) {
            return Err(Error::Backend { backend: b.name(), reason: format!("returned a {}×{}×{} tile for 3×{ph}×{pw}", r.c, r.h, r.w) });
        }
        Ok(Tensor { c: 7, h: ph, w: pw, data: [t.data, r.data].concat() })
    };
    // blend one finished tile into the accumulators: the outer half of each overlap is never
    // used (the network's zero padding shows there); the inner half cross-fades with the neighbour
    let blend = |s: &mut Shared, (x0, y0, pw, ph): (isize, isize, usize, usize), t: &Tensor| {
        let n = pw * ph;
        let core_w = pw.saturating_sub(2 * ov).max(1);
        let core_h = ph.saturating_sub(2 * ov).max(1);
        let ramp = |d: usize, core: usize| {
            let margin = (ov / 2) as f32;
            let len = (ov as f32 - margin).max(1.0);
            let up = (d as f32 + 0.5 - margin) / len;
            let down = (ov as f32 * 2.0 + core as f32 - d as f32 - 0.5 - margin) / len;
            up.min(down).clamp(0.0, 1.0)
        };
        for ty in 0..ph {
            let iy = y0 + ty as isize;
            if iy < 0 || iy >= h as isize {
                continue;
            }
            for tx in 0..pw {
                let ix = x0 + tx as isize;
                if ix < 0 || ix >= w as isize {
                    continue;
                }
                let wt = ramp(tx, core_w) * ramp(ty, core_h);
                if wt <= 0.0 {
                    continue;
                }
                let (k, i) = (iy as usize * w + ix as usize, ty * pw + tx);
                if let Some(a) = s.acc.get_mut(k) {
                    for c in 0..3 {
                        let yv = t.data.get(c * n + i).copied().unwrap_or(0.0);
                        let res = t.data.get((4 + c) * n + i).copied().unwrap_or(0.0);
                        a[c] += finish(yv, res) * wt;
                    }
                }
                if let Some(ws) = s.wsum.get_mut(k) {
                    *ws += wt;
                }
            }
        }
    };
    // a worker: pull tiles until the queue is empty, its backend fails or the job is cancelled
    let worker = |bi: usize, b: &dyn Backend| loop {
        if cancelled.load(Ordering::Relaxed) || dead.get(bi).is_some_and(|d| d.load(Ordering::Relaxed)) {
            break;
        }
        let Some(i) = queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop_front() else { break };
        let Some(r) = region(i) else { continue };
        match run_tile(b, r) {
            Ok(t) => {
                let mut s = shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                blend(&mut s, r, &t);
                s.done += 1;
                if let Some(c) = s.per_backend.get_mut(bi) {
                    *c += 1;
                }
                let f = s.done as f32 / n_tiles as f32;
                drop(s);
                if !progress(f) {
                    cancelled.store(true, Ordering::Relaxed);
                }
            }
            Err(e) => {
                // hand the tile back for the others, and stop using this backend
                queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push_front(i);
                if dead.get(bi).is_some_and(|d| !d.swap(true, Ordering::Relaxed)) {
                    fallbacks.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(format!("{}: {e}", b.name()));
                }
                break;
            }
        }
    };
    // every backend in `set` works on the queue at once (each with its own concurrency)
    let run_wave = |set: &[usize]| {
        std::thread::scope(|sc| {
            for &bi in set {
                let Some(b) = backends.get(bi).copied() else { continue };
                for _ in 0..b.concurrency().clamp(1, 16) {
                    if std::thread::Builder::new().name("denoise-tile".into()).spawn_scoped(sc, move || worker(bi, b)).is_err() {
                        worker(bi, b); // no threads here (e.g. the browser): run in place
                    }
                }
            }
        });
    };
    // the accelerators share the work; last-resort backends (the CPU) only pick up what they leave
    let (first, rest): (Vec<usize>, Vec<usize>) = (0..backends.len()).partition(|&i| backends.get(i).is_some_and(|b| !b.last_resort()));
    let first = if first.is_empty() { rest.clone() } else { first };
    run_wave(&first);
    if !cancelled.load(Ordering::Relaxed) && !queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_empty() {
        let left: Vec<usize> = rest.into_iter().filter(|i| !first.contains(i) && dead.get(*i).is_some_and(|d| !d.load(Ordering::Relaxed))).collect();
        run_wave(&left);
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    let fallbacks = fallbacks.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    let s = shared.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    if s.done < n_tiles {
        let reason = if fallbacks.is_empty() { "tiles left undone".to_string() } else { fallbacks.join("; ") };
        return Err(Error::Backend { backend: "all".into(), reason });
    }
    let mut out = img.clone();
    for ((o, a), ws) in out.data.iter_mut().zip(&s.acc).zip(&s.wsum) {
        if *ws > 0.0 {
            for c in 0..3 {
                let d = a[c] / ws;
                o[c] += (d - o[c]) * amount;
            }
        }
    }
    let used: Vec<(String, usize)> = backends.iter().zip(&s.per_backend).filter(|(_, n)| **n > 0).map(|(b, n)| (b.name(), *n)).collect();
    let report =
        Report { tiles: n_tiles, backends: used.iter().map(|u| u.0.clone()).collect(), tiles_by_backend: used, fallbacks, noise: Some(*noise) };
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::{Header, Preprocess, unet_header};

    /// A model that outputs zero residual: denoise must return the input (through the gamma round
    /// trip) whatever the tiling.
    fn zero_model() -> Model {
        let (h, n) = unet_header("zero", "CC0", 4, 3);
        Model::from_bytes(&Model::to_bytes(&h, &vec![0.0; n]).unwrap_or_default()).unwrap()
    }

    fn ramp(w: usize, h: usize) -> Rgb32f {
        Rgb32f::from_fn(w, h, |x, y| [x as f32 / w as f32, y as f32 / h as f32, 0.25])
    }

    #[test]
    fn zero_residual_is_identity_across_tiles() {
        let img = ramp(70, 45);
        let nm = NoiseModel { a: [1e-4; 3], b: [1e-6; 3] };
        let (out, rep) = denoise(&img, &nm, &zero_model(), &[&Cpu], 1.0, Tiling { core: 16, overlap: 6 }, &|_| true).unwrap();
        assert_eq!(rep.tiles, 5 * 3);
        assert_eq!(rep.backends, vec!["CPU".to_string()]);
        for (a, b) in out.data.iter().zip(&img.data) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-4, "{a:?} {b:?}");
            }
        }
    }

    struct Broken;
    impl Backend for Broken {
        fn name(&self) -> String {
            "remote test".into()
        }
        fn run(&self, _: &Model, _: &Tensor) -> Result<Tensor> {
            Err(Error::Backend { backend: self.name(), reason: "connection refused".into() })
        }
        fn concurrency(&self) -> usize {
            4
        }
    }

    #[test]
    fn a_failing_backend_falls_back_and_says_why() {
        let img = ramp(40, 40);
        let nm = NoiseModel { a: [1e-4; 3], b: [1e-6; 3] };
        let (_, rep) = denoise(&img, &nm, &zero_model(), &[&Broken, &Cpu], 1.0, Tiling { core: 16, overlap: 4 }, &|_| true).unwrap();
        assert_eq!(rep.backends, vec!["CPU".to_string()]);
        assert!(rep.fallbacks[0].contains("connection refused"), "{:?}", rep.fallbacks);
        let e = denoise(&img, &nm, &zero_model(), &[&Broken], 1.0, Tiling::default(), &|_| true).unwrap_err();
        assert!(e.to_string().contains("connection refused"));
        assert!(matches!(denoise(&img, &nm, &zero_model(), &[&Cpu], 1.0, Tiling { core: 16, overlap: 4 }, &|_| false), Err(Error::Cancelled)));
    }

    /// A CPU-backed accelerator for tests: slow enough that tiles interleave, failing after
    /// `fail_after` tiles when set.
    struct Accel {
        name: &'static str,
        fail_after: Option<usize>,
        ran: std::sync::atomic::AtomicUsize,
    }
    impl Backend for Accel {
        fn name(&self) -> String {
            self.name.into()
        }
        fn run(&self, m: &Model, x: &Tensor) -> Result<Tensor> {
            let n = self.ran.fetch_add(1, Ordering::SeqCst);
            if self.fail_after.is_some_and(|f| n >= f) {
                return Err(Error::Backend { backend: self.name.into(), reason: "link dropped".into() });
            }
            std::thread::sleep(std::time::Duration::from_millis(3));
            model::run_cpu(m, x)
        }
    }

    #[test]
    fn accelerators_share_tiles_and_a_failing_one_hands_its_work_back() {
        let img = ramp(96, 80);
        let nm = NoiseModel { a: [1e-4; 3], b: [1e-6; 3] };
        let t = Tiling { core: 16, overlap: 4 };
        let a = Accel { name: "gpu", fail_after: None, ran: Default::default() };
        let b = Accel { name: "remote", fail_after: None, ran: Default::default() };
        let (both, rep) = denoise(&img, &nm, &zero_model(), &[&b, &a, &Cpu], 1.0, t, &|_| true).unwrap();
        assert_eq!(rep.tiles, 30);
        let counts: std::collections::HashMap<_, _> = rep.tiles_by_backend.iter().cloned().collect();
        assert!(counts.get("gpu").is_some_and(|n| *n > 0) && counts.get("remote").is_some_and(|n| *n > 0), "{counts:?}");
        assert!(!counts.contains_key("CPU"), "the CPU only picks up leftovers");
        let (one, _) = denoise(&img, &nm, &zero_model(), &[&Cpu], 1.0, t, &|_| true).unwrap();
        // tiles blend in arrival order, so only float rounding may differ
        let close = |a: &Rgb32f, b: &Rgb32f| a.data.iter().zip(&b.data).all(|(p, q)| (0..3).all(|c| (p[c] - q[c]).abs() < 1e-5));
        assert!(close(&both, &one), "sharing doesn't change the result");
        // the remote link dies after 3 tiles: the GPU finishes, the reason is reported
        let a = Accel { name: "gpu", fail_after: None, ran: Default::default() };
        let b = Accel { name: "remote", fail_after: Some(3), ran: Default::default() };
        let (out, rep) = denoise(&img, &nm, &zero_model(), &[&b, &a, &Cpu], 1.0, t, &|_| true).unwrap();
        assert!(close(&out, &one));
        assert!(rep.fallbacks.iter().any(|f| f.contains("link dropped")), "{:?}", rep.fallbacks);
        assert_eq!(rep.tiles_by_backend.iter().map(|t| t.1).sum::<usize>(), 30);
        // every accelerator fails: the CPU takes over
        let a = Accel { name: "gpu", fail_after: Some(0), ran: Default::default() };
        let (_, rep) = denoise(&img, &nm, &zero_model(), &[&a, &Cpu], 1.0, t, &|_| true).unwrap();
        assert_eq!(rep.backends, vec!["CPU".to_string()]);
    }

    #[test]
    fn a_smoothing_model_reduces_noise_without_seams() {
        // a hand-built "denoiser": the residual is a 3×3 box blur of the input minus the input
        let mut wts = Vec::new();
        let mut ops = Vec::new();
        // conv 4→3, k3: out_c = mean of 3×3 of channel c − centre of channel c
        let w0 = 0;
        for oc in 0..3 {
            for ic in 0..4 {
                for k in 0..9 {
                    let v = if ic == oc { if k == 4 { 1.0 / 9.0 - 1.0 } else { 1.0 / 9.0 } } else { 0.0 };
                    wts.push(v);
                }
            }
        }
        let b0 = wts.len();
        wts.extend([0.0; 3]);
        ops.push(Op::Conv { cin: 4, cout: 3, k: 3, stride: 1, relu: false, w: w0, b: b0 });
        let h = Header { name: "box".into(), license: "CC0".into(), preprocess: Preprocess::Gamma22Sigma, cin: 4, cout: 3, multiple: 1, ops };
        let m = Model::from_bytes(&Model::to_bytes(&h, &wts).unwrap()).unwrap();
        // flat grey + deterministic noise
        let (w, hh) = (64, 48);
        let img = Rgb32f::from_fn(w, hh, |x, y| [0.3 + 0.02 * gauss(y * w + x); 3]);
        let nm = NoiseModel::estimate(&img);
        assert!(nm.b[1] > 0.0 || nm.a[1] > 0.0);
        let (out, _) = denoise(&img, &nm, &m, &[&Cpu], 1.0, Tiling { core: 16, overlap: 4 }, &|_| true).unwrap();
        let sd = |im: &Rgb32f| {
            let v: Vec<f32> = im.data.iter().map(|p| p[1]).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(sd(&out) < sd(&img) * 0.6, "{} vs {}", sd(&out), sd(&img));
        // no seams: small tiles give the same result as one big tile
        let (one, _) = denoise(&img, &nm, &m, &[&Cpu], 1.0, Tiling { core: 1024, overlap: 4 }, &|_| true).unwrap();
        let worst = out.data.iter().zip(&one.data).map(|(a, b)| (a[1] - b[1]).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "tiled vs whole: {worst}");
        // half the amount sits in between
        let (half, _) = denoise(&img, &nm, &m, &[&Cpu], 0.5, Tiling { core: 16, overlap: 4 }, &|_| true).unwrap();
        assert!(sd(&half) < sd(&img) && sd(&half) > sd(&out));
    }

    /// Deterministic ≈ N(0, 1) noise: a sum of 12 hashed uniforms.
    fn gauss(i: usize) -> f32 {
        let mut s = 0.0;
        for k in 0..12u64 {
            let mut v = (i as u64).wrapping_mul(6_364_136_223_846_793_005).wrapping_add(k.wrapping_mul(1_442_695_040_888_963_407));
            v ^= v >> 33;
            v = v.wrapping_mul(0xff51_afd7_ed55_8ccd);
            v ^= v >> 33;
            s += (v >> 11) as f32 / (1u64 << 53) as f32;
        }
        s - 6.0
    }

    #[test]
    fn noise_estimate_finds_poisson_gaussian_parameters() {
        // a smooth gradient plus noise with variance a·x + b (deterministic hash noise)
        let (a, b) = (4e-4f32, 1e-5f32);
        let (w, h) = (512, 384);
        let img = Rgb32f::from_fn(w, h, |x, y| {
            let base = 0.02 + 0.6 * x as f32 / w as f32;
            let i = y * w + x;
            let sd = (a * base + b).sqrt();
            [base + sd * gauss(i * 3), base + sd * gauss(i * 3 + 1), base + sd * gauss(i * 3 + 2)]
        });
        let nm = NoiseModel::estimate(&img);
        for c in 0..3 {
            assert!((nm.a[c] / a - 1.0).abs() < 0.35, "a[{c}] = {} vs {a}", nm.a[c]);
            assert!(nm.sigma(c, 0.3) / (a * 0.3 + b).sqrt() > 0.8 && nm.sigma(c, 0.3) / (a * 0.3 + b).sqrt() < 1.2);
        }
        assert_eq!(NoiseModel::from_dng_profile(&[1e-4, 2e-6]).map(|n| n.a[2]), Some(1e-4));
        assert!(NoiseModel::from_dng_profile(&[f64::NAN, 1.0]).is_none());
    }
}
