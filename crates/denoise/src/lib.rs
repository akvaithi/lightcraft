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

/// The level and noise [`prepare`] takes the noise channel from: green's 3×3 mean (green is the
/// densest and least noisy channel of a Bayer sensor).
fn local_level(img: &Rgb32f, x: usize, y: usize) -> f32 {
    let (w, h) = (img.width, img.height);
    let mut s = 0.0;
    let mut n = 0.0;
    for yy in y.saturating_sub(1)..(y + 2).min(h) {
        for xx in x.saturating_sub(1)..(x + 2).min(w) {
            s += img.data.get(yy * w + xx).map_or(0.0, |p| p[1]);
            n += 1.0;
        }
    }
    if n > 0.0 { s / n } else { 0.0 }
}

const GAMMA: f32 = 2.2;

/// The network input for the `w × h` region of `img` at (`x0`, `y0`) under [`model::Preprocess::Gamma22Sigma`]:
/// `x^(1/2.2)` per channel, then the noise standard deviation in that domain. The region may
/// extend past the image: the edge pixels repeat there.
pub fn prepare(img: &Rgb32f, noise: &NoiseModel, x0: isize, y0: isize, w: usize, h: usize) -> Tensor {
    let mut t = Tensor::zeros(4, h, w);
    let n = w * h;
    let a = (noise.a[0] + noise.a[1] + noise.a[2]) / 3.0;
    let b = (noise.b[0] + noise.b[1] + noise.b[2]) / 3.0;
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
            let m = local_level(img, sx, sy).max(1e-4);
            let sd = (a * m + b).max(0.0).sqrt();
            // d(x^(1/γ))/dx = x^(1/γ − 1) / γ
            let s = (sd * m.powf(1.0 / GAMMA - 1.0) / GAMMA).clamp(0.0, 1.0);
            if let Some(v) = t.data.get_mut(3 * n + i) {
                *v = s;
            }
        }
    }
    t
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
}

/// How [`denoise`] went.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Number of tiles.
    pub tiles: usize,
    /// The backends that ran tiles, in order of use.
    pub backends: Vec<String>,
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
/// result over the input. `backends` are tried in order; a failing backend is dropped for the
/// rest of the image. `progress(fraction)` returns false to cancel.
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
    let mut acc = vec![[0f32; 3]; w * h];
    let mut wsum = vec![0f32; w * h];
    let mut report = Report { tiles: cores.len(), noise: Some(*noise), ..Default::default() };
    let mut current = 0usize;
    let amount = amount.clamp(0.0, 1.0);
    let mut done = 0usize;
    let mut next = 0usize;
    while next < cores.len() {
        let Some(backend) = backends.get(current) else {
            return Err(Error::Backend { backend: "all".into(), reason: report.fallbacks.join("; ") });
        };
        let batch: Vec<usize> = (next..(next + backend.concurrency().clamp(1, 64)).min(cores.len())).collect();
        // the extended region of each tile: `ov` beyond its core on every side (past the image
        // edge too, so the network never meets a hard border inside the picture), padded up to
        // the model's multiple
        let regions: Vec<(isize, isize, usize, usize)> = batch
            .iter()
            .filter_map(|&i| cores.get(i))
            .map(|&(cx, cy, cw, ch)| {
                (cx as isize - ov as isize, cy as isize - ov as isize, (cw + 2 * ov).next_multiple_of(m), (ch + 2 * ov).next_multiple_of(m))
            })
            .collect();
        let run_tile = |&(x0, y0, pw, ph): &(isize, isize, usize, usize)| -> Result<Tensor> {
            let t = prepare(img, noise, x0, y0, pw, ph);
            let r = backend.run(model, &t)?;
            if (r.c, r.h, r.w) != (3, ph, pw) || r.data.iter().any(|v| !v.is_finite()) {
                return Err(Error::Backend {
                    backend: backend.name(), reason: format!("returned a {}×{}×{} tile for 3×{ph}×{pw}", r.c, r.h, r.w)
                });
            }
            Ok(Tensor { c: 7, h: ph, w: pw, data: [t.data, r.data].concat() })
        };
        let results: Vec<Result<Tensor>> = if regions.len() == 1 {
            regions.iter().map(run_tile).collect()
        } else {
            // one thread per tile in flight; where threads can't be spawned, run in place
            std::thread::scope(|s| {
                let handles: Vec<_> = regions
                    .iter()
                    .map(|r| (r, std::thread::Builder::new().name("denoise-tile".into()).spawn_scoped(s, move || run_tile(r))))
                    .collect();
                handles
                    .into_iter()
                    .map(|(r, h)| match h {
                        Ok(h) => h.join().unwrap_or_else(|_| Err(Error::Backend { backend: backend.name(), reason: "tile thread panicked".into() })),
                        Err(_) => run_tile(r),
                    })
                    .collect()
            })
        };
        if let Some(Err(e)) = results.iter().find(|r| r.is_err()) {
            report.fallbacks.push(format!("{}: {e}", backend.name()));
            current += 1;
            continue; // the same batch again on the next backend
        }
        for (&(x0, y0, pw, ph), r) in regions.iter().zip(results) {
            let Ok(t) = r else { continue };
            let n = pw * ph;
            // the tile's core spans [ov, ov + core) of the region; weights ramp up over the
            // overlap and back down after the core, so neighbouring tiles cross-fade
            let core_w = pw.saturating_sub(2 * ov).max(1);
            let core_h = ph.saturating_sub(2 * ov).max(1);
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
                    let (ix, iy) = (ix as usize, iy as usize);
                    // the outer half of each overlap is never used (the network's zero padding
                    // shows there); the inner half cross-fades with the neighbour
                    let ramp = |d: usize, core: usize| {
                        let margin = (ov / 2) as f32;
                        let len = (ov as f32 - margin).max(1.0);
                        let up = (d as f32 + 0.5 - margin) / len;
                        let down = (ov as f32 * 2.0 + core as f32 - d as f32 - 0.5 - margin) / len;
                        up.min(down).clamp(0.0, 1.0)
                    };
                    let wt = ramp(tx, core_w) * ramp(ty, core_h);
                    if wt <= 0.0 {
                        continue;
                    }
                    let i = ty * pw + tx;
                    let Some(a) = acc.get_mut(iy * w + ix) else { continue };
                    for c in 0..3 {
                        let yv = t.data.get(c * n + i).copied().unwrap_or(0.0);
                        let res = t.data.get((4 + c) * n + i).copied().unwrap_or(0.0);
                        a[c] += finish(yv, res) * wt;
                    }
                    if let Some(s) = wsum.get_mut(iy * w + ix) {
                        *s += wt;
                    }
                }
            }
        }
        let name = backend.name();
        report.backends.extend(std::iter::repeat_n(name, batch.len()));
        done += batch.len();
        next += batch.len();
        if !progress(done as f32 / cores.len() as f32) {
            return Err(Error::Cancelled);
        }
    }
    let mut out = img.clone();
    for ((o, a), s) in out.data.iter_mut().zip(&acc).zip(&wsum) {
        if *s > 0.0 {
            for c in 0..3 {
                let d = a[c] / s;
                o[c] += (d - o[c]) * amount;
            }
        }
    }
    report.backends.dedup();
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
