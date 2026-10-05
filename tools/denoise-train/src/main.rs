//! Train LightCraft's AI Denoise network and write it as a `.lcdn` model.
//!
//! ```text
//! cargo run --release -- [--raws DIR] [--steps N] [--batch B] [--crop C] [--lr LR] [--eval-every N] [-o OUT.lcdn]
//! cargo run --release -- --selftest | --check-gradients | --check-data | --bench-data | --check-model M.lcdn
//! ```
//!
//! `--selftest`: the training network and the app's inference agree on the exported weights.
//! `--check-gradients`: one optimizer step reaches every layer. `--check-data`: sizes and noise
//! statistics of a generated pair. `--bench-data`: data generation speed. `--check-model`: L1/MSE of
//! a model on training-style crops.
//!
//! Training pairs are made the way real noise is made: a clean image is mosaicked to an RGGB
//! sensor, Poisson-Gaussian noise (variance a·x + b, random a and b over the range of real
//! cameras and ISOs) is added per photosite, and both the clean and the noisy mosaic are
//! demosaicked with LightCraft's own raw developer. The network sees exactly what the app gives it
//! (`lightcraft_denoise::prepare`, with the noise level *estimated* from the noisy image as the
//! app does) and learns the residual to the clean demosaicked image in the same gamma domain.
//!
//! Clean images: low-ISO raws from `--raws DIR` (DNG and other raws LightCraft decodes;
//! downscaled 2× so their own noise averages out), else LightCraft's procedural demo scenes
//! (license-free, but too smooth for a production model: use them to check the pipeline).
//!
//! The network is built from `lightcraft_denoise::model::unet_header` and runs the header's
//! operations, so the trained weights drop into the same layout the app reads.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};
use lightcraft_denoise::model::{Header, Op, unet_header};
use lightcraft_denoise::{Model, NoiseModel};
use lightcraft_raster::Rgb32f;
use rayon::prelude::*;

#[cfg(not(feature = "cuda"))]
type Inner = burn::backend::Wgpu;
#[cfg(feature = "cuda")]
type Inner = burn::backend::Cuda;
type Train = burn::backend::Autodiff<Inner>;

// ------------------------------------------------------------------------------------- model

/// The header's network: one `Conv2d` per convolution op, run by interpreting the ops.
#[derive(Module, Debug)]
struct Net<B: Backend> {
    convs: Vec<Conv2d<B>>,
}

fn build<B: Backend>(h: &Header, device: &B::Device) -> Net<B> {
    let convs = h
        .ops
        .iter()
        .filter_map(|op| match *op {
            Op::Conv { cin, cout, k, stride, .. } => Some(
                Conv2dConfig::new([cin, cout], [k, k])
                    .with_stride([stride, stride])
                    .with_padding(PaddingConfig2d::Explicit(k / 2, k / 2, k / 2, k / 2))
                    .init(device),
            ),
            _ => None,
        })
        .collect();
    Net { convs }
}

/// Bilinear ×2, pixel-centre aligned with clamped edges — exactly `model::upsample2`:
/// out[2i] = ¼·in[i−1] + ¾·in[i], out[2i+1] = ¾·in[i] + ¼·in[i+1].
fn up2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let along = |x: Tensor<B, 4>, dim: usize| -> Tensor<B, 4> {
        let n = x.dims()[dim];
        let prev = Tensor::cat(vec![x.clone().narrow(dim, 0, 1), x.clone().narrow(dim, 0, n - 1)], dim);
        let next = Tensor::cat(vec![x.clone().narrow(dim, 1, n - 1), x.clone().narrow(dim, n - 1, 1)], dim);
        let even = prev.mul_scalar(0.25) + x.clone().mul_scalar(0.75);
        let odd = x.mul_scalar(0.75) + next.mul_scalar(0.25);
        let [b, c, h, w] = even.dims();
        let s: Tensor<B, 5> = Tensor::stack(vec![even, odd], dim + 1);
        if dim == 2 { s.reshape([b, c, h * 2, w]) } else { s.reshape([b, c, h, w * 2]) }
    };
    along(along(x, 2), 3)
}

impl<B: Backend> Net<B> {
    fn forward(&self, h: &Header, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cur = x;
        let mut slots: Vec<Option<Tensor<B, 4>>> = vec![None; 16];
        let mut i = 0;
        for op in &h.ops {
            cur = match *op {
                Op::Conv { relu, .. } => {
                    let y = self.convs[i].forward(cur);
                    i += 1;
                    if relu { activation::relu(y) } else { y }
                }
                Op::Save(s) => {
                    slots[s] = Some(cur.clone());
                    cur
                }
                Op::Cat(s) => Tensor::cat(vec![cur, slots[s].clone().expect("slot")], 1),
                Op::Up => up2(cur),
            };
        }
        cur
    }

    /// The weights in the header's layout (kernel `[out][in][ky][kx]`, then biases).
    fn export(&self, h: &Header, n: usize) -> Vec<f32> {
        let mut w = vec![0f32; n];
        let mut i = 0;
        for op in &h.ops {
            if let Op::Conv { w: wo, b: bo, .. } = *op {
                let c = &self.convs[i];
                let kv = c.weight.val().into_data().convert::<f32>().to_vec::<f32>().expect("weights");
                w[wo..wo + kv.len()].copy_from_slice(&kv);
                if let Some(b) = &c.bias {
                    let bv = b.val().into_data().convert::<f32>().to_vec::<f32>().expect("bias");
                    w[bo..bo + bv.len()].copy_from_slice(&bv);
                }
                i += 1;
            }
        }
        w
    }
}

// -------------------------------------------------------------------------------------- data

/// A tiny deterministic generator (xorshift*), one per sample.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    fn gauss(&mut self) -> f32 {
        // Box–Muller
        let u = self.unit().max(1e-7);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// Where clean images come from.
enum Clean {
    Scenes(Vec<lightcraft_scenes::Scene>),
    /// Decoded, demosaicked, 2×-downscaled low-ISO raws (linear camera RGB, white = 1).
    Raws(Vec<Rgb32f>),
}

fn load_raws(dir: &Path) -> Vec<Rgb32f> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    files.sort();
    files
        .par_iter()
        .filter_map(|p| {
            let bytes = std::fs::read(p).ok()?;
            lightcraft_raw::probe(&bytes)?;
            let raw = lightcraft_raw::decode(&bytes).ok()?;
            let img = raw.develop(lightcraft_raw::Method::Ahd).ok()?;
            // 2× box downscale: the raw's own noise drops by half, detail stays sharp
            let (w, h) = (img.width / 2, img.height / 2);
            let small = Rgb32f::from_fn(w, h, |x, y| {
                let mut s = [0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = img.data[(2 * y + dy) * img.width + 2 * x + dx];
                    for c in 0..3 {
                        s[c] += p[c] * 0.25;
                    }
                }
                s
            });
            eprintln!("  {} → {}×{}", p.display(), w, h);
            Some(small)
        })
        .collect()
}

/// A clean `size × size` linear image (white = 1, values clipped at the sensor's white).
fn clean_image(src: &Clean, rng: &mut Rng, size: usize) -> Rgb32f {
    let img = match src {
        Clean::Scenes(s) => {
            let sc = &s[(rng.next() % s.len() as u64) as usize];
            // render bigger than needed and crop, for detail at varied scales
            let big = (size as f32 * rng.range(1.0, 4.0)) as usize;
            let full = sc.render(big, big * 2 / 3 + 1);
            crop(&full, rng, size)
        }
        Clean::Raws(r) => crop(&r[(rng.next() % r.len() as u64) as usize], rng, size),
    };
    let gain = 2f32.powf(rng.range(-3.0, 1.0));
    Rgb32f { data: img.data.iter().map(|p| p.map(|v| (v * gain).clamp(0.0, 1.0))).collect(), ..img }
}

fn crop(img: &Rgb32f, rng: &mut Rng, size: usize) -> Rgb32f {
    let (w, h) = (img.width, img.height);
    let x0 = if w > size { (rng.next() % (w - size) as u64) as usize } else { 0 };
    let y0 = if h > size { (rng.next() % (h - size) as u64) as usize } else { 0 };
    Rgb32f::from_fn(size, size, |x, y| img.data[(y0 + y).min(h - 1) * w + (x0 + x).min(w - 1)])
}

/// Mosaic `img` to RGGB, optionally add Poisson-Gaussian noise per photosite, and demosaic with
/// LightCraft's raw developer.
fn sensor(img: &Rgb32f, noise: Option<(f32, f32, &mut Rng)>) -> Rgb32f {
    use lightcraft_raw::*;
    let (w, h) = (img.width, img.height);
    let mut data: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let c = match (y % 2, x % 2) {
                (0, 0) => 0,
                (1, 1) => 2,
                _ => 1,
            };
            img.data[i][c]
        })
        .collect();
    if let Some((a, b, rng)) = noise {
        for v in &mut data {
            *v = (*v + (a * *v + b).sqrt() * rng.gauss()).clamp(0.0, 1.0);
        }
    }
    let raw = RawImage {
        format: RawFormat::Dng,
        width: w,
        height: h,
        cpp: 1,
        data: RawData::F32(data),
        cfa: Cfa::bayer("RGGB"),
        bits: 16,
        black: BlackLevel::uniform(0.0),
        white: vec![1.0],
        active_area: Rect::new(0, 0, w, h),
        crop: Rect::new(0, 0, w, h),
        orientation: lightcraft_raw::Orientation::Normal,
        color: ColorData::default(),
        wb_multipliers: None,
        linearized: true,
        opcodes: OpcodeLists::default(),
        metadata: Default::default(),
    };
    raw.develop(Method::Ahd).unwrap_or_else(|_| img.clone())
}

/// One training pair: (network input 4×c×c, target residual-domain image 3×c×c).
fn sample(src: &Clean, seed: u64, crop_px: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    // a bigger image, so the noise estimate (like the app's, on a whole photo) is stable
    let big = (crop_px * 2).max(192);
    let clean = clean_image(src, &mut rng, big);
    // shot noise a (ISO 100 … 25600 on typical sensors) and read noise b
    let a = 10f32.powf(rng.range(-5.0, -1.6));
    let b = 10f32.powf(rng.range(-7.5, -3.5));
    let target = sensor(&clean, None);
    let noisy = sensor(&clean, Some((a, b, &mut rng)));
    let nm = NoiseModel::estimate(&noisy);
    let x0 = (rng.next() % (big - crop_px) as u64) as isize;
    let y0 = (rng.next() % (big - crop_px) as u64) as isize;
    let input = lightcraft_denoise::prepare(&noisy, &nm, x0, y0, crop_px, crop_px);
    let n = crop_px * crop_px;
    let mut t = vec![0f32; 3 * n];
    for y in 0..crop_px {
        for x in 0..crop_px {
            let p = target.data[(y0 as usize + y) * big + x0 as usize + x];
            for c in 0..3 {
                t[c * n + y * crop_px + x] = p[c].max(0.0).powf(1.0 / 2.2);
            }
        }
    }
    (input.data, t)
}

/// A batch as flat vectors (inputs b×4×c×c, targets b×3×c×c).
fn batch_data(src: &Clean, step: u64, b: usize, crop_px: usize) -> (Vec<f32>, Vec<f32>) {
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..b).into_par_iter().map(|i| sample(src, step * 1000 + i as u64, crop_px)).collect();
    let (mut xi, mut ti) = (Vec::new(), Vec::new());
    for (x, t) in pairs {
        xi.extend(x);
        ti.extend(t);
    }
    (xi, ti)
}

// ----------------------------------------------------------------------------------- eval

fn psnr(a: &Rgb32f, b: &Rgb32f) -> f64 {
    // in the gamma domain, where errors are seen
    let mut se = 0f64;
    for (p, q) in a.data.iter().zip(&b.data) {
        for c in 0..3 {
            let d = p[c].max(0.0).powf(1.0 / 2.2) - q[c].max(0.0).powf(1.0 / 2.2);
            se += (d * d) as f64;
        }
    }
    let mse = se / (a.data.len() * 3) as f64;
    10.0 * (1.0 / mse.max(1e-12)).log10()
}

/// Denoise held-out images through the app's own inference path (`lightcraft_denoise::denoise`
/// on the CPU) with the exported model; mean PSNR before and after.
fn evaluate(src: &Clean, model: &Model) -> (f64, f64) {
    let mut before = 0.0;
    let mut after = 0.0;
    let n = 6;
    for k in 0..n {
        let mut rng = Rng::new(9_000_000 + k);
        let clean = clean_image(src, &mut rng, 256);
        let a = 10f32.powf(-4.0 + 0.4 * k as f32);
        let target = sensor(&clean, None);
        let noisy = sensor(&clean, Some((a, a * 0.01, &mut rng)));
        let nm = NoiseModel::estimate(&noisy);
        let (out, _) =
            lightcraft_denoise::denoise(&noisy, &nm, model, &[&lightcraft_denoise::Cpu], 1.0, lightcraft_denoise::Tiling::default(), &|_| true)
                .expect("denoise");
        before += psnr(&noisy, &target);
        after += psnr(&out, &target);
    }
    (before / n as f64, after / n as f64)
}

// ------------------------------------------------------------------------------------- main

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let steps: u64 = get("--steps").and_then(|v| v.parse().ok()).unwrap_or(3000);
    let bsz: usize = get("--batch").and_then(|v| v.parse().ok()).unwrap_or(8);
    let crop_px: usize = get("--crop").and_then(|v| v.parse::<usize>().ok()).unwrap_or(96).next_multiple_of(4);
    let lr0: f64 = get("--lr").and_then(|v| v.parse().ok()).unwrap_or(1e-3);
    let out = PathBuf::from(get("-o").unwrap_or_else(|| "denoise.lcdn".into()));
    let name = get("--name").unwrap_or_else(|| "LightCraft Denoise (dev)".into());
    let eval_every: u64 = get("--eval-every").and_then(|v| v.parse().ok()).unwrap_or(1000);

    let src = match get("--raws") {
        Some(d) => {
            eprintln!("loading raws from {d}");
            let r = load_raws(Path::new(&d));
            assert!(!r.is_empty(), "no decodable raws in {d}");
            Clean::Raws(r)
        }
        None => {
            eprintln!("no --raws: training on procedural scenes (a pipeline check, not a production model)");
            Clean::Scenes(lightcraft_scenes::demo_library())
        }
    };

    let (header, n) = unet_header(&name, "CC0-1.0", 4, 3);
    let device = Default::default();
    if let Some(path) = get("--check-model") {
        let model = Model::from_bytes(&std::fs::read(&path).expect("read")).expect("model");
        let (mut l1a, mut l1b, mut l2a, mut l2b, mut bias) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let k = 16;
        for i in 0..k {
            let (x, t) = sample(&src, 5_000_000 + i, crop_px);
            let n = crop_px * crop_px;
            let r = lightcraft_denoise::model::run_cpu(&model, &lightcraft_denoise::Tensor { c: 4, h: crop_px, w: crop_px, data: x.clone() })
                .expect("run");
            for j in 0..3 * n {
                let (y, tv, rv) = (x[j] as f64, t[j] as f64, r.data[j] as f64);
                l1a += (y - tv).abs();
                l1b += (y + rv - tv).abs();
                l2a += (y - tv).powi(2);
                l2b += (y + rv - tv).powi(2);
                bias += rv;
            }
        }
        let m = (k as usize * 3 * crop_px * crop_px) as f64;
        eprintln!("L1  noisy {:.5}  denoised {:.5}", l1a / m, l1b / m);
        eprintln!("MSE noisy {:.6}  denoised {:.6}  (PSNR {:.2} → {:.2} dB)", l2a / m, l2b / m, 10.0 * (m / l2a).log10(), 10.0 * (m / l2b).log10());
        eprintln!("mean residual {:.5}", bias / m);
        return;
    }
    if args.iter().any(|a| a == "--check-data") {
        let mut rng = Rng::new(7);
        let clean = clean_image(&src, &mut rng, 192);
        let t = sensor(&clean, None);
        let nz = sensor(&clean, Some((1e-3, 1e-5, &mut rng)));
        eprintln!("clean {}×{}, target {}×{}, noisy {}×{}", clean.width, clean.height, t.width, t.height, nz.width, nz.height);
        let d = |a: &Rgb32f, b: &Rgb32f| a.data.iter().zip(&b.data).map(|(p, q)| (p[1] - q[1]).abs()).sum::<f32>() / a.data.len() as f32;
        eprintln!("mean |target − clean| {:.4}, mean |noisy − target| {:.4}", d(&t, &clean), d(&nz, &t));
        let nm = NoiseModel::estimate(&nz);
        eprintln!("estimated noise a {:?} b {:?} (true a 1e-3 b 1e-5 per photosite)", nm.a, nm.b);
        return;
    }
    if args.iter().any(|a| a == "--bench-data") {
        let t = std::time::Instant::now();
        let n = 32;
        let _: Vec<_> = (0..n).into_par_iter().map(|i| sample(&src, i, crop_px)).collect();
        eprintln!("data: {:.0} ms per sample ({n} in parallel)", t.elapsed().as_secs_f64() * 1e3 / n as f64);
        return;
    }
    if args.iter().any(|a| a == "--selftest") {
        // burn's forward pass and the app's inference on the exported weights must agree
        let net: Net<Inner> = build(&header, &device);
        let (c, h, w) = (4, 40, 56);
        let xs: Vec<f32> = (0..c * h * w).map(|i| (i as f32 * 0.37).sin() * 0.5 + 0.5).collect();
        let x = Tensor::<Inner, 4>::from_data(TensorData::new(xs.clone(), [1, c, h, w]), &device);
        let ours = net.forward(&header, x).into_data().convert::<f32>().to_vec::<f32>().expect("data");
        let model = Model::from_bytes(&Model::to_bytes(&header, &net.export(&header, n)).expect("ser")).expect("model");
        let app = lightcraft_denoise::model::run_cpu(&model, &lightcraft_denoise::Tensor { c, h, w, data: xs }).expect("run");
        let scale = app.data.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = ours.iter().zip(&app.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("selftest: burn vs app inference, worst |Δ| = {worst:.2e} (output scale {scale:.2e}, f16 weights)");
        assert!(worst / scale < 5e-3, "the training network and the app's inference disagree");
        eprintln!("selftest: OK");
        return;
    }
    let mut net: Net<Train> = build(&header, &device);
    // start from "change nothing": the last convolution (the residual) begins at zero, so
    // early training can't make images worse than the input
    if let Some(last) = net.convs.last_mut() {
        let [o, i, kh, kw] = last.weight.val().dims();
        last.weight = burn::module::Param::from_tensor(Tensor::zeros([o, i, kh, kw], &device).require_grad());
        if let Some(b) = &mut last.bias {
            *b = burn::module::Param::from_tensor(Tensor::zeros([o], &device).require_grad());
        }
    }
    let mut optim = AdamConfig::new().init::<Train, Net<Train>>();
    if args.iter().any(|a| a == "--check-gradients") {
        let sums = |n: &Net<Train>| -> Vec<f32> { n.convs.iter().map(|c| c.weight.val().abs().sum().into_scalar().elem::<f32>()).collect() };
        let before = sums(&net);
        eprintln!("parameters: {}", net.num_params());
        let (xi, ti) = batch_data(&src, 0, 2, crop_px);
        let x = Tensor::<Train, 4>::from_data(TensorData::new(xi, [2, 4, crop_px, crop_px]), &device);
        let t = Tensor::<Train, 4>::from_data(TensorData::new(ti, [2, 3, crop_px, crop_px]), &device);
        let y = x.clone().narrow(1, 0, 3);
        let loss = (y + net.forward(&header, x) - t).abs().mean();
        eprintln!("loss {}", loss.clone().into_scalar().elem::<f32>());
        let g = loss.backward();
        let last = net.convs.last().map(|c| c.weight.grad(&g).map(|gr| gr.abs().sum().into_scalar().elem::<f32>()));
        let first = net.convs.first().map(|c| c.weight.grad(&g).map(|gr| gr.abs().sum().into_scalar().elem::<f32>()));
        eprintln!("grad |sum| first conv {first:?}, last conv {last:?}");
        let grads = GradientsParams::from_grads::<Train, Net<Train>>(g, &net);
        eprintln!(
            "grads collected for {} params; last conv weight id present: {}",
            grads.len(),
            net.convs.last().map(|c| grads.get::<Inner, 4>(c.weight.id).is_some()).unwrap_or(false)
        );
        net = optim.step(1e-3, net, grads);
        let after = sums(&net);
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            eprintln!("conv {i}: |w| {a:.4} → {b:.4}");
        }
        return;
    }
    let t0 = std::time::Instant::now();
    let mut avg = 0.0f64;
    // batches are made on other threads, a few steps ahead of the GPU
    let src = std::sync::Arc::new(src);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(3);
    {
        let src = src.clone();
        std::thread::spawn(move || {
            for step in 0..steps {
                if tx.send(batch_data(&src, step, bsz, crop_px)).is_err() {
                    break;
                }
            }
        });
    }
    for step in 0..steps {
        let (xi, ti) = rx.recv().expect("batch producer");
        let x = Tensor::<Train, 4>::from_data(TensorData::new(xi, [bsz, 4, crop_px, crop_px]), &device);
        let t = Tensor::<Train, 4>::from_data(TensorData::new(ti, [bsz, 3, crop_px, crop_px]), &device);
        // the network predicts the residual from the noisy gamma image (input channels 0..3)
        let y = x.clone().narrow(1, 0, 3);
        let pred = y + net.forward(&header, x);
        let loss = (pred - t).abs().mean();
        let l: f64 = loss.clone().into_scalar().elem();
        avg = if step == 0 { l } else { avg * 0.98 + l * 0.02 };
        let grads = GradientsParams::from_grads::<Train, Net<Train>>(loss.backward(), &net);
        // cosine decay to 5 %
        let lr = lr0 * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / steps as f64).cos()));
        net = optim.step(lr, net, grads);
        if step % 100 == 0 || step + 1 == steps {
            eprintln!("step {step:>6}  loss {avg:.5}  lr {lr:.2e}  {:.0} s", t0.elapsed().as_secs_f64());
        }
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let bytes = Model::to_bytes(&header, &net.export(&header, n)).expect("serialize");
            std::fs::write(&out, &bytes).expect("write model");
            let model = Model::from_bytes(&bytes).expect("reload");
            let (b, a) = evaluate(&src, &model);
            eprintln!("  wrote {} — held-out PSNR {b:.2} dB → {a:.2} dB", out.display());
        }
    }
}

use burn::tensor::ElementConversion;
