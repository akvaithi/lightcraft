//! AI Denoise on the GPU: [`lightcraft_denoise`]'s network operations as compute kernels
//! (`wgsl/nn.wgsl`), as a [`Backend`]. The weights stay on the device between tiles.

use std::sync::{Arc, Mutex};

use lightcraft_denoise::{Backend, Error, Model, Op, Result, Tensor};

use crate::ctx::{self, Buf, Gpu};

/// This machine's GPU as a denoise backend (`None` when there is no usable GPU or it is off).
pub fn backend() -> Option<Box<dyn Backend + Send>> {
    if !crate::enabled() {
        return None;
    }
    let g = crate::device()?;
    Some(Box::new(GpuDenoise { name: format!("GPU: {} ({:?})", g.info.name, g.info.backend) }))
}

pub struct GpuDenoise {
    name: String,
}

/// The weights of the last model used, by hash.
static WEIGHTS: Mutex<Option<(u64, Arc<Buf>)>> = Mutex::new(None);

fn weights(g: &Gpu, m: &Model) -> Arc<Buf> {
    let mut w = WEIGHTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((h, b)) = &*w
        && *h == m.hash
    {
        return b.clone();
    }
    let b = Arc::new(g.upload(&m.weights));
    *w = Some((m.hash, b.clone()));
    b
}

impl Backend for GpuDenoise {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn run(&self, model: &Model, x: &Tensor) -> Result<Tensor> {
        let err = |reason: String| Error::Backend { backend: self.name.clone(), reason };
        if !crate::enabled() {
            return Err(err(crate::unavailable_reason().unwrap_or_else(|| "GPU off".into())));
        }
        let g = crate::device().ok_or_else(|| err("no GPU".into()))?;
        let _ = ctx::take_failure();
        let (r, scoped) = {
            let _scope = ctx::RenderScope::new(g);
            let errors = ctx::ErrorScopes::push(g);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_net(g, model, x)));
            (r, errors.pop())
        };
        let failure = ctx::take_failure().or(scoped);
        match (r, failure) {
            (Err(_), _) => Err(err("a GPU denoise pass panicked".into())),
            (Ok(_), Some(f)) => {
                if f.kind == ctx::FailKind::OutOfMemory {
                    g.trim(0);
                }
                Err(err(f.reason))
            }
            (Ok(r), None) => r,
        }
    }
}

/// Output channels per `nn_conv` invocation (`OCB` in `wgsl/nn.wgsl`).
const OCB: usize = 8;

fn run_net(g: &Gpu, model: &Model, x: &Tensor) -> Result<Tensor> {
    let h = &model.header;
    if x.c != h.cin || x.h == 0 || x.w == 0 || !x.h.is_multiple_of(h.multiple) || !x.w.is_multiple_of(h.multiple) || x.data.len() != x.c * x.h * x.w {
        return Err(Error::Shape(format!("{}×{}×{} input for a {}-channel model", x.c, x.h, x.w, h.cin)));
    }
    let wb = weights(g, model);
    let mut enc = g.encoder();
    // (buffer, channels, height, width)
    let mut cur: (Arc<Buf>, usize, usize, usize) = (Arc::new(g.upload(&x.data)), x.c, x.h, x.w);
    let mut slots: Vec<Option<(Arc<Buf>, usize, usize, usize)>> = vec![None; 16];
    let u = |v: usize| v as u32;
    for op in &h.ops {
        cur = match *op {
            Op::Conv { cin, cout, k, stride, relu, w, b } => {
                let (ih, iw) = (cur.2, cur.3);
                let (oh, ow) = (ih.div_ceil(stride), iw.div_ceil(stride));
                let out = g.buffer(cout * oh * ow);
                let p = [u(cin), u(cout), u(k), u(stride), u32::from(relu), u(w), u(b), u(ih), u(iw), u(oh), u(ow)];
                g.run(
                    &mut enc,
                    "nn_conv",
                    &p,
                    &[Some(&cur.0), Some(&wb), Some(&out)],
                    [ow.div_ceil(16) as u32, oh.div_ceil(16) as u32, u(cout.div_ceil(OCB))],
                );
                (Arc::new(out), cout, oh, ow)
            }
            Op::Save(s) => {
                if let Some(slot) = slots.get_mut(s) {
                    *slot = Some(cur.clone());
                }
                cur
            }
            Op::Cat(s) => {
                let other = slots.get(s).cloned().flatten().ok_or_else(|| Error::Model(format!("slot {s} is empty")))?;
                if (other.2, other.3) != (cur.2, cur.3) {
                    return Err(Error::Shape("concatenating different sizes".into()));
                }
                let (a, b) = ((cur.1 * cur.2 * cur.3) as u64 * 4, (other.1 * other.2 * other.3) as u64 * 4);
                let out = g.buffer(cur.1 * cur.2 * cur.3 + other.1 * other.2 * other.3);
                enc.copy_buffer_to_buffer(cur.0.raw(), 0, out.raw(), 0, a);
                enc.copy_buffer_to_buffer(other.0.raw(), 0, out.raw(), a, b);
                (Arc::new(out), cur.1 + other.1, cur.2, cur.3)
            }
            Op::Up => {
                let (c, ih, iw) = (cur.1, cur.2, cur.3);
                let out = g.buffer(c * ih * iw * 4);
                g.run(
                    &mut enc,
                    "nn_up",
                    &[u(c), u(ih), u(iw)],
                    &[Some(&cur.0), None, Some(&out)],
                    [(iw * 2).div_ceil(16) as u32, (ih * 2).div_ceil(16) as u32, u(c)],
                );
                (Arc::new(out), c, ih * 2, iw * 2)
            }
        };
    }
    let n = cur.1 * cur.2 * cur.3;
    let data: Vec<f32> = g.finish_and_read(enc, &cur.0, n);
    Ok(Tensor { c: cur.1, h: cur.2, w: cur.3, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_denoise::model::unet_header;

    #[test]
    fn gpu_tiles_match_the_cpu() {
        let Some(b) = backend() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (h, n) = unet_header("t", "CC0", 4, 3);
        let w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.618).fract() - 0.5) * 0.08).collect();
        let m = Model::from_bytes(&Model::to_bytes(&h, &w).unwrap()).unwrap();
        let x = Tensor { c: 4, h: 40, w: 56, data: (0..4 * 40 * 56).map(|i| (i as f32 * 0.37).sin() * 0.5 + 0.5).collect() };
        let cpu = lightcraft_denoise::model::run_cpu(&m, &x).unwrap();
        let gpu = b.run(&m, &x).unwrap();
        assert_eq!((gpu.c, gpu.h, gpu.w), (cpu.c, cpu.h, cpu.w));
        let scale = cpu.data.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let worst = cpu.data.iter().zip(&gpu.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst / scale < 1e-4, "GPU vs CPU: {worst} (scale {scale})");
        // a second tile reuses the weights on the device
        assert!(b.run(&m, &x).is_ok());
    }

    /// Speed of a 304 × 304 tile (256 core + overlap) on the GPU and the CPU:
    /// `cargo test -p lightcraft-gpu --release --lib denoise_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn denoise_speed() {
        let (h, n) = unet_header("t", "CC0", 4, 3);
        let m = Model::from_bytes(&Model::to_bytes(&h, &vec![0.01; n]).unwrap()).unwrap();
        let x = Tensor { c: 4, h: 304, w: 304, data: vec![0.3; 4 * 304 * 304] };
        let time = |f: &dyn Fn()| {
            f();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                f();
            }
            t.elapsed().as_secs_f64() / 3.0
        };
        let cpu = time(&|| {
            lightcraft_denoise::model::run_cpu(&m, &x).unwrap();
        });
        let mp_per_tile = 256.0 * 256.0 / 1e6;
        eprintln!("CPU: {:.0} ms/tile, {:.1} s per 24 MP", cpu * 1e3, cpu / mp_per_tile * 24.0);
        if let Some(b) = backend() {
            let gpu = time(&|| {
                b.run(&m, &x).unwrap();
            });
            eprintln!("{}: {:.0} ms/tile, {:.1} s per 24 MP", b.name(), gpu * 1e3, gpu / mp_per_tile * 24.0);
            let small = Tensor { c: 4, h: 16, w: 16, data: vec![0.3; 4 * 16 * 16] };
            let tiny = time(&|| {
                b.run(&m, &small).unwrap();
            });
            eprintln!("  16 × 16 tile: {:.1} ms (fixed cost per tile)", tiny * 1e3);
        }
    }
}
