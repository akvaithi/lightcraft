//! The denoise model file (`.lcdn`): a small convolutional network as a list of operations plus
//! its weights, so a retrained or reshaped network ships as a new file, not new code.
//!
//! ```text
//! "LCDN" | u32 LE version (1) | u32 LE header length | header (JSON, UTF-8) | f16 LE weights
//! ```
//!
//! The header names the model, its licence and preprocessing, the input/output channel counts, the
//! size multiple tiles must have, and the operations. Weights are addressed by element offset into
//! the f16 block; a convolution's kernel is laid out `[out][in][ky][kx]`, then `out` biases.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const MAGIC: &[u8; 4] = b"LCDN";
pub const VERSION: u32 = 1;
/// Largest model file accepted (weights and header): a guard against hostile files.
pub const MAX_BYTES: usize = 256 << 20;
const MAX_HEADER: usize = 1 << 20;
const MAX_OPS: usize = 512;
const MAX_CHANNELS: usize = 1024;
const MAX_SLOTS: usize = 16;

/// One network operation, applied to the running activation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Op {
    /// Convolution (`k` 1 or 3, zero padding `k / 2`, `stride` 1 or 2), then ReLU if `relu`.
    Conv { cin: usize, cout: usize, k: usize, stride: usize, relu: bool, w: usize, b: usize },
    /// Keep the current activation in `slot` (for a skip connection).
    Save(usize),
    /// Concatenate `slot` after the current activation's channels.
    Cat(usize),
    /// Bilinear ×2 upsampling (pixel-centre aligned).
    Up,
}

/// Input preparation and output reconstruction (see [`crate::prepare`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Preprocess {
    /// Linear camera RGB → `x^(1/2.2)` per channel plus one noise-level channel (the standard
    /// deviation of the noise in that domain); the network predicts a residual in it.
    #[default]
    Gamma22Sigma,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Header {
    /// Name and version tag (shown to the user, compared by remote servers).
    pub name: String,
    pub license: String,
    #[serde(default)]
    pub preprocess: Preprocess,
    /// Input and output channels.
    pub cin: usize,
    pub cout: usize,
    /// Tile sides must be multiples of this (2^number of stride-2 convolutions).
    pub multiple: usize,
    pub ops: Vec<Op>,
}

/// A loaded, validated model.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub header: Header,
    pub weights: Vec<f32>,
    /// FNV-1a hash of the file: identifies the exact weights (local vs remote must agree).
    pub hash: u64,
}

fn fnv1a(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Model {
    /// Parse and validate a model file.
    pub fn from_bytes(b: &[u8]) -> Result<Model> {
        let bad = |m: &str| Error::Model(m.into());
        if b.len() > MAX_BYTES {
            return Err(bad("model file too large"));
        }
        if b.get(..4) != Some(MAGIC) {
            return Err(bad("not a LightCraft denoise model"));
        }
        let u32_at = |o: usize| b.get(o..o + 4).and_then(|s| s.try_into().ok()).map(u32::from_le_bytes);
        let version = u32_at(4).ok_or_else(|| bad("truncated"))?;
        if version != VERSION {
            return Err(Error::Model(format!("model format version {version} (this build reads {VERSION})")));
        }
        let hlen = u32_at(8).ok_or_else(|| bad("truncated"))? as usize;
        if hlen > MAX_HEADER {
            return Err(bad("model header too large"));
        }
        let hb = b.get(12..12 + hlen).ok_or_else(|| bad("truncated header"))?;
        let header: Header = serde_json::from_slice(hb).map_err(|e| Error::Model(format!("header: {e}")))?;
        let wb = b.get(12 + hlen..).ok_or_else(|| bad("truncated weights"))?;
        if wb.len() % 2 != 0 {
            return Err(bad("odd weight block length"));
        }
        let weights: Vec<f32> = wb.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect();
        let m = Model { header, weights, hash: fnv1a(b) };
        m.validate()?;
        Ok(m)
    }

    /// Serialize (f16 weights).
    pub fn to_bytes(header: &Header, weights: &[f32]) -> Result<Vec<u8>> {
        let h = serde_json::to_vec(header).map_err(|e| Error::Model(e.to_string()))?;
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(h.len() as u32).to_le_bytes());
        out.extend_from_slice(&h);
        for w in weights {
            out.extend_from_slice(&half::f16::from_f32(*w).to_le_bytes());
        }
        Ok(out)
    }

    /// Walk the operations: channel counts and slots must line up, weights must be in range, and
    /// the network must end with `cout` channels at the input's resolution.
    fn validate(&self) -> Result<()> {
        let h = &self.header;
        let bad = |m: String| Err(Error::Model(m));
        if h.ops.len() > MAX_OPS || h.cin == 0 || h.cin > MAX_CHANNELS || h.cout == 0 || h.cout > MAX_CHANNELS {
            return bad("implausible model shape".into());
        }
        if !matches!(h.multiple, 1 | 2 | 4 | 8 | 16 | 32) {
            return bad(format!("tile multiple {} must be a power of two ≤ 32", h.multiple));
        }
        let mut c = h.cin;
        let mut scale = 0i32; // log2 of downsampling
        let mut slots: [Option<(usize, i32)>; MAX_SLOTS] = [None; MAX_SLOTS];
        for (i, op) in h.ops.iter().enumerate() {
            match *op {
                Op::Conv { cin, cout, k, stride, w, b, .. } => {
                    if cin != c || cout == 0 || cout > MAX_CHANNELS || !matches!(k, 1 | 3) || !matches!(stride, 1 | 2) {
                        return bad(format!("op {i}: convolution {cin}→{cout} k{k} s{stride} doesn't fit {c} channels"));
                    }
                    let nw = cout.checked_mul(cin).and_then(|v| v.checked_mul(k * k)).unwrap_or(usize::MAX);
                    if w.checked_add(nw).is_none_or(|e| e > self.weights.len()) || b.checked_add(cout).is_none_or(|e| e > self.weights.len()) {
                        return bad(format!("op {i}: weights out of range"));
                    }
                    c = cout;
                    if stride == 2 {
                        scale += 1;
                    }
                }
                Op::Save(s) => match slots.get_mut(s) {
                    Some(slot) => *slot = Some((c, scale)),
                    None => return bad(format!("op {i}: slot {s} out of range")),
                },
                Op::Cat(s) => match slots.get(s).copied().flatten() {
                    Some((sc, ss)) if ss == scale => c += sc,
                    Some(_) => return bad(format!("op {i}: slot {s} has another resolution")),
                    None => return bad(format!("op {i}: slot {s} is empty")),
                },
                Op::Up => scale -= 1,
            }
            if c > MAX_CHANNELS || scale < 0 || (1usize << scale.clamp(0, 30)) > h.multiple {
                return bad(format!("op {i}: resolution or channels out of range"));
            }
        }
        if c != h.cout || scale != 0 {
            return bad(format!("the network ends with {c} channels at 1/{} resolution, not {} at full", 1 << scale.clamp(0, 30), h.cout));
        }
        Ok(())
    }

    /// Multiply-accumulates per input pixel (a speed estimate).
    pub fn macs_per_pixel(&self) -> f64 {
        let mut area = 1.0f64;
        let mut macs = 0.0;
        for op in &self.header.ops {
            match *op {
                Op::Conv { cin, cout, k, stride, .. } => {
                    if stride == 2 {
                        area /= 4.0;
                    }
                    macs += (cin * cout * k * k) as f64 * area;
                }
                Op::Up => area *= 4.0,
                _ => {}
            }
        }
        macs
    }
}

/// A channels × height × width activation.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(c: usize, h: usize, w: usize) -> Tensor {
        Tensor { c, h, w, data: vec![0.0; c * h * w] }
    }
    fn plane(&self, i: usize) -> &[f32] {
        let n = self.h * self.w;
        self.data.get(i * n..(i + 1) * n).unwrap_or(&[])
    }
}

/// Run `model` on `x` (`cin` channels, sides multiples of `multiple`) on the CPU.
pub fn run_cpu(model: &Model, x: &Tensor) -> Result<Tensor> {
    use rayon::prelude::*;
    let h = &model.header;
    if x.c != h.cin || x.h == 0 || x.w == 0 || !x.h.is_multiple_of(h.multiple) || !x.w.is_multiple_of(h.multiple) || x.data.len() != x.c * x.h * x.w {
        return Err(Error::Shape(format!("{}×{}×{} input for a {}-channel model (multiple {})", x.c, x.h, x.w, h.cin, h.multiple)));
    }
    let wts = &model.weights;
    let mut cur = x.clone();
    let mut slots: Vec<Option<Tensor>> = vec![None; MAX_SLOTS];
    for op in &h.ops {
        cur = match *op {
            Op::Conv { cin, cout, k, stride, relu, w, b } => {
                let (oh, ow) = (cur.h.div_ceil(stride), cur.w.div_ceil(stride));
                let pad = (k / 2) as isize;
                let mut out = Tensor::zeros(cout, oh, ow);
                let src = &cur;
                out.data.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, o)| {
                    let bias = wts.get(b + oc).copied().unwrap_or(0.0);
                    o.fill(bias);
                    for ic in 0..cin {
                        let inp = src.plane(ic);
                        for ky in 0..k {
                            for kx in 0..k {
                                let wv = wts.get(w + ((oc * cin + ic) * k + ky) * k + kx).copied().unwrap_or(0.0);
                                if wv == 0.0 {
                                    continue;
                                }
                                for oy in 0..oh {
                                    let iy = (oy * stride) as isize + ky as isize - pad;
                                    if iy < 0 || iy >= src.h as isize {
                                        continue;
                                    }
                                    let irow = &inp[iy as usize * src.w..(iy as usize + 1) * src.w];
                                    let orow = &mut o[oy * ow..(oy + 1) * ow];
                                    let dx = kx as isize - pad;
                                    if stride == 1 {
                                        // in-range span of x for this tap
                                        let x0 = (-dx).max(0) as usize;
                                        let x1 = ((src.w as isize - dx).min(ow as isize)).max(0) as usize;
                                        if x0 < x1 {
                                            let ii = &irow[(x0 as isize + dx) as usize..(x1 as isize + dx) as usize];
                                            for (ov, iv) in orow[x0..x1].iter_mut().zip(ii) {
                                                *ov += wv * iv;
                                            }
                                        }
                                    } else {
                                        for (ox, ov) in orow.iter_mut().enumerate() {
                                            let ix = (ox * 2) as isize + dx;
                                            if ix >= 0 && (ix as usize) < src.w {
                                                *ov += wv * irow[ix as usize];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if relu {
                        o.iter_mut().for_each(|v| *v = v.max(0.0));
                    }
                });
                out
            }
            Op::Save(s) => {
                if let Some(slot) = slots.get_mut(s) {
                    *slot = Some(cur.clone());
                }
                cur
            }
            Op::Cat(s) => {
                let other = slots.get(s).cloned().flatten().ok_or_else(|| Error::Model(format!("slot {s} is empty")))?;
                if (other.h, other.w) != (cur.h, cur.w) {
                    return Err(Error::Shape(format!("concatenating {}×{} with {}×{}", cur.h, cur.w, other.h, other.w)));
                }
                let mut data = cur.data;
                data.extend_from_slice(&other.data);
                Tensor { c: cur.c + other.c, h: cur.h, w: cur.w, data }
            }
            Op::Up => upsample2(&cur),
        };
    }
    Ok(cur)
}

/// Bilinear ×2 upsampling with pixel-centre alignment (edge-clamped).
pub fn upsample2(t: &Tensor) -> Tensor {
    use rayon::prelude::*;
    let (h, w) = (t.h * 2, t.w * 2);
    let mut out = Tensor::zeros(t.c, h, w);
    out.data.par_chunks_mut(h * w).enumerate().for_each(|(c, o)| {
        let p = t.plane(c);
        let at = |y: isize, x: isize| p[y.clamp(0, t.h as isize - 1) as usize * t.w + x.clamp(0, t.w as isize - 1) as usize];
        for y in 0..h {
            // output centre (y + 0.5) / 2 − 0.5 in input pixels
            let sy = (y as f32 + 0.5) * 0.5 - 0.5;
            let y0 = sy.floor();
            let fy = sy - y0;
            for x in 0..w {
                let sx = (x as f32 + 0.5) * 0.5 - 0.5;
                let x0 = sx.floor();
                let fx = sx - x0;
                let (y0, x0) = (y0 as isize, x0 as isize);
                let a = at(y0, x0) + (at(y0, x0 + 1) - at(y0, x0)) * fx;
                let b = at(y0 + 1, x0) + (at(y0 + 1, x0 + 1) - at(y0 + 1, x0)) * fx;
                o[y * w + x] = a + (b - a) * fy;
            }
        }
    });
    out
}

/// The default architecture: a three-level U-Net (32, 64, 128 channels) over `cin` inputs with
/// `cout` residual outputs. Returns the header and the number of weights it needs.
pub fn unet_header(name: &str, license: &str, cin: usize, cout: usize) -> (Header, usize) {
    let mut ops = Vec::new();
    let mut off = 0usize;
    let mut conv = |ops: &mut Vec<Op>, cin: usize, cout: usize, k: usize, stride: usize, relu: bool| {
        let w = off;
        off += cin * cout * k * k;
        let b = off;
        off += cout;
        ops.push(Op::Conv { cin, cout, k, stride, relu, w, b });
    };
    conv(&mut ops, cin, 32, 3, 1, true);
    conv(&mut ops, 32, 32, 3, 1, true);
    ops.push(Op::Save(0));
    conv(&mut ops, 32, 64, 3, 2, true);
    conv(&mut ops, 64, 64, 3, 1, true);
    ops.push(Op::Save(1));
    conv(&mut ops, 64, 128, 3, 2, true);
    conv(&mut ops, 128, 128, 3, 1, true);
    conv(&mut ops, 128, 128, 3, 1, true);
    ops.push(Op::Up);
    conv(&mut ops, 128, 64, 3, 1, true);
    ops.push(Op::Cat(1));
    conv(&mut ops, 128, 64, 3, 1, true);
    ops.push(Op::Up);
    conv(&mut ops, 64, 32, 3, 1, true);
    ops.push(Op::Cat(0));
    conv(&mut ops, 64, 32, 3, 1, true);
    conv(&mut ops, 32, cout, 3, 1, false);
    let header = Header { name: name.into(), license: license.into(), preprocess: Preprocess::Gamma22Sigma, cin, cout, multiple: 4, ops };
    (header, off)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Header, Vec<f32>) {
        let (h, n) = unet_header("test", "CC0", 4, 3);
        let w: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).map(|v| v * 0.05).collect();
        (h, w)
    }

    #[test]
    fn round_trips_and_validates() {
        let (h, w) = tiny();
        let b = Model::to_bytes(&h, &w).unwrap();
        let m = Model::from_bytes(&b).unwrap();
        assert_eq!(m.header, h);
        assert_eq!(m.weights.len(), w.len());
        assert!(m.weights.iter().zip(&w).all(|(a, b)| (a - b).abs() < 1e-3));
        assert!(m.macs_per_pixel() > 50_000.0 && m.macs_per_pixel() < 200_000.0, "{}", m.macs_per_pixel());
        // corrupt files are refused, not trusted
        assert!(Model::from_bytes(&b[..b.len() - 3]).is_err());
        assert!(Model::from_bytes(b"LCDN").is_err());
        let mut bad = h.clone();
        bad.ops.pop();
        assert!(Model::from_bytes(&Model::to_bytes(&bad, &w).unwrap()).is_err(), "must end with cout channels");
        let mut bad = h.clone();
        bad.ops.insert(0, Op::Cat(3));
        assert!(Model::from_bytes(&Model::to_bytes(&bad, &w).unwrap()).is_err(), "empty slot");
        assert!(Model::from_bytes(&Model::to_bytes(&h, &w[..w.len() - 10]).unwrap()).is_err(), "weights out of range");
    }

    #[test]
    fn conv_matches_a_naive_reference() {
        // one 3×3 stride-2 convolution, checked against direct summation
        let h = Header {
            name: "c".into(),
            license: "CC0".into(),
            preprocess: Preprocess::default(),
            cin: 2,
            cout: 3,
            multiple: 2,
            ops: vec![Op::Conv { cin: 2, cout: 3, k: 3, stride: 2, relu: false, w: 0, b: 54 }, Op::Up],
        };
        let w: Vec<f32> = (0..57).map(|i| (i as f32 * 0.37).sin()).collect();
        let m = Model::from_bytes(&Model::to_bytes(&h, &w).unwrap()).unwrap();
        let x = Tensor { c: 2, h: 6, w: 8, data: (0..96).map(|i| (i as f32 * 0.11).cos()).collect() };
        // the conv alone ends at half size (not a valid model), so check conv + Up against
        // the naive conv upsampled
        let full = run_cpu(&m, &x).unwrap();
        assert_eq!((full.c, full.h, full.w), (3, 6, 8));
        let mut naive = Tensor::zeros(3, 3, 4);
        for oc in 0..3 {
            for oy in 0..3 {
                for ox in 0..4 {
                    let mut s = m.weights[54 + oc];
                    for ic in 0..2 {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let (iy, ix) = (oy as isize * 2 + ky as isize - 1, ox as isize * 2 + kx as isize - 1);
                                if (0..6).contains(&iy) && (0..8).contains(&ix) {
                                    s += m.weights[((oc * 2 + ic) * 3 + ky) * 3 + kx] * x.data[ic * 48 + iy as usize * 8 + ix as usize];
                                }
                            }
                        }
                    }
                    naive.data[oc * 12 + oy * 4 + ox] = s;
                }
            }
        }
        let up = upsample2(&naive);
        for (a, b) in full.data.iter().zip(&up.data) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn unet_runs_and_keeps_the_shape() {
        let (h, w) = tiny();
        let m = Model::from_bytes(&Model::to_bytes(&h, &w).unwrap()).unwrap();
        let x = Tensor { c: 4, h: 16, w: 24, data: vec![0.3; 4 * 16 * 24] };
        let y = run_cpu(&m, &x).unwrap();
        assert_eq!((y.c, y.h, y.w), (3, 16, 24));
        assert!(y.data.iter().all(|v| v.is_finite()));
        assert!(run_cpu(&m, &Tensor { c: 4, h: 15, w: 24, data: vec![0.0; 4 * 15 * 24] }).is_err(), "not a multiple of 4");
    }
}
