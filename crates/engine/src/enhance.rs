//! Enhance → Denoise: a learned denoiser applied to the demosaiced raw data, written as a new
//! linear DNG (`<name>-Enhanced-NR.dng`) next to the original, imported with the original's edits
//! (manual noise reduction zeroed) and stacked on it, as Lightroom does.
//!
//! Tiles run on a chain of backends: a remote GPU server when one is configured
//! (`LIGHTCRAFT_DENOISE_URL` + `LIGHTCRAFT_DENOISE_TOKEN`, or the library preferences), this
//! machine's GPU, then the CPU; a backend that fails hands the rest to the next one.
//!
//! The model is a `.lcdn` file ([`lightcraft_denoise::model`]): `LIGHTCRAFT_DENOISE_MODEL`, else
//! the library preference.

use std::sync::Arc;

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_denoise::{Backend, Model, NoiseModel, Tiling, remote::Remote};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::merge::ByteReader;
use crate::{EngineError, Result, Session};

/// Denoise settings kept with the library. Environment variables take precedence.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DenoisePrefs {
    /// Path of the `.lcdn` model.
    pub model: Option<String>,
    /// Remote server `host[:port]` (tailnet / LAN).
    pub remote: Option<String>,
    /// The remote server's token. Stored in the library's prefs.json in plain text: prefer
    /// `LIGHTCRAFT_DENOISE_TOKEN`.
    pub token: Option<String>,
    /// Use the remote server when configured (default on).
    pub use_remote: Option<bool>,
}

impl DenoisePrefs {
    fn env(k: &str) -> Option<String> {
        std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
    }

    /// The model path in effect.
    pub fn model_path(&self) -> Option<String> {
        Self::env("LIGHTCRAFT_DENOISE_MODEL").or_else(|| self.model.clone().filter(|m| !m.trim().is_empty()))
    }

    /// The remote server and token in effect (`None` when not configured or switched off).
    pub fn remote(&self) -> Option<(String, String)> {
        if self.use_remote == Some(false) {
            return None;
        }
        let addr = Self::env("LIGHTCRAFT_DENOISE_URL").or_else(|| self.remote.clone().filter(|r| !r.trim().is_empty()))?;
        let token = Self::env("LIGHTCRAFT_DENOISE_TOKEN").or_else(|| self.token.clone()).unwrap_or_default();
        // accept a URL-ish form: tcp://host:port, http://host:port
        let addr = addr.split("://").last().unwrap_or(&addr).trim_end_matches('/').to_string();
        Some((addr, token))
    }
}

/// Load the model at `path`.
pub fn load_model(path: &str) -> Result<Arc<Model>> {
    let b = std::fs::read(path).map_err(|e| EngineError::Other(format!("denoise model {path}: {e}")))?;
    Model::from_bytes(&b).map(Arc::new).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}

/// A planned denoise of one or more photos (runs without the session, e.g. on a worker thread).
pub struct DenoiseJob {
    pub sources: Vec<(PhotoId, String)>,
    /// 0..1.
    pub amount: f32,
    pub stack: bool,
    model: Arc<Model>,
    remote: Option<(String, String)>,
    read: ByteReader,
}

/// One denoised photo.
pub struct DenoiseOutput {
    pub source: PhotoId,
    pub dng: Vec<u8>,
    pub report: lightcraft_denoise::Report,
    pub width: usize,
    pub height: usize,
}

impl DenoiseJob {
    /// The backends in order: remote, GPU, CPU.
    fn backends(&self) -> Vec<Box<dyn Backend + Send>> {
        let mut v: Vec<Box<dyn Backend + Send>> = Vec::new();
        if let Some((addr, token)) = &self.remote {
            v.push(Box::new(Remote::new(addr, token)));
        }
        if let Some(gpu) = lightcraft_gpu::denoise::backend() {
            v.push(gpu);
        }
        v.push(Box::new(lightcraft_denoise::Cpu));
        v
    }

    /// Denoise every source. `progress(fraction, stage)` returns false to cancel.
    pub fn run(&self, progress: &(dyn Fn(f32, &str) -> bool + Sync)) -> std::result::Result<Vec<DenoiseOutput>, String> {
        let owned = self.backends();
        let backends: Vec<&dyn Backend> = owned.iter().map(|b| b.as_ref() as &dyn Backend).collect();
        let n = self.sources.len().max(1) as f32;
        let mut outs = Vec::new();
        for (k, (id, path)) in self.sources.iter().enumerate() {
            let base = k as f32 / n;
            let name = std::path::Path::new(path).file_name().and_then(|f| f.to_str()).unwrap_or(path).to_string();
            if !progress(base, &format!("Reading {name}")) {
                return Err("cancelled".into());
            }
            let bytes = (self.read)(path)?;
            if lightcraft_raw::probe(&bytes).is_none() {
                return Err(format!("{name}: Denoise works on raw photos"));
            }
            let frame = lightcraft_merge::load_frame(&bytes, None, false).map_err(|e| format!("{name}: {e}"))?;
            drop(bytes);
            let noise = NoiseModel::estimate(&frame.image);
            let tile_progress = |f: f32| progress(base + f * 0.9 / n, &format!("Denoising {name}"));
            let (img, report) =
                lightcraft_denoise::denoise(&frame.image, &noise, &self.model, &backends, self.amount, Tiling::default(), &tile_progress)
                    .map_err(|e| format!("{name}: {e}"))?;
            if !progress(base + 0.95 / n, "Writing DNG") {
                return Err("cancelled".into());
            }
            let mut metadata = frame.metadata.clone();
            metadata.software = Some(format!("LightCraft Enhance (Denoise: {})", self.model.header.name));
            let dng = lightcraft_merge::write_linear_dng(
                &img,
                &frame.color,
                frame.orientation,
                &metadata,
                frame.baseline_exposure,
                lightcraft_merge::DngSamples::Half,
            )
            .map_err(|e| format!("{name}: {e}"))?;
            outs.push(DenoiseOutput { source: *id, dng, report, width: img.width, height: img.height });
        }
        progress(1.0, "Done");
        Ok(outs)
    }
}

/// `photo.dng` → `photo-Enhanced-NR.dng` (numbered when taken).
fn output_path(source: &str) -> String {
    let p = std::path::Path::new(source);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("photo");
    let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
    let mut out = dir.join(format!("{stem}-Enhanced-NR.dng"));
    let mut i = 2;
    while out.exists() && i < 1000 {
        out = dir.join(format!("{stem}-Enhanced-NR-{i}.dng"));
        i += 1;
    }
    out.to_string_lossy().to_string()
}

impl Session {
    /// The denoise model in effect, loaded.
    pub fn denoise_model(&self) -> Result<Arc<Model>> {
        let path = self.denoise.model_path().ok_or_else(|| {
            EngineError::Other("no denoise model: set LIGHTCRAFT_DENOISE_MODEL or `denoise.preferences {model}` to a .lcdn file".into())
        })?;
        load_model(&path)
    }

    /// Plan Enhance → Denoise of `ids` at `amount` (0..100).
    pub fn plan_denoise(&self, ids: &[PhotoId], amount: f64, stack: bool) -> Result<DenoiseJob> {
        let mut sources = Vec::new();
        for id in ids {
            let p = self.catalog.photo(*id).ok_or(lightcraft_catalog::CatalogError::NoPhoto(*id))?;
            match &p.source {
                Source::File { path } => sources.push((*id, path.clone())),
                Source::Demo { .. } => return Err(EngineError::Other(format!("{} is a demo photo; Denoise needs raw files", p.file_name))),
            }
        }
        if sources.is_empty() {
            return Err(EngineError::Other("select the raw photos to denoise".into()));
        }
        let read: ByteReader = match &self.media.file_bytes {
            Some(r) => r.clone(),
            None => Arc::new(|path: &str| std::fs::read(path).map_err(|e| format!("{path}: {e}"))),
        };
        let amount = if amount.is_finite() { (amount / 100.0).clamp(0.0, 1.0) as f32 } else { 0.5 };
        Ok(DenoiseJob { sources, amount, stack, model: self.denoise_model()?, remote: self.denoise.remote(), read })
    }

    /// Write, import and stack finished denoise outputs; returns one entry per photo.
    pub fn finish_denoise(&mut self, job: &DenoiseJob, outs: Vec<DenoiseOutput>) -> Result<Value> {
        let mut done = Vec::new();
        for out in outs {
            let Some((_, src_path)) = job.sources.iter().find(|(id, _)| *id == out.source) else { continue };
            let path = output_path(src_path);
            std::fs::write(&path, &out.dng).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
            let report = crate::import::import(self, std::slice::from_ref(&path), crate::import::ImportMode::Add)?;
            let id = report
                .imported
                .first()
                .copied()
                .map(PhotoId)
                .ok_or_else(|| EngineError::Other(format!("the denoised file {path} could not be imported: {:?}", report.failed)))?;
            // the original's edits, without the manual noise reduction the AI replaced
            if let Some(src) = self.develop_of(out.source) {
                let mut d = (*src).clone();
                d.detail.nr_luminance = 0.0;
                d.detail.nr_color = 0.0;
                d.enhance.denoise = (job.amount * 100.0) as f64;
                self.set_develop(id, d, "Denoise")?;
            }
            if job.stack
                && let Some(op) = self.catalog.stack_with_ops(id, &[out.source])
            {
                self.commit("Stack with Original", op)?;
            }
            done.push(json!({"id": id.0, "source": out.source.0, "path": path, "width": out.width, "height": out.height, "report": out.report}));
        }
        if let Some(id) = done.last().and_then(|d| d["id"].as_u64()) {
            self.selection = crate::Selection::single(PhotoId(id));
        }
        Ok(json!({"photos": done}))
    }

    /// Plan, run and finish synchronously (CLI, MCP, control channel).
    pub fn denoise_now(&mut self, ids: &[PhotoId], amount: f64, stack: bool) -> Result<Value> {
        let job = self.plan_denoise(ids, amount, stack)?;
        let outs = job.run(&|_, _| true).map_err(EngineError::Other)?;
        self.finish_denoise(&job, outs)
    }

    /// The model and the remote server in effect, and whether the server answers.
    pub fn denoise_status(&self) -> Value {
        let model = match self.denoise.model_path() {
            None => json!({"error": "no model configured"}),
            Some(p) => match load_model(&p) {
                Ok(m) => {
                    json!({"path": p, "name": m.header.name, "license": m.header.license, "hash": lightcraft_denoise::remote::hash_hex(m.hash), "macsPerPixel": m.macs_per_pixel().round()})
                }
                Err(e) => json!({"path": p, "error": e.to_string()}),
            },
        };
        let remote = match self.denoise.remote() {
            None => Value::Null,
            Some((addr, token)) => match Remote::new(&addr, &token).hello() {
                Ok(info) => {
                    let matches = model.get("hash").and_then(Value::as_str).map(|h| h == info.model_hash);
                    json!({"address": info.address, "ok": true, "backend": info.backend, "model": info.model_name, "modelHash": info.model_hash, "sameModel": matches})
                }
                Err(e) => json!({"address": addr, "ok": false, "error": e.to_string()}),
            },
        };
        let gpu = lightcraft_gpu::denoise::backend().map(|b| b.name());
        json!({"model": model, "remote": remote, "gpu": gpu, "tokenSet": self.denoise.remote().is_some_and(|r| !r.1.is_empty())})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_pick_env_over_library_and_strip_schemes() {
        let p = DenoisePrefs { remote: Some("tcp://100.105.122.87:7990/".into()), token: Some("t".into()), ..Default::default() };
        // (environment variables are not set in tests)
        if std::env::var("LIGHTCRAFT_DENOISE_URL").is_err() {
            assert_eq!(p.remote(), Some(("100.105.122.87:7990".into(), "t".into())));
        }
        let off = DenoisePrefs { use_remote: Some(false), ..p.clone() };
        assert_eq!(off.remote(), None);
        assert_eq!(output_path("/x/IMG_1.CR2"), "/x/IMG_1-Enhanced-NR.dng");
    }
}
