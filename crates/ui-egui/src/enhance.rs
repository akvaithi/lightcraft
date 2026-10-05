//! Photo ▸ Enhance…: the AI Denoise dialog and its background job.
//!
//! The job ([`lightcraft_engine::enhance::DenoiseJob`]) runs on a worker thread with progress in a
//! toast and keeps running after the dialog closes; the result is written, imported, stacked and
//! selected by [`lightcraft_engine::Session::finish_denoise`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use lightcraft_catalog::PhotoId;
use lightcraft_engine::enhance::{DenoiseJob, DenoiseOutput};
use serde_json::{Value, json};

use crate::LightcraftApp;

pub struct EnhanceTask {
    job: DenoiseJob,
    pub progress: Arc<Mutex<(f32, String)>>,
    pub cancel: Arc<AtomicBool>,
    rx: Receiver<Result<Vec<DenoiseOutput>, String>>,
}

#[derive(Default)]
pub struct EnhanceState {
    pub task: Option<EnhanceTask>,
    /// Photos the dialog was opened for.
    pub ids: Vec<PhotoId>,
    /// `denoise.status` when the dialog opened (model, remote server, GPU).
    pub status: Option<Value>,
    pub last_result: Option<Value>,
}

/// Open the dialog for the selection.
pub fn open(app: &mut LightcraftApp) -> Result<Value, String> {
    let ids = app.session.targets(&json!({}));
    if ids.is_empty() {
        return Err("select the raw photos to denoise".into());
    }
    app.enhance.ids = ids;
    app.enhance.status = Some(app.session.denoise_status());
    app.ui.dialog = Some(crate::state::Dialog::Enhance { amount: 50.0, stack: true });
    Ok(json!({"photos": app.enhance.ids.len()}))
}

/// Start denoising (the dialog's Denoise button).
pub fn start(app: &mut LightcraftApp, amount: f64, stack: bool) -> Result<Value, String> {
    if app.enhance.task.is_some() {
        return Err("Denoise is already running".into());
    }
    let ids = app.enhance.ids.clone();
    let job = app.session.plan_denoise(&ids, amount, stack).map_err(|e| e.to_string())?;
    let progress = Arc::new(Mutex::new((0.0, "Starting".to_string())));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = channel();
    let (j, p, c) = (job.clone(), progress.clone(), cancel.clone());
    let work = move || {
        let run = || {
            j.run(&|f, stage| {
                if let Ok(mut g) = p.lock() {
                    *g = (f, stage.to_string());
                }
                !c.load(Ordering::Relaxed)
            })
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|_| Err("Denoise failed unexpectedly".into()));
        let _ = tx.send(r);
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name("denoise".into()).spawn(work).map_err(|e| e.to_string())?;
    #[cfg(target_arch = "wasm32")]
    work();
    app.enhance.task = Some(EnhanceTask { job, progress, cancel, rx });
    Ok(json!({"started": true, "photos": ids.len()}))
}

/// Per frame: progress toast, and the finished job.
pub fn poll(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(t) = &app.enhance.task else { return };
    match t.rx.try_recv() {
        Ok(r) => {
            let Some(t) = app.enhance.task.take() else { return };
            match r.map_err(lightcraft_engine::EngineError::Other).and_then(|outs| app.session.finish_denoise(&t.job, outs)) {
                Ok(v) => {
                    let n = v["photos"].as_array().map_or(0, Vec::len);
                    let via =
                        v["photos"][0]["report"]["backends"].as_array().and_then(|b| b.first()).and_then(Value::as_str).unwrap_or("").to_string();
                    let fell_back =
                        v["photos"].as_array().is_some_and(|a| a.iter().any(|p| p["report"]["fallbacks"].as_array().is_some_and(|f| !f.is_empty())));
                    app.enhance.last_result = Some(v);
                    let mut msg = format!("Denoised {n} photo{} on {via}", if n == 1 { "" } else { "s" });
                    if fell_back {
                        msg += " (the remote server was unavailable)";
                    }
                    app.toast(ctx, msg);
                }
                Err(e) => {
                    if e.to_string() != "cancelled" {
                        app.ui.status = e.to_string();
                        app.toast(ctx, format!("Denoise failed: {e}"));
                    }
                }
            }
        }
        Err(_) => {
            let (f, stage) = t.progress.lock().map(|g| g.clone()).unwrap_or_default();
            let now = ctx.input(|i| i.time);
            app.ui.toast = Some((format!("{stage} {:.0}%", f * 100.0), now + 0.5));
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// Cancel a running denoise.
pub fn cancel(app: &mut LightcraftApp) {
    if let Some(t) = &app.enhance.task {
        t.cancel.store(true, Ordering::Relaxed);
    }
}

const AMOUNT: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "enhance.amount",
    label: "Amount",
    section: lightcraft_develop::Section::Detail,
    min: 0.0,
    max: 100.0,
    default: 50.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};

/// The dialog body.
pub fn body(app: &mut LightcraftApp, ui: &mut egui::Ui, amount: &mut f64, stack: &mut bool) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let n = app.enhance.ids.len();
    ui.label(format!("AI Denoise of {n} raw photo{}: each is saved as a new DNG next to the original.", if n == 1 { "" } else { "s" }));
    ui.add_space(6.0);
    let out = crate::widgets::slider(ui, &AMOUNT, *amount, true, None);
    if let Some(v) = out.value {
        *amount = v;
    }
    ui.checkbox(stack, "Stack with the original");
    ui.add_space(8.0);
    let st = app.enhance.status.clone().unwrap_or_default();
    let line = |ui: &mut egui::Ui, k: &str, v: String, ok: bool| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(k).color(t.text_dim));
            ui.label(egui::RichText::new(v).color(if ok { t.text } else { t.caution }));
        });
    };
    match st["model"]["name"].as_str() {
        Some(name) => line(ui, "Model", name.to_string(), true),
        None => line(ui, "Model", st["model"]["error"].as_str().unwrap_or("none").to_string(), false),
    }
    let runs_on = if st["remote"]["ok"] == true {
        let same = st["remote"]["sameModel"] != false;
        (
            format!(
                "{} — {}{}",
                st["remote"]["address"].as_str().unwrap_or(""),
                st["remote"]["backend"].as_str().unwrap_or(""),
                if same { "" } else { " (different model: runs here)" }
            ),
            same,
        )
    } else if st["remote"].is_object() {
        (format!("this computer — remote {} unreachable", st["remote"]["address"].as_str().unwrap_or("")), false)
    } else {
        (st["gpu"].as_str().map_or("this computer (CPU)".to_string(), |g| format!("this computer ({g})")), true)
    };
    line(ui, "Runs on", runs_on.0, runs_on.1);
}
