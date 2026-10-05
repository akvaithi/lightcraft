//! Enhance commands: `enhance.denoise`, `denoise.status`, `denoise.preferences`.
//!
//! `enhance.denoise` runs synchronously here (CLI, MCP, control channel); the desktop UI plans the
//! same job with [`Session::plan_denoise`] and runs it on a worker thread.

use serde_json::{Value, json};

use super::{CommandSpec, always, bad, bool_or, cmd, f64_or};
use crate::{Result, Session};

fn can_denoise(s: &Session) -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") {
        return Err("Denoise runs in the desktop app".into());
    }
    if s.active().is_none() && s.selection.ids.is_empty() { Err("select the raw photos to denoise".into()) } else { Ok(()) }
}

const PREFS: &str = "denoise.preferences";

fn prefs(s: &mut Session, p: &Value) -> Result<Value> {
    let mut d = s.denoise.clone();
    let text = |k: &str| -> Result<Option<Option<String>>> {
        match p.get(k) {
            None => Ok(None),
            Some(Value::Null) => Ok(Some(None)),
            Some(Value::String(v)) => Ok(Some(Some(v.trim().to_string()).filter(|v| !v.is_empty()))),
            Some(_) => Err(bad(PREFS, format!("`{k}` must be text or null"))),
        }
    };
    if let Some(m) = text("model")? {
        if let Some(path) = &m {
            crate::enhance::load_model(path)?; // refuse a path that isn't a usable model
        }
        d.model = m;
    }
    if let Some(r) = text("remote")? {
        d.remote = r;
    }
    if let Some(t) = text("token")? {
        d.token = t;
    }
    if let Some(u) = p.get("useRemote").and_then(Value::as_bool) {
        d.use_remote = Some(u);
    }
    if d != s.denoise {
        s.denoise = d;
        s.save_prefs()?;
    }
    Ok(
        json!({"model": s.denoise.model, "remote": s.denoise.remote, "useRemote": s.denoise.use_remote.unwrap_or(true), "tokenSet": s.denoise.token.is_some()}),
    )
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "enhance.denoise",
            "Denoise…",
            ["Photo", "Enhance"],
            Some("Cmd+Alt+I"),
            "{ids?, amount=50 (0..100), stack=true} — AI Denoise of raw photos: writes <name>-Enhanced-NR.dng next to each, imports it with the original's edits (manual noise reduction off) and stacks it on the original. Runs on the remote GPU server when configured, else this machine → {photos: [{id, source, path, width, height, report: {tiles, backends, fallbacks, noise}}]}",
            can_denoise,
            |s, p| {
                let ids = s.targets(p);
                s.denoise_now(&ids, f64_or(p, "amount", 50.0), bool_or(p, "stack", true))
            }
        ),
        cmd!(
            "denoise.status",
            "Denoise Status",
            [],
            None,
            "{} — the denoise model in effect, the remote server (reachable? same model?) and the GPU → {model, remote, gpu, tokenSet}",
            always,
            |s, _| Ok(s.denoise_status())
        ),
        cmd!(
            "denoise.preferences",
            "Denoise Preferences",
            [],
            None,
            "{model?: path to a .lcdn model | null, remote?: \"host[:port]\" (default port 7990) | null, token?: text | null, useRemote?: bool} — saved with the library; LIGHTCRAFT_DENOISE_MODEL, LIGHTCRAFT_DENOISE_URL and LIGHTCRAFT_DENOISE_TOKEN override them → {model, remote, useRemote, tokenSet}",
            always,
            prefs
        ),
    ]
}
