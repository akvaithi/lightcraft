//! Enhance → Denoise end to end: a raw file in, an Enhanced-NR DNG out (imported, edited like the
//! original, stacked), on the CPU and through a remote server.

use std::sync::Arc;

use serde_json::json;

use crate::Session;
use crate::tests_xmp::{synthetic_dng_with, temp_dir};

/// A small model file (random weights: the test checks plumbing, not image quality).
fn write_model(dir: &std::path::Path) -> String {
    let (h, n) = lightcraft_denoise::model::unet_header("test-unet", "CC0", 4, 3);
    let w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.618).fract() - 0.5) * 0.01).collect();
    let path = dir.join("test.lcdn");
    std::fs::write(&path, lightcraft_denoise::Model::to_bytes(&h, &w).unwrap()).unwrap();
    path.to_string_lossy().to_string()
}

fn session_with_raw(tag: &str) -> (Session, lightcraft_catalog::PhotoId, std::path::PathBuf) {
    let dir = temp_dir(tag);
    let src = dir.join("Noisy.dng");
    std::fs::write(&src, synthetic_dng_with(None, Default::default())).unwrap();
    let mut s = Session::new().with_fs();
    s.execute("library.import", &json!({"paths": [src.to_string_lossy()]})).unwrap();
    let id = s.catalog.photos().next().unwrap().id;
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    (s, id, dir)
}

#[test]
fn denoise_writes_imports_and_stacks_an_enhanced_dng() {
    let (mut s, id, dir) = session_with_raw("denoise");
    // no model yet: a clear error, nothing written
    if std::env::var("LIGHTCRAFT_DENOISE_MODEL").is_err() {
        let e = s.execute("enhance.denoise", &json!({})).unwrap_err().to_string();
        assert!(e.contains("no denoise model"), "{e}");
    }
    let model = write_model(&dir);
    s.execute("denoise.preferences", &json!({"model": model})).unwrap();
    assert!(s.execute("denoise.preferences", &json!({"model": "/nope.lcdn"})).is_err(), "a bad path is refused");
    s.execute("develop.set", &json!({"values": {"light.exposure": 0.4, "detail.nrLuminance": 30}})).unwrap();

    let r = s.execute("enhance.denoise", &json!({"amount": 70})).unwrap();
    let p = &r["photos"][0];
    let path = p["path"].as_str().unwrap();
    assert!(path.ends_with("Noisy-Enhanced-NR.dng"), "{path}");
    assert!(std::path::Path::new(path).exists());
    let local =
        |b: &serde_json::Value| b.as_array().is_some_and(|a| a.len() == 1 && a[0].as_str().is_some_and(|n| n == "CPU" || n.starts_with("GPU")));
    assert!(local(&p["report"]["backends"]), "{}", p["report"]);
    let new = lightcraft_catalog::PhotoId(p["id"].as_u64().unwrap());
    let d = s.develop_of(new).unwrap();
    assert_eq!(d.light.exposure, 0.4, "the original's edits carry over");
    assert_eq!(d.detail.nr_luminance, 0.0, "manual NR is off on the denoised copy");
    assert!((d.enhance.denoise - 70.0).abs() < 1e-6);
    let st = s.catalog.stack_of(new).expect("stacked with the original");
    assert!(st.photos.contains(&id) && st.photos[0] == new, "the denoised copy tops the stack");
    // a second run doesn't overwrite the first
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    let r2 = s.execute("enhance.denoise", &json!({"stack": false})).unwrap();
    assert!(r2["photos"][0]["path"].as_str().unwrap().ends_with("Noisy-Enhanced-NR-2.dng"));
}

#[test]
fn denoise_runs_on_a_remote_server_and_falls_back_when_it_is_gone() {
    let (mut s, _, dir) = session_with_raw("denoise-remote");
    let model = write_model(&dir);
    let m = crate::enhance::load_model(&model).unwrap();
    let token = "test-token-0123456789";
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let opts = lightcraft_denoise::remote::ServerOptions { token: token.into(), ..Default::default() };
    std::thread::spawn(move || lightcraft_denoise::remote::serve(l, m, Arc::new(lightcraft_denoise::Cpu), opts, Arc::new(|_| {})));
    s.execute("denoise.preferences", &json!({"model": model, "remote": addr, "token": token})).unwrap();
    if std::env::var("LIGHTCRAFT_DENOISE_URL").is_ok() {
        return; // the environment points elsewhere
    }
    let st = s.execute("denoise.status", &json!({})).unwrap();
    assert_eq!(st["remote"]["ok"], true, "{st}");
    assert_eq!(st["remote"]["sameModel"], true);
    let r = s.execute("enhance.denoise", &json!({"stack": false})).unwrap();
    let backends = r["photos"][0]["report"]["backends"].as_array().unwrap().clone();
    assert!(backends[0].as_str().unwrap().starts_with("remote"), "{backends:?}");
    // a dead server: this machine takes over and the report says why
    s.execute("denoise.preferences", &json!({"remote": "127.0.0.1:1"})).unwrap();
    let id = s.catalog.photos().next().unwrap().id;
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    let r = s.execute("enhance.denoise", &json!({"stack": false})).unwrap();
    let rep = &r["photos"][0]["report"];
    let local = rep["backends"].as_array().is_some_and(|a| a.len() == 1 && a[0].as_str().is_some_and(|n| n == "CPU" || n.starts_with("GPU")));
    assert!(local, "{rep}");
    assert!(rep["fallbacks"][0].as_str().unwrap().contains("remote 127.0.0.1:1"), "{rep}");
    assert_eq!(s.execute("denoise.status", &json!({})).unwrap()["remote"]["ok"], false);
}
