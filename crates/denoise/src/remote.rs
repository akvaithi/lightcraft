//! Remote denoise: run tiles on another machine's GPU over TCP (meant for a LAN or a VPN such as
//! Tailscale/WireGuard: the connection itself is not encrypted).
//!
//! Protocol (version 1). Each message is one JSON line (`\n`-terminated, ≤ 4 KiB), optionally
//! followed by a binary payload of `bytes` bytes: the tile's c·h·w samples as little-endian f16,
//! byte-shuffled (every sample's low byte, then every high byte) and deflated. Shuffling puts the
//! slowly varying high bytes together, which roughly halves what crosses a slow link.
//!
//! ```text
//! → {"v":1,"token":"…","op":"hello"}
//! ← {"ok":true,"v":1,"backend":"GPU: …","model":{"name":"…","hash":"…"}}
//! → {"v":1,"token":"…","op":"run","model":"<hash>","c":4,"h":H,"w":W,"bytes":N}  + N bytes
//! ← {"ok":true,"c":3,"h":H,"w":W,"bytes":N}                                      + N bytes
//! ← {"ok":false,"error":"…"}                                                     (any failure)
//! ```
//!
//! A connection serves any number of requests. The server refuses a wrong token, another model
//! than its own (by hash), oversized headers or tiles, and caps concurrent connections.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

use crate::{Backend, Error, Model, Result, Tensor};

pub const PROTOCOL: u64 = 1;
const MAX_LINE: usize = 4096;
/// Largest tile accepted, in samples (4 channels of 1024²).
pub const MAX_SAMPLES: usize = 4 << 20;
pub const DEFAULT_PORT: u16 = 7990;

/// The model hash as written on the wire.
pub fn hash_hex(h: u64) -> String {
    format!("{h:016x}")
}

fn read_line(r: &mut impl BufRead) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let n = r.by_ref().take(MAX_LINE as u64 + 1).read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(None);
    }
    if buf.last() != Some(&b'\n') {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "message header too long or truncated"));
    }
    String::from_utf8(buf).map(Some).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "header is not UTF-8"))
}

/// Samples → f16 → byte-shuffled → deflated (see the protocol above).
pub fn pack(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    let mut b = vec![0u8; n * 2];
    let (lo, hi) = b.split_at_mut(n);
    for ((v, l), h) in data.iter().zip(lo.iter_mut()).zip(hi.iter_mut()) {
        [*l, *h] = half::f16::from_f32(*v).to_le_bytes();
    }
    miniz_oxide::deflate::compress_to_vec(&b, 1)
}

/// The inverse of [`pack`] for exactly `n` samples (refuses anything that inflates to more).
pub fn unpack(bytes: &[u8], n: usize) -> std::io::Result<Vec<f32>> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let b = miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, n * 2).map_err(|_| bad("payload doesn't inflate to the tile size"))?;
    if b.len() != n * 2 {
        return Err(bad("payload size doesn't match the tile"));
    }
    let (lo, hi) = b.split_at(n);
    Ok(lo.iter().zip(hi).map(|(l, h)| half::f16::from_le_bytes([*l, *h]).to_f32()).collect())
}

fn write_payload(w: &mut impl Write, header: Value, data: &[f32]) -> std::io::Result<()> {
    let packed = pack(data);
    let mut header = header;
    header["bytes"] = json!(packed.len());
    writeln!(w, "{header}")?;
    w.write_all(&packed)
}

/// Read a payload of `bytes` bytes holding `n` samples (`bytes` is capped: a hostile header can't
/// make us allocate more than an uncompressed tile, plus slack).
fn read_payload(r: &mut impl Read, bytes: usize, n: usize) -> std::io::Result<Vec<f32>> {
    if bytes > n * 2 + 4096 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "payload larger than its tile"));
    }
    let mut b = vec![0u8; bytes];
    r.read_exact(&mut b)?;
    unpack(&b, n)
}

fn payload_bytes(v: &Value) -> Option<usize> {
    v.get("bytes").and_then(Value::as_u64).map(|b| b as usize)
}

fn shape(v: &Value) -> Option<(usize, usize, usize)> {
    let g = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
    let (c, h, w) = (g("c")?, g("h")?, g("w")?);
    let n = c.checked_mul(h)?.checked_mul(w)?;
    (n > 0 && n <= MAX_SAMPLES).then_some((c, h, w))
}

/// Constant-time comparison (the token must not leak through timing).
fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut d = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        d |= (a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)) as usize;
    }
    d == 0
}

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerOptions {
    /// Shared secret every request must carry (empty refuses everything).
    pub token: String,
    pub max_connections: usize,
    /// Idle and transfer timeout per connection.
    pub timeout: Duration,
}

impl Default for ServerOptions {
    fn default() -> Self {
        ServerOptions { token: String::new(), max_connections: 8, timeout: Duration::from_secs(120) }
    }
}

/// Serve `model` on `backend` until the listener fails. `log` gets one line per event.
pub fn serve(
    listener: TcpListener,
    model: Arc<Model>,
    backend: Arc<dyn Backend + Send>,
    opts: ServerOptions,
    log: Arc<dyn Fn(&str) + Send + Sync>,
) -> Result<()> {
    if opts.token.trim().len() < 16 {
        return Err(Error::Backend { backend: "server".into(), reason: "the token must be at least 16 characters".into() });
    }
    let active = Arc::new(AtomicUsize::new(0));
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".into());
        if active.load(Ordering::SeqCst) >= opts.max_connections.max(1) {
            let mut s = stream;
            let _ = writeln!(s, "{}", json!({"ok": false, "error": "server busy"}));
            log(&format!("{peer}: refused (busy)"));
            continue;
        }
        active.fetch_add(1, Ordering::SeqCst);
        let (m, b, o, l, n) = (model.clone(), backend.clone(), opts.clone(), log.clone(), active.clone());
        let spawned = std::thread::Builder::new().name(format!("denoise {peer}")).spawn(move || {
            if let Err(e) = handle(stream, &m, b.as_ref(), &o, &*l) {
                l(&format!("{peer}: {e}"));
            }
            n.fetch_sub(1, Ordering::SeqCst);
        });
        if spawned.is_err() {
            active.fetch_sub(1, Ordering::SeqCst);
            log("could not start a connection thread");
        }
    }
    Ok(())
}

fn handle(stream: TcpStream, model: &Model, backend: &dyn Backend, opts: &ServerOptions, log: &dyn Fn(&str)) -> std::io::Result<()> {
    stream.set_read_timeout(Some(opts.timeout))?;
    stream.set_write_timeout(Some(opts.timeout))?;
    stream.set_nodelay(true)?;
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let mut r = BufReader::new(stream.try_clone()?);
    let mut w = stream;
    let hash = hash_hex(model.hash);
    let mut tiles = 0usize;
    while let Some(line) = read_line(&mut r)? {
        let fail = |w: &mut TcpStream, msg: &str| writeln!(w, "{}", json!({"ok": false, "error": msg}));
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            fail(&mut w, "malformed request")?;
            break;
        };
        if req.get("v").and_then(Value::as_u64) != Some(PROTOCOL) {
            fail(&mut w, &format!("protocol version {PROTOCOL} only"))?;
            break;
        }
        if !same(req.get("token").and_then(Value::as_str).unwrap_or(""), &opts.token) {
            fail(&mut w, "bad token")?;
            log(&format!("{peer}: bad token"));
            break;
        }
        match req.get("op").and_then(Value::as_str) {
            Some("hello") => {
                writeln!(w, "{}", json!({"ok": true, "v": PROTOCOL, "backend": backend.name(), "model": {"name": model.header.name, "hash": hash}}))?;
            }
            Some("run") => {
                let (Some((c, h, wd)), Some(bytes)) = (shape(&req), payload_bytes(&req)) else {
                    fail(&mut w, "bad or oversized tile shape")?;
                    break;
                };
                // the payload is read before any refusal, so the stream stays in step
                let data = match read_payload(&mut r, bytes, c * h * wd) {
                    Ok(d) => d,
                    Err(e) => {
                        fail(&mut w, &e.to_string())?;
                        break;
                    }
                };
                if req.get("model").and_then(Value::as_str) != Some(hash.as_str()) {
                    fail(&mut w, &format!("model mismatch: this server runs {} ({hash})", model.header.name))?;
                    continue;
                }
                match backend.run(model, &Tensor { c, h, w: wd, data }) {
                    Ok(out) => {
                        write_payload(&mut w, json!({"ok": true, "c": out.c, "h": out.h, "w": out.w}), &out.data)?;
                        tiles += 1;
                    }
                    Err(e) => fail(&mut w, &e.to_string())?,
                }
            }
            _ => {
                fail(&mut w, "unknown op")?;
                break;
            }
        }
        w.flush()?;
    }
    if tiles > 0 {
        log(&format!("{peer}: {tiles} tiles"));
    }
    Ok(())
}

/// What a server reported in its greeting.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub address: String,
    pub backend: String,
    pub model_name: String,
    pub model_hash: String,
}

/// A remote server as a [`Backend`]: keeps up to `concurrency` connections open.
pub struct Remote {
    pub address: String,
    token: String,
    concurrency: usize,
    timeout: Duration,
    pool: Mutex<Vec<(BufReader<TcpStream>, TcpStream)>>,
    backend_name: Mutex<Option<String>>,
}

impl Remote {
    /// `address` as `host:port` (the port defaults to [`DEFAULT_PORT`]).
    pub fn new(address: &str, token: &str) -> Remote {
        let address = if address.rsplit_once(':').is_some_and(|(_, p)| p.parse::<u16>().is_ok()) && !address.ends_with(']') {
            address.to_string()
        } else {
            format!("{address}:{DEFAULT_PORT}")
        };
        Remote {
            address,
            token: token.into(),
            concurrency: 4,
            timeout: Duration::from_secs(60),
            pool: Mutex::new(Vec::new()),
            backend_name: Mutex::new(None),
        }
    }

    pub fn with_concurrency(mut self, n: usize) -> Remote {
        self.concurrency = n.clamp(1, 16);
        self
    }

    fn err(&self, reason: impl std::fmt::Display) -> Error {
        Error::Backend { backend: format!("remote {}", self.address), reason: reason.to_string() }
    }

    fn connect(&self) -> Result<(BufReader<TcpStream>, TcpStream)> {
        let addr = self.address.to_socket_addrs().map_err(|e| self.err(e))?.next().ok_or_else(|| self.err("address doesn't resolve"))?;
        let s = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| self.err(e))?;
        s.set_read_timeout(Some(self.timeout)).map_err(|e| self.err(e))?;
        s.set_write_timeout(Some(self.timeout)).map_err(|e| self.err(e))?;
        s.set_nodelay(true).map_err(|e| self.err(e))?;
        let r = BufReader::new(s.try_clone().map_err(|e| self.err(e))?);
        Ok((r, s))
    }

    fn take(&self) -> Result<(BufReader<TcpStream>, TcpStream)> {
        let pooled = self.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop();
        match pooled {
            Some(c) => Ok(c),
            None => self.connect(),
        }
    }

    fn put(&self, c: (BufReader<TcpStream>, TcpStream)) {
        self.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(c);
    }

    fn reply(&self, r: &mut BufReader<TcpStream>) -> Result<Value> {
        let line = read_line(r).map_err(|e| self.err(e))?.ok_or_else(|| self.err("connection closed"))?;
        let v: Value = serde_json::from_str(&line).map_err(|_| self.err("malformed reply"))?;
        if v.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(self.err(v.get("error").and_then(Value::as_str).unwrap_or("request failed")));
        }
        Ok(v)
    }

    /// Greet the server: its backend and model.
    pub fn hello(&self) -> Result<ServerInfo> {
        let (mut r, mut s) = self.connect()?;
        writeln!(s, "{}", json!({"v": PROTOCOL, "token": self.token, "op": "hello"})).map_err(|e| self.err(e))?;
        let v = self.reply(&mut r)?;
        let m = v.get("model").cloned().unwrap_or_default();
        let info = ServerInfo {
            address: self.address.clone(),
            backend: v.get("backend").and_then(Value::as_str).unwrap_or("?").into(),
            model_name: m.get("name").and_then(Value::as_str).unwrap_or("?").into(),
            model_hash: m.get("hash").and_then(Value::as_str).unwrap_or("").into(),
        };
        *self.backend_name.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(info.backend.clone());
        self.put((r, s));
        Ok(info)
    }
}

impl Backend for Remote {
    fn name(&self) -> String {
        match &*self.backend_name.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
            Some(b) => format!("remote {} ({b})", self.address),
            None => format!("remote {}", self.address),
        }
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn run(&self, model: &Model, x: &Tensor) -> Result<Tensor> {
        if x.data.len() > MAX_SAMPLES {
            return Err(self.err("tile too large for the protocol"));
        }
        let (mut r, mut s) = self.take()?;
        let req = json!({"v": PROTOCOL, "token": self.token, "op": "run", "model": hash_hex(model.hash), "c": x.c, "h": x.h, "w": x.w});
        write_payload(&mut s, req, &x.data).map_err(|e| self.err(e))?;
        s.flush().map_err(|e| self.err(e))?;
        let v = self.reply(&mut r)?;
        let (c, h, w) = shape(&v).ok_or_else(|| self.err("bad tile shape in reply"))?;
        let bytes = payload_bytes(&v).ok_or_else(|| self.err("reply without a payload size"))?;
        let data = read_payload(&mut r, bytes, c * h * w).map_err(|e| self.err(e))?;
        self.put((r, s));
        Ok(Tensor { c, h, w, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::unet_header;
    use crate::{Cpu, NoiseModel, Tiling, denoise};

    fn model(seed: f32) -> Arc<Model> {
        let (h, n) = unet_header("test", "CC0", 4, 3);
        let w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.618 + seed).fract() - 0.5) * 0.02).collect();
        Arc::new(Model::from_bytes(&Model::to_bytes(&h, &w).unwrap()).unwrap())
    }

    fn start(m: Arc<Model>, token: &str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let opts = ServerOptions { token: token.into(), ..Default::default() };
        std::thread::spawn(move || serve(l, m, Arc::new(Cpu), opts, Arc::new(|_| {})));
        addr
    }

    const TOKEN: &str = "0123456789abcdef-test";

    #[test]
    fn remote_tiles_match_local_ones() {
        let m = model(0.3);
        let addr = start(m.clone(), TOKEN);
        let remote = Remote::new(&addr, TOKEN).with_concurrency(3);
        let info = remote.hello().unwrap();
        assert_eq!(info.model_hash, hash_hex(m.hash));
        assert!(remote.name().contains("CPU"));
        let img = lightcraft_raster::Rgb32f::from_fn(60, 44, |x, y| [x as f32 / 60.0, y as f32 / 44.0, 0.2]);
        let nm = NoiseModel { a: [1e-3; 3], b: [1e-5; 3] };
        let t = Tiling { core: 16, overlap: 8 };
        let (a, ra) = denoise(&img, &nm, &m, &[&remote, &Cpu], 1.0, t, &|_| true).unwrap();
        let (b, _) = denoise(&img, &nm, &m, &[&Cpu], 1.0, t, &|_| true).unwrap();
        assert!(ra.fallbacks.is_empty(), "{:?}", ra.fallbacks);
        assert!(ra.backends[0].starts_with("remote"));
        // f16 on the wire: close, not identical
        let worst = a.data.iter().zip(&b.data).flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs())).fold(0.0f32, f32::max);
        assert!(worst < 3e-3, "{worst}");
    }

    #[test]
    fn refusals_fall_back_with_the_reason() {
        let m = model(0.3);
        let addr = start(m.clone(), TOKEN);
        let img = lightcraft_raster::Rgb32f::from_fn(20, 20, |_, _| [0.2; 3]);
        let nm = NoiseModel { a: [1e-3; 3], b: [1e-5; 3] };
        // wrong token
        let bad = Remote::new(&addr, "wrong-token-000000");
        assert!(bad.hello().unwrap_err().to_string().contains("bad token"));
        let (_, rep) = denoise(&img, &nm, &m, &[&bad, &Cpu], 1.0, Tiling::default(), &|_| true).unwrap();
        assert!(rep.fallbacks[0].contains("bad token") && rep.backends == ["CPU"]);
        // another model than the server's
        let other = model(0.7);
        let r = Remote::new(&addr, TOKEN);
        let e = r.run(&other, &Tensor::zeros(4, 8, 8)).unwrap_err();
        assert!(e.to_string().contains("model mismatch"), "{e}");
        // the connection stays usable after a refusal
        assert!(r.run(&m, &Tensor::zeros(4, 8, 8)).is_ok());
        // nobody listening
        let dead = Remote::new("127.0.0.1:1", TOKEN);
        assert!(dead.hello().is_err());
        // a short token is refused at startup
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(serve(l, m, Arc::new(Cpu), ServerOptions { token: "short".into(), ..Default::default() }, Arc::new(|_| {})).is_err());
    }

    #[test]
    fn hostile_requests_are_refused_without_harm() {
        let m = model(0.3);
        let addr = start(m.clone(), TOKEN);
        let talk = |bytes: &[u8]| -> String {
            let mut s = TcpStream::connect(&addr).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            s.write_all(bytes).unwrap();
            let mut r = BufReader::new(s);
            let mut line = String::new();
            let _ = r.read_line(&mut line);
            line
        };
        assert!(talk(b"not json\n").contains("malformed"));
        assert!(talk(&[b'x'; 10_000]).is_empty() || talk(&[b'x'; 10_000]).contains("false"));
        let huge = format!("{}\n", json!({"v": 1, "token": TOKEN, "op": "run", "model": "x", "c": 4, "h": 100_000, "w": 100_000}));
        assert!(talk(huge.as_bytes()).contains("oversized"));
        let v2 = format!("{}\n", json!({"v": 2, "token": TOKEN, "op": "hello"}));
        assert!(talk(v2.as_bytes()).contains("protocol"));
        // the server still works
        assert!(Remote::new(&addr, TOKEN).hello().is_ok());
        assert!(same("abc", "abc") && !same("abc", "abd") && !same("abc", "abcd"));
        // a payload that inflates past its tile is refused
        let bomb = miniz_oxide::deflate::compress_to_vec(&vec![0u8; 1 << 20], 9);
        let hdr = format!("{}\n", json!({"v": 1, "token": TOKEN, "op": "run", "model": "x", "c": 1, "h": 8, "w": 8, "bytes": bomb.len()}));
        assert!(talk(&[hdr.as_bytes(), &bomb].concat()).contains("false"));
        assert!(Remote::new(&addr, TOKEN).hello().is_ok());
    }

    #[test]
    fn packing_round_trips_and_shrinks_smooth_data() {
        let noisy: Vec<f32> = (0..4096).map(|i| 0.5 + ((i * 7919 % 101) as f32 - 50.0) * 0.002).collect();
        let back = unpack(&pack(&noisy), noisy.len()).unwrap();
        assert!(noisy.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-3));
        let smooth: Vec<f32> = (0..4096).map(|i| 0.01 + i as f32 * 1e-6).collect();
        assert!(pack(&smooth).len() < smooth.len() * 2 / 4, "smooth planes compress well");
        assert!(unpack(&pack(&noisy), noisy.len() + 1).is_err());
    }
}
