# AI Denoise on another machine's GPU

AI Denoise (`enhance.denoise`, Photo ▸ Enhance…) runs a small learned network over the
demosaiced raw data in tiles. The tiles can also run on **another computer's GPU**: a desktop with
a big graphics card serves them to a laptop over your LAN or VPN. The laptop keeps the files and
writes the result. Its own GPU and the server **work at the same time**, pulling tiles from one
queue, so a server behind a slow link still adds speed instead of costing it; a server that can't
be reached (or drops out mid-job) hands its tiles back, and the CPU only finishes what the GPUs
leave.

```text
laptop (LightCraft)  ── tiles (f16, compressed) ──▶  desktop (lightcraft-cli denoise-serve)
   its own GPU too   ◀── denoised residuals ───────    GPU: wgpu (Vulkan / DX12 / Metal)
```

The server runs the **same** network code as the app (`crates/denoise` + the WGSL kernels in
`crates/gpu/src/wgsl/nn.wgsl`), so a remote result matches a local one up to f16 rounding on the
wire.

## Server

```bash
export LIGHTCRAFT_DENOISE_TOKEN='a long random string, ≥ 16 characters'
lightcraft-cli denoise-serve --model denoise.lcdn --listen 100.101.102.103:7990
```

- `--listen` takes `address:port` (default `127.0.0.1:7990`). Bind it to the VPN address (for
  example the machine's Tailscale IP) or to `0.0.0.0` behind a firewall rule that only admits the
  VPN. **The protocol is not encrypted**: use it on a LAN or inside a VPN such as Tailscale or
  WireGuard, which encrypt the traffic themselves. Never expose the port to the internet.
- `--token-env VAR` reads the shared token from another variable (default
  `LIGHTCRAFT_DENOISE_TOKEN`); requests without it are refused.
- `--cpu` serves from the CPU; `--max-connections N` caps concurrent clients (default 8).
- It logs one line per client and refuses: a wrong token, another model than its own (by hash),
  oversized requests (> 4 Mi samples per tile, > 4 KiB headers) and a different protocol version.

To keep it running, start it from your system's service manager (launchd, systemd, a Windows
scheduled task at boot) with the token in the service's environment.

## Client (the app)

Either environment variables (they win, so a token never has to sit in a file):

```bash
export LIGHTCRAFT_DENOISE_MODEL=/path/to/denoise.lcdn
export LIGHTCRAFT_DENOISE_URL=100.101.102.103:7990
export LIGHTCRAFT_DENOISE_TOKEN='…the same token…'
```

or the library preferences (`denoise.preferences {model, remote, token, useRemote}`; the token is
stored in the library's `prefs.json` in plain text).

`denoise.status` reports the model in effect, whether the server answers, its backend (e.g.
`GPU: NVIDIA GeForce RTX 4090 (Vulkan)`) and whether it runs the same model. Every
`enhance.denoise` result carries a report: the backends that ran tiles, and why any were given up.

## Protocol (version 1)

One JSON line per message (`\n`-terminated, ≤ 4 KiB), optionally followed by a payload of
`bytes` bytes: the tile's samples as little-endian f16, byte-shuffled (all low bytes, then all
high bytes) and deflated. A payload that inflates past its tile is refused. A connection serves any
number of requests; the client keeps four open to hide latency.

```text
→ {"v":1,"token":"…","op":"hello"}
← {"ok":true,"v":1,"backend":"GPU: …","model":{"name":"…","hash":"…"}}
→ {"v":1,"token":"…","op":"run","model":"<hash>","c":4,"h":H,"w":W,"bytes":N}  + N bytes
← {"ok":true,"c":3,"h":H,"w":W,"bytes":N}                                      + N bytes
← {"ok":false,"error":"…"}                                                     (any failure)
```

## Speed

One 304 × 304 tile (256 px core plus overlap) of the default 3-level U-Net (~113 k multiply-adds
per pixel), measured with `cargo test -p lightcraft-gpu --release --lib denoise_speed -- --ignored --nocapture`:

| Backend | per tile | 24 MP raw |
|---|---:|---:|
| CPU, Apple M3 | 376 ms | ~2 min 18 s |
| GPU, Apple M3 (Metal) | 70 ms | ~26 s |

A whole raw (Nikon D6, 20.8 MP, 330 tiles), Apple M3 laptop and an RTX 3060 Ti server reached
over Tailscale through a **relay** (DERP: no direct path between the two networks):

| Where the tiles ran | time |
|---|---:|
| laptop GPU only | 24.2 s |
| server only, uncompressed tiles | 166 s |
| server only, compressed tiles | 88 s |
| laptop GPU + server together | 20.4 s (server: 77 of 330 tiles) |

Over a relay the link, not the server's GPU, sets the pace; on a LAN or a direct Tailscale path
(`tailscale ping <server>` says "via <ip>" rather than "via DERP") the server takes most of the
tiles. Either way, running both is never slower than the laptop alone.
