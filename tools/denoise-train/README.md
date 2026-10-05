# lightcraft-denoise-train

Trains LightCraft's AI Denoise network and writes the `.lcdn` model the app loads
(`LIGHTCRAFT_DENOISE_MODEL`, or `denoise.preferences {model}`). Dev tooling only: it is its own
Cargo workspace, not part of the app or `cargo xtask ci`, and uses
[burn](https://burn.dev) (pure Rust, MIT OR Apache-2.0) on wgpu (Metal, Vulkan, DX12) or CUDA.

```bash
cargo run --release -- --selftest                      # training net == app inference?
cargo run --release -- --raws ~/corpus/raws --steps 60000 --batch 16 --crop 128 -o denoise.lcdn
cargo run --release --features cuda -- --raws …        # NVIDIA, with the CUDA toolkit
```

Pipeline check (procedural scenes, Apple M3 GPU, batch 8 × 96 px, 400 steps, ~13 min): held-out
PSNR through the app's inference 34.31 → 37.99 dB. A production model wants real raws and tens of
thousands of steps on a desktop GPU.

## How training pairs are made

Real raw noise is per photosite, before demosaicing, and its variance grows with the signal. So a
pair is made the same way:

1. A **clean image**: a crop of a low-ISO raw from `--raws DIR`, decoded and demosaicked by
   `lightcraft-raw` and downscaled 2× (its own noise halves, detail stays), at a random exposure.
   Without `--raws`, LightCraft's procedural demo scenes stand in: they need no licence at all, but
   they are too smooth to train a production model. Use them to check the pipeline.
2. Mosaic it to RGGB and add **Poisson–Gaussian noise** per photosite: variance `a·x + b`, with
   `a` from 10⁻⁵ to 10⁻¹·⁶ (roughly ISO 100 to 25600) and `b` from 10⁻⁷·⁵ to 10⁻³·⁵.
3. **Demosaic** the clean and the noisy mosaic with LightCraft's own developer (AHD), so the noise
   has exactly the spatial structure the app feeds the network.
4. Prepare the input with `lightcraft_denoise::prepare`, the app's own function, using the noise
   level the app would **estimate** from the noisy image (`NoiseModel::estimate`), not the true
   one. The target is the clean demosaicked image in the same gamma domain, and the network
   learns the residual (L1 loss, Adam, cosine learning-rate decay).

Every 1000 steps the model is written and checked on held-out images **through the app's
inference path** (`lightcraft_denoise::denoise`), reporting PSNR before and after.

## Data and licence

The weights are only as clean, licence-wise, as the data. Use raws you may use for training, and
record where each came from in the corpus folder (it is gitignored, like LightCraft's `corpus/`):

- [raw.pixls.us](https://raw.pixls.us/) publishes its sample raws under **CC0**. It covers
  hundreds of camera models, which is good for generalising across sensors.
- Your own photographs.

Models trained here are released as **CC0-1.0** (the header's `license`), with this recipe
(command line, step count, seed range and the list of training files) so anyone can retrain
them.
