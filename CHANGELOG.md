# Changelog

## [0.4.0] - 2026-06-27

### Fixed

- Candle backend: pool per the model's convention instead of always mean-pooling. BGE-family models pool the CLS token; mean-pooling them silently degraded retrieval (BEIR SciFact bge-base nDCG@10 0.474 vs 0.731 with correct pooling). Pooling is now read from the model's `1_Pooling/config.json` (CLS when `pooling_mode_cls_token`, else mean). This changes the embeddings BGE models produce through the Candle backend, so re-embed BGE stores built with 0.3.

## [0.3.0] - 2026-06-26

### Added

- Compute-device acceleration for the Candle backend: new `metal`, `cuda`, and `accelerate` features. The backend auto-selects the best available device via `pick_device` (Metal on Apple Silicon, CUDA on NVIDIA, else CPU), falling back to CPU on any accelerator init failure so a missing GPU never breaks embedding. Measured 6.2x faster than CPU for BGE-small on Apple Silicon (512 texts: 0.75s vs 4.63s; `examples/metal_bench.rs`). Default build is unchanged.

## [0.2.2] - 2026-06-10

### Fixed

- ort >= 2.0.0-rc.12 compatibility: session-builder errors became `Error<SessionBuilder>` (not `Send + Sync`), breaking `?`-conversion into `anyhow`. The two builder chains in the `ort-tokenizers` backend now map errors explicitly; compiles against rc.11 and rc.12.

### Changed

- Real README (usage from `examples/hello_embed.rs`, backends/features table); innr 0.4.

Earlier releases predate this changelog; see git history.
