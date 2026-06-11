# Changelog

## [0.2.2] - 2026-06-10

### Fixed

- ort >= 2.0.0-rc.12 compatibility: session-builder errors became `Error<SessionBuilder>` (not `Send + Sync`), breaking `?`-conversion into `anyhow`. The two builder chains in the `ort-tokenizers` backend now map errors explicitly; compiles against rc.11 and rc.12.

### Changed

- Real README (usage from `examples/hello_embed.rs`, backends/features table); innr 0.4.

Earlier releases predate this changelog; see git history.
