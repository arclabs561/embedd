# embedd

Embedding interfaces + local backends (Candle/HF).

This repo’s publishable crate lives at `crates/embedd/` (package name: `embedd`).

## What it is

`embedd` is the embedding substrate intended to:
- centralize model loading/caching
- keep downstream crates backend-agnostic
- expose a small set of stable embedding calls

## Crate docs

- Library + feature flags: `crates/embedd/README.md`

## License

MIT OR Apache-2.0

