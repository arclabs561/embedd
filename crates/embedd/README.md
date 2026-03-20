# embedd

Trait-based embedding interface for Rust with pluggable backends.

Provides `TextEmbedder`, `ImageEmbedder`, `AudioEmbedder`, and extension traits for
token-level and sparse embeddings. Backends are feature-gated so dependents only pull
what they need.

## Features

| Feature         | Backend                        | Requires            |
|-----------------|--------------------------------|----------------------|
| `candle-hf`     | Local BERT via Candle + HF Hub | CPU (no GPU needed)  |
| `fastembed`     | fastembed (ONNX)               | downloads models     |
| `openai`        | OpenAI-compatible API          | API key + network    |
| `tei`           | TEI server                     | running TEI instance |
| `hf-inference`  | HF Inference API               | HF token + network   |
| `ort-tokenizers` | ONNX Runtime (stub)           | --                   |
| `burn-backend`  | Burn (stub)                    | --                   |
| `siglip`        | SigLIP image (stub)            | --                   |
| `serde`         | Serde derives on core types    | --                   |

## Quick start

```toml
[dependencies]
embedd = { version = "0.1", features = ["fastembed"] }
```

```rust
use embedd::{EmbedMode, TextEmbedder};
use embedd::fastembed::FastembedEmbedder;

let embedder = FastembedEmbedder::new_default().unwrap();
let vec = embedder.embed_text("hello world", EmbedMode::Document).unwrap();
println!("dim={}", vec.len());
```

## Trait overview

- **`TextEmbedder`** -- `embed_texts(&[String], EmbedMode) -> Vec<Vec<f32>>` + single-text
  convenience `embed_text(&str, EmbedMode) -> Vec<f32>`.
- **`ImageEmbedder`** -- `embed_images(&[Vec<u8>]) -> Vec<Vec<f32>>`.
- **`TokenEmbedder`** -- multi-vector (late interaction) embeddings.
- **`SparseEmbedder`** -- sparse lexical embeddings (`(term_id, weight)` pairs).

Wrappers: `PromptedTextEmbedder` (instruction prefix), `L2NormalizedTextEmbedder`,
`TruncateDimTextEmbedder` (matryoshka truncation). Compose via `apply_scoping_policy`,
`apply_normalization_policy`, `apply_output_dim`.

## Related crates

- [innr](https://crates.io/crates/innr) -- SIMD vector ops, binary quantization, matryoshka truncation
- [vicinity](https://crates.io/crates/vicinity) -- approximate nearest neighbor search
- [rankops](https://crates.io/crates/rankops) -- score fusion, reranking (MaxSim, MMR, DPP)

## License

MIT OR Apache-2.0
