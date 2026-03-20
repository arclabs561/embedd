# embedd

Trait-based embedding interface for Rust with pluggable backends.

Provides `TextEmbedder`, `AsyncTextEmbedder`, `ImageEmbedder`, `AudioEmbedder`, and
extension traits for token-level and sparse embeddings. Backends are feature-gated so
dependents only pull what they need.

## Features

### Sync backends (ureq)

| Feature         | Backend                                   | Requires            |
|-----------------|-------------------------------------------|----------------------|
| `candle-hf`     | Local BERT/JinaBERT/DistilBERT via Candle | CPU (no GPU needed)  |
| `fastembed`     | fastembed dense + sparse (ONNX)           | downloads models     |
| `openai`        | OpenAI-compatible API                     | API key + network    |
| `tei`           | TEI server                                | running TEI instance |
| `hf-inference`  | HF Inference API                          | HF token + network   |

### Async backends (reqwest)

| Feature              | Backend                     |
|----------------------|-----------------------------|
| `async-openai`       | OpenAI-compatible API       |
| `async-tei`          | TEI server                  |
| `async-hf-inference` | HF Inference API            |

### Other

| Feature         | What it does                     |
|-----------------|----------------------------------|
| `serde`         | Serde derives on core types      |
| `ort-tokenizers`| ONNX Runtime (stub)              |
| `burn-backend`  | Burn (stub)                      |
| `siglip`        | SigLIP image (stub)              |

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

### Sparse embeddings

```rust
use embedd::{EmbedMode, SparseEmbedder};
use embedd::fastembed::FastembedSparseEmbedder;

let sparse = FastembedSparseEmbedder::new_default().unwrap();
let vecs = sparse.embed_sparse(&["hello world".into()], EmbedMode::Document).unwrap();
// Each vec is Vec<(term_id, weight)>
```

### Async

```rust
use embedd::{EmbedMode, AsyncTextEmbedder};
use embedd::async_openai::AsyncOpenAiEmbedder;

let embedder = AsyncOpenAiEmbedder::new("sk-...", "text-embedding-3-small");
let vec = embedder.embed_text("hello", EmbedMode::Query).await.unwrap();
```

## Trait overview

- **`TextEmbedder`** -- `embed_texts(&[String], EmbedMode) -> Vec<Vec<f32>>` + single-text
  convenience `embed_text(&str, EmbedMode) -> Vec<f32>`.
- **`AsyncTextEmbedder`** -- async counterpart, object-safe via `BoxFuture`.
- **`SparseEmbedder`** -- sparse lexical embeddings (`(term_id, weight)` pairs).
- **`ImageEmbedder`** -- `embed_images(&[Vec<u8>]) -> Vec<Vec<f32>>`.
- **`TokenEmbedder`** -- multi-vector (late interaction) embeddings.

Wrappers: `PromptedTextEmbedder` (instruction prefix), `L2NormalizedTextEmbedder`,
`TruncateDimTextEmbedder` (matryoshka truncation), `BatchingTextEmbedder` (batch size control).
Compose via `apply_scoping_policy`, `apply_normalization_policy`, `apply_output_dim`.

## Candle architectures

The `candle-hf` backend auto-detects model architecture from `config.json`:

| `model_type`  | Architecture | Notes                          |
|---------------|--------------|--------------------------------|
| `bert`        | BERT         | default fallback               |
| `bert` + ALiBi| JinaBERT     | detected via position_embedding_type |
| `distilbert`  | DistilBERT   | no token_type_ids              |

## Related crates

- [innr](https://crates.io/crates/innr) -- SIMD vector ops, binary quantization, matryoshka truncation
- [vicinity](https://crates.io/crates/vicinity) -- approximate nearest neighbor search
- [rankops](https://crates.io/crates/rankops) -- score fusion, reranking (MaxSim, MMR, DPP)

## License

MIT OR Apache-2.0
