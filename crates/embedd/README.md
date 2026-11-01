# embedd

Embedding interfaces and backends. One `TextEmbedder` trait shared
across local (fastembed, candle) and remote (OpenAI-compatible, TEI, HF
Inference) providers. Backends are feature-gated; the default build is traits
and wrappers only.

```toml
[dependencies]
embedd = { version = "0.4", features = ["fastembed"] }
```

## The trait

```rust,ignore
pub trait TextEmbedder: Send + Sync {
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> Result<Vec<Vec<f32>>>;

    fn embed_text(&self, text: &str, mode: EmbedMode) -> Result<Vec<f32>> {
        // default: single-text convenience wrapper
    }
}
```

Any backend implements `TextEmbedder`. Swap by changing the feature flag and constructor;
nothing else changes.

## Quick start

Local ONNX inference via fastembed:

```rust,no_run
# #[cfg(feature = "fastembed")]
# fn main() -> anyhow::Result<()> {
use embedd::{EmbedMode, TextEmbedder};
use embedd::fastembed::FastembedEmbedder;

let embedder = FastembedEmbedder::new_default()?;
let vec = embedder.embed_text("hello world", EmbedMode::Document)?;
println!("dim={}", vec.len());
# Ok(())
# }
# #[cfg(not(feature = "fastembed"))]
# fn main() {}
```

Remote via OpenAI-compatible API (enable `openai` in the dependency above
and set `OPENAI_API_KEY`):

```rust,no_run
# #[cfg(feature = "openai")]
# fn main() -> anyhow::Result<()> {
use embedd::{EmbedMode, TextEmbedder};
use embedd::openai::OpenAiEmbedder;  // sync

let embedder = OpenAiEmbedder::new(std::env::var("OPENAI_API_KEY")?, "text-embedding-3-small");
let vec = embedder.embed_text("hello world", EmbedMode::Query)?;
# Ok(())
# }
# #[cfg(not(feature = "openai"))]
# fn main() {}
```

Async remote (enable `async-openai` and set `OPENAI_API_KEY`):

```rust,no_run
# #[cfg(feature = "async-openai")]
# async fn run() -> anyhow::Result<()> {
use embedd::{EmbedMode, AsyncTextEmbedder};
use embedd::async_openai::AsyncOpenAiEmbedder;

let embedder = AsyncOpenAiEmbedder::new(std::env::var("OPENAI_API_KEY")?, "text-embedding-3-small");
let vec = embedder.embed_text("hello", EmbedMode::Query).await?;
# Ok(())
# }
# fn main() {}
```

## Backends

### Sync

| Feature | Backend | Notes |
|---|---|---|
| `fastembed` | fastembed dense, sparse, and reranker backends (ONNX) | downloads models on first use |
| `candle-hf` | `LocalHfEmbedder` (BERT, JinaBERT, DistilBERT, XLM-RoBERTa, ModernBERT), `StellaEmbedder` | local weights or HF Hub |
| `ort-tokenizers` | `OrtReranker` cross-encoder via ONNX Runtime | local `model.onnx` + `tokenizer.json` |
| `openai` | OpenAI-compatible API | API key + network |
| `tei` | TEI server | running TEI instance |
| `hf-inference` | HF Inference API | HF token + network |

### Async (reqwest)

| Feature              | Backend               |
|----------------------|-----------------------|
| `async-openai`       | OpenAI-compatible API |
| `async-tei`          | TEI server            |
| `async-hf-inference` | HF Inference API      |

## Traits

- `TextEmbedder`: `embed_texts(&[String], EmbedMode) -> Vec<Vec<f32>>` plus a single-text convenience.
- `AsyncTextEmbedder`: async counterpart, object-safe via `BoxFuture`.
- `SparseEmbedder`: sparse lexical embeddings as `(term_id, weight)` pairs.
- `TokenEmbedder`: multi-vector (late interaction) embeddings.
- `ImageEmbedder` / `AudioEmbedder`: bytes to vectors.
- `Reranker` / `AsyncReranker`: cross-encoder relevance scoring.

For span pooling, `LocalHfEmbedder` also exposes
`embed_tokens_with_offsets` under the `candle-hf` feature. It returns token
vectors plus byte offsets into the original input text. This is backend-specific
because remote embedding services do not all expose tokenizer offsets.

Wrappers: `PromptedTextEmbedder` (instruction prefix), `L2NormalizedTextEmbedder`,
`TruncateDimTextEmbedder` (matryoshka truncation), `BatchingTextEmbedder` (batch
size control), `CachingTextEmbedder`, and `BatchingReranker`.

Run the no-network policy example to see these wrappers compose:

```sh
cargo run -p embedd --example policy_pipeline
```

## Sparse embeddings

```rust,no_run
# #[cfg(feature = "fastembed")]
# fn main() -> anyhow::Result<()> {
use embedd::{EmbedMode, SparseEmbedder};
use embedd::fastembed::FastembedSparseEmbedder;

let sparse = FastembedSparseEmbedder::new_default()?;
let vecs = sparse.embed_sparse(&["hello world".into()], EmbedMode::Document)?;
// Each vec is Vec<(term_id, weight)>
# Ok(())
# }
# #[cfg(not(feature = "fastembed"))]
# fn main() {}
```

## Candle architectures

The `candle-hf` backend auto-detects model architecture from `config.json`:

| `model_type`   | Architecture | Notes                                      |
|----------------|--------------|--------------------------------------------|
| `bert`         | BERT         | default fallback                           |
| `bert` + ALiBi | JinaBERT     | detected via `position_embedding_type`     |
| `distilbert`   | DistilBERT   | no token_type_ids                          |
| `xlm-roberta`  | XLM-RoBERTa  | multilingual                               |
| `modernbert`   | ModernBERT   | RoPE, sliding window attention             |

## Vector acceleration

Vector helpers and normalization use safe scalar Rust by default. The `simd`
feature (also part of `all`) uses [innr](https://crates.io/crates/innr) instead
and requires Rust 1.89. Floating-point results can differ slightly because the
reduction order changes, and another dependency can turn `simd` on through
Cargo feature unification.

## License

MIT OR Apache-2.0
