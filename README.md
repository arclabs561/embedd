# embedd

[![crates.io](https://img.shields.io/crates/v/embedd.svg)](https://crates.io/crates/embedd)
[![Documentation](https://docs.rs/embedd/badge.svg)](https://docs.rs/embedd)
[![CI](https://github.com/arclabs561/embedd/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/embedd/actions/workflows/ci.yml)

Embedding interfaces and local backends. One `TextEmbedder` trait shared
across local (fastembed, candle) and remote (OpenAI-compatible, TEI, HF
Inference) providers. Backends are feature-gated; the default build is
traits and wrappers only.

```toml
[dependencies]
embedd = { version = "0.2", features = ["fastembed"] }
```

## Quick start

Local ONNX inference via fastembed (downloads the model on first use):

```rust
use embedd::fastembed::FastembedEmbedder;
use embedd::{EmbedMode, TextEmbedder};

let embedder = FastembedEmbedder::new_default()?;
let a = embedder.embed_text("the cat sat on the mat", EmbedMode::Document)?;
let b = embedder.embed_text("a dog lay on the rug", EmbedMode::Document)?;
println!("cosine similarity: {:.4}", embedd::vector::cosine_f32(&a, &b));
```

Same trait, remote backend:

```rust
use embedd::openai::OpenAiEmbedder;
use embedd::{EmbedMode, TextEmbedder};

let embedder = OpenAiEmbedder::new("sk-...", "text-embedding-3-small");
let vec = embedder.embed_text("hello world", EmbedMode::Query)?;
```

All trait methods return `anyhow::Result`. Swapping backends changes the
constructor and the feature flag; nothing else changes.

## Traits

- `TextEmbedder`: `embed_texts(&[String], EmbedMode) -> Vec<Vec<f32>>`, plus a
  single-text convenience, `model_id()`, `dimension()`, and `capabilities()`
  (declares normalization, truncation, and where prompts are applied, so
  callers can detect double-prompting and normalization drift).
- `AsyncTextEmbedder`: async counterpart, object-safe via boxed futures.
- `SparseEmbedder`: sparse lexical vectors as `(term_id, weight)` pairs.
- `TokenEmbedder`: multi-vector (late interaction) embeddings.
- `ImageEmbedder` / `AudioEmbedder`: bytes to vectors.
- `Reranker` / `AsyncReranker`: cross-encoder relevance scoring.

Wrappers compose over any implementation: `PromptedTextEmbedder` (instruction
prefix), `L2NormalizedTextEmbedder`, `TruncateDimTextEmbedder` (matryoshka
truncation), `BatchingTextEmbedder`, `CachingTextEmbedder`, `BatchingReranker`.

## Backends

| Feature | Provides | Needs |
|---------|----------|-------|
| `fastembed` | `FastembedEmbedder`, `FastembedSparseEmbedder`, `FastembedReranker` (ONNX) | model download on first use |
| `candle-hf` | `LocalHfEmbedder` (BERT, JinaBERT, DistilBERT, XLM-RoBERTa, ModernBERT), `StellaEmbedder`; CPU inference | local weights or HF Hub |
| `ort-tokenizers` | `OrtReranker` cross-encoder via ONNX Runtime | local `model.onnx` + `tokenizer.json` |
| `openai` | `OpenAiEmbedder` for any `/v1/embeddings` API (sync) | API key + network |
| `tei` | `TeiEmbedder` for a text-embeddings-inference server (sync) | running TEI instance |
| `hf-inference` | `HfInferenceEmbedder` (text, image, audio; sync) | HF token + network |
| `async-openai`, `async-tei`, `async-hf-inference` | reqwest/tokio variants of the sync clients | as above |
| `qdrant` | `embed_and_upsert` / `embed_and_search` against Qdrant | running Qdrant instance |

Also: `serde` (derives on config types), `cli` (minimal `embedd` binary for
local validation), `all` (everything, for local dev).

`candle-hf` auto-detects the architecture from the model's `config.json`;
see [crates/embedd/README.md](crates/embedd/README.md) for the detection table.

## Sparse embeddings

```rust
use embedd::fastembed::FastembedSparseEmbedder;
use embedd::{EmbedMode, SparseEmbedder};

let sparse = FastembedSparseEmbedder::new_default()?;
let vecs = sparse.embed_sparse(&["hello world".into()], EmbedMode::Document)?;
// each entry: Vec<(term_id, weight)>
```

## Examples

```sh
cargo run -p embedd --example policy_pipeline
cargo run -p embedd --example hello_embed --features fastembed
```

Also `semantic_search`, `sparse_retrieval`, `batched_embed` (all `fastembed`),
`backend_matrix`, `backend_compare`, and `rerank_ort` (`ort-tokenizers`).

## Related

- [innr](https://crates.io/crates/innr): SIMD vector ops backing the `vector` module
- [vicinity](https://crates.io/crates/vicinity): approximate nearest neighbor search
- [rankops](https://crates.io/crates/rankops): score fusion and reranking on top of embeddings

## License

MIT OR Apache-2.0
