# embedd

[![crates.io](https://img.shields.io/crates/v/embedd.svg)](https://crates.io/crates/embedd)
[![Documentation](https://docs.rs/embedd/badge.svg)](https://docs.rs/embedd)

Embedding interfaces and backends.

One `TextEmbedder` trait covers local and remote providers. Backends are
feature-gated; the default build is traits and wrappers only.

```toml
[dependencies]
embedd = { version = "0.4", features = ["fastembed"] }
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

For the remote backend, enable `openai` in the dependency above (instead of
`fastembed`, or alongside it). This example requires an API key and network
access:

```rust
use embedd::openai::OpenAiEmbedder;
use embedd::{EmbedMode, TextEmbedder};

let embedder = OpenAiEmbedder::new(std::env::var("OPENAI_API_KEY")?, "text-embedding-3-small");
let vec = embedder.embed_text("hello world", EmbedMode::Query)?;
```

Embedding and scoring methods return `anyhow::Result`. Swapping backends uses
the same trait interface, with a backend-specific constructor and feature flag.

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

For span pooling, `LocalHfEmbedder` also exposes
`embed_tokens_with_offsets` under the `candle-hf` feature. It returns token
vectors plus byte offsets into the original input text. This is backend-specific
because remote embedding services do not all expose tokenizer offsets.

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

Also `semantic_search`, `sparse_retrieval`, and `batched_embed` (all
`fastembed`), `span_pool_offsets` (`candle-hf`), `rerank_ort`
(`ort-tokenizers`), and `backend_matrix` and `backend_compare` (whichever
backend features you enable).

## Vector acceleration

Vector helpers and normalization use safe scalar Rust by default. The `simd`
feature (also part of `all`) uses [innr](https://crates.io/crates/innr) instead
and requires Rust 1.89. Floating-point results can differ slightly because the
reduction order changes, and another dependency can turn `simd` on through
Cargo feature unification.

## License

MIT OR Apache-2.0
