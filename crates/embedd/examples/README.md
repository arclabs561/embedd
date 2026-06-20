# Examples

## Which example should I run?

| I want to... | Example |
|---|---|
| Understand the trait and policy wrappers without downloading a model | `policy_pipeline` |
| Embed two strings with a local ONNX backend | `hello_embed` |
| Rank a small corpus by cosine similarity | `semantic_search` |
| Split large requests into fixed-size batches | `batched_embed` |
| Use sparse lexical embeddings | `sparse_retrieval` |
| Compare backend capabilities and output policies | `backend_matrix`, `backend_compare` |
| Run an ONNX cross-encoder reranker | `rerank_ort` |

## Example descriptions

- `policy_pipeline`: no-network example using a deterministic toy embedder. Shows client-side prompt scoping, output-dimension truncation, L2 normalization, batching, and cache behavior.
- `hello_embed`: embeds two strings with `fastembed` and prints cosine similarity.
- `semantic_search`: embeds a small corpus, embeds several queries, and prints the top matches.
- `batched_embed`: wraps a backend with `BatchingTextEmbedder` and verifies large inputs are chunked.
- `sparse_retrieval`: uses the fastembed sparse backend to retrieve with weighted sparse vectors.
- `backend_matrix`: prints capability metadata and policy effects across configured backends.
- `backend_compare`: compares backend behavior on a shared corpus.
- `rerank_ort`: runs a local ONNX cross-encoder reranker. Requires `ort-tokenizers`.

## Running

```sh
cargo run -p embedd --example policy_pipeline
cargo run -p embedd --example semantic_search --features fastembed
cargo run -p embedd --example rerank_ort --features ort-tokenizers
```
