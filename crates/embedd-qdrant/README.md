# embedd-qdrant

Bridge between [embedd](https://crates.io/crates/embedd) embedders and [Qdrant](https://qdrant.tech/) vector database.

Provides `embed_and_upsert` and `embed_and_search` for the common "embed text, then store/query" workflow.

## Usage

```rust
use embedd::fastembed::FastembedEmbedder;
use embedd::{EmbedMode, TextEmbedder};
use embedd_qdrant::{embed_and_upsert, embed_and_search};
use qdrant_client::Qdrant;

let client = Qdrant::from_url("http://localhost:6334").build()?;
let embedder = FastembedEmbedder::new_default()?;

// Upsert
let docs = vec!["cats are great".into(), "dogs are loyal".into()];
embed_and_upsert(&client, "my_collection", &embedder, &docs, 0, EmbedMode::Document).await?;

// Search
let results = embed_and_search(&client, "my_collection", &embedder, "pets", 5).await?;
```

## License

MIT OR Apache-2.0
