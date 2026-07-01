//! Pool local token embeddings into span embeddings using tokenizer offsets.
//!
//! This example is model-gated. Set `EMBEDD_MODEL_DIR` or `EMBEDD_MODEL`
//! before running; otherwise it exits successfully without loading weights.
//!
//! ```sh
//! EMBEDD_MODEL_DIR=/path/to/model cargo run -p embedd --example span_pool_offsets --features candle-hf
//! ```

use embedd::{EmbedMode, LocalHfEmbedder, ModelSource};
use slabs::{LateChunkingPooler as SpanPooler, Slab};

fn main() -> anyhow::Result<()> {
    if std::env::var("EMBEDD_MODEL_DIR").is_err()
        && std::env::var("EMBEDD_MODEL").is_err()
        && std::env::var("IKSH_EMBED_MODEL_DIR").is_err()
        && std::env::var("IKSH_EMBED_MODEL").is_err()
    {
        println!("set EMBEDD_MODEL_DIR or EMBEDD_MODEL to run this example");
        return Ok(());
    }

    let document = "Ada designed the engine. It tabulated values.";
    let first_end = document.find('.').expect("example document has a sentence") + 1;
    let second_start = first_end + 1;
    let spans = vec![
        Slab::from_byte_range(document, 0..first_end, 0)?,
        Slab::from_byte_range(document, second_start..document.len(), 1)?,
    ];

    let embedder = LocalHfEmbedder::new(&ModelSource::from_env_any())?;
    let batch = embedder.embed_tokens_with_offsets(&[document.to_string()], EmbedMode::Document)?;
    let Some(token_output) = batch.into_iter().next() else {
        println!("no token output");
        return Ok(());
    };
    let (token_embeddings, byte_offsets) = token_output.into_parts();
    let Some(dim) = token_embeddings.first().map(Vec::len) else {
        println!("no token embeddings");
        return Ok(());
    };

    let pooler = SpanPooler::new(dim);
    let span_embeddings = pooler.pool_with_offsets(&token_embeddings, &byte_offsets, &spans);

    println!(
        "pooled {} spans from {} token vectors (dim={dim})",
        span_embeddings.len(),
        token_embeddings.len()
    );

    Ok(())
}
