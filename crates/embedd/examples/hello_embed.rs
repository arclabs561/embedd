//! Minimal example: embed two strings and compute cosine similarity.
//!
//! Run: `cargo run -p embedd --example hello_embed --features fastembed`

use embedd::fastembed::FastembedEmbedder;
use embedd::{EmbedMode, TextEmbedder};

fn main() -> anyhow::Result<()> {
    let embedder = FastembedEmbedder::new_default()?;
    println!(
        "model: {}  dim: {}",
        embedder.model_id().unwrap_or("?"),
        embedder.dimension().unwrap_or(0),
    );

    let a = embedder.embed_text("the cat sat on the mat", EmbedMode::Document)?;
    let b = embedder.embed_text("a dog lay on the rug", EmbedMode::Document)?;

    let sim = embedd::vector::cosine_f32(&a, &b);
    println!("cosine similarity: {sim:.4}");

    Ok(())
}
