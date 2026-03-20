//! Batch-controlled embedding: split large inputs into fixed-size chunks.
//!
//! Run: `cargo run -p embedd --example batched_embed --features fastembed`

use embedd::fastembed::FastembedEmbedder;
use embedd::{BatchingTextEmbedder, EmbedMode, TextEmbedder};

fn main() -> anyhow::Result<()> {
    let inner = FastembedEmbedder::new_default()?;
    let embedder = BatchingTextEmbedder::new(inner, 4);

    // Generate 15 texts -- will be split into batches of 4.
    let texts: Vec<String> = (0..15)
        .map(|i| format!("Sentence number {} about various topics.", i))
        .collect();

    println!("Embedding {} texts in batches of 4...", texts.len());

    let vecs = embedder.embed_texts(&texts, EmbedMode::Document)?;
    println!("Got {} embeddings, each dim={}", vecs.len(), vecs[0].len());

    // Show pairwise similarity of first 3.
    for i in 0..3 {
        for j in (i + 1)..3 {
            let sim = embedd::vector::cosine_f32(&vecs[i], &vecs[j]);
            println!("  sim({i}, {j}) = {sim:.4}");
        }
    }

    Ok(())
}
