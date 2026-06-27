//! Bench the Candle backend on CPU vs Apple Metal for a BGE embedding model.
//!
//! The device is auto-selected by feature (pick_device): the `metal` feature
//! requests Metal, plain `candle-hf` stays on CPU. Run both and compare:
//!
//!   cargo run --release --example metal_bench --features candle-hf   # CPU
//!   cargo run --release --example metal_bench --features metal        # Metal
//!
//! Downloads BAAI/bge-small-en-v1.5 (BERT, 384d) from the HF hub on first run.

use embedd::{EmbedMode, LocalHfEmbedder, ModelSource, TextEmbedder};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = if cfg!(feature = "metal") {
        "metal (requested)"
    } else {
        "cpu"
    };
    let texts: Vec<String> = (0..512)
        .map(|i| {
            format!(
                "Document {i}: notes on retrieval, embeddings, vector search, \
                 cosine similarity, and ranking in software systems."
            )
        })
        .collect();

    let src = ModelSource::HuggingFaceModelId("BAAI/bge-small-en-v1.5".to_string());
    let emb = LocalHfEmbedder::new(&src)?;

    // Warm up (graph + device init), then time the full batch.
    let _ = emb.embed_texts(&texts[..16.min(texts.len())], EmbedMode::Document)?;
    let t = Instant::now();
    let out = emb.embed_texts(&texts, EmbedMode::Document)?;
    let dt = t.elapsed().as_secs_f64();

    println!(
        "device={device}  {} texts ({} dim) in {dt:.2}s  ({:.0} texts/s)",
        out.len(),
        out.first().map(|e| e.len()).unwrap_or(0),
        texts.len() as f64 / dt
    );
    Ok(())
}
