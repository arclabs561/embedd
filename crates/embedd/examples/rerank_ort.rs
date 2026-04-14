//! Rerank documents using a cross-encoder ONNX model from HuggingFace Hub.
//!
//! Downloads `ms-marco-MiniLM-L-6-v2` (~80 MB, cached after first run)
//! and scores query-document pairs.
//!
//! Run: `cargo run -p embedd --example rerank_ort --features ort-tokenizers`

use embedd::ort::OrtReranker;
use embedd::Reranker;

fn main() -> anyhow::Result<()> {
    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string());

    println!("Loading {model_id}...");
    let reranker = OrtReranker::from_hf_hub(&model_id)?;
    println!("Model: {}", reranker.model_id().unwrap_or("unknown"));

    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital and most populous city of France.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "London is the capital of the United Kingdom.".to_string(),
        "France is a country in Western Europe.".to_string(),
    ];

    println!("\nQuery: {query}\n");

    let results = reranker.rerank(query, &documents, Some(3))?;

    for result in &results {
        println!("  [{:.4}] {}", result.score, documents[result.index]);
    }

    Ok(())
}
