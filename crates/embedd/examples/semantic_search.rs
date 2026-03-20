//! Semantic search: embed a corpus, query it, show ranked results.
//!
//! Run: `cargo run -p embedd --example semantic_search --features fastembed`

use embedd::fastembed::FastembedEmbedder;
use embedd::{EmbedMode, TextEmbedder};

fn main() -> anyhow::Result<()> {
    let embedder = FastembedEmbedder::new_default()?;
    println!(
        "model: {}  dim: {}",
        embedder.model_id().unwrap_or("?"),
        embedder.dimension().unwrap_or(0),
    );

    // A small corpus about programming languages.
    let corpus = vec![
        "Rust is a systems programming language focused on safety and performance.",
        "Python is widely used for data science, machine learning, and scripting.",
        "JavaScript runs in browsers and powers most interactive web applications.",
        "Go was designed at Google for concurrent server-side programming.",
        "Haskell is a purely functional programming language with strong static typing.",
        "C++ provides low-level memory control and is used in game engines and databases.",
        "TypeScript adds static types to JavaScript for large-scale applications.",
        "Julia is designed for high-performance numerical and scientific computing.",
    ];

    let corpus_strings: Vec<String> = corpus.iter().map(|s| s.to_string()).collect();
    let corpus_vecs = embedder.embed_texts(&corpus_strings, EmbedMode::Document)?;

    // Interactive-style queries.
    let queries = [
        "fast compiled language for systems work",
        "language for machine learning",
        "functional programming",
        "web development",
    ];

    for query in &queries {
        let query_vec = embedder.embed_text(query, EmbedMode::Query)?;

        // Rank by cosine similarity.
        let mut scored: Vec<(usize, f32)> = corpus_vecs
            .iter()
            .enumerate()
            .map(|(i, doc_vec)| (i, embedd::vector::cosine_f32(&query_vec, doc_vec)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nQuery: \"{query}\"");
        for (i, (idx, score)) in scored.iter().take(3).enumerate() {
            println!("  #{}: {:.4}  {}", i + 1, score, corpus[*idx]);
        }
    }

    Ok(())
}
