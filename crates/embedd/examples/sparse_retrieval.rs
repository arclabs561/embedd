//! Sparse retrieval: SPLADE lexical embeddings for term-weighted search.
//!
//! Run: `cargo run -p embedd --example sparse_retrieval --features fastembed`

use embedd::fastembed::FastembedSparseEmbedder;
use embedd::{EmbedMode, SparseEmbedder};

fn main() -> anyhow::Result<()> {
    let sparse = FastembedSparseEmbedder::new_default()?;
    println!("model: {}", sparse.model_id());

    let docs = vec![
        "the quick brown fox jumps over the lazy dog".to_string(),
        "a fast red car drives along the highway".to_string(),
        "the lazy cat sleeps on the warm windowsill".to_string(),
    ];

    let query = "quick fox".to_string();

    let doc_vecs = sparse.embed_sparse(&docs, EmbedMode::Document)?;
    let query_vecs = sparse.embed_sparse(std::slice::from_ref(&query), EmbedMode::Query)?;
    let query_vec = &query_vecs[0];

    println!("\nQuery: \"{query}\"");
    println!("Query sparse vector: {} non-zero terms", query_vec.len());

    // Dot-product scoring between sparse vectors.
    let mut scores: Vec<(usize, f32)> = doc_vecs
        .iter()
        .enumerate()
        .map(|(i, doc_vec)| {
            let score = sparse_dot(query_vec, doc_vec);
            (i, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (idx, score)) in scores.iter().enumerate() {
        println!(
            "  #{}: {:.4}  {} ({} terms)",
            i + 1,
            score,
            &docs[*idx],
            doc_vecs[*idx].len(),
        );
    }

    Ok(())
}

/// Dot product between two sparse vectors.
fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    // For small vectors, nested iteration is fine.
    // For production, use sorted merge or hash lookup.
    let mut score = 0.0f32;
    for &(ai, av) in a {
        for &(bi, bv) in b {
            if ai == bi {
                score += av * bv;
            }
        }
    }
    score
}
