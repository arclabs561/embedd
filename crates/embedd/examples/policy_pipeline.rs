//! Compose embedding policies without downloading a model.
//!
//! Uses a deterministic toy embedder with prompt scoping, output-dimension
//! truncation, L2 normalization, batching, and caching.
//!
//! Run:
//!
//! ```sh
//! cargo run -p embedd --example policy_pipeline
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use embedd::{
    apply_normalization_policy, apply_output_dim, apply_scoping_policy, BatchingTextEmbedder,
    CachingTextEmbedder, EmbedMode, Normalization, NormalizationPolicy, PromptApplication,
    PromptTemplate, ScopingPolicy, TextEmbedder, TextEmbedderCapabilities, TruncationPolicy,
};

#[derive(Clone)]
struct ToyEmbedder {
    calls: Arc<AtomicUsize>,
}

impl ToyEmbedder {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn embed_one(text: &str) -> Vec<f32> {
        let mut v = vec![0.0; 6];
        for (i, b) in text.bytes().enumerate() {
            let lane = i % v.len();
            v[lane] += (f32::from(b) / 255.0) + (lane as f32 * 0.01);
        }
        v
    }
}

impl TextEmbedder for ToyEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(texts.iter().map(|text| Self::embed_one(text)).collect())
    }

    fn model_id(&self) -> Option<&str> {
        Some("toy-hash-vectors")
    }

    fn dimension(&self) -> Option<usize> {
        Some(6)
    }

    fn capabilities(&self) -> TextEmbedderCapabilities {
        TextEmbedderCapabilities {
            uses_embed_mode: PromptApplication::None,
            normalization: Normalization::NotNormalized,
            truncation: TruncationPolicy::None,
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> anyhow::Result<()> {
    let toy = ToyEmbedder::new();
    let call_counter = toy.clone();

    let prompt = PromptTemplate {
        query_prefix: "query: ".to_string(),
        doc_prefix: "passage: ".to_string(),
    };

    let scoped = apply_scoping_policy(toy, ScopingPolicy::ClientPrefix(prompt))?;
    let truncated = apply_output_dim(scoped, Some(4))?;
    let normalized = apply_normalization_policy(truncated, NormalizationPolicy::RequireL2)?;
    let batched = BatchingTextEmbedder::new(normalized, 2);
    let cached = CachingTextEmbedder::new(batched);

    println!("model: {}", cached.model_id().unwrap_or("?"));
    println!("reported dimension: {:?}", cached.dimension());
    println!();

    let docs = vec![
        "hybrid retrieval combines lexical and dense scores".to_string(),
        "matryoshka embeddings allow shorter vectors at query time".to_string(),
        "rerankers score query document pairs".to_string(),
    ];

    let doc_vecs = cached.embed_texts(&docs, EmbedMode::Document)?;
    println!("document embeddings");
    for (doc, v) in docs.iter().zip(doc_vecs.iter()) {
        println!("  dim={} norm={:.3} text=\"{}\"", v.len(), l2_norm(v), doc);
        assert_eq!(v.len(), 4);
        assert!((l2_norm(v) - 1.0).abs() < 1e-5);
    }
    println!();

    let text = "matryoshka embeddings";
    let query_vec = cached.embed_text(text, EmbedMode::Query)?;
    let doc_vec = cached.embed_text(text, EmbedMode::Document)?;
    let same_text_similarity = embedd::vector::cosine_f32(&query_vec, &doc_vec);

    println!("same text under different scopes");
    println!("  cosine(query, document) = {same_text_similarity:.4}");
    assert!(same_text_similarity < 0.999);
    println!();

    let calls_after_first_pass = call_counter.calls();
    let _again = cached.embed_texts(&docs, EmbedMode::Document)?;
    println!("cache");
    println!("  backend calls after first pass: {calls_after_first_pass}");
    println!("  backend calls after cache hit:  {}", call_counter.calls());
    println!("  cache entries: {}", cached.cache_len());
    assert_eq!(calls_after_first_pass, call_counter.calls());

    Ok(())
}
