//! E2E test against a real Qdrant instance.
//!
//! Requires Qdrant running on localhost:6334 (gRPC port).
//! Skips gracefully if Qdrant is not available.

use embedd::{EmbedMode, TextEmbedder};
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
use qdrant_client::Qdrant;

/// Stub embedder for deterministic testing (no model download needed).
struct FixedDimEmbedder {
    dim: usize,
}

impl TextEmbedder for FixedDimEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        // Generate deterministic vectors based on text content.
        // Each character contributes to a different dimension slot.
        Ok(texts
            .iter()
            .map(|t| {
                let mut vec = vec![0.0f32; self.dim];
                for (i, c) in t.chars().enumerate() {
                    vec[i % self.dim] += (c as u32) as f32 / 1000.0;
                }
                // L2 normalize.
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-9 {
                    for x in &mut vec {
                        *x /= norm;
                    }
                }
                vec
            })
            .collect())
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
}

async fn qdrant_client() -> Option<Qdrant> {
    let client = Qdrant::from_url("http://localhost:6334").build().ok()?;
    // Quick health check.
    client.health_check().await.ok()?;
    Some(client)
}

#[tokio::test]
async fn embed_upsert_and_search_roundtrip() {
    let Some(client) = qdrant_client().await else {
        eprintln!("skipping: Qdrant not available on localhost:6334");
        return;
    };

    let collection = "embedd_test_e2e";
    let dim = 32;
    let embedder = FixedDimEmbedder { dim };

    // Clean up from any previous run.
    let _ = client.delete_collection(collection).await;

    // Create collection.
    client
        .create_collection(
            CreateCollectionBuilder::new(collection)
                .vectors_config(VectorParamsBuilder::new(dim as u64, Distance::Cosine)),
        )
        .await
        .expect("failed to create collection");

    // Upsert.
    let docs = vec![
        "the cat sat on the mat".to_string(),
        "a dog played in the park".to_string(),
        "the sun shone brightly".to_string(),
    ];

    embedd_qdrant::embed_and_upsert(
        &client,
        collection,
        &embedder,
        &docs,
        0,
        EmbedMode::Document,
    )
    .await
    .expect("upsert failed");

    // Search.
    let results = embedd_qdrant::embed_and_search(&client, collection, &embedder, "cat mat", 3)
        .await
        .expect("search failed");

    assert_eq!(results.result.len(), 3, "expected 3 results");

    // The top result should be the most similar text.
    let top = &results.result[0];
    let top_text = top
        .payload
        .get("text")
        .and_then(|v| v.kind.as_ref())
        .map(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => s.as_str(),
            _ => "",
        })
        .unwrap_or("");

    // "cat mat" should be closest to "the cat sat on the mat" given our
    // character-based deterministic embedder.
    assert_eq!(
        top_text, "the cat sat on the mat",
        "expected cat sentence as top result, got: {top_text}"
    );

    // Verify scores are sorted descending.
    let scores: Vec<f32> = results.result.iter().map(|r| r.score).collect();
    for w in scores.windows(2) {
        assert!(w[0] >= w[1], "scores not sorted descending: {:?}", scores);
    }

    // Clean up.
    client
        .delete_collection(collection)
        .await
        .expect("cleanup failed");
}

#[tokio::test]
async fn embed_and_upsert_empty_is_noop() {
    let Some(client) = qdrant_client().await else {
        eprintln!("skipping: Qdrant not available on localhost:6334");
        return;
    };

    let embedder = FixedDimEmbedder { dim: 32 };

    // Empty upsert should succeed without creating/needing a collection.
    embedd_qdrant::embed_and_upsert(
        &client,
        "nonexistent_collection",
        &embedder,
        &[],
        0,
        EmbedMode::Document,
    )
    .await
    .expect("empty upsert should be a no-op");
}
