use embedd::{EmbedMode, L2NormalizedTextEmbedder, Normalization, TextEmbedder};

#[derive(Debug, Clone)]
struct NonNormalizedDummy;

impl TextEmbedder for NonNormalizedDummy {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        // Return vectors with varying norms.
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| vec![1.0 + i as f32, 2.0, 3.0])
            .collect())
    }

    fn capabilities(&self) -> embedd::TextEmbedderCapabilities {
        embedd::TextEmbedderCapabilities {
            uses_embed_mode: embedd::PromptApplication::None,
            normalization: Normalization::NotNormalized,
            truncation: embedd::TruncationPolicy::Unknown,
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn l2_normalized_wrapper_enforces_unit_norm() {
    let inner = NonNormalizedDummy;
    let e = L2NormalizedTextEmbedder::new(inner);
    let xs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let embs = e.embed_texts(&xs, EmbedMode::Query).unwrap();
    assert_eq!(embs.len(), xs.len());
    for v in &embs {
        let n = l2_norm(v);
        assert!((n - 1.0).abs() < 1e-4, "expected ~1.0, got {n}");
    }
    assert_eq!(e.capabilities().normalization, Normalization::L2Normalized);
}
