use embedd::{CachingTextEmbedder, EmbedMode, TextEmbedder};

/// Returns one fewer vector than requested.
#[derive(Debug, Clone)]
struct ShortOutput;

impl TextEmbedder for ShortOutput {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .take(texts.len().saturating_sub(1))
            .map(|_| vec![1.0, 0.0])
            .collect())
    }
}

#[test]
fn cache_errors_when_inner_returns_fewer_vectors_than_texts() {
    let cached = CachingTextEmbedder::new(ShortOutput);
    let xs = vec!["a".to_string(), "b".to_string()];
    let err = cached.embed_texts(&xs, EmbedMode::Query).unwrap_err();
    assert!(format!("{err:#}").contains("returned 1 vectors for 2 texts"));
    assert_eq!(cached.cache_len(), 0);
}
