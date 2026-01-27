use std::sync::{Arc, Mutex};

use embedd::{EmbedMode, PromptTemplate, PromptedTextEmbedder, TextEmbedder};

#[derive(Clone, Default)]
struct RecordingEmbedder {
    last_inputs: Arc<Mutex<Vec<String>>>,
}

impl RecordingEmbedder {
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.last_inputs.lock().unwrap())
    }
}

impl TextEmbedder for RecordingEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        *self.last_inputs.lock().unwrap() = texts.to_vec();
        // Deterministic dummy embedding.
        Ok(texts.iter().map(|_| vec![0.0f32, 1.0f32]).collect())
    }
}

#[test]
fn prompted_embedder_applies_query_prefix() {
    let inner = RecordingEmbedder::default();
    let prompt = PromptTemplate {
        query_prefix: "Q: ".into(),
        doc_prefix: "D: ".into(),
    };
    let wrapped = PromptedTextEmbedder::new(inner.clone(), prompt);

    let raw = vec!["hello".to_string()];
    let _ = wrapped.embed_texts(&raw, EmbedMode::Query).unwrap();

    let seen = inner.take();
    assert_eq!(seen, vec!["Q: hello".to_string()]);
}

#[test]
fn prompted_embedder_applies_doc_prefix() {
    let inner = RecordingEmbedder::default();
    let prompt = PromptTemplate {
        query_prefix: "Q: ".into(),
        doc_prefix: "D: ".into(),
    };
    let wrapped = PromptedTextEmbedder::new(inner.clone(), prompt);

    let raw = vec!["hello".to_string()];
    let _ = wrapped.embed_texts(&raw, EmbedMode::Document).unwrap();

    let seen = inner.take();
    assert_eq!(seen, vec!["D: hello".to_string()]);
}

#[test]
fn prompted_embedder_does_not_mutate_original_inputs() {
    let inner = RecordingEmbedder::default();
    let prompt = PromptTemplate {
        query_prefix: "Q: ".into(),
        doc_prefix: "D: ".into(),
    };
    let wrapped = PromptedTextEmbedder::new(inner, prompt);

    let raw = vec!["hello".to_string()];
    let raw_clone = raw.clone();
    let _ = wrapped.embed_texts(&raw, EmbedMode::Query).unwrap();
    assert_eq!(raw, raw_clone);
}
