use embedd::{BatchingTextEmbedder, EmbedMode, TextEmbedder};
use std::sync::{Arc, Mutex};

/// Stub embedder that records batch sizes it was called with.
struct RecordingEmbedder {
    dim: usize,
    calls: Arc<Mutex<Vec<usize>>>,
}

impl RecordingEmbedder {
    fn new(dim: usize) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                dim,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl TextEmbedder for RecordingEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        self.calls.lock().unwrap().push(texts.len());
        Ok(texts.iter().map(|_| vec![1.0; self.dim]).collect())
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
}

#[test]
fn batching_passes_through_small_batches() {
    let (inner, calls) = RecordingEmbedder::new(4);
    let batched = BatchingTextEmbedder::new(inner, 10);
    let texts: Vec<String> = (0..5).map(|i| format!("text {i}")).collect();
    let result = batched.embed_texts(&texts, EmbedMode::Document).unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(*calls.lock().unwrap(), vec![5]); // single call, no splitting
}

#[test]
fn batching_splits_large_batches() {
    let (inner, calls) = RecordingEmbedder::new(4);
    let batched = BatchingTextEmbedder::new(inner, 3);
    let texts: Vec<String> = (0..10).map(|i| format!("text {i}")).collect();
    let result = batched.embed_texts(&texts, EmbedMode::Document).unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(*calls.lock().unwrap(), vec![3, 3, 3, 1]);
}

#[test]
fn batching_exact_multiple() {
    let (inner, calls) = RecordingEmbedder::new(4);
    let batched = BatchingTextEmbedder::new(inner, 4);
    let texts: Vec<String> = (0..8).map(|i| format!("text {i}")).collect();
    let result = batched.embed_texts(&texts, EmbedMode::Document).unwrap();
    assert_eq!(result.len(), 8);
    assert_eq!(*calls.lock().unwrap(), vec![4, 4]);
}

#[test]
fn batching_preserves_capabilities() {
    let (inner, _) = RecordingEmbedder::new(128);
    let batched = BatchingTextEmbedder::new(inner, 4);
    assert_eq!(batched.dimension(), Some(128));
}

#[test]
fn batching_single_text_convenience() {
    let (inner, _) = RecordingEmbedder::new(4);
    let batched = BatchingTextEmbedder::new(inner, 32);
    let vec = batched.embed_text("hello", EmbedMode::Query).unwrap();
    assert_eq!(vec.len(), 4);
}

#[test]
fn batching_empty_input() {
    let (inner, calls) = RecordingEmbedder::new(4);
    let batched = BatchingTextEmbedder::new(inner, 4);
    let result = batched.embed_texts(&[], EmbedMode::Document).unwrap();
    assert!(result.is_empty());
    assert_eq!(*calls.lock().unwrap(), vec![0]); // one call with empty slice
}
