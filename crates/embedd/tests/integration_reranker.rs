//! Integration test for real fastembed reranker.
//! Requires model download -- run with: cargo test --features fastembed -- --ignored

#[cfg(feature = "fastembed")]
mod fastembed_reranker {
    use embedd::{FastembedReranker, Reranker};

    #[test]
    #[ignore]
    fn test_fastembed_reranker_real() {
        let reranker = FastembedReranker::new_default()
            .expect("failed to initialize fastembed reranker (model download required)");

        let query = "What is machine learning?";
        let documents = vec![
            "Machine learning is a subset of artificial intelligence.".to_string(),
            "The weather today is sunny and warm.".to_string(),
            "Deep learning uses neural networks with many layers.".to_string(),
            "I enjoy cooking pasta for dinner.".to_string(),
        ];

        let results = reranker
            .rerank(query, &documents, Some(2))
            .expect("reranking failed");

        assert_eq!(results.len(), 2, "top_k=2 should return 2 results");
        assert!(
            results[0].score >= results[1].score,
            "results should be sorted by descending score"
        );
        // The ML-related documents should rank higher than cooking/weather
        assert!(
            results[0].index == 0 || results[0].index == 2,
            "top result should be an ML-related document, got index {}",
            results[0].index
        );
    }
}
