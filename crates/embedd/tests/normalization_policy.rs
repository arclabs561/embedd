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

#[test]
fn wrapper_preserves_zero_and_tiny_vectors_but_normalizes_ordinary_outputs() {
    struct BoundaryDummy;
    impl TextEmbedder for BoundaryDummy {
        fn embed_texts(&self, _: &[String], _: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0, 0.0], vec![1e-10, 0.0], vec![3.0, 4.0]])
        }
    }
    let wrapped = L2NormalizedTextEmbedder::new(BoundaryDummy);
    let result = wrapped
        .embed_texts(
            &["zero".into(), "tiny".into(), "normal".into()],
            EmbedMode::Query,
        )
        .unwrap();
    assert_eq!(result[0], [0.0, 0.0]);
    assert_eq!(result[1], [1e-10, 0.0]);
    assert!((result[2][0] - 0.6).abs() < 1e-6);
    assert!((result[2][1] - 0.8).abs() < 1e-6);
    assert_eq!(
        wrapped.capabilities().normalization,
        Normalization::L2Normalized
    );
}

#[test]
fn wrapper_handles_adjacent_threshold_values_in_long_vectors() {
    struct ThresholdDummy;
    impl TextEmbedder for ThresholdDummy {
        fn embed_texts(&self, _: &[String], _: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
            let epsilon = 1e-9_f32;
            Ok([
                epsilon.to_bits() - 1,
                epsilon.to_bits(),
                epsilon.to_bits() + 1,
            ]
            .into_iter()
            .map(|bits| {
                let mut v = vec![0.0; 129];
                v[128] = f32::from_bits(bits);
                v
            })
            .collect())
        }
    }
    let wrapped = L2NormalizedTextEmbedder::new(ThresholdDummy);
    let result = wrapped
        .embed_texts(
            &["below".into(), "at".into(), "above".into()],
            EmbedMode::Document,
        )
        .unwrap();
    assert_eq!(result[0][128], f32::from_bits(1e-9_f32.to_bits() - 1));
    assert_eq!(result[1][128], 1e-9);
    assert!((result[2][128] - 1.0).abs() < 1e-6);
    assert!(result.iter().all(|v| v[..128].iter().all(|&x| x == 0.0)));
}
