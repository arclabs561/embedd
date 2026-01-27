use embedd::{
    apply_normalization_policy, apply_output_dim, EmbedMode, Normalization, NormalizationPolicy,
    TextEmbedder,
};

#[derive(Debug, Clone)]
struct FixedDimDummy {
    dim: usize,
}

impl TextEmbedder for FixedDimDummy {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                // Nonzero vectors so L2 normalization is meaningful.
                (0..self.dim)
                    .map(|j| 1.0 + (i as f32) + (j as f32))
                    .collect()
            })
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
fn output_dim_truncates_vectors() {
    let inner = FixedDimDummy { dim: 8 };
    let e = apply_output_dim(inner, Some(3)).unwrap();
    let xs = vec!["a".to_string(), "b".to_string()];
    let out = e.embed_texts(&xs, EmbedMode::Query).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 3);
    assert_eq!(out[1].len(), 3);
}

#[test]
fn output_dim_errors_if_requested_dim_exceeds_actual_dim() {
    let inner = FixedDimDummy { dim: 4 };
    let e = apply_output_dim(inner, Some(8)).unwrap();
    let xs = vec!["a".to_string()];
    let err = e.embed_texts(&xs, EmbedMode::Query).unwrap_err();
    let s = format!("{err:#}");
    assert!(s.contains("exceeds embedding dim"), "err={s}");
}

#[test]
fn output_dim_then_require_l2_yields_unit_norm_vectors() {
    let inner = FixedDimDummy { dim: 8 };
    let e = apply_output_dim(inner, Some(3)).unwrap();
    let e = apply_normalization_policy(e, NormalizationPolicy::RequireL2).unwrap();

    let xs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let out = e.embed_texts(&xs, EmbedMode::Query).unwrap();
    for v in &out {
        assert_eq!(v.len(), 3);
        let n = l2_norm(v);
        assert!((n - 1.0).abs() < 1e-4, "expected ~1.0, got {n}");
    }

    // Wrapper policy should report L2Normalized after RequireL2.
    assert_eq!(e.capabilities().normalization, Normalization::L2Normalized);
}
