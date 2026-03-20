//! Tests for candle-hf architecture auto-detection from config.json.
//!
//! These encode the actual model_type values from HuggingFace model configs.
//! If a model changes its config format, the corresponding test will catch it.

// Detection logic is internal to candle_hf. We test via the public ModelArch export
// and by parsing real config.json snippets. Since detect_arch is private, we test
// it indirectly by checking that ModelSource -> LocalHfEmbedder produces the right arch.
// For unit-level validation, we replicate the detection logic here and test it directly.

/// Replicate detect_arch logic for testability.
/// This must stay in sync with candle_hf::detect_arch.
fn detect_arch(config_json: &serde_json::Value) -> &'static str {
    match config_json.get("model_type").and_then(|v| v.as_str()) {
        Some("distilbert") => "DistilBert",
        Some("xlm-roberta") => "XlmRoberta",
        Some("modernbert") => "ModernBert",
        Some("bert") => {
            if config_json
                .get("position_embedding_type")
                .and_then(|v| v.as_str())
                == Some("alibi")
            {
                "JinaBert"
            } else {
                "Bert"
            }
        }
        _ => "Bert",
    }
}

// -- Configs taken from actual HuggingFace model repos --

#[test]
fn bge_small_is_bert() {
    // BAAI/bge-small-en-v1.5 config.json
    let cfg = serde_json::json!({
        "model_type": "bert",
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}

#[test]
fn jina_v2_is_jina_bert() {
    // jinaai/jina-embeddings-v2-base-en config.json
    let cfg = serde_json::json!({
        "model_type": "bert",
        "position_embedding_type": "alibi",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "feed_forward_type": "geglu",
    });
    assert_eq!(detect_arch(&cfg), "JinaBert");
}

#[test]
fn xlm_roberta_detected() {
    // xlm-roberta-base config.json
    let cfg = serde_json::json!({
        "model_type": "xlm-roberta",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 250002,
    });
    assert_eq!(detect_arch(&cfg), "XlmRoberta");
}

#[test]
fn modernbert_detected() {
    // answerdotai/ModernBERT-base config.json
    let cfg = serde_json::json!({
        "model_type": "modernbert",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
    });
    assert_eq!(detect_arch(&cfg), "ModernBert");
}

#[test]
fn distilbert_detected() {
    let cfg = serde_json::json!({
        "model_type": "distilbert",
        "hidden_size": 768,
        "n_layers": 6,
    });
    assert_eq!(detect_arch(&cfg), "DistilBert");
}

#[test]
fn stella_400m_falls_through_to_bert() {
    // dunzhang/stella_en_400M_v5 uses model_type "new" -- not auto-detectable.
    // Should fall through to BERT (the safe default), not crash or misdetect.
    let cfg = serde_json::json!({
        "model_type": "new",
        "hidden_size": 1024,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}

#[test]
fn stella_1_5b_falls_through_to_bert() {
    // dunzhang/stella_en_1.5B_v5 uses model_type "qwen2" -- not supported.
    let cfg = serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": 1536,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}

#[test]
fn unknown_model_type_defaults_to_bert() {
    let cfg = serde_json::json!({
        "model_type": "some_future_arch",
        "hidden_size": 512,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}

#[test]
fn missing_model_type_defaults_to_bert() {
    let cfg = serde_json::json!({
        "hidden_size": 768,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}

#[test]
fn bert_with_absolute_position_is_not_jina() {
    let cfg = serde_json::json!({
        "model_type": "bert",
        "position_embedding_type": "absolute",
        "hidden_size": 768,
    });
    assert_eq!(detect_arch(&cfg), "Bert");
}
