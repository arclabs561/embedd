//! Attention masking in the local Candle backend for architectures whose
//! candle-transformers forward pass does not take a Hugging Face style
//! keep-mask (DistilBERT takes a drop-mask, JinaBERT takes none).
//!
//! Each test writes a tiny random model (config.json, tokenizer.json,
//! model.safetensors) to a temporary directory, so no download is needed.
//! A correct mask makes a text's embedding finite and independent of the
//! padding added for longer texts in the same batch.
#![cfg(feature = "candle-hf")]

use candle_core::{Device, Tensor};
use embedd::{EmbedMode, LocalHfEmbedder, PromptTemplate, TextEmbedder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const VOCAB: usize = 16;
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 16;
const POSITIONS: usize = 64;

const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]},
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "alpha": 4, "beta": 5,
      "gamma": 6, "delta": 7, "epsilon": 8, "zeta": 9, "eta": 10, "theta": 11,
      "iota": 12, "kappa": 13, "lambda": 14, "mu": 15
    },
    "unk_token": "[UNK]"
  }
}"#;

/// Deterministic weights in [-1, 1) from a 64-bit LCG.
struct Weights {
    state: u64,
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            tensors: HashMap::new(),
        }
    }

    fn random(&mut self, name: &str, shape: &[usize]) {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|_| {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        self.insert(name, values, shape);
    }

    fn layer_norm(&mut self, prefix: &str) {
        self.insert(&format!("{prefix}.weight"), vec![1.0; HIDDEN], &[HIDDEN]);
        self.insert(&format!("{prefix}.bias"), vec![0.0; HIDDEN], &[HIDDEN]);
    }

    fn linear(&mut self, prefix: &str, out_dim: usize, in_dim: usize) {
        self.random(&format!("{prefix}.weight"), &[out_dim, in_dim]);
        self.random(&format!("{prefix}.bias"), &[out_dim]);
    }

    fn insert(&mut self, name: &str, values: Vec<f32>, shape: &[usize]) {
        let t = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        self.tensors.insert(name.to_string(), t);
    }

    fn save(&self, path: &Path) {
        candle_core::safetensors::save(&self.tensors, path).unwrap();
    }
}

fn model_dir(tag: &str, config: serde_json::Value, weights: &Weights) -> PathBuf {
    // Tests run on parallel threads in one process, so each call needs its
    // own directory: a shared one is removed while another test reads it.
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "embedd-attention-mask-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
    weights.save(&dir.join("model.safetensors"));
    dir
}

fn tiny_distilbert() -> PathBuf {
    let config = serde_json::json!({
        "model_type": "distilbert", "vocab_size": VOCAB, "dim": HIDDEN, "n_layers": 1,
        "n_heads": 2, "hidden_dim": INTERMEDIATE, "activation": "gelu",
        "max_position_embeddings": POSITIONS, "initializer_range": 0.02, "pad_token_id": 0
    });
    let mut w = Weights::new(1);
    w.random("embeddings.word_embeddings.weight", &[VOCAB, HIDDEN]);
    w.random(
        "embeddings.position_embeddings.weight",
        &[POSITIONS, HIDDEN],
    );
    w.layer_norm("embeddings.LayerNorm");
    let p = "transformer.layer.0";
    for name in ["q_lin", "k_lin", "v_lin", "out_lin"] {
        w.linear(&format!("{p}.attention.{name}"), HIDDEN, HIDDEN);
    }
    w.layer_norm(&format!("{p}.sa_layer_norm"));
    w.linear(&format!("{p}.ffn.lin1"), INTERMEDIATE, HIDDEN);
    w.linear(&format!("{p}.ffn.lin2"), HIDDEN, INTERMEDIATE);
    w.layer_norm(&format!("{p}.output_layer_norm"));
    model_dir("distilbert", config, &w)
}

fn tiny_jinabert() -> PathBuf {
    let config = serde_json::json!({
        "model_type": "bert", "vocab_size": VOCAB, "hidden_size": HIDDEN,
        "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": INTERMEDIATE,
        "hidden_act": "gelu", "max_position_embeddings": POSITIONS, "type_vocab_size": 2,
        "initializer_range": 0.02, "layer_norm_eps": 1e-12, "pad_token_id": 0,
        "position_embedding_type": "alibi"
    });
    let mut w = Weights::new(2);
    w.random("embeddings.word_embeddings.weight", &[VOCAB, HIDDEN]);
    w.random("embeddings.token_type_embeddings.weight", &[2, HIDDEN]);
    w.layer_norm("embeddings.LayerNorm");
    let p = "encoder.layer.0";
    for name in ["query", "key", "value"] {
        w.linear(&format!("{p}.attention.self.{name}"), HIDDEN, HIDDEN);
    }
    w.linear(&format!("{p}.attention.output.dense"), HIDDEN, HIDDEN);
    w.layer_norm(&format!("{p}.attention.output.LayerNorm"));
    w.random(
        &format!("{p}.mlp.gated_layers.weight"),
        &[2 * INTERMEDIATE, HIDDEN],
    );
    w.linear(&format!("{p}.mlp.wo"), HIDDEN, INTERMEDIATE);
    w.layer_norm(&format!("{p}.mlp.layernorm"));
    model_dir("jinabert", config, &w)
}

fn load(dir: &Path) -> LocalHfEmbedder {
    LocalHfEmbedder::from_dir("tiny", dir)
        .unwrap()
        .with_prompt(PromptTemplate {
            query_prefix: String::new(),
            doc_prefix: String::new(),
        })
}

fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Embed `short` alone and next to a longer text that forces padding, and
/// return both embeddings of `short`.
fn alone_and_padded(m: &LocalHfEmbedder) -> (Vec<f32>, Vec<f32>) {
    let short = "alpha beta".to_string();
    let long = "gamma delta epsilon zeta eta theta iota".to_string();
    let alone = m
        .embed_texts(std::slice::from_ref(&short), EmbedMode::Document)
        .unwrap();
    let batched = m.embed_texts(&[short, long], EmbedMode::Document).unwrap();
    assert_eq!(batched.len(), 2);
    (alone[0].clone(), batched[0].clone())
}

#[test]
fn distilbert_embedding_is_finite() {
    let m = load(&tiny_distilbert());
    let v = m.embed_text("alpha beta", EmbedMode::Document).unwrap();
    assert_eq!(v.len(), HIDDEN);
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite embedding {v:?}"
    );
}

#[test]
fn distilbert_embedding_ignores_batch_padding() {
    let m = load(&tiny_distilbert());
    let (alone, padded) = alone_and_padded(&m);
    assert!(alone.iter().chain(&padded).all(|x| x.is_finite()));
    let d = l1(&alone, &padded);
    assert!(d < 1e-4, "padding changed the DistilBERT embedding: L1 {d}");
}

#[test]
fn jinabert_embedding_ignores_batch_padding() {
    let m = load(&tiny_jinabert());
    let (alone, padded) = alone_and_padded(&m);
    assert!(alone.iter().chain(&padded).all(|x| x.is_finite()));
    let d = l1(&alone, &padded);
    assert!(d < 1e-4, "padding changed the JinaBERT embedding: L1 {d}");
}
