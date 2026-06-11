use crate::{EmbedMode, RerankResult, Reranker, TextEmbedder};
use anyhow::{Context, Result};
use std::path::Path;

/// Placeholder embedder for ORT-backed models.
pub struct OrtEmbedder;

impl OrtEmbedder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OrtEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl TextEmbedder for OrtEmbedder {
    fn embed_texts(&self, _texts: &[String], _mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!(
            "embedd::ort: not implemented yet (need model/tokenizer loading conventions)"
        ))
    }
}

/// ONNX Runtime cross-encoder reranker.
///
/// Loads a cross-encoder model (e.g. `ms-marco-MiniLM-L-6-v2`) from a
/// directory containing `model.onnx` (or `onnx/model.onnx`) and
/// `tokenizer.json`. Scores query-document pairs via batch forward pass.
///
/// ```ignore
/// let reranker = OrtReranker::from_dir("./reranker-model")?;
/// let results = reranker.rerank("what is rust?", &docs, Some(5))?;
/// ```
pub struct OrtReranker {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    needs_token_type_ids: bool,
    model_id: Option<String>,
}

impl OrtReranker {
    /// Load from a directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        let model_path = if dir.join("model.onnx").exists() {
            dir.join("model.onnx")
        } else if dir.join("onnx/model.onnx").exists() {
            dir.join("onnx/model.onnx")
        } else {
            anyhow::bail!(
                "no model.onnx found in {} or {}/onnx/",
                dir.display(),
                dir.display()
            );
        };

        let tokenizer_path = dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found in {}",
            dir.display()
        );

        // map_err: ort >= 2.0.0-rc.12 builder errors carry the builder back
        // (Error<SessionBuilder>), which is not Send + Sync, so `?` into
        // anyhow no longer converts directly.
        let builder = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("creating ONNX session builder: {e}"))?;
        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting ONNX optimization level: {e}"))?
            .commit_from_file(&model_path)
            .with_context(|| format!("loading ONNX model from {}", model_path.display()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let model_id = dir.file_name().map(|n| n.to_string_lossy().into_owned());

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            needs_token_type_ids,
            model_id,
        })
    }

    /// Load a cross-encoder from HuggingFace Hub by model ID.
    ///
    /// Downloads `model.onnx` (or `onnx/model.onnx`) and `tokenizer.json`
    /// to the HF cache directory. Subsequent calls use the cached files.
    ///
    /// ```ignore
    /// // Default: ms-marco-MiniLM-L-6-v2 (80 MB, fastest)
    /// let reranker = OrtReranker::from_hf_hub(
    ///     "cross-encoder/ms-marco-MiniLM-L-6-v2",
    /// )?;
    /// ```
    pub fn from_hf_hub(model_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        let tokenizer_path = repo
            .get("tokenizer.json")
            .with_context(|| format!("downloading tokenizer.json from {model_id}"))?;

        // Try onnx/model.onnx first (optimum export convention), then model.onnx.
        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .with_context(|| {
                format!(
                    "downloading model.onnx from {model_id} (tried onnx/model.onnx and model.onnx)"
                )
            })?;

        // map_err: ort >= 2.0.0-rc.12 builder errors carry the builder back
        // (Error<SessionBuilder>), which is not Send + Sync, so `?` into
        // anyhow no longer converts directly.
        let builder = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("creating ONNX session builder: {e}"))?;
        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting ONNX optimization level: {e}"))?
            .commit_from_file(&model_path)
            .with_context(|| format!("loading ONNX model from {}", model_path.display()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            needs_token_type_ids,
            model_id: Some(model_id.to_string()),
        })
    }

    /// Run a dummy inference to warm up the ONNX Runtime session.
    ///
    /// ONNX Runtime JIT-compiles kernels on first use, causing a multi-second
    /// latency spike. Call this once after construction to absorb that cost
    /// before serving live traffic.
    pub fn warmup(&self) -> Result<()> {
        self.score_pairs("warmup", &["warmup".to_string()])?;
        Ok(())
    }

    /// Score a batch of query-document pairs. Returns raw logit scores.
    fn score_pairs(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let pairs: Vec<tokenizers::EncodeInput> = documents
            .iter()
            .map(|doc| {
                tokenizers::EncodeInput::Dual(
                    tokenizers::InputSequence::from(query),
                    tokenizers::InputSequence::from(doc.as_str()),
                )
            })
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids_data = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let offset = i * max_len;
            for (j, (&id, &m)) in enc
                .get_ids()
                .iter()
                .zip(enc.get_attention_mask())
                .enumerate()
            {
                input_ids[offset + j] = id as i64;
                attention_mask[offset + j] = m as i64;
            }
            if self.needs_token_type_ids {
                for (j, &t) in enc.get_type_ids().iter().enumerate() {
                    token_type_ids_data[offset + j] = t as i64;
                }
            }
        }

        let shape = vec![batch_size as i64, max_len as i64];

        let ids_tensor =
            ort::value::Tensor::from_array((shape.clone(), input_ids.into_boxed_slice()))?;
        let mask_tensor =
            ort::value::Tensor::from_array((shape.clone(), attention_mask.into_boxed_slice()))?;

        let mut inputs = ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        ];

        if self.needs_token_type_ids {
            let types_tensor =
                ort::value::Tensor::from_array((shape, token_type_ids_data.into_boxed_slice()))?;
            inputs.push((
                std::borrow::Cow::from("token_type_ids"),
                ort::session::SessionInputValue::from(types_tensor),
            ));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;
        let outputs = session.run(inputs)?;

        // Extract logits[:, 0] as scores.
        let logits_value = &outputs[0];
        let (_shape, data) = logits_value.try_extract_tensor::<f32>()?;

        // data is flat [batch_size * num_labels]; take every num_labels-th element.
        let num_labels = _shape.last().copied().unwrap_or(1) as usize;
        let scores: Vec<f32> = data.chunks(num_labels).map(|chunk| chunk[0]).collect();

        Ok(scores)
    }
}

impl Reranker for OrtReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let scores = self.score_pairs(query, documents)?;

        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| RerankResult { index: i, score })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = top_k {
            results.truncate(k);
        }

        Ok(results)
    }

    fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }
}
