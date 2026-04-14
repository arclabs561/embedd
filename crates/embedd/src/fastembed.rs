use crate::{EmbedMode, SparseEmbedder, TextEmbedder};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Global model cache to prevent multiple initializations.
///
/// Rationale:
/// - Some `fastembed` backends are FFI-backed and can misbehave if repeatedly initialized/dropped.
/// - For a tool-heavy workspace (tests, benches), "drop order" is hard to reason about.
/// - We accept a bounded leak (models live until process exit) to avoid teardown UAF/segfaults.
///
/// Invariant: the cache is initialized once and never dropped.
#[allow(clippy::type_complexity)]
static MODEL_CACHE: OnceLock<&'static Mutex<HashMap<String, Arc<Mutex<TextEmbedding>>>>> =
    OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Arc<Mutex<TextEmbedding>>>> {
    MODEL_CACHE.get_or_init(|| Box::leak(Box::new(Mutex::new(HashMap::new()))))
}

/// A `fastembed`-backed embedder with process-wide model caching.
pub struct FastembedEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    model_id: String,
    dimension: usize,
}

impl FastembedEmbedder {
    /// Create a default embedder (currently `AllMiniLML6V2`).
    pub fn new_default() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create an embedder with an explicit fastembed model.
    pub fn with_model(model_name: EmbeddingModel) -> Result<Self> {
        let model = Self::get_or_init_model(model_name.clone())?;
        let model_id = format!("fastembed:{:?}", model_name);

        // Probe dimension once.
        let dimension = {
            let mut guard = model.lock().expect("fastembed model mutex poisoned");
            let out = guard
                .embed(vec!["probe"], None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.first()
                .map(|v| v.len())
                .filter(|&d| d > 0)
                .ok_or_else(|| anyhow::anyhow!("model returned zero-dim or empty embedding"))?
        };

        Ok(Self {
            model,
            model_id,
            dimension,
        })
    }

    fn get_or_init_model(model_name: EmbeddingModel) -> Result<Arc<Mutex<TextEmbedding>>> {
        let key = format!("{:?}", model_name);

        let mut guard = cache()
            .lock()
            .expect("fastembed model cache mutex poisoned");
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let model =
            TextEmbedding::try_new(InitOptions::new(model_name).with_show_download_progress(false))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        let arc = Arc::new(Mutex::new(model));
        guard.insert(key, Arc::clone(&arc));
        Ok(arc)
    }
}

impl TextEmbedder for FastembedEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut guard = self.model.lock().expect("fastembed model mutex poisoned");
        let embs = guard
            .embed(refs, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(embs)
    }

    fn model_id(&self) -> Option<&str> {
        Some(self.model_id.as_str())
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dimension)
    }

    fn capabilities(&self) -> crate::TextEmbedderCapabilities {
        crate::TextEmbedderCapabilities {
            uses_embed_mode: crate::PromptApplication::None,
            normalization: crate::Normalization::Unknown,
            truncation: crate::TruncationPolicy::Unknown,
        }
    }
}

// --- Sparse embedding via fastembed ---

use fastembed::{SparseModel, SparseTextEmbedding};

#[allow(clippy::type_complexity)]
static SPARSE_CACHE: OnceLock<&'static Mutex<HashMap<String, Arc<Mutex<SparseTextEmbedding>>>>> =
    OnceLock::new();

fn sparse_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<SparseTextEmbedding>>>> {
    SPARSE_CACHE.get_or_init(|| Box::leak(Box::new(Mutex::new(HashMap::new()))))
}

/// Sparse lexical embedder backed by fastembed (SPLADE, BGE-M3).
pub struct FastembedSparseEmbedder {
    model: Arc<Mutex<SparseTextEmbedding>>,
    model_id: String,
}

impl FastembedSparseEmbedder {
    /// Create with the default sparse model (SPLADE PP v1).
    pub fn new_default() -> Result<Self> {
        Self::with_model(SparseModel::default())
    }

    /// Create with an explicit fastembed sparse model.
    pub fn with_model(model_name: SparseModel) -> Result<Self> {
        let model = Self::get_or_init(model_name.clone())?;
        let model_id = format!("fastembed-sparse:{:?}", model_name);
        Ok(Self { model, model_id })
    }

    fn get_or_init(model_name: SparseModel) -> Result<Arc<Mutex<SparseTextEmbedding>>> {
        let key = format!("{:?}", model_name);
        let mut guard = sparse_cache()
            .lock()
            .expect("sparse model cache mutex poisoned");
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let opts = fastembed::SparseInitOptions::new(model_name).with_show_download_progress(false);
        let model = SparseTextEmbedding::try_new(opts).map_err(|e| anyhow::anyhow!("{e}"))?;
        let arc = Arc::new(Mutex::new(model));
        guard.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl SparseEmbedder for FastembedSparseEmbedder {
    fn embed_sparse(&self, texts: &[String], _mode: EmbedMode) -> Result<Vec<Vec<(u32, f32)>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut guard = self.model.lock().expect("sparse model mutex poisoned");
        let sparse = guard
            .embed(refs, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Bridge fastembed's (Vec<usize>, Vec<f32>) to our (u32, f32) pairs.
        Ok(sparse
            .into_iter()
            .map(|s| {
                s.indices
                    .into_iter()
                    .zip(s.values)
                    .map(|(idx, val)| (idx as u32, val))
                    .collect()
            })
            .collect())
    }
}

// --- Reranker via fastembed ---

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

#[allow(clippy::type_complexity)]
static RERANK_CACHE: OnceLock<&'static Mutex<HashMap<String, Arc<Mutex<TextRerank>>>>> =
    OnceLock::new();

fn rerank_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<TextRerank>>>> {
    RERANK_CACHE.get_or_init(|| Box::leak(Box::new(Mutex::new(HashMap::new()))))
}

/// Cross-encoder reranker backed by fastembed with process-wide model caching.
pub struct FastembedReranker {
    model: Arc<Mutex<TextRerank>>,
    model_id: String,
}

impl FastembedReranker {
    /// Create a reranker with the default model (BGE Reranker Base).
    pub fn new_default() -> Result<Self> {
        Self::with_model(RerankerModel::BGERerankerBase)
    }

    /// Create a reranker with an explicit fastembed reranker model.
    pub fn with_model(model_name: RerankerModel) -> Result<Self> {
        let model = Self::get_or_init(model_name.clone())?;
        let model_id = format!("fastembed-rerank:{:?}", model_name);
        Ok(Self { model, model_id })
    }

    fn get_or_init(model_name: RerankerModel) -> Result<Arc<Mutex<TextRerank>>> {
        let key = format!("{:?}", model_name);
        let mut guard = rerank_cache()
            .lock()
            .expect("rerank model cache mutex poisoned");
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let model = TextRerank::try_new(
            RerankInitOptions::new(model_name).with_show_download_progress(false),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let arc = Arc::new(Mutex::new(model));
        guard.insert(key, Arc::clone(&arc));
        Ok(arc)
    }
}

impl crate::Reranker for FastembedReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<Vec<crate::RerankResult>> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let mut guard = self.model.lock().expect("rerank model mutex poisoned");
        let results = guard
            .rerank(query, refs, false, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut out: Vec<crate::RerankResult> = results
            .into_iter()
            .map(|r| crate::RerankResult {
                index: r.index,
                score: r.score,
            })
            .collect();
        // fastembed returns sorted by descending score, but re-sort to be safe.
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = top_k {
            out.truncate(k);
        }
        Ok(out)
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }
}
