//! `embedd`: embedding interfaces + reusable backends (multi-modality substrate).
//!
//! This crate is the “shared embedding substrate”: consumers should depend on `embedd`
//! (traits + basic types), and enable backend **features** (`candle-hf`, `openai`, `tei`, etc.)
//! as needed.
//!
//! Scope: **any modality** (text, images, audio, …) as long as the interface remains small and
//! testable. Concretely, we provide:
//! - `TextEmbedder` (text -> vectors)
//! - `ImageEmbedder` (bytes -> vectors)
//! - `AudioEmbedder` (bytes -> vectors; placeholder contract)
//! - extension traits for token-level / sparse embeddings.
//!
//! ## Compatibility note (env vars)
//! For now, we preserve `iksh`'s env-var surface to avoid behavior drift:
//! - `IKSH_EMBED_MODEL`
//! - `IKSH_EMBED_MAX_LEN`
//! - `IKSH_EMBED_QUERY_PREFIX`
//! - `IKSH_EMBED_DOC_PREFIX`

/// Whether an embedding is for a query or a document/passage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedMode {
    /// Query embedding (may use a different instruction/prefix).
    Query,
    /// Document / passage embedding.
    Document,
}

/// Prompt template applied before tokenization, used for instruction-tuned / prompt-tuned embedders.
///
/// Common patterns in the ecosystem:
/// - **E5**: `query: {text}` vs `passage: {text}`
/// - **BGE** (often): no prompt, or `query: ...` / `passage: ...` depending on checkpoint
/// - **Instruct models**: sometimes use longer task prompts; we keep this as plain prefixing
///   for now because it composes with both local and remote backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    pub query_prefix: String,
    pub doc_prefix: String,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self {
            query_prefix: "query: ".to_string(),
            doc_prefix: "passage: ".to_string(),
        }
    }
}

impl PromptTemplate {
    pub fn apply(&self, mode: EmbedMode, text: &str) -> String {
        match mode {
            EmbedMode::Query => format!("{}{}", self.query_prefix, text),
            EmbedMode::Document => format!("{}{}", self.doc_prefix, text),
        }
    }

    /// Load prompt prefixes from the legacy `iksh` env vars (compat).
    pub fn from_iksh_env() -> Self {
        let query_prefix = std::env::var("IKSH_EMBED_QUERY_PREFIX")
            .unwrap_or_else(|_| Self::default().query_prefix);
        let doc_prefix =
            std::env::var("IKSH_EMBED_DOC_PREFIX").unwrap_or_else(|_| Self::default().doc_prefix);
        Self {
            query_prefix,
            doc_prefix,
        }
    }

    /// Load prompt prefixes from `EMBEDD_QUERY_PREFIX` / `EMBEDD_DOC_PREFIX`.
    ///
    /// Falls back to defaults when unset.
    pub fn from_embedd_env() -> Self {
        let query_prefix =
            std::env::var("EMBEDD_QUERY_PREFIX").unwrap_or_else(|_| Self::default().query_prefix);
        let doc_prefix =
            std::env::var("EMBEDD_DOC_PREFIX").unwrap_or_else(|_| Self::default().doc_prefix);
        Self {
            query_prefix,
            doc_prefix,
        }
    }

    /// Prefer `EMBEDD_*` prompt env vars, else fall back to `IKSH_*`, else defaults.
    ///
    /// This is useful for examples/benchmarks that want a single “prompt surface” without
    /// accidentally drifting existing `iksh` behavior.
    pub fn from_env_any() -> Self {
        let has_embedd = std::env::var("EMBEDD_QUERY_PREFIX").is_ok()
            || std::env::var("EMBEDD_DOC_PREFIX").is_ok();
        if has_embedd {
            return Self::from_embedd_env();
        }

        let has_iksh = std::env::var("IKSH_EMBED_QUERY_PREFIX").is_ok()
            || std::env::var("IKSH_EMBED_DOC_PREFIX").is_ok();
        if has_iksh {
            return Self::from_iksh_env();
        }

        Self::default()
    }
}

/// Wrapper that applies a `PromptTemplate` before calling an inner `TextEmbedder`.
///
/// This is the “instruction/scoped embedding” adapter when a backend:
/// - ignores `EmbedMode`, or
/// - expects explicit prompt prefixes.
///
/// **Nuance**: If your inner backend already applies its own prompt logic (e.g. a backend that
/// uses `EmbedMode` internally), wrapping may double-prefix. Keep this opt-in.
#[derive(Debug, Clone)]
pub struct PromptedTextEmbedder<E> {
    inner: E,
    prompt: PromptTemplate,
}

impl<E> PromptedTextEmbedder<E> {
    pub fn new(inner: E, prompt: PromptTemplate) -> Self {
        Self { inner, prompt }
    }

    pub fn prompt(&self) -> &PromptTemplate {
        &self.prompt
    }

    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: TextEmbedder> TextEmbedder for PromptedTextEmbedder<E> {
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        let prompted: Vec<String> = texts.iter().map(|t| self.prompt.apply(mode, t)).collect();
        // Pass the same mode through; the inner may ignore it or use it for other behavior.
        self.inner.embed_texts(&prompted, mode)
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }

    fn dimension(&self) -> Option<usize> {
        self.inner.dimension()
    }

    fn capabilities(&self) -> TextEmbedderCapabilities {
        let mut caps = self.inner.capabilities();
        // This wrapper definitely applies a client-side prefix and uses EmbedMode for that.
        caps.uses_embed_mode = PromptApplication::ClientPrefix;
        caps
    }
}

/// Wrapper that enforces L2-normalized outputs.
///
/// This is the “design around normalization drift” adapter: downstream code can rely on cosine==dot.
#[derive(Debug, Clone)]
pub struct L2NormalizedTextEmbedder<E> {
    inner: E,
}

impl<E> L2NormalizedTextEmbedder<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E: TextEmbedder> TextEmbedder for L2NormalizedTextEmbedder<E> {
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut embs = self.inner.embed_texts(texts, mode)?;
        for e in &mut embs {
            vector::l2_normalize_in_place(e);
        }
        Ok(embs)
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }

    fn dimension(&self) -> Option<usize> {
        self.inner.dimension()
    }

    fn capabilities(&self) -> TextEmbedderCapabilities {
        let mut caps = self.inner.capabilities();
        caps.normalization = Normalization::L2Normalized;
        caps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationPolicy {
    /// Keep backend behavior.
    Preserve,
    /// Ensure outputs are L2-normalized (wrap if needed).
    RequireL2,
}

pub fn apply_normalization_policy<E: TextEmbedder + 'static>(
    inner: E,
    policy: NormalizationPolicy,
) -> anyhow::Result<Box<dyn TextEmbedder>> {
    let caps = inner.capabilities();
    match policy {
        NormalizationPolicy::Preserve => Ok(Box::new(inner)),
        NormalizationPolicy::RequireL2 => {
            if caps.normalization == Normalization::L2Normalized {
                Ok(Box::new(inner))
            } else {
                Ok(Box::new(L2NormalizedTextEmbedder::new(inner)))
            }
        }
    }
}

/// Wrapper that truncates output vectors to the first `dim` dimensions.
///
/// This matches the common “truncate_dim” / “dimensions” knob (e.g. SentenceTransformers, TEI).
#[derive(Debug, Clone)]
pub struct TruncateDimTextEmbedder<E> {
    inner: E,
    dim: usize,
}

impl<E> TruncateDimTextEmbedder<E> {
    pub fn new(inner: E, dim: usize) -> anyhow::Result<Self> {
        if dim == 0 {
            return Err(anyhow::anyhow!("truncate dim must be > 0"));
        }
        Ok(Self { inner, dim })
    }
}

impl<E: TextEmbedder> TextEmbedder for TruncateDimTextEmbedder<E> {
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut embs = self.inner.embed_texts(texts, mode)?;
        for v in &mut embs {
            if v.len() < self.dim {
                return Err(anyhow::anyhow!(
                    "truncate dim {} exceeds embedding dim {}",
                    self.dim,
                    v.len()
                ));
            }
            v.truncate(self.dim);
        }
        Ok(embs)
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn capabilities(&self) -> TextEmbedderCapabilities {
        // This wrapper doesn't claim anything about prompt/truncation tokenization behavior.
        self.inner.capabilities()
    }
}

pub fn apply_output_dim<E: TextEmbedder + 'static>(
    inner: E,
    dim: Option<usize>,
) -> anyhow::Result<Box<dyn TextEmbedder>> {
    match dim {
        None => Ok(Box::new(inner)),
        Some(d) => Ok(Box::new(TruncateDimTextEmbedder::new(inner, d)?)),
    }
}

/// Minimal interface for “text → dense vector” encoders (bi-encoder style).
///
/// This covers the common “sentence embedding” family: one vector per input string.
///
/// **Important**: there are multiple *kinds* of “embeddings” used in retrieval:
///
/// - **Dense sentence embeddings** (this trait): one \(d\)-dim vector per string.
/// - **Dense token embeddings / late interaction** (e.g. ColBERT): many vectors per string.
/// - **Sparse embeddings** (e.g. SPLADE): a weighted sparse vector over vocabulary IDs.
/// - **Binary / quantized embeddings**: compressed representations for ANN speed/memory.
///
/// We start with the dense-sentence contract because it’s the smallest stable surface that
/// many parts of the workspace can share (iksh, chunking, retrieval prototypes).
pub trait TextEmbedder: Send + Sync {
    /// Embed texts into vectors.
    ///
    /// Recommended invariant for backends that can afford it: **L2-normalize** outputs so
    /// cosine similarity equals dot product and downstream fusion is less brittle.
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Optional: model identifier for debugging and provenance.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Optional: embedding dimension **as returned by this embedder**.
    ///
    /// Notes:
    /// - If you wrap an embedder with `apply_output_dim(Some(d))`, this should become `Some(d)`.
    /// - Many remote backends can't report this without a request; returning `None` is fine.
    fn dimension(&self) -> Option<usize> {
        None
    }

    /// Optional: backend capability declaration.
    ///
    /// This exists to design around “silent drift” failure modes:
    /// - prompt applied client-side vs server-side vs internally
    /// - whether `EmbedMode` is actually used
    /// - whether outputs are L2-normalized
    ///
    /// Backends that don't override this should be treated as "unknown" by callers.
    fn capabilities(&self) -> TextEmbedderCapabilities {
        TextEmbedderCapabilities::unknown()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    Unknown,
    L2Normalized,
    NotNormalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationDirection {
    Unknown,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationPolicy {
    Unknown,
    None,
    Truncate {
        max_len: Option<usize>,
        direction: TruncationDirection,
    },
}

/// Where prompt/scoping is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptApplication {
    Unknown,
    None,
    /// Prefix applied in the client before sending to the model/backend.
    ClientPrefix,
    /// Prompt selection is delegated to the backend/server (e.g. TEI `prompt_name`).
    ServerPromptName,
    /// Backend applies scope/prompt internally (e.g. local HF embedder uses `EmbedMode`).
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEmbedderCapabilities {
    pub uses_embed_mode: PromptApplication,
    pub normalization: Normalization,
    pub truncation: TruncationPolicy,
}

impl TextEmbedderCapabilities {
    pub const fn unknown() -> Self {
        Self {
            uses_embed_mode: PromptApplication::Unknown,
            normalization: Normalization::Unknown,
            truncation: TruncationPolicy::Unknown,
        }
    }
}

/// Declarative scoping/prompt policy for a text embedder call site.
///
/// This is intentionally small: it exists to prevent the two most common regression classes:
/// 1) **double prompting** (client prefix + server/internal prompt)
/// 2) **silent mode ignore** (caller thinks Query/Document scope matters, but backend ignores it)
#[derive(Debug, Clone)]
pub enum ScopingPolicy {
    /// No scoping/prompting beyond what the backend does by default.
    None,
    /// Apply a client-side prefix template based on `EmbedMode`.
    ClientPrefix(PromptTemplate),
    /// Require that the backend applies scope via server-side prompt selection (e.g. TEI `prompt_name`).
    RequireServerPromptName,
    /// Require that the backend applies scope internally (e.g. local embedder with per-mode prompt fields).
    RequireInternal,
}

/// Apply a `ScopingPolicy` to an embedder, returning a boxed embedder.
///
/// Note: this only *wraps* for `ClientPrefix`. The other policies are validation gates.
pub fn apply_scoping_policy<E: TextEmbedder + 'static>(
    inner: E,
    policy: ScopingPolicy,
) -> anyhow::Result<Box<dyn TextEmbedder>> {
    let caps = inner.capabilities();
    match policy {
        ScopingPolicy::None => Ok(Box::new(inner)),
        ScopingPolicy::ClientPrefix(prompt) => {
            // Disallow obvious double-prompting.
            match caps.uses_embed_mode {
                PromptApplication::ServerPromptName | PromptApplication::Internal => {
                    return Err(anyhow::anyhow!(
                        "scoping policy ClientPrefix conflicts with backend prompt application {:?}",
                        caps.uses_embed_mode
                    ));
                }
                _ => {}
            }
            Ok(Box::new(PromptedTextEmbedder::new(inner, prompt)))
        }
        ScopingPolicy::RequireServerPromptName => {
            if caps.uses_embed_mode != PromptApplication::ServerPromptName {
                return Err(anyhow::anyhow!(
                    "expected ServerPromptName scoping, but backend reports {:?}",
                    caps.uses_embed_mode
                ));
            }
            Ok(Box::new(inner))
        }
        ScopingPolicy::RequireInternal => {
            if caps.uses_embed_mode != PromptApplication::Internal {
                return Err(anyhow::anyhow!(
                    "expected Internal scoping, but backend reports {:?}",
                    caps.uses_embed_mode
                ));
            }
            Ok(Box::new(inner))
        }
    }
}

// Ergonomics: allow trait objects (`Box<dyn TextEmbedder>`) to be used wherever a `TextEmbedder` is expected.
impl<T: TextEmbedder + ?Sized> TextEmbedder for Box<T> {
    fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).embed_texts(texts, mode)
    }

    fn model_id(&self) -> Option<&str> {
        (**self).model_id()
    }

    fn dimension(&self) -> Option<usize> {
        (**self).dimension()
    }

    fn capabilities(&self) -> TextEmbedderCapabilities {
        (**self).capabilities()
    }
}

/// Optional extension trait for “multi-vector” (late-interaction) embeddings.
///
/// Shape: `batch -> tokens -> dim`.
pub trait TokenEmbedder: Send + Sync {
    fn embed_tokens(&self, texts: &[String], mode: EmbedMode)
        -> anyhow::Result<Vec<Vec<Vec<f32>>>>;
}

/// Optional extension trait for sparse lexical embeddings.
///
/// Each string becomes a sparse vector: `(term_id, weight)` pairs.
pub trait SparseEmbedder: Send + Sync {
    fn embed_sparse(
        &self,
        texts: &[String],
        mode: EmbedMode,
    ) -> anyhow::Result<Vec<Vec<(u32, f32)>>>;
}

/// Minimal image embedder interface (bytes -> vectors).
///
/// `images` are opaque bytes (e.g. JPEG/PNG). Format detection/decoding is backend-specific.
pub trait ImageEmbedder: Send + Sync {
    fn embed_images(&self, images: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<f32>>>;

    fn model_id(&self) -> Option<&str> {
        None
    }
}

/// Minimal audio embedder interface (bytes -> vectors).
///
/// `audios` are opaque bytes (e.g. WAV/MP3). Decoding/resampling is backend-specific.
pub trait AudioEmbedder: Send + Sync {
    fn embed_audios(&self, audios: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<f32>>>;

    fn model_id(&self) -> Option<&str> {
        None
    }
}

// --- Optional backends (feature-gated) ---
//
// Goal: `embedd` is the only public crate name. Backends live behind features,
// not as separate publishable crates.

#[cfg(feature = "hf-inference")]
pub mod hf_inference {
    use super::{AudioEmbedder, ImageEmbedder, TextEmbedder};
    use anyhow::Result;

    /// HuggingFace Inference API “feature-extraction” embedder.
    ///
    /// This is a **real** multimodal e2e path (network), useful for:
    /// - quickly smoke-testing text/image/audio modality plumbing
    /// - comparing remote models without pulling weights locally
    ///
    /// Auth: set `HF_API_TOKEN` env var, or pass the token explicitly.
    ///
    /// Endpoint shape is assumed to be `POST https://api-inference.huggingface.co/models/{model}`.
    /// The response is expected to be numeric arrays (often nested). We reduce to a single vector
    /// via mean pooling over leading dimensions.
    #[derive(Debug, Clone)]
    pub struct HfInferenceEmbedder {
        base_url: String,
        token: Option<String>,
        model: String,
        client: ureq::Agent,
    }

    impl HfInferenceEmbedder {
        pub fn new(model: impl Into<String>) -> Self {
            Self::new_with_base_url(model, "https://api-inference.huggingface.co")
        }

        pub fn new_with_base_url(model: impl Into<String>, base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into().trim_end_matches('/').to_string(),
                token: std::env::var("HF_API_TOKEN").ok(),
                model: model.into(),
                client: ureq::AgentBuilder::new().build(),
            }
        }

        pub fn with_token(mut self, token: impl Into<String>) -> Self {
            self.token = Some(token.into());
            self
        }

        pub fn model(&self) -> &str {
            &self.model
        }

        fn endpoint(&self) -> String {
            format!("{}/models/{}", self.base_url, self.model)
        }

        fn auth_header_value(&self) -> Option<String> {
            self.token.as_ref().map(|t| format!("Bearer {t}"))
        }

        fn post_json<T: serde::Serialize>(&self, payload: &T) -> Result<serde_json::Value> {
            let mut req = self
                .client
                .post(&self.endpoint())
                .set("Content-Type", "application/json");
            if let Some(auth) = self.auth_header_value() {
                req = req.set("Authorization", &auth);
            }

            let resp = req.send_string(&serde_json::to_string(payload)?)?;
            let status = resp.status();
            let body = resp.into_string().unwrap_or_default();
            if !(200..300).contains(&status) {
                return Err(anyhow::anyhow!(
                    "hf-inference failed: status={} body={}",
                    status,
                    body
                ));
            }
            Ok(serde_json::from_str(&body)?)
        }

        fn post_bytes(&self, bytes: &[u8]) -> Result<serde_json::Value> {
            let mut req = self
                .client
                .post(&self.endpoint())
                .set("Content-Type", "application/octet-stream");
            if let Some(auth) = self.auth_header_value() {
                req = req.set("Authorization", &auth);
            }

            // Note: HF accepts raw bytes for image/audio feature extraction models.
            let resp = req.send_bytes(bytes)?;
            let status = resp.status();
            let body = resp.into_string().unwrap_or_default();
            if !(200..300).contains(&status) {
                return Err(anyhow::anyhow!(
                    "hf-inference failed: status={} body={}",
                    status,
                    body
                ));
            }
            Ok(serde_json::from_str(&body)?)
        }
    }

    fn mean_pool_to_vec(v: &serde_json::Value) -> Result<Vec<f32>> {
        // Accept shapes like:
        // - [d]
        // - [[d]]
        // - [[[d]]] (token/patch embeddings)
        // and mean-pool all but the last dimension.
        fn flatten_numbers(v: &serde_json::Value, out: &mut Vec<f64>) -> Result<()> {
            match v {
                serde_json::Value::Number(n) => {
                    out.push(
                        n.as_f64()
                            .ok_or_else(|| anyhow::anyhow!("non-f64 number"))?,
                    );
                    Ok(())
                }
                serde_json::Value::Array(xs) => {
                    for x in xs {
                        flatten_numbers(x, out)?;
                    }
                    Ok(())
                }
                _ => Err(anyhow::anyhow!("expected numeric arrays, got: {}", v)),
            }
        }

        // Strategy:
        // - If it’s a flat vector already, return it.
        // - Otherwise: interpret the last dimension as `d`, average across leading elements.
        fn as_vec_f32(v: &serde_json::Value) -> Option<Vec<f32>> {
            match v {
                serde_json::Value::Array(xs) if xs.iter().all(|x| x.is_number()) => xs
                    .iter()
                    .map(|x| x.as_f64().map(|f| f as f32))
                    .collect::<Option<Vec<f32>>>(),
                _ => None,
            }
        }

        if let Some(vec) = as_vec_f32(v) {
            return Ok(vec);
        }

        // Try to treat v as nested arrays whose leaves are vectors of the same dimension.
        fn collect_leaf_vectors(v: &serde_json::Value, leaves: &mut Vec<Vec<f32>>) -> Result<()> {
            if let Some(vec) = as_vec_f32(v) {
                leaves.push(vec);
                return Ok(());
            }
            match v {
                serde_json::Value::Array(xs) => {
                    for x in xs {
                        collect_leaf_vectors(x, leaves)?;
                    }
                    Ok(())
                }
                _ => Err(anyhow::anyhow!("expected numeric arrays, got: {}", v)),
            }
        }

        let mut leaves = Vec::new();
        collect_leaf_vectors(v, &mut leaves)?;
        if leaves.is_empty() {
            // fallback: full flatten (shouldn’t happen for feature-extraction, but keep a clear error)
            let mut flat = Vec::new();
            flatten_numbers(v, &mut flat)?;
            return Ok(flat.into_iter().map(|x| x as f32).collect());
        }

        let d = leaves[0].len();
        if d == 0 {
            return Err(anyhow::anyhow!("zero-dim embedding"));
        }
        if !leaves.iter().all(|e| e.len() == d) {
            return Err(anyhow::anyhow!(
                "inconsistent embedding dims in HF response"
            ));
        }

        let mut acc = vec![0.0f32; d];
        for e in &leaves {
            for (i, x) in e.iter().enumerate() {
                acc[i] += *x;
            }
        }
        let n = leaves.len() as f32;
        for x in &mut acc {
            *x /= n;
        }
        Ok(acc)
    }

    impl TextEmbedder for HfInferenceEmbedder {
        fn embed_texts(&self, texts: &[String], _mode: super::EmbedMode) -> Result<Vec<Vec<f32>>> {
            let payload = serde_json::json!({ "inputs": texts });
            let v = self.post_json(&payload)?;
            // Response is typically: Vec<Vec<f32>> (batch -> vector), but can vary.
            // We accept:
            // - [ [d], [d], ... ]
            // - [d] (single input) -> wrap
            match v {
                serde_json::Value::Array(items) => {
                    // If items are numbers, treat as single embedding vector.
                    if items.iter().all(|x| x.is_number()) {
                        return Ok(vec![mean_pool_to_vec(&serde_json::Value::Array(items))?]);
                    }
                    // Otherwise, each item should be a (possibly nested) embedding.
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(mean_pool_to_vec(&item)?);
                    }
                    Ok(out)
                }
                other => Err(anyhow::anyhow!("unexpected HF response: {}", other)),
            }
        }

        fn model_id(&self) -> Option<&str> {
            Some(&self.model)
        }

        fn capabilities(&self) -> super::TextEmbedderCapabilities {
            super::TextEmbedderCapabilities {
                uses_embed_mode: super::PromptApplication::None,
                normalization: super::Normalization::Unknown,
                truncation: super::TruncationPolicy::Unknown,
            }
        }
    }

    impl ImageEmbedder for HfInferenceEmbedder {
        fn embed_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(images.len());
            for img in images {
                let v = self.post_bytes(img)?;
                out.push(mean_pool_to_vec(&v)?);
            }
            Ok(out)
        }

        fn model_id(&self) -> Option<&str> {
            Some(&self.model)
        }
    }

    impl AudioEmbedder for HfInferenceEmbedder {
        fn embed_audios(&self, audios: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(audios.len());
            for a in audios {
                let v = self.post_bytes(a)?;
                out.push(mean_pool_to_vec(&v)?);
            }
            Ok(out)
        }

        fn model_id(&self) -> Option<&str> {
            Some(&self.model)
        }
    }
}

#[cfg(feature = "openai")]
pub mod openai {
    use super::{EmbedMode, TextEmbedder};
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone)]
    pub struct OpenAiEmbedder {
        base_url: String,
        api_key: String,
        model: String,
        client: ureq::Agent,
    }

    impl OpenAiEmbedder {
        pub fn new(
            base_url: impl Into<String>,
            api_key: impl Into<String>,
            model: impl Into<String>,
        ) -> Self {
            Self {
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: api_key.into(),
                model: model.into(),
                client: ureq::AgentBuilder::new().build(),
            }
        }

        fn embeddings_endpoint(&self) -> String {
            format!("{}/v1/embeddings", self.base_url)
        }
    }

    #[derive(Debug, Serialize)]
    struct EmbeddingsRequest<'a> {
        model: &'a str,
        input: &'a [String],
    }

    #[derive(Debug, Deserialize)]
    struct EmbeddingsResponse {
        data: Vec<EmbeddingDatum>,
    }

    #[derive(Debug, Deserialize)]
    struct EmbeddingDatum {
        embedding: Vec<f32>,
    }

    impl TextEmbedder for OpenAiEmbedder {
        fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
            let payload = EmbeddingsRequest {
                model: &self.model,
                input: texts,
            };

            let resp = self
                .client
                .post(&self.embeddings_endpoint())
                .set("Content-Type", "application/json")
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .send_string(&serde_json::to_string(&payload)?)?;

            let status = resp.status();
            if status < 200 || status >= 300 {
                let body = resp.into_string().unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "openai embeddings failed: status={} body={}",
                    status,
                    body
                ));
            }

            let body = resp.into_string()?;
            let parsed: EmbeddingsResponse = serde_json::from_str(&body)?;
            Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
        }

        fn model_id(&self) -> Option<&str> {
            Some(&self.model)
        }

        fn capabilities(&self) -> super::TextEmbedderCapabilities {
            super::TextEmbedderCapabilities {
                uses_embed_mode: super::PromptApplication::None,
                normalization: super::Normalization::Unknown,
                truncation: super::TruncationPolicy::Unknown,
            }
        }
    }
}

#[cfg(feature = "tei")]
pub mod tei {
    use super::{
        EmbedMode, PromptApplication, TextEmbedder, TextEmbedderCapabilities, TruncationDirection,
        TruncationPolicy,
    };
    use anyhow::Result;
    use serde_json::Value;

    /// TEI client embedder.
    #[derive(Debug, Clone)]
    pub struct TeiEmbedder {
        base_url: String,
        api_key: Option<String>,
        prompt_name_query: Option<String>,
        prompt_name_doc: Option<String>,
        dimensions: Option<usize>,
        normalize: Option<bool>,
        truncate: Option<bool>,
        truncation_direction: Option<String>,
        client: ureq::Agent,
    }

    impl TeiEmbedder {
        /// Create a new TEI client targeting `base_url` (e.g. `http://127.0.0.1:8080`).
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: None,
                prompt_name_query: None,
                prompt_name_doc: None,
                dimensions: None,
                normalize: None,
                truncate: None,
                truncation_direction: None,
                client: ureq::AgentBuilder::new().build(),
            }
        }

        /// Set a TEI API key (TEI supports an `--api-key` flag; this sends `Authorization: Bearer <key>`).
        pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
            self.api_key = Some(api_key.into());
            self
        }

        /// Configure TEI `prompt_name` per scope (`EmbedMode`).
        ///
        /// TEI applies prompts server-side based on the model’s SentenceTransformers `prompts` config.
        pub fn with_prompt_names(
            mut self,
            query_prompt_name: impl Into<String>,
            doc_prompt_name: impl Into<String>,
        ) -> Self {
            self.prompt_name_query = Some(query_prompt_name.into());
            self.prompt_name_doc = Some(doc_prompt_name.into());
            self
        }

        /// TEI request option: force output embedding dimension (TEI `dimensions`).
        pub fn with_dimensions(mut self, dimensions: usize) -> Self {
            self.dimensions = Some(dimensions);
            self
        }

        /// TEI request option: whether to L2-normalize outputs (TEI defaults `true`).
        pub fn with_normalize(mut self, normalize: bool) -> Self {
            self.normalize = Some(normalize);
            self
        }

        /// TEI request option: whether to truncate inputs instead of erroring (TEI defaults `false`).
        pub fn with_truncate(mut self, truncate: bool) -> Self {
            self.truncate = Some(truncate);
            self
        }

        /// TEI request option: truncation direction ("left" or "right"; TEI defaults "right").
        pub fn with_truncation_direction(mut self, dir: impl Into<String>) -> Self {
            self.truncation_direction = Some(dir.into());
            self
        }

        fn prompt_name_for_mode(&self, mode: EmbedMode) -> Option<&str> {
            match mode {
                EmbedMode::Query => self.prompt_name_query.as_deref(),
                EmbedMode::Document => self.prompt_name_doc.as_deref(),
            }
        }

        fn embed_endpoint(&self) -> String {
            format!("{}/embed", self.base_url)
        }

        fn build_embed_payload(&self, texts: &[String], mode: EmbedMode) -> Value {
            let mut payload = serde_json::Map::new();
            payload.insert("inputs".to_string(), serde_json::Value::from(texts));
            if let Some(d) = self.dimensions {
                payload.insert("dimensions".to_string(), serde_json::Value::from(d as u64));
            }
            if let Some(pn) = self.prompt_name_for_mode(mode) {
                payload.insert("prompt_name".to_string(), serde_json::Value::from(pn));
            }
            if let Some(n) = self.normalize {
                payload.insert("normalize".to_string(), serde_json::Value::from(n));
            }
            if let Some(t) = self.truncate {
                payload.insert("truncate".to_string(), serde_json::Value::from(t));
            }
            if let Some(dir) = self.truncation_direction.as_deref() {
                payload.insert(
                    "truncation_direction".to_string(),
                    serde_json::Value::from(dir),
                );
            }
            Value::Object(payload)
        }
    }

    impl TextEmbedder for TeiEmbedder {
        fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
            let payload = self.build_embed_payload(texts, mode);

            let mut req = self
                .client
                .post(&self.embed_endpoint())
                .set("Content-Type", "application/json");
            if let Some(k) = &self.api_key {
                req = req.set("Authorization", &format!("Bearer {k}"));
            }
            let resp = req.send_string(&payload.to_string())?;

            let status = resp.status();
            if status < 200 || status >= 300 {
                let body = resp.into_string().unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "tei /embed failed: status={} body={}",
                    status,
                    body
                ));
            }

            let body = resp.into_string()?;
            let embs: Vec<Vec<f32>> = serde_json::from_str(&body)?;
            Ok(embs)
        }

        fn dimension(&self) -> Option<usize> {
            self.dimensions
        }

        fn capabilities(&self) -> TextEmbedderCapabilities {
            // TEI can apply prompts server-side if `prompt_name_*` is configured; otherwise it ignores mode.
            let uses = if self.prompt_name_query.is_some() || self.prompt_name_doc.is_some() {
                PromptApplication::ServerPromptName
            } else {
                PromptApplication::None
            };
            let truncation = match self.truncate {
                Some(true) => TruncationPolicy::Truncate {
                    max_len: None,
                    direction: match self.truncation_direction.as_deref() {
                        Some("left") => TruncationDirection::Left,
                        Some("right") => TruncationDirection::Right,
                        Some(_) => TruncationDirection::Unknown,
                        None => TruncationDirection::Right,
                    },
                },
                _ => TruncationPolicy::None,
            };
            let normalization = match self.normalize {
                Some(true) => super::Normalization::L2Normalized,
                Some(false) => super::Normalization::NotNormalized,
                None => super::Normalization::Unknown,
            };
            TextEmbedderCapabilities {
                uses_embed_mode: uses,
                normalization,
                truncation,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::TeiEmbedder;
        use crate::{
            EmbedMode, Normalization, PromptApplication, TruncationDirection, TruncationPolicy,
        };

        fn obj(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
            v.as_object().cloned().expect("expected object")
        }

        #[test]
        fn tei_payload_includes_prompt_name_per_mode() {
            let e = TeiEmbedder::new("http://127.0.0.1:8080").with_prompt_names("q", "d");
            let xs = vec!["hello".to_string()];

            let q = obj(e.build_embed_payload(&xs, EmbedMode::Query));
            assert_eq!(q.get("prompt_name").and_then(|v| v.as_str()), Some("q"));

            let d = obj(e.build_embed_payload(&xs, EmbedMode::Document));
            assert_eq!(d.get("prompt_name").and_then(|v| v.as_str()), Some("d"));
        }

        #[test]
        fn tei_payload_includes_dimensions_normalize_truncate_fields_when_set() {
            let e = TeiEmbedder::new("http://127.0.0.1:8080")
                .with_dimensions(256)
                .with_normalize(false)
                .with_truncate(true)
                .with_truncation_direction("left");
            let xs = vec!["hello".to_string(), "world".to_string()];
            let p = obj(e.build_embed_payload(&xs, EmbedMode::Query));

            assert_eq!(p.get("dimensions").and_then(|v| v.as_u64()), Some(256));
            assert_eq!(p.get("normalize").and_then(|v| v.as_bool()), Some(false));
            assert_eq!(p.get("truncate").and_then(|v| v.as_bool()), Some(true));
            assert_eq!(
                p.get("truncation_direction").and_then(|v| v.as_str()),
                Some("left")
            );
        }

        #[test]
        fn tei_capabilities_reflect_prompt_and_truncation_and_normalization() {
            let e = TeiEmbedder::new("http://127.0.0.1:8080")
                .with_prompt_names("query", "doc")
                .with_normalize(true)
                .with_truncate(true)
                .with_truncation_direction("right");
            let caps = e.capabilities();

            assert_eq!(caps.uses_embed_mode, PromptApplication::ServerPromptName);
            assert_eq!(caps.normalization, Normalization::L2Normalized);
            assert_eq!(
                caps.truncation,
                TruncationPolicy::Truncate {
                    max_len: None,
                    direction: TruncationDirection::Right
                }
            );
        }
    }
}

#[cfg(feature = "fastembed")]
pub mod fastembed {
    use super::{EmbedMode, TextEmbedder};
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

    /// Global model cache to prevent multiple initializations.
    ///
    /// Rationale:
    /// - Some `fastembed` backends are FFI-backed and can misbehave if repeatedly initialized/dropped.
    /// - For a tool-heavy workspace (tests, benches), “drop order” is hard to reason about.
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
                // fastembed requires a non-empty batch
                let out = guard
                    .embed(vec!["probe"], None)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                out.first().map(|v| v.len()).unwrap_or(0)
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

            let model = TextEmbedding::try_new(
                InitOptions::new(model_name).with_show_download_progress(false),
            )
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

        fn capabilities(&self) -> super::TextEmbedderCapabilities {
            super::TextEmbedderCapabilities {
                uses_embed_mode: super::PromptApplication::None,
                normalization: super::Normalization::Unknown,
                truncation: super::TruncationPolicy::Unknown,
            }
        }
    }
}

#[cfg(feature = "ort-tokenizers")]
pub mod ort {
    use super::{EmbedMode, TextEmbedder};
    use anyhow::Result;

    /// Placeholder embedder for ORT-backed models.
    pub struct OrtEmbedder;

    impl OrtEmbedder {
        pub fn new() -> Self {
            Self
        }
    }

    impl TextEmbedder for OrtEmbedder {
        fn embed_texts(&self, _texts: &[String], _mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
            Err(anyhow::anyhow!(
                "embedd::ort: not implemented yet (need model/tokenizer loading conventions)"
            ))
        }
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn {
    /// Placeholder burn embedder.
    pub struct BurnEmbedder;

    impl BurnEmbedder {
        pub fn new() -> Self {
            Self
        }
    }

    impl super::TextEmbedder for BurnEmbedder {
        fn embed_texts(
            &self,
            _texts: &[String],
            _mode: super::EmbedMode,
        ) -> anyhow::Result<Vec<Vec<f32>>> {
            Err(anyhow::anyhow!("embedd::burn: not implemented yet"))
        }
    }
}

#[cfg(feature = "siglip")]
pub mod siglip {
    use super::ImageEmbedder;

    /// Placeholder SigLIP embedder.
    pub struct SiglipEmbedder;

    impl SiglipEmbedder {
        pub fn new() -> Self {
            Self
        }
    }

    impl ImageEmbedder for SiglipEmbedder {
        fn embed_images(&self, _images: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<f32>>> {
            Err(anyhow::anyhow!("embedd(siglip): not implemented yet"))
        }
    }
}

/// Configuration of where embedding model artifacts come from.
///
/// This is intentionally minimal and testable: it just answers “local dir or hub id?”.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Load from a local directory containing `config.json`, `tokenizer.json`, `model.safetensors`.
    LocalDir(std::path::PathBuf),
    /// Load via HuggingFace Hub model id (downloaded/cached by `hf-hub`).
    HuggingFaceModelId(String),
}

impl ModelSource {
    /// Resolve from the legacy `iksh` environment variables.
    ///
    /// Priority:
    /// 1) `IKSH_EMBED_MODEL_DIR` (local, no network)
    /// 2) `IKSH_EMBED_MODEL` (hub model id)
    pub fn from_iksh_env() -> Self {
        if let Ok(dir) = std::env::var("IKSH_EMBED_MODEL_DIR") {
            return Self::LocalDir(std::path::PathBuf::from(dir));
        }
        let model_id = std::env::var("IKSH_EMBED_MODEL")
            .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
        Self::HuggingFaceModelId(model_id)
    }
}

/// Safetensors validation utilities.
///
/// This is a “native safety rail” inspired by:
/// - HuggingFace `safetensors` (Rust) invariants (bounds, contiguity, overflow checks)
/// - tinygrad’s minimal `safe_load` implementation (header length + JSON + offsets)
pub mod safetensors {
    use anyhow::Result;
    use serde_json::Value;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    /// Hard cap to avoid allocating absurd headers.
    ///
    /// HF safetensors uses 100MB; we keep the same default.
    pub const MAX_HEADER_SIZE: usize = 100_000_000;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TensorInfo {
        pub dtype: String,
        pub shape: Vec<usize>,
        /// (begin, end) offsets in bytes, relative to the data buffer start.
        pub data_offsets: (usize, usize),
    }

    fn dtype_size_bytes(dtype: &str) -> Option<usize> {
        // Conservative: cover common safetensors dtypes. Unknown dtypes fail validation.
        Some(match dtype {
            "BOOL" => 1,
            "U8" => 1,
            "I8" => 1,
            "U16" => 2,
            "I16" => 2,
            "F16" => 2,
            "BF16" => 2,
            "U32" => 4,
            "I32" => 4,
            "F32" => 4,
            "U64" => 8,
            "I64" => 8,
            "F64" => 8,
            _ => return None,
        })
    }

    fn checked_numel(shape: &[usize]) -> Option<usize> {
        let mut n: usize = 1;
        for &d in shape {
            n = n.checked_mul(d)?;
        }
        Some(n)
    }

    fn parse_usize(v: &Value) -> Option<usize> {
        v.as_u64().and_then(|x| usize::try_from(x).ok())
    }

    fn parse_tensor_info(v: &Value) -> Result<TensorInfo> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("tensor entry must be an object"))?;

        let dtype = obj
            .get("dtype")
            .and_then(|x| x.as_str())
            .ok_or_else(|| anyhow::anyhow!("tensor entry missing dtype"))?
            .to_string();

        let shape = obj
            .get("shape")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("tensor entry missing shape"))?
            .iter()
            .map(|x| parse_usize(x).ok_or_else(|| anyhow::anyhow!("shape must be u64 array")))
            .collect::<Result<Vec<usize>>>()?;

        let offs = obj
            .get("data_offsets")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("tensor entry missing data_offsets"))?;
        if offs.len() != 2 {
            return Err(anyhow::anyhow!("data_offsets must have length 2"));
        }
        let begin = parse_usize(&offs[0]).ok_or_else(|| anyhow::anyhow!("offset begin invalid"))?;
        let end = parse_usize(&offs[1]).ok_or_else(|| anyhow::anyhow!("offset end invalid"))?;

        Ok(TensorInfo {
            dtype,
            shape,
            data_offsets: (begin, end),
        })
    }

    /// Validate a safetensors header JSON blob against a given data buffer length.
    ///
    /// This checks:
    /// - header is valid JSON object
    /// - each tensor has known dtype, shape, and offsets
    /// - per-tensor byte size matches dtype*shape
    /// - offsets are within bounds, non-overlapping, and contiguous (no holes)
    /// - final end offset equals `data_len` (full coverage, no trailing junk)
    pub fn validate_header_and_data_len(header_bytes: &[u8], data_len: usize) -> Result<()> {
        if header_bytes.is_empty() {
            return Err(anyhow::anyhow!("header is empty"));
        }
        if header_bytes[0] != b'{' {
            return Err(anyhow::anyhow!("header must start with '{{'"));
        }

        // safetensors allows trailing whitespace padding.
        let mut trimmed = header_bytes;
        while trimmed.last() == Some(&b' ') {
            trimmed = &trimmed[..trimmed.len() - 1];
        }

        let root: Value = serde_json::from_slice(trimmed)
            .map_err(|e| anyhow::anyhow!("invalid header JSON: {e}"))?;
        let obj = root
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("header must be a JSON object"))?;

        // Collect tensor infos and sort by begin offset.
        //
        // Note: empty tensors (0 bytes) are allowed by safetensors. That means multiple tensors
        // may legitimately share the same `(begin=end)` offset. We therefore cannot key by
        // `begin` alone.
        let mut tensors: Vec<(usize, usize, String, TensorInfo)> = Vec::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                // safetensors restricts metadata to string->string, but we treat it as opaque.
                continue;
            }
            let info = parse_tensor_info(v)?;

            // dtype must be known
            let Some(elt) = dtype_size_bytes(&info.dtype) else {
                return Err(anyhow::anyhow!("unknown dtype: {}", info.dtype));
            };

            // offsets must be ordered and within buffer
            let (begin, end) = info.data_offsets;
            if begin > end {
                return Err(anyhow::anyhow!("invalid offsets (begin > end) for {k}"));
            }
            if end > data_len {
                return Err(anyhow::anyhow!("offset end out of bounds for {k}"));
            }

            // dtype*shape must match byte length, with overflow checks.
            let Some(numel) = checked_numel(&info.shape) else {
                return Err(anyhow::anyhow!("overflow computing numel for {k}"));
            };
            let Some(expected_bytes) = numel.checked_mul(elt) else {
                return Err(anyhow::anyhow!("overflow computing byte size for {k}"));
            };
            let actual_bytes = end.saturating_sub(begin);
            if expected_bytes != actual_bytes {
                return Err(anyhow::anyhow!(
                    "tensor byte size mismatch for {k}: expected {expected_bytes} got {actual_bytes}"
                ));
            }

            tensors.push((begin, end, k.clone(), info));
        }

        // Contiguity: 0..data_len must be fully covered without holes/overlaps.
        let mut cursor = 0usize;
        tensors.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (begin, end, name, _info) in tensors {
            if begin != cursor {
                return Err(anyhow::anyhow!(
                    "metadata does not fully cover buffer (hole or overlap): cursor={cursor} begin={begin} tensor={name}"
                ));
            }
            cursor = end;
        }
        if cursor != data_len {
            return Err(anyhow::anyhow!(
                "metadata does not fully cover buffer: end={cursor} data_len={data_len}"
            ));
        }

        Ok(())
    }

    /// Validate a safetensors file on disk (header + offsets).
    ///
    /// Reads the header only (bounded) and validates that offsets cover the data buffer.
    pub fn validate_file(path: &Path) -> Result<()> {
        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len();
        if file_len < 8 {
            return Err(anyhow::anyhow!("file too small for safetensors header"));
        }

        let mut n_bytes = [0u8; 8];
        f.read_exact(&mut n_bytes)?;
        let n = u64::from_le_bytes(n_bytes);
        let n_usize = usize::try_from(n).map_err(|_| anyhow::anyhow!("header length overflow"))?;
        if n_usize == 0 {
            return Err(anyhow::anyhow!("header length is zero"));
        }
        if n_usize > MAX_HEADER_SIZE {
            return Err(anyhow::anyhow!("header too large"));
        }

        let data_start = 8u64
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("header length overflow"))?;
        if data_start > file_len {
            return Err(anyhow::anyhow!("invalid header length (past end of file)"));
        }

        let mut header = vec![0u8; n_usize];
        f.read_exact(&mut header)?;

        let data_len_u64 = file_len - data_start;
        let data_len =
            usize::try_from(data_len_u64).map_err(|_| anyhow::anyhow!("data length overflow"))?;
        validate_header_and_data_len(&header, data_len)
    }
}

/// Vector post-processing helpers (L0-backed).
pub mod vector {
    use innr::{cosine, dot, norm, NORM_EPSILON};

    /// Compute the L2 norm of a vector.
    pub fn l2_norm(v: &[f32]) -> f32 {
        norm(v)
    }

    /// In-place L2 normalization.
    ///
    /// Returns the original norm. If the vector is near-zero (\(<\) `innr::NORM_EPSILON`),
    /// this is a no-op and returns 0.0.
    pub fn l2_normalize_in_place(v: &mut [f32]) -> f32 {
        let n = norm(v);
        if n <= NORM_EPSILON {
            return 0.0;
        }
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
        n
    }

    /// Dot product.
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        dot(a, b)
    }

    /// Cosine similarity (handles zero vectors).
    pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        cosine(a, b)
    }
}

#[cfg(feature = "candle-hf")]
mod candle_hf {
    use super::{EmbedMode, TextEmbedder};
    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use hf_hub::api::sync::Api as HfApi;
    use once_cell::sync::OnceCell;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;
    use tokenizers::Tokenizer;

    static EMBEDDER: OnceCell<std::sync::Mutex<LocalHfEmbedder>> = OnceCell::new();

    /// Local embedding inference via HuggingFace Hub + Candle (CPU).
    ///
    /// Note: this is the backend `iksh` currently uses.
    pub struct LocalHfEmbedder {
        pub model_id: String,
        tokenizer: Tokenizer,
        model: BertModel,
        device: Device,
        #[allow(dead_code)]
        hidden_dim: usize,
        query_prefix: String,
        doc_prefix: String,
        max_len: usize,
    }

    impl LocalHfEmbedder {
        fn load_from_paths(model_label: &str, dir: &Path) -> Result<Self> {
            let config_path = dir.join("config.json");
            let tok_path = dir.join("tokenizer.json");
            let weights_path = dir.join("model.safetensors");

            // Preflight: validate safetensors structure before mmap loading.
            super::safetensors::validate_file(&weights_path)?;

            let config: BertConfig =
                serde_json::from_reader(BufReader::new(File::open(config_path)?))?;
            let device = Device::Cpu;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
            };
            let model = BertModel::load(vb, &config)?;

            let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

            // Reasonable defaults; can be overridden.
            let max_len = std::env::var("IKSH_EMBED_MAX_LEN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(384)
                .min(2048);

            let prompt = super::PromptTemplate::from_iksh_env();

            Ok(Self {
                model_id: model_label.to_string(),
                tokenizer,
                model,
                device,
                hidden_dim: config.hidden_size,
                query_prefix: prompt.query_prefix,
                doc_prefix: prompt.doc_prefix,
                max_len,
            })
        }

        fn load(model_source: &super::ModelSource) -> Result<Self> {
            match model_source {
                super::ModelSource::LocalDir(dir) => {
                    Self::load_from_paths(&dir.to_string_lossy(), dir)
                }
                super::ModelSource::HuggingFaceModelId(model_id) => {
                    // HuggingFace Hub download (cached on disk by hf-hub).
                    let api = HfApi::new()?;
                    let repo = api.model(model_id.to_string());
                    let config_path = repo.get("config.json")?;
                    let tok_path = repo.get("tokenizer.json")?;
                    let weights_path = repo.get("model.safetensors")?;

                    // Preflight: validate safetensors structure before mmap loading.
                    super::safetensors::validate_file(&weights_path)?;

                    // Reuse the same loader via a temp “dir” abstraction: we just pass the parent.
                    // (We keep it simple: read config/tokenizer from their actual paths.)
                    let config: BertConfig =
                        serde_json::from_reader(BufReader::new(File::open(config_path)?))?;
                    let device = Device::Cpu;
                    let vb = unsafe {
                        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
                    };
                    let model = BertModel::load(vb, &config)?;

                    let tokenizer =
                        Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

                    let max_len = std::env::var("IKSH_EMBED_MAX_LEN")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(384)
                        .min(2048);

                    let prompt = super::PromptTemplate::from_iksh_env();

                    Ok(Self {
                        model_id: model_id.to_string(),
                        tokenizer,
                        model,
                        device,
                        hidden_dim: config.hidden_size,
                        query_prefix: prompt.query_prefix,
                        doc_prefix: prompt.doc_prefix,
                        max_len,
                    })
                }
            }
        }

        /// Singleton access (reuses model weights across calls).
        pub fn get() -> Result<std::sync::MutexGuard<'static, LocalHfEmbedder>> {
            let m = EMBEDDER.get_or_try_init(|| {
                Ok::<std::sync::Mutex<LocalHfEmbedder>, anyhow::Error>(std::sync::Mutex::new(
                    Self::load(&super::ModelSource::from_iksh_env())?,
                ))
            })?;
            Ok(m.lock().expect("embedder mutex poisoned"))
        }

        fn embed_texts_inner(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
            // Tokenize (with manual truncation + padding).
            let mut toks = Vec::with_capacity(texts.len());
            let mut masks = Vec::with_capacity(texts.len());
            let mut max_seq = 0usize;

            for t in texts {
                let prefix = if is_query {
                    &self.query_prefix
                } else {
                    &self.doc_prefix
                };
                let s = format!("{prefix}{t}");
                let enc = self
                    .tokenizer
                    .encode(s, true)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let mut ids: Vec<u32> = enc.get_ids().iter().copied().collect();
                let mut mask: Vec<u32> = enc.get_attention_mask().iter().copied().collect();

                if ids.len() > self.max_len {
                    ids.truncate(self.max_len);
                    mask.truncate(self.max_len);
                }
                max_seq = max_seq.max(ids.len());
                toks.push(ids);
                masks.push(mask);
            }

            // Pad.
            let pad_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0);
            for (ids, mask) in toks.iter_mut().zip(masks.iter_mut()) {
                while ids.len() < max_seq {
                    ids.push(pad_id);
                    mask.push(0);
                }
            }

            // Build tensors.
            let bsz = toks.len();
            let flat_ids: Vec<u32> = toks.into_iter().flatten().collect();
            let flat_mask: Vec<u32> = masks.into_iter().flatten().collect();
            let input_ids =
                Tensor::from_vec(flat_ids, (bsz, max_seq), &self.device)?.to_dtype(DType::I64)?;
            let attention_mask =
                Tensor::from_vec(flat_mask, (bsz, max_seq), &self.device)?.to_dtype(DType::I64)?;
            let token_type_ids = Tensor::zeros((bsz, max_seq), DType::I64, &self.device)?;

            // Forward: last hidden state [bsz, seq, hidden]
            let ys = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

            // Mean-pool with attention mask.
            let ys = ys.to_dtype(DType::F32)?;
            let mask_f = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
            let masked = ys.broadcast_mul(&mask_f)?;
            let sum = masked.sum(1)?;
            let denom = mask_f.sum(1)?.clamp(1e-6, f32::MAX)?;
            let pooled = sum.broadcast_div(&denom)?;

            // L2 normalize.
            let norms = pooled
                .sqr()?
                .sum(1)?
                .sqrt()?
                .clamp(1e-12, f32::MAX)?
                .unsqueeze(1)?;
            let normed = pooled.broadcast_div(&norms)?;

            Ok(normed.to_vec2()?)
        }
    }

    impl TextEmbedder for LocalHfEmbedder {
        fn embed_texts(&self, texts: &[String], mode: EmbedMode) -> Result<Vec<Vec<f32>>> {
            self.embed_texts_inner(texts, matches!(mode, EmbedMode::Query))
        }

        fn model_id(&self) -> Option<&str> {
            Some(&self.model_id)
        }

        fn dimension(&self) -> Option<usize> {
            Some(self.hidden_dim)
        }

        fn capabilities(&self) -> super::TextEmbedderCapabilities {
            super::TextEmbedderCapabilities {
                uses_embed_mode: super::PromptApplication::Internal,
                normalization: super::Normalization::L2Normalized,
                truncation: super::TruncationPolicy::Truncate {
                    max_len: Some(self.max_len),
                    direction: super::TruncationDirection::Right,
                },
            }
        }
    }
}

#[cfg(feature = "candle-hf")]
pub use candle_hf::LocalHfEmbedder;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn prompt_template_apply_starts_with_prefix() {
        let p = PromptTemplate {
            query_prefix: "Q: ".into(),
            doc_prefix: "D: ".into(),
        };
        assert_eq!(p.apply(EmbedMode::Query, "x"), "Q: x");
        assert_eq!(p.apply(EmbedMode::Document, "x"), "D: x");
    }

    proptest! {
        #[test]
        fn prompt_applies_as_prefix(qp in ".*", dp in ".*", t in ".*") {
            let p = PromptTemplate { query_prefix: qp.clone(), doc_prefix: dp.clone() };

            let q = p.apply(EmbedMode::Query, &t);
            prop_assert!(q.starts_with(&qp));
            prop_assert!(q.ends_with(&t));

            let d = p.apply(EmbedMode::Document, &t);
            prop_assert!(d.starts_with(&dp));
            prop_assert!(d.ends_with(&t));
        }
    }

    proptest! {
        #[test]
        fn normalize_then_cosine_is_dot_for_nonzero(
            a in prop::collection::vec(-10.0f32..10.0f32, 1..64),
            b in prop::collection::vec(-10.0f32..10.0f32, 1..64),
        ) {
            // Match lengths.
            let n = a.len().min(b.len());
            let mut a = a[..n].to_vec();
            let mut b = b[..n].to_vec();

            let na = vector::l2_normalize_in_place(&mut a);
            let nb = vector::l2_normalize_in_place(&mut b);
            prop_assume!(na > 0.0 && nb > 0.0);

            let d = vector::dot_f32(&a, &b);
            let c = vector::cosine_f32(&a, &b);
            prop_assert!((d - c).abs() < 1e-3);
        }
    }

    #[test]
    fn model_source_prefers_local_dir() {
        std::env::set_var("IKSH_EMBED_MODEL_DIR", "/tmp/somewhere");
        std::env::remove_var("IKSH_EMBED_MODEL");
        let src = ModelSource::from_iksh_env();
        assert!(matches!(src, ModelSource::LocalDir(_)));
        std::env::remove_var("IKSH_EMBED_MODEL_DIR");
    }

    #[test]
    fn safetensors_rejects_holes() {
        // One tensor covering 0..4, but data_len claims 8.
        let header = br#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let err = safetensors::validate_header_and_data_len(header, 8).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("does not fully cover buffer"));
    }

    proptest! {
        #[test]
        fn safetensors_accepts_contiguous_layout(
            // Up to 16 tensors, each 0..128 bytes.
            sizes in prop::collection::vec(0usize..128, 1..16),
        ) {
            // Build a contiguous layout with dtype=U8 and shape=[size].
            let mut offset = 0usize;
            let mut entries = Vec::new();
            for (i, sz) in sizes.iter().enumerate() {
                let begin = offset;
                let end = offset + *sz;
                offset = end;
                entries.push(format!(
                    "\"t{i}\":{{\"dtype\":\"U8\",\"shape\":[{sz}],\"data_offsets\":[{begin},{end}]}}"
                ));
            }
            let json = format!("{{{}}}", entries.join(","));
            safetensors::validate_header_and_data_len(json.as_bytes(), offset).unwrap();
        }
    }
}
