#![allow(dead_code)]

use embedd::{
    apply_normalization_policy, apply_output_dim, apply_scoping_policy, EmbedMode, Normalization,
    NormalizationPolicy, PromptApplication, PromptTemplate, ScopingPolicy, TextEmbedder,
    TruncationPolicy,
};
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-inference")]
use embedd::{AudioEmbedder, ImageEmbedder};

fn corpus_fallback() -> Vec<String> {
    vec!["Marie Curie discovered radium in Paris.".into()]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modality {
    Text,
    Image,
    Audio,
}

fn parse_modality() -> anyhow::Result<Modality> {
    let m = std::env::var("EMBEDD_MODALITY").unwrap_or_else(|_| "text".to_string());
    match m.as_str() {
        "text" => Ok(Modality::Text),
        "image" => Ok(Modality::Image),
        "audio" => Ok(Modality::Audio),
        other => Err(anyhow::anyhow!(
            "unknown EMBEDD_MODALITY={other} (use text|image|audio)"
        )),
    }
}

#[derive(Debug, Clone, Copy)]
enum CorpusSource {
    Default,
    Env,
    Fallback,
    Builtin,
}

#[derive(Debug, Clone)]
struct CorpusMeta {
    source: CorpusSource,
    // A display string intentionally kept stable-ish:
    // - default corpus: a relative path
    // - env corpus: whatever the user provided
    // - fallback: None
    path_display: Option<String>,
}

fn base64_decode(s: &str) -> Vec<u8> {
    fn val(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u8 = 0;
    for &b in bytes {
        if b == b'=' {
            break;
        }
        if b == b'\n' || b == b'\r' || b == b' ' || b == b'\t' {
            continue;
        }
        let v = match val(b) {
            Some(v) => v as u32,
            None => continue,
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    out
}

fn png_1x1_transparent() -> Vec<u8> {
    base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xc1cAAAAASUVORK5CYII=")
}

fn png_1x1_black() -> Vec<u8> {
    base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP8z8AABfsC/VE0l9QAAAAASUVORK5CYII=")
}

fn wav_silence_8khz_mono_u8(samples: usize) -> Vec<u8> {
    let sample_rate: u32 = 8_000;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 8;
    let byte_rate: u32 = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align: u16 = num_channels * (bits_per_sample / 8);
    let data_len: u32 = samples as u32;

    let riff_len: u32 = 4 /*WAVE*/ + 8 + 16 /*fmt*/ + 8 + data_len;

    let mut b = Vec::with_capacity((8 + riff_len) as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&riff_len.to_le_bytes());
    b.extend_from_slice(b"WAVE");
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&num_channels.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&byte_rate.to_le_bytes());
    b.extend_from_slice(&block_align.to_le_bytes());
    b.extend_from_slice(&bits_per_sample.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    b.extend(std::iter::repeat(0x80u8).take(samples));
    b
}

fn wav_tone_8khz_mono_u8(freq_hz: f32, seconds: f32) -> Vec<u8> {
    let sample_rate: u32 = 8_000;
    let n = (seconds * sample_rate as f32).round().max(1.0) as usize;
    let amp: f32 = 40.0;
    let two_pi = std::f32::consts::PI * 2.0;
    let samples: Vec<u8> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let s = (two_pi * freq_hz * t).sin();
            (128.0 + amp * s).round().clamp(0.0, 255.0) as u8
        })
        .collect();

    let mut b = wav_silence_8khz_mono_u8(0);
    let data_len: u32 = samples.len() as u32;
    let riff_len: u32 = 4 /*WAVE*/ + 8 + 16 /*fmt*/ + 8 + data_len;
    b[4..8].copy_from_slice(&riff_len.to_le_bytes());
    b[40..44].copy_from_slice(&data_len.to_le_bytes());
    b.extend_from_slice(&samples);
    b
}

fn l2_norm(v: &[f32]) -> f32 {
    let ss: f32 = v.iter().map(|x| x * x).sum();
    ss.sqrt()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    // Stable, tiny hash for corpus identity (not cryptographic).
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn prompt_hash_and_lens(p: &PromptTemplate) -> (u64, usize, usize) {
    let mut b = Vec::new();
    b.extend_from_slice(p.query_prefix.as_bytes());
    b.push(0);
    b.extend_from_slice(p.doc_prefix.as_bytes());
    (fnv1a64(&b), p.query_prefix.len(), p.doc_prefix.len())
}

fn env_bool(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn default_corpus_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("embedding_corpus.txt")
}

fn default_corpus_display() -> String {
    "testdata/embedding_corpus.txt".to_string()
}

fn load_corpus_from_path(path: &Path) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(line.to_string());
        if out.len() > 10_000 {
            return Err(anyhow::anyhow!(
                "corpus too large (>10k lines); refusing to proceed (line {i})"
            ));
        }
    }
    Ok(out)
}

fn load_corpus() -> (Vec<String>, CorpusMeta) {
    if let Ok(p) = std::env::var("EMBEDD_CORPUS_PATH") {
        let path = PathBuf::from(p);
        match load_corpus_from_path(&path) {
            Ok(v) => {
                return (
                    v,
                    CorpusMeta {
                        source: CorpusSource::Env,
                        path_display: Some(path.display().to_string()),
                    },
                )
            }
            Err(e) => {
                eprintln!("failed to read EMBEDD_CORPUS_PATH corpus: {e}");
                return (
                    corpus_fallback(),
                    CorpusMeta {
                        source: CorpusSource::Fallback,
                        path_display: None,
                    },
                );
            }
        }
    }

    let path = default_corpus_path();
    if path.exists() {
        match load_corpus_from_path(&path) {
            Ok(v) => {
                return (
                    v,
                    CorpusMeta {
                        source: CorpusSource::Default,
                        path_display: Some(default_corpus_display()),
                    },
                )
            }
            Err(e) => {
                eprintln!("failed to read default corpus: {e}");
                return (
                    corpus_fallback(),
                    CorpusMeta {
                        source: CorpusSource::Fallback,
                        path_display: None,
                    },
                );
            }
        }
    }

    (
        corpus_fallback(),
        CorpusMeta {
            source: CorpusSource::Fallback,
            path_display: None,
        },
    )
}

fn builtin_blob_corpus(modality: Modality) -> (Vec<Vec<u8>>, CorpusMeta) {
    let (blobs, label) = match modality {
        Modality::Image => (
            vec![png_1x1_transparent(), png_1x1_black()],
            "builtin:image:png1x1:v1",
        ),
        Modality::Audio => (
            vec![
                wav_silence_8khz_mono_u8(800),
                wav_tone_8khz_mono_u8(440.0, 0.1),
            ],
            "builtin:audio:wav8khz:v1",
        ),
        Modality::Text => (vec![], "builtin:text:none"),
    };
    (
        blobs,
        CorpusMeta {
            source: CorpusSource::Builtin,
            path_display: Some(label.to_string()),
        },
    )
}

fn parse_csv_paths(v: &str) -> Vec<PathBuf> {
    v.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn load_blob_corpus_from_paths(paths: &[PathBuf]) -> anyhow::Result<Vec<Vec<u8>>> {
    const MAX_ITEMS: usize = 256;
    const MAX_ITEM_BYTES: usize = 10 * 1024 * 1024; // 10 MiB
    const MAX_TOTAL_BYTES: usize = 50 * 1024 * 1024; // 50 MiB

    if paths.is_empty() {
        return Ok(Vec::new());
    }
    if paths.len() > MAX_ITEMS {
        return Err(anyhow::anyhow!(
            "too many blob inputs (>{MAX_ITEMS}); refusing"
        ));
    }

    let mut total = 0usize;
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let b =
            std::fs::read(p).map_err(|e| anyhow::anyhow!("failed to read {}: {e}", p.display()))?;
        if b.len() > MAX_ITEM_BYTES {
            return Err(anyhow::anyhow!(
                "blob too large (>{MAX_ITEM_BYTES} bytes): {}",
                p.display()
            ));
        }
        total = total.saturating_add(b.len());
        if total > MAX_TOTAL_BYTES {
            return Err(anyhow::anyhow!(
                "total blob bytes too large (>{MAX_TOTAL_BYTES} bytes)"
            ));
        }
        out.push(b);
    }
    Ok(out)
}

fn load_blob_corpus_from_env(
    modality: Modality,
) -> anyhow::Result<Option<(Vec<Vec<u8>>, CorpusMeta)>> {
    let key = match modality {
        Modality::Image => "EMBEDD_IMAGE_PATHS",
        Modality::Audio => "EMBEDD_AUDIO_PATHS",
        Modality::Text => return Ok(None),
    };

    let v = match std::env::var(key) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let paths = parse_csv_paths(&v);
    if paths.is_empty() {
        return Ok(None);
    }
    let blobs = load_blob_corpus_from_paths(&paths)?;
    let display = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(",");
    Ok(Some((
        blobs,
        CorpusMeta {
            source: CorpusSource::Env,
            path_display: Some(display),
        },
    )))
}

fn compute_stats(embs: &[Vec<f32>]) -> (usize, usize, usize, usize, Option<(f32, f32, f32, f32)>) {
    // Returns:
    // (dim, wrong_dim, non_finite, n_valid, norm_stats(min,max,mean,std))
    if embs.is_empty() {
        return (0, 0, 0, 0, None);
    }

    let dim = embs[0].len();
    let mut min_norm = f32::INFINITY;
    let mut max_norm = 0.0f32;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut non_finite = 0usize;
    let mut wrong_dim = 0usize;
    let mut n_valid = 0usize;

    for e in embs {
        if e.len() != dim {
            wrong_dim += 1;
            continue;
        }
        if !e.iter().all(|x| x.is_finite()) {
            non_finite += 1;
            continue;
        }
        let n = l2_norm(e);
        if !n.is_finite() {
            non_finite += 1;
            continue;
        }
        min_norm = min_norm.min(n);
        max_norm = max_norm.max(n);
        sum += n as f64;
        sumsq += (n as f64) * (n as f64);
        n_valid += 1;
    }

    if n_valid == 0 {
        return (dim, wrong_dim, non_finite, n_valid, None);
    }

    let mean = (sum / n_valid as f64) as f32;
    let var = (sumsq / n_valid as f64) - (mean as f64) * (mean as f64);
    let std = if var.is_finite() && var >= 0.0 {
        (var as f32).sqrt()
    } else {
        f32::NAN
    };

    (
        dim,
        wrong_dim,
        non_finite,
        n_valid,
        Some((min_norm, max_norm, mean, std)),
    )
}

fn write_stats_artifact(
    out_path: &Path,
    backend: &str,
    model_id: Option<&str>,
    modality: Modality,
    mode: Option<EmbedMode>,
    n_inputs: usize,
    corpus_meta: &CorpusMeta,
    corpus_bytes: &[u8],
    corpus_item_hashes: Option<Vec<String>>,
    embs: &[Vec<f32>],
    embed_ms_total: Option<f64>,
    prompt: Option<&PromptTemplate>,
    prompt_application: Option<PromptApplication>,
    prompt_name_effective: Option<&str>,
    normalization: Option<Normalization>,
    truncation: Option<TruncationPolicy>,
    output_dim: Option<usize>,
) -> anyhow::Result<()> {
    use serde_json::{Map, Number, Value};

    let (dim, wrong_dim, non_finite, n_valid, norms) = compute_stats(embs);

    let is_jsonl = out_path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("jsonl"));

    let corpus_hash = fnv1a64(corpus_bytes);

    let include_timestamp = is_jsonl || env_bool("EMBEDD_STATS_INCLUDE_TIMESTAMP");
    let generated_at_unix_s = if include_timestamp {
        Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        )
    } else {
        None
    };

    let mut obj = Map::new();
    obj.insert("schema_version".into(), Value::Number(Number::from(2u64)));
    obj.insert("crate".into(), Value::String("embedd".to_string()));
    obj.insert(
        "embedd_version".into(),
        Value::String(env!("CARGO_PKG_VERSION").to_string()),
    );
    obj.insert("backend".into(), Value::String(backend.to_string()));
    obj.insert(
        "modality".into(),
        Value::String(
            match modality {
                Modality::Text => "text",
                Modality::Image => "image",
                Modality::Audio => "audio",
            }
            .to_string(),
        ),
    );
    obj.insert(
        "mode".into(),
        Value::String(match mode {
            Some(m) => format!("{m:?}"),
            None => "N/A".to_string(),
        }),
    );
    if let Some(ts) = generated_at_unix_s {
        obj.insert(
            "generated_at_unix_s".into(),
            Value::Number(Number::from(ts)),
        );
    }
    obj.insert(
        "n_texts".into(),
        Value::Number(Number::from(n_inputs as u64)),
    );
    // Back-compat / clarity: "inputs" is the real concept across modalities.
    obj.insert(
        "n_inputs".into(),
        Value::Number(Number::from(n_inputs as u64)),
    );
    obj.insert(
        "n_embs".into(),
        Value::Number(Number::from(embs.len() as u64)),
    );
    obj.insert("dim".into(), Value::Number(Number::from(dim as u64)));
    obj.insert(
        "wrong_dim".into(),
        Value::Number(Number::from(wrong_dim as u64)),
    );
    obj.insert(
        "non_finite".into(),
        Value::Number(Number::from(non_finite as u64)),
    );
    obj.insert(
        "n_valid".into(),
        Value::Number(Number::from(n_valid as u64)),
    );
    obj.insert(
        "corpus_hash_fnv1a64".into(),
        Value::String(format!("{corpus_hash:016x}")),
    );
    obj.insert(
        "corpus_n_lines".into(),
        Value::Number(Number::from(n_inputs as u64)),
    );

    obj.insert(
        "corpus_source".into(),
        Value::String(
            match corpus_meta.source {
                CorpusSource::Default => "default",
                CorpusSource::Env => "env",
                CorpusSource::Fallback => "fallback",
                CorpusSource::Builtin => "builtin",
            }
            .to_string(),
        ),
    );

    if let Some(p) = &corpus_meta.path_display {
        obj.insert("corpus_path".into(), Value::String(p.clone()));
    }
    if let Some(mid) = model_id {
        obj.insert("model_id".into(), Value::String(mid.to_string()));
    }

    if let Some(p) = prompt {
        let (h, ql, dl) = prompt_hash_and_lens(p);
        obj.insert(
            "prompt_hash_fnv1a64".into(),
            Value::String(format!("{h:016x}")),
        );
        obj.insert(
            "prompt_query_prefix_len".into(),
            Value::Number(Number::from(ql as u64)),
        );
        obj.insert(
            "prompt_doc_prefix_len".into(),
            Value::Number(Number::from(dl as u64)),
        );
    }

    // Record "how prompts/scoping were applied" to avoid silent drift.
    if let Some(app) = prompt_application {
        obj.insert("prompt_apply".into(), Value::String(format!("{app:?}")));
    }
    if let Some(pn) = prompt_name_effective {
        obj.insert("prompt_name".into(), Value::String(pn.to_string()));
    }
    if let Some(n) = normalization {
        obj.insert("normalization".into(), Value::String(format!("{n:?}")));
    }
    if let Some(t) = truncation {
        obj.insert("truncation_policy".into(), Value::String(format!("{t:?}")));
        // Expand a bit for dashboard friendliness.
        if let TruncationPolicy::Truncate { max_len, direction } = t {
            if let Some(ml) = max_len {
                obj.insert(
                    "truncation_max_len".into(),
                    Value::Number(Number::from(ml as u64)),
                );
            }
            obj.insert(
                "truncation_direction".into(),
                Value::String(format!("{direction:?}")),
            );
        }
    }
    if let Some(d) = output_dim {
        obj.insert("output_dim".into(), Value::Number(Number::from(d as u64)));
    }

    // Benchmarking: wall-clock time for the embedder call, in milliseconds.
    //
    // Notes:
    // - This includes whatever the backend does (network, queueing, model compute).
    // - It is intentionally coarse; for high-fidelity latency profiling, prefer backend-specific tools
    //   (e.g. TEI exposes timing headers and ships k6 load tests).
    if let Some(ms) = embed_ms_total {
        if let Some(n) = Number::from_f64(ms) {
            obj.insert("embed_ms_total".into(), Value::Number(n));
        }
        if n_inputs > 0 {
            let per = ms / (n_inputs as f64);
            if let Some(n) = Number::from_f64(per) {
                obj.insert("embed_ms_per_text".into(), Value::Number(n));
            }
            // Back-compat / clarity across modalities:
            if let Some(n) = Number::from_f64(per) {
                obj.insert("embed_ms_per_input".into(), Value::Number(n));
            }
        }
    }

    // Optional: record item-level hashes to localize corpus changes without storing inputs.
    if let Some(hs) = corpus_item_hashes {
        if !hs.is_empty() {
            obj.insert(
                "corpus_line_hashes_fnv1a64".into(),
                Value::Array(hs.into_iter().map(Value::String).collect()),
            );
        }
    }

    if let Some((min_n, max_n, mean_n, std_n)) = norms {
        if let Some(n) = Number::from_f64(min_n as f64) {
            obj.insert("l2_norm_min".into(), Value::Number(n));
        }
        if let Some(n) = Number::from_f64(max_n as f64) {
            obj.insert("l2_norm_max".into(), Value::Number(n));
        }
        if let Some(n) = Number::from_f64(mean_n as f64) {
            obj.insert("l2_norm_mean".into(), Value::Number(n));
        }
        if let Some(n) = Number::from_f64(std_n as f64) {
            obj.insert("l2_norm_std".into(), Value::Number(n));
        }
    }

    let v = Value::Object(obj);

    if is_jsonl {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out_path)?;
        writeln!(f, "{}", serde_json::to_string(&v)?)?;
    } else {
        std::fs::write(out_path, serde_json::to_string_pretty(&v)?)?;
    }

    Ok(())
}

#[cfg(feature = "tei")]
fn make_tei() -> anyhow::Result<Box<dyn TextEmbedder>> {
    let base_url = std::env::var("EMBEDD_TEI_BASE_URL")
        .map_err(|_| anyhow::anyhow!("missing EMBEDD_TEI_BASE_URL (e.g. http://127.0.0.1:8080)"))?;
    let mut tei = embedd::tei::TeiEmbedder::new(base_url);
    if let Ok(k) = std::env::var("EMBEDD_TEI_API_KEY") {
        tei = tei.with_api_key(k);
    }
    // Optional TEI-native prompt selection.
    if let (Ok(q), Ok(d)) = (
        std::env::var("EMBEDD_TEI_PROMPT_NAME_QUERY"),
        std::env::var("EMBEDD_TEI_PROMPT_NAME_DOC"),
    ) {
        tei = tei.with_prompt_names(q, d);
    }
    if let Ok(v) = std::env::var("EMBEDD_TEI_TRUNCATE") {
        if v == "1" || v.eq_ignore_ascii_case("true") {
            tei = tei.with_truncate(true);
        }
        if v == "0" || v.eq_ignore_ascii_case("false") {
            tei = tei.with_truncate(false);
        }
    }
    if let Ok(v) = std::env::var("EMBEDD_TEI_TRUNCATION_DIRECTION") {
        tei = tei.with_truncation_direction(v);
    }
    if let Ok(v) = std::env::var("EMBEDD_TEI_NORMALIZE") {
        if v == "1" || v.eq_ignore_ascii_case("true") {
            tei = tei.with_normalize(true);
        }
        if v == "0" || v.eq_ignore_ascii_case("false") {
            tei = tei.with_normalize(false);
        }
    }
    if let Ok(v) = std::env::var("EMBEDD_OUTPUT_DIM") {
        if let Ok(d) = v.parse::<usize>() {
            tei = tei.with_dimensions(d);
        }
    }
    Ok(Box::new(tei))
}

#[cfg(not(feature = "tei"))]
fn make_tei() -> anyhow::Result<Box<dyn TextEmbedder>> {
    Err(anyhow::anyhow!("backend tei requires `--features tei`"))
}

#[cfg(feature = "openai")]
fn make_openai() -> anyhow::Result<Box<dyn TextEmbedder>> {
    let api_key = std::env::var("EMBEDD_OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("missing EMBEDD_OPENAI_API_KEY"))?;
    let base_url = std::env::var("EMBEDD_OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("EMBEDD_OPENAI_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());
    Ok(Box::new(embedd::openai::OpenAiEmbedder::new(
        base_url, api_key, model,
    )))
}

#[cfg(not(feature = "openai"))]
fn make_openai() -> anyhow::Result<Box<dyn TextEmbedder>> {
    Err(anyhow::anyhow!(
        "backend openai requires `--features openai`"
    ))
}

#[cfg(feature = "fastembed")]
fn make_fastembed() -> anyhow::Result<Box<dyn TextEmbedder>> {
    Ok(Box::new(
        embedd::fastembed::FastembedEmbedder::new_default()?
    ))
}

#[cfg(not(feature = "fastembed"))]
fn make_fastembed() -> anyhow::Result<Box<dyn TextEmbedder>> {
    Err(anyhow::anyhow!(
        "backend fastembed requires `--features fastembed`"
    ))
}

#[cfg(feature = "hf-inference")]
fn make_hf_inference() -> anyhow::Result<embedd::hf_inference::HfInferenceEmbedder> {
    let model = std::env::var("EMBEDD_HF_MODEL").unwrap_or_else(|_| {
        // Reasonable default for text; for image/audio, recommend overriding.
        "sentence-transformers/all-MiniLM-L6-v2".to_string()
    });
    Ok(embedd::hf_inference::HfInferenceEmbedder::new(model))
}

#[cfg(not(feature = "hf-inference"))]
fn make_hf_inference() -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "backend hf-inference requires `--features hf-inference`"
    ))
}

fn main() -> anyhow::Result<()> {
    let backend = std::env::var("EMBEDD_BACKEND").unwrap_or_else(|_| "tei".to_string());
    let modality = parse_modality()?;
    let mode_s = std::env::var("EMBEDD_MODE").unwrap_or_else(|_| "query".to_string());
    let mode = match mode_s.as_str() {
        "query" => EmbedMode::Query,
        "doc" | "document" => EmbedMode::Document,
        other => {
            return Err(anyhow::anyhow!(
                "unknown EMBEDD_MODE={other} (use query|doc)"
            ))
        }
    };

    println!("backend={backend} modality={modality:?}");

    let include_item_hashes = env_bool("EMBEDD_STATS_INCLUDE_LINE_HASHES");
    let apply_prompt = env_bool("EMBEDD_APPLY_PROMPT") && modality == Modality::Text;
    let prompt = apply_prompt.then_some(PromptTemplate::from_env_any());

    // Prepare inputs + corpus identity bytes + optional per-item hashes (for stable diffing).
    let (
        n_inputs,
        corpus_meta,
        corpus_bytes,
        corpus_item_hashes,
        embs,
        embed_ms_total,
        model_id,
        prompt_application,
        prompt_name_effective,
        normalization,
        truncation,
    ): (
        usize,
        CorpusMeta,
        Vec<u8>,
        Option<Vec<String>>,
        Vec<Vec<f32>>,
        f64,
        Option<String>,
        Option<PromptApplication>,
        Option<String>,
        Option<Normalization>,
        Option<TruncationPolicy>,
    ) = match modality {
        Modality::Text => {
            let embedder_raw = match backend.as_str() {
                "tei" => make_tei()?,
                "openai" => make_openai()?,
                "fastembed" => make_fastembed()?,
                "hf-inference" => {
                    return Err(anyhow::anyhow!(
                        "hf-inference backend requires EMBEDD_MODALITY=image|audio|text with --features hf-inference; use EMBEDD_HF_MODEL"
                    ))
                }
                other => {
                    return Err(anyhow::anyhow!(
                        "unknown EMBEDD_BACKEND={other} (tei|openai|fastembed)"
                    ))
                }
            };
            let (texts, corpus_meta) = load_corpus();
            println!("mode={mode:?} n_texts={}", texts.len());
            let model_id_s = embedder_raw.model_id().map(|s| s.to_string());

            // Choose a scoping policy:
            // - If client prompt is requested: ClientPrefix
            // - Else if TEI prompt name vars are present: RequireServerPromptName (TEI applies it)
            // - Else: None
            let tei_has_prompt_names = backend == "tei"
                && (std::env::var("EMBEDD_TEI_PROMPT_NAME_QUERY").is_ok()
                    || std::env::var("EMBEDD_TEI_PROMPT_NAME_DOC").is_ok());
            let policy = if let Some(p) = prompt.clone() {
                ScopingPolicy::ClientPrefix(p)
            } else if tei_has_prompt_names {
                ScopingPolicy::RequireServerPromptName
            } else {
                ScopingPolicy::None
            };

            // Apply/validate policy.
            let embedder = apply_scoping_policy(embedder_raw, policy)?;
            // Optional output-dim truncation (applies before normalization).
            let output_dim = std::env::var("EMBEDD_OUTPUT_DIM")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            let embedder = apply_output_dim(embedder, output_dim)?;
            // Apply/validate normalization policy (default: Preserve).
            let norm_policy = if env_bool("EMBEDD_REQUIRE_L2") {
                NormalizationPolicy::RequireL2
            } else {
                NormalizationPolicy::Preserve
            };
            let embedder = apply_normalization_policy(embedder, norm_policy)?;
            let caps = embedder.capabilities();
            let prompt_application = Some(caps.uses_embed_mode);
            let normalization = Some(caps.normalization);
            let truncation = Some(caps.truncation);

            // Record TEI prompt_name (effective for this mode) when configured.
            let prompt_name_effective = if backend == "tei" {
                match mode {
                    EmbedMode::Query => std::env::var("EMBEDD_TEI_PROMPT_NAME_QUERY").ok(),
                    EmbedMode::Document => std::env::var("EMBEDD_TEI_PROMPT_NAME_DOC").ok(),
                }
            } else {
                None
            };

            let mut corpus_bytes = Vec::new();
            for t in &texts {
                corpus_bytes.extend_from_slice(t.as_bytes());
                corpus_bytes.push(b'\n');
            }
            let corpus_item_hashes = if include_item_hashes {
                Some(
                    texts
                        .iter()
                        .map(|t| format!("{:016x}", fnv1a64(t.as_bytes())))
                        .collect(),
                )
            } else {
                None
            };

            let t0 = std::time::Instant::now();
            let embs = embedder.embed_texts(&texts, mode)?;
            let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;

            (
                texts.len(),
                corpus_meta,
                corpus_bytes,
                corpus_item_hashes,
                embs,
                embed_ms_total,
                model_id_s,
                prompt_application,
                prompt_name_effective,
                normalization,
                truncation,
            )
        }
        Modality::Image => {
            // Real e2e path for image/audio currently: hf-inference.
            if backend.as_str() != "hf-inference" {
                return Err(anyhow::anyhow!(
                    "EMBEDD_MODALITY=image currently requires EMBEDD_BACKEND=hf-inference"
                ));
            }
            #[cfg(feature = "hf-inference")]
            {
                let e = make_hf_inference()?;
                let (blobs, corpus_meta) = match load_blob_corpus_from_env(Modality::Image)? {
                    Some(v) => v,
                    None => builtin_blob_corpus(Modality::Image),
                };
                let mut corpus_bytes = Vec::new();
                for b in &blobs {
                    corpus_bytes.extend_from_slice(b);
                    corpus_bytes.push(0);
                }
                let corpus_item_hashes = if include_item_hashes {
                    Some(
                        blobs
                            .iter()
                            .map(|b| format!("{:016x}", fnv1a64(b)))
                            .collect(),
                    )
                } else {
                    None
                };
                println!("n_images={}", blobs.len());
                let t0 = std::time::Instant::now();
                let embs = e.embed_images(&blobs)?;
                let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;
                (
                    blobs.len(),
                    corpus_meta,
                    corpus_bytes,
                    corpus_item_hashes,
                    embs,
                    embed_ms_total,
                    TextEmbedder::model_id(&e).map(|s| s.to_string()),
                    None,
                    None,
                    None,
                    None,
                )
            }
            #[cfg(not(feature = "hf-inference"))]
            {
                return Err(anyhow::anyhow!(
                    "hf-inference backend requires `--features hf-inference`"
                ));
            }
        }
        Modality::Audio => {
            if backend.as_str() != "hf-inference" {
                return Err(anyhow::anyhow!(
                    "EMBEDD_MODALITY=audio currently requires EMBEDD_BACKEND=hf-inference"
                ));
            }
            #[cfg(feature = "hf-inference")]
            {
                let e = make_hf_inference()?;
                let (blobs, corpus_meta) = match load_blob_corpus_from_env(Modality::Audio)? {
                    Some(v) => v,
                    None => builtin_blob_corpus(Modality::Audio),
                };
                let mut corpus_bytes = Vec::new();
                for b in &blobs {
                    corpus_bytes.extend_from_slice(b);
                    corpus_bytes.push(0);
                }
                let corpus_item_hashes = if include_item_hashes {
                    Some(
                        blobs
                            .iter()
                            .map(|b| format!("{:016x}", fnv1a64(b)))
                            .collect(),
                    )
                } else {
                    None
                };
                println!("n_audios={}", blobs.len());
                let t0 = std::time::Instant::now();
                let embs = e.embed_audios(&blobs)?;
                let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;
                (
                    blobs.len(),
                    corpus_meta,
                    corpus_bytes,
                    corpus_item_hashes,
                    embs,
                    embed_ms_total,
                    TextEmbedder::model_id(&e).map(|s| s.to_string()),
                    None,
                    None,
                    None,
                    None,
                )
            }
            #[cfg(not(feature = "hf-inference"))]
            {
                return Err(anyhow::anyhow!(
                    "hf-inference backend requires `--features hf-inference`"
                ));
            }
        }
    };

    let model_id = model_id.as_deref();
    if let Some(mid) = model_id {
        println!("model_id={mid}");
    }
    if let Some(p) = &corpus_meta.path_display {
        println!("corpus_path={p}");
    }
    println!("n_embs={}", embs.len());
    let (dim, wrong_dim, non_finite, _n_valid, norms) = compute_stats(&embs);
    println!("dim={dim}");
    println!("wrong_dim={wrong_dim} non_finite={non_finite}");
    println!("embed_ms_total={embed_ms_total:.3}");
    if let Some((min_norm, max_norm, mean, std)) = norms {
        println!(
            "l2_norm_min={min_norm:.6} l2_norm_max={max_norm:.6} l2_norm_mean={mean:.6} l2_norm_std={std:.6}"
        );
    }

    if let Ok(out_path) = std::env::var("EMBEDD_STATS_OUT") {
        let out_path = PathBuf::from(out_path);
        write_stats_artifact(
            &out_path,
            &backend,
            model_id,
            modality,
            if modality == Modality::Text {
                Some(mode)
            } else {
                None
            },
            n_inputs,
            &corpus_meta,
            &corpus_bytes,
            corpus_item_hashes,
            &embs,
            Some(embed_ms_total),
            prompt.as_ref(),
            prompt_application,
            prompt_name_effective.as_deref(),
            normalization,
            truncation,
            std::env::var("EMBEDD_OUTPUT_DIM")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
        )?;
        println!("stats_out={}", out_path.display());
    }

    Ok(())
}
