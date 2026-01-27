#![allow(dead_code)]

use embedd::{
    apply_normalization_policy, apply_output_dim, apply_scoping_policy, EmbedMode,
    NormalizationPolicy, PromptTemplate, ScopingPolicy, TextEmbedder,
};
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-inference")]
use embedd::{AudioEmbedder, ImageEmbedder};

fn env_bool(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
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

fn default_corpus_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("embedding_corpus.txt")
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

fn load_corpus() -> anyhow::Result<Vec<String>> {
    if let Ok(p) = std::env::var("EMBEDD_CORPUS_PATH") {
        return load_corpus_from_path(Path::new(&p));
    }
    load_corpus_from_path(&default_corpus_path())
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

fn load_blob_corpus_from_env(modality: Modality) -> anyhow::Result<Option<Vec<Vec<u8>>>> {
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
    Ok(Some(load_blob_corpus_from_paths(&paths)?))
}

fn compute_stats(embs: &[Vec<f32>]) -> (usize, usize, usize, usize, Option<(f32, f32, f32, f32)>) {
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

fn write_jsonl_line(out_path: &Path, v: serde_json::Value) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)?;
    writeln!(f, "{}", serde_json::to_string(&v)?)?;
    Ok(())
}

#[cfg(feature = "tei")]
fn make_tei() -> anyhow::Result<Box<dyn TextEmbedder>> {
    let base_url = std::env::var("EMBEDD_TEI_BASE_URL")
        .map_err(|_| anyhow::anyhow!("missing EMBEDD_TEI_BASE_URL (e.g. http://127.0.0.1:8080)"))?;
    Ok(Box::new(embedd::tei::TeiEmbedder::new(base_url)))
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

fn make_backend(name: &str) -> anyhow::Result<Box<dyn TextEmbedder>> {
    match name {
        "tei" => make_tei(),
        "openai" => make_openai(),
        "fastembed" => make_fastembed(),
        other => Err(anyhow::anyhow!("unknown backend {other}")),
    }
}

#[cfg(feature = "hf-inference")]
fn make_hf_inference() -> anyhow::Result<embedd::hf_inference::HfInferenceEmbedder> {
    let model = std::env::var("EMBEDD_HF_MODEL")
        .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    Ok(embedd::hf_inference::HfInferenceEmbedder::new(model))
}

fn main() -> anyhow::Result<()> {
    // Comma-separated list of backends to try. Defaults to whatever is enabled.
    let backends =
        std::env::var("EMBEDD_BACKENDS").unwrap_or_else(|_| "tei,fastembed,openai".to_string());
    let modes = std::env::var("EMBEDD_MODES").unwrap_or_else(|_| "query,doc".to_string());
    let modality = parse_modality()?;

    let out_path = std::env::var("EMBEDD_STATS_OUT").map_err(|_| {
        anyhow::anyhow!("missing EMBEDD_STATS_OUT (must end with .jsonl for matrix mode)")
    })?;
    let out_path = PathBuf::from(out_path);

    if out_path
        .extension()
        .and_then(|s| s.to_str())
        .is_none_or(|s| !s.eq_ignore_ascii_case("jsonl"))
    {
        return Err(anyhow::anyhow!(
            "EMBEDD_STATS_OUT must be a .jsonl path for backend_matrix"
        ));
    }

    let include_line_hashes = env_bool("EMBEDD_STATS_INCLUDE_LINE_HASHES");
    let apply_prompt = env_bool("EMBEDD_APPLY_PROMPT") && modality == Modality::Text;
    let prompt = if apply_prompt {
        Some(PromptTemplate::from_env_any())
    } else {
        None
    };
    let norm_policy = if env_bool("EMBEDD_REQUIRE_L2") {
        NormalizationPolicy::RequireL2
    } else {
        NormalizationPolicy::Preserve
    };
    let output_dim = std::env::var("EMBEDD_OUTPUT_DIM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    for backend in backends
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        // For image/audio we currently only support hf-inference in this “matrix” harness.
        if modality != Modality::Text && backend != "hf-inference" {
            eprintln!("skipping backend={backend}: modality={modality:?} requires hf-inference");
            continue;
        }

        for mode_s in modes.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let mode = match mode_s {
                "query" => EmbedMode::Query,
                "doc" | "document" => EmbedMode::Document,
                other => {
                    eprintln!("skipping unknown mode {other}");
                    continue;
                }
            };

            let (
                n_inputs,
                corpus_hash,
                corpus_line_hashes,
                embs,
                embed_ms_total,
                model_id,
                mode_out,
                prompt_apply,
                prompt_name_effective,
                normalization,
                truncation,
            ) = match modality {
                Modality::Text => {
                    let texts = match load_corpus() {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("failed to load corpus: {e}");
                            continue;
                        }
                    };
                    let mut corpus_bytes = Vec::new();
                    for t in &texts {
                        corpus_bytes.extend_from_slice(t.as_bytes());
                        corpus_bytes.push(b'\n');
                    }
                    let corpus_hash = fnv1a64(&corpus_bytes);
                    let corpus_line_hashes: Option<Vec<String>> = if include_line_hashes {
                        Some(
                            texts
                                .iter()
                                .map(|t| format!("{:016x}", fnv1a64(t.as_bytes())))
                                .collect(),
                        )
                    } else {
                        None
                    };

                    let embedder_raw = match make_backend(backend) {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("skipping backend={backend}: {e}");
                            continue;
                        }
                    };
                    let model_id = embedder_raw.model_id().map(|s| s.to_string());

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
                    let embedder = match apply_scoping_policy(embedder_raw, policy) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("skipping backend={backend}: {e}");
                            continue;
                        }
                    };
                    let embedder = match apply_output_dim(embedder, output_dim) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("skipping backend={backend}: {e}");
                            continue;
                        }
                    };
                    let embedder = match apply_normalization_policy(embedder, norm_policy) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("skipping backend={backend}: {e}");
                            continue;
                        }
                    };
                    let caps = embedder.capabilities();
                    let prompt_apply = caps.uses_embed_mode;
                    let normalization = caps.normalization;
                    let truncation = caps.truncation;
                    let prompt_name_effective = if backend == "tei" {
                        match mode {
                            EmbedMode::Query => std::env::var("EMBEDD_TEI_PROMPT_NAME_QUERY").ok(),
                            EmbedMode::Document => std::env::var("EMBEDD_TEI_PROMPT_NAME_DOC").ok(),
                        }
                    } else {
                        None
                    };

                    let t0 = std::time::Instant::now();
                    let embs = match embedder.embed_texts(&texts, mode) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("backend={backend} mode={mode:?} failed: {e}");
                            continue;
                        }
                    };
                    let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;
                    (
                        texts.len(),
                        corpus_hash,
                        corpus_line_hashes,
                        embs,
                        embed_ms_total,
                        model_id,
                        Some(format!("{mode:?}")),
                        Some(prompt_apply),
                        prompt_name_effective,
                        Some(normalization),
                        Some(truncation),
                    )
                }
                Modality::Image => {
                    #[cfg(feature = "hf-inference")]
                    {
                        let e = match make_hf_inference() {
                            Ok(e) => e,
                            Err(err) => {
                                eprintln!("skipping backend={backend}: {err}");
                                continue;
                            }
                        };
                        let blobs = match load_blob_corpus_from_env(Modality::Image)? {
                            Some(b) => b,
                            None => vec![png_1x1_transparent(), png_1x1_black()],
                        };
                        let mut corpus_bytes = Vec::new();
                        for b in &blobs {
                            corpus_bytes.extend_from_slice(b);
                            corpus_bytes.push(0);
                        }
                        let corpus_hash = fnv1a64(&corpus_bytes);
                        let corpus_line_hashes: Option<Vec<String>> = if include_line_hashes {
                            Some(
                                blobs
                                    .iter()
                                    .map(|b| format!("{:016x}", fnv1a64(b)))
                                    .collect(),
                            )
                        } else {
                            None
                        };
                        let model_id = TextEmbedder::model_id(&e).map(|s| s.to_string());
                        let t0 = std::time::Instant::now();
                        let embs = match e.embed_images(&blobs) {
                            Ok(v) => v,
                            Err(err) => {
                                eprintln!("backend={backend} image failed: {err}");
                                continue;
                            }
                        };
                        let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;
                        (
                            blobs.len(),
                            corpus_hash,
                            corpus_line_hashes,
                            embs,
                            embed_ms_total,
                            model_id,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    }
                    #[cfg(not(feature = "hf-inference"))]
                    {
                        eprintln!("skipping backend={backend}: needs --features hf-inference");
                        continue;
                    }
                }
                Modality::Audio => {
                    #[cfg(feature = "hf-inference")]
                    {
                        let e = match make_hf_inference() {
                            Ok(e) => e,
                            Err(err) => {
                                eprintln!("skipping backend={backend}: {err}");
                                continue;
                            }
                        };
                        let blobs = match load_blob_corpus_from_env(Modality::Audio)? {
                            Some(b) => b,
                            None => vec![
                                wav_silence_8khz_mono_u8(800),
                                wav_tone_8khz_mono_u8(440.0, 0.1),
                            ],
                        };
                        let mut corpus_bytes = Vec::new();
                        for b in &blobs {
                            corpus_bytes.extend_from_slice(b);
                            corpus_bytes.push(0);
                        }
                        let corpus_hash = fnv1a64(&corpus_bytes);
                        let corpus_line_hashes: Option<Vec<String>> = if include_line_hashes {
                            Some(
                                blobs
                                    .iter()
                                    .map(|b| format!("{:016x}", fnv1a64(b)))
                                    .collect(),
                            )
                        } else {
                            None
                        };
                        let model_id = TextEmbedder::model_id(&e).map(|s| s.to_string());
                        let t0 = std::time::Instant::now();
                        let embs = match e.embed_audios(&blobs) {
                            Ok(v) => v,
                            Err(err) => {
                                eprintln!("backend={backend} audio failed: {err}");
                                continue;
                            }
                        };
                        let embed_ms_total = t0.elapsed().as_secs_f64() * 1000.0;
                        (
                            blobs.len(),
                            corpus_hash,
                            corpus_line_hashes,
                            embs,
                            embed_ms_total,
                            model_id,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    }
                    #[cfg(not(feature = "hf-inference"))]
                    {
                        eprintln!("skipping backend={backend}: needs --features hf-inference");
                        continue;
                    }
                }
            };

            let (dim, wrong_dim, non_finite, n_valid, norms) = compute_stats(&embs);

            let mut obj = serde_json::Map::new();
            obj.insert("schema_version".into(), serde_json::Value::from(2u64));
            obj.insert("crate".into(), serde_json::Value::from("embedd"));
            obj.insert(
                "embedd_version".into(),
                serde_json::Value::from(env!("CARGO_PKG_VERSION")),
            );
            obj.insert("backend".into(), serde_json::Value::from(backend));
            obj.insert(
                "modality".into(),
                serde_json::Value::from(match modality {
                    Modality::Text => "text",
                    Modality::Image => "image",
                    Modality::Audio => "audio",
                }),
            );
            obj.insert(
                "mode".into(),
                serde_json::Value::from(mode_out.clone().unwrap_or_else(|| "N/A".to_string())),
            );

            // Matrix runs are logs: timestamp is always included.
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            obj.insert("generated_at_unix_s".into(), serde_json::Value::from(ts));

            if let Some(mid) = model_id.as_deref() {
                obj.insert("model_id".into(), serde_json::Value::from(mid));
            }
            if let Some(p) = prompt.as_ref() {
                let (h, ql, dl) = prompt_hash_and_lens(p);
                obj.insert(
                    "prompt_hash_fnv1a64".into(),
                    serde_json::Value::from(format!("{h:016x}")),
                );
                obj.insert(
                    "prompt_query_prefix_len".into(),
                    serde_json::Value::from(ql as u64),
                );
                obj.insert(
                    "prompt_doc_prefix_len".into(),
                    serde_json::Value::from(dl as u64),
                );
            }
            if let Some(app) = prompt_apply {
                obj.insert(
                    "prompt_apply".into(),
                    serde_json::Value::from(format!("{app:?}")),
                );
            }
            if let Some(pn) = prompt_name_effective.as_deref() {
                obj.insert("prompt_name".into(), serde_json::Value::from(pn));
            }
            if let Some(n) = normalization {
                obj.insert(
                    "normalization".into(),
                    serde_json::Value::from(format!("{n:?}")),
                );
            }
            if let Some(t) = truncation {
                obj.insert(
                    "truncation_policy".into(),
                    serde_json::Value::from(format!("{t:?}")),
                );
            }
            if let Some(d) = output_dim {
                obj.insert("output_dim".into(), serde_json::Value::from(d as u64));
            }

            obj.insert("n_texts".into(), serde_json::Value::from(n_inputs as u64));
            obj.insert("n_inputs".into(), serde_json::Value::from(n_inputs as u64));
            obj.insert("n_embs".into(), serde_json::Value::from(embs.len() as u64));
            obj.insert("dim".into(), serde_json::Value::from(dim as u64));
            obj.insert(
                "wrong_dim".into(),
                serde_json::Value::from(wrong_dim as u64),
            );
            obj.insert(
                "non_finite".into(),
                serde_json::Value::from(non_finite as u64),
            );
            obj.insert("n_valid".into(), serde_json::Value::from(n_valid as u64));
            obj.insert(
                "embed_ms_total".into(),
                serde_json::Value::from(embed_ms_total),
            );
            if n_inputs > 0 {
                obj.insert(
                    "embed_ms_per_text".into(),
                    serde_json::Value::from(embed_ms_total / (n_inputs as f64)),
                );
                obj.insert(
                    "embed_ms_per_input".into(),
                    serde_json::Value::from(embed_ms_total / (n_inputs as f64)),
                );
                let secs = embed_ms_total / 1000.0;
                if secs > 0.0 {
                    obj.insert(
                        "embed_texts_per_s".into(),
                        serde_json::Value::from((n_inputs as f64) / secs),
                    );
                }
            }
            obj.insert(
                "corpus_hash_fnv1a64".into(),
                serde_json::Value::from(format!("{corpus_hash:016x}")),
            );
            obj.insert(
                "corpus_n_lines".into(),
                serde_json::Value::from(n_inputs as u64),
            );

            if let Some(hs) = &corpus_line_hashes {
                obj.insert(
                    "corpus_line_hashes_fnv1a64".into(),
                    serde_json::Value::Array(
                        hs.iter()
                            .cloned()
                            .map(serde_json::Value::String)
                            .collect::<Vec<_>>(),
                    ),
                );
            }

            if let Some((min_n, max_n, mean_n, std_n)) = norms {
                obj.insert("l2_norm_min".into(), serde_json::Value::from(min_n));
                obj.insert("l2_norm_max".into(), serde_json::Value::from(max_n));
                obj.insert("l2_norm_mean".into(), serde_json::Value::from(mean_n));
                obj.insert("l2_norm_std".into(), serde_json::Value::from(std_n));
            }

            write_jsonl_line(&out_path, serde_json::Value::Object(obj))?;
            println!(
                "wrote backend={backend} mode={mode:?} -> {}",
                out_path.display()
            );
        }
    }

    Ok(())
}
