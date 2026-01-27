#[cfg(feature = "hf-inference")]
use embedd::hf_inference::HfInferenceEmbedder;

#[cfg(feature = "hf-inference")]
use embedd::{AudioEmbedder, ImageEmbedder, TextEmbedder};

#[cfg(feature = "hf-inference")]
fn net_tests_enabled() -> bool {
    std::env::var("EMBEDD_RUN_NET_TESTS")
        .ok()
        .as_deref()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(feature = "hf-inference")]
fn assert_embeddings_sane(embs: &[Vec<f32>]) {
    assert!(!embs.is_empty(), "no embeddings returned");
    let d = embs[0].len();
    assert!(d > 0, "zero-dim embedding");
    for (i, e) in embs.iter().enumerate() {
        assert_eq!(e.len(), d, "embedding {i} has inconsistent dim");
        assert!(
            e.iter().all(|x| x.is_finite()),
            "embedding {i} has non-finite"
        );
    }
}

#[cfg(feature = "hf-inference")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    embedd::vector::cosine_f32(a, b)
}

#[cfg(feature = "hf-inference")]
fn assert_identicals_are_most_similar(
    embs: &[Vec<f32>],
    idx_a: usize,
    idx_b: usize,
    others: &[usize],
) {
    let ab = cosine(&embs[idx_a], &embs[idx_b]);
    assert!(ab.is_finite(), "cosine(ab) not finite");
    assert!(
        ab > 0.95,
        "identical inputs not highly similar: cosine={ab}"
    );
    for &j in others {
        let aj = cosine(&embs[idx_a], &embs[j]);
        assert!(aj.is_finite(), "cosine(a,j) not finite");
        assert!(
            ab >= aj + 0.01,
            "identical similarity not best: cosine(ab)={ab} vs cosine(a,{j})={aj}"
        );
    }
}

// 1x1 transparent PNG (valid PNG file).
#[cfg(feature = "hf-inference")]
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
            let byte = ((buf >> bits) & 0xFF) as u8;
            out.push(byte);
        }
    }
    out
}

#[cfg(feature = "hf-inference")]
fn assert_looks_like_png(bytes: &[u8]) {
    assert!(bytes.len() > 16, "png too small");
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n", "bad png signature");
    // Minimal cheap checks; actual decoding happens on the server side.
    assert!(bytes.windows(4).any(|w| w == b"IHDR"), "missing IHDR");
    assert!(bytes.windows(4).any(|w| w == b"IEND"), "missing IEND");
}

#[cfg(feature = "hf-inference")]
fn png_1x1_transparent() -> Vec<u8> {
    // Widely used fixture: 1x1 transparent PNG.
    let b = base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xc1cAAAAASUVORK5CYII=");
    assert_looks_like_png(&b);
    b
}

#[cfg(feature = "hf-inference")]
fn png_1x1_black() -> Vec<u8> {
    // Widely used fixture: 1x1 black PNG.
    let b = base64_decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP8z8AABfsC/VE0l9QAAAAASUVORK5CYII=");
    assert_looks_like_png(&b);
    b
}

// Minimal mono PCM WAV with short silence (valid RIFF/WAVE container).
#[cfg(feature = "hf-inference")]
fn wav_silence_8khz_mono_u8(samples: usize) -> Vec<u8> {
    let sample_rate: u32 = 8_000;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 8;
    let byte_rate: u32 = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align: u16 = num_channels * (bits_per_sample / 8);
    let data_len: u32 = samples as u32; // 8-bit mono => 1 byte/sample

    let riff_len: u32 = 4 /*WAVE*/ + 8 + 16 /*fmt*/ + 8 + data_len;

    let mut b = Vec::with_capacity((8 + riff_len) as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&riff_len.to_le_bytes());
    b.extend_from_slice(b"WAVE");

    // fmt chunk
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&num_channels.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&byte_rate.to_le_bytes());
    b.extend_from_slice(&block_align.to_le_bytes());
    b.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    // 8-bit PCM silence is 0x80 (midpoint).
    b.extend(std::iter::repeat(0x80u8).take(samples));
    b
}

#[cfg(feature = "hf-inference")]
fn wav_tone_8khz_mono_u8(freq_hz: f32, seconds: f32) -> Vec<u8> {
    let sample_rate: u32 = 8_000;
    let n = (seconds * sample_rate as f32).round().max(1.0) as usize;

    // Generate 8-bit PCM samples: mid=0x80, amplitude small to avoid clipping.
    let amp: f32 = 40.0;
    let two_pi = std::f32::consts::PI * 2.0;
    let samples: Vec<u8> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let s = (two_pi * freq_hz * t).sin();
            let v = 128.0 + amp * s;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect();

    // Reuse the WAV framing from the silence generator.
    let mut b = wav_silence_8khz_mono_u8(0);
    // Patch in the correct data chunk length and RIFF length, then append samples.
    //
    // Layout offsets:
    // - RIFF size at bytes 4..8
    // - data size at bytes (44-4)..44? Actually "data" chunk header begins at 36.
    //   data size field is at 40..44.
    let data_len: u32 = samples.len() as u32;
    let riff_len: u32 = 4 /*WAVE*/ + 8 + 16 /*fmt*/ + 8 + data_len;
    b[4..8].copy_from_slice(&riff_len.to_le_bytes());
    b[40..44].copy_from_slice(&data_len.to_le_bytes());
    b.extend_from_slice(&samples);
    b
}

#[cfg(feature = "hf-inference")]
#[test]
fn integration_hf_text_opt_in() {
    if !net_tests_enabled() {
        eprintln!("Skipping: set EMBEDD_RUN_NET_TESTS=1 to enable.");
        return;
    }
    let model = std::env::var("EMBEDD_HF_TEXT_MODEL")
        .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string());

    let e = HfInferenceEmbedder::new(model);
    let a = "Marie Curie discovered radium in Paris.".to_string();
    let b = "This is unrelated: xqzv 12345 — 東京 (Tokyo) delegation.".to_string();
    let texts = vec![a.clone(), a, b];
    let out = e.embed_texts(&texts, embedd::EmbedMode::Document).unwrap();
    assert_eq!(out.len(), texts.len());
    assert_embeddings_sane(&out);
    assert_identicals_are_most_similar(&out, 0, 1, &[2]);
}

#[cfg(feature = "hf-inference")]
#[test]
fn integration_hf_image_opt_in() {
    if !net_tests_enabled() {
        eprintln!("Skipping: set EMBEDD_RUN_NET_TESTS=1 to enable.");
        return;
    }
    // A CLIP-like model is common for image feature extraction on HF Inference.
    let model = std::env::var("EMBEDD_HF_IMAGE_MODEL")
        .unwrap_or_else(|_| "openai/clip-vit-base-patch32".to_string());

    let e = HfInferenceEmbedder::new(model);
    let a = png_1x1_transparent();
    let b = png_1x1_black();
    let images = vec![a.clone(), a, b];
    let out = e.embed_images(&images).unwrap();
    assert_eq!(out.len(), images.len());
    assert_embeddings_sane(&out);
    assert_identicals_are_most_similar(&out, 0, 1, &[2]);
}

#[cfg(feature = "hf-inference")]
#[test]
fn integration_hf_audio_opt_in() {
    if !net_tests_enabled() {
        eprintln!("Skipping: set EMBEDD_RUN_NET_TESTS=1 to enable.");
        return;
    }
    // Audio embedding models vary; user can override with env var.
    // We default to a wav2vec2 model; HF Inference may or may not return feature vectors depending on pipeline.
    let model = std::env::var("EMBEDD_HF_AUDIO_MODEL")
        .unwrap_or_else(|_| "facebook/wav2vec2-base-960h".to_string());

    let e = HfInferenceEmbedder::new(model);
    let a = wav_silence_8khz_mono_u8(800); // 0.1s silence
    let b = wav_tone_8khz_mono_u8(440.0, 0.1);
    let audios = vec![a.clone(), a, b];
    let out = e.embed_audios(&audios).unwrap();
    assert_eq!(out.len(), audios.len());
    assert_embeddings_sane(&out);
    assert_identicals_are_most_similar(&out, 0, 1, &[2]);
}
