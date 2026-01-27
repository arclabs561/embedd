use embedd::{AudioEmbedder, ImageEmbedder, TextEmbedder};
use proptest::prelude::*;

/// Contract-level tests for all modalities supported by `embedd`.
///
/// These are deliberately **backend-agnostic**: they test invariants about the trait surfaces,
/// using tiny dummy embedders rather than real models.

#[derive(Clone, Default)]
struct DummyTextEmbedder {
    dim: usize,
}

impl embedd::TextEmbedder for DummyTextEmbedder {
    fn embed_texts(
        &self,
        texts: &[String],
        _mode: embedd::EmbedMode,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let d = self.dim.max(1);
        Ok(texts
            .iter()
            .map(|t| {
                let h = fnv1a64(t.as_bytes());
                (0..d)
                    .map(|i| ((h ^ (i as u64)) as f32) * 1e-6)
                    .collect::<Vec<f32>>()
            })
            .collect())
    }
}

#[derive(Clone, Default)]
struct DummyImageEmbedder {
    dim: usize,
}

impl embedd::ImageEmbedder for DummyImageEmbedder {
    fn embed_images(&self, images: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<f32>>> {
        let d = self.dim.max(1);
        Ok(images
            .iter()
            .map(|b| {
                let h = fnv1a64(b);
                (0..d)
                    .map(|i| ((h.wrapping_add(i as u64)) as f32) * 1e-6)
                    .collect::<Vec<f32>>()
            })
            .collect())
    }
}

#[derive(Clone, Default)]
struct DummyAudioEmbedder {
    dim: usize,
}

impl embedd::AudioEmbedder for DummyAudioEmbedder {
    fn embed_audios(&self, audios: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<f32>>> {
        let d = self.dim.max(1);
        Ok(audios
            .iter()
            .map(|b| {
                let h = fnv1a64(b);
                (0..d)
                    .map(|i| ((h.rotate_left((i % 63) as u32)) as f32) * 1e-6)
                    .collect::<Vec<f32>>()
            })
            .collect())
    }
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

fn assert_vectors_sane(embs: &[Vec<f32>], expected_len: usize) {
    assert_eq!(embs.len(), expected_len);
    if embs.is_empty() {
        return;
    }
    let dim = embs[0].len();
    assert!(dim > 0, "zero-dim embedding");
    for (i, e) in embs.iter().enumerate() {
        assert_eq!(e.len(), dim, "embedding {i} has inconsistent dim");
        assert!(
            e.iter().all(|x| x.is_finite()),
            "embedding {i} contains non-finite"
        );
    }
}

proptest! {
    #[test]
    fn text_embedder_contract(
        dim in 1usize..256,
        texts in prop::collection::vec(".*", 0..64),
    ) {
        let e = DummyTextEmbedder { dim };
        let texts: Vec<String> = texts.into_iter().collect();
        let out = e.embed_texts(&texts, embedd::EmbedMode::Query).unwrap();
        assert_vectors_sane(&out, texts.len());
    }

    #[test]
    fn image_embedder_contract(
        dim in 1usize..256,
        blobs in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..2048), 0..64),
    ) {
        let e = DummyImageEmbedder { dim };
        let out = e.embed_images(&blobs).unwrap();
        assert_vectors_sane(&out, blobs.len());
    }

    #[test]
    fn audio_embedder_contract(
        dim in 1usize..256,
        blobs in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..4096), 0..64),
    ) {
        let e = DummyAudioEmbedder { dim };
        let out = e.embed_audios(&blobs).unwrap();
        assert_vectors_sane(&out, blobs.len());
    }
}
