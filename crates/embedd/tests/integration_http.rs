// These tests are intentionally opt-in, because they can hit the network and may cost money.
//
// Run (example):
//   EMBEDD_RUN_NET_TESTS=1 EMBEDD_OPENAI_API_KEY=... \
//   cargo test --manifest-path embedd/Cargo.toml -p embedd --features openai -- integration_openai

#[cfg(any(feature = "tei", feature = "openai"))]
use embedd::TextEmbedder;

#[cfg(any(feature = "tei", feature = "openai"))]
fn net_tests_enabled() -> bool {
    matches!(
        std::env::var("EMBEDD_RUN_NET_TESTS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

#[cfg(any(feature = "tei", feature = "openai"))]
fn assert_embeddings_sane(embs: &[Vec<f32>]) {
    assert!(!embs.is_empty(), "no embeddings returned");
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

#[cfg(feature = "tei")]
#[test]
fn integration_tei_smoke_opt_in() {
    if !net_tests_enabled() {
        eprintln!("skipping: set EMBEDD_RUN_NET_TESTS=1 to enable");
        return;
    }

    let base_url = match std::env::var("EMBEDD_TEI_BASE_URL") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: set EMBEDD_TEI_BASE_URL (e.g. http://127.0.0.1:8080)");
            return;
        }
    };

    let embedder = embedd::tei::TeiEmbedder::new(base_url);

    let texts = vec![
        "Marie Curie discovered radium in Paris.".to_string(),
        "習近平在北京會見了普京。".to_string(),
    ];

    let embs = embedder
        .embed_texts(&texts, embedd::EmbedMode::Query)
        .expect("tei embed_texts failed");

    assert_eq!(embs.len(), texts.len());
    assert_embeddings_sane(&embs);
}

#[cfg(feature = "openai")]
#[test]
fn integration_openai_smoke_opt_in() {
    if !net_tests_enabled() {
        eprintln!("skipping: set EMBEDD_RUN_NET_TESTS=1 to enable");
        return;
    }

    let api_key = match std::env::var("EMBEDD_OPENAI_API_KEY") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: set EMBEDD_OPENAI_API_KEY");
            return;
        }
    };

    let base_url = std::env::var("EMBEDD_OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("EMBEDD_OPENAI_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());

    let embedder = embedd::openai::OpenAiEmbedder::new(base_url, api_key, model);

    let texts = vec![
        "Marie Curie discovered radium in Paris.".to_string(),
        "Путин встретился с Си Цзиньпином в Москве.".to_string(),
    ];

    let embs = embedder
        .embed_texts(&texts, embedd::EmbedMode::Query)
        .expect("openai embed_texts failed");

    assert_eq!(embs.len(), texts.len());
    assert_embeddings_sane(&embs);
}
