use embedd::{EmbedMode, PromptTemplate};
use proptest::prelude::*;

fn diverse_texts() -> Vec<&'static str> {
    vec![
        // Latin
        "Marie Curie discovered radium in Paris.",
        // CJK
        "習近平在北京會見了普京。",
        // Arabic (RTL)
        "التقى محمد بن سلمان بالرئيس في الرياض",
        // Cyrillic
        "Путин встретился с Си Цзиньпином в Москве.",
        // Mixed
        "Dr. 田中 presented her research at MIT's AI conference.",
        // Diacritics
        "François Müller and José García met in São Paulo.",
        // Sanskrit (Devanagari)
        "रामायणे रामः सीतां अयोध्यायाः वनं नयति",
        // Emoji ZWJ
        "👨\u{200D}👩\u{200D}👧\u{200D}👦",
        // Normalization hazard (NFD)
        "cafe\u{0301}",
    ]
}

#[test]
fn prompt_template_default_prefixes_are_nonempty() {
    let p = PromptTemplate::default();
    assert!(!p.query_prefix.is_empty());
    assert!(!p.doc_prefix.is_empty());
}

#[test]
fn prompt_template_apply_examples() {
    let p = PromptTemplate {
        query_prefix: "Q: ".into(),
        doc_prefix: "D: ".into(),
    };
    assert_eq!(p.apply(EmbedMode::Query, "x"), "Q: x");
    assert_eq!(p.apply(EmbedMode::Document, "x"), "D: x");
}

#[test]
fn prompt_template_handles_diverse_texts() {
    let p = PromptTemplate::default();
    for t in diverse_texts() {
        let q = p.apply(EmbedMode::Query, t);
        let d = p.apply(EmbedMode::Document, t);
        assert!(q.contains(t));
        assert!(d.contains(t));
    }
}

proptest! {
    #[test]
    fn prompt_apply_is_prefixing(qp in ".*", dp in ".*", t in ".*") {
        let p = PromptTemplate { query_prefix: qp.clone(), doc_prefix: dp.clone() };

        let q = p.apply(EmbedMode::Query, &t);
        prop_assert!(q.starts_with(&qp));
        prop_assert!(q.ends_with(&t));

        let d = p.apply(EmbedMode::Document, &t);
        prop_assert!(d.starts_with(&dp));
        prop_assert!(d.ends_with(&t));
    }
}
