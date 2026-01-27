use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use embedd::{EmbedMode, TextEmbedder};

fn bench_prompt_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_apply");

    let texts = [
        "Marie Curie discovered radium in Paris.",
        "習近平在北京會見了普京。",
        "التقى محمد بن سلمان بالرئيس في الرياض",
        "Путин встретился с Си Цзиньпином в Москве.",
        "Hello 👋 مرحبا 09:35",
        "👨\u{200D}👩\u{200D}👧\u{200D}👦",
        "Rōma āterna est. Cīvis Rōmānus sum.",
        "रामायणे रामः सीतां अयोध्यायाः वनं नयति",
    ];

    let p = embedd::PromptTemplate::default();

    for (i, t) in texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("query", i), t, |b, t| {
            b.iter(|| p.apply(embedd::EmbedMode::Query, t))
        });
        group.bench_with_input(BenchmarkId::new("doc", i), t, |b, t| {
            b.iter(|| p.apply(embedd::EmbedMode::Document, t))
        });
    }

    group.finish();
}

fn bench_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops");

    let n = 768usize;
    let a: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();

    group.bench_function("dot", |bench| {
        bench.iter(|| embedd::vector::dot_f32(&a, &b))
    });
    group.bench_function("cosine", |bench| {
        bench.iter(|| embedd::vector::cosine_f32(&a, &b))
    });

    group.bench_function("l2_normalize_in_place", |bench| {
        bench.iter(|| {
            let mut v = a.clone();
            let _ = embedd::vector::l2_normalize_in_place(&mut v);
            v
        })
    });

    group.finish();
}

#[derive(Clone)]
struct PrecomputedEmbedder {
    v: Vec<f32>,
}

impl TextEmbedder for PrecomputedEmbedder {
    fn embed_texts(&self, texts: &[String], _mode: EmbedMode) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| self.v.clone()).collect())
    }

    fn capabilities(&self) -> embedd::TextEmbedderCapabilities {
        embedd::TextEmbedderCapabilities {
            uses_embed_mode: embedd::PromptApplication::None,
            normalization: embedd::Normalization::NotNormalized,
            truncation: embedd::TruncationPolicy::Unknown,
        }
    }
}

fn bench_wrapper_overheads(c: &mut Criterion) {
    let mut group = c.benchmark_group("wrapper_overheads");

    let dim = 768usize;
    let v: Vec<f32> = (0..dim).map(|i| (i as f32).sin() + 0.01).collect();
    let base = PrecomputedEmbedder { v };

    let texts: Vec<String> = (0..32).map(|i| format!("text {i}")).collect();

    group.bench_function("base", |b| {
        b.iter(|| base.embed_texts(&texts, EmbedMode::Query))
    });

    let out_dim = embedd::apply_output_dim(base.clone(), Some(256)).unwrap();
    group.bench_function("output_dim_256", |b| {
        b.iter(|| out_dim.embed_texts(&texts, EmbedMode::Query))
    });

    let l2 =
        embedd::apply_normalization_policy(base.clone(), embedd::NormalizationPolicy::RequireL2)
            .unwrap();
    group.bench_function("require_l2", |b| {
        b.iter(|| l2.embed_texts(&texts, EmbedMode::Query))
    });

    let out_dim_then_l2 = embedd::apply_output_dim(base, Some(256)).unwrap();
    let out_dim_then_l2 =
        embedd::apply_normalization_policy(out_dim_then_l2, embedd::NormalizationPolicy::RequireL2)
            .unwrap();
    group.bench_function("output_dim_256_then_require_l2", |b| {
        b.iter(|| out_dim_then_l2.embed_texts(&texts, EmbedMode::Query))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_prompt_apply,
    bench_vector_ops,
    bench_wrapper_overheads
);
criterion_main!(benches);
