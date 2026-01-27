use clap::{Parser, ValueEnum};
use embedd::{
    apply_normalization_policy, apply_output_dim, apply_scoping_policy, EmbedMode,
    NormalizationPolicy, PromptTemplate, ScopingPolicy, TextEmbedder,
};

#[derive(Debug, Clone, ValueEnum)]
enum ModeArg {
    Query,
    Doc,
}

impl From<ModeArg> for EmbedMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Query => EmbedMode::Query,
            ModeArg::Doc => EmbedMode::Document,
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
#[command(after_help = r#"EXAMPLES:
  # Embed stdin lines as queries (default)
  printf "hello\nworld\n" | embedd --backend fastembed --mode query

  # Embed from a file as documents
  embedd --input-file inputs.txt --mode doc

  # Enforce L2 normalization and truncate output dimension
  embedd --require-l2 --output-dim 256 < inputs.txt
"#)]
struct Cli {
    /// Backend name: tei | fastembed | openai
    #[arg(long, default_value = "fastembed")]
    backend: String,

    /// Mode: query | doc
    #[arg(long, value_enum, default_value_t = ModeArg::Query)]
    mode: ModeArg,

    /// Read one input per line from this file (otherwise stdin).
    #[arg(long)]
    input_file: Option<std::path::PathBuf>,

    /// If set, apply prompt prefixes (from env) on the client side.
    #[arg(long, default_value_t = false)]
    apply_prompt: bool,

    /// Require L2 normalization (client-side wrapper if backend doesn't do it).
    #[arg(long, default_value_t = false)]
    require_l2: bool,

    /// Truncate output embeddings to this dimension (client-side wrapper).
    #[arg(long)]
    output_dim: Option<usize>,

    /// TEI base URL (e.g. http://127.0.0.1:8080)
    #[arg(long)]
    tei_url: Option<String>,

    /// TEI API key (sent as Authorization: Bearer <key>)
    #[arg(long)]
    tei_api_key: Option<String>,
}

fn read_lines(path: Option<std::path::PathBuf>) -> anyhow::Result<Vec<String>> {
    let s = match path {
        Some(p) => std::fs::read_to_string(p)?,
        None => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };
    let mut out = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if !line.is_empty() {
            out.push(line.to_string());
        }
    }
    if out.is_empty() {
        return Err(anyhow::anyhow!(
            "no inputs (provide lines via --input-file or stdin)"
        ));
    }
    Ok(out)
}

fn l2_stats(embs: &[Vec<f32>]) -> (usize, usize, usize, Option<(f64, f64, f64, f64)>) {
    let mut wrong_dim = 0usize;
    let mut non_finite = 0usize;
    let dim = embs.first().map(|v| v.len()).unwrap_or(0);
    let mut norms = Vec::new();
    for v in embs {
        if v.len() != dim {
            wrong_dim += 1;
            continue;
        }
        let mut ss = 0.0f64;
        for &x in v {
            if !x.is_finite() {
                non_finite += 1;
                break;
            }
            ss += (x as f64) * (x as f64);
        }
        norms.push(ss.sqrt());
    }
    let stats = if norms.is_empty() {
        None
    } else {
        let min = norms.iter().copied().fold(f64::INFINITY, f64::min);
        let max = norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = norms.iter().sum::<f64>() / norms.len() as f64;
        let var = norms
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / norms.len() as f64;
        Some((min, max, mean, var.sqrt()))
    };
    (dim, wrong_dim, non_finite, stats)
}

fn make_backend(cli: &Cli) -> anyhow::Result<Box<dyn TextEmbedder>> {
    match cli.backend.as_str() {
        "tei" => {
            #[cfg(feature = "tei")]
            {
                let url = cli
                    .tei_url
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("--tei-url is required for --backend tei"))?;
                let mut e = embedd::tei::TeiEmbedder::new(url);
                if let Some(k) = &cli.tei_api_key {
                    e = e.with_api_key(k.clone());
                }
                Ok(Box::new(e))
            }
            #[cfg(not(feature = "tei"))]
            {
                Err(anyhow::anyhow!(
                    "backend tei requires building with --features tei"
                ))
            }
        }
        "openai" => {
            #[cfg(feature = "openai")]
            {
                let base_url = std::env::var("EMBEDD_OPENAI_BASE_URL")
                    .map_err(|_| anyhow::anyhow!("missing EMBEDD_OPENAI_BASE_URL"))?;
                let api_key = std::env::var("EMBEDD_OPENAI_API_KEY")
                    .map_err(|_| anyhow::anyhow!("missing EMBEDD_OPENAI_API_KEY"))?;
                let model = std::env::var("EMBEDD_OPENAI_MODEL")
                    .map_err(|_| anyhow::anyhow!("missing EMBEDD_OPENAI_MODEL"))?;
                Ok(Box::new(embedd::openai::OpenAiEmbedder::new(
                    base_url, api_key, model,
                )))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(anyhow::anyhow!(
                    "backend openai requires building with --features openai"
                ))
            }
        }
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                Ok(Box::new(
                    embedd::fastembed::FastembedEmbedder::new_default()?
                ))
            }
            #[cfg(not(feature = "fastembed"))]
            {
                Err(anyhow::anyhow!(
                    "backend fastembed requires building with --features fastembed"
                ))
            }
        }
        other => Err(anyhow::anyhow!(
            "unknown --backend {other} (use tei|fastembed|openai)"
        )),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mode: EmbedMode = cli.mode.clone().into();
    let texts = read_lines(cli.input_file.clone())?;

    let raw = make_backend(&cli)?;

    // Scoping / prompting policy.
    let tei_has_prompt_names = cli.backend == "tei"
        && (std::env::var("EMBEDD_TEI_PROMPT_NAME_QUERY").is_ok()
            || std::env::var("EMBEDD_TEI_PROMPT_NAME_DOC").is_ok());
    let policy = if cli.apply_prompt {
        ScopingPolicy::ClientPrefix(PromptTemplate::from_env_any())
    } else if tei_has_prompt_names {
        ScopingPolicy::RequireServerPromptName
    } else {
        ScopingPolicy::None
    };
    let e = apply_scoping_policy(raw, policy)?;

    // Output dim truncation then optional normalization.
    let e = apply_output_dim(e, cli.output_dim)?;
    let norm_policy = if cli.require_l2 {
        NormalizationPolicy::RequireL2
    } else {
        NormalizationPolicy::Preserve
    };
    let e = apply_normalization_policy(e, norm_policy)?;

    let caps = e.capabilities();
    let dim_reported = e.dimension();
    let embs = e.embed_texts(&texts, mode)?;

    let (dim, wrong_dim, non_finite, norm_stats) = l2_stats(&embs);

    // Keep output stable and easy to diff.
    let mut obj = serde_json::Map::new();
    obj.insert("schema_version".into(), serde_json::Value::from(1u64));
    obj.insert("backend".into(), serde_json::Value::from(cli.backend));
    obj.insert("mode".into(), serde_json::Value::from(format!("{mode:?}")));
    if let Some(mid) = e.model_id() {
        obj.insert("model_id".into(), serde_json::Value::from(mid));
    }
    obj.insert(
        "n_inputs".into(),
        serde_json::Value::from(texts.len() as u64),
    );
    obj.insert("dim".into(), serde_json::Value::from(dim as u64));
    if let Some(d) = dim_reported {
        obj.insert(
            "dimension_reported".into(),
            serde_json::Value::from(d as u64),
        );
    }
    obj.insert(
        "wrong_dim".into(),
        serde_json::Value::from(wrong_dim as u64),
    );
    obj.insert(
        "non_finite".into(),
        serde_json::Value::from(non_finite as u64),
    );
    obj.insert(
        "prompt_apply".into(),
        serde_json::Value::from(format!("{:?}", caps.uses_embed_mode)),
    );
    obj.insert(
        "normalization".into(),
        serde_json::Value::from(format!("{:?}", caps.normalization)),
    );
    obj.insert(
        "truncation_policy".into(),
        serde_json::Value::from(format!("{:?}", caps.truncation)),
    );
    if let Some(d) = cli.output_dim {
        obj.insert("output_dim".into(), serde_json::Value::from(d as u64));
    }

    if let Some((min, max, mean, std)) = norm_stats {
        obj.insert("l2_norm_min".into(), serde_json::Value::from(min));
        obj.insert("l2_norm_max".into(), serde_json::Value::from(max));
        obj.insert("l2_norm_mean".into(), serde_json::Value::from(mean));
        obj.insert("l2_norm_std".into(), serde_json::Value::from(std));
    }

    println!("{}", serde_json::Value::Object(obj).to_string());
    Ok(())
}
