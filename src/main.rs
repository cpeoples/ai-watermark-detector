mod c2pa;

use ai_watermark_detector::{
    parse_token_ids, render, score, Config, OutputFormat, Scheme, RELIABLE_TOKEN_FLOOR,
};
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;

/// Resolve the effective output format from the shared `--format` flag and the legacy
/// `--json` shortcut. `--json` wins when set so existing scripts keep working.
fn resolve_format(format: &str, json: bool) -> OutputFormat {
    if json {
        return OutputFormat::Json;
    }
    OutputFormat::parse(format).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    })
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Statistical text-watermark scorer + C2PA file-provenance checker"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Score text you watermarked yourself (needs the key). NOT a "paste ChatGPT text" detector.
    ///
    /// Scores a token-id sequence against a statistical text watermark (kgw, synthid, exp,
    /// unigram, sweet, exp-edit).
    ///
    /// IMPORTANT: this only gives a real answer when you hold the watermark KEY — i.e. you
    /// generated the text yourself, or a vendor published their key. Keyed text watermarks
    /// leave no trace you can read without the exact key, so this CANNOT tell you whether an
    /// arbitrary ChatGPT/Claude/Gemini paragraph is AI-written; there is no key-free method
    /// by cryptographic design. To check a real file (image/video/audio/pdf) for AI
    /// provenance instead, use `check` — that works today with no key.
    Score(ScoreArgs),
    /// Check files, folders, or URLs for C2PA content provenance (images/video/audio/pdf).
    Check(CheckArgs),
    /// Scan files/folders and report per-format C2PA manifest coverage.
    Scan(ScanArgs),
}

#[derive(Parser, Debug)]
struct ScoreArgs {
    /// Comma/space separated token IDs, e.g. "1,42,983,17"
    #[arg(long)]
    tokens: Option<String>,

    /// File containing whitespace/comma separated token IDs.
    #[arg(long)]
    token_file: Option<String>,

    /// JSON config containing ngram_len, keys, and (KGW only) gamma/vocab_size.
    #[arg(long)]
    config: String,

    /// Watermark family: `synthid` (Gemini-style) or `kgw` (Claude-style green-list).
    #[arg(long, default_value = "synthid")]
    scheme: String,

    /// Optional minimum score for a local "positive" result.
    #[arg(long, default_value_t = 0.95)]
    threshold: f64,

    /// Output format: text (default), json, xml, or yaml.
    #[arg(long, default_value = "text")]
    format: String,

    /// Legacy shortcut for `--format json`.
    #[arg(long, hide = true)]
    json: bool,
}

#[derive(Parser, Debug)]
struct CheckArgs {
    /// Files, folders, or http(s) URLs to inspect.
    files: Vec<String>,

    /// Download real, publicly C2PA-signed samples (image/video/audio) and check them.
    #[arg(long)]
    fetch_samples: bool,

    /// Directory for fetched samples (persistent).
    #[arg(long, default_value = "samples")]
    samples_dir: String,

    /// URL or path to a PEM trust-anchor list. When set, signatures are validated against
    /// it and reported as trusted vs. untrusted (real cert-chain verification).
    #[arg(long)]
    trust_anchors: Option<String>,

    /// Output format: text (default), json, xml, or yaml.
    #[arg(long, default_value = "text")]
    format: String,

    /// Legacy shortcut for `--format json`.
    #[arg(long, hide = true)]
    json: bool,
}

#[derive(Parser, Debug)]
struct ScanArgs {
    /// Files, folders, or http(s) URLs to scan (folders are walked recursively).
    files: Vec<String>,

    /// URL or path to a PEM trust-anchor list for cert-chain validation.
    #[arg(long)]
    trust_anchors: Option<String>,

    /// Output format: text (default), json, xml, or yaml.
    #[arg(long, default_value = "text")]
    format: String,

    /// Legacy shortcut for `--format json`.
    #[arg(long, hide = true)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Score(args) => run_score(args),
        Commands::Check(args) => run_check(args),
        Commands::Scan(args) => run_scan(args),
    }
}

fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.tokens.is_some() == args.token_file.is_some() {
        eprintln!("Provide exactly one of --tokens or --token-file");
        std::process::exit(2);
    }

    let scheme = Scheme::parse(&args.scheme).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });

    let config_text = fs::read_to_string(&args.config).unwrap_or_else(|e| {
        eprintln!(
            "error: could not read config file '{}': {e}\n\
             (pass --config with a path to a JSON config; see config.example.json)",
            args.config
        );
        std::process::exit(2);
    });
    let cfg: Config = serde_json::from_str(&config_text).unwrap_or_else(|e| {
        eprintln!(
            "error: config '{}' is not valid JSON: {e}\n\
             (expected keys like ngram_len, keys, gamma; see config.example.json)",
            args.config
        );
        std::process::exit(2);
    });
    if cfg.ngram_len < 2 || cfg.keys.is_empty() {
        eprintln!("config must contain ngram_len >= 2 and at least one key");
        std::process::exit(2);
    }
    if scheme == Scheme::Kgw && !(cfg.gamma > 0.0 && cfg.gamma < 1.0) {
        eprintln!("kgw scheme requires gamma in (0,1)");
        std::process::exit(2);
    }

    let token_text = if let Some(s) = args.tokens {
        s
    } else {
        let path = args.token_file.unwrap();
        fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: could not read token file '{path}': {e}");
            std::process::exit(2);
        })
    };

    let tokens = parse_token_ids(&token_text).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let result = score(&tokens, &cfg, scheme);

    let positive = result.reliable
        && result.total > 0
        && result.z > 0.0
        && result.approx_p_value <= (1.0 - args.threshold);

    let format = resolve_format(&args.format, args.json);
    let out = ScoreOutput {
        scheme: result.scheme.as_str().to_string(),
        tokens: result.tokens,
        positions: result.positions,
        usable_positions: result.usable_positions,
        green_or_ones: result.ones,
        total: result.total,
        mean_g: result.mean_g,
        weighted_mean_g: result.weighted_mean_g,
        grounded_p_value: result.grounded_p_value,
        z: result.z,
        approx_p_value: result.approx_p_value,
        reliable: result.reliable,
        screen_positive: positive,
        warning: "not an authoritative vendor verifier",
    };
    if let Some(rendered) = render(&out, format, "score")? {
        println!("{rendered}");
    } else {
        // Plain-English verdict first, details after — so a non-expert gets the answer up top.
        let verdict = if !result.reliable {
            Verdict::TooShort
        } else if positive {
            Verdict::Found
        } else {
            Verdict::None
        };
        println!("=== AI watermark screen: {} ===", verdict.headline());
        match verdict {
            Verdict::Found => {
                println!(
                    "This text carries the '{}' statistical watermark for the key you supplied.",
                    result.scheme.as_str()
                );
                println!(
                    "Chance a random unwatermarked text would score this high: about {}.",
                    human_odds(result.approx_p_value)
                );
            }
            Verdict::None => {
                println!(
                    "No '{}' watermark detectable with the key you supplied. This does NOT prove",
                    result.scheme.as_str()
                );
                println!(
                    "the text is human-written — only that THIS scheme+key leaves no trace here."
                );
            }
            Verdict::TooShort => {
                println!(
                    "Only {} usable tokens (need ~{}+). Too little text for a reliable statistical call.",
                    result.usable_positions, RELIABLE_TOKEN_FLOOR
                );
            }
        }
        println!();
        println!("--- details (for the technically inclined) ---");
        println!("scheme:            {}", result.scheme.as_str());
        println!("tokens:            {}", result.tokens);
        println!("positions:         {}", result.positions);
        println!("usable positions:  {}", result.usable_positions);
        let label = if result.scheme == Scheme::Kgw {
            "green count"
        } else {
            "g=1 count"
        };
        println!("{label:<18} {} / {}", result.ones, result.total);
        println!("mean rate:         {:.4}", result.mean_g);
        if let Some(w) = result.weighted_mean_g {
            println!("weighted-mean g:   {w:.4}   (SynthID stronger detector)");
        }
        println!("z-score:           {:.3}", result.z);
        println!("approx p-value:    {:.6}", result.approx_p_value);
        if let Some(gp) = result.grounded_p_value {
            println!(
                "grounded p-value:  {gp:.3e}   (Three Bricks exact null, better on short text)"
            );
        }
        println!();
        println!(
            "NOTE: This is a RESEARCH tool, not an authoritative vendor verifier. It gives a real"
        );
        println!(
            "answer only when the watermark scheme, config/keys, and tokenizer match the text that"
        );
        println!("was generated. It cannot detect a vendor's production text without that vendor's secret");
        println!("key. A result for one scheme (e.g. KGW/Claude) says nothing about another (SynthID/Gemini).");
    }

    Ok(())
}

/// Machine-readable `score` result. Field names/order define the JSON/XML/YAML contract
/// that the contributor Python tools parse, so keep them stable. `weighted_mean_g` and
/// `grounded_p_value` are omitted when a scheme doesn't produce them.
#[derive(Serialize)]
struct ScoreOutput {
    scheme: String,
    tokens: usize,
    positions: usize,
    usable_positions: usize,
    green_or_ones: usize,
    total: usize,
    mean_g: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    weighted_mean_g: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounded_p_value: Option<f64>,
    z: f64,
    approx_p_value: f64,
    reliable: bool,
    screen_positive: bool,
    warning: &'static str,
}

/// Machine-readable wrapper for `check`/`scan`: the per-file reports under one root.
#[derive(Serialize)]
struct ReportsOutput<'a> {
    results: &'a [c2pa::C2paReport],
}

/// Plain-English verdict shown at the top of a `score` run.
enum Verdict {
    Found,
    None,
    TooShort,
}

impl Verdict {
    fn headline(&self) -> &'static str {
        match self {
            Verdict::Found => "WATERMARK SIGNAL FOUND",
            Verdict::None => "NO WATERMARK SIGNAL",
            Verdict::TooShort => "TOO SHORT TO TELL",
        }
    }
}

/// Turn a p-value into plain-English odds a lay reader can grasp.
fn human_odds(p: f64) -> String {
    if p <= 0.0 {
        return "less than 1 in a trillion".to_string();
    }
    let denom = (1.0 / p).round() as u128;
    if denom >= 1_000_000_000_000 {
        "less than 1 in a trillion".to_string()
    } else if denom >= 1_000_000_000 {
        format!("1 in {} billion", denom / 1_000_000_000)
    } else if denom >= 1_000_000 {
        format!("1 in {} million", denom / 1_000_000)
    } else if denom >= 1_000 {
        format!("1 in {},000", denom / 1_000)
    } else {
        format!("1 in {denom}")
    }
}

fn run_check(args: CheckArgs) -> Result<(), Box<dyn std::error::Error>> {
    ensure_c2patool();

    let mut files = c2pa::gather_files(&args.files);
    if args.fetch_samples {
        let mut fetched = c2pa::fetch_samples(&args.samples_dir)?;
        fetched.extend(files);
        files = fetched;
    }
    if files.is_empty() {
        eprintln!("error: no files given. Pass files/folders, or use --fetch-samples.");
        std::process::exit(2);
    }

    let reports: Vec<c2pa::C2paReport> = files
        .iter()
        .map(|f| c2pa::check_file(f, args.trust_anchors.as_deref()))
        .collect();

    let format = resolve_format(&args.format, args.json);
    if let Some(rendered) = render(&ReportsOutput { results: &reports }, format, "check")? {
        println!("{rendered}");
        return Ok(());
    }

    println!("C2PA provenance check");
    println!("{}", "=".repeat(74));
    for r in &reports {
        println!("file: {}", r.file);
        println!("  plain answer:    {}", plain_c2pa_answer(r));
        println!(
            "  manifest:        {}",
            if r.has_manifest { "yes" } else { "no" }
        );
        println!("  verdict:         {}", r.verdict.to_uppercase());
        if let Some(gen) = &r.claim_generator {
            println!("  claim generator: {gen}");
        }
        if let Some(title) = &r.title {
            println!("  title:           {title}");
        }
        if let Some(ai) = &r.ai_source_type {
            println!("  AI marker:       {ai}  <-- flagged AI-generated");
        }
        if r.synthid_assertion {
            println!("  SynthID:         SynthID provenance assertion present (Google marker)");
        }
        if !r.assertions.is_empty() {
            println!("  assertions:      {}", r.assertions.join(", "));
        }
        println!("  status:          {}", r.status);
        println!("{}", "-".repeat(74));
    }
    println!("Note: C2PA is signed provenance living alongside the file; format conversion,");
    println!("re-saving, or screenshots strip it. A present valid manifest is strong positive");
    println!("evidence; its absence is inconclusive.");
    Ok(())
}

fn run_scan(args: ScanArgs) -> Result<(), Box<dyn std::error::Error>> {
    ensure_c2patool();

    let files = c2pa::gather_files(&args.files);
    if files.is_empty() {
        eprintln!("error: no files given. Pass files or folders to scan.");
        std::process::exit(2);
    }

    let reports: Vec<c2pa::C2paReport> = files
        .iter()
        .map(|f| c2pa::check_file(f, args.trust_anchors.as_deref()))
        .collect();

    let format = resolve_format(&args.format, args.json);
    if let Some(rendered) = render(&ReportsOutput { results: &reports }, format, "scan")? {
        println!("{rendered}");
        return Ok(());
    }

    // Aggregate per extension: (total, with_manifest, ai_marked).
    let mut by_ext: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for r in &reports {
        let e = by_ext.entry(r.ext.clone()).or_insert((0, 0, 0));
        e.0 += 1;
        if r.has_manifest {
            e.1 += 1;
        }
        if r.ai_source_type.is_some() || r.synthid_assertion {
            e.2 += 1;
        }
    }

    println!("C2PA format-coverage report");
    println!("{}", "=".repeat(60));
    println!(
        "{:<10}{:>7}{:>12}{:>11}",
        "ext", "files", "w/manifest", "AI-marked"
    );
    println!("{}", "-".repeat(60));
    for (ext, (total, manifest, ai)) in &by_ext {
        println!("{ext:<10}{total:>7}{manifest:>12}{ai:>11}");
    }
    println!("{}", "-".repeat(60));
    println!("Formats with 0 w/manifest carry NO checkable provenance in practice");
    println!("(e.g. .docx/.pptx/.txt/source code) - their only possible trace is the");
    println!("key-gated statistical text watermark, which needs the vendor's secret key.");
    Ok(())
}

fn ensure_c2patool() {
    if which_c2patool().is_none() {
        eprintln!(
            "error: `c2patool` was not found on PATH. Install the official Content Authenticity CLI:\n\
             \x20 macOS:    brew install c2patool\n\
             \x20 any OS:   cargo install c2patool\n\
             \x20 Windows:  cargo install c2patool  (or download c2patool.exe from Releases)\n\
             \x20 releases: https://github.com/contentauth/c2patool/releases\n\
             (The text-watermark `score` command does NOT need c2patool — only `check`/`scan` do.)"
        );
        std::process::exit(2);
    }
}

fn which_c2patool() -> Option<String> {
    let path = std::env::var_os("PATH")?;
    // Look for both the plain name and the Windows executable name.
    let names = ["c2patool", "c2patool.exe"];
    for dir in std::env::split_paths(&path) {
        for name in &names {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate.to_str().map(|s| s.to_string());
            }
        }
    }
    None
}

/// One-line, non-expert answer for a C2PA report.
fn plain_c2pa_answer(r: &c2pa::C2paReport) -> String {
    match r.verdict.as_str() {
        "trusted" => {
            if r.ai_source_type.is_some() || r.synthid_assertion {
                "AI-generated, and the provenance is cryptographically TRUSTED (signer verified)."
                    .to_string()
            } else {
                "Has a TRUSTED signed provenance record (signer verified).".to_string()
            }
        }
        "valid-untrusted" => {
            if r.ai_source_type.is_some() || r.synthid_assertion {
                "Claims to be AI-generated; signature is valid but signer is NOT on your trust list.".to_string()
            } else {
                "Has a valid signed record, but the signer is NOT on your trust list.".to_string()
            }
        }
        "tampered" => {
            "TAMPERED: the file was altered after it was signed — do not trust its provenance."
                .to_string()
        }
        _ => "No provenance record found. This is inconclusive — absence is not proof of anything."
            .to_string(),
    }
}
