//! Controlled-corpus generator for validating the all-in-one detector.
//!
//! Emits token-id streams (as text files) using KNOWN watermark keys for BOTH schemes
//! (`kgw` = Kirchenbauer green-list, `synthid` = Gemini-style), plus unwatermarked
//! baselines. This lets the harness confirm each detector recovers its own signal,
//! observe degradation under simulated copy-paste / edits / rewrites, and verify that
//! detectors do NOT cross schemes. These are research fixtures, NOT real model output.
//!
//! Usage: gen_corpus [OUT_DIR] [SCHEME]
//!   SCHEME = kgw | synthid   (default: kgw)

use ai_watermark_detector::{generate, generate_control, Config, Scheme, XorShift64};
use std::fs;
use std::path::Path;

fn write_ids(dir: &Path, name: &str, ids: &[u64]) {
    fs::create_dir_all(dir).unwrap();
    let body = ids
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(dir.join(name), body).unwrap();
}

/// Simulate light edits: replace a fraction of tokens with random vocab tokens.
fn corrupt(ids: &[u64], frac: f64, vocab: u64, rng: &mut XorShift64) -> Vec<u64> {
    ids.iter()
        .map(|&t| {
            if (rng.next_u64() as f64 / u64::MAX as f64) < frac {
                rng.below(vocab)
            } else {
                t
            }
        })
        .collect()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpus".to_string());
    let scheme_arg = std::env::args().nth(2).unwrap_or_else(|| "kgw".to_string());
    let scheme = Scheme::parse(&scheme_arg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let root = Path::new(&out);

    let vocab = 8000u64;
    let cfg = Config::demo();
    let len = 1500usize;
    let per_cat = 12usize;

    // Watermarked samples under the chosen scheme (strong tournament).
    let mut rng = XorShift64::new(0xC1A0DE);
    for i in 0..per_cat {
        let ids = generate(&cfg, scheme, vocab, len, 12, &mut rng);
        write_ids(
            &root.join("watermarked"),
            &format!("sample_{i:02}.txt"),
            &ids,
        );

        // copy-paste: identical (watermark designed to survive this).
        write_ids(
            &root.join("watermarked-copy-paste"),
            &format!("sample_{i:02}.txt"),
            &ids,
        );
        // minor edits: ~5% tokens changed.
        let minor = corrupt(&ids, 0.05, vocab, &mut rng);
        write_ids(
            &root.join("watermarked-minor-edits"),
            &format!("sample_{i:02}.txt"),
            &minor,
        );
        // heavy edits: ~30% tokens changed.
        let heavy = corrupt(&ids, 0.30, vocab, &mut rng);
        write_ids(
            &root.join("watermarked-heavy-edits"),
            &format!("sample_{i:02}.txt"),
            &heavy,
        );
        // rewrite: ~70% tokens changed (approximates a full paraphrase).
        let rewrite = corrupt(&ids, 0.70, vocab, &mut rng);
        write_ids(
            &root.join("watermarked-rewrite"),
            &format!("sample_{i:02}.txt"),
            &rewrite,
        );
    }

    // A "wrong-scheme" watermarked category: watermarked under the OTHER family, to
    // confirm the harness detector (run with --scheme <scheme>) does NOT flag it.
    let other = match scheme {
        Scheme::Kgw => Scheme::SynthId,
        Scheme::SynthId => Scheme::Kgw,
        Scheme::Exp | Scheme::Unigram | Scheme::Sweet | Scheme::ExpEdit => Scheme::Kgw,
    };
    let mut rng_other = XorShift64::new(0x0DDBA11);
    for i in 0..per_cat {
        let ids = generate(&cfg, other, vocab, len, 12, &mut rng_other);
        write_ids(
            &root.join("other-scheme-watermark"),
            &format!("sample_{i:02}.txt"),
            &ids,
        );
    }

    // Unwatermarked baselines: human / gpt / gemini modeled as no-watermark text.
    for (name, seed) in [("human", 0xA), ("gpt", 0xB), ("gemini", 0xC)] {
        let mut r = XorShift64::new(seed);
        for i in 0..per_cat {
            let ids = generate_control(vocab, len, &mut r);
            write_ids(&root.join(name), &format!("sample_{i:02}.txt"), &ids);
        }
    }

    println!(
        "Wrote controlled corpus to {}/ (scheme: {})",
        root.display(),
        scheme.as_str()
    );
    println!("Categories: watermarked, watermarked-copy-paste, watermarked-minor-edits, watermarked-heavy-edits, watermarked-rewrite, other-scheme-watermark, human, gpt, gemini");
    println!("Score with: python3 tools/corpus_harness.py --corpus {} --config config.example.json --pretokenized --scheme {}", root.display(), scheme.as_str());
}
