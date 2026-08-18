//! Detector validation suite: the math anchor for the whole project.
//!
//! This is the in-tree Rust validation (run by `cargo test`); it complements
//! `tools/validate.py`, which cross-checks the same detectors against the external reference
//! implementations. These tests do NOT check against real Claude/Gemini output (impossible
//! without the vendors' private keys/tokenizers). Instead each uses a *known* watermark key
//! to generate a watermarked stream and an unwatermarked control, then asserts the detector
//! reports a strong signal on the former and noise on the latter — across every scheme
//! (SynthID, KGW, exp, unigram, SWEET, exp-edit) plus the grounded p-value special
//! functions. If any of these fail, the core g-value/statistic machinery is broken.

use ai_watermark_detector::{
    generate_control, generate_kgw, generate_watermarked, kgw_scores, ngram_g_values, score,
    Config, Scheme, XorShift64,
};

fn test_config() -> Config {
    Config::demo()
}
#[test]
fn watermarked_stream_scores_positive() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let candidates = 8usize;
    let mut rng = XorShift64::new(0xDEADBEEF);

    let wm = generate_watermarked(&cfg, vocab, len, candidates, &mut rng);
    let r = ngram_g_values(&wm, &cfg);

    // Tournament sampling should push mean_g well above 0.5 and produce a large z.
    assert!(
        r.mean_g > 0.55,
        "watermarked mean_g should be clearly above 0.5, got {}",
        r.mean_g
    );
    assert!(
        r.z > 6.0,
        "watermarked z-score should be strongly positive, got {}",
        r.z
    );
    assert!(
        r.approx_p_value < 1e-6,
        "watermarked p-value should be tiny, got {}",
        r.approx_p_value
    );
}

#[test]
fn control_stream_scores_neutral() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let mut rng = XorShift64::new(0x1234_5678);

    let ctrl = generate_control(vocab, len, &mut rng);
    let r = ngram_g_values(&ctrl, &cfg);

    // Random tokens should look like a fair coin: mean_g near 0.5, modest |z|.
    assert!(
        (r.mean_g - 0.5).abs() < 0.03,
        "control mean_g should be near 0.5, got {}",
        r.mean_g
    );
    assert!(
        r.z.abs() < 4.0,
        "control z-score should be modest, got {}",
        r.z
    );
}

#[test]
fn detector_separates_watermarked_from_control() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let candidates = 8usize;

    let mut rng_wm = XorShift64::new(1);
    let mut rng_ctrl = XorShift64::new(2);

    let wm = generate_watermarked(&cfg, vocab, len, candidates, &mut rng_wm);
    let ctrl = generate_control(vocab, len, &mut rng_ctrl);

    let z_wm = ngram_g_values(&wm, &cfg).z;
    let z_ctrl = ngram_g_values(&ctrl, &cfg).z;

    assert!(
        z_wm > z_ctrl + 5.0,
        "watermarked z ({z_wm}) should clearly exceed control z ({z_ctrl})"
    );
}

#[test]
fn stronger_tournament_yields_stronger_signal() {
    // More candidates per step => stronger bias => larger z. Sanity check on monotonicity.
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;

    let mut rng_weak = XorShift64::new(10);
    let mut rng_strong = XorShift64::new(10);

    let weak = generate_watermarked(&cfg, vocab, len, 2, &mut rng_weak);
    let strong = generate_watermarked(&cfg, vocab, len, 16, &mut rng_strong);

    let z_weak = ngram_g_values(&weak, &cfg).z;
    let z_strong = ngram_g_values(&strong, &cfg).z;

    assert!(
        z_strong > z_weak,
        "16-candidate tournament z ({z_strong}) should exceed 2-candidate z ({z_weak})"
    );
}

// ---------------------------------------------------------------------------
// KGW (green-list) scheme
// ---------------------------------------------------------------------------

#[test]
fn kgw_watermarked_stream_scores_positive() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let mut rng = XorShift64::new(0xABCDEF);

    let wm = generate_kgw(&cfg, vocab, len, 12, &mut rng);
    let r = kgw_scores(&wm, &cfg);

    // Green fraction should sit well above gamma (0.25) with a large z.
    assert!(
        r.mean_g > 0.30,
        "kgw watermarked green rate should exceed gamma, got {}",
        r.mean_g
    );
    assert!(r.z > 6.0, "kgw watermarked z should be strong, got {}", r.z);
    assert!(
        r.reliable,
        "2000-token sample should clear the reliability floor"
    );
}

#[test]
fn kgw_control_stream_scores_neutral() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let mut rng = XorShift64::new(0x9999);

    let ctrl = generate_control(vocab, len, &mut rng);
    let r = kgw_scores(&ctrl, &cfg);

    // Random tokens: green rate near gamma (0.25), modest |z|.
    assert!(
        (r.mean_g - cfg.gamma).abs() < 0.03,
        "kgw control green rate should be near gamma, got {}",
        r.mean_g
    );
    assert!(
        r.z.abs() < 4.0,
        "kgw control z should be modest, got {}",
        r.z
    );
}

#[test]
fn short_text_is_flagged_unreliable() {
    let cfg = test_config();
    let mut rng = XorShift64::new(7);
    // ~40 tokens: below the 100-position floor.
    let short = generate_kgw(&cfg, 8000, 40, 12, &mut rng);
    let r = kgw_scores(&short, &cfg);
    assert!(
        !r.reliable,
        "short samples must be flagged unreliable (usable={})",
        r.usable_positions
    );
}

// ---------------------------------------------------------------------------
// Cross-scheme independence: a watermark from one family must NOT masquerade as
// the other. This mirrors the real-world fact that detectors do not cross vendors.
// ---------------------------------------------------------------------------

#[test]
fn kgw_watermark_is_invisible_to_synthid_detector() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let mut rng = XorShift64::new(0x5150);

    let kgw_wm = generate_kgw(&cfg, vocab, len, 12, &mut rng);
    // Scored under the WRONG scheme (SynthID), it should look like noise.
    let wrong = score(&kgw_wm, &cfg, Scheme::SynthId);
    assert!(
        wrong.z.abs() < 5.0,
        "KGW watermark should not strongly trigger the SynthID detector, got z={}",
        wrong.z
    );
    // Scored under the RIGHT scheme it must be strong.
    let right = score(&kgw_wm, &cfg, Scheme::Kgw);
    assert!(
        right.z > 6.0,
        "KGW detector should flag KGW text, got z={}",
        right.z
    );
}

#[test]
fn synthid_watermark_is_invisible_to_kgw_detector() {
    let cfg = test_config();
    let vocab = 8000u64;
    let len = 2000usize;
    let mut rng = XorShift64::new(0x1D10);

    let synth_wm = generate_watermarked(&cfg, vocab, len, 12, &mut rng);
    let wrong = score(&synth_wm, &cfg, Scheme::Kgw);
    assert!(
        wrong.z.abs() < 5.0,
        "SynthID watermark should not strongly trigger the KGW detector, got z={}",
        wrong.z
    );
    let right = score(&synth_wm, &cfg, Scheme::SynthId);
    assert!(
        right.z > 6.0,
        "SynthID detector should flag SynthID text, got z={}",
        right.z
    );
}

// ---------------------------------------------------------------------------
// HF-faithful SynthID path: driven by an external sampling table (as exported from
// Google's transformers implementation). A degenerate all-ones/all-zeros table gives
// a deterministic mean-g, proving the table controls g-values exactly like Google's.
// ---------------------------------------------------------------------------

fn hf_config(table: Vec<u8>) -> Config {
    let mut c = test_config();
    c.sampling_table = Some(table);
    c
}

#[test]
fn synthid_hf_uses_sampling_table() {
    let mut rng = XorShift64::new(42);
    let tokens = generate_control(8000, 500, &mut rng);

    // All-ones table: every position is green => mean_g == 1.0.
    let ones = score(&tokens, &hf_config(vec![1u8; 1 << 12]), Scheme::SynthId);
    assert!(
        (ones.mean_g - 1.0).abs() < 1e-9,
        "all-ones table => mean_g 1.0, got {}",
        ones.mean_g
    );

    // All-zeros table: mean_g == 0.0.
    let zeros = score(&tokens, &hf_config(vec![0u8; 1 << 12]), Scheme::SynthId);
    assert!(
        zeros.mean_g.abs() < 1e-9,
        "all-zeros table => mean_g 0.0, got {}",
        zeros.mean_g
    );
}

// ---------------------------------------------------------------------------
// HF-faithful KGW path: driven by an external green-list oracle (as exported from
// HF's real lefthash WatermarkLogitsProcessor). A stream whose every token is green
// for its predecessor scores mean_g == 1.0; a stream with no green tokens scores 0.0.
// ---------------------------------------------------------------------------

#[test]
fn kgw_hf_uses_green_oracle() {
    use std::collections::HashMap;

    // Oracle over a tiny vocab: token t's green set is {t+1 mod V} (deterministic).
    let vocab = 50u64;
    let mut oracle: HashMap<String, Vec<u64>> = HashMap::new();
    for t in 0..vocab {
        oracle.insert(t.to_string(), vec![(t + 1) % vocab]);
    }
    let mut cfg = test_config();
    cfg.green_oracle = Some(oracle);

    // All-green stream: 0,1,2,... each token is green for its predecessor.
    let all_green: Vec<u64> = (0..40).map(|t| t % vocab).collect();
    let g = score(&all_green, &cfg, Scheme::Kgw);
    assert!(
        (g.mean_g - 1.0).abs() < 1e-9,
        "all-green stream => mean_g 1.0, got {}",
        g.mean_g
    );

    // No-green stream: repeated same token => successor never in green set {t+1}.
    let none_green: Vec<u64> = vec![7u64; 40];
    let n = score(&none_green, &cfg, Scheme::Kgw);
    assert!(
        n.mean_g.abs() < 1e-9,
        "no-green stream => mean_g 0.0, got {}",
        n.mean_g
    );
}

// ---------------------------------------------------------------------------
// Exponential (Aaronson/Gumbel) scheme: per-token score is -ln(1 - u). A stream whose
// u-values are all near 1 scores a large mean and z; u-values at 0.5 give mean ln(2).
// ---------------------------------------------------------------------------

#[test]
fn exp_scheme_uses_u_oracle() {
    use std::collections::HashMap;

    let mut cfg = test_config();
    cfg.context_width = 1;

    // Build a stream 0,1,2,...,149 with a u-oracle assigning every (prev|cur) a high u.
    let toks: Vec<u64> = (0..150u64).collect();
    let mut oracle: HashMap<String, f64> = HashMap::new();
    for w in toks.windows(2) {
        oracle.insert(format!("{}|{}", w[0], w[1]), 0.9); // -ln(0.1) ~= 2.302
    }
    cfg.u_oracle = Some(oracle);
    let r = score(&toks, &cfg, Scheme::Exp);
    let expected = -(1.0f64 - 0.9).ln();
    assert!(
        (r.mean_g - expected).abs() < 1e-9,
        "mean score should be -ln(0.1), got {}",
        r.mean_g
    );
    assert!(r.z > 5.0, "high-u stream should have large z, got {}", r.z);
    assert!(r.reliable, "150 tokens should clear the reliability floor");

    // Neutral u=0.5 => per-token score ln(2) ~= 0.693 < 1 => negative z (looks unwatermarked).
    let mut cfg2 = test_config();
    cfg2.context_width = 1;
    let mut oracle2: HashMap<String, f64> = HashMap::new();
    for w in toks.windows(2) {
        oracle2.insert(format!("{}|{}", w[0], w[1]), 0.5);
    }
    cfg2.u_oracle = Some(oracle2);
    let r2 = score(&toks, &cfg2, Scheme::Exp);
    assert!(
        r2.z < 0.0,
        "u=0.5 stream scores below H0 mean => negative z, got {}",
        r2.z
    );
}

// ---------------------------------------------------------------------------
// Unigram (Zhao 2024): a single GLOBAL green set applies at every position. A stream made
// entirely of green tokens scores mean_g ~ 1 and a large z; a stream of red tokens ~ 0.
// ---------------------------------------------------------------------------

#[test]
fn unigram_uses_global_greenlist() {
    let mut cfg = test_config();
    cfg.gamma = 0.5;
    // Global green set = even token ids in [0, 200).
    let greens: Vec<u64> = (0..200u64).filter(|t| t % 2 == 0).collect();
    cfg.unigram_greenlist = Some(greens);

    // All-green stream (even ids) => mean_g == 1, z large.
    let all_green: Vec<u64> = (0..150u64).map(|i| (i * 2) % 200).collect();
    let g = score(&all_green, &cfg, Scheme::Unigram);
    assert!(
        (g.mean_g - 1.0).abs() < 1e-9,
        "all-green => mean_g 1.0, got {}",
        g.mean_g
    );
    assert!(
        g.z > 10.0,
        "all-green stream should have large z, got {}",
        g.z
    );
    assert!(g.reliable);

    // All-red stream (odd ids) => mean_g == 0, z very negative.
    let all_red: Vec<u64> = (0..150u64).map(|i| (i * 2 + 1) % 200).collect();
    let r = score(&all_red, &cfg, Scheme::Unigram);
    assert!(
        r.mean_g.abs() < 1e-9,
        "all-red => mean_g 0.0, got {}",
        r.mean_g
    );
    assert!(
        r.z < -10.0,
        "all-red stream should have large negative z, got {}",
        r.z
    );
}

// ---------------------------------------------------------------------------
// SWEET (Lee 2024): only HIGH-ENTROPY positions (entropy_mask == true) are scored. Low-entropy
// positions are ignored entirely, so a green-heavy stream masked to few positions has small T.
// ---------------------------------------------------------------------------

#[test]
fn sweet_scores_only_high_entropy_positions() {
    use std::collections::HashMap;

    let mut cfg = test_config();
    cfg.gamma = 0.5;
    // Greenlist oracle keyed by previous token: successor t+1 is green for prev t.
    let toks: Vec<u64> = (0..60u64).collect();
    let mut oracle: HashMap<String, Vec<u64>> = HashMap::new();
    for w in toks.windows(2) {
        oracle.insert(w[0].to_string(), vec![w[1]]);
    }
    cfg.green_oracle = Some(oracle);

    // Mask only even generated positions high-entropy => ~half the positions are scored.
    let mask: Vec<bool> = (0..toks.len()).map(|i| i % 2 == 0).collect();
    cfg.entropy_mask = Some(mask);

    let r = score(&toks, &cfg, Scheme::Sweet);
    // Every scored position is green (successor always in greenlist) => mean_g == 1.
    assert!(
        (r.mean_g - 1.0).abs() < 1e-9,
        "green successors => mean_g 1.0, got {}",
        r.mean_g
    );
    // Only masked positions counted, so total < positions.
    assert!(
        r.total < r.positions,
        "SWEET should skip low-entropy positions"
    );
    assert!(r.total > 0, "some high-entropy positions should be scored");
}

// ---------------------------------------------------------------------------
// EXP-Edit (Kuditipudi 2024): the statistic is the minimum edit-distance alignment cost
// between the token stream and the key's uniform matrix xi. A stream aligned to high-xi
// tokens (near 1) gets a very negative (strong) statistic; a mismatched stream is near 0.
// ---------------------------------------------------------------------------

#[test]
fn exp_edit_alignment_statistic() {
    let mut cfg = test_config();
    // Tiny key matrix: pseudo_length = 6 rows, vocab = 4. Row t makes token t high (0.99).
    let n = 6usize;
    let vocab = 4usize;
    let mut xi: Vec<Vec<f64>> = Vec::new();
    for t in 0..n {
        let mut row = vec![0.01f64; vocab];
        row[t % vocab] = 0.99;
        xi.push(row);
    }
    cfg.exp_edit_xi = Some(xi);
    cfg.exp_edit_gamma = Some(0.0);

    // Aligned stream: tokens follow the argmax of each row => matches produce log(1-0.99) each.
    let aligned: Vec<u64> = (0..6u64).map(|t| (t as usize % vocab) as u64).collect();
    let a = score(&aligned, &cfg, Scheme::ExpEdit);
    // Mismatched stream: constant token that is LOW in most rows => alignment cost near 0.
    let mismatched: Vec<u64> = vec![3u64; 6];
    let m = score(&mismatched, &cfg, Scheme::ExpEdit);

    assert!(
        a.mean_g < m.mean_g,
        "aligned stream should have a smaller (stronger) statistic: aligned={} mismatched={}",
        a.mean_g,
        m.mean_g
    );
    assert!(
        a.mean_g < -5.0,
        "aligned stream statistic should be strongly negative, got {}",
        a.mean_g
    );
}

// ---------------------------------------------------------------------------
// Three Bricks (Fernandez et al. 2023) grounded p-values: exact null distributions.
// betainc / gammaincc must match known reference values (cross-checked against scipy).
// ---------------------------------------------------------------------------

#[test]
fn grounded_special_functions_match_reference() {
    use ai_watermark_detector::{betainc, gammaincc, kgw_binomial_p};

    // gammaincc(a, x) reference values (scipy.special.gammaincc).
    assert!(
        (gammaincc(1.0, 1.0) - 0.367_879_441_171_442_33).abs() < 1e-12,
        "gammaincc(1,1) should be exp(-1), got {}",
        gammaincc(1.0, 1.0)
    );
    assert!(
        (gammaincc(10.0, 10.0) - 0.457_929_714_331_402_2).abs() < 1e-9,
        "gammaincc(10,10) mismatch, got {}",
        gammaincc(10.0, 10.0)
    );

    // betainc(a, b, x) reference values (scipy.special.betainc / regularized I_x(a,b)).
    assert!(
        (betainc(2.0, 3.0, 0.5) - 0.687_5).abs() < 1e-9,
        "betainc(2,3,0.5) should be 0.6875, got {}",
        betainc(2.0, 3.0, 0.5)
    );
    assert!(
        (betainc(0.5, 0.5, 0.5) - 0.5).abs() < 1e-9,
        "betainc(0.5,0.5,0.5) should be 0.5, got {}",
        betainc(0.5, 0.5, 0.5)
    );

    // Grounded KGW p: all-green short stream should give a tiny p-value; half-green ~ 0.5-ish.
    let all_green = kgw_binomial_p(20, 20, 0.5);
    assert!(
        all_green < 1e-5,
        "20/20 green => tiny grounded p, got {all_green}"
    );
    let half = kgw_binomial_p(10, 20, 0.5);
    assert!(
        half > 0.3 && half < 0.7,
        "10/20 green => moderate grounded p, got {half}"
    );
}
