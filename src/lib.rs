//! All-in-one statistical watermark scoring primitives for two public schemes.
//!
//! This is a *research reproduction* of two published logit-biasing text-watermark
//! families:
//!   * `synthid` - the SynthID-Text g-value / tournament mechanism (used by Gemini).
//!   * `kgw` - the Kirchenbauer-Geiping-Wen-Katz-Miers-Goldstein (2023) green-list
//!     mechanism (the category Anthropic's public description of Claude matches).
//!
//! These do NOT constitute an authoritative verifier for any production model.
//! Authoritative detection requires the originating vendor's private key, exact
//! tokenizer, and sampling configuration - none of which are published. Detectors do
//! not cross vendors: a Claude/KGW detector says nothing about Gemini/SynthID text
//! and vice versa.

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub mod output;
pub use output::{render, OutputFormat};

pub const MULTIPLIER: u64 = 6364136223846793005;
pub const INCREMENT: u64 = 1;
pub const HASH_ROUNDS: usize = 12;
pub const I64_MAX: u64 = 9_223_372_036_854_775_807;

/// Public-research reliability floor: below this many usable positions the statistical
/// test is not trustworthy (KGW paper reports ~100+ continuous tokens).
pub const RELIABLE_TOKEN_FLOOR: usize = 100;

/// Which watermark family to score/generate against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    SynthId,
    Kgw,
    /// Aaronson / Kuditipudi-style exponential (Gumbel) scheme: a keyed PRNG assigns each
    /// token a uniform u in (0,1); the per-token detection score is -ln(1 - u_chosen),
    /// which has mean 1 under H0 and > 1 when watermarked.
    Exp,
    /// Unigram (Zhao et al. 2024): KGW with a single GLOBAL green/red split (no per-token
    /// seeding), which makes it markedly more robust to paraphrase/edits. Every token is
    /// scored against one fixed green set.
    Unigram,
    /// SWEET (Lee et al. 2024): KGW restricted to HIGH-ENTROPY positions (a per-position
    /// entropy mask decides which tokens are scored), the canonical code watermark.
    Sweet,
    /// EXP-Edit / ITS-Edit (Kuditipudi et al. 2024): distortion-free AND edit-robust. The
    /// detection statistic is the minimum edit-distance alignment cost between the token
    /// stream and the key's uniform sequence; significance is a permutation test. Unlike the
    /// plain exp scheme, this survives insertions/deletions (paraphrase-ish edits).
    ExpEdit,
}

impl Scheme {
    pub fn parse(s: &str) -> Result<Scheme, String> {
        match s.to_ascii_lowercase().as_str() {
            "synthid" | "synthid-text" | "gemini" => Ok(Scheme::SynthId),
            "kgw" | "claude" | "greenlist" | "green-list" => Ok(Scheme::Kgw),
            "exp" | "exponential" | "gumbel" | "aaronson" => Ok(Scheme::Exp),
            "unigram" | "unigram-fixed" | "zhao" => Ok(Scheme::Unigram),
            "sweet" | "sweet-code" => Ok(Scheme::Sweet),
            "exp-edit" | "expedit" | "exp_edit" | "its-edit" | "kuditipudi" => Ok(Scheme::ExpEdit),
            other => Err(format!(
                "unknown scheme: {other} (use `synthid`, `kgw`, `exp`, `unigram`, `sweet`, or `exp-edit`)"
            )),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Scheme::SynthId => "synthid",
            Scheme::Kgw => "kgw",
            Scheme::Exp => "exp",
            Scheme::Unigram => "unigram",
            Scheme::Sweet => "sweet",
            Scheme::ExpEdit => "exp-edit",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub ngram_len: usize,
    pub keys: Vec<u64>,
    /// KGW-only: fraction of the vocabulary placed on the green list (default 0.25).
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// KGW-only vocab size used to derive the green list. If absent, the detector
    /// infers it from the maximum observed token id + 1.
    #[serde(default)]
    pub vocab_size: Option<u64>,
    /// SynthID (HF-faithful): the precomputed Bernoulli sampling table (values 0/1),
    /// exported from Google's `SynthIDTextWatermarkLogitsProcessor`. When present, the
    /// SynthID scheme uses Google's exact hashing + this table, matching the official
    /// g-function bit-for-bit. When absent, a standalone reproduction is used instead.
    #[serde(default)]
    pub sampling_table: Option<Vec<u8>>,
    /// KGW (HF-faithful): a green-list oracle mapping a context signature (as a string key)
    /// to the list of green token IDs, exported from HF's real `WatermarkLogitsProcessor`.
    /// The signature is the space-joined last `context_width` tokens for `selfhash`, or the
    /// single last token for `lefthash` (whose seed only depends on the previous token).
    /// When present, the KGW scheme matches HF bit-for-bit.
    #[serde(default)]
    pub green_oracle: Option<std::collections::HashMap<String, Vec<u64>>>,
    /// KGW seeding scheme: `lefthash` (default) or `selfhash`. Controls how the oracle key
    /// is derived from the context window when `green_oracle` is present.
    #[serde(default = "default_seeding")]
    pub seeding_scheme: String,
    /// KGW context width (number of trailing tokens the seed depends on). For `lefthash`
    /// the seed uses only the last token regardless; for `selfhash` it uses the full window.
    #[serde(default = "default_context_width")]
    pub context_width: usize,
    /// Exponential/Gumbel scheme: a map from a `"context|token"` signature to the uniform
    /// value u in (0,1) that the keyed PRNG assigns that (context, token) pair. Exported
    /// from a reference implementation so the per-token score -ln(1 - u) matches exactly.
    #[serde(default)]
    pub u_oracle: Option<std::collections::HashMap<String, f64>>,
    /// Unigram: the single GLOBAL green token set (sorted), exported from the reference.
    /// Every token is scored against this fixed set (no per-token seeding).
    #[serde(default)]
    pub unigram_greenlist: Option<Vec<u64>>,
    /// SWEET: per-position boolean entropy mask (true = high-entropy, scored). Exported from
    /// the generator's own entropy computation. Only masked positions count toward the z.
    #[serde(default)]
    pub entropy_mask: Option<Vec<bool>>,
    /// EXP-Edit (Kuditipudi et al. 2024): the key's uniform matrix `xi` of shape
    /// [pseudo_length, vocab_size], row-major, exported from the reference MersenneRNG. Used
    /// by the edit-distance alignment detector. Small vocab/pseudo_length for tractability.
    #[serde(default)]
    pub exp_edit_xi: Option<Vec<Vec<f64>>>,
    /// EXP-Edit: block size k for the alignment statistic (default = len(tokens)).
    #[serde(default)]
    pub exp_edit_k: Option<usize>,
    /// EXP-Edit: insertion/deletion penalty gamma for the Levenshtein alignment (default 0).
    #[serde(default)]
    pub exp_edit_gamma: Option<f64>,
}

fn default_seeding() -> String {
    "lefthash".to_string()
}
fn default_context_width() -> usize {
    1
}

fn default_gamma() -> f64 {
    0.25
}

impl Config {
    /// The canonical research/demo config: the fixed 30-key `lefthash` KGW/SynthID setup used
    /// by the corpus generator and the in-tree tests, mirroring `config.example.json`. Kept in
    /// one place so those call sites can't drift apart. Optional oracle fields are left unset
    /// (the standalone reproductions are used unless an exported table/oracle is supplied).
    pub fn demo() -> Config {
        Config {
            ngram_len: 5,
            keys: vec![
                654, 400, 836, 123, 340, 443, 597, 160, 57, 901, 712, 333, 812, 449, 218, 777, 102,
                503, 911, 284, 620, 145, 388, 731, 266, 594, 847, 319, 705, 432,
            ],
            gamma: 0.25,
            vocab_size: Some(8000),
            sampling_table: None,
            green_oracle: None,
            seeding_scheme: default_seeding(),
            context_width: default_context_width(),
            u_oracle: None,
            unigram_greenlist: None,
            entropy_mask: None,
            exp_edit_xi: None,
            exp_edit_k: None,
            exp_edit_gamma: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResultRow {
    pub scheme: Scheme,
    pub tokens: usize,
    pub positions: usize,
    pub usable_positions: usize,
    pub ones: usize,
    pub total: usize,
    pub mean_g: f64,
    pub z: f64,
    pub approx_p_value: f64,
    /// True when there are enough usable positions to trust the statistic.
    pub reliable: bool,
    /// SynthID only: the "weighted mean" detector score (Nature 2024 / MarkLLM), which
    /// re-weights the g-values across watermarking depth (default linear weights 10..1
    /// normalized to sum to depth). None for other schemes/paths.
    pub weighted_mean_g: Option<f64>,
    /// "Three Bricks" (Fernandez et al. 2023) GROUNDED p-value using the exact null
    /// distribution instead of the Gaussian z-approximation: binomial (betainc) for KGW /
    /// unigram / sweet, Gamma (gammaincc) for exp. Better calibrated on short texts. None
    /// for schemes without a closed-form null here (synthid, exp-edit).
    pub grounded_p_value: Option<f64>,
}

impl ResultRow {
    /// The "no usable signal" row returned when a stream is too short or lacks the oracle
    /// a scheme needs. Every detector shares this shape, so building it in one place keeps
    /// the early-return paths honest (an unreliable, p=1.0 result).
    pub fn empty(scheme: Scheme, tokens: usize) -> ResultRow {
        ResultRow {
            scheme,
            tokens,
            positions: 0,
            usable_positions: 0,
            ones: 0,
            total: 0,
            mean_g: 0.0,
            z: 0.0,
            approx_p_value: 1.0,
            reliable: false,
            weighted_mean_g: None,
            grounded_p_value: None,
        }
    }
}

/// Binomial z-score for a green/red count under H0 P(green) = gamma. Returns 0 for the
/// degenerate empty/zero-variance case so callers need no extra guard.
fn binomial_z(green: usize, total: usize, gamma: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let mu = total as f64 * gamma;
    let sigma = (total as f64 * gamma * (1.0 - gamma)).sqrt();
    if sigma > 0.0 {
        (green as f64 - mu) / sigma
    } else {
        0.0
    }
}

/// Bernoulli(0.5) z-score for a g==1 count (the SynthID null: a fair sampling table).
fn bernoulli_z(ones: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    (ones as f64 - total as f64 * 0.5) / (total as f64 * 0.25).sqrt()
}

fn rate(numerator: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        numerator as f64 / total as f64
    }
}

/// Space-joined signature of a token slice, the key shape used by the exported HF green /
/// exp oracles.
fn signature(tokens: &[u64]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `token` is on the greenlist that `oracle` maps `prev` to. Oracle lists are
/// exported sorted, so a binary search is exact.
fn is_green(oracle: &std::collections::HashMap<String, Vec<u64>>, prev: u64, token: u64) -> bool {
    oracle
        .get(&prev.to_string())
        .is_some_and(|greens| greens.binary_search(&token).is_ok())
}

pub fn parse_token_ids(s: &str) -> Result<Vec<u64>, String> {
    s.split(|c: char| c == ',' || c.is_ascii_whitespace())
        .filter(|x| !x.is_empty())
        .map(|x| {
            x.parse::<u64>()
                .map_err(|_| format!("invalid token id: {x}"))
        })
        .collect()
}

pub fn lcg_accumulate(mut hash: u64, data: &[u64]) -> u64 {
    for &x in data {
        hash = hash.wrapping_add(x);
        hash = hash.wrapping_mul(MULTIPLIER);
        hash = hash.wrapping_add(INCREMENT);
    }
    hash
}

pub fn hash_iv(keys: &[u64]) -> u64 {
    // Match the public SynthID reference: SHA-256(keys bytes) reduced modulo int64 max.
    let mut hasher = Sha256::new();
    for &key in keys {
        hasher.update((key as i64).to_ne_bytes());
    }
    let digest = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first) % I64_MAX
}

pub fn g_value(mut key: u64) -> u8 {
    // Public reference repeatedly hashes, then takes bit 30.
    for _ in 0..HASH_ROUNDS {
        key = lcg_accumulate(key, &[1]);
        key >>= 64 / HASH_ROUNDS;
    }
    ((key >> 30) & 1) as u8
}

pub fn context_hash(iv: u64, context: &[u64]) -> u64 {
    lcg_accumulate(iv, context)
}

// ---------------------------------------------------------------------------
// HF-faithful SynthID hashing (matches transformers' SynthIDTextWatermarkLogitsProcessor)
// ---------------------------------------------------------------------------

/// Google's `compute_ngram_keys` for a single n-gram + single key:
///   h = accumulate_hash(1, full_ngram); h = accumulate_hash(h, [key])
/// Uses i64 wrapping arithmetic, matching torch int64 semantics.
pub fn hf_ngram_key(ngram: &[u64], key: u64) -> u64 {
    let mut h: u64 = 1;
    h = lcg_accumulate(h, ngram);
    lcg_accumulate(h, &[key])
}

/// Google's `sample_g_values`: g = sampling_table[ngram_key % table_size].
pub fn hf_g_value(ngram_key: u64, table: &[u8]) -> u8 {
    let idx = (ngram_key % table.len() as u64) as usize;
    table[idx]
}

/// Google's context-repetition key: accumulate_hash(1, n_minus_1_gram). Two contexts
/// collide iff this value matches (same criterion the official detector uses).
pub fn hf_context_key(context: &[u64]) -> u64 {
    lcg_accumulate(1, context)
}

/// HF-faithful SynthID detector. Requires `sampling_table` in config; matches Google's
/// official g-function exactly. Mean g under H0 is ~0.5 (fair Bernoulli table).
pub fn synthid_hf_scores(tokens: &[u64], cfg: &Config, table: &[u8]) -> ResultRow {
    let n = cfg.ngram_len;
    assert!(n >= 2);
    assert!(!cfg.keys.is_empty());

    if tokens.len() < n {
        return ResultRow::empty(Scheme::SynthId, tokens.len());
    }

    let positions = tokens.len() - n + 1;
    let mut seen = std::collections::HashSet::with_capacity(positions);
    let mut ones = 0usize;
    let mut total = 0usize;
    let depth = cfg.keys.len();
    // Per-depth g sums and per-depth unmasked counts, for the weighted-mean detector.
    let mut per_depth_ones = vec![0f64; depth];
    let mut unmasked = 0usize;

    for i in 0..positions {
        let ngram = &tokens[i..i + n];
        let context = &ngram[..n - 1];
        if !seen.insert(hf_context_key(context)) {
            continue;
        }
        unmasked += 1;
        for (d, &key) in cfg.keys.iter().enumerate() {
            let nk = hf_ngram_key(ngram, key);
            let g = hf_g_value(nk, table) as usize;
            ones += g;
            per_depth_ones[d] += g as f64;
            total += 1;
        }
    }

    let mean = rate(ones, total);
    // Weighted-mean detector (MarkLLM/Nature): default weights linear 10..1 across depth,
    // normalized to sum to `depth`; score = sum_d(w_d * sum_pos g) / (depth * unmasked).
    let weighted_mean_g = if total == 0 || unmasked == 0 {
        None
    } else if depth == 1 {
        // With a single depth the weighted mean equals the plain mean.
        Some(mean)
    } else {
        let mut weights: Vec<f64> = (0..depth)
            .map(|d| 10.0 - 9.0 * (d as f64) / ((depth - 1) as f64))
            .collect();
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w *= depth as f64 / wsum;
        }
        let weighted: f64 = per_depth_ones
            .iter()
            .zip(&weights)
            .map(|(g, w)| g * w)
            .sum();
        Some(weighted / (depth as f64 * unmasked as f64))
    };
    let z = bernoulli_z(ones, total);
    let p = normal_upper_tail(z);
    let usable = total.checked_div(depth).unwrap_or(0);

    ResultRow {
        scheme: Scheme::SynthId,
        tokens: tokens.len(),
        positions,
        usable_positions: usable,
        ones,
        total,
        mean_g: mean,
        z,
        approx_p_value: p,
        reliable: usable >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g,
        grounded_p_value: None,
    }
}

/// g-value for a single candidate token at a given context, aggregated over all keys.
/// Returns the count of keys for which g == 1. This is the quantity the generator
/// maximizes and the detector measures, so they must share this function.
pub fn candidate_g_sum(iv: u64, context: &[u64], candidate: u64, keys: &[u64]) -> usize {
    let ch = context_hash(iv, context);
    keys.iter()
        .map(|&key| g_value(lcg_accumulate(ch, &[candidate, key])) as usize)
        .sum()
}

pub fn ngram_g_values(tokens: &[u64], cfg: &Config) -> ResultRow {
    let n = cfg.ngram_len;
    assert!(n >= 2);
    assert!(!cfg.keys.is_empty());

    if tokens.len() < n {
        return ResultRow::empty(Scheme::SynthId, tokens.len());
    }

    let iv = hash_iv(&cfg.keys);
    let positions = tokens.len() - n + 1;

    let mut seen = std::collections::HashSet::with_capacity(positions);
    let mut ones = 0usize;
    let mut total = 0usize;
    let depth = cfg.keys.len();

    for i in 0..positions {
        let context = &tokens[i..i + n - 1];
        let repeated = !seen.insert(context_hash(iv, context));
        if repeated {
            continue;
        }

        let s = candidate_g_sum(iv, context, tokens[i + n - 1], &cfg.keys);
        ones += s;
        total += cfg.keys.len();
    }

    let mean = rate(ones, total);
    let z = bernoulli_z(ones, total);

    let p = normal_upper_tail(z);
    let usable = total.checked_div(depth).unwrap_or(0);
    ResultRow {
        scheme: Scheme::SynthId,
        tokens: tokens.len(),
        positions,
        usable_positions: usable,
        ones,
        total,
        mean_g: mean,
        z,
        approx_p_value: p,
        reliable: usable >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: None,
    }
}

/// Score a token stream under the requested scheme.
pub fn score(tokens: &[u64], cfg: &Config, scheme: Scheme) -> ResultRow {
    match scheme {
        Scheme::SynthId => match &cfg.sampling_table {
            // HF-faithful path: matches Google's official g-function bit-for-bit.
            Some(table) if !table.is_empty() => synthid_hf_scores(tokens, cfg, table),
            // Standalone reproduction (no external table available).
            _ => ngram_g_values(tokens, cfg),
        },
        Scheme::Kgw => match &cfg.green_oracle {
            // HF-faithful path: matches HF's real lefthash WatermarkDetector bit-for-bit.
            Some(oracle) if !oracle.is_empty() => kgw_hf_scores(tokens, cfg, oracle),
            // Standalone reproduction (no external oracle available).
            _ => kgw_scores(tokens, cfg),
        },
        Scheme::Exp => exp_scores(tokens, cfg),
        Scheme::Unigram => unigram_scores(tokens, cfg),
        Scheme::Sweet => sweet_scores(tokens, cfg),
        Scheme::ExpEdit => exp_edit_scores(tokens, cfg),
    }
}

/// EXP-Edit / ITS-Edit (Kuditipudi et al. 2024) detection statistic. This reproduces the
/// reference `test_stat` / `levenshtein` exactly: for each alignment offset j into the key
/// sequence, compute the edit-distance alignment cost between the token block and the key's
/// uniform rows, and return the minimum (lower = stronger watermark). The permutation-test
/// p-value is orchestrated by the caller (it needs random reference keys); here we compute
/// the exact statistic, reported in `mean_g` (the alignment cost). This scheme is the only
/// text watermark that is BOTH distortion-free and robust to insertions/deletions.
pub fn exp_edit_scores(tokens: &[u64], cfg: &Config) -> ResultRow {
    let empty = ResultRow::empty(Scheme::ExpEdit, tokens.len());
    let xi = match &cfg.exp_edit_xi {
        Some(x) if !x.is_empty() => x,
        _ => return empty,
    };
    if tokens.is_empty() {
        return empty;
    }
    let n = xi.len(); // pseudo_length (number of key rows)
    let gamma = cfg.exp_edit_gamma.unwrap_or(0.0);
    let k = cfg.exp_edit_k.unwrap_or(tokens.len()).min(tokens.len());
    let m = tokens.len();

    // Reference `adjacency` uses i in 0..(m-(k-1)); with k == m there is a single block.
    let num_blocks = m - (k - 1);
    let mut best = f64::INFINITY;
    for i in 0..num_blocks {
        let block = &tokens[i..i + k];
        for j in 0..n {
            let cost = levenshtein_align(block, xi, j, n, gamma);
            if cost < best {
                best = cost;
            }
        }
    }
    let stat = if best.is_finite() { best } else { 0.0 };
    ResultRow {
        scheme: Scheme::ExpEdit,
        tokens: tokens.len(),
        positions: num_blocks,
        usable_positions: m,
        ones: 0,
        total: m,
        mean_g: stat,
        z: 0.0,
        approx_p_value: 1.0,
        reliable: m >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: None,
    }
}

/// Reference Levenshtein alignment cost (MarkLLM `levenshtein.pyx`): align token block `x`
/// against the key rows starting at offset `j0` (wrapping mod n). Match cost at (i,j) is
/// `log(1 - xi[(j0+j) % n][x[i]])`; insertion/deletion cost is `gamma`. Returns A[len(x)][k].
fn levenshtein_align(x: &[u64], xi: &[Vec<f64>], j0: usize, n: usize, gamma: f64) -> f64 {
    let ln = x.len();
    // y has `ln` rows too (the block xi[(j0+arange(k))%n], k == len(x) in one_run).
    let lm = ln;
    let mut a = vec![vec![0f64; lm + 1]; ln + 1];
    for i in 0..=ln {
        for j in 0..=lm {
            if i == 0 {
                a[i][j] = j as f64 * gamma;
            } else if j == 0 {
                a[i][j] = i as f64 * gamma;
            } else {
                let row = &xi[(j0 + (j - 1)) % n];
                let tok = x[i - 1] as usize;
                let p = row.get(tok).copied().unwrap_or(0.0);
                let cost = (1.0 - p).max(1e-300).ln();
                let mut best = a[i - 1][j] + gamma;
                if a[i][j - 1] + gamma < best {
                    best = a[i][j - 1] + gamma;
                }
                if a[i - 1][j - 1] + cost < best {
                    best = a[i - 1][j - 1] + cost;
                }
                a[i][j] = best;
            }
        }
    }
    a[ln][lm]
}

/// Unigram (Zhao et al. 2024) detector. A single GLOBAL green set (from `unigram_greenlist`)
/// applies at every position; there is no per-token seeding, which is what makes Unigram far
/// more robust to edits/paraphrase. Each unique token is scored once; z uses gamma as the H0
/// green rate (gamma = |greenlist| / vocab).
pub fn unigram_scores(tokens: &[u64], cfg: &Config) -> ResultRow {
    let empty = ResultRow::empty(Scheme::Unigram, tokens.len());
    let greens = match &cfg.unigram_greenlist {
        Some(g) if !g.is_empty() => g,
        _ => return empty,
    };
    let gamma = cfg.gamma;
    // Score every position (Unigram scores each generated token against the fixed split).
    let mut green = 0usize;
    let mut total = 0usize;
    for &t in tokens {
        total += 1;
        if greens.binary_search(&t).is_ok() {
            green += 1;
        }
    }
    if total == 0 {
        return empty;
    }
    let mean = rate(green, total);
    let z = binomial_z(green, total, gamma);
    ResultRow {
        scheme: Scheme::Unigram,
        tokens: tokens.len(),
        positions: total,
        usable_positions: total,
        ones: green,
        total,
        mean_g: mean,
        z,
        approx_p_value: normal_upper_tail(z),
        reliable: total >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: Some(kgw_binomial_p(green, total, gamma)),
    }
}

/// SWEET (Lee et al. 2024) detector. KGW `lefthash` scoring, but restricted to HIGH-ENTROPY
/// positions per the `entropy_mask` (the generator only watermarks where the model was
/// uncertain, preserving low-entropy code tokens). Only masked positions count toward z.
/// Requires a `green_oracle` (the KGW greenlist) plus the per-position `entropy_mask`.
pub fn sweet_scores(tokens: &[u64], cfg: &Config) -> ResultRow {
    let empty = ResultRow::empty(Scheme::Sweet, tokens.len());
    let oracle = match &cfg.green_oracle {
        Some(o) if !o.is_empty() => o,
        _ => return empty,
    };
    let mask = match &cfg.entropy_mask {
        Some(m) if !m.is_empty() => m,
        _ => return empty,
    };
    if tokens.len() < 2 {
        return empty;
    }
    let gamma = cfg.gamma;
    let positions = tokens.len() - 1;
    let mut green = 0usize;
    let mut total = 0usize;
    // Position i scores token[i+1] using greenlist(prev=token[i]); mask indexes the scored
    // position (the token being generated), i.e. mask[i+1] or mask over generated positions.
    for i in 0..positions {
        // The mask is aligned to the generated tokens (index i+1); guard length.
        let scored_idx = i + 1;
        let high_entropy = mask.get(scored_idx).copied().unwrap_or(false);
        if !high_entropy {
            continue;
        }
        total += 1;
        if is_green(oracle, tokens[i], tokens[i + 1]) {
            green += 1;
        }
    }
    if total == 0 {
        return empty;
    }
    let mean = rate(green, total);
    let z = binomial_z(green, total, gamma);
    ResultRow {
        scheme: Scheme::Sweet,
        tokens: tokens.len(),
        positions,
        usable_positions: total,
        ones: green,
        total,
        mean_g: mean,
        z,
        approx_p_value: normal_upper_tail(z),
        reliable: total >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: Some(kgw_binomial_p(green, total, gamma)),
    }
}

/// "Three Bricks" grounded p-value for a greenlist scheme: the exact upper-tail of a
/// Binomial(total, gamma) at `green` green tokens, via the regularized incomplete beta
/// (matching `special.betainc(green, 1 + total - green, gamma)` in the reference).
pub fn kgw_binomial_p(green: usize, total: usize, gamma: f64) -> f64 {
    if total == 0 || green == 0 {
        return 1.0;
    }
    betainc(green as f64, (1 + total - green) as f64, gamma).clamp(0.0, 1.0)
}

/// Aaronson/Kuditipudi-style exponential (Gumbel) detector.
///
/// Each position `i` has a keyed uniform value `u_i = U(context_i, token_i)` in (0,1). The
/// per-token score is `-ln(1 - u_i)`, which is Exp(1)-distributed (mean 1, var 1) under H0
/// (unwatermarked text, u independent of the chosen token) and inflated when the generator
/// picked tokens with large `u`. The aggregate statistic `S = Σ -ln(1 - u_i)` is Gamma(n,1)
/// under H0; we report the normalized z = (S - n) / sqrt(n) and its upper-tail p-value.
///
/// The uniform values that actually occur are supplied via `u_oracle`, keyed by the
/// `"<space-joined context>|<token>"` signature, matching a reference implementation exactly.
pub fn exp_scores(tokens: &[u64], cfg: &Config) -> ResultRow {
    let cw = cfg.context_width.max(1);
    let empty = ResultRow::empty(Scheme::Exp, tokens.len());
    if tokens.len() <= cw {
        return empty;
    }
    let oracle = match &cfg.u_oracle {
        Some(o) if !o.is_empty() => o,
        _ => return empty,
    };

    let positions = tokens.len() - cw;
    let mut sum_score = 0.0f64;
    let mut total = 0usize;
    // Count each unique (context, token) once to avoid over-counting repeats.
    let mut seen = std::collections::HashSet::with_capacity(positions);

    for i in cw..tokens.len() {
        let context = &tokens[i - cw..i];
        let key = format!("{}|{}", signature(context), tokens[i]);
        if !seen.insert(key.clone()) {
            continue;
        }
        if let Some(&u) = oracle.get(&key) {
            let u = u.clamp(1e-12, 1.0 - 1e-12);
            sum_score += -(1.0 - u).ln();
            total += 1;
        }
    }

    if total == 0 {
        return empty;
    }
    let n = total as f64;
    let mean = sum_score / n;
    // S ~ Gamma(n, 1) under H0; normalize by mean 1, variance 1 per token.
    let z = (sum_score - n) / n.sqrt();
    let p = normal_upper_tail(z);

    ResultRow {
        scheme: Scheme::Exp,
        tokens: tokens.len(),
        positions,
        usable_positions: total,
        ones: 0,
        total,
        mean_g: mean,
        z,
        approx_p_value: p,
        reliable: total >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        // Three Bricks grounded p: S = sum(-ln(1-u)) ~ Gamma(total, 1) under H0.
        grounded_p_value: Some(gammaincc(total as f64, sum_score).clamp(0.0, 1.0)),
    }
}

/// HF-faithful KGW detector. Reproduces HF's `WatermarkDetector._score_ngrams_in_passage`
/// exactly for both `lefthash` and `selfhash` seeding at any `context_width`.
///
/// HF forms n-grams of length `n = context_width + 1 - selfhash`. For each unique n-gram it
/// takes `prefix = ngram` (selfhash) or `ngram[:-1]` (lefthash) and `target = ngram[-1]`,
/// then checks whether `target` is in the greenlist seeded by `prefix`. The greenlist for
/// each prefix that actually occurs is supplied via `oracle` (exported from HF), keyed by
/// the space-joined prefix tokens. With `ignore_repeated_ngrams`, each unique n-gram counts
/// once; the z-score uses `gamma` as the H0 green rate.
pub fn kgw_hf_scores(
    tokens: &[u64],
    cfg: &Config,
    oracle: &std::collections::HashMap<String, Vec<u64>>,
) -> ResultRow {
    let selfhash = cfg.seeding_scheme.eq_ignore_ascii_case("selfhash");
    let n = cfg.context_width + 1 - usize::from(selfhash);

    let empty = ResultRow::empty(Scheme::Kgw, tokens.len());
    if n == 0 || tokens.len() < n {
        return empty;
    }

    let positions = tokens.len() - n + 1;
    // Count each unique n-gram once (HF's ignore_repeated_ngrams=True).
    let mut seen = std::collections::HashSet::with_capacity(positions);
    let mut green = 0usize;
    let mut total = 0usize;

    for i in 0..positions {
        let ngram = &tokens[i..i + n];
        if !seen.insert(ngram.to_vec()) {
            continue;
        }
        total += 1;
        // prefix = full ngram (selfhash) or ngram[:-1] (lefthash); target = last token.
        let prefix: &[u64] = if selfhash { ngram } else { &ngram[..n - 1] };
        let target = ngram[n - 1];
        if let Some(greens) = oracle.get(&signature(prefix)) {
            if greens.binary_search(&target).is_ok() {
                green += 1;
            }
        }
    }

    let mean = rate(green, total);
    let z = binomial_z(green, total, cfg.gamma);
    let p = normal_upper_tail(z);

    ResultRow {
        scheme: Scheme::Kgw,
        tokens: tokens.len(),
        positions,
        usable_positions: total,
        ones: green,
        total,
        mean_g: mean,
        z,
        approx_p_value: p,
        reliable: total >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: Some(kgw_binomial_p(green, total, cfg.gamma)),
    }
}

/// Effective vocab size for KGW: explicit config value, else inferred from tokens.
fn kgw_vocab(tokens: &[u64], cfg: &Config) -> u64 {
    if let Some(v) = cfg.vocab_size {
        return v.max(2);
    }
    tokens
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(2)
        .max(2)
}

/// KGW green-list membership for `token` given the preceding context and a single key.
/// Reproduces the described mechanism: hash(key, context) seeds a pseudo-random split
/// of the vocab; a token is "green" if it falls in the first `gamma` fraction of a
/// keyed permutation. We realize the permutation implicitly by hashing (seed, token)
/// and thresholding, which is a standard hash-based green-list construction.
pub fn kgw_is_green(
    iv: u64,
    context: &[u64],
    token: u64,
    key: u64,
    gamma: f64,
    vocab: u64,
) -> bool {
    let seed = lcg_accumulate(context_hash(iv, context), &[key]);
    // Map (seed, token) -> uniform in [0, vocab); green iff below gamma*vocab.
    let h = lcg_accumulate(seed, &[token]);
    let bucket = h % vocab;
    let threshold = (gamma * vocab as f64).round() as u64;
    bucket < threshold
}

/// KGW detector: fraction of tokens landing on the green list vs. the gamma baseline.
pub fn kgw_scores(tokens: &[u64], cfg: &Config) -> ResultRow {
    let n = cfg.ngram_len;
    assert!(n >= 2);
    assert!(!cfg.keys.is_empty());
    let gamma = cfg.gamma;
    assert!(gamma > 0.0 && gamma < 1.0, "gamma must be in (0,1)");

    if tokens.len() < n {
        return ResultRow::empty(Scheme::Kgw, tokens.len());
    }

    let iv = hash_iv(&cfg.keys);
    let vocab = kgw_vocab(tokens, cfg);
    let positions = tokens.len() - n + 1;

    // Skip repeated (n-1)-token contexts, matching the SynthID-side uniqueness rule so
    // both schemes count comparable "usable positions".
    let mut seen = std::collections::HashSet::with_capacity(positions);
    let mut green = 0usize;
    let mut total = 0usize;
    let depth = cfg.keys.len();

    for i in 0..positions {
        let context = &tokens[i..i + n - 1];
        if !seen.insert(context_hash(iv, context)) {
            continue;
        }
        let token = tokens[i + n - 1];
        for &key in &cfg.keys {
            if kgw_is_green(iv, context, token, key, gamma, vocab) {
                green += 1;
            }
            total += 1;
        }
    }

    // Under H0 (no watermark), P(green) = gamma. z uses the binomial mean/variance.
    let mean = rate(green, total);
    let z = binomial_z(green, total, gamma);
    let p = normal_upper_tail(z);
    let usable = total.checked_div(depth).unwrap_or(0);

    ResultRow {
        scheme: Scheme::Kgw,
        tokens: tokens.len(),
        positions,
        usable_positions: usable,
        ones: green,
        total,
        mean_g: mean,
        z,
        approx_p_value: p,
        reliable: usable >= RELIABLE_TOKEN_FLOOR,
        weighted_mean_g: None,
        grounded_p_value: Some(kgw_binomial_p(green, total, gamma)),
    }
}

pub fn normal_upper_tail(z: f64) -> f64 {
    // 0.5 * erfc(z / sqrt(2)); Abramowitz-Stegun approximation.
    let x = z.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    let erf = 1.0 - poly * (-x * x).exp();
    if z >= 0.0 {
        0.5 * (1.0 - erf)
    } else {
        0.5 * (1.0 + erf)
    }
}

// ---------------------------------------------------------------------------
// "Three Bricks" (Fernandez et al. 2023) grounded p-values. Instead of a Gaussian
// z-approximation, these use the EXACT null distributions:
//   * KGW green count ~ Binomial(ntoks, gamma)   => p = I_gamma(k, ntoks-k+1)  (betainc)
//   * exp sum of -ln(1-u) ~ Gamma(ntoks, 1)       => p = Q(ntoks, score)       (gammaincc)
// These are well-calibrated even on short texts where the normal approximation is poor.
// ---------------------------------------------------------------------------

/// ln(Gamma(x)) via the Lanczos approximation (g=7, n=9); accurate to ~1e-13 for x>0.
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized upper incomplete gamma Q(a, x) = Gamma(a,x)/Gamma(a). Numerical Recipes:
/// series expansion for x < a+1, continued fraction otherwise. Returns P(sum >= x) for a
/// Gamma(a,1) variate, i.e. the exp-scheme grounded p-value `gammaincc(ntoks, score)`.
pub fn gammaincc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // Lower series P(a,x); Q = 1 - P.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..500 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - ln_gamma(a)).exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        // Continued fraction for Q(a,x) (Lentz's method).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = (-x + a * x.ln() - ln_gamma(a)).exp() * h;
        q.clamp(0.0, 1.0)
    }
}

/// Regularized incomplete beta I_x(a, b). Numerical Recipes continued fraction (betacf).
/// Used for the KGW grounded p-value `betainc(k, ntoks-k+1, gamma)` = binomial upper tail.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Deterministic xorshift RNG so tests are reproducible without external crates.
pub struct XorShift64(pub u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        XorShift64(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform integer in [0, bound).
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Generate a *watermarked* token stream by mimicking SynthID's tournament sampling:
/// at each step draw `num_candidates` tokens uniformly from the vocab, then emit the
/// candidate whose aggregated g-value over the keys is highest (ties broken by the
/// RNG order). This biases emitted tokens toward g==1, exactly what the detector
/// measures. With a matching config, `ngram_g_values` should report a strong signal.
pub fn generate_watermarked(
    cfg: &Config,
    vocab_size: u64,
    len: usize,
    num_candidates: usize,
    rng: &mut XorShift64,
) -> Vec<u64> {
    assert!(vocab_size >= 2);
    assert!(num_candidates >= 1);
    let n = cfg.ngram_len;
    let iv = hash_iv(&cfg.keys);

    let mut out: Vec<u64> = Vec::with_capacity(len);
    // Seed the first n-1 tokens with unbiased draws so a context exists.
    for _ in 0..(n - 1).min(len) {
        out.push(rng.below(vocab_size));
    }

    while out.len() < len {
        let ctx_start = out.len() - (n - 1);
        let context = out[ctx_start..].to_vec();

        let mut best_tok = 0u64;
        let mut best_score = -1i64;
        for _ in 0..num_candidates {
            let cand = rng.below(vocab_size);
            let score = candidate_g_sum(iv, &context, cand, &cfg.keys) as i64;
            if score > best_score {
                best_score = score;
                best_tok = cand;
            }
        }
        out.push(best_tok);
    }
    out
}

/// Generate an unwatermarked control stream: uniform random tokens.
pub fn generate_control(vocab_size: u64, len: usize, rng: &mut XorShift64) -> Vec<u64> {
    (0..len).map(|_| rng.below(vocab_size)).collect()
}

/// Generate a *KGW-watermarked* token stream. At each step we draw `num_candidates`
/// uniform candidates and emit the one with the most green-list "votes" across keys,
/// simulating the logit boost toward the green list. With a matching config,
/// `kgw_scores` should report a strong signal (green fraction well above gamma).
pub fn generate_kgw(
    cfg: &Config,
    vocab_size: u64,
    len: usize,
    num_candidates: usize,
    rng: &mut XorShift64,
) -> Vec<u64> {
    assert!(vocab_size >= 2);
    assert!(num_candidates >= 1);
    let n = cfg.ngram_len;
    let gamma = cfg.gamma;
    let iv = hash_iv(&cfg.keys);

    let mut out: Vec<u64> = Vec::with_capacity(len);
    for _ in 0..(n - 1).min(len) {
        out.push(rng.below(vocab_size));
    }

    while out.len() < len {
        let ctx_start = out.len() - (n - 1);
        let context = out[ctx_start..].to_vec();

        let mut best_tok = 0u64;
        let mut best_votes = -1i64;
        for _ in 0..num_candidates {
            let cand = rng.below(vocab_size);
            let votes = cfg
                .keys
                .iter()
                .filter(|&&key| kgw_is_green(iv, &context, cand, key, gamma, vocab_size))
                .count() as i64;
            if votes > best_votes {
                best_votes = votes;
                best_tok = cand;
            }
        }
        out.push(best_tok);
    }
    out
}

/// Generate a watermarked stream under the requested scheme.
pub fn generate(
    cfg: &Config,
    scheme: Scheme,
    vocab_size: u64,
    len: usize,
    num_candidates: usize,
    rng: &mut XorShift64,
) -> Vec<u64> {
    match scheme {
        Scheme::SynthId => generate_watermarked(cfg, vocab_size, len, num_candidates, rng),
        Scheme::Kgw => generate_kgw(cfg, vocab_size, len, num_candidates, rng),
        // The exponential scheme's authoritative generation is driven from the reference
        // implementation in the Python validator (it needs the keyed PRNG); the built-in
        // corpus generator does not synthesize exp-watermarked streams.
        Scheme::Exp => generate_control(vocab_size, len, rng),
        // Unigram/SWEET authoritative generation is likewise driven from the reference
        // (Unigram needs the global split; SWEET needs model entropy). The built-in
        // generator emits control streams for these.
        Scheme::Unigram | Scheme::Sweet => generate_control(vocab_size, len, rng),
        // EXP-Edit generation needs the keyed uniform matrix + inverse-CDF sampling; it is
        // driven from the reference in the Python validator, not the built-in generator.
        Scheme::ExpEdit => generate_control(vocab_size, len, rng),
    }
}
