//! C2PA content-provenance checking, implemented natively in Rust.
//!
//! Unlike the secret-keyed statistical TEXT watermarks in this crate, C2PA manifests are
//! PUBLIC-key signed provenance records embedded in generated files. They can be verified
//! today with no private key. This module wraps the official `c2patool` (itself a Rust
//! binary from the Content Authenticity Initiative), parses its JSON, and surfaces:
//!   * whether a manifest is present + signature validation status,
//!   * the claim generator (tool/vendor),
//!   * AI-generation markers: the IPTC `digitalSourceType`, any SynthID assertion, the
//!     forensic `soft-binding` watermark, and a SynthID implied by a SynthID-pairing issuer.
//!     Markers are collected across the whole manifest chain, since AI provenance often lives
//!     on a parent/ingredient rather than the active manifest; a mismatch (clean active over an
//!     AI ingredient) is reported as a provenance conflict.
//!
//! C2PA is container-agnostic: images (JPEG/PNG/WebP/AVIF/…), video (MP4/MOV), audio
//! (MP3/WAV/M4A), and PDF. `c2patool` handles those binary containers; C2PA 2.4 text
//! embeddings (in `.txt`/source/HTML) are detected separately in `signals.rs`.

use crate::signals::{self, Signal};
use serde_json::Value;
use std::path::Path;
use std::process::Command;

/// IPTC digital-source-type suffixes that are *normatively* Generative AI per C2PA
/// §18.15.4.5, mapped to a plain description. Matching is on the exact code after the final
/// '/'. Only these three flip the AI verdict.
pub fn ai_source_meaning(code: &str) -> Option<&'static str> {
    match code.to_ascii_lowercase().as_str() {
        "trainedalgorithmicmedia" => Some("fully AI-generated (trained algorithmic media)"),
        "compositewithtrainedalgorithmicmedia" => Some("AI-assisted / composite with AI media"),
        "compositesynthetic" => Some("composite with at least one AI-generated element"),
        _ => None,
    }
}

/// IPTC digital-source-type suffixes that are algorithmic but NOT trained/generative AI (e.g.
/// fractal or formula-based art, or a minor algorithmic correction). Recognized and surfaced,
/// but deliberately does not flip the AI verdict.
fn algorithmic_source_meaning(code: &str) -> Option<&'static str> {
    match code.to_ascii_lowercase().as_str() {
        "algorithmicmedia" => Some("algorithmic media (formula-based, not trained AI)"),
        "algorithmicallyenhanced" => {
            Some("algorithmically enhanced (minor correction, not trained AI)")
        }
        _ => None,
    }
}

/// Turn a C2PA soft-binding `alg` identifier (from the C2PA Soft Binding Algorithm List, which
/// uses reverse-DNS names like `com.adobe.trustmark.Q`) into a readable vendor label. Known
/// vendors are named explicitly; anything else keeps the raw identifier and notes that it is a
/// registered algorithm, so new registry entries still read sensibly without hardcoding them.
fn soft_binding_label(alg: &str) -> String {
    let lower = alg.to_ascii_lowercase();
    let vendor = if lower.contains("trustmark") {
        "Adobe TrustMark"
    } else if lower.contains("digimarc") {
        "Digimarc"
    } else if lower.contains("imatag") {
        "IMATAG"
    } else if lower.contains("audioseal")
        || lower.contains("videoseal")
        || lower.contains("pixelseal")
    {
        "Meta Seal"
    } else if lower.contains("nexguard") {
        "NAGRA NexGuard"
    } else {
        return format!("{alg} (C2PA-registered soft binding)");
    };
    format!("{vendor} ({alg})")
}

/// Inspect an actions assertion for a `c2pa.watermarked{,.bound,.unbound}` action (C2PA 2.2+),
/// returning a readable label for the declared watermark. `bound` watermarks back a soft-binding
/// lookup; `unbound` ones are a vendor mark with no recovery, which we note explicitly.
fn watermarked_action_label(assertion: &Value) -> Option<String> {
    let actions = assertion.get("data")?.get("actions")?.as_array()?;
    for act in actions {
        let name = act.get("action").and_then(|s| s.as_str()).unwrap_or("");
        match name {
            "c2pa.watermarked.unbound" => {
                return Some("declared watermark (unbound, no manifest lookup)".to_string())
            }
            "c2pa.watermarked" | "c2pa.watermarked.bound" => {
                return Some("declared soft-binding watermark".to_string())
            }
            _ => {}
        }
    }
    None
}

/// Real, publicly-hosted C2PA-signed sample files from the official Content Authenticity
/// project test fixtures, spanning image / video / audio / PDF formats.
pub const SAMPLE_URLS: &[(&str, &str)] = &[
    ("cai_image_C.jpg", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/C.jpg"),
    ("cai_image_cloud.jpg", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/cloud.jpg"),
    ("cai_image.png", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/libpng-test.png"),
    ("cai_image.gif", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/sample1.gif"),
    ("cai_video.mp4", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/video1.mp4"),
    ("cai_audio.wav", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/sample1.wav"),
    ("cai_audio.mp3", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/sample1.mp3"),
    ("cai_audio.m4a", "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/sample1.m4a"),
    // Official C2PA public test corpus (Adobe-signed provenance chains).
    ("adobe_signed_C.jpg", "https://raw.githubusercontent.com/c2pa-org/public-testfiles/main/legacy/1.4/image/jpeg/adobe-20220124-C.jpg"),
    ("adobe_signed_CA.jpg", "https://raw.githubusercontent.com/c2pa-org/public-testfiles/main/legacy/1.4/image/jpeg/adobe-20220124-CA.jpg"),
    ("adobe_signed.pdf", "https://raw.githubusercontent.com/c2pa-org/public-testfiles/main/legacy/1.4/pdf/adobe-20240110-single_manifest_store.pdf"),
];

/// Result of inspecting a single file.
#[derive(Debug, serde::Serialize)]
pub struct C2paReport {
    pub file: String,
    pub ext: String,
    pub has_manifest: bool,
    pub status: String,
    /// Forensic verdict: "trusted", "valid-untrusted", "tampered", "no-manifest", or "error".
    pub verdict: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claim_generator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_source_type: Option<String>,
    /// An algorithmic-but-not-generative-AI digitalSourceType (e.g. fractal/formula art). Shown
    /// for transparency; does NOT set the AI verdict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithmic_source: Option<String>,
    pub synthid_assertion: bool,
    /// Forensic soft-binding watermark named in the manifest (e.g. `com.adobe.trustmark.Q`,
    /// Digimarc, Imatag). Present only when the signed manifest declares one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft_binding: Option<String>,
    /// A validly signed manifest from a vendor that always pairs C2PA with a SynthID pixel
    /// watermark (Google, OpenAI), so the watermark is implied even without a SynthID assertion.
    pub implied_synthid: bool,
    /// The active (current signer's) manifest declares no AI generation, yet an ingredient/
    /// parent in the signed chain does - a provenance conflict worth surfacing (the "Integrity
    /// Clash": a valid signature can assert a human-only edit over an AI-generated original).
    pub provenance_conflict: bool,
    pub assertions: Vec<String>,
    pub validation_codes: Vec<String>,
    /// Lower-confidence AI-generation hints from unsigned metadata / filename (never override
    /// the C2PA `verdict`; empty when none are found or the bytes can't be read).
    pub signals: Vec<Signal>,
}

impl C2paReport {
    /// True when the signed manifest establishes AI generation, by any of its markers.
    pub fn is_ai_generated(&self) -> bool {
        self.ai_source_type.is_some() || self.synthid_assertion || self.implied_synthid
    }
}

/// Validation codes that indicate the content or manifest was altered after signing.
fn is_tamper_code(code: &str) -> bool {
    let c = code.to_ascii_lowercase();
    c.contains("mismatch")
        || c.contains(".invalid")
        || c.contains("hash")
        || (c.contains("signature") && !c.contains("untrusted"))
}

/// Classify the manifest into a forensic verdict from its validation codes.
fn classify(has_manifest: bool, codes: &[String]) -> String {
    if !has_manifest {
        return "no-manifest".to_string();
    }
    if codes.iter().any(|c| is_tamper_code(c)) {
        return "tampered".to_string();
    }
    // Only benign/trust-related codes remain.
    let only_untrusted = codes
        .iter()
        .all(|c| c.to_ascii_lowercase().contains("untrusted"));
    if codes.is_empty() {
        "trusted".to_string()
    } else if only_untrusted {
        "valid-untrusted".to_string()
    } else {
        "valid-with-warnings".to_string()
    }
}

fn ext_of(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_else(|| "(none)".to_string())
}

/// Recursively collect every `digitalSourceType` string found in a JSON value.
fn collect_dst(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                if k == "digitalSourceType" {
                    if let Some(s) = val.as_str() {
                        out.push(s.to_string());
                    }
                } else {
                    collect_dst(val, out);
                }
            }
        }
        Value::Array(arr) => arr.iter().for_each(|x| collect_dst(x, out)),
        _ => {}
    }
}

/// AI-generation markers extracted from a signed manifest's assertions.
#[derive(Default)]
struct AiMarkers {
    source_type: Option<String>,
    synthid_assertion: bool,
    soft_binding: Option<String>,
    /// An algorithmic-but-not-AI digitalSourceType (recognized for transparency; does not
    /// contribute to `is_ai`).
    algorithmic_source: Option<String>,
}

impl AiMarkers {
    /// Fold another manifest's markers in, keeping any value already found (the active
    /// manifest is scanned first, so its values win).
    fn merge(&mut self, other: AiMarkers) {
        self.synthid_assertion |= other.synthid_assertion;
        self.source_type = self.source_type.take().or(other.source_type);
        self.soft_binding = self.soft_binding.take().or(other.soft_binding);
        self.algorithmic_source = self.algorithmic_source.take().or(other.algorithmic_source);
    }

    /// Whether this manifest declares AI generation (source type or SynthID assertion).
    fn is_ai(&self) -> bool {
        self.source_type.is_some() || self.synthid_assertion
    }
}

fn find_ai_markers(active: &Value) -> AiMarkers {
    let mut markers = AiMarkers::default();
    let Some(assertions) = active.get("assertions").and_then(|a| a.as_array()) else {
        return markers;
    };
    for a in assertions {
        let label = a
            .get("label")
            .and_then(|l| l.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let blob = a.to_string().to_ascii_lowercase();
        if label.contains("synthid") || blob.contains("synthid") {
            markers.synthid_assertion = true;
        }
        if label.contains("soft-binding") {
            markers.soft_binding = a
                .get("data")
                .and_then(|d| d.get("alg"))
                .and_then(|s| s.as_str())
                .map(soft_binding_label)
                .or_else(|| Some("soft-binding watermark".to_string()));
        }
        // A `c2pa.watermarked.bound/unbound` action (inside an actions assertion) declares an
        // inserted watermark even when no separate soft-binding assertion names the algorithm -
        // notably the `unbound` case, which carries a vendor mark with no recovery lookup.
        if label.contains("actions") {
            if let Some(w) = watermarked_action_label(a) {
                markers.soft_binding.get_or_insert(w);
            }
        }
        let mut dsts = Vec::new();
        if let Some(data) = a.get("data") {
            collect_dst(data, &mut dsts);
        }
        for dst in dsts {
            let code = dst.rsplit('/').next().unwrap_or(&dst);
            if let Some(meaning) = ai_source_meaning(code) {
                markers.source_type = Some(meaning.to_string());
            } else if let Some(meaning) = algorithmic_source_meaning(code) {
                markers.algorithmic_source = Some(meaning.to_string());
            }
        }
    }
    markers
}

/// Collect AI markers across the whole signed manifest store, not just the active manifest.
/// AI provenance often lives on a parent/ingredient manifest (e.g. an editor re-signs an
/// AI-generated original), so the active manifest alone can miss it. The active manifest is
/// scanned first so its values take precedence.
fn find_ai_markers_in_store(store: &Value, active: &Value) -> AiMarkers {
    let mut markers = find_ai_markers(active);
    if let Some(manifests) = store.get("manifests").and_then(|m| m.as_object()) {
        for manifest in manifests.values() {
            if manifest == active {
                continue;
            }
            markers.merge(find_ai_markers(manifest));
        }
    }
    markers
}

/// Vendors whose validly signed C2PA manifests always carry a SynthID pixel watermark, so a
/// good signature from one implies the invisible watermark even absent a SynthID assertion.
/// Note: only pixel-SynthID vendors belong here. Anthropic/Claude and Adobe/Microsoft sign
/// C2PA as provenance only (Claude uses a SynthID-*Text* token watermark, not a pixel one), so
/// they must not be added.
fn issuer_implies_synthid(claim_generator: Option<&str>) -> bool {
    let Some(gen) = claim_generator else {
        return false;
    };
    let gen = gen.to_ascii_lowercase();
    [
        "google", "imagen", "gemini", "deepmind", "openai", "dall", "gpt",
    ]
    .iter()
    .any(|v| gen.contains(v))
}

/// Run `c2patool <path>` (optionally with a trust anchor list) and summarize the result.
pub fn check_file(path: &str, trust_anchors: Option<&str>) -> C2paReport {
    let ext = ext_of(path);
    let mut report = C2paReport {
        file: path.to_string(),
        ext,
        has_manifest: false,
        status: String::new(),
        verdict: "error".to_string(),
        claim_generator: None,
        title: None,
        ai_source_type: None,
        algorithmic_source: None,
        synthid_assertion: false,
        soft_binding: None,
        implied_synthid: false,
        provenance_conflict: false,
        assertions: Vec::new(),
        validation_codes: Vec::new(),
        signals: Vec::new(),
    };

    // Metadata/filename signals are independent of c2patool and worth surfacing even when the
    // C2PA path errors or finds nothing, so gather them up front from the raw bytes.
    if let Ok(bytes) = std::fs::read(path) {
        report.signals = signals::scan(path, &report.ext, &bytes);
    }

    let mut cmd = Command::new("c2patool");
    cmd.arg(path);
    // Trust options are a subcommand that must follow the path.
    if let Some(anchors) = trust_anchors {
        cmd.arg("trust").arg("--trust_anchors").arg(anchors);
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            report.status = format!("error: could not run c2patool ({e})");
            return report;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        let msg = if !stderr.trim().is_empty() {
            stderr.trim()
        } else {
            stdout.trim()
        };
        let low = msg.to_ascii_lowercase();
        report.status = if ["no claim", "no manifest", "jumbf", "not found"]
            .iter()
            .any(|k| low.contains(k))
        {
            "no C2PA manifest".to_string()
        } else if low.contains("unsupported")
            || low.contains("invalid file header")
            || low.contains("could not be parsed")
        {
            "no C2PA manifest (format not C2PA-capable or unsigned)".to_string()
        } else {
            format!("error: {}", msg.lines().next().unwrap_or("unknown"))
        };
        report.verdict = if report.status.starts_with("no C2PA") {
            "no-manifest".to_string()
        } else {
            "error".to_string()
        };
        return report;
    }

    let store: Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(_) => {
            report.status = "unparseable c2patool output".to_string();
            return report;
        }
    };

    let active_id = store.get("active_manifest").and_then(|a| a.as_str());
    let manifests = store.get("manifests");
    report.has_manifest = manifests
        .map(|m| m.as_object().map(|o| !o.is_empty()).unwrap_or(false))
        .unwrap_or(false);

    let active = active_id
        .and_then(|id| manifests.and_then(|m| m.get(id)))
        .cloned()
        .unwrap_or(Value::Null);

    if let Some(gen) = active.get("claim_generator").and_then(|g| g.as_str()) {
        report.claim_generator = Some(gen.to_string());
    } else if let Some(info) = active.get("claim_generator_info") {
        report.claim_generator = Some(info.to_string());
    }
    report.title = active
        .get("title")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string());

    if let Some(assertions) = active.get("assertions").and_then(|a| a.as_array()) {
        report.assertions = assertions
            .iter()
            .filter_map(|a| {
                a.get("label")
                    .and_then(|l| l.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
    }

    let active_markers = find_ai_markers(&active);
    let active_is_ai = active_markers.is_ai();
    let markers = find_ai_markers_in_store(&store, &active);
    report.provenance_conflict = markers.is_ai() && !active_is_ai;
    report.ai_source_type = markers.source_type;
    report.algorithmic_source = markers.algorithmic_source;
    report.synthid_assertion = markers.synthid_assertion;
    report.soft_binding = markers.soft_binding;

    let vs = store
        .get("validation_status")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    report.validation_codes = vs
        .iter()
        .filter_map(|v| {
            v.get("code")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    report.verdict = classify(report.has_manifest, &report.validation_codes);
    // A SynthID pixel watermark is implied only when the manifest is a genuinely valid
    // signature (tampered/error manifests prove nothing) from a SynthID-pairing issuer, and no
    // explicit SynthID assertion already established it.
    let validly_signed = matches!(
        report.verdict.as_str(),
        "trusted" | "valid-untrusted" | "valid-with-warnings"
    );
    report.implied_synthid = validly_signed
        && !report.synthid_assertion
        && issuer_implies_synthid(report.claim_generator.as_deref());
    report.status = match report.verdict.as_str() {
        "trusted" => {
            "manifest present; signature TRUSTED (chains to a provided anchor)".to_string()
        }
        "valid-untrusted" => {
            "manifest present; signature cryptographically valid but signer NOT in trust list"
                .to_string()
        }
        "tampered" => format!(
            "TAMPERED: content/manifest altered after signing ({})",
            report.validation_codes.join(", ")
        ),
        _ => format!(
            "manifest present; validation: {}",
            report.validation_codes.join(", ")
        ),
    };

    report
}

/// Download `url` into `dest` bytes-for-bytes, returning the response body.
fn download(url: &str) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Download the official signed sample fixtures into `dest`, returning saved paths.
pub fn fetch_samples(dest: &str) -> std::io::Result<Vec<String>> {
    std::fs::create_dir_all(dest)?;
    let mut saved = Vec::new();
    for (name, url) in SAMPLE_URLS {
        let out = format!("{dest}/{name}");
        match download(url) {
            Ok(buf) => {
                std::fs::write(&out, &buf)?;
                eprintln!("fetched {name} ({} bytes)", buf.len());
                saved.push(out);
            }
            Err(e) => eprintln!("warning: could not fetch {name}: {e}"),
        }
    }
    Ok(saved)
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Download a URL to a temp file (keeping the URL's filename extension so `c2patool` sees the
/// right container) and return the local path.
fn fetch_url(url: &str) -> Result<String, String> {
    let buf = download(url)?;
    let name = url
        .rsplit('/')
        .next()
        .filter(|n| !n.is_empty())
        .unwrap_or("download");
    let dir = std::env::temp_dir().join("ai-watermark-detector-urls");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let out = dir.join(name);
    std::fs::write(&out, &buf).map_err(|e| e.to_string())?;
    out.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "non-UTF8 temp path".to_string())
}

/// Recursively gather files under the given paths. Directories are walked; `http(s)://`
/// arguments are downloaded to a temp file and checked like any local file.
pub fn gather_files(paths: &[String]) -> Vec<String> {
    let mut files = Vec::new();
    for p in paths {
        if is_url(p) {
            match fetch_url(p) {
                Ok(local) => files.push(local),
                Err(e) => eprintln!("warning: could not fetch {p}: {e}"),
            }
            continue;
        }
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, &mut files);
        } else {
            files.push(p.clone());
        }
    }
    files
}

fn walk(dir: &Path, out: &mut Vec<String>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut names: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        names.sort();
        for path in names {
            let is_hidden = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false);
            if is_hidden {
                continue;
            }
            if path.is_dir() {
                walk(&path, out);
            } else if let Some(s) = path.to_str() {
                out.push(s.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ai_source_meaning_covers_genai_types() {
        assert!(ai_source_meaning("trainedAlgorithmicMedia").is_some());
        assert!(ai_source_meaning("compositeWithTrainedAlgorithmicMedia").is_some());
        assert!(ai_source_meaning("compositeSynthetic").is_some());
        // Pure algorithmic art and camera capture are not generative-AI markers.
        assert!(ai_source_meaning("algorithmicMedia").is_none());
        assert!(ai_source_meaning("digitalCapture").is_none());
        assert!(ai_source_meaning("something-else").is_none());
    }

    #[test]
    fn algorithmic_source_is_recognized_but_not_ai() {
        // Recognized for transparency, but must not appear in the AI mapping.
        assert!(algorithmic_source_meaning("algorithmicMedia").is_some());
        assert!(algorithmic_source_meaning("algorithmicallyEnhanced").is_some());
        assert!(ai_source_meaning("algorithmicMedia").is_none());
    }

    #[test]
    fn soft_binding_alg_is_extracted() {
        let manifest = json!({
            "assertions": [
                {"label": "c2pa.soft-binding", "data": {"alg": "com.adobe.trustmark.Q"}}
            ]
        });
        let markers = find_ai_markers(&manifest);
        assert_eq!(
            markers.soft_binding.as_deref(),
            Some("Adobe TrustMark (com.adobe.trustmark.Q)")
        );
    }

    #[test]
    fn soft_binding_label_names_known_and_unknown_vendors() {
        assert!(soft_binding_label("com.digimarc.validate.1").starts_with("Digimarc"));
        assert!(soft_binding_label("com.aiwatermark.audioseal.1").starts_with("Meta Seal"));
        // Unregistered-but-well-formed ids fall back to naming themselves, not a crash.
        assert!(soft_binding_label("io.example.newmark.1").contains("C2PA-registered"));
    }

    #[test]
    fn watermarked_action_is_detected_from_actions_assertion() {
        let manifest = json!({
            "assertions": [
                {"label": "c2pa.actions.v2", "data": {"actions": [
                    {"action": "c2pa.watermarked.unbound"}
                ]}}
            ]
        });
        let m = find_ai_markers(&manifest);
        assert!(m.soft_binding.as_deref().unwrap().contains("unbound"));
        // An unbound watermark is not, by itself, an AI-generation claim.
        assert!(!m.is_ai());
    }

    #[test]
    fn synthid_assertion_is_detected() {
        let manifest = json!({ "assertions": [{"label": "com.google.synthid"}] });
        assert!(find_ai_markers(&manifest).synthid_assertion);
    }

    #[test]
    fn ai_marker_on_ingredient_manifest_is_found() {
        // Active manifest is a plain edit; the AI marker lives on the parent/ingredient.
        let store = json!({
            "active_manifest": "edit",
            "manifests": {
                "edit": {"assertions": [{"label": "c2pa.actions"}]},
                "original": {"assertions": [{
                    "label": "c2pa.actions",
                    "data": {"digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}
                }]}
            }
        });
        let active = store["manifests"]["edit"].clone();
        let markers = find_ai_markers_in_store(&store, &active);
        assert!(
            markers.source_type.is_some(),
            "AI marker on an ingredient manifest must be found"
        );
    }

    #[test]
    fn provenance_conflict_when_active_clean_but_ingredient_ai() {
        let active_ai = json!({"assertions": [{
            "label": "c2pa.actions",
            "data": {"digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}
        }]});
        let active_clean = json!({"assertions": [{"label": "c2pa.actions"}]});
        let store_markers = |active: &Value, ingredient_ai: bool| {
            let store = json!({
                "active_manifest": "a",
                "manifests": {
                    "a": active.clone(),
                    "b": if ingredient_ai { active_ai.clone() } else { active_clean.clone() }
                }
            });
            let m = find_ai_markers_in_store(&store, active);
            m.is_ai() && !find_ai_markers(active).is_ai()
        };
        // Clean active + AI ingredient => conflict; AI active => no conflict; all clean => none.
        assert!(store_markers(&active_clean, true));
        assert!(!store_markers(&active_ai, true));
        assert!(!store_markers(&active_clean, false));
    }

    #[test]
    fn issuer_synthid_inference() {
        assert!(issuer_implies_synthid(Some("Made by Google Gemini")));
        assert!(issuer_implies_synthid(Some("OpenAI DALL-E 3")));
        assert!(!issuer_implies_synthid(Some("Adobe Firefly")));
        // Claude signs C2PA as provenance only (SynthID-Text, not a pixel watermark).
        assert!(!issuer_implies_synthid(Some("Claude by Anthropic")));
        assert!(!issuer_implies_synthid(None));
    }

    #[test]
    fn is_ai_generated_covers_every_marker() {
        let mut r = C2paReport {
            file: String::new(),
            ext: String::new(),
            has_manifest: true,
            status: String::new(),
            verdict: "trusted".to_string(),
            claim_generator: None,
            title: None,
            ai_source_type: None,
            algorithmic_source: None,
            synthid_assertion: false,
            soft_binding: None,
            implied_synthid: false,
            provenance_conflict: false,
            assertions: Vec::new(),
            validation_codes: Vec::new(),
            signals: Vec::new(),
        };
        assert!(!r.is_ai_generated());
        r.implied_synthid = true;
        assert!(r.is_ai_generated());
    }
}
