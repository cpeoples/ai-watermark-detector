//! C2PA content-provenance checking, implemented natively in Rust.
//!
//! Unlike the secret-keyed statistical TEXT watermarks in this crate, C2PA manifests are
//! PUBLIC-key signed provenance records embedded in generated files. They can be verified
//! today with no private key. This module wraps the official `c2patool` (itself a Rust
//! binary from the Content Authenticity Initiative), parses its JSON, and surfaces:
//!   * whether a manifest is present + signature validation status,
//!   * the claim generator (tool/vendor),
//!   * AI-generation markers: the IPTC `digitalSourceType` and any SynthID assertion
//!     (Google's marker on Gemini/Imagen images & Veo video).
//!
//! C2PA is container-agnostic: images (JPEG/PNG/WebP/AVIF/…), video (MP4/MOV), audio
//! (MP3/WAV/M4A), and PDF. Text/code containers (.docx/.txt/source) carry no manifest in
//! practice — this module reports that honestly.

use serde_json::Value;
use std::path::Path;
use std::process::Command;

/// IPTC digital-source-type suffixes that indicate AI involvement, mapped to a plain
/// description. Matching is on the exact code after the final '/'.
pub fn ai_source_meaning(code: &str) -> Option<&'static str> {
    match code.to_ascii_lowercase().as_str() {
        "trainedalgorithmicmedia" => Some("fully AI-generated (trained algorithmic media)"),
        "compositewithtrainedalgorithmicmedia" => Some("AI-assisted / composite with AI media"),
        "algorithmicmedia" => Some("algorithmically generated media"),
        _ => None,
    }
}

/// Real, publicly-hosted C2PA-signed sample files from the official Content Authenticity
/// project test fixtures, spanning image / video / audio formats.
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
    pub synthid_assertion: bool,
    pub assertions: Vec<String>,
    pub validation_codes: Vec<String>,
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

fn find_ai_markers(active: &Value) -> (Option<String>, bool) {
    let mut ai_source = None;
    let mut synthid = false;
    if let Some(assertions) = active.get("assertions").and_then(|a| a.as_array()) {
        for a in assertions {
            let label = a
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let blob = a.to_string().to_ascii_lowercase();
            if label.contains("synthid") || blob.contains("synthid") {
                synthid = true;
            }
            let mut dsts = Vec::new();
            if let Some(data) = a.get("data") {
                collect_dst(data, &mut dsts);
            }
            for dst in dsts {
                let code = dst.rsplit('/').next().unwrap_or(&dst);
                if let Some(meaning) = ai_source_meaning(code) {
                    ai_source = Some(meaning.to_string());
                }
            }
        }
    }
    (ai_source, synthid)
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
        synthid_assertion: false,
        assertions: Vec::new(),
        validation_codes: Vec::new(),
    };

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

    let (ai_source, synthid) = find_ai_markers(&active);
    report.ai_source_type = ai_source;
    report.synthid_assertion = synthid;

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
