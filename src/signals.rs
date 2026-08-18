//! Lower-confidence AI-generation signals from file metadata and filenames.
//!
//! These complement the cryptographic C2PA check in `c2pa.rs`: C2PA is signed provenance
//! (HIGH confidence, tamper-evident), whereas the signals here come from unsigned metadata
//! that can be edited or stripped - so they are deliberately reported at MEDIUM/LOW
//! confidence and never override a C2PA forensic verdict. Everything is parsed natively from
//! the file bytes (no external tools, no ML): PNG text chunks, embedded XMP/EXIF/IPTC
//! strings, MP4/MKV/WebM/AVI/FLV video and WAV/FLAC/OGG/AAC audio container tags, the TC260
//! `AIGC` label, ID3 tags, embedded C2PA-in-text manifests (C2PA 2.4 A.7/A.8/A.9), SPDX-AI
//! source-file disclosure tags, and filename conventions.
//!
//! Deliberately out of scope (to stay a deterministic, dependency-light tool): visible-logo
//! template matching and neural invisible-watermark decoders such as Adobe TrustMark, which
//! need a CNN model and inference. Those belong to the ML-based removers, not here.

use serde::Serialize;

/// How much trust to place in an unsigned signal. C2PA (the signed, tamper-evident tier) is
/// reported separately on `C2paReport`; these signals are only ever `Medium` (structured
/// metadata) or `Low` (filename/loose heuristic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    Low,
    Medium,
}

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::Low => "low",
            Confidence::Medium => "medium",
        }
    }
}

/// A single AI-generation hint found outside the C2PA manifest.
#[derive(Debug, Clone, Serialize)]
pub struct Signal {
    pub confidence: Confidence,
    /// Where the hint came from, e.g. `png-text`, `xmp`, `iptc`, `container`, `filename`.
    pub source: String,
    /// Human-readable description of what matched.
    pub detail: String,
    /// The AI tool inferred from the hint, when one is recognizable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
}

impl Signal {
    fn new(confidence: Confidence, source: &str, detail: String, tool: Option<String>) -> Signal {
        Signal {
            confidence,
            source: source.to_string(),
            detail,
            tool,
        }
    }
}

/// Known AI tools keyed by a substring that appears in metadata/claim-generator strings,
/// mapped to the canonical tool name. Matching is case-insensitive on the lowercased haystack.
const TOOL_MARKERS: &[(&str, &str)] = &[
    ("dall-e", "DALL-E"),
    ("dalle", "DALL-E"),
    ("midjourney", "Midjourney"),
    ("stable diffusion", "Stable Diffusion"),
    ("stable-diffusion", "Stable Diffusion"),
    ("stablediffusion", "Stable Diffusion"),
    ("firefly", "Adobe Firefly"),
    ("imagen", "Imagen"),
    ("gemini", "Gemini"),
    ("veo", "Google Veo"),
    ("flux", "Flux"),
    ("ideogram", "Ideogram"),
    ("leonardo", "Leonardo.ai"),
    ("novelai", "NovelAI"),
    ("grok", "Grok"),
    ("comfyui", "ComfyUI"),
    ("automatic1111", "Automatic1111"),
    ("invokeai", "InvokeAI"),
    ("fooocus", "Fooocus"),
    ("dreamstudio", "DreamStudio"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-image", "GPT Image"),
    ("chatgpt", "ChatGPT"),
    ("openai", "OpenAI"),
    ("claude", "Claude"),
    ("anthropic", "Claude"),
    ("sora", "Sora"),
    ("runway", "Runway"),
    ("pika", "Pika"),
    ("kling", "Kling"),
    ("luma", "Luma"),
    ("pixverse", "Pixverse"),
    ("suno", "Suno"),
    ("udio", "Udio"),
    ("elevenlabs", "ElevenLabs"),
    ("soundraw", "SoundRaw"),
    ("image playground", "Apple Image Playground"),
    ("microsoft designer", "Microsoft Designer"),
    ("bing image creator", "Bing Image Creator"),
    ("copilot", "Microsoft Copilot"),
    ("canva", "Canva"),
    ("recraft", "Recraft"),
    ("qwen", "Qwen"),
    ("seedream", "Seedream"),
];

/// Return the canonical AI tool name if `haystack` (already lowercased by the caller) mentions
/// a known marker.
pub fn tool_from(haystack: &str) -> Option<&'static str> {
    TOOL_MARKERS
        .iter()
        .find(|(marker, _)| haystack.contains(marker))
        .map(|(_, name)| *name)
}

/// Metadata tokens whose presence indicates AI generation: either an AI-specific key name
/// (`AISystemUsed`, `digitalSourceType`) or a normative IPTC `digitalSourceType` value such as
/// `trainedAlgorithmicMedia`. Matched as a lowercased substring of the metadata text.
const AI_METADATA_KEYS: &[&str] = &[
    "aisystemused",
    "aigenerated",
    "digitalsourcetype",
    "aipromptinformation",
    "genai",
    // digitalSourceType values, not just the key: some files carry the value under a
    // vendor-specific field name we don't track.
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
];

/// Scan a file's raw bytes plus its name for AI-generation hints. `ext` is the lowercased
/// extension (used to pick container parsers); an empty `Vec` means no signals found.
pub fn scan(path: &str, ext: &str, bytes: &[u8]) -> Vec<Signal> {
    let mut signals = Vec::new();

    if let Some(name) = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
    {
        if let Some(sig) = filename_signal(name) {
            signals.push(sig);
        }
    }

    match ext {
        "png" => scan_png(bytes, &mut signals),
        "mp4" | "mov" | "m4a" | "m4v" => scan_bytes_for_tools(bytes, "mp4", &mut signals),
        "mkv" | "webm" | "avi" | "flv" => scan_bytes_for_tools(bytes, "container", &mut signals),
        "mp3" => scan_id3(bytes, &mut signals),
        "wav" | "flac" | "ogg" | "aac" => scan_bytes_for_tools(bytes, "audio", &mut signals),
        _ => {}
    }
    // XMP/EXIF/IPTC text and the TC260 AIGC label can appear across many containers; scan the
    // byte stream for these embedded strings regardless of format.
    scan_embedded_text(bytes, &mut signals);
    scan_aigc_label(bytes, &mut signals);
    scan_grok_signature(bytes, &mut signals);
    scan_c2pa_text(bytes, &mut signals);
    // Source files carry no signed provenance, but authors may declare AI involvement in a
    // top-of-file comment (SPDX-AI tags or a tool header). Unlike git trailers these survive
    // rebases/squashes, so scan any text-ish file for them.
    scan_source_disclosure(bytes, &mut signals);

    dedupe(signals)
}

fn filename_signal(name: &str) -> Option<Signal> {
    let lower = name.to_ascii_lowercase();
    // ElevenLabs exports as `ElevenLabs_YYYY-MM-DDT...`.
    if lower.starts_with("elevenlabs_") {
        return Some(Signal::new(
            Confidence::Low,
            "filename",
            format!("filename matches ElevenLabs export pattern: {name}"),
            Some("ElevenLabs".to_string()),
        ));
    }
    let tool = tool_from(&lower)?;
    Some(Signal::new(
        Confidence::Low,
        "filename",
        format!("filename mentions {tool}: {name}"),
        Some(tool.to_string()),
    ))
}

/// PNG `tEXt`/`iTXt`/`zTXt` chunks carry generation parameters for local diffusion UIs
/// (`parameters`, `prompt`) and tool names. We read the keyword + a slice of the text value.
fn scan_png(bytes: &[u8], out: &mut Vec<Signal>) {
    const SIG: &[u8] = &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    if !bytes.starts_with(SIG) {
        return;
    }
    let mut pos = SIG.len();
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let ctype = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = match data_start.checked_add(len) {
            Some(e) if e <= bytes.len() => e,
            _ => break,
        };
        if matches!(ctype, b"tEXt" | b"iTXt" | b"zTXt") {
            let data = &bytes[data_start..data_end];
            if let Some((keyword, text)) = png_text_kv(ctype, data) {
                let combined = format!("{keyword} {text}").to_ascii_lowercase();
                let key_lower = keyword.to_ascii_lowercase();
                let is_gen_key = matches!(
                    key_lower.as_str(),
                    "parameters" | "prompt" | "sd-metadata" | "comfy" | "workflow"
                );
                let tool = tool_from(&combined);
                if is_gen_key || tool.is_some() {
                    let snippet: String = text.chars().take(80).collect();
                    out.push(Signal::new(
                        Confidence::Medium,
                        "png-text",
                        format!("PNG {keyword} chunk: {snippet}"),
                        tool.map(|t| t.to_string()),
                    ));
                }
            }
        }
        if ctype == b"IEND" {
            break;
        }
        pos = data_end + 4; // skip the trailing CRC
    }
}

/// Extract (keyword, text) from a PNG text chunk. `tEXt` is `keyword\0text`; `iTXt` adds
/// compression/language fields we skip; `zTXt` is compressed and we only read its keyword.
fn png_text_kv(ctype: &[u8], data: &[u8]) -> Option<(String, String)> {
    let nul = data.iter().position(|&b| b == 0)?;
    let keyword = String::from_utf8_lossy(&data[..nul]).to_string();
    let rest = &data[nul + 1..];
    let text = match ctype {
        b"tEXt" => String::from_utf8_lossy(rest).to_string(),
        b"iTXt" => {
            // iTXt: compression flag, compression method, language\0, translated keyword\0, text.
            let after_flags = rest.get(2..).unwrap_or(&[]);
            let mut it = after_flags.splitn(3, |&b| b == 0);
            it.next();
            it.next();
            it.next()
                .map(|t| String::from_utf8_lossy(t).to_string())
                .unwrap_or_default()
        }
        _ => String::new(), // zTXt payload is zlib-compressed; keyword alone is enough to flag
    };
    Some((keyword, text))
}

/// ID3v2 tags on MP3s carry encoder/publisher/comment fields that AI music tools populate
/// (e.g. Suno). We scan the ID3 region for the tool markers.
fn scan_id3(bytes: &[u8], out: &mut Vec<Signal>) {
    if !bytes.starts_with(b"ID3") {
        return;
    }
    scan_bytes_for_tools(bytes, "id3", out);
}

/// Scan embedded XMP/EXIF/IPTC text found in most image containers. We look for AI metadata
/// keys and tool markers in the printable-string content of the file header region.
fn scan_embedded_text(bytes: &[u8], out: &mut Vec<Signal>) {
    // XMP is embedded as an XML packet; find it and scan its text for AI keys.
    if let Some(xmp) = find_xmp(bytes) {
        let lower = xmp.to_ascii_lowercase();
        for key in AI_METADATA_KEYS {
            if lower.contains(key) {
                out.push(Signal::new(
                    Confidence::Medium,
                    "xmp",
                    format!("XMP metadata contains AI key `{key}`"),
                    tool_from(&lower).map(|t| t.to_string()),
                ));
                break;
            }
        }
        if let Some(tool) = tool_from(&lower) {
            out.push(Signal::new(
                Confidence::Medium,
                "xmp",
                format!("XMP metadata mentions {tool}"),
                Some(tool.to_string()),
            ));
        }
    }

    // IPTC lives in a Photoshop `8BIM` resource marked by `Photoshop 3.0`. Rather than decode
    // the binary dataset tags, flag AI keys/tools in that region's printable text.
    if let Some(start) = find_subslice(bytes, b"Photoshop 3.0") {
        let region = &bytes[start..bytes.len().min(start + 8192)];
        let lower = printable_ascii(region).to_ascii_lowercase();
        if let Some(tool) = tool_from(&lower) {
            out.push(Signal::new(
                Confidence::Medium,
                "iptc",
                format!("IPTC metadata mentions {tool}"),
                Some(tool.to_string()),
            ));
        } else if AI_METADATA_KEYS.iter().any(|k| lower.contains(k)) {
            out.push(Signal::new(
                Confidence::Medium,
                "iptc",
                "IPTC metadata contains an AI-generation key".to_string(),
                None,
            ));
        }
    }
}

/// China TC260 AIGC labeling standard: generated media embeds an `AIGC` label plus a
/// `ContentProducer`/`ProduceID` record (in XMP, EXIF UserComment, or a container atom). The
/// presence of the `AIGC` marker alongside a producer field is a deterministic MEDIUM signal.
fn scan_aigc_label(bytes: &[u8], out: &mut Vec<Signal>) {
    let text = printable_ascii(bytes);
    let lower = text.to_ascii_lowercase();
    if lower.contains("aigc") && (lower.contains("contentproducer") || lower.contains("produceid"))
    {
        out.push(Signal::new(
            Confidence::Medium,
            "aigc",
            "TC260 AIGC label present (ContentProducer/ProduceID)".to_string(),
            tool_from(&lower).map(|t| t.to_string()),
        ));
    }
}

/// xAI/Grok (Aurora) images carry no C2PA; instead they embed an EXIF `Signature:` blob
/// alongside a UUID in the `Artist` field. The blob plus a UUID-shaped token is a distinctive
/// pair that ordinary "signature" text won't produce.
fn scan_grok_signature(bytes: &[u8], out: &mut Vec<Signal>) {
    let text = printable_ascii(bytes);
    if text.contains("Signature:") && contains_uuid(&text) {
        out.push(Signal::new(
            Confidence::Medium,
            "exif",
            "xAI/Grok EXIF signature scheme (Signature blob + UUID artist)".to_string(),
            Some("Grok".to_string()),
        ));
    }
}

/// True if `text` contains a canonical 8-4-4-4-12 hex UUID token.
fn contains_uuid(text: &str) -> bool {
    let is_hex_run = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    text.split(|c: char| !c.is_ascii_hexdigit() && c != '-')
        .any(|tok| {
            let parts: Vec<&str> = tok.split('-').collect();
            parts.len() == 5
                && is_hex_run(parts[0], 8)
                && is_hex_run(parts[1], 4)
                && is_hex_run(parts[2], 4)
                && is_hex_run(parts[3], 4)
                && is_hex_run(parts[4], 12)
        })
}

/// C2PA 2.4 defines embedding a signed manifest directly in text/source assets, which the
/// binary-focused `c2patool` doesn't scan: the A.9 ASCII-armour block, the A.8 invisible
/// Unicode variation-selector wrapper, and the A.7 inline HTML `application/c2pa` script. We
/// flag the presence of any of these (signature not verified here) as a MEDIUM provenance hint.
fn scan_c2pa_text(bytes: &[u8], out: &mut Vec<Signal>) {
    let method = if find_subslice(bytes, b"-----BEGIN C2PA MANIFEST-----").is_some() {
        Some("A.9 ASCII-armour block")
    } else if find_subslice(bytes, b"application/c2pa").is_some() {
        Some("A.7 HTML manifest element")
    } else if has_variation_selector_run(bytes) {
        Some("A.8 invisible variation-selector wrapper")
    } else {
        None
    };
    if let Some(method) = method {
        out.push(Signal::new(
            Confidence::Medium,
            "c2pa-text",
            format!("embedded C2PA text manifest ({method}); signature not verified"),
            None,
        ));
    }
}

/// True if the bytes contain a run of UTF-8-encoded Unicode variation selectors long enough to
/// be a C2PA A.8 wrapper rather than an incidental emoji selector. Basic selectors U+FE00..FE0F
/// encode as `EF B8 80..8F`; supplementary U+E0100..E01EF as `F3 A0 84/85 ..`.
fn has_variation_selector_run(bytes: &[u8]) -> bool {
    const MIN_RUN: usize = 16;
    let mut run = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let is_basic = i + 3 <= bytes.len()
            && bytes[i] == 0xEF
            && bytes[i + 1] == 0xB8
            && (0x80..=0x8F).contains(&bytes[i + 2]);
        let is_supp = i + 4 <= bytes.len()
            && bytes[i] == 0xF3
            && bytes[i + 1] == 0xA0
            && matches!(bytes[i + 2], 0x84 | 0x85);
        if is_basic {
            run += 1;
            i += 3;
        } else if is_supp {
            run += 1;
            i += 4;
        } else {
            run = 0;
            i += 1;
        }
        if run >= MIN_RUN {
            return true;
        }
    }
    false
}

/// Locate the XMP packet in a file's byte stream, returning its inner text if present.
fn find_xmp(bytes: &[u8]) -> Option<String> {
    let start = find_subslice(bytes, b"<x:xmpmeta")?;
    let end_marker = b"</x:xmpmeta>";
    let end = find_subslice(&bytes[start..], end_marker)? + start + end_marker.len();
    Some(String::from_utf8_lossy(&bytes[start..end]).to_string())
}

/// Emit a MEDIUM signal for any tool marker found in the printable ASCII of `bytes`.
fn scan_bytes_for_tools(bytes: &[u8], source: &str, out: &mut Vec<Signal>) {
    let text = printable_ascii(bytes).to_ascii_lowercase();
    if let Some(tool) = tool_from(&text) {
        out.push(Signal::new(
            Confidence::Medium,
            source,
            format!("{source} metadata mentions {tool}"),
            Some(tool.to_string()),
        ));
    }
}

/// Detect an author's self-declaration of AI involvement in a text/source file. Two stable
/// conventions exist: the SPDX-AI line tags (W3C AI Content Disclosure + SPDX line-tag format,
/// e.g. `SPDX-AI-Disclosure: ai-generated`) and ad-hoc tool headers (`@cursor-generated`,
/// `Generated by GitHub Copilot`). These live in the file itself, so unlike git trailers they
/// survive rebases and squashes. Only the file head is scanned so a chance match deep in a data
/// blob does not trip it.
fn scan_source_disclosure(bytes: &[u8], out: &mut Vec<Signal>) {
    let head = &bytes[..bytes.len().min(4096)];
    let text = printable_ascii(head);
    let lower = text.to_ascii_lowercase();

    if let Some(pos) = lower.find("spdx-ai-disclosure:") {
        let value = text[pos + "spdx-ai-disclosure:".len()..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        // The vocabulary marks generated/assisted as AI; `none`/`ai-reviewed` are human-authored.
        if value.starts_with("ai-generated") || value.starts_with("ai-assisted") {
            let tool = tool_from(&lower).map(str::to_string);
            out.push(Signal::new(
                Confidence::Medium,
                "source-disclosure",
                format!("SPDX-AI-Disclosure tag declares {value}"),
                tool,
            ));
        }
        return;
    }

    for (marker, tool) in SOURCE_HEADER_MARKERS {
        if lower.contains(marker) {
            out.push(Signal::new(
                Confidence::Medium,
                "source-disclosure",
                format!("source header declares AI generation ({tool})"),
                Some((*tool).to_string()),
            ));
            return;
        }
    }
}

/// Ad-hoc AI-generation headers some tools write into source files, keyed by a lowercased
/// substring. Distinct from `TOOL_MARKERS` (which matches media metadata): these are the
/// specific comment forms seen in code files.
const SOURCE_HEADER_MARKERS: &[(&str, &str)] = &[
    ("@cursor-generated", "Cursor"),
    ("generated by github copilot", "GitHub Copilot"),
    ("generated with [claude code]", "Claude Code"),
    ("@windsurf-generated", "Windsurf"),
];

/// Collapse the bytes to their printable-ASCII runs joined by spaces, so tool markers that
/// straddle binary fields are still found by substring search.
fn printable_ascii(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        s.push(if b.is_ascii_graphic() || b == b' ' {
            b as char
        } else {
            ' '
        });
    }
    s
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Drop duplicate signals (same source+detail) that byte-scanning can produce.
fn dedupe(mut signals: Vec<Signal>) -> Vec<Signal> {
    signals.sort_by(|a, b| {
        b.confidence
            .cmp(&a.confidence)
            .then(a.source.cmp(&b.source))
            .then(a.detail.cmp(&b.detail))
    });
    signals.dedup_by(|a, b| a.source == b.source && a.detail == b.detail);
    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_marker_lookup() {
        assert_eq!(
            tool_from("made with stable diffusion"),
            Some("Stable Diffusion")
        );
        assert_eq!(tool_from("elevenlabs_2025-01-01"), Some("ElevenLabs"));
        assert_eq!(tool_from("generated with anthropic claude"), Some("Claude"));
        assert_eq!(tool_from("a normal photo"), None);
    }

    #[test]
    fn filename_patterns() {
        let s = filename_signal("ElevenLabs_2025-01-01T10_00_00_voice.mp3").unwrap();
        assert_eq!(s.confidence, Confidence::Low);
        assert_eq!(s.tool.as_deref(), Some("ElevenLabs"));
        assert!(filename_signal("IMG_1234.jpg").is_none());
        assert_eq!(
            filename_signal("midjourney_grid.png")
                .unwrap()
                .tool
                .as_deref(),
            Some("Midjourney")
        );
    }

    #[test]
    fn spdx_ai_disclosure_tag_is_detected() {
        let src =
            b"// SPDX-AI-Disclosure: ai-generated\n// SPDX-AI-Provider: Anthropic\nfn main() {}\n";
        let sigs = scan("src/main.rs", "rs", src);
        let s = sigs
            .iter()
            .find(|s| s.source == "source-disclosure")
            .expect("expected a source-disclosure signal");
        assert_eq!(s.confidence, Confidence::Medium);
        assert!(s.detail.contains("ai-generated"));
    }

    #[test]
    fn spdx_ai_none_is_not_flagged() {
        let src = b"// SPDX-AI-Disclosure: none\nfn main() {}\n";
        assert!(scan("src/main.rs", "rs", src)
            .iter()
            .all(|s| s.source != "source-disclosure"));
    }

    #[test]
    fn tool_header_is_detected() {
        let src = b"# @cursor-generated\nprint('hi')\n";
        let s = scan("app.py", "py", src)
            .into_iter()
            .find(|s| s.source == "source-disclosure")
            .expect("expected a tool-header signal");
        assert_eq!(s.tool.as_deref(), Some("Cursor"));
    }

    #[test]
    fn xmp_digitalsourcetype_value_is_flagged() {
        // Midjourney/Firefly write the IPTC value even without an AI-named key we track.
        let xmp = br#"x<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>y"#;
        let mut out = Vec::new();
        scan_embedded_text(xmp, &mut out);
        assert!(out.iter().any(|s| s.source == "xmp"));
    }

    #[test]
    fn png_parameters_chunk_is_flagged() {
        let png = make_png_text(b"parameters", b"masterpiece, Stable Diffusion, steps: 20");
        let mut out = Vec::new();
        scan_png(&png, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].confidence, Confidence::Medium);
        assert_eq!(out[0].source, "png-text");
        assert_eq!(out[0].tool.as_deref(), Some("Stable Diffusion"));
    }

    #[test]
    fn png_without_ai_text_is_ignored() {
        let png = make_png_text(b"Comment", b"just a holiday photo");
        let mut out = Vec::new();
        scan_png(&png, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn xmp_ai_key_is_flagged() {
        let xmp =
            br#"garbage<x:xmpmeta><rdf:Description AISystemUsed="DALL-E 3"/></x:xmpmeta>tail"#;
        let mut out = Vec::new();
        scan_embedded_text(xmp, &mut out);
        assert!(out.iter().any(|s| s.source == "xmp"));
        assert!(out.iter().any(|s| s.tool.as_deref() == Some("DALL-E")));
    }

    #[test]
    fn iptc_block_with_tool_is_flagged() {
        let mut bytes = b"\xff\xed\x00\x40Photoshop 3.0\x008BIM".to_vec();
        bytes.extend_from_slice(b"...Midjourney...");
        let mut out = Vec::new();
        scan_embedded_text(&bytes, &mut out);
        assert!(out
            .iter()
            .any(|s| s.source == "iptc" && s.tool.as_deref() == Some("Midjourney")));
    }

    #[test]
    fn aigc_label_needs_producer_field() {
        let mut only_label = Vec::new();
        scan_aigc_label(b"contains AIGC somewhere", &mut only_label);
        assert!(only_label.is_empty(), "AIGC alone is too weak to flag");

        let mut full = Vec::new();
        scan_aigc_label(b"AIGC ContentProducer=Wan ProduceID=123", &mut full);
        assert_eq!(full.len(), 1);
        assert_eq!(full[0].source, "aigc");
    }

    #[test]
    fn grok_signature_needs_blob_and_uuid() {
        let mut none = Vec::new();
        scan_grok_signature(b"a photo with a Signature: but no id", &mut none);
        assert!(none.is_empty(), "Signature without a UUID must not fire");

        let mut hit = Vec::new();
        scan_grok_signature(
            b"Artist=550e8400-e29b-41d4-a716-446655440000 Signature: AbCdEf00",
            &mut hit,
        );
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].source, "exif");
        assert_eq!(hit[0].tool.as_deref(), Some("Grok"));
    }

    #[test]
    fn uuid_detection() {
        assert!(contains_uuid("x 550e8400-e29b-41d4-a716-446655440000 y"));
        assert!(!contains_uuid("550e8400-e29b-41d4-a716"));
        assert!(!contains_uuid("not-a-uuid-at-all-here"));
    }

    #[test]
    fn c2pa_text_armour_is_flagged() {
        let doc = b"# notes\n-----BEGIN C2PA MANIFEST-----\nAAAA\n-----END C2PA MANIFEST-----\n";
        let mut out = Vec::new();
        scan_c2pa_text(doc, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, "c2pa-text");
        assert!(out[0].detail.contains("A.9"));
    }

    #[test]
    fn c2pa_text_variation_selectors_flagged() {
        // A run of basic variation selectors (EF B8 80..8F) is the A.8 invisible wrapper.
        let mut doc = b"hello".to_vec();
        for _ in 0..20 {
            doc.extend_from_slice(&[0xEF, 0xB8, 0x81]);
        }
        let mut out = Vec::new();
        scan_c2pa_text(&doc, &mut out);
        assert!(out.iter().any(|s| s.detail.contains("A.8")));
    }

    #[test]
    fn plain_text_has_no_c2pa_text_signal() {
        let mut out = Vec::new();
        scan_c2pa_text(b"just a normal README with an emoji or two", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn id3_requires_header() {
        let mut out = Vec::new();
        scan_id3(b"not an mp3 suno", &mut out);
        assert!(out.is_empty(), "no ID3 header => no scan");
    }

    #[test]
    fn id3_with_header_finds_tool() {
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
        bytes.extend_from_slice(b"COMMengmade with Suno");
        let mut out = Vec::new();
        scan_id3(&bytes, &mut out);
        assert!(out.iter().any(|s| s.tool.as_deref() == Some("Suno")));
    }

    #[test]
    fn wav_riff_info_tool_is_flagged() {
        // WAV with a RIFF INFO software tag naming an AI audio tool.
        let mut wav = b"RIFF\x00\x00\x00\x00WAVELIST\x00\x00\x00\x00INFOISFT".to_vec();
        wav.extend_from_slice(b"Suno");
        let signals = scan("clip.wav", "wav", &wav);
        assert!(signals
            .iter()
            .any(|s| s.source == "audio" && s.tool.as_deref() == Some("Suno")));
    }

    #[test]
    fn full_scan_dedupes_and_orders_by_confidence() {
        // A PNG named after a tool: filename (LOW) + png-text (MEDIUM); MEDIUM sorts first.
        let png = make_png_text(b"parameters", b"ComfyUI workflow");
        std::fs::create_dir_all(std::env::temp_dir().join("awd-sig-test")).unwrap();
        let path = std::env::temp_dir().join("awd-sig-test/midjourney_art.png");
        std::fs::write(&path, &png).unwrap();
        let signals = scan(path.to_str().unwrap(), "png", &png);
        assert!(signals.len() >= 2);
        assert_eq!(signals[0].confidence, Confidence::Medium);
        assert!(signals.iter().any(|s| s.source == "filename"));
    }

    /// Build a minimal valid-enough PNG with a single text chunk for testing the parser.
    fn make_png_text(keyword: &[u8], text: &[u8]) -> Vec<u8> {
        let mut v = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        let mut data = Vec::new();
        data.extend_from_slice(keyword);
        data.push(0);
        data.extend_from_slice(text);
        v.extend_from_slice(&(data.len() as u32).to_be_bytes());
        v.extend_from_slice(b"tEXt");
        v.extend_from_slice(&data);
        v.extend_from_slice(&[0, 0, 0, 0]); // fake CRC
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(b"IEND");
        v.extend_from_slice(&[0, 0, 0, 0]);
        v
    }
}
