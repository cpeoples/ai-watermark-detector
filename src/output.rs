//! Shared output rendering for the CLI.
//!
//! Every subcommand serializes its result through one path so the machine-readable formats
//! stay consistent: `text` is each command's own human layout, while `json`, `xml`, and
//! `yaml` all go through `serde` on the same result structs (never hand-built strings).

use serde::Serialize;

/// Machine-readable formats the CLI can emit, plus the human `text` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Xml,
    Yaml,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Result<OutputFormat, String> {
        match s.to_ascii_lowercase().as_str() {
            "text" | "plain" | "human" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "xml" => Ok(OutputFormat::Xml),
            "yaml" | "yml" => Ok(OutputFormat::Yaml),
            other => Err(format!(
                "unknown format: {other} (use `text`, `json`, `xml`, or `yaml`)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Text => "text",
            OutputFormat::Json => "json",
            OutputFormat::Xml => "xml",
            OutputFormat::Yaml => "yaml",
        }
    }
}

/// Serialize `value` into one of the machine-readable formats. `root` names the XML root
/// element (ignored by JSON/YAML). Returns `None` for `Text`, which each command renders
/// itself.
pub fn render<T: Serialize>(
    value: &T,
    format: OutputFormat,
    root: &str,
) -> Result<Option<String>, String> {
    let out = match format {
        OutputFormat::Text => return Ok(None),
        OutputFormat::Json => serde_json::to_string_pretty(value).map_err(|e| e.to_string())?,
        OutputFormat::Yaml => yaml_serde::to_string(value).map_err(|e| e.to_string())?,
        OutputFormat::Xml => {
            let mut buf = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            let mut ser = quick_xml::se::Serializer::with_root(&mut buf, Some(root))
                .map_err(|e| e.to_string())?;
            ser.indent(' ', 2);
            value.serialize(ser).map_err(|e| e.to_string())?;
            buf
        }
    };
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Sample {
        name: String,
        count: u32,
        ratio: f64,
        flag: bool,
    }

    fn sample() -> Sample {
        Sample {
            name: "kgw".to_string(),
            count: 42,
            ratio: 0.25,
            flag: true,
        }
    }

    #[test]
    fn format_parse_roundtrips() {
        for f in [
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::Xml,
            OutputFormat::Yaml,
        ] {
            assert_eq!(OutputFormat::parse(f.as_str()), Ok(f));
        }
        assert_eq!(OutputFormat::parse("yml"), Ok(OutputFormat::Yaml));
        assert!(OutputFormat::parse("toml").is_err());
    }

    #[test]
    fn text_renders_nothing() {
        assert_eq!(render(&sample(), OutputFormat::Text, "row").unwrap(), None);
    }

    #[test]
    fn json_is_parseable_and_faithful() {
        let s = render(&sample(), OutputFormat::Json, "row")
            .unwrap()
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["name"], "kgw");
        assert_eq!(v["count"], 42);
        assert_eq!(v["flag"], true);
    }

    #[test]
    fn yaml_is_parseable_and_faithful() {
        let s = render(&sample(), OutputFormat::Yaml, "row")
            .unwrap()
            .unwrap();
        let v: serde_json::Value = yaml_serde::from_str(&s).unwrap();
        assert_eq!(v["name"], "kgw");
        assert_eq!(v["count"], 42);
    }

    #[test]
    fn xml_has_root_and_fields() {
        let s = render(&sample(), OutputFormat::Xml, "row")
            .unwrap()
            .unwrap();
        assert!(s.starts_with("<?xml"));
        assert!(s.contains("<row>"));
        assert!(s.contains("<name>kgw</name>"));
        assert!(s.contains("<count>42</count>"));
    }
}
