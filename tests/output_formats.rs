//! End-to-end output-format tests for the `score` subcommand.
//!
//! These run the actual compiled binary (Cargo exposes its path via CARGO_BIN_EXE_*) over a
//! known token stream and assert that each `--format` produces the expected, parseable shape.
//! `score` needs no external tools, so it exercises the shared renderer for all four formats
//! without depending on `c2patool`.

use std::process::Command;

const CONFIG: &str = "config.example.json";

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ai-watermark-detector"))
}

/// A short green-heavy KGW stream: enough to run, exercised only for output shape.
fn tokens() -> String {
    (0..40).map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
}

fn run(format_args: &[&str]) -> String {
    let out = bin()
        .args(["score", "--config", CONFIG, "--scheme", "kgw", "--tokens"])
        .arg(tokens())
        .args(format_args)
        .output()
        .expect("failed to run binary");
    assert!(
        out.status.success(),
        "score exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout was not UTF-8")
}

#[test]
fn text_is_the_default_and_human_readable() {
    let default = run(&[]);
    assert!(default.contains("AI watermark screen"));
    // Explicit --format text matches the default.
    assert!(run(&["--format", "text"]).contains("AI watermark screen"));
}

#[test]
fn json_output_is_valid_and_has_contract_fields() {
    let s = run(&["--format", "json"]);
    let v: serde_json::Value = serde_json::from_str(&s).expect("not valid JSON");
    for key in [
        "scheme",
        "tokens",
        "mean_g",
        "z",
        "approx_p_value",
        "reliable",
    ] {
        assert!(v.get(key).is_some(), "JSON missing field {key}");
    }
    assert_eq!(v["scheme"], "kgw");
}

#[test]
fn legacy_json_flag_matches_format_json() {
    let a: serde_json::Value = serde_json::from_str(&run(&["--json"])).unwrap();
    let b: serde_json::Value = serde_json::from_str(&run(&["--format", "json"])).unwrap();
    assert_eq!(a, b, "--json and --format json must agree");
}

#[test]
fn yaml_output_is_valid_and_has_contract_fields() {
    let s = run(&["--format", "yaml"]);
    let v: serde_json::Value = yaml_serde::from_str(&s).expect("not valid YAML");
    assert_eq!(v["scheme"], "kgw");
    assert!(v.get("z").is_some());
}

#[test]
fn xml_output_is_well_formed_with_root_and_fields() {
    let s = run(&["--format", "xml"]);
    assert!(s.starts_with("<?xml"), "missing XML declaration");
    assert!(
        s.contains("<score>") && s.contains("</score>"),
        "missing root"
    );
    assert!(s.contains("<scheme>kgw</scheme>"), "missing scheme field");
}

#[test]
fn unknown_format_is_rejected() {
    let out = bin()
        .args(["score", "--config", CONFIG, "--scheme", "kgw", "--tokens"])
        .arg(tokens())
        .args(["--format", "toml"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "unknown format should fail");
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown format"));
}
