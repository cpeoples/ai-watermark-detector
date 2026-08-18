//! Git commit provenance: which commits were written by an AI coding agent.
//!
//! This is the code analogue of the C2PA check. AI coding tools declare themselves in the
//! commit's identity and trailers, and those markers are deterministic and greppable (unlike
//! statistical code watermarks such as SWEET, which need the model key). We shell out to
//! `git log` - same approach as the C2PA path shelling out to `c2patool` - and read the author,
//! committer, and message trailers for the well-known agent signals. The markers are
//! self-declared by the tooling, so a match is HIGH confidence, but absence proves nothing.

use serde::Serialize;
use std::process::Command;

/// A record separator unlikely to appear in a commit message, so we can split `git log` output
/// into whole commits and then into fields without a fragile per-line parser.
const REC_SEP: &str = "\u{1e}AWD\u{1e}";
const FIELD_SEP: &str = "\u{1f}";

/// AI-agent signals keyed by a case-insensitive substring, mapped to the tool they identify.
/// Author/committer emails and message trailers are all checked against these. Emails are the
/// stable key (display names drift between releases), matched as a lowercased substring.
const AGENT_MARKERS: &[(&str, &str)] = &[
    ("copilot-swe-agent[bot]", "GitHub Copilot agent"),
    ("copilot@users.noreply.github.com", "GitHub Copilot"),
    ("copilot[bot]@users.noreply.github.com", "GitHub Copilot"),
    ("copilot@github.com", "GitHub Copilot"),
    ("agent-logs-url:", "GitHub Copilot agent"),
    ("cursoragent@cursor.com", "Cursor agent"),
    ("devin-ai-integration[bot]", "Devin"),
    ("devin@cognition", "Devin"),
    ("claude-code", "Claude Code"),
    ("noreply@anthropic.com", "Claude Code"),
    ("generated with [claude code]", "Claude Code"),
    ("noreply@openai.com", "OpenAI Codex"),
    ("codex@openai.com", "OpenAI Codex"),
    ("chatgpt-codex-connector[bot]", "OpenAI Codex"),
    ("aider@aider.chat", "Aider"),
    ("google-labs-jules[bot]", "Google Jules"),
    ("jules@google.com", "Google Jules"),
    ("gemini-code-assist[bot]", "Gemini Code Assist"),
    ("gemini@google.com", "Gemini Code Assist"),
    ("@codeium.com", "Windsurf/Codeium"),
    ("amazon-q-developer", "Amazon Q Developer"),
    ("cline-bot", "Cline"),
    ("cline@cline.ai", "Cline"),
    ("continue-agent", "Continue"),
    ("continue@continue.dev", "Continue"),
    ("noreply@sourcegraph.com", "Sourcegraph Amp/Cody"),
    ("codegen", "Codegen"),
    ("generated-by:", "declared via Generated-By trailer"),
    ("co-authored-by: copilot", "GitHub Copilot"),
    ("co-authored-by: opencode", "OpenCode"),
];

/// One commit and the agent marker found in it, if any.
#[derive(Serialize)]
pub struct CommitReport {
    pub hash: String,
    pub author: String,
    pub subject: String,
    pub ai_authored: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// The marker text that matched (e.g. the trailer or bot email), for transparency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

/// Match `haystack` (raw, any case) against the agent markers, returning (tool, marker).
fn agent_from(haystack: &str) -> Option<(&'static str, &'static str)> {
    let lower = haystack.to_ascii_lowercase();
    AGENT_MARKERS
        .iter()
        .find(|(marker, _)| lower.contains(marker))
        .map(|(marker, tool)| (*tool, *marker))
}

/// Read up to `limit` commits from the repo at `repo` and classify each. Errors carry a
/// human-readable reason (not a git repo, git missing, etc.).
pub fn scan_repo(repo: &str, limit: usize) -> Result<Vec<CommitReport>, String> {
    let format = ["%H", "%an <%ae>", "%cn <%ce>", "%s", "%b"].join(FIELD_SEP);
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("log")
        .arg(format!("--max-count={limit}"))
        .arg(format!("--pretty=format:{REC_SEP}{format}"))
        .output()
        .map_err(|e| format!("could not run git: {e} (is git installed and on PATH?)"))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git log failed in '{repo}': {}", err.trim()));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text
        .split(REC_SEP)
        .filter(|r| !r.trim().is_empty())
        .map(parse_commit)
        .collect())
}

fn parse_commit(record: &str) -> CommitReport {
    let mut fields = record.splitn(5, FIELD_SEP);
    let hash = fields.next().unwrap_or("").trim().to_string();
    let author = fields.next().unwrap_or("").trim().to_string();
    let committer = fields.next().unwrap_or("").to_string();
    let subject = fields.next().unwrap_or("").trim().to_string();
    let body = fields.next().unwrap_or("");

    // Author and committer identities plus the message body all carry agent markers; scan the
    // combined text so a bot committer or a trailer in the body is enough on its own.
    let haystack = format!("{author}\n{committer}\n{body}");
    let (tool, evidence) = match agent_from(&haystack) {
        Some((tool, marker)) => (Some(tool.to_string()), Some(marker.to_string())),
        None => (None, None),
    };

    CommitReport {
        hash,
        author,
        subject,
        ai_authored: tool.is_some(),
        tool,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copilot_coauthor_trailer_is_detected() {
        let (tool, _) = agent_from(
            "feat: x\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
        )
        .unwrap();
        assert_eq!(tool, "GitHub Copilot");
    }

    #[test]
    fn agent_logs_url_trailer_is_detected() {
        assert!(agent_from("Agent-Logs-Url: https://github.com/x/y/actions/runs/1").is_some());
    }

    #[test]
    fn codex_and_aider_and_jules_are_detected() {
        assert_eq!(
            agent_from("Co-authored-by: Codex <noreply@openai.com>")
                .unwrap()
                .0,
            "OpenAI Codex"
        );
        assert_eq!(
            agent_from("Co-authored-by: aider <aider@aider.chat>")
                .unwrap()
                .0,
            "Aider"
        );
        assert_eq!(
            agent_from("Jane <jane@x.com>\ngoogle-labs-jules[bot] <jules@google.com>\n")
                .unwrap()
                .0,
            "Google Jules"
        );
    }

    #[test]
    fn human_commit_has_no_marker() {
        assert!(agent_from("fix typo\n\nSigned-off-by: Jane <jane@example.com>").is_none());
    }

    #[test]
    fn numbered_copilot_bot_email_is_detected() {
        let (tool, _) = agent_from(
            "feat: x\n\nCo-authored-by: Copilot <198982749+Copilot[bot]@users.noreply.github.com>",
        )
        .unwrap();
        assert_eq!(tool, "GitHub Copilot");
    }

    #[test]
    fn parse_commit_splits_fields() {
        let rec = format!(
            "abc123{FIELD_SEP}Jane <jane@x.com>{FIELD_SEP}Jane <jane@x.com>{FIELD_SEP}fix bug{FIELD_SEP}body"
        );
        let c = parse_commit(&rec);
        assert_eq!(c.hash, "abc123");
        assert_eq!(c.subject, "fix bug");
        assert!(!c.ai_authored);
    }

    #[test]
    fn parse_commit_flags_bot_committer() {
        let rec = format!(
            "def456{FIELD_SEP}Jane <jane@x.com>{FIELD_SEP}Copilot <copilot-swe-agent[bot]@users.noreply.github.com>{FIELD_SEP}add endpoint{FIELD_SEP}"
        );
        let c = parse_commit(&rec);
        assert!(c.ai_authored);
        assert_eq!(c.tool.as_deref(), Some("GitHub Copilot agent"));
    }
}
