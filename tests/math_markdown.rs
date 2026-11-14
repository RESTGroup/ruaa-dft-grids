//! Enforce the math-in-Markdown convention used by the long-form docs.
//!
//! Rustdoc parses doc comments as Markdown before KaTeX sees them.  A bare
//! `_` that does not immediately follow a letter or digit can open emphasis,
//! and a LaTeX command written as backslash plus ASCII punctuation is
//! consumed by Markdown.  The rules checked here are:
//!
//! * inside math, a `_` must follow an ASCII letter/digit or be written `\_`;
//! * inside math, no `\` + ASCII punctuation other than `\_`;
//! * inside math, no bare `*` (write `\ast`).

use std::fs;
use std::path::{Path, PathBuf};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn math_markdown_convention() {
    let root = Path::new(ROOT);
    let mut files = vec![root.join("README.md")];
    collect(&root.join("docs"), "md", &mut files);
    collect(&root.join("src"), "rs", &mut files);
    files.sort();

    let mut errors = Vec::new();
    for path in files {
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => panic!("cannot read {}: {err}", path.display()),
        };
        let view = if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            doc_comment_view(&text)
        } else {
            text
        };
        check_file(root, &path, &view, &mut errors);
    }

    assert!(
        errors.is_empty(),
        "math-in-Markdown convention violations:\n{}",
        errors.join("\n")
    );
}

fn collect(dir: &Path, extension: &str, files: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, extension, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
            files.push(path);
        }
    }
}

/// Keep only the text that rustdoc treats as Markdown.  Line numbers are
/// preserved so lint messages point at the source line.
fn doc_comment_view(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
        }
        let trimmed = line.trim_start();
        let content = if trimmed.starts_with("///") && !trimmed.starts_with("////") {
            trimmed.strip_prefix("///")
        } else {
            trimmed.strip_prefix("//!")
        };
        if let Some(content) = content {
            out.push_str(&line[..line.len() - trimmed.len()]);
            out.push_str(content);
        }
    }
    out
}

/// Blank out fenced code blocks and inline code spans, preserving newlines and
/// character positions.  Math is only searched in the remaining text.
fn mask_code(text: &str) -> Vec<char> {
    let mut chars: Vec<char> = text.chars().collect();
    let mut fence: Option<&str> = None;
    let mut line_start = 0;
    let mut index = 0;
    while index < chars.len() {
        if index == line_start {
            let line_end = chars[index..]
                .iter()
                .position(|ch| *ch == '\n')
                .map_or(chars.len(), |offset| index + offset);
            let line: String = chars[index..line_end].iter().collect();
            let trimmed = line.trim_start();
            if let Some(marker) = fence {
                let closing = trimmed.starts_with(marker);
                blank(&mut chars, index, line_end);
                if closing {
                    fence = None;
                }
                index = line_end;
                line_start = line_end + 1;
                continue;
            }
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                fence = Some(if trimmed.starts_with("```") {
                    "```"
                } else {
                    "~~~"
                });
                blank(&mut chars, index, line_end);
                index = line_end;
                line_start = line_end + 1;
                continue;
            }
        }
        if chars[index] == '\n' {
            line_start = index + 1;
        }
        index += 1;
    }

    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '`' {
            index += 1;
            continue;
        }
        let open_start = index;
        while index < chars.len() && chars[index] == '`' {
            index += 1;
        }
        let run = index - open_start;
        let mut cursor = index;
        let mut close = None;
        while cursor < chars.len() {
            if chars[cursor] != '`' {
                cursor += 1;
                continue;
            }
            let close_start = cursor;
            while cursor < chars.len() && chars[cursor] == '`' {
                cursor += 1;
            }
            if cursor - close_start == run {
                close = Some(cursor);
                break;
            }
        }
        if let Some(close_end) = close {
            blank(&mut chars, open_start, close_end);
            index = close_end;
        }
    }
    chars
}

fn blank(chars: &mut [char], start: usize, end: usize) {
    for ch in &mut chars[start..end] {
        if *ch != '\n' {
            *ch = ' ';
        }
    }
}

fn check_file(root: &Path, path: &Path, view: &str, errors: &mut Vec<String>) {
    let chars = mask_code(view);
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '$' || is_escaped(&chars, index) {
            index += 1;
            continue;
        }
        if index + 1 < chars.len() && chars[index + 1] == '$' {
            if let Some(end) = find_display_end(&chars, index + 2) {
                check_math(root, path, view, &chars, index + 2, end, errors);
                index = end + 2;
                continue;
            }
        } else if let Some(end) = find_inline_end(&chars, index + 1) {
            check_math(root, path, view, &chars, index + 1, end, errors);
            index = end + 1;
            continue;
        }
        index += 1;
    }
}

fn find_inline_end(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    while index < chars.len() {
        if chars[index] == '\n' && index + 1 < chars.len() && chars[index + 1] == '\n' {
            return None;
        }
        if chars[index] == '$' && !is_escaped(chars, index) {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn find_display_end(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    while index + 1 < chars.len() {
        if chars[index] == '$' && chars[index + 1] == '$' && !is_escaped(chars, index) {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn is_escaped(chars: &[char], index: usize) -> bool {
    let mut backslashes = 0;
    let mut cursor = index;
    while cursor > 0 && chars[cursor - 1] == '\\' {
        backslashes += 1;
        cursor -= 1;
    }
    backslashes % 2 == 1
}

fn check_math(
    root: &Path,
    path: &Path,
    view: &str,
    chars: &[char],
    start: usize,
    end: usize,
    errors: &mut Vec<String>,
) {
    let mut index = start;
    while index < end {
        let ch = chars[index];
        if ch == '\\' && index + 1 < end {
            let next = chars[index + 1];
            if next.is_ascii_punctuation() && next != '_' {
                push_error(
                    root,
                    path,
                    view,
                    chars,
                    index,
                    &format!("LaTeX `\\{next}` is consumed by Markdown; use a named command"),
                    errors,
                );
            }
            index += 2;
            continue;
        }
        if ch == '_' {
            let safe = index > start && chars[index - 1].is_ascii_alphanumeric();
            if !safe {
                push_error(
                    root,
                    path,
                    view,
                    chars,
                    index,
                    "a bare `_` here can open Markdown emphasis; write `\\_`",
                    errors,
                );
            }
        } else if ch == '*' {
            push_error(
                root,
                path,
                view,
                chars,
                index,
                "a bare `*` in math can open emphasis; write `\\ast`",
                errors,
            );
        }
        index += 1;
    }
}

fn push_error(
    root: &Path,
    path: &Path,
    view: &str,
    chars: &[char],
    index: usize,
    message: &str,
    errors: &mut Vec<String>,
) {
    let line = 1 + chars[..index].iter().filter(|ch| **ch == '\n').count();
    let snippet = view.lines().nth(line - 1).unwrap_or("").trim();
    let relative = path.strip_prefix(root).unwrap_or(path);
    errors.push(format!(
        "{}:{line}: {message}\n    {snippet}",
        relative.display()
    ));
}

fn violations(view: &str) -> Vec<String> {
    let mut errors = Vec::new();
    check_file(Path::new("."), Path::new("sample.md"), view, &mut errors);
    errors
}

#[test]
fn scanner_self_check() {
    for text in [
        "$\\hat{\\mathbf u}_{AB}$",
        "$$\n\\hat{\\mathbf u}_{AB}\n$$",
        "$a\\,b$",
        "$x^*$",
    ] {
        assert!(
            !violations(text).is_empty(),
            "expected a violation for {text:?}"
        );
    }
    for text in [
        "$w_g$",
        "$\\hat{\\mathbf u}\\_{AB}$",
        "$a\\thinspace b$",
        "`$x_{i}$`",
        "```text\n$x_{i}$\n```",
    ] {
        let errors = violations(text);
        assert!(
            errors.is_empty(),
            "unexpected violation for {text:?}: {errors:?}"
        );
    }
}
