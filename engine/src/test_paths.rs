//! Host locations of the model weights, fixtures and scratch roots that tests read.
//!
//! A location is looked up by one name: the environment variable of that name first, then the
//! key of that name in `~/.config/fifi/test-paths.json`, a flat JSON object of name to path.
//! The file lives outside the repository, so every worktree on a host shares it.
//!
//! A test that skips when its location is unresolved fails instead when `FIFI_TEST_REQUIRE_DATA`
//! is `1`, as the other data-dependent tests do. A deliberate `#[ignore]`d run always fails on a missing location.

use std::path::{Path, PathBuf};

const CONFIG_RELATIVE: &str = ".config/fifi/test-paths.json";

/// The location named `name`, from the environment or from the user config file.
pub(crate) fn lookup(name: &str) -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(name).filter(|value| !value.is_empty()) {
        return Some(value.into());
    }
    let home = std::env::var_os("HOME")?;
    let text = std::fs::read_to_string(Path::new(&home).join(CONFIG_RELATIVE)).ok()?;
    config_value(&text, name).map(PathBuf::from)
}

/// The string at key `name` of the config object. A file that is not a JSON object panics, so
/// a typo cannot hide a path.
fn config_value(text: &str, name: &str) -> Option<String> {
    let config: serde_json::Value = serde_json::from_str(text)
        .unwrap_or_else(|error| panic!("~/{CONFIG_RELATIVE} is not valid JSON: {error}"));
    let object = config
        .as_object()
        .unwrap_or_else(|| panic!("~/{CONFIG_RELATIVE} must be a JSON object"));
    object.get(name)?.as_str().map(str::to_owned)
}

/// Whether the run must not skip for a missing input: `FIFI_TEST_REQUIRE_DATA` is `1`.
fn require_mode() -> bool {
    std::env::var("FIFI_TEST_REQUIRE_DATA").as_deref() == Ok("1")
}

/// The location named `name`, for a deliberate run (an `#[ignore]`d test): an unresolved
/// name panics and says where to set it.
pub(crate) fn required(name: &str) -> PathBuf {
    lookup(name).unwrap_or_else(|| panic!("{}", unresolved(name)))
}

/// The location named `name`, or `None` after printing a SKIPPING line so a host without the
/// input still runs the rest of the suite; under `FIFI_TEST_REQUIRE_DATA` an unresolved
/// name panics instead, so a gate never passes without its input.
pub(crate) fn or_skip(name: &str) -> Option<PathBuf> {
    let found = lookup(name);
    if found.is_none() {
        assert!(!require_mode(), "{}", unresolved(name));
        eprintln!("{}: SKIPPING", unresolved(name));
    }
    found
}

fn unresolved(name: &str) -> String {
    format!("{name} is not set and has no entry in ~/{CONFIG_RELATIVE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"A_DIR": "/data/a", "N": 3}"#;

    #[test]
    fn config_value_reads_a_string_key() {
        assert_eq!(config_value(SAMPLE, "A_DIR").as_deref(), Some("/data/a"));
    }

    #[test]
    fn config_value_misses_unknown_and_non_string_keys() {
        assert_eq!(config_value(SAMPLE, "MISSING"), None);
        assert_eq!(config_value(SAMPLE, "N"), None);
    }

    #[test]
    #[should_panic(expected = "must be a JSON object")]
    fn config_value_rejects_a_file_that_is_not_an_object() {
        config_value("[1]", "A_DIR");
    }
}
