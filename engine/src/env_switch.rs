//! The fail-safe default-on environment switch shared by every default-on
//! flag (tokenizer replay flags, the Flint family switches, the CPU Q8g
//! kernel switches): an absent variable keeps the new default (on); once
//! set, only the explicit enable words ("1"/"true"/"on", trimmed,
//! case-folded) keep the feature on, and anything else — disable words,
//! case games, whitespace, CRLF, typos, a value that is not UTF-8 — rolls
//! back. A rollback can never silently fail.

/// The fail-safe default-on env switch shared by the tokenizer replay flags
/// and the Flint family switches (`FIFI_FLINT_MLX_ASYNC_CHAIN`,
/// `FIFI_FLINT_MLX_COALESCE_RESIDUAL`, `FIFI_FLINT_MLX_DECODE_ATTN`,
/// `FIFI_FLINT_MLX_RING_VIEW`, `FIFI_FLINT_CUDA_ALLOC_CACHE`): an absent
/// variable keeps the new default (on); once set, only the explicit enable
/// words ("1"/"true"/"on", trimmed, case-folded) keep the feature on, and
/// anything else — disable words, case games, whitespace, CRLF, typos, a
/// value that is not UTF-8 — rolls back. A rollback can never silently
/// fail.
pub fn env_allows_default_on(name: &str) -> bool {
    env_result_allows_default_on(std::env::var(name))
}

/// The pure `Result`-shaped core of [`env_allows_default_on`]: absent keeps
/// the default on; non-UTF-8 and any set value resolve through
/// [`env_allows_value`]. Shared with switch helpers that receive an
/// already-read env result (e.g. the ring-view flag) so the polarity match
/// exists in exactly one place.
pub fn env_result_allows_default_on(value: Result<String, std::env::VarError>) -> bool {
    match value {
        // Only a genuinely absent variable keeps the new default; a set
        // value that is not UTF-8 is a mis-set rollback and disables.
        Err(std::env::VarError::NotPresent) => true,
        Err(std::env::VarError::NotUnicode(_)) => false,
        Ok(value) => env_allows_value(Some(&value)),
    }
}

/// The pure polarity half of [`env_allows_default_on`], testable without
/// touching process-global environment.
pub fn env_allows_value(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fail-safe polarity of the default-on switches: unset keeps the new
    /// default; once set, only the explicit enable words keep the feature on
    /// and everything else — disable words, case games, whitespace, CRLF,
    /// unrecognized values, typos — rolls back.
    #[test]
    fn default_on_flags_fail_safe_on_any_set_value() {
        let allows = env_allows_value;
        assert!(allows(None), "absent keeps the new default");
        assert!(allows(Some("1")), "the enable words keep it on");
        assert!(allows(Some("true")), "the enable words keep it on");
        assert!(allows(Some("on")), "the enable words keep it on");
        assert!(
            allows(Some(" ON ")),
            "trimmed and case-folded enable words keep it on"
        );
        assert!(!allows(Some("0")), "the documented rollback disables");
        assert!(!allows(Some("false")), "false disables");
        assert!(!allows(Some("off")), "off disables");
        assert!(!allows(Some("no")), "no disables");
        assert!(!allows(Some("FALSE")), "case-folded disable words disable");
        assert!(
            !allows(Some("0 ")),
            "trailing whitespace cannot rescue a rollback"
        );
        assert!(!allows(Some("0\r")), "a CRLF rollback still disables");
        assert!(!allows(Some("")), "an empty export disables");
        assert!(
            !allows(Some("yes")),
            "unrecognized values roll back, fail safe"
        );
        assert!(!allows(Some("flase")), "typos roll back, fail safe");
    }

    /// The lifted default-on reader is pinned by REAL environment rows in
    /// child processes (the parent re-execs the test binary per case):
    /// absent keeps the default; `0`/`off`/padded/empty/non-UTF-8 roll
    /// back; the enable words keep it on. This is the rollback authority
    /// for every default-on switch, so the fail-open mutants
    /// (`Ok(_) => true`, `Ok(_) => env_allows_value(None)`,
    /// `NotUnicode(_) => true`) all die here.
    #[test]
    fn the_default_on_environment_parses_in_child_processes() {
        const NAME: &str = "FIFI_ENGINE_TEST_DEFAULT_ON";
        const CHILD: &str =
            "env_switch::tests::the_default_on_environment_parses_in_child_processes";
        if std::env::var_os("DEFAULT_ON_CHILD").is_none() {
            for (value, expected) in [
                ("0", false),
                ("off", false),
                (" 0 ", false),
                ("", false),
                ("maybe", false),
                (" ON ", true),
                ("1", true),
                ("true", true),
            ] {
                let mut command =
                    std::process::Command::new(std::env::current_exe().expect("test binary"));
                command
                    .args(["--exact", CHILD, "--test-threads", "1", "--nocapture"])
                    .env("DEFAULT_ON_CHILD", "1")
                    .env_remove(NAME)
                    .env(NAME, value);
                let output = command.output().expect("spawn the default-on child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success(),
                    "default-on child for {value:?} exited with {}\nstdout:\n{stdout}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(
                    stdout.contains(&format!("default on child: {expected}")),
                    "child for {value:?} missing 'default on child: {expected}':\n{stdout}"
                );
            }
            // A non-UTF-8 value cannot ride the &str case table; spawn it
            // directly: a mis-set rollback value disables (the fail-safe rule).
            {
                use std::os::unix::ffi::OsStrExt;
                let mut command =
                    std::process::Command::new(std::env::current_exe().expect("test binary"));
                command
                    .args(["--exact", CHILD, "--test-threads", "1", "--nocapture"])
                    .env("DEFAULT_ON_CHILD", "1")
                    .env_remove(NAME)
                    .env(NAME, std::ffi::OsStr::from_bytes(b"\xff"));
                let output = command
                    .output()
                    .expect("spawn the non-UTF-8 default-on child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success() && stdout.contains("default on child: false"),
                    "non-UTF-8 value must roll back:\n{stdout}"
                );
            }
            // The absent arm in the same table shape: no variable at all.
            {
                let mut command =
                    std::process::Command::new(std::env::current_exe().expect("test binary"));
                command
                    .args(["--exact", CHILD, "--test-threads", "1", "--nocapture"])
                    .env("DEFAULT_ON_CHILD", "1")
                    .env_remove(NAME);
                let output = command.output().expect("spawn the absent default-on child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success() && stdout.contains("default on child: true"),
                    "an absent variable must keep the default:\n{stdout}"
                );
            }
            return;
        }
        println!("default on child: {}", env_allows_default_on(NAME));
    }
}
