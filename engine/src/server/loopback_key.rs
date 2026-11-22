//! The address and API key of a simplified start.
//!
//! A simplified start (`serve`, with or without `--model`) listens on loopback only. When the key
//! variable is unset or empty there, it uses a key it generated once and keeps in the user's
//! config directory (a regular file, mode 0600, owned by the user), so the same key works across
//! restarts and no setup step stands between the user and the first request. A start from a
//! `--config` file, and any start whose address is not loopback, still refuses without the
//! variable.

use std::ffi::OsString;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use super::config::{Config, StartupError};

const FILE_NAME: &str = "loopback-key";
const KEY_BYTES: usize = 32;

/// The file the key is kept in: `$XDG_CONFIG_HOME/fifi` or `~/.config/fifi`.
pub(crate) fn key_path(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let base = match xdg.filter(|value| !value.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(home.filter(|home| !home.is_empty())?).join(".config"),
    };
    Some(base.join("fifi").join(FILE_NAME))
}

/// What `listen_and_key` needs from the process: the environment, where the key lives, and
/// whether stderr is a terminal.
pub(crate) struct Host<'a> {
    pub(crate) env: &'a dyn Fn(&str) -> Option<String>,
    pub(crate) key_file: Option<PathBuf>,
    pub(crate) terminal: bool,
}

/// The loopback address to listen on and the key requests must carry. The address is checked
/// first: a non-loopback address is refused before any key is read or made.
pub(crate) fn listen_and_key(
    config: &Config,
    simplified: bool,
    host: &Host<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<(SocketAddr, String), StartupError> {
    let address = config.listen_address()?;
    let key = match config.api_key(host.env) {
        Ok(key) => key,
        Err(_) if simplified && address.ip().is_loopback() => {
            let variable = &config.api_key_env;
            let path = host.key_file.as_deref().ok_or_else(|| {
                StartupError(format!(
                    "no home directory to keep a generated API key in; set {variable} to name one"
                ))
            })?;
            let (key, created) = load_or_create(path)?;
            for line in announce(variable, path, &key, created, host.terminal) {
                say(&line);
            }
            key
        }
        Err(error) => return Err(error),
    };
    Ok((address, key))
}

/// What the user is told about the generated key. The key itself appears only when it was just
/// made and stderr is a terminal; a log never carries it.
fn announce(variable: &str, path: &Path, key: &str, created: bool, terminal: bool) -> Vec<String> {
    let mut lines = vec![format!(
        "api key: {variable} is not set; using the key {} {}",
        if created { "made in" } else { "kept in" },
        path.display()
    )];
    if terminal {
        lines.push(if created {
            format!("api key: send it as `Authorization: Bearer {key}`")
        } else {
            format!(
                "api key: send it as `Authorization: Bearer $(cat {})`",
                path.display()
            )
        });
    }
    lines
}

/// The kept key, made first when there is none. `created` says whether this call made it.
pub(crate) fn load_or_create(path: &Path) -> Result<(String, bool), StartupError> {
    if let Some(key) = read_existing(path)? {
        return Ok((key, false));
    }
    let fail = |error: &dyn std::fmt::Display| {
        StartupError(format!(
            "cannot keep the generated API key in {}: {error}; set the key variable to name one",
            path.display()
        ))
    };
    if let Some(parent) = path.parent() {
        make_private_dir(parent).map_err(|error| fail(&error))?;
    }
    let key = generate().map_err(|error| fail(&error))?;
    // The key goes to a private temporary file first and is linked into place whole: a crash
    // never leaves an empty or half-written key file, and a start that raced us keeps its key.
    let temporary = path.with_file_name(format!(".{FILE_NAME}.{}.tmp", std::process::id()));
    let written = (|| -> std::io::Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&temporary)?;
        file.write_all(key.as_bytes())?;
        file.sync_all()
    })();
    let linked = written.and_then(|()| std::fs::hard_link(&temporary, path));
    let _ = std::fs::remove_file(&temporary);
    match linked {
        Ok(()) => Ok((key, true)),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => read_existing(path)?
            .map(|key| (key, false))
            .ok_or_else(|| fail(&"the file vanished")),
        Err(error) => Err(fail(&error)),
    }
}

/// The key in `path`, or `None` when there is no file. A file that is not a regular file, is
/// owned by someone else, is readable by others, or holds no key is refused with what to do.
fn read_existing(path: &Path) -> Result<Option<String>, StartupError> {
    use std::os::unix::fs::MetadataExt;
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(StartupError(format!(
                "cannot read the API key file {}: {error}",
                path.display()
            )));
        }
    };
    let refuse = |why: &str| {
        StartupError(format!(
            "the API key file {} {why}; delete it and start again to make a new key",
            path.display()
        ))
    };
    if !meta.file_type().is_file() {
        return Err(refuse("is not a regular file (a link or a directory)"));
    }
    if meta.uid() != rustix::process::geteuid().as_raw() {
        return Err(refuse("belongs to another user"));
    }
    if meta.mode() & 0o077 != 0 {
        return Err(refuse("can be read by other users (it must be mode 0600)"));
    }
    let text = std::fs::read_to_string(path).map_err(|error| {
        StartupError(format!(
            "cannot read the API key file {}: {error}",
            path.display()
        ))
    })?;
    let key = text.trim();
    if key.len() < KEY_BYTES {
        return Err(refuse("holds no key"));
    }
    Ok(Some(key.to_string()))
}

/// Creates `dir` and any missing parent, a directory this call makes with mode 0700.
fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

fn generate() -> std::io::Result<String> {
    let mut bytes = [0_u8; KEY_BYTES];
    let mut filled = 0;
    while filled < bytes.len() {
        filled +=
            rustix::rand::getrandom(&mut bytes[filled..], rustix::rand::GetRandomFlags::empty())
                .map_err(std::io::Error::from)?;
    }
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    fn config(listen: &str) -> Config {
        serde_json::from_value(serde_json::json!({
            "listen": listen,
            "api_key_env": "ENGINESRV_API_KEY",
            "engines": [{"id": "e", "model_dir": "m", "tokenizer_dir": "t", "family": "fake"}]
        }))
        .expect("config")
    }

    fn start(
        listen: &str,
        simplified: bool,
        env: Option<&str>,
        file: &Path,
        terminal: bool,
    ) -> (Result<(SocketAddr, String), StartupError>, Vec<String>) {
        let value = env.map(str::to_string);
        let lookup = move |_: &str| value.clone();
        let mut said = Vec::new();
        let result = listen_and_key(
            &config(listen),
            simplified,
            &Host {
                env: &lookup,
                key_file: Some(file.to_path_buf()),
                terminal,
            },
            &mut |line| said.push(line.to_string()),
        );
        (result, said)
    }

    #[test]
    fn the_key_is_made_once_kept_private_and_reused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fifi").join("loopback-key");
        let (first, created) = load_or_create(&path).expect("created");
        assert!(created);
        assert!(first.len() >= 32 && first.bytes().all(|b| b.is_ascii_hexdigit()));
        let (again, created) = load_or_create(&path).expect("reused");
        assert!(!created);
        assert_eq!(first, again);
        assert_eq!(
            std::fs::metadata(&path).expect("meta").mode() & 0o777,
            0o600
        );
        // A directory this call made is private; no temporary file is left.
        let parent = path.parent().expect("parent");
        assert_eq!(
            std::fs::metadata(parent).expect("meta").mode() & 0o777,
            0o700
        );
        assert_eq!(std::fs::read_dir(parent).expect("dir").count(), 1);
    }

    #[test]
    fn a_key_file_that_could_be_read_or_replaced_by_others_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("loopback-key");
        let good = "k".repeat(64);
        std::fs::write(&path, &good).expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let error = load_or_create(&path).unwrap_err().0;
        assert!(
            error.contains("mode 0600") && error.contains("delete it"),
            "{error}"
        );
        // A link is not followed, even to a good file.
        let target = dir.path().join("real");
        std::fs::write(&target, &good).expect("write");
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).expect("link");
        let error = load_or_create(&link).unwrap_err().0;
        assert!(error.contains("not a regular file"), "{error}");
        // A file with no key in it is refused, naming the file to delete, and left alone.
        let empty = dir.path().join("empty");
        std::fs::write(&empty, "short").expect("write");
        std::fs::set_permissions(&empty, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        let error = load_or_create(&empty).unwrap_err().0;
        assert!(
            error.contains("holds no key") && error.contains("empty"),
            "{error}"
        );
        assert_eq!(std::fs::read_to_string(&empty).expect("read"), "short");
    }

    #[test]
    fn a_non_loopback_address_is_refused_before_any_key_is_made() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("fifi").join("loopback-key");
        for listen in ["0.0.0.0:18791", "[::]:18791", "192.168.1.5:18791"] {
            let (result, said) = start(listen, true, None, &file, true);
            let error = result.unwrap_err().0;
            assert!(error.contains("loopback"), "{listen}: {error}");
            assert!(said.is_empty());
            assert!(
                !file.exists() && !file.parent().expect("parent").exists(),
                "{listen}"
            );
        }
        let (result, _) = start("[::1]:18791", true, None, &file, false);
        let (address, key) = result.expect("::1 is loopback");
        assert!(address.ip().is_loopback() && key.len() >= 32);
    }

    #[test]
    fn a_set_variable_is_the_key_and_no_file_is_made() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("loopback-key");
        let (result, said) = start("127.0.0.1:18791", true, Some("sk-given"), &file, true);
        assert_eq!(result.expect("key").1, "sk-given");
        assert!(said.is_empty() && !file.exists());
    }

    #[test]
    fn a_config_start_without_the_variable_still_refuses() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("loopback-key");
        let (result, _) = start("127.0.0.1:18791", false, None, &file, true);
        assert!(result.unwrap_err().0.contains("ENGINESRV_API_KEY"));
        assert!(!file.exists());
        // An empty variable counts as unset in both.
        let (result, _) = start("127.0.0.1:18791", false, Some(""), &file, true);
        assert!(result.is_err());
    }

    #[test]
    fn the_key_is_shown_once_on_a_terminal_and_never_in_a_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("loopback-key");
        let (result, said) = start("127.0.0.1:18791", true, None, &file, true);
        let key = result.expect("key").1;
        assert!(said.iter().any(|line| line.contains(&key)), "{said:?}");
        let (_, said) = start("127.0.0.1:18791", true, None, &file, true);
        assert!(said.iter().all(|line| !line.contains(&key)), "{said:?}");
        assert!(said.iter().any(|line| line.contains("$(cat ")), "{said:?}");
        let (_, said) = start("127.0.0.1:18791", true, None, &file, false);
        assert!(
            said.iter()
                .all(|line| !line.contains(&key) && !line.contains("Bearer")),
            "{said:?}"
        );
        assert!(
            said.iter()
                .any(|line| line.contains(file.to_str().expect("utf8")))
        );
    }

    #[test]
    fn the_key_file_follows_the_config_directory_rules() {
        let from = |xdg: Option<&str>, home: Option<&str>| {
            key_path(xdg.map(OsString::from), home.map(OsString::from))
        };
        assert_eq!(
            from(Some("/x"), Some("/h")),
            Some(PathBuf::from("/x/fifi/loopback-key"))
        );
        assert_eq!(
            from(None, Some("/h")),
            Some(PathBuf::from("/h/.config/fifi/loopback-key"))
        );
        assert_eq!(
            from(Some(""), Some("/h")),
            Some(PathBuf::from("/h/.config/fifi/loopback-key"))
        );
        assert_eq!(from(None, None), None);
        assert_eq!(from(None, Some("")), None);
    }

    #[test]
    fn a_file_owned_by_the_user_is_what_the_owner_check_accepts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("loopback-key");
        load_or_create(&path).expect("created");
        assert_eq!(
            std::fs::metadata(&path).expect("meta").uid(),
            rustix::process::geteuid().as_raw()
        );
    }
}
