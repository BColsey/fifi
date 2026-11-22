//! The API key of a start that names none.
//!
//! A simplified start (`serve`, with or without `--model`) listens on loopback only. When its key
//! variable is not set, it uses a key it generated once and keeps in the user's config directory
//! (mode 0600), so the same key works across restarts and no setup step stands between the user
//! and the first request. A start that names a key variable and finds it empty still refuses.

use std::io::Write;
use std::path::{Path, PathBuf};

use super::config::StartupError;

const FILE_NAME: &str = "loopback-key";
const KEY_BYTES: usize = 32;

/// The file the key is kept in: `$XDG_CONFIG_HOME/fifi` or `~/.config/fifi`.
pub(crate) fn key_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => {
            PathBuf::from(std::env::var_os("HOME").filter(|home| !home.is_empty())?).join(".config")
        }
    };
    Some(base.join("fifi").join(FILE_NAME))
}

/// The kept key, made first when there is none. `created` says whether this call made it.
pub(crate) fn load_or_create(path: &Path) -> Result<(String, bool), StartupError> {
    let fail = |error: &dyn std::fmt::Display| {
        StartupError(format!(
            "cannot keep the generated API key in {}: {error}; set the key variable to name one",
            path.display()
        ))
    };
    if let Some(key) = read(path) {
        return Ok((key, false));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| fail(&error))?;
    }
    let key = generate().map_err(|error| fail(&error))?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    match options.open(path) {
        Ok(mut file) => {
            file.write_all(key.as_bytes())
                .map_err(|error| fail(&error))?;
            Ok((key, true))
        }
        // Another start made it first: use theirs.
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => read(path)
            .map(|key| (key, false))
            .ok_or_else(|| fail(&"the file exists and holds no key")),
        Err(error) => Err(fail(&error)),
    }
}

fn read(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let key = text.trim();
    (key.len() >= KEY_BYTES).then(|| key.to_string())
}

fn generate() -> std::io::Result<String> {
    let mut bytes = [0_u8; KEY_BYTES];
    std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom")?, &mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).expect("meta").permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn a_file_with_no_key_in_it_is_refused_not_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("loopback-key");
        std::fs::write(&path, "short").expect("write");
        let error = load_or_create(&path).unwrap_err().0;
        assert!(error.contains("holds no key"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "short");
    }
}
