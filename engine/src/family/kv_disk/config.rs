//! The configuration of the on-disk prefix cache, shared by every family that wires it: the
//! `kv_cache` object of an engine config, its validation, and the build default.

use serde::Deserialize;
use std::path::PathBuf;

/// The default of the on-disk prefix cache when the config and the environment say nothing: on,
/// at the default directory and quota. `kv_cache.enabled: false` or `FIFI_KV_CACHE=0` turns it
/// off; the cache also stays off when the host cannot name a directory or has too little free
/// space.
pub(crate) const KV_CACHE_DEFAULT_ON: bool = true;

/// The least quota an operator may set.
const QUOTA_MIN_BYTES: u64 = 1 << 30;

/// The `kv_cache` member of an engine's family object.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct KvCacheOptions {
    /// The cache directory, absolute. Naming it turns the cache on. Created 0700 when missing;
    /// one process owns it at a time. Absent means the user cache directory
    /// (`$XDG_CACHE_HOME/fifi/kv`, else `$HOME/.cache/fifi/kv`).
    pub(crate) dir: Option<PathBuf>,
    /// The byte quota of the directory, from 1 GiB. Absent means a quarter of the free space of
    /// the filesystem at start, at most 64 GiB; under 4 GiB the cache stays off.
    pub(crate) quota_bytes: Option<u64>,
    /// `true` turns the cache on at the default directory, `false` keeps it off whatever else
    /// is set. Absent follows the default of the build.
    pub(crate) enabled: Option<bool>,
}

/// The checked `kv_cache` member: what the operator asked for, before the host is probed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct KvCacheSettings {
    pub(crate) enabled: Option<bool>,
    pub(crate) dir: Option<PathBuf>,
    pub(crate) quota_bytes: Option<u64>,
}

impl KvCacheOptions {
    /// Checks the members that do not depend on the family: an absolute directory and a quota of
    /// at least 1 GiB. The error is the text of the refusal.
    pub(crate) fn checked(&self) -> Result<KvCacheSettings, String> {
        if let Some(dir) = &self.dir
            && !dir.is_absolute()
        {
            return Err(format!(
                "kv_cache.dir must be an absolute path, got {}",
                dir.display()
            ));
        }
        if let Some(quota) = self.quota_bytes
            && quota < QUOTA_MIN_BYTES
        {
            return Err(format!(
                "kv_cache.quota_bytes must be at least {QUOTA_MIN_BYTES} (1 GiB), got {quota}"
            ));
        }
        Ok(KvCacheSettings {
            enabled: self.enabled,
            dir: self.dir.clone(),
            quota_bytes: self.quota_bytes,
        })
    }
}
