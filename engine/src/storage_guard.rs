//! Refuses model weights and runtime caches on network-backed filesystems.
//!
//! Weights and caches are read on the inference path, where network latency
//! and availability are not acceptable, and a network filesystem is neither a
//! storage tier nor a fallback. [`require_local`] checks one path;
//! [`LocalGuard`] reads the platform's filesystem facts once and checks many,
//! for a loader that opens a path per shard.
//!
//! Symlinks are resolved first, so a local shard that links into a network
//! mount is refused, and a path that does not exist yet is judged by its
//! nearest existing ancestor.
//!
//! On Linux the owner of a path comes from the kernel mount table
//! (`/proc/self/mountinfo`): the mount point that is the longest prefix of the
//! resolved path. Mounts stacked at one mount point (an automount placeholder
//! under the real mount) resolve to the topmost one, the mount that no other
//! mount at that point names as its parent. On macOS the owner is the
//! filesystem `statfs` reports for the resolved path. Other platforms are not
//! checked and every check returns `Ok`.
//!
//! A mount hidden under a later mount on one of its parent directories is not
//! visible to a path walk, so it is skipped and the visible mount judges the
//! path. An `autofs` placeholder may be a local disk that has not mounted yet:
//! resolving the path is what triggers the mount, so the table is read again
//! once after resolving, and a path that still sits on a placeholder is
//! refused.
//!
//! What a filesystem type cannot reveal is not caught: a union filesystem
//! such as mergerfs over network branches, and a FUSE mount whose source is
//! a local-looking path. The tokenizer directory and the Flash-Next manifest
//! hold small text files and are not checked.
//!
//! Every loader that opens model weight files calls this module: the server
//! start (all families), the inspection, the EXL3 and Flint checkpoint opens
//! (directory and each shard), the GGUF open, the Nemotron shard reader and
//! the read-only expert mapping. The Flash-Next and Qwen3.6 resident
//! executors open their shards the same way, but they are built only for
//! macOS with the MLX backend, so their guarded form is compiled and checked
//! on a Mac and has no Linux test.

use std::path::Path;

use crate::EngError;

#[cfg(all(test, target_os = "linux"))]
pub(crate) mod seam;

/// The filesystem facts of one check, read once. Every
/// [`LocalGuard::check`] after [`LocalGuard::new`] reuses them.
#[cfg(target_os = "linux")]
pub struct LocalGuard {
    mounts: Vec<mounts::Entry>,
}

#[cfg(target_os = "linux")]
impl LocalGuard {
    /// Reads the mount table. An unreadable table is an error: without it no
    /// path can be confirmed local.
    pub fn new() -> Result<Self, EngError> {
        let table = linux::read_table()
            .map_err(|error| unresolved(Path::new(linux::MOUNTINFO), "the mount table", &error))?;
        Ok(Self {
            mounts: mounts::parse_mountinfo(&table),
        })
    }

    /// Refuses `path` when it lives on a network-backed filesystem. `what`
    /// names the stored thing in the refusal ("model weights", "prefix
    /// cache").
    pub fn check(&self, path: &Path, what: &str) -> Result<(), EngError> {
        let resolved = resolve::resolve(path).map_err(|error| unresolved(path, what, &error))?;
        match mounts::owning_mount(&self.mounts, &resolved) {
            Some(mount) if mount.fstype == "autofs" => {
                // Resolving may have just triggered the real mount that
                // replaces the placeholder, which the table read before it
                // cannot show. Judge against a fresh table.
                Self::new()?.judge(&resolved, path, what)
            }
            _ => self.judge(&resolved, path, what),
        }
    }

    fn judge(&self, resolved: &Path, shown: &Path, what: &str) -> Result<(), EngError> {
        match mounts::owning_mount(&self.mounts, resolved) {
            Some(mount) => refuse(mount, resolved, shown, what),
            None => Ok(()),
        }
    }
}

/// `statfs` answers per path, so there is nothing to read ahead.
#[cfg(target_os = "macos")]
pub struct LocalGuard;

#[cfg(target_os = "macos")]
impl LocalGuard {
    pub fn new() -> Result<Self, EngError> {
        Ok(Self)
    }

    /// Refuses `path` when it lives on a network-backed filesystem. `what`
    /// names the stored thing in the refusal ("model weights", "prefix
    /// cache").
    pub fn check(&self, path: &Path, what: &str) -> Result<(), EngError> {
        let resolved = resolve::resolve(path).map_err(|error| unresolved(path, what, &error))?;
        let mount = macos::owner(&resolved).map_err(|error| unresolved(path, what, &error))?;
        refuse(&mount, &resolved, path, what)
    }
}

/// Not checked on this platform: every check returns `Ok`.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub struct LocalGuard;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl LocalGuard {
    pub fn new() -> Result<Self, EngError> {
        Ok(Self)
    }

    pub fn check(&self, path: &Path, what: &str) -> Result<(), EngError> {
        let _ = (path, what);
        Ok(())
    }
}

/// [`LocalGuard::check`] for a caller with one path.
pub fn require_local(path: &Path, what: &str) -> Result<(), EngError> {
    LocalGuard::new()?.check(path, what)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unresolved(path: &Path, what: &str, error: &std::io::Error) -> EngError {
    EngError::Checkpoint(format!(
        "cannot confirm that {what} at {} is on local storage: {error}",
        path.display()
    ))
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
use filesystem::{Mount, refuse};

/// What a filesystem is and which types the check refuses; shared by the
/// Linux and the macOS owner lookups, and pure so it is tested everywhere.
#[cfg(any(target_os = "linux", target_os = "macos", test))]
mod filesystem {
    use std::path::{Path, PathBuf};

    use crate::EngError;

    /// Filesystem types the check refuses. Network filesystems put the bytes
    /// behind a network. `autofs` is a placeholder with nothing mounted over
    /// it, so what backs the path is unknown. The macOS names (`afpfs`,
    /// `webdav`) sit beside the Linux ones. A FUSE mount whose source reads
    /// `host:path` is also refused (see [`is_refused`]), which covers remote
    /// FUSE filesystems missing from this list.
    const REFUSED_FILESYSTEMS: &[&str] = &[
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "smbfs",
        "afpfs",
        "webdav",
        "9p",
        "ceph",
        "glusterfs",
        "lustre",
        "afs",
        "ncpfs",
        "davfs",
        "gpfs",
        "beegfs",
        "wekafs",
        "panfs",
        "orangefs",
        "sshfs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.mount-s3",
        "fuse.gcsfuse",
        "fuse.glusterfs",
        "fuse.ceph",
        "fuse.ceph-fuse",
        "fuse.mountpoint-s3",
        "fuse.juicefs",
        "autofs",
    ];

    /// The mount that owns a path, reduced to what the check needs.
    #[derive(Debug, PartialEq, Eq)]
    pub(in crate::storage_guard) struct Mount {
        pub(in crate::storage_guard) point: PathBuf,
        pub(in crate::storage_guard) fstype: String,
        pub(in crate::storage_guard) source: String,
    }

    /// The refusal for `mount` as the owner of `resolved`, or `Ok` when it is
    /// local. `shown` is the path as the caller gave it.
    pub(in crate::storage_guard) fn refuse(
        mount: &Mount,
        resolved: &Path,
        shown: &Path,
        what: &str,
    ) -> Result<(), EngError> {
        if !is_refused(mount) {
            return Ok(());
        }
        let given = if shown == resolved {
            String::new()
        } else {
            format!(" (given as {})", shown.display())
        };
        Err(EngError::Checkpoint(format!(
            "{what} must be on local storage, but {}{given} is on a {} filesystem \
             (mount {}, source {})",
            resolved.display(),
            mount.fstype,
            mount.point.display(),
            mount.source,
        )))
    }

    pub(in crate::storage_guard) fn is_refused(mount: &Mount) -> bool {
        let fstype = mount.fstype.to_ascii_lowercase();
        if REFUSED_FILESYSTEMS.contains(&fstype.as_str()) {
            return true;
        }
        // A remote FUSE filesystem names its server as `host:path` and a local
        // one names a device or a program, never a colon-separated host.
        let fuse = fstype.starts_with("fuse") || fstype == "macfuse" || fstype == "osxfuse";
        fuse && !mount.source.starts_with('/') && mount.source.contains(':')
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn mount(fstype: &str, source: &str) -> Mount {
            Mount {
                point: "/m".into(),
                fstype: fstype.into(),
                source: source.into(),
            }
        }

        // Fails if any listed type is dropped, or a local type is added.
        #[test]
        fn every_listed_type_is_refused_and_local_types_pass() {
            // Spelled out, not read from the const, so dropping an entry fails.
            for fstype in [
                "nfs",
                "nfs4",
                "cifs",
                "smb3",
                "smbfs",
                "afpfs",
                "webdav",
                "9p",
                "ceph",
                "glusterfs",
                "lustre",
                "afs",
                "ncpfs",
                "davfs",
                "gpfs",
                "beegfs",
                "wekafs",
                "panfs",
                "orangefs",
                "sshfs",
                "fuse.sshfs",
                "fuse.rclone",
                "fuse.s3fs",
                "fuse.mount-s3",
                "fuse.gcsfuse",
                "fuse.glusterfs",
                "fuse.ceph",
                "fuse.ceph-fuse",
                "fuse.mountpoint-s3",
                "fuse.juicefs",
                "autofs",
                "NFS4",
            ] {
                assert!(is_refused(&mount(fstype, "/dev/x")), "{fstype}");
            }
            for fstype in [
                "ext4",
                "btrfs",
                "xfs",
                "zfs",
                "tmpfs",
                "overlay",
                "apfs",
                "hfs",
                "devfs",
                "fuseblk",
                "f2fs",
                "fuse.cephfs",
            ] {
                assert!(!is_refused(&mount(fstype, "/dev/x")), "{fstype}");
            }
        }

        // Fails if the host:path rule is removed, applies to local FUSE, or
        // misses the macOS FUSE type names.
        #[test]
        fn a_fuse_mount_with_a_host_source_is_network_and_a_device_source_is_not() {
            assert!(is_refused(&mount("fuse.unknownfs", "user@box:/data")));
            assert!(is_refused(&mount("macfuse", "user@box:/data")));
            assert!(is_refused(&mount("osxfuse", "user@box:/data")));
            assert!(!is_refused(&mount("macfuse", "/dev/disk4")));
            assert!(!is_refused(&mount("fuse.unknownfs", "/dev/sdb1")));
            assert!(!is_refused(&mount("fuse.gvfsd-fuse", "gvfsd-fuse")));
            assert!(!is_refused(&mount("ext4", "a:b")));
        }

        // Fails if the message drops the resolved path, the filesystem type or
        // `what`, repeats itself, or hides the path the caller gave.
        #[test]
        fn the_refusal_names_the_resolved_path_once_and_the_given_one() {
            let nfs = Mount {
                point: "/srv/shared".into(),
                fstype: "nfs4".into(),
                source: "nas:/shared".into(),
            };
            let resolved = Path::new("/srv/shared/cache");
            let error = refuse(&nfs, resolved, resolved, "prefix cache").unwrap_err();
            let text = error.describe();
            assert!(text.contains("prefix cache"), "{text}");
            assert!(text.contains("nfs4"), "{text}");
            assert!(text.contains("nas:/shared"), "{text}");
            assert_eq!(text.matches("/srv/shared/cache").count(), 1, "{text}");
            assert_eq!(text.matches("prefix cache").count(), 1, "{text}");
            assert!(!text.contains("given as"), "{text}");

            let given = Path::new("link/cache");
            let error = refuse(&nfs, resolved, given, "prefix cache").unwrap_err();
            assert!(error.describe().contains("(given as link/cache)"));
        }
    }
}

/// The absolute path with its nearest existing prefix canonicalized (so
/// symlinks resolve) and the not-yet-created rest appended lexically, where
/// `..` can only cancel a missing directory.
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod resolve {
    use std::{
        io,
        path::{Component, Path, PathBuf},
    };

    /// A prefix that exists but cannot be canonicalized, such as a dangling
    /// symlink, is an error naming it: its target is unknown, and creating
    /// through it could land anywhere.
    pub(super) fn resolve(path: &Path) -> io::Result<PathBuf> {
        let absolute = std::env::current_dir()?.join(path);
        let parts: Vec<Component> = absolute.components().collect();
        for split in (1..=parts.len()).rev() {
            let existing: PathBuf = parts[..split].iter().collect();
            if std::fs::symlink_metadata(&existing).is_err() {
                continue;
            }
            let mut resolved = std::fs::canonicalize(&existing).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "{} exists but does not resolve (a dangling symlink?): {error}",
                        existing.display()
                    ),
                )
            })?;
            for part in &parts[split..] {
                match part {
                    Component::ParentDir => {
                        resolved.pop();
                    }
                    Component::CurDir => {}
                    other => resolved.push(other),
                }
            }
            return Ok(resolved);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no existing ancestor",
        ))
    }
}

/// The macOS owner lookup: `statfs` of the resolved path, or of its nearest
/// existing ancestor when the path does not exist yet.
#[cfg(target_os = "macos")]
mod macos {
    use std::{ffi::c_char, io, path::Path};

    use super::Mount;

    /// A path that does not exist yet lives on the filesystem of its nearest
    /// existing ancestor, which `statfs` can answer for.
    pub(super) fn owner(resolved: &Path) -> io::Result<Mount> {
        for candidate in resolved.ancestors() {
            match rustix::fs::statfs(candidate) {
                Ok(stat) => {
                    return Ok(Mount {
                        point: text(&stat.f_mntonname).into(),
                        fstype: text(&stat.f_fstypename),
                        source: text(&stat.f_mntfromname),
                    });
                }
                Err(rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(io::Error::from(io::ErrorKind::NotFound))
    }

    /// A NUL-terminated C char array as text.
    fn text(field: &[c_char]) -> String {
        let bytes: Vec<u8> = field
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn text_stops_at_the_first_nul() {
            let field: [c_char; 8] = [
                b'n' as c_char,
                b'f' as c_char,
                b's' as c_char,
                0,
                7,
                7,
                0,
                0,
            ];
            assert_eq!(text(&field), "nfs");
        }

        // Fails if the statfs lookup errors on, or refuses, the local root.
        #[test]
        fn the_root_filesystem_is_local() {
            let mount = owner(Path::new("/")).unwrap();
            assert!(!super::super::filesystem::is_refused(&mount), "{mount:?}");
            assert!(super::super::require_local(Path::new("/"), "model weights").is_ok());
        }

        // Fails if a real macOS autofs placeholder passes. macOS mounts one
        // at this path (`map auto_home`); skipped with a note on a Mac
        // without it. This Mac has no NFS or SMB volume to test with.
        #[test]
        fn an_autofs_placeholder_is_refused() {
            let point = Path::new("/System/Volumes/Data/home");
            let Ok(mount) = owner(point) else {
                eprintln!("skipped: {} cannot be stat'ed", point.display());
                return;
            };
            eprintln!("statfs reports {mount:?}");
            if mount.fstype != "autofs" {
                eprintln!("skipped: {} is not an autofs mount here", point.display());
                return;
            }
            let child = point.join("storage-guard-test-absent");
            let error = super::super::require_local(&child, "model weights").unwrap_err();
            assert!(error.describe().contains("autofs"), "{}", error.describe());
        }

        // Fails if the check skips symlink resolution or missing children.
        #[test]
        fn a_missing_child_of_a_local_directory_is_local() {
            let dir = tempfile::tempdir().unwrap();
            let missing = dir.path().join("cache").join("new");
            assert!(super::super::require_local(&missing, "prefix cache").is_ok());
        }
    }
}

/// Where the Linux mount table comes from.
#[cfg(target_os = "linux")]
mod linux {
    use std::io;

    pub(super) const MOUNTINFO: &str = "/proc/self/mountinfo";

    /// The kernel table, or the fixture a test installed for this thread.
    pub(super) fn read_table() -> io::Result<Vec<u8>> {
        #[cfg(test)]
        if let Some(table) = super::seam::installed() {
            return Ok(table);
        }
        std::fs::read(MOUNTINFO)
    }
}

/// The Linux owner lookup: the kernel mount table, as bytes.
#[cfg(any(target_os = "linux", test))]
mod mounts {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt, path::Path};

    use super::Mount;

    /// One line of the table: the mount and its place in the stack.
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Entry {
        id: u32,
        parent: u32,
        mount: Mount,
    }

    /// The mount with the most path components that is a prefix of `path`.
    /// Among mounts at that one point, the topmost wins: the one no other
    /// mount there lists as its parent, else the later table line.
    pub(super) fn owning_mount<'a>(entries: &'a [Entry], path: &Path) -> Option<&'a Mount> {
        let prefixes = || {
            entries
                .iter()
                .filter(|entry| path.starts_with(&entry.mount.point))
                .filter(|entry| !is_hidden(entries, entry))
        };
        let depth = prefixes()
            .map(|entry| entry.mount.point.components().count())
            .max()?;
        let stacked: Vec<&Entry> = prefixes()
            .filter(|entry| entry.mount.point.components().count() == depth)
            .collect();
        let is_covered = |entry: &Entry| {
            stacked
                .iter()
                .any(|other| other.parent == entry.id && other.id != entry.id)
        };
        stacked
            .iter()
            .rev()
            .find(|entry| !is_covered(entry))
            .or(stacked.last())
            .map(|entry| &entry.mount)
    }

    /// Whether a later mount (a higher id; the kernel numbers mounts in
    /// creation order) on a parent directory of `entry` covers it. A path walk
    /// then reaches the covering mount and never the covered one.
    fn is_hidden(entries: &[Entry], entry: &Entry) -> bool {
        entries.iter().any(|later| {
            later.id > entry.id
                && later.mount.point != entry.mount.point
                && entry.mount.point.starts_with(&later.mount.point)
        })
    }

    /// Parses `mountinfo` as bytes, because a mount point may hold any byte
    /// but NUL. A line reads
    /// `id parent major:minor root mount-point options [optional...] - fstype source super-options`;
    /// the lone `-` ends the optional fields, whose count varies. Lines that do
    /// not fit are skipped.
    pub(super) fn parse_mountinfo(table: &[u8]) -> Vec<Entry> {
        table
            .split(|byte| *byte == b'\n')
            .filter_map(parse_line)
            .collect()
    }

    fn parse_line(line: &[u8]) -> Option<Entry> {
        let separator = line.windows(3).position(|window| window == b" - ")?;
        let mut head = line[..separator].split(|byte| *byte == b' ');
        let id = number(head.next()?)?;
        let parent = number(head.next()?)?;
        let point = head.nth(2)?;
        let mut tail = line[separator + 3..].split(|byte| *byte == b' ');
        let fstype = tail.next()?;
        let source = tail.next().unwrap_or_default();
        Some(Entry {
            id,
            parent,
            mount: Mount {
                point: OsString::from_vec(unescape(point)).into(),
                fstype: String::from_utf8_lossy(fstype).into_owned(),
                source: String::from_utf8_lossy(&unescape(source)).into_owned(),
            },
        })
    }

    fn number(field: &[u8]) -> Option<u32> {
        std::str::from_utf8(field).ok()?.parse().ok()
    }

    /// Decodes the kernel's `\NNN` octal escapes (space is `\040`, tab `\011`,
    /// newline `\012`, backslash `\134`). A backslash not followed by three octal
    /// digits stays literal.
    fn unescape(field: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(field.len());
        let mut index = 0;
        while index < field.len() {
            let octal = field.get(index + 1..index + 4).filter(|digits| {
                field[index] == b'\\' && digits.iter().all(|digit| (b'0'..=b'7').contains(digit))
            });
            match octal {
                Some(digits) => {
                    let value = digits
                        .iter()
                        .fold(0u32, |acc, digit| acc * 8 + u32::from(digit - b'0'));
                    out.push((value & 0xff) as u8);
                    index += 4;
                }
                None => {
                    out.push(field[index]);
                    index += 1;
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::EngError;
        use crate::storage_guard::refuse;

        /// The shape of a host with an automounted share: an automount
        /// placeholder and the real nfs4 mount stacked at one mount point.
        const STACKED: &str = "\
    24 1 0:21 / / rw,relatime shared:1 - btrfs /dev/nvme0n1p2 rw,ssd\n\
    68 24 0:48 / /srv/shared rw,relatime shared:230 - autofs systemd-1 rw,fd=71,direct\n\
    77 68 0:91 / /srv/shared rw,noatime shared:1630 - nfs4 nas:/shared rw,vers=4.2\n\
    80 24 8:1 / /data rw,relatime shared:5 - ext4 /dev/nvme1n1 rw\n\
    ";

        fn parse(text: &str) -> Vec<Entry> {
            parse_mountinfo(text.as_bytes())
        }

        fn owner<'a>(mounts: &'a [Entry], path: &str) -> &'a Mount {
            owning_mount(mounts, Path::new(path)).expect("some mount owns the path")
        }

        fn refusal(mounts: &[Entry], path: &str, what: &str) -> Result<(), EngError> {
            let path = Path::new(path);
            match owning_mount(mounts, path) {
                Some(mount) => refuse(mount, path, path, what),
                None => Ok(()),
            }
        }

        // Fails if the parser reads the mount point from the wrong column.
        #[test]
        fn the_parser_reads_ids_point_type_and_source() {
            let mounts = parse(STACKED);
            assert_eq!(mounts.len(), 4);
            assert_eq!((mounts[2].id, mounts[2].parent), (77, 68));
            assert_eq!(mounts[2].mount.point, Path::new("/srv/shared"));
            assert_eq!(mounts[2].mount.fstype, "nfs4");
            assert_eq!(mounts[2].mount.source, "nas:/shared");
        }

        // Fails if the longest prefix is not chosen, or the stacked placeholder
        // wins over the mount above it.
        #[test]
        fn stacked_mounts_resolve_to_the_topmost_and_the_longest_prefix() {
            let mounts = parse(STACKED);
            assert_eq!(owner(&mounts, "/srv/shared/models/a").fstype, "nfs4");
            assert_eq!(owner(&mounts, "/srv/other-local").fstype, "btrfs");
            assert_eq!(owner(&mounts, "/data/x").fstype, "ext4");
        }

        // Fails if stacking follows line order instead of the parent ids: here
        // the placeholder is listed last but the nfs4 mount sits above it.
        #[test]
        fn the_topmost_mount_is_found_by_parent_id_not_line_order() {
            let reversed = "\
    1 0 0:1 / / rw - ext4 /dev/sda1 rw\n\
    77 68 0:91 / /srv/shared rw - nfs4 nas:/shared rw\n\
    68 1 0:48 / /srv/shared rw - autofs systemd-1 rw\n\
    ";
            let mounts = parse(reversed);
            assert_eq!(owner(&mounts, "/srv/shared/models").fstype, "nfs4");
        }

        // Fails if an autofs placeholder with nothing mounted over it passes.
        #[test]
        fn an_autofs_owner_alone_is_refused() {
            let text = "\
    1 0 0:1 / / rw - ext4 /dev/sda1 rw\n\
    68 1 0:48 / /srv/shared rw - autofs systemd-1 rw\n\
    ";
            let mounts = parse(text);
            let error = refusal(&mounts, "/srv/shared/models", "model weights").unwrap_err();
            assert!(error.describe().contains("autofs"), "{}", error.describe());
        }

        // Fails if a mount hidden under a later mount on its parent directory
        // still owns paths, which would pass an nfs4 mount laid over a local
        // disk's mount point.
        #[test]
        fn a_mount_hidden_under_a_later_overmount_does_not_own_the_path() {
            let hidden = "\
    1 0 0:1 / / rw - ext4 /dev/sda1 rw\n\
    10 1 8:2 / /srv/data/cache rw - ext4 /dev/sdb1 rw\n\
    20 1 0:9 / /srv/data rw - nfs4 nas:/data rw\n\
    ";
            let mounts = parse(hidden);
            assert_eq!(owner(&mounts, "/srv/data/cache/w.gguf").fstype, "nfs4");
            // The reverse order is an ordinary nested mount: the child owns.
            let nested = "\
    1 0 0:1 / / rw - ext4 /dev/sda1 rw\n\
    10 1 0:9 / /srv/data rw - nfs4 nas:/data rw\n\
    20 10 8:2 / /srv/data/cache rw - ext4 /dev/sdb1 rw\n\
    ";
            let mounts = parse(nested);
            assert_eq!(owner(&mounts, "/srv/data/cache/w.gguf").fstype, "ext4");
        }

        // Fails if prefix matching is textual: /srv/shared2 is not below
        // /srv/shared.
        #[test]
        fn a_sibling_with_a_shared_name_prefix_is_not_below_the_mount() {
            let mounts = parse(STACKED);
            assert_eq!(owner(&mounts, "/srv/shared2/models").fstype, "btrfs");
        }

        // Fails if \040 stays literal or the optional-field count is assumed.
        #[test]
        fn escaped_spaces_and_variable_optional_fields_parse() {
            let text = "\
    30 24 0:40 / /mnt/my\\040models rw shared:1 master:2 - nfs 10.0.0.1:/m rw\n\
    31 24 0:41 / /mnt/plain rw - ext4 /dev/sdb rw\n\
    ";
            let mounts = parse(text);
            assert_eq!(mounts[0].mount.point, Path::new("/mnt/my models"));
            assert_eq!(mounts[0].mount.fstype, "nfs");
            assert_eq!(mounts[1].mount.point, Path::new("/mnt/plain"));
            assert!(refusal(&mounts, "/mnt/my models/q.gguf", "model weights").is_err());
            assert!(refusal(&mounts, "/mnt/plain/q.gguf", "model weights").is_ok());
        }

        // Fails if the table is decoded as UTF-8 text: a lossy decode turns the
        // 0xff byte into U+FFFD and the mount no longer matches its own path.
        #[test]
        fn a_mount_point_with_a_non_utf8_byte_still_matches() {
            use std::os::unix::ffi::OsStrExt;
            let mut table = b"1 0 0:1 / / rw - ext4 /dev/sda1 rw\n".to_vec();
            table.extend_from_slice(b"2 1 0:2 / /mnt/\xff rw - nfs4 nas:/m rw\n");
            let mounts = parse_mountinfo(&table);
            let path = Path::new(std::ffi::OsStr::from_bytes(b"/mnt/\xff/models"));
            let mount = owning_mount(&mounts, path).expect("the nfs mount owns the path");
            assert_eq!(mount.fstype, "nfs4");
            assert!(refuse(mount, path, path, "model weights").is_err());
        }

        #[test]
        fn octal_escapes_decode_and_a_bare_backslash_stays() {
            assert_eq!(unescape(b"a\\040b\\134c"), b"a b\\c");
            assert_eq!(unescape(b"a\\04"), b"a\\04");
            assert_eq!(unescape(b"a\\9zz"), b"a\\9zz");
        }

        #[test]
        fn malformed_lines_are_skipped() {
            assert!(parse("garbage\n\n1 2 3\n").is_empty());
        }

        // Fails if a bind mount of a network directory is judged by its parent
        // filesystem, or a local bind inside an nfs tree by the nfs parent.
        #[test]
        fn paths_below_a_bind_mount_follow_the_bind_mounts_filesystem() {
            let text = "\
    24 1 0:21 / / rw - ext4 /dev/sda1 rw\n\
    40 24 0:50 /export /srv/models rw - nfs4 nas:/export rw\n\
    41 40 8:2 /fast /srv/models/fast rw - ext4 /dev/nvme0n1 rw\n\
    ";
            let mounts = parse(text);
            assert!(refusal(&mounts, "/srv/models/q/w.gguf", "model weights").is_err());
            assert!(refusal(&mounts, "/srv/models/fast/w.gguf", "model weights").is_ok());
            assert!(refusal(&mounts, "/srv/other", "model weights").is_ok());
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod live_tests {
    use super::*;

    // Fails if the real check stops resolving the path or rejects local disks.
    #[test]
    fn the_root_and_a_missing_child_of_a_local_directory_pass() {
        assert!(require_local(Path::new("/"), "model weights").is_ok());
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("cache").join("..").join("new").join("x");
        assert!(require_local(&missing, "prefix cache").is_ok());
        // A relative path is judged where it lands, whatever the working
        // directory (a checkout on a network share must not fail this test).
        let relative = Path::new("relative/missing");
        let absolute = std::env::current_dir().unwrap().join(relative);
        assert_eq!(
            require_local(relative, "prefix cache").is_ok(),
            require_local(&absolute, "prefix cache").is_ok()
        );
    }

    // Fails if a dangling symlink passes, or its refusal does not name it.
    #[test]
    fn a_dangling_symlink_is_refused_and_named() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("cache");
        std::os::unix::fs::symlink(dir.path().join("gone"), &link).unwrap();
        let error = require_local(&link.join("x"), "prefix cache").unwrap_err();
        let text = error.describe();
        assert!(text.contains(&link.display().to_string()), "{text}");
        assert!(text.contains("dangling symlink"), "{text}");
    }

    // Fails if a `..` after a missing directory is kept instead of cancelling
    // it, which would let `<nfs>/missing/../..` be judged as a deeper path.
    #[test]
    fn dot_dot_cancels_a_missing_directory_and_symlinks_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let base = std::fs::canonicalize(dir.path()).unwrap();
        let resolved = resolve::resolve(&dir.path().join("a").join("..").join("b")).unwrap();
        assert_eq!(resolved, base.join("b"));
        std::fs::create_dir(base.join("real")).unwrap();
        std::os::unix::fs::symlink(base.join("real"), base.join("link")).unwrap();
        let through = resolve::resolve(&base.join("link").join("new")).unwrap();
        assert_eq!(through, base.join("real").join("new"));
    }

    // Fails if a symlink into a network mount is judged by where it lives
    // instead of where it points (the shard layout of a downloaded snapshot).
    #[test]
    fn a_local_symlink_into_a_network_mount_is_refused() {
        let local = tempfile::tempdir().unwrap();
        let share = tempfile::tempdir().unwrap();
        std::fs::write(share.path().join("shard.safetensors"), b"x").unwrap();
        let link = local.path().join("shard.safetensors");
        std::os::unix::fs::symlink(share.path().join("shard.safetensors"), &link).unwrap();
        seam::with_nfs_at(share.path(), || {
            let error = require_local(&link, "model weights").unwrap_err();
            let text = error.describe();
            assert!(text.contains("nfs4"), "{text}");
            assert!(text.contains("(given as"), "{text}");
            assert!(require_local(local.path(), "model weights").is_ok());
        });
    }

    /// A fixture table with `dir` as an autofs placeholder, plus `above` (a
    /// mount line stacked on the placeholder) when given.
    fn automount_table(dir: &Path, above: &str) -> String {
        let point = std::fs::canonicalize(dir).unwrap().display().to_string();
        let mut table = format!(
            "1 0 0:1 / / rw - ext4 /dev/fixture rw\n2 1 0:2 / {point} rw - autofs systemd-1 rw\n"
        );
        if !above.is_empty() {
            table.push_str(&format!("3 2 0:3 / {point} rw - {above}\n"));
        }
        table
    }

    // Fails if the table is read only before the path is resolved: resolving
    // is what mounts an idle `x-systemd.automount` disk, so the first table
    // still shows the placeholder.
    #[test]
    fn an_idle_automount_of_a_local_disk_is_judged_after_it_mounts() {
        let dir = tempfile::tempdir().unwrap();
        let idle = automount_table(dir.path(), "");
        let mounted = automount_table(dir.path(), "ext4 /dev/fixture2 rw");
        let path = dir.path().join("models");
        seam::with_tables(&[idle.as_bytes(), mounted.as_bytes()], || {
            assert!(require_local(&path, "model weights").is_ok());
        });
    }

    // Fails if a real network automount stops being refused, or the refusal
    // reports the placeholder instead of the filesystem behind it.
    #[test]
    fn a_network_automount_is_refused_with_its_real_type() {
        let dir = tempfile::tempdir().unwrap();
        let idle = automount_table(dir.path(), "");
        let mounted = automount_table(dir.path(), "nfs4 nas:/fixture rw");
        let path = dir.path().join("models");
        seam::with_tables(&[idle.as_bytes(), mounted.as_bytes()], || {
            let text = require_local(&path, "model weights")
                .unwrap_err()
                .describe();
            assert!(text.contains("nfs4"), "{text}");
        });
    }

    // Fails if a placeholder that never mounts anything passes.
    #[test]
    fn an_automount_that_never_mounts_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let idle = automount_table(dir.path(), "");
        seam::with_table(idle.as_bytes(), || {
            let text = require_local(dir.path(), "model weights")
                .unwrap_err()
                .describe();
            assert!(text.contains("autofs"), "{text}");
        });
    }

    // Fails if require_local does not consult the live mount table. Skipped
    // with a note on a host with no network mount.
    #[test]
    fn a_path_under_a_live_network_mount_is_refused() {
        let Some(path) = seam::live_network_path() else {
            eprintln!("skipped: no network mount on this host");
            return;
        };
        let error = require_local(&path, "model weights").unwrap_err();
        assert!(
            error.describe().contains("local storage"),
            "{}",
            error.describe()
        );
    }
}
