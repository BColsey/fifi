//! Refuses model weights and runtime caches on network-backed filesystems.
//!
//! Weights and caches are read on the inference path, where network latency
//! and availability are not acceptable, and a network filesystem is neither a
//! storage tier nor a fallback. [`require_local`] is the one check, called
//! where the engine opens a model directory or weight file.
//!
//! The check finds the filesystem that owns a path from the kernel mount
//! table (`/proc/self/mountinfo`): the mount point that is the longest prefix
//! of the resolved path, the later table line winning a tie (an automount
//! placeholder and the real mount stack at one mount point, and the real one
//! is listed last). Only Linux is checked; on other platforms the function
//! returns `Ok` without looking, because no std-only way to read the
//! filesystem type exists there and the engine adds no unsafe code for it.

use std::path::Path;

use crate::EngError;

/// Refuses `path` when it lives on a network-backed filesystem. `what` names
/// the stored thing in the refusal ("model weights", "prefix cache").
///
/// A path that does not exist yet is judged by its nearest existing ancestor,
/// so a cache directory about to be created is checked too.
#[cfg(target_os = "linux")]
pub fn require_local(path: &Path, what: &str) -> Result<(), EngError> {
    let resolved = linux::resolve(path).map_err(|error| unresolved(path, what, &error))?;
    let table = std::fs::read(linux::MOUNTINFO)
        .map_err(|error| unresolved(Path::new(linux::MOUNTINFO), what, &error))?;
    let mounts = mounts::parse_mountinfo(&String::from_utf8_lossy(&table));
    mounts::refuse_network(&mounts, &resolved, path, what)
}

/// Not checked on this platform: always `Ok`. See the module documentation.
#[cfg(not(target_os = "linux"))]
pub fn require_local(path: &Path, what: &str) -> Result<(), EngError> {
    let _ = (path, what);
    Ok(())
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        io,
        path::{Component, Path, PathBuf},
    };

    pub(super) const MOUNTINFO: &str = "/proc/self/mountinfo";

    /// The absolute path with its nearest existing prefix canonicalized (so
    /// symlinks resolve) and the not-yet-created rest appended lexically,
    /// where `..` can only cancel a missing directory. A prefix that exists
    /// but cannot be canonicalized, such as a dangling symlink, is an error:
    /// its target is unknown, and creating through it could land anywhere.
    pub(super) fn resolve(path: &Path) -> io::Result<PathBuf> {
        let absolute = std::env::current_dir()?.join(path);
        let parts: Vec<Component> = absolute.components().collect();
        for split in (1..=parts.len()).rev() {
            let existing: PathBuf = parts[..split].iter().collect();
            if std::fs::symlink_metadata(&existing).is_err() {
                continue;
            }
            let mut resolved = std::fs::canonicalize(&existing)?;
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

#[cfg(target_os = "linux")]
fn unresolved(path: &Path, what: &str, error: &std::io::Error) -> EngError {
    EngError::Checkpoint(format!(
        "cannot confirm that {what} at {} is on local storage: {error}",
        path.display()
    ))
}

/// The mount-table side of the check: pure text and path logic, so it is
/// tested on fixtures on every platform.
#[cfg(any(target_os = "linux", test))]
mod mounts {
    use std::path::{Path, PathBuf};

    use crate::EngError;

    /// Filesystem types that put the bytes behind a network. A `fuse` mount whose
    /// source reads `host:path` is also refused (see [`is_network`]), which covers
    /// remote FUSE filesystems missing from this list.
    const NETWORK_FILESYSTEMS: &[&str] = &[
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "smbfs",
        "9p",
        "ceph",
        "glusterfs",
        "lustre",
        "afs",
        "ncpfs",
        "davfs",
        "sshfs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.gcsfuse",
        "fuse.glusterfs",
        "fuse.cephfs",
    ];

    /// One entry of the kernel mount table, reduced to what the check needs.
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Mount {
        point: PathBuf,
        fstype: String,
        source: String,
    }

    /// The refusal for the mount that owns `resolved`, if it is network-backed.
    pub(super) fn refuse_network(
        mounts: &[Mount],
        resolved: &Path,
        shown: &Path,
        what: &str,
    ) -> Result<(), EngError> {
        let Some(mount) = owning_mount(mounts, resolved).filter(|mount| is_network(mount)) else {
            return Ok(());
        };
        Err(EngError::Checkpoint(format!(
            "{what} must be on local storage, but {} is on a {} filesystem (mount {}, source {}); \
             network-backed storage is not accepted for {what}",
            shown.display(),
            mount.fstype,
            mount.point.display(),
            mount.source,
        )))
    }

    fn is_network(mount: &Mount) -> bool {
        let fstype = mount.fstype.to_ascii_lowercase();
        if NETWORK_FILESYSTEMS.contains(&fstype.as_str()) {
            return true;
        }
        // A remote FUSE filesystem names its server as `host:path` and a local
        // one names a device or a program, never a colon-separated host.
        fstype.starts_with("fuse") && !mount.source.starts_with('/') && mount.source.contains(':')
    }

    /// The mount with the most path components that is a prefix of `path`; among
    /// equally long mount points the later table line wins.
    fn owning_mount<'a>(mounts: &'a [Mount], path: &Path) -> Option<&'a Mount> {
        let mut best: Option<(usize, &Mount)> = None;
        for mount in mounts.iter().filter(|mount| path.starts_with(&mount.point)) {
            let depth = mount.point.components().count();
            if best.is_none_or(|(deepest, _)| depth >= deepest) {
                best = Some((depth, mount));
            }
        }
        best.map(|(_, mount)| mount)
    }

    /// Parses `mountinfo`. A line reads
    /// `id parent major:minor root mount-point options [optional...] - fstype source super-options`;
    /// the lone `-` ends the optional fields, whose count varies. Lines that do
    /// not fit are skipped.
    pub(super) fn parse_mountinfo(text: &str) -> Vec<Mount> {
        text.lines().filter_map(parse_line).collect()
    }

    fn parse_line(line: &str) -> Option<Mount> {
        let (head, tail) = line.split_once(" - ")?;
        let point = head.split(' ').nth(4)?;
        let mut tail = tail.split(' ');
        let fstype = tail.next()?;
        let source = tail.next().unwrap_or_default();
        Some(Mount {
            point: unescape(point),
            fstype: fstype.to_string(),
            source: String::from_utf8_lossy(&unescape_bytes(source)).into_owned(),
        })
    }

    fn unescape(field: &str) -> PathBuf {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(unescape_bytes(field)).into()
    }

    /// Decodes the kernel's `\NNN` octal escapes (space is `\040`, tab `\011`,
    /// newline `\012`, backslash `\134`). A backslash not followed by three octal
    /// digits stays literal.
    fn unescape_bytes(field: &str) -> Vec<u8> {
        let bytes = field.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            let octal = bytes.get(index + 1..index + 4).filter(|digits| {
                bytes[index] == b'\\' && digits.iter().all(|digit| (b'0'..=b'7').contains(digit))
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
                    out.push(bytes[index]);
                    index += 1;
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The shape of a host with an automounted share: an automount placeholder and the real nfs4
        /// mount stacked at one mount point, listed in that order.
        const STACKED: &str = "\
    24 1 0:21 / / rw,relatime shared:1 - btrfs /dev/nvme0n1p2 rw,ssd\n\
    68 24 0:48 / /srv/shared rw,relatime shared:230 - autofs systemd-1 rw,fd=71,direct\n\
    77 68 0:91 / /srv/shared rw,noatime shared:1630 - nfs4 nas:/shared rw,vers=4.2\n\
    80 24 8:1 / /data rw,relatime shared:5 - ext4 /dev/nvme1n1 rw\n\
    ";

        fn owner<'a>(mounts: &'a [Mount], path: &str) -> &'a Mount {
            owning_mount(mounts, Path::new(path)).expect("some mount owns the path")
        }

        fn refusal(mounts: &[Mount], path: &str, what: &str) -> Result<(), EngError> {
            refuse_network(mounts, Path::new(path), Path::new(path), what)
        }

        // Fails if the parser reads the mount point from the wrong column.
        #[test]
        fn the_parser_reads_point_type_and_source() {
            let mounts = parse_mountinfo(STACKED);
            assert_eq!(mounts.len(), 4);
            assert_eq!(mounts[2].point, Path::new("/srv/shared"));
            assert_eq!(mounts[2].fstype, "nfs4");
            assert_eq!(mounts[2].source, "nas:/shared");
        }

        // Fails if a tie keeps the earlier line (the autofs placeholder) instead
        // of the later nfs4 mount, or if the longest prefix is not chosen.
        #[test]
        fn stacked_mounts_resolve_to_the_later_line_and_the_longest_prefix() {
            let mounts = parse_mountinfo(STACKED);
            assert_eq!(owner(&mounts, "/srv/shared/models/a").fstype, "nfs4");
            assert_eq!(owner(&mounts, "/srv/other-local").fstype, "btrfs");
            assert_eq!(owner(&mounts, "/data/x").fstype, "ext4");
        }

        // Fails if prefix matching is textual: /srv/shared2 is not below
        // /srv/shared.
        #[test]
        fn a_sibling_with_a_shared_name_prefix_is_not_below_the_mount() {
            let mounts = parse_mountinfo(STACKED);
            assert_eq!(owner(&mounts, "/srv/shared2/models").fstype, "btrfs");
        }

        // Fails if \040 stays literal or the optional-field count is assumed.
        #[test]
        fn escaped_spaces_and_variable_optional_fields_parse() {
            let text = "\
    30 24 0:40 / /mnt/my\\040models rw shared:1 master:2 - nfs 10.0.0.1:/m rw\n\
    31 24 0:41 / /mnt/plain rw - ext4 /dev/sdb rw\n\
    ";
            let mounts = parse_mountinfo(text);
            assert_eq!(mounts[0].point, Path::new("/mnt/my models"));
            assert_eq!(mounts[0].fstype, "nfs");
            assert_eq!(mounts[1].point, Path::new("/mnt/plain"));
            assert!(refusal(&mounts, "/mnt/my models/q.gguf", "model weights").is_err());
            assert!(refusal(&mounts, "/mnt/plain/q.gguf", "model weights").is_ok());
        }

        #[test]
        fn octal_escapes_decode_and_a_bare_backslash_stays() {
            assert_eq!(unescape_bytes("a\\040b\\134c"), b"a b\\c");
            assert_eq!(unescape_bytes("a\\04"), b"a\\04");
            assert_eq!(unescape_bytes("a\\9zz"), b"a\\9zz");
        }

        #[test]
        fn malformed_lines_are_skipped() {
            assert!(parse_mountinfo("garbage\n\n1 2 3\n").is_empty());
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
            let mounts = parse_mountinfo(text);
            assert!(refusal(&mounts, "/srv/models/q/w.gguf", "model weights").is_err());
            assert!(refusal(&mounts, "/srv/models/fast/w.gguf", "model weights").is_ok());
            assert!(refusal(&mounts, "/srv/other", "model weights").is_ok());
        }

        // Fails if the message drops the path, the filesystem type or `what`.
        #[test]
        fn the_refusal_names_the_path_the_type_and_the_stored_thing() {
            let mounts = parse_mountinfo(STACKED);
            let error = refusal(&mounts, "/srv/shared/cache", "prefix cache").unwrap_err();
            let EngError::Checkpoint(text) = error else {
                panic!("a refusal is a checkpoint error");
            };
            assert!(text.contains("/srv/shared/cache"), "{text}");
            assert!(text.contains("nfs4"), "{text}");
            assert!(text.contains("prefix cache"), "{text}");
            assert!(text.contains("nas:/shared"), "{text}");
        }

        // Fails if any listed type is dropped, or a local type is added.
        #[test]
        fn every_listed_network_type_is_refused_and_local_types_pass() {
            for fstype in NETWORK_FILESYSTEMS {
                let mount = Mount {
                    point: "/m".into(),
                    fstype: (*fstype).into(),
                    source: "/dev/x".into(),
                };
                assert!(is_network(&mount), "{fstype}");
            }
            for fstype in [
                "ext4", "btrfs", "xfs", "zfs", "tmpfs", "overlay", "apfs", "fuseblk", "f2fs",
            ] {
                let mount = Mount {
                    point: "/m".into(),
                    fstype: fstype.into(),
                    source: "/dev/x".into(),
                };
                assert!(!is_network(&mount), "{fstype}");
            }
        }

        // Fails if the host:path rule is removed, or applies to local FUSE.
        #[test]
        fn a_fuse_mount_with_a_host_source_is_network_and_a_device_source_is_not() {
            let fuse = |fstype: &str, source: &str| Mount {
                point: "/m".into(),
                fstype: fstype.into(),
                source: source.into(),
            };
            assert!(is_network(&fuse("fuse.unknownfs", "user@box:/data")));
            assert!(!is_network(&fuse("fuse.unknownfs", "/dev/sdb1")));
            assert!(!is_network(&fuse("fuse.gvfsd-fuse", "gvfsd-fuse")));
            assert!(!is_network(&fuse("ext4", "a:b")));
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod live_tests {
    use super::*;

    // Fails if a `..` after a missing directory is kept instead of cancelling
    // it, which would let `<nfs>/missing/../..` be judged as a deeper path.
    #[test]
    fn dot_dot_cancels_a_missing_directory_and_symlinks_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let base = std::fs::canonicalize(dir.path()).unwrap();
        let resolved = linux::resolve(&dir.path().join("a").join("..").join("b")).unwrap();
        assert_eq!(resolved, base.join("b"));
        std::fs::create_dir(base.join("real")).unwrap();
        std::os::unix::fs::symlink(base.join("real"), base.join("link")).unwrap();
        let through = linux::resolve(&base.join("link").join("new")).unwrap();
        assert_eq!(through, base.join("real").join("new"));
    }

    // Fails if the real check stops resolving the path or rejects local disks.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_root_and_a_missing_child_of_a_local_directory_pass() {
        assert!(require_local(Path::new("/"), "model weights").is_ok());
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("cache").join("..").join("new").join("x");
        assert!(require_local(&missing, "prefix cache").is_ok());
        assert!(require_local(Path::new("relative/missing"), "prefix cache").is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_dangling_symlink_is_refused_because_its_target_is_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("cache");
        std::os::unix::fs::symlink(dir.path().join("gone"), &link).unwrap();
        assert!(require_local(&link.join("x"), "prefix cache").is_err());
    }

    // Fails if require_local does not consult the live mount table. Runs only
    // where some nfs4 mount exists, found by reading /proc/self/mounts
    // independently of the parser under test.
    #[test]
    fn a_path_under_a_live_nfs4_mount_is_refused() {
        let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
        let nfs4_point = mounts.lines().find_map(|line| {
            let mut fields = line.split(' ');
            let point = fields.nth(1)?;
            (fields.next() == Some("nfs4")).then_some(point.replace("\\040", " "))
        });
        let Some(point) = nfs4_point else {
            eprintln!("skipped: no nfs4 mount on this host");
            return;
        };
        let path = Path::new(&point).join("models/not-created-yet");
        let error = require_local(&path, "model weights").unwrap_err();
        assert!(error.describe().contains("nfs4"), "{}", error.describe());
    }
}
