//! Test seam: a mount table installed for the current thread, so a test can
//! make any directory look like an nfs4 mount without an NFS mount on the
//! host. Production reads `/proc/self/mountinfo` and never reaches this.

use std::{cell::RefCell, path::Path};

thread_local! {
    static TABLE: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

pub(super) fn installed() -> Option<Vec<u8>> {
    TABLE.with(|table| table.borrow().clone())
}

struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        TABLE.with(|table| *table.borrow_mut() = None);
    }
}

/// Runs `run` with `table` as the mount table of this thread, restoring the
/// real one afterwards even when `run` panics.
pub(crate) fn with_table<R>(table: &[u8], run: impl FnOnce() -> R) -> R {
    TABLE.with(|slot| *slot.borrow_mut() = Some(table.to_vec()));
    let _reset = Reset;
    run()
}

/// Runs `run` while `dir` (and everything below it) is an nfs4 mount and the
/// rest of the filesystem is ext4.
pub(crate) fn with_nfs_at<R>(dir: &Path, run: impl FnOnce() -> R) -> R {
    let point = std::fs::canonicalize(dir).expect("fixture directory exists");
    let table = format!(
        "1 0 0:1 / / rw - ext4 /dev/fixture rw\n2 1 0:2 / {} rw - nfs4 nas:/fixture rw\n",
        point.display().to_string().replace(' ', "\\040")
    );
    with_table(table.as_bytes(), run)
}

/// A path below a network mount that this host really has, or `None`. The
/// mounts come from `/proc/self/mounts`, read independently of the code under
/// test. The child does not exist, so a test that reaches the real
/// filesystem creates nothing and reads nothing there.
pub(crate) fn live_network_path() -> Option<std::path::PathBuf> {
    let mounts = std::fs::read_to_string("/proc/self/mounts").ok()?;
    mounts.lines().find_map(|line| {
        let mut fields = line.split(' ');
        let point = fields.nth(1)?;
        matches!(fields.next()?, "nfs" | "nfs4" | "cifs" | "smb3")
            .then(|| Path::new(&point.replace("\\040", " ")).join("storage-guard-test-absent"))
    })
}
