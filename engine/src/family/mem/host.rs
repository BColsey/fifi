//! The host half of the memory facts: `/proc/meminfo` clamped by the process's cgroup tree.
//!
//! [`HostFacts::read`] reads `/proc/meminfo` and the process's cgroup tree (or, when the tree
//! cannot be verified, the bare figures with their provenance). The admission checks refuse a
//! payload that would break the OS floor, the enclosing cgroup limit, or the process's own
//! overhead inside it. Every refusal names the caller that refused; the frozen families' exact
//! texts are pinned by the full-text tests in this module.

use crate::EngError;

/// How sure the host facts are about the cgroup clamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Provenance {
    /// The process's cgroup tree was found, verified and read: the budget is the cgroup-clamped one.
    Verified,
    /// The tree could not be verified (a namespace root, a hybrid layout, a missing controller):
    /// the budget is the bare `MemAvailable`. The text names what was refused.
    Bare(String),
    /// A unified-memory machine: one pool the whole machine shares, no cgroup. The budget is the
    /// physical memory and the floor is the unified-memory one ([`super::host_budget`]).
    Unified,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HostFacts {
    mem_total: u64,
    mem_available: u64,
    // None means a confirmed unlimited cgroup, never a failed limit read.
    cgroup_remaining: Option<u64>,
    provenance: Provenance,
    // This process's resident set when the facts were read; 0 when it could not be read.
    process_rss: u64,
}

impl HostFacts {
    /// The host facts of this process: `/proc/meminfo` clamped by the process's cgroup tree.
    ///
    /// Every caller reads the host through this one function and gets the same answer for the
    /// same machine. When the cgroup tree cannot be verified the facts carry the bare
    /// `MemAvailable` figure and [`Provenance::Bare`], and one line is logged: a start inside a
    /// container the probe cannot parse must keep serving on the figure it always served on, and
    /// a memory refusal where a smaller setting fits is a defect. Only an unreadable or
    /// inconsistent `/proc/meminfo` is an error.
    #[cfg(any(test, target_os = "linux"))]
    #[allow(
        dead_code,
        reason = "the callers are Linux-only; the macOS test build exercises read_in"
    )]
    pub(crate) fn read(label: &'static str, root_label: &'static str) -> Result<Self, EngError> {
        Self::read_in(
            label,
            root_label,
            std::path::Path::new("/proc"),
            std::path::Path::new("/sys/fs/cgroup"),
            std::process::id(),
        )
    }

    #[cfg(any(test, target_os = "linux"))]
    fn read_in(
        label: &'static str,
        root_label: &'static str,
        proc: &std::path::Path,
        root: &std::path::Path,
        pid: u32,
    ) -> Result<Self, EngError> {
        let mut facts = match Self::verified_in(label, root_label, proc, root, pid) {
            Ok(facts) => facts,
            Err(refusal) => {
                let meminfo = read_admission_file(label, &proc.join("meminfo"))?;
                Self::bare(label, root_label, &refusal, &meminfo)?
            }
        };
        facts.process_rss = resident_set_bytes(proc);
        Ok(facts)
    }

    /// The bare figures of the given meminfo text, with one log line (per process) naming the
    /// refusal they replace.
    #[cfg(any(test, target_os = "linux"))]
    pub(crate) fn bare(
        label: &'static str,
        root_label: &'static str,
        refusal: &EngError,
        meminfo: &str,
    ) -> Result<Self, EngError> {
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut facts = Self::from_meminfo(label, meminfo, None)?;
        facts.provenance = Provenance::Bare(format!("{refusal:?}"));
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "{root_label}: cgroup budget unknown ({refusal:?}); using the bare MemAvailable figure"
            );
        }
        Ok(facts)
    }

    /// The host facts of a Mac: the unified memory the machine has, read by the one Mac reader
    /// ([`super::physical_memory_bytes`]) and floored by the one Mac rule ([`super::host_budget`]).
    /// There is no cgroup and no `MemAvailable`: the pool is the whole physical memory.
    #[cfg(not(any(test, target_os = "linux")))]
    pub(crate) fn read(label: &'static str, _root_label: &'static str) -> Result<Self, EngError> {
        let physical = super::physical_memory_bytes().ok_or_else(|| {
            EngError::MemoryPlan(format!(
                "{label} could not read the physical memory (sysctl hw.memsize)."
            ))
        })?;
        Ok(Self::from_unified(physical))
    }

    /// The facts of a unified-memory machine of `physical` bytes.
    pub(crate) fn from_unified(physical: u64) -> Self {
        Self {
            mem_total: physical,
            mem_available: physical,
            cgroup_remaining: None,
            provenance: Provenance::Unified,
            process_rss: 0,
        }
    }

    /// One line for a start line or a report: the budget and whether its cgroup clamp was
    /// verified or the bare `MemAvailable` figure stands in for it.
    pub(crate) fn provenance_note(&self) -> String {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        match &self.provenance {
            Provenance::Verified => format!(
                "host budget {:.1} GiB, cgroup limit verified",
                gib(self.budget())
            ),
            Provenance::Unified => {
                format!("host budget {:.1} GiB, unified memory", gib(self.budget()))
            }
            Provenance::Bare(reason) => format!(
                "host budget {:.1} GiB, cgroup limit unverified (bare MemAvailable): {reason}",
                gib(self.budget())
            ),
        }
    }

    /// `MemTotal` alone, from `/proc/meminfo`: what a plan that only needs the machine's size
    /// asks for (the unified-memory fractions), with no cgroup read.
    #[cfg(any(test, target_os = "linux"))]
    pub(crate) fn physical_total() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo_bytes("the host", &text, "MemTotal:").ok()
    }

    /// Whether the cgroup clamp was verified or the facts are the bare figures.
    pub(crate) fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The verified path: finds the process's cgroup tree, verifies it and reads every limit on
    /// the way to the root. Refuses when anything cannot be verified; [`HostFacts::read`] turns
    /// that refusal into the bare figures.
    #[cfg(any(test, target_os = "linux"))]
    fn verified_in(
        label: &'static str,
        root_label: &'static str,
        proc: &std::path::Path,
        root: &std::path::Path,
        pid: u32,
    ) -> Result<Self, EngError> {
        let meminfo = read_admission_file(label, &proc.join("meminfo"))?;
        let membership = read_admission_file(label, &proc.join("self/cgroup"))?;
        let mounts = read_admission_file(label, &proc.join("self/mountinfo"))?;
        let group = cgroup_from_proc(label, root, &membership, &mounts, pid)?;
        let root = std::fs::canonicalize(root).map_err(|error| root_error(root_label, error))?;
        Self::from_meminfo(label, &meminfo, cgroup_remaining_in(label, &root, &group)?)
    }

    /// Private to `verified_in` except for the host tests of the memory plans, which build the
    /// synthetic cgroup-clamped snapshot a start reads through `read` (the same seam).
    #[cfg(any(test, target_os = "linux"))]
    pub(crate) fn from_meminfo(
        label: &'static str,
        contents: &str,
        cgroup_remaining: Option<u64>,
    ) -> Result<Self, EngError> {
        let total = meminfo_bytes(label, contents, "MemTotal:")?;
        let available = meminfo_bytes(label, contents, "MemAvailable:")?;
        if total == 0 || available > total {
            return Err(EngError::MemoryPlan(format!(
                "{label} memory snapshot has inconsistent totals."
            )));
        }
        Ok(Self {
            mem_total: total,
            mem_available: available,
            cgroup_remaining,
            provenance: Provenance::Verified,
            process_rss: 0,
        })
    }

    /// The byte budget admission admits against: `MemAvailable` clamped by every enclosing
    /// cgroup's remaining bytes. The same figure the admission checks refuse against,
    /// so a host plan sized from it cannot disagree with the guard that checks it. Names no
    /// model family, so any family's host plan may reuse it.
    pub(crate) fn budget(&self) -> u64 {
        self.cgroup_remaining
            .map_or(self.mem_available, |remaining| {
                remaining.min(self.mem_available)
            })
    }

    /// `MemTotal`, bytes.
    pub(crate) fn total(&self) -> u64 {
        self.mem_total
    }

    /// The bytes the OS keeps out of every host plan on this machine ([`os_floor_of`]).
    pub(crate) fn os_floor(&self) -> u64 {
        match self.provenance {
            Provenance::Unified => self.mem_total - super::host_budget(self.mem_total),
            _ => super::os_floor_of(self.mem_total),
        }
    }

    /// What a plan may spend on the process and its stores: `MemAvailable` above the OS floor,
    /// and no more than the enclosing cgroup has left.
    ///
    /// The floor guards the rest of the host (the OS, other processes) from this one. A cgroup
    /// limit already fences this process off from the host, so it is spent whole and the floor is
    /// not charged inside it: a start under `MemoryMax=12G` plans against 12 GiB, not 4.
    pub(crate) fn spare(&self) -> u64 {
        let above_floor = self.mem_available.saturating_sub(self.os_floor());
        self.cgroup_remaining
            .map_or(above_floor, |remaining| remaining.min(above_floor))
    }

    /// The bytes of the budget a plan with its own process terms keeps back: the budget less
    /// what [`HostFacts::spare`] allows spending. The whole OS floor while the host is the binding
    /// limit, less while a cgroup is.
    pub(crate) fn kept_back(&self) -> u64 {
        self.budget() - self.spare()
    }

    /// What this process still adds to its own footprint beyond the payload an admission names:
    /// heap growth, the CUDA context's host side, staging and allocator slack. Taken from the
    /// resident set now (the process is about to roughly double what it holds while it loads)
    /// plus the shared host margin.
    pub(crate) fn process_overhead(&self) -> u64 {
        self.process_rss.saturating_add(super::HOST_MARGIN_BYTES)
    }

    /// What an admission that charges only its payload may spend: [`HostFacts::spare`], and
    /// inside a binding cgroup (where no OS floor is charged) the cgroup's remainder less the
    /// process's own overhead, so a payload that fills the cgroup cannot be killed by the heap
    /// the process still has to grow.
    pub(crate) fn payload_spare(&self) -> u64 {
        let above_floor = self.mem_available.saturating_sub(self.os_floor());
        self.cgroup_remaining.map_or(above_floor, |remaining| {
            above_floor.min(remaining.saturating_sub(self.process_overhead()))
        })
    }

    /// The bytes of the budget a payload-only admission keeps back.
    pub(crate) fn payload_kept_back(&self) -> u64 {
        self.budget() - self.payload_spare()
    }

    /// `MemTotal` paired with [`HostFacts::budget`], for host plans that carry
    /// both figures (the tier and bank plans read them as a pair).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the CUDA-build facts paths read the pair")
    )]
    pub(crate) fn host_memory(&self) -> (u64, u64) {
        (self.mem_total, self.budget())
    }

    /// [`HostFacts::admit_payload`] with the caller's label as the refusing name.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the refusals are pinned by this module's tests")
    )]
    pub(crate) fn admit(&self, label: &'static str, required: u64) -> Result<(), EngError> {
        self.admit_payload(required, label)
    }

    /// The admission check for a payload (a store, a snapshot, a pinned pool): the payload must
    /// fit [`HostFacts::payload_spare`], so the process's own overhead is always charged beside
    /// it and no caller can leave it out. The refusal names the caller.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the refusals are pinned by this module's tests")
    )]
    pub(crate) fn admit_payload(&self, required: u64, what: &str) -> Result<(), EngError> {
        self.admit_within(required, what, self.payload_spare())
    }

    /// The admission check for bytes that already include the process's own terms (a plan that
    /// charges its heap, margin and staging itself, as the Flash-Next host plan does): they must
    /// fit [`HostFacts::spare`]. A caller that charges only a payload uses
    /// [`HostFacts::admit_payload`].
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the refusals are pinned by this module's tests")
    )]
    pub(crate) fn admit_process_bytes(&self, required: u64, what: &str) -> Result<(), EngError> {
        self.admit_within(required, what, self.spare())
    }

    fn admit_within(&self, required: u64, what: &str, spare: u64) -> Result<(), EngError> {
        let available = self.budget();
        if spare == 0 || required > spare {
            return Err(EngError::MemoryPlan(format!(
                "{what} needs {required} bytes plus {} bytes for the OS floor. Available host/cgroup budget: {available} bytes.",
                available.saturating_sub(spare)
            )));
        }
        Ok(())
    }
}

/// This process's resident set in bytes (`VmRSS` of `<proc>/self/status`), 0 when unreadable.
#[cfg(any(test, target_os = "linux"))]
fn resident_set_bytes(proc: &std::path::Path) -> u64 {
    std::fs::read_to_string(proc.join("self/status"))
        .ok()
        .and_then(|text| {
            let line = text.lines().find_map(|line| line.strip_prefix("VmRSS:"))?;
            let mut fields = line.split_whitespace();
            let kib = fields.next()?.parse::<u64>().ok()?;
            (fields.next() == Some("kB")).then(|| kib.saturating_mul(1024))
        })
        .unwrap_or(0)
}

#[cfg(any(test, target_os = "linux"))]
fn meminfo_bytes(label: &'static str, contents: &str, name: &str) -> Result<u64, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs one valid {name} value in kB from /proc/meminfo."
        ))
    };
    let mut matches = contents.lines().filter_map(|line| line.strip_prefix(name));
    let mut fields = matches.next().ok_or_else(invalid)?.split_whitespace();
    let bytes = fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(invalid)?;
    if fields.next() != Some("kB") || fields.next().is_some() || matches.next().is_some() {
        return Err(invalid());
    }
    Ok(bytes)
}

#[cfg(any(test, target_os = "linux"))]
fn cgroup_remaining_bytes(
    label: &'static str,
    maximum: &str,
    current: &str,
) -> Result<Option<u64>, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
        ))
    };
    let parse = |value: &str| {
        let value = value.trim();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid());
        }
        value.parse::<u64>().map_err(|_| invalid())
    };
    let current = parse(current)?;
    match maximum.trim() {
        "max" => Ok(None),
        maximum => Ok(Some(parse(maximum)?.saturating_sub(current))),
    }
}

#[cfg(any(test, target_os = "linux"))]
fn root_error(root_label: &str, error: std::io::Error) -> EngError {
    EngError::MemoryPlan(format!("{root_label} cgroup root: {error}"))
}

#[cfg(any(test, target_os = "linux"))]
fn read_admission_file(label: &'static str, path: &std::path::Path) -> Result<String, EngError> {
    std::fs::read_to_string(path).map_err(|error| {
        EngError::MemoryPlan(format!("{label} cannot read {}: {error}", path.display()))
    })
}

// Discovery must verify the global controller root and reject hidden ancestors first.
#[cfg(any(test, target_os = "linux"))]
fn cgroup_remaining_in(
    label: &'static str,
    root: &std::path::Path,
    cgroup: &std::path::Path,
) -> Result<Option<u64>, EngError> {
    if !root.is_absolute()
        || !cgroup.is_absolute()
        || !cgroup.starts_with(root)
        || root
            .components()
            .chain(cgroup.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(EngError::MemoryPlan(format!(
            "{label} cgroup must be inside its verified controller root."
        )));
    }
    let mut minimum: Option<u64> = None;
    let mut group = cgroup;
    while group != root {
        let parent = group.parent().ok_or_else(|| {
            EngError::MemoryPlan(format!("{label} cgroup has no controller ancestor."))
        })?;
        if read_admission_file(label, &parent.join("cgroup.subtree_control"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
            && let Some(remaining) = cgroup_remaining_bytes(
                label,
                &read_admission_file(label, &group.join("memory.max"))?,
                &read_admission_file(label, &group.join("memory.current"))?,
            )?
        {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
        group = parent;
    }
    // The root of a cgroup namespace is the container's own cgroup: no parent is visible, so
    // its limit is the outermost one. The global root has no limit files.
    if root.join("memory.max").exists() {
        let remaining = cgroup_remaining_bytes(
            label,
            &read_admission_file(label, &root.join("memory.max"))?,
            &read_admission_file(label, &root.join("memory.current"))?,
        )?;
        if let Some(remaining) = remaining {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
    }
    Ok(minimum)
}

#[cfg(any(test, target_os = "linux"))]
fn verify_cgroup_root(
    label: &'static str,
    root: &std::path::Path,
    mounts: &str,
) -> Result<(), EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs one complete cgroup2 root with the memory controller."
        ))
    };
    let point = root.to_str().ok_or_else(invalid)?;
    let mut mounted = false;
    for line in mounts.lines() {
        let Some((head, tail)) = line.split_once(" - ") else {
            continue;
        };
        let fields: Vec<_> = head.split_ascii_whitespace().collect();
        if fields.get(4) != Some(&point) {
            continue;
        }
        if mounted
            || fields.get(3) != Some(&"/")
            || tail.split_ascii_whitespace().next() != Some("cgroup2")
        {
            return Err(invalid());
        }
        mounted = true;
    }
    if !mounted
        || !read_admission_file(label, &root.join("cgroup.controllers"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
    {
        return Err(invalid());
    }
    // These interfaces exist on namespace/subtree roots (a container's own cgroup), but not on
    // the global root. All three together mean a namespace root, whose own limit
    // `cgroup_remaining_in` applies; some of them alone is an ambiguous tree.
    let mut present = 0;
    for name in ["cgroup.type", "memory.max", "memory.current"] {
        let path = root.join(name);
        match std::fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => present += 1,
            Err(error) => {
                return Err(EngError::MemoryPlan(format!(
                    "{label} cannot verify {}: {error}",
                    path.display()
                )));
            }
        }
    }
    if present != 0 && present != 3 {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(any(test, target_os = "linux"))]
fn cgroup_from_proc(
    label: &'static str,
    root: &std::path::Path,
    membership: &str,
    mounts: &str,
    pid: u32,
) -> Result<std::path::PathBuf, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} cannot map its process inside a complete unified cgroup tree."
        ))
    };
    // Legacy hierarchies such as net_cls can coexist with v2 memory accounting.
    let mut unified_path = None;
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let id = fields.next().ok_or_else(invalid)?;
        let controllers = fields.next().ok_or_else(invalid)?;
        let path = std::path::Path::new(fields.next().ok_or_else(invalid)?);
        if !path.is_absolute() {
            return Err(invalid());
        }
        if id == "0" {
            if !controllers.is_empty() || unified_path.replace(path).is_some() {
                return Err(invalid());
            }
        } else if !id.bytes().all(|b| b.is_ascii_digit())
            || id.parse::<u32>().ok().is_none_or(|id| id == 0)
            || controllers
                .split(',')
                .any(|c| c.is_empty() || c == "memory" || c.contains(char::is_whitespace))
        {
            return Err(invalid());
        }
    }
    let path = unified_path.ok_or_else(invalid)?;
    if pid == 0
        || !root.is_absolute()
        || root
            .components()
            .chain(path.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(invalid());
    }
    let resolve = |path: &std::path::Path| {
        std::fs::canonicalize(path).map_err(|error| {
            EngError::MemoryPlan(format!(
                "{label} cannot resolve {}: {error}",
                path.display()
            ))
        })
    };
    let root = resolve(root)?;
    verify_cgroup_root(label, &root, mounts)?;
    let group = resolve(&root.join(path.strip_prefix("/").map_err(|_| invalid())?))?;
    if !group.starts_with(&root) {
        return Err(invalid());
    }
    if group != root && read_admission_file(label, &group.join("cgroup.type"))?.trim() != "domain" {
        return Err(invalid());
    }
    let mut found = false;
    for owner in read_admission_file(label, &group.join("cgroup.procs"))?.split_whitespace() {
        found |= owner.parse::<u32>().map_err(|_| invalid())? == pid;
    }
    if !found {
        return Err(invalid());
    }
    Ok(group)
}

#[cfg(test)]
mod tests {
    const LABEL: &str = "the caller";
    const ROOT_LABEL: &str = "the caller";
    use super::*;

    fn refusal_text<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::MemoryPlan(message)) => message,
            Err(other) => panic!("unexpected refusal variant: {other:?}"),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    #[test]
    fn refusal_texts_carry_the_neutral_label_and_name_no_family() {
        let no_family = |message: &str| {
            assert!(message.contains(LABEL), "{message}");
            assert!(
                !message.contains("Flint") && !message.contains("Flash"),
                "{message}"
            );
        };
        // Every reachable refusal, produced, not quoted from the source.
        no_family(&refusal_text(HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            None,
        )));
        no_family(&refusal_text(meminfo_bytes(LABEL, "junk", "MemAvailable:")));
        no_family(&refusal_text(cgroup_remaining_bytes(LABEL, "bogus", "0")));
        no_family(&refusal_text(read_admission_file(
            LABEL,
            std::path::Path::new("/nonexistent-admission-file"),
        )));
        no_family(&refusal_text(cgroup_remaining_in(
            LABEL,
            std::path::Path::new("/sys/fs/cgroup"),
            std::path::Path::new("/var"),
        )));
        no_family(&refusal_text(verify_cgroup_root(
            LABEL,
            std::path::Path::new("/definitely/not/here"),
            "",
        )));
        let (_root, _keep, directory, proc_dir) = namespace_root("max", "memory\n");
        no_family(&refusal_text(cgroup_from_proc(
            LABEL,
            &directory,
            "0::\nbroken",
            "",
            42,
        )));
        let mounts = std::fs::read_to_string(proc_dir.join("self/mountinfo")).unwrap();
        let looped = directory.join("cgroup.type");
        std::fs::remove_file(&looped).unwrap();
        std::os::unix::fs::symlink("cgroup.type", &looped).unwrap();
        no_family(&refusal_text(verify_cgroup_root(
            LABEL, &directory, &mounts,
        )));
        // The no-ancestor refusal is structurally unreachable: a cgroup that
        // starts_with(root) always walks up to root, so that branch stays
        // untriggered by design.
        // The root-label line only fires in the window between
        // cgroup_from_proc's canonicalize and verified_in's own, so it is
        // pinned through root_error rather than produced end to end.
        assert_eq!(
            refusal_text::<()>(Err(root_error(ROOT_LABEL, std::io::Error::other("x")))),
            "the caller cgroup root: x"
        );
        no_family(&refusal_text(cgroup_from_proc(
            LABEL,
            &directory,
            "0::/missing-group\n",
            &std::fs::read_to_string(proc_dir.join("self/mountinfo")).unwrap(),
            42,
        )));
        assert_eq!(
            refusal_text(HostFacts::from_meminfo(
                LABEL,
                "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
                None
            )),
            "the caller memory snapshot has inconsistent totals."
        );
        assert_eq!(
            refusal_text(meminfo_bytes(LABEL, "junk", "MemAvailable:")),
            "the caller needs one valid MemAvailable: value in kB from /proc/meminfo."
        );
        assert_eq!(
            refusal_text(cgroup_remaining_bytes(LABEL, "bogus", "0")),
            "the caller needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
        );
        assert!(
            refusal_text(read_admission_file(
                LABEL,
                std::path::Path::new("/nonexistent-admission-file")
            ))
            .starts_with("the caller cannot read /nonexistent-admission-file: ")
        );
        assert_eq!(
            refusal_text(cgroup_remaining_in(
                LABEL,
                std::path::Path::new("/sys/fs/cgroup"),
                std::path::Path::new("/var")
            )),
            "the caller cgroup must be inside its verified controller root."
        );
        assert_eq!(
            refusal_text(verify_cgroup_root(
                LABEL,
                std::path::Path::new("/definitely/not/here"),
                ""
            )),
            "the caller needs one complete cgroup2 root with the memory controller."
        );
        assert_eq!(
            refusal_text(cgroup_from_proc(
                LABEL,
                std::path::Path::new("/sys/fs/cgroup"),
                "garbage",
                "",
                42
            )),
            "the caller cannot map its process inside a complete unified cgroup tree."
        );
    }

    #[test]
    fn flint_label_refusals_keep_the_frozen_texts_in_full() {
        const FLINT: &str = "Flint expert cache";
        assert_eq!(
            refusal_text(HostFacts::from_meminfo(
                FLINT,
                "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
                None
            )),
            "Flint expert cache memory snapshot has inconsistent totals."
        );
        assert_eq!(
            refusal_text(cgroup_remaining_in(
                FLINT,
                std::path::Path::new("/sys/fs/cgroup"),
                std::path::Path::new("/var")
            )),
            "Flint expert cache cgroup must be inside its verified controller root."
        );
        assert_eq!(
            refusal_text(verify_cgroup_root(
                FLINT,
                std::path::Path::new("/definitely/not/here"),
                ""
            )),
            "Flint expert cache needs one complete cgroup2 root with the memory controller."
        );
        assert_eq!(
            refusal_text(cgroup_from_proc(
                FLINT,
                std::path::Path::new("/sys/fs/cgroup"),
                "garbage",
                "",
                42
            )),
            "Flint expert cache cannot map its process inside a complete unified cgroup tree."
        );
        assert_eq!(
            refusal_text::<()>(Err(root_error("Flint", std::io::Error::other("x")))),
            "Flint cgroup root: x"
        );
    }

    #[test]
    fn captured_cgroup_roots_match_real_admission() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/cgroup-views-20261028.json")).unwrap();
        let hostns: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/cgroup-hostns-20261028.json")).unwrap();
        let host_files = captures["host"]["root_file_names"].as_array().unwrap();
        for name in ["cgroup.type", "memory.max", "memory.current"] {
            assert!(!host_files.iter().any(|value| value.as_str() == Some(name)));
        }
        for (name, view, accepted) in [
            ("container", &captures["container"], false),
            ("host", &captures["host"], true),
            ("hostns", &hostns, true),
        ] {
            let root = tempfile::tempdir().unwrap();
            let directory = root.path().canonicalize().unwrap();
            let controllers = view["controllers"].as_str().unwrap();
            assert!(
                controllers
                    .split_whitespace()
                    .any(|value| value == "memory")
            );
            std::fs::write(directory.join("cgroup.controllers"), controllers).unwrap();
            for (file, key) in [
                ("cgroup.type", "cgroup_type"),
                ("memory.max", "memory_max"),
                ("memory.current", "memory_current"),
            ] {
                if let Some(contents) = view[key].as_str() {
                    std::fs::write(directory.join(file), contents).unwrap();
                }
            }
            let mounts = view["mountinfo"]
                .as_str()
                .unwrap()
                .replace("/sys/fs/cgroup", &directory.display().to_string());
            let result = verify_cgroup_root(LABEL, &directory, &mounts);
            eprintln!("captured view {name}: accepted={}", result.is_ok());
            if accepted {
                assert!(result.is_ok(), "{name}: {result:?}");
            } else {
                assert!(matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("one complete cgroup2 root")));
            }
        }
    }

    #[test]
    fn an_unverified_budget_falls_back_to_the_bare_figure() {
        let refusal = EngError::MemoryPlan(String::from("admission: cannot map its process"));
        let bare = HostFacts::bare(
            LABEL,
            ROOT_LABEL,
            &refusal,
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        // The bare figures: machine-wide, no cgroup minimum applied.
        assert_eq!(bare.host_memory(), (32u64 << 30, 24u64 << 30));
        assert!(matches!(bare.provenance(), Provenance::Bare(text) if text.contains("cannot map")));
    }

    #[test]
    fn captured_jetson_layout_refuses_cleanly_to_the_bare_fallback() {
        // The Xavier (L4T) runs hybrid cgroup v1 with a controller-less unified tree: the
        // membership's first line is a v1 controller line, not `0::`, and no memory limit
        // lives anywhere the unified reader looks. The capture pins that this layout refuses
        // with the clean "cannot map" error — the refusal `read` turns into the
        // bare `/proc/meminfo` fallback — before any filesystem access at all.
        let jetson: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/cgroup-jetson-20261118.json")).unwrap();
        let view = &jetson["jetson"];
        assert_eq!(view["unified_controllers"].as_str().unwrap(), "");
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let result = cgroup_from_proc(
            LABEL,
            &directory,
            view["membership"].as_str().unwrap(),
            view["mountinfo"].as_str().unwrap(),
            std::process::id(),
        );
        assert!(
            matches!(&result, Err(EngError::MemoryPlan(message))
                if message.contains("cannot map its process")),
            "the Jetson layout must refuse, not resolve or panic: {result:?}"
        );
        // The refusal happens before the tree is read: the untouched directory proves it.
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        eprintln!("captured jetson view: refused cleanly, bare fallback applies");
    }

    #[test]
    fn captured_cgroup_host_member_maps_to_nested_group() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/cgroup-views-20261028.json")).unwrap();
        let view = &captures["host"];
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        std::fs::write(
            directory.join("cgroup.controllers"),
            view["controllers"].as_str().unwrap(),
        )
        .unwrap();
        let membership = view["membership"].as_str().unwrap();
        let group = directory.join(membership.trim().strip_prefix("0::/").unwrap());
        std::fs::create_dir_all(&group).unwrap();
        std::fs::write(
            group.join("cgroup.type"),
            view["member_type"].as_str().unwrap(),
        )
        .unwrap();
        std::fs::write(
            group.join("cgroup.procs"),
            view["member_procs"].as_str().unwrap(),
        )
        .unwrap();
        let mounts = view["mountinfo"]
            .as_str()
            .unwrap()
            .replace("/sys/fs/cgroup", &directory.display().to_string());
        let pid = u32::try_from(view["pid"].as_u64().unwrap()).unwrap();
        let result = cgroup_from_proc(LABEL, &directory, membership, &mounts, pid);
        assert_eq!(result.unwrap(), group);
        eprintln!("captured host member: mapped=true pid={pid}");
    }

    /// The canonical controller root and its member group, plus the mounts that cover it.
    /// Paths are canonical so the strict ParentDir refusal in `cgroup_from_proc` judges the
    /// fixture tree, never the caller's `TMPDIR` spelling.
    fn discovery_fixture() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
        String,
    ) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let group = directory.join("work");
        std::fs::create_dir(&group).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(group.join("cgroup.type"), "domain\n").unwrap();
        std::fs::write(group.join("cgroup.procs"), "41\n42\n").unwrap();
        let mounts = format!(
            "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
            directory.display()
        );
        (root, directory, group, mounts)
    }

    #[test]
    fn cgroup_discovery_resolves_own_pid_in_global_tree() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let resolved = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
        assert!(
            resolved.is_ok(),
            "a complete global cgroup tree must resolve its own process group"
        );
        assert_eq!(resolved.unwrap(), group);
    }

    #[cfg(unix)]
    #[test]
    fn cgroup_discovery_matches_canonical_mount_paths() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases
            .path()
            .canonicalize()
            .unwrap()
            .join("controller-root");
        std::os::unix::fs::symlink(&directory, &alias).unwrap();
        let resolved = cgroup_from_proc(LABEL, &alias, "0::/work\n", &mounts, 42).unwrap();
        assert_eq!(resolved, group);
    }

    #[cfg(unix)]
    #[test]
    fn a_root_spelled_through_parent_components_is_refused() {
        // The strictness is the point: a root that merely RESOLVES to the fixture
        // tree but is spelled with ".." (a TMPDIR like "/x/wt/../tmp") is refused
        // before canonicalization, so an ambiguous tree never becomes a budget.
        let (_root, directory, _group, mounts) = discovery_fixture();
        let dotted = directory.join("..").join(directory.file_name().unwrap());
        // The dotted spelling must resolve to the very fixture tree, so the
        // refusal below can only come from the strict ParentDir check.
        assert_eq!(
            std::fs::canonicalize(&dotted).unwrap(),
            std::fs::canonicalize(&directory).unwrap()
        );
        let result = cgroup_from_proc(LABEL, &dotted, "0::/work\n", &mounts, 42);
        assert!(
            matches!(&result, Err(EngError::MemoryPlan(message))
                if message.contains("cannot map its process")),
            "a root spelled with .. must be refused, got {result:?}"
        );
    }

    #[test]
    fn cgroup_discovery_rejects_hidden_or_ambiguous_roots() {
        for (name, contents) in [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
            ("cgroup.controllers", "cpu\n"),
        ] {
            let (_root, directory, _, mounts) = discovery_fixture();
            std::fs::write(directory.join(name), contents).unwrap();
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject root interface {name}"
            );
        }
        for (source, filesystem, copies) in [
            ("/delegated", "cgroup2", 1),
            ("/", "tmpfs", 1),
            ("/", "cgroup2", 2),
        ] {
            let (_root, directory, group, _) = discovery_fixture();
            let mounts = format!(
                "36 29 0:33 {source} {} rw - {filesystem} cgroup rw\n",
                group.parent().unwrap().display()
            )
            .repeat(copies);
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject mount {source}/{filesystem}/{copies}"
            );
        }
    }

    /// A container's view: the cgroup root is the container's own cgroup, with limit files.
    /// Both returned paths are canonical so the strict ParentDir refusals judge the fixture
    /// tree, never the caller's `TMPDIR` spelling; the TempDirs only keep the trees alive.
    fn namespace_root(
        maximum: &str,
        controllers: &str,
    ) -> (
        tempfile::TempDir,
        tempfile::TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        for (name, contents) in [
            ("cgroup.controllers", controllers),
            ("cgroup.subtree_control", ""),
            ("cgroup.type", "domain\n"),
            ("cgroup.procs", "1\n42\n"),
            ("memory.max", maximum),
            ("memory.current", "2147483648\n"),
        ] {
            std::fs::write(directory.join(name), contents).unwrap();
        }
        let proc = tempfile::tempdir().unwrap();
        let proc_dir = proc.path().canonicalize().unwrap();
        std::fs::create_dir(proc_dir.join("self")).unwrap();
        std::fs::write(
            proc_dir.join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc_dir.join("self/cgroup"), "0::/\n").unwrap();
        std::fs::write(
            proc_dir.join("self/mountinfo"),
            format!(
                "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
                directory.display()
            ),
        )
        .unwrap();
        (root, proc, directory, proc_dir)
    }

    #[test]
    fn a_namespace_root_applies_its_own_limit_or_falls_back_to_available_memory() {
        let (_root, _proc, directory, proc_dir) = namespace_root("8589934592\n", "memory\n");
        let budget = HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(6u64 << 30));
        assert_eq!(budget.os_floor(), crate::family::mem::os_floor_of(32 << 30));
        // 6 GiB left in the cgroup: the cgroup is the binding limit, so the OS floor is not
        // charged inside it. A plan that carries its own process terms may spend the whole 6 GiB;
        // a payload keeps the process's overhead (the 2 GiB host margin here, no resident set
        // in this fixture) back inside it.
        assert!(budget.admit_process_bytes(6u64 << 30, LABEL).is_ok());
        assert!(budget.admit_process_bytes((6u64 << 30) + 1, LABEL).is_err());
        assert!(budget.admit(LABEL, 4u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (4u64 << 30) + 1).is_err());
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "memory\n");
        let budget = HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, None);
        // MemAvailable is 24 GiB; the floor of a 32 GiB host is a fifth of it.
        let spare = (24u64 << 30) - crate::family::mem::os_floor_of(32 << 30);
        assert!(budget.admit(LABEL, spare).is_ok());
        assert!(budget.admit(LABEL, spare + 1).is_err());
    }

    #[test]
    fn a_root_with_some_but_not_all_namespace_files_is_refused() {
        let files = [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
        ];
        for skipped in 0..files.len() {
            let (_root, directory, _group, mounts) = discovery_fixture();
            for (index, (name, contents)) in files.iter().enumerate() {
                if index != skipped {
                    std::fs::write(directory.join(name), contents).unwrap();
                }
            }
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "two of three root files without {} must be refused",
                files[skipped].0
            );
        }
    }

    #[test]
    fn a_namespace_root_without_the_memory_controller_is_refused_clearly() {
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "cpu\n");
        let result = HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42);
        assert!(matches!(result, Err(EngError::MemoryPlan(message))
                if message.contains("complete cgroup2 root with the memory controller")));
    }

    #[test]
    fn hybrid_membership_uses_the_unified_memory_budget_in_either_order() {
        let (_root, _proc, directory, proc_dir) = namespace_root("8589934592\n", "memory\n");
        for membership in [
            "1:net_cls:/\n0::/\n",
            "0::/\n2:cpu,cpuacct:/work:bench\n",
            // A v1 line for a different controller whose path differs from the unified
            // path must not change which cgroup's limit is read.
            "0::/\n1:devices:/elsewhere\n",
        ] {
            std::fs::write(proc_dir.join("self/cgroup"), membership).unwrap();
            let budget =
                HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
            // Machine-wide MemAvailable is 24 GiB, but only 6 GiB remain in this group.
            assert_eq!(budget.budget(), 6u64 << 30);
        }
    }

    #[test]
    fn cgroup_discovery_rejects_bad_process_mapping() {
        for (membership, pid, owners, kind) in [
            ("0::/../work\n", 42, "41\n42\n", "domain"),
            ("0::work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n0::/work\n", 42, "41\n42\n", "domain"),
            ("1:memory:/work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n1:cpu,memory:/\n", 42, "42\n", "domain"),
            ("0::/work\n0:net_cls:/\n", 42, "42\n", "domain"),
            ("0::/work\n1::/\n", 42, "42\n", "domain"),
            ("0::/work\n1:cpu,:/\n", 42, "42\n", "domain"),
            ("0::/work\nx:cpu:/\n", 42, "42\n", "domain"),
            ("0::/work\n1:net_cls:relative\n", 42, "42\n", "domain"),
            ("0::/work\n", 0, "41\n42\n", "domain"),
            ("0::/work\n", 42, "41\n", "domain"),
            ("0::/work\n", 42, "41\n42\n", "threaded"),
            ("0::/work\n", 42, "41\n42\nnot-a-pid\n", "domain"),
        ] {
            let (_root, directory, group, mounts) = discovery_fixture();
            std::fs::write(group.join("cgroup.procs"), owners).unwrap();
            std::fs::write(group.join("cgroup.type"), kind).unwrap();
            let result = cgroup_from_proc(LABEL, &directory, membership, &mounts, pid);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("cannot map its process")),
                "must reject process mapping {membership:?}/{pid}/{owners:?}/{kind}"
            );
        }
    }

    #[test]
    fn cgroup_reader_rejects_missing_current_with_unlimited_max() {
        let _root = tempfile::tempdir().unwrap();
        let directory = _root.path().canonicalize().unwrap();
        let child = directory.join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(child.join("memory.max"), "max\n").unwrap();
        let result = cgroup_remaining_in(LABEL, &directory, &child);
        assert!(
            matches!(
                result,
                Err(EngError::MemoryPlan(message)) if message.contains("memory.current")
            ),
            "an unreadable active charge file must not become an unlimited budget"
        );
    }

    #[test]
    fn cgroup_reader_applies_finite_ancestor_charges() {
        let _root = tempfile::tempdir().unwrap();
        let directory = _root.path().canonicalize().unwrap();
        let parent = directory.join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        for dir in [directory.as_path(), parent.as_path()] {
            std::fs::write(dir.join("cgroup.subtree_control"), "memory\n").unwrap();
        }
        for (dir, maximum, current) in [
            (&parent, "17179869184\n", "4294967296\n"),
            (&child, "68719476736\n", "1073741824\n"),
        ] {
            std::fs::write(dir.join("memory.max"), maximum).unwrap();
            std::fs::write(dir.join("memory.current"), current).unwrap();
        }
        let remaining = cgroup_remaining_in(LABEL, &directory, &child);
        assert!(
            remaining.is_ok(),
            "a readable hierarchy must honor finite ancestor charges"
        );
        assert_eq!(remaining.unwrap(), Some(12u64 << 30));
    }

    #[test]
    fn cgroup_snapshot_accounts_for_current_charges() {
        let remaining = cgroup_remaining_bytes(LABEL, "17179869184\n", "4294967296\n");
        assert!(
            remaining.is_ok(),
            "valid finite cgroup limit and current charges must produce a budget"
        );
        let remaining = remaining.unwrap();
        assert_eq!(remaining, Some(12u64 << 30));
        let budget = HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
            remaining,
        )
        .unwrap();
        assert!(budget.admit_process_bytes(12u64 << 30, LABEL).is_ok());
        assert!(
            budget
                .admit_process_bytes((12u64 << 30) + 1, LABEL)
                .is_err()
        );
        assert!(budget.admit(LABEL, 10u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (10u64 << 30) + 1).is_err());
    }

    #[test]
    fn budget_is_the_cgroup_minimum_in_either_order() {
        // The plan-facing accessor must hand a plan the same number the guard refuses against:
        // MemAvailable clamped by the cgroup remaining bytes, whichever side is smaller, and
        // the bare host figure when the cgroup is unlimited.
        let unlimited = HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 25165824 kB\n",
            None,
        )
        .unwrap();
        assert_eq!(unlimited.budget(), 24 << 30);
        assert_eq!(unlimited.host_memory(), (64 << 30, 24 << 30));

        let cgroup_smaller = HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 25165824 kB\n",
            Some(12 << 30),
        )
        .unwrap();
        assert_eq!(cgroup_smaller.budget(), 12 << 30);

        let host_smaller = HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 8388608 kB\n",
            Some(12 << 30),
        )
        .unwrap();
        assert_eq!(host_smaller.budget(), 8 << 30);

        // One budget for plan and guard: the last byte the guard admits is `spare`. With the
        // cgroup binding (12 GiB left of 24 GiB available) that is the whole cgroup remainder;
        // with the host binding it is the available figure minus the 8 GiB OS floor of the
        // 64 GiB host.
        assert_eq!(cgroup_smaller.spare(), 12 << 30);
        assert!(
            cgroup_smaller
                .admit_process_bytes(12u64 << 30, LABEL)
                .is_ok()
        );
        assert!(
            cgroup_smaller
                .admit_process_bytes((12u64 << 30) + 1, LABEL)
                .is_err()
        );
        assert_eq!(
            cgroup_smaller.payload_spare(),
            10 << 30,
            "the process overhead stays back"
        );
        assert_eq!(
            unlimited.payload_spare(),
            unlimited.spare(),
            "uncapped: the floor covers it"
        );
        assert_eq!(unlimited.spare(), 16 << 30);
        assert_eq!(unlimited.kept_back(), 8 << 30);
        assert_eq!(
            host_smaller.spare(),
            0,
            "the floor is above what the host has"
        );
        // A cgroup roomier than the host above its floor does not raise the spare.
        let roomy = HostFacts::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 25165824 kB\n",
            Some(20 << 30),
        )
        .unwrap();
        assert_eq!(roomy.spare(), 16 << 30);
    }

    #[test]
    fn cgroup_snapshot_distinguishes_unlimited_and_exhausted_limits() {
        for (maximum, current, remaining) in [
            ("max\n", "0\n", None),
            (" max\n", "18446744073709551615\n", None),
            ("0", "0", Some(0)),
            ("1024", "1024", Some(0)),
            ("4", "8", Some(0)),
            ("18446744073709551615", "0", Some(u64::MAX)),
            ("18446744073709551615", "18446744073709551615", Some(0)),
        ] {
            assert_eq!(
                cgroup_remaining_bytes(LABEL, maximum, current).unwrap(),
                remaining,
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn cgroup_snapshot_rejects_malformed_limits_and_current_charges() {
        for (maximum, current) in [
            ("", "0"),
            ("MAX", "0"),
            ("bad", "0"),
            ("1 kB", "0"),
            ("1 2", "0"),
            ("1.5", "0"),
            ("-1", "0"),
            ("+1", "0"),
            ("18446744073709551616", "0"),
            ("1", ""),
            ("1", "bad"),
            ("1", "1 kB"),
            ("1", "-1"),
            ("1", "+1"),
            ("1", "18446744073709551616"),
            ("max", ""),
            ("max", "bad"),
            ("max", "max"),
            ("max", "-1"),
            ("max", "18446744073709551616"),
        ] {
            assert!(
                cgroup_remaining_bytes(LABEL, maximum, current).is_err(),
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn host_snapshot_uses_mem_available_and_model_scaled_os_floor() {
        let result = HostFacts::from_meminfo(
            LABEL,
            "MemTotal:       268435456 kB\nMemAvailable:    50331648 kB\n",
            None,
        );
        assert!(
            result.is_ok(),
            "valid memory snapshot must produce an admission budget"
        );
        let budget = result.unwrap();
        assert_eq!(budget.mem_available, 48u64 << 30);
        assert_eq!(budget.os_floor(), 27_487_790_695);
        assert_eq!(budget.cgroup_remaining, None);
        assert!(budget.admit(LABEL, 256 * 47 * 1_603_584u64).is_ok());
    }

    #[test]
    fn host_snapshot_preserves_minimum_floor_and_finite_cgroup() {
        let text = "MemAvailable: 25165824 kB\nMemFree: 0 kB\nMemTotal: 33554432 kB\n";
        let floor = crate::family::mem::os_floor_of(32u64 << 30);
        let budget = HostFacts::from_meminfo(LABEL, text, Some(floor + 1)).unwrap();
        assert_eq!(budget.os_floor(), floor);
        assert_eq!(budget.cgroup_remaining, Some(floor + 1));
        assert!(budget.admit_process_bytes(floor + 1, LABEL).is_ok());
        assert!(budget.admit_process_bytes(floor + 2, LABEL).is_err());
        assert!(
            HostFacts::from_meminfo(LABEL, text, Some(0))
                .unwrap()
                .admit(LABEL, 0)
                .is_err()
        );
        let empty =
            HostFacts::from_meminfo(LABEL, "MemTotal: 1 kB\nMemAvailable: 0 kB\n", None).unwrap();
        assert_eq!(empty.mem_available, 0);
        assert!(empty.admit(LABEL, 0).is_err());
    }

    #[test]
    fn host_snapshot_rejects_malformed_or_inconsistent_fields() {
        for text in [
            "",
            "MemTotal: 1 kB\n",
            "MemAvailable: 1 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            "MemTotal: -1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 MB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: bad kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 MB\n",
            "MemTotal: 1 kB trailing\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemTotal: 1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 18014398509481984 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 18014398509481984 kB\n",
        ] {
            assert!(
                HostFacts::from_meminfo(LABEL, text, None).is_err(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn host_budget_rejects_unrepresentable_payload_plus_floor() {
        let budget = HostFacts {
            mem_total: 40 << 30,
            mem_available: u64::MAX,
            cgroup_remaining: None,
            provenance: Provenance::Verified,
            process_rss: 0,
        };
        let floor = budget.os_floor();
        assert!(budget.admit(LABEL, u64::MAX).is_err());
        assert!(budget.admit(LABEL, u64::MAX - floor + 1).is_err());
        assert!(budget.admit(LABEL, u64::MAX - floor).is_ok());
    }

    #[test]
    fn host_budget_accepts_exact_model_demand_only_after_os_floor() {
        let required = 256 * 47 * 1_603_584u64;
        let floor = 8u64 << 30;
        let exact = HostFacts {
            mem_total: 40 << 30,
            mem_available: required + floor,
            cgroup_remaining: Some(required + floor),
            provenance: Provenance::Verified,
            process_rss: 0,
        };
        assert!(
            exact.admit_process_bytes(required, LABEL).is_ok(),
            "exact model demand plus OS floor must admit"
        );
        assert!(
            HostFacts {
                mem_available: exact.mem_available - 1,
                ..exact.clone()
            }
            .admit_process_bytes(required, LABEL)
            .is_err()
        );
        assert!(
            HostFacts {
                cgroup_remaining: Some(required),
                ..exact.clone()
            }
            .admit_process_bytes(required, LABEL)
            .is_ok(),
            "a cgroup that holds the payload is not asked to hold the OS floor as well"
        );
        assert!(
            HostFacts {
                cgroup_remaining: Some(required - 1),
                ..exact.clone()
            }
            .admit_process_bytes(required, LABEL)
            .is_err()
        );
        assert!(
            HostFacts {
                cgroup_remaining: None,
                ..exact.clone()
            }
            .admit_process_bytes(required, LABEL)
            .is_ok(),
            "an unlimited cgroup still obeys host availability and the OS floor"
        );
    }

    #[test]
    fn composed_host_probe_uses_verified_tree_and_fails_closed() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let proc = tempfile::tempdir().unwrap();
        let proc_dir = proc.path().canonicalize().unwrap();
        std::fs::create_dir(proc_dir.join("self")).unwrap();
        std::fs::write(
            proc_dir.join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc_dir.join("self/cgroup"), "0::/work\n").unwrap();
        std::fs::write(proc_dir.join("self/mountinfo"), mounts).unwrap();
        std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(group.join("memory.max"), "17179869184\n").unwrap();
        std::fs::write(group.join("memory.current"), "1073741824\n").unwrap();
        let budget = HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(15u64 << 30));
        assert!(budget.admit_process_bytes(15u64 << 30, LABEL).is_ok());
        assert!(
            budget
                .admit_process_bytes((15u64 << 30) + 1, LABEL)
                .is_err()
        );
        assert!(budget.admit(LABEL, 13u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (13u64 << 30) + 1).is_err());
        std::fs::remove_file(group.join("memory.current")).unwrap();
        assert!(HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).is_err());
        // The one read does not fail closed: the same unverifiable tree reads as the bare
        // figures with their provenance, so every caller of one machine gets one budget.
        let bare = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(bare.host_memory(), (32u64 << 30, 24u64 << 30));
        assert!(matches!(bare.provenance(), Provenance::Bare(_)));
    }

    /// Every cgroup layout the verifier refuses reads as the bare figures through `read_in`,
    /// where the old strict probe refused the start (Flash-Next, Flint) while the lenient one
    /// carried on (check, Bolt, Nemotron, Qwen3.6): one rule for all of them.
    #[test]
    fn a_tree_the_verifier_refuses_reads_as_the_bare_figures_never_as_a_refusal() {
        // A namespace root without the memory controller is refused by the verifier.
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "cpu\n");
        assert!(HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).is_err());
        let read = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert!(matches!(read.provenance(), Provenance::Bare(_)), "{read:?}");
        // The same root with the controller is verified, and its own limit applies.
        let (_root, _proc, directory, proc_dir) = namespace_root("8589934592\n", "memory\n");
        let read = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(read.provenance(), &Provenance::Verified);
        assert_eq!(read.budget(), 6u64 << 30);
        // A membership file that maps the process nowhere.
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "memory\n");
        std::fs::write(proc_dir.join("self/cgroup"), "1:net_cls:/\n").unwrap();
        assert!(HostFacts::verified_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).is_err());
        let read = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(read.budget(), 24u64 << 30, "the bare MemAvailable figure");
        assert!(matches!(read.provenance(), Provenance::Bare(_)));
        // An unreadable /proc/meminfo is the only refusal left.
        std::fs::remove_file(proc_dir.join("meminfo")).unwrap();
        assert!(HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).is_err());
    }

    /// A payload inside a binding cgroup keeps the process's own overhead back: the Flint expert
    /// cache charges only its payload, and under `MemoryMax=12G` a payload that fills the cgroup
    /// would be killed by the heap and CUDA context the process still grows while it loads.
    #[test]
    fn a_payload_under_a_twelve_gib_cap_keeps_the_process_overhead_back() {
        const FLINT: &str = "Flint expert cache";
        // 62 GiB host, 50 GiB available, a 12 GiB cgroup with 1.5 GiB of it already resident.
        let meminfo = "MemTotal: 65011712 kB\nMemAvailable: 52428800 kB\n";
        let mut host =
            HostFacts::from_meminfo(FLINT, meminfo, Some(12 * (1 << 30) - (3 << 29))).unwrap();
        host.process_rss = 3 << 29;
        assert_eq!(host.process_overhead(), (3 << 29) + (2 << 30));
        // 10.5 GiB remain in the cgroup; 3.5 GiB of overhead leave 7 GiB for the payload.
        assert_eq!(host.payload_spare(), 7 << 30);
        assert!(host.admit(FLINT, 7 << 30).is_ok());
        let refusal = host.admit(FLINT, 9 << 30).unwrap_err().describe();
        assert!(refusal.contains("Flint expert cache needs"), "{refusal}");
        // The same host without a cap charges the floor, as before.
        let uncapped = HostFacts::from_meminfo(FLINT, meminfo, None).unwrap();
        assert_eq!(uncapped.payload_spare(), (50u64 << 30) - 8 * (1 << 30));
    }

    /// The resident set comes from `<proc>/self/status`, and a missing file reads as zero.
    #[test]
    fn the_process_overhead_reads_the_resident_set() {
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "memory\n");
        std::fs::write(
            proc_dir.join("self/status"),
            "Name:\tx\nVmRSS:\t  2097152 kB\n",
        )
        .unwrap();
        let read = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(read.process_rss, 2 << 30);
        std::fs::remove_file(proc_dir.join("self/status")).unwrap();
        let read = HostFacts::read_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(read.process_rss, 0);
    }

    #[test]
    fn a_unified_memory_machine_spends_the_unified_budget_and_keeps_the_unified_floor() {
        const GIB: u64 = 1 << 30;
        // The Mac rule is the one in `unified`: a sixth of the memory, at least 4 GiB, is the
        // floor. A 16 GiB Mac keeps 4 GiB and spends 12; a 64 GiB Mac keeps its sixth.
        for physical in [8 * GIB, 16 * GIB, 36 * GIB, 64 * GIB, 192 * GIB] {
            let host = HostFacts::from_unified(physical);
            assert_eq!(host.provenance(), &Provenance::Unified);
            assert_eq!(host.total(), physical);
            assert_eq!(host.spare(), crate::family::mem::host_budget(physical));
            assert_eq!(host.os_floor() + host.spare(), physical);
            assert_eq!(host.kept_back(), host.os_floor());
        }
        let sixteen = HostFacts::from_unified(16 * GIB);
        assert_eq!(sixteen.os_floor(), 4 * GIB);
        // There is no cgroup: the payload admission spends the whole spare, less nothing.
        assert!(sixteen.admit(LABEL, 12 * GIB).is_ok());
        assert!(sixteen.admit(LABEL, 12 * GIB + 1).is_err());
        assert!(sixteen.provenance_note().contains("unified memory"));
    }
}
