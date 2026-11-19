//! Bounded disk-record envelope for shared-KV persistence.
//!
//! Host-only and UNWIRED: no configuration keys, no session or executor
//! integration, and deliberately none of the lease machinery — no lease, no
//! `ACCOUNTING` baseline, no recovery fence, no touches/retired/quarantine
//! directories. A root created here is therefore not yet a production root
//! for writes; that arrives when the lease machinery is wired. This module
//! exists so the manifest schema, publication order and O_DIRECT read path
//! can be qualified independently of it.
//!
//! Privacy: files carry prompt-derived state and exact tokens, so every
//! directory is created 0700 and every file 0600, and this module's own
//! miss reasons are opaque — no token text, no manifest strings, no paths.
//! Two non-promises: io::Error values from the kernel can still carry a
//! path some OTHER layer added (nothing here adds one), and the root's
//! owner can read the bytes by design — the modes keep everyone else out.

use std::{
    fmt,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use rustix::fs::OFlags;
use serde::{Deserialize, Serialize};

use super::host_tier::{ODirectRing, SECTOR, Slot};

/// The byte cap on one manifest before anything is parsed.
const MANIFEST_BYTE_CAP: u64 = 8 * 1024 * 1024;
/// Objects are 1..=1 GiB; larger snapshots must be split by a
/// registered adapter, and this envelope refuses to stage one read that
/// could not have been written.
const OBJECT_BYTE_MAX: u64 = 1_073_741_824;
/// One dependency may reference at most this many objects; the count is
/// checked before any file is opened.
const DEPENDENCY_MAX: usize = 256;
const PAGE_VERSION_MAX: usize = 65_536;
const FIXED_STATE_MAX: usize = 4_096;
const TOKEN_MAX: u64 = 262_144;
const LATTICE_MAX: usize = 262_144;
const STRING_BYTE_MAX: usize = 128;

/// Why a disk record did not produce a usable entry. Typed so a caller can
/// fall to the next candidate or stop, and never has to string-match an
/// error to learn which happened.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DiskMiss {
    /// The named object or entry does not exist under the root.
    Absent,
    /// The record belongs to an envelope, schema, ABI or version this
    /// build does not implement. Decided before any payload bytes are
    /// interpreted.
    Incompatible(String),
    /// The record exists but fails its own integrity contract — bad CRC,
    /// bad hash, torn staging, structural violation of the manifest.
    Corrupt(String),
    /// The environment refused the operation (O_DIRECT on this
    /// filesystem, a symlinked or foreign root, ENOSPC-style failures).
    Refused(String),
}

impl fmt::Display for DiskMiss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskMiss::Absent => f.write_str("absent"),
            DiskMiss::Incompatible(r) => write!(f, "incompatible: {r}"),
            DiskMiss::Corrupt(r) => write!(f, "corrupt: {r}"),
            DiskMiss::Refused(r) => write!(f, "refused: {r}"),
        }
    }
}

impl std::error::Error for DiskMiss {}

fn corrupt(reason: impl Into<String>) -> DiskMiss {
    DiskMiss::Corrupt(reason.into())
}

fn incompatible(reason: impl Into<String>) -> DiskMiss {
    DiskMiss::Incompatible(reason.into())
}

/// The format's H32/H64/H8 digests: exactly `$digits` lowercase hex digits,
/// stored as the `$bytes` they name. Three concrete types, not one
/// generic over digit count, because array-length arithmetic on const
/// generics needs unstable `generic_const_exprs`.
macro_rules! hex_digest {
    ($name:ident, $digits:literal, $bytes:literal) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) struct $name(pub(crate) [u8; $bytes]);

        impl $name {
            pub(crate) fn from_digest(bytes: [u8; $bytes]) -> Self {
                Self(bytes)
            }

            fn parse(text: &str) -> Result<Self, String> {
                if text.len() != $digits
                    || !text
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(format!("expected {} lowercase hex digits", $digits));
                }
                let mut out = [0u8; $bytes];
                for (i, chunk) in text.as_bytes().chunks(2).enumerate() {
                    let hi = (chunk[0] as char).to_digit(16).unwrap() as u8;
                    let lo = (chunk[1] as char).to_digit(16).unwrap() as u8;
                    out[i] = hi << 4 | lo;
                }
                Ok(Self(out))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for byte in self.0 {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_str(self)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                struct V;
                impl<'de> serde::de::Visitor<'de> for V {
                    type Value = $name;
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, "{} lowercase hex digits", $digits)
                    }
                    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                        $name::parse(v).map_err(E::custom)
                    }
                }
                d.deserialize_str(V)
            }
        }
    };
}

hex_digest!(H32, 32, 16);
hex_digest!(H64, 64, 32);
hex_digest!(H8, 8, 4);

/// CRC32C, reflected Castagnoli 0x82f63b78 with initial and final xor
/// 0xffffffff, printed as eight lowercase hex digits. Table over
/// the low byte keeps the loop cheap without a new dependency.
pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_resume(0xffff_ffff, bytes) ^ 0xffff_ffff
}

fn crc32c_resume(state: u32, bytes: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0x82f6_3b78 & ((crc & 1).wrapping_neg()));
            }
            *slot = crc;
        }
        table
    });
    let mut crc = state;
    for &byte in bytes {
        crc = table[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// The twelve identity fields in the fixed namespace order.
const IDENTITY_ORDER: [&str; 12] = [
    "model_digest",
    "weight_layout_digest",
    "tokenizer_digest",
    "template_digest",
    "adapter_id",
    "backend_id",
    "snapshot_schema",
    "payload_abi",
    "numeric_recipe_digest",
    "position_policy_digest",
    "render_mode",
    "kv_layout_digest",
];

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Identity {
    pub(crate) model_digest: H64,
    pub(crate) weight_layout_digest: H64,
    pub(crate) tokenizer_digest: H64,
    pub(crate) template_digest: H64,
    pub(crate) adapter_id: String,
    pub(crate) backend_id: String,
    pub(crate) snapshot_schema: String,
    pub(crate) payload_abi: String,
    pub(crate) numeric_recipe_digest: H64,
    pub(crate) position_policy_digest: H64,
    pub(crate) render_mode: String,
    pub(crate) kv_layout_digest: H64,
}

impl Identity {
    fn values_in_order(&self) -> [String; 12] {
        [
            self.model_digest.to_string(),
            self.weight_layout_digest.to_string(),
            self.tokenizer_digest.to_string(),
            self.template_digest.to_string(),
            self.adapter_id.clone(),
            self.backend_id.clone(),
            self.snapshot_schema.clone(),
            self.payload_abi.clone(),
            self.numeric_recipe_digest.to_string(),
            self.position_policy_digest.to_string(),
            self.render_mode.clone(),
            self.kv_layout_digest.to_string(),
        ]
    }

    fn check(&self) -> Result<(), DiskMiss> {
        for (i, value) in self.values_in_order().into_iter().enumerate() {
            let field = IDENTITY_ORDER[i];
            let is_digest = field.ends_with("_digest");
            if is_digest {
                continue;
            }
            if value.is_empty()
                || value.len() > STRING_BYTE_MAX
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-'))
            {
                return Err(incompatible(format!(
                    "identity field {field} is not a nonempty [A-Za-z0-9._:-] identifier of at most {STRING_BYTE_MAX} bytes"
                )));
            }
        }
        Ok(())
    }

    /// `SHA256("fifi-kv-namespace-v1" NUL || values…, each u32-LE-length
    /// prefixed)` over the fixed field order.
    pub(crate) fn namespace(&self) -> [u8; 32] {
        let mut buf = b"fifi-kv-namespace-v1\0".to_vec();
        for value in self.values_in_order() {
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value.as_bytes());
        }
        sha256(&buf)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveLattice {
    pub(crate) origin_q: u64,
    pub(crate) stride_q: u64,
    pub(crate) selected_q: Vec<u64>,
    pub(crate) skipped_q: Vec<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    pub(crate) capture_kind: String,
    pub(crate) numeric_recipe_digest: H64,
    pub(crate) actual_fed_cuts: Vec<u64>,
    pub(crate) save_lattice: SaveLattice,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Dependency {
    pub(crate) id: H64,
    pub(crate) file_bytes: u64,
    pub(crate) crc32c: H8,
    pub(crate) billing_engine: H64,
}

/// `ring_origin` must be PRESENT as either a u64 or JSON null; omitting
/// the field is rejected. `Option<Option<u64>>` cannot express that
/// (serde defaults any Option field to None when absent), so the field is
/// a plain untagged enum: absent → "missing field", `null` → `Null`,
/// a number → `Pos`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum RingOrigin {
    Null(()),
    Pos(u64),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PageVersion {
    pub(crate) layer: u32,
    pub(crate) dep: H64,
    pub(crate) byte_offset: u64,
    pub(crate) byte_length: u64,
    pub(crate) first_pos: u64,
    pub(crate) valid_rows: u64,
    pub(crate) page_rows: u64,
    pub(crate) tail_rows: u32,
    pub(crate) version: H32,
    pub(crate) ring_origin: RingOrigin,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FixedState {
    pub(crate) layer: u32,
    pub(crate) kind: String,
    pub(crate) dep: H64,
    pub(crate) byte_offset: u64,
    pub(crate) byte_length: u64,
    pub(crate) at_p: u64,
    pub(crate) schema: String,
}

/// Manifest version 1, field-for-field. Serialized with the
/// declaration order above, which is the manifest's canonical order; the
/// entry filename is the SHA256 of these exact bytes.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) format: String,
    pub(crate) format_version: u64,
    pub(crate) entry_id: H32,
    pub(crate) engine_id: H64,
    pub(crate) credential_scope: H32,
    pub(crate) namespace: H64,
    pub(crate) identity: Identity,
    pub(crate) created_unix_ms: u64,
    pub(crate) p: u64,
    pub(crate) q: u64,
    pub(crate) pending_token: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) block_size: u64,
    pub(crate) block_hashes: Vec<H64>,
    pub(crate) provenance: Provenance,
    pub(crate) dependencies: Vec<Dependency>,
    pub(crate) page_versions: Vec<PageVersion>,
    pub(crate) fixed_state: Vec<FixedState>,
}

impl std::fmt::Debug for Manifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Exact tokens never appear in a Debug dump: they are the payload.
        f.debug_struct("Manifest")
            .field("format", &self.format)
            .field("format_version", &self.format_version)
            .field("entry_id", &self.entry_id)
            .field("engine_id", &self.engine_id)
            .field("credential_scope", &self.credential_scope)
            .field("namespace", &self.namespace)
            .field("identity", &self.identity)
            .field("created_unix_ms", &self.created_unix_ms)
            .field("p", &self.p)
            .field("q", &self.q)
            .field("tokens", &format!("<{} redacted>", self.tokens.len()))
            .field("block_size", &self.block_size)
            .field("block_hashes", &self.block_hashes)
            .field("provenance", &self.provenance)
            .field("dependencies", &self.dependencies)
            .field("page_versions", &self.page_versions)
            .field("fixed_state", &self.fixed_state)
            .finish()
    }
}

impl Manifest {
    /// Strict parse: byte cap, BOM and trailing-content rejection happen
    /// before serde; `deny_unknown_fields`, the typed digests and the
    /// non-optional scalars make serde itself refuse unknown/extra fields,
    /// duplicate keys, floats and negative or overflowing numbers. Every
    /// cross-field rule then runs in [`Self::check`], before any
    /// caller opens a payload file.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, DiskMiss> {
        if bytes.len() as u64 > MANIFEST_BYTE_CAP {
            return Err(corrupt(format!(
                "manifest is {} bytes, over the {MANIFEST_BYTE_CAP} cap; rejected before parsing",
                bytes.len()
            )));
        }
        if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
            return Err(corrupt("manifest carries a UTF-8 BOM"));
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| corrupt("manifest is not valid UTF-8"))?;
        // Envelope peek before the strict parse: a manifest from a future
        // version may carry fields this schema has never heard of, and the
        // strict schema would read that as corruption. Peek permissively
        // first so a foreign envelope stays Incompatible — preserved, not
        // quarantined — whatever else its version changes.
        #[derive(Deserialize)]
        struct EnvelopePeek {
            format: String,
            format_version: u64,
        }
        if let Ok(peek) = serde_json::from_str::<EnvelopePeek>(text)
            && (peek.format != "fifi-kv-v3" || peek.format_version != 1)
        {
            return Err(incompatible(
                "the manifest belongs to another envelope version",
            ));
        }
        let raw: Manifest =
            serde_json::from_str(text).map_err(|_| corrupt(String::from("manifest rejected")))?;
        raw.check()?;
        Ok(raw)
    }

    /// The canonical serialized bytes: the write path stores exactly these
    /// and names the entry by their SHA256.
    pub(crate) fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("manifest serialization is infallible")
    }

    /// Block `i` hashes `"fifi-kv-block-v1" NUL || namespace || previous
    /// hash || u64-LE absolute offset || u32-LE count || u32-LE tokens`
    /// (32 zero bytes as the previous hash of block 0).
    pub(crate) fn block_hash(&self, i: usize) -> [u8; 32] {
        let mut buf = b"fifi-kv-block-v1\0".to_vec();
        buf.extend_from_slice(&self.identity.namespace());
        let prev = if i == 0 {
            [0u8; 32]
        } else {
            self.block_hashes[i - 1].0
        };
        buf.extend_from_slice(&prev);
        let start = (i * 256) as u64;
        let count = self.tokens[i * 256..].len().min(256) as u64;
        buf.extend_from_slice(&start.to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        for &token in &self.tokens[i * 256..i * 256 + count as usize] {
            buf.extend_from_slice(&token.to_le_bytes());
        }
        sha256(&buf)
    }

    fn dep_bytes(&self, id: &H64) -> Option<u64> {
        self.dependencies
            .iter()
            .find(|d| &d.id == id)
            .map(|d| d.file_bytes)
    }

    fn check(&self) -> Result<(), DiskMiss> {
        self.identity.check()?;
        if self.q
            != self
                .p
                .checked_add(1)
                .ok_or_else(|| corrupt("p+1 overflows"))?
        {
            return Err(corrupt("q is not p+1"));
        }
        if self.q == 0 || self.q > TOKEN_MAX {
            return Err(corrupt(format!("q={} is outside 1..={TOKEN_MAX}", self.q)));
        }
        if self.tokens.len() as u64 != self.q {
            return Err(corrupt(format!(
                "tokens lists {} IDs, but q={}",
                self.tokens.len(),
                self.q
            )));
        }
        if self.tokens.last().copied() != Some(self.pending_token) {
            return Err(corrupt("the final token is not the pending token"));
        }
        if self.block_size != 256 {
            return Err(corrupt("block_size is not the literal 256"));
        }
        let want_blocks = self.q.div_ceil(256) as usize;
        if self.block_hashes.len() != want_blocks {
            return Err(corrupt(format!(
                "{} block hashes for {} blocks",
                self.block_hashes.len(),
                want_blocks
            )));
        }
        if self.identity.namespace() != self.namespace.0 {
            return Err(corrupt("the namespace does not match the identity"));
        }
        for (i, stored) in self.block_hashes.iter().enumerate() {
            if self.block_hash(i) != stored.0 {
                return Err(corrupt(format!("block hash {i} does not verify")));
            }
        }
        self.check_provenance()?;
        if self.dependencies.is_empty() || self.dependencies.len() > DEPENDENCY_MAX {
            return Err(corrupt(format!(
                "manifest lists {} dependencies, outside 1..={DEPENDENCY_MAX}",
                self.dependencies.len()
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for dep in &self.dependencies {
            if !seen.insert(dep.id) {
                return Err(corrupt("a dependency object is listed twice"));
            }
            if dep.file_bytes == 0 || dep.file_bytes > OBJECT_BYTE_MAX {
                return Err(corrupt(format!(
                    "dependency {} is {} bytes, outside 1..={OBJECT_BYTE_MAX}",
                    dep.id, dep.file_bytes
                )));
            }
        }
        self.check_coverage()?;
        Ok(())
    }

    fn check_provenance(&self) -> Result<(), DiskMiss> {
        let p = &self.provenance;
        if !matches!(p.capture_kind.as_str(), "root" | "periodic" | "tip") {
            return Err(corrupt("capture_kind is not root|periodic|tip".to_string()));
        }
        if p.numeric_recipe_digest != self.identity.numeric_recipe_digest {
            return Err(corrupt(
                "provenance numeric_recipe_digest differs from identity",
            ));
        }
        if p.actual_fed_cuts.len() > LATTICE_MAX {
            return Err(corrupt("actual_fed_cuts exceeds the lattice cap"));
        }
        if self.p == 0 {
            if !p.actual_fed_cuts.is_empty() {
                return Err(corrupt("p=0 but actual_fed_cuts is nonempty"));
            }
        } else {
            let cuts = &p.actual_fed_cuts;
            if cuts.is_empty()
                || cuts.last().copied() != Some(self.p)
                || cuts.first().copied().unwrap_or(0) == 0
                || cuts.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(corrupt(
                    "actual_fed_cuts must be positive, strictly increasing and end at p",
                ));
            }
        }
        let lattice = &p.save_lattice;
        if lattice.origin_q == 0 || lattice.stride_q == 0 {
            return Err(corrupt("the save lattice has a zero origin or stride"));
        }
        let on_grid = |v: u64| -> bool {
            v >= lattice.origin_q
                && v <= self.q
                && (v - lattice.origin_q).is_multiple_of(lattice.stride_q)
        };
        for (name, list) in [
            ("selected_q", &lattice.selected_q),
            ("skipped_q", &lattice.skipped_q),
        ] {
            if list.len() > LATTICE_MAX {
                return Err(corrupt(format!("{name} exceeds the lattice cap")));
            }
            if list.windows(2).any(|w| w[0] >= w[1]) {
                return Err(corrupt(format!("{name} is not sorted distinct")));
            }
            if list.iter().any(|&v| !on_grid(v)) {
                return Err(corrupt(format!(
                    "{name} holds a value that is not a lattice point <= q"
                )));
            }
        }
        let mut all = lattice.selected_q.clone();
        all.extend(lattice.skipped_q.iter().copied());
        all.sort_unstable();
        if all.windows(2).any(|w| w[0] == w[1]) {
            return Err(corrupt("selected_q and skipped_q overlap"));
        }
        if p.capture_kind == "periodic" && !lattice.selected_q.contains(&self.q) {
            return Err(corrupt("a periodic capture must select q"));
        }
        Ok(())
    }

    fn check_coverage(&self) -> Result<(), DiskMiss> {
        if self.page_versions.len() > PAGE_VERSION_MAX {
            return Err(corrupt("page_versions exceeds its cap"));
        }
        if self.fixed_state.len() > FIXED_STATE_MAX {
            return Err(corrupt("fixed_state exceeds its cap"));
        }
        let mut referenced = std::collections::HashSet::new();
        for (i, pv) in self.page_versions.iter().enumerate() {
            let Some(file_bytes) = self.dep_bytes(&pv.dep) else {
                return Err(corrupt(format!(
                    "page version {i} names dependency {} outside the manifest",
                    pv.dep
                )));
            };
            let end = pv
                .byte_offset
                .checked_add(pv.byte_length)
                .filter(|&e| e <= file_bytes);
            if pv.byte_length == 0 || end.is_none() {
                // The bound is decided from the manifest ALONE, before any
                // file is opened: a lying length never stages a read.
                return Err(corrupt(format!(
                    "page version {i} range escapes dependency {} ({} bytes)",
                    pv.dep, file_bytes
                )));
            }
            if pv.page_rows != 256 || pv.valid_rows == 0 {
                return Err(corrupt("page_rows must be 256 and valid_rows positive"));
            }
            if pv.tail_rows as u64 != 1 + (pv.valid_rows - 1) % 256 {
                return Err(corrupt("tail_rows does not match valid_rows"));
            }
            referenced.insert(pv.dep);
        }
        for (i, fs) in self.fixed_state.iter().enumerate() {
            let Some(file_bytes) = self.dep_bytes(&fs.dep) else {
                return Err(corrupt(format!(
                    "fixed state {i} names dependency {} outside the manifest",
                    fs.dep
                )));
            };
            let end = fs
                .byte_offset
                .checked_add(fs.byte_length)
                .filter(|&e| e <= file_bytes);
            if fs.byte_length == 0 || end.is_none() {
                return Err(corrupt(format!(
                    "fixed state {i} range escapes dependency {} ({} bytes)",
                    fs.dep, file_bytes
                )));
            }
            if fs.at_p != self.p {
                return Err(corrupt("a fixed-state record is not at p"));
            }
            if !matches!(
                fs.kind.as_str(),
                "deltanet" | "mamba" | "convolution" | "adapter"
            ) {
                return Err(corrupt("fixed-state kind is unknown".to_string()));
            }
            if fs.schema.is_empty() || fs.schema.len() > STRING_BYTE_MAX {
                return Err(corrupt("fixed-state schema is not a bounded identifier"));
            }
            referenced.insert(fs.dep);
        }
        for dep in &self.dependencies {
            if !referenced.contains(&dep.id) {
                return Err(corrupt(format!(
                    "dependency {} is listed but never referenced by coverage",
                    dep.id
                )));
            }
        }
        Ok(())
    }

    /// Hashes accelerate lookup; they never replace the equality check
    ///. `true` only on exact stored-token equality.
    pub(crate) fn tokens_equal(&self, other: &[u32]) -> bool {
        self.tokens == other
    }
}

/// The root marker, exactly `{format, format_version, root_id,
/// credential_scope, billing_engine}` and at most 512 bytes on disk.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RootMarker {
    format: String,
    format_version: u64,
    root_id: H32,
    credential_scope: H32,
    billing_engine: H64,
}

/// An opened (or freshly created) bounded record root. Creation makes the
/// root, `objects/`, `entries/` and `staging/` 0700 and the marker 0600;
/// `touches/`, `retired/`, `quarantine/`, the lease and `ACCOUNTING` are
/// retirement machinery this module intentionally does not create.
pub(crate) struct DiskRoot {
    path: PathBuf,
    marker: RootMarker,
    /// The Unix UID that owns the root directory. Every file the root
    /// touches must carry it, and no group or other access bits.
    owner_uid: u32,
}

impl std::fmt::Debug for DiskRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The path never appears: a root location is operator-territory and
        // a Debug dump is not the place for it.
        f.debug_struct("DiskRoot")
            .field("marker", &self.marker)
            .field("owner_uid", &self.owner_uid)
            .finish_non_exhaustive()
    }
}

impl DiskRoot {
    /// Opens an existing root or creates one. A path that resolves through
    /// symlinks, or a marker whose credential scope does not match, is
    /// refused: a different credential may not adopt a root just because
    /// the Unix UID matches.
    pub(crate) fn open_or_create(
        path: &Path,
        credential_scope: H32,
        billing_engine: H64,
    ) -> Result<Self, DiskMiss> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            let existed = path.is_dir();
            if existed {
                // A symlinked root (or one under one) is not openable:
                // operations must be anchored to a real directory.
                let resolved = std::fs::canonicalize(path)
                    .map_err(|e| DiskMiss::Refused(format!("root unresolvable: {e}")))?;
                if resolved != path {
                    return Err(DiskMiss::Refused(
                        "the root path resolves through a symlink".into(),
                    ));
                }
            } else {
                let parent = path
                    .parent()
                    .ok_or_else(|| DiskMiss::Refused("root has no parent".into()))?;
                let mut builder = std::fs::DirBuilder::new();
                builder.mode(0o700).recursive(true);
                match builder.create(parent) {
                    Err(e) if e.kind() != io::ErrorKind::AlreadyExists => {
                        return Err(DiskMiss::Refused(format!("parent creation: {e}")));
                    }
                    _ => {}
                }
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(path)
                    .map_err(|e| DiskMiss::Refused(format!("root creation: {e}")))?;
                // The parent's entry for the new root must be durable too.
                fsync_dir(parent)?;
            }
            // Verify the existing root BEFORE creating anything inside it:
            // the owner, mode and link checks name a root this process may
            // use, not one it would widen by adding directories to it.
            use std::os::unix::fs::MetadataExt;
            let owner_uid = std::fs::symlink_metadata(path)
                .map_err(|e| DiskMiss::Refused(format!("root stat: {e}")))?
                .uid();
            if existed {
                owned_dir(path, owner_uid)?;
            }
            #[cfg(target_os = "linux")]
            if owner_uid != effective_uid() {
                return Err(DiskMiss::Refused("the root belongs to another user".into()));
            }
            // recursive tolerates an existing root's fixed directories (the
            // mode applies to what this call creates; an existing directory's
            // mode is not re-asserted).
            for dir in ["objects", "entries", "staging"] {
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .recursive(true)
                    .create(path.join(dir))
                    .map_err(|e| DiskMiss::Refused(format!("directory {dir}: {e}")))?;
            }
            for dir in ["objects", "entries", "staging"] {
                owned_dir(&path.join(dir), owner_uid)?;
            }
            let marker_path = path.join("CACHE.json");
            let marker = if marker_path.exists() {
                let file = open_owned(&marker_path, owner_uid)?;
                let meta = file
                    .metadata()
                    .map_err(|e| DiskMiss::Refused(format!("marker read: {e}")))?;
                if meta.len() > 512 {
                    return Err(corrupt("CACHE.json exceeds 512 bytes"));
                }
                let mut bytes = Vec::with_capacity(meta.len() as usize);
                (&file)
                    .take(513)
                    .read_to_end(&mut bytes)
                    .map_err(|e| DiskMiss::Refused(format!("marker read: {e}")))?;
                if bytes.len() > 512 {
                    return Err(corrupt("CACHE.json exceeds 512 bytes"));
                }
                let m: RootMarker = serde_json::from_slice(&bytes)
                    .map_err(|_| corrupt(String::from("CACHE.json rejected")))?;
                if m.format != "fifi-kv-root" || m.format_version != 1 {
                    return Err(incompatible("CACHE.json is not fifi-kv-root v1"));
                }
                m
            } else {
                let m = RootMarker {
                    format: "fifi-kv-root".into(),
                    format_version: 1,
                    root_id: fresh_h32(b"root"),
                    credential_scope,
                    billing_engine,
                };
                let bytes = serde_json::to_vec(&m).expect("marker serialization");
                // Through staging and rename, never in place: a torn in-place
                // CACHE.json would brick the root on every later open, while
                // a crash before the rename leaves only staging junk the
                // sweeper discards and the root opens as fresh. A fresh name
                // can collide only with leftovers from a torn attempt; a new
                // name is free, so retry instead of refusing the open.
                #[cfg(unix)]
                let staged = {
                    use std::os::unix::fs::OpenOptionsExt;
                    let mut written = None;
                    for _ in 0..3 {
                        let candidate = path
                            .join("staging")
                            .join(format!("{}.json", fresh_h32(b"mark")));
                        match std::fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .mode(0o600)
                            .open(&candidate)
                        {
                            Ok(mut f) => {
                                let outcome = f.write_all(&bytes).and_then(|()| f.sync_all());
                                if let Err(e) = outcome {
                                    let _ = std::fs::remove_file(&candidate);
                                    return Err(DiskMiss::Refused(format!("marker write: {e}")));
                                }
                                written = Some(candidate);
                                break;
                            }
                            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                            Err(e) => {
                                return Err(DiskMiss::Refused(format!("marker write: {e}")));
                            }
                        }
                    }
                    written.ok_or_else(|| {
                        DiskMiss::Refused("marker staging: three name collisions".into())
                    })?
                };
                // Linked, never renamed: a rename would silently replace a
                // marker a concurrent opener just created; a link fails
                // EEXIST and this opener loses cleanly instead.
                std::fs::hard_link(&staged, &marker_path).map_err(|e| {
                    let _ = std::fs::remove_file(&staged);
                    DiskMiss::Refused(format!("marker commit: {e}"))
                })?;
                fsync_dir(path)?;
                let _ = std::fs::remove_file(&staged);
                m
            };
            if marker.credential_scope != credential_scope {
                return Err(DiskMiss::Refused(
                    "the root belongs to a different credential scope".into(),
                ));
            }
            sweep_staging(path, owner_uid)?;
            Ok(Self {
                path: path.to_path_buf(),
                marker,
                owner_uid,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (path, credential_scope, billing_engine);
            Err(DiskMiss::Refused(
                "the disk record envelope is unix-only for now".into(),
            ))
        }
    }

    pub(crate) fn entry_path(&self, manifest_sha: &H64) -> PathBuf {
        self.path
            .join("entries")
            .join(format!("{manifest_sha}.json"))
    }

    pub(crate) fn object_path(&self, dep: &H64) -> PathBuf {
        self.path.join("objects").join(format!("{dep}.bin"))
    }

    /// Reads and parses the authoritative entry named by the manifest
    /// digest, refusing before any payload interpretation when the
    /// envelope is not one this build implements.
    pub(crate) fn read_entry(&self, manifest_sha: &H64) -> Result<Manifest, DiskMiss> {
        let path = self.entry_path(manifest_sha);
        let file = open_owned(&path, self.owner_uid)?;
        let meta = file
            .metadata()
            .map_err(|e| DiskMiss::Refused(format!("entry read: {e}")))?;
        if !meta.is_file() || meta.len() > MANIFEST_BYTE_CAP {
            return Err(corrupt(format!(
                "manifest is {} bytes, over the {MANIFEST_BYTE_CAP} cap; rejected before parsing",
                meta.len()
            )));
        }
        let mut bytes = Vec::with_capacity(meta.len() as usize);
        (&file)
            .take(MANIFEST_BYTE_CAP + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| DiskMiss::Refused(format!("entry read: {e}")))?;
        let manifest = Manifest::parse(&bytes)?;
        if sha256(&bytes) != manifest_sha.0 {
            return Err(corrupt(
                "the entry file name does not match the manifest bytes",
            ));
        }
        if manifest.credential_scope != self.marker.credential_scope {
            return Err(DiskMiss::Refused(
                "the entry belongs to a different credential scope".into(),
            ));
        }
        Ok(manifest)
    }
}

fn fsync_dir(dir: &Path) -> Result<(), DiskMiss> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)
            .and_then(|f| f.sync_all())
            .map_err(|e| DiskMiss::Refused(format!("directory sync failed: {e}")))
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// A fresh staging transaction id: an H32 of the payload identity, wall
/// clock and process — unique per attempt, carrying no user text.
fn fresh_h32(seed: &[u8]) -> H32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut buf = seed.to_vec();
    buf.extend_from_slice(&nanos.to_le_bytes());
    buf.extend_from_slice(&std::process::id().to_le_bytes());
    buf.extend_from_slice(
        &std::sync::atomic::AtomicU64::fetch_add(
            &STAGING_COUNTER,
            1,
            std::sync::atomic::Ordering::Relaxed,
        )
        .to_le_bytes(),
    );
    let mut id = [0u8; 16];
    id.copy_from_slice(&sha256(&buf)[..16]);
    H32::from_digest(id)
}

static STAGING_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The payloads one save wants to publish, keyed by dependency id.
pub(crate) struct PendingObject<'a> {
    pub(crate) id: H64,
    pub(crate) bytes: &'a [u8],
}

/// Publishes one entry exactly: stage and fsync every
/// object, verify the complete bytes against the manifest, install under
/// the content name without replacing anything, fsync `objects/`, stage
/// and fsync the manifest, rename it into `entries/`, then fsync both
/// directories. Any error abandons the save — staging is discarded, never
/// resumed — and nothing under `entries/` is ever overwritten in place.
/// Test-only write path: no lease, no accounting, and the name says so.
/// A production root is never written through this entry point; when the
/// wiring lands, the accounted writer is a separate function.
pub(crate) fn write_entry_unaccounted(
    root: &DiskRoot,
    manifest: &Manifest,
    objects: &[PendingObject<'_>],
) -> Result<H64, DiskMiss> {
    if objects.len() != manifest.dependencies.len() {
        return Err(corrupt(
            "the save carries a different object count than the manifest",
        ));
    }
    for dep in &manifest.dependencies {
        if dep.billing_engine != root.marker.billing_engine {
            return Err(DiskMiss::Refused(
                "a dependency is billed to a different engine than the root".into(),
            ));
        }
    }
    // The save validates exactly what its own reader will, before any
    // file moves: a manifest this build would reject must never reach
    // entries/, and an entry billed under another credential scope never
    // adopts this root (the read path enforces the same rule).
    manifest.check()?;
    if Manifest::parse(&manifest.canonical_bytes())? != *manifest {
        return Err(corrupt(
            "the manifest does not survive its canonical round trip",
        ));
    }
    if manifest.credential_scope != root.marker.credential_scope {
        return Err(DiskMiss::Refused(
            "the entry belongs to a different credential scope than the root".into(),
        ));
    }
    // Stage and fsync per object, then one objects/ sync after all installs.
    for dep in &manifest.dependencies {
        let Some(obj) = objects.iter().find(|o| o.id == dep.id) else {
            return Err(corrupt(format!(
                "payload for dependency {} is missing from the save",
                dep.id
            )));
        };
        if obj.bytes.len() as u64 != dep.file_bytes {
            return Err(corrupt(format!(
                "dependency {} carries {} bytes, the manifest says {}",
                dep.id,
                obj.bytes.len(),
                dep.file_bytes
            )));
        }
        let sha = sha256(obj.bytes);
        if sha != dep.id.0 {
            return Err(corrupt(format!(
                "dependency {}'s bytes do not hash to its name",
                dep.id
            )));
        }
        let crc = H8::from_digest(crc32c(obj.bytes).to_be_bytes());
        if crc != dep.crc32c {
            return Err(corrupt(format!(
                "dependency {}'s CRC32C does not match the manifest",
                dep.id
            )));
        }
        let installed = root.object_path(&dep.id);
        if regular_file(&installed, root.owner_uid)?.is_some() {
            // Content-addressed: an existing file with this name either
            // holds the same bytes (validated before reuse) or the root
            // is corrupt.
            verify_object_bytes(
                &installed,
                &dep.id,
                dep.file_bytes,
                dep.crc32c,
                root.owner_uid,
            )?;
            continue;
        }
        let staging = root
            .path
            .join("staging")
            .join(format!("{}.part", fresh_h32(b"obj")));
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&staging)
                .and_then(|mut f| {
                    f.write_all(obj.bytes)?;
                    f.sync_all()
                })
                .map_err(|e| {
                    // A failed write leaves a partial file behind: staging
                    // must be empty after any failure, not only success.
                    let _ = std::fs::remove_file(&staging);
                    DiskMiss::Refused(format!("staging write: {e}"))
                })?;
        }
        // No-replace install: a hard link makes the content name durable
        // atomically and fails EEXIST instead of clobbering a racing twin.
        match std::fs::hard_link(&staging, &installed) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(&staging);
                verify_object_bytes(
                    &installed,
                    &dep.id,
                    dep.file_bytes,
                    dep.crc32c,
                    root.owner_uid,
                )?;
                continue;
            }
            Err(e) => {
                let _ = std::fs::remove_file(&staging);
                return Err(DiskMiss::Refused(format!("object install: {e}")));
            }
        }
        let _ = std::fs::remove_file(&staging);
    }
    fsync_dir(&root.path.join("objects"))?;
    // The manifest in staging, fsynced, unpublished.
    let bytes = manifest.canonical_bytes();
    let entry_sha = H64::from_digest(sha256(&bytes));
    let staged = root
        .path
        .join("staging")
        .join(format!("{}.json", fresh_h32(b"man")));
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&staged)
            .and_then(|mut f| {
                f.write_all(&bytes)?;
                f.sync_all()
            })
            .map_err(|e| {
                let _ = std::fs::remove_file(&staged);
                DiskMiss::Refused(format!("manifest staging: {e}"))
            })?;
    }
    // The commit is the rename; then both directories are synced.
    std::fs::rename(&staged, root.entry_path(&entry_sha)).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        DiskMiss::Refused(format!("entry rename: {e}"))
    })?;
    fsync_dir(&root.path.join("entries"))?;
    fsync_dir(&root.path.join("staging"))?;
    Ok(entry_sha)
}

/// Completes or discards interrupted installs. A crash between an
/// object's hard link and its staging unlink leaves the content name with
/// two links plus a dead `.part`; anything else in `staging/` is a write
/// that never reached its link. Each leftover is unlinked from staging
/// only — never from `objects/` — so the content name returns to one link
/// and the root reopens instead of wedging behind the single-link rule.
///
/// A second opener while a first is writing ABORTS the first's live
/// save: this sweep unlinks its in-flight staging file, so the first
/// writer's install or rename fails and the save is refused. One writer
/// per root at a time is the contract until the lease machinery lands.
fn sweep_staging(path: &Path, owner_uid: u32) -> Result<(), DiskMiss> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    for entry in std::fs::read_dir(path.join("staging"))
        .map_err(|e| DiskMiss::Refused(format!("staging sweep: {e}")))?
    {
        let entry = entry.map_err(|e| DiskMiss::Refused(format!("staging sweep: {e}")))?;
        let meta = entry
            .metadata()
            .map_err(|e| DiskMiss::Refused(format!("staging sweep: {e}")))?;
        if !meta.is_file() || meta.uid() != owner_uid || meta.permissions().mode() & 0o077 != 0 {
            return Err(corrupt("staging holds a file this root does not own"));
        }
        std::fs::remove_file(entry.path())
            .map_err(|e| DiskMiss::Refused(format!("staging sweep: {e}")))?;
    }
    Ok(())
}

/// Opens one file of the root with `O_NOFOLLOW` and validates it on the
/// open descriptor: a symlink is refused by the open itself, and the uid,
/// mode, link count and regularity are read from the same file the caller
/// then reads bytes through. A path check cannot give this — a plain
/// `File::open` follows a symlink swapped in after it.
fn open_owned(path: &Path, owner_uid: u32) -> Result<std::fs::File, DiskMiss> {
    open_owned_with(path, owner_uid, OFlags::NOFOLLOW)
}

/// `open_owned` with `O_DIRECT` added for the ring readers: the same
/// descriptor rule, and the alignment the ring needs. The flag values come
/// from rustix, which carries the per-arch Linux tables — the literal values
/// differ between x86_64 and aarch64.
fn open_owned_direct(path: &Path, owner_uid: u32) -> Result<std::fs::File, DiskMiss> {
    // Off Linux the ring falls back to buffered reads, so only `O_NOFOLLOW`
    // applies.
    #[cfg(target_os = "linux")]
    let flags = OFlags::NOFOLLOW | OFlags::DIRECT;
    #[cfg(not(target_os = "linux"))]
    let flags = OFlags::NOFOLLOW;
    open_owned_with(path, owner_uid, flags)
}

fn open_owned_with(path: &Path, owner_uid: u32, flags: OFlags) -> Result<std::fs::File, DiskMiss> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(flags.bits() as i32)
        .open(path)
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => DiskMiss::Absent,
            _ => DiskMiss::Refused(String::from("open refused")),
        })?;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let meta = file
        .metadata()
        .map_err(|e| DiskMiss::Refused(format!("descriptor stat: {e}")))?;
    if !meta.is_file()
        || meta.nlink() != 1
        || meta.uid() != owner_uid
        || meta.permissions().mode() & 0o077 != 0
    {
        return Err(corrupt(String::from(
            "a file of the root is not a regular single-link owner-only file",
        )));
    }
    Ok(file)
}

/// The file is a regular single-link owner-only file (`None` when absent)
/// — symlinks, special files, group or other access bits, extra links and
/// foreign owners are refused before they are ever opened.
fn regular_file(path: &Path, owner_uid: u32) -> Result<Option<()>, DiskMiss> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if !meta.is_file()
                || meta.nlink() != 1
                || meta.uid() != owner_uid
                || meta.permissions().mode() & 0o077 != 0
            {
                return Err(corrupt(
                    "a file of the root is not a regular single-link owner-only file",
                ));
            }
            Ok(Some(()))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DiskMiss::Refused(format!("file stat: {e}"))),
    }
}

/// A directory this credential owns alone: not a symlink, no group or
/// other access bits, owned by `owner_uid`.
/// The effective uid, read from the kernel's own entry for this process;
/// `u32::MAX` when unreadable, so any owner comparison refuses.
#[cfg(target_os = "linux")]
fn effective_uid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self")
        .map(|meta| meta.uid())
        .unwrap_or(u32::MAX)
}

fn owned_dir(path: &Path, owner_uid: u32) -> Result<(), DiskMiss> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| DiskMiss::Refused(format!("directory stat: {e}")))?;
    if !meta.is_dir() || meta.uid() != owner_uid || meta.permissions().mode() & 0o077 != 0 {
        return Err(DiskMiss::Refused(
            "a directory of the root is not owner-only".into(),
        ));
    }
    Ok(())
}

/// Full-identity validation of an installed object: its exact bytes must
/// hash to its name and match the recorded length and CRC32C.
fn verify_object_bytes(
    path: &Path,
    id: &H64,
    file_bytes: u64,
    crc: H8,
    owner_uid: u32,
) -> Result<(), DiskMiss> {
    let file = open_owned(path, owner_uid)?;
    let meta = file
        .metadata()
        .map_err(|e| DiskMiss::Refused(format!("object read: {e}")))?;
    if meta.len() != file_bytes {
        return Err(corrupt(format!(
            "installed object {} is {} bytes on disk, the manifest says {file_bytes}",
            id,
            meta.len()
        )));
    }
    let mut bytes = Vec::with_capacity(file_bytes as usize);
    (&file)
        .take(file_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| DiskMiss::Refused(format!("object read: {e}")))?;
    if bytes.len() as u64 != file_bytes
        || sha256(&bytes) != id.0
        || H8::from_digest(crc32c(&bytes).to_be_bytes()) != crc
    {
        return Err(corrupt(format!(
            "installed object {} fails its length, hash or CRC",
            id
        )));
    }
    Ok(())
}

/// Stages the exact wanted bytes of one dependency into a ring slot via
/// the shared O_DIRECT reader, and returns where they landed inside the
/// slot. The bytes are NOT identity-checked here: the manifest's hash and
/// CRC32C cover the whole file, so a span cannot be verified in isolation
/// — callers treat what lands in the slot as unverified until the owning
/// restore path validates the state it builds from it. Padding beyond EOF
/// is tolerated; the wanted span is not. The open itself carries the
/// shape check — regular, single-link, owner-only, no symlink — on the
/// descriptor, and a file shorter than the manifest says is `Corrupt`.
pub(crate) fn read_unverified_span(
    root: &DiskRoot,
    dep: &Dependency,
    file_offset: u64,
    len: u64,
    ring: &mut ODirectRing,
    slot: Slot,
    stage_offset: u64,
) -> Result<u64, DiskMiss> {
    // A hand-built Dependency never passed the manifest parser, so the
    // byte cap is re-asserted here before aligned_span can see the value.
    if dep.file_bytes == 0 || dep.file_bytes > OBJECT_BYTE_MAX {
        return Err(corrupt(format!(
            "dependency {} is {} bytes, outside 1..={OBJECT_BYTE_MAX}",
            dep.id, dep.file_bytes
        )));
    }
    let end = file_offset
        .checked_add(len)
        .filter(|&e| e <= dep.file_bytes);
    if len == 0 || end.is_none() {
        return Err(corrupt(format!(
            "the requested span escapes dependency {} ({} bytes)",
            dep.id, dep.file_bytes
        )));
    }
    let path = root.object_path(&dep.id);
    let file = open_owned_direct(&path, root.owner_uid)?;
    let actual_len = file
        .metadata()
        .map_err(|e| DiskMiss::Refused(format!("descriptor stat: {e}")))?
        .len();
    if actual_len < dep.file_bytes {
        return Err(corrupt(format!(
            "object {} is {actual_len} bytes on disk, the manifest says {} — truncated",
            dep.id, dep.file_bytes
        )));
    }
    ring.stage_span(&file, file_offset, len, slot, stage_offset)
        .map_err(|e| DiskMiss::Refused(format!("ring stage: {e}")))
}

/// Verifies one dependency end-to-end through the O_DIRECT ring — the
/// same read path a restore will use — by streaming its exact bytes in
/// sector-sized chunks and checking SHA256 and CRC32C against the
/// manifest. A flipped bit is a typed `Corrupt` miss, decided before any
/// caller interprets the payload.
pub(crate) fn verify_dependency(
    root: &DiskRoot,
    dep: &Dependency,
    ring: &mut ODirectRing,
    slot: Slot,
) -> Result<(), DiskMiss> {
    use sha2::Digest;
    // Same re-assertion as read_unverified_span: the parser's cap does
    // not bind a hand-built Dependency.
    if dep.file_bytes == 0 || dep.file_bytes > OBJECT_BYTE_MAX {
        return Err(corrupt(format!(
            "dependency {} is {} bytes, outside 1..={OBJECT_BYTE_MAX}",
            dep.id, dep.file_bytes
        )));
    }
    let path = root.object_path(&dep.id);
    let file = open_owned_direct(&path, root.owner_uid)?;
    let len = file
        .metadata()
        .map_err(|e| DiskMiss::Refused(format!("object read: {e}")))?
        .len();
    if len != dep.file_bytes {
        return Err(corrupt(format!(
            "object {} is {len} bytes on disk, the manifest says {} — truncated or padded",
            dep.id, dep.file_bytes
        )));
    }
    let mut hasher = sha2::Sha256::new();
    let mut crc_state = 0xffff_ffff;
    let mut done: u64 = 0;
    // An aligned span can be up to SECTOR bytes longer than the wanted
    // bytes (round-down at the front, round-up at the back), so every
    // chunk must leave one sector of headroom inside the slot.
    let chunk = ring.slot_span().saturating_sub(SECTOR);
    if chunk == 0 {
        return Err(DiskMiss::Refused(format!(
            "the {}-byte slot span cannot stage a sector-aligned read",
            ring.slot_span()
        )));
    }
    while done < dep.file_bytes {
        let want = (dep.file_bytes - done).min(chunk);
        let in_slot = ring
            .stage_span(&file, done, want, slot, 0)
            .map_err(|e| DiskMiss::Refused(format!("ring stage: {e}")))?;
        let window: &[u8] = ring
            .slot_range(slot, in_slot, want)
            .map_err(|e| corrupt(format!("ring window: {e}")))?;
        hasher.update(window);
        crc_state = crc32c_resume(crc_state, window);
        done += want;
    }
    let crc_ok = (crc32c_resume(crc_state, &[]) ^ 0xffff_ffff) == u32::from_be_bytes(dep.crc32c.0);
    if hasher.finalize()[..] != dep.id.0 || !crc_ok {
        return Err(corrupt(format!(
            "object {} fails its hash or CRC32C on read-back",
            dep.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::{NonZeroU64, NonZeroUsize};

    /// The format's complete published fixture, byte for byte. Every
    /// golden digest in it was cross-checked against an independent
    /// implementation.
    const FIXTURE: &str = r#"{
  "format": "fifi-kv-v3", "format_version": 1,
  "entry_id": "11111111111111111111111111111111",
  "engine_id": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "credential_scope": "cccccccccccccccccccccccccccccccc",
  "namespace": "94671bc0719914c52026961adefbed2cc996d5f3e37559228154fa4a2bd2634a",
  "identity": {
    "model_digest": "a2180e3331e72c839099bc3ac48fca5bc8a0c258ed69b50165228f0ca72ae9d7",
    "weight_layout_digest": "3edc15b240a139ce8efead621406ffe93ea68a9b0b6a445887fe911cde881a91",
    "tokenizer_digest": "d33d4209cfd6937ce44ecbf698b88fbb6622fd4ccf125bb1a4be9a61f16d29a8",
    "template_digest": "941e36cd5fcfb959b89b9c93dec823584963d237fe77182730ac19f59e4c758d",
    "adapter_id": "fixture.attention", "backend_id": "fixture.cpu",
    "snapshot_schema": "fixture.complete.v1", "payload_abi": "fixture.kv-u8.v1",
    "numeric_recipe_digest": "c41f2c4c8dc38107aa91580bd926ddfe8f3a36092f444119d166ff1b9e14e26b",
    "position_policy_digest": "d778dc9b0c20462b4b5d9ed616f86dda0913beb3dd29367332c64f34eac3e040",
    "render_mode": "completion",
    "kv_layout_digest": "85d7c1ddc144b66c13025c209d2d849e0f2b64d3e422ce5072d7240e4cb85e43"
  },
  "created_unix_ms": 1700000000000,
  "p": 2, "q": 3, "pending_token": 13, "tokens": [7, 11, 13],
  "block_size": 256,
  "block_hashes": ["47c2265873ac4f52a1fe0a0cb5f89379c43ed7b0e29b3011dff9c914561d7fc7"],
  "provenance": {
    "capture_kind": "periodic",
    "numeric_recipe_digest": "c41f2c4c8dc38107aa91580bd926ddfe8f3a36092f444119d166ff1b9e14e26b",
    "actual_fed_cuts": [2],
    "save_lattice": {"origin_q": 3, "stride_q": 256, "selected_q": [3], "skipped_q": []}
  },
  "dependencies": [{
    "id": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    "file_bytes": 4, "crc32c": "48674bc7",
    "billing_engine": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
  }],
  "page_versions": [{
    "layer": 0, "dep": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    "byte_offset": 0, "byte_length": 4, "first_pos": 0, "valid_rows": 2,
    "page_rows": 256, "tail_rows": 2, "version": "22222222222222222222222222222222", "ring_origin": null
  }],
  "fixed_state": []
}"#;

    fn fixture() -> Manifest {
        Manifest::parse(FIXTURE.as_bytes()).expect("the fixture must parse")
    }

    fn scope() -> H32 {
        H32::parse(&"c".repeat(32)).unwrap()
    }

    fn engine() -> H64 {
        H64::parse(&"e".repeat(64)).unwrap()
    }

    /// A sector-aligned leaked buffer, as the production ring memory
    /// (cudaHostAlloc/mmap) would be.
    fn pinned(len: usize) -> &'static mut [u8] {
        let slop = SECTOR as usize - 1;
        let box_ = Box::leak(vec![0u8; len + slop].into_boxed_slice());
        let base = box_.as_ptr() as usize;
        let off = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
        &mut box_[off..off + len]
    }

    fn one_ring() -> ODirectRing {
        ODirectRing::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(2 * SECTOR).unwrap(),
            pinned((2 * SECTOR) as usize),
        )
        .unwrap()
    }

    fn assert_corrupt(bytes: &[u8], needle: &str) {
        match Manifest::parse(bytes) {
            Err(DiskMiss::Corrupt(r)) => {
                assert!(r.contains(needle), "reason {r:?} lacks {needle:?}")
            }
            other => panic!("expected Corrupt({needle:?}), got {other:?}"),
        }
    }

    #[test]
    fn crc32c_matches_the_reference_vectors() {
        assert_eq!(crc32c(&[0, 0, 0, 0]), 0x4867_4bc7);
        // The canonical CRC32C check value.
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(&[]), 0);
    }

    #[test]
    fn the_published_fixture_parses_and_every_digest_verifies() {
        let m = fixture();
        assert_eq!(m.identity.namespace(), m.namespace.0);
        assert_eq!(m.block_hash(0), m.block_hashes[0].0);
        assert_eq!(
            m.dependencies[0].id.to_string(),
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
        );
        // The dependency object IS the four zero bytes: its name hashes.
        assert_eq!(sha256(&[0, 0, 0, 0]), m.dependencies[0].id.0);
        assert_eq!(
            H8::from_digest(crc32c(&[0, 0, 0, 0]).to_be_bytes()),
            m.dependencies[0].crc32c
        );
    }

    #[test]
    fn canonical_bytes_round_trip_and_the_entry_name_is_their_sha() {
        let m = fixture();
        let bytes = m.canonical_bytes();
        let again = Manifest::parse(&bytes).expect("canonical bytes re-parse");
        assert_eq!(again, m);
        let name = H64::from_digest(sha256(&bytes));
        assert_eq!(name.to_string().len(), 64);
    }

    #[test]
    fn strict_parsing_rejects_before_any_payload_work() {
        assert_corrupt(
            format!("\u{feff}{}", FIXTURE.trim_start()).as_bytes(),
            "BOM",
        );
        assert_corrupt(format!("{FIXTURE} trailing").as_bytes(), "rejected");
        // Duplicate keys: serde's derived struct visitor rejects them.
        assert_corrupt(
            FIXTURE
                .replace(
                    "\"tokens\": [7, 11, 13],",
                    "\"tokens\": [7, 11, 13], \"tokens\": [7, 11, 13],",
                )
                .as_bytes(),
            "rejected",
        );
        // Floats never stand in for integers.
        assert_corrupt(
            FIXTURE.replace("\"q\": 3", "\"q\": 3.0").as_bytes(),
            "rejected",
        );
        assert_corrupt(
            FIXTURE
                .replace(
                    "\"created_unix_ms\": 1700000000000,",
                    "\"created_unix_ms\": 1700000000000, \"surprise\": 1,",
                )
                .as_bytes(),
            "rejected",
        );
        assert_corrupt(
            FIXTURE
                .replace("\"created_unix_ms\": 1700000000000,", "")
                .as_bytes(),
            "rejected",
        );
        // Uppercase hex is not the canonical form.
        assert_corrupt(
            FIXTURE.replace("48674bc7", "48674BC7").as_bytes(),
            "rejected",
        );
        // ring_origin must be present-and-null, not absent.
        assert_corrupt(
            FIXTURE.replace(", \"ring_origin\": null", "").as_bytes(),
            "rejected",
        );
        // Structural cross-field rules.
        assert_corrupt(
            FIXTURE
                .replace("\"p\": 2, \"q\": 3", "\"p\": 2, \"q\": 4")
                .as_bytes(),
            "q is not p+1",
        );
        assert_corrupt(
            FIXTURE
                .replace("\"tail_rows\": 2", "\"tail_rows\": 3")
                .as_bytes(),
            "tail_rows",
        );
        assert_corrupt(
            FIXTURE
                .replace("\"actual_fed_cuts\": [2]", "\"actual_fed_cuts\": [1]")
                .as_bytes(),
            "end at p",
        );
        // The namespace must match the identity.
        assert_corrupt(
            FIXTURE
                .replace("94671bc0719914c5", "94671bc0719914c6")
                .as_bytes(),
            "namespace",
        );
    }

    /// Structural rules easier to reach by mutating a valid manifest than
    /// by string surgery on the fixture.
    #[test]
    fn structural_rules_reject_after_parse() {
        // A dependency nothing references is not publishable coverage.
        let mut m = fixture();
        m.page_versions.clear();
        assert_corrupt(&m.canonical_bytes(), "never referenced");
        // A lying byte_length is decided from the manifest alone, before
        // any file is opened.
        let mut m = fixture();
        m.page_versions[0].byte_length = 8;
        assert_corrupt(&m.canonical_bytes(), "escapes");
        // A block hash that does not match its tokens is corruption.
        let mut m = fixture();
        m.block_hashes[0].0[0] ^= 1;
        assert_corrupt(&m.canonical_bytes(), "block hash");
        // A save point off the lattice grid is refused.
        let mut m = fixture();
        m.provenance.save_lattice.selected_q = vec![4];
        assert_corrupt(&m.canonical_bytes(), "lattice");
    }

    #[test]
    fn foreign_envelopes_are_incompatible_not_corrupt() {
        for (text, why) in [
            (FIXTURE.replace("fifi-kv-v3", "fifi-kv-v9"), "envelope"),
            (
                FIXTURE.replace("\"format_version\": 1", "\"format_version\": 2"),
                "envelope",
            ),
        ] {
            match Manifest::parse(text.as_bytes()) {
                Err(DiskMiss::Incompatible(r)) => assert!(r.contains(why), "{r}"),
                other => panic!("expected Incompatible, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_future_envelope_stays_incompatible_whatever_fields_it_adds() {
        // A version-2 manifest may carry a field this schema has never
        // heard of; without the permissive envelope peek the strict schema
        // reads that as corruption and the record would be quarantined
        // instead of preserved for the newer build.
        let text = FIXTURE
            .replace("\"format_version\": 1", "\"format_version\": 2")
            .replace(
                "\"created_unix_ms\": 1700000000000,",
                "\"created_unix_ms\": 1700000000000, \"v2_field\": true,",
            );
        match Manifest::parse(text.as_bytes()) {
            Err(DiskMiss::Incompatible(r)) => assert!(r.contains("envelope"), "{r}"),
            other => panic!("expected Incompatible, got {other:?}"),
        }
    }

    #[test]
    fn a_save_validates_before_it_publishes_anything() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let payload = [PendingObject {
            id: fixture().dependencies[0].id,
            bytes: &[0, 0, 0, 0],
        }];
        let staging_empty = || {
            std::fs::read_dir(root.path.join("staging"))
                .unwrap()
                .count()
                == 0
        };
        // A manifest the save's own reader would reject (the tokens no
        // longer hash to the stored block hash) must never be published.
        let mut lying = fixture();
        lying.tokens[0] ^= 1;
        assert!(write_entry_unaccounted(&root, &lying, &payload).is_err());
        assert!(staging_empty(), "a refused save left staging residue");
        // An entry billed under another credential scope never adopts the
        // root, exactly as the read path refuses it.
        let mut foreign = fixture();
        foreign.credential_scope = H32::parse(&"d".repeat(32)).unwrap();
        assert!(write_entry_unaccounted(&root, &foreign, &payload).is_err());
        assert!(staging_empty(), "a refused save left staging residue");
        let name = H64::from_digest(sha256(&foreign.canonical_bytes()));
        assert!(!root.entry_path(&name).exists());
    }

    #[test]
    fn publication_follows_its_four_step_order() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        let name = write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        // The publication order leaves no staging residue, exactly one
        // object, one entry.
        assert!(
            std::fs::read_dir(root.path.join("staging"))
                .unwrap()
                .next()
                .is_none()
        );
        let objects: Vec<_> = std::fs::read_dir(root.path.join("objects"))
            .unwrap()
            .collect();
        assert_eq!(objects.len(), 1);
        assert!(root.object_path(&m.dependencies[0].id).is_file());
        assert!(root.entry_path(&name).is_file());
        // The published entry re-reads and re-verifies.
        assert_eq!(root.read_entry(&name).unwrap(), m);
        // Re-publication of the identical save is a no-op that still succeeds.
        write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let objects: Vec<_> = std::fs::read_dir(root.path.join("objects"))
            .unwrap()
            .collect();
        assert_eq!(objects.len(), 1, "content-named objects never duplicate");
    }

    #[test]
    fn payloads_that_do_not_match_the_manifest_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        let wrong = write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[1, 2, 3, 4],
            }],
        );
        match wrong {
            Err(DiskMiss::Corrupt(r)) => assert!(r.contains("hash"), "{r}"),
            other => panic!("expected a hash refusal, got {other:?}"),
        }
        // Nothing leaked into the published namespaces.
        assert!(
            std::fs::read_dir(root.path.join("entries"))
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn published_objects_verify_through_the_o_direct_ring() {
        let dir = tempfile::tempdir().unwrap();
        // tmpfs and some container filesystems refuse O_DIRECT outright; like
        // host_tier's ring test, skip with a printed note where the kernel
        // will not serve the mode instead of failing on environment.
        #[cfg(target_os = "linux")]
        {
            use rustix::fs::OFlags;
            use std::os::unix::fs::OpenOptionsExt;
            let probe = dir.path().join("direct-probe");
            std::fs::write(&probe, vec![0u8; 8192]).unwrap();
            if std::fs::OpenOptions::new()
                .read(true)
                .custom_flags((OFlags::DIRECT).bits() as i32)
                .open(&probe)
                .is_err()
            {
                eprintln!(
                    "direct-read test: O_DIRECT refused on {}; the byte-exact read skips",
                    dir.path().display()
                );
                return;
            }
        }
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        verify_dependency(&root, &m.dependencies[0], &mut ring, slot).unwrap();
        ring.abandon(slot).unwrap();
        // A flipped bit is a typed miss decided before any decode.
        let path = root.object_path(&m.dependencies[0].id);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[2] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        match verify_dependency(&root, &m.dependencies[0], &mut ring, slot) {
            Err(DiskMiss::Corrupt(r)) => assert!(r.contains("CRC") || r.contains("hash"), "{r}"),
            other => panic!("expected corruption, got {other:?}"),
        }
        // And an absent object is Absent, not an error to decode.
        std::fs::remove_file(&path).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        assert_eq!(
            verify_dependency(&root, &m.dependencies[0], &mut ring, slot),
            Err(DiskMiss::Absent)
        );
    }

    #[test]
    fn a_fresh_root_creates_directories_and_files_owner_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&root.path), 0o700);
            for d in ["objects", "entries", "staging"] {
                assert_eq!(mode(&root.path.join(d)), 0o700, "{d}");
            }
            assert_eq!(mode(&root.path.join("CACHE.json")), 0o600);
        }
        // A published entry and its object land owner-only as well.
        let m = fixture();
        let name = write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&root.object_path(&m.dependencies[0].id)), 0o600);
            assert_eq!(mode(&root.entry_path(&name)), 0o600);
        }
    }

    #[test]
    fn a_torn_published_manifest_is_corrupt_not_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        let name = H64::from_digest(sha256(&m.canonical_bytes()));
        let bytes = m.canonical_bytes();
        let torn = &bytes[..bytes.len() / 2];
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(root.entry_path(&name), torn).unwrap();
            std::fs::set_permissions(
                root.entry_path(&name),
                std::fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        match root.read_entry(&name) {
            Err(DiskMiss::Corrupt(_)) => {}
            other => panic!("expected Corrupt for a torn manifest, got {other:?}"),
        }
    }

    #[test]
    fn a_crc_only_mismatch_in_a_placed_manifest_is_corrupt() {
        // The object hashes to its id but the placed manifest carries a
        // wrong CRC32C: the write path refuses such a pair, so this file
        // can only arrive by placement; the read path must name it Corrupt.
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let mut lying = m.clone();
        lying.dependencies[0].crc32c = H8::parse(&"f".repeat(8)).unwrap();
        let lying_name = H64::from_digest(sha256(&lying.canonical_bytes()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(root.entry_path(&lying_name), lying.canonical_bytes()).unwrap();
            std::fs::set_permissions(
                root.entry_path(&lying_name),
                std::fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        root.read_entry(&lying_name).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        match verify_dependency(&root, &lying.dependencies[0], &mut ring, slot) {
            Err(DiskMiss::Corrupt(r)) => assert!(r.contains("CRC"), "{r}"),
            other => panic!("expected a CRC corruption, got {other:?}"),
        }
    }

    #[test]
    fn staging_leftovers_stay_invisible() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        // A torn staging manifest naming an entry that "exists" only in
        // staging must not be readable.
        let m = fixture();
        let staged_name = H64::from_digest(sha256(&m.canonical_bytes()));
        std::fs::write(
            root.path
                .join("staging")
                .join(format!("{staged_name}.json")),
            m.canonical_bytes(),
        )
        .unwrap();
        assert_eq!(root.read_entry(&staged_name), Err(DiskMiss::Absent));
    }

    #[test]
    fn a_foreign_credential_cannot_adopt_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let root_path = dir.path().join("kv");
        DiskRoot::open_or_create(&root_path, scope(), engine()).unwrap();
        let other = H32::parse(&"d".repeat(32)).unwrap();
        match DiskRoot::open_or_create(&root_path, other, engine()) {
            Err(DiskMiss::Refused(r)) => assert!(r.contains("credential"), "{r}"),
            other => panic!("expected a scope refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_second_opener_sweeps_a_live_writers_staging_file() {
        // The documented contract: the sweep at open removes even a file
        // a first writer still considers in flight, aborting that save.
        let dir = tempfile::tempdir().unwrap();
        let root_path = dir.path().join("kv");
        let writer = DiskRoot::open_or_create(&root_path, scope(), engine()).unwrap();
        let in_flight = root_path.join("staging").join("live.0.part");
        std::fs::write(&in_flight, b"partial").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&in_flight, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        DiskRoot::open_or_create(&root_path, scope(), engine()).unwrap();
        assert!(!in_flight.exists());
        drop(writer);
    }

    #[test]
    fn a_symlinked_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("kv");
        DiskRoot::open_or_create(&real, scope(), engine()).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        match DiskRoot::open_or_create(&link, scope(), engine()) {
            Err(DiskMiss::Refused(r)) => assert!(r.contains("symlink"), "{r}"),
            other => panic!("expected a symlink refusal, got {other:?}"),
        }
    }

    #[test]
    fn reopening_completes_an_interrupted_object_install() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        // The crash window by hand: the staging file exists and the content
        // name is its second link, so the object sits at nlink 2 and the
        // single-link rule poisons it.
        let staging = root.path.join("staging").join("torn.part");
        {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&staging)
                .and_then(|mut f| std::io::Write::write_all(&mut f, &[0, 0, 0, 0]))
                .unwrap();
        }
        let installed = root.object_path(&m.dependencies[0].id);
        std::fs::hard_link(&staging, &installed).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        assert!(verify_dependency(&root, &m.dependencies[0], &mut ring, slot).is_err());
        ring.abandon(slot).unwrap();
        // Reopen sweeps staging: the leftover disappears and the content
        // name returns to one link with the right bytes.
        let root = DiskRoot::open_or_create(&root.path, scope(), engine()).unwrap();
        assert!(
            std::fs::read_dir(root.path.join("staging"))
                .unwrap()
                .next()
                .is_none()
        );
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        verify_dependency(&root, &m.dependencies[0], &mut ring, slot).unwrap();
        ring.abandon(slot).unwrap();
    }

    #[test]
    fn a_truncated_or_padded_object_is_corrupt_not_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let object = root.object_path(&m.dependencies[0].id);
        for (label, bytes) in [("truncated", &[][..]), ("padded", &[0u8, 0, 0, 0, 0][..])] {
            std::fs::write(&object, bytes).unwrap();
            let mut ring = one_ring();
            let slot = ring.acquire().unwrap();
            match verify_dependency(&root, &m.dependencies[0], &mut ring, slot) {
                Err(DiskMiss::Corrupt(r)) => {
                    assert!(r.contains("truncated or padded"), "{label}: {r}")
                }
                other => panic!("{label}: expected Corrupt, got {other:?}"),
            }
            ring.abandon(slot).unwrap();
        }
    }

    #[test]
    fn token_equality_is_exact_not_hashed() {
        let m = fixture();
        assert!(m.tokens_equal(&[7, 11, 13]));
        assert!(!m.tokens_equal(&[7, 11, 14]));
    }

    #[test]
    fn a_symlink_swapped_for_a_published_file_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        let name = write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        // The entry, replaced by a symlink to a byte-perfect copy of
        // itself: the path check refuses before the open, so the swap never
        // reads, even though following the link would "work".
        let decoy = dir.path().join("decoy.json");
        std::fs::write(&decoy, m.canonical_bytes()).unwrap();
        // A perfectly acceptable target (owner-only, single link, exact
        // bytes): only its symlink-ness can cause the refusal, so the
        // test fails if the O_NOFOLLOW defence is ever removed.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&decoy, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let entry = root.entry_path(&name);
        std::fs::remove_file(&entry).unwrap();
        std::os::unix::fs::symlink(&decoy, &entry).unwrap();
        match root.read_entry(&name) {
            // The open itself refuses a symlink now (O_NOFOLLOW), so the
            // swap never reaches the shape checks that named it a
            // non-regular file; any refusal is the defence.
            Err(_) => {}
            other => panic!("expected a refusal, got {other:?}"),
        }
        // The object, likewise: a symlink to the correct object bytes is
        // still refused, and the refusal precedes any hash pass.
        let object = root.object_path(&m.dependencies[0].id);
        let decoy_object = dir.path().join("decoy.bin");
        std::fs::write(&decoy_object, [0u8; 4]).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&decoy_object, std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        std::fs::remove_file(&object).unwrap();
        std::os::unix::fs::symlink(&decoy_object, &object).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        match verify_dependency(&root, &m.dependencies[0], &mut ring, slot) {
            // The open refuses a symlink (O_NOFOLLOW) before any hash pass.
            Err(_) => {}
            other => panic!("expected a refusal, got {other:?}"),
        }
        ring.abandon(slot).unwrap();
    }

    #[test]
    fn a_dependency_over_the_cap_is_refused_at_parse() {
        let oversize = FIXTURE.replace("\"file_bytes\": 4", "\"file_bytes\": 1073741825");
        assert_ne!(oversize, FIXTURE, "the surgery must land");
        match Manifest::parse(oversize.as_bytes()) {
            Err(DiskMiss::Corrupt(r)) => assert!(
                r.contains("outside 1..=1073741824") && r.contains("1073741825"),
                "{r}"
            ),
            other => panic!("expected a cap refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_group_readable_published_object_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let root = DiskRoot::open_or_create(&dir.path().join("kv"), scope(), engine()).unwrap();
        let m = fixture();
        write_entry_unaccounted(
            &root,
            &m,
            &[PendingObject {
                id: m.dependencies[0].id,
                bytes: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let object = root.object_path(&m.dependencies[0].id);
        std::fs::set_permissions(&object, std::fs::Permissions::from_mode(0o644)).unwrap();
        let mut ring = one_ring();
        let slot = ring.acquire().unwrap();
        match verify_dependency(&root, &m.dependencies[0], &mut ring, slot) {
            Err(DiskMiss::Corrupt(r)) => {
                assert!(r.contains("not a regular"), "{r}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        ring.abandon(slot).unwrap();
    }

    #[test]
    fn a_group_accessible_root_directory_refuses_reopen() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let root_path = dir.path().join("kv");
        DiskRoot::open_or_create(&root_path, scope(), engine()).unwrap();
        std::fs::set_permissions(&root_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        match DiskRoot::open_or_create(&root_path, scope(), engine()) {
            Err(DiskMiss::Refused(r)) => assert!(r.contains("owner-only"), "{r}"),
            other => panic!("expected a directory refusal, got {other:?}"),
        }
    }
}
