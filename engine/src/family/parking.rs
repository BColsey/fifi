//! Conversation parking (CPU mechanics): a host byte-budgeted
//! LRU store of parked conversation states, keyed by identity and
//! token prefix. The identity is the pack fingerprint (model), the
//! pinned chat-template digest, the cache-config hash (kv profile,
//! prefill numeric path, row block, context limit, layer major), the
//! format version, the profile and the image list; identities are
//! assembled in `flint::park_identity`. This module is the store, its
//! exact byte accounting and its tests; the device-side wiring (cut at
//! the server `keep` position, upload through the pinned staging pair,
//! adoption as the restored prefix) is not part of this module.

use crate::EngError;
use std::mem::size_of;

/// The KV profile a park was taken under. A park restores only into the
/// same profile: parked bytes are the profile's own encoding (q8 codes
/// stay q8), so cross-profile adoption is refused rather than lossy.
/// KV dtype stays a per-profile parameter; nothing here locks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ParkProfile {
    F16,
    Q8,
}

/// The stable identity half of a park key: everything a restore must
/// match exactly before token-prefix matching may even look at entries.
/// `model` is the pack fingerprint (sha256 of the pack's SHA256SUMS
/// manifest, or of all shards in name order when the pack ships none);
/// `template` is the chat-template content hash; `cache_config` is a
/// hash of everything the payload encoding depends on beside the
/// profile (serialized row-block size and layout constants) — a
/// requantized or relaid-out model can never restore another's park,
/// the same rule as the upstream tier-3 fingerprint; `format` is the
/// payload layout version. `images` reserves the per-image content
/// hashes of image turns from the start, so a park that ever carries
/// images never matches one that does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParkIdentity {
    pub(crate) model: [u8; 32],
    pub(crate) template: [u8; 32],
    pub(crate) cache_config: [u8; 32],
    pub(crate) format: u32,
    pub(crate) profile: ParkProfile,
    pub(crate) images: Vec<[u8; 32]>,
}

/// Identity of a parked conversation state: the stable half plus the
/// token prefix that was parked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParkKey {
    pub(crate) identity: ParkIdentity,
    pub(crate) tokens: Vec<u32>,
}

/// The park payload layout version. Bump when `ParkedPayload`'s
/// encoding changes: an old-format park then simply never matches a
/// new-format identity, so no stale bytes can restore.
pub(crate) const PARK_FORMAT_VERSION: u32 = 1;

impl ParkKey {
    /// Logical byte size this key charges against the park budget: the
    /// identity's fixed bytes (three 32-byte hashes, the format version
    /// and the profile tag) plus the variable bytes (4 per token id, 32
    /// per reserved image hash). Logical, not exact: container and
    /// struct overhead is not charged. Charging the variable bytes
    /// bounds key-only growth: no count of tiny parks can grow memory
    /// outside the budget.
    pub(crate) fn byte_len(&self) -> u64 {
        let fixed = 3 * size_of::<[u8; 32]>() + size_of::<u32>() + size_of::<ParkProfile>();
        (fixed
            + self.tokens.len() * size_of::<u32>()
            + self.identity.images.len() * size_of::<[u8; 32]>()) as u64
    }
}

/// The serialized state of a parked conversation: the checkpoint payload
/// (recurrent/conv/PLE and QSA-ring state) and the K/V rows, both in the
/// profile's own encoding. `encode`/`decode` round-trip exactly; there is
/// no lossy step anywhere in the park path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParkedPayload {
    pub(crate) state: Vec<u8>,
    pub(crate) kv_rows: Vec<u8>,
}

impl ParkedPayload {
    /// Logical byte size this payload charges against the park budget.
    pub(crate) fn byte_len(&self) -> u64 {
        (self.state.len() + self.kv_rows.len()) as u64
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.state.len() + self.kv_rows.len());
        out.extend_from_slice(&(self.state.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.state);
        out.extend_from_slice(&(self.kv_rows.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.kv_rows);
        out
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, EngError> {
        let short = |what: &str| EngError::Checkpoint(format!("parked payload truncated: {what}"));
        let (len, rest) = bytes
            .split_at_checked(8)
            .ok_or_else(|| short("state length"))?;
        let state_len = usize::try_from(u64::from_le_bytes(len.try_into().expect("8 bytes")))
            .map_err(|_| short("state length overflows usize"))?;
        let state = rest
            .split_at_checked(state_len)
            .ok_or_else(|| short("state bytes"))?
            .0;
        let rest = &rest[state_len..];
        let (len, rest) = rest.split_at_checked(8).ok_or_else(|| short("kv length"))?;
        let kv_len = usize::try_from(u64::from_le_bytes(len.try_into().expect("8 bytes")))
            .map_err(|_| short("kv length overflows usize"))?;
        let kv_rows = rest
            .split_at_checked(kv_len)
            .ok_or_else(|| short("kv bytes"))?
            .0;
        if !rest[kv_len..].is_empty() {
            return Err(EngError::Checkpoint(
                "parked payload has trailing bytes".to_string(),
            ));
        }
        Ok(Self {
            state: state.to_vec(),
            kv_rows: kv_rows.to_vec(),
        })
    }
}

struct ParkEntry {
    key: ParkKey,
    payload: ParkedPayload,
}

impl ParkEntry {
    fn charge(&self) -> u64 {
        self.key.byte_len() + self.payload.byte_len()
    }
}

/// The host-tier park store: LRU entries under one absolute byte budget
/// (`conversation_cache_bytes`; 0 = parking off). The store refuses the
/// park, never the process: when the payload cannot fit even after
/// evicting every evictable entry, `park` returns false and the store is
/// unchanged. Accounting is logical bytes: `used_bytes` always
/// equals the sum of each stored entry's key and payload charge
/// bytes (container and struct overhead is not counted).
pub(crate) struct ParkStore {
    /// LRU order: least recently used first.
    entries: Vec<ParkEntry>,
    budget: u64,
    used: u64,
}

impl ParkStore {
    pub(crate) fn new(budget_bytes: u64) -> Self {
        Self {
            entries: Vec::new(),
            budget: budget_bytes,
            used: 0,
        }
    }

    pub(crate) fn budget_bytes(&self) -> u64 {
        self.budget
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Parks `payload` under `key`. The charge is the payload plus the
    /// key's bytes, evicting LRU entries until it fits; refuses (returns
    /// false, store otherwise unchanged) when the charge alone exceeds
    /// the budget — for budget 0 that is every park, since any key
    /// charges at least its fixed bytes (parking off). Replacing the same
    /// key charges only once and a refused replacement keeps the old
    /// entry.
    pub(crate) fn park(&mut self, key: ParkKey, payload: ParkedPayload) -> bool {
        let charge = key.byte_len() + payload.byte_len();
        if charge > self.budget {
            return false;
        }
        if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            self.used -= self.entries[pos].charge();
            self.entries.remove(pos);
        }
        // Overflow-safe room check: `used <= budget` always holds, so the
        // subtraction cannot underflow and no `used + charge` is computed.
        while self.budget - self.used < charge {
            if self.entries.is_empty() {
                return false;
            }
            let evicted = self.entries.remove(0);
            self.used -= evicted.charge();
        }
        self.used += charge;
        self.entries.push(ParkEntry { key, payload });
        true
    }

    /// The longest restore for `prompt`: the entry whose full identity
    /// matches and whose tokens are a prefix of `prompt`, maximal in
    /// tokens. The match may cover the entire `prompt` (`matched ==
    /// prompt.len()`): the caller must still evaluate at least one fresh
    /// token of its own — restore never short-circuits generation.
    /// Restoring touches the entry (LRU): a conversation in active
    /// rotation is the last thing evicted.
    pub(crate) fn restore(
        &mut self,
        prompt: &[u32],
        identity: &ParkIdentity,
    ) -> Option<(usize, ParkedPayload)> {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.key.identity == *identity
                    && e.key.tokens.len() <= prompt.len()
                    && e.key.tokens.iter().zip(prompt).all(|(k, p)| k == p)
            })
            .max_by_key(|(_, e)| e.key.tokens.len())
            .map(|(pos, _)| pos)?;
        let entry = self.entries.remove(best);
        let matched = entry.key.tokens.len();
        let payload = entry.payload.clone();
        // Reinsert at the MRU end: the restore marks the conversation live.
        self.entries.push(entry);
        Some((matched, payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(profile: ParkProfile) -> ParkIdentity {
        ParkIdentity {
            model: [1; 32],
            template: [2; 32],
            cache_config: [4; 32],
            format: PARK_FORMAT_VERSION,
            profile,
            images: Vec::new(),
        }
    }

    fn key(tokens: &[u32], profile: ParkProfile) -> ParkKey {
        ParkKey {
            identity: ident(profile),
            tokens: tokens.to_vec(),
        }
    }

    fn payload(state: usize, kv: usize) -> ParkedPayload {
        ParkedPayload {
            state: vec![0xA5; state],
            kv_rows: vec![0x5A; kv],
        }
    }

    #[test]
    fn key_charge_is_pinned_and_the_budget_edge_admits() {
        // The fixed identity charge: three 32-byte hashes, the format
        // version, and the one-byte (repr(u8)) profile tag. A new key
        // field must change both sides here.
        let bare = key(&[], ParkProfile::F16);
        assert_eq!(bare.byte_len(), 3 * 32 + 4 + 1);
        assert_eq!(bare.byte_len(), 101);
        // A charge exactly at the budget is admitted, filling it.
        let mut store = ParkStore::new(101);
        assert!(store.park(
            bare,
            ParkedPayload {
                state: Vec::new(),
                kv_rows: Vec::new(),
            }
        ));
        assert_eq!(store.used_bytes(), store.budget_bytes());
    }

    #[test]
    fn accounting_is_exact_and_replacement_charges_once() {
        let mut store = ParkStore::new(1_000);
        let first = (key(&[1, 2, 3], ParkProfile::F16), payload(100, 50));
        assert!(store.park(first.0.clone(), first.1.clone()));
        assert_eq!(store.used_bytes(), first.0.byte_len() + first.1.byte_len());
        let second = (key(&[1, 2, 3], ParkProfile::F16), payload(200, 0));
        assert!(store.park(second.0.clone(), second.1.clone()));
        assert_eq!(
            store.used_bytes(),
            second.0.byte_len() + second.1.byte_len()
        );
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.entries.iter().map(|e| e.charge()).sum::<u64>(),
            store.used_bytes()
        );
    }

    #[test]
    fn over_budget_park_is_refused_and_the_store_is_unchanged() {
        let mut store = ParkStore::new(200);
        assert!(store.park(key(&[1], ParkProfile::F16), payload(60, 0)));
        let used = store.used_bytes();
        assert!(!store.park(key(&[2], ParkProfile::F16), payload(120, 0)));
        assert_eq!(store.used_bytes(), used);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn budget_zero_is_off() {
        let mut store = ParkStore::new(0);
        assert!(!store.park(key(&[1], ParkProfile::F16), payload(1, 1)));
        assert!(store.is_empty());
    }

    #[test]
    fn duplicate_repark_larger_than_budget_keeps_the_old_entry() {
        let mut store = ParkStore::new(200);
        assert!(store.park(key(&[1], ParkProfile::F16), payload(50, 0)));
        assert!(!store.park(key(&[1], ParkProfile::F16), payload(200, 0)));
        let (matched, got) = store
            .restore(&[1], &ident(ParkProfile::F16))
            .expect("the refused re-park keeps the old entry");
        assert_eq!(matched, 1);
        assert_eq!(got, payload(50, 0));
    }

    #[test]
    fn park_evicts_lru_until_it_fits() {
        let mut store = ParkStore::new(400);
        assert!(store.park(key(&[1], ParkProfile::F16), payload(90, 0)));
        assert!(store.park(key(&[2], ParkProfile::F16), payload(90, 0)));
        assert!(store.park(key(&[3], ParkProfile::F16), payload(90, 0)));
        assert_eq!(store.len(), 2);
        assert!(
            store.restore(&[1], &ident(ParkProfile::F16)).is_none(),
            "the evicted entry is gone through the whole eviction chain"
        );
        assert!(store.restore(&[2], &ident(ParkProfile::F16)).is_some());
    }

    #[test]
    fn restore_takes_the_longest_token_prefix() {
        let mut store = ParkStore::new(10_000);
        assert!(store.park(key(&[1, 2, 3, 4], ParkProfile::F16), payload(1, 1)));
        assert!(store.park(key(&[1, 2], ParkProfile::F16), payload(1, 1)));
        let (matched, _) = store
            .restore(&[1, 2, 3, 4, 5], &ident(ParkProfile::F16))
            .expect("longest prefix wins");
        assert_eq!(matched, 4);
        let (matched, _) = store
            .restore(&[1, 2, 9], &ident(ParkProfile::F16))
            .expect("shorter entry matches the divergent prompt");
        assert_eq!(matched, 2);
        assert!(store.restore(&[9, 9], &ident(ParkProfile::F16)).is_none());
    }

    #[test]
    fn restore_never_crosses_cache_config_or_format() {
        let mut store = ParkStore::new(10_000);
        assert!(store.park(key(&[1, 2], ParkProfile::F16), payload(1, 1)));
        // A different cache layout (row-block size, layout constants)
        // parks beside the original and never restores it.
        let mut reconfigured = ident(ParkProfile::F16);
        reconfigured.cache_config = [9; 32];
        assert!(store.park(
            ParkKey {
                identity: reconfigured.clone(),
                tokens: vec![1, 2],
            },
            payload(2, 2)
        ));
        let (matched, got) = store
            .restore(&[1, 2], &ident(ParkProfile::F16))
            .expect("the original cache config still matches its own park");
        assert_eq!(matched, 2);
        assert_eq!(got, payload(1, 1), "the reconfigured park must not win");
        let (matched, got) = store
            .restore(&[1, 2], &reconfigured)
            .expect("the reconfigured identity restores its own park");
        assert_eq!(matched, 2);
        assert_eq!(got, payload(2, 2));
        // A layout-version change never matches either side.
        let mut bumped = ident(ParkProfile::F16);
        bumped.format = PARK_FORMAT_VERSION + 1;
        assert!(
            store.restore(&[1, 2], &bumped).is_none(),
            "a bumped park-format version never matches"
        );
    }

    #[test]
    fn restore_never_crosses_model_template_profile_or_images() {
        let mut store = ParkStore::new(10_000);
        assert!(store.park(key(&[1, 2], ParkProfile::F16), payload(1, 1)));
        let mut other_model = ident(ParkProfile::F16);
        other_model.model = [9; 32];
        assert!(
            store.restore(&[1, 2], &other_model).is_none(),
            "different model never matches"
        );
        let mut other_template = ident(ParkProfile::F16);
        other_template.template = [9; 32];
        assert!(
            store.restore(&[1, 2], &other_template).is_none(),
            "different template never matches"
        );
        assert!(
            store.restore(&[1, 2], &ident(ParkProfile::Q8)).is_none(),
            "different profile never matches"
        );
        // Two parks over the same tokens that differ ONLY in the image
        // list, with distinct payloads so a wrong restore is observable.
        let mut imaged = ident(ParkProfile::F16);
        imaged.images = vec![[7; 32]];
        assert!(store.park(
            ParkKey {
                identity: imaged.clone(),
                tokens: vec![1, 2],
            },
            payload(3, 1)
        ));
        let (matched, got) = store
            .restore(&[1, 2], &ident(ParkProfile::F16))
            .expect("the imageless prompt restores the imageless park");
        assert_eq!(matched, 2);
        assert_eq!(
            got,
            payload(1, 1),
            "the imageless prompt must not restore the imaged park"
        );
        let (matched, got) = store
            .restore(&[1, 2], &imaged)
            .expect("the imaged prompt restores the imaged park");
        assert_eq!(matched, 2);
        assert_eq!(
            got,
            payload(3, 1),
            "the imaged prompt must not restore the imageless park"
        );
        let mut only_imageless = ParkStore::new(10_000);
        assert!(only_imageless.park(key(&[1, 2], ParkProfile::F16), payload(1, 1)));
        assert!(
            only_imageless.restore(&[1, 2], &imaged).is_none(),
            "an imaged prompt never restores an imageless park"
        );
        let mut only_imaged = ParkStore::new(10_000);
        assert!(only_imaged.park(
            ParkKey {
                identity: imaged,
                tokens: vec![1, 2],
            },
            payload(3, 1)
        ));
        assert!(
            only_imaged
                .restore(&[1, 2], &ident(ParkProfile::F16))
                .is_none(),
            "an imageless prompt never restores an imaged park"
        );
    }

    #[test]
    fn restore_touches_lru_so_live_conversations_are_evicted_last() {
        let mut store = ParkStore::new(400);
        assert!(store.park(key(&[1], ParkProfile::F16), payload(90, 0)));
        assert!(store.park(key(&[2], ParkProfile::F16), payload(90, 0)));
        // Touch the first entry.
        store.restore(&[1, 5], &ident(ParkProfile::F16));
        // A park that needs room evicts the untouched (second) entry.
        assert!(store.park(key(&[3], ParkProfile::F16), payload(90, 0)));
        assert!(store.restore(&[2], &ident(ParkProfile::F16)).is_none());
        assert!(store.restore(&[1], &ident(ParkProfile::F16)).is_some());
    }

    #[test]
    fn payload_round_trip_is_byte_exact() {
        let p = payload(1_024, 2_048);
        let bytes = p.encode();
        assert_eq!(ParkedPayload::decode(&bytes).expect("decode"), p);
        assert_eq!(bytes.len() as u64, p.byte_len() + 16);
    }

    #[test]
    fn payload_decode_rejects_impossible_lengths() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(ParkedPayload::decode(&bytes).is_err());
    }

    #[test]
    fn payload_decode_rejects_corruption() {
        let bytes = ParkedPayload {
            state: vec![1; 10],
            kv_rows: Vec::new(),
        }
        .encode();
        let mut corrupt = bytes.clone();
        corrupt[0] ^= 0xFF;
        assert!(ParkedPayload::decode(&corrupt).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(ParkedPayload::decode(&trailing).is_err());
        let mut truncated = trailing;
        truncated.pop();
        assert!(ParkedPayload::decode(&truncated[..truncated.len() - 1]).is_err());
    }
}
