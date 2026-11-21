//! The peer payload of a stored prefix: the totals a report gives follow the shards a slot holds.
use crate::family::{
    decode_step::run_report_tests::PrefixHostStep,
    session::{DecodeSession, PrefixSlot, SaveKind, StoreLimits},
};
use crate::sched::job::{PrefixHit, PrefixReport};

#[test]
fn peer_payload_tracks_published_shards_through_replacement_and_eviction() {
    use super::state::{
        StateIdentity,
        export::{DeltaExport, ExportedState, KvExport},
    };
    use super::{common_tests, dims::KvMode, prefix::stored_prefix_report};

    let mut store = PrefixSlot::with_limits(StoreLimits {
        roots: 1,
        entries: 1,
    });
    let lookup = PrefixReport {
        hit: PrefixHit::Root,
        uncached_shared: 7,
        ..PrefixReport::default()
    };
    let totals = |slot: &PrefixSlot<Vec<ExportedState>, u8>| {
        let report = stored_prefix_report(slot, lookup);
        assert_eq!((report.hit, report.uncached_shared), (PrefixHit::Root, 7));
        (report.entries, report.bytes)
    };
    let identity = StateIdentity::new(&common_tests::dims(), KvMode::Fp16);
    let save = |slot: &mut _, key: &[u32], kind| {
        let mut session = DecodeSession::default();
        let mut step = PrefixHostStep::default();
        session.prefill(&mut step, key, 2048).unwrap();
        let rows = step.fed;
        let shard = |rec, conv, k, v| ExportedState {
            identity: identity.clone(),
            tokens: rows,
            delta: vec![DeltaExport {
                recurrent: vec![0.0; rec],
                conv: vec![0.0; conv],
            }],
            kv: vec![KvExport {
                planes: vec![vec![0; k * rows], vec![0; v * rows]],
            }],
        };
        assert!(
            session
                .save_prefix(slot, key, 0, rows, kind, || Ok(vec![
                    shard(2, 1, 1, 2),
                    shard(1, 3, 4, 2),
                ]))
                .unwrap()
        );
    };
    // The two shards retain 28 fixed bytes plus nine bytes per committed row.
    assert_eq!(totals(&store), (0, 0));
    save(&mut store, &[7, 6], SaveKind::Root);
    assert_eq!(totals(&store), (1, 37));
    save(&mut store, &[1, 2, 3], SaveKind::Tip);
    assert_eq!(totals(&store), (2, 83));
    save(&mut store, &[3, 4, 5, 6], SaveKind::Tip);
    save(&mut store, &[3, 4, 5, 6], SaveKind::Tip);
    assert_eq!(totals(&store), (2, 92));
    assert!(store.evict_lru_one(None));
    assert_eq!(totals(&store), (1, 37));
}
