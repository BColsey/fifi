//! Host tests of the window-cost arithmetic and identities.

use super::window_cost::{
    DeviceWindowTransfers, PackManifest, RECORD_SCHEMA, StateShape, Transfer, WindowTransfers,
    check_identity, ids_sha256, median, parse_stat_faults, sha256_file, sha256_hex, summary,
};
use serde_json::json;

/// A state of 3 DeltaNet layers and 2 attention layers with 2 planes each. The runs are what
/// `prefix_ranges` would give at some length: one run per plane, two for the second layer's
/// second plane (a page boundary).
fn shape() -> StateShape {
    StateShape {
        delta_layers: 3,
        recurrent_bytes: 1_000,
        conv_bytes: 100,
        kv_runs: vec![vec![vec![400], vec![400]], vec![vec![400], vec![300, 100]]],
        words_bytes: 4_096,
        logits_row_values: 248_320,
        start_block_bytes: 32,
        result_pair_bytes: 8,
    }
}

#[test]
fn the_median_of_an_even_count_is_the_mean_of_the_middle_pair() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[3.0]), Some(3.0));
    assert_eq!(median(&[5.0, 1.0, 3.0]), Some(3.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    let summed = summary(&[2.0, 8.0, 5.0]);
    assert_eq!(summed["n"], 3);
    assert_eq!(summed["median"], 5.0);
    assert_eq!(
        (summed["min"].as_f64(), summed["max"].as_f64()),
        (Some(2.0), Some(8.0))
    );
    assert!(summary(&[]).is_null());
}

#[test]
fn the_digests_match_known_vectors() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // The packing is little endian: [1] is the bytes 01 00 00 00.
    assert_eq!(ids_sha256(&[1]), sha256_hex(&[1, 0, 0, 0]));
    assert_ne!(ids_sha256(&[1, 2]), ids_sha256(&[2, 1]));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f");
    std::fs::write(&path, b"abc").unwrap();
    assert_eq!(sha256_file(&path).unwrap(), sha256_hex(b"abc"));
}

#[test]
fn the_pack_manifest_changes_with_a_name_a_size_or_a_blob() {
    let dir = tempfile::tempdir().unwrap();
    let blobs = dir.path().join("blobs");
    let snap = dir.path().join("snap");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    // A sparse file over the 64 MiB hashing limit: it is never read.
    let blob = |name: &str, bytes: u64| {
        std::fs::File::create(blobs.join(name))
            .unwrap()
            .set_len(bytes)
            .unwrap();
    };
    blob(&"a".repeat(64), 65 << 20);
    std::fs::write(blobs.join("cfg"), b"{}").unwrap();
    std::os::unix::fs::symlink(
        blobs.join("a".repeat(64)),
        snap.join("model-00001.safetensors"),
    )
    .unwrap();
    std::os::unix::fs::symlink(blobs.join("cfg"), snap.join("config.json")).unwrap();
    let first = PackManifest::read(&snap).unwrap();
    assert_eq!(first.files.len(), 2);
    let shard = first
        .files
        .iter()
        .find(|f| f.name.ends_with("safetensors"))
        .unwrap();
    assert_eq!(shard.blob_name.as_deref(), Some("a".repeat(64).as_str()));
    assert_eq!(
        shard.content_sha256, None,
        "a shard is identified by its blob name and size"
    );
    let config = first
        .files
        .iter()
        .find(|f| f.name == "config.json")
        .unwrap();
    assert_eq!(
        config.content_sha256.as_deref(),
        Some(sha256_hex(b"{}").as_str())
    );
    assert_eq!(first, PackManifest::read(&snap).unwrap(), "stable");
    // A different config body changes the manifest.
    std::fs::write(blobs.join("cfg"), b"{ }").unwrap();
    let edited = PackManifest::read(&snap).unwrap();
    assert_ne!(first.manifest_sha256, edited.manifest_sha256);
    // A shard of the same size under another blob name changes it too.
    blob(&"b".repeat(64), 65 << 20);
    std::fs::remove_file(snap.join("model-00001.safetensors")).unwrap();
    std::os::unix::fs::symlink(
        blobs.join("b".repeat(64)),
        snap.join("model-00001.safetensors"),
    )
    .unwrap();
    let swapped = PackManifest::read(&snap).unwrap();
    assert_ne!(edited.manifest_sha256, swapped.manifest_sha256);
    assert_eq!(swapped.to_json()["files"].as_array().unwrap().len(), 2);
}

#[test]
fn the_state_copies_follow_the_layer_and_run_counts() {
    let state = shape();
    assert_eq!(state.delta_bytes(), 3 * 1_100);
    assert_eq!(state.kv_bytes(), 400 + 400 + 400 + 300 + 100);
    assert_eq!(state.state_bytes(), 3_300 + 1_600);
    let export = state.export();
    assert_eq!((export.d2h_bytes, export.h2d_bytes), (4_900, 0));
    // Two blocks per DeltaNet layer plus one call per non-empty KV run.
    assert_eq!(export.d2h_calls, 2 * 3 + 5);
    let import = state.import();
    assert_eq!((import.h2d_bytes, import.d2h_bytes), (4_900 + 4_096, 0));
    assert_eq!(
        import.h2d_calls,
        2 * 3 + 5 + 1,
        "and one call for the words block"
    );
}

#[test]
fn a_window_costs_its_export_its_rows_and_after_a_partial_accept_an_import_and_a_replay() {
    let state = shape();
    let head = state.logits_row_values;
    let full = WindowTransfers::new(&state, 4, 4);
    assert_eq!(
        full.import,
        Transfer::default(),
        "a full accept restores nothing"
    );
    assert_eq!(full.replay, Transfer::default());
    assert_eq!(full.rows.d2h_bytes, 4 * (8 + head * 4));
    assert_eq!(full.rows.d2h_calls, 4 * 2, "the pair and the logits row");
    assert_eq!(full.rows.h2d_calls, 4);
    let partial = WindowTransfers::new(&state, 4, 1);
    assert_eq!(partial.export, full.export);
    assert_eq!(partial.import, state.import());
    assert_eq!(
        partial.replay,
        state.step(),
        "one kept row fed again, no logits"
    );
    assert_eq!(
        partial.total().d2h_bytes,
        partial.export.d2h_bytes + partial.rows.d2h_bytes + 8
    );
    assert_eq!(state.logits_row().d2h_bytes, 993_280);
    let json = partial.to_json();
    assert_eq!(json["total"]["h2d_calls"], partial.total().h2d_calls);
    assert!(json["basis"].as_str().unwrap().contains("computed"));
}

#[test]
fn a_record_without_a_complete_identity_is_refused() {
    let complete = json!({
        "schema": RECORD_SCHEMA,
        "identity": {
            "commit": "abc", "binary_sha256": "d", "kv_mode": "Fp16", "capture": true,
            "single_row_kernel": "wide1",
            "pack": {"manifest_sha256": "e"},
            "token_ids": {"text_sha256": "f", "tokenizer_sha256": "g", "ids_sha256": "h", "source": "text"},
        }
    });
    check_identity(&complete).expect("complete");
    for (path, patch) in [
        ("commit", json!("")),
        ("binary_sha256", json!(null)),
        ("single_row_kernel", json!(null)),
    ] {
        let mut broken = complete.clone();
        broken["identity"][path] = patch;
        let error = check_identity(&broken).expect_err(path);
        assert!(error.contains(path), "{error}");
    }
    let mut no_pack = complete.clone();
    no_pack["identity"]["pack"] = json!({});
    assert!(
        check_identity(&no_pack)
            .unwrap_err()
            .contains("pack.manifest_sha256")
    );
    assert!(check_identity(&json!({})).is_err());
}

/// Prints the identity of a pack directory for the run plan: `FIFI_EXL3_CHECKPOINT` names it.
/// Host only; it hashes the small files and reads the sizes of the shards.
#[test]
#[ignore = "needs FIFI_EXL3_CHECKPOINT, the pack directory"]
fn print_the_pack_manifest() {
    let dir = std::env::var("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let manifest = PackManifest::read(std::path::Path::new(&dir)).expect("manifest");
    println!("MRV_PACK manifest_sha256={}", manifest.manifest_sha256);
    println!("MRV_PACK_JSON {}", manifest.to_json());
}

/// Prints the identity of the teacher-forced token stream for the run plan: the text
/// (`FIFI_SPEC_TEXT`) encoded by the tokenizer (`FIFI_SPEC_TOKENIZER`), the first `FIFI_MRV_NEED`
/// ids (default 34,304: the deepest depth of 32,768 plus the harness slack). Host only.
#[test]
#[ignore = "needs FIFI_SPEC_TEXT and FIFI_SPEC_TOKENIZER"]
fn print_the_token_ids_identity() {
    let tokenizer_path = std::env::var("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER");
    let text_path = std::env::var("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT");
    let need: usize = std::env::var("FIFI_MRV_NEED")
        .ok()
        .map_or(34_304, |text| text.parse().expect("FIFI_MRV_NEED"));
    let tokenizer = crate::tokenizer::Tokenizer::from_bytes(
        &std::fs::read(&tokenizer_path).expect("tokenizer"),
    )
    .expect("parse");
    let ids = tokenizer
        .encode(&std::fs::read_to_string(&text_path).expect("text"))
        .expect("encode");
    assert!(
        ids.len() >= need,
        "the text holds {} tokens, the run needs {need}",
        ids.len()
    );
    assert!(
        ids.iter().all(|&id| id < 248_077),
        "an id outside the sampler vocabulary"
    );
    println!(
        "MRV_IDS available={} used={need} ids_sha256={} text_sha256={} tokenizer_sha256={}",
        ids.len(),
        ids_sha256(&ids[..need]),
        sha256_file(std::path::Path::new(&text_path)).expect("text sha"),
        sha256_file(std::path::Path::new(&tokenizer_path)).expect("tokenizer sha"),
    );
}

#[test]
fn a_device_rollback_window_copies_the_slots_and_never_the_state_to_the_host() {
    let state = shape();
    const SLOT: u64 = 65_863_680;
    let all = DeviceWindowTransfers::new(&state, SLOT, 4, 4);
    assert_eq!(all.capture.d2d_bytes, 3 * SLOT);
    assert_eq!(all.capture.d2d_calls, 3);
    assert_eq!(
        all.restore,
        Transfer::default(),
        "a full accept restores nothing"
    );
    let partial = DeviceWindowTransfers::new(&state, SLOT, 4, 1);
    assert_eq!(partial.restore.d2d_bytes, SLOT);
    assert_eq!(partial.restore.d2d_calls, 1);
    assert_eq!(
        partial.restore.h2d_bytes, state.words_bytes,
        "the words block is rewritten"
    );
    // The rows are the host window's rows; the export, the import and the replay are gone.
    let host = WindowTransfers::new(&state, 4, 1);
    assert_eq!(partial.rows, host.rows);
    let total = partial.total();
    assert_eq!(total.d2d_bytes, 4 * SLOT);
    assert_eq!(
        total.d2h_bytes, host.rows.d2h_bytes,
        "no state goes to the host"
    );
    assert!(total.d2h_bytes < host.total().d2h_bytes);
    let two = DeviceWindowTransfers::new(&state, SLOT, 2, 1);
    assert_eq!(two.capture.d2d_calls, 1);
    assert_eq!(two.to_json()["total"]["d2d_bytes"], 2 * SLOT);
}

#[test]
fn the_fault_counters_are_read_after_the_command_name() {
    // Fields 10 and 12 of /proc/self/stat; the name may hold spaces and parentheses.
    let stat = "4242 (a (weird) name) S 1 4242 4242 0 -1 4194304 12345 0 7 0 5 6 0 0 20 0 1";
    assert_eq!(parse_stat_faults(stat), Some((12345, 7)));
    assert_eq!(parse_stat_faults("no parenthesis"), None);
    assert_eq!(parse_stat_faults("1 (x) S 1 2"), None, "too few fields");
    assert_eq!(
        parse_stat_faults("1 (x) S 1 2 3 4 5 6 abc 8 9"),
        None,
        "not a number"
    );
}
