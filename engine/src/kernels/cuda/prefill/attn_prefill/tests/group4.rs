//! Test 7 of the parent module at the SenseNova Loop geometry: 16 query heads over 4 kv heads,
//! head dimension 256. The parent runs it at 16:2. This is the same f64 reference, the same five
//! cases and the same derived bound at 16:4, where one kv head serves 4 query heads instead of 8.

use super::*;

const KV_HEADS: usize = 4;

/// The Loop shape of one chunk: `m` tokens at `p0`, a `rows`-row linear cache.
fn shape(m: usize, rows: usize, p0: i64) -> AttnPrefillShape {
    AttnPrefillShape {
        m,
        n_head: 16,
        n_kv_head: KV_HEADS,
        head_dim: D,
        cache_rows: rows,
        ld: 16 * 2 * D,
        p0,
    }
}

#[test]
fn attn_prefill_group4_matches_the_f64_host_reference() {
    if !testkit::device_present("attn_prefill_group4_matches_the_f64_host_reference") {
        return;
    }
    check_matches_the_f64_host_reference(shape, KV_HEADS);
}
