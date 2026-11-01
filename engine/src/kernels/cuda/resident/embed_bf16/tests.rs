//! Gather, bad-id and refusal tests of embed_bf16 against a host widening.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

const ROWS: usize = 8;
const K: usize = 24;

/// A table of distinct BF16 codes: the code of row `r`, column `c` is `r * 977 + c * 31 + 5`
/// with the exponent bits kept finite.
fn table() -> Vec<u8> {
    (0..ROWS * K)
        .flat_map(|i| {
            let code = (((i / K) * 977 + (i % K) * 31 + 5) as u16 & 0x3fff) | 0x3000;
            code.to_le_bytes()
        })
        .collect()
}

fn widened(table: &[u8], id: usize, column: usize) -> f32 {
    let at = (id * K + column) * 2;
    f32::from_bits(u32::from(u16::from_le_bytes([table[at], table[at + 1]])) << 16)
}

#[test]
fn gathers_rows_as_the_exact_widening_of_the_stored_codes() {
    if !testkit::device_present("embed_bf16 gather") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let table = table();
    let w = testkit::upload_bytes(&ctx, &table);
    let ids = testkit::upload_bytes(&ctx, &[5i32.to_le_bytes(), 0i32.to_le_bytes()].concat());
    let x = testkit::upload_filled(&ctx, 0xff, 2 * K * 4);
    let status = testkit::upload_filled(&ctx, 0, 4);
    ctx.embed_bf16(
        &stream,
        EmbedBf16Call {
            table: w.range(0, table.len()).unwrap(),
            ids: ids.range(0, 8).unwrap(),
            x: x.range(0, 2 * K * 4).unwrap(),
            status: status.range(0, 4).unwrap(),
            m: 2,
            n_rows: ROWS,
            k: K,
        },
    )
    .expect("gather");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &x, 2 * K * 4));
    for (token, id) in [5usize, 0].into_iter().enumerate() {
        for column in 0..K {
            assert_eq!(
                got[token * K + column].to_bits(),
                widened(&table, id, column).to_bits(),
                "token {token} column {column}"
            );
        }
    }
    assert_eq!(testkit::read_bytes(&ctx, &status, 4), [0; 4]);
}

#[test]
fn a_bad_id_writes_a_nan_row_and_sets_status_bit_four_only() {
    if !testkit::device_present("embed_bf16 bad id") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let table = table();
    let w = testkit::upload_bytes(&ctx, &table);
    for bad in [-1i32, ROWS as i32, i32::MAX] {
        let ids = testkit::upload_bytes(&ctx, &[2i32.to_le_bytes(), bad.to_le_bytes()].concat());
        let x = testkit::upload_filled(&ctx, 0, 2 * K * 4);
        let status = testkit::upload_filled(&ctx, 0, 4);
        ctx.embed_bf16(
            &stream,
            EmbedBf16Call {
                table: w.range(0, table.len()).unwrap(),
                ids: ids.range(0, 8).unwrap(),
                x: x.range(0, 2 * K * 4).unwrap(),
                status: status.range(0, 4).unwrap(),
                m: 2,
                n_rows: ROWS,
                k: K,
            },
        )
        .expect("launch");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &x, 2 * K * 4));
        assert!(
            (0..K).all(|c| got[c] == widened(&table, 2, c)),
            "valid row {bad}"
        );
        assert!(got[K..].iter().all(|v| v.is_nan()), "bad row {bad}");
        assert_eq!(
            testkit::read_bytes(&ctx, &status, 4),
            4i32.to_le_bytes(),
            "{bad}"
        );
    }
}

#[test]
fn the_wrapper_refuses_bad_shapes_and_overlaps_before_any_launch() {
    if !testkit::device_present("embed_bf16 refusals") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let w = testkit::upload_bytes(&ctx, &table());
    let ids = testkit::upload_bytes(&ctx, &[0u8; 8]);
    let x = testkit::upload_filled(&ctx, 0, 2 * K * 4);
    let status = testkit::upload_filled(&ctx, 0, 4);
    let call = |m, n_rows, k| EmbedBf16Call {
        table: w.range(0, ROWS * K * 2).unwrap(),
        ids: ids.range(0, 8).unwrap(),
        x: x.range(0, 2 * K * 4).unwrap(),
        status: status.range(0, 4).unwrap(),
        m,
        n_rows,
        k,
    };
    for (m, n_rows, k) in [
        (0, ROWS, K),
        (4097, ROWS, K),
        (2, 0, K),
        (2, ROWS, 20),
        (2, ROWS, 0),
    ] {
        let refused = ctx.embed_bf16(&stream, call(m, n_rows, k));
        assert!(
            matches!(refused, Err(crate::EngError::BadRequest(_))),
            "{m} {n_rows} {k}"
        );
    }
    // More rows than the table holds, and the status word inside the output.
    assert!(ctx.embed_bf16(&stream, call(2, ROWS + 1, K)).is_err());
    let mut aliased = call(2, ROWS, K);
    aliased.status = x.range(8, 4).unwrap();
    assert!(ctx.embed_bf16(&stream, aliased).is_err());
}
