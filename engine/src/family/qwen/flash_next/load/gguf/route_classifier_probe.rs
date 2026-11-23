//! Test-only validation of device router picks through the resident slot classifier.
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
    resident::tiered_classify::{
        BASE_PICKS, CTRL_BYTES, ClassifyCall, ClassifyShape, MAX_TOP_K, REQUEST_BYTES,
        request_pick_at,
    },
};

pub(super) struct Input<'a> {
    pub ids: DeviceRange<'a>,
    pub weights: DeviceRange<'a>,
    pub x: DeviceRange<'a>,
    pub slot_row: &'a [i32],
    pub slots: usize,
    pub top_k: usize,
    pub hidden: usize,
}

pub(super) struct Classified {
    pub slots: DeviceBuffer,
    pub weights: DeviceBuffer,
    pub picks: Vec<i32>,
    pub misses: usize,
}

fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}
fn read(stream: &DeviceStream, range: DeviceRange<'_>, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    range
        .buffer()
        .read_at(stream, range.offset(), &mut bytes)
        .unwrap();
    bytes
}
fn i32_at(bytes: &[u8], at: usize) -> i32 {
    i32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_ne_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Queues and checks the real GPU classifier, including the miss request and log.
pub(super) fn classify(ctx: &CudaContext, stream: &DeviceStream, input: Input<'_>) -> Classified {
    let row_bytes: Vec<u8> = input
        .slot_row
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let slot_row = ctx.upload(&row_bytes).unwrap();
    let slots = ctx.allocate(input.top_k * 4).unwrap();
    let weights = ctx.allocate(input.top_k * 4).unwrap();
    let log = ctx.allocate(input.top_k * 8).unwrap();
    let ctrl = ctx.upload(&[0; CTRL_BYTES]).unwrap();
    let word = ctx.upload(&[0; 4]).unwrap();
    let req = ctx.upload(&[0xA5; REQUEST_BYTES]).unwrap();
    let row = ctx.allocate(input.hidden * 4).unwrap();
    ctx.tiered_classify(
        stream,
        ClassifyCall {
            shape: ClassifyShape {
                n_expert: input.slot_row.len(),
                slots: input.slots,
                top_k: input.top_k,
                hidden: input.hidden,
                layer: 0,
            },
            ids: input.ids,
            weights: input.weights,
            slot_row: all(&slot_row),
            norm_row: input.x,
            slot_ids_out: all(&slots),
            weights_out: all(&weights),
            log_out: all(&log),
            ctrl: all(&ctrl),
            word: all(&word),
            mbox_req: all(&req),
            mbox_row: all(&row),
        },
    )
    .unwrap();
    let picks_raw = read(stream, input.ids, input.top_k * 4);
    let weights_raw = read(stream, input.weights, input.top_k * 4);
    let slot_raw = read(stream, all(&slots), input.top_k * 4);
    let out_weights = read(stream, all(&weights), input.top_k * 4);
    let log_raw = read(stream, all(&log), input.top_k * 8);
    let ctrl_raw = read(stream, all(&ctrl), CTRL_BYTES);
    let req_raw = read(stream, all(&req), REQUEST_BYTES);
    assert_eq!(i32_at(&read(stream, all(&word), 4), 0), 0);
    let picks: Vec<_> = (0..input.top_k).map(|i| i32_at(&log_raw, i * 8)).collect();
    let mut missing = Vec::new();
    for (rank, &id) in picks.iter().enumerate() {
        assert!(id >= 0 && (id as usize) < input.slot_row.len());
        assert_eq!(i32_at(&picks_raw, rank * 4), id);
        assert_eq!(
            &log_raw[rank * 8 + 4..rank * 8 + 8],
            &weights_raw[rank * 4..rank * 4 + 4]
        );
        let slot = input.slot_row[id as usize];
        if slot < 0 {
            assert_eq!(i32_at(&slot_raw, rank * 4), 0);
            assert_eq!(&out_weights[rank * 4..rank * 4 + 4], &[0; 4]);
            missing.push(rank);
        } else {
            assert_eq!(i32_at(&slot_raw, rank * 4), slot);
            assert_eq!(
                &out_weights[rank * 4..rank * 4 + 4],
                &weights_raw[rank * 4..rank * 4 + 4]
            );
        }
    }
    assert_eq!(u64_at(&ctrl_raw, 0), missing.len() as u64);
    if missing.is_empty() {
        assert!(req_raw.iter().all(|v| *v == 0xA5));
        assert_eq!((u64_at(&ctrl_raw, 8), u64_at(&ctrl_raw, 16)), (0, 0));
    } else {
        assert_eq!(
            (
                u64_at(&ctrl_raw, 8),
                u64_at(&ctrl_raw, 16),
                u64_at(&req_raw, 0)
            ),
            (1, 1, 1)
        );
        assert_eq!(u64_at(&req_raw, 8), (missing.len() as u64) << 32);
        for (position, &rank) in missing.iter().enumerate() {
            let (id_at, weight_at) = request_pick_at(position);
            assert_eq!(i32_at(&req_raw, id_at), picks[rank]);
            assert_eq!(
                &req_raw[weight_at..weight_at + 4],
                &weights_raw[rank * 4..rank * 4 + 4]
            );
        }
        let request_slots = if input.top_k > BASE_PICKS {
            MAX_TOP_K
        } else {
            BASE_PICKS
        };
        for position in missing.len()..request_slots {
            let (id_at, weight_at) = request_pick_at(position);
            assert_eq!(i32_at(&req_raw, id_at), -1);
            assert_eq!(&req_raw[weight_at..weight_at + 4], &[0; 4]);
        }
        assert_eq!(
            read(stream, all(&row), input.hidden * 4),
            read(stream, input.x, input.hidden * 4)
        );
    }
    Classified {
        slots,
        weights,
        picks,
        misses: missing.len(),
    }
}
