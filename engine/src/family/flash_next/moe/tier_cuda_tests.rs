//! Layer 3 of the real pack through the device classifier at ten picks: the real router's routes, the
//! tier plan's slot row of the layer, `tiered_classify` publishing into a real mailbox, and the host
//! oracle (`SlotTable::classify`) and the polled request line as the expected values.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack with the headers of every shard and the router of
//! the layer (`FIFI_FLASH_NEXT_EXL3_MOE_LAYER`, 3 by default). Needs a GPU, under 1 GiB. The
//! classifier does not read expert weights; this checks the 512-expert, 10-pick, 2560-wide
//! geometry and the request line's tail (picks 8 and 9) against the real routing.

use super::{
    real_layer_tests::{input, open, pack_sizes, plan_at},
    route,
    tier_plan::RouteCounts,
};
use crate::{
    EngError,
    family::project::cuda::CudaProjector,
    family::{
        exl3_pack::{Root, router_name_in},
        flash_next::{load::model_shape, stored::host_f32},
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, device_present,
        resident::tiered_classify::{CTRL_BYTES, ClassifyCall, ClassifyShape, REQUEST_BYTES},
    },
};

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

fn le_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs a GPU (under 1 GiB) and FIFI_FLASH_NEXT_EXL3_PACK with the router of layer 3"]
fn the_device_classifier_agrees_with_the_oracle_on_real_routes_at_ten_picks() {
    assert!(
        device_present("real Flash-Next tier classify"),
        "a GPU is required"
    );
    let (checkpoint, layer) = open();
    let (tier, trunk, hidden) = pack_sizes(&checkpoint);
    let (shape, kinds) = model_shape(&checkpoint.model_config).expect("the model shape");
    let moe = shape.moe;
    assert_eq!((moe.experts, moe.top_k, hidden), (512, 10, 2560));
    let router = host_f32(
        &checkpoint.source,
        &router_name_in(Root::Layers, layer),
        moe.experts * hidden,
    )
    .expect("the real router");
    let routes: Vec<Vec<(u32, f32)>> = (0..64)
        .map(|seed| route(&moe, &router, &input(hidden, seed)).expect("route"))
        .collect();
    let plan = plan_at(tier, trunk, 131_072);
    let mut counts = RouteCounts::new(kinds.len(), moe.experts);
    for picks in &routes[..32] {
        let ids: Vec<u32> = picks.iter().map(|p| p.0).collect();
        counts.record(layer, &ids).expect("record");
    }
    let hot = plan.hot_set(counts.counts()).expect("the hot set");
    let row = hot.table.row(layer).to_vec();

    let projector = CudaProjector::new().expect("projector");
    let ctx: &CudaContext = projector.context();
    let stream = ctx.stream().expect("stream");
    let mailbox = ctx.mailbox(hidden).expect("mailbox");
    let port = mailbox.port().expect("port");
    let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]).expect("upload");
    let ids = zeros(40);
    let weights = zeros(40);
    let slot_row = ctx.upload(&le_i32(&row)).expect("upload");
    let norm_row = zeros(hidden * 4);
    let slot_ids_out = zeros(40);
    let weights_out = zeros(40);
    let log_out = zeros(80);
    let ctrl = zeros(CTRL_BYTES);
    let word = zeros(4);
    let unused_req = zeros(REQUEST_BYTES);
    let unused_row = zeros(hidden * 4);
    let (mut after, mut tail_misses) = (0u64, 0usize);
    for (index, picks) in routes[32..].iter().enumerate() {
        let want_ids: Vec<i32> = picks.iter().map(|p| p.0 as i32).collect();
        let want_weights: Vec<f32> = picks.iter().map(|p| p.1).collect();
        let want = hot
            .table
            .classify(layer, &want_ids, &want_weights)
            .expect("oracle");
        ids.write_at(&stream, 0, &le_i32(&want_ids)).expect("ids");
        weights
            .write_at(&stream, 0, &le_f32(&want_weights))
            .expect("weights");
        norm_row
            .write_at(&stream, 0, &le_f32(&input(hidden, 64 + index)))
            .expect("row");
        stream.synchronize().expect("sync");
        let call = ClassifyCall {
            shape: ClassifyShape {
                n_expert: 512,
                slots: hot.table.slots(),
                top_k: 10,
                hidden,
                layer: 0,
            },
            ids: whole(&ids).unwrap(),
            weights: whole(&weights).unwrap(),
            slot_row: whole(&slot_row).unwrap(),
            norm_row: whole(&norm_row).unwrap(),
            slot_ids_out: whole(&slot_ids_out).unwrap(),
            weights_out: whole(&weights_out).unwrap(),
            log_out: whole(&log_out).unwrap(),
            ctrl: whole(&ctrl).unwrap(),
            word: whole(&word).unwrap(),
            mbox_req: whole(&unused_req).unwrap(),
            mbox_row: whole(&unused_row).unwrap(),
        };
        ctx.tiered_classify_to_mailbox(&stream, call, &mailbox)
            .expect("classify");
        stream.synchronize().expect("sync");
        let (mut slot_bytes, mut weight_bytes) = ([0u8; 40], [0u8; 40]);
        slot_ids_out
            .read_at(&stream, 0, &mut slot_bytes)
            .expect("read");
        weights_out
            .read_at(&stream, 0, &mut weight_bytes)
            .expect("read");
        let slots: Vec<i32> = slot_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        let weights_got: Vec<u32> = weight_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        assert_eq!(slots, want.slot_ids, "route {index}: slots");
        assert_eq!(
            weights_got,
            want.weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            "route {index}: weights"
        );
        if want.misses.is_empty() {
            assert!(
                port.poll(after).is_none(),
                "route {index}: no request without a miss"
            );
            continue;
        }
        let request = port.poll(after).expect("a request for the misses");
        assert_eq!(
            request.count as usize,
            want.misses.len(),
            "route {index}: count"
        );
        for (k, &(id, weight)) in want.misses.iter().enumerate() {
            let got = request.miss(k).expect("a published miss");
            assert_eq!(
                (got.0, got.1.to_bits()),
                (id, weight.to_bits()),
                "route {index}: miss {k}"
            );
        }
        tail_misses += usize::from(want.misses.len() > 8);
        let mut partial_row = vec![0f32; hidden];
        assert!(port.read_row(&mut partial_row));
        assert!(port.answer(request.seq, Ok(&vec![0f32; hidden])));
        after = request.seq;
    }
    eprintln!("routes with more than 8 misses (the request tail): {tail_misses}");
}
