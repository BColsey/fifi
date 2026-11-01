use super::*;
use crate::family::flint::exl3_resident::{FlintResidentLimits, load_flint_resident};
use crate::family::project::cuda::Reserve;
use crate::kernels::attention_reference::oracle::{sdpa_f64, ulp_bf16};

struct WriterPlane {
    input: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    decoded: Vec<f64>,
}

fn writer_plane(factor: f32) -> WriterPlane {
    let input: Vec<_> = (0_usize..8 * 512)
        .map(|i| {
            let (row, head, group, lane) = (i / 512, i % 512 / 128, i % 128 / 32, i % 32);
            let amplitude = 0.25 * (row * 16 + head * 4 + group + 1) as f32;
            let magnitude = ((lane + head * 5 + row * 3 + group * 7) % 32 + 1) as f32 / 32.0;
            let sign = if (lane + head + row).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            half::bf16::from_f32(factor * amplitude * magnitude * sign).to_f32()
        })
        .collect();
    let mut codes = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / 16);
    let mut decoded = vec![0.0; 527 * 512];
    for values in input.as_chunks::<32>().0 {
        let amax = values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = half::f16::from_f32(amax / 127.0);
        scales.extend_from_slice(&scale.to_bits().to_ne_bytes());
        for &value in values {
            // StoredHalf uses rounded division and nearest-even integer conversion.
            let code = (value / scale.to_f32())
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            decoded[519 * 512 + codes.len()] = f64::from(f32::from(code) * scale.to_f32());
            codes.push(code as u8);
        }
    }
    assert_ne!(
        codes[..128],
        codes[128..256],
        "head permutation must be observable"
    );
    assert_ne!(
        codes[..512],
        codes[512..1024],
        "time permutation must be observable"
    );
    WriterPlane {
        input,
        codes,
        scales,
        decoded,
    }
}

fn check_written_plane(
    stream: &DeviceStream,
    view: AttnKvView<'_>,
    source: &WriterPlane,
    cells: usize,
) {
    let extents = view.layout.extents(cells, 4, 128).unwrap();
    assert_eq!(extents.payload_bytes, cells * 512);
    assert_eq!(extents.scale_bytes, Some(cells * 32));
    assert_eq!(extents.bias_bytes, None);
    let scalar = view.layout.scales.unwrap();
    let mut expected_codes = vec![0x55; cells * 512];
    let mut expected_scales = vec![0xff; cells * 32];
    for row in 0..8 {
        let cell = (519 + row) % cells;
        for head in 0..4 {
            let from = (row * 4 + head) * 128;
            let to = head * view.layout.payload_strides.head_stride_bytes
                + cell * view.layout.payload_strides.time_stride_bytes;
            expected_codes[to..to + 128].copy_from_slice(&source.codes[from..from + 128]);
            for group in 0..4 {
                let from = ((row * 4 + head) * 4 + group) * 2;
                let to = head * scalar.strides.head_stride_bytes
                    + cell * scalar.strides.time_stride_bytes
                    + group * scalar.group_stride_bytes;
                expected_scales[to..to + 2].copy_from_slice(&source.scales[from..from + 2]);
            }
        }
    }
    for (range, expected, what) in [
        (view.payload, expected_codes, "codes"),
        (view.scales.unwrap(), expected_scales, "scales"),
    ] {
        let mut actual = vec![0; range.bytes()];
        range
            .buffer()
            .read_at(stream, range.offset(), &mut actual)
            .unwrap();
        if actual != expected {
            let index = actual
                .iter()
                .zip(&expected)
                .position(|(a, e)| a != e)
                .unwrap_or(actual.len());
            panic!(
                "Q8 {what} mismatch at byte {index}: actual {:02x?} expected {:02x?} (of {} bytes)",
                &actual[index.saturating_sub(2)..(index + 14).min(actual.len())],
                &expected[index.saturating_sub(2)..(index + 14).min(expected.len())],
                actual.len(),
            );
        }
    }
}

fn attention_error(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f64;
    for (actual, expected) in actual
        .as_chunks::<128>()
        .0
        .iter()
        .zip(expected.as_chunks::<128>().0)
    {
        let magnitude = expected.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let bound = 2.0 * ulp_bf16(magnitude);
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            worst = worst.max((f64::from(actual) - expected).abs() / bound);
        }
    }
    worst
}

fn check_writer_sink(
    group: &Group<'_, '_>,
    views: [AttnKvView<'_>; 2],
    source: &[WriterPlane; 2],
    sliding: bool,
) {
    let (ctx, stream) = (group.ctx, group.stream);
    let cells = views[0].payload.bytes() / 512;
    let first = if sliding { 7 } else { 0 };
    // Seed valid old history only after the whole-plane writer/poison checks. The ABI,
    // not the view under test, defines these head-major addresses and the reference.
    for view in views {
        for head in 0..4 {
            let offset = (head * cells + first) * 128;
            view.payload
                .buffer()
                .write_at(
                    stream,
                    view.payload.offset() + offset,
                    &vec![0; (519 - first) * 128],
                )
                .unwrap();
            let scales = view.scales.unwrap();
            scales
                .buffer()
                .write_at(
                    stream,
                    scales.offset() + (head * cells + first) * 8,
                    &half::f16::ONE
                        .to_bits()
                        .to_ne_bytes()
                        .repeat((519 - first) * 4),
                )
                .unwrap();
        }
    }
    let sinks: Vec<f32> = (0..32)
        .map(|head| [-4.0, 0.5, 12.0][head % 3] + head as f32 / 64.0)
        .collect();
    let sink_buffer = ctx
        .upload(
            &sinks
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sinks_f64: Vec<_> = sinks.iter().copied().map(f64::from).collect();
    let span = CpfAttnSpan {
        window: if sliding { 513 } else { 0 },
        ring: if sliding { cells as i64 } else { 0 },
    };
    let mut swapped = views;
    for (correct, wrong) in views.iter().zip(&mut swapped) {
        wrong.layout.payload_strides = AttnStrides {
            time_stride_bytes: 512,
            head_stride_bytes: 128,
        };
        assert_eq!(
            correct.layout.extents(cells, 4, 128).unwrap().payload_bytes,
            wrong.layout.extents(cells, 4, 128).unwrap().payload_bytes,
            "the negative control must pass the same extent check",
        );
    }
    for row in 0..8 {
        let position = 519 + row;
        let q: Vec<f32> = (0..4096)
            .map(|i| ((i * 7 + row * 5) % 31) as f32 / 512.0 - 15.0 / 512.0)
            .collect();
        let q_f64: Vec<_> = q.iter().copied().map(f64::from).collect();
        let scale = 128.0_f32.sqrt().recip();
        let expected = sdpa_f64(
            &q_f64,
            &source[0].decoded,
            &source[1].decoded,
            1,
            32,
            527,
            128,
            8,
            f64::from(scale),
            position as u32,
            0,
            sliding.then_some(513),
            Some(&sinks_f64),
        );
        let query = ctx
            .upload(&q.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>())
            .unwrap();
        let plan = cpf_attn_launch_plan(
            HotScratch::geometry(),
            position + 1,
            span.window as usize,
            CPF_ATTN_SM_COUNT,
        )
        .unwrap();
        let part_bytes = cpf_attn_part_bytes(HotScratch::geometry(), plan).unwrap();
        let part = ctx.upload(&vec![0xff; part_bytes]).unwrap();
        let output = ctx.upload(&[0xff; 4096 * 4]).unwrap();
        let status = ctx.upload(&0_u32.to_ne_bytes()).unwrap();
        for (mutated, kv_views) in [(false, views), (true, swapped)] {
            output.write_at(stream, 0, &[0xff; 4096 * 4]).unwrap();
            part.write_at(stream, 0, &vec![0xff; part_bytes]).unwrap();
            status.write_at(stream, 0, &0_u32.to_ne_bytes()).unwrap();
            ctx.attn_decode_sinks_via_prefill(
                stream,
                AttnPrefillCall {
                    storage: kv_views[0].layout.storage,
                    kv_views: Some(kv_views),
                    shape: AttnPrefillShape {
                        m: 1,
                        n_head: 32,
                        n_kv_head: 4,
                        head_dim: 128,
                        cache_rows: cells,
                        ld: 0,
                        p0: position as i64,
                    },
                    span,
                    split: CpfAttnSplit {
                        max_splits: plan.max_splits,
                        min_keys: plan.min_keys,
                    },
                    q_tile: plan.q_tile,
                    scale,
                    q: query.range(0, query.bytes()).unwrap(),
                    proj: None,
                    k_cache: kv_views[0].payload,
                    v_cache: kv_views[1].payload,
                    part: part.range(0, part_bytes).unwrap(),
                    out: output.range(0, 4096 * 4).unwrap(),
                    status: Some(status.range(0, 4).unwrap()),
                    sinks: Some(sink_buffer.range(0, 32 * 4).unwrap()),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut flag = [0; 4];
            status.read_at(stream, 0, &mut flag).unwrap();
            assert_eq!(u32::from_ne_bytes(flag), 0);
            let actual = values(stream, output.range(0, 4096 * 4).unwrap()).unwrap();
            let error = attention_error(&actual, &expected);
            if mutated {
                assert!(
                    error > 1.0,
                    "stride mutation was vacuous: sliding={sliding} P{position} error/bound={error}"
                );
            } else {
                assert!(
                    error <= 1.0,
                    "writer-to-sink CPF mismatch: sliding={sliding} P{position} error/bound={error}"
                );
            }
            eprintln!(
                "Q8 writer_to_sink_CPF sliding={sliding} P{position} stride_mutation={mutated} error/bound={error}"
            );
        }
        if row == 0 && std::env::var_os("Q8_TILED_WRITER_ROUTE_PROBE").is_some() {
            // Production-shaped runtime planes (real Group and write_cache views,
            // sinks Some) driven through the tiled route entry — the same safe-layer
            // entry runtime attention() dispatches to when armed. This does NOT
            // exercise the runtime env branch itself; q8_tiled_runtime_attention_
            // routes_when_armed covers that through a real decoder step. Here the
            // counters must prove the tile kernel ran and nothing fell back; the
            // output must agree numerically.
            output.write_at(stream, 0, &[0xff; 4096 * 4]).unwrap();
            part.write_at(stream, 0, &vec![0xff; part_bytes]).unwrap();
            status.write_at(stream, 0, &0_u32.to_ne_bytes()).unwrap();
            let routes = CudaContext::q8_tiled_route_count();
            let fallbacks = CudaContext::q8_tiled_fallback_count();
            ctx.attn_prefill_q8_tiled_opt_in(
                stream,
                AttnPrefillCall {
                    storage: views[0].layout.storage,
                    kv_views: Some(views),
                    shape: AttnPrefillShape {
                        m: 1,
                        n_head: 32,
                        n_kv_head: 4,
                        head_dim: 128,
                        cache_rows: cells,
                        ld: 0,
                        p0: position as i64,
                    },
                    span,
                    split: CpfAttnSplit {
                        max_splits: plan.max_splits,
                        min_keys: plan.min_keys,
                    },
                    q_tile: plan.q_tile,
                    scale,
                    q: query.range(0, query.bytes()).unwrap(),
                    proj: None,
                    k_cache: views[0].payload,
                    v_cache: views[1].payload,
                    part: part.range(0, part_bytes).unwrap(),
                    out: output.range(0, 4096 * 4).unwrap(),
                    status: Some(status.range(0, 4).unwrap()),
                    sinks: Some(sink_buffer.range(0, 32 * 4).unwrap()),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut flag = [0; 4];
            status.read_at(stream, 0, &mut flag).unwrap();
            assert_eq!(u32::from_ne_bytes(flag), 0);
            assert_eq!(CudaContext::q8_tiled_route_count(), routes + 1);
            assert_eq!(CudaContext::q8_tiled_fallback_count(), fallbacks);
            let actual = values(stream, output.range(0, 4096 * 4).unwrap()).unwrap();
            let error = attention_error(&actual, &expected);
            assert!(
                error <= 1.0,
                "armed-route CPF mismatch: sliding={sliding} P{position} error/bound={error}"
            );
            println!(
                "ROUTE_W=1 sliding={sliding} routes={} fallbacks={} error/bound={error}",
                routes + 1,
                fallbacks
            );
        }
    }
}

fn writer_sink_case(ctx: &CudaContext, stream: &DeviceStream, layer: usize) {
    let host_writes = test_counts()[1];
    let q8_launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    let plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_storage(KvStorage::Q8G32);
    let mut cache = LayerKvAllocation::new(&plan, layer)
        .unwrap()
        .allocate(ctx)
        .unwrap();
    let cells = plan.cache_rows(layer).unwrap();
    let source = [writer_plane(1.0), writer_plane(-2.0)];
    let inputs = source.each_ref().map(|plane| {
        ctx.upload(
            &plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    });
    for plane in [&cache.keys, &cache.values] {
        plane
            .write_at(stream, 0, &vec![0x55; plane.bytes()])
            .unwrap();
    }
    for plane in cache.scales.as_ref().unwrap() {
        plane
            .write_at(stream, 0, &vec![0xff; plane.bytes()])
            .unwrap();
    }
    while cache.cursor.committed_prefix() < 519 {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(519 - position))
            .unwrap()
            .commit();
    }
    let append = cache.cursor.begin_append(519, 8).unwrap();
    let group = Group::new(ctx, stream, None, 0).unwrap();
    let planes = [&cache.keys, &cache.values];
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, None, &append), 8, None),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache storage and scale planes disagree"
    ));
    let mut f16_cursor = LayerKvAllocation::new(&KvPlan::new(2048, 8, true).unwrap(), layer)
        .unwrap()
        .cursor();
    let f16_append = f16_cursor.begin_append(0, 8).unwrap();
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, cache.scales.as_ref(), &f16_append), 8, None),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache storage and scale planes disagree"
    ));
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, cache.scales.as_ref(), &append), 7, None),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache append runs do not cover source rows"
    ));
    let views = write_cache(
        &group,
        [&inputs[0], &inputs[1]],
        (planes, cache.scales.as_ref(), &append),
        8,
        None,
    )
    .unwrap()
    .unwrap();
    for (view, source) in views.into_iter().zip(&source) {
        check_written_plane(stream, view, source, cells);
    }
    check_writer_sink(&group, views, &source, layer == 2);
    stream.synchronize().unwrap();
    append.commit();
    assert_eq!(cache.cursor.committed_prefix(), 527);
    assert_eq!(test_counts()[1], host_writes);
    assert_eq!(
        TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get),
        q8_launches + if layer == 2 { 2 } else { 1 }
    );
}

#[test]
fn device_q8_wrap_has_zero_host_kv_writes_and_two_device_launches() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer-to-sink wrap fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    writer_sink_case(&ctx, &ctx.stream().unwrap(), 2);
}

#[test]
fn device_q8_full_layer_writer_views_match_sink_reference() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer-to-sink full-layer fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    writer_sink_case(&ctx, &ctx.stream().unwrap(), 0);
}

const REFUSED_BOUNDARY_VALUES: [f32; 8] = [
    f32::from_bits(1.0_f32.to_bits() + 1),
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    65536.0,
    -65536.0,
    8_388_608.0,
    -8_388_608.0,
];

fn boundary_cache(group: &Group<'_, '_>) -> LayerKv {
    let plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_storage(KvStorage::Q8G32);
    let mut cache = LayerKvAllocation::new(&plan, 2)
        .unwrap()
        .allocate(group.ctx)
        .unwrap();
    for (plane, poison) in [
        (&cache.keys, 0x55),
        (&cache.values, 0x55),
        (&cache.scales.as_ref().unwrap()[0], 0xff),
        (&cache.scales.as_ref().unwrap()[1], 0xff),
    ] {
        plane
            .write_at(group.stream, 0, &vec![poison; plane.bytes()])
            .unwrap();
    }
    while cache.cursor.committed_prefix() < 519 {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(519 - position))
            .unwrap()
            .commit();
    }
    cache
}

fn check_refused_boundary(group: &Group<'_, '_>, invalid_plane: usize, value: f32) {
    let mut cache = boundary_cache(group);
    let mut input = [vec![1.0_f32; 8 * 512], vec![-1.0_f32; 8 * 512]];
    input[invalid_plane][8 * 512 - 1] = value;
    let inputs = input.each_ref().map(|values| {
        group
            .ctx
            .upload(
                &values
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    });
    let counts = test_counts();
    let launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    {
        let append = cache.cursor.begin_append(519, 8).unwrap();
        let error = write_cache(
            group,
            [&inputs[0], &inputs[1]],
            ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
            8,
            None,
        )
        .err()
        .expect("invalid Q8 writer input must be refused");
        assert_eq!(
            error,
            EngError::ModelOutput(
                "Flint invalid BF16 input or overflow in physical KV cache".into()
            ),
            "plane={invalid_plane} value={value:?}"
        );
    }
    group.stream.synchronize().unwrap();
    assert_eq!(test_counts(), counts);
    assert_eq!(TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get), launches);
    assert_eq!(cache.cursor.committed_prefix(), 519);
    assert!(cache.cursor.test_poisoned());
    for (plane, poison) in [
        (&cache.keys, 0x55),
        (&cache.values, 0x55),
        (&cache.scales.as_ref().unwrap()[0], 0xff),
        (&cache.scales.as_ref().unwrap()[1], 0xff),
    ] {
        let mut actual = vec![0; plane.bytes()];
        plane.read_at(group.stream, 0, &mut actual).unwrap();
        assert!(
            actual.iter().all(|&byte| byte == poison),
            "refusal changed a Q8 plane: input={invalid_plane} value={value:?}"
        );
    }
    assert!(cache.cursor.begin_append(519, 8).is_err());
    eprintln!(
        "Q8 numerical refusal plane={invalid_plane} value={value:?} launches=0 planes_unchanged=4 prefix=519 poisoned=true"
    );
}

#[test]
fn device_q8_writer_refuses_non_bf16_nonfinite_and_overflow_without_writes() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer numerical refusal fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    for invalid_plane in 0..2 {
        for value in REFUSED_BOUNDARY_VALUES {
            check_refused_boundary(&group, invalid_plane, value);
        }
    }
}

fn stored_half_boundary_plane(factor: f32) -> WriterPlane {
    let unit = 2.0_f32.powi(-25);
    // At half the least F16 scale, ties-to-even stores zero; the next BF16 amax stores 2^-24.
    let pattern = [-1.0, 1.0, 0.5, 0.0, -1.0, 1.0, 0.5, 0.0];
    let groups = [
        (pattern.map(|x| x * 0.0), 0_u16, [0_i8; 8]),
        (pattern.map(|x| x * (126.0 * unit)), 0, [0; 8]),
        (pattern.map(|x| x * (127.0 * unit)), 0, [0; 8]),
        (
            pattern.map(|x| x * (128.0 * unit)),
            1,
            [-64, 64, 32, 0, -64, 64, 32, 0],
        ),
        (
            pattern.map(|x| x * 65280.0),
            0x6004,
            [-127, 127, 64, 0, -127, 127, 64, 0],
        ),
        (
            [-127.0, 127.0, -0.5, 0.5, -2.5, 2.5, 0.0, 0.0],
            0x3c00,
            [-127, 127, 0, 0, -2, 2, 0, 0],
        ),
        (
            pattern.map(|x| x * (190.0 * 2.0_f32.powi(-24))),
            1,
            [-127, 127, 95, 0, -127, 127, 95, 0],
        ),
    ];
    let input: Vec<_> = (0..8 * 512)
        .map(|i| factor * groups[(i / 32) % groups.len()].0[i % 8])
        .collect();
    assert!(input.iter().all(|x| x.to_bits() & 0xffff == 0));
    let codes = (0..input.len())
        .map(|i| (factor as i8 * groups[(i / 32) % groups.len()].2[i % 8]) as u8)
        .collect();
    let scales = (0..input.len() / 32)
        .flat_map(|i| groups[i % groups.len()].1.to_ne_bytes())
        .collect();
    WriterPlane {
        input,
        codes,
        scales,
        decoded: Vec::new(),
    }
}

#[test]
fn device_q8_writer_pins_zero_tiny_and_maximum_stored_half_groups() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer StoredHalf boundary fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    let mut cache = boundary_cache(&group);
    let source = [
        stored_half_boundary_plane(1.0),
        stored_half_boundary_plane(-1.0),
    ];
    let inputs = source.each_ref().map(|plane| {
        ctx.upload(
            &plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    });
    let host_writes = test_counts()[1];
    let launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    let append = cache.cursor.begin_append(519, 8).unwrap();
    let views = write_cache(
        &group,
        [&inputs[0], &inputs[1]],
        ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
        8,
        None,
    )
    .unwrap()
    .unwrap();
    for (view, expected) in views.into_iter().zip(&source) {
        check_written_plane(&stream, view, expected, 520);
    }
    stream.synchronize().unwrap();
    append.commit();
    assert_eq!(cache.cursor.committed_prefix(), 527);
    assert!(!cache.cursor.test_poisoned());
    assert_eq!(test_counts()[1], host_writes);
    assert_eq!(
        TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get),
        launches + 2
    );
    eprintln!(
        "Q8 StoredHalf zero/below-tie/tie/above-tie/maximum/code-RNE-ties/subnormal-saturation groups byte-exact K+V launches=2 prefix=527"
    );
}

#[test]
fn q8_writer_boundary_inputs_pin_host_refusals_and_stored_half_goldens() {
    for value in REFUSED_BOUNDARY_VALUES {
        assert_eq!(
            f16_row(&[value; 512]).unwrap_err(),
            EngError::ModelOutput(
                "Flint invalid BF16 input or overflow in physical KV cache".into()
            )
        );
    }
    for factor in [1.0, -1.0] {
        let source = stored_half_boundary_plane(factor);
        for row in source.input.as_chunks::<512>().0 {
            assert_eq!(f16_row(row).unwrap().len(), 1024);
        }
        for (index, group) in source.input.as_chunks::<32>().0.iter().enumerate() {
            let amax = group.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = half::f16::from_f32(amax / 127.0);
            assert_eq!(
                source.scales[index * 2..index * 2 + 2],
                scale.to_bits().to_ne_bytes()
            );
            for (lane, value) in group.iter().enumerate() {
                let code = if scale.to_bits() == 0 {
                    0
                } else {
                    (value / scale.to_f32())
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i8
                };
                assert_eq!(
                    source.codes[index * 32 + lane],
                    code as u8,
                    "factor={factor} group={index} lane={lane}"
                );
            }
        }
    }
}

/// The runtime probe marker, shared by the child that prints it and the parent
/// that parses it so the wire format cannot drift between them.
fn q8_route_rt_marker(routes: u64, fallbacks: u64, layers: u64) -> String {
    format!("ROUTE_RT=1 routes={routes} fallbacks={fallbacks} layers={layers}")
}

/// Parses the fields of a [`q8_route_rt_marker`] line. Key-addressed tokens,
/// so field order or trailing text cannot break the read (the parser is bound
/// to the marker helper by this pair and the round-trip test below).
fn parse_q8_route_rt(line: &str) -> (u64, u64, u64) {
    let mut routes = None;
    let mut fallbacks = None;
    let mut layers = None;
    for token in line.split_whitespace() {
        let (key, value) = token.split_once('=').unwrap_or(("", ""));
        match key {
            "routes" => routes = value.parse().ok(),
            "fallbacks" => fallbacks = value.parse().ok(),
            "layers" => layers = value.parse().ok(),
            _ => {}
        }
    }
    (
        routes.expect("routes field"),
        fallbacks.expect("fallbacks field"),
        layers.expect("layers field"),
    )
}

/// Device-independent guard: the parent parses exactly what the child emits.
#[test]
fn q8_route_rt_marker_round_trip() {
    for &(routes, fallbacks, layers) in &[
        (0, 0, 49),
        (49, 0, 49),
        (7, 3, 7),
        (u64::MAX, 0, 1),
        (2_114_445_375, 0, 64),
    ] {
        let line = q8_route_rt_marker(routes, fallbacks, layers);
        assert_eq!(parse_q8_route_rt(&line), (routes, fallbacks, layers));
    }
}

/// Armed-route check through the REAL runtime branch: one decoder step on the
/// real pack, with FIFI_FLINT_CUDA_Q8_TILED set to `1`, rolled back to `0`, and
/// unset (default ON), in separate child processes
/// (the OnceLock is read once per process). Rolled back, a step must leave the
/// route counter untouched; armed AND unset, every layer's attention() must go
/// through the tiled dispatch: the routed child's route delta equals the model's
/// layer count (one attention() call per layer per single-token step), and
/// fallbacks stay zero. Children are matched by FULL module path (--exact with
/// a bare name matches nothing) and the probe marker is read from stdout and
/// stderr. Ignored: admitted device window only — three children each load the
/// full pack with all free VRAM (single-GPU-owner rule).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; armed-route runtime attention proof"]
fn q8_tiled_runtime_attention_routes_when_armed() {
    use std::process::Command;
    // An --ignored run is an admitted device window by definition: missing
    // prerequisites must fail loudly, not report ok without exercising the
    // branch (which would let an inverted dispatch stay green).
    assert!(
        std::env::var_os("FLINT_EXL3_PACK").is_some(),
        "--ignored run requires FLINT_EXL3_PACK"
    );
    assert!(
        crate::kernels::cuda::device_present("q8_tiled_runtime_attention_routes_when_armed"),
        "--ignored run requires a CUDA device"
    );
    #[derive(Clone, Copy, Debug)]
    enum Knob {
        Unset,
        Armed,
        Rollback,
    }
    let leg = |knob: Knob| -> (u64, u64) {
        let mut cmd = Command::new(std::env::current_exe().expect("exe"));
        cmd.arg("--exact")
            .arg(concat!(
                "family::flint::cuda_ffn::device::runtime::tests::q8::",
                "q8_tiled_runtime_attention_step_probe"
            ))
            .arg("--test-threads=1")
            .arg("--nocapture")
            .arg("--quiet")
            .env("Q8_TILED_RUNTIME_STEP_PROBE", "1");
        match knob {
            Knob::Unset => {
                cmd.env_remove("FIFI_FLINT_CUDA_Q8_TILED");
            }
            Knob::Armed => {
                cmd.env("FIFI_FLINT_CUDA_Q8_TILED", "1");
            }
            Knob::Rollback => {
                cmd.env("FIFI_FLINT_CUDA_Q8_TILED", "0");
            }
        }
        let out = cmd.output().expect("child run");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.status.success(),
            "runtime-attention child (knob={knob:?}) failed: {}\n{text}",
            out.status
        );
        let row = text
            .lines()
            .find(|l| l.contains("ROUTE_RT=1"))
            .expect("runtime probe marker line");
        println!("ROUTE_RT_CHILD knob={knob:?}: {row}");
        let (routes, fallbacks, layers) = parse_q8_route_rt(row);
        // The child asserts routes == layers against its own model; this
        // re-checks the same fact through the marker the parent parsed.
        assert_eq!(
            routes,
            if matches!(knob, Knob::Rollback) {
                0
            } else {
                layers
            },
            "marker routes disagree with layers: {row}"
        );
        (routes, fallbacks)
    };
    let (routes_off, fallbacks_off) = leg(Knob::Rollback);
    assert_eq!(
        (routes_off, fallbacks_off),
        (0, 0),
        "rolled-back step must never route or fall back"
    );
    let (routes_on, fallbacks_on) = leg(Knob::Armed);
    assert!(
        routes_on >= 1,
        "armed step routed nothing: routes={routes_on} fallbacks={fallbacks_on}"
    );
    assert_eq!(fallbacks_on, 0, "armed step had fallbacks: {fallbacks_on}");
    let (routes_default, fallbacks_default) = leg(Knob::Unset);
    assert!(
        routes_default >= 1,
        "default (unset) step routed nothing: routes={routes_default} fallbacks={fallbacks_default}"
    );
    assert_eq!(
        fallbacks_default, 0,
        "default (unset) step had fallbacks: {fallbacks_default}"
    );
    println!(
        "ROUTE_RT_SUMMARY routes={routes_on} fallbacks={fallbacks_on} (rollback control: 0/0; unset default: {routes_default}/{fallbacks_default})"
    );
}

/// The child probe for the test above: one decoder step through runtime
/// attention() with q8 KV planes, printing the counter deltas it caused.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn q8_tiled_runtime_attention_step_probe() {
    use std::path::PathBuf;
    if std::env::var_os("Q8_TILED_RUNTIME_STEP_PROBE").is_none() {
        return; // only meaningful as the child of the parent test
    }
    let pack = PathBuf::from(
        std::env::var_os("FLINT_EXL3_PACK").expect("the parent test gates on FLINT_EXL3_PACK"),
    );
    let ctx = CudaContext::new().expect("probe requires CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(1027, 1, false)
        .unwrap()
        .with_storage(KvStorage::Q8G32);
    let kv_bytes = plan.bytes(4 * 128 * 2 * 2).unwrap();
    let hot_bytes = State::scratch_bytes(&plan).unwrap();
    let model = load_flint_resident(
        &ctx,
        &stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: ctx.memory_info().unwrap().free as u64,
            reserve: Reserve {
                kv_cache: kv_bytes as u64,
                work: (512_u64 << 20).checked_add(hot_bytes).unwrap(),
                headroom: 512 << 20,
            },
        },
    )
    .unwrap();
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut state = State::new(&ctx, plan).unwrap();
    let routes = CudaContext::q8_tiled_route_count();
    let fallbacks = CudaContext::q8_tiled_fallback_count();
    decoder
        .step(&mut state, 1_u32, |_, _| -> Result<()> { Ok(()) })
        .unwrap();
    stream.synchronize().unwrap();
    let routes = CudaContext::q8_tiled_route_count() - routes;
    let fallbacks = CudaContext::q8_tiled_fallback_count() - fallbacks;
    // One attention() call per layer per single-token step: the armed child must
    // route exactly layer_count times, the rolled-back child zero times. A partial
    // count (silent fallbacks) or any fallback fails here, in the child, where
    // the layer count is known.
    let layers = model.layer_count() as u64;
    let armed = crate::family::flint::settings::q8_tiled_armed();
    assert_eq!(
        routes,
        if armed { layers } else { 0 },
        "step routed {routes} of {layers} layer attentions (armed={armed})"
    );
    assert_eq!(fallbacks, 0, "step had fallbacks");
    println!("{}", q8_route_rt_marker(routes, fallbacks, layers));
}

/// Runs the writer-sink route leg (the Q8_TILED_WRITER_ROUTE_PROBE arm inside
/// check_writer_sink): a production-shaped call on real Group/write_cache ring
/// planes launched through the tiled route entry, with counter and f64-oracle
/// checks. Without this parent the leg is dead code. Ignored like the runtime
/// probe: device window only.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires admitted GPU; writer-plane tiled route entry proof"]
fn q8_tiled_writer_route_entry_passes_oracle() {
    use std::process::Command;
    assert!(
        crate::kernels::cuda::device_present("q8_tiled_writer_route_entry_passes_oracle"),
        "--ignored run requires a CUDA device"
    );
    let out = Command::new(std::env::current_exe().expect("exe"))
        .arg("--exact")
        .arg(concat!(
            "family::flint::cuda_ffn::device::runtime::tests::q8::",
            "device_q8_wrap_has_zero_host_kv_writes_and_two_device_launches"
        ))
        .arg("--test-threads=1")
        .arg("--nocapture")
        .arg("--quiet")
        .env("Q8_TILED_WRITER_ROUTE_PROBE", "1")
        .output()
        .expect("child run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "writer-route child failed: {}\n{text}",
        out.status
    );
    let row = text
        .lines()
        .find(|l| l.contains("ROUTE_W=1"))
        .expect("writer route marker line");
    println!("ROUTE_W_CHILD: {row}");
    let routes: u64 = row
        .split_once(" routes=")
        .expect("routes field")
        .1
        .split_whitespace()
        .next()
        .expect("routes value")
        .parse()
        .expect("routes number");
    assert!(routes >= 1, "writer route leg counted no launches: {row}");
}

/// A layer cache advanced to `start` committed rows with the same poison in every
/// plane, so a byte compare also covers the cells an append never touches.
fn sentinel_cache(
    ctx: &CudaContext,
    stream: &DeviceStream,
    plan: &KvPlan,
    start: usize,
) -> LayerKv {
    // Layer 3 is a sliding layer: its 520-row ring makes the append at 519 straddle the end.
    let mut cache = LayerKvAllocation::new(plan, 3)
        .unwrap()
        .allocate(ctx)
        .unwrap();
    for plane in [&cache.keys, &cache.values] {
        plane
            .write_at(stream, 0, &vec![0x55; plane.bytes()])
            .unwrap();
    }
    // The mixed profile's second scale plane is the zero-byte placeholder;
    // a write through it would resolve a freed (absent) backing allocation.
    for plane in cache.scales.iter().flatten().filter(|p| p.bytes() > 0) {
        plane
            .write_at(stream, 0, &vec![0xff; plane.bytes()])
            .unwrap();
    }
    while cache.cursor.committed_prefix() < start {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(start - position))
            .unwrap()
            .commit();
    }
    cache
}

/// The kv_device_f16 arm through write_cache on one layer, for either KV profile.
/// ON and OFF appends of the same rows must leave every plane byte-identical,
/// poison cells included; NaN in either source plane, in either run of the
/// straddling append, must set the sticky word that the token gate reads.
fn check_kv_device_f16_arm(storage: KvStorage, fixture: &str) {
    if !crate::kernels::cuda::device_present(fixture) {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(2048, 8, true).unwrap().with_storage(storage);
    let bytes = [writer_plane(1.0), writer_plane(-2.0)].map(|plane| {
        plane
            .input
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>()
    });
    let upload = |bytes: &[Vec<u8>; 2]| bytes.each_ref().map(|b| ctx.upload(b).unwrap());
    let read_all = |buffer: &DeviceBuffer| {
        let mut out = vec![0u8; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut out).unwrap();
        out
    };
    // Eight rows from `start`, through the host guard (no word) or the device guard.
    let append_planes =
        |start: usize, runs: usize, inputs: &[DeviceBuffer; 2], guard: Option<&DeviceBuffer>| {
            let mut cache = sentinel_cache(&ctx, &stream, &plan, start);
            let append = cache.cursor.begin_append(start, 8).unwrap();
            let launched = append.runs().unwrap();
            assert_eq!(
                launched.iter().filter(|(_, rows)| !rows.is_empty()).count(),
                runs,
                "append at {start} must launch {runs} run(s)"
            );
            let group = Group::new(&ctx, &stream, None, 0).unwrap();
            write_cache(
                &group,
                [&inputs[0], &inputs[1]],
                ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
                8,
                guard,
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut planes = vec![read_all(&cache.keys), read_all(&cache.values)];
            planes.extend(cache.scales.iter().flatten().map(read_all));
            planes
        };
    let clean = upload(&bytes);
    for (start, runs) in [(0, 1), (519, 2)] {
        let host = append_planes(start, runs, &clean, None);
        assert!(
            host[0].iter().any(|&byte| byte != 0x55),
            "the host append at {start} must write K"
        );
        let guard = ctx.upload(&0u32.to_ne_bytes()).unwrap();
        let device = append_planes(start, runs, &clean, Some(&guard));
        assert_eq!(host.len(), device.len());
        for (index, (host, device)) in host.iter().zip(&device).enumerate() {
            assert!(host == device, "plane {index} at {start} differs ON vs OFF");
        }
        assert_eq!(
            read_all(&guard),
            0u32.to_ne_bytes(),
            "clean rows at {start}"
        );
    }
    // Row 0 sits in the first run of the append at 519, row 5 in the second.
    for (plane, row) in [(0, 0), (0, 5), (1, 0), (1, 5)] {
        let mut poisoned = bytes.clone();
        let at = row * 512 * 4;
        poisoned[plane][at..at + 4].copy_from_slice(&f32::NAN.to_ne_bytes());
        let guard = ctx.upload(&0u32.to_ne_bytes()).unwrap();
        append_planes(519, 2, &upload(&poisoned), Some(&guard));
        assert_eq!(
            read_all(&guard),
            1u32.to_ne_bytes(),
            "NaN in plane {plane}, row {row} must set the sticky word"
        );
    }
}

#[test]
fn kv_device_f16_q8_planes_identical_on_off_and_fault_flag_sticky() {
    check_kv_device_f16_arm(KvStorage::Q8G32, "Flint Q8 kv_device_f16 arm fixture");
}

#[test]
fn kv_device_f16_f16_planes_identical_on_off_and_fault_flag_sticky() {
    check_kv_device_f16_arm(KvStorage::F16, "Flint F16 kv_device_f16 arm fixture");
}

/// The mixed "q8k" device round-trip: K quantizes with byte-identical codes
/// and scales to the uniform profile, V keeps the exact f16 rows, and the
/// attention that reads the mixed views matches an oracle built from the
/// quantized K and the exact V.
#[test]
fn device_q8k_mixed_write_quantizes_k_and_keeps_v_f16() {
    if !crate::kernels::cuda::device_present("Flint Q8K mixed writer fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let owned_stream = ctx.stream().unwrap();
    let stream = &owned_stream;
    let layer = 0;
    let plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_kv_storage(KvStorage::Q8G32, KvStorage::F16);
    let mut cache = LayerKvAllocation::new(&plan, layer)
        .unwrap()
        .allocate(&ctx)
        .unwrap();
    let cells = plan.cache_rows(layer).unwrap();
    assert_eq!(cells, 2048);
    // K codes 512 B/row, V f16 1024 B/row: the planes are sized apart.
    assert_eq!(cache.keys.bytes(), cells * 512);
    assert_eq!(cache.values.bytes(), cells * 1024);
    let scales = cache.scales.as_ref().unwrap();
    assert_eq!(scales[0].bytes(), cells * 32);
    assert_eq!(scales[1].bytes(), 0);

    let source = [writer_plane(1.0), writer_plane(-2.0)];
    let inputs = source.each_ref().map(|plane| {
        ctx.upload(
            &plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    });
    for plane in [&cache.keys, &cache.values] {
        plane
            .write_at(stream, 0, &vec![0x55; plane.bytes()])
            .unwrap();
    }
    scales[0]
        .write_at(stream, 0, &vec![0xff; scales[0].bytes()])
        .unwrap();
    while cache.cursor.committed_prefix() < 519 {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(519 - position))
            .unwrap()
            .commit();
    }
    let append = cache.cursor.begin_append(519, 8).unwrap();
    let group = Group::new(&ctx, stream, None, 0).unwrap();
    let views = write_cache(
        &group,
        [&inputs[0], &inputs[1]],
        ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
        8,
        None,
    )
    .unwrap()
    .unwrap();

    // K: the q8 view with byte-identical codes and scales to the uniform path.
    check_written_plane(stream, views[0], &source[0], cells);
    // V: the f16 view; exact rows at cells 519..527, poison elsewhere.
    let mut expected_v = vec![0x55; cells * 1024];
    let mut row_bytes = Vec::with_capacity(8 * 1024);
    for row in source[1].input.as_chunks::<512>().0 {
        row_bytes.extend(f16_row(row).unwrap());
    }
    expected_v[519 * 1024..527 * 1024].copy_from_slice(&row_bytes);
    let mut actual_v = vec![0; cache.values.bytes()];
    cache.values.read_at(stream, 0, &mut actual_v).unwrap();
    assert!(
        actual_v == expected_v,
        "Q8K V plane must hold the exact f16 rows and untouched poison cells"
    );

    // Attention over the mixed views: oracle from quantized K, exact V.
    // Seed zero history below 519 first, exactly as the oracle assumes.
    for head in 0..4 {
        views[0]
            .payload
            .buffer()
            .write_at(
                stream,
                views[0].payload.offset() + head * cells * 128,
                &vec![0; 519 * 128],
            )
            .unwrap();
        let scales = views[0].scales.unwrap();
        scales
            .buffer()
            .write_at(
                stream,
                scales.offset() + head * cells * 8,
                &half::f16::ONE.to_bits().to_ne_bytes().repeat(519 * 4),
            )
            .unwrap();
    }
    views[1]
        .payload
        .buffer()
        .write_at(stream, views[1].payload.offset(), &vec![0; 519 * 1024])
        .unwrap();
    let k_decoded = source[0].decoded.clone();
    let sinks: Vec<f32> = (0..32)
        .map(|head| [-4.0, 0.5, 12.0][head % 3] + head as f32 / 64.0)
        .collect();
    let sink_buffer = ctx
        .upload(
            &sinks
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sinks_f64: Vec<_> = sinks.iter().copied().map(f64::from).collect();
    let mut v_exact = vec![0.0; 527 * 512];
    for (index, &value) in source[1].input.iter().enumerate() {
        v_exact[519 * 512 + index] = f64::from(value);
    }
    for row in 0..8 {
        let position = 519 + row;
        let q: Vec<f32> = (0..4096)
            .map(|i| ((i * 7 + row * 5) % 31) as f32 / 512.0 - 15.0 / 512.0)
            .collect();
        let q_f64: Vec<_> = q.iter().copied().map(f64::from).collect();
        let scale = 128.0_f32.sqrt().recip();
        let expected = sdpa_f64(
            &q_f64,
            &k_decoded,
            &v_exact,
            1,
            32,
            527,
            128,
            8,
            f64::from(scale),
            position as u32,
            0,
            None,
            Some(&sinks_f64),
        );
        let query = ctx
            .upload(&q.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>())
            .unwrap();
        let plan = cpf_attn_launch_plan(HotScratch::geometry(), position + 1, 0, CPF_ATTN_SM_COUNT)
            .unwrap();
        let part_bytes = cpf_attn_part_bytes(HotScratch::geometry(), plan).unwrap();
        let part = ctx.upload(&vec![0xff; part_bytes]).unwrap();
        let output = ctx.upload(&[0xff; 4096 * 4]).unwrap();
        let status = ctx.upload(&0_u32.to_ne_bytes()).unwrap();
        ctx.attn_decode_sinks_via_prefill(
            stream,
            AttnPrefillCall {
                storage: views[0].layout.storage,
                kv_views: Some(views),
                shape: AttnPrefillShape {
                    m: 1,
                    n_head: 32,
                    n_kv_head: 4,
                    head_dim: 128,
                    cache_rows: cells,
                    ld: 0,
                    p0: position as i64,
                },
                span: CpfAttnSpan { window: 0, ring: 0 },
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: plan.q_tile,
                scale,
                q: query.range(0, query.bytes()).unwrap(),
                proj: None,
                k_cache: views[0].payload,
                v_cache: views[1].payload,
                part: part.range(0, part_bytes).unwrap(),
                out: output.range(0, 4096 * 4).unwrap(),
                status: Some(status.range(0, 4).unwrap()),
                sinks: Some(sink_buffer.range(0, 32 * 4).unwrap()),
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut flag = [0; 4];
        status.read_at(stream, 0, &mut flag).unwrap();
        assert_eq!(u32::from_ne_bytes(flag), 0);
        let actual = values(stream, output.range(0, 4096 * 4).unwrap()).unwrap();
        let error = attention_error(&actual, &expected);
        assert!(
            error <= 1.0,
            "Q8K mixed attention mismatch at P{position}: error/bound={error}"
        );
        eprintln!("Q8K mixed attention P{position} error/bound={error}");
    }
    append.commit();
}

/// The mixed "q8k" write under the DEFAULT-ARMED device KV guard: the V f16
/// plane still takes the real row launches (only K is flag-only), so the V
/// bytes equal the pure f16 path's V plane — at a fresh append and at the
/// 519 straddle that wraps the 520-row ring. The guard comes from the
/// production default-env arming (kv_guard_word), the guarded K codes and
/// scales equal the unguarded mixed write's, and a NaN in the V source
/// latches the sticky word.
#[test]
fn device_q8k_guard_armed_keeps_v_rows_on_the_f16_path() {
    for key in [
        "FIFI_FLINT_CUDA_KV_DEVICE_F16",
        "FIFI_FLINT_CUDA_Q8_TILED",
        "FIFI_EXL3_LAYER_MAJOR",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "{key} must be unset: this regression pins the default configuration"
        );
    }
    // The default env arms the device guard (settings memoizes one read).
    assert!(crate::family::flint::settings::kv_device_f16_armed());
    if !crate::kernels::cuda::device_present("Flint Q8K guard-arm fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // The guard word exactly as the session allocates it under the default
    // env — not a hand-built buffer.
    let guard = kv_guard_word(&ctx)
        .unwrap()
        .expect("default env arms the guard");
    let bytes = [writer_plane(1.0), writer_plane(-2.0)].map(|plane| {
        plane
            .input
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>()
    });
    let inputs = bytes.each_ref().map(|b| ctx.upload(b).unwrap());
    let read_all = |buffer: &DeviceBuffer| {
        let mut out = vec![0u8; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut out).unwrap();
        out
    };
    for start in [0usize, 519] {
        let group = Group::new(&ctx, &stream, None, 0).unwrap();
        // The f16 path's V plane: same rows, both planes f16, guard armed.
        let f16_plan = KvPlan::new(2048, 8, true)
            .unwrap()
            .with_storage(KvStorage::F16);
        let mut f16_cache = sentinel_cache(&ctx, &stream, &f16_plan, start);
        let f16_append = f16_cache.cursor.begin_append(start, 8).unwrap();
        write_cache(
            &group,
            [&inputs[0], &inputs[1]],
            (
                [&f16_cache.keys, &f16_cache.values],
                f16_cache.scales.as_ref(),
                &f16_append,
            ),
            8,
            Some(&guard),
        )
        .unwrap();
        // The q8k write under the same armed guard, and the same write with
        // the guard off: the K quantize must be guard-invariant.
        let mixed_plan = KvPlan::new(2048, 8, true)
            .unwrap()
            .with_kv_storage(KvStorage::Q8G32, KvStorage::F16);
        let mut cache = sentinel_cache(&ctx, &stream, &mixed_plan, start);
        let append = cache.cursor.begin_append(start, 8).unwrap();
        write_cache(
            &group,
            [&inputs[0], &inputs[1]],
            ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
            8,
            Some(&guard),
        )
        .unwrap();
        let mut unguarded = sentinel_cache(&ctx, &stream, &mixed_plan, start);
        let bare_append = unguarded.cursor.begin_append(start, 8).unwrap();
        write_cache(
            &group,
            [&inputs[0], &inputs[1]],
            (
                [&unguarded.keys, &unguarded.values],
                unguarded.scales.as_ref(),
                &bare_append,
            ),
            8,
            None,
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            read_all(&cache.values),
            read_all(&f16_cache.values),
            "q8k V plane at {start} must equal the f16 path's V"
        );
        assert_eq!(
            read_all(&cache.keys),
            read_all(&unguarded.keys),
            "guarded K codes at {start} must equal the unguarded mixed write's"
        );
        let scales = cache.scales.as_ref().unwrap();
        let bare_scales = unguarded.scales.as_ref().unwrap();
        assert_eq!(
            read_all(&scales[0]),
            read_all(&bare_scales[0]),
            "guarded K scales at {start} must equal the unguarded mixed write's"
        );
        assert_eq!(
            read_all(&guard),
            0u32.to_ne_bytes(),
            "clean rows at {start}"
        );
    }
    // A NaN in the V source latches the sticky word under the armed guard:
    // the bf16 NaN pattern goes into the HIGH half (bytes 2..4, low half
    // zeroed) so the f32 is a true NaN and still bf16-exact — the low half
    // alone would leave the value finite.
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    let mixed_plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_kv_storage(KvStorage::Q8G32, KvStorage::F16);
    let mut cache = sentinel_cache(&ctx, &stream, &mixed_plan, 0);
    let mut nan_bytes = bytes[1].clone();
    nan_bytes[0..2].copy_from_slice(&[0, 0]);
    nan_bytes[2..4].copy_from_slice(&half::bf16::NAN.to_ne_bytes());
    let bad_v = ctx.upload(&nan_bytes).unwrap();
    let append = cache.cursor.begin_append(0, 8).unwrap();
    write_cache(
        &group,
        [&inputs[0], &bad_v],
        ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
        8,
        Some(&guard),
    )
    .unwrap();
    stream.synchronize().unwrap();
    assert_ne!(
        read_all(&guard),
        0u32.to_ne_bytes(),
        "NaN in the V source must latch the sticky guard word"
    );
}

/// The q8k regression under the DEFAULT configuration (the device KV guard
/// on, every env knob unset), on the sliding layer's ring. Lap 1 fills the
/// whole 520-row ring with real factor-A writes; lap 2 appends 8 rows at
/// 520, wrapping into rows 0..7 which hold STALE-but-valid lap-1 bytes.
/// Checks: V bytes equal the f16 twin's for the same rows (fresh and stale
/// slots), K codes/scales and V rows sit at the plan's strides, a decode over
/// the returned views reads the newly written K/V (oracle from the same
/// values; stale lap-1 rows cannot pass), and the ring wrap replaces exactly
/// the straddling slots.
#[test]
fn device_q8k_guard_default_env_ring_reuse_and_decode() {
    for key in [
        "FIFI_FLINT_CUDA_KV_DEVICE_F16",
        "FIFI_FLINT_CUDA_Q8_TILED",
        "FIFI_EXL3_LAYER_MAJOR",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "{key} must be unset: this regression pins the default configuration"
        );
    }
    // The default env arms the device KV guard; the guard word comes from
    // the production allocator, not a hand-built buffer.
    assert!(crate::family::flint::settings::kv_device_f16_armed());
    if !crate::kernels::cuda::device_present("Flint Q8K guard ring fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let guard = kv_guard_word(&ctx)
        .unwrap()
        .expect("the default env arms the device guard");
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    let upload = |factor: f32| {
        [writer_plane(factor), writer_plane(-0.5 * factor)].map(|plane| {
            plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>()
        })
    };
    let inputs_a = upload(1.0);
    let inputs_b = upload(0.5);
    let dev = |bytes: &[Vec<u8>; 2]| bytes.each_ref().map(|b| ctx.upload(b).unwrap());
    let dev_a = dev(&inputs_a);
    let dev_b = dev(&inputs_b);
    let mixed_plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_kv_storage(KvStorage::Q8G32, KvStorage::F16);
    let f16_plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_storage(KvStorage::F16);
    let mut q8k = sentinel_cache(&ctx, &stream, &mixed_plan, 0);
    let mut twin = sentinel_cache(&ctx, &stream, &f16_plan, 0);
    let cells = mixed_plan.cache_rows(3).unwrap();
    assert_eq!(cells, 520, "layer 3 is the 520-row sliding ring");
    // Plane geometry at the q8k plan: K codes 512 B/row + scales 32 B/row,
    // V f16 1024 B/row, no V scales.
    assert_eq!(q8k.keys.bytes(), cells * 512);
    assert_eq!(q8k.values.bytes(), cells * 1024);
    {
        let scales = q8k.scales.as_ref().unwrap();
        assert_eq!(scales[0].bytes(), cells * 32);
        assert_eq!(scales[1].bytes(), 0);
    }
    assert_eq!(twin.values.bytes(), cells * 1024);
    let fixture = |factor: f32| [writer_plane(factor), writer_plane(-0.5 * factor)];
    let write = |cache: &mut LayerKv, start: usize, rows: usize, inputs: &[DeviceBuffer; 2]| {
        let append = cache.cursor.begin_append(start, rows).unwrap();
        write_cache(
            &group,
            [&inputs[0], &inputs[1]],
            ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
            rows,
            Some(&guard),
        )
        .unwrap(); // views unused here; the F16 twin legitimately returns None
        append.commit();
    };
    // Lap 1: the whole ring holds REAL factor-A bytes (stale-but-valid later):
    // 65 eight-row appends fill the whole 520-row ring.
    for start in (0..520).step_by(8) {
        write(&mut q8k, start, 8, &dev_a);
        write(&mut twin, start, 8, &dev_a);
    }
    // Lap 2: append 520..527 wraps into rows 0..7, all stale-but-valid. The in-flight
    // append supplies the decode views.
    let lap2 = q8k.cursor.begin_append(520, 8).unwrap();
    let views = write_cache(
        &group,
        [&dev_b[0], &dev_b[1]],
        ([&q8k.keys, &q8k.values], q8k.scales.as_ref(), &lap2),
        8,
        Some(&guard),
    )
    .unwrap()
    .unwrap();
    let twin_lap2 = twin.cursor.begin_append(520, 8).unwrap();
    write_cache(
        &group,
        [&dev_b[0], &dev_b[1]],
        ([&twin.keys, &twin.values], None, &twin_lap2),
        8,
        Some(&guard),
    )
    .unwrap();
    stream.synchronize().unwrap();
    let read_all = |buffer: &DeviceBuffer| {
        let mut out = vec![0u8; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut out).unwrap();
        out
    };
    // (1) V equals the f16 twin's for the same rows: the straddling slots in
    // this lap hold lap-2 values, every other slot keeps its lap-1 values.
    assert_eq!(
        read_all(&q8k.values),
        read_all(&twin.values),
        "q8k V plane must equal the f16 path's V across the ring reuse"
    );
    assert_eq!(
        read_all(&guard),
        0u32.to_ne_bytes(),
        "both laps are clean under the armed guard"
    );
    // (2) K codes/scales and V rows at the plan strides: rebuild both planes
    // from the fixtures: A below 520, B at the wrapped cells 0..7.
    let extents = views[0].layout.extents(cells, 4, 128).unwrap();
    assert_eq!(extents.payload_bytes, cells * 512);
    assert_eq!(extents.scale_bytes, Some(cells * 32));
    assert_eq!(extents.bias_bytes, None);
    let strides = views[0].layout.payload_strides;
    let scalar = views[0].layout.scales.unwrap();
    let [k_a, v_a] = fixture(1.0);
    let [k_b, v_b] = fixture(0.5);
    let mut expected_codes = vec![0u8; cells * 512];
    let mut expected_scales = vec![0u8; cells * 32];
    let mut expected_v = vec![0u8; cells * 1024];
    // Logical position -> source row: 0..520 is lap 1 (fixture A tiled),
    // 520..528 is the wrapped lap-2 append (fixture B). Cell = position % ring.
    let mut k_full = vec![0.0_f64; 528 * 512];
    let mut v_full = vec![0.0_f64; 528 * 512];
    for position in 0..528 {
        let lap2 = position >= 520;
        let row = if lap2 { position - 520 } else { position % 8 };
        let (k_src, v_src) = if lap2 { (&k_b, &v_b) } else { (&k_a, &v_a) };
        let cell = position % cells;
        for head in 0..4 {
            let from = (row * 4 + head) * 128;
            let to = head * strides.head_stride_bytes + cell * strides.time_stride_bytes;
            expected_codes[to..to + 128].copy_from_slice(&k_src.codes[from..from + 128]);
            for group in 0..4 {
                let from = ((row * 4 + head) * 4 + group) * 2;
                let to = head * scalar.strides.head_stride_bytes
                    + cell * scalar.strides.time_stride_bytes
                    + group * scalar.group_stride_bytes;
                expected_scales[to..to + 2].copy_from_slice(&k_src.scales[from..from + 2]);
            }
        }
        let input_row = &v_src.input[row * 512..(row + 1) * 512];
        expected_v[cell * 1024..(cell + 1) * 1024].copy_from_slice(&f16_row(input_row).unwrap());
        k_full[position * 512..(position + 1) * 512]
            .copy_from_slice(&k_src.decoded[519 * 512 + row * 512..519 * 512 + (row + 1) * 512]);
        v_full[position * 512..(position + 1) * 512].copy_from_slice(
            &v_src.input[row * 512..(row + 1) * 512]
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(
        read_all(&q8k.keys),
        expected_codes,
        "K codes at the plan strides"
    );
    assert_eq!(
        read_all(&q8k.scales.as_ref().unwrap()[0]),
        expected_scales,
        "K scales at the plan strides"
    );
    assert_eq!(
        read_all(&q8k.values),
        expected_v,
        "V rows: lap-2 at the wrapped cells 0..7, lap-1 kept elsewhere"
    );
    // (3) A decode step over the returned views must read the newly written
    // K/V: oracle from the same fixture values, windowed like the sliding
    // layer (window 513, ring 520). Stale lap-1 rows at cells 0..6 cannot
    // pass: the oracle holds lap-2 values there.
    let sinks: Vec<f32> = (0..32)
        .map(|head| [-4.0, 0.5, 12.0][head % 3] + head as f32 / 64.0)
        .collect();
    let sink_buffer = ctx
        .upload(
            &sinks
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sinks_f64: Vec<_> = sinks.iter().copied().map(f64::from).collect();
    let position = 527;
    let q: Vec<f32> = (0..4096)
        .map(|i| ((i * 7 + 7 * 5) % 31) as f32 / 512.0 - 15.0 / 512.0)
        .collect();
    let q_f64: Vec<_> = q.iter().copied().map(f64::from).collect();
    let scale = 128.0_f32.sqrt().recip();
    let expected = sdpa_f64(
        &q_f64,
        &k_full,
        &v_full,
        1,
        32,
        528,
        128,
        8,
        f64::from(scale),
        position as u32,
        0,
        Some(513),
        Some(&sinks_f64),
    );
    let query = ctx
        .upload(&q.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>())
        .unwrap();
    let launch = cpf_attn_launch_plan(HotScratch::geometry(), 528, 0, CPF_ATTN_SM_COUNT).unwrap();
    let part_bytes = cpf_attn_part_bytes(HotScratch::geometry(), launch).unwrap();
    let part = ctx.upload(&vec![0xff; part_bytes]).unwrap();
    let output = ctx.upload(&[0xff; 4096 * 4]).unwrap();
    let status = ctx.upload(&0_u32.to_ne_bytes()).unwrap();
    ctx.attn_decode_sinks_via_prefill(
        &stream,
        AttnPrefillCall {
            storage: views[0].layout.storage,
            kv_views: Some(views),
            shape: AttnPrefillShape {
                m: 1,
                n_head: 32,
                n_kv_head: 4,
                head_dim: 128,
                cache_rows: cells,
                ld: 0,
                p0: position as i64,
            },
            span: CpfAttnSpan {
                window: 513,
                ring: cells as i64,
            },
            split: CpfAttnSplit {
                max_splits: launch.max_splits,
                min_keys: launch.min_keys,
            },
            q_tile: launch.q_tile,
            scale,
            q: query.range(0, query.bytes()).unwrap(),
            proj: None,
            k_cache: views[0].payload,
            v_cache: views[1].payload,
            part: part.range(0, part_bytes).unwrap(),
            out: output.range(0, 4096 * 4).unwrap(),
            status: Some(status.range(0, 4).unwrap()),
            sinks: Some(sink_buffer.range(0, 32 * 4).unwrap()),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut flag = [0; 4];
    status.read_at(&stream, 0, &mut flag).unwrap();
    assert_eq!(u32::from_ne_bytes(flag), 0);
    let actual = values(&stream, output.range(0, 4096 * 4).unwrap()).unwrap();
    let error = attention_error(&actual, &expected);
    assert!(
        error <= 1.0,
        "q8k guarded decode at P{position}: error/bound={error}"
    );
    eprintln!("Q8K guarded ring decode P{position} error/bound={error}");
}
