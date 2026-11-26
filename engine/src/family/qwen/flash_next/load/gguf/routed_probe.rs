//! Test-only ten-pick routed MoE probe with GPU classification and staged disk refills.
//! The host drives the existing bank policy; this is not a production executor.
#[path = "routed_probe/matched_precision.rs"]
mod matched_precision;

use super::{
    GgufModel,
    expert_layout::{
        ExpertDeviceStacks, ExpertLayout, ExpertProjection, ExpertProjectionInputs,
        PinnedDiskRefill,
    },
    route_classifier_probe::{self, Input as ClassifyInput},
    tests::PinnedRouterReference,
};
use crate::{
    family::{
        expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru},
        qwen::flash_next::step::GgufDecodePrecision,
        recipe::Role,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        resident::{
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    },
};

pub(in super::super::super) struct Inputs<'a> {
    pub layer: usize,
    /// The arithmetic the step under test runs; the probe's experts run the same one.
    pub precision: GgufDecodePrecision,
    pub host_capture: &'a std::path::Path,
    pub x: DeviceRange<'a>,
    pub route_ids: DeviceRange<'a>,
    pub route_weights: DeviceRange<'a>,
    pub initial_misses: usize,
    /// Diagnostic only: compare a different actual GPU set to the true host aggregate.
    pub diagnose_route_mismatch: bool,
    pub shared_gate: DeviceRange<'a>,
    pub shared_out: DeviceRange<'a>,
    pub block_out: DeviceRange<'a>,
}

struct ExpertCapture {
    input: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
}

fn capture(layer: usize, path: &std::path::Path, picks: &[u32]) -> (Vec<ExpertCapture>, Vec<f32>) {
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let floats = |row: &serde_json::Value, field: &str| -> Vec<f32> {
        serde_json::from_value(row[field].clone()).unwrap()
    };
    let expert_row = |name: &str, id: u32| {
        let mut matches = rows
            .iter()
            .filter(|row| row["name"] == name && row["expert"].as_u64() == Some(u64::from(id)));
        let first = matches.next().expect("missing captured expert projection");
        assert!(
            matches.next().is_none(),
            "duplicate captured expert projection"
        );
        first
    };
    let experts = picks
        .iter()
        .map(|id| {
            let gate = expert_row(&format!("blk.{layer}.ffn_gate_exps.weight"), *id);
            let up = expert_row(&format!("blk.{layer}.ffn_up_exps.weight"), *id);
            let down = expert_row(&format!("blk.{layer}.ffn_down_exps.weight"), *id);
            let input = floats(gate, "input");
            assert_eq!(input, floats(up, "input"), "host gate/up inputs differ");
            ExpertCapture {
                input,
                gate: floats(gate, "output"),
                up: floats(up, "output"),
                act: floats(down, "input"),
                down: floats(down, "output"),
            }
        })
        .collect();
    let shared = rows
        .iter()
        .find(|row| row["name"] == format!("blk.{layer}.ffn_down_shexp.weight"))
        .unwrap();
    (experts, floats(shared, "output"))
}

#[test]
fn capture_rejects_duplicate_expert_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.jsonl");
    let gate = serde_json::json!({"name":"blk.0.ffn_gate_exps.weight","expert":1,"input":[1.0],"output":[2.0]});
    let up = serde_json::json!({"name":"blk.0.ffn_up_exps.weight","expert":1,"input":[1.0],"output":[3.0]});
    let down = serde_json::json!({"name":"blk.0.ffn_down_exps.weight","expert":1,"input":[4.0],"output":[5.0]});
    let shared =
        serde_json::json!({"name":"blk.0.ffn_down_shexp.weight","expert":null,"output":[6.0]});
    let rows = [&gate, &up, &down, &shared];
    std::fs::write(
        &path,
        rows.iter().map(|x| format!("{x}\n")).collect::<String>(),
    )
    .unwrap();
    assert_eq!(capture(0, &path, &[1]).0.len(), 1);
    std::fs::write(
        &path,
        rows.iter()
            .chain([&gate].iter())
            .map(|x| format!("{x}\n"))
            .collect::<String>(),
    )
    .unwrap();
    assert!(std::panic::catch_unwind(|| capture(0, &path, &[1])).is_err());
}

fn read_f32(stream: &DeviceStream, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_ne_bytes(*v))
        .collect()
}
/// `ncols` activation rows of `k` values, quantized to q8_1: the opt-out's input.
fn quantized(
    ctx: &CudaContext,
    stream: &DeviceStream,
    x: DeviceRange<'_>,
    k: usize,
    ncols: usize,
) -> DeviceBuffer {
    let q = ctx.allocate(q8_1_bytes(k, ncols).unwrap()).unwrap();
    let call = QuantizeCall {
        x,
        y: all(&q),
        k,
        ncols,
    };
    if ncols == 1 {
        ctx.quantize_q8_1(stream, call).unwrap();
    } else {
        ctx.quantize_q8_1_rows(stream, call).unwrap();
    }
    q
}

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (err, norm) = got.iter().zip(want).fold((0f64, 0f64), |(e, n), (&a, &b)| {
        (e + f64::from(a - b).powi(2), n + f64::from(b).powi(2))
    });
    assert!(norm > 0.0 && err.is_finite());
    (err / norm).sqrt()
}
fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

struct MatchedDown<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    projection: ExpertProjection<'a>,
    slots: DeviceRange<'a>,
    input: DeviceRange<'a>,
    picks: &'a [u32],
    host_ids: &'a [u32],
    host: &'a [ExpertCapture],
}

/// Replay each selected down projection with the exact host intermediate, without changing
/// the actual propagated activation or aggregate output. This isolates the projection boundary.
fn matched_down(run: MatchedDown<'_>) {
    let MatchedDown {
        ctx,
        stream,
        projection,
        slots,
        input,
        picks,
        host_ids,
        host,
    } = run;
    let (inter, hidden, used) = (projection.k, projection.rows, picks.len());
    assert_eq!(host.len(), used);
    assert_eq!(host_ids.len(), used);
    let mut input_bytes = vec![0; hidden * 4];
    input
        .buffer()
        .read_at(stream, input.offset(), &mut input_bytes)
        .unwrap();
    let actual_input: Vec<_> = input_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_ne_bytes(*v))
        .collect();
    let mut activations = Vec::with_capacity(used * inter);
    let mut ordered = Vec::with_capacity(used);
    for id in picks {
        let rank = host_ids
            .iter()
            .position(|candidate| candidate == id)
            .unwrap();
        let reference = &host[rank];
        assert_eq!(reference.input.len(), hidden);
        assert!(
            actual_input
                .iter()
                .zip(&reference.input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "matched expert input differs from host"
        );
        assert_eq!(reference.act.len(), inter);
        assert_eq!(reference.down.len(), hidden);
        activations.extend_from_slice(&reference.act);
        ordered.push(reference);
    }
    let bytes: Vec<_> = activations.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let source = ctx.upload(&bytes).unwrap();
    assert_eq!(
        read_f32(stream, &source),
        activations,
        "host activation upload changed values"
    );
    let q8 = ctx.allocate(q8_1_bytes(inter, used).unwrap()).unwrap();
    ctx.quantize_q8_1_rows(
        stream,
        QuantizeCall {
            x: all(&source),
            y: all(&q8),
            k: inter,
            ncols: used,
        },
    )
    .unwrap();
    let output = ctx.allocate(used * hidden * 4).unwrap();
    let precision_projection =
        std::env::var_os("FIFI_FLASH_GGUF_MATCHED_F16_OUT").map(|_| ExpertProjection {
            ty: projection.ty,
            weights: projection.weights,
            k: projection.k,
            rows: projection.rows,
            slots: projection.slots,
        });
    projection
        .launch(
            ctx,
            stream,
            ExpertProjectionInputs {
                ids: slots,
                x: all(&q8),
                x_per_slot: true,
                output: all(&output),
                output_stride: hidden,
                used,
            },
        )
        .unwrap();
    let result = read_f32(stream, &output);
    if let Some(projection) = precision_projection {
        matched_precision::run(matched_precision::Experiment {
            ctx,
            stream,
            projection,
            slots,
            source: &source,
            q8: &q8,
            picks,
            host: &ordered,
            q8_reference: &result,
        });
    }
    if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_MATCHED_EXPERT_DOWN_OUT") {
        let mut q8_bytes = vec![0; q8.bytes()];
        q8.read_at(stream, 0, &mut q8_bytes).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(
            file,
            &serde_json::json!({
                "picks": picks, "input_width": inter, "output_width": hidden,
                "q8_bytes": q8_bytes, "gpu_outputs": result,
            }),
        )
        .unwrap();
    }
    let mut max_error = 0.0f64;
    for (rank, reference) in ordered.iter().enumerate() {
        let error = relative_l2(&result[rank * hidden..(rank + 1) * hidden], &reference.down);
        eprintln!(
            "matched expert down: expert={} relative_l2={error:.9}",
            picks[rank]
        );
        max_error = max_error.max(error);
    }
    assert!(
        max_error < 0.08,
        "matched expert down max relative L2={max_error}"
    );
    eprintln!(
        "matched expert down: experts={used} max_relative_l2={max_error:.9}; isolated host intermediate, not propagated execution"
    );
}

/// Isolates warm-ish source reads from the same staged synchronous device copies. The second
/// phase is not a full refill: it excludes policy and disk reads, and uses pre-read host records.
fn profile_refill(
    ctx: &CudaContext,
    stream: &DeviceStream,
    source: (&GgufModel, &ExpertLayout, usize),
    picks: &[u32],
) {
    let (model, layout, layer) = source;
    let started = std::time::Instant::now();
    let mut records = Vec::with_capacity(picks.len());
    for &id in picks {
        let mut record = vec![0; layout.bytes];
        layout
            .read(model, layer, u64::from(id), &mut record)
            .unwrap();
        records.push(record);
    }
    let read_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut bank = ExpertDeviceStacks::new(ctx, layout.clone(), picks.len()).unwrap();
    let started = std::time::Instant::now();
    for (slot, record) in records.iter().enumerate() {
        bank.upload(stream, slot, record).unwrap();
    }
    stream.synchronize().unwrap();
    let upload_ms = started.elapsed().as_secs_f64() * 1e3;
    for (index, part) in layout.parts.iter().enumerate() {
        let range = bank.part(index).unwrap();
        let mut actual = vec![0; range.bytes()];
        range
            .buffer()
            .read_at(stream, range.offset(), &mut actual)
            .unwrap();
        for (slot, record) in records.iter().enumerate() {
            let part_bytes = &record[part.bytes.clone()];
            assert_eq!(
                &actual[slot * part_bytes.len()..(slot + 1) * part_bytes.len()],
                part_bytes
            );
        }
    }
    eprintln!(
        "staged GGUF refill phases: records={} record_bytes={} read_alloc_ms={read_ms:.3} synchronous_upload_ms={upload_ms:.3} (cache state uncontrolled; separately timed)",
        picks.len(),
        layout.bytes
    );
}

/// Uses actual device-propagated input and route. On the explicit diagnostic path, compare
/// only common expert intermediates and compare the actual aggregate to the host route.
pub(in super::super::super) fn run(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: Inputs<'_>,
    want: &PinnedRouterReference,
) -> (f64, f64) {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let layer = input.layer;
    let layout = ExpertLayout::for_layer(&model, layer).unwrap();
    let used = want.route.len();
    let mut bytes = vec![0; used * 4];
    input
        .route_ids
        .buffer()
        .read_at(stream, input.route_ids.offset(), &mut bytes)
        .unwrap();
    let picks: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::try_from(i32::from_ne_bytes(*v)).unwrap())
        .collect();
    let mut gpu_set = picks.clone();
    let mut host_set: Vec<_> = want.route.iter().map(|(id, _)| *id).collect();
    gpu_set.sort_unstable();
    host_set.sort_unstable();
    let sets_match = gpu_set == host_set;
    assert!(
        sets_match || input.diagnose_route_mismatch,
        "GPU and host expert sets differ"
    );
    if !sets_match {
        eprintln!(
            "route diagnostic: gpu-only={:?}, host-only={:?}; GPU-only expert intermediate capture unavailable",
            gpu_set
                .iter()
                .filter(|id| !host_set.contains(id))
                .collect::<Vec<_>>(),
            host_set
                .iter()
                .filter(|id| !gpu_set.contains(id))
                .collect::<Vec<_>>()
        );
    }
    let experts = model.config().dims.experts as usize;
    assert!(BankGeometry::new(1, experts, used, used, layout.bytes).is_err());
    assert!(input.initial_misses <= used);
    let filler = (0..experts as u32).find(|id| !picks.contains(id)).unwrap();
    let mut resident = vec![filler];
    resident.extend(
        picks
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(rank, &id)| (rank >= input.initial_misses).then_some(id)),
    );
    for id in 0..experts as u32 {
        if resident.len() == used + 1 {
            break;
        }
        if !picks.contains(&id) && !resident.contains(&id) {
            resident.push(id);
        }
    }
    assert_eq!(resident.len(), used + 1);
    let mut bank = ExpertDeviceStacks::from_model(ctx, stream, &model, layer, &resident).unwrap();
    // Explicit test opt-in; allocation and its resource admission precede timing and policy work.
    let mut pinned = std::env::var_os("FIFI_FLASH_GGUF_PINNED_REFILL")
        .map(|_| PinnedDiskRefill::new(ctx, stream, &model, layer).unwrap());
    let geometry =
        BankGeometry::new(layer + 1, experts, used, resident.len(), layout.bytes).unwrap();
    let mut mapping = vec![-1; experts];
    for (slot, &id) in resident.iter().enumerate() {
        assert_eq!(mapping[id as usize], -1);
        mapping[id as usize] = slot as i32;
    }
    let mappings = vec![mapping.as_slice(); layer + 1];
    let counts = vec![vec![0; experts]; layer + 1];
    let policy = ReplaceOnMissLru::from_fill(geometry, &mappings, &counts, None).unwrap();
    let mut state = ExpertBankState::with_policy(geometry, HostMode::Disk, policy).unwrap();
    let classify = |state: &ExpertBankState<ReplaceOnMissLru>| {
        route_classifier_probe::classify(
            ctx,
            stream,
            ClassifyInput {
                ids: input.route_ids,
                weights: input.route_weights,
                x: input.x,
                slot_row: state.slot_row(layer),
                slots: resident.len(),
                top_k: used,
                hidden: model.config().dims.hidden as usize,
            },
        )
    };
    let mut classified = classify(&state);
    assert_eq!(classified.misses, input.initial_misses);
    assert_eq!(
        classified.picks,
        picks.iter().map(|id| *id as i32).collect::<Vec<_>>()
    );
    if input.initial_misses != 0 {
        let before = state.slot_row(layer).to_vec();
        let started = std::time::Instant::now();
        let copies = match pinned.as_mut() {
            Some(pinned) => pinned.step(&mut bank, &mut state, &classified.picks),
            None => bank.step_disk(stream, &model, &mut state, layer, &classified.picks),
        }
        .unwrap();
        stream.synchronize().unwrap();
        eprintln!(
            "staged GGUF disk refill: mode={} misses={} record_bytes={} part_copies={} elapsed_ms={:.3} (cache state uncontrolled)",
            if pinned.is_some() {
                "pinned"
            } else {
                "pageable"
            },
            copies.ops.len(),
            layout.bytes,
            copies.ops.len() * layout.parts.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(
            (copies.distinct, copies.hits, copies.ops.len()),
            (used, used - input.initial_misses, input.initial_misses)
        );
        let mut expected: Vec<_> = classified
            .picks
            .iter()
            .copied()
            .filter(|id| before[*id as usize] < 0)
            .collect();
        let mut incoming: Vec<_> = copies.ops.iter().map(|op| i32::from(op.incoming)).collect();
        expected.sort_unstable();
        incoming.sort_unstable();
        assert_eq!(incoming, expected);
        for &id in &classified.picks {
            if before[id as usize] >= 0 {
                assert_eq!(state.slot_row(layer)[id as usize], before[id as usize]);
            }
        }
        classified = classify(&state);
        assert_eq!(classified.misses, 0);
        if input.initial_misses == used
            && std::env::var_os("FIFI_FLASH_GGUF_PROFILE_REFILL").is_some()
        {
            profile_refill(ctx, stream, (&model, &layout, layer), &picks);
        }
    }
    let ids = &classified.slots;
    let gate = bank.projection(&state, Role::ExpertGate).unwrap();
    let up = bank.projection(&state, Role::ExpertUp).unwrap();
    let down = bank.projection(&state, Role::ExpertDown).unwrap();
    assert_eq!((gate.k, gate.rows), (up.k, up.rows));
    let (hidden, inter) = (gate.k, gate.rows);
    assert_eq!((down.k, down.rows), (inter, hidden));
    // The arithmetic the step runs: the f32 row for Exact, its q8_1 form only for the opt-out.
    let exact = input.precision == GgufDecodePrecision::Exact;
    let row = (!exact).then(|| quantized(ctx, stream, input.x, hidden, 1));
    let gu = ctx.allocate(used * 2 * inter * 4).unwrap();
    for (projection, offset) in [(gate, 0), (up, inter * 4)] {
        let inputs = ExpertProjectionInputs {
            ids: all(ids),
            x: row.as_ref().map_or(input.x, |q| all(q)),
            x_per_slot: false,
            output: gu.range(offset, gu.bytes() - offset).unwrap(),
            output_stride: 2 * inter,
            used,
        };
        if exact {
            projection.launch_exact(ctx, stream, inputs).unwrap();
        } else {
            projection.launch(ctx, stream, inputs).unwrap();
        }
    }
    let act = ctx.allocate(used * inter * 4).unwrap();
    ctx.swiglu_rows(
        stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows: used,
                n_ff: inter,
                up_off: inter,
                ld_in: 2 * inter,
                ld_out: inter,
            },
            gu: all(&gu),
            act: all(&act),
        },
    )
    .unwrap();
    let rows_q = (!exact).then(|| quantized(ctx, stream, all(&act), inter, used));
    let outputs = ctx.allocate(used * hidden * 4).unwrap();
    let inputs = ExpertProjectionInputs {
        ids: all(ids),
        x: rows_q.as_ref().map_or(all(&act), |q| all(q)),
        x_per_slot: true,
        output: all(&outputs),
        output_stride: hidden,
        used,
    };
    if exact {
        down.launch_exact(ctx, stream, inputs).unwrap();
    } else {
        down.launch(ctx, stream, inputs).unwrap();
    }
    let routed_buffer = ctx.allocate(hidden * 4).unwrap();
    let routed = combine(
        ctx,
        stream,
        &outputs,
        all(&classified.weights),
        None,
        all(&routed_buffer),
        hidden,
        used,
    );
    let total = combine(
        ctx,
        stream,
        &outputs,
        all(&classified.weights),
        Some(SharedTerm {
            gate: input.shared_gate,
            row: input.shared_out,
        }),
        input.block_out,
        hidden,
        used,
    );
    let host_ids: Vec<_> = want.route.iter().map(|(id, _)| *id).collect();
    let (captures, shared) = capture(layer, input.host_capture, &host_ids);
    if std::env::var_os("FIFI_FLASH_GGUF_MATCHED_EXPERT_DOWN").is_some() {
        assert!(
            sets_match,
            "matched expert diagnostic requires the host route set"
        );
        matched_down(MatchedDown {
            ctx,
            stream,
            projection: bank.projection(&state, Role::ExpertDown).unwrap(),
            slots: all(ids),
            input: input.x,
            picks: &picks,
            host_ids: &host_ids,
            host: &captures,
        });
    }
    let gu = read_f32(stream, &gu);
    let act = read_f32(stream, &act);
    let outputs = read_f32(stream, &outputs);
    let mut maxima = [0.0f64; 4];
    let mut compared = 0;
    for (rank, id) in picks.iter().enumerate() {
        let Some(host_rank) = host_ids.iter().position(|host| host == id) else {
            continue;
        };
        let cap = &captures[host_rank];
        compared += 1;
        let errors = [
            relative_l2(&gu[rank * 2 * inter..rank * 2 * inter + inter], &cap.gate),
            relative_l2(
                &gu[rank * 2 * inter + inter..(rank + 1) * 2 * inter],
                &cap.up,
            ),
            relative_l2(&act[rank * inter..(rank + 1) * inter], &cap.act),
            relative_l2(&outputs[rank * hidden..(rank + 1) * hidden], &cap.down),
        ];
        for (max, error) in maxima.iter_mut().zip(errors) {
            *max = max.max(error);
        }
    }
    let mut expected = vec![0.0f32; hidden];
    for (cap, (_, weight)) in captures.iter().zip(&want.route) {
        for (out, &x) in expected.iter_mut().zip(&cap.down) {
            *out += weight * x;
        }
    }
    let routed_error = relative_l2(&routed, &expected);
    for (out, &x) in expected.iter_mut().zip(&shared) {
        *out += want.shared_gate * x;
    }
    let total_error = relative_l2(&total, &expected);
    if compared > 0 {
        eprintln!(
            "pinned composed layer={layer} precision={:?} routed MoE initial misses={} common_experts={compared}/{used} max per-expert relative L2 gate={:.6} up={:.6} act={:.6} down={:.6}; routed={routed_error:.6} combined={total_error:.6}",
            input.precision, input.initial_misses, maxima[0], maxima[1], maxima[2], maxima[3]
        );
    } else {
        eprintln!(
            "pinned composed layer={layer}: no common expert captures; routed={routed_error:.6} combined={total_error:.6}"
        );
    }
    if sets_match {
        assert_eq!(compared, used);
        assert!(maxima.iter().all(|v| *v < 0.08), "expert errors {maxima:?}");
    }
    (routed_error, total_error)
}

#[allow(clippy::too_many_arguments)] // test helper: the kernel's inputs go in one by one
fn combine(
    ctx: &CudaContext,
    stream: &DeviceStream,
    down: &DeviceBuffer,
    weights: DeviceRange<'_>,
    shared: Option<SharedTerm<'_>>,
    out: DeviceRange<'_>,
    hidden: usize,
    used: usize,
) -> Vec<f32> {
    let zeros = ctx.upload(&vec![0; hidden * 4]).unwrap();
    ctx.moe_combine(
        stream,
        CombineCall {
            shape: CombineShape {
                m: 1,
                n: hidden,
                top_k: used,
                ld_down: hidden,
            },
            res: all(&zeros),
            down: all(down),
            weights,
            shared,
            out: CombineOut::To(out),
        },
    )
    .unwrap();
    let mut bytes = vec![0; hidden * 4];
    out.buffer()
        .read_at(stream, out.offset(), &mut bytes)
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_ne_bytes(*v))
        .collect()
}
