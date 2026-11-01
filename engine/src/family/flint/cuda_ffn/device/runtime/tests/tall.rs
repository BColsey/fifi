//! The tall projection and the jobs piece on the real pack: the rows entry against the
//! small-row entry, the slicing against another slicing, and one jobs piece against the
//! token-major walk. Run with `FLINT_EXL3_PACK` and an admitted GPU.

use super::real::load_real_model;
use super::*;
use crate::family::flint::exl3_resident::FlintFfnView;
use crate::family::project::cuda::exl3::Exl3Io;

/// Deterministic unit-scale activations: a sum of three uniforms, centred.
fn activations(rows: usize, width: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..rows * width)
        .map(|_| (next() + next() + next() - 1.5) * 2.0)
        .collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn read(stream: &DeviceStream, buffer: &DeviceBuffer) -> Vec<f32> {
    stream.synchronize().unwrap();
    values(stream, buffer.range(0, buffer.bytes()).unwrap()).unwrap()
}

/// How one run of `rows` rows reaches `weight`.
#[derive(Clone, Copy)]
enum Route {
    /// `exl3_linear` in groups of at most eight rows.
    Small,
    /// `linear_rows`, which cuts the rows itself.
    Tall,
    /// The rows entry over these `(first, count)` runs.
    Runs(&'static [(usize, usize)]),
}

fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: Exl3MatrixView<'_>,
    x: &[f32],
    route: Route,
) -> Vec<f32> {
    let [k, n] = weight.shape;
    let rows = x.len() / k;
    let input = ctx.upload(&floats(x)).unwrap();
    let flag = ctx.upload(&[0; 4]).unwrap();
    let xh = ctx.allocate(rows * k * 4).unwrap();
    let z = ctx.allocate(rows * n * 4).unwrap();
    let out = ctx.allocate(rows * n * 4).unwrap();
    let part = |first: usize, count: usize| Exl3Io {
        x: input.range(first * k * 4, count * k * 4).unwrap(),
        xh: xh.range(first * k * 4, count * k * 4).unwrap(),
        z: z.range(first * n * 4, count * n * 4).unwrap(),
        out: out.range(first * n * 4, count * n * 4).unwrap(),
        flag: flag.range(0, 4).unwrap(),
        rows: count,
        fused: false,
    };
    match route {
        Route::Small => {
            for first in (0..rows).step_by(8) {
                let count = (rows - first).min(8);
                ctx.exl3_linear(stream, weight.linear_call(&part(first, count)))
                    .unwrap();
            }
        }
        Route::Tall => exl3::linear_rows(ctx, stream, weight, part(0, rows)).unwrap(),
        Route::Runs(runs) => {
            for &(first, count) in runs {
                ctx.exl3_linear_midrows(stream, weight.linear_call(&part(first, count)))
                    .unwrap();
            }
        }
    }
    let status = read(stream, &flag);
    assert_eq!(status[0].to_bits(), 0, "the arithmetic flag stayed clear");
    read(stream, &out)
}

fn relative_error(reference: &[f32], candidate: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(candidate) {
        diff += f64::from(a - b).powi(2);
        norm += f64::from(*a).powi(2);
    }
    (diff / norm).sqrt()
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; tall projections against the small-row entry"]
fn tall_projections_track_the_small_entry_and_ignore_the_slicing() {
    // 300 rows are three even runs of 100; this slicing differs from that one. Every run is a
    // tensor-core run (64 rows or more): a shorter run takes the SIMT tile, whose sums differ.
    const OTHER: &[(usize, usize)] = &[(0, 64), (64, 136), (200, 100)];
    let ctx = CudaContext::new().expect("real weights need CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(576, 8, true).unwrap();
    let model = load_real_model(&ctx, &stream, &[&plan]);
    let attention = model.layer(3).unwrap();
    let shared = match model.layer(10).unwrap().ffn {
        FlintFfnView::Moe { shared, .. } => shared,
        FlintFfnView::Dense(_) => panic!("layer 10 is a MoE layer"),
    };
    let weights = [
        ("q_proj", attention.q),
        ("k_proj", attention.k),
        ("o_proj", attention.o),
        ("shared gate", shared.gate),
        ("shared down", shared.down),
    ];
    for (name, weight) in weights {
        let x = activations(300, weight.shape[0], 11);
        let tall = project(&ctx, &stream, weight, &x, Route::Tall);
        let other = project(&ctx, &stream, weight, &x, Route::Runs(OTHER));
        assert!(
            tall.iter()
                .zip(&other)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: a row does not depend on which run it sits in"
        );
        let small = project(&ctx, &stream, weight, &x, Route::Small);
        let error = relative_error(&small, &tall);
        println!("{name}: 300 rows, tall against small, relative L2 error {error:e}");
        assert!(error < 5e-3, "{name}: error {error:e}");
        // The class switches: 9 rows and a 37-row tail take the SIMT tile, 63 the last SIMT
        // height, 64 the first tensor-core height, 145 two even tensor-core runs.
        for rows in [9usize, 37, 63, 64, 145] {
            let head = &x[..rows * weight.shape[0]];
            let tall = project(&ctx, &stream, weight, head, Route::Tall);
            let small = project(&ctx, &stream, weight, head, Route::Small);
            let error = relative_error(&small, &tall);
            println!("{name}: {rows} rows, tall against small, relative L2 error {error:e}");
            assert!(error < 5e-3, "{name}, {rows} rows: error {error:e}");
        }
    }
}

/// The F16 values of the first `rows` positions of one KV plane.
fn plane_rows(stream: &DeviceStream, plane: &DeviceBuffer, rows: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; rows * 4 * 128 * 2];
    plane.read_at(stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half::f16::from_bits(u16::from_ne_bytes(*pair)).to_f32())
        .collect()
}

/// Relative L2 error of the committed K and V of every layer of `candidate` against `reference`:
/// the hidden-state error each layer inherited, so the figures show how error grows with depth.
/// A ring has not wrapped for the rows these tests feed, so position `p` is at slot `p`.
fn layer_errors(
    stream: &DeviceStream,
    reference: &State,
    candidate: &State,
    rows: usize,
) -> Vec<(f64, f64)> {
    reference
        .kv
        .iter()
        .zip(&candidate.kv)
        .map(|(left, right)| {
            let error = |a: &DeviceBuffer, b: &DeviceBuffer| {
                relative_error(&plane_rows(stream, a, rows), &plane_rows(stream, b, rows))
            };
            (
                error(&left.keys, &right.keys),
                error(&left.values, &right.values),
            )
        })
        .collect()
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; a jobs piece against the token-major walk"]
fn a_jobs_piece_tracks_the_token_major_walk() {
    jobs_piece_against_token_major(KvStorage::F16);
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; a q8 jobs piece against the q8 token-major walk"]
fn a_q8_jobs_piece_tracks_the_q8_token_major_walk() {
    jobs_piece_against_token_major(KvStorage::Q8G32);
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; the runner pins FIFI_FLINT_CUDA_Q8_TILED=0"]
fn a_q8_jobs_piece_tracks_the_q8_token_major_walk_on_the_legacy_route() {
    // The tiled route is read once per process, so the runner sets the pin; without it this
    // would repeat the tiled run.
    assert!(
        !crate::family::flint::settings::q8_tiled_armed(),
        "the GPU runner must pin FIFI_FLINT_CUDA_Q8_TILED=0"
    );
    jobs_piece_against_token_major(KvStorage::Q8G32);
}

/// One jobs piece against the token-major walk and the tile pass on one KV profile. The per-layer
/// K/V comparison reads f16 planes, so the q8 profile is judged on its logits.
fn jobs_piece_against_token_major(storage: KvStorage) {
    let ctx = CudaContext::new().expect("real weights need CUDA");
    let stream = ctx.stream().unwrap();
    let rows_list = [9usize, 100];
    let top = *rows_list.last().unwrap();
    let narrow = KvPlan::new(576, 8, true).unwrap().with_storage(storage);
    let wide = KvPlan::new(576, top, true).unwrap().with_storage(storage);
    let model = load_real_model(&ctx, &stream, &[&narrow, &narrow, &wide]);
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    // The token-major walk is the reference; the tile pass is the earlier layer-major pass, whose
    // distance from the reference puts the jobs pass's distance in scale.
    let mut reference = State::new(&ctx, narrow).unwrap();
    let mut tiles = State::new(&ctx, narrow).unwrap();
    tiles.arm_layer_major(&ctx, top, false).unwrap();
    let mut jobs = State::new(&ctx, wide).unwrap();
    jobs.arm_layer_major(&ctx, top, true).unwrap();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    for rows in rows_list {
        let tokens: Vec<u32> = (0..rows)
            .map(|position| 630 + 31 * position as u32)
            .collect();
        for state in [&mut reference, &mut tiles, &mut jobs] {
            state.reset(&stream).unwrap();
        }
        for chunk in tokens.chunks(8) {
            decoder.feed_rows(&mut reference, chunk).unwrap();
        }
        decoder
            .feed_rows_layer_major(&mut tiles, &tokens, 8)
            .unwrap();
        decoder
            .feed_rows_layer_major(&mut jobs, &tokens, rows)
            .unwrap();
        let next = 630 + 31 * rows as u32;
        let mut logits = Vec::new();
        for state in [&mut reference, &mut tiles, &mut jobs] {
            assert_eq!(state.fed(), rows);
            logits.push(
                decoder
                    .step(state, next, |_, logits| Ok(logits.to_vec()))
                    .unwrap(),
            );
        }
        for (name, candidate) in [("tiles", &logits[1]), ("jobs", &logits[2])] {
            let error = relative_error(&logits[0], candidate);
            println!(
                "rows {rows}: {name} logits relative L2 error {error:e}, argmax {} / {}",
                argmax(&logits[0]),
                argmax(candidate)
            );
            assert_eq!(argmax(&logits[0]), argmax(candidate), "rows {rows} {name}");
            assert!(error < 5e-2, "rows {rows} {name}: error {error:e}");
        }
        if rows == top && storage == KvStorage::F16 {
            for (name, state) in [("tiles", &tiles), ("jobs", &jobs)] {
                let growth = layer_errors(&stream, &reference, state, rows);
                for (layer, (keys, values)) in growth.iter().enumerate() {
                    println!("{name} layer {layer:2}: K rel L2 {keys:.3e}  V rel L2 {values:.3e}");
                }
                // The figures are the result; a bound waits for what the tile pass shows. A layer
                // whose cache is unrelated to the walk's (error near 1) is still a failure.
                let worst = growth.iter().map(|&(k, v)| k.max(v)).fold(0f64, f64::max);
                assert!(worst < 1.0, "{name}: a layer's KV is unrelated: {worst:e}");
            }
        }
    }
}

/// Rows of K or V as the attention half publishes them: finite and BF16-exact, so the cache
/// guards accept every row.
fn published_rows(rows: usize, seed: u64) -> Vec<f32> {
    activations(rows, 512, seed)
        .into_iter()
        .map(|value| f32::from_bits(value.to_bits() & 0xffff_0000))
        .collect()
}

/// Appends `rows` rows at `position` to one layer, `height` rows per `write_cache` call.
fn append_in_blocks(
    ctx: &CudaContext,
    stream: &DeviceStream,
    state: &mut State,
    layer: usize,
    (position, rows, height): (usize, usize, usize),
    seed: u64,
) {
    let (keys, values) = (published_rows(rows, seed), published_rows(rows, seed + 1));
    let group = Group::new(ctx, stream, None, 0).unwrap();
    let cache = &mut state.kv[layer];
    for first in (0..rows).step_by(height) {
        let count = (rows - first).min(height);
        let append = cache.cursor.begin_append(position + first, count).unwrap();
        let k = ctx
            .upload(&floats(&keys[first * 512..(first + count) * 512]))
            .unwrap();
        let v = ctx
            .upload(&floats(&values[first * 512..(first + count) * 512]))
            .unwrap();
        write_cache(
            &group,
            [&k, &v],
            ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
            count,
            None,
        )
        .unwrap();
        append.commit();
    }
    stream.synchronize().unwrap();
}

fn cache_bytes(stream: &DeviceStream, state: &State, layer: usize) -> Vec<Vec<u8>> {
    let cache = &state.kv[layer];
    let mut planes = vec![&cache.keys, &cache.values];
    if let Some(scales) = &cache.scales {
        planes.extend(scales.iter());
    }
    planes
        .into_iter()
        .map(|plane| {
            let mut bytes = vec![0u8; plane.bytes()];
            plane.read_at(stream, 0, &mut bytes).unwrap();
            bytes
        })
        .collect()
}

/// A jobs piece appends its rows in one `write_cache` call where the token-major walk appends
/// them eight at a time; the stored bytes (f16 halves, or q8 codes and scales) are the same, on
/// a full layer and on a sliding layer whose ring the piece crosses: the piece then lands in two
/// runs, the second starting part-way into the piece's source rows.
#[test]
#[ignore = "requires an admitted GPU"]
fn a_piece_append_stores_the_bytes_of_block_appends() {
    let ctx = CudaContext::new().expect("CUDA");
    let stream = ctx.stream().unwrap();
    const PIECE: usize = 100;
    for storage in [KvStorage::F16, KvStorage::Q8G32] {
        let plan = || {
            KvPlan::new(1500, PIECE, true)
                .unwrap()
                .with_storage(storage)
        };
        let mut whole = State::new(&ctx, plan()).unwrap();
        let mut blocks = State::new(&ctx, plan()).unwrap();
        for layer in [0, 2] {
            // The piece starts 50 rows before the end of the layer's ring; a full layer's ring is
            // the whole context, so there it does not cross.
            let ring = whole.plan.cache_rows(layer).unwrap();
            let start = if layer == 2 { ring - 50 } else { ring / 2 };
            assert!(layer == 0 || (start < ring && start + PIECE > ring));
            for state in [&mut whole, &mut blocks] {
                append_in_blocks(&ctx, &stream, state, layer, (0, start, 8), 11);
            }
            append_in_blocks(&ctx, &stream, &mut whole, layer, (start, PIECE, PIECE), 29);
            append_in_blocks(&ctx, &stream, &mut blocks, layer, (start, PIECE, 8), 29);
            let (left, right) = (
                cache_bytes(&stream, &whole, layer),
                cache_bytes(&stream, &blocks, layer),
            );
            assert_eq!(left.len(), right.len());
            for (index, (a, b)) in left.iter().zip(&right).enumerate() {
                assert!(a == b, "{storage:?} layer {layer} plane {index} differs");
            }
        }
    }
}

/// Where one jobs piece of real text goes. Run in release: the marks wait for the stream, so the
/// table ranks stages; the unmarked wall clock after it is the figure to quote.
#[test]
#[ignore = "requires admitted GPU, full EXL3 weights and FLINT_WINDOW_TOKENS; timing, run in release"]
fn a_jobs_piece_stage_split() {
    use crate::family::flint::cuda_ffn::device::stage_clock;
    let path = std::env::var_os("FLINT_WINDOW_TOKENS").expect("real token ids");
    let all: Vec<u32> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let rows: usize = std::env::var("FLINT_SPLIT_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    // The piece is timed after `depth` rows of earlier context, so its attention reads that many
    // keys. The real ids of FLINT_WINDOW_TOKENS repeat to fill the depth; attention cost does
    // not depend on them.
    let depth: usize = std::env::var("FLINT_SPLIT_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let cycled: Vec<u32> = all.iter().copied().cycle().take(depth + rows).collect();
    let (before, tokens) = cycled.split_at(depth);
    // The serving context: the allocation cache on, with the limits the jobs pass raises.
    let ctx =
        CudaContext::with_admission_cache(&crate::family::project::cuda::served_admission(), true)
            .expect("real weights need CUDA");
    ctx.set_alloc_cache_limits(
        64 << 20,
        usize::try_from(State::layer_jobs_idle_bytes(rows).unwrap()).unwrap(),
    );
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(depth + rows + 64, rows, true).unwrap();
    let model = load_real_model(&ctx, &stream, &[&plan]);
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut state = State::new(&ctx, plan).unwrap();
    state.arm_layer_major(&ctx, rows, true).unwrap();
    // One warm-up piece compiles nothing but fills the allocator caches and the clocks.
    for piece in before.chunks(rows) {
        decoder
            .feed_rows_layer_major(&mut state, piece, rows)
            .unwrap();
    }
    decoder
        .feed_rows_layer_major(&mut state, tokens, rows)
        .unwrap();
    state.reset(&stream).unwrap();
    for piece in before.chunks(rows) {
        decoder
            .feed_rows_layer_major(&mut state, piece, rows)
            .unwrap();
    }
    stage_clock::arm();
    decoder
        .feed_rows_layer_major(&mut state, tokens, rows)
        .unwrap();
    let table = stage_clock::take();
    let total: f64 = table.iter().map(|row| row.1).sum();
    println!("marked piece of {rows} rows at depth {depth}: {total:.1} ms");
    for (stage, ms, marks) in &table {
        println!(
            "STAGE {stage:48} {ms:9.1} ms {:5.1}%  marks {marks}",
            100.0 * ms / total
        );
    }
    state.reset(&stream).unwrap();
    for piece in before.chunks(rows) {
        decoder
            .feed_rows_layer_major(&mut state, piece, rows)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let begin = std::time::Instant::now();
    decoder
        .feed_rows_layer_major(&mut state, tokens, rows)
        .unwrap();
    stream.synchronize().unwrap();
    println!(
        "WALL unmarked piece of {rows} rows at depth {depth}: {:.1} ms",
        begin.elapsed().as_secs_f64() * 1e3
    );
}

/// The error against ideal FP64 of the F32 small-row entry (the token-major walk's arithmetic)
/// and of the F16-operand rows entry (the jobs pass), for the matrices of real layers: attention
/// q, k, v, o at layers 3, 20, 47 and 48 and the gate, up and down of expert 7 at layers 3, 20
/// and 47. 64 unit-scale rows each, the shortest run that takes the tensor-core kernel. The
/// reference reconstructs the dense weights from the pack on the host.
#[test]
#[ignore = "requires admitted GPU, full EXL3 weights; reads about 30 matrices on the host, run in release"]
fn projections_against_ideal_f64() {
    use crate::weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix};
    let pack = std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in pack"));
    let checkpoint = Checkpoint::open(&pack).unwrap();
    let reference = |prefix: &str, x: &[f32], rows: usize| -> Vec<f32> {
        let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let keys = names.iter().map(String::as_str).collect::<Vec<_>>();
        let selected = checkpoint.source.read_selected(&keys, 1 << 30).unwrap();
        let matrix = Matrix::load(&selected, prefix, &checkpoint.config).unwrap();
        let [_, n] = matrix.shape();
        let input: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let mut out = vec![0.0f64; rows * n];
        matrix.project_fp64_into(&input, rows, &mut out).unwrap();
        out.into_iter().map(|v| v as f32).collect()
    };
    let ctx = CudaContext::new().expect("real weights need CUDA");
    let stream = ctx.stream().unwrap();
    let plan = KvPlan::new(576, 8, true).unwrap();
    let model = load_real_model(&ctx, &stream, &[&plan]);
    println!(
        "F64ANCHOR matrix                                   F32 small entry    F16-operand rows"
    );
    for layer in [3usize, 20, 47, 48] {
        let view = model.layer(layer).unwrap();
        let FlintLayerView {
            q, k, v, o, ffn, ..
        } = view;
        let mut cases = vec![
            (format!("model.layers.{layer}.self_attn.q_proj"), q),
            (format!("model.layers.{layer}.self_attn.k_proj"), k),
            (format!("model.layers.{layer}.self_attn.v_proj"), v),
            (format!("model.layers.{layer}.self_attn.o_proj"), o),
        ];
        let moe = match &ffn {
            FlintFfnView::Moe { experts, .. } if layer != 48 => Some(experts),
            _ => None,
        };
        if let Some(experts) = moe {
            let expert = experts.get(7).unwrap();
            for (role, weight) in [
                ("gate_proj", expert.gate),
                ("up_proj", expert.up),
                ("down_proj", expert.down),
            ] {
                cases.push((format!("model.layers.{layer}.mlp.experts.7.{role}"), weight));
            }
        }
        for (prefix, weight) in cases {
            let x = activations(64, weight.shape[0], 29 + layer as u64);
            let ideal = reference(&prefix, &x, 64);
            let small = project(&ctx, &stream, weight, &x, Route::Small);
            let tall = project(&ctx, &stream, weight, &x, Route::Tall);
            println!(
                "F64ANCHOR {prefix:46} {:.3e}          {:.3e}",
                relative_error(&ideal, &small),
                relative_error(&ideal, &tall)
            );
            assert!(
                relative_error(&ideal, &small) < 1e-4,
                "{prefix}: the F32 entry is off"
            );
            assert!(
                relative_error(&ideal, &tall) < 5e-3,
                "{prefix}: the rows entry is off"
            );
        }
    }
}
