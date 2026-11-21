use std::{hint::black_box, time::Instant};

use super::{QKV_WEIGHTS, SafetensorDense, bf16_worker_limit, full_range, widen_bf16};
use crate::{
    family::mlx_budget::{DeviceBudget, PRESSURE_MARGIN},
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use half::bf16;

const BATCHES: [usize; 3] = [1, 4, 16];
const WIDTH: usize = 2304;
const HEIGHTS: [usize; 3] = [3072, 1024, 1024];
const ROUNDS: usize = 9;
const RESIDENT_BYTES: usize = WIDTH * (3072 + 1024 + 1024) * 4;
type Results = [Vec<f32>; 3];
type Inputs = [Vec<f32>; 3];

fn f32_tensor(rows: usize, cols: usize, data: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        vec![rows as u32, cols as u32],
        data.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn inputs(layer: usize) -> Inputs {
    std::array::from_fn(|projection| {
        (0..BATCHES[2] * WIDTH)
            .map(|i| {
                let n = (i * 73 + i / WIDTH * 137 + projection * 331 + layer * 19) % 4093;
                bf16::from_f32((n as f32 - 2046.0) / 1024.0).to_f32()
            })
            .collect()
    })
}

fn cpu(dense: &SafetensorDense, names: &[String; 3], input: &Inputs, rows: usize) -> Results {
    let mut out = std::array::from_fn(|i| Vec::with_capacity(rows * HEIGHTS[i]));
    for row in 0..rows {
        for i in 0..3 {
            let input = &input[i][row * WIDTH..(row + 1) * WIDTH];
            out[i].extend(dense.matvec(&names[i], input).unwrap());
        }
    }
    out
}

fn metal(
    kernels: &mut MlxKernels,
    weights: &[ExecutionTensor],
    input: &Inputs,
    rows: usize,
    piece: usize,
) -> Results {
    let mut out = std::array::from_fn(|i| Vec::with_capacity(rows * HEIGHTS[i]));
    for row in (0..rows).step_by(piece) {
        for i in 0..3 {
            let activation =
                f32_tensor(piece, WIDTH, &input[i][row * WIDTH..(row + piece) * WIDTH]);
            let activation = kernels.import_tensor(&activation).unwrap();
            let projected = kernels.mm(&activation, &weights[i]).unwrap();
            let result = kernels.readback(&projected).unwrap();
            assert_eq!(result.shape(), &[piece as u32, HEIGHTS[i] as u32]);
            assert_eq!(result.dtype(), Dtype::F32);
            out[i].extend(
                result
                    .host_bytes()
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&word| f32::from_le_bytes(word)),
            );
        }
    }
    out
}

fn timed(mut run: impl FnMut() -> Results) -> (Results, Vec<u128>) {
    let reference = run();
    let mut durations = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let started = Instant::now();
        black_box(run());
        durations.push(started.elapsed().as_nanos());
    }
    durations.sort_unstable();
    (reference, durations)
}

fn cpu_measurement(name: &str, layer: usize, rows: usize, time: &[u128]) {
    println!(
        "inkling_dense_placement {}",
        serde_json::json!({
            "backend": name, "layer": layer, "rows": rows,
            "samples_ns": time, "median_ns": time[ROUNDS / 2],
        })
    );
}

fn compare_metal(
    kernel: &str,
    layer: usize,
    rows: usize,
    expected: &Results,
    actual: &Results,
    time: &[u128],
) {
    let mut mismatches = 0;
    let mut compare_count = 0;
    let mut max_abs = 0.0_f32;
    let mut max_scaled = 0.0_f32;
    for (expected, actual) in expected.iter().zip(actual) {
        assert_eq!(expected.len(), actual.len());
        for (&a, &b) in expected.iter().zip(actual) {
            assert!(a.is_finite() && b.is_finite());
            compare_count += 1;
            mismatches += usize::from(bf16::from_f32(a) != bf16::from_f32(b));
            let err = (a - b).abs();
            max_abs = max_abs.max(err);
            max_scaled = max_scaled.max(err / a.abs().max(1.0));
        }
    }
    println!(
        "inkling_dense_placement {}",
        serde_json::json!({
            "backend": kernel, "layer": layer, "rows": rows,
            "samples_ns": time, "median_ns": time[ROUNDS / 2],
            "max_abs_error_f32": max_abs, "max_abs_error_scaled": max_scaled,
            "bf16_mismatches": mismatches, "values_compared": compare_count,
        })
    );
    assert!(
        max_scaled <= 2e-4,
        "{kernel} layer={layer} rows={rows} changed output: {max_scaled}"
    );
}

#[test]
#[ignore = "requires the publisher safetensors checkpoint and native Metal"]
fn publisher_qkv_placement_real_weights() {
    let checkpoint = std::env::var_os("FIFI_TEST_INKLING_CHECKPOINT")
        .expect("set FIFI_TEST_INKLING_CHECKPOINT to the publisher safetensors directory");
    let checkpoint = std::path::Path::new(&checkpoint);
    let _lock = crate::native_test_lock();
    assert_eq!(
        bf16_worker_limit(),
        4,
        "benchmark requires four CPU workers"
    );

    // This component benchmark owns one layer at a time. Charge the active
    // weights, a full 128 MiB transient allowance for transpose/import, arena
    // and GEMM work, and the existing family pressure margin.
    DeviceBudget::detect_with_default(true)
        .unwrap()
        .expect("keep automatic Metal admission enabled for the experiment")
        .remaining(
            "Inkling QKV benchmark",
            &[
                ("persistent_f32_qkv", RESIDENT_BYTES as u64),
                ("transient_allowance", 128 << 20),
                ("pressure_margin", PRESSURE_MARGIN),
            ],
        )
        .unwrap();

    for layer in [0, 5] {
        let names = QKV_WEIGHTS.map(|weight| format!("model.llm.layers.{layer}.attn.{weight}"));
        let input = inputs(layer);
        let mut dense = SafetensorDense::open(checkpoint).unwrap();
        let mut expected = Vec::with_capacity(BATCHES.len());
        for rows in BATCHES {
            dense.attention_workers = 1;
            let (serial, serial_time) = timed(|| cpu(&dense, &names, &input, rows));
            dense.attention_workers = 4;
            let (parallel, parallel_time) = timed(|| cpu(&dense, &names, &input, rows));
            for (a, b) in serial.iter().zip(&parallel) {
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            cpu_measurement("cpu_1", layer, rows, &serial_time);
            cpu_measurement("cpu_4", layer, rows, &parallel_time);
            expected.push(serial);
        }
        drop(dense);

        let mut kernels = MlxKernels::try_new().unwrap();
        let before = kernels.memory_stats().unwrap();
        let source = SafetensorDense::open(checkpoint).unwrap();
        let start = Instant::now();
        let mut weights = Vec::with_capacity(3);
        for (name, h) in names.iter().zip(HEIGHTS) {
            let entry = source.entry(name).unwrap();
            assert_eq!(entry.info.shape, [h, WIDTH]);
            assert_eq!(entry.info.dtype, safetensors::Dtype::BF16);
            // The model's row-major packed plane survives only this import.
            let host = source
                .with_bytes(name, entry, full_range(entry), false, |packed| {
                    let mut transposed = vec![0; h * WIDTH * 4];
                    for (row, packed) in packed.chunks_exact(WIDTH * 2).enumerate() {
                        for (col, &bytes) in packed.as_chunks::<2>().0.iter().enumerate() {
                            let at = (col * h + row) * 4;
                            transposed[at..at + 4]
                                .copy_from_slice(&widen_bf16(bytes).to_le_bytes());
                        }
                    }
                    Tensor::try_from_host(Dtype::F32, vec![WIDTH as u32, h as u32], transposed)
                })
                .unwrap()
                .unwrap();
            assert!(entry.bytes.get().is_none());
            weights.push(kernels.import_tensor(&host).unwrap());
        }
        let import_elapsed = start.elapsed().as_nanos();
        drop(source);

        // Measure serial Metal to match the current input dependency graph,
        // and batched Metal separately to price a future changed graph.
        for (rows, expected) in BATCHES.into_iter().zip(&expected) {
            let (serial, timings) = timed(|| metal(&mut kernels, &weights, &input, rows, 1));
            compare_metal("metal_serial", layer, rows, expected, &serial, &timings);
            if rows > 1 {
                let (batched, timings) =
                    timed(|| metal(&mut kernels, &weights, &input, rows, rows));
                compare_metal("metal_batched", layer, rows, expected, &batched, &timings);
            }
        }
        let after = kernels.memory_stats().unwrap();
        println!(
            "inkling_dense_placement_memory {}",
            serde_json::json!({
                "layer": layer, "weight_payload_bytes": RESIDENT_BYTES, "import_elapsed_ns": import_elapsed,
                "process_active_before": before.active, "process_pool_before": before.cache,
                "process_active_after": after.active, "process_pool_after": after.cache,
                "process_peak_active": after.peak,
            })
        );
    }
}
