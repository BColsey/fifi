//! Test-only precision and timing comparison on already resident routed expert slots.
use super::*;
use crate::{
    family::qwen::flash_next::load::gguf::expert_f16_probe,
    kernels::cuda::resident::matvec_q::QuantType,
};

pub(super) struct Experiment<'a> {
    pub ctx: &'a CudaContext,
    pub stream: &'a DeviceStream,
    pub projection: ExpertProjection<'a>,
    pub slots: DeviceRange<'a>,
    pub source: &'a DeviceBuffer,
    pub q8: &'a DeviceBuffer,
    pub picks: &'a [u32],
    pub host: &'a [&'a ExpertCapture],
    pub q8_reference: &'a [f32],
}

fn elapsed(ctx: &CudaContext, stream: &DeviceStream, repeat: usize, run: impl Fn()) -> f64 {
    for _ in 0..5 {
        run();
    }
    stream.synchronize().unwrap();
    let start = ctx.event_timing().unwrap();
    let end = ctx.event_timing().unwrap();
    stream.record_timing(&start).unwrap();
    for _ in 0..repeat {
        run();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    f64::from(start.elapsed_ms_since(&end).unwrap()) / repeat as f64
}

/// Compare indexed Q8_1 against ten independent f16-intermediate GEMMs, using identical
/// resident IQ4_NL slots and each selected expert's captured F32 host activation.
pub(super) fn run(experiment: Experiment<'_>) {
    let Experiment {
        ctx,
        stream,
        projection,
        slots,
        source,
        q8,
        picks,
        host,
        q8_reference,
    } = experiment;
    assert_eq!(
        projection.ty,
        QuantType::Iq4Nl,
        "this fixture selects IQ4_NL down weights"
    );
    let (k, rows, used) = (projection.k, projection.rows, picks.len());
    assert_eq!(used, host.len());
    assert_eq!(q8_reference.len(), used * rows);
    assert_eq!(source.bytes(), used * k * 4);
    assert_eq!(slots.bytes(), used * 4);
    assert_eq!(q8.bytes(), q8_1_bytes(k, used).unwrap());
    let mut raw = vec![0; slots.bytes()];
    slots
        .buffer()
        .read_at(stream, slots.offset(), &mut raw)
        .unwrap();
    let slot_ids: Vec<i32> = raw
        .chunks_exact(4)
        .map(|x| i32::from_ne_bytes(x.try_into().unwrap()))
        .collect();
    let f16_output = ctx.allocate(used * rows * 4).unwrap();
    let q8_output = ctx.allocate(used * rows * 4).unwrap();
    let poison = f32::NAN.to_ne_bytes().repeat(used * rows);
    f16_output.write_at(stream, 0, &poison).unwrap();
    q8_output.write_at(stream, 0, &poison).unwrap();
    let calls =
        expert_f16_probe::plan(&projection, &slot_ids, all(source), all(&f16_output)).unwrap();
    let f16 = || expert_f16_probe::launch(ctx, stream, &calls).unwrap();
    let q8_launch = |quantize: bool| {
        if quantize {
            ctx.quantize_q8_1_rows(
                stream,
                QuantizeCall {
                    x: all(source),
                    y: all(q8),
                    k,
                    ncols: used,
                },
            )
            .unwrap();
        }
        ExpertProjection {
            ty: projection.ty,
            weights: projection.weights,
            k,
            rows,
            slots: projection.slots,
        }
        .launch(
            ctx,
            stream,
            ExpertProjectionInputs {
                ids: slots,
                x_q8: all(q8),
                x_per_slot: true,
                output: all(&q8_output),
                output_stride: rows,
                used,
            },
        )
        .unwrap();
    };
    f16();
    q8_launch(true);
    let f16_values = read_f32(stream, &f16_output);
    let q8_values = read_f32(stream, &q8_output);
    assert!(f16_values.iter().all(|v| v.is_finite()));
    assert!(
        q8_values
            .iter()
            .zip(q8_reference)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "timed Q8 path differs from the matched baseline"
    );
    let mut max_f16 = 0.0f64;
    for (rank, expected) in host.iter().enumerate() {
        let error = relative_l2(&f16_values[rank * rows..(rank + 1) * rows], &expected.down);
        eprintln!(
            "matched expert f16: expert={} relative_l2={error:.9}",
            picks[rank]
        );
        max_f16 = max_f16.max(error);
    }
    assert!(max_f16 < 0.01, "FP16 diagnostic max rel L2={max_f16}");
    let repeat = 40;
    let mut measurements = Vec::new();
    for mode in [
        "q8_projection",
        "f16_gemm",
        "q8_quant_projection",
        "q8_quant_projection",
        "f16_gemm",
        "q8_projection",
    ] {
        let time = match mode {
            "q8_projection" => elapsed(ctx, stream, repeat, || q8_launch(false)),
            "q8_quant_projection" => elapsed(ctx, stream, repeat, || q8_launch(true)),
            "f16_gemm" => elapsed(ctx, stream, repeat, &f16),
            _ => unreachable!(),
        };
        assert!(time.is_finite() && time > 0.0);
        eprintln!(
            "matched expert event timing: mode={mode} total_experts={used} repeat={repeat} ms_per_ten={time:.6} (resident weights; excludes refill, routing, allocation and readback)"
        );
        measurements.push((mode, time));
    }
    let after = read_f32(stream, &f16_output);
    assert!(
        after
            .iter()
            .zip(&f16_values)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let after = read_f32(stream, &q8_output);
    assert!(
        after
            .iter()
            .zip(q8_reference)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let path =
        std::env::var_os("FIFI_FLASH_GGUF_MATCHED_F16_OUT").expect("new matched precision receipt");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(
        file,
        &serde_json::json!({"picks":picks, "slot_ids":slot_ids,
        "input_width":k, "output_width":rows, "f16_outputs":f16_values,
        "measurements_ms_per_all_experts":measurements, "repeat":repeat}),
    )
    .unwrap();
    eprintln!(
        "matched expert f16: experts={used} max_relative_l2={max_f16:.9}; isolated precision experiment, not model parity"
    );
}
