//! Opt-in routing and diagnostics for Inkling's existing multirow attention projections.

use std::sync::OnceLock;

use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceStream,
        resident::dense_gemv::{DenseDtype, DenseGemvCall},
    },
    weights::GgufType,
};

fn flags() -> (bool, bool) {
    static FLAGS: OnceLock<(bool, bool)> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        (
            std::env::var("FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE").as_deref() == Ok("1"),
            std::env::var("FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE_REPORT").as_deref() == Ok("1"),
        )
    })
}

pub(super) fn project_group(
    context: &CudaContext,
    stream: &DeviceStream,
    calls: &[DenseGemvCall<'_>],
    stage: &str,
) -> Result<(), EngError> {
    let (reuse, report) = flags();
    let paired = if reuse {
        context.dense_gemv_group_with_q8_row_reuse(stream, calls)?
    } else {
        context.dense_gemv_group(stream, calls)?;
        0
    };
    if report {
        report_counts(calls, stage, paired);
    }
    Ok(())
}

pub(super) fn project_output(
    context: &CudaContext,
    stream: &DeviceStream,
    call: DenseGemvCall<'_>,
) -> Result<(), EngError> {
    let (reuse, report) = flags();
    let shape = call.shape;
    let q8 = call.dtype == DenseDtype::PackedGguf(GgufType::Q80);
    let width = if reuse {
        static FOUR: OnceLock<bool> = OnceLock::new();
        let four = *FOUR.get_or_init(|| {
            std::env::var("FIFI_INKLING_CUDA_PREFILL_Q8_OUTPUT_EXACT4").as_deref() == Ok("1")
        });
        context.dense_gemv_with_q8_row_reuse(stream, call, four)?
    } else {
        context.dense_gemv(stream, call)?;
        0
    };
    if width == 4 {
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used exact four-row Q8 attention output"));
        if report {
            eprintln!("Inkling CUDA Q8 output exact4 rows {}", shape.m);
        }
    } else if report && q8 {
        emit("attention_output", shape.m, 1, usize::from(width == 2));
    }
    Ok(())
}

fn report_counts(calls: &[DenseGemvCall<'_>], stage: &str, paired: usize) {
    // No WMMA/expert opt-in here: every Q8_0 call took the pair or ordinary GEMV entry.
    let mut q8 = calls
        .iter()
        .filter(|call| call.dtype == DenseDtype::PackedGguf(GgufType::Q80));
    if let Some(first) = q8.next() {
        emit(stage, first.shape.m, 1 + q8.count(), paired);
    }
}

fn emit(stage: &str, rows: usize, total: usize, paired: usize) {
    eprintln!(
        "Inkling CUDA Q8 prefill {}",
        serde_json::json!({
            "stage": stage,
            "rows": rows,
            "q8_pair_calls": paired,
            "q8_gemv_calls": total - paired,
        })
    );
}
