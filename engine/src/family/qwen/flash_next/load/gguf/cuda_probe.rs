//! Bounded one-row projection probe: native GGUF blocks in, host-visible logits out.
//! This is a test adapter, not a serving path or a weight placement policy.

use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceStream,
        resident::{
            matvec_q::{MatvecCall, QuantType, ROWS_MULTIPLE},
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    },
    weights::GgufType,
};

pub(super) fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: GgufType,
    blocks: &[u8],
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let bad = |message: &str| {
        EngError::BadRequest(format!("Flash-Next CUDA projection probe: {message}"))
    };
    let ty = QuantType::from_gguf(ty).ok_or_else(|| bad("no native quantized matvec"))?;
    let k = x.len();
    let row_bytes = ty
        .weight_bytes(1, k)
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("invalid input width"))?;
    if blocks.is_empty() || !blocks.len().is_multiple_of(row_bytes) {
        return Err(bad("expected whole weight rows"));
    }
    let live = blocks.len() / row_bytes;
    let rows = live.next_multiple_of(ROWS_MULTIPLE);
    let mut padded = blocks.to_vec();
    padded.resize(rows * row_bytes, 0);
    let x_host: Vec<u8> = x.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let w = ctx.allocate(padded.len())?;
    let input = ctx.allocate(x_host.len())?;
    let q8_bytes = q8_1_bytes(k, 1).ok_or_else(|| bad("invalid q8 activation size"))?;
    let q8 = ctx.allocate(q8_bytes)?;
    let y = ctx.allocate(rows * size_of::<f32>())?;
    w.write_at(stream, 0, &padded)?;
    input.write_at(stream, 0, &x_host)?;
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: input.range(0, x_host.len())?,
            y: q8.range(0, q8_bytes)?,
            k,
            ncols: 1,
        },
    )?;
    ctx.matvec_q(
        stream,
        MatvecCall {
            ty,
            w: w.range(0, padded.len())?,
            x_q8: q8.range(0, q8_bytes)?,
            y: y.range(0, rows * size_of::<f32>())?,
            k,
            rows,
            ncols: 1,
        },
    )?;
    let mut out = vec![0_u8; live * size_of::<f32>()];
    y.read_at(stream, 0, &mut out)?;
    Ok(out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_ne_bytes(bytes))
        .collect())
}

#[test]
#[ignore = "needs an admitted native CUDA device; no checkpoint required"]
fn a_three_row_gguf_projection_matches_the_host_dot_products() {
    use crate::weights::dequantize;
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let x = [1.0; 32];
    let mut blocks = Vec::new();
    for code in [1_i8, -2, 3] {
        blocks.extend(0x3800_u16.to_le_bytes()); // Q8_0 scale 0.5.
        blocks.extend([code as u8; 32]);
    }
    let weights = dequantize(GgufType::Q80, &blocks).expect("host weights");
    let expected: Vec<f32> = weights
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect();
    let got = project(&ctx, &stream, GgufType::Q80, &blocks, &x).expect("native projection");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.into_iter().zip(expected) {
        assert!((a - b).abs() <= b.abs() * 0.005, "native {a}, host {b}");
    }
}
