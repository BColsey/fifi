//! Bounded one-row projection probe: native GGUF blocks in, host-visible logits out.
//! This is a test adapter, not a serving path or a weight placement policy.

use std::ops::Range;

use super::{GgufModel, tensors::Mapped};
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

/// Read an output row range or one expert slice from the checkpoint's mapped tensor.
fn mapped_blocks(
    model: &GgufModel,
    tensor: &Mapped,
    expert: Option<u64>,
    rows: Range<u64>,
) -> Result<Vec<u8>, EngError> {
    let invalid = || EngError::BadRequest(format!("invalid row slice for {}", tensor.name));
    let (k, outputs, expert) = match (tensor.dims.as_slice(), expert) {
        ([k, outputs], None) => (*k, *outputs, 0),
        ([k, outputs, experts], Some(expert)) if expert < *experts => (*k, *outputs, expert),
        _ => return Err(invalid()),
    };
    let block = tensor.ty.block_elems();
    if k == 0 || !k.is_multiple_of(block) || rows.start >= rows.end || rows.end > outputs {
        return Err(invalid());
    }
    let stride = (k / block)
        .checked_mul(tensor.ty.block_bytes())
        .ok_or_else(invalid)?;
    let begin = expert
        .checked_mul(outputs)
        .and_then(|base| base.checked_add(rows.start))
        .and_then(|first| first.checked_mul(stride))
        .ok_or_else(invalid)?;
    let bytes = (rows.end - rows.start)
        .checked_mul(stride)
        .filter(|bytes| {
            begin
                .checked_add(*bytes)
                .is_some_and(|end| end <= tensor.bytes)
        })
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(invalid)?;
    let mut blocks = vec![0; bytes];
    model.weights.read_range(&tensor.name, begin, &mut blocks)?;
    Ok(blocks)
}

#[test]
fn reads_only_the_mapped_rows_from_the_expected_split_shard() {
    use super::testkit::{Spec, write_flash};
    let file = write_flash(&Spec {
        shards: 4,
        ..Spec::default()
    });
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let tensor = model
        .map
        .entries()
        .iter()
        .find(|t| t.ty == GgufType::Q80 && t.dims.len() == 2)
        .expect("mapped Q8_0 matrix");
    assert!(tensor.shard > 0, "the first shard holds only metadata");
    let stride = tensor.dims[0] / 32 * 34;
    let wanted = vec![0xb7; 3 * stride as usize];
    file.patch(&tensor.name, 2 * stride, &wanted);
    assert_eq!(mapped_blocks(&model, tensor, None, 2..5).unwrap(), wanted);
    assert_eq!(
        mapped_blocks(&model, tensor, None, 1..2).unwrap(),
        vec![0; stride as usize]
    );
    assert!(mapped_blocks(&model, tensor, Some(0), 2..5).is_err());
    let end = tensor.dims[1];
    assert!(mapped_blocks(&model, tensor, None, end..end + 1).is_err());

    let stack = model
        .map
        .entries()
        .iter()
        .find(|t| t.dims.len() == 3 && t.dims[2] > 1)
        .expect("mapped expert stack");
    let stride = stack.dims[0] / stack.ty.block_elems() * stack.ty.block_bytes();
    let wanted = vec![0x6d; stride as usize];
    file.patch(&stack.name, (stack.dims[1] + 2) * stride, &wanted);
    assert_eq!(mapped_blocks(&model, stack, Some(1), 2..3).unwrap(), wanted);
    assert_eq!(
        mapped_blocks(&model, stack, Some(0), 2..3).unwrap(),
        vec![0; stride as usize]
    );
    assert!(mapped_blocks(&model, stack, Some(stack.dims[2]), 2..3).is_err());
}

#[test]
#[ignore = "needs the pinned Flash-Next GGUF and an admitted CUDA device"]
fn mapped_checkpoint_projections_match_host_dots() {
    use crate::weights::dequantize;
    let path =
        std::env::var_os("FIFI_FLASH_GGUF").expect("set FIFI_FLASH_GGUF to the verified pack");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .expect("Flash-Next GGUF");
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let mut checked = 0;
    for t in model.map.entries().iter().filter(|t| t.layer == Some(0)) {
        if QuantType::from_gguf(t.ty).is_none() || !matches!(t.dims.len(), 2 | 3) || t.dims[1] < 4 {
            continue;
        }
        let expert = t.dims.get(2).map(|n| n / 2);
        let x: Vec<_> = (0..t.dims[0])
            .map(|n| if n % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        let blocks = mapped_blocks(&model, t, expert, 1..4).expect("mapped rows");
        let weights = dequantize(t.ty, &blocks).expect("host weights");
        let actual = project(&ctx, &stream, t.ty, &blocks, &x).expect("native projection");
        assert_eq!(actual.len(), 3);
        for (actual, row) in actual.into_iter().zip(weights.chunks_exact(x.len())) {
            let expected: f32 = row.iter().zip(&x).map(|(w, x)| w * x).sum();
            let scale: f32 = row.iter().map(|w| w.abs()).sum();
            assert!(
                (actual - expected).abs() <= 0.003 * scale + 1e-5,
                "{}: native {actual}, host {expected}",
                t.name
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no native mapped projections were checked");
    eprintln!("Flash-Next layer 0: checked {checked} mapped projections");
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
