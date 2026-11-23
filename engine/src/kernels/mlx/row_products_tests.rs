//! The products a decode window and a lone row depend on: MLX picks the kernel from the row count,
//! and the engine relies on its choice, so these tests name it. Mac only, ignored, no model.
use crate::{
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

const ROWS: usize = 8;
const K: usize = 2048;
const N: usize = 512;

fn next(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) as u32
}

fn bf16(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

/// A 4-bit affine weight of `[N, K]` with pseudo-random codes, scales and biases.
fn weight() -> PackedWeight {
    let mut seed = 0x5eed_u64;
    let payload: Vec<u8> = (0..N * K / 8)
        .flat_map(|_| next(&mut seed).to_le_bytes())
        .collect();
    let groups = N * K / 64;
    let scales: Vec<u8> = (0..groups)
        .flat_map(|_| bf16(0.01 + (next(&mut seed) % 100) as f32 / 4000.0))
        .collect();
    let biases: Vec<u8> = (0..groups)
        .flat_map(|_| bf16((next(&mut seed) % 200) as f32 / 100.0 - 1.0))
        .collect();
    let host = |dtype, last: usize, bytes| {
        Tensor::try_from_host(dtype, vec![N as u32, last as u32], bytes).unwrap()
    };
    PackedWeight::try_new(
        host(Dtype::U32, K / 8, payload),
        host(Dtype::Bf16, K / 64, scales),
        Some(host(Dtype::Bf16, K / 64, biases)),
        vec![N as u32, K as u32],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap()
}

/// `rows` BF16 activation rows of `[rows, K]`, the first `rows` of one fixed sequence.
fn rows(k: &mut MlxKernels, rows: usize) -> ExecutionTensor {
    let mut seed = 0xac7_u64;
    let bytes: Vec<u8> = (0..ROWS * K)
        .flat_map(|_| bf16((next(&mut seed) % 2001) as f32 / 1000.0 - 1.0))
        .take(rows * K * 2)
        .collect();
    k.import_tensor(
        &Tensor::try_from_host(Dtype::Bf16, vec![rows as u32, K as u32], bytes).unwrap(),
    )
    .unwrap()
}

fn bytes(k: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<u8> {
    k.readback(tensor).unwrap().host_bytes().unwrap().to_vec()
}

/// With the one-row products tiled, a lone row's product equals its row of a window of 2, 4 or 8
/// bit for bit. This is the property the batch-invariant decode stands on; it fails if MLX
/// changes which kernel a given row count takes (for instance a vector kernel for a window of 2),
/// and then the stacked-copy trick of the native wrapper needs a new row count.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn a_tiled_one_row_product_equals_its_row_of_every_window() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let weight = k.import_packed(&weight()).unwrap();
    let row_bytes = N * 2;
    let widest = {
        let x = rows(&mut k, ROWS);
        let y = k.qmm(&x, &weight).unwrap();
        bytes(&mut k, &y)
    };
    for window in [2, 4] {
        let x = rows(&mut k, window);
        let y = k.qmm(&x, &weight).unwrap();
        assert_eq!(
            bytes(&mut k, &y),
            widest[..window * row_bytes],
            "the rows of a window of {window} differ from the same rows in a window of {ROWS}"
        );
    }
    k.set_row_invariant_products(true).unwrap();
    assert!(k.row_invariant_products());
    for row in 0..ROWS {
        // The one-row activation is the row's own bytes: the first row of a fresh import at
        // offset `row` is built by taking the sequence up to it.
        let all = rows(&mut k, ROWS);
        let host = bytes(&mut k, &all);
        let one = k
            .import_tensor(
                &Tensor::try_from_host(
                    Dtype::Bf16,
                    vec![1, K as u32],
                    host[row * K * 2..(row + 1) * K * 2].to_vec(),
                )
                .unwrap(),
            )
            .unwrap();
        let y = k.qmm(&one, &weight).unwrap();
        assert_eq!(
            bytes(&mut k, &y),
            widest[row * row_bytes..(row + 1) * row_bytes],
            "a lone row {row} differs from its row of a window"
        );
    }
    k.set_row_invariant_products(false).unwrap();
}
