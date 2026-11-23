use super::*;
use crate::weights::{GgufType, dequantize};

#[test]
fn q6_f32_geometry_is_checked_before_launch() {
    assert_eq!(sizes(9, 512), Some((9 * 2 * 210, 512 * 4, 9 * 4)));
    assert_eq!(sizes(0, 256), None);
    assert_eq!(sizes(1, 0), None);
    assert_eq!(sizes(1, 257), None);
    assert_eq!(sizes(usize::MAX, 256), None);
    assert_eq!(sizes(1, usize::MAX), None);
    assert_eq!(sizes(i32::MAX as usize, 256), None);
}

fn fixture(rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut state = 0x6f32_1234_u32;
    let mut bytes = vec![0_u8; rows * (k / 256) * 210];
    for (i, block) in bytes.chunks_exact_mut(210).enumerate() {
        for value in &mut block[..208] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = (state >> 24) as u8;
        }
        block[192] = 128;
        block[193] = 127;
        let d = [0x0001_u16, 0x2c00, 0xac00, 0x0000][i % 4];
        block[208..210].copy_from_slice(&d.to_le_bytes());
    }
    let x = (0..k)
        .map(|i| ((i * 13 % 37) as f32 - 18.0) / 9.0)
        .collect();
    (bytes, x)
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn q6_f32_matches_dequantized_f64_dots_and_guards() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for (rows, k) in [(1, 256), (7, 512), (8, 768), (9, 1024), (17, 512)] {
        let (blocks, x) = fixture(rows, k);
        let w = ctx.upload(&blocks).unwrap();
        let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let input = ctx.upload(&xbytes).unwrap();
        let output = ctx.upload(&vec![0xa5; rows * 4 + 8]).unwrap();
        ctx.q6_f32(
            &stream,
            Q6F32Call {
                w: w.range(0, blocks.len()).unwrap(),
                x: input.range(0, xbytes.len()).unwrap(),
                y: output.range(4, rows * 4).unwrap(),
                rows,
                k,
            },
        )
        .unwrap();
        let mut actual = vec![0_u8; rows * 4 + 8];
        output.read_at(&stream, 0, &mut actual).unwrap();
        assert_eq!(&actual[..4], &[0xa5; 4]);
        assert_eq!(&actual[actual.len() - 4..], &[0xa5; 4]);
        let weights = dequantize(GgufType::Q6K, &blocks).unwrap();
        assert_eq!(weights.len(), rows * k);
        for (row, (weight_row, bytes)) in weights
            .chunks_exact(k)
            .zip(actual[4..actual.len() - 4].chunks_exact(4))
            .enumerate()
        {
            let got = f32::from_ne_bytes(bytes.try_into().unwrap()) as f64;
            let (expected, absolute) =
                weight_row
                    .iter()
                    .zip(&x)
                    .fold((0.0_f64, 0.0_f64), |(sum, abs), (&w, &a)| {
                        let p = f64::from(w) * f64::from(a);
                        (sum + p, abs + p.abs())
                    });
            assert!(
                got.is_finite() && (got - expected).abs() <= 4e-6 * absolute + 2e-7,
                "rows={rows} k={k} row={row} got={got} expected={expected} abs_sum={absolute}"
            );
        }
    }
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn q6_f32_rejects_bad_geometry_ranges_aliases_and_foreign_owners() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let w = ctx.upload(&vec![0; 210 + 4]).unwrap();
    let x = ctx.upload(&vec![0; 256 * 4 + 4]).unwrap();
    let y = ctx.upload(&[0xa5; 16]).unwrap();
    let invoke = |w, x, y, rows, k| ctx.q6_f32(&stream, Q6F32Call { w, x, y, rows, k });
    for (rows, k) in [
        (0, 256),
        (1, 0),
        (1, 257),
        (usize::MAX, 256),
        (1, usize::MAX),
    ] {
        assert!(matches!(
            invoke(
                w.range(0, 210).unwrap(),
                x.range(0, 1024).unwrap(),
                y.range(0, 4).unwrap(),
                rows,
                k
            ),
            Err(EngError::BadRequest(_))
        ));
    }
    for (weight, input, output) in [
        (
            w.range(0, 209).unwrap(),
            x.range(0, 1024).unwrap(),
            y.range(0, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(0, 1020).unwrap(),
            y.range(0, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(0, 1024).unwrap(),
            y.range(0, 3).unwrap(),
        ),
        (
            w.range(1, 210).unwrap(),
            x.range(0, 1024).unwrap(),
            y.range(0, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(1, 1024).unwrap(),
            y.range(0, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(0, 1024).unwrap(),
            y.range(1, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(0, 1024).unwrap(),
            x.range(0, 4).unwrap(),
        ),
        (
            w.range(0, 210).unwrap(),
            x.range(0, 1024).unwrap(),
            w.range(0, 4).unwrap(),
        ),
    ] {
        assert!(matches!(
            invoke(weight, input, output, 1, 256),
            Err(EngError::BadRequest(_))
        ));
    }
    let foreign = CudaContext::new().unwrap();
    let other_stream = foreign.stream().unwrap();
    let other = foreign.upload(&vec![0; 1024]).unwrap();
    assert!(matches!(
        invoke(
            w.range(0, 210).unwrap(),
            other.range(0, 1024).unwrap(),
            y.range(0, 4).unwrap(),
            1,
            256
        ),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        ctx.q6_f32(
            &other_stream,
            Q6F32Call {
                w: w.range(0, 210).unwrap(),
                x: x.range(0, 1024).unwrap(),
                y: y.range(0, 4).unwrap(),
                rows: 1,
                k: 256,
            }
        ),
        Err(EngError::BadRequest(_))
    ));
    let mut unchanged = [0; 16];
    y.read_at(&stream, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0xa5; 16]);
}
