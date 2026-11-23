use super::*;
use crate::kernels::cuda::testkit;

#[test]
fn head_tile_preserves_bits_and_guards() {
    if !testkit::device_present("head_tile bits") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for (keys, heads, dim) in [(16, 48, 128), (2, 6, 7), (3, 3, 1), (1, 7, 13)] {
        let count = heads * dim;
        let source: Vec<u32> = (0..count)
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x7fc0_0001)
            .collect();
        let raw: Vec<u8> = source.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let src = ctx.upload(&raw).unwrap();
        let dst = ctx.upload(&vec![0xa5; raw.len() + 8]).unwrap();
        ctx.head_tile(
            &stream,
            HeadTileCall {
                src: src.range(0, raw.len()).unwrap(),
                dst: dst.range(4, raw.len()).unwrap(),
                shape: HeadTileShape {
                    key_heads: keys,
                    value_heads: heads,
                    head_dim: dim,
                },
            },
        )
        .unwrap();
        let mut bytes = vec![0; raw.len() + 8];
        dst.read_at(&stream, 0, &mut bytes).unwrap();
        assert_eq!(&bytes[..4], &[0xa5; 4]);
        assert_eq!(&bytes[bytes.len() - 4..], &[0xa5; 4]);
        let mut expected = vec![0u32; count];
        for h in 0..heads {
            let tiled = (h % (heads / keys)) * keys + h / (heads / keys);
            expected[tiled * dim..(tiled + 1) * dim]
                .copy_from_slice(&source[h * dim..(h + 1) * dim]);
        }
        let want: Vec<u8> = expected.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(&bytes[4..bytes.len() - 4], want);
    }
}

#[test]
#[ignore = "synthetic CUDA microbenchmark; requires an admitted device"]
fn head_tile_microbenchmark_against_diagnostic_head_copies() {
    if !testkit::device_present("head_tile benchmark") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let shape = HeadTileShape {
        key_heads: 16,
        value_heads: 48,
        head_dim: 128,
    };
    let bytes = shape.value_heads * shape.head_dim * 4;
    let src = ctx
        .upload(&(0..bytes).map(|i| (i % 251) as u8).collect::<Vec<_>>())
        .unwrap();
    let single = ctx.allocate(bytes).unwrap();
    let copies = ctx.allocate(bytes).unwrap();
    let one = || {
        ctx.head_tile(
            &stream,
            HeadTileCall {
                src: src.range(0, bytes)?,
                dst: single.range(0, bytes)?,
                shape,
            },
        )
    };
    let many = || {
        let per_key = shape.value_heads / shape.key_heads;
        let hb = shape.head_dim * 4;
        for h in 0..shape.value_heads {
            let t = (h % per_key) * shape.key_heads + h / per_key;
            ctx.copy_device(&stream, src.range(h * hb, hb)?, copies.range(t * hb, hb)?)?;
        }
        Ok::<(), EngError>(())
    };
    for _ in 0..20 {
        one().unwrap();
        many().unwrap();
    }
    stream.synchronize().unwrap();
    let start = ctx.event_timing().unwrap();
    let end = ctx.event_timing().unwrap();
    const N: usize = 200;
    stream.record_timing(&start).unwrap();
    for _ in 0..N {
        one().unwrap();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    let kernel_us = start.elapsed_ms_since(&end).unwrap() * 1000.0 / N as f32;
    stream.record_timing(&start).unwrap();
    for _ in 0..N {
        many().unwrap();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    let copies_us = start.elapsed_ms_since(&end).unwrap() * 1000.0 / N as f32;
    let mut a = vec![0; bytes];
    let mut b = vec![0; bytes];
    single.read_at(&stream, 0, &mut a).unwrap();
    copies.read_at(&stream, 0, &mut b).unwrap();
    assert_eq!(a, b);
    assert!(kernel_us.is_finite() && kernel_us > 0.0 && copies_us.is_finite() && copies_us > 0.0);
    eprintln!(
        "synthetic 16-key/48-value heads x 128 floats, {N} iterations: one_kernel_us={kernel_us:.3}, 48_d2d_copies_us={copies_us:.3}; not model throughput"
    );
}

#[test]
fn head_tile_rejects_bad_geometry_and_ranges_without_writes() {
    if !testkit::device_present("head_tile rejection") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let src = ctx.upload(&[0x31; 64]).unwrap();
    let dst = ctx.upload(&[0x82; 64]).unwrap();
    let good = HeadTileShape {
        key_heads: 2,
        value_heads: 4,
        head_dim: 4,
    };
    for shape in [
        HeadTileShape {
            key_heads: 0,
            ..good
        },
        HeadTileShape {
            value_heads: 0,
            ..good
        },
        HeadTileShape {
            head_dim: 0,
            ..good
        },
        HeadTileShape {
            value_heads: 3,
            ..good
        },
        HeadTileShape {
            head_dim: usize::MAX,
            ..good
        },
        HeadTileShape {
            value_heads: 1 << 24,
            head_dim: 2,
            ..good
        },
    ] {
        assert!(matches!(
            ctx.head_tile(
                &stream,
                HeadTileCall {
                    src: src.range(0, 64).unwrap(),
                    dst: dst.range(0, 64).unwrap(),
                    shape
                }
            ),
            Err(EngError::BadRequest(_))
        ));
    }
    for (input, output) in [
        (src.range(0, 60).unwrap(), dst.range(0, 64).unwrap()),
        (src.range(0, 64).unwrap(), dst.range(0, 60).unwrap()),
        (src.range(1, 63).unwrap(), dst.range(0, 64).unwrap()),
        (src.range(0, 64).unwrap(), src.range(0, 64).unwrap()),
    ] {
        assert!(matches!(
            ctx.head_tile(
                &stream,
                HeadTileCall {
                    src: input,
                    dst: output,
                    shape: good
                }
            ),
            Err(EngError::BadRequest(_))
        ));
    }
    let mut untouched = [0; 64];
    dst.read_at(&stream, 0, &mut untouched).unwrap();
    assert_eq!(untouched, [0x82; 64]);
}
