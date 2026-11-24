use super::*;
use crate::{kernels::cuda::native::resident::dense_gemv::try_attention_group, weights::GgufType};

#[test]
fn decode_attention_group_checks_all_cross_call_writes_before_launch() {
    if !testkit::device_present("decode_attention_group_checks_all_cross_call_writes_before_launch")
    {
        return;
    }
    let rig = Rig::new();
    let rows = [3072, 1024, 1024, 384, 24, 3072, 32, 32];
    let mut blocks = vec![0u8; 3072 * 2304 / 32 * 34];
    for (i, b) in blocks.chunks_exact_mut(34).enumerate() {
        b[..2].copy_from_slice(&0x2800u16.to_le_bytes());
        for (k, v) in b[2..].iter_mut().enumerate() {
            *v = ((i + k) % 27) as u8;
        }
    }
    let w = Padded::new(&rig.ctx, &blocks, 0);
    let bf = Padded::new(&rig.ctx, &vec![0x3d; 32 * 2304 * 2], 0);
    let x = Padded::new(
        &rig.ctx,
        &f32_bytes(
            &(0..2305)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
        ),
        0,
    );
    let outputs = || rows.map(|n| Padded::new(&rig.ctx, &sentinel(n * 4), 0));
    let dst = outputs();
    let reference = outputs();
    let make = |i: usize, input, y| DenseGemvCall {
        shape: DenseGemvShape::packed(1, rows[i], 2304),
        dtype: if rows[i] <= 32 {
            DenseDtype::Bf16
        } else {
            DenseDtype::PackedGguf(GgufType::Q80)
        },
        w: if rows[i] <= 32 { bf.all() } else { w.all() },
        x: input,
        y,
    };
    for i in 0..8 {
        rig.ctx
            .dense_gemv(&rig.stream, make(i, x.all(), reference[i].all()))
            .unwrap();
    }
    let attempt = |mode: usize, n| {
        let calls = (0..n)
            .map(|i| {
                let input = match (mode, i) {
                    (1, 1) => dst[0].all(),
                    (3, 0) => x.range(4, 2304 * 4),
                    _ => x.all(),
                };
                let y = match (mode, i) {
                    (2, 1) => dst[0].all(),
                    (4, 1) => bf.all(),
                    _ => dst[i].all(),
                };
                let mut call = make(i, input, y);
                if mode == 5 && i == 4 {
                    call.w = bf.range(2, 24 * 2304 * 2);
                }
                rig.ctx.plan_dense_gemv(&call).unwrap()
            })
            .collect::<Vec<_>>();
        rig.ctx
            .launch_on(&rig.stream, |s| try_attention_group(s, calls.iter()))
            .unwrap()
    };
    for mode in 1..=5 {
        assert!(!attempt(mode, 8));
    }
    assert!(!attempt(0, 5));
    assert!(attempt(0, 6));
    assert!(attempt(0, 8));
    rig.stream.synchronize().unwrap();
    for i in 0..8 {
        let r = reference[i].read(&rig.ctx, "serial");
        let d = dst[i].read(&rig.ctx, "group");
        if rows[i] <= 32 {
            assert_eq!(r, d, "the BF16 logical lane order must match serial GEMV");
        }
        for (a, b) in r.chunks_exact(4).zip(d.chunks_exact(4)) {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            assert!(b.is_finite() && (a - b).abs() <= 1e-4 + 2e-5 * a.abs());
        }
    }
}
