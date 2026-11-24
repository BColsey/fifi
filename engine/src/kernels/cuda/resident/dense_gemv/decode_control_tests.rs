use super::*;
use crate::kernels::cuda::native::resident::dense_gemv::try_control_pair;

#[test]
fn decode_control_pair_is_exact_and_refuses_cross_call_aliases() {
    if !testkit::device_present("decode_control_pair_is_exact_and_refuses_cross_call_aliases") {
        return;
    }
    let rig = Rig::new();
    let shape = DenseGemvShape::packed(1, 2304, 64);
    let data = |n: usize, seed: f32| {
        f32_bytes(
            &(0..n)
                .map(|i| ((i % 113) as f32 * seed).sin())
                .collect::<Vec<_>>(),
        )
    };
    let w = Padded::new(&rig.ctx, &data(64 * 2304, 0.01), 0);
    let v = Padded::new(&rig.ctx, &data(64 * 2304, 0.02), 0);
    let x = Padded::new(&rig.ctx, &data(64, 0.3), 0);
    let output = || Padded::new(&rig.ctx, &sentinel(2304 * 4), 0);
    let [a, b, u, t] = [output(), output(), output(), output()];
    let call = |w, y| DenseGemvCall {
        shape,
        dtype: DenseDtype::F32,
        w,
        x: x.all(),
        y,
    };
    for (w, y) in [(&w, &a), (&v, &b)] {
        rig.ctx
            .dense_gemv(&rig.stream, call(w.all(), y.all()))
            .unwrap();
    }
    let pair = |first: DenseGemvCall<'_>, second: DenseGemvCall<'_>| {
        let first = rig.ctx.plan_dense_gemv(&first).unwrap();
        let second = rig.ctx.plan_dense_gemv(&second).unwrap();
        rig.ctx
            .launch_on(&rig.stream, |s| try_control_pair(s, &first, &second))
            .unwrap()
    };
    assert!(!pair(call(w.all(), u.all()), call(v.all(), u.all())));
    assert!(!pair(
        call(w.all(), v.range(0, 2304 * 4)),
        call(v.all(), t.all())
    ));
    assert!(pair(call(w.all(), u.all()), call(v.all(), t.all())));
    rig.stream.synchronize().unwrap();
    // Bitwise comparison (not an epsilon); Padded::read also checks guards.
    assert_eq!(a.read(&rig.ctx, "gain ref"), u.read(&rig.ctx, "gain"));
    assert_eq!(b.read(&rig.ctx, "forget ref"), t.read(&rig.ctx, "forget"));
}
