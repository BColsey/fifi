use super::*;
use crate::family::inkling2::cpu_forward::TapRow;

/// Warm a real continuation, replay the prelude and first pass, then stop at the
/// same pair boundary in both arms. Compare the unnormalized continuation, not a
/// sampled id: max(abs delta) <= 5e-3 and relative L2 <= 5e-4.
#[test]
#[ignore = "requires leased CUDA, packed resident Inkling Mini and FIFI_INKLING_MODEL_DIR"]
fn real_consecutive_banked_rows() {
    let run = |candidate| {
        let path = std::env::var_os("FIFI_INKLING_MODEL_DIR").unwrap();
        let mut f = CpuForward::open(std::path::Path::new(&path), 768).unwrap();
        f.enable_cuda(0).unwrap();
        f.token(1, false).unwrap();
        let mut state = std::mem::take(&mut f.state);
        let mut depth = f.embed(2).unwrap();
        let mut taps = TapRow::new(&[], f.hidden());
        let core = &f.spec.core;
        f.run_visits(
            &f.visits[..core.core_start],
            &mut state,
            &mut depth,
            None,
            &mut taps,
            true,
        )
        .unwrap();
        let mut first_pass = depth.clone();
        f.run_visits(
            &f.visits[core.core_start..core.core_end],
            &mut state,
            &mut first_pass,
            None,
            &mut taps,
            true,
        )
        .unwrap();
        let bank = MessageBank {
            first_block: depth.block_count() + 1,
            stream: first_pass,
        };
        let pair = [f.visits[core.core_end], f.visits[core.core_end + 1]];
        if candidate {
            assert!(
                f.try_resident_pair(pair, &mut state, &mut depth, &bank)
                    .unwrap(),
                "pair was declined"
            );
        } else {
            for visit in pair {
                f.layer(
                    visit,
                    &mut state[visit.state_slot],
                    &mut depth,
                    Some(&bank),
                    true,
                )
                .unwrap();
            }
        }
        depth.into_current()
    };
    let oracle = run(false);
    let candidate = run(true);
    assert_eq!(candidate.len(), oracle.len());
    let mut max = 0.0_f64;
    let (mut err, mut norm, mut exact) = (0., 0., 0);
    for (&a, b) in oracle.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite());
        exact += usize::from(a.to_bits() == b.to_bits());
        let delta = f64::from(a) - f64::from(b);
        max = max.max(delta.abs());
        err += delta * delta;
        norm += f64::from(a).powi(2);
    }
    let rel = (err / norm).sqrt();
    eprintln!(
        "pair row exact={exact}/{} max_abs={max} rel_l2={rel} tolerances=0.005,0.0005",
        oracle.len()
    );
    assert!(max <= 0.005 && rel <= 0.0005);
}
