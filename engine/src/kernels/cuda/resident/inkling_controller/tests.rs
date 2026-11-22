use crate::kernels::cuda::{CudaContext, device_present};

fn pack(roles: usize, width: usize, broken: Option<usize>) -> Vec<f32> {
    let mut words = vec![0.0; roles * width * 6];
    for role in 0..roles {
        for i in 0..width {
            let value = |seed| (((role * 13 + i * 7 + seed) % 97) as f32 - 48.0) / 32.0;
            for (j, seed) in [1, 3, 7].into_iter().enumerate() {
                words[(role * 3 + j) * width + i] = value(seed);
            }
            for (j, seed) in [11, 17].into_iter().enumerate() {
                words[(3 * roles + role * 2 + j) * width + i] = value(seed);
            }
        }
    }
    if let Some(role) = broken {
        words[(3 * roles + role * 2) * width] = f32::INFINITY;
    }
    words
}

fn run(ctx: &CudaContext, width: usize, roles: usize, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let stream = ctx.stream().unwrap();
    let words = input.len();
    let workspace = ctx.allocate(words * 4).unwrap();
    let health = ctx.allocate(roles * 4).unwrap();
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
    workspace.write_at(&stream, 0, &bytes).unwrap();
    ctx.inkling_controller_norm_roles(
        &stream,
        workspace.range(0, words * 4).unwrap(),
        health.range(0, roles * 4).unwrap(),
        ([width, roles], 1e-5),
    )
    .unwrap();
    let mut output = vec![0u8; roles * width * 4];
    workspace
        .read_at(&stream, 5 * roles * width * 4, &mut output)
        .unwrap();
    let mut status = vec![0u8; roles * 4];
    health.read_at(&stream, 0, &mut status).unwrap();
    let floats = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    };
    (floats(&output), floats(&status))
}

#[test]
fn batched_controller_norm_matches_independent_roles_and_reports_health() {
    if !device_present("batched controller norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let width = 37;
    for roles in [1, 3, 6, 195, 1536] {
        let packed = pack(roles, width, None);
        let (out, health) = run(&ctx, width, roles, &packed);
        assert!(health.iter().all(|h| h.to_bits() == 0));
        for role in [0, roles / 2, roles - 1] {
            let mut single = vec![0.0; width * 6];
            single[..width * 3].copy_from_slice(&packed[role * 3 * width..(role + 1) * 3 * width]);
            single[width * 3..width * 5].copy_from_slice(
                &packed[(3 * roles + 2 * role) * width..(3 * roles + 2 * role + 2) * width],
            );
            let (expected, state) = run(&ctx, width, 1, &single);
            assert_eq!(state[0].to_bits(), 0);
            for (a, b) in out[role * width..(role + 1) * width].iter().zip(expected) {
                assert_eq!(a.to_bits(), b.to_bits(), "roles={roles} role={role}");
            }
        }
    }
    let (_, health) = run(&ctx, width, 6, &pack(6, width, Some(2)));
    assert!(health[2].is_nan());
    assert!(
        health
            .iter()
            .enumerate()
            .all(|(i, h)| i == 2 || h.to_bits() == 0)
    );
}
