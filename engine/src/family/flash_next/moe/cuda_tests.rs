//! The device router (`CudaContext::router`) at the Flash-Next MoE shape (hidden 2560, 512
//! experts, top 10, a sigmoid shared-expert gate) against this module's host `route`, which the
//! Mac MLX path is checked against.
//!
//! Every dot is exact on both sides: router row `e` holds `v_e` in column 0 and zeros elsewhere,
//! the shared-gate row holds `g` in column 1, and token `t` has `x[0] = s_t` and `x[1] = 1`, all
//! small binary fractions. So the logits are `v_e * s_t` bit for bit on the host and the device,
//! and the comparison measures the softmax, the ranking and the renormalization. The 512 values
//! are distinct and 1/64 apart, so no rank is within rounding of the next. The ids must equal the
//! host's in rank order; each weight and the gate must lie within `2^-17` of the host's. Calls of
//! 3 tokens (the decode logits kernel) and 16 tokens (the tiled prefill kernel) are both checked.

use super::{MoeShape, route};
use crate::EngError;
use crate::family::flash_next::math::sigmoid;
use crate::kernels::cuda::resident::router::{
    RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present};

const HIDDEN: usize = 2560;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GATE: f32 = -0.75;
/// The bound on a weight and on the gate, `2^-17` (the router's class R1).
const BOUND: f64 = 1.0 / 131_072.0;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer, count: usize) -> Result<Vec<[u8; 4]>, EngError> {
    let mut bytes = vec![0u8; count * 4];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes.as_chunks::<4>().0.to_vec())
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The router matrix: row `e` is `v_e` in column 0, distinct values 1/64 apart in [-4, 4).
fn router_rows() -> Vec<f32> {
    let mut w = vec![0.0f32; EXPERTS * HIDDEN];
    for e in 0..EXPERTS {
        w[e * HIDDEN] = ((e * 197) % EXPERTS) as f32 / 64.0 - 4.0;
    }
    w
}

/// Token `t`: `x[0]` cycles through 1, 0.5, 2 and 0.25, `x[1] = 1`, the rest any finite value.
fn tokens(m: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; m * HIDDEN];
    for t in 0..m {
        let row = &mut x[t * HIDDEN..(t + 1) * HIDDEN];
        for (i, v) in row.iter_mut().enumerate() {
            *v = (i % 13) as f32 - 6.0;
        }
        row[0] = [1.0, 0.5, 2.0, 0.25][t % 4];
        row[1] = 1.0;
    }
    x
}

struct DeviceRoutes {
    ids: Vec<i32>,
    weights: Vec<f32>,
    gate: Vec<f32>,
    status: i32,
}

fn device_routes(
    ctx: &CudaContext,
    stream: &DeviceStream,
    (x, w, w_shared): (&[f32], &[f32], &[f32]),
    m: usize,
) -> Result<DeviceRoutes, EngError> {
    let shape = RouterShape {
        m,
        hidden: HIDDEN,
        n_expert: EXPERTS,
        top_k: TOP_K,
    };
    let (x_buf, w_buf, s_buf) = (
        ctx.upload(&f32_bytes(x))?,
        ctx.upload(&f32_bytes(w))?,
        ctx.upload(&f32_bytes(w_shared))?,
    );
    let logits = ctx.upload(&vec![0x55; router_logits_bytes(shape)?])?;
    let ids = ctx.upload(&vec![0x55; m * TOP_K * 4])?;
    let weights = ctx.upload(&vec![0x55; m * TOP_K * 4])?;
    let gate = ctx.upload(&vec![0x55; m * 4])?;
    let status = ctx.upload(&[0u8; 4])?;
    ctx.router(
        stream,
        RouterCall {
            shape,
            x: whole(&x_buf)?,
            weights: RouterWeights {
                w: whole(&w_buf)?,
                w_shared: Some(whole(&s_buf)?),
            },
            logits: whole(&logits)?,
            out: RouterOut {
                ids: whole(&ids)?,
                weights: whole(&weights)?,
                shared_gate: Some(whole(&gate)?),
                status: whole(&status)?,
            },
        },
    )?;
    stream.synchronize()?;
    let floats = |words: Vec<[u8; 4]>| words.into_iter().map(f32::from_le_bytes).collect();
    Ok(DeviceRoutes {
        ids: read(ctx, &ids, m * TOP_K)?
            .into_iter()
            .map(i32::from_le_bytes)
            .collect(),
        weights: floats(read(ctx, &weights, m * TOP_K)?),
        gate: floats(read(ctx, &gate, m)?),
        status: i32::from_le_bytes(read(ctx, &status, 1)?[0]),
    })
}

fn relative(dev: f32, host: f32) -> f64 {
    (f64::from(dev) - f64::from(host)).abs() / f64::from(host).abs().max(f64::MIN_POSITIVE)
}

#[test]
fn cuda_router_matches_the_flash_next_route_at_512_experts() -> Result<(), EngError> {
    if !device_present("cuda_router_matches_the_flash_next_route_at_512_experts") {
        return Ok(());
    }
    let shape = MoeShape::new(EXPERTS as u32, TOP_K as u32, 640, 640, HIDDEN as u32, true)?;
    let w = router_rows();
    let mut w_shared = vec![0.0f32; HIDDEN];
    w_shared[1] = GATE;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    for m in [3, 16] {
        let x = tokens(m);
        let dev = device_routes(&ctx, &stream, (&x, &w, &w_shared), m)?;
        assert_eq!(dev.status, 0, "m {m}: status");
        let mut worst = 0.0f64;
        for t in 0..m {
            let row = &x[t * HIDDEN..(t + 1) * HIDDEN];
            let host = route(&shape, &w, row)?;
            let slots = t * TOP_K..(t + 1) * TOP_K;
            let host_ids: Vec<i32> = host.iter().map(|&(e, _)| e as i32).collect();
            assert_eq!(dev.ids[slots.clone()], host_ids, "m {m} token {t}: ids");
            for (d, &(_, h)) in dev.weights[slots].iter().zip(&host) {
                worst = worst.max(relative(*d, h));
            }
            worst = worst.max(relative(dev.gate[t], sigmoid(GATE * row[1])));
        }
        println!("REPORTED: flash-next router m {m}: worst weight or gate {worst:.3e}");
        assert!(worst <= BOUND, "m {m}: {worst:.3e} above 2^-17");
    }
    Ok(())
}
