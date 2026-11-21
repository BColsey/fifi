//! The device against the host on the small model: the same tokens through the CUDA driver
//! (trunk upload, native projections, DeltaNet scan in the tiled order, native MoE block, host
//! PLE feed) and through the host forward over the same file. The weights are one set of bytes;
//! the device quantizes its activations to q8_1 and widens some tensors to half precision, so the
//! logits agree to a tolerance and the arg-max agrees exactly.

use super::{GgufModel, host, smallmodel::write_small, trunk::gguf_trunk};
use crate::{
    family::{
        flash_next::cuda::gguf_device::gguf_driver, project::cuda::CudaProjector,
        session::TokenStepper,
    },
    kernels::cuda::DeviceStream,
};

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("nonempty")
}

#[test]
fn the_device_logits_follow_the_host_forward_on_the_small_model() {
    let small = write_small(7);
    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let projector: &'static CudaProjector =
        Box::leak(Box::new(CudaProjector::new().expect("the card")));
    let ctx = projector.context();
    let stream: &'static DeviceStream = Box::leak(Box::new(ctx.stream().expect("stream")));
    let weights = Box::leak(Box::new(
        gguf_trunk(model, ctx, true, 8 << 20).expect("the trunk uploads"),
    ));
    let mut driver = gguf_driver(model, ctx, stream, weights, 64, 2).expect("the driver");

    let tokens = [5u32, 9, 200, 17, 33, 481];
    let reference = host::HostModel::load(model, 4).expect("host layers");
    let want = host::run(model, &reference, &tokens, 2).expect("host forward");
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (at, &token) in tokens.iter().enumerate() {
        let got = driver.logits(token).expect("device logits");
        assert_eq!(got.len(), want.logits[at].len());
        assert!(got.iter().all(|v| v.is_finite()), "position {at}");
        let diff = got
            .iter()
            .zip(&want.logits[at])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst = worst.max(diff);
        scale = scale.max(want.logits[at].iter().fold(0f32, |m, v| m.max(v.abs())));
        eprintln!(
            "position {at}: device top-1 {}, host top-1 {}, max abs difference {diff}",
            argmax(&got),
            argmax(&want.logits[at])
        );
        assert_eq!(argmax(&got), argmax(&want.logits[at]), "position {at}");
    }
    eprintln!("worst max abs difference {worst} on logits up to {scale}");
    assert!(worst < 0.05 * scale.max(1.0), "{worst} against {scale}");
}
