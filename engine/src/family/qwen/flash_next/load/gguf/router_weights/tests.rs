use super::*;
use crate::{
    family::{
        exl3_pack::layer_name,
        qwen::flash_next::{
            load::gguf::{
                ple_weights::read_floats,
                testkit::{Spec, write_flash},
            },
            moe::{MoeShape, route},
        },
    },
    kernels::cuda::{
        DeviceBuffer,
        resident::router::{
            RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
        },
    },
};

#[test]
fn router_float_plan_rejects_invalid_layouts() {
    let file = write_flash(&Spec::default());
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let entry = model.map.get(Role::Router, Some(0)).unwrap();
    let dims = &model.config.dims;
    let expected = [u64::from(dims.hidden), u64::from(dims.experts)];
    assert!(FloatPlan::new(entry, &expected).is_ok());
    assert!(FloatPlan::new(entry, &[expected[1], expected[0]]).is_err());
    let mut invalid = entry.clone();
    invalid.bytes -= 1;
    assert!(FloatPlan::new(&invalid, &expected).is_err());
    invalid.bytes = entry.bytes;
    invalid.ty = GgufType::Q80;
    assert!(FloatPlan::new(&invalid, &expected).is_err());
    invalid.ty = entry.ty;
    invalid.convention = Convention::GammaOnePlusW;
    assert!(FloatPlan::new(&invalid, &expected).is_err());
    invalid.convention = Convention::AsStored;
    invalid.dims = vec![u64::MAX, u64::MAX];
    assert!(FloatPlan::new(&invalid, &[u64::MAX, u64::MAX]).is_err());
    invalid.dims = vec![0];
    assert!(FloatPlan::new(&invalid, &[0]).is_err());
}

fn read(stream: &DeviceStream, buffer: &DeviceBuffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    bytes
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_ne_bytes(*v))
        .collect()
}

fn compare_router(model: &GgufModel, layer: usize, input: &[f32]) {
    let dims = &model.config.dims;
    let (hidden, experts, top_k) = (
        dims.hidden as usize,
        dims.experts as usize,
        dims.top_k as usize,
    );
    assert_eq!(input.len(), hidden);
    assert!(
        dims.renorm,
        "the current device router renormalizes selected weights"
    );
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let uploaded = model.upload_router(layer, &ctx, &stream).unwrap();
    assert_eq!(uploaded.len(), 2);
    let CudaWeight::Raw {
        name: router_name,
        ty: router_ty,
        buffer: router,
    } = &uploaded[0]
    else {
        panic!("raw router expected")
    };
    let CudaWeight::Raw {
        name: shared_name,
        ty: shared_ty,
        buffer: shared,
    } = &uploaded[1]
    else {
        panic!("raw shared gate expected")
    };
    assert_eq!(router_name, &layer_name(layer, "mlp.gate.weight"));
    assert_eq!(
        shared_name,
        &layer_name(layer, "mlp.shared_expert_gate.weight")
    );
    assert_eq!((*router_ty, *shared_ty), (GgufType::F32, GgufType::F32));
    let router_host = read_floats(
        &model.weights,
        model.map.get(Role::Router, Some(layer)).unwrap(),
    )
    .unwrap();
    let shared_host = read_floats(
        &model.weights,
        model.map.get(Role::SharedRouterGate, Some(layer)).unwrap(),
    )
    .unwrap();
    assert_eq!(f32s(&read(&stream, router)), router_host);
    assert_eq!(f32s(&read(&stream, shared)), shared_host);
    assert!(router_host.iter().any(|x| *x != 0.0));
    assert!(shared_host.iter().any(|x| *x != 0.0));
    let host_shape = MoeShape::new(
        dims.experts,
        dims.top_k,
        dims.moe_inter,
        dims.shared_inter,
        dims.hidden,
        dims.renorm,
    )
    .unwrap();
    let expected = route(&host_shape, &router_host, input).unwrap();
    let dot: f32 = shared_host.iter().zip(input).map(|(w, x)| w * x).sum();
    let gate_expected = 1.0 / (1.0 + (-dot).exp());
    let shape = RouterShape {
        m: 1,
        hidden,
        n_expert: experts,
        top_k,
    };
    let x = ctx
        .upload(
            &input
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let logits = ctx.allocate(router_logits_bytes(shape).unwrap()).unwrap();
    let ids = ctx.allocate(top_k * 4).unwrap();
    let scores = ctx.allocate(top_k * 4).unwrap();
    let gate = ctx.allocate(4).unwrap();
    let status = ctx.upload(&[0; 4]).unwrap();
    ctx.router(
        &stream,
        RouterCall {
            shape,
            x: x.range(0, x.bytes()).unwrap(),
            weights: RouterWeights {
                w: router.range(0, router.bytes()).unwrap(),
                w_shared: Some(shared.range(0, shared.bytes()).unwrap()),
            },
            logits: logits.range(0, logits.bytes()).unwrap(),
            out: RouterOut {
                ids: ids.range(0, ids.bytes()).unwrap(),
                weights: scores.range(0, scores.bytes()).unwrap(),
                shared_gate: Some(gate.range(0, 4).unwrap()),
                status: status.range(0, 4).unwrap(),
            },
        },
    )
    .unwrap();
    let actual_ids: Vec<i32> = read(&stream, &ids)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_ne_bytes(*v))
        .collect();
    let actual_scores = f32s(&read(&stream, &scores));
    let actual_gate = f32s(&read(&stream, &gate))[0];
    assert_eq!(read(&stream, &status), vec![0; 4]);
    assert_eq!(
        actual_ids,
        expected
            .iter()
            .map(|(id, _)| *id as i32)
            .collect::<Vec<_>>()
    );
    for (got, (_, want)) in actual_scores.iter().zip(&expected) {
        assert!((got - want).abs() < 2e-5, "router weight {got} vs {want}");
    }
    assert!(
        (actual_gate - gate_expected).abs() < 2e-5,
        "shared gate {actual_gate} vs {gate_expected}"
    );
    eprintln!(
        "layer {layer} router: top IDs {actual_ids:?}, shared gate GPU={actual_gate:.8} host={gate_expected:.8}"
    );
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn synthetic_router_upload_and_selection_match_host() {
    let mut spec = Spec::default();
    for (layer, ty) in [GgufType::F32, GgufType::F16, GgufType::Bf16]
        .into_iter()
        .enumerate()
    {
        for suffix in ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"] {
            spec.retype.push((format!("blk.{layer}.{suffix}"), ty));
        }
    }
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let hidden = model.config.dims.hidden as usize;
    let input: Vec<f32> = (0..hidden).map(|i| 0.1 + (i % 17) as f32 * 0.002).collect();
    for layer in 0..3 {
        for role in [Role::Router, Role::SharedRouterGate] {
            let entry = model.map.get(role, Some(layer)).unwrap();
            let count = entry.dims.iter().product::<u64>() as usize;
            let mut bytes = Vec::with_capacity(entry.bytes as usize);
            for i in 0..count {
                let value = if role == Role::Router {
                    (i / hidden) as f32 * 0.00001
                        + ((i % hidden) % 13) as f32 * 0.00001 * ((i / hidden % 3) as f32 - 1.0)
                } else {
                    ((i % 19) as f32 - 9.0) * 0.0001
                };
                match entry.ty {
                    GgufType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                    GgufType::F16 => {
                        bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                    }
                    GgufType::Bf16 => bytes
                        .extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes()),
                    _ => unreachable!(),
                }
            }
            assert_eq!(bytes.len(), entry.bytes as usize);
            file.patch(&entry.name, 0, &bytes);
        }
        compare_router(&model, layer, &input);
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    assert!(
        model
            .upload_router(model.config.dims.kinds.len(), &ctx, &stream)
            .is_err()
    );
}

#[test]
#[ignore = "requires the verified pinned GGUF, layer-0 host capture and an admitted CUDA device"]
fn pinned_router_upload_and_selection_match_host() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let capture = std::fs::read_to_string(
        std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set the host capture"),
    )
    .unwrap();
    let input: Vec<f32> = capture
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|row| row["name"] == "blk.0.ffn_gate_shexp.weight")
        .map(|row| serde_json::from_value(row["input"].clone()).unwrap())
        .expect("layer-0 MoE input in host capture");
    compare_router(&model, 0, &input);
}
