use super::{
    GgufModel,
    config::{ARCHITECTURE, GgufConfig},
    plan::Card,
    tensors::{KernelGap, Place, Slot, TensorMap},
    testkit::{REAL_EXPERT_TYPES, Spec, flash_kvs, write_flash, write_gguf},
};
use crate::{
    EngError,
    family::qwen::flash_next::host_plan::HostMemory,
    family::qwen::flash_next::load::{Dims, tests::CONFIG},
    family::recipe::{Global, Role as Wanted, Site, SiteTensor, WeightMap},
    weights::GgufType,
};

const GIB: u64 = 1 << 30;

/// Two independent, same-shaped sparse checkpoints with different router payloads.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn provenance_fixture_pair()
-> (tempfile::TempDir, GgufModel, tempfile::TempDir, GgufModel) {
    let mut spec = Spec::default();
    spec.expert_types[0] = (GgufType::Q80, GgufType::Q80);
    let (first, a) = open(&spec);
    let (second, b) = open(&spec);
    let router = b.map().get(Wanted::Router, Some(0)).unwrap();
    second.patch(&router.name, 0, &[1]);
    let mut left = [0u8; 1];
    let mut right = [0u8; 1];
    a.weights.read_range(&router.name, 0, &mut left).unwrap();
    b.weights.read_range(&router.name, 0, &mut right).unwrap();
    assert_ne!(left, right);
    assert_eq!(a.config().dims.hidden, b.config().dims.hidden);
    assert_eq!(a.config().dims.experts, b.config().dims.experts);
    (first.dir, a, second.dir, b)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn synthetic_qsa_fixture() -> (super::testkit::Written, GgufModel) {
    let retype = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
    ]
    .into_iter()
    .map(|suffix| (format!("blk.3.{suffix}"), GgufType::Q80))
    .collect();
    let (file, model) = open(&Spec {
        retype,
        ..Spec::default()
    });
    for (role, value) in [
        (Wanted::QProj, 1u8),
        (Wanted::KProj, 2),
        (Wanted::VProj, 3),
        (Wanted::OProj, 4),
    ] {
        let entry = model.map().get(role, Some(3)).unwrap();
        let mut block = 0x3c00u16.to_le_bytes().to_vec();
        block.extend_from_slice(&[value; 32]);
        file.patch(
            &entry.name,
            0,
            &block.repeat(entry.bytes as usize / block.len()),
        );
    }
    for (role, value) in [
        (Wanted::IndexProj, 0x3f80u16),
        (Wanted::IndexKey, 0x4000u16),
    ] {
        let entry = model.map().get(role, Some(3)).unwrap();
        file.patch(
            &entry.name,
            0,
            &value.to_le_bytes().repeat(entry.bytes as usize / 2),
        );
    }
    (file, model)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn synthetic_qsa_trunk(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
) -> Vec<crate::family::project::cuda::CudaWeight> {
    let (_file, model) = synthetic_qsa_fixture();
    model.upload_qsa_trunk(3, ctx, stream).unwrap()
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn synthetic_qsa_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    inputs: &[Vec<f32>],
) -> (Vec<crate::family::project::cuda::CudaWeight>, Vec<Vec<f32>>) {
    use super::projector::{HostProjector, HostWeight, Remap};
    use crate::family::qwen::flash_next::qsa::{self, QsaCache, QsaWeights};
    let (file, model) = synthetic_qsa_fixture();
    let get = |role| model.map().get(role, Some(3)).unwrap();
    for role in [Wanted::QProj, Wanted::KProj, Wanted::VProj, Wanted::OProj] {
        let entry = get(role);
        let mut payload = Vec::with_capacity(entry.bytes as usize);
        for row in 0..entry.dims[1] {
            let mut block = 0x1000u16.to_le_bytes().to_vec();
            block.extend_from_slice(&[(1 + row % 5) as u8; 32]);
            payload.extend_from_slice(&block.repeat(entry.dims[0] as usize / 32));
        }
        file.patch(&entry.name, 0, &payload);
    }
    for role in [Wanted::IndexProj, Wanted::IndexKey] {
        let entry = get(role);
        let mut payload = Vec::with_capacity(entry.bytes as usize);
        for row in 0..entry.dims[1] {
            let value = if row % 2 == 0 { 0x3b80u16 } else { 0x3c00u16 };
            payload.extend_from_slice(&value.to_le_bytes().repeat(entry.dims[0] as usize));
        }
        file.patch(&entry.name, 0, &payload);
    }
    for role in [
        Wanted::QNorm,
        Wanted::KNorm,
        Wanted::IndexQNorm,
        Wanted::IndexKNorm,
    ] {
        let entry = get(role);
        let values: Vec<u8> = (0..entry.dims[0])
            .flat_map(|i| (0.75f32 + (i % 3) as f32 * 0.25).to_le_bytes())
            .collect();
        file.patch(&entry.name, 0, &values);
    }
    let device = model.upload_qsa_trunk(3, ctx, stream).unwrap();
    let plain = |role| HostWeight::new(&[get(role)], Remap::None);
    let (q, k, v, o) = (
        plain(Wanted::QProj),
        plain(Wanted::KProj),
        plain(Wanted::VProj),
        plain(Wanted::OProj),
    );
    let idx = HostWeight::new(
        &[get(Wanted::IndexProj), get(Wanted::IndexKey)],
        Remap::None,
    );
    let norms: Vec<Vec<f32>> = [
        Wanted::QNorm,
        Wanted::KNorm,
        Wanted::IndexQNorm,
        Wanted::IndexKNorm,
    ]
    .into_iter()
    .map(|role| super::ple_weights::read_floats(&model.weights, get(role)).unwrap())
    .collect();
    let weights = QsaWeights {
        q: &q,
        k: &k,
        v: &v,
        o: &o,
        idx: &idx,
        q_norm: &norms[0],
        k_norm: &norms[1],
        idx_q_norm: &norms[2],
        idx_k_norm: &norms[3],
    };
    let ngram = model.ngram().unwrap();
    let mut projector = HostProjector::new(&model.weights, &ngram, 1);
    let shape = model.config().model_shape().unwrap().qsa;
    let mut cache = QsaCache::new();
    let outputs = inputs
        .iter()
        .enumerate()
        .map(|(i, x)| qsa::step(&mut projector, &shape, &weights, &mut cache, i as u32, x).unwrap())
        .collect();
    (device, outputs)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn synthetic_delta_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    inputs: &[Vec<f32>],
    out_ty: GgufType,
) -> (
    Vec<crate::family::project::cuda::CudaWeight>,
    Vec<Vec<f32>>,
    Vec<f32>,
    Vec<f32>,
) {
    assert!(matches!(out_ty, GgufType::Q80 | GgufType::Q6K));
    let (file, model) = open(&Spec {
        retype: vec![
            ("blk.0.attn_qkv.weight".into(), GgufType::Q80),
            ("blk.0.attn_gate.weight".into(), GgufType::Q80),
            ("blk.0.ssm_out.weight".into(), out_ty),
            ("blk.0.ssm_beta.weight".into(), GgufType::F32),
            ("blk.0.ssm_alpha.weight".into(), GgufType::F32),
            ("blk.0.ssm_conv1d.weight".into(), GgufType::F32),
            ("blk.0.ssm_a".into(), GgufType::F32),
            ("blk.0.ssm_dt.bias".into(), GgufType::F32),
            ("blk.0.ssm_norm.weight".into(), GgufType::F32),
        ],
        ..Spec::default()
    });
    let get = |role| model.map().get(role, Some(0)).unwrap();
    for role in [Wanted::DeltaQkv, Wanted::DeltaGate, Wanted::DeltaOut] {
        let entry = get(role);
        let mut bytes = Vec::with_capacity(entry.bytes as usize);
        for row in 0..entry.dims[1] as usize {
            if entry.ty == GgufType::Q80 {
                for block in 0..entry.dims[0] as usize / 32 {
                    bytes.extend_from_slice(&0x1400u16.to_le_bytes());
                    for col in 0..32 {
                        bytes.push((1 + (row * 7 + block * 3 + col) % 11) as u8);
                    }
                }
            } else {
                assert_eq!(entry.ty, GgufType::Q6K);
                for block in 0..entry.dims[0] as usize / 256 {
                    let mut q = [0u8; 210];
                    for (i, byte) in q[..128].iter_mut().enumerate() {
                        let lo = 1 + (row * 7 + block * 3 + i) % 11;
                        let hi = 1 + (row * 3 + block + i * 2) % 11;
                        *byte = (lo | hi << 4) as u8;
                    }
                    q[128..192].fill(0xaa);
                    for (i, scale) in q[192..208].iter_mut().enumerate() {
                        *scale = (((row * 17 + block * 31 + i * 11) % 13) as i8 - 6) as u8;
                    }
                    q[208..].copy_from_slice(&0x1000u16.to_le_bytes());
                    bytes.extend_from_slice(&q);
                }
            }
        }
        assert_eq!(bytes.len(), entry.bytes as usize);
        file.patch(&entry.name, 0, &bytes);
    }
    for role in [
        Wanted::DeltaBeta,
        Wanted::DeltaAlpha,
        Wanted::DeltaConv,
        Wanted::DeltaDecay,
        Wanted::DeltaDtBias,
        Wanted::DeltaNorm,
    ] {
        let entry = get(role);
        let width = entry.dims[0] as usize;
        let elements = entry.bytes as usize / 4;
        let bytes: Vec<u8> = (0..elements)
            .flat_map(|i| {
                let value = match role {
                    Wanted::DeltaBeta | Wanted::DeltaAlpha => {
                        (1 + (i / width + i % width % 5) % 7) as f32 * 0.000_01
                    }
                    Wanted::DeltaConv => 0.15 + ((i / width + i % width) % 5) as f32 * 0.025,
                    Wanted::DeltaDecay => -0.25 - (i % 7) as f32 * 0.025,
                    Wanted::DeltaDtBias => (i % 5) as f32 * 0.02 - 0.04,
                    Wanted::DeltaNorm => 0.7 + (i % 7) as f32 * 0.05,
                    _ => unreachable!(),
                };
                value.to_le_bytes()
            })
            .collect();
        file.patch(&entry.name, 0, &bytes);
    }
    let (reference, rec, ring) = super::host::delta_reference(&model, inputs).unwrap();
    let weights = model.upload_delta_trunk(0, ctx, stream).unwrap();
    (weights, reference, rec, ring)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_delta_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    inputs: &[Vec<f32>],
) -> (
    Vec<crate::family::project::cuda::CudaWeight>,
    Vec<Vec<f32>>,
    Vec<f32>,
    Vec<f32>,
) {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    assert_eq!(
        model.map().get(Wanted::DeltaOut, Some(0)).unwrap().ty,
        GgufType::Q6K
    );
    let (reference, rec, ring) = super::host::delta_reference(&model, inputs).unwrap();
    let weights = model.upload_delta_trunk(0, ctx, stream).unwrap();
    (weights, reference, rec, ring)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_hyper_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    site: Site,
) -> (
    Vec<crate::family::project::cuda::CudaWeight>,
    Vec<f32>,
    crate::family::qwen::flash_next::hyper::HyperRead,
) {
    use super::projector::{HostProjector, HostWeight, Remap};
    use crate::family::{
        project::Projector,
        qwen::flash_next::hyper::{self, HyperWeights},
    };
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let layer = if site == Site::Head { None } else { Some(0) };
    let entry = |part| model.map().get(Slot::hyper(site, part), layer).unwrap();
    let norm_entry = entry(SiteTensor::Norm);
    let down_entry = entry(SiteTensor::Down);
    let up_entry = entry(SiteTensor::Up);
    let inject_entry = (site != Site::Head).then(|| entry(SiteTensor::Inject));
    let norm = super::ple_weights::read_floats(&model.weights, norm_entry).unwrap();
    let down = HostWeight::new(&[down_entry], Remap::None);
    let up = HostWeight::new(&[up_entry], Remap::None);
    let inject = inject_entry.map(|v| HostWeight::new(&[v], Remap::None));
    let embedding = model.map().get(Global::Embedding, None).unwrap();
    let embedding = HostWeight::new(&[embedding], Remap::None);
    let ngram = model.ngram().unwrap();
    let mut projector = HostProjector::new(&model.weights, &ngram, 8);
    let row = projector.rows(&embedding, &[9707]).unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_ne!(
        model.config().dims.ple_layer,
        0,
        "the first residual must precede PLE"
    );
    let input = row.repeat(shape.hyper.branches());
    let host = hyper::read(
        &mut projector,
        &shape.hyper,
        &HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: inject.as_ref(),
        },
        &input,
    )
    .unwrap();
    if site == Site::Mixer {
        let capture = std::fs::read_to_string(
            std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set host capture"),
        )
        .unwrap();
        let captured: Vec<f32> = capture
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .find(|row| row["name"] == "blk.0.attn_qkv.weight")
            .map(|row| serde_json::from_value(row["input"].clone()).unwrap())
            .unwrap();
        assert_eq!(host.mixed.len(), captured.len());
        let (err, norm) =
            host.mixed
                .iter()
                .zip(&captured)
                .fold((0.0f64, 0.0f64), |(e, n), (&got, &want)| {
                    (
                        e + f64::from(got - want).powi(2),
                        n + f64::from(want).powi(2),
                    )
                });
        assert!(
            (err / norm).sqrt() < 1e-6,
            "reconstructed token-9707 hyper input must match host capture"
        );
    }
    let source_bytes: u64 = [
        Some(norm_entry),
        Some(down_entry),
        Some(up_entry),
        inject_entry,
    ]
    .into_iter()
    .flatten()
    .map(|v| v.bytes)
    .sum();
    let weights = model.upload_hyper(site, layer, ctx, stream).unwrap();
    let device_bytes: usize = weights
        .iter()
        .map(|w| match w {
            crate::family::project::cuda::CudaWeight::Raw { buffer, .. } => buffer.bytes(),
            _ => panic!("a raw hyper weight"),
        })
        .sum();
    eprintln!("pinned hyper {site:?}: encoded bytes={source_bytes}, staged bytes={device_bytes}");
    (weights, input, host)
}

/// Host final mixer and head on an already captured full-model residual stream.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_final_host_reference(
    model: &GgufModel,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    use super::projector::{HostProjector, HostWeight, Remap};
    use crate::family::{
        project::Projector,
        qwen::flash_next::hyper::{self, HyperWeights},
    };
    let entry = |part| {
        model
            .map()
            .get(Slot::hyper(Site::Head, part), None)
            .unwrap()
    };
    let norm = super::ple_weights::read_floats(&model.weights, entry(SiteTensor::Norm)).unwrap();
    let down = HostWeight::new(&[entry(SiteTensor::Down)], Remap::None);
    let up = HostWeight::new(&[entry(SiteTensor::Up)], Remap::None);
    let head = HostWeight::new(&[model.map().get(Global::Head, None).unwrap()], Remap::None);
    let ngram = model.ngram().unwrap();
    let mut projector = HostProjector::new(&model.weights, &ngram, 8);
    let shape = model.config().model_shape().unwrap();
    let mixed = hyper::read(
        &mut projector,
        &shape.hyper,
        &HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: None,
        },
        input,
    )
    .unwrap()
    .mixed;
    let logits = projector.linear(&head, &mixed).unwrap();
    (mixed, logits)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_shared_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
) -> (
    Vec<crate::family::project::cuda::CudaWeight>,
    Vec<f32>,
    Vec<Vec<f32>>,
) {
    let capture = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set host capture");
    pinned_shared_host_reference_at(ctx, 0, std::path::Path::new(&capture))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_shared_host_reference_at(
    ctx: &crate::kernels::cuda::CudaContext,
    layer: usize,
    capture: &std::path::Path,
) -> (
    Vec<crate::family::project::cuda::CudaWeight>,
    Vec<f32>,
    Vec<Vec<f32>>,
) {
    let (input, values) = pinned_shared_host_data(layer, capture);
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    (
        model.upload_shared_expert(layer, ctx).unwrap(),
        input,
        values,
    )
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_shared_host_data(
    layer: usize,
    capture: &std::path::Path,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let capture = std::fs::read_to_string(capture).unwrap();
    let rows: Vec<serde_json::Value> = capture
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let at = |name: &str| rows.iter().find(|row| row["name"] == name).unwrap();
    let floats = |row: &serde_json::Value, field: &str| -> Vec<f32> {
        serde_json::from_value(row[field].clone()).unwrap()
    };
    let gate = at(&format!("blk.{layer}.ffn_gate_shexp.weight"));
    let up = at(&format!("blk.{layer}.ffn_up_shexp.weight"));
    let down = at(&format!("blk.{layer}.ffn_down_shexp.weight"));
    let input = floats(gate, "input");
    assert_eq!(input, floats(up, "input"));
    let (gate, up, act, down) = (
        floats(gate, "output"),
        floats(up, "output"),
        floats(down, "input"),
        floats(down, "output"),
    );
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), act.len());
    for ((&g, &u), &a) in gate.iter().zip(&up).zip(&act) {
        assert!((crate::family::math::silu(g) * u - a).abs() < 1e-6);
    }
    (input, vec![gate, up, act, down])
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) struct PinnedRouterReference {
    pub weights: Vec<crate::family::project::cuda::CudaWeight>,
    pub logits: Vec<f32>,
    pub route: Vec<(u32, f32)>,
    pub shared_gate: f32,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_router_host_reference(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    input: &[f32],
) -> PinnedRouterReference {
    pinned_router_host_reference_at(ctx, stream, input, 0)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_router_host_reference_at(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
    input: &[f32],
    layer: usize,
) -> PinnedRouterReference {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set the verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let mut reference = pinned_router_host_data(&model, input, layer);
    reference.weights = model.upload_router(layer, ctx, stream).unwrap();
    reference
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn pinned_router_host_data(
    model: &GgufModel,
    input: &[f32],
    layer: usize,
) -> PinnedRouterReference {
    use crate::family::qwen::flash_next::moe::{MoeShape, route};
    let d = &model.config.dims;
    let router = super::ple_weights::read_floats(
        &model.weights,
        model.map.get(Wanted::Router, Some(layer)).unwrap(),
    )
    .unwrap();
    let shared = super::ple_weights::read_floats(
        &model.weights,
        model
            .map
            .get(Wanted::SharedRouterGate, Some(layer))
            .unwrap(),
    )
    .unwrap();
    let shape = MoeShape::new(
        d.experts,
        d.top_k,
        d.moe_inter,
        d.shared_inter,
        d.hidden,
        d.renorm,
    )
    .unwrap();
    let logits = router
        .chunks_exact(input.len())
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect();
    let dot: f32 = shared.iter().zip(input).map(|(w, x)| w * x).sum();
    PinnedRouterReference {
        weights: Vec::new(),
        logits,
        route: route(&shape, &router, input).unwrap(),
        shared_gate: 1.0 / (1.0 + (-dot).exp()),
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn synthetic_native_head(
    ctx: &crate::kernels::cuda::CudaContext,
) -> crate::family::project::cuda::CudaWeight {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head.gguf");
    let mut payload = Vec::new();
    for _ in 0..32 {
        payload.extend_from_slice(&0x3c00u16.to_le_bytes());
        payload.extend_from_slice(&[1u8; 32]);
    }
    write_gguf(
        &path,
        &[],
        &[super::testkit::TensorSpec {
            name: "output.weight".into(),
            dims: vec![256, 4],
            ty: GgufType::Q80,
            payload: Some(payload.clone()),
        }],
    );
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::LIMITS, None).unwrap();
    let (_file, model) = open(&Spec::default());
    let mut mapped = model.map().get(Global::Head, None).unwrap().clone();
    mapped.dims = vec![256, 4];
    mapped.bytes = payload.len() as u64;
    mapped.ty = GgufType::Q80;
    super::upload_native_head(&weights, &mapped, "lm_head".into(), ctx).unwrap()
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) fn synthetic_indexer(
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
) -> crate::family::project::cuda::CudaWeight {
    let (_file, model) = open(&Spec::default());
    let q = model.map().get(Wanted::IndexProj, Some(3)).unwrap().clone();
    let mut k = model.map().get(Wanted::IndexKey, Some(3)).unwrap().clone();
    assert_eq!((q.ty, k.ty), (GgufType::Bf16, GgufType::Bf16));
    let tensor = |entry: &super::tensors::Mapped, value: u16| super::testkit::TensorSpec {
        name: entry.name.clone(),
        dims: entry.dims.clone(),
        ty: entry.ty,
        payload: Some(value.to_le_bytes().repeat(entry.bytes as usize / 2)),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("indexer.gguf");
    write_gguf(&path, &[], &[tensor(&q, 0x3f80), tensor(&k, 0x4000)]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::LIMITS, None).unwrap();
    let result = super::upload_indexer_pair(&weights, [&q, &k], 3, ctx, stream).unwrap();
    k.ty = GgufType::F16;
    assert!(super::upload_indexer_pair(&weights, [&q, &k], 3, ctx, stream).is_err());
    k.ty = GgufType::Bf16;
    k.dims[0] -= 1;
    assert!(super::upload_indexer_pair(&weights, [&q, &k], 3, ctx, stream).is_err());
    result
}

fn open(spec: &Spec) -> (super::testkit::Written, GgufModel) {
    let written = write_flash(spec);
    let model = GgufModel::open(&written.first)
        .expect("open")
        .expect("a qwen4exp model");
    (written, model)
}

#[test]
fn delta_trunk_checks_layer_kinds_without_cuda() {
    let (_file, model) = open(&Spec::default());
    for (layer, &qsa) in model.config().kinds().iter().enumerate() {
        assert_eq!(
            model.validate_delta_layer(layer).is_ok(),
            !qsa,
            "layer {layer}"
        );
    }
    for layer in [model.config().kinds().len(), usize::MAX] {
        let error = model.validate_delta_layer(layer).unwrap_err().describe();
        assert!(error.contains("outside the model"), "{error}");
    }
    let error = model.validate_delta_layer(3).unwrap_err().describe();
    assert!(error.contains("QSA, not DeltaNet"), "{error}");
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn delta_trunk_upload_matches_role_loaders_and_refuses_before_allocation() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;
    let (file, model) = open(&Spec::default());
    let roles = [
        Wanted::DeltaQkv,
        Wanted::DeltaGate,
        Wanted::DeltaBeta,
        Wanted::DeltaAlpha,
        Wanted::DeltaOut,
        Wanted::DeltaConv,
        Wanted::DeltaDecay,
        Wanted::DeltaDtBias,
        Wanted::DeltaNorm,
    ];
    // Nonzero finite patterns distinguish the staged sources while keeping the rest sparse.
    for (i, role) in roles.into_iter().enumerate() {
        let entry = model.map().get(role, Some(0)).unwrap();
        let bytes = match entry.ty {
            GgufType::F32 => ((i + 1) as f32 / 16.0).to_le_bytes().repeat(16),
            GgufType::F16 => 0x3400_u16.to_le_bytes().repeat(32),
            GgufType::Bf16 => 0x3f00_u16.to_le_bytes().repeat(32),
            _ => vec![(i as u8) + 1; 64],
        };
        assert!(entry.bytes >= bytes.len() as u64);
        file.patch(&entry.name, 0, &bytes);
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let foreign = CudaContext::new().unwrap();
    let foreign_stream = foreign.stream().unwrap();
    let before = ctx.live_bytes();
    assert!(model.upload_delta_trunk(3, &ctx, &stream).is_err());
    assert!(model.upload_delta_trunk(usize::MAX, &ctx, &stream).is_err());
    assert!(model.upload_delta_trunk(0, &ctx, &foreign_stream).is_err());
    assert_eq!(
        ctx.live_bytes(),
        before,
        "refusal allocated a device buffer"
    );

    let actual = model.upload_delta_trunk(0, &ctx, &stream).unwrap();
    let mut expected = model.upload_delta_qkv_gate(0, &ctx).unwrap();
    expected.extend(model.upload_delta_beta_alpha(0, &ctx, &stream).unwrap());
    expected.push(model.upload_delta_out(0, &ctx).unwrap());
    expected.extend(model.upload_delta_state(0, &ctx, &stream).unwrap());
    let suffixes = [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.out_proj",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
    ];
    assert_eq!(actual.len(), suffixes.len());
    assert_eq!(expected.len(), suffixes.len());
    for ((actual, expected), suffix) in actual.iter().zip(&expected).zip(suffixes) {
        assert_eq!(actual.name(), layer_name(0, suffix));
        assert_eq!(expected.name(), actual.name());
        let (left, right) = match (actual, expected) {
            (CudaWeight::Native { weight: a, .. }, CudaWeight::Native { weight: b, .. }) => {
                assert_eq!(
                    (a.quant(), a.k(), a.n_rows(), a.experts()),
                    (b.quant(), b.k(), b.n_rows(), b.experts())
                );
                (a.buffer(), b.buffer())
            }
            (
                CudaWeight::Raw {
                    ty: a, buffer: x, ..
                },
                CudaWeight::Raw {
                    ty: b, buffer: y, ..
                },
            ) => {
                assert_eq!((*a, *b), (GgufType::F32, GgufType::F32));
                (x, y)
            }
            _ => panic!("mismatched DeltaNet layouts for {suffix}"),
        };
        assert_eq!(left.bytes(), right.bytes(), "{suffix}");
        for offset in (0..left.bytes()).step_by(256 * 1024) {
            let count = (left.bytes() - offset).min(256 * 1024);
            let mut a = vec![0; count];
            let mut b = vec![0; count];
            left.read_at(&stream, offset, &mut a).unwrap();
            right.read_at(&stream, offset, &mut b).unwrap();
            assert_eq!(a, b, "{suffix} offset={offset}");
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device; truncates an owned synthetic checkpoint"]
fn delta_trunk_upload_releases_partial_buffers_after_late_read_failure() {
    use super::testkit::{flash_tensors, split_kvs};
    use crate::kernels::cuda::CudaContext;
    let (seed_file, seed) = open(&Spec::default());
    let norm = seed
        .map()
        .get(Wanted::DeltaNorm, Some(0))
        .unwrap()
        .name
        .clone();
    let mut tensors = flash_tensors(&Spec::default(), seed.config());
    let total = tensors.len();
    let index = tensors
        .iter()
        .position(|tensor| tensor.name == norm)
        .unwrap();
    let isolated = tensors.remove(index);
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (1..=3)
        .map(|i| dir.path().join(format!("flash-{i:05}-of-00003.gguf")))
        .collect();
    let mut metadata = flash_kvs();
    metadata.extend(split_kvs(0, 3, total));
    write_gguf(&paths[0], &metadata, &[]);
    write_gguf(&paths[1], &split_kvs(1, 3, total), &tensors);
    write_gguf(&paths[2], &split_kvs(2, 3, total), &[isolated]);
    drop(seed);
    drop(seed_file);
    let model = GgufModel::open(&paths[0]).unwrap().unwrap();
    let last = model.map().get(Wanted::DeltaNorm, Some(0)).unwrap();
    let tensor = model.weights.header().tensor(&last.name).unwrap();
    assert_eq!(tensor.shard(), 2);
    let end = tensor.abs_offset();
    for role in [
        Wanted::DeltaQkv,
        Wanted::DeltaGate,
        Wanted::DeltaBeta,
        Wanted::DeltaAlpha,
        Wanted::DeltaOut,
        Wanted::DeltaConv,
        Wanted::DeltaDecay,
        Wanted::DeltaDtBias,
    ] {
        let mapped = model.map().get(role, Some(0)).unwrap();
        let tensor = model.weights.header().tensor(&mapped.name).unwrap();
        assert_eq!(tensor.shard(), 1, "{role:?}");
    }
    // Only a shard within this newly created temporary fixture is truncated.
    let shard = paths[2].canonicalize().unwrap();
    assert_eq!(shard.parent().unwrap(), dir.path().canonicalize().unwrap());
    std::fs::OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .set_len(end)
        .unwrap();
    for role in [
        Wanted::DeltaQkv,
        Wanted::DeltaGate,
        Wanted::DeltaBeta,
        Wanted::DeltaAlpha,
        Wanted::DeltaOut,
        Wanted::DeltaConv,
        Wanted::DeltaDecay,
        Wanted::DeltaDtBias,
    ] {
        let mapped = model.map().get(role, Some(0)).unwrap();
        model.weights.read_range(&mapped.name, 0, &mut [0]).unwrap();
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let before = ctx.live_bytes();
    let error = match model.upload_delta_trunk(0, &ctx, &stream) {
        Ok(_) => panic!("truncated DeltaNet norm was accepted"),
        Err(error) => error,
    };
    let message = error.describe();
    assert!(message.contains("file length changed"), "{message}");
    eprintln!("expected isolated-norm shard failure: {message}");
    stream.synchronize().unwrap();
    assert_eq!(
        ctx.live_bytes(),
        before,
        "partial DeltaNet upload leaked device buffers"
    );
    let usable = ctx.allocate(4).unwrap();
    assert_eq!(usable.bytes(), 4);
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn layer_trunk_upload_matches_role_loaders_and_planned_bytes() {
    use crate::family::project::cuda::CudaWeight;
    use crate::kernels::cuda::CudaContext;
    let (_file, model) = open(&Spec::default());
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let foreign = CudaContext::new().unwrap();
    let foreign_stream = foreign.stream().unwrap();
    let before = ctx.live_bytes();
    assert!(model.upload_layer_trunk(usize::MAX, &ctx, &stream).is_err());
    assert!(model.upload_layer_trunk(0, &ctx, &foreign_stream).is_err());
    assert_eq!(
        ctx.live_bytes(),
        before,
        "invalid layer or stream allocated"
    );
    let kinds = model.config().kinds();
    let delta = kinds.iter().position(|&qsa| !qsa).unwrap();
    let qsa = kinds.iter().position(|&qsa| qsa).unwrap();
    for layer in [delta, model.config().ple_layer(), qsa] {
        let actual = model.upload_layer_trunk(layer, &ctx, &stream).unwrap();
        let mut expected = model
            .upload_hyper(Site::Mixer, Some(layer), &ctx, &stream)
            .unwrap();
        if model.config().kinds()[layer] {
            expected.extend(model.upload_qsa_trunk(layer, &ctx, &stream).unwrap());
        } else {
            expected.extend(model.upload_delta_trunk(layer, &ctx, &stream).unwrap());
        }
        expected.extend(
            model
                .upload_hyper(Site::Ffn, Some(layer), &ctx, &stream)
                .unwrap(),
        );
        expected.extend(model.upload_shared_expert(layer, &ctx).unwrap());
        expected.extend(model.upload_router(layer, &ctx, &stream).unwrap());
        assert!(!actual.is_empty());
        assert_eq!(actual.len(), expected.len());
        let mut names = std::collections::HashSet::new();
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.name(), b.name());
            assert!(names.insert(a.name()), "duplicate {}", a.name());
            let (left, right) = match (a, b) {
                (CudaWeight::Native { weight: a, .. }, CudaWeight::Native { weight: b, .. }) => {
                    assert_eq!(
                        (a.quant(), a.k(), a.n_rows(), a.experts()),
                        (b.quant(), b.k(), b.n_rows(), b.experts())
                    );
                    (a.buffer(), b.buffer())
                }
                (
                    CudaWeight::Raw {
                        ty: a, buffer: x, ..
                    },
                    CudaWeight::Raw {
                        ty: b, buffer: y, ..
                    },
                ) => {
                    assert_eq!(a, b);
                    (x, y)
                }
                _ => panic!("mismatched GGUF layout for {}", a.name()),
            };
            assert_eq!(left.bytes(), right.bytes());
            for offset in (0..left.bytes()).step_by(256 * 1024) {
                let count = (left.bytes() - offset).min(256 * 1024);
                let mut a = vec![0; count];
                let mut b = vec![0; count];
                left.read_at(&stream, offset, &mut a).unwrap();
                right.read_at(&stream, offset, &mut b).unwrap();
                assert_eq!(a, b, "offset={offset}");
            }
        }
        // PLE has a separate owner. The mapped device forms conservatively charge hyper
        // norms as F32; the staged hyper kernel actually stores them as F16.
        let planned: u64 = model
            .map()
            .entries()
            .iter()
            .filter(|entry| {
                entry.layer == Some(layer)
                    && entry.place == Place::Device
                    && !matches!(entry.role, Slot::Layer(Wanted::PleKey | Wanted::PleValue))
            })
            .map(|entry| entry.device_bytes())
            .sum();
        let actual_bytes: u64 = actual
            .iter()
            .map(|weight| weight.device_bytes() as u64)
            .sum();
        let norm_overcharge: u64 = [Site::Mixer, Site::Ffn]
            .into_iter()
            .map(|site| {
                let norm = model
                    .map()
                    .get(Slot::hyper(site, SiteTensor::Norm), Some(layer))
                    .unwrap();
                norm.dims.iter().product::<u64>() * 2
            })
            .sum();
        assert_eq!(planned.checked_sub(actual_bytes), Some(norm_overcharge));
        eprintln!(
            "layer trunk accounting: layer={layer} planned_bytes={planned} staged_bytes={actual_bytes} conservative_hyper_norm_bytes={norm_overcharge}"
        );
    }
}

#[test]
fn delta_projection_row_plan_matches_host_untile() {
    let (_file, model) = open(&Spec::default());
    let dims = &model.config().dims;
    for role in [
        Wanted::DeltaQkv,
        Wanted::DeltaGate,
        Wanted::DeltaBeta,
        Wanted::DeltaAlpha,
    ] {
        let entry = model.map().get(role, Some(0)).unwrap();
        let (skip, unit) = match role {
            Wanted::DeltaQkv => (
                2 * dims.key_heads as usize * dims.key_dim as usize,
                dims.value_dim as usize,
            ),
            Wanted::DeltaGate => (0, dims.value_dim as usize),
            _ => (0, 1),
        };
        let stored: Vec<f32> = (0..entry.dims[1]).map(|n| n as f32).collect();
        let reference = super::projector::untile(
            &stored,
            skip,
            unit,
            dims.key_heads as usize,
            dims.value_heads as usize,
        );
        let plan = model.delta_projection_rows(0, role).unwrap();
        assert_eq!(
            plan.iter().map(|&i| stored[i]).collect::<Vec<_>>(),
            reference
        );
        assert!(model.delta_projection_rows(3, role).is_err());
    }
    assert!(model.delta_projection_rows(0, Wanted::DeltaOut).is_err());
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn delta_state_upload_groups_heads_and_preserves_decay() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;
    let roles = [
        (
            Wanted::DeltaConv,
            "linear_attn.conv1d.weight",
            GgufType::F16,
        ),
        (Wanted::DeltaDecay, "linear_attn.A_log", GgufType::F32),
        (Wanted::DeltaDtBias, "linear_attn.dt_bias", GgufType::Bf16),
        (Wanted::DeltaNorm, "linear_attn.norm.weight", GgufType::F32),
    ];
    let (file, model) = open(&Spec {
        retype: vec![
            ("blk.0.ssm_conv1d.weight".into(), GgufType::F16),
            ("blk.0.ssm_a".into(), GgufType::F32),
            ("blk.0.ssm_dt.bias".into(), GgufType::Bf16),
            ("blk.0.ssm_norm.weight".into(), GgufType::F32),
        ],
        ..Spec::default()
    });
    let dims = &model.config().dims;
    let (kh, vh, vd, ck) = (
        dims.key_heads as usize,
        dims.value_heads as usize,
        dims.value_dim as usize,
        dims.conv_kernel as usize,
    );
    assert!(kh > 1 && vh > kh && vh % kh == 0);
    let mut expected = Vec::new();
    for (role, _, ty) in roles {
        let entry = model.map().get(role, Some(0)).unwrap();
        assert_eq!(entry.ty, ty);
        let row_elements = if role == Wanted::DeltaConv { ck } else { 1 };
        let count = entry.bytes as usize / ty.block_bytes() as usize;
        let values: Vec<f32> = (0..count)
            .map(|i| {
                let v = (1 + (i / row_elements) % 31) as f32 + (i % row_elements) as f32 * 0.25;
                if role == Wanted::DeltaDecay { -v } else { v }
            })
            .collect();
        let mut bytes = Vec::new();
        for &value in &values {
            match ty {
                GgufType::F16 => {
                    bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                }
                GgufType::Bf16 => {
                    bytes.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes())
                }
                GgufType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                _ => unreachable!(),
            }
        }
        file.patch(&entry.name, 0, &bytes);
        let want = match role {
            Wanted::DeltaConv => super::projector::untile(
                &values,
                2 * kh * dims.key_dim as usize * ck,
                vd * ck,
                kh,
                vh,
            ),
            Wanted::DeltaDecay | Wanted::DeltaDtBias => {
                super::projector::untile(&values, 0, 1, kh, vh)
            }
            Wanted::DeltaNorm => values.clone(),
            _ => unreachable!(),
        };
        if role != Wanted::DeltaNorm {
            assert_ne!(want, values, "fixture must exercise untile");
        }
        expected.push(want);
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    assert!(model.upload_delta_state(3, &ctx, &stream).is_err());
    let uploaded = model.upload_delta_state(0, &ctx, &stream).unwrap();
    assert_eq!(uploaded.len(), roles.len());
    for ((weight, (_, suffix, _)), expected) in uploaded.iter().zip(roles).zip(expected) {
        let CudaWeight::Raw { name, ty, buffer } = weight else {
            panic!("F32 raw weight expected")
        };
        assert_eq!(*name, layer_name(0, suffix));
        assert_eq!(*ty, GgufType::F32);
        assert_eq!(buffer.bytes(), expected.len() * 4);
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut bytes).unwrap();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, expected, "{suffix}");
        if suffix == "linear_attn.A_log" {
            assert!(values.iter().all(|&v| v < 0.0));
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn delta_qkv_gate_upload_gathers_logical_rows() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;
    let (file, model) = open(&Spec {
        retype: vec![
            ("blk.0.attn_qkv.weight".into(), GgufType::Q80),
            ("blk.0.attn_gate.weight".into(), GgufType::Q80),
        ],
        ..Spec::default()
    });
    let dims = &model.config().dims;
    let roles = [
        (Wanted::DeltaQkv, "linear_attn.in_proj_qkv"),
        (Wanted::DeltaGate, "linear_attn.in_proj_z"),
    ];
    let mut expected = Vec::new();
    for (role, _) in roles {
        let entry = model.map().get(role, Some(0)).unwrap();
        let stride = entry.bytes as usize / entry.dims[1] as usize;
        let mut payload = Vec::with_capacity(entry.bytes as usize);
        for row in 0..entry.dims[1] as usize {
            let mut block = 0x3c00u16.to_le_bytes().to_vec();
            block.extend_from_slice(&[(1 + row % 127) as u8; 32]);
            payload.extend_from_slice(&block.repeat(stride / block.len()));
        }
        file.patch(&entry.name, 0, &payload);
        let skip = if role == Wanted::DeltaQkv {
            2 * dims.key_heads as usize * dims.key_dim as usize
        } else {
            0
        };
        let stored: Vec<f32> = (0..entry.dims[1]).map(|i| i as f32).collect();
        let order = super::projector::untile(
            &stored,
            skip,
            dims.value_dim as usize,
            dims.key_heads as usize,
            dims.value_heads as usize,
        );
        assert!(order.iter().enumerate().any(|(i, &row)| i != row as usize));
        expected.push(
            order
                .into_iter()
                .flat_map(|row| {
                    let start = row as usize * stride;
                    payload[start..start + stride].to_vec()
                })
                .collect::<Vec<_>>(),
        );
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = model.upload_delta_qkv_gate(0, &ctx).unwrap();
    assert_eq!(weights.len(), roles.len());
    for ((weight, (role, suffix)), expected) in weights.iter().zip(roles).zip(expected) {
        let CudaWeight::Native { name, weight } = weight else {
            panic!("native projection expected")
        };
        let entry = model.map().get(role, Some(0)).unwrap();
        assert_eq!(*name, layer_name(0, suffix));
        assert_eq!(
            (weight.k(), weight.n_rows()),
            (entry.dims[0] as usize, entry.dims[1] as usize)
        );
        let mut actual = vec![0; expected.len()];
        weight.buffer().read_at(&stream, 0, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn delta_out_upload_groups_encoded_heads_per_row() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;

    let (file, model) = open(&Spec {
        retype: vec![("blk.0.ssm_out.weight".into(), GgufType::Q80)],
        ..Spec::default()
    });
    let dims = &model.config().dims;
    let (keys, heads, dim) = (
        dims.key_heads as usize,
        dims.value_heads as usize,
        dims.value_dim as usize,
    );
    assert!(keys > 1 && heads > keys && heads % keys == 0 && dim % 32 == 0);
    let entry = model.map().get(Wanted::DeltaOut, Some(0)).unwrap();
    let rows = entry.dims[1] as usize;
    assert!(rows >= 2);
    assert_eq!(entry.dims[0] as usize, heads * dim);
    let head_bytes = dim / 32 * 34;
    let mut payload = Vec::with_capacity(entry.bytes as usize);
    for row in 0..rows {
        for head in 0..heads {
            let mut block = 0x3c00u16.to_le_bytes().to_vec();
            block.extend_from_slice(&[(1 + (row + head) % 127) as u8; 32]);
            payload.extend_from_slice(&block.repeat(dim / 32));
        }
    }
    assert_ne!(&payload[..head_bytes], &payload[head_bytes..2 * head_bytes]);
    file.patch(&entry.name, 0, &payload);
    let order = super::projector::untile(
        &(0..heads).map(|h| h as f32).collect::<Vec<_>>(),
        0,
        1,
        keys,
        heads,
    );
    assert!(order.iter().enumerate().any(|(h, &src)| h != src as usize));
    let mut expected = Vec::with_capacity(payload.len());
    for row in 0..rows {
        for &src in &order {
            let start = (row * heads + src as usize) * head_bytes;
            expected.extend_from_slice(&payload[start..start + head_bytes]);
        }
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let CudaWeight::Native { name, weight } = model.upload_delta_out(0, &ctx).unwrap() else {
        panic!("native projection expected")
    };
    assert_eq!(name, layer_name(0, "linear_attn.out_proj"));
    assert_eq!((weight.k(), weight.n_rows()), (heads * dim, rows));
    assert_eq!(weight.buffer().bytes(), expected.len());
    let mut actual = vec![0; expected.len()];
    weight.buffer().read_at(&stream, 0, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn delta_beta_alpha_upload_gathers_float_rows() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;

    let (file, model) = open(&Spec {
        retype: vec![
            ("blk.0.ssm_beta.weight".into(), GgufType::F16),
            ("blk.0.ssm_alpha.weight".into(), GgufType::F32),
        ],
        ..Spec::default()
    });
    let roles = [
        (Wanted::DeltaBeta, "linear_attn.in_proj_b.weight"),
        (Wanted::DeltaAlpha, "linear_attn.in_proj_a.weight"),
    ];
    let dims = &model.config().dims;
    let mut expected = Vec::new();
    for (index, (role, _)) in roles.iter().enumerate() {
        let entry = model.map().get(*role, Some(0)).unwrap();
        assert_eq!(
            entry.ty,
            if index == 0 {
                GgufType::F16
            } else {
                GgufType::F32
            }
        );
        let width = entry.dims[0] as usize;
        let rows = entry.dims[1] as usize;
        let mut payload = Vec::new();
        for row in 0..rows {
            for col in 0..width {
                let v = 1 + (row * 7 + col % 7 + index * 3) % 16;
                if index == 0 {
                    // Exactly representable integers 1..16 in binary16.
                    let bits = match v {
                        1 => 0x3c00u16,
                        2 => 0x4000,
                        3 => 0x4200,
                        4 => 0x4400,
                        5 => 0x4500,
                        6 => 0x4600,
                        7 => 0x4700,
                        8 => 0x4800,
                        9 => 0x4880,
                        10 => 0x4900,
                        11 => 0x4980,
                        12 => 0x4a00,
                        13 => 0x4a80,
                        14 => 0x4b00,
                        15 => 0x4b80,
                        _ => 0x4c00,
                    };
                    payload.extend_from_slice(&bits.to_le_bytes());
                } else {
                    payload.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
        }
        file.patch(&entry.name, 0, &payload);
        let stored: Vec<f32> = (0..rows).map(|r| r as f32).collect();
        let order = super::projector::untile(
            &stored,
            0,
            1,
            dims.key_heads as usize,
            dims.value_heads as usize,
        );
        assert!(order.iter().enumerate().any(|(i, &r)| i != r as usize));
        expected.push(
            order
                .into_iter()
                .flat_map(|r| {
                    (0..width)
                        .map(move |col| (1 + (r as usize * 7 + col % 7 + index * 3) % 16) as f32)
                })
                .collect::<Vec<_>>(),
        );
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = model.upload_delta_beta_alpha(0, &ctx, &stream).unwrap();
    assert_eq!(weights.len(), roles.len());
    for ((weight, (_, suffix)), expected) in weights.iter().zip(roles).zip(expected) {
        let CudaWeight::Raw { name, ty, buffer } = weight else {
            panic!("raw weight expected")
        };
        assert_eq!(*name, layer_name(0, suffix));
        assert_eq!(*ty, GgufType::F32);
        assert_eq!(buffer.bytes(), expected.len() * 4);
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut bytes).unwrap();
        let actual: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn output_head_mapping_prefers_explicit_head_and_checks_fallback() {
    let (_file, model) = open(&Spec::default());
    assert_eq!(
        model.map().get(Global::Head, None).unwrap().name,
        "output.weight"
    );
    let spec = Spec {
        output: false,
        ..Spec::default()
    };
    let (_file, tied) = open(&spec);
    assert_eq!(
        tied.map().get(Global::Head, None).unwrap().name,
        "token_embd.weight"
    );
    let invalid = Spec {
        reshape: vec![("output.weight".into(), vec![256, 4])],
        ..Spec::default()
    };
    let fixture = write_flash(&invalid);
    assert!(GgufModel::open(&fixture.first).is_err());
    let absent = Spec {
        output: false,
        omit: vec!["token_embd.weight".into()],
        ..Spec::default()
    };
    let fixture = write_flash(&absent);
    assert!(GgufModel::open(&fixture.first).is_err());
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn native_head_upload_reads_mapped_bytes_under_executor_name() {
    use crate::family::project::cuda::CudaWeight;
    use crate::kernels::cuda::CudaContext;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head.gguf");
    let payload = vec![7u8; 4 * 8 * 34];
    let tensor = super::testkit::TensorSpec {
        name: "output.weight".into(),
        dims: vec![256, 4],
        ty: GgufType::Q80,
        payload: Some(payload.clone()),
    };
    write_gguf(&path, &[], &[tensor]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::LIMITS, None).unwrap();
    let (_file, model) = open(&Spec::default());
    let mut mapped = model.map().get(Global::Head, None).unwrap().clone();
    mapped.name = "output.weight".into();
    mapped.dims = vec![256, 4];
    mapped.bytes = payload.len() as u64;
    mapped.ty = GgufType::Q80;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weight = super::upload_native_head(&weights, &mapped, "lm_head".into(), &ctx).unwrap();
    let CudaWeight::Native { name, weight } = weight else {
        panic!("native head expected")
    };
    assert_eq!(name, "lm_head");
    assert_eq!((weight.k(), weight.n_rows()), (256, 4));
    let mut bytes = vec![0; payload.len()];
    weight.buffer().read_at(&stream, 0, &mut bytes).unwrap();
    assert_eq!(bytes, payload);
    mapped.dims = vec![256, 4, 1];
    assert!(super::upload_native_head(&weights, &mapped, "lm_head".into(), &ctx).is_err());
    mapped.dims = vec![256, 4];
    mapped.ty = GgufType::F32;
    assert!(super::upload_native_head(&weights, &mapped, "lm_head".into(), &ctx).is_err());
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn qsa_norm_loader_preserves_consumer_conventions() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;
    let (file, model) = open(&Spec::default());
    let roles = [
        (Wanted::QNorm, "self_attn.q_norm.weight"),
        (Wanted::KNorm, "self_attn.k_norm.weight"),
        (Wanted::IndexQNorm, "self_attn.indexer.q_layernorm.weight"),
        (Wanted::IndexKNorm, "self_attn.indexer.k_layernorm.weight"),
    ];
    for (i, (role, _)) in roles.iter().enumerate() {
        let entry = model.map().get(*role, Some(3)).unwrap();
        assert_eq!(entry.ty, GgufType::F32);
        let gamma = 1.25f32 + i as f32 * 0.25;
        file.patch(
            &entry.name,
            0,
            &gamma.to_le_bytes().repeat(entry.dims[0] as usize),
        );
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    assert!(model.upload_qsa_norms(0, &ctx, &stream).is_err());
    let weights = model.upload_qsa_norms(3, &ctx, &stream).unwrap();
    assert_eq!(weights.len(), 4);
    for (i, (weight, (_, suffix))) in weights.iter().zip(roles).enumerate() {
        let CudaWeight::Raw { name, ty, buffer } = weight else {
            panic!("raw norm expected")
        };
        assert_eq!(*name, layer_name(3, suffix));
        assert_eq!(*ty, GgufType::F32);
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut bytes).unwrap();
        let gamma = 1.25f32 + i as f32 * 0.25;
        let expected = if i < 2 { gamma } else { gamma - 1.0 };
        for value in bytes.chunks_exact(4) {
            assert_eq!(f32::from_ne_bytes(value.try_into().unwrap()), expected);
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires an admitted CUDA device"]
fn qsa_projection_loader_uploads_each_mapped_role() {
    use crate::family::{exl3_pack::layer_name, project::cuda::CudaWeight};
    use crate::kernels::cuda::CudaContext;
    let (file, model) = open(&Spec::default());
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    assert!(model.upload_qsa_projections(0, &ctx).is_err());
    let roles = [
        (Wanted::QProj, "self_attn.q_proj"),
        (Wanted::KProj, "self_attn.k_proj"),
        (Wanted::VProj, "self_attn.v_proj"),
        (Wanted::OProj, "self_attn.o_proj"),
    ];
    for (i, (role, _)) in roles.iter().enumerate() {
        let entry = model.map().get(*role, Some(3)).unwrap();
        file.patch(&entry.name, 0, &[i as u8 + 1; 16]);
        file.patch(&entry.name, entry.bytes - 16, &[i as u8 + 11; 16]);
    }
    let weights = model.upload_qsa_projections(3, &ctx).unwrap();
    assert_eq!(weights.len(), roles.len());
    for (i, (weight, (role, suffix))) in weights.iter().zip(roles).enumerate() {
        let CudaWeight::Native { name, weight } = weight else {
            panic!("native projection expected")
        };
        let entry = model.map().get(role, Some(3)).unwrap();
        assert_eq!(*name, layer_name(3, suffix));
        assert_eq!(
            (weight.k(), weight.n_rows()),
            (entry.dims[0] as usize, entry.dims[1] as usize)
        );
        assert_eq!(weight.buffer().bytes(), entry.bytes as usize);
        let mut bytes = vec![0u8; entry.bytes as usize];
        weight.buffer().read_at(&stream, 0, &mut bytes).unwrap();
        assert_eq!(&bytes[..16], &[i as u8 + 1; 16]);
        assert!(bytes[16..bytes.len() - 16].iter().all(|&b| b == 0));
        assert_eq!(&bytes[bytes.len() - 16..], &[i as u8 + 11; 16]);
    }
}

#[test]
fn mapped_native_projection_width_bounds_one_serialized_activation() {
    let (_written, model) = open(&Spec::default());
    let max = model.map().max_quantized_input_width().unwrap();
    eprintln!("synthetic mapped native projection maximum k={max}");
    assert_eq!(max, 6144);
    for entry in model.map().entries().iter().filter(|entry| {
        (entry.place == Place::Experts || entry.form == super::tensors::Form::Blocks)
            && !super::config::is_float(entry.ty)
    }) {
        assert!(entry.dims[0] <= max as u64, "{}", entry.name);
    }
}

#[test]
#[ignore = "requires the projection-only sparse GGUF header fixture"]
fn sparse_gguf_headers_report_native_projection_width() {
    let path = std::env::var_os("FIFI_FLASH_GGUF_PROJECTION_FIXTURE")
        .expect("FIFI_FLASH_GGUF_PROJECTION_FIXTURE is required");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let max = model.map().max_quantized_input_width().unwrap();
    assert!(max > 0 && max.is_multiple_of(32));
    eprintln!("sparse header metadata only: native projection maximum k={max}");
    let head = model.map().get(Global::Head, None).expect("mapped head");
    eprintln!(
        "sparse header metadata only: head={} type={:?} dims={:?} bytes={}",
        head.name, head.ty, head.dims, head.bytes
    );
    for layer in 0..model.config().layers() {
        for role in [Wanted::IndexProj, Wanted::IndexKey] {
            if let Some(entry) = model.map().get(role, Some(layer)) {
                eprintln!(
                    "sparse header metadata only: layer={layer} role={role:?} name={} type={:?} dims={:?} bytes={}",
                    entry.name, entry.ty, entry.dims, entry.bytes
                );
            }
        }
    }
}

#[test]
fn expert_allocation_accounts_for_each_layer_and_staging() {
    use super::expert_layout::{ExpertAllocation, ExpertLayout, ExpertStacks};
    let (_file, model) = open(&Spec::default());
    assert!(ExpertAllocation::for_model(&model, 0).is_err());
    assert!(ExpertAllocation::for_model(&model, model.config().dims.experts as usize + 1).is_err());
    let allocation = ExpertAllocation::for_model(&model, 11).unwrap();
    let layouts: Vec<_> = (0..model.config().layers())
        .map(|layer| ExpertLayout::for_layer(&model, layer).unwrap())
        .collect();
    let expected: Vec<_> = layouts
        .iter()
        .map(|layout| ExpertStacks::new(layout, 11).unwrap().bytes)
        .collect();
    assert_eq!(allocation.layer_bytes, expected);
    assert_eq!(allocation.device_bytes, expected.iter().sum::<usize>());
    assert_eq!(
        allocation.staging_bytes,
        layouts.iter().map(|layout| layout.bytes).max().unwrap()
    );
}

#[test]
fn part_major_alignment_can_exceed_packed_record_accounting() {
    use super::expert_layout::{ExpertLayout, ExpertPart, ExpertStacks};
    let roles = [Wanted::ExpertGate, Wanted::ExpertUp, Wanted::ExpertDown];
    let layout = ExpertLayout {
        parts: std::array::from_fn(|i| ExpertPart {
            role: roles[i],
            ty: GgufType::Q80,
            bytes: i * 34..(i + 1) * 34,
        }),
        bytes: 102,
    };
    let stacks = ExpertStacks::new(&layout, 1).unwrap();
    assert_eq!(stacks.parts, [0..34, 36..70, 72..106]);
    assert_eq!(stacks.bytes, 106);
    assert!(stacks.bytes > layout.bytes);
    let reserve = super::plan::stack_alignment_reserve(1).unwrap() as usize;
    assert_eq!(reserve, 6);
    for slots in 1..=512 {
        let aligned = ExpertStacks::new(&layout, slots).unwrap().bytes;
        assert!(aligned <= layout.bytes * slots + reserve);
    }
}

/// The EXL3 configuration of the same model, as the pack path reads it.
fn exl3_dims() -> Dims {
    let mut dims = Dims::read(CONFIG).expect("the fixture config");
    // The pack splits its n-gram table over parts; the GGUF holds one tensor.
    dims.ngram_parts = None;
    dims
}

#[test]
fn the_metadata_gives_the_dimensions_the_exl3_configuration_gives() {
    let written = write_flash(&Spec::default());
    let weights = written.open();
    let config = GgufConfig::read(&weights).expect("config");
    assert_eq!(config.dims, exl3_dims());
    assert_eq!(config.layers(), 48);
    assert_eq!(config.ple_layer(), 1);
    assert_eq!(config.max_positions, 262_144);
    assert_eq!(config.rope_sections, [11, 11, 10, 0]);
    assert_eq!(
        config.hash.multipliers,
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    assert_eq!(config.hash.offsets.len(), 16);
    assert_eq!(config.hash.offsets[1], 20_000_003);
    assert_eq!(config.hash.vocab[0], 20_000_003);
    assert_eq!(config.ngram_rows_needed, 320_001_446);
    config
        .model_shape()
        .expect("the shape constructors accept it");
}

#[test]
fn every_tensor_of_the_real_layout_is_mapped_with_its_place_and_bytes() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    assert_eq!(map.total(), 1224, "tensors written");
    assert_eq!(map.mapped(), map.total());
    assert!(map.unmapped().is_empty(), "{:?}", map.unmapped());
    // The expert bytes the type table implies: 1,638,400 weights per matrix per expert.
    let per_matrix = |ty: GgufType| 640u64 * 2560 / ty.block_elems() * ty.block_bytes() * 512;
    let experts: u64 = REAL_EXPERT_TYPES
        .iter()
        .map(|&(gate_up, down)| 2 * per_matrix(gate_up) + per_matrix(down))
        .sum();
    assert_eq!(map.file_bytes(Place::Experts), experts);
    assert_eq!(model.expert_bytes(), experts);
    assert_eq!(map.file_bytes(Place::Ngram), 320_001_536 * 90);
    assert_eq!(
        map.get(Slot::Global(Global::Embedding), None)
            .expect("embedding")
            .bytes,
        521_472_000
    );
    // Layer 1 holds the PLE block; no other layer does.
    assert!(map.get(Slot::Layer(Wanted::PleKey), Some(1)).is_some());
    assert!(map.get(Slot::Layer(Wanted::PleKey), Some(5)).is_none());
    // Layer 3 attends, layer 0 runs DeltaNet.
    assert!(
        map.get(Slot::Layer(Wanted::QProj), Some(3)).is_some()
            && map.get(Slot::Layer(Wanted::DeltaQkv), Some(3)).is_none()
    );
    assert!(
        map.get(Slot::Layer(Wanted::DeltaQkv), Some(0)).is_some()
            && map.get(Slot::Layer(Wanted::QProj), Some(0)).is_none()
    );
    // The mixer has no inject projection; every layer site does.
    assert!(
        map.get(Slot::hyper(Site::Head, SiteTensor::Inject), None)
            .is_none()
    );
    assert!(
        map.get(Slot::hyper(Site::Mixer, SiteTensor::Inject), Some(7))
            .is_some()
    );
}

#[test]
fn the_device_trunk_counts_each_tensor_in_the_form_the_executor_holds_it() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    let at = |role: Slot, layer: Option<usize>| map.get(role, layer).expect("mapped");
    // Quantized matrices stay as stored blocks.
    let qkv = at(Slot::Layer(Wanted::DeltaQkv), Some(0));
    assert_eq!(qkv.device_bytes(), qkv.bytes);
    // A Q8_0 hyper-connection matrix stays as stored, 34 bytes a block of 32 values (10240 x 320
    // values, 7.0 MB a site against 13.1 MB widened); its inject row, F32 here, is widened to F16.
    let down = at(Slot::hyper(Site::Mixer, SiteTensor::Down), Some(0));
    assert_eq!(down.ty, GgufType::Q80);
    assert_eq!(down.device_bytes(), 10240 * 320 / 32 * 34);
    assert_eq!(
        at(Slot::hyper(Site::Mixer, SiteTensor::Inject), Some(0)).device_bytes(),
        4 * 10240 * 2
    );
    // The router is F32 as stored.
    assert_eq!(
        at(Slot::Layer(Wanted::Router), Some(0)).device_bytes(),
        2560 * 512 * 4
    );
    // The embedding and the experts are not on the card.
    assert_eq!(at(Slot::Global(Global::Embedding), None).device_bytes(), 0);
    assert_eq!(
        at(Slot::Layer(Wanted::ExpertGate), Some(0)).device_bytes(),
        0
    );
    let trunk = map.device_bytes();
    assert!(trunk > 2 * GIB && trunk < 6 * GIB, "trunk {trunk}");
}

#[test]
fn a_model_without_its_own_output_matrix_uses_the_embedding() {
    let (_written, model) = open(&Spec {
        output: false,
        ..Spec::default()
    });
    let head = model
        .map()
        .get(Slot::Global(Global::Head), None)
        .expect("head");
    assert_eq!(head.name, "token_embd.weight");
    assert_eq!(model.map().mapped(), model.map().total());
}

fn refusal(spec: &Spec) -> String {
    let written = write_flash(spec);
    match GgufModel::open(&written.first) {
        Err(error) => error.describe(),
        Ok(_) => panic!("expected a refusal"),
    }
}

#[test]
fn a_missing_tensor_is_named() {
    let text = refusal(&Spec {
        omit: vec!["blk.5.ffn_up_exps.weight".into()],
        ..Spec::default()
    });
    assert!(
        text.contains("blk.5.ffn_up_exps.weight is missing"),
        "{text}"
    );
    assert!(text.contains("1 of the model's"), "{text}");
}

#[test]
fn a_tensor_of_another_shape_is_named_with_both_shapes() {
    let text = refusal(&Spec {
        reshape: vec![("blk.3.attn_k.weight".into(), vec![2560, 256])],
        ..Spec::default()
    });
    assert!(
        text.contains(
            "blk.3.attn_k.weight has dimensions [2560, 256], the model needs [2560, 512]"
        ),
        "{text}"
    );
}

#[test]
fn a_vector_stored_as_a_quantized_type_is_refused() {
    let text = refusal(&Spec {
        retype: vec![("blk.0.ffn_gate_inp.weight".into(), GgufType::Q80)],
        ..Spec::default()
    });
    assert!(text.contains("blk.0.ffn_gate_inp.weight is Q8_0"), "{text}");
}

#[test]
fn more_than_five_problems_are_counted_and_five_are_listed() {
    let omit: Vec<String> = (0..8)
        .map(|l| format!("blk.{l}.ffn_gate_inp.weight"))
        .collect();
    let text = refusal(&Spec {
        omit,
        ..Spec::default()
    });
    assert!(text.contains("8 of the model's"), "{text}");
    assert_eq!(text.matches("is missing").count(), 5, "{text}");
}

#[test]
fn a_tensor_no_role_claims_is_reported_and_not_refused() {
    let (_written, model) = open(&Spec {
        extra: vec!["mtp.0.eh_proj.weight".into()],
        ..Spec::default()
    });
    assert_eq!(model.map().unmapped(), ["mtp.0.eh_proj.weight"]);
    assert_eq!(model.map().mapped() + 1, model.map().total());
    assert!(
        model.summary().contains("1 unmapped"),
        "{}",
        model.summary()
    );
}

#[test]
fn a_split_set_maps_like_one_file() {
    let (_one, single) = open(&Spec::default());
    let (written, split) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    assert!(split.weights.is_split());
    assert_eq!(split.weights.paths().len(), 4);
    assert_eq!(split.map().mapped(), split.map().total());
    let key = |m: &super::tensors::Mapped| (m.name.clone(), m.ty, m.bytes, m.place);
    let a: Vec<_> = single.map().entries().iter().map(key).collect();
    let b: Vec<_> = split.map().entries().iter().map(key).collect();
    assert_eq!(a, b);
    // The tensors sit in shards 2 to 4; shard 1 holds only metadata.
    let shards: std::collections::BTreeSet<usize> =
        split.map().entries().iter().map(|m| m.shard).collect();
    assert_eq!(shards, [1, 2, 3].into());
    // Opening from the last shard finds the same set.
    let last = written.dir.path().join("flash-00004-of-00004.gguf");
    let again = GgufModel::open(&last).expect("open").expect("model");
    assert_eq!(again.first_path(), written.first);
}

#[test]
fn a_directory_opens_the_flash_next_file_and_skips_other_models() {
    let written = write_flash(&Spec::default());
    // A vision projector of another architecture in the same directory.
    let other = written.dir.path().join("mmproj-F16.gguf");
    write_gguf(
        &other,
        &[super::testkit::kv_str("general.architecture", "clip")],
        &[],
    );
    let model = GgufModel::open(written.dir.path())
        .expect("open")
        .expect("model");
    assert_eq!(model.first_path(), written.first);
    // A directory with no file of this architecture is not an error.
    let empty = tempfile::tempdir().expect("tempdir");
    write_gguf(
        &empty.path().join("m.gguf"),
        &[super::testkit::kv_str("general.architecture", "other")],
        &[],
    );
    assert!(GgufModel::open(empty.path()).expect("open").is_none());
    let nothing = tempfile::tempdir().expect("tempdir");
    assert!(GgufModel::open(nothing.path()).expect("open").is_none());
}

#[test]
fn another_architecture_is_refused_by_the_config_reader() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    write_gguf(
        &path,
        &[super::testkit::kv_str("general.architecture", "other")],
        &[],
    );
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains(r#"general.architecture is "other", expected "qwen4exp""#),
        "{error}"
    );
    assert_eq!(ARCHITECTURE, "qwen4exp");
}

#[test]
fn a_missing_metadata_key_is_named() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    let kvs: Vec<Vec<u8>> = flash_kvs()
        .into_iter()
        .filter(|kv| !kv.windows(20).any(|w| w == b"hyper_connection.cou"))
        .collect();
    let embed = super::testkit::TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![2560, 248_320],
        ty: GgufType::Q6K,
        payload: None,
    };
    write_gguf(&path, &kvs, &[embed]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains("metadata key qwen4exp.hyper_connection.count is missing"),
        "{error}"
    );
}

#[test]
fn the_real_file_has_no_kernel_gaps() {
    // Its expert types (IQ2_S, IQ3_XXS, IQ4_NL, Q2_0, IQ4_XS, Q5_0) are all read by the
    // vendored or the codebook expert kernel.
    let (_written, model) = open(&Spec::default());
    let gaps = model.kernel_gaps();
    assert!(gaps.is_empty(), "{gaps:?}");
}

#[test]
fn the_refusal_names_each_type_without_a_kernel_and_the_types_that_have_one() {
    let (_written, model) = open(&Spec::default());
    let gaps = [KernelGap {
        ty: GgufType::Q41,
        tensors: 3,
        bytes: 3 << 30,
        first: "blk.0.ffn_down_exps.weight".to_owned(),
    }];
    let message = model.kernel_refusal(&gaps).describe();
    assert!(
        message.contains("cannot be served on CUDA yet"),
        "{message}"
    );
    assert!(
        message.contains("Q4_1") && message.contains("blk.0.ffn_down_exps.weight"),
        "{message}"
    );
    assert!(
        message.contains("dense and routed-expert kernels read Q4_0, Q5_0"),
        "{message}"
    );
}

#[test]
fn expert_reader_selects_layer_role_and_expert_without_touching_other_rows() {
    let (written, model) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    let entry = model.map().get(Wanted::ExpertGate, Some(0)).unwrap();
    let [k, rows, experts] = entry.dims[..] else {
        panic!("expected expert stack")
    };
    assert!(experts > 1 && rows > 2);
    let stride = (k / entry.ty.block_elems() * entry.ty.block_bytes()) as usize;
    let payload = vec![0xa7; 2 * stride];
    written.patch(&entry.name, (rows + 1) * stride as u64, &payload);
    let mut output = vec![0; 2 * stride];
    model
        .read_expert_rows(0, Wanted::ExpertGate, 1, 1..3, &mut output)
        .unwrap();
    assert_eq!(output, payload);
    model
        .read_expert_rows(0, Wanted::ExpertGate, 0, 1..3, &mut output)
        .unwrap();
    assert_eq!(output, vec![0; 2 * stride]);
    output.fill(0x55);
    assert!(
        model
            .read_expert_rows(0, Wanted::ExpertGate, experts, 1..3, &mut output)
            .is_err()
    );
    assert!(
        model
            .read_expert_rows(0, Wanted::ExpertGate, 1, 1..2, &mut output)
            .is_err()
    );
    assert!(
        model
            .read_expert_rows(0, Wanted::QProj, 1, 1..3, &mut output)
            .is_err()
    );
    assert!(
        model
            .read_expert_rows(usize::MAX, Wanted::ExpertGate, 1, 1..3, &mut output)
            .is_err()
    );
    assert_eq!(output, vec![0x55; 2 * stride]);
}

#[test]
fn expert_reader_uses_each_layers_mixed_quantization_and_role() {
    let (written, model) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    let roles = [Wanted::ExpertGate, Wanted::ExpertUp, Wanted::ExpertDown];
    for (layer, &(gate_ty, down_ty)) in REAL_EXPERT_TYPES.iter().enumerate() {
        for (index, role) in roles.iter().copied().enumerate() {
            let entry = model.map().get(role, Some(layer)).unwrap();
            assert_eq!(entry.ty, if index == 2 { down_ty } else { gate_ty });
            let [k, rows, experts] = entry.dims[..] else {
                panic!("expected expert stack")
            };
            let stride = (k / entry.ty.block_elems() * entry.ty.block_bytes()) as usize;
            let payload = vec![1 + ((layer * 3 + index) % 254) as u8; stride];
            let expert = experts - 1;
            let row = rows - 1;
            written.patch(&entry.name, (expert * rows + row) * stride as u64, &payload);
            let mut output = vec![0; stride];
            model
                .read_expert_rows(layer, role, expert, row..row + 1, &mut output)
                .unwrap();
            assert_eq!(output, payload, "layer {layer} role {role:?}");
        }
    }
}

#[test]
fn stored_expert_layout_matches_per_layer_and_global_file_sizes() {
    use super::expert_layout::ExpertLayout;
    let (_written, model) = open(&Spec::default());
    let mut sum = 0u64;
    let mut max = 0u64;
    let mut sizes = std::collections::BTreeSet::new();
    for layer in 0..model.config().layers() {
        let layout = ExpertLayout::for_layer(&model, layer).unwrap();
        let mut offset = 0;
        for part in &layout.parts {
            let tensor = model.map().get(part.role, Some(layer)).unwrap();
            assert_eq!(part.ty, tensor.ty);
            assert_eq!(part.bytes.start, offset);
            assert_eq!(part.bytes.len() as u64 * tensor.dims[2], tensor.bytes);
            offset = part.bytes.end;
        }
        assert_eq!(offset, layout.bytes);
        let experts = model.config().dims.experts as u64;
        sum += layout.bytes as u64 * experts;
        max = max.max(layout.bytes as u64);
        sizes.insert(layout.bytes);
    }
    assert!(
        sizes.len() > 1,
        "fixture should exercise mixed layer record sizes"
    );
    let planned = model.sizes().unwrap();
    assert_eq!(sum, planned.expert_bytes);
    assert_eq!(max, planned.record_bytes_max);
    assert!(ExpertLayout::for_layer(&model, model.config().layers()).is_err());
}

#[test]
fn expert_stacks_are_part_major_aligned_and_checked() {
    use super::expert_layout::{ExpertLayout, ExpertStacks};
    let (_written, model) = open(&Spec::default());
    for layer in 0..model.config().layers() {
        let record = ExpertLayout::for_layer(&model, layer).unwrap();
        let stack = ExpertStacks::new(&record, 7).unwrap();
        for (index, part) in record.parts.iter().enumerate() {
            assert_eq!(stack.parts[index].start % 4, 0);
            assert_eq!(stack.parts[index].len(), part.bytes.len() * 7);
            let first = stack.slot(index, 0).unwrap();
            let last = stack.slot(index, 6).unwrap();
            assert_eq!(first.start, stack.parts[index].start);
            assert_eq!(last.end, stack.parts[index].end);
            assert_eq!(first.len(), part.bytes.len());
            assert!(stack.slot(index, 7).is_err());
        }
        assert!(stack.slot(3, 0).is_err());
        assert!(ExpertStacks::new(&record, 0).is_err());
        assert!(ExpertStacks::new(&record, usize::MAX).is_err());
    }
}

#[test]
fn expert_stack_copy_preserves_parts_slots_and_rejects_bad_input() {
    use super::expert_layout::{ExpertLayout, ExpertStacks};
    let (_written, model) = open(&Spec::default());
    let layout = ExpertLayout::for_layer(&model, 0).unwrap();
    let stacks = ExpertStacks::new(&layout, 2).unwrap();
    let mut target = vec![0; stacks.bytes];
    let mut record = vec![0; layout.bytes];
    for (i, part) in layout.parts.iter().enumerate() {
        record[part.bytes.clone()].fill(0x10 + i as u8);
    }
    stacks
        .copy_record(&layout, 1, &record, &mut target)
        .unwrap();
    for (i, _) in layout.parts.iter().enumerate() {
        assert!(
            target[stacks.slot(i, 1).unwrap()]
                .iter()
                .all(|&v| v == 0x10 + i as u8)
        );
        assert!(target[stacks.slot(i, 0).unwrap()].iter().all(|&v| v == 0));
    }
    let before = target.clone();
    assert!(
        stacks
            .copy_record(&layout, 2, &record, &mut target)
            .is_err()
    );
    assert!(
        stacks
            .copy_record(&layout, 0, &record[..1], &mut target)
            .is_err()
    );
    let mut malformed = layout.clone();
    malformed.parts[2].bytes.end = usize::MAX;
    assert!(
        stacks
            .copy_record(&malformed, 0, &record, &mut target)
            .is_err()
    );
    assert_eq!(target, before);
}

#[test]
fn split_shard_experts_load_into_selected_part_major_slots() {
    use super::expert_layout::{ExpertLayout, ExpertStacks};
    let (written, model) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    for layer in [0, 3] {
        let layout = ExpertLayout::for_layer(&model, layer).unwrap();
        let stacks = ExpertStacks::new(&layout, 3).unwrap();
        let mut bank = vec![0; stacks.bytes];
        let mut record = vec![0; layout.bytes];
        for (expert, slot) in [(1u64, 2usize), (2, 0)] {
            for (part_index, part) in layout.parts.iter().enumerate() {
                let tensor = model.map().get(part.role, Some(layer)).unwrap();
                let value = (expert as u8) * 16 + part_index as u8 + 1;
                written.patch(
                    &tensor.name,
                    expert * part.bytes.len() as u64,
                    &vec![value; part.bytes.len()],
                );
            }
            layout.read(&model, layer, expert, &mut record).unwrap();
            stacks
                .copy_record(&layout, slot, &record, &mut bank)
                .unwrap();
        }
        for (expert, slot) in [(1u8, 2usize), (2, 0)] {
            for part_index in 0..3 {
                let expected = expert * 16 + part_index as u8 + 1;
                assert!(
                    bank[stacks.slot(part_index, slot).unwrap()]
                        .iter()
                        .all(|&v| v == expected)
                );
                assert!(
                    bank[stacks.slot(part_index, 1).unwrap()]
                        .iter()
                        .all(|&v| v == 0)
                );
            }
        }
    }
}

#[test]
fn packed_expert_read_preserves_role_boundaries_and_rejects_bad_requests() {
    use super::expert_layout::ExpertLayout;
    let (written, model) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    let layer = 0;
    let layout = ExpertLayout::for_layer(&model, layer).unwrap();
    let expert = 1;
    for (index, part) in layout.parts.iter().enumerate() {
        let tensor = model.map().get(part.role, Some(layer)).unwrap();
        written.patch(
            &tensor.name,
            expert * part.bytes.len() as u64,
            &vec![0x31 + index as u8; part.bytes.len()],
        );
    }
    let mut output = vec![0; layout.bytes];
    layout.read(&model, layer, expert, &mut output).unwrap();
    for (index, part) in layout.parts.iter().enumerate() {
        assert!(
            output[part.bytes.clone()]
                .iter()
                .all(|&byte| byte == 0x31 + index as u8)
        );
    }
    output.fill(0x55);
    assert!(
        layout
            .read(
                &model,
                layer,
                model.config().dims.experts as u64,
                &mut output
            )
            .is_err()
    );
    assert!(output.iter().all(|&byte| byte == 0x55));
    assert!(
        layout
            .read(&model, layer, expert, &mut output[..1])
            .is_err()
    );
    let mut stale = layout.clone();
    stale.parts[0].bytes.start += 1;
    assert!(stale.read(&model, layer, expert, &mut output).is_err());
    assert!(output.iter().all(|&byte| byte == 0x55));
}

#[test]
fn expert_row_spans_check_bounds_alignment_and_overflow() {
    let (_written, model) = open(&Spec::default());
    let mut stack = model
        .map()
        .entries()
        .iter()
        .find(|t| t.place == Place::Experts)
        .unwrap()
        .clone();
    let [k, rows, experts] = stack.dims[..] else {
        panic!("expert stack must have three dimensions")
    };
    let stride = k / stack.ty.block_elems() * stack.ty.block_bytes();
    assert_eq!(
        stack.row_span(Some(0), 0..rows).unwrap(),
        (0, (rows * stride) as usize)
    );
    assert_eq!(
        stack.row_span(Some(experts - 1), rows - 1..rows).unwrap().0,
        (experts * rows - 1) * stride
    );
    assert!(stack.row_span(None, 0..1).is_err());
    assert!(stack.row_span(Some(experts), 0..1).is_err());
    assert!(stack.row_span(Some(0), 0..0).is_err());
    assert!(stack.row_span(Some(0), rows..rows + 1).is_err());
    stack.bytes = 0;
    assert!(stack.row_span(Some(0), 0..1).is_err());
    stack.bytes = u64::MAX;
    stack.dims = vec![k, u64::MAX, 3];
    assert!(stack.row_span(Some(2), 0..1).is_err());
    stack.dims = vec![k + 1, rows, experts];
    assert!(stack.row_span(Some(0), 0..1).is_err());
}

#[test]
fn a_native_matrix_is_not_a_gap_anywhere_and_q4_1_is_one() {
    use super::tensors::has_decode_kernel;
    assert!(has_decode_kernel(Place::Device, GgufType::Q3K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Q3K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Q4K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Iq3Xxs));
    assert!(!has_decode_kernel(Place::Experts, GgufType::Q41));
    assert_eq!(GgufType::native_types().len(), 18);
}

#[test]
fn a_file_of_supported_types_has_no_gaps() {
    let all = |_: Place, _: GgufType| true;
    let (_written, model) = open(&Spec::default());
    assert!(model.map().kernel_gaps(all).is_empty());
    assert_eq!(GgufType::native_types().len(), 18);
}

#[test]
fn the_n_gram_rows_come_from_the_ends_of_the_28_gb_table() {
    let (written, model) = open(&Spec::default());
    // Row 320,001,535 is the last row: its blocks start past byte 2^34.
    let last = 320_001_535u64;
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
    block[2] = 0x98; // element 0: value index 8 (+1), element 16: index 9 (+13)
    written.patch("per_layer_token_embd.weight", last * 90, &block);
    let rows = model.ngram().expect("open the table");
    assert_eq!(
        (rows.rows(), rows.row_dim(), rows.row_bytes()),
        (320_001_536, 160, 90)
    );
    let row = rows.row(last).expect("last row");
    assert_eq!((row[0], row[16]), (1.0, 13.0));
    assert!(
        row[1..16].iter().all(|&v| v == -127.0),
        "zero nibbles decode to the first value"
    );
    // A row nobody wrote decodes to zero: the scale is zero.
    assert!(rows.row(0).expect("first row").iter().all(|&v| v == 0.0));
    assert!(rows.row(last + 1).is_err());
}

fn card(free: u64, total: u64, available: u64) -> Card {
    Card {
        free_vram: free,
        host: HostMemory::from_pair(total, available),
        ring_budget: 0,
    }
}

#[test]
fn card_budget_charges_ring_and_expert_alignment_without_underflow() {
    let mut machine = card(1000, 62 * GIB, 40 * GIB);
    machine.ring_budget = 200;
    assert_eq!(machine.free_after_reserves(48).unwrap(), 512);
    machine.free_vram = 487;
    assert_eq!(machine.free_after_reserves(48).unwrap(), 0);
    machine.free_vram = 199;
    assert_eq!(machine.free_after_reserves(48).unwrap(), 0);
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn staged_decode_arena_is_charged_at_the_exact_admission_boundary() {
    use super::super::super::moe::context_budget::KvFormat;
    let (_file, model) = open(&Spec::default());
    for positions in [1usize, 4096, 32768] {
        for sm_count in [1u32, 68, 82] {
            let (_, arena) = model
                .load_plan_with_decode_arena(
                    card(64 * GIB, 62 * GIB, 40 * GIB),
                    positions,
                    sm_count,
                )
                .unwrap();
            let boundary = |with_arena: bool| {
                let admits = |free| {
                    let machine = card(free, 62 * GIB, 40 * GIB);
                    let plan = if with_arena {
                        model
                            .load_plan_with_decode_arena(machine, positions, sm_count)
                            .map(|(plan, _)| plan)
                    } else {
                        model.load_plan(machine)
                    };
                    plan.and_then(|plan| plan.at_context(positions as u64, KvFormat::F16, 11))
                        .is_ok()
                };
                let (mut low, mut high) = (0, 64 * GIB);
                assert!(!admits(low) && admits(high));
                while high - low > 1 {
                    let mid = low + (high - low) / 2;
                    if admits(mid) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                high
            };
            assert!(arena > 0);
            assert_eq!(
                boundary(true) - boundary(false),
                arena,
                "positions={positions}, sm={sm_count}"
            );
        }
    }
    assert!(
        model
            .load_plan_with_decode_arena(card(0, 62 * GIB, 40 * GIB), 4096, 82)
            .is_err()
    );
    assert!(
        model
            .load_plan_with_decode_arena(card(24 * GIB, 62 * GIB, 40 * GIB), 0, 82)
            .is_err()
    );
}

#[test]
fn context_admission_charges_ring_budget_at_the_exact_boundary() {
    use super::super::super::moe::context_budget::KvFormat;
    let (_file, model) = open(&Spec::default());
    let min_free = |ring: u64| {
        let admits = |free| {
            let mut machine = card(free, 62 * GIB, 40 * GIB);
            machine.ring_budget = ring;
            model
                .load_plan(machine)
                .and_then(|plan| plan.at_context(4096, KvFormat::F16, 11))
                .is_ok()
        };
        let mut low = 0;
        let mut high = 24 * GIB;
        assert!(!admits(low) && admits(high));
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            if admits(mid) {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    };
    assert_eq!(min_free(288), min_free(0) + 288);
}

#[test]
fn gguf_ple_weights_load_without_transformer_trunk() {
    let (_written, model) = open(&Spec::default());
    let ple = super::ple_weights::GgufPleWeights::load(&model).expect("PLE weights");
    let view = ple.view();
    let shape = model.config().model_shape().unwrap();
    let dims = shape.ple.dims();
    let width = dims.hidden as usize * dims.branches as usize;
    assert_eq!(view.shards.len(), 1);
    assert_eq!(view.shard_rows.len(), 1);
    assert!(view.shard_rows[0] > 0);
    assert_eq!(view.norm_key.len(), width);
    assert_eq!(view.norm_query.len(), width);
    assert_eq!(view.norm_conv.len(), width);
    assert_eq!(view.conv.len(), width * dims.conv_kernel as usize);
    assert_eq!(view.weight_scale, 1.0);
}

#[test]
fn gguf_feed_drives_token_driver_with_fake_device() {
    let (_written, model) = open(&Spec::default());
    let rows = model.ngram().unwrap();
    let feed = super::token_feed::GgufTokenFeeds::new(&model, &rows, 2).unwrap();
    let shape = model.config().model_shape().unwrap();
    super::super::super::cuda::driver::fakes::exercise_host_feed(feed, &shape);
}

#[test]
fn gguf_feed_drives_chunk_driver_with_fake_device() {
    let (_written, model) = open(&Spec::default());
    let rows = model.ngram().unwrap();
    let feed = super::token_feed::GgufTokenFeeds::new(&model, &rows, 2).unwrap();
    let shape = model.config().model_shape().unwrap();
    super::super::super::cuda::driver::fakes::exercise_chunk_feed(feed, &shape);
}

#[test]
fn gguf_token_feed_rejects_a_different_ngram_shard() {
    let (_a, model_a) = open(&Spec::default());
    let (_b, model_b) = open(&Spec::default());
    let rows_a = model_a.ngram().unwrap();
    assert!(super::token_feed::GgufTokenFeeds::new(&model_a, &rows_a, 1).is_ok());
    let error = super::token_feed::GgufTokenFeeds::new(&model_b, &rows_a, 1)
        .err()
        .expect("different source must be refused");
    assert!(error.describe().contains("n-gram source mismatch"));

    // Replace only a path owned by this fixture: the model must keep using its opened shard.
    let table = model_a
        .map()
        .get(crate::family::recipe::Global::NgramTable, None)
        .unwrap();
    let shard = model_a
        .weights
        .header()
        .tensor(&table.name)
        .unwrap()
        .shard();
    let path_a = &model_a.weights.paths()[shard];
    assert_eq!(path_a.parent(), Some(_a.dir.path()));
    let old = path_a.with_extension("saved-gguf");
    assert!(!old.exists());
    std::fs::rename(path_a, &old).unwrap();
    std::fs::hard_link(&model_b.weights.paths()[shard], path_a).unwrap();
    let reopened = model_a.ngram().unwrap();
    assert!(rows_a.same_source(&reopened).unwrap());
    let rows_b = model_b.ngram().unwrap();
    assert!(super::token_feed::GgufTokenFeeds::new(&model_a, &rows_b, 1).is_err());
    assert!(super::token_feed::GgufTokenFeeds::new(&model_a, &rows_a, 1).is_ok());
}

#[test]
fn gguf_ple_load_fixture_payloads_are_visible_to_admission_work() {
    let (_written, model) = open(&Spec::default());
    let layer = Some(model.config().dims.ple_layer);
    let roles = [
        Wanted::PleNormKey,
        Wanted::PleNormQuery,
        Wanted::PleNormConv,
        Wanted::PleConv,
    ];
    let mut retained = 0u64;
    let mut peak = 0u64;
    for role in roles {
        let entry = model.map().get(role, layer).unwrap();
        let decoded = entry
            .dims
            .iter()
            .try_fold(4u64, |n, d| n.checked_mul(*d))
            .unwrap();
        peak = peak.max(
            retained
                .checked_add(entry.bytes)
                .unwrap()
                .checked_add(decoded)
                .unwrap(),
        );
        retained = retained.checked_add(decoded).unwrap();
    }
    let loaded = super::ple_weights::GgufPleWeights::load(&model).unwrap();
    let view = loaded.view();
    let actual = 4
        * (view.norm_key.len() + view.norm_query.len() + view.norm_conv.len() + view.conv.len())
            as u64;
    assert_eq!(retained, actual);
    assert_eq!(retained, 286_720);
    assert_eq!(peak, 450_560);
    eprintln!("PLE load fixture decoded={retained} peak_encoded_plus_decoded={peak}");
}

#[test]
fn gguf_feed_fixture_chunk_payloads_are_visible_to_admission_work() {
    let (_written, model) = open(&Spec::default());
    let shape = model.config().model_shape().unwrap();
    let ple = shape.ple.dims();
    let rows = u64::from(ple.ngram - 1) * u64::from(ple.heads_per_ngram);
    let embedding = rows * u64::from(ple.head_dim);
    let stream = u64::from(ple.hidden) * u64::from(ple.branches);
    let tokens = u64::from(super::super::super::settings::MAX_PREFILL_CHUNK);
    let pending = tokens * embedding * 4;
    let starts = tokens * (stream + embedding) * 4;
    let staging = super::super::super::host_plan::HostTerms::DEFAULT
        .ngram_staging_bytes()
        .unwrap();
    assert_eq!((rows, embedding), (16, 2560));
    assert_eq!(
        (pending, starts, staging),
        (20_971_520, 104_857_600, 85_983_232)
    );
    eprintln!(
        "{tokens}-token fixture payloads: pending={pending} starts={starts} existing staging={staging}; excludes metadata and other allocations"
    );
}

/// Measurement probe only: max chunk of synthetic sparse rows, not an admission test.
#[test]
#[ignore = "allocates a maximum-size synthetic token-feed chunk for memory measurement"]
fn gguf_token_feed_max_chunk_memory_probe() {
    use super::super::super::cuda::driver::TokenFeeds;
    use super::token_feed::GgufTokenFeeds;
    let (_written, model) = open(&Spec::default());
    let rows = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &rows, 8).unwrap();
    let tokens = vec![1; super::super::super::settings::MAX_PREFILL_CHUNK as usize];
    let starts = feed.begin_chunk(&tokens, false).unwrap();
    assert_eq!(starts.len(), tokens.len());
    let update = feed.ple_update(tokens[0], &starts[0].stream).unwrap();
    assert_eq!(update.len(), starts[0].stream.len());
    std::hint::black_box((&starts, &feed));
}

#[test]
fn gguf_ple_projector_fixture_worker_buffer_bounds() {
    use crate::family::recipe::Role;
    let (_written, model) = open(&Spec::default());
    let layer = Some(model.config().dims.ple_layer);
    for role in [Role::PleKey, Role::PleValue] {
        let tensor = model.map().get(role, layer).unwrap();
        let input = tensor.dims[0];
        let output = tensor.dims[1];
        let row_bytes = input / tensor.ty.block_elems() * tensor.ty.block_bytes();
        for threads in [1u64, 8, 64] {
            // Mirrors projector.rs matvec worker partitioning and per-worker chunk size.
            let workers = threads.min(output.div_ceil(64)).max(1);
            let share = output.div_ceil(workers);
            let chunk_rows = (24 * 1024 * 1024 / row_bytes).max(1).min(share);
            let encoded_and_decoded = workers * chunk_rows * (row_bytes + 4 * input);
            assert!(encoded_and_decoded > 0);
            eprintln!(
                "{role:?} threads={threads} workers={workers} encoded+decoded payload <= {encoded_and_decoded} bytes"
            );
        }
    }
}

#[test]
fn gguf_token_feed_chunk_matches_serial_and_resume() {
    use super::super::super::cuda::driver::TokenFeeds;
    use super::token_feed::GgufTokenFeeds;
    let (_written, model) = open(&Spec::default());
    let rows = model.ngram().unwrap();
    let mut chunk = GgufTokenFeeds::new(&model, &rows, 2).unwrap();
    let mut serial = GgufTokenFeeds::new(&model, &rows, 2).unwrap();
    let shape = model.config().model_shape().unwrap();
    let weights = super::ple_weights::GgufPleWeights::load(&model).unwrap();
    // Populate only the planned IQ4_NL n-gram rows in this sparse synthetic file.
    // These are test bytes, not a sample from the pinned checkpoint.
    {
        use std::os::unix::fs::FileExt;
        let mapped = model
            .map()
            .get(crate::family::recipe::Global::NgramTable, None)
            .unwrap();
        let tensor = model.weights.header().tensor(&mapped.name).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&model.weights.paths()[tensor.shard()])
            .unwrap();
        let mut block = vec![0x00, 0x3c]; // F16 scale 1.0
        block.extend([0x88; 16]);
        let encoded = block.repeat(mapped.dims[0] as usize / 32);
        let mut history = Vec::new();
        for token in [1, 2, shape.ple.dims().eos] {
            let plan =
                super::super::super::ple::plan_rows(&shape.ple, &weights.view(), &history, token)
                    .unwrap();
            for (_, row) in plan.rows {
                file.write_all_at(
                    &encoded,
                    tensor.abs_offset() + u64::from(row) * encoded.len() as u64,
                )
                .unwrap();
            }
            history = plan.history;
        }
        // One nonzero Q8_0 block of the PLE value matrix gives a nonzero update.
        let mapped = model
            .map()
            .get(
                crate::family::recipe::Role::PleValue,
                Some(model.config().dims.ple_layer),
            )
            .unwrap();
        assert_eq!(mapped.ty, crate::weights::GgufType::Q80);
        let tensor = model.weights.header().tensor(&mapped.name).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&model.weights.paths()[tensor.shard()])
            .unwrap();
        let mut block = vec![0x00, 0x2c]; // F16 scale 1/16
        block.extend([1; 32]);
        file.write_all_at(&block, tensor.abs_offset()).unwrap();
        // Populate different Q6_K token-embedding blocks to check branch replication.
        let mapped = model
            .map()
            .get(crate::family::recipe::Global::Embedding, None)
            .unwrap();
        assert_eq!(mapped.ty, crate::weights::GgufType::Q6K);
        let tensor = model.weights.header().tensor(&mapped.name).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&model.weights.paths()[tensor.shard()])
            .unwrap();
        let row_bytes = mapped.dims[0] / mapped.ty.block_elems() * mapped.ty.block_bytes();
        for (token, low) in [(1u64, 0x11), (2, 0x22)] {
            let mut block = vec![low; 128];
            block.extend([0xaa; 64]);
            block.extend([1; 16]);
            block.extend([0x00, 0x2c]);
            file.write_all_at(&block, tensor.abs_offset() + token * row_bytes)
                .unwrap();
        }
    }
    let mut projector = super::projector::HostProjector::new(&model.weights, &rows, 2);
    let mut state = super::super::super::ple::PleState::new(&shape.ple);
    let tokens = [1, 2];
    let starts = chunk.begin_chunk(&tokens, false).unwrap();
    assert_eq!(starts.len(), 2);
    for (start, value) in starts.iter().zip([0.0625f32, 0.125]) {
        for branch in start.stream.chunks_exact(shape.hyper.hidden()) {
            assert!(branch[..256].iter().all(|&x| x == value));
            assert!(branch[256..].iter().all(|&x| x == 0.0));
        }
    }
    assert!(
        starts
            .iter()
            .all(|start| start.ple_embedding.iter().any(|&x| x != 0.0)),
        "synthetic n-gram rows must be nonzero"
    );
    // A late failure after preparing token 1 must preserve the previous pending chunk.
    assert!(chunk.begin_chunk(&[1, u32::MAX], false).is_err());
    assert!(chunk.ple_update(2, &starts[0].stream).is_err());
    for (token, start) in tokens.into_iter().zip(&starts) {
        let reference = serial.begin(token).unwrap();
        assert_eq!(*start, reference);
        let a = chunk.ple_update(token, &start.stream).unwrap();
        let b = serial.ple_update(token, &reference.stream).unwrap();
        let oracle = super::super::super::ple::step(
            &mut projector,
            &shape.ple,
            &weights.view(),
            &mut state,
            token,
            &start.stream,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, oracle);
        assert!(
            a.iter().any(|&value| value != 0.0),
            "synthetic PLE update must be nonzero"
        );
    }
    assert_eq!(chunk.ple_state(), serial.ple_state());
    let (history, ring) = chunk.ple_state().unwrap();
    let (history, ring) = (history.to_vec(), ring.to_vec());
    chunk.reset();
    assert_ne!(chunk.ple_state().unwrap().0, history);
    chunk.ple_resume(&history, &ring).unwrap();
    assert_eq!(chunk.ple_state(), serial.ple_state());
    assert!(chunk.ple_resume(&history, &[]).is_err());
    assert_eq!(chunk.ple_state(), serial.ple_state());
    let start = chunk.begin(1).unwrap();
    assert!(chunk.ple_update(1, &[]).is_err());
    assert_eq!(chunk.ple_state(), serial.ple_state());
    assert!(
        chunk.ple_update(1, &start.stream).is_err(),
        "failed update consumes matching rows"
    );
    chunk.reset();
    serial.reset();
    let eos = shape.ple.dims().eos;
    let tokens = [1, eos, 2];
    let starts = chunk.begin_chunk(&tokens, false).unwrap();
    for (index, (token, start)) in tokens.into_iter().zip(&starts).enumerate() {
        let reference = serial.begin(token).unwrap();
        assert_eq!(*start, reference, "token {index}");
        assert_eq!(
            chunk.ple_update(token, &start.stream).unwrap(),
            serial.ple_update(token, &reference.stream).unwrap()
        );
        assert_eq!(chunk.ple_state(), serial.ple_state());
        if index == 1 {
            assert!(
                chunk.ple_state().unwrap().0.is_empty(),
                "EOS clears history"
            );
        }
    }
    let pending = chunk.begin(1).unwrap();
    assert!(chunk.begin_chunk(&[], false).unwrap().is_empty());
    assert!(
        chunk.ple_update(1, &pending.stream).is_err(),
        "empty chunk clears pending rows"
    );
    assert_eq!(chunk.ple_state(), serial.ple_state());
}

#[test]
fn the_plan_on_a_24_gib_card_and_a_62_gib_host_does_not_refuse() {
    let (_written, model) = open(&Spec::default());
    let sizes = model.sizes().expect("sizes");
    // 83.6 GB in the files: the experts and the n-gram table dominate.
    let total = sizes.expert_bytes + sizes.ngram_bytes + sizes.embedding_bytes + sizes.trunk_bytes;
    assert!(
        total > 70 * 1_000_000_000 && total < 90 * 1_000_000_000,
        "{total}"
    );
    // A 24 GiB card with about 1.2 GiB held by the desktop, a 62 GiB host with 40 GiB free.
    for (available, label) in [
        (40 * GIB, "40 GiB free"),
        (20 * GIB, "20 GiB free"),
        (16 * GIB, "16 GiB free"),
    ] {
        let plan = model
            .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, available))
            .expect(label);
        let load = plan
            .at_context(
                4096,
                crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16,
                11,
            )
            .unwrap_or_else(|error| panic!("{label}: {}", error.describe()));
        assert!(load.context.plan.slots_per_layer() >= 11, "{label}");
        eprintln!("{label}: {load}; tier {:?}", load.context.plan.store_tier());
    }
}

#[test]
fn a_host_that_cannot_pin_the_cold_experts_plans_the_disk_tier() {
    use crate::family::expert_bank::bank_tier::TierChoice;
    let (_written, model) = open(&Spec::default());
    let kv = crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16;
    let tier = |available| {
        model
            .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, available))
            .and_then(|plan| plan.at_context(4096, kv, 11))
            .map(|load| load.context.plan.store_tier())
    };
    // A host with room for the whole store pins it; the same card with 16 GiB free (the 8 GiB OS
    // floor and the process terms leave no room for a pinned store) reads from disk.
    assert!(matches!(
        tier(200 * GIB),
        Ok(TierChoice::Inclusive | TierChoice::Exclusive { .. })
    ));
    assert!(matches!(tier(16 * GIB), Ok(TierChoice::Disk)));
}

#[test]
fn the_tier_shape_uses_the_mean_expert_and_the_ring_the_largest_record() {
    let (_written, model) = open(&Spec::default());
    let sizes = model.sizes().expect("sizes");
    let records = 48 * 512;
    assert_eq!(
        sizes.shape.expert_bytes,
        sizes.expert_bytes.div_ceil(records)
    );
    assert!(sizes.record_bytes_max > sizes.shape.expert_bytes);
    // Layer 47: IQ4_XS gate and up, IQ4_NL down.
    let xs = 640u64 * 2560 / 256 * 136;
    let nl = 640u64 * 2560 / 32 * 18;
    assert_eq!(sizes.record_bytes_max, 2 * xs + nl);
}

#[test]
fn the_tensor_map_type_is_send() {
    fn needs_send<T: Send>() {}
    needs_send::<TensorMap>();
    let _ = EngError::BadRequest(String::new());
}

/// Real-file checks: set FIFI_FLASH_GGUF to the model directory (or its first shard) and run
/// with `--nocapture`. Skipped when unset. Set FIFI_FLASH_EXL3 to the EXL3 pack directory to
/// also compare the configuration and the n-gram hash constants with the pack's.
fn real_model() -> Option<GgufModel> {
    let path = std::env::var("FIFI_FLASH_GGUF").ok()?;
    Some(
        GgufModel::open(std::path::Path::new(&path))
            .expect("open the real model")
            .expect("a qwen4exp model at FIFI_FLASH_GGUF"),
    )
}

#[test]
fn real_file_maps_every_tensor_and_agrees_with_the_exl3_configuration() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let map = model.map();
    eprintln!("{}", model.summary());
    eprintln!("tensors mapped {} of {}", map.mapped(), map.total());
    assert_eq!(map.mapped(), map.total(), "unmapped: {:?}", map.unmapped());
    let sizes = model.sizes().expect("sizes");
    eprintln!("{sizes:?}");
    for gap in model.kernel_gaps() {
        eprintln!("gap {gap:?}");
    }
    eprintln!("{}", model.kernel_refusal(&model.kernel_gaps()).describe());
    if let Ok(exl3) = std::env::var("FIFI_FLASH_EXL3") {
        let config =
            std::fs::read(std::path::Path::new(&exl3).join("config.json")).expect("config");
        let mut want = Dims::read(&config).expect("exl3 dims");
        want.ngram_parts = None;
        assert_eq!(model.config().dims, want);
        eprintln!("dims equal the EXL3 config.json: {:?}", want);
    }
    // The three default-context lines for a 24 GiB card and a 62 GiB host.
    let kv = crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16;
    let plan = model
        .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, 40 * GIB))
        .expect("plan");
    let load = plan.at_context(4096, kv, 11).expect("context");
    eprintln!("{load}; tier {:?}", load.context.plan.store_tier());
}

#[test]
fn real_file_ngram_rows_decode_to_finite_values() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let rows = model.ngram().expect("table");
    assert_eq!(
        (rows.rows(), rows.row_dim(), rows.row_bytes()),
        (320_001_536, 160, 90)
    );
    let mut sum = 0f64;
    for row in [0u64, 1, 20_000_003, 160_000_000, 320_001_445] {
        let values = rows.row(row).expect("row");
        assert!(values.iter().all(|v| v.is_finite()), "row {row}");
        sum += values.iter().map(|v| f64::from(v.abs())).sum::<f64>();
    }
    assert!(sum > 0.0, "all sampled rows are zero");
    eprintln!("mean |value| over five rows: {}", sum / 800.0);
}

#[test]
fn real_file_host_forward_of_the_first_layers() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let layers: usize = std::env::var("FIFI_FLASH_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let host = super::host::HostModel::load(&model, layers).expect("load the layers");
    let tokens = [9707u32, 11, 1879];
    let run = super::host::run(&model, &host, &tokens, 8).expect("host forward");
    let rms = super::host::rms_by_layer(&run);
    for (layer, values) in &rms {
        eprintln!("layer {layer}: rms per token {values:?}");
        assert!(
            values.iter().all(|v| v.is_finite() && *v > 0.0),
            "layer {layer}"
        );
    }
    // Different tokens at different positions give different residuals.
    assert_ne!(run.layer_out[0][layers - 1], run.layer_out[2][layers - 1]);
    assert!(run.logits.iter().flatten().all(|v| v.is_finite()));
    // A second run reproduces the first.
    let again = super::host::run(&model, &host, &tokens[..1], 3).expect("again");
    assert_eq!(again.layer_out[0], run.layer_out[0]);
}

/// Save the true final residual for each token in the same uninterrupted host prompt.
#[test]
#[ignore = "requires the pinned GGUF and independently saved full-prompt oracle"]
fn pinned_prompt_final_streams_host_capture() {
    let oracle: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE").unwrap()).unwrap(),
    )
    .unwrap();
    let tokens = [9707u32, 11, 1879];
    assert_eq!(oracle["prompt"], serde_json::json!(tokens));
    let expected: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("FIFI_FLASH_GGUF_FINAL_STREAMS_OUT").unwrap())
        .unwrap();
    let model = real_model().expect("set verified pinned GGUF");
    let host = super::host::HostModel::load(&model, model.config().layers()).unwrap();
    let run = super::host::run(&model, &host, &tokens, 8).unwrap();
    assert_eq!(run.layer_out.len(), tokens.len());
    assert_eq!(run.logits.len(), tokens.len());
    assert_eq!(run.logits[2].len(), expected.len());
    let (err, norm) =
        run.logits[2]
            .iter()
            .zip(&expected)
            .fold((0.0f64, 0.0f64), |(e, n), (&got, &want)| {
                (
                    e + f64::from(got - want).powi(2),
                    n + f64::from(want).powi(2),
                )
            });
    let relative = (err / norm).sqrt();
    assert!(
        relative.is_finite() && relative < 1e-6,
        "oracle logit error {relative}"
    );
    assert_eq!(
        super::host::greedy_token(&run.logits[2]).unwrap(),
        oracle["greedy"][0].as_u64().unwrap() as u32
    );
    let streams: Vec<_> = run
        .layer_out
        .iter()
        .map(|layers| {
            assert_eq!(layers.len(), model.config().layers());
            let stream = layers.last().unwrap();
            assert_eq!(
                stream.len(),
                model.config().model_shape().unwrap().hyper.width()
            );
            assert!(stream.iter().all(|v| v.is_finite()));
            stream
        })
        .collect();
    serde_json::to_writer(
        &mut output,
        &serde_json::json!({"prompt": tokens, "layers": model.config().layers(), "streams": streams}),
    )
    .unwrap();
    std::io::Write::flush(&mut output).unwrap();
    eprintln!(
        "pinned full-prompt host final logit relative L2={relative:.9}; saved {} final streams",
        streams.len()
    );
}

/// Captures one pinned token through a selected prefix and checks every residual against
/// the saved full-model trace before a device probe consumes its projection rows.
#[test]
#[ignore = "requires pinned GGUF and independently saved full-model layer trace"]
fn pinned_prefix_host_capture() {
    let target: usize = std::env::var("FIFI_FLASH_GGUF_CAPTURE_LAYER")
        .expect("set capture layer")
        .parse()
        .expect("capture layer is an integer");
    let capture = std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE").expect("set new capture path");
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&capture)
        .unwrap();
    let expected: Vec<Vec<Vec<f32>>> = serde_json::from_reader(
        std::fs::File::open(
            std::env::var_os("FIFI_FLASH_GGUF_HOST_LAYERS").expect("set verified full-model trace"),
        )
        .unwrap(),
    )
    .unwrap();
    let model = real_model().expect("set verified pinned checkpoint");
    assert_eq!(
        model.config().dims.ple_layer,
        1,
        "probe must cross the PLE boundary"
    );
    assert!(target >= model.config().dims.ple_layer);
    assert!(target < expected[0].len() && target < model.config().layers());
    let host = super::host::HostModel::load(&model, target + 1).unwrap();
    let run = super::host::run(&model, &host, &[9707], 8).unwrap();
    for (layer, (actual, want)) in run.layer_out[0][..=target]
        .iter()
        .zip(&expected[0][..=target])
        .enumerate()
    {
        assert_eq!(actual.len(), want.len());
        let (err, norm) = actual
            .iter()
            .zip(want)
            .fold((0f64, 0f64), |(err, norm), (&a, &b)| {
                (err + f64::from(a - b).powi(2), norm + f64::from(b).powi(2))
            });
        let relative = (err / norm).sqrt();
        eprintln!(
            "pinned token 9707 prefix host trace: layer={layer} residual relative L2={relative:.9}"
        );
        assert!(relative.is_finite() && relative < 1e-6);
    }
    let text = std::fs::read_to_string(capture).unwrap();
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|row| {
        row["name"]
            .as_str()
            .unwrap()
            .starts_with(&format!("blk.{target}."))
    }));
    for suffix in [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_down_shexp.weight",
    ] {
        assert!(
            rows.iter()
                .any(|row| row["name"] == format!("blk.{target}.{suffix}"))
        );
    }
    eprintln!(
        "pinned token 9707 layer-{target} projection rows={}",
        rows.len()
    );
}

/// Capture three prompt positions on one uninterrupted full-model host state, and
/// bind the trace to the earlier single-token trace and full-prompt logits oracle.
#[test]
#[ignore = "requires the pinned GGUF, verified full host trace and prompt oracle"]
fn pinned_prompt_prefix_host_capture() {
    let capture = std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE").expect("set new capture");
    assert_eq!(
        std::env::var("FIFI_FLASH_GGUF_CAPTURE_PREFIX").unwrap(),
        "4"
    );
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&capture)
        .unwrap();
    let trace: Vec<Vec<Vec<f32>>> = serde_json::from_reader(
        std::fs::File::open(std::env::var_os("FIFI_FLASH_GGUF_HOST_LAYERS").unwrap()).unwrap(),
    )
    .unwrap();
    let oracle: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE").unwrap()).unwrap(),
    )
    .unwrap();
    let tokens = [9707u32, 11, 1879];
    assert_eq!(oracle["prompt"], serde_json::json!(tokens));
    let logits: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    let model = real_model().expect("set verified pinned GGUF");
    let host = super::host::HostModel::load(&model, model.config().layers()).unwrap();
    let run = super::host::run(&model, &host, &tokens, 8).unwrap();
    assert_eq!(run.layer_out.len(), tokens.len());
    assert_eq!(run.layer_out[0].len(), trace[0].len());
    let relative = |actual: &[f32], want: &[f32]| {
        assert_eq!(actual.len(), want.len());
        let (err, norm) = actual
            .iter()
            .zip(want)
            .fold((0f64, 0f64), |(e, n), (&a, &b)| {
                (e + f64::from(a - b).powi(2), n + f64::from(b).powi(2))
            });
        let value = (err / norm).sqrt();
        assert!(value.is_finite() && value < 1e-6, "relative L2 {value}");
        value
    };
    for (layer, (actual, want)) in run.layer_out[0].iter().zip(&trace[0]).enumerate() {
        eprintln!(
            "prompt host first-token layer={layer} residual relative L2={:.9}",
            relative(actual, want)
        );
    }
    eprintln!(
        "prompt host final logits relative L2={:.9}",
        relative(&run.logits[2], &logits)
    );
    assert_eq!(super::host::greedy_token(&run.logits[2]).unwrap(), 5854);
    let prefix: Vec<_> = run.layer_out.iter().map(|layers| &layers[..4]).collect();
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("FIFI_FLASH_GGUF_PROMPT_LAYERS_OUT").unwrap())
        .unwrap();
    serde_json::to_writer(&mut output, &prefix).unwrap();
    std::io::Write::flush(&mut output).unwrap();
    let text = std::fs::read_to_string(capture).unwrap();
    let boundaries: Vec<_> = text
        .lines()
        .filter_map(|line| {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            row.get("position").map(|_| {
                (
                    row["position"].as_u64().unwrap(),
                    row["token"].as_u64().unwrap(),
                )
            })
        })
        .collect();
    assert_eq!(boundaries, vec![(0, 9707), (1, 11), (2, 1879)]);
    eprintln!("captured prompt positions 0..3 across layers 0..4");
}

/// Preserve all layer residuals and ordered route logs for one uninterrupted prompt state.
#[test]
#[ignore = "requires the pinned GGUF, verified first-token trace and prompt oracle"]
fn pinned_prompt_stateful_host_trace() {
    assert!(std::env::var_os("FIFI_FLASH_GGUF_STATEFUL_TRACE").is_some());
    assert!(std::env::var_os("FIFI_FLASH_GGUF_TRACE_ROUTES").is_some());
    assert!(std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE").is_none());
    use sha2::{Digest, Sha256};
    let read_pinned = |variable: &str, hash: &str| {
        let path = std::env::var_os(variable).expect(variable);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            hash,
            "unexpected {variable}"
        );
        bytes
    };
    let trace: Vec<Vec<Vec<f32>>> = serde_json::from_slice(&read_pinned(
        "FIFI_FLASH_GGUF_HOST_LAYERS",
        "62fdd25ecb4f3e14f2f4417de026421e6e6557c972eadee3e779e9cc6656bace",
    ))
    .unwrap();
    let oracle: serde_json::Value = serde_json::from_slice(&read_pinned(
        "FIFI_FLASH_GGUF_HOST_ORACLE",
        "2c2133dccb289ad6f3e244d304f655189c04497bf19cd3b609779eb5af235ffa",
    ))
    .unwrap();
    let tokens = [9707u32, 11, 1879];
    assert_eq!(oracle["prompt"], serde_json::json!(tokens));
    let logits: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    let model = real_model().expect("set verified pinned GGUF");
    let host = super::host::HostModel::load(&model, model.config().layers()).unwrap();
    let run = super::host::run(&model, &host, &tokens, 8).unwrap();
    assert_eq!(run.layer_out.len(), tokens.len());
    assert_eq!(trace.len(), 1);
    assert_eq!(run.layer_out[0], trace[0], "first-token reference changed");
    let width = trace[0][0].len();
    assert!(
        width > 0
            && run.layer_out.iter().all(|position| {
                position.len() == model.config().layers()
                    && position
                        .iter()
                        .all(|row| row.len() == width && row.iter().all(|v| v.is_finite()))
            })
    );
    let actual = &run.logits[2];
    assert_eq!(actual.len(), logits.len());
    let (err, norm) = actual
        .iter()
        .zip(&logits)
        .fold((0f64, 0f64), |(e, n), (&a, &b)| {
            (e + f64::from(a - b).powi(2), n + f64::from(b).powi(2))
        });
    let relative = (err / norm).sqrt();
    assert!(
        relative.is_finite() && relative < 1e-6,
        "final logits relative L2 {relative}"
    );
    assert_eq!(
        super::host::greedy_token(actual).unwrap(),
        super::host::greedy_token(&logits).unwrap()
    );
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("FIFI_FLASH_GGUF_STATEFUL_HOST_OUT").unwrap())
        .unwrap();
    serde_json::to_writer(file, &run.layer_out).unwrap();
    eprintln!(
        "stateful host trace complete: positions={} layers={} width={} final_logits_relative_l2={relative:.9}",
        tokens.len(),
        model.config().layers(),
        width
    );
}

/// Check exact token boundaries, projection identities and routed expert sets.
fn validate_single_layer_prompt_capture(
    reader: impl std::io::BufRead,
    layer: usize,
    top_k: usize,
    experts: usize,
) -> [usize; 3] {
    use std::collections::BTreeSet;
    let tokens = [9707u32, 11, 1879];
    let mut keys = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
    let mut positions = 0usize;
    for line in reader.lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row.get("position").is_some() {
            assert!(positions < tokens.len());
            assert_eq!(
                row,
                serde_json::json!({"position":positions,"token":tokens[positions]})
            );
            if positions > 0 {
                assert!(!keys[positions - 1].is_empty());
            }
            positions += 1;
            continue;
        }
        assert!(positions > 0);
        let name = row["name"].as_str().unwrap();
        assert!(name.starts_with(&format!("blk.{layer}.")));
        let expert_value = row.get("expert").unwrap();
        assert!(expert_value.is_null() || expert_value.as_u64().is_some());
        let expert = expert_value.as_u64();
        assert!(expert.is_none_or(|id| id < experts as u64));
        assert!(row["type"].as_str().is_some());
        for field in ["input", "output"] {
            let values = row[field].as_array().unwrap();
            assert!(
                !values.is_empty()
                    && values
                        .iter()
                        .all(|v| v.as_f64().is_some_and(f64::is_finite))
            );
        }
        assert!(
            keys[positions - 1].insert((name.to_string(), expert)),
            "duplicate captured projection"
        );
    }
    assert_eq!(positions, tokens.len());
    for position in &keys {
        let selected = |part: &str| -> BTreeSet<_> {
            let name = format!("blk.{layer}.ffn_{part}_exps.weight");
            position
                .iter()
                .filter(|(n, _)| n == &name)
                .map(|(_, expert)| expert.expect("expert ID required"))
                .collect()
        };
        let gate = selected("gate");
        assert_eq!(gate.len(), top_k);
        assert_eq!(gate, selected("up"));
        assert_eq!(gate, selected("down"));
        for part in ["gate", "up", "down"] {
            assert!(position.contains(&(format!("blk.{layer}.ffn_{part}_shexp.weight"), None)));
        }
    }
    keys.map(|position| position.len())
}

#[test]
fn selected_layer_capture_rejects_corrupt_boundaries_and_experts() {
    let mut rows = Vec::new();
    for (position, token) in [9707, 11, 1879].into_iter().enumerate() {
        rows.push(serde_json::json!({"position":position,"token":token}).to_string());
        for part in ["gate", "up", "down"] {
            for expert in [1, 3] {
                rows.push(serde_json::json!({"name":format!("blk.5.ffn_{part}_exps.weight"),"expert":expert,"type":"Q80","input":[1.0],"output":[2.0]}).to_string());
            }
            rows.push(serde_json::json!({"name":format!("blk.5.ffn_{part}_shexp.weight"),"expert":null,"type":"Q80","input":[1.0],"output":[2.0]}).to_string());
        }
    }
    let validate = |lines: &[String]| {
        validate_single_layer_prompt_capture(std::io::Cursor::new(lines.join("\n")), 5, 2, 4)
    };
    assert_eq!(validate(&rows), [9, 9, 9]);
    let mut duplicate = rows.clone();
    duplicate.insert(2, duplicate[1].clone());
    let mut token = rows.clone();
    token[10] = token[10].replace("11", "12");
    let mut wrong_expert = rows.clone();
    wrong_expert[1] = wrong_expert[1].replace("\"expert\":1", "\"expert\":4");
    let mut missing = rows.clone();
    missing.remove(1);
    for broken in [duplicate, token, wrong_expert, missing] {
        assert!(std::panic::catch_unwind(|| validate(&broken)).is_err());
    }
}

/// Capture one selected layer on one uninterrupted prompt state, without changing the projector.
#[test]
#[ignore = "requires the pinned GGUF, stateful host residuals and prompt oracle"]
fn pinned_prompt_single_layer_host_capture() {
    use sha2::{Digest, Sha256};
    assert!(std::env::var_os("FIFI_FLASH_GGUF_CAPTURE_PREFIX").is_none());
    assert!(std::env::var_os("FIFI_FLASH_GGUF_CAPTURE_MARK_TOKENS").is_some());
    let layer: usize = std::env::var("FIFI_FLASH_GGUF_CAPTURE_LAYER")
        .unwrap()
        .parse()
        .unwrap();
    let path = std::env::var_os("FIFI_FLASH_GGUF_LAYER0_CAPTURE").expect("new capture path");
    let pinned = |variable: &str, hash: &str| {
        let bytes = std::fs::read(std::env::var_os(variable).expect(variable)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            hash,
            "unexpected {variable}"
        );
        bytes
    };
    let _: Vec<u8> = pinned(
        "FIFI_FLASH_GGUF",
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let expected: Vec<Vec<Vec<f32>>> = serde_json::from_slice(&pinned(
        "FIFI_FLASH_GGUF_HOST_STATEFUL_LAYERS",
        "ca258df303d22dd6e804ed362728570a12a13e1308418abdf50cc76d86268b39",
    ))
    .unwrap();
    let oracle: serde_json::Value = serde_json::from_slice(&pinned(
        "FIFI_FLASH_GGUF_HOST_ORACLE",
        "2c2133dccb289ad6f3e244d304f655189c04497bf19cd3b609779eb5af235ffa",
    ))
    .unwrap();
    let tokens = [9707u32, 11, 1879];
    assert_eq!(oracle["prompt"], serde_json::json!(tokens));
    let logits: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    let model = real_model().expect("set verified pinned GGUF");
    assert!(layer < model.config().layers());
    let host = super::host::HostModel::load(&model, model.config().layers()).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let run = super::host::run(&model, &host, &tokens, 8).unwrap();
    assert_eq!(
        run.layer_out, expected,
        "stateful host residuals changed during capture"
    );
    let actual = &run.logits[2];
    assert_eq!(actual.len(), logits.len());
    let (err, norm) = actual
        .iter()
        .zip(&logits)
        .fold((0.0f64, 0.0f64), |(e, n), (&a, &b)| {
            (e + f64::from(a - b).powi(2), n + f64::from(b).powi(2))
        });
    let relative = (err / norm).sqrt();
    assert!(
        relative.is_finite() && relative < 1e-6,
        "logit relative L2 {relative}"
    );
    assert_eq!(
        super::host::greedy_token(actual).unwrap(),
        super::host::greedy_token(&logits).unwrap()
    );
    let file = std::fs::File::open(path).unwrap();
    let moe = model.config().model_shape().unwrap().moe;
    let counts = validate_single_layer_prompt_capture(
        std::io::BufReader::new(file),
        layer,
        moe.top_k(),
        moe.experts(),
    );
    eprintln!(
        "single-layer prompt host capture: layer={layer} positions=3 counts={counts:?} residuals_exact=true final_logits_relative_l2={relative:.9}"
    );
}

/// The whole model on the host: a plain-text prompt and the five most likely next tokens, of which the first must be Paris.
/// Needs FIFI_FLASH_GGUF, FIFI_FLASH_EXL3 (for `tokenizer.json`) and FIFI_FLASH_FULL=1; it
/// decodes every layer of every token at CPU speed (minutes).
#[test]
fn real_file_host_forward_continues_a_sentence() {
    let (Some(model), Ok(exl3), Ok(_)) = (
        real_model(),
        std::env::var("FIFI_FLASH_EXL3"),
        std::env::var("FIFI_FLASH_FULL"),
    ) else {
        eprintln!("skipped: FIFI_FLASH_GGUF, FIFI_FLASH_EXL3 or FIFI_FLASH_FULL is unset");
        return;
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&exl3).join("tokenizer.json"))
            .expect("tokenizer.json");
    let prompt = "The capital of France is";
    let ids = tokenizer
        .encode(prompt, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt ids {ids:?}");
    let host = super::host::HostModel::load(&model, model.config().layers()).expect("load");
    let run = super::host::run(&model, &host, &ids, 12).expect("host forward");
    let logits = run.logits.last().expect("logits");
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<(String, f32)> = order[..5]
        .iter()
        .map(|&id| {
            (
                tokenizer.decode(&[id as u32], false).unwrap_or_default(),
                logits[id],
            )
        })
        .collect();
    eprintln!("top five after {prompt:?}: {top:?}");
    assert_eq!(top[0].0.trim(), "Paris", "top five: {top:?}");
}

/// Fixed token IDs, so the CPU and CUDA probes can compare greedy decisions without a
/// separate tokenizer or a text comparison. Unlike exploratory probes, this never skips.
#[test]
#[ignore = "requires the complete pinned GGUF and runs full-model host decoding"]
fn real_file_host_greedy_parity_baseline() {
    let model = real_model().expect("set FIFI_FLASH_GGUF to the pinned Flash-Next checkpoint");
    let host = super::host::HostModel::load(&model, model.config().layers()).expect("load");
    let prompt = [9707, 11, 1879];
    let rows = super::host::greedy_rows(&model, &host, &prompt, 3, 8).expect("host greedy trace");
    assert_eq!(rows.len(), 3);
    let width = rows[0].len();
    let tokens: Vec<_> = rows
        .iter()
        .map(|row| {
            assert_eq!(row.len(), width);
            super::host::greedy_token(row).expect("finite logits")
        })
        .collect();
    eprintln!("Flash-Next GGUF host baseline: prompt={prompt:?}, vocab={width}, greedy={tokens:?}");
    if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_ORACLE_OUT") {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new oracle receipt");
        let mut out = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut out, &serde_json::json!({"prompt": prompt, "greedy": tokens, "vocab": width, "logits": rows})).expect("write oracle receipt");
        std::io::Write::flush(&mut out).expect("flush oracle receipt");
    }
}

#[test]
fn an_unrelated_gguf_that_does_not_open_leaves_the_exl3_start_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A partial download of another model: the magic and nothing else.
    std::fs::write(
        dir.path().join("other-model-Q4_K_M.gguf"),
        b"GGUF\x03\0\0\0",
    )
    .expect("write");
    assert!(GgufModel::open(dir.path()).expect("not an error").is_none());
    // A truncated set whose first shard is not ours either.
    std::fs::write(dir.path().join("part-00002-of-00003.gguf"), b"GGUF").expect("write");
    assert!(GgufModel::open(dir.path()).expect("not an error").is_none());
}

#[test]
fn a_truncated_file_that_names_the_architecture_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("Qwen3.8-Flash-Next-Q3.gguf"),
        b"GGUF\x03\0\0\0",
    )
    .expect("write");
    let error = GgufModel::open(dir.path()).expect_err("the model is there and unusable");
    assert!(error.describe().contains("gguf"), "{}", error.describe());
    // The same by the first bytes: the architecture key leads the metadata.
    let other = tempfile::tempdir().expect("tempdir");
    let mut head = b"GGUF\x03\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0".to_vec();
    head.extend_from_slice(
        b"\x14\0\0\0\0\0\0\0general.architecture\x08\0\0\0\x08\0\0\0\0\0\0\0qwen4exp",
    );
    std::fs::write(other.path().join("model.gguf"), head).expect("write");
    assert!(GgufModel::open(other.path()).is_err());
}

#[test]
fn a_set_with_a_truncated_shard_is_an_error_named_for_the_set() {
    let written = write_flash(&Spec {
        shards: 3,
        ..Spec::default()
    });
    let last = written.dir.path().join("flash-00003-of-00003.gguf");
    let len = std::fs::metadata(&last).expect("stat").len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&last)
        .expect("open")
        .set_len(len / 2)
        .expect("truncate");
    let error = GgufModel::open(written.dir.path()).expect_err("an unusable set");
    assert!(
        error.describe().contains("extends beyond"),
        "{}",
        error.describe()
    );
}

#[test]
fn a_file_that_says_the_expert_weights_are_not_normalised_is_refused_by_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    let mut kvs = flash_kvs();
    let mut flag = super::testkit::string("qwen4exp.expert_weights_norm");
    flag.extend_from_slice(&7u32.to_le_bytes());
    flag.push(0);
    kvs.push(flag);
    let embed = super::testkit::TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![2560, 248_320],
        ty: GgufType::Q6K,
        payload: None,
    };
    write_gguf(&path, &kvs, &[embed]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains("qwen4exp.expert_weights_norm is false"),
        "{error}"
    );
}

/// The rate of the real file's expert extents: one buffered `pread` of a stack against the
/// direct parallel reads of every candidate plan, on a layer far from the ones other runs read
/// (`FIFI_FLASH_EXTENT_LAYER`, default 40), and the bytes of the best plan against the buffered
/// bytes. A direct read bypasses the page cache, so its rates are cold-cache rates by
/// construction; the buffered rate is for the pages as found.
#[test]
fn real_file_extents_read_in_parallel_with_direct_io() {
    use crate::family::expert_bank::{
        extent_read::{ExtentPlan, read_extent, span_of},
        host_tier::{SECTOR, open_direct},
    };
    use std::os::unix::fs::FileExt;
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let layer: usize = std::env::var("FIFI_FLASH_EXTENT_LAYER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(40);
    let experts = model.config().dims.experts as usize;
    let lens: Vec<u64> = model
        .weights
        .paths()
        .iter()
        .map(|path| std::fs::metadata(path).expect("shard").len())
        .collect();
    let dims = &model.config().dims;
    let index = crate::weights::GgufExpertIndex::new(
        model.weights.header(),
        &lens,
        &crate::weights::ExpertTensorNames::standard(),
        model.config().layers(),
        experts,
        (dims.hidden as usize, dims.moe_inter as usize),
        None,
    )
    .expect("index");
    let slices = index.slices(layer, 0).expect("slices");
    let part = slices.parts().next().expect("a stack");
    let bytes = ((part.file_range.end - part.file_range.start) as usize) * experts;
    let offset = part.file_range.start;
    let path = &model.weights.paths()[part.shard];
    let buffered = std::fs::File::open(path).expect("open");
    let direct = open_direct(path).expect("a drive that takes O_DIRECT");
    eprintln!(
        "layer {layer}: one stack of {} MiB at offset {offset} of {}",
        bytes >> 20,
        path.display()
    );
    let mut want = vec![0u8; bytes];
    let started = std::time::Instant::now();
    let mut done = 0usize;
    while done < bytes {
        let n = buffered
            .read_at(&mut want[done..], offset + done as u64)
            .expect("pread");
        assert!(n > 0, "end of file");
        done += n;
    }
    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "buffered pread, one thread: {:.2} GB/s ({seconds:.2} s)",
        bytes as f64 / seconds / 1e9
    );
    let (span, delta) = span_of(offset, bytes);
    let mut backing = vec![0u8; span + SECTOR as usize];
    let shift = (SECTOR as usize - backing.as_ptr() as usize % SECTOR as usize) % SECTOR as usize;
    let mut best: Option<(f64, ExtentPlan)> = None;
    for plan in ExtentPlan::CANDIDATES {
        let dst = &mut backing[shift..shift + span];
        let stats = read_extent(&direct, offset, bytes, dst, plan).expect("direct read");
        eprintln!(
            "direct {} threads x {} KiB: {:.2} GB/s ({:.2} s)",
            plan.threads,
            plan.chunk >> 10,
            stats.gbps(),
            stats.seconds
        );
        if best.is_none_or(|(rate, _)| stats.gbps() > rate) {
            best = Some((stats.gbps(), plan));
        }
        assert!(
            backing[shift + delta..shift + delta + bytes] == want[..],
            "{plan:?} read other bytes than the buffered read"
        );
    }
    let (rate, plan) = best.expect("a plan");
    eprintln!(
        "best: {} threads x {} KiB at {rate:.2} GB/s",
        plan.threads,
        plan.chunk >> 10
    );
}

#[test]
fn the_weight_map_names_each_recipe_role_and_gives_its_stored_type() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    // A DeltaNet layer holds the gated recurrence, no attention and no indexer.
    assert_eq!(
        map.layer(0, Wanted::DeltaQkv).as_deref(),
        Some("blk.0.attn_qkv.weight")
    );
    assert_eq!(map.layer(0, Wanted::QProj), None);
    assert_eq!(map.layer(0, Wanted::IndexProj), None);
    // Every fourth layer is the QSA layer, with the indexer's two matrices.
    assert_eq!(
        map.layer(3, Wanted::QProj).as_deref(),
        Some("blk.3.attn_q.weight")
    );
    assert!(map.layer(3, Wanted::IndexProj).is_some());
    assert!(map.layer(3, Wanted::IndexKey).is_some());
    assert_eq!(map.layer(3, Wanted::DeltaQkv), None);
    // The hyper-connection sites carry the norm the plain model holds as a separate role.
    assert_eq!(
        map.layer(5, Wanted::Hyper(Site::Mixer, SiteTensor::Down))
            .as_deref(),
        Some("blk.5.hc_attn_down.weight")
    );
    assert_eq!(
        map.layer(5, Wanted::Hyper(Site::Ffn, SiteTensor::Inject))
            .as_deref(),
        Some("blk.5.hc_ffn_inject.weight")
    );
    assert_eq!(map.layer(5, Wanted::AttnNorm), None);
    // Only the layer that carries the per-layer embedding step has its tensors.
    assert!(map.layer(1, Wanted::PleKey).is_some());
    assert_eq!(map.layer(2, Wanted::PleKey), None);
    // The stored type is the file's own, per layer and role.
    let expert = model
        .map()
        .get(Slot::Layer(Wanted::ExpertGate), Some(2))
        .expect("expert gate");
    assert_eq!(map.storage(2, Wanted::ExpertGate), Some(expert.ty));
    assert!(map.storage(2, Wanted::DeltaQkv).is_some());
    assert_eq!(
        map.global(Global::Embedding).as_deref(),
        Some("token_embd.weight")
    );
    assert!(map.global(Global::NgramTable).is_some());
    assert!(map.global(Global::HeadSite(SiteTensor::Norm)).is_some());
    assert_eq!(map.global(Global::FinalNorm), None);
}
