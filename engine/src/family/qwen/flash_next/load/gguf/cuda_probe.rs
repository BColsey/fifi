//! Bounded one-row projection probe: native GGUF blocks in, host-visible logits out.
//! This is a test adapter, not a serving path or a weight placement policy.

use std::{cell::Cell, ops::Range};

use super::{
    GgufModel,
    tensors::{Mapped, Slot},
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceStream,
        resident::{
            matvec_q::{MatvecCall, QuantType, ROWS_MULTIPLE},
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    },
    weights::GgufType,
};

pub(super) struct Projection<'a> {
    pub(super) ctx: &'a CudaContext,
    pub(super) stream: &'a DeviceStream,
    pub(super) launches: &'a Cell<usize>,
}

impl Projection<'_> {
    pub(super) fn supports(&self, ty: GgufType) -> bool {
        QuantType::from_gguf(ty).is_some()
    }

    pub(super) fn run(&self, ty: GgufType, blocks: &[u8], x: &[f32]) -> Result<Vec<f32>, EngError> {
        let out = project(self.ctx, self.stream, ty, blocks, x)?;
        self.launches.set(self.launches.get() + 1);
        Ok(out)
    }
}

fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: GgufType,
    blocks: &[u8],
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let bad = |message: &str| {
        EngError::BadRequest(format!("Flash-Next CUDA projection probe: {message}"))
    };
    let ty = QuantType::from_gguf(ty).ok_or_else(|| bad("no native quantized matvec"))?;
    let k = x.len();
    let row_bytes = ty
        .weight_bytes(1, k)
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("invalid input width"))?;
    if blocks.is_empty() || !blocks.len().is_multiple_of(row_bytes) {
        return Err(bad("expected whole weight rows"));
    }
    let live = blocks.len() / row_bytes;
    let rows = live.next_multiple_of(ROWS_MULTIPLE);
    let mut padded = blocks.to_vec();
    padded.resize(rows * row_bytes, 0);
    let x_host: Vec<u8> = x.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let w = ctx.allocate(padded.len())?;
    let input = ctx.allocate(x_host.len())?;
    let q8_bytes = q8_1_bytes(k, 1).ok_or_else(|| bad("invalid q8 activation size"))?;
    let q8 = ctx.allocate(q8_bytes)?;
    let y = ctx.allocate(rows * size_of::<f32>())?;
    w.write_at(stream, 0, &padded)?;
    input.write_at(stream, 0, &x_host)?;
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: input.range(0, x_host.len())?,
            y: q8.range(0, q8_bytes)?,
            k,
            ncols: 1,
        },
    )?;
    ctx.matvec_q(
        stream,
        MatvecCall {
            ty,
            w: w.range(0, padded.len())?,
            x_q8: q8.range(0, q8_bytes)?,
            y: y.range(0, rows * size_of::<f32>())?,
            k,
            rows,
            ncols: 1,
        },
    )?;
    let mut out = vec![0_u8; live * size_of::<f32>()];
    y.read_at(stream, 0, &mut out)?;
    Ok(out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_ne_bytes(bytes))
        .collect())
}

#[test]
#[ignore = "requires an admitted CUDA device; synthetic packed expert stacks"]
fn packed_expert_stacks_run_indexed_cuda_matvec() {
    use super::expert_layout::{ExpertDeviceStacks, ExpertLayout, ExpertPart, ExpertStacks};
    use crate::{family::recipe::Role, kernels::cuda::resident::matvec_expert::ExpertMatvecCall};
    let per_part = 4 * 34;
    let parts = [Role::ExpertGate, Role::ExpertUp, Role::ExpertDown].map(|role| ExpertPart {
        role,
        ty: GgufType::Q80,
        bytes: 0..0,
    });
    let mut layout = ExpertLayout {
        parts,
        bytes: 3 * per_part,
    };
    for (i, part) in layout.parts.iter_mut().enumerate() {
        part.bytes = i * per_part..(i + 1) * per_part;
    }
    let stacks = ExpertStacks::new(&layout, 2).unwrap();
    let mut packed = vec![0; stacks.bytes];
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let w = ctx.allocate(packed.len()).unwrap();
    w.write_at(&stream, 0, &packed).unwrap();
    let mut device = ExpertDeviceStacks::new(&ctx, layout.clone(), 2).unwrap();
    assert!(device.part(0).is_err());
    for slot in 0..2 {
        let mut record = vec![0; layout.bytes];
        for (part_index, part) in layout.parts.iter().enumerate() {
            for row in record[part.bytes.clone()].chunks_exact_mut(34) {
                row[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                row[2..].fill((1 + slot + part_index * 2) as u8);
            }
        }
        stacks
            .copy_record(&layout, slot, &record, &mut packed)
            .unwrap();
        stacks
            .upload_record(&layout, slot, &record, &w, &stream)
            .unwrap();
        device.upload(&stream, slot, &record).unwrap();
        if slot == 0 {
            assert!(device.part(0).is_err());
        }
    }
    assert!(device.part(0).is_ok());
    assert!(device.upload(&stream, 1, &[0]).is_err());
    assert!(device.part(0).is_err());
    let mut restored = vec![0; layout.bytes];
    for (i, part) in layout.parts.iter().enumerate() {
        for row in restored[part.bytes.clone()].chunks_exact_mut(34) {
            row[..2].copy_from_slice(&0x3800u16.to_le_bytes());
            row[2..].fill((2 + i * 2) as u8);
        }
    }
    device.upload(&stream, 1, &restored).unwrap();
    assert!(device.part(3).is_err());
    assert!(
        stacks
            .upload_record(&layout, 2, &vec![0; layout.bytes], &w, &stream)
            .is_err()
    );
    let mut copied = vec![0; packed.len()];
    w.read_at(&stream, 0, &mut copied).unwrap();
    assert_eq!(copied, packed);
    let ids = ctx.allocate(8).unwrap();
    ids.write_at(&stream, 0, &[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    let x = ctx.allocate(32 * 4).unwrap();
    let input: Vec<u8> = vec![1.0f32; 32]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    x.write_at(&stream, 0, &input).unwrap();
    let qbytes = q8_1_bytes(32, 1).unwrap();
    let q = ctx.allocate(qbytes).unwrap();
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: x.range(0, 128).unwrap(),
            y: q.range(0, qbytes).unwrap(),
            k: 32,
            ncols: 1,
        },
    )
    .unwrap();
    let y = ctx.allocate(8 * 4).unwrap();
    for (i, _) in stacks.parts.iter().enumerate() {
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty: QuantType::Q80,
                w_stack: device.part(i).unwrap(),
                ids: ids.range(0, 8).unwrap(),
                x_q8: q.range(0, qbytes).unwrap(),
                x_per_slot: false,
                y: y.range(0, 32).unwrap(),
                y_slot_stride: 4,
                k: 32,
                rows: 4,
                n_expert: 2,
                n_used: 2,
                glu_gate: None,
            },
        )
        .unwrap();
        let mut out = vec![0u8; 32];
        y.read_at(&stream, 0, &mut out).unwrap();
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        for (index, value) in got.iter().enumerate() {
            let expected = 16.0 * (1 + (if index < 4 { 1 } else { 0 }) + i * 2) as f32;
            assert!(
                (value - expected).abs() < 0.1,
                "part {i}, output {index}: {value} != {expected}"
            );
        }
    }
}

fn split_expert_fixture() -> (
    super::testkit::Written,
    GgufModel,
    super::expert_layout::ExpertLayout,
    super::expert_layout::ExpertStacks,
    Vec<u8>,
) {
    use super::{
        expert_layout::{ExpertLayout, ExpertStacks},
        testkit::{Spec, write_flash},
    };
    let mut spec = Spec {
        shards: 4,
        ..Spec::default()
    };
    spec.expert_types[0] = (GgufType::Q80, GgufType::Q80);
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let layout = ExpertLayout::for_layer(&model, 0).unwrap();
    let stacks = ExpertStacks::new(&layout, 2).unwrap();
    for (part_index, part) in layout.parts.iter().enumerate() {
        let tensor = model.map().get(part.role, Some(0)).unwrap();
        let mut payload = vec![0u8; part.bytes.len()];
        for block in payload.chunks_exact_mut(34) {
            block[..2].copy_from_slice(&0x3800u16.to_le_bytes());
            block[2..].fill((part_index + 1) as u8);
        }
        file.patch(&tensor.name, part.bytes.len() as u64, &payload);
    }
    let mut record = vec![0; layout.bytes];
    layout.read(&model, 0, 1, &mut record).unwrap();
    (file, model, layout, stacks, record)
}

#[test]
fn split_gguf_expert_fixture_builds_valid_q8_records() {
    let (_file, _model, layout, stacks, record) = split_expert_fixture();
    let mut bank = vec![0; stacks.bytes];
    stacks.copy_record(&layout, 1, &record, &mut bank).unwrap();
    for (i, part) in layout.parts.iter().enumerate() {
        let data = &bank[stacks.slot(i, 1).unwrap()];
        assert_eq!(data.len(), part.bytes.len());
        for block in data.chunks_exact(34) {
            assert_eq!(&block[..2], &0x3800u16.to_le_bytes());
            assert!(block[2..].iter().all(|v| *v == (i + 1) as u8));
        }
    }
}

#[test]
#[ignore = "requires admitted CUDA device; truncates only an owned synthetic shard"]
fn disk_bank_read_failure_poisons_policy_and_device() {
    use super::expert_layout::ExpertDeviceStacks;
    use crate::family::{
        expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru},
        recipe::Role,
    };
    let (file, model, layout, _, _) = split_expert_fixture();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1]).unwrap();
    let experts = model.config().dims.experts as usize;
    let geometry = BankGeometry::new(1, experts, 1, 2, layout.bytes).unwrap();
    let mut row = vec![-1; experts];
    row[0] = 0;
    row[1] = 1;
    let policy = ReplaceOnMissLru::from_fill(geometry, &[&row], &[vec![0; experts]], None).unwrap();
    let mut state = ExpertBankState::with_policy(geometry, HostMode::Disk, policy).unwrap();
    let tensor = model.map().get(Role::ExpertGate, Some(0)).unwrap();
    let weights = file.open();
    let path = weights.paths()[tensor.shard].canonicalize().unwrap();
    let dir = file.dir.path().canonicalize().unwrap();
    assert_eq!(path.parent(), Some(dir.as_path()));
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(0)
        .unwrap();
    let error = bank
        .step_disk(&stream, &model, &mut state, 0, &[256])
        .unwrap_err();
    eprintln!(
        "expected truncated-shard read failure: {}",
        error.describe()
    );
    assert!(
        state.slot_row(0)[256] >= 0,
        "policy advanced before the read"
    );
    assert!(state.is_poisoned());
    assert!(state.begin_pass(0, &[256]).is_err());
    assert!(bank.part(0).is_err());
    assert!(bank.projection(&state, Role::ExpertGate).is_err());
    assert!(
        bank.step_disk(&stream, &model, &mut state, 0, &[0])
            .is_err()
    );
}

#[test]
#[ignore = "requires admitted CUDA device; synthetic split GGUF full expert"]
fn split_gguf_expert_upload_and_indexed_matvec() {
    use super::expert_layout::ExpertDeviceStacks;
    use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
    let (_file, model, layout, _stacks, _record) = split_expert_fixture();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    assert!(ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[]).is_err());
    assert!(ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[1, 1]).is_err());
    assert!(
        ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[model.config().dims.experts])
            .is_err()
    );
    let bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1]).unwrap();
    let ids = ctx.allocate(4).unwrap();
    ids.write_at(&stream, 0, &1i32.to_ne_bytes()).unwrap();
    for (i, part) in layout.parts.iter().enumerate() {
        let tensor = model.map().get(part.role, Some(0)).unwrap();
        let k = tensor.dims[0] as usize;
        let rows = tensor.dims[1] as usize;
        let input: Vec<u8> = vec![1.0f32; k]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let x = ctx.allocate(input.len()).unwrap();
        x.write_at(&stream, 0, &input).unwrap();
        let qbytes = q8_1_bytes(k, 1).unwrap();
        let q = ctx.allocate(qbytes).unwrap();
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x.range(0, input.len()).unwrap(),
                y: q.range(0, qbytes).unwrap(),
                k,
                ncols: 1,
            },
        )
        .unwrap();
        let y = ctx.allocate(rows * 4).unwrap();
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty: QuantType::Q80,
                w_stack: bank.part(i).unwrap(),
                ids: ids.range(0, 4).unwrap(),
                x_q8: q.range(0, qbytes).unwrap(),
                x_per_slot: false,
                y: y.range(0, rows * 4).unwrap(),
                y_slot_stride: rows,
                k,
                rows,
                n_expert: 2,
                n_used: 1,
                glu_gate: None,
            },
        )
        .unwrap();
        let mut out = vec![0; rows * 4];
        y.read_at(&stream, 0, &mut out).unwrap();
        let expected = k as f32 * 0.5 * (i + 1) as f32;
        for chunk in out.chunks_exact(4) {
            let got = f32::from_ne_bytes(chunk.try_into().unwrap());
            assert!(
                (got - expected).abs() <= expected * 0.005,
                "part {i}: {got} != {expected}"
            );
        }
    }
}

/// Read an output row range or one expert slice from the checkpoint's mapped tensor.
fn mapped_blocks(
    model: &GgufModel,
    tensor: &Mapped,
    expert: Option<u64>,
    rows: Range<u64>,
) -> Result<Vec<u8>, EngError> {
    let (begin, bytes) = tensor.row_span(expert, rows.clone())?;
    let mut blocks = vec![0; bytes];
    if let Some(expert) = expert {
        let (Slot::Layer(role), Some(layer)) = (tensor.role, tensor.layer) else {
            return Err(EngError::BadRequest(
                "expert tensor lacks layer role".into(),
            ));
        };
        model.read_expert_rows(layer, role, expert, rows, &mut blocks)?;
    } else {
        model.weights.read_range(&tensor.name, begin, &mut blocks)?;
    }
    Ok(blocks)
}

#[test]
fn reads_only_the_mapped_rows_from_the_expected_split_shard() {
    use super::testkit::{Spec, write_flash};
    let file = write_flash(&Spec {
        shards: 4,
        ..Spec::default()
    });
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let tensor = model
        .map
        .entries()
        .iter()
        .find(|t| t.ty == GgufType::Q80 && t.dims.len() == 2)
        .expect("mapped Q8_0 matrix");
    assert!(tensor.shard > 0, "the first shard holds only metadata");
    let stride = tensor.dims[0] / 32 * 34;
    let wanted = vec![0xb7; 3 * stride as usize];
    file.patch(&tensor.name, 2 * stride, &wanted);
    assert_eq!(mapped_blocks(&model, tensor, None, 2..5).unwrap(), wanted);
    assert_eq!(
        mapped_blocks(&model, tensor, None, 1..2).unwrap(),
        vec![0; stride as usize]
    );
    assert!(mapped_blocks(&model, tensor, Some(0), 2..5).is_err());
    let end = tensor.dims[1];
    assert!(mapped_blocks(&model, tensor, None, end..end + 1).is_err());

    let stack = model
        .map
        .entries()
        .iter()
        .find(|t| t.dims.len() == 3 && t.dims[2] > 1)
        .expect("mapped expert stack");
    let stride = stack.dims[0] / stack.ty.block_elems() * stack.ty.block_bytes();
    let wanted = vec![0x6d; stride as usize];
    file.patch(&stack.name, (stack.dims[1] + 2) * stride, &wanted);
    assert_eq!(mapped_blocks(&model, stack, Some(1), 2..3).unwrap(), wanted);
    assert_eq!(
        mapped_blocks(&model, stack, Some(0), 2..3).unwrap(),
        vec![0; stride as usize]
    );
    assert!(mapped_blocks(&model, stack, Some(stack.dims[2]), 2..3).is_err());
}

fn projection_samples(model: &GgufModel) -> impl Iterator<Item = (&Mapped, Option<u64>)> {
    model.map.entries().iter().filter_map(|t| {
        (t.layer == Some(0)
            && QuantType::from_gguf(t.ty).is_some()
            && matches!(t.dims.len(), 2 | 3)
            && t.dims[1] >= 4)
            .then(|| (t, t.dims.get(2).map(|n| n / 2)))
    })
}

fn projection_model() -> GgufModel {
    let path = std::env::var_os("FIFI_FLASH_GGUF_PROJECTION_FIXTURE")
        .or_else(|| std::env::var_os("FIFI_FLASH_GGUF"))
        .expect("set FIFI_FLASH_GGUF or the projection-only fixture path");
    GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .expect("Flash-Next GGUF")
}

#[test]
#[ignore = "needs complete expert samples and a sparse negative fixture"]
fn full_expert_fixture_integrity() {
    use super::expert_layout::ExpertLayout;
    use sha2::{Digest, Sha256};
    let hashes = [
        "3da6977d2c06568b213e1e79fd49f3de3816c594c7b6f0928cbc4f74eed6d13b",
        "923942e4b406fc5602397c2444bc2fbb961e1e79d64ddb2469a3e0da99249e32",
        "f3564536713b47dd71045f35e7129099fc843b239628f2a053f6b2e9f1af8a61",
    ];
    for (key, expected_match) in [
        ("FIFI_FLASH_EXPERT_FIXTURE", true),
        ("FIFI_FLASH_GGUF_PROJECTION_FIXTURE", false),
    ] {
        let path = std::env::var_os(key).unwrap();
        let model = GgufModel::open(std::path::Path::new(&path))
            .unwrap()
            .unwrap();
        let layout = ExpertLayout::for_layer(&model, 0).unwrap();
        let mut record = vec![0; layout.bytes];
        layout.read(&model, 0, 256, &mut record).unwrap();
        let valid = layout.parts.iter().zip(hashes).all(|(part, hash)| {
            format!("{:x}", Sha256::digest(&record[part.bytes.clone()])) == hash
        });
        assert_eq!(valid, expected_match, "{key}");
    }
}

#[test]
#[ignore = "needs two-expert sparse fixture and single-expert negative control"]
fn second_expert_fixture_integrity() {
    use super::expert_layout::ExpertLayout;
    use sha2::{Digest, Sha256};
    let hashes = [
        "ac0f9bfdd7ec13f0d982a235686c747b7e31737d170ed1d7a54faa71f424cd6a",
        "3730d2e7670e0c512ccbbd9c5d21e15badb7c21096f2c336222a8915e1dede6f",
        "113aa44f068b315a90dd7ece5b35e836bf595af188b345af3110f073fac1fb19",
    ];
    for (key, expected) in [
        ("FIFI_FLASH_TWO_EXPERT_FIXTURE", true),
        ("FIFI_FLASH_EXPERT_FIXTURE", false),
    ] {
        let path = std::env::var_os(key).unwrap();
        let model = GgufModel::open(std::path::Path::new(&path))
            .unwrap()
            .unwrap();
        let layout = ExpertLayout::for_layer(&model, 0).unwrap();
        let mut record = vec![0; layout.bytes];
        layout.read(&model, 0, 257, &mut record).unwrap();
        let valid = layout.parts.iter().zip(hashes).all(|(part, hash)| {
            format!("{:x}", Sha256::digest(&record[part.bytes.clone()])) == hash
        });
        assert_eq!(valid, expected, "{key}");
    }
}

#[test]
#[ignore = "emits a complete expert sample plan from pinned GGUF headers"]
fn write_full_expert_sample_plan() {
    let model = projection_model();
    let requested: u64 = std::env::var("FIFI_FLASH_EXPERT_ID")
        .unwrap_or_else(|_| "256".into())
        .parse()
        .unwrap();
    let samples: Vec<_> = projection_samples(&model).filter_map(|(tensor, expert)| {
        expert?;
        let expert = requested;
        let (begin, bytes) = tensor.row_span(Some(expert), 0..tensor.dims[1]).unwrap();
        let stored = model.weights.header().tensor(&tensor.name).unwrap();
        Some(serde_json::json!({"name": tensor.name, "shard": stored.shard(),
            "start": stored.abs_offset().checked_add(begin).unwrap(), "bytes": bytes, "expert": expert}))
    }).collect();
    assert_eq!(samples.len(), 3);
    let path = std::env::var_os("FIFI_FLASH_EXPERT_PLAN").expect("set expert plan output path");
    std::fs::write(path, serde_json::to_vec_pretty(&samples).unwrap()).unwrap();
}

#[test]
#[ignore = "emits a sample plan from a complete or header-only split set; no CUDA device needed"]
fn write_mapped_projection_sample_plan() {
    let model = projection_model();
    let samples: Vec<_> = projection_samples(&model)
        .map(|(tensor, expert)| {
            let (begin, bytes) = tensor.row_span(expert, 1..4).unwrap();
            let stored = model.weights.header().tensor(&tensor.name).unwrap();
            serde_json::json!({
                "name": tensor.name,
                "shard": stored.shard(),
                "start": stored.abs_offset().checked_add(begin).unwrap(),
                "bytes": bytes,
            })
        })
        .collect();
    assert!(!samples.is_empty());
    let path = std::env::var_os("FIFI_FLASH_GGUF_SAMPLE_PLAN").expect("set plan output path");
    std::fs::write(path, serde_json::to_vec_pretty(&samples).unwrap()).expect("write plan");
}

#[test]
#[ignore = "needs pinned Flash-Next projection samples and an admitted CUDA device"]
fn mapped_checkpoint_projections_match_host_dots() {
    use crate::weights::dequantize;
    let model = projection_model();
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let mut checked = 0;
    let mut expert_checked = 0;
    for (t, expert) in projection_samples(&model) {
        let x: Vec<_> = (0..t.dims[0])
            .map(|n| if n % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        let blocks = mapped_blocks(&model, t, expert, 1..4).expect("mapped rows");
        assert!(
            blocks.iter().any(|&byte| byte != 0),
            "missing sample for {}",
            t.name
        );
        let weights = dequantize(t.ty, &blocks).expect("host weights");
        let actual = project(&ctx, &stream, t.ty, &blocks, &x).expect("native projection");
        assert_eq!(actual.len(), 3);
        for (actual, row) in actual.into_iter().zip(weights.chunks_exact(x.len())) {
            let expected: f32 = row.iter().zip(&x).map(|(w, x)| w * x).sum();
            let scale: f32 = row.iter().map(|w| w.abs()).sum();
            assert!(
                (actual - expected).abs() <= 0.003 * scale + 1e-5,
                "{}: native {actual}, host {expected}",
                t.name
            );
        }
        checked += 1;
        expert_checked += usize::from(expert.is_some());
    }
    assert!(checked > 0, "no native mapped projections were checked");
    assert!(
        expert_checked > 0,
        "no expert reader projections were checked"
    );
    eprintln!(
        "Flash-Next layer 0: checked {checked} mapped projections, {expert_checked} expert stacks"
    );
}

// Range hashes recorded from the pinned revision, not independent whole-shard hashes.
fn verify_pinned_expert_sample(name: &str, bytes: &[u8]) -> bool {
    use sha2::{Digest, Sha256};
    let expected = match name {
        "blk.0.ffn_gate_exps.weight" => {
            "37e0e5454cce1f22a7ef56e2e6eaa48eb7d036e790e39a7b8db52de738080b2e"
        }
        "blk.0.ffn_up_exps.weight" => {
            "cd93f9529b635f30a0741d883938fe910c34f5bb6030186d8a957d78b4eda73d"
        }
        "blk.0.ffn_down_exps.weight" => {
            "ac3b3689a319b6651372c430cab53f1d328a5b928263ec5248da2c097075e3d2"
        }
        _ => return false,
    };
    format!("{:x}", Sha256::digest(bytes)) == expected
}

#[test]
fn pinned_expert_sample_rejects_missing_and_unknown_data() {
    assert!(!verify_pinned_expert_sample(
        "blk.0.ffn_gate_exps.weight",
        &[]
    ));
    assert!(!verify_pinned_expert_sample(
        "blk.0.ffn_gate_exps.weight",
        &[0; 2460]
    ));
    assert!(!verify_pinned_expert_sample("unknown", &[1]));
}

#[test]
#[ignore = "needs pinned expert row samples and an admitted CUDA device"]
fn mapped_expert_rows_match_indexed_cuda_matvec() {
    use crate::{kernels::cuda::resident::matvec_expert::ExpertMatvecCall, weights::dequantize};
    let model = projection_model();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut checked = 0;
    for (tensor, expert) in projection_samples(&model).filter(|(_, e)| e.is_some()) {
        let x: Vec<f32> = (0..tensor.dims[0])
            .map(|n| if n % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        let bytes = mapped_blocks(&model, tensor, expert, 1..4).unwrap();
        assert!(
            verify_pinned_expert_sample(&tensor.name, &bytes),
            "pinned sample hash mismatch: {}",
            tensor.name
        );
        let weights = dequantize(tensor.ty, &bytes).unwrap();
        let stride = bytes.len() / 3;
        let mut padded = bytes.clone();
        padded.resize(4 * stride, 0);
        let w = ctx.allocate(padded.len()).unwrap();
        w.write_at(&stream, 0, &padded).unwrap();
        let input: Vec<u8> = x.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let xd = ctx.allocate(input.len()).unwrap();
        xd.write_at(&stream, 0, &input).unwrap();
        let qbytes = q8_1_bytes(x.len(), 1).unwrap();
        let q = ctx.allocate(qbytes).unwrap();
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: xd.range(0, input.len()).unwrap(),
                y: q.range(0, qbytes).unwrap(),
                k: x.len(),
                ncols: 1,
            },
        )
        .unwrap();
        let ids = ctx.allocate(4).unwrap();
        ids.write_at(&stream, 0, &0i32.to_ne_bytes()).unwrap();
        let y = ctx.allocate(16).unwrap();
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty: QuantType::from_gguf(tensor.ty).unwrap(),
                w_stack: w.range(0, padded.len()).unwrap(),
                ids: ids.range(0, 4).unwrap(),
                x_q8: q.range(0, qbytes).unwrap(),
                x_per_slot: false,
                y: y.range(0, 16).unwrap(),
                y_slot_stride: 4,
                k: x.len(),
                rows: 4,
                n_expert: 1,
                n_used: 1,
                glu_gate: None,
            },
        )
        .unwrap();
        let mut out = [0u8; 16];
        y.read_at(&stream, 0, &mut out).unwrap();
        for (chunk, row) in out
            .chunks_exact(4)
            .take(3)
            .zip(weights.chunks_exact(x.len()))
        {
            let got = f32::from_ne_bytes(chunk.try_into().unwrap());
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            let scale: f32 = row.iter().map(|v| v.abs()).sum();
            assert!(
                (got - want).abs() <= 0.003 * scale + 1e-5,
                "{} got {got}, expected {want}",
                tensor.name
            );
        }
        eprintln!("checked {} {:?}", tensor.name, tensor.ty);
        checked += 1;
    }
    assert_eq!(checked, 3, "expected gate, up and down expert stacks");
}

#[test]
#[ignore = "needs verified full expert samples and an admitted CUDA device"]
fn full_expert_device_stacks_match_host() {
    use super::expert_layout::{ExpertDeviceStacks, ExpertProjectionInputs};
    use crate::weights::dequantize;
    full_expert_fixture_integrity();
    let path = std::env::var_os("FIFI_FLASH_EXPERT_FIXTURE").unwrap();
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    use super::expert_layout::ExpertLayout;
    use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
    let experts = model.config().dims.experts as usize;
    let geometry = BankGeometry::new(
        1,
        experts,
        1,
        2,
        ExpertLayout::for_layer(&model, 0).unwrap().bytes,
    )
    .unwrap();
    let mut row = vec![-1; experts];
    row[0] = 0;
    row[1] = 1;
    let state = || {
        let policy =
            ReplaceOnMissLru::from_fill(geometry, &[&row], &[vec![0; experts]], None).unwrap();
        ExpertBankState::with_policy(geometry, HostMode::Disk, policy).unwrap()
    };
    let mut bad_state = state();
    let mut short_bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0]).unwrap();
    assert!(
        short_bank
            .step_disk(&stream, &model, &mut bad_state, 0, &[256])
            .is_err()
    );
    assert!(!bad_state.is_poisoned());
    assert_eq!(bad_state.slot_row(0)[256], -1);
    bad_state.poison();
    assert!(
        short_bank
            .step_disk(&stream, &model, &mut bad_state, 0, &[0])
            .is_err()
    );
    assert!(short_bank.part(0).is_err());
    assert!(short_bank.upload(&stream, 0, &[]).is_err());
    assert!(
        short_bank
            .step_disk(&stream, &model, &mut state(), 0, &[0])
            .is_err()
    );
    let mut swapped = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[1, 0]).unwrap();
    let mut mismatch = state();
    assert!(
        swapped
            .step_disk(&stream, &model, &mut mismatch, 0, &[0])
            .is_err()
    );
    assert!(!mismatch.is_poisoned());
    assert!(
        swapped
            .projection(&mismatch, crate::family::recipe::Role::ExpertGate)
            .is_err()
    );
    swapped
        .replace_from_model(&stream, &model, 0, 0, 256)
        .unwrap();
    assert!(swapped.part(0).is_ok());
    assert!(
        swapped
            .projection(&mismatch, crate::family::recipe::Role::ExpertGate)
            .is_err()
    );
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1]).unwrap();
    let mut state = state();
    let other = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    assert!(
        bank.step_disk(&stream, &other, &mut state, 0, &[0])
            .is_err()
    );
    assert!(
        bank.step_disk(&stream, &model, &mut state, 1, &[0])
            .is_err()
    );
    assert!(!state.is_poisoned());
    assert_eq!(state.slot_row(0)[0], 0);
    let copies = bank
        .step_disk(&stream, &model, &mut state, 0, &[256])
        .unwrap();
    assert_eq!(copies.ops.len(), 1);
    let slot = copies.ops[0].slot;
    assert_eq!(state.slot_row(0)[256], i32::from(slot));
    assert!(
        bank.step_disk(&stream, &model, &mut state, 0, &[256])
            .unwrap()
            .ops
            .is_empty()
    );
    let ids = ctx.allocate(4).unwrap();
    ids.write_at(&stream, 0, &i32::from(slot).to_ne_bytes())
        .unwrap();
    assert!(
        bank.projection(&state, crate::family::recipe::Role::Router)
            .is_err()
    );
    let mut checked = 0;
    for (tensor, expert) in projection_samples(&model).filter(|(_, expert)| expert.is_some()) {
        let Slot::Layer(role) = tensor.role else {
            panic!("expert role")
        };
        let projection = bank.projection(&state, role).unwrap();
        let k = projection.k;
        let rows = projection.rows;
        assert_eq!(
            (k, rows, projection.slots),
            (tensor.dims[0] as usize, tensor.dims[1] as usize, 2)
        );
        assert_eq!(rows % 4, 0);
        let input: Vec<f32> = (0..k).map(|n| if n % 5 < 2 { -1.0 } else { 1.0 }).collect();
        let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let x = ctx.allocate(bytes.len()).unwrap();
        x.write_at(&stream, 0, &bytes).unwrap();
        let qbytes = q8_1_bytes(k, 1).unwrap();
        let q = ctx.allocate(qbytes).unwrap();
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x.range(0, bytes.len()).unwrap(),
                y: q.range(0, qbytes).unwrap(),
                k,
                ncols: 1,
            },
        )
        .unwrap();
        let y = ctx.allocate(rows * 4).unwrap();
        projection
            .launch(
                &ctx,
                &stream,
                ExpertProjectionInputs {
                    ids: ids.range(0, 4).unwrap(),
                    x_q8: q.range(0, qbytes).unwrap(),
                    x_per_slot: false,
                    output: y.range(0, rows * 4).unwrap(),
                    output_stride: rows,
                    used: 1,
                },
            )
            .unwrap();
        let mut out = vec![0; rows * 4];
        y.read_at(&stream, 0, &mut out).unwrap();
        let blocks = mapped_blocks(&model, tensor, expert, 0..rows as u64).unwrap();
        let weights = dequantize(tensor.ty, &blocks).unwrap();
        for (index, (chunk, row)) in out.chunks_exact(4).zip(weights.chunks_exact(k)).enumerate() {
            let got = f32::from_ne_bytes(chunk.try_into().unwrap());
            let want: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
            let scale: f32 = row.iter().map(|v| v.abs()).sum();
            assert!(
                (got - want).abs() <= 0.003 * scale + 1e-5,
                "{} row {index}: {got} != {want}",
                tensor.name
            );
        }
        if role == crate::family::recipe::Role::ExpertDown {
            let two_inputs: Vec<u8> = input
                .iter()
                .chain(input.iter())
                .enumerate()
                .flat_map(|(i, &v)| (if i < k { v } else { -v }).to_ne_bytes())
                .collect();
            let x2 = ctx.allocate(two_inputs.len()).unwrap();
            x2.write_at(&stream, 0, &two_inputs).unwrap();
            let q2bytes = q8_1_bytes(k, 2).unwrap();
            let q2 = ctx.allocate(q2bytes).unwrap();
            ctx.quantize_q8_1(
                &stream,
                QuantizeCall {
                    x: x2.range(0, two_inputs.len()).unwrap(),
                    y: q2.range(0, q2bytes).unwrap(),
                    k,
                    ncols: 2,
                },
            )
            .unwrap();
            let ids2 = ctx.allocate(8).unwrap();
            let slot_bytes = i32::from(slot).to_ne_bytes();
            ids2.write_at(&stream, 0, &[slot_bytes, slot_bytes].concat())
                .unwrap();
            let y2 = ctx.allocate(rows * 8).unwrap();
            bank.projection(&state, role)
                .unwrap()
                .launch(
                    &ctx,
                    &stream,
                    ExpertProjectionInputs {
                        ids: ids2.range(0, 8).unwrap(),
                        x_q8: q2.range(0, q2bytes).unwrap(),
                        x_per_slot: true,
                        output: y2.range(0, rows * 8).unwrap(),
                        output_stride: rows,
                        used: 2,
                    },
                )
                .unwrap();
            let mut both = vec![0u8; rows * 8];
            y2.read_at(&stream, 0, &mut both).unwrap();
            for (index, row) in weights.chunks_exact(k).enumerate() {
                let expected: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
                let tolerance = 0.003 * row.iter().map(|v| v.abs()).sum::<f32>() + 1e-5;
                for (slot_index, sign) in [1.0, -1.0].into_iter().enumerate() {
                    let offset = (slot_index * rows + index) * 4;
                    let got = f32::from_ne_bytes(both[offset..offset + 4].try_into().unwrap());
                    assert!(
                        (got - sign * expected).abs() <= tolerance,
                        "down slot {slot_index} row {index}"
                    );
                }
            }
            eprintln!("checked {rows} down rows with two distinct slot activations");
        }
        eprintln!("checked {} {:?} {rows} rows", tensor.name, tensor.ty);
        checked += 1;
    }
    assert_eq!(checked, 3);
    state.poison();
    assert!(
        bank.projection(&state, crate::family::recipe::Role::ExpertGate)
            .is_err()
    );
}

#[test]
#[ignore = "needs two verified expert ranges and an admitted CUDA device"]
fn two_real_expert_slots_match_host() {
    use super::expert_layout::{ExpertDeviceStacks, ExpertLayout, ExpertProjectionInputs};
    use crate::{
        family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru},
        weights::dequantize,
    };
    use sha2::{Digest, Sha256};
    let path = std::env::var_os("FIFI_FLASH_TWO_EXPERT_FIXTURE").unwrap();
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let layout = ExpertLayout::for_layer(&model, 0).unwrap();
    let hashes = [
        [
            "3da6977d2c06568b213e1e79fd49f3de3816c594c7b6f0928cbc4f74eed6d13b",
            "923942e4b406fc5602397c2444bc2fbb961e1e79d64ddb2469a3e0da99249e32",
            "f3564536713b47dd71045f35e7129099fc843b239628f2a053f6b2e9f1af8a61",
        ],
        [
            "ac0f9bfdd7ec13f0d982a235686c747b7e31737d170ed1d7a54faa71f424cd6a",
            "3730d2e7670e0c512ccbbd9c5d21e15badb7c21096f2c336222a8915e1dede6f",
            "113aa44f068b315a90dd7ece5b35e836bf595af188b345af3110f073fac1fb19",
        ],
    ];
    for (id, expected) in [256u64, 257].into_iter().zip(hashes) {
        let mut record = vec![0; layout.bytes];
        layout.read(&model, 0, id, &mut record).unwrap();
        for (part, hash) in layout.parts.iter().zip(expected) {
            assert_eq!(
                format!("{:x}", Sha256::digest(&record[part.bytes.clone()])),
                hash
            );
        }
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // Initial zero-valued synthetic holes are not dispatched; only the two verified experts are.
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1, 256]).unwrap();
    let experts = model.config().dims.experts as usize;
    let geometry = BankGeometry::new(1, experts, 2, 3, layout.bytes).unwrap();
    let mut mapping = vec![-1; experts];
    mapping[0] = 0;
    mapping[1] = 1;
    mapping[256] = 2;
    let policy =
        ReplaceOnMissLru::from_fill(geometry, &[&mapping], &[vec![0; experts]], None).unwrap();
    let mut state = ExpertBankState::with_policy(geometry, HostMode::Disk, policy).unwrap();
    let copies = bank
        .step_disk(&stream, &model, &mut state, 0, &[257, 256])
        .unwrap();
    assert_eq!(
        (copies.hits, copies.ops.len(), copies.ops[0].incoming),
        (1, 1, 257)
    );
    let slots = [state.slot_row(0)[257], state.slot_row(0)[256]];
    assert!(slots.iter().all(|&slot| slot >= 0) && slots[0] != slots[1]);
    let ids = ctx.allocate(8).unwrap();
    ids.write_at(
        &stream,
        0,
        &[slots[0].to_ne_bytes(), slots[1].to_ne_bytes()].concat(),
    )
    .unwrap();
    for part in &layout.parts {
        let tensor = model.map().get(part.role, Some(0)).unwrap();
        let projection = bank.projection(&state, part.role).unwrap();
        let (k, rows) = (projection.k, projection.rows);
        let input: Vec<f32> = (0..k).map(|i| if i % 5 < 2 { -1.0 } else { 1.0 }).collect();
        let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let x = ctx.allocate(bytes.len()).unwrap();
        x.write_at(&stream, 0, &bytes).unwrap();
        let qbytes = q8_1_bytes(k, 1).unwrap();
        let q = ctx.allocate(qbytes).unwrap();
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x.range(0, bytes.len()).unwrap(),
                y: q.range(0, qbytes).unwrap(),
                k,
                ncols: 1,
            },
        )
        .unwrap();
        let y = ctx.allocate(rows * 8).unwrap();
        projection
            .launch(
                &ctx,
                &stream,
                ExpertProjectionInputs {
                    ids: ids.range(0, 8).unwrap(),
                    x_q8: q.range(0, qbytes).unwrap(),
                    x_per_slot: false,
                    output: y.range(0, rows * 8).unwrap(),
                    output_stride: rows,
                    used: 2,
                },
            )
            .unwrap();
        let mut output = vec![0; rows * 8];
        y.read_at(&stream, 0, &mut output).unwrap();
        for (rank, expert) in [257u64, 256].into_iter().enumerate() {
            let blocks = mapped_blocks(&model, tensor, Some(expert), 0..rows as u64).unwrap();
            let weights = dequantize(tensor.ty, &blocks).unwrap();
            for (index, row) in weights.chunks_exact(k).enumerate() {
                let off = (rank * rows + index) * 4;
                let got = f32::from_ne_bytes(output[off..off + 4].try_into().unwrap());
                let want: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
                let tolerance = 0.003 * row.iter().map(|v| v.abs()).sum::<f32>() + 1e-5;
                assert!(
                    (got - want).abs() <= tolerance,
                    "{:?} expert {expert} row {index}",
                    part.role
                );
            }
        }
        eprintln!(
            "checked {:?} {rows} rows for two distinct experts",
            part.role
        );
    }
}

/// The token loop and unquantized operators stay on the host; this is not a serving test.
#[test]
#[ignore = "requires the complete pinned GGUF and an admitted CUDA device"]
fn checkpoint_greedy_rows_with_cuda_projections() {
    use super::host::{HostModel, HostProjector, greedy_rows, greedy_rows_with, greedy_token};
    let path =
        std::env::var_os("FIFI_FLASH_GGUF").expect("set FIFI_FLASH_GGUF to the verified pack");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .expect("Flash-Next GGUF");
    let host = HostModel::load(&model, model.config().layers()).expect("host model");
    let prompt = [9707, 11, 1879];
    let expected = greedy_rows(&model, &host, &prompt, 3, 8).expect("host greedy rows");
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let ngram = model.ngram().expect("n-gram table");
    let launches = Cell::new(0);
    let projector = HostProjector::new(&model.weights, &ngram, 8).with_cuda(Projection {
        ctx: &ctx,
        stream: &stream,
        launches: &launches,
    });
    let actual = greedy_rows_with(projector, &host, &prompt, 3).expect("hybrid greedy rows");
    assert!(launches.get() > 0, "no CUDA projections ran");
    assert_eq!(actual.len(), expected.len());
    for (position, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(got.len(), want.len());
        assert_eq!(
            greedy_token(got).unwrap(),
            greedy_token(want).unwrap(),
            "decision {position}"
        );
        let (err, norm) = got
            .iter()
            .zip(want)
            .fold((0.0, 0.0), |(err, norm), (&a, &b)| {
                let (a, b) = (f64::from(a), f64::from(b));
                (err + (a - b).powi(2), norm + b.powi(2))
            });
        let relative_l2 = (err / norm).sqrt();
        eprintln!(
            "hybrid decision {position}: relative L2={relative_l2}, CUDA projections={}",
            launches.get()
        );
        // Provisional bring-up threshold, not a model quality criterion.
        assert!(
            relative_l2 <= 0.05,
            "decision {position}: relative L2 {relative_l2}"
        );
    }
}

#[test]
#[ignore = "needs an admitted native CUDA device; no checkpoint required"]
fn a_three_row_gguf_projection_matches_the_host_dot_products() {
    use crate::weights::dequantize;
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let x = [1.0; 32];
    let mut blocks = Vec::new();
    for code in [1_i8, -2, 3] {
        blocks.extend(0x3800_u16.to_le_bytes()); // Q8_0 scale 0.5.
        blocks.extend([code as u8; 32]);
    }
    let weights = dequantize(GgufType::Q80, &blocks).expect("host weights");
    let expected: Vec<f32> = weights
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect();
    let got = project(&ctx, &stream, GgufType::Q80, &blocks, &x).expect("native projection");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.into_iter().zip(expected) {
        assert!((a - b).abs() <= b.abs() * 0.005, "native {a}, host {b}");
    }
}
