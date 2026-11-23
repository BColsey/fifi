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

    pub(super) fn run_f16(
        &self,
        ty: GgufType,
        blocks: &[u8],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        use crate::kernels::cuda::prefill::gemm_dq::GemmDqTypedCall;
        let bad = || {
            EngError::BadRequest(
                "Flash-Next FP16 diagnostic projection: invalid weight rows".into(),
            )
        };
        let ty = QuantType::from_gguf(ty).ok_or_else(bad)?;
        let row_bytes = ty
            .weight_bytes(1, x.len())
            .filter(|&n| n > 0)
            .ok_or_else(bad)?;
        if blocks.is_empty() || !blocks.len().is_multiple_of(row_bytes) {
            return Err(bad());
        }
        let rows = blocks.len() / row_bytes;
        let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let w = self.ctx.allocate(blocks.len())?;
        let input = self.ctx.allocate(xbytes.len())?;
        let output = self.ctx.allocate(rows * 4)?;
        w.write_at(self.stream, 0, blocks)?;
        input.write_at(self.stream, 0, &xbytes)?;
        let call = GemmDqTypedCall {
            w: w.range(0, blocks.len())?,
            x: input.range(0, xbytes.len())?,
            y: output.range(0, rows * 4)?,
            k: x.len(),
            rows,
            ncols: 1,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: true,
        };
        if matches!(ty, QuantType::Q80 | QuantType::Q6K) {
            self.ctx.gemm_dq_codebook(self.stream, ty, call)?;
        } else {
            self.ctx.gemm_dq(self.stream, ty, call)?;
        }
        let mut bytes = vec![0_u8; rows * 4];
        output.read_at(self.stream, 0, &mut bytes)?;
        self.launches.set(self.launches.get() + 1);
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
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
#[ignore = "requires admitted CUDA device; synthetic split GGUF"]
fn persistent_routed_synthetic_hits_eviction_and_poison() {
    use super::routed::PersistentRouted;
    use crate::family::{
        exl3::{
            launch::{MoeDims, RouteBufs},
            moe::{Exl3MoeBufs, Scratch},
        },
        recipe::Role,
    };
    let (file, model, layout, _, _) = split_expert_fixture();
    let shape = model.config().model_shape().unwrap();
    let dims = MoeDims {
        rows: 1,
        hidden: shape.hyper.hidden(),
        n_expert: shape.moe.experts(),
        top_k: shape.moe.top_k(),
        inter: shape.moe.inter(),
        shared_inter: shape.moe.shared_inter(),
    };
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let zeros = |n| ctx.upload(&vec![0; n]).unwrap();
    let resid = zeros(dims.hidden * 4);
    let x = zeros(dims.hidden * 4);
    let out = zeros(dims.hidden * 4);
    let ids = zeros(dims.top_k * 4);
    let weights = ctx
        .upload(
            &(0..dims.top_k)
                .flat_map(|_| (1.0f32 / dims.top_k as f32).to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let shared_gate = ctx.upload(&0.5f32.to_ne_bytes()).unwrap();
    let shared = ctx
        .upload(
            &(0..dims.hidden)
                .flat_map(|i| ((i % 11) as f32 * 0.125).to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let gu = zeros(dims.top_k * 2 * dims.inter * 4);
    let act = zeros(dims.top_k * dims.inter * 4);
    let down = zeros(dims.top_k * dims.hidden * 4);
    let dummy = zeros(4);
    fn range(b: &crate::kernels::cuda::DeviceBuffer) -> crate::kernels::cuda::DeviceRange<'_> {
        b.range(0, b.bytes()).unwrap()
    }
    let bufs = Exl3MoeBufs {
        route: RouteBufs {
            resid_in: range(&resid),
            resid_out: range(&out),
            norm_row: range(&x),
            logits: range(&dummy),
            ids: range(&ids),
            weights: range(&weights),
            shared_gate: range(&shared_gate),
            status: range(&dummy),
        },
        expert_gu: range(&gu),
        expert_act: range(&act),
        expert_down: range(&down),
        shared_gu: range(&dummy),
        shared_act: range(&dummy),
        shared_out: range(&shared),
        scratch: Scratch {
            xh: range(&dummy),
            z: range(&dummy),
            flag: range(&dummy),
        },
    };
    let mut bank = PersistentRouted::new(&ctx, &stream, &model, 0..1, dims.top_k + 1).unwrap();
    let write_ids = |values: &[i32]| {
        ids.write_at(
            &stream,
            0,
            &values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let original: Vec<i32> = (1..=dims.top_k as i32).collect();
    write_ids(&original);
    let first = bank.run(0, &bufs, &dims).unwrap();
    assert_eq!((first.hits, first.ops.len()), (dims.top_k, 0));
    let mut changed = original.clone();
    changed[0] = dims.top_k as i32 + 1;
    write_ids(&changed);
    let evict = bank.run(0, &bufs, &dims).unwrap();
    assert_eq!((evict.hits, evict.ops.len()), (dims.top_k - 1, 1));
    write_ids(&original);
    let restore = bank.run(0, &bufs, &dims).unwrap();
    assert_eq!((restore.hits, restore.ops.len()), (dims.top_k - 1, 1));
    let repeat = bank.run(0, &bufs, &dims).unwrap();
    assert_eq!((repeat.hits, repeat.ops.len()), (dims.top_k, 0));
    let mut actual = vec![0; dims.hidden * 4];
    out.read_at(&stream, 0, &mut actual).unwrap();
    for (i, bytes) in actual.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_ne_bytes(bytes.try_into().unwrap()),
            (i % 11) as f32 * 0.0625
        );
    }
    let mut invalid = original.clone();
    invalid[0] = invalid[1];
    write_ids(&invalid);
    assert!(bank.run(0, &bufs, &dims).is_err());
    write_ids(&original);
    assert!(
        bank.run(0, &bufs, &dims).is_err(),
        "invalid input must poison the owner"
    );
    drop(bank);
    let mut io_bank = PersistentRouted::new(&ctx, &stream, &model, 0..1, dims.top_k + 1).unwrap();
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
    assert!(
        layout
            .read(&model, 0, changed[0] as u64, &mut vec![0; layout.bytes])
            .is_err()
    );
    write_ids(&changed);
    assert!(
        io_bank.run(0, &bufs, &dims).is_err(),
        "truncated shard must fail a miss"
    );
    write_ids(&original);
    assert!(
        io_bank.run(0, &bufs, &dims).is_err(),
        "refill failure must poison the owner"
    );
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires admitted CUDA device; synthetic split GGUF"]
fn pinned_disk_refill_reuses_record_and_checks_ownership() {
    use super::expert_layout::{ExpertDeviceStacks, PinnedDiskRefill};
    use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
    let (_file, model, layout, _, _) = split_expert_fixture();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let experts = model.config().dims.experts as usize;
    let geometry = BankGeometry::new(1, experts, 2, 3, layout.bytes).unwrap();
    let mut row = vec![-1; experts];
    for (slot, id) in [0usize, 1, 2].into_iter().enumerate() {
        row[id] = slot as i32;
    }
    let state = || {
        let policy =
            ReplaceOnMissLru::from_fill(geometry, &[&row], &[vec![0; experts]], None).unwrap();
        ExpertBankState::with_policy(geometry, HostMode::Disk, policy).unwrap()
    };
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1, 2]).unwrap();
    let mut policy = state();
    let mut refill = PinnedDiskRefill::new(&ctx, &stream, &model, 0).unwrap();
    for picks in [[1, 256], [257, 256]] {
        let copies = refill.step(&mut bank, &mut policy, &picks).unwrap();
        assert!(!copies.ops.is_empty());
        for slot in 0..3 {
            let id = policy.expert_in_slot(0, slot).unwrap();
            let mut record = vec![0; layout.bytes];
            layout.read(&model, 0, u64::from(id), &mut record).unwrap();
            for (i, part) in layout.parts.iter().enumerate() {
                let range = bank.part(i).unwrap();
                let mut got = vec![0; part.bytes.len()];
                range
                    .buffer()
                    .read_at(&stream, range.offset() + slot * got.len(), &mut got)
                    .unwrap();
                assert_eq!(got, record[part.bytes.clone()]);
            }
        }
    }
    assert!(
        refill
            .step(&mut bank, &mut policy, &[257, 256])
            .unwrap()
            .ops
            .is_empty()
    );

    let foreign = CudaContext::new().unwrap();
    let foreign_stream = foreign.stream().unwrap();
    assert!(PinnedDiskRefill::new(&ctx, &foreign_stream, &model, 0).is_err());
    let mut foreign_refill = PinnedDiskRefill::new(&foreign, &foreign_stream, &model, 0).unwrap();
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1, 2]).unwrap();
    let mut policy = state();
    for picks in [[0, 1], [1, 256]] {
        assert!(foreign_refill.step(&mut bank, &mut policy, &picks).is_err());
        assert!(!policy.is_poisoned());
        assert_eq!(policy.slot_row(0), row);
        assert!(bank.part(0).is_ok());
    }
    assert!(refill.step(&mut bank, &mut policy, &[1, 256]).is_ok());
}

#[test]
#[ignore = "requires admitted CUDA device; truncates only an owned synthetic shard"]
fn pinned_disk_refill_read_failure_poisons_policy_and_device() {
    use super::expert_layout::{ExpertDeviceStacks, PinnedDiskRefill};
    use crate::family::{
        expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru},
        recipe::Role,
    };
    let (file, model, layout, _, _) = split_expert_fixture();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut bank = ExpertDeviceStacks::from_model(&ctx, &stream, &model, 0, &[0, 1]).unwrap();
    let mut refill = PinnedDiskRefill::new(&ctx, &stream, &model, 0).unwrap();
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
    assert!(refill.step(&mut bank, &mut state, &[256]).is_err());
    assert!(state.is_poisoned());
    assert!(bank.part(0).is_err());
    assert!(refill.step(&mut bank, &mut state, &[0]).is_err());
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
                    x: q.range(0, qbytes).unwrap(),
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
                        x: q2.range(0, q2bytes).unwrap(),
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

fn classify_expert_pair(
    ctx: &CudaContext,
    stream: &DeviceStream,
    row: &[i32],
    expected: [i32; 2],
) -> crate::kernels::cuda::DeviceBuffer {
    use crate::kernels::cuda::resident::tiered_classify::{
        CTRL_BYTES, ClassifyCall, ClassifyShape, REQUEST_BYTES,
    };
    let allocate = |bytes: usize| {
        let buffer = ctx.allocate(bytes).unwrap();
        buffer.write_at(stream, 0, &vec![0; bytes]).unwrap();
        buffer
    };
    let ids = allocate(8);
    ids.write_at(
        stream,
        0,
        &[257i32.to_ne_bytes(), 256i32.to_ne_bytes()].concat(),
    )
    .unwrap();
    let weights = allocate(8);
    weights
        .write_at(
            stream,
            0,
            &[0.25f32.to_ne_bytes(), 0.75f32.to_ne_bytes()].concat(),
        )
        .unwrap();
    let table = allocate(row.len() * 4);
    table
        .write_at(
            stream,
            0,
            &row.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
    let norm = allocate(16);
    let out = allocate(8);
    let out_weights = allocate(8);
    let log = allocate(16);
    let ctrl = allocate(CTRL_BYTES);
    let word = allocate(4);
    let request = allocate(REQUEST_BYTES);
    let mailbox = allocate(16);
    ctx.tiered_classify(
        stream,
        ClassifyCall {
            shape: ClassifyShape {
                n_expert: row.len(),
                slots: 3,
                top_k: 2,
                hidden: 4,
                layer: 0,
            },
            ids: ids.range(0, 8).unwrap(),
            weights: weights.range(0, 8).unwrap(),
            slot_row: table.range(0, row.len() * 4).unwrap(),
            norm_row: norm.range(0, 16).unwrap(),
            slot_ids_out: out.range(0, 8).unwrap(),
            weights_out: out_weights.range(0, 8).unwrap(),
            log_out: log.range(0, 16).unwrap(),
            ctrl: ctrl.range(0, CTRL_BYTES).unwrap(),
            word: word.range(0, 4).unwrap(),
            mbox_req: request.range(0, REQUEST_BYTES).unwrap(),
            mbox_row: mailbox.range(0, 16).unwrap(),
        },
    )
    .unwrap();
    let mut observed = [0u8; 8];
    out.read_at(stream, 0, &mut observed).unwrap();
    let actual: Vec<i32> = observed
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(actual, expected);
    let mut weight_bytes = [0u8; 8];
    out_weights.read_at(stream, 0, &mut weight_bytes).unwrap();
    let actual_weights: Vec<f32> = weight_bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        actual_weights,
        [if row[257] < 0 { 0.0 } else { 0.25 }, 0.75]
    );
    out
}

#[test]
#[ignore = "needs the locally hashed two-expert sparse fixture; CPU-only preflight"]
fn two_real_expert_gate_preflight() {
    use crate::{family::recipe::Role, weights::dequantize};
    use sha2::{Digest, Sha256};
    let path = std::env::var_os("FIFI_FLASH_TWO_EXPERT_FIXTURE").unwrap();
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let tensor = model.map().get(Role::ExpertGate, Some(0)).unwrap();
    let k = tensor.dims[0] as usize;
    let input: Vec<f32> = (0..k).map(|i| if i % 5 < 2 { -1.0 } else { 1.0 }).collect();
    let mut weights = Vec::new();
    for (expert, hash) in [
        (
            256,
            "3da6977d2c06568b213e1e79fd49f3de3816c594c7b6f0928cbc4f74eed6d13b",
        ),
        (
            257,
            "ac0f9bfdd7ec13f0d982a235686c747b7e31737d170ed1d7a54faa71f424cd6a",
        ),
    ] {
        let blocks = mapped_blocks(&model, tensor, Some(expert), 0..tensor.dims[1]).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&blocks)), hash);
        weights.push(dequantize(tensor.ty, &blocks).unwrap());
    }
    let mut distinguishing = 0;
    for (a, b) in weights[0].chunks_exact(k).zip(weights[1].chunks_exact(k)) {
        let dot = |row: &[f32]| -> f32 { row.iter().zip(&input).map(|(w, x)| w * x).sum() };
        let tol = 0.003
            * (0.75 * a.iter().map(|v| v.abs()).sum::<f32>()
                + 0.25 * b.iter().map(|v| v.abs()).sum::<f32>())
            + 1e-5;
        if (0.5 * (dot(a) - dot(b))).abs() > tol {
            distinguishing += 1;
        }
    }
    assert!(
        distinguishing > 0,
        "swapping expert weights must be detectable"
    );
    eprintln!("{distinguishing} gate rows detect a swapped 0.25/0.75 expert weighting");
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
    // Slot zero is a synthetic zero filler; expert 256 is the real resident hit.
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
    let before = classify_expert_pair(&ctx, &stream, state.slot_row(0), [0, 2]);
    let gate = bank
        .projection(&state, crate::family::recipe::Role::ExpertGate)
        .unwrap();
    let (k, rows) = (gate.k, gate.rows);
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
    gate.launch(
        &ctx,
        &stream,
        ExpertProjectionInputs {
            ids: before.range(0, 8).unwrap(),
            x: q.range(0, qbytes).unwrap(),
            x_per_slot: false,
            output: y.range(0, rows * 8).unwrap(),
            output_stride: rows,
            used: 2,
        },
    )
    .unwrap();
    let mut output = vec![0; rows * 8];
    y.read_at(&stream, 0, &mut output).unwrap();
    let tensor = model
        .map()
        .get(crate::family::recipe::Role::ExpertGate, Some(0))
        .unwrap();
    let blocks = mapped_blocks(&model, tensor, Some(256), 0..rows as u64).unwrap();
    let resident = dequantize(tensor.ty, &blocks).unwrap();
    for (index, row) in resident.chunks_exact(k).enumerate() {
        let filler = f32::from_ne_bytes(output[index * 4..index * 4 + 4].try_into().unwrap());
        let offset = (rows + index) * 4;
        let hit = f32::from_ne_bytes(output[offset..offset + 4].try_into().unwrap());
        let want: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
        let tolerance = 0.003 * row.iter().map(|v| v.abs()).sum::<f32>() + 1e-5;
        assert_eq!(filler, 0.0);
        assert!((hit - want).abs() <= tolerance, "resident gate row {index}");
    }
    eprintln!("checked {rows} resident gate rows and zero filler before miss");
    let hidden = bank
        .projection(&state, crate::family::recipe::Role::ExpertDown)
        .unwrap()
        .rows;
    let repaired_down = ctx.allocate(2 * hidden * 4).unwrap();
    macro_rules! full_pass {
        ($ids:expr, $count:expr) => {{
            use crate::family::recipe::Role;
            use crate::kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};
            let count = $count;
            let gu = ctx.allocate(count * 2 * rows * 4).unwrap();
            for (role, off) in [(Role::ExpertGate, 0), (Role::ExpertUp, rows * 4)] {
                bank.projection(&state, role)
                    .unwrap()
                    .launch(
                        &ctx,
                        &stream,
                        ExpertProjectionInputs {
                            ids: $ids.range(0, count * 4).unwrap(),
                            x: q.range(0, qbytes).unwrap(),
                            x_per_slot: false,
                            output: gu.range(off, (2 * count - 1) * rows * 4).unwrap(),
                            output_stride: 2 * rows,
                            used: count,
                        },
                    )
                    .unwrap();
            }
            let act = ctx.allocate(count * rows * 4).unwrap();
            ctx.swiglu_rows(
                &stream,
                SwigluRowsCall {
                    shape: SwigluRowsShape {
                        rows: count,
                        n_ff: rows,
                        up_off: rows,
                        ld_in: 2 * rows,
                        ld_out: rows,
                    },
                    gu: gu.range(0, count * 2 * rows * 4).unwrap(),
                    act: act.range(0, count * rows * 4).unwrap(),
                },
            )
            .unwrap();
            let act_qbytes = q8_1_bytes(rows, count).unwrap();
            let act_q = ctx.allocate(act_qbytes).unwrap();
            ctx.quantize_q8_1(
                &stream,
                QuantizeCall {
                    x: act.range(0, count * rows * 4).unwrap(),
                    y: act_q.range(0, act_qbytes).unwrap(),
                    k: rows,
                    ncols: count,
                },
            )
            .unwrap();
            bank.projection(&state, Role::ExpertDown)
                .unwrap()
                .launch(
                    &ctx,
                    &stream,
                    ExpertProjectionInputs {
                        ids: $ids.range(0, count * 4).unwrap(),
                        x: act_q.range(0, act_qbytes).unwrap(),
                        x_per_slot: true,
                        output: repaired_down.range(0, count * hidden * 4).unwrap(),
                        output_stride: hidden,
                        used: count,
                    },
                )
                .unwrap();
        }};
    }
    full_pass!(before, 2);
    let mut resident_down = vec![0; hidden * 8];
    repaired_down
        .read_at(&stream, 0, &mut resident_down)
        .unwrap();
    assert!(
        resident_down[..hidden * 4]
            .chunks_exact(4)
            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()) == 0.0),
        "filler down must be zero"
    );
    let copies = bank
        .step_disk(&stream, &model, &mut state, 0, &[257, 256])
        .unwrap();
    assert_eq!(
        (copies.hits, copies.ops.len(), copies.ops[0].incoming),
        (1, 1, 257)
    );
    let slots = [state.slot_row(0)[257], state.slot_row(0)[256]];
    assert!(slots.iter().all(|&slot| slot >= 0) && slots[0] != slots[1]);
    assert_eq!(i32::from(copies.ops[0].slot), slots[0]);
    let miss_id = ctx.allocate(4).unwrap();
    miss_id
        .write_at(&stream, 0, &slots[0].to_ne_bytes())
        .unwrap();
    full_pass!(miss_id, 1);
    let mut repaired = vec![0; hidden * 8];
    repaired_down.read_at(&stream, 0, &mut repaired).unwrap();
    assert_eq!(
        &repaired[hidden * 4..],
        &resident_down[hidden * 4..],
        "resident down must remain unchanged"
    );
    assert_ne!(
        &repaired[..hidden * 4],
        &resident_down[..hidden * 4],
        "miss must overwrite filler down"
    );
    bank.projection(&state, crate::family::recipe::Role::ExpertGate)
        .unwrap()
        .launch(
            &ctx,
            &stream,
            ExpertProjectionInputs {
                ids: miss_id.range(0, 4).unwrap(),
                x: q.range(0, qbytes).unwrap(),
                x_per_slot: false,
                output: y.range(0, rows * 4).unwrap(),
                output_stride: rows,
                used: 1,
            },
        )
        .unwrap();
    let mut after = vec![0; rows * 8];
    y.read_at(&stream, 0, &mut after).unwrap();
    assert_eq!(
        &after[rows * 4..],
        &output[rows * 4..],
        "resident rank remains untouched"
    );
    use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
    let residual = ctx.allocate(rows * 4).unwrap();
    residual.write_at(&stream, 0, &vec![0; rows * 4]).unwrap();
    let weights = ctx.allocate(8).unwrap();
    weights
        .write_at(
            &stream,
            0,
            &[0.25f32.to_ne_bytes(), 0.75f32.to_ne_bytes()].concat(),
        )
        .unwrap();
    let combined = ctx.allocate(rows * 4).unwrap();
    ctx.moe_combine(
        &stream,
        CombineCall {
            shape: CombineShape {
                m: 1,
                n: rows,
                top_k: 2,
                ld_down: rows,
            },
            res: residual.range(0, rows * 4).unwrap(),
            down: y.range(0, rows * 8).unwrap(),
            weights: weights.range(0, 8).unwrap(),
            shared: None,
            out: CombineOut::To(combined.range(0, rows * 4).unwrap()),
        },
    )
    .unwrap();
    let mut combined_bytes = vec![0; rows * 4];
    combined.read_at(&stream, 0, &mut combined_bytes).unwrap();
    let missing_blocks = mapped_blocks(&model, tensor, Some(257), 0..rows as u64).unwrap();
    let missing = dequantize(tensor.ty, &missing_blocks).unwrap();
    for (index, (hit_row, miss_row)) in resident
        .chunks_exact(k)
        .zip(missing.chunks_exact(k))
        .enumerate()
    {
        let off = index * 4;
        let got_miss = f32::from_ne_bytes(after[off..off + 4].try_into().unwrap());
        let hit_off = (rows + index) * 4;
        let got_hit = f32::from_ne_bytes(after[hit_off..hit_off + 4].try_into().unwrap());
        let dot = |row: &[f32]| -> f32 { row.iter().zip(&input).map(|(a, b)| a * b).sum() };
        let expected = 0.25 * dot(miss_row) + 0.75 * dot(hit_row);
        let tolerance = 0.003
            * (0.25 * miss_row.iter().map(|v| v.abs()).sum::<f32>()
                + 0.75 * hit_row.iter().map(|v| v.abs()).sum::<f32>())
            + 1e-5;
        let device_combined = f32::from_ne_bytes(combined_bytes[off..off + 4].try_into().unwrap());
        assert!(
            (0.25 * got_miss + 0.75 * got_hit - expected).abs() <= tolerance,
            "host combined gate row {index}"
        );
        assert!(
            (device_combined - expected).abs() <= tolerance,
            "device combined gate row {index}"
        );
    }
    eprintln!("checked {rows} miss-only gate rows and device weighted gate combination");
    let ids = classify_expert_pair(&ctx, &stream, state.slot_row(0), slots);
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
                    x: q.range(0, qbytes).unwrap(),
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
    use crate::family::recipe::Role;
    use crate::kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};
    let gate = bank.projection(&state, Role::ExpertGate).unwrap();
    let up = bank.projection(&state, Role::ExpertUp).unwrap();
    assert_eq!((gate.k, gate.rows), (up.k, up.rows));
    let (k, n_ff) = (gate.k, gate.rows);
    let input: Vec<u8> = (0..k)
        .flat_map(|i| (if i % 5 < 2 { -1.0f32 } else { 1.0 }).to_ne_bytes())
        .collect();
    let x = ctx.allocate(input.len()).unwrap();
    x.write_at(&stream, 0, &input).unwrap();
    let qb = q8_1_bytes(k, 1).unwrap();
    let q = ctx.allocate(qb).unwrap();
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: x.range(0, input.len()).unwrap(),
            y: q.range(0, qb).unwrap(),
            k,
            ncols: 1,
        },
    )
    .unwrap();
    let gu = ctx.allocate(4 * n_ff * 4).unwrap();
    for (projection, offset) in [(gate, 0), (up, n_ff * 4)] {
        projection
            .launch(
                &ctx,
                &stream,
                ExpertProjectionInputs {
                    ids: ids.range(0, 8).unwrap(),
                    x: q.range(0, qb).unwrap(),
                    x_per_slot: false,
                    output: gu.range(offset, 3 * n_ff * 4).unwrap(),
                    output_stride: 2 * n_ff,
                    used: 2,
                },
            )
            .unwrap();
    }
    let act = ctx.allocate(2 * n_ff * 4).unwrap();
    ctx.swiglu_rows(
        &stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows: 2,
                n_ff,
                up_off: n_ff,
                ld_in: 2 * n_ff,
                ld_out: n_ff,
            },
            gu: gu.range(0, 4 * n_ff * 4).unwrap(),
            act: act.range(0, 2 * n_ff * 4).unwrap(),
        },
    )
    .unwrap();
    let mut gu_bytes = vec![0; 4 * n_ff * 4];
    gu.read_at(&stream, 0, &mut gu_bytes).unwrap();
    let mut act_bytes = vec![0; 2 * n_ff * 4];
    act.read_at(&stream, 0, &mut act_bytes).unwrap();
    for rank in 0..2 {
        for index in 0..n_ff {
            let f =
                |off: usize| f32::from_ne_bytes(gu_bytes[off * 4..off * 4 + 4].try_into().unwrap());
            let g = f(rank * 2 * n_ff + index);
            let u = f(rank * 2 * n_ff + n_ff + index);
            let off = (rank * n_ff + index) * 4;
            let got = f32::from_ne_bytes(act_bytes[off..off + 4].try_into().unwrap());
            let want = g / (1.0 + (-g).exp()) * u;
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "SwiGLU rank {rank} row {index}"
            );
        }
    }
    eprintln!("checked {n_ff} SwiGLU values for two real experts");
    let qa_bytes = q8_1_bytes(n_ff, 2).unwrap();
    let qa = ctx.allocate(qa_bytes).unwrap();
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: act.range(0, 2 * n_ff * 4).unwrap(),
            y: qa.range(0, qa_bytes).unwrap(),
            k: n_ff,
            ncols: 2,
        },
    )
    .unwrap();
    let down = bank.projection(&state, Role::ExpertDown).unwrap();
    assert_eq!(down.k, n_ff);
    let hidden = down.rows;
    let y_down = ctx.allocate(2 * hidden * 4).unwrap();
    down.launch(
        &ctx,
        &stream,
        ExpertProjectionInputs {
            ids: ids.range(0, 8).unwrap(),
            x: qa.range(0, qa_bytes).unwrap(),
            x_per_slot: true,
            output: y_down.range(0, 2 * hidden * 4).unwrap(),
            output_stride: hidden,
            used: 2,
        },
    )
    .unwrap();
    let mut down_bytes = vec![0; 2 * hidden * 4];
    y_down.read_at(&stream, 0, &mut down_bytes).unwrap();
    let tensor = model.map().get(Role::ExpertDown, Some(0)).unwrap();
    let host_input: Vec<f32> = (0..k).map(|i| if i % 5 < 2 { -1.0 } else { 1.0 }).collect();
    let mut error_sq = 0.0f64;
    let mut norm_sq = 0.0f64;
    let mut host_rows = vec![vec![0.0f32; hidden]; 2];
    for (rank, expert) in [257u64, 256].into_iter().enumerate() {
        let input: Vec<f32> = act_bytes[rank * n_ff * 4..(rank + 1) * n_ff * 4]
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let max_abs = input.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
        let host_projections: Vec<Vec<f32>> = [Role::ExpertGate, Role::ExpertUp]
            .into_iter()
            .map(|role| {
                let t = model.map().get(role, Some(0)).unwrap();
                let blocks = mapped_blocks(&model, t, Some(expert), 0..n_ff as u64).unwrap();
                dequantize(t.ty, &blocks)
                    .unwrap()
                    .chunks_exact(k)
                    .map(|row| row.iter().zip(&host_input).map(|(w, x)| w * x).sum())
                    .collect()
            })
            .collect();
        let host_activation: Vec<f32> = host_projections[0]
            .iter()
            .zip(&host_projections[1])
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let max_activation_delta = input
            .iter()
            .zip(&host_activation)
            .fold(0.0f32, |max, (&a, &b)| max.max((a - b).abs()));
        let blocks = mapped_blocks(&model, tensor, Some(expert), 0..hidden as u64).unwrap();
        let weights = dequantize(tensor.ty, &blocks).unwrap();
        for (index, row) in weights.chunks_exact(n_ff).enumerate() {
            let off = (rank * hidden + index) * 4;
            let got = f32::from_ne_bytes(down_bytes[off..off + 4].try_into().unwrap());
            let want: f32 = row.iter().zip(&input).map(|(w, x)| w * x).sum();
            let tolerance = 0.01 * max_abs * row.iter().map(|v| v.abs()).sum::<f32>() + 1e-5;
            assert!(
                (got - want).abs() <= tolerance,
                "down after SwiGLU rank {rank} row {index}"
            );
            let host_want: f32 = row.iter().zip(&host_activation).map(|(w, x)| w * x).sum();
            host_rows[rank][index] = host_want;
            let bound = tolerance + max_activation_delta * row.iter().map(|w| w.abs()).sum::<f32>();
            assert!(
                (got - host_want).abs() <= bound,
                "independent expert rank {rank} row {index}"
            );
            error_sq += f64::from(got - host_want).powi(2);
            norm_sq += f64::from(host_want).powi(2);
        }
    }
    let relative_l2 = (error_sq / norm_sq).sqrt();
    eprintln!(
        "checked {hidden} down rows for two independent host expert MLPs; relative L2 {relative_l2}"
    );
    assert!(norm_sq > 0.0 && relative_l2 <= 0.05);
    let residual = ctx.allocate(hidden * 4).unwrap();
    residual.write_at(&stream, 0, &vec![0; hidden * 4]).unwrap();
    let route_weights = ctx.allocate(8).unwrap();
    route_weights
        .write_at(
            &stream,
            0,
            &[0.25f32.to_ne_bytes(), 0.75f32.to_ne_bytes()].concat(),
        )
        .unwrap();
    let combined = ctx.allocate(hidden * 4).unwrap();
    ctx.moe_combine(
        &stream,
        CombineCall {
            shape: CombineShape {
                m: 1,
                n: hidden,
                top_k: 2,
                ld_down: hidden,
            },
            res: residual.range(0, hidden * 4).unwrap(),
            down: repaired_down.range(0, hidden * 8).unwrap(),
            weights: route_weights.range(0, 8).unwrap(),
            shared: None,
            out: CombineOut::To(combined.range(0, hidden * 4).unwrap()),
        },
    )
    .unwrap();
    let mut result = vec![0; hidden * 4];
    combined.read_at(&stream, 0, &mut result).unwrap();
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (index, chunk) in result.chunks_exact(4).enumerate() {
        let got = f32::from_ne_bytes(chunk.try_into().unwrap());
        let want = 0.25 * host_rows[0][index] + 0.75 * host_rows[1][index];
        err += f64::from(got - want).powi(2);
        norm += f64::from(want).powi(2);
    }
    let combined_l2 = (err / norm).sqrt();
    eprintln!("two-expert weighted output relative L2 {combined_l2}");
    assert!(norm > 0.0 && combined_l2 <= 0.05);
}

/// The token loop and unquantized operators stay on the host; this is not a serving test.
#[test]
#[ignore = "requires the complete pinned GGUF and an admitted CUDA device"]
fn checkpoint_greedy_rows_with_cuda_projections() {
    use super::host::{HostModel, greedy_rows, greedy_rows_with, greedy_token};
    use super::projector::HostProjector;
    let path =
        std::env::var_os("FIFI_FLASH_GGUF").expect("set FIFI_FLASH_GGUF to the verified pack");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .expect("Flash-Next GGUF");
    let host = HostModel::load(&model, model.config().layers()).expect("host model");
    let prompt = [9707, 11, 1879];
    let decisions = std::env::var("FIFI_FLASH_GGUF_DECISIONS")
        .map(|value| value.parse::<usize>().expect("valid decision count"))
        .unwrap_or(3);
    assert!((1..=3).contains(&decisions));
    let expected = if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_ORACLE") {
        #[derive(serde::Deserialize)]
        struct Receipt {
            prompt: Vec<u32>,
            vocab: usize,
            greedy: Vec<u32>,
            logits: Vec<Vec<f32>>,
        }
        let file = std::fs::File::open(path).expect("open host oracle receipt");
        let receipt: Receipt = serde_json::from_reader(std::io::BufReader::new(file))
            .expect("parse host oracle receipt");
        assert_eq!(receipt.prompt, prompt);
        assert_eq!(receipt.logits.len(), 3);
        assert_eq!(receipt.greedy.len(), 3);
        for (row, &token) in receipt.logits.iter().zip(&receipt.greedy) {
            assert_eq!(row.len(), receipt.vocab);
            assert_eq!(greedy_token(row).expect("finite oracle logits"), token);
        }
        receipt.logits.into_iter().take(decisions).collect()
    } else {
        greedy_rows(&model, &host, &prompt, decisions, 8).expect("host greedy rows")
    };
    eprintln!("hybrid probe: oracle ready, decisions={decisions}");
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let ngram = model.ngram().expect("n-gram table");
    eprintln!("hybrid probe: CUDA context ready");
    let launches = Cell::new(0);
    let projector = HostProjector::new(&model.weights, &ngram, 8).with_cuda(Projection {
        ctx: &ctx,
        stream: &stream,
        launches: &launches,
    });
    let actual =
        greedy_rows_with(projector, &host, &prompt, decisions).expect("hybrid greedy rows");
    assert!(launches.get() > 0, "no CUDA projections ran");
    assert_eq!(actual.len(), expected.len());
    if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_HYBRID_OUT") {
        let greedy: Vec<_> = actual
            .iter()
            .map(|row| greedy_token(row).expect("finite hybrid logits"))
            .collect();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new hybrid receipt");
        let mut out = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut out, &serde_json::json!({"prompt": prompt, "greedy": greedy, "logits": actual, "cuda_projections": launches.get()})).expect("write hybrid receipt");
        std::io::Write::flush(&mut out).expect("flush hybrid receipt");
    }
    for (position, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(got.len(), want.len());
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
        assert_eq!(
            greedy_token(got).unwrap(),
            greedy_token(want).unwrap(),
            "decision {position}"
        );
        // Provisional bring-up threshold, not a model quality criterion.
        assert!(
            relative_l2 <= 0.05,
            "decision {position}: relative L2 {relative_l2}"
        );
    }
}

/// Records or compares first-token per-layer residuals; diagnostic, not parity qualification.
#[test]
#[ignore = "requires the pinned GGUF and optionally an admitted CUDA device"]
fn checkpoint_first_token_layer_trace() {
    use super::host::{HostModel, run, run_with};
    use super::projector::HostProjector;
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("verified checkpoint");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let host = HostModel::load(&model, model.config().layers()).expect("host model");
    let token = [9707];
    if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_LAYER_OUT") {
        let trace = run(&model, &host, &token, 8).expect("host trace");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("new layer receipt");
        let mut out = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut out, &trace.layer_out).expect("write layer receipt");
        std::io::Write::flush(&mut out).expect("flush layer receipt");
        return;
    }
    let path = std::env::var_os("FIFI_FLASH_GGUF_LAYER_ORACLE").expect("layer oracle");
    let expected: Vec<Vec<Vec<f32>>> = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(path).expect("open layer oracle"),
    ))
    .expect("parse layer oracle");
    let ctx = CudaContext::new().expect("CUDA device");
    let stream = ctx.stream().expect("stream");
    let ngram = model.ngram().expect("ngram");
    let launches = Cell::new(0);
    let projector = HostProjector::new(&model.weights, &ngram, 8).with_cuda(Projection {
        ctx: &ctx,
        stream: &stream,
        launches: &launches,
    });
    let actual = run_with(projector, &host, &token).expect("hybrid trace");
    assert!(launches.get() > 0);
    assert_eq!(actual.layer_out.len(), 1);
    assert_eq!(expected.len(), 1);
    assert_eq!(actual.layer_out[0].len(), expected[0].len());
    for (index, (got, want)) in actual.layer_out[0].iter().zip(&expected[0]).enumerate() {
        assert_eq!(got.len(), want.len());
        let (error, norm) = got
            .iter()
            .zip(want)
            .fold((0.0_f64, 0.0_f64), |(e, n), (&a, &b)| {
                (
                    e + (f64::from(a) - f64::from(b)).powi(2),
                    n + f64::from(b).powi(2),
                )
            });
        assert!(error.is_finite() && norm.is_finite() && norm > 0.0);
        eprintln!(
            "first-token layer {index}: relative L2={}",
            (error / norm).sqrt()
        );
    }
    eprintln!(
        "first-token layer trace: CUDA projections={}",
        launches.get()
    );
}

/// Attribute first real projection error to activation quantization versus the native matvec.
#[test]
#[ignore = "requires the pinned GGUF and captured first-token host and hybrid projections"]
fn checkpoint_first_projection_q8_activation_diagnostic() {
    use std::io::BufRead;
    #[derive(serde::Deserialize)]
    struct Capture {
        name: String,
        input: Vec<f32>,
        output: Vec<f32>,
    }
    fn first(path: &str) -> Capture {
        let file = std::fs::File::open(std::env::var_os(path).expect("capture path"))
            .expect("open capture");
        serde_json::from_str(
            &std::io::BufReader::new(file)
                .lines()
                .next()
                .expect("first record")
                .expect("read record"),
        )
        .expect("parse record")
    }
    let host = first("FIFI_FLASH_GGUF_HOST_CAPTURE");
    let device = first("FIFI_FLASH_GGUF_HYBRID_CAPTURE");
    assert_eq!(host.name, "blk.0.hc_attn_down.weight");
    assert_eq!(host.name, device.name);
    assert_eq!(host.input, device.input);
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("checkpoint");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let tensor = model
        .map
        .entries()
        .iter()
        .find(|t| t.name == host.name)
        .expect("tensor");
    assert_eq!(tensor.ty, GgufType::Q80);
    assert_eq!(tensor.dims[0] as usize, host.input.len());
    let blocks = mapped_blocks(&model, tensor, None, 0..tensor.dims[1]).expect("weight rows");
    let weights = crate::weights::dequantize(tensor.ty, &blocks).expect("decode weights");
    let quantized: Vec<f32> = host
        .input
        .chunks(32)
        .flat_map(|chunk| {
            let max = chunk.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let d = max / 127.0;
            let stored_d = half::f16::from_f32(d).to_f32();
            chunk.iter().map(move |&x| {
                if max == 0.0 {
                    0.0
                } else {
                    (x / d).round() * stored_d
                }
            })
        })
        .collect();
    let qdot: Vec<f32> = weights
        .chunks_exact(host.input.len())
        .map(|row| row.iter().zip(&quantized).map(|(w, x)| w * x).sum())
        .collect();
    assert_eq!(qdot.len(), host.output.len());
    assert_eq!(qdot.len(), device.output.len());
    let rel = |a: &[f32], b: &[f32]| {
        let (err, norm) = a
            .iter()
            .zip(b)
            .fold((0.0_f64, 0.0_f64), |(e, n), (&x, &y)| {
                (
                    e + (f64::from(x) - f64::from(y)).powi(2),
                    n + f64::from(y).powi(2),
                )
            });
        assert!(norm > 0.0 && err.is_finite());
        (err / norm).sqrt()
    };
    eprintln!(
        "first projection relative L2: gpu_vs_f32={}, host_q8_vs_f32={}, gpu_vs_host_q8={}",
        rel(&device.output, &host.output),
        rel(&qdot, &host.output),
        rel(&device.output, &qdot)
    );
}

/// Compare native Q8 activation and FP16 GEMM on identical real expert inputs.
#[test]
#[ignore = "requires pinned GGUF, host layer-zero capture, and admitted CUDA device"]
fn checkpoint_expert_gemm_precision_diagnostic() {
    use std::io::BufRead;
    #[derive(serde::Deserialize)]
    struct Capture {
        name: String,
        expert: Option<u32>,
        input: Vec<f32>,
        output: Vec<f32>,
    }
    let path = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("host capture");
    let entries: Vec<Capture> =
        std::io::BufReader::new(std::fs::File::open(path).expect("open capture"))
            .lines()
            .map(|line| serde_json::from_str(&line.expect("line")).expect("record"))
            .collect();
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("checkpoint");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let ctx = CudaContext::new().expect("device");
    let stream = ctx.stream().expect("stream");
    for name in ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"] {
        let capture = entries
            .iter()
            .find(|e| e.name == name && e.expert == Some(342))
            .expect("captured expert");
        let tensor = model
            .map
            .entries()
            .iter()
            .find(|t| t.name == name)
            .expect("tensor");
        let rows = tensor.dims[1] as usize;
        let k = capture.input.len();
        assert_eq!(tensor.dims[0] as usize, k);
        assert_eq!(capture.output.len(), rows);
        let blocks = mapped_blocks(&model, tensor, Some(342), 0..rows as u64).expect("expert rows");
        let q8 = project(&ctx, &stream, tensor.ty, &blocks, &capture.input)
            .expect("native q8 projection");
        let launches = Cell::new(0);
        let gemm = Projection {
            ctx: &ctx,
            stream: &stream,
            launches: &launches,
        }
        .run_f16(tensor.ty, &blocks, &capture.input)
        .expect("fp16 gemm");
        assert_eq!(launches.get(), 1);
        let rel = |a: &[f32]| {
            let (err, norm) =
                a.iter()
                    .zip(&capture.output)
                    .fold((0.0_f64, 0.0_f64), |(e, n), (&x, &y)| {
                        (
                            e + (f64::from(x) - f64::from(y)).powi(2),
                            n + f64::from(y).powi(2),
                        )
                    });
            assert!(err.is_finite() && norm > 0.0);
            (err / norm).sqrt()
        };
        eprintln!(
            "expert precision {name} type={:?} k={k} rows={rows} q8_rel_l2={} gemm_f16_rel_l2={}",
            tensor.ty,
            rel(&q8),
            rel(&gemm)
        );
    }
}

#[test]
#[ignore = "requires pinned GGUF, host layer-zero capture, and admitted CUDA device"]
fn checkpoint_trunk_gemm_precision_diagnostic() {
    use std::io::BufRead;
    #[derive(serde::Deserialize)]
    struct Capture {
        name: String,
        input: Vec<f32>,
        output: Vec<f32>,
    }
    let path = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("host capture");
    let entries: Vec<Capture> =
        std::io::BufReader::new(std::fs::File::open(path).expect("open capture"))
            .lines()
            .map(|line| serde_json::from_str(&line.expect("line")).expect("record"))
            .collect();
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("checkpoint");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let ctx = CudaContext::new().expect("device");
    let stream = ctx.stream().expect("stream");
    for name in ["blk.0.hc_attn_down.weight", "blk.0.attn_qkv.weight"] {
        let capture = entries
            .iter()
            .find(|e| e.name == name)
            .expect("captured trunk");
        let tensor = model
            .map
            .entries()
            .iter()
            .find(|t| t.name == name)
            .expect("tensor");
        let rows = tensor.dims[1] as usize;
        assert_eq!(tensor.dims[0] as usize, capture.input.len());
        assert_eq!(capture.output.len(), rows);
        let blocks = mapped_blocks(&model, tensor, None, 0..rows as u64).expect("weight rows");
        let launches = Cell::new(0);
        let actual = Projection {
            ctx: &ctx,
            stream: &stream,
            launches: &launches,
        }
        .run_f16(tensor.ty, &blocks, &capture.input)
        .expect("fp16 gemm");
        assert_eq!(launches.get(), 1);
        let (error, norm) =
            actual
                .iter()
                .zip(&capture.output)
                .fold((0.0_f64, 0.0_f64), |(e, n), (&x, &y)| {
                    (
                        e + (f64::from(x) - f64::from(y)).powi(2),
                        n + f64::from(y).powi(2),
                    )
                });
        assert!(error.is_finite() && norm > 0.0);
        let rel = (error / norm).sqrt();
        eprintln!(
            "trunk precision {name} type={:?} rows={rows} gemm_f16_rel_l2={rel}",
            tensor.ty
        );
        assert!(rel < 0.005, "{name} relative L2 {rel}");
    }
}

#[test]
#[ignore = "requires the verified pinned GGUF and an admitted CUDA device"]
fn checkpoint_delta_out_q6_grouped_activation_matches_host() {
    use super::projector::{HostProjector, HostWeight, Remap, tile};
    use crate::family::{
        project::{Projector, cuda::CudaWeight},
        recipe::Role,
    };
    use crate::kernels::cuda::resident::head_tile::{HeadTileCall, HeadTileShape};

    let path =
        std::env::var_os("FIFI_FLASH_GGUF").expect("set FIFI_FLASH_GGUF to verified first shard");
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .expect("Flash-Next GGUF");
    let entry = model.map().get(Role::DeltaOut, Some(0)).unwrap();
    assert_eq!(
        entry.ty,
        GgufType::Q6K,
        "this diagnostic targets the pinned Q6_K layer"
    );
    let dims = &model.config().dims;
    let (kh, vh, hd) = (
        dims.key_heads as usize,
        dims.value_heads as usize,
        dims.value_dim as usize,
    );
    assert!(kh > 1 && vh > kh && vh % kh == 0);
    assert_eq!(entry.dims[0] as usize, vh * hd);
    assert!(!hd.is_multiple_of(entry.ty.block_elems() as usize));
    let inputs: Vec<f32> = (0..vh * hd)
        .map(|i| {
            let head = i / hd;
            let j = i % hd;
            (((head * 31 + j * 17) % 101) as f32 - 50.0) / 50.0 + (head % 7) as f32 * 0.025
        })
        .collect();
    let ngram = model.ngram().unwrap();
    let mut host = HostProjector::new(&model.weights, &ngram, 1);
    let reference = host
        .linear(
            &HostWeight::new(
                &[entry],
                Remap::ValueIn {
                    hd,
                    key_heads: kh,
                    value_heads: vh,
                },
            ),
            &inputs,
        )
        .unwrap();
    let expected_tiled = tile(&inputs, hd, kh, vh);
    assert_ne!(
        inputs, expected_tiled,
        "fixture must exercise the permutation"
    );

    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let raw = model.upload_delta_out(0, &ctx).unwrap();
    let CudaWeight::Native { name, weight } = raw else {
        panic!("native Q6_K expected")
    };
    assert_eq!(
        name,
        crate::family::exl3_pack::layer_name(0, "linear_attn.out_proj")
    );
    let input_bytes: Vec<u8> = inputs.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let grouped = ctx.allocate(input_bytes.len()).unwrap();
    let tiled = ctx.allocate(input_bytes.len()).unwrap();
    grouped.write_at(&stream, 0, &input_bytes).unwrap();
    let head_bytes = hd * size_of::<f32>();
    let reps = vh / kh;
    for h in 0..vh {
        let stored = (h % reps) * kh + h / reps;
        ctx.copy_device(
            &stream,
            grouped.range(h * head_bytes, head_bytes).unwrap(),
            tiled.range(stored * head_bytes, head_bytes).unwrap(),
        )
        .unwrap();
    }
    let single = ctx.allocate(input_bytes.len()).unwrap();
    ctx.head_tile(
        &stream,
        HeadTileCall {
            src: grouped.range(0, input_bytes.len()).unwrap(),
            dst: single.range(0, input_bytes.len()).unwrap(),
            shape: HeadTileShape {
                key_heads: kh,
                value_heads: vh,
                head_dim: hd,
            },
        },
    )
    .unwrap();
    let mut reordered = vec![0; input_bytes.len()];
    tiled.read_at(&stream, 0, &mut reordered).unwrap();
    let mut one_launch = vec![0; input_bytes.len()];
    single.read_at(&stream, 0, &mut one_launch).unwrap();
    assert_eq!(
        one_launch, reordered,
        "one launch must match the per-head copies"
    );
    let expected_bytes: Vec<u8> = expected_tiled
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    assert_eq!(
        reordered, expected_bytes,
        "device permutation must preserve every input bit"
    );
    let qbytes = q8_1_bytes(inputs.len(), 1).unwrap();
    let q8 = ctx.allocate(qbytes).unwrap();
    let out_bytes = reference.len() * size_of::<f32>();
    let out = ctx.allocate(out_bytes).unwrap();
    let run = |source: &crate::kernels::cuda::DeviceBuffer| {
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: source.range(0, input_bytes.len()).unwrap(),
                y: q8.range(0, qbytes).unwrap(),
                k: inputs.len(),
                ncols: 1,
            },
        )
        .unwrap();
        ctx.matvec_native(
            &stream,
            &weight,
            q8.range(0, qbytes).unwrap(),
            out.range(0, out_bytes).unwrap(),
            1,
        )
        .unwrap();
        let mut raw = vec![0u8; out_bytes];
        out.read_at(&stream, 0, &mut raw).unwrap();
        raw.chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let got = run(&single);
    assert_eq!(
        got,
        run(&tiled),
        "equivalent reorderings must project identically"
    );
    let wrong = run(&grouped);
    let rel = |actual: &[f32]| {
        let numerator: f64 = actual
            .iter()
            .zip(&reference)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let denominator: f64 = reference.iter().map(|&v| (v as f64).powi(2)).sum();
        assert!(denominator > 0.0 && numerator.is_finite());
        (numerator / denominator).sqrt()
    };
    let (error, wrong_error) = (rel(&got), rel(&wrong));
    eprintln!(
        "pinned DeltaOut Q6_K rows={} input_heads={vh} head_dim={hd}: reordered relative_l2={error:.6}, identity relative_l2={wrong_error:.6}, one_kernel_vs_device_copies={vh}",
        reference.len()
    );
    assert!(error < 0.03, "reordered output relative L2 {error}");
    assert!(
        wrong_error > 0.1 && wrong_error > 5.0 * error,
        "identity negative control {wrong_error}, reordered {error}"
    );
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
