use std::path::Path;

use half::{bf16, f16};

use super::super::{
    FlintResidentLimits, ResidentPlan,
    cache::{
        BankPlan, BankRequest, CACHE_VAR, ExpertCacheMode, SLOTS_VAR, cpu_moe_bank, plan_bank,
    },
    plan::plan_resident,
};
use super::{FlintExl3Resident, PackedMatrix, PackedRange, host::BankAsk, packing::Packing};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, plan},
    tensor::Dtype,
    weights::{
        WeightSource,
        exl3::{
            Checkpoint, Matrix,
            flint_pack::{Geometry, Plain},
        },
    },
};

#[cfg(test)]
pub(super) fn load_cached_fixture(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
    slots: usize,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::from_config(&checkpoint.model_config)?;
    let ask = BankAsk {
        mode: ExpertCacheMode::On,
        request: BankRequest::Exactly(slots),
        rows: 1,
    };
    super::host::load_banked(context, stream, path, checkpoint, geometry, limits, ask)
}

pub(crate) fn load_flint_resident(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    load_with_prefill(
        context,
        stream,
        path,
        limits,
        crate::family::flint::settings::PREFILL_MAX_ROWS,
    )
}

/// Device bytes of the pack with every expert resident.
pub(super) fn resident_weight_bytes(path: &Path) -> Result<u64, EngError> {
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::published(&checkpoint.model_config)?;
    let limits = FlintResidentLimits {
        max_staging_bytes: u64::MAX,
        max_weight_bytes: u64::MAX,
        reserve: Default::default(),
    };
    let roles = plan_resident(path, &checkpoint, &geometry, limits)?;
    u64::try_from(Packing::new(&roles.layout, &limits)?.bytes)
        .map_err(|_| EngError::MemoryPlan("Flint EXL3 resident bytes overflow".into()))
}

pub(super) fn cache_forced_off() -> bool {
    ExpertCacheMode::from_env(std::env::var(CACHE_VAR)) == Ok(ExpertCacheMode::Off)
}

pub(super) fn cache_forced_on() -> bool {
    ExpertCacheMode::from_env(std::env::var(CACHE_VAR)) == Ok(ExpertCacheMode::On)
}

pub(super) fn load_with_prefill(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
    rows: usize,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::published(&checkpoint.model_config)?;
    let mode = ExpertCacheMode::from_env(std::env::var(CACHE_VAR))?;
    let request = BankRequest::from_env(std::env::var(SLOTS_VAR))?;
    let config = crate::family::flint::cpu_moe::active().map(|moe| moe.config());
    let (mode, request) = cpu_moe_bank(config, mode, request)?;
    if mode == ExpertCacheMode::Off {
        // The plan never says Bank for the off switch; the arm refuses rather than banks if it ever does.
        let BankPlan::Resident = plan_bank(mode, request, false)? else {
            return Err(EngError::BadRequest(
                "Flint expert cache: the off switch must keep every expert resident".into(),
            ));
        };
        return load_checked(context, stream, path, checkpoint, geometry, limits);
    }
    let ask = BankAsk {
        mode,
        request,
        rows,
    };
    super::host::load_banked(context, stream, path, checkpoint, geometry, limits, ask)
}

#[cfg(test)]
pub(crate) fn load_flint_fixture(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    context.ensure_usable()?;
    let checkpoint = Checkpoint::open(path)?;
    let geometry = Geometry::from_config(&checkpoint.model_config)?;
    load_checked(context, stream, path, checkpoint, geometry, limits)
}

fn load_checked(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    checkpoint: Checkpoint,
    geometry: Geometry,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    let roles = plan_resident(path, &checkpoint, &geometry, limits)?;
    load_planned(context, stream, checkpoint, roles, limits)
}

/// The fully resident load of an already audited pack: every expert of every layer on the device.
pub(super) fn load_planned(
    context: &CudaContext,
    stream: &DeviceStream,
    checkpoint: Checkpoint,
    roles: ResidentPlan,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    let packing = Packing::new(&roles.layout, &limits)?;
    plan::admit(
        &plan::Demand {
            tensors: packing.bytes as u64,
            kv_cache: limits.reserve.kv_cache,
            work: limits.reserve.work,
            headroom: limits.reserve.headroom,
        },
        u64::try_from(context.memory_info()?.free).unwrap_or(u64::MAX),
    )?;
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(roles.layout.modules.len())
        .map_err(|error| {
            EngError::MemoryPlan(format!("Flint EXL3 resident matrix descriptors: {error}"))
        })?;
    let buffer = context.allocate(packing.bytes)?;
    for (module, slots) in roles.layout.modules.iter().zip(&packing.matrices) {
        let specs = module.specs();
        let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        let selected = checkpoint
            .source
            .read_selected(&keys, limits.max_staging_bytes)?;
        let matrix = Matrix::load(&selected, &module.prefix, &checkpoint.config)?;
        let trellis = matrix.trellis().host_bytes().ok_or_else(|| {
            EngError::Checkpoint("Flint EXL3 trellis requires host storage".into())
        })?;
        if trellis.len() != slots[0].bytes || matrix.bias().is_some() {
            return Err(EngError::Checkpoint(
                "Flint EXL3 matrix disagrees with audited placement".into(),
            ));
        }
        buffer.write_at(stream, slots[0].offset, trellis)?;
        let [suh_factor, svh_factor] = scale_factors(&module.prefix);
        for (tensor, &slot, factor) in [
            (matrix.suh(), &slots[1], suh_factor),
            (matrix.svh(), &slots[2], svh_factor),
        ] {
            let raw = tensor.host_bytes().ok_or_else(|| {
                EngError::Checkpoint("Flint EXL3 scales require host storage".into())
            })?;
            write_halves_by(&buffer, stream, slot, (raw, tensor.dtype()), factor)?;
        }
        matrices.push(PackedMatrix {
            trellis: slots[0],
            suh: slots[1],
            svh: slots[2],
            bias: None,
            shape: matrix.shape(),
            bits_twice: matrix.bitrate_twice(),
            book: matrix.codebook(),
        });
    }
    for (plain, &slot) in roles.layout.plain.iter().zip(&packing.plain) {
        upload_plain(
            &checkpoint,
            &buffer,
            stream,
            plain,
            slot,
            limits.max_staging_bytes,
        )?;
    }
    Ok(FlintExl3Resident {
        buffer,
        host: None,
        pick_tables: (0..roles.layers.len())
            .map(|_| std::cell::OnceCell::new())
            .collect(),
        matrices,
        plain: packing.plain,
        roles,
        model_config: checkpoint.model_config,
    })
}

pub(super) fn upload_plain(
    checkpoint: &Checkpoint,
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    plain: &Plain,
    slot: PackedRange,
    max_staging: u64,
) -> Result<(), EngError> {
    if !plain.bulk {
        let selected = checkpoint
            .source
            .read_selected(&[&plain.name], max_staging)?;
        let tensor = selected.tensor(&plain.name).ok_or_else(|| {
            EngError::Checkpoint(format!("Flint EXL3 missing auxiliary {}", plain.name))
        })?;
        let raw = tensor.host_bytes().ok_or_else(|| {
            EngError::Checkpoint("Flint EXL3 auxiliary requires host storage".into())
        })?;
        return write_halves(buffer, stream, slot, raw, tensor.dtype());
    }
    // The embedding is read one existing leading slice at a time, never staged whole.
    let rows = plain.shape[0];
    let row_bytes = slot.bytes / rows as usize;
    let mut raw = Vec::new();
    raw.try_reserve_exact(row_bytes / 2)
        .map_err(|error| EngError::MemoryPlan(format!("Flint EXL3 embedding row: {error}")))?;
    raw.resize(row_bytes / 2, 0);
    for row in 0..rows {
        checkpoint
            .source
            .read_slice_into(&plain.name, row, &mut raw)?;
        let row_slot = PackedRange {
            offset: slot.offset + row as usize * row_bytes,
            bytes: row_bytes,
        };
        write_halves(buffer, stream, row_slot, &raw, plain.dtype)?;
    }
    Ok(())
}

fn write_halves(
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    slot: PackedRange,
    raw: &[u8],
    dtype: Dtype,
) -> Result<(), EngError> {
    write_halves_by(buffer, stream, slot, (raw, dtype), 1.0)
}

fn write_halves_by(
    buffer: &DeviceBuffer,
    stream: &DeviceStream,
    slot: PackedRange,
    (raw, dtype): (&[u8], Dtype),
    factor: f32,
) -> Result<(), EngError> {
    let converted = convert_halves_by(raw, dtype, factor)?;
    if converted.len() != slot.bytes {
        return Err(EngError::Checkpoint(
            "Flint EXL3 F32 auxiliary disagrees with audited placement".into(),
        ));
    }
    buffer.write_at(stream, slot.offset, &converted)
}

/// The power-of-two factors the input and output scales of a matrix are stored with. The routed
/// up projections of the pack fold /128 into their input scales, which leaves nearly all of them
/// FP16 subnormals; multiplying the input scale by 2^12 and the output scale by 2^-12 changes
/// no F32 result (power-of-two scaling is exact and every product after it is scaled alike) and
/// keeps the rotated input of the FP16-operand kernels in the normal FP16 range.
pub(super) fn scale_factors(prefix: &str) -> [f32; 2] {
    if prefix.contains(".mlp.experts.") && prefix.ends_with(".up_proj") {
        [4096.0, 1.0 / 4096.0]
    } else {
        [1.0, 1.0]
    }
}

pub(super) fn convert_halves(raw: &[u8], dtype: Dtype) -> Result<Vec<u8>, EngError> {
    convert_halves_by(raw, dtype, 1.0)
}

/// [`convert_halves`] with every value multiplied by `factor`, a power of two.
pub(super) fn convert_halves_by(
    raw: &[u8],
    dtype: Dtype,
    factor: f32,
) -> Result<Vec<u8>, EngError> {
    if !matches!(dtype, Dtype::F16 | Dtype::Bf16) || !raw.len().is_multiple_of(2) {
        return Err(EngError::Checkpoint(
            "Flint EXL3 auxiliary requires complete F16/BF16 values".into(),
        ));
    }
    let bytes = raw
        .len()
        .checked_mul(2)
        .ok_or_else(|| EngError::MemoryPlan("Flint EXL3 F32 staging length overflows".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|error| EngError::MemoryPlan(format!("Flint EXL3 F32 staging: {error}")))?;
    for word in raw.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*word);
        let value = factor
            * if dtype == Dtype::F16 {
                f16::from_bits(bits).to_f32()
            } else {
                bf16::from_bits(bits).to_f32()
            };
        if !value.is_finite() {
            return Err(EngError::Checkpoint(
                "Flint EXL3 auxiliary contains nonfinite values".into(),
            ));
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_prefill_floor_is_six_experts_per_row() {
        use crate::family::flint::cuda_ffn::device::exl3::cache_prefill_floor;
        for (rows, floor) in [(1, 6), (8, 48), (37, 222), (38, 228), (64, 384)] {
            assert_eq!(cache_prefill_floor(rows).unwrap(), floor, "rows={rows}");
        }
        for rows in [0, usize::MAX] {
            assert!(cache_prefill_floor(rows).is_err(), "rows={rows}");
        }
    }

    #[test]
    fn routed_up_scales_move_a_power_of_two_between_input_and_output() {
        let up = "model.layers.5.mlp.experts.9.up_proj";
        let [suh, svh] = scale_factors(up);
        assert_eq!((suh, svh), (4096.0, 1.0 / 4096.0));
        for other in [
            "model.layers.5.mlp.experts.9.gate_proj",
            "model.layers.5.mlp.experts.9.down_proj",
            "model.layers.5.mlp.shared_expert.up_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.3.self_attn.q_proj",
        ] {
            assert_eq!(scale_factors(other), [1.0, 1.0], "{other}");
        }
        // A subnormal FP16 input scale becomes a normal FP16 value, and the pair multiplies back
        // to the same F32 product, bit for bit.
        let tiny = [0x0001u16, 0x0100, 0x03ff, 0x0400];
        for bits in tiny {
            let value = f16::from_bits(bits).to_f32();
            let scaled = convert_halves_by(&bits.to_le_bytes(), Dtype::F16, suh).unwrap();
            let scaled = f32::from_le_bytes(scaled[..4].try_into().unwrap());
            assert_eq!(scaled, value * 4096.0);
            assert!(scaled >= f16::MIN_POSITIVE.to_f32(), "bits {bits:#06x}");
            for x in [0.37f32, 1.0, 1234.5, -7.25e-3] {
                let out = [0.91f32, -3.5, 1.0e-3];
                for y in out {
                    assert_eq!(
                        ((x * scaled) * y) * svh,
                        ((x * value) * y),
                        "bits {bits:#06x}"
                    );
                }
            }
        }
        assert!(convert_halves_by(&0x7c00u16.to_le_bytes(), Dtype::F16, suh).is_err());
    }

    /// The ranges of every routed up projection's scales in the real pack, before and after the
    /// 2^12 fold. Reads about 50 MB of headers and scales; no device.
    #[test]
    #[ignore = "reads the real EXL3 pack from FLINT_EXL3_PACK; no device"]
    fn the_fold_keeps_the_real_up_scales_in_range() {
        use crate::weights::exl3::Checkpoint;
        let pack = std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("pack"));
        let checkpoint = Checkpoint::open(&pack).unwrap();
        let (mut suh_max, mut suh_min, mut svh_max, mut svh_min) = (0f32, f32::MAX, 0f32, f32::MAX);
        let (mut vectors, mut subnormal, mut values) = (0usize, 0usize, 0usize);
        for layer in 0..49 {
            let names: Vec<String> = (0..256)
                .flat_map(|expert| {
                    let prefix = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj");
                    [format!("{prefix}.suh"), format!("{prefix}.svh")]
                })
                .filter(|name| checkpoint.source.contains(name))
                .collect();
            if names.is_empty() {
                continue;
            }
            let keys: Vec<&str> = names.iter().map(String::as_str).collect();
            let selected = checkpoint.source.read_selected(&keys, 1 << 30).unwrap();
            for name in &names {
                let tensor = selected.tensor(name).unwrap();
                let raw = tensor.host_bytes().unwrap();
                assert_eq!(tensor.dtype(), Dtype::F16);
                let input = name.ends_with(".suh");
                vectors += usize::from(input);
                for word in raw.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*word);
                    let value = f16::from_bits(bits).to_f32().abs();
                    if input {
                        values += 1;
                        subnormal +=
                            usize::from(value != 0.0 && value < f16::MIN_POSITIVE.to_f32());
                        suh_max = suh_max.max(value);
                        if value != 0.0 {
                            suh_min = suh_min.min(value);
                        }
                    } else {
                        svh_max = svh_max.max(value);
                        if value != 0.0 {
                            svh_min = svh_min.min(value);
                        }
                    }
                }
            }
        }
        let [up, down] = scale_factors("model.layers.3.mlp.experts.7.up_proj");
        println!(
            "FOLD up_proj vectors {vectors}, suh values {values} ({subnormal} subnormal): suh [{suh_min:e}, {suh_max:e}] x {up} = [{:e}, {:e}]; svh [{svh_min:e}, {svh_max:e}] x {down} = [{:e}, {:e}]",
            suh_min * up,
            suh_max * up,
            svh_min * down,
            svh_max * down
        );
        // The input scale after the fold stays inside FP16, and the output scale stays a normal
        // F32 (the kernels read both as F32; no FP16 copy of either exists on the device path).
        assert!(suh_max * up < 65504.0);
        assert!(svh_min * down >= f32::MIN_POSITIVE);
    }

    #[test]
    fn raw_f32_conversion_preserves_norms_signed_zero_and_subnormals() {
        for (dtype, bits, expected) in [
            (Dtype::F16, 0x3c00u16, 1f32.to_bits()),
            (Dtype::Bf16, 0x3f80, 1f32.to_bits()),
            (Dtype::F16, 0x8000, (-0f32).to_bits()),
            (Dtype::F16, 0x0001, 0x33800000),
            (Dtype::Bf16, 0x0001, 0x00010000),
        ] {
            assert_eq!(
                convert_halves(&bits.to_le_bytes(), dtype).unwrap(),
                expected.to_le_bytes()
            );
        }
        for (dtype, bits) in [(Dtype::F16, 0x7c00u16), (Dtype::Bf16, 0x7f80)] {
            assert!(matches!(
                convert_halves(&bits.to_le_bytes(), dtype),
                Err(EngError::Checkpoint(_))
            ));
        }
        assert!(convert_halves(&[0], Dtype::F16).is_err());
        assert!(convert_halves(&[0, 0], Dtype::F32).is_err());
    }
}
