use super::*;
use crate::family::qwen::flash_next::load::gguf::testkit::{Spec, Written, write_flash};

#[test]
fn staging_validates_layout_and_centers_before_rounding() {
    let file = write_flash(&Spec::default());
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let d = &model.config.dims;
    let width = u64::from(d.hidden) * u64::from(d.branches);
    let norm = model
        .map
        .get(Slot::hyper(Site::Mixer, SiteTensor::Norm), Some(0))
        .unwrap();
    let plan = Staging::new(norm, &[width], true, false).unwrap();
    assert_eq!(plan.output_bytes, width as usize * 2);
    assert!(!plan.stored);
    assert!(Staging::new(norm, &[width], false, false).is_err());
    assert!(Staging::new(norm, &[width + 1], true, false).is_err());
    let mut wrong = norm.clone();
    wrong.bytes -= 1;
    assert!(Staging::new(&wrong, &[width], true, false).is_err());
    wrong = norm.clone();
    wrong.ty = GgufType::Q6K;
    assert!(Staging::new(&wrong, &[width], true, false).is_err());
    wrong = norm.clone();
    wrong.dims = vec![u64::MAX, u64::MAX];
    assert!(Staging::new(&wrong, &[u64::MAX, u64::MAX], true, false).is_err());
    wrong.dims = vec![u64::MAX / 2];
    assert!(Staging::new(&wrong, &[u64::MAX / 2], true, false).is_err());
    let encoded = half_bytes(&[1.0001, 0.75, 1.25], true).unwrap();
    let decoded: Vec<f32> = encoded
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    assert_ne!(decoded[0], 0.0, "subtract before rounding the stored gamma");
    assert_eq!(decoded[1..], [-0.25, 0.25]);
    assert!(half_bytes(&[f32::INFINITY], false).is_err());
    assert!(half_bytes(&[f32::NAN], false).is_err());
    assert!(half_bytes(&[70_000.0], false).is_err());
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn synthetic_mixed_types_upload_to_expected_f16() {
    let roles = [
        (SiteTensor::Norm, "hc_norm.weight", GgufType::F32),
        (
            SiteTensor::Down,
            "input_mix_weight_down.weight",
            GgufType::F16,
        ),
        (SiteTensor::Up, "input_mix_weight_up.weight", GgufType::Bf16),
        (
            SiteTensor::Inject,
            "block_inject_weight.weight",
            GgufType::Q80,
        ),
    ];
    let spec = Spec {
        retype: vec![
            ("blk.0.hc_attn_norm.weight".into(), GgufType::F32),
            ("blk.0.hc_attn_down.weight".into(), GgufType::F16),
            ("blk.0.hc_attn_up.weight".into(), GgufType::Bf16),
            ("blk.0.hc_attn_inject.weight".into(), GgufType::Q80),
        ],
        ..Spec::default()
    };
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let mut expected = Vec::new();
    for (part, _, ty) in roles {
        let entry = model
            .map
            .get(Slot::hyper(Site::Mixer, part), Some(0))
            .unwrap();
        assert_eq!(entry.ty, ty);
        let elements = entry.dims.iter().product::<u64>() as usize;
        let mut raw = Vec::with_capacity(entry.bytes as usize);
        if ty == GgufType::Q80 {
            for block in 0..elements / 32 {
                raw.extend_from_slice(&0x1400u16.to_le_bytes());
                for i in 0..32 {
                    raw.push((((block * 7 + i * 3) % 21) as i8 - 10) as u8);
                }
            }
        } else {
            for i in 0..elements {
                let value = if part == SiteTensor::Norm {
                    1.0 + ((i % 7) as f32 - 3.0) * 0.0001
                } else {
                    ((i % 101) as f32 - 50.0) * 0.0001
                };
                match ty {
                    GgufType::F32 => raw.extend_from_slice(&value.to_le_bytes()),
                    GgufType::F16 => {
                        raw.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                    }
                    GgufType::Bf16 => {
                        raw.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes())
                    }
                    _ => unreachable!(),
                }
            }
        }
        assert_eq!(raw.len(), entry.bytes as usize);
        file.patch(&entry.name, 0, &raw);
        let reference: Vec<f32> = match ty {
            GgufType::Q80 => (0..elements)
                .map(|i| {
                    let q = (((i / 32 * 7 + i % 32 * 3) % 21) as i8 - 10) as f32;
                    half::f16::from_bits(0x1400).to_f32() * q
                })
                .collect(),
            GgufType::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            GgufType::F16 => raw
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
            GgufType::Bf16 => raw
                .chunks_exact(2)
                .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
            _ => unreachable!(),
        };
        let bytes: Vec<u8> = reference
            .into_iter()
            .flat_map(|v| {
                let v = if part == SiteTensor::Norm { v - 1.0 } else { v };
                half::f16::from_f32(v).to_bits().to_ne_bytes()
            })
            .collect();
        expected.push(bytes);
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = model
        .upload_hyper(Site::Mixer, Some(0), &ctx, &stream)
        .unwrap();
    assert_eq!(weights.len(), 4);
    for ((weight, want), (_, suffix, _)) in weights.iter().zip(expected).zip(roles) {
        let CudaWeight::Raw { name, ty, buffer } = weight else {
            panic!("raw F16 expected")
        };
        assert_eq!(*ty, GgufType::F16);
        assert_eq!(
            name,
            &format!("{}.{suffix}", layer_name(0, "attn_hyper_connection"))
        );
        let mut got = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut got).unwrap();
        assert_eq!(got, want);
    }
    assert!(
        model
            .upload_hyper(Site::Head, Some(0), &ctx, &stream)
            .is_err()
    );
    assert!(
        model
            .upload_hyper(Site::Mixer, None, &ctx, &stream)
            .is_err()
    );
    assert!(
        model
            .upload_hyper(
                Site::Mixer,
                Some(model.config.dims.kinds.len()),
                &ctx,
                &stream
            )
            .is_err()
    );
}

/// A Q8_0 matrix is staged as stored, as many bytes as the file holds; the same tensor asked for
/// as a vector would be widened.
#[test]
fn a_q80_matrix_is_kept_as_stored() {
    let spec = Spec {
        retype: vec![
            ("blk.0.hc_attn_down.weight".into(), GgufType::Q80),
            ("blk.0.hc_attn_up.weight".into(), GgufType::Q80),
            ("blk.0.hc_attn_inject.weight".into(), GgufType::Q80),
        ],
        ..Spec::default()
    };
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let entry = |part| {
        model
            .map
            .get(Slot::hyper(Site::Mixer, part), Some(0))
            .unwrap()
    };
    for part in [SiteTensor::Down, SiteTensor::Up] {
        let entry = entry(part);
        assert_eq!(entry.ty, GgufType::Q80);
        let stored = Staging::new(entry, &entry.dims, false, true).unwrap();
        assert!(stored.stored, "{part:?}");
        assert_eq!(stored.output_bytes, entry.bytes as usize);
        assert!(stored.output_bytes.is_multiple_of(34));
        let widened = Staging::new(entry, &entry.dims, false, false).unwrap();
        assert!(!widened.stored);
        assert_eq!(
            widened.output_bytes,
            entry.dims.iter().product::<u64>() as usize * 2
        );
        // The map charges what the loader stages: the stored blocks.
        assert_eq!(entry.device_bytes(), entry.bytes, "{part:?}");
    }
    let inject = entry(SiteTensor::Inject);
    assert_eq!(inject.ty, GgufType::Q80);
    assert!(
        !Staging::new(inject, &inject.dims, false, false)
            .unwrap()
            .stored
    );
    assert_eq!(
        inject.device_bytes(),
        inject.dims.iter().product::<u64>() * 2,
        "a Q8_0 vector is charged as the F16 it is widened to"
    );
    // A centered tensor is never Q8_0, stored or widened.
    let norm = entry(SiteTensor::Norm);
    let mut centered_q80 = norm.clone();
    centered_q80.ty = GgufType::Q80;
    assert!(Staging::new(&centered_q80, &[norm.dims[0]], true, false).is_err());
}

/// Q8_0 blocks of known bytes for a hyper tensor, patched into the file: a block's scale grows
/// with its index and its lanes walk a short cycle.
fn q80_blocks(file: &Written, entry: &Mapped) -> Vec<u8> {
    assert_eq!(entry.ty, GgufType::Q80);
    let blocks = entry.bytes as usize / 34;
    let mut raw = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        let scale = half::f16::from_f32(0.03 * (block as f32 + 1.0));
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32 {
            raw.push((((block * 5 + i * 11) % 17) as i8 - 8) as u8);
        }
    }
    file.patch(&entry.name, 0, &raw);
    raw
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn q80_matrices_upload_byte_for_byte_and_need_their_pair() {
    let spec = Spec {
        retype: vec![
            ("blk.0.hc_attn_down.weight".into(), GgufType::Q80),
            ("blk.0.hc_attn_up.weight".into(), GgufType::Q80),
        ],
        ..Spec::default()
    };
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let entry = |part| {
        model
            .map
            .get(Slot::hyper(Site::Mixer, part), Some(0))
            .unwrap()
    };
    let stored = [
        (
            "input_mix_weight_down.weight",
            q80_blocks(&file, entry(SiteTensor::Down)),
        ),
        (
            "input_mix_weight_up.weight",
            q80_blocks(&file, entry(SiteTensor::Up)),
        ),
    ];
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = model
        .upload_hyper(Site::Mixer, Some(0), &ctx, &stream)
        .unwrap();
    assert_eq!(weights.len(), 4);
    for weight in [weights.first().unwrap(), weights.last().unwrap()] {
        let CudaWeight::Raw { ty, .. } = weight else {
            panic!("a raw weight")
        };
        assert_eq!(*ty, GgufType::F16, "the vectors stay F16");
    }
    for (weight, (suffix, raw)) in weights[1..3].iter().zip(stored) {
        let CudaWeight::Raw { name, ty, buffer } = weight else {
            panic!("a raw weight")
        };
        assert_eq!(*ty, GgufType::Q80);
        assert_eq!(
            name,
            &format!("{}.{suffix}", layer_name(0, "attn_hyper_connection"))
        );
        let mut got = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut got).unwrap();
        assert_eq!(got, raw, "{suffix}: the stored blocks changed");
    }

    // One Q8_0 matrix without the other is refused before any upload (the synthetic model stores
    // both as Q8_0, so the other one is retyped).
    let spec = Spec {
        retype: vec![("blk.0.hc_attn_up.weight".into(), GgufType::F16)],
        ..Spec::default()
    };
    let file = write_flash(&spec);
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let error = model
        .upload_hyper(Site::Mixer, Some(0), &ctx, &stream)
        .err()
        .expect("an unpaired Q8_0 matrix");
    assert!(
        format!("{error:?}").contains("not both Q8_0"),
        "refused as {error:?}"
    );
}
