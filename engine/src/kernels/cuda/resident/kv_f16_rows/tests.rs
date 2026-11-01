use super::*;
use crate::kernels::cuda::testkit;

/// Host twin of the production guard (runtime.rs f16_row): accept iff finite,
/// low 16 f32 mantissa bits zero, and the RNE f16 conversion stays finite.
/// Returns the half-word bits the host publication would write. Accepted
/// half-words match the host RNE conversion bit for bit, including the
/// rounding clean subnormal-range inputs take on both sides.
fn host_guard(value: f32) -> Option<u16> {
    let stored = half::f16::from_f32(value);
    if !value.is_finite() || value.to_bits() & 0xffff != 0 || !stored.is_finite() {
        return None;
    }
    Some(stored.to_bits())
}

/// Named bit cases over every accept and reject class. Expectations are
/// DERIVED from host_guard (never hand-written constants); the host twin is
/// pinned first, before the device-presence return, on values whose f16
/// facts are known independently: 0x477ff000 is 65520 with dirty low bits
/// (reject — NOT max-finite-f16), 65504 itself is not bf16-clean so the
/// largest clean accept is 65280 -> 0x7bf8, and clean subnormal-range values
/// round RNE on both sides (2^-24 -> 0x0001, 2^-20 -> 0x0010).
#[test]
fn kv_f16_rows_match_the_host_guard_bit_for_bit() {
    // Host-twin self-consistency: hand-verified f16 facts.
    assert_eq!(
        host_guard(f32::from_bits(0x477ff000)),
        None,
        "65520 is dirty"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x477f0000)),
        Some(0x7bf8),
        "65280"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x477fe000)),
        None,
        "65504 dirty low"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x38000000)),
        Some(0x0200),
        "2^-15"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x33800000)),
        Some(0x0001),
        "2^-24"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x35800000)),
        Some(0x0010),
        "2^-20"
    );
    assert_eq!(
        host_guard(f32::from_bits(0x33800800)),
        None,
        "subnormal dirty"
    );
    if !testkit::device_present("kv f16 rows host parity") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let baseline = ctx.live_bytes();
    {
        // (f32 bits, label) — the expectation comes from host_guard(bits).
        let cases: [(u32, &str); 26] = [
            (0x00000000, "+0"),
            (0x80000000, "-0 (f16-exact accept)"),
            (0x3f800000, "1.0"),
            (0xbf800000, "-1.0"),
            (0x477f0000, "65280, largest clean accept"),
            (0x477ff000, "65520, dirty low bits"),
            (0x477fe000, "65504 itself is dirty"),
            (0x47800000, "65536: clean overflow"),
            (0x38000000, "2^-15"),
            (0x33800000, "2^-24 min subnormal"),
            (0x33800400, "2^-24 dirty"),
            (0x35800000, "2^-20"),
            (0x33800800, "subnormal-range dirty"),
            (0x00010001, "subnormal-adjacent dirty"),
            (0x3f800001, "dirty normal"),
            (0x477ff001, "just past max finite, dirty"),
            (0x7f800000, "+inf"),
            (0xff800000, "-inf"),
            (0x7fc00000, "NaN"),
            (0xffc00000, "-NaN"),
            (0x7f800001, "NaN dirty payload"),
            (0x461ff000, "bf16-style dirty"),
            (0xc61ff000, "its negative"),
            (0x461c0000, "bf16-clean normal"),
            (0xc61c0000, "bf16-clean negative"),
            (0x00008000, "tiny clean subnormal"),
        ];
        let (rows, columns) = (5, cases.len());
        let selected: Vec<_> = (0..rows)
            .flat_map(|row| (0..columns).map(move |col| cases[(row + col) % cases.len()]))
            .collect();
        let input: Vec<_> = selected
            .iter()
            .map(|&(bits, _)| f32::from_bits(bits))
            .collect();
        let wants: Vec<Option<u16>> = selected
            .iter()
            .map(|&(bits, _)| host_guard(f32::from_bits(bits)))
            .collect();
        let bytes = input.len() * 4;
        let half_bytes = input.len() * 2;
        let x = testkit::upload_f32(&ctx, &input);
        // Fill with poison so an unwritten half-word cannot pass as correct.
        let y = testkit::upload_filled(&ctx, 0xa5, half_bytes + 8);
        let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
        ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                x: x.range(0, bytes).unwrap(),
                out: Some(y.range(0, half_bytes).unwrap()),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns,
            },
        )
        .expect("one kv f16 rows launch");
        stream.synchronize().expect("completion before health/read");
        // Sticky word ORs onto its start value; rejects never clear it.
        let expected_flag = if wants.iter().any(Option::is_none) {
            9u32
        } else {
            8u32
        };
        assert_eq!(
            testkit::read_bytes(&ctx, &flag, 4),
            expected_flag.to_ne_bytes()
        );
        let written = testkit::read_bytes(&ctx, &y, half_bytes + 8);
        let halves: Vec<u16> = written[..half_bytes]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_ne_bytes(*pair))
            .collect();
        for (i, (&got, (&(bits, label), want))) in
            halves.iter().zip(selected.iter().zip(&wants)).enumerate()
        {
            match want {
                Some(want) => assert_eq!(got, *want, "accept {label} {bits:#010x} at {i}"),
                None => assert_eq!(got, 0, "reject {label} {bits:#010x} at {i}"),
            }
        }
        // The tail past the region stays untouched.
        assert!(written[half_bytes..].iter().all(|&byte| byte == 0xa5));
    }
    stream.synchronize().unwrap();
    assert_eq!(ctx.live_bytes(), baseline, "device allocations returned");
}

/// Exhaustive clean-input sweep: every f32 whose low 16 bits are zero
/// (hi16 << 16 over the full 65,536-entry range) plus dirty low-bit samples
/// (1, 0x8000, 0xf000, 0xffff on several exponents), all against host_guard.
#[test]
fn kv_f16_exhaustive_clean_sweep_matches_host() {
    // Host twin pinned on sweep corners before any device need.
    let clean = |hi: u32| f32::from_bits(hi << 16);
    assert_eq!(host_guard(clean(0x477f)), Some(0x7bf8));
    assert_eq!(host_guard(clean(0x4780)), None, "overflow with clean bits");
    assert_eq!(host_guard(clean(0x3380)), Some(0x0001), "min f16 subnormal");
    assert_eq!(host_guard(clean(0x0001)), Some(0x0000), "underflow to zero");
    assert_eq!(host_guard(f32::from_bits(0x3f800001)), None, "dirty sample");
    if !testkit::device_present("kv f16 exhaustive sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let baseline = ctx.live_bytes();
    {
        let mut bits: Vec<u32> = (0..=u16::MAX).map(|hi| (hi as u32) << 16).collect();
        for hi in [0x0000u32, 0x0010, 0x0080, 0x3f80, 0x477f, 0x7f7f, 0x3380] {
            for dirty in [1u32, 0x8000, 0xf000, 0xffff] {
                bits.push((hi << 16) | dirty);
            }
        }
        let input: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let wants: Vec<Option<u16>> = input.iter().map(|&v| host_guard(v)).collect();
        let bytes = input.len() * 4;
        let half_bytes = input.len() * 2;
        let x = testkit::upload_f32(&ctx, &input);
        let y = testkit::upload_filled(&ctx, 0x5a, half_bytes);
        let flag = ctx.upload(&0u32.to_ne_bytes()).unwrap();
        ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                x: x.range(0, bytes).unwrap(),
                out: Some(y.range(0, half_bytes).unwrap()),
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                columns: input.len(),
            },
        )
        .expect("one sweep launch");
        stream.synchronize().expect("completion before health/read");
        let written = testkit::read_bytes(&ctx, &y, half_bytes);
        let mut flagged = false;
        for (i, ((&bits, &want), raw)) in bits
            .iter()
            .zip(&wants)
            .zip(written.as_chunks::<2>().0.iter())
            .enumerate()
        {
            let got = u16::from_ne_bytes(*raw);
            match want {
                Some(want) => assert_eq!(got, want, "sweep {i} of {bits:#010x}"),
                None => {
                    flagged = true;
                    assert_eq!(got, 0, "sweep reject {i} of {bits:#010x}");
                }
            }
        }
        assert!(flagged, "sweep must cross reject classes");
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 1u32.to_ne_bytes());
    }
    stream.synchronize().unwrap();
    assert_eq!(ctx.live_bytes(), baseline, "device allocations returned");
}

/// The Q8G32 shape: y is absent, a reject still sets the sticky word, and a
/// second launch onto the same word only ever ORs (sticky within the token).
#[test]
fn kv_f16_guard_only_mode_flags_without_writing() {
    if !testkit::device_present("kv f16 guard-only mode") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let baseline = ctx.live_bytes();
    {
        let rows = [(0x3f800000u32), (0x7f800000), (0x3f800000)];
        let input: Vec<f32> = rows.iter().map(|&bits| f32::from_bits(bits)).collect();
        let bytes = input.len() * 4;
        let x = testkit::upload_f32(&ctx, &input);
        let flag = ctx.upload(&0u32.to_ne_bytes()).unwrap();
        ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                x: x.range(0, bytes).unwrap(),
                out: None,
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                columns: input.len(),
            },
        )
        .expect("one guard-only launch");
        stream.synchronize().expect("completion before health/read");
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 1u32.to_ne_bytes());
        // An all-clean second launch must not clear the sticky word.
        let clean = testkit::upload_f32(&ctx, &[1.0f32, -1.0, 0.0, -0.0]);
        ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                x: clean.range(0, 16).unwrap(),
                out: None,
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                columns: 4,
            },
        )
        .expect("one clean guard-only launch");
        stream.synchronize().expect("completion before health/read");
        assert_eq!(
            testkit::read_bytes(&ctx, &flag, 4),
            1u32.to_ne_bytes(),
            "sticky word survives a clean pass"
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(ctx.live_bytes(), baseline, "device allocations returned");
}

/// The wrapper rejects a short output range and overlapping regions by its
/// documented error text (require_fit / require_apart in context/launch.rs)
/// before any native launch, and refuses out-of-range geometry.
#[test]
fn kv_f16_wrapper_rejects_bad_ranges() {
    if !testkit::device_present("kv f16 wrapper rejections") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let baseline = ctx.live_bytes();
    {
        let x = testkit::upload_f32(&ctx, &[1.0f32; 512]);
        let y = testkit::upload_filled(&ctx, 0, 512 * 2 - 2);
        let flag = ctx.upload(&0u32.to_ne_bytes()).unwrap();
        let short = ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                x: x.range(0, 512 * 4).unwrap(),
                out: Some(y.range(0, 512 * 2 - 2).unwrap()),
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                columns: 512,
            },
        );
        assert!(matches!(
            &short,
            Err(EngError::BadRequest(message))
                if message.contains("holds 1022 bytes but the call needs 1024")
        ));
        let overlap = ctx.kv_f16_rows(
            &stream,
            KvF16RowsCall {
                // The flag word inside the read region: must be refused.
                x: x.range(0, 512 * 4).unwrap(),
                out: None,
                flag: x.range(0, 4).unwrap(),
                rows: 1,
                columns: 512,
            },
        );
        assert!(matches!(
            &overlap,
            Err(EngError::BadRequest(message)) if message.contains("overlap")
        ));
        for (rows, columns) in [(0usize, 512usize), (65537, 512), (1, 0), (1, 1 << 23)] {
            let rejected = ctx.kv_f16_rows(
                &stream,
                KvF16RowsCall {
                    x: x.range(0, 512 * 4).unwrap(),
                    out: None,
                    flag: flag.range(0, 4).unwrap(),
                    rows,
                    columns,
                },
            );
            assert!(
                matches!(&rejected, Err(EngError::BadRequest(message))
                    if message.contains("KV F16 rows: invalid geometry or overflow")),
                "geometry {rows}x{columns} must be refused by text"
            );
        }
    }
    stream.synchronize().unwrap();
    assert_eq!(ctx.live_bytes(), baseline, "device allocations returned");
}
