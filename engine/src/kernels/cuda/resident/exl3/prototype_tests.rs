//! The bench-only prototypes of the id-indexed launch ([`IndexedPrototype`]) against the
//! production launch.
//!
//! - The DP4A decode of book 2 must give the bits `decode_state` gives for every one of the 65536
//!   states, and both must equal the host reference (`Codebook::Mul1`).
//! - Its byte sum must be exact for every byte value in every lane, with the 0x6400 offset and
//!   nothing above bit 15.
//! - Every prototype launch must write the bytes of the production launch of the same 20 calls:
//!   rotated inputs, partials, outputs and flag, for every codebook, several rates and an
//!   unaligned bank.
//! - An id outside the bank in the tail pairs (16 to 19, the 20-pair launch's own) skips exactly
//!   those pairs and raises the flag.
//!
//! A host without a device skips the device tests.

use std::ffi::OsStr;

use super::{Exl3IndexedCall, Exl3LinearCall, INDEXED_CAPACITY, IndexedPrototype};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};
use crate::weights::exl3::Codebook;

/// The multiplier of book 2's hash (`decode_state`).
const MUL1: u32 = 0x83dc_d12d;
const PAIRS: usize = 20;

#[test]
fn the_prototype_names_parse_strictly() {
    for prototype in IndexedPrototype::ALL {
        let name = OsStr::new(prototype.name());
        assert_eq!(IndexedPrototype::parse(Some(name)).unwrap(), prototype);
        let padded = format!(" {} ", prototype.name());
        assert_eq!(
            IndexedPrototype::parse(Some(OsStr::new(&padded))).unwrap(),
            prototype
        );
    }
    assert_eq!(
        IndexedPrototype::parse(None).unwrap(),
        IndexedPrototype::Off
    );
    for bad in ["", "DP4A", "on", "1", "pairs16", "dp4a,pairs20"] {
        assert!(
            IndexedPrototype::parse(Some(OsStr::new(bad))).is_err(),
            "{bad:?}"
        );
    }
}

#[test]
fn a_prototype_changes_only_the_launches_it_names() {
    use IndexedPrototype::*;
    for calls in 1..=PAIRS {
        let over = calls > INDEXED_CAPACITY;
        assert_eq!(Off.variant(calls), None);
        assert_eq!(Dp4a.variant(calls), Some(1));
        assert_eq!(Pairs20.variant(calls), over.then_some(2));
        assert_eq!(Both.variant(calls), Some(if over { 3 } else { 1 }));
    }
    for (prototype, capacity) in [(Off, 16), (Dp4a, 16), (Pairs20, 20), (Both, 20)] {
        assert_eq!(prototype.capacity(), capacity, "{prototype:?}");
    }
    assert_eq!(IndexedPrototype::default(), Off);
}

/// 1024 + s is an F16 integer for every byte sum s (0 to 1020), and its bits are 0x6400 + s:
/// the claim the DP4A decode rests on.
#[test]
fn the_offset_byte_sum_is_an_exact_f16_bit_pattern() {
    for sum in 0u16..=1020 {
        let value = half::f16::from_f64(1024.0 + f64::from(sum));
        assert_eq!(value.to_bits(), 0x6400 + sum, "sum {sum}");
        assert_eq!(value.to_f64(), 1024.0 + f64::from(sum), "sum {sum}");
    }
}

/// Words with every byte value in every lane, every byte repeated in all four lanes, the hash
/// boundaries, the hashed word of every state and random words.
fn probe_words() -> Vec<u32> {
    let mut words = Vec::new();
    for lane in 0..4 {
        words.extend((0u32..256).map(|v| v << (8 * lane)));
    }
    words.extend((0u32..256).map(|v| v * 0x0101_0101));
    words.extend([
        0,
        u32::MAX,
        0x8080_8080,
        0x7f7f_7f7f,
        0x7fff_ffff,
        0x8000_0000,
        0x0000_00ff,
        0xff00_0000,
        0x00ff_00ff,
        0xff00_ff00,
    ]);
    words.extend((0u32..=0xffff).map(|state| state.wrapping_mul(MUL1)));
    let mut rng = testkit::Rng::new(0x5347_00d4);
    words.extend((0..4096).map(|_| rng.next_u64() as u32));
    words
}

#[test]
fn the_dp4a_decode_equals_the_book2_decode_for_every_state_and_byte() {
    if !testkit::device_present("EXL3 DP4A decode probe") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let words = probe_words();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let input = testkit::upload_bytes(&ctx, &bytes);
    let sums = testkit::upload_filled(&ctx, 0xff, words.len() * 8);
    let decoded = testkit::upload_filled(&ctx, 0xff, 65536 * 4);
    ctx.exl3_book2_probe(
        &stream,
        &input.range(0, input.bytes()).unwrap(),
        &sums.range(0, sums.bytes()).unwrap(),
        &decoded.range(0, decoded.bytes()).unwrap(),
    )
    .expect("probe");
    stream.synchronize().unwrap();
    let sums: Vec<u32> = testkit::read_bytes(&ctx, &sums, words.len() * 8)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    for (i, &word) in words.iter().enumerate() {
        let want: u32 = word.to_le_bytes().iter().map(|&b| u32::from(b)).sum();
        assert_eq!(sums[2 * i], want, "masked byte sum of {word:#010x}");
        // Unsigned lanes: a byte of 0x80 or more adds its value, not a negative one, and the
        // offset never carries past bit 15.
        assert_eq!(
            sums[2 * i + 1],
            0x6400 + want,
            "DP4A byte sum of {word:#010x}"
        );
    }
    let decoded: Vec<u16> = testkit::read_bytes(&ctx, &decoded, 65536 * 4)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let mut differ = Vec::new();
    for state in 0..=u16::MAX {
        let host = Codebook::Mul1.decode_state(state);
        let s = usize::from(state);
        if decoded[2 * s] != host || decoded[2 * s + 1] != host {
            differ.push((state, host, decoded[2 * s], decoded[2 * s + 1]));
        }
    }
    assert!(
        differ.is_empty(),
        "{} states differ (state, host, decode_state, DP4A), first: {:x?}",
        differ.len(),
        &differ[..differ.len().min(8)]
    );
}

/// One bank of `members` random matrices of a rate and codebook, 20 calls on it and the buffers
/// they write.
struct Bank {
    book: Codebook,
    bits_twice: u8,
    k: usize,
    n: usize,
    members: usize,
    offset: usize,
    trellis: DeviceBuffer,
    scales: DeviceBuffer,
    x: DeviceBuffer,
}

/// What a launch wrote: rotated inputs, partials, outputs and the flag.
type Written = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

impl Bank {
    fn new(ctx: &CudaContext, rng: &mut testkit::Rng, kind: (Codebook, u8, usize)) -> Self {
        let (book, bits_twice, offset) = kind;
        let (k, n, members) = (512usize, 384usize, 12usize);
        let stride = Self::stride(k, n, bits_twice);
        let mut bytes = vec![0xa5; offset];
        bytes.extend((0..members * stride).map(|_| rng.next_u64() as u8));
        let scale_values: Vec<f32> = (0..members * (k + n))
            .map(|i| 0.4 + (i % 11) as f32 / 16.0)
            .collect();
        let x_values: Vec<f32> = (0..PAIRS * k)
            .map(|_| (rng.unit() as f32 - 0.5) * 8.0)
            .collect();
        Self {
            book,
            bits_twice,
            k,
            n,
            members,
            offset,
            trellis: testkit::upload_bytes(ctx, &bytes),
            scales: testkit::upload_f32(ctx, &scale_values),
            x: testkit::upload_f32(ctx, &x_values),
        }
    }

    fn stride(k: usize, n: usize, bits_twice: u8) -> usize {
        k * n / 256 * usize::from(bits_twice) * 16
    }

    /// The 20 calls of `ids` in launches of the context's capacity; what they wrote.
    fn run(&self, ctx: &CudaContext, stream: &DeviceStream, ids: &[i32; PAIRS]) -> Written {
        self.run_in(ctx, stream, ids, (ctx.exl3_indexed_capacity(), 1))
            .expect("the launches")
    }

    /// The 20 calls of `ids` (call p reads member `ids[p]`) in launches of `chunk` calls with
    /// `cols` tiles per block, each pair writing its own rows of buffers filled with 0xff; what
    /// they wrote, or the refusal of a launch.
    fn run_in(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        ids: &[i32; PAIRS],
        (chunk, cols): (usize, u32),
    ) -> Result<Written, crate::EngError> {
        let (k, n) = (self.k, self.n);
        let xh = testkit::upload_filled(ctx, 0xff, PAIRS * k * 4);
        let z = testkit::upload_filled(ctx, 0xff, PAIRS * n * 4);
        let out = testkit::upload_filled(ctx, 0xff, PAIRS * n * 4);
        let flag = testkit::upload_bytes(ctx, &0u32.to_le_bytes());
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let id_buffer = testkit::upload_bytes(ctx, &id_bytes);
        let stride = Self::stride(k, n, self.bits_twice);
        let calls: Vec<Exl3IndexedCall<'_>> = (0..PAIRS)
            .map(|p| Exl3IndexedCall {
                call: Exl3LinearCall {
                    trellis: self
                        .trellis
                        .range(self.offset, self.members * stride)
                        .unwrap(),
                    suh: self.scales.range(0, self.scales.bytes()).unwrap(),
                    svh: self
                        .scales
                        .range(k * 4, self.scales.bytes() - k * 4)
                        .unwrap(),
                    bias: None,
                    x: self.x.range(p * k * 4, k * 4).unwrap(),
                    xh: xh.range(p * k * 4, k * 4).unwrap(),
                    z: z.range(p * n * 4, n * 4).unwrap(),
                    out: out.range(p * n * 4, n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    rows: 1,
                    split_k: 2,
                    k,
                    n,
                    bits_twice: self.bits_twice,
                    book: self.book,
                    fused: false,
                },
                ids: id_buffer.range(0, id_buffer.bytes()).unwrap(),
                slot: p as u32,
                members: self.members as u32,
                trellis_stride: stride,
                scale_stride: k + n,
                offsets: None,
            })
            .collect();
        for launch in calls.chunks(chunk) {
            ctx.exl3_grouped_indexed(stream, launch, cols)?;
        }
        stream.synchronize()?;
        Ok((
            testkit::read_bytes(ctx, &xh, PAIRS * k * 4),
            testkit::read_bytes(ctx, &z, PAIRS * n * 4),
            testkit::read_bytes(ctx, &out, PAIRS * n * 4),
            testkit::read_bytes(ctx, &flag, 4),
        ))
    }
}

/// The bytes pair `p` owns in each of the rotated inputs, partials and outputs.
fn pair_spans(written: &Written, p: usize, k: usize, n: usize) -> [&[u8]; 3] {
    [
        &written.0[p * k * 4..(p + 1) * k * 4],
        &written.1[p * n * 4..(p + 1) * n * 4],
        &written.2[p * n * 4..(p + 1) * n * 4],
    ]
}

#[test]
fn indexed_prototypes_write_the_bytes_of_the_production_launch() {
    if !testkit::device_present("EXL3 indexed prototypes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0020);
    // Ten picks laid out as the block's gate and up pairs: pick s owns calls 2 s and 2 s + 1.
    let picks = [3i32, 0, 11, 7, 2, 9, 5, 10, 1, 6];
    let ids: [i32; PAIRS] = std::array::from_fn(|p| picks[p / 2]);
    // Book 2 at Flash-Next's rate (6), at the rates with their own bodies (8, 12), at an odd rate,
    // and with a bank 2 bytes into its buffer; the other books must be unchanged by the DP4A
    // prototype too.
    for kind in [
        (Codebook::Mul1, 6u8, 0usize),
        (Codebook::Mul1, 8, 0),
        (Codebook::Mul1, 12, 0),
        (Codebook::Mul1, 5, 0),
        (Codebook::Mul1, 6, 2),
        (Codebook::Mcg, 8, 0),
        (Codebook::ThreeInst, 12, 2),
    ] {
        let bank = Bank::new(&ctx, &mut rng, kind);
        let (k, n) = (bank.k, bank.n);
        ctx.set_indexed_prototype(IndexedPrototype::Off);
        let want = bank.run(&ctx, &stream, &ids);
        assert_eq!(want.3, [0; 4], "{kind:?}: production flag");
        // The production launches wrote every pair, the second launch's included, so equality
        // below shows the prototypes wrote them too.
        for p in 0..PAIRS {
            for span in pair_spans(&want, p, k, n) {
                assert!(
                    span.iter().any(|&b| b != 0xff),
                    "{kind:?}: pair {p} unwritten"
                );
            }
        }
        let mut bad = ids;
        bad[17] = bank.members as i32;
        bad[19] = -1;
        for prototype in [
            IndexedPrototype::Dp4a,
            IndexedPrototype::Pairs20,
            IndexedPrototype::Both,
        ] {
            ctx.set_indexed_prototype(prototype);
            let got = bank.run(&ctx, &stream, &ids);
            assert_eq!(got, want, "{kind:?} {prototype:?}");
            // Ids outside the bank in the tail: exactly those pairs are skipped, the flag is
            // raised, every other pair matches.
            let gapped = bank.run(&ctx, &stream, &bad);
            assert_eq!(gapped.3, 1u32.to_le_bytes(), "{kind:?} {prototype:?}: flag");
            for p in 0..PAIRS {
                let spans = pair_spans(&gapped, p, k, n);
                if p == 17 || p == 19 {
                    for span in spans {
                        assert!(
                            span.iter().all(|&b| b == 0xff),
                            "{kind:?} {prototype:?}: skipped pair {p} was written"
                        );
                    }
                } else {
                    assert_eq!(
                        spans,
                        pair_spans(&want, p, k, n),
                        "{kind:?} {prototype:?}: pair {p}"
                    );
                }
            }
        }
    }
    ctx.set_indexed_prototype(IndexedPrototype::Off);
}

/// One launch of 20 calls: refused by the production launch and the 16-pair DP4A prototype,
/// taken by the 20-pair prototypes with the bytes of 16 + 4 production launches. A prototype
/// refuses more than one tile per block.
#[test]
fn only_a_twenty_pair_prototype_takes_twenty_calls_in_one_launch() {
    if !testkit::device_present("EXL3 indexed prototype capacity") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0021);
    let bank = Bank::new(&ctx, &mut rng, (Codebook::Mul1, 6, 0));
    let ids: [i32; PAIRS] = std::array::from_fn(|p| (p * 5 % bank.members) as i32);
    let want = bank.run(&ctx, &stream, &ids);
    for prototype in [IndexedPrototype::Off, IndexedPrototype::Dp4a] {
        ctx.set_indexed_prototype(prototype);
        assert_eq!(ctx.exl3_indexed_capacity(), INDEXED_CAPACITY);
        assert!(
            bank.run_in(&ctx, &stream, &ids, (PAIRS, 1)).is_err(),
            "{prototype:?} took 20 calls"
        );
    }
    for prototype in [IndexedPrototype::Pairs20, IndexedPrototype::Both] {
        ctx.set_indexed_prototype(prototype);
        assert_eq!(ctx.exl3_indexed_capacity(), PAIRS);
        let got = bank.run_in(&ctx, &stream, &ids, (PAIRS, 1)).unwrap();
        assert_eq!(got, want, "{prototype:?}");
    }
    ctx.set_indexed_prototype(IndexedPrototype::Dp4a);
    assert!(bank.run_in(&ctx, &stream, &ids, (16, 2)).is_err());
    ctx.set_indexed_prototype(IndexedPrototype::Off);
}
