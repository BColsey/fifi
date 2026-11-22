//! One routed expert on the CPU for any GGUF type with a decoder: the kernel of the tier when a
//! file stores an expert stack in a type the Q4_K kernels do not read.
//!
//! The three projections are decoded a row at a time to f32 (`weights::dequantize_into`, the
//! oracle every device kernel is checked against) and dotted with the f32 activation, so the
//! result is the expert evaluated in f32 over the decoded weights. The dot accumulates into eight
//! lanes in index order and reduces in a fixed order, so it does not depend on how the compiler
//! vectorizes it.

use super::expert_cpu::{ExpertCpuError, ExpertShape};
use crate::weights::{GgufType, dequantize_into};

/// The raw slices of one expert, each in its own type's blocks.
pub struct AnyExpertSlices<'a> {
    pub gate: (GgufType, &'a [u8]),
    pub up: (GgufType, &'a [u8]),
    pub down: (GgufType, &'a [u8]),
}

/// Reusable buffers of [`expert_forward_any`]: the two projections, the activation and one decoded
/// row.
pub struct ExpertScratchAny {
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    row: Vec<f32>,
}

impl ExpertScratchAny {
    pub fn new(shape: ExpertShape) -> Self {
        Self {
            gate: vec![0.0; shape.inter],
            up: vec![0.0; shape.inter],
            act: vec![0.0; shape.inter],
            row: vec![0.0; shape.hidden.max(shape.inter)],
        }
    }
}

/// Bytes of `rows` rows of `k` values in `ty` blocks, `None` when `k` is off the block grid or the
/// size overflows.
pub fn slice_bytes(ty: GgufType, rows: usize, k: usize) -> Option<usize> {
    let (elems, bytes) = (
        usize::try_from(ty.block_elems()).ok()?,
        usize::try_from(ty.block_bytes()).ok()?,
    );
    if elems == 0 || !k.is_multiple_of(elems) {
        return None;
    }
    rows.checked_mul(k / elems)?.checked_mul(bytes)
}

/// Bytes of the gate, up and down slices of one expert of `shape` in `quant`, `None` when a row
/// length is off its type's block grid.
pub fn expert_slice_bytes(shape: ExpertShape, quant: [GgufType; 3]) -> Option<[usize; 3]> {
    Some([
        slice_bytes(quant[0], shape.inter, shape.hidden)?,
        slice_bytes(quant[1], shape.inter, shape.hidden)?,
        slice_bytes(quant[2], shape.hidden, shape.inter)?,
    ])
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, at) = a.as_chunks::<8>();
    let (b8, bt) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for l in 0..8 {
            lanes[l] += x[l] * y[l];
        }
    }
    let tail: f32 = at.iter().zip(bt).map(|(x, y)| x * y).sum();
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// `out[r] = w[r] . x` for the rows of `bytes`.
fn matvec(
    ty: GgufType,
    bytes: &[u8],
    k: usize,
    x: &[f32],
    row: &mut [f32],
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let row_bytes = slice_bytes(ty, 1, k).ok_or_else(|| {
        ExpertCpuError::BadShape(format!(
            "{} rows of {k} values are off the block grid",
            ty.name()
        ))
    })?;
    let row = &mut row[..k];
    for (chunk, y) in bytes.chunks_exact(row_bytes).zip(out.iter_mut()) {
        dequantize_into(ty, chunk, row)
            .map_err(|error| ExpertCpuError::BadShape(format!("{error:?}")))?;
        *y = dot(row, x);
    }
    Ok(())
}

/// [`expert_forward_any`] over the three raw slices of one expert in `quant`, building the
/// scratch in `scratch` on first use.
pub fn expert_forward_typed(
    shape: ExpertShape,
    quant: [GgufType; 3],
    [gate, up, down]: [&[u8]; 3],
    x: &[f32],
    scratch: &mut Option<ExpertScratchAny>,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let scratch = scratch.get_or_insert_with(|| ExpertScratchAny::new(shape));
    let slices = AnyExpertSlices {
        gate: (quant[0], gate),
        up: (quant[1], up),
        down: (quant[2], down),
    };
    expert_forward_any(shape, &slices, x, scratch, out)
}

/// Runs one expert: `out = down(silu(gate(x)) * up(x))`. Every length implied by `shape` is checked
/// before any computation, so on error `out` is untouched.
pub fn expert_forward_any(
    shape: ExpertShape,
    s: &AnyExpertSlices<'_>,
    x: &[f32],
    scratch: &mut ExpertScratchAny,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let ExpertShape { hidden, inter } = shape;
    if hidden == 0 || inter == 0 {
        return Err(ExpertCpuError::BadShape(format!(
            "expert shape {hidden}x{inter} has a zero extent"
        )));
    }
    let want = |ty: GgufType, rows: usize, k: usize| {
        slice_bytes(ty, rows, k).ok_or_else(|| {
            ExpertCpuError::BadShape(format!(
                "{} rows of {k} values are off the block grid",
                ty.name()
            ))
        })
    };
    for (what, (ty, bytes), rows, k) in [
        ("gate", s.gate, inter, hidden),
        ("up", s.up, inter, hidden),
        ("down", s.down, hidden, inter),
    ] {
        let wanted = want(ty, rows, k)?;
        if bytes.len() != wanted {
            return Err(ExpertCpuError::BadLength {
                what,
                got: bytes.len(),
                want: wanted,
            });
        }
    }
    for (what, got, wanted) in [("x", x.len(), hidden), ("out", out.len(), hidden)] {
        if got != wanted {
            return Err(ExpertCpuError::BadLength {
                what,
                got,
                want: wanted,
            });
        }
    }
    let ExpertScratchAny { gate, up, act, row } = scratch;
    matvec(s.gate.0, s.gate.1, hidden, x, row, gate)?;
    matvec(s.up.0, s.up.1, hidden, x, row, up)?;
    for ((a, g), u) in act.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *a = g / (1.0 + (-g).exp()) * u;
    }
    matvec(s.down.0, s.down.1, inter, act, row, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::expert_cpu::{ExpertScratch, ExpertSlices, expert_forward};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 2001) as f32 / 1000.0 - 1.0
        }
    }

    /// Encodes `values` in `ty` where an encoder exists: Q8_0 (and F32 through it is not used).
    fn q8_0_bytes(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let values: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
        crate::weights::encode_q8_0(&values).expect("encode")
    }

    fn q4k_bytes(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; rows * k / 256 * 144];
        for block in bytes.as_chunks_mut::<144>().0 {
            block.fill_with(|| (rng.next() & 0xff) as u8);
            // Small finite binary16 scales: 0x2c00 = 0.0625.
            block[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
        }
        bytes
    }

    const SHAPE: ExpertShape = ExpertShape {
        hidden: 512,
        inter: 256,
    };

    #[test]
    fn q4_k_experts_match_the_q4_k_kernel_to_rounding() {
        let mut rng = Lcg(7);
        let gate = q4k_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let up = q4k_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let down = q4k_bytes(&mut rng, SHAPE.hidden, SHAPE.inter);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.unit()).collect();
        let mut want = vec![0.0; SHAPE.hidden];
        expert_forward(
            SHAPE,
            &ExpertSlices {
                gate: &gate,
                up: &up,
                down: &down,
            },
            &x,
            &mut ExpertScratch::new(SHAPE),
            &mut want,
        )
        .expect("q4_k kernel");
        let mut got = vec![0.0; SHAPE.hidden];
        expert_forward_any(
            SHAPE,
            &AnyExpertSlices {
                gate: (GgufType::Q4K, &gate),
                up: (GgufType::Q4K, &up),
                down: (GgufType::Q4K, &down),
            },
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut got,
        )
        .expect("any kernel");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-3 * scale, "{a} vs {b}");
        }
    }

    #[test]
    fn mixed_types_run_and_a_wrong_length_is_refused_before_any_write() {
        let mut rng = Lcg(11);
        let gate = q8_0_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let up = q8_0_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let down = q4k_bytes(&mut rng, SHAPE.hidden, SHAPE.inter);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.unit()).collect();
        let slices = AnyExpertSlices {
            gate: (GgufType::Q80, &gate),
            up: (GgufType::Q80, &up),
            down: (GgufType::Q4K, &down),
        };
        let mut out = vec![0.0; SHAPE.hidden];
        expert_forward_any(
            SHAPE,
            &slices,
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut out,
        )
        .expect("mixed");
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|v| *v != 0.0));
        let short = AnyExpertSlices {
            gate: (GgufType::Q80, &gate[..gate.len() - 34]),
            ..slices
        };
        let mut untouched = vec![9.0; SHAPE.hidden];
        let error = expert_forward_any(
            SHAPE,
            &short,
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut untouched,
        )
        .expect_err("short gate");
        assert!(matches!(
            error,
            ExpertCpuError::BadLength { what: "gate", .. }
        ));
        assert!(untouched.iter().all(|v| *v == 9.0));
    }

    #[test]
    fn slice_bytes_follow_the_block_grid() {
        assert_eq!(slice_bytes(GgufType::Q4K, 512, 2048), Some(512 * 8 * 144));
        assert_eq!(slice_bytes(GgufType::Q80, 4, 64), Some(4 * 2 * 34));
        assert_eq!(slice_bytes(GgufType::Q4K, 4, 100), None);
    }
}
