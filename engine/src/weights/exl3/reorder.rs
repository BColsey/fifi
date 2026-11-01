//! Reordering of the 128-wide groups of a stored matrix along one axis.
//!
//! The rotation of the format works on blocks of 128 input channels and blocks of 128 output
//! channels, and every 16 by 16 trellis tile is coded alone. A permutation of whole groups
//! therefore only moves tiles and scale entries: the reordered matrix decodes to the original
//! with its groups permuted, bit for bit, and no tile is re-encoded. A family uses it to match
//! a checkpoint whose head order differs from the order its kernels expect, where one head is
//! a whole number of groups.

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

use super::Matrix;

/// Channels in one rotation block.
pub const GROUP: usize = 128;
/// Trellis tiles along an axis in one group.
const TILES_PER_GROUP: usize = GROUP / 16;

/// The axis of the stored `[input, output]` matrix whose groups move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Input,
    Output,
}

impl Matrix {
    /// Returns the matrix whose group `g` along `axis` is group `order[g]` of this one.
    /// `order` must hold every group index exactly once.
    pub fn reorder_groups(&self, axis: Axis, order: &[usize]) -> Result<Matrix, EngError> {
        let [k, n] = self.shape();
        let groups = match axis {
            Axis::Input => k / GROUP,
            Axis::Output => n / GROUP,
        };
        check_order(order, groups)?;
        let trellis = self
            .trellis()
            .host_bytes()
            .ok_or_else(|| host_only("trellis"))?;
        let tile_bytes = usize::from(self.bitrate_twice()) * 8 * 2;
        let (k_tiles, n_tiles) = (k / 16, n / 16);
        let mut moved = Vec::new();
        moved
            .try_reserve_exact(trellis.len())
            .map_err(|error| EngError::MemoryPlan(format!("EXL3 reorder buffer: {error}")))?;
        match axis {
            Axis::Output => {
                let run = TILES_PER_GROUP * tile_bytes;
                for tile_row in 0..k_tiles {
                    let row = tile_row * n_tiles * tile_bytes;
                    for &old in order {
                        let from = row + old * run;
                        moved.extend_from_slice(&trellis[from..from + run]);
                    }
                }
            }
            Axis::Input => {
                let run = TILES_PER_GROUP * n_tiles * tile_bytes;
                for &old in order {
                    moved.extend_from_slice(&trellis[old * run..(old + 1) * run]);
                }
            }
        }
        let trellis = Tensor::try_from_host(
            self.trellis().dtype(),
            self.trellis().shape().to_vec(),
            moved,
        )?;
        let (input_side, output_side) = match axis {
            Axis::Input => (Some(order), None),
            Axis::Output => (None, Some(order)),
        };
        let suh = regroup(self.suh(), input_side)?;
        let svh = regroup(self.svh(), output_side)?;
        let bias = self
            .bias()
            .map(|bias| regroup(bias, output_side))
            .transpose()?;
        Ok(Matrix::from_parts(
            (trellis, suh, svh, bias),
            self.codebook(),
            self.shape(),
            self.bitrate_twice(),
        ))
    }
}

/// A vector of the matrix with its groups moved by `order`, or a plain copy without one.
fn regroup(vector: &Tensor, order: Option<&[usize]>) -> Result<Tensor, EngError> {
    let bytes = vector
        .host_bytes()
        .ok_or_else(|| host_only("scale or bias"))?;
    let width = match vector.dtype() {
        Dtype::F16 => 2,
        Dtype::F32 => 4,
        other => {
            return Err(EngError::Checkpoint(format!(
                "EXL3 reorder: scale or bias dtype {other:?} is not F16 or F32"
            )));
        }
    };
    let moved = match order {
        None => bytes.to_vec(),
        Some(order) => {
            let run = GROUP * width;
            let mut moved = Vec::with_capacity(bytes.len());
            for &old in order {
                moved.extend_from_slice(&bytes[old * run..(old + 1) * run]);
            }
            moved
        }
    };
    Tensor::try_from_host(vector.dtype(), vector.shape().to_vec(), moved)
}

pub fn check_order(order: &[usize], groups: usize) -> Result<(), EngError> {
    let mut seen = vec![false; groups];
    let valid = order.len() == groups
        && order
            .iter()
            .all(|&old| old < groups && !std::mem::replace(&mut seen[old], true));
    if valid {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "EXL3 reorder: the order is not a permutation of {groups} groups"
    )))
}

fn host_only(what: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 reorder: {what} requires host storage"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::exl3::Codebook;

    const K: usize = 384;
    const N: usize = 384;
    const BITS_TWICE: u8 = 6;

    fn pseudo(seed: u32, bytes: usize) -> Vec<u8> {
        let mut state = seed;
        (0..bytes)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    /// Small finite scales: the F16 values `i / 64` with the sign alternating.
    fn scales(values: usize, seed: usize) -> Vec<u8> {
        (0..values)
            .flat_map(|i| {
                let value = ((i * 7 + seed) % 61) as f32 / 64.0;
                half::f16::from_f32(if i % 2 == 0 { value } else { -value })
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }

    fn matrix(with_bias: bool) -> Matrix {
        let words = usize::from(BITS_TWICE) * 8;
        let trellis = Tensor::try_from_host(
            Dtype::I16,
            vec![(K / 16) as u32, (N / 16) as u32, words as u32],
            pseudo(7, K / 16 * (N / 16) * words * 2),
        )
        .unwrap();
        let vector = |values, seed| {
            Tensor::try_from_host(Dtype::F16, vec![values as u32], scales(values, seed)).unwrap()
        };
        let bias = with_bias.then(|| {
            let bytes = (0..N)
                .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
                .collect();
            Tensor::try_from_host(Dtype::F32, vec![N as u32], bytes).unwrap()
        });
        Matrix::from_parts(
            (trellis, vector(K, 3), vector(N, 11), bias),
            Codebook::Mul1,
            [K, N],
            BITS_TWICE,
        )
    }

    fn weights(matrix: &Matrix) -> Vec<f64> {
        let mut out = vec![0.0; K * N];
        matrix.reconstruct_fp64_into(&mut out).unwrap();
        out
    }

    #[test]
    fn output_groups_decode_to_the_original_columns_in_the_new_order() {
        let original = matrix(false);
        let order = [2, 0, 1];
        let moved = original.reorder_groups(Axis::Output, &order).unwrap();
        let (before, after) = (weights(&original), weights(&moved));
        for row in 0..K {
            for (new, &old) in order.iter().enumerate() {
                let to = row * N + new * GROUP;
                let from = row * N + old * GROUP;
                assert_eq!(after[to..to + GROUP], before[from..from + GROUP], "{row}");
            }
        }
    }

    #[test]
    fn input_groups_decode_to_the_original_rows_in_the_new_order() {
        let original = matrix(false);
        // Not an involution: a swapped direction would put group 1 where group 2 belongs.
        let order = [2, 0, 1];
        let moved = original.reorder_groups(Axis::Input, &order).unwrap();
        let (before, after) = (weights(&original), weights(&moved));
        for (new, &old) in order.iter().enumerate() {
            let to = new * GROUP * N;
            let from = old * GROUP * N;
            assert_eq!(after[to..to + GROUP * N], before[from..from + GROUP * N]);
        }
    }

    #[test]
    fn projection_follows_the_reordered_outputs() {
        let original = matrix(true);
        let order = [1, 2, 0];
        let moved = original.reorder_groups(Axis::Output, &order).unwrap();
        let input: Vec<f64> = (0..K).map(|i| ((i * 5 % 17) as f64 - 8.0) / 8.0).collect();
        let (mut before, mut after) = (vec![0.0; N], vec![0.0; N]);
        original.project_fp64_into(&input, 1, &mut before).unwrap();
        moved.project_fp64_into(&input, 1, &mut after).unwrap();
        for (new, &old) in order.iter().enumerate() {
            assert_eq!(
                after[new * GROUP..(new + 1) * GROUP],
                before[old * GROUP..(old + 1) * GROUP]
            );
        }
    }

    #[test]
    fn the_identity_order_keeps_every_component() {
        let original = matrix(true);
        let same = original
            .reorder_groups(Axis::Output, &[0, 1, 2])
            .unwrap()
            .reorder_groups(Axis::Input, &[0, 1, 2])
            .unwrap();
        assert_eq!(same.trellis().host_bytes(), original.trellis().host_bytes());
        assert_eq!(same.suh().host_bytes(), original.suh().host_bytes());
        assert_eq!(same.svh().host_bytes(), original.svh().host_bytes());
        assert_eq!(
            same.bias().unwrap().host_bytes(),
            original.bias().unwrap().host_bytes()
        );
        assert_eq!(same.shape(), original.shape());
        assert_eq!(same.bitrate_twice(), original.bitrate_twice());
    }

    #[test]
    fn an_order_that_is_not_a_permutation_is_refused() {
        let original = matrix(false);
        for (axis, order) in [
            (Axis::Output, vec![0, 1]),
            (Axis::Output, vec![0, 1, 1]),
            (Axis::Output, vec![0, 1, 3]),
            (Axis::Input, vec![0, 0, 1]),
            (Axis::Input, vec![0, 1, 2, 3]),
        ] {
            assert!(
                matches!(
                    original.reorder_groups(axis, &order),
                    Err(EngError::BadRequest(_))
                ),
                "{axis:?} {order:?}"
            );
        }
    }
}
