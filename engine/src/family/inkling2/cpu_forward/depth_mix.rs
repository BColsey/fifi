//! Depth residual blocks are immutable once a layer boundary publishes them.

use std::{cell::OnceCell, sync::Arc};

use super::{MIX_EPS, add, dot, rms_norm};

#[derive(Clone)]
pub(super) struct Stream {
    current: Vec<f32>,
    blocks: Blocks,
}

#[derive(Clone)]
enum Blocks {
    Raw(Vec<Vec<f32>>),
    Normalized {
        sealed: Vec<Arc<Vec<f32>>>,
        current: OnceCell<Arc<Vec<f32>>>,
    },
}

impl Stream {
    pub(super) fn new(current: Vec<f32>, cache: bool) -> Self {
        let blocks = if cache {
            Blocks::Normalized {
                sealed: Vec::new(),
                current: OnceCell::new(),
            }
        } else {
            Blocks::Raw(Vec::new())
        };
        Self { current, blocks }
    }

    pub(super) fn block_count(&self) -> usize {
        match &self.blocks {
            Blocks::Raw(rows) => rows.len(),
            Blocks::Normalized { sealed, .. } => sealed.len(),
        }
    }

    pub(super) fn current(&self) -> &[f32] {
        &self.current
    }

    pub(super) fn into_current(self) -> Vec<f32> {
        self.current
    }

    pub(super) fn add_current(&mut self, row: &[f32]) {
        if let Blocks::Normalized { current, .. } = &mut self.blocks {
            current.take();
        }
        add(&mut self.current, row, 1.0);
    }

    pub(super) fn seal(&mut self, next: Vec<f32>) {
        let mut previous = std::mem::replace(&mut self.current, next);
        match &mut self.blocks {
            Blocks::Raw(rows) => rows.push(previous),
            Blocks::Normalized { sealed, current } => {
                let normalized = current.take().unwrap_or_else(|| {
                    rms_norm(&mut previous, MIX_EPS);
                    Arc::new(previous)
                });
                sealed.push(normalized);
            }
        }
    }

    pub(super) fn mix(&self, first_block: usize, query: &[f32]) -> Vec<f32> {
        match &self.blocks {
            Blocks::Raw(rows) => {
                let scale = (query.len() as f32).sqrt().recip();
                let rows: Vec<_> = rows[first_block..]
                    .iter()
                    .rev()
                    .chain(std::iter::once(&self.current))
                    .map(|row| {
                        let normalized = normalize(row);
                        (dot(&normalized, query) * scale, normalized)
                    })
                    .collect();
                let peak = rows
                    .iter()
                    .map(|(score, _)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                weighted_mix(
                    rows.iter().map(|(score, row)| (*score, row.as_slice())),
                    query.len(),
                    peak,
                )
            }
            Blocks::Normalized { sealed, current } => {
                let last = current.get_or_init(|| Arc::new(normalize(&self.current)));
                let rows = sealed[first_block..]
                    .iter()
                    .rev()
                    .map(|row| row.as_slice())
                    .chain(std::iter::once(last.as_slice()));
                mix_rows(rows, query)
            }
        }
    }
}

fn normalize(row: &[f32]) -> Vec<f32> {
    let mut copy = row.to_vec();
    rms_norm(&mut copy, MIX_EPS);
    copy
}

fn mix_rows<'a>(rows: impl Iterator<Item = &'a [f32]> + Clone, query: &[f32]) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt().recip();
    static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let scores: Vec<_> = if *BATCH
        .get_or_init(|| std::env::var("FIFI_INKLING_HOST_DEPTH_SCORE4").as_deref() == Ok("1"))
    {
        score_rows4(rows.clone(), query, scale)
    } else {
        rows.clone().map(|row| dot(row, query) * scale).collect()
    };
    let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    weighted_mix(scores.into_iter().zip(rows), query.len(), peak)
}

fn score_rows4<'a>(
    mut rows: impl Iterator<Item = &'a [f32]>,
    query: &[f32],
    scale: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(rows.size_hint().0);
    loop {
        let group = std::array::from_fn(|_| rows.next());
        let [Some(a), Some(b), Some(c), Some(d)] = group else {
            scores.extend(
                group
                    .into_iter()
                    .flatten()
                    .map(|row| dot(row, query) * scale),
            );
            return scores;
        };
        // Independent row accumulators permit parallel execution without reassociating any dot.
        let mut sums = [-0.0; 4];
        for ((((q, a), b), c), d) in query.iter().zip(a).zip(b).zip(c).zip(d) {
            sums[0] += a * q;
            sums[1] += b * q;
            sums[2] += c * q;
            sums[3] += d * q;
        }
        scores.extend(sums.map(|sum| sum * scale));
    }
}

fn weighted_mix<'a>(
    rows: impl Iterator<Item = (f32, &'a [f32])>,
    width: usize,
    peak: f32,
) -> Vec<f32> {
    let mut mass = 0.0;
    let mut output = vec![0.0; width];
    for (score, row) in rows {
        let weight = (score - peak).exp();
        mass += weight;
        add(&mut output, row, weight);
    }
    for value in &mut output {
        *value /= mass;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_mix_can_exclude_the_first_core_boundary() {
        for cache in [false, true] {
            let mut stream = Stream::new(vec![0.0, 2.0], cache);
            stream.seal(vec![2.0, 0.0]);
            let both = stream.mix(0, &[0.0; 2]);
            let current = stream.mix(1, &[0.0; 2]);
            assert!((both[0] - both[1]).abs() < 1e-6);
            assert!((current[0] - std::f32::consts::SQRT_2).abs() < 1e-6);
            assert_eq!(current[1], 0.0);
        }
    }

    #[test]
    fn cached_depth_mix_keeps_exact_bits_across_branch_and_invalidation() {
        let input = vec![3.0, -6.0, 0.75, -0.0];
        let mut raw = Stream::new(input.clone(), false);
        let mut cached = Stream::new(input, true);
        for new in [[-1.0, -0.0, 0.0, 2.0], [0.0, -4.0, 0.5, 0.0]] {
            same_bits(&raw, &cached);
            raw.seal(new.to_vec());
            cached.seal(new.to_vec());
        }
        same_bits(&raw, &cached);
        let mut raw_fork = raw.clone();
        let mut cached_fork = cached.clone();
        raw_fork.add_current(&[0.0, 2.0, -8.0, 1.0]);
        cached_fork.add_current(&[0.0, 2.0, -8.0, 1.0]);
        same_bits(&raw_fork, &cached_fork);
        same_bits(&raw, &cached);
        raw_fork.add_current(&[0.0, 3.0, 0.0, -1.0]);
        cached_fork.add_current(&[0.0, 3.0, 0.0, -1.0]);
        raw_fork.seal(vec![0.0; 4]);
        cached_fork.seal(vec![0.0; 4]);
        same_bits(&raw_fork, &cached_fork);
    }

    #[test]
    fn interleaved_scores_preserve_scalar_reductions_and_tail_order() {
        let q = [0.5, 1.0, -3.0, 1.0, -0.0];
        let rows: Vec<_> = (0..11)
            .map(|i| [1e10, i as f32, -0.25, -5e9, 0.0])
            .collect();
        for n in 0..=rows.len() {
            let iter = rows[..n].iter().rev().map(|r| r.as_slice());
            let actual = score_rows4(iter.clone(), &q, 0.37);
            let expected = iter.map(|r| (dot(r, &q) * 0.37).to_bits());
            assert!(actual.into_iter().map(f32::to_bits).eq(expected));
        }
    }

    fn same_bits(raw: &Stream, cached: &Stream) {
        assert_eq!(raw.current(), cached.current());
        assert_eq!(raw.block_count(), cached.block_count());
        for first in 0..=raw.block_count() {
            for query in [[0.0; 4], [2.0, -1.0, 0.0, 3.0], [-5.0, 4.0, 0.0, 0.0]] {
                let bits = |stream: &Stream| {
                    stream
                        .mix(first, &query)
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(raw), bits(cached));
            }
        }
    }
}
