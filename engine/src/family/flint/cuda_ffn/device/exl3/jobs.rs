//! Host plan of the job-table routed pass: which bank member each expert of a round is, and how
//! the round's rows are cut into jobs and launch batches.
//!
//! The device side lives in `wide.rs`. Everything here is arithmetic over offsets and row
//! counts, so it is tested without a device.

use super::bad;
use super::layer_plan::ExpertVisit;
use crate::{EngError, kernels::cuda::resident::exl3::Exl3Job};

/// Rows of one job: the narrowest CTA of the job-table kernel, so a job never wastes a wide one.
pub(super) const JOB_ROWS: usize = 64;

/// Rows of one launch batch. A batch keeps the rotated-input and partial scratch of a launch
/// bounded however many experts a round holds; it must hold the tallest tile of a plan.
pub(super) const BATCH_ROWS: usize = 2048;

/// The members of a bank that the experts of one round occupy, as the job-table kernel
/// addresses them: member `i` starts `i * stride` into each of the three components, counted
/// from the lowest expert of the round.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct MemberPlan {
    /// Byte offset of member 0 in the trellis, `suh` and `svh` components.
    pub(super) base: [usize; 3],
    /// Bytes between two members of each component; 0 when the round holds one expert.
    pub(super) stride: [usize; 3],
    /// The member of each expert, in the order of the offsets given.
    pub(super) member: Vec<u32>,
    /// Members the addressing spans, up to the highest used.
    pub(super) count: usize,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Places experts whose three components start at `offsets` (bytes) in one addressing. Slots of a
/// bank repeat one layout at a fixed stride, so the offsets of each component differ by multiples
/// of that stride; the greatest common divisor of the differences is a valid stride whatever
/// slots the experts occupy. The three components must agree on every expert's member, and no
/// two experts may share one.
pub(super) fn plan_members(offsets: &[[usize; 3]]) -> Result<MemberPlan, EngError> {
    if offsets.is_empty() {
        return Err(bad("a job-table round needs at least one expert"));
    }
    let base: [usize; 3] = std::array::from_fn(|c| offsets.iter().map(|o| o[c]).min().unwrap_or(0));
    let stride: [usize; 3] =
        std::array::from_fn(|c| offsets.iter().map(|o| o[c] - base[c]).fold(0, gcd));
    // Slots repeat one layout, so the three components advance by the same bytes per slot; the
    // kernel takes one stride for all of them.
    if offsets.len() > 1 && (stride[0] != stride[1] || stride[0] != stride[2]) {
        return Err(bad("expert components sit at different bank strides"));
    }
    let mut member = Vec::with_capacity(offsets.len());
    for expert in offsets {
        let index = |c: usize| (stride[c] != 0).then(|| (expert[c] - base[c]) / stride[c]);
        let first = index(0).unwrap_or(0);
        if (1..3).any(|c| index(c).unwrap_or(0) != first) {
            return Err(bad("expert components sit at different bank strides"));
        }
        member.push(u32::try_from(first).map_err(|_| bad("bank member index overflows"))?);
    }
    let mut seen = member.clone();
    seen.sort_unstable();
    if seen.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(bad("two experts of a round share a bank slot"));
    }
    if !stride[0].is_multiple_of(2) || !stride[1].is_multiple_of(4) || !stride[2].is_multiple_of(4)
    {
        return Err(bad("bank stride is not a whole number of elements"));
    }
    let count = seen.last().map_or(1, |&last| last as usize + 1);
    Ok(MemberPlan {
        base,
        stride,
        member,
        count,
    })
}

/// One launch of the job-table kernels: a run of consecutive sorted rows and the jobs over them.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Batch {
    /// Position of the batch's first row in the sorted pair order.
    pub(super) first: usize,
    pub(super) rows: usize,
    /// Jobs with `row0` counted from `first`.
    pub(super) jobs: Vec<Exl3Job>,
}

/// Cuts the visits of a round into batches of at most [`BATCH_ROWS`] rows and jobs of at most
/// [`JOB_ROWS`]. `member[i]` is the bank member of `visits[i]`. The rows of consecutive visits
/// must be back to back, which resident-first rounds are: tiles keep the sorted order.
pub(super) fn plan_batches(visits: &[ExpertVisit], member: &[u32]) -> Result<Vec<Batch>, EngError> {
    if visits.len() != member.len() {
        return Err(bad("a job-table round needs one member per expert"));
    }
    let mut batches: Vec<Batch> = Vec::new();
    let mut expected: Option<usize> = None;
    for (visit, &member) in visits.iter().zip(member) {
        for tile in &visit.tiles {
            if expected.is_some_and(|next| next != tile.first) {
                return Err(bad("the rows of a round are not back to back"));
            }
            expected = Some(tile.first + tile.rows);
            let full = batches
                .last()
                .is_none_or(|batch| batch.rows + tile.rows > BATCH_ROWS);
            if full {
                batches.push(Batch {
                    first: tile.first,
                    rows: 0,
                    jobs: Vec::new(),
                });
            }
            let batch = batches.last_mut().ok_or_else(|| bad("no batch"))?;
            let mut done = 0;
            while done < tile.rows {
                let rows = (tile.rows - done).min(JOB_ROWS);
                let row0 = tile.first + done - batch.first;
                batch.jobs.push(Exl3Job {
                    member,
                    row0: u32::try_from(row0).map_err(|_| bad("job row overflows"))?,
                    rows: rows as u32,
                });
                done += rows;
            }
            batch.rows += tile.rows;
        }
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::super::layer_plan::Tile;
    use super::*;

    const STRIDE: usize = 96;

    fn at(slot: usize) -> [usize; 3] {
        [
            1000 + slot * STRIDE,
            7000 + slot * STRIDE,
            9000 + slot * STRIDE,
        ]
    }

    #[test]
    fn a_round_reads_its_members_at_one_stride() {
        let slots = [5usize, 2, 9, 0, 14];
        let offsets: Vec<[usize; 3]> = slots.iter().map(|&slot| at(slot)).collect();
        let plan = plan_members(&offsets).unwrap();
        assert_eq!(plan.base, at(0));
        assert_eq!(plan.stride, [STRIDE; 3]);
        assert_eq!(plan.member, [5, 2, 9, 0, 14]);
        assert_eq!(plan.count, 15);
        // Addressing: base + member * stride lands on the expert's own offsets.
        for (offset, &member) in offsets.iter().zip(&plan.member) {
            for ((base, stride), want) in plan.base.iter().zip(&plan.stride).zip(offset) {
                assert_eq!(base + member as usize * stride, *want);
            }
        }
    }

    #[test]
    fn slots_two_apart_use_the_common_divisor() {
        let plan = plan_members(&[at(3), at(5), at(9)]).unwrap();
        assert_eq!(plan.stride, [2 * STRIDE; 3]);
        assert_eq!(plan.member, [0, 1, 3]);
        assert_eq!(plan.count, 4);
    }

    #[test]
    fn one_expert_is_member_zero_with_no_stride() {
        let plan = plan_members(&[at(7)]).unwrap();
        assert_eq!(plan.member, [0]);
        assert_eq!(plan.stride, [0, 0, 0]);
        assert_eq!(plan.count, 1);
    }

    #[test]
    fn components_that_disagree_or_collide_are_refused() {
        let mut skew = at(4);
        skew[2] += STRIDE;
        assert!(plan_members(&[at(0), skew]).is_err(), "svh at another slot");
        let mut three = [at(0), at(2), at(5)];
        three[2][1] += STRIDE;
        assert!(plan_members(&three).is_err(), "suh one slot off");
        assert!(
            plan_members(&[at(2), at(2)]).is_err(),
            "two experts, one slot"
        );
        assert!(plan_members(&[]).is_err());
        let odd = [[0, 0, 0], [3, 8, 8]];
        assert!(plan_members(&odd).is_err(), "a trellis stride of odd bytes");
    }

    fn visit(id: usize, first: usize, rows: usize) -> ExpertVisit {
        ExpertVisit {
            id,
            tiles: vec![Tile { first, rows }],
        }
    }

    #[test]
    fn jobs_cover_the_rows_once_in_batches_of_bounded_height() {
        let counts = [3usize, 64, 65, 200, 1, 144, 700, 900, 5, 31];
        let mut first = 10;
        let visits: Vec<ExpertVisit> = counts
            .iter()
            .enumerate()
            .map(|(id, &rows)| {
                let v = visit(id, first, rows);
                first += rows;
                v
            })
            .collect();
        // A tile is at most 144 rows in a real plan; split the large ones as the plan does.
        let visits: Vec<ExpertVisit> = visits
            .into_iter()
            .map(|v| ExpertVisit {
                id: v.id,
                tiles: v
                    .tiles
                    .iter()
                    .flat_map(|t| {
                        let mut out = Vec::new();
                        let mut done = 0;
                        while done < t.rows {
                            let rows = (t.rows - done).min(144);
                            out.push(Tile {
                                first: t.first + done,
                                rows,
                            });
                            done += rows;
                        }
                        out
                    })
                    .collect(),
            })
            .collect();
        let member: Vec<u32> = (0..visits.len() as u32).map(|m| m * 2).collect();
        let batches = plan_batches(&visits, &member).unwrap();
        let total: usize = counts.iter().sum();
        assert_eq!(batches.iter().map(|b| b.rows).sum::<usize>(), total);
        let mut covered = vec![0u32; total];
        for batch in &batches {
            assert!(batch.rows <= BATCH_ROWS && batch.rows > 0);
            for job in &batch.jobs {
                assert!((1..=JOB_ROWS as u32).contains(&job.rows));
                assert!(job.row0 as usize + job.rows as usize <= batch.rows);
                for row in 0..job.rows as usize {
                    covered[batch.first - 10 + job.row0 as usize + row] += 1;
                }
                assert_eq!(job.member % 2, 0);
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "every row is in exactly one job"
        );
        assert!(batches.len() >= 2, "{} rows need several batches", total);
        // A gap between two experts' rows is refused.
        let gapped = [visit(0, 0, 10), visit(1, 11, 10)];
        assert!(plan_batches(&gapped, &[0, 1]).is_err());
        assert!(plan_batches(&gapped, &[0]).is_err());
    }
}
