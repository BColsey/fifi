//! The sequence planner: how many sequences an engine can hold at once.
//!
//! The planner is a pure function of byte counts, so every family feeds it the numbers it already
//! computes for its single-session plan and the host tests cover the arithmetic without a device.
//!
//! The lone-user rule: sequence 1 keeps the plan it has today. The input `free` is what remains
//! after sequence 1 is resident or reserved, so extra sequences are funded only from slack and a
//! single client never pays for them. The count is then bounded by the family's kernel row limit
//! and by a policy ceiling, and an explicit request overrides the automatic count but never
//! exceeds what the arithmetic funds.

use crate::EngError;

/// What one more sequence costs beyond the shared weights and scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SeqCost {
    /// Bytes every sequence holds whatever its length: recurrent state, position words, the rows
    /// a window adds to the shared scratch.
    pub(crate) fixed: u64,
    /// Bytes one cached position costs.
    pub(crate) per_position: u64,
}

/// The inputs of one plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SeqInputs {
    /// Free device bytes with sequence 1 already resident or reserved.
    pub(crate) free: u64,
    /// Bytes that must stay free after the extra sequences.
    pub(crate) headroom: u64,
    pub(crate) cost: SeqCost,
    /// The positions each sequence can hold.
    pub(crate) context: u64,
    /// The most rows the family's kernels take in one window.
    pub(crate) row_cap: usize,
    /// The most sequences the host policy allows.
    pub(crate) ceiling: usize,
}

/// What bound the count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    Memory,
    RowCap,
    Ceiling,
    Requested,
}

impl Bound {
    fn name(self) -> &'static str {
        match self {
            Self::Memory => "free memory",
            Self::RowCap => "the kernel row cap",
            Self::Ceiling => "the policy ceiling",
            Self::Requested => "the requested count",
        }
    }
}

/// The planned count and the numbers behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SeqPlan {
    pub(crate) seqs: usize,
    /// Bytes one extra sequence charges.
    pub(crate) per_seq: u64,
    /// Bytes the extra sequences charge together.
    pub(crate) extra_bytes: u64,
    /// Free bytes that would remain if every extra sequence were allocated: a figure of the plan.
    pub(crate) free_after: u64,
    pub(crate) bound: Bound,
    inputs: SeqInputs,
}

impl SeqPlan {
    /// The numbers for the start line, in MiB. Sessions after the first are allocated when
    /// requests need them: a lone client holds one.
    pub(crate) fn describe(&self) -> String {
        let mib = |bytes: u64| bytes >> 20;
        format!(
            "sessions {} of at most {} (bound by {}); each extra session is {} MiB ({} bytes fixed + {} positions x {} bytes) and is allocated when a request needs it; {} extra for {} MiB; {} MiB free after the first session, {} MiB kept as headroom, {} MiB would be left if every extra session were allocated (a figure of the plan, not a reservation)",
            self.seqs,
            self.inputs.row_cap.min(self.inputs.ceiling),
            self.bound.name(),
            mib(self.per_seq),
            self.inputs.cost.fixed,
            self.inputs.context,
            self.inputs.cost.per_position,
            self.seqs.saturating_sub(1),
            mib(self.extra_bytes),
            mib(self.inputs.free),
            mib(self.inputs.headroom),
            mib(self.free_after),
        )
    }
}

fn per_sequence(inputs: &SeqInputs) -> Result<u64, EngError> {
    inputs
        .cost
        .per_position
        .checked_mul(inputs.context)
        .and_then(|positions| positions.checked_add(inputs.cost.fixed))
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| EngError::MemoryPlan("the per-sequence cost overflows or is zero".into()))
}

/// Plans the sequence count. `requested` is an explicit count: it must lie within the row cap and
/// the ceiling and be funded by `free`, otherwise the plan is a `MemoryPlan` refusal that names
/// the numbers. Without a request the count is as many sequences as the free bytes fund, at least
/// one, within the row cap and the ceiling.
pub(crate) fn plan_seqs(inputs: SeqInputs, requested: Option<usize>) -> Result<SeqPlan, EngError> {
    let per_seq = per_sequence(&inputs)?;
    let spare = inputs.free.saturating_sub(inputs.headroom);
    let funded = usize::try_from(spare / per_seq).unwrap_or(usize::MAX);
    let limit = inputs.row_cap.min(inputs.ceiling).max(1);
    let (seqs, bound) = match requested {
        Some(0) => {
            return Err(EngError::BadRequest(
                "the sequence count must be at least 1".into(),
            ));
        }
        Some(count) => {
            if count > limit {
                return Err(EngError::BadRequest(format!(
                    "{count} sequences exceed the limit of {limit} (kernel row cap {}, policy ceiling {})",
                    inputs.row_cap, inputs.ceiling
                )));
            }
            if count - 1 > funded {
                return Err(EngError::MemoryPlan(format!(
                    "{count} sequences need {} extra bytes of {per_seq} each beyond the first, but {} bytes are free after the first with {} kept as headroom",
                    (count as u64 - 1).saturating_mul(per_seq),
                    inputs.free,
                    inputs.headroom
                )));
            }
            (count, Bound::Requested)
        }
        None => {
            let wanted = funded.saturating_add(1);
            if wanted <= limit {
                (wanted, Bound::Memory)
            } else if inputs.row_cap <= inputs.ceiling {
                (limit, Bound::RowCap)
            } else {
                (limit, Bound::Ceiling)
            }
        }
    };
    let extra_bytes = (seqs as u64 - 1).saturating_mul(per_seq);
    Ok(SeqPlan {
        seqs,
        per_seq,
        extra_bytes,
        free_after: inputs.free.saturating_sub(extra_bytes),
        bound,
        inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn inputs(free: u64) -> SeqInputs {
        SeqInputs {
            free,
            headroom: GIB / 2,
            cost: SeqCost {
                fixed: 2 << 20,
                per_position: 30_464,
            },
            context: 32_768,
            row_cap: 8,
            ceiling: 8,
        }
    }

    #[test]
    fn a_lone_user_plan_is_one_sequence_whatever_is_free() {
        let plan = plan_seqs(inputs(0), None).expect("plan");
        assert_eq!(plan.seqs, 1);
        assert_eq!(plan.extra_bytes, 0);
        assert_eq!(plan.free_after, 0);
        let plan = plan_seqs(inputs(GIB / 2), None).expect("only headroom free");
        assert_eq!(plan.seqs, 1);
    }

    #[test]
    fn extra_sequences_are_funded_from_slack_above_the_headroom() {
        let per_seq = 30_464 * 32_768 + (2 << 20);
        let plan = plan_seqs(inputs(GIB / 2 + 3 * per_seq), None).expect("plan");
        assert_eq!(plan.seqs, 4);
        assert_eq!(plan.bound, Bound::Memory);
        assert_eq!(plan.extra_bytes, 3 * per_seq);
        assert_eq!(plan.free_after, GIB / 2);
        let short = plan_seqs(inputs(GIB / 2 + 3 * per_seq - 1), None).expect("plan");
        assert_eq!(short.seqs, 3, "one byte short of the fourth");
    }

    #[test]
    fn the_row_cap_and_the_ceiling_bound_the_count() {
        let rich = inputs(100 * GIB);
        let plan = plan_seqs(rich, None).expect("plan");
        assert_eq!((plan.seqs, plan.bound), (8, Bound::RowCap));
        let plan = plan_seqs(SeqInputs { ceiling: 4, ..rich }, None).expect("plan");
        assert_eq!((plan.seqs, plan.bound), (4, Bound::Ceiling));
        let plan = plan_seqs(SeqInputs { row_cap: 2, ..rich }, None).expect("plan");
        assert_eq!((plan.seqs, plan.bound), (2, Bound::RowCap));
    }

    #[test]
    fn a_request_is_honoured_when_funded_and_refused_with_numbers_when_not() {
        let per_seq = 30_464 * 32_768 + (2 << 20);
        let free = GIB / 2 + 2 * per_seq;
        let plan = plan_seqs(inputs(free), Some(3)).expect("funded");
        assert_eq!((plan.seqs, plan.bound), (3, Bound::Requested));
        let plan = plan_seqs(inputs(free), Some(1)).expect("one");
        assert_eq!(plan.seqs, 1);
        let error = plan_seqs(inputs(free), Some(4)).expect_err("unfunded");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected a memory plan refusal");
        };
        assert!(message.contains("4 sequences"), "{message}");
        assert!(message.contains(&per_seq.to_string()), "{message}");
        assert!(matches!(
            plan_seqs(inputs(100 * GIB), Some(9)),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            plan_seqs(inputs(100 * GIB), Some(0)),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn overflow_and_a_zero_cost_are_refused() {
        let huge = SeqInputs {
            cost: SeqCost {
                fixed: 0,
                per_position: u64::MAX,
            },
            ..inputs(GIB)
        };
        assert!(matches!(
            plan_seqs(huge, None),
            Err(EngError::MemoryPlan(_))
        ));
        let free_cost = SeqInputs {
            cost: SeqCost {
                fixed: 0,
                per_position: 0,
            },
            ..inputs(GIB)
        };
        assert!(matches!(
            plan_seqs(free_cost, None),
            Err(EngError::MemoryPlan(_))
        ));
        let edge = SeqInputs {
            free: u64::MAX,
            headroom: 0,
            cost: SeqCost {
                fixed: u64::MAX - 1,
                per_position: 0,
            },
            context: 1,
            ..inputs(0)
        };
        let plan = plan_seqs(edge, None).expect("a huge fixed cost plans one extra at most");
        assert_eq!(plan.seqs, 2);
    }

    #[test]
    fn the_start_line_carries_the_numbers_in_mib() {
        let plan = plan_seqs(inputs(5 * GIB), None).expect("plan");
        let line = plan.describe();
        assert!(line.contains(&format!("sessions {}", plan.seqs)), "{line}");
        assert!(line.contains("30464"), "{line}");
        assert!(line.contains("32768"), "{line}");
        assert!(
            line.contains(&format!(
                "{} MiB would be left if every extra session were allocated",
                plan.free_after >> 20
            )),
            "{line}"
        );
        assert!(
            line.contains(&format!("for {} MiB", plan.extra_bytes >> 20)),
            "{line}"
        );
        assert!(line.contains("allocated when a request needs it"), "{line}");
        assert!(
            !line.contains(&plan.extra_bytes.to_string()),
            "no raw byte counts: {line}"
        );
    }
}
