//! Pure summary of the inputs of a distributed router fault.

pub(super) fn enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Summary {
    pub(super) experts_nonfinite: usize,
    pub(super) first_bad_expert: Option<usize>,
    pub(super) gate_finite: bool,
    pub(super) input_nonfinite: usize,
    pub(super) first_bad_input: Option<usize>,
}

pub(super) fn summarize(experts: &[f32], gate: f32, input: &[f32]) -> Summary {
    Summary {
        experts_nonfinite: experts.iter().filter(|x| !x.is_finite()).count(),
        first_bad_expert: experts.iter().position(|x| !x.is_finite()),
        gate_finite: gate.is_finite(),
        input_nonfinite: input.iter().filter(|x| !x.is_finite()).count(),
        first_bad_input: input.iter().position(|x| !x.is_finite()),
    }
}

#[cfg(test)]
mod tests {
    use super::{enabled, summarize};

    #[test]
    fn trace_requires_explicit_one() {
        use std::ffi::OsStr;
        assert!(!enabled(None));
        assert!(!enabled(Some(OsStr::new(""))));
        assert!(!enabled(Some(OsStr::new("0"))));
        assert!(!enabled(Some(OsStr::new("true"))));
        assert!(enabled(Some(OsStr::new("1"))));
    }

    #[test]
    fn identifies_invalid_experts_gate_and_input_without_exposing_values() {
        let report = summarize(
            &[1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            f32::NAN,
            &[0.0, f32::INFINITY],
        );
        assert_eq!(report.experts_nonfinite, 3);
        assert_eq!(report.first_bad_expert, Some(1));
        assert!(!report.gate_finite);
        assert_eq!(report.input_nonfinite, 1);
        assert_eq!(report.first_bad_input, Some(1));
    }

    #[test]
    fn clean_inputs_do_not_claim_a_gpu_normalization_predicate() {
        let report = summarize(&[1.0, -100.0], 0.0, &[0.0, 2.0]);
        assert_eq!(report.experts_nonfinite, 0);
        assert_eq!(report.first_bad_expert, None);
        assert!(report.gate_finite);
        assert_eq!(report.input_nonfinite, 0);
        assert_eq!(report.first_bad_input, None);
    }
}
