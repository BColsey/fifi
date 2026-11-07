//! The decode step with the shared expert inside the routed launches, against the reference step,
//! on the real 4.00 pack (`FIFI_EXL3_CHECKPOINT`). `#[ignore]`: the 3090 held exclusively, in a
//! window with its own authorization.
//!
//! Each leg is a fresh step over the same loaded weights, run through the same positions, with
//! the logits row of EVERY position (the prompt's and the generated ones) read back. The legs run
//! as reference (off), merged (on), reference again (off), on the eager path (picks read on the
//! host) and on the captured path (picks read on the device, graph replayed). Two prompts give two
//! different routing patterns. Everything is compared as bytes against the first leg of its
//! prompt: the next token of every position, the logits row bytes and the sha256 of all rows. The
//! first difference stops the test and prints the first differing 32-bit word.
//!
//! The intermediates of one layer (shared expert output, routed outputs, combine output) are
//! compared by `exl3_moe_merge_device_tests`.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::{
    common_tests::{dims as step_dims, inventory},
    exl3_load::load_exl3_dir,
    exl3_step_device_tests::{EOS, HEAD_ROWS, PROMPT, SAMPLER_VOCAB, options, row_values},
    native_device_rig::device_present,
    resources::StepResources,
    rules::QwenRules,
    run_report::CaptureOutcome,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::project::cuda::{CudaProjector, Reserve},
};

/// Positions generated after the prompt.
const GENERATED: usize = 12;
/// A second prompt: token ids of a recorded decode of the pinned model (a different text than the
/// first prompt, hence different routing). They are valid ids of the model's vocabulary.
const PROMPT_B: [u32; 16] = [
    248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983, 2570,
];

struct Position {
    next: u32,
    row: Vec<u8>,
}

struct Leg {
    label: String,
    positions: Vec<Position>,
}

impl Leg {
    fn digest(&self) -> String {
        let mut hash = Sha256::new();
        for position in &self.positions {
            hash.update(&position.row);
        }
        format!("{:x}", hash.finalize())
    }
}

/// One fresh step: `merged` selects the block, `capture` the path of the picks.
fn leg(
    projector: &CudaProjector,
    weights: &crate::family::project::cuda::CudaWeights,
    prompt: &[u32],
    (merged, capture): (bool, bool),
) -> Result<Leg, EngError> {
    let context = projector.context();
    let mut options = options();
    options.context_tokens = prompt.len() + GENERATED + 1;
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS)?;
    let resources = StepResources::new(context, step_dims(), &inventory(), &options)?;
    let mut step = QwenResidentStep::new(context, &resources, weights, &options, rules)?;
    step.set_shared_in_routed(merged)?;
    // The merged block must really be in this step: the load-line suffix is the proof the
    // comparison of a merged leg is not reference against reference.
    let want = if merged {
        " exl3_shared_in_routed=on"
    } else {
        ""
    };
    assert_eq!(step.shared_in_routed_suffix(), want, "the block of the leg");
    step.set_capture_config(CaptureConfig {
        capture,
        section_marks: false,
    })?;
    let mut positions = Vec::new();
    let mut next = 0;
    for &id in prompt {
        next = step.run_token(id, true)?;
        positions.push(Position {
            next,
            row: step.row_bytes().to_vec(),
        });
    }
    for _ in 0..GENERATED {
        next = step.run_token(next, true)?;
        positions.push(Position {
            next,
            row: step.row_bytes().to_vec(),
        });
    }
    if capture {
        // The first token runs eager (the warm-up); a refused capture would make this comparison
        // eager against eager.
        assert_eq!(
            step.last_report().capture,
            CaptureOutcome::Captured,
            "the last token must have replayed the captured graph"
        );
    }
    Ok(Leg {
        label: format!(
            "{} {}",
            if merged { "merged" } else { "reference" },
            if capture { "captured" } else { "eager" }
        ),
        positions,
    })
}

/// Stops on the first difference between two legs, naming the position and the first word.
fn require_equal(reference: &Leg, other: &Leg, prompt: &str) {
    assert_eq!(reference.positions.len(), other.positions.len());
    for (at, (a, b)) in reference.positions.iter().zip(&other.positions).enumerate() {
        assert_eq!(
            a.next, b.next,
            "{prompt}: {} against {}: next token at position {at}",
            reference.label, other.label
        );
        if a.row != b.row {
            let (x, y) = (row_values(&a.row), row_values(&b.row));
            let word = x
                .iter()
                .zip(&y)
                .position(|(p, q)| p.to_bits() != q.to_bits())
                .expect("rows of equal length that differ");
            panic!(
                "{prompt}: {} against {}: logits row at position {at}, first differing word {word}: {:#010x} ({}) and {:#010x} ({})",
                reference.label,
                other.label,
                x[word].to_bits(),
                x[word],
                y[word].to_bits(),
                y[word]
            );
        }
    }
    assert_eq!(reference.digest(), other.digest());
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real EXL3 pack; run by name"]
fn shared_in_routed_step_is_byte_identical_off_on_off() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("shared in routed step"), "a GPU is required");
    let projector = CudaProjector::new().expect("projector");
    let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("pack");
    // The kernel the Qwen3.6 start registers; the merged block refuses any other.
    projector
        .context()
        .set_single_row_default(crate::kernels::cuda::resident::exl3::SingleRow::Wide(1));
    let mut routes = Vec::new();
    for (name, prompt) in [("prompt A", &PROMPT[..]), ("prompt B", &PROMPT_B[..])] {
        // off, on, off on both paths; the first leg is the reference of the prompt.
        let legs: Vec<Leg> = [
            (false, false),
            (true, false),
            (false, false),
            (false, true),
            (true, true),
            (false, true),
        ]
        .into_iter()
        .map(|mode| leg(&projector, &weights, prompt, mode).expect("a leg"))
        .collect();
        for other in &legs[1..] {
            require_equal(&legs[0], other, name);
            println!(
                "{name}: {} == {} over {} positions, sha256 {}",
                legs[0].label,
                other.label,
                other.positions.len(),
                other.digest()
            );
        }
        routes.push(legs[0].positions.iter().map(|p| p.next).collect::<Vec<_>>());
    }
    // Different prompts must have walked different token paths, or the two routing patterns are one.
    assert_ne!(routes[0], routes[1], "the two prompts decoded the same ids");
}
