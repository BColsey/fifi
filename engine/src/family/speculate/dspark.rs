//! A GGUF DSpark block drafter using the target's embeddings and output head.
//! The shared speculative driver verifies candidates and samples committed tokens.

use half::bf16;

use super::{
    accept::Accepted,
    driver::{DraftOutput, Drafter, Proposal, TargetConditioning},
    window::Window,
};
use crate::{
    EngError,
    weights::dspark_gguf::{
        DsparkBlock, DsparkConfig, DsparkGguf, DsparkMarkovWeights, MarkovMatrix,
    },
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::{
    kernels::cuda::CudaContext,
    weights::dspark_gguf::{DeviceDraft, DeviceDraftOptions},
};

pub(crate) const CUDA_MARKOV_ENV: &str = "FIFI_INKLING_DSPARK_CUDA_MARKOV";

pub(crate) fn cuda_markov_requested() -> bool {
    std::env::var(CUDA_MARKOV_ENV).as_deref() == Ok("1")
}

struct MarkovHead<'a> {
    weights: &'a DsparkMarkovWeights,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    device: Option<(&'a DeviceDraft, &'a CudaContext)>,
}

impl MarkovHead<'_> {
    fn project(&self, kind: MarkovMatrix, input: &[f32]) -> Result<Vec<f32>, EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some((device, cuda)) = self.device
            && let Some(values) = device.project_markov(cuda, kind, input)?
        {
            return Ok(values);
        }
        Ok(self.weights.matrix(kind).project(input))
    }
}

pub(crate) struct DsparkDrafter {
    model: DsparkGguf,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    device: Option<DeviceDraft>,
}

impl DsparkDrafter {
    pub(crate) fn new(model: DsparkGguf) -> Self {
        Self {
            model,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            device: None,
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn enable_cuda(
        &mut self,
        cuda: &CudaContext,
        context: usize,
        reserve_floor: impl Fn(u64) -> u64,
    ) -> Result<(), EngError> {
        self.device = Some(DeviceDraft::load(
            cuda,
            &self.model,
            DeviceDraftOptions {
                context,
                markov: cuda_markov_requested(),
                reserve_floor,
            },
        )?);
        Ok(())
    }

    fn forward(
        &mut self,
        embedding: &[f32],
        target: &TargetConditioning<'_>,
    ) -> Result<DsparkBlock, EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(device) = &mut self.device {
            let cuda = target
                .output
                .cuda_context()
                .ok_or_else(|| invalid("CUDA draft requires the target CUDA context"))?;
            return device.forward(
                cuda,
                &self.model,
                embedding,
                target.features,
                target.lineage,
            );
        }
        self.model
            .forward(embedding, target.features, target.features.len())
    }
}

impl Drafter for DsparkDrafter {
    fn propose(&mut self, _history: &[u32], _room: usize) -> Result<Proposal, EngError> {
        Ok(Proposal::default())
    }

    fn propose_conditioned(
        &mut self,
        history: &[u32],
        room: usize,
        target: Option<TargetConditioning<'_>>,
    ) -> Result<Proposal, EngError> {
        let (Some(target), Some((&anchor, committed))) = (target, history.split_last()) else {
            return Ok(Proposal::default());
        };
        if target.features.len() != committed.len() {
            return Err(invalid("target features and committed history differ"));
        }
        let config = self.model.config();
        if room == 0
            || committed.is_empty()
            || config.context_length.saturating_sub(committed.len()) < config.block_size
        {
            return Ok(Proposal::default());
        }
        if anchor as usize >= config.vocab {
            return Err(invalid("anchor token is outside the drafter vocabulary"));
        }
        let embedding = target.output.embedding(anchor)?;
        let block = self.forward(&embedding, &target)?;
        let config = self.model.config();
        let tokens = select_block(
            MarkovHead {
                weights: self.model.markov(),
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                device: self.device.as_ref().zip(target.output.cuda_context()),
            },
            config,
            target.output,
            block,
            (anchor, room),
        )?;
        Ok(Proposal {
            tokens,
            scores: Vec::new(),
        })
    }

    fn observe(&mut self, _window: &Window, _accepted: &Accepted) -> Result<(), EngError> {
        // The next proposal reads the target's committed features. Candidate KV
        // and recurrent state live for one block.
        Ok(())
    }
}

fn select_block(
    head: MarkovHead<'_>,
    config: &DsparkConfig,
    output: &dyn DraftOutput,
    block: DsparkBlock,
    (mut previous, room): (u32, usize),
) -> Result<Vec<u32>, EngError> {
    let rank = config.markov_rank;
    let mut state = vec![0.0; rank];
    let mut joint_input = Vec::with_capacity(2 * (rank + config.width));
    let mut latent = vec![0.0; rank];
    let recurrent_context: Vec<_> = block.recurrent_context.iter().copied().map(round).collect();
    let mut tokens = Vec::with_capacity(room.min(block.hidden.len()));
    for mut hidden in block.hidden.into_iter().take(room) {
        hidden.iter_mut().for_each(|value| *value = round(*value));
        joint_input.clear();
        joint_input.extend_from_slice(&state);
        joint_input.extend(head.weights.previous.row(previous as usize));
        joint_input.extend_from_slice(&hidden);
        joint_input.extend_from_slice(&recurrent_context);

        let mut gates = head.project(MarkovMatrix::Joint, &joint_input)?;
        for (gate, &bias) in gates.iter_mut().zip(&head.weights.joint_bias) {
            *gate += bias;
        }
        update_markov(&mut state, &mut latent, &gates)?;
        let mut correction = head.project(MarkovMatrix::Output, &latent)?;
        for (value, &bias) in correction.iter_mut().zip(&head.weights.output_bias) {
            *value += bias;
        }

        let cap = config.final_logit_softcap;
        previous =
            match output.project_corrected_greedy([&hidden, &correction], config.vocab, cap)? {
                Some(token) => token,
                None => {
                    // The two tied projections have separate BF16 rounding boundaries.
                    let base = output.project(&hidden)?;
                    let correction = output.project(&correction)?;
                    choose_token(&base, &correction, config.vocab, cap)?
                }
            };
        tokens.push(previous);
    }
    Ok(tokens)
}

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn update_markov(state: &mut [f32], latent: &mut [f32], gates: &[f32]) -> Result<(), EngError> {
    let rank = state.len();
    if latent.len() != rank || gates.len() != 3 * rank || gates.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid recurrent gate output"));
    }
    let (gate, rest) = gates.split_at(rank);
    let (candidate, output) = rest.split_at(rank);
    for (i, (state, latent)) in state.iter_mut().zip(latent).enumerate() {
        // The checkpoint's recurrent head rounds each elementwise operation
        // to BF16. Keep the products separate even in the host reference.
        let keep = round((1.0 + (-round(gate[i])).exp()).recip());
        let next = round(round(candidate[i]).tanh());
        *state = round(round(keep * *state) + round(round(1.0 - keep) * next));
        *latent = round(round(output[i]).tanh());
    }
    Ok(())
}

pub(crate) fn softcap_logit_bf16(merged: f32, cap: f32) -> bf16 {
    bf16::from_f32(cap * (merged / cap).tanh())
}

pub(in crate::family) fn choose_token(
    base: &[f32],
    correction: &[f32],
    vocab: usize,
    cap: f32,
) -> Result<u32, EngError> {
    if vocab == 0
        || base.len() < vocab
        || correction.len() < vocab
        || !cap.is_finite()
        || cap <= 0.0
    {
        return Err(invalid("invalid target output shape or draft logit cap"));
    }
    let mut best = (0, f32::NEG_INFINITY);
    for (id, (&base, &bias)) in base.iter().zip(correction).take(vocab).enumerate() {
        let merged = round(round(base) + round(bias));
        if !merged.is_finite() {
            return Err(invalid("nonfinite corrected draft logit"));
        }
        let logit = softcap_logit_bf16(merged, cap).to_f32();
        // Strict comparison preserves the lowest vocabulary ID on a BF16 tie.
        if logit > best.1 {
            best = (id, logit);
        }
    }
    u32::try_from(best.0).map_err(|_| invalid("draft token ID overflows"))
}

fn invalid(message: &str) -> EngError {
    EngError::Backend {
        operation: "dspark_draft",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrence_keeps_prior_state_and_publishes_separate_latent() {
        let mut state = [0.25, -0.5];
        let mut latent = [0.0; 2];
        // Gates sigmoid(0) are 1/2. The first candidate saturates at 1.
        update_markov(&mut state, &mut latent, &[0.0, 0.0, 20.0, 0.0, 0.0, -20.0]).unwrap();
        assert_eq!(state, [0.625, -0.25]);
        assert_eq!(latent, [0.0, -1.0]);
        update_markov(&mut state, &mut latent, &[0.0; 6]).unwrap();
        assert_eq!(state, [0.3125, -0.125]);
        assert_eq!(latent, [0.0, 0.0]);
    }

    #[test]
    fn correction_can_change_the_choice_and_padding_is_never_sampled() {
        assert_eq!(
            choose_token(&[4., 3., 100.], &[0., 2., 0.], 2, 30.).unwrap(),
            1
        );
        assert_eq!(choose_token(&[4., 3.], &[0., 1.], 2, 30.).unwrap(), 0);
        assert!(choose_token(&[f32::NAN], &[0.], 1, 30.).is_err());
        assert!(update_markov(&mut [0.], &mut [0.], &[0.; 2]).is_err());
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "requires the published Inkling DSpark GGUF and CUDA"]
    fn published_markov_cuda_matches_cpu_recurrence() {
        use std::{path::Path, sync::Arc, time::Instant};

        let sidecar = std::env::var_os("FIFI_TEST_INKLING_DSPARK_GGUF")
            .expect("set FIFI_TEST_INKLING_DSPARK_GGUF");
        let model = DsparkGguf::open(Path::new(&sidecar)).unwrap();
        let cfg = model.config();
        let probe = |i| ((i % 19) as f32 - 9.0) / 19.0;
        let anchor: Vec<_> = (0..cfg.width).map(probe).collect();
        let taps: Arc<[f32]> = (0..cfg.width * cfg.target_layers.len())
            .map(probe)
            .collect();
        let block = model.forward(&anchor, &[taps], 1).unwrap();
        let cuda = CudaContext::new().unwrap();
        let device = DeviceDraft::load(
            &cuda,
            &model,
            DeviceDraftOptions {
                context: 32,
                markov: true,
                reserve_floor: |_| crate::expert_tier::VRAM_HEADROOM_BYTES,
            },
        )
        .unwrap();
        let weights = model.markov();
        let heads = [
            MarkovHead {
                weights,
                device: None,
            },
            MarkovHead {
                weights,
                device: Some((&device, &cuda)),
            },
        ];
        let mut states = [vec![0.0; cfg.markov_rank], vec![0.0; cfg.markov_rank]];
        let mut latent = states.clone();
        let context: Vec<_> = block.recurrent_context.into_iter().map(round).collect();
        let mut times = [std::time::Duration::ZERO; 2];
        let mut max_error = [0.0f64; 2];
        for (step, hidden) in block.hidden.iter().enumerate() {
            // Teacher forcing gives both projection backends the same token history.
            let previous = (8 + 2309 * step) % cfg.vocab;
            let mut inputs = states.clone();
            for input in &mut inputs {
                input.extend(weights.previous.row(previous));
                input.extend(hidden.iter().copied().map(round));
                input.extend_from_slice(&context);
            }
            let mut gates = [Vec::new(), Vec::new()];
            let mut correction = [Vec::new(), Vec::new()];
            for (arm, head) in heads.iter().enumerate() {
                let start = Instant::now();
                gates[arm] = head.project(MarkovMatrix::Joint, &inputs[arm]).unwrap();
                times[arm] += start.elapsed();
                for (value, &bias) in gates[arm].iter_mut().zip(&weights.joint_bias) {
                    *value += bias;
                }
                update_markov(&mut states[arm], &mut latent[arm], &gates[arm]).unwrap();
                let start = Instant::now();
                correction[arm] = head.project(MarkovMatrix::Output, &latent[arm]).unwrap();
                times[arm] += start.elapsed();
                for (value, &bias) in correction[arm].iter_mut().zip(&weights.output_bias) {
                    *value += bias;
                }
            }
            for (index, [reference, candidate]) in [gates, correction].into_iter().enumerate() {
                assert_eq!(reference.len(), candidate.len());
                let norm: f64 = reference.iter().map(|&x| f64::from(x).powi(2)).sum();
                let error: f64 = reference
                    .iter()
                    .zip(&candidate)
                    .map(|(&want, &got)| f64::from(want - got).powi(2))
                    .sum();
                max_error[index] = max_error[index].max((error / norm).sqrt());
            }
            assert_eq!(
                states[0], states[1],
                "BF16 recurrent state at proposal {step}"
            );
            assert_eq!(latent[0], latent[1], "BF16 latent at proposal {step}");
        }
        eprintln!(
            "published Markov projections, {} proposals: CPU {:.3} ms, CUDA {:.3} ms, joint/correction max relative RMS {:.3e}/{:.3e}",
            block.hidden.len(),
            times[0].as_secs_f64() * 1000.0,
            times[1].as_secs_f64() * 1000.0,
            max_error[0],
            max_error[1],
        );
        assert!(max_error.into_iter().all(|value| value < 2e-5));
    }
}
