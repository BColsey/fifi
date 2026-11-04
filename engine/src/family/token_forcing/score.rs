//! Teacher-forced scoring: the reference model reads a generated sequence and judges every token
//! in the context the generator actually had.

use crate::EngError;
use crate::family::session::TokenStepper;
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, EngError>;

/// The f64 rounding allowed between a mismatch's margin and the top-two gap, in nats.
const MARGIN_SLACK: f64 = 1e-9;

/// The reference's judgement of one generated token.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TokenRecord {
    /// The index of the token in the generated sequence; 0 is the first generated token.
    pub(crate) position: usize,
    pub(crate) generated: u32,
    /// The reference's greedy choice. A tie goes to the lower id, as in the greedy sampler.
    pub(crate) argmax: u32,
    /// The best id other than `argmax`.
    pub(crate) runner_up: u32,
    /// `log p(generated)` in nats: the reference's f32 logits, normalised in f64.
    pub(crate) logprob: f64,
    /// `log p(argmax)` in nats.
    pub(crate) argmax_logprob: f64,
    /// `logit(argmax) - logit(runner_up)`: how close the reference itself is to a tie here.
    pub(crate) top2_gap: f64,
}

impl TokenRecord {
    pub(crate) fn matched(&self) -> bool {
        self.generated == self.argmax
    }

    /// The log-probability gap from the generated token to the reference's argmax; zero for a
    /// match.
    pub(crate) fn margin(&self) -> f64 {
        self.argmax_logprob - self.logprob
    }

    /// Checks that the numbers of a record agree with each other, as `from_logits` writes them.
    /// A record read from a file can contradict itself, for example a mismatch whose log-
    /// probability was raised to hide its margin.
    pub(crate) fn check(&self) -> std::result::Result<(), String> {
        let values = [self.logprob, self.argmax_logprob, self.top2_gap];
        if values.iter().any(|value| !value.is_finite()) {
            return Err("a value is not finite".into());
        }
        if self.runner_up == self.argmax {
            return Err(format!("id {} is both argmax and runner-up", self.argmax));
        }
        if self.argmax_logprob > 0.0 || self.logprob > self.argmax_logprob {
            return Err(format!(
                "log p(generated) {} and log p(argmax) {} are not ordered below zero",
                self.logprob, self.argmax_logprob
            ));
        }
        if self.top2_gap < 0.0 {
            return Err(format!("the top-two gap {} is negative", self.top2_gap));
        }
        if self.matched() {
            if self.margin() != 0.0 {
                return Err(format!("a matched token has margin {}", self.margin()));
            }
            return Ok(());
        }
        // A mismatched token is at best the runner-up. The slack covers the f64 rounding of
        // the two normalised log-probabilities; logits are f32, far coarser.
        if self.margin() + MARGIN_SLACK < self.top2_gap {
            return Err(format!(
                "a mismatch with margin {} sits above the runner-up at gap {}",
                self.margin(),
                self.top2_gap
            ));
        }
        // A tie goes to the lower id, so a mismatch at zero margin names a higher id.
        if self.margin() == 0.0 && self.generated < self.argmax {
            return Err(format!(
                "id {} ties argmax {} but would have won the tie",
                self.generated, self.argmax
            ));
        }
        Ok(())
    }

    /// Judges `generated` against one logits row of the reference. Only the first `vocab`
    /// entries count: the head can be wider than the tokenizer, and the extra ids have no text.
    pub(crate) fn from_logits(
        position: usize,
        logits: &[f32],
        generated: u32,
        vocab: usize,
    ) -> Result<Self> {
        let visible = visible_row(logits, vocab)?;
        let generated_logit = *visible.get(generated as usize).ok_or_else(|| {
            score_error(format!(
                "generated id {generated} at position {position} is outside the {vocab}-id vocabulary"
            ))
        })?;
        let (argmax, runner_up) = top_two(visible);
        let top = f64::from(visible[argmax]);
        let sum: f64 = visible
            .iter()
            .map(|&logit| (f64::from(logit) - top).exp())
            .sum();
        let log_normaliser = top + sum.ln();
        Ok(Self {
            position,
            generated,
            argmax: argmax as u32,
            runner_up: runner_up as u32,
            logprob: f64::from(generated_logit) - log_normaliser,
            argmax_logprob: top - log_normaliser,
            top2_gap: top - f64::from(visible[runner_up]),
        })
    }
}

/// The reference model as token forcing drives it. Both reads return the full logits row of the
/// next token.
pub(crate) trait ForcedModel {
    /// Clears the model state.
    fn reset(&mut self) -> Result<()>;
    /// Consumes a non-empty `context` through the model's prompt path and returns the row that
    /// follows its last token.
    fn prefill_logits(&mut self, context: &[u32]) -> Result<Vec<f32>>;
    /// Consumes one more token through the one-row step path and returns the row that follows it.
    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>>;
}

/// Any executor's model side as a reference: the prompt goes through `feed_rows`, the executor's
/// own chunked prompt path, and every later token through `logits`, its one-row step.
pub(crate) struct StepperModel<S>(pub(crate) S);

impl<S: TokenStepper> ForcedModel for StepperModel<S> {
    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }

    fn prefill_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        let (&last, rows) = context
            .split_last()
            .ok_or_else(|| score_error("a forced context needs at least one token".into()))?;
        if !rows.is_empty() {
            self.0.feed_rows(rows)?;
        }
        self.0.logits(last)
    }

    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.0.logits(token)
    }
}

/// Scores `generated` after `prompt`: the first row follows the prompt, read through the prompt
/// path, and row `i` follows `generated[i - 1]`, read through the step path. The last generated
/// token is never fed.
pub(crate) fn score_sequence(
    model: &mut dyn ForcedModel,
    prompt: &[u32],
    generated: &[u32],
    vocab: usize,
) -> Result<Vec<TokenRecord>> {
    if prompt.is_empty() {
        return Err(score_error("token forcing needs a non-empty prompt".into()));
    }
    let mut records = Vec::with_capacity(generated.len());
    let Some((&first, _)) = generated.split_first() else {
        return Ok(records);
    };
    model.reset()?;
    let row = model.prefill_logits(prompt)?;
    records.push(TokenRecord::from_logits(0, &row, first, vocab)?);
    for position in 1..generated.len() {
        let row = model.step_logits(generated[position - 1])?;
        records.push(TokenRecord::from_logits(
            position,
            &row,
            generated[position],
            vocab,
        )?);
    }
    Ok(records)
}

/// Greedy generation of `tokens` ids through the model's own two paths: the prompt path for the
/// prompt, the step path for every new token, ties to the lower id. Run on the scorer, this is
/// the reference execution of the gate, whose rows the scorer reads again unchanged.
pub(crate) fn greedy_generate(
    model: &mut dyn ForcedModel,
    prompt: &[u32],
    tokens: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    if prompt.is_empty() {
        return Err(score_error("token forcing needs a non-empty prompt".into()));
    }
    let mut generated = Vec::with_capacity(tokens);
    if tokens == 0 {
        return Ok(generated);
    }
    model.reset()?;
    let mut row = model.prefill_logits(prompt)?;
    loop {
        let (next, _) = top_two(visible_row(&row, vocab)?);
        generated.push(next as u32);
        if generated.len() == tokens {
            return Ok(generated);
        }
        row = model.step_logits(next as u32)?;
    }
}

/// SpecPilot's injected fault (section 6.2) around any model: from output position `from` on,
/// every `stride`-th id gains `bias`. A harness self-test: a candidate that carries it must be
/// rejected, and a task metric that reads only the start of each answer cannot see it.
///
/// It generates; it is not an audit model. Every prefill counts as the prompt and restarts at
/// output position 0, so one prefill of a long context would be read without the fault.
pub(crate) struct LogitBiasFault<M> {
    pub(crate) inner: M,
    pub(crate) from: usize,
    pub(crate) stride: usize,
    pub(crate) bias: f32,
    /// The output position of the next row: 0 after the prompt, plus one per step.
    position: usize,
}

impl<M> LogitBiasFault<M> {
    /// The fault of SpecPilot's sealed finals: bias 2 on every 7th id after 2,048 tokens.
    pub(crate) fn specpilot(inner: M) -> Self {
        Self {
            inner,
            from: 2048,
            stride: 7,
            bias: 2.0,
            position: 0,
        }
    }

    /// `row` as the fault reads it at output position `position`.
    pub(crate) fn bias_at(&self, position: usize, mut row: Vec<f32>) -> Vec<f32> {
        if position >= self.from {
            for logit in row.iter_mut().step_by(self.stride) {
                *logit += self.bias;
            }
        }
        row
    }

    /// What the fault does to the unbiased `row` at output position `position`: the token it
    /// chooses instead of the row's argmax, judged by that row, so the record's margin is the
    /// margin a scorer reads. `None` when the bias leaves the argmax chosen.
    pub(crate) fn departure_at(
        &self,
        position: usize,
        row: &[f32],
        vocab: usize,
    ) -> Result<Option<TokenRecord>> {
        let biased = self.bias_at(position, row.to_vec());
        let (choice, _) = top_two(visible_row(&biased, vocab)?);
        let record = TokenRecord::from_logits(position, row, choice as u32, vocab)?;
        Ok((!record.matched()).then_some(record))
    }

    fn corrupt(&self, row: Vec<f32>) -> Vec<f32> {
        self.bias_at(self.position, row)
    }
}

impl<M: ForcedModel> ForcedModel for LogitBiasFault<M> {
    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.inner.reset()
    }

    fn prefill_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        self.position = 0;
        let row = self.inner.prefill_logits(context)?;
        Ok(self.corrupt(row))
    }

    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.position += 1;
        let row = self.inner.step_logits(token)?;
        Ok(self.corrupt(row))
    }
}

/// The model's judgement of `generated[position]` when the prompt and every earlier generated
/// token go through the prompt path in one prefill: the same token in the same history as the
/// step path judged it, read through another legal execution.
pub(crate) fn prefill_record(
    model: &mut dyn ForcedModel,
    prompt: &[u32],
    generated: &[u32],
    position: usize,
    vocab: usize,
) -> Result<TokenRecord> {
    let &token = generated.get(position).ok_or_else(|| {
        score_error(format!(
            "position {position} is past the {} generated tokens",
            generated.len()
        ))
    })?;
    if prompt.is_empty() {
        return Err(score_error("token forcing needs a non-empty prompt".into()));
    }
    let mut context = Vec::with_capacity(prompt.len() + position);
    context.extend_from_slice(prompt);
    context.extend_from_slice(&generated[..position]);
    model.reset()?;
    let row = model.prefill_logits(&context)?;
    TokenRecord::from_logits(position, &row, token, vocab)
}

fn visible_row(logits: &[f32], vocab: usize) -> Result<&[f32]> {
    if vocab < 2 {
        return Err(score_error(format!(
            "a vocabulary of {vocab} ids has no runner-up"
        )));
    }
    let visible = logits.get(..vocab).ok_or_else(|| {
        score_error(format!(
            "a logits row of {} values is narrower than the {vocab}-id vocabulary",
            logits.len()
        ))
    })?;
    if let Some(index) = visible.iter().position(|logit| !logit.is_finite()) {
        return Err(score_error(format!("logit {index} is not finite")));
    }
    Ok(visible)
}

/// The indices of the largest and second-largest values; ties go to the lower index. The row has
/// at least two finite values.
fn top_two(row: &[f32]) -> (usize, usize) {
    let (mut best, mut second) = if row[1] > row[0] { (1, 0) } else { (0, 1) };
    for (index, &value) in row.iter().enumerate().skip(2) {
        if value > row[best] {
            second = best;
            best = index;
        } else if value > row[second] {
            second = index;
        }
    }
    (best, second)
}

fn score_error(message: String) -> EngError {
    EngError::BadRequest(format!("token forcing: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_measures_the_margin_from_the_logits() {
        let logits = [1.0_f32, 3.0, 2.5, -1.0, 9.0];
        // The fifth logit is outside the four-id vocabulary and is ignored.
        let record = TokenRecord::from_logits(7, &logits, 2, 4).unwrap();
        assert_eq!(record.argmax, 1);
        assert_eq!(record.runner_up, 2);
        assert!(!record.matched());
        assert!((record.margin() - 0.5).abs() < 1e-12);
        assert!((record.top2_gap - 0.5).abs() < 1e-12);
        let mass: f64 = logits[..4]
            .iter()
            .map(|&logit| (f64::from(logit) - f64::from(logits[1]) + record.argmax_logprob).exp())
            .sum();
        assert!((mass - 1.0).abs() < 1e-12, "{mass}");
        let matched = TokenRecord::from_logits(7, &logits, 1, 4).unwrap();
        assert!(matched.matched());
        assert_eq!(matched.margin(), 0.0);
    }

    #[test]
    fn ties_go_to_the_lower_id_like_the_greedy_sampler() {
        let record = TokenRecord::from_logits(0, &[0.5, 2.0, 2.0, 2.0], 2, 4).unwrap();
        assert_eq!(record.argmax, 1);
        assert_eq!(record.runner_up, 2);
        assert_eq!(record.top2_gap, 0.0);
        assert_eq!(record.margin(), 0.0);
        assert!(!record.matched());
        let first = TokenRecord::from_logits(0, &[3.0, 3.0], 1, 2).unwrap();
        assert_eq!((first.argmax, first.runner_up), (0, 1));
    }

    #[test]
    fn malformed_rows_and_ids_are_refused() {
        assert!(TokenRecord::from_logits(0, &[1.0, f32::NAN, 0.0], 0, 3).is_err());
        assert!(TokenRecord::from_logits(0, &[1.0, 2.0], 0, 3).is_err());
        assert!(TokenRecord::from_logits(0, &[1.0, 2.0, 3.0], 3, 3).is_err());
        assert!(TokenRecord::from_logits(0, &[1.0], 0, 1).is_err());
    }
}
