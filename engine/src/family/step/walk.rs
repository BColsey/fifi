//! The feed of a step: tokens in pieces of the step's chunk size, with the bounds and the head
//! rows. A model supplies the chunk itself.

use crate::EngError;

/// Which rows of a fed chunk run through the output head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeadRows {
    None,
    Last,
    #[cfg(test)]
    All,
}

/// A model's forward pass over one chunk of token rows.
pub(crate) trait ChunkStep {
    /// The name of the model in refusals.
    fn family(&self) -> &'static str;
    /// The most rows of one chunk.
    fn chunk_rows(&self) -> usize;
    /// Positions the state holds.
    fn cells(&self) -> usize;
    fn vocab(&self) -> usize;
    /// Runs `tokens` (at most [`Self::chunk_rows`]) at positions `pos0..` through every layer.
    fn run_chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError>;
    /// The logits of row `row` of the chunk just run.
    fn head_row(&self, row: usize) -> Result<Vec<f32>, EngError>;
    /// Waits for the queued work of the step.
    fn drain(&self) -> Result<(), EngError>;
}

/// Feeds `tokens` at positions `pos0..`, in chunks. Returns the logits of the rows `head` names
/// (one row of `vocab` values each).
pub(crate) fn feed<S: ChunkStep>(
    step: &mut S,
    tokens: &[u32],
    pos0: usize,
    head: HeadRows,
) -> Result<Vec<Vec<f32>>, EngError> {
    let family = step.family();
    if tokens.is_empty() {
        return Err(EngError::BadRequest(format!(
            "{family}: a feed needs a token"
        )));
    }
    if pos0
        .checked_add(tokens.len())
        .is_none_or(|end| end > step.cells())
    {
        return Err(EngError::ContextLimit(format!(
            "{family}: {} tokens at position {pos0} exceed the {} cached positions",
            tokens.len(),
            step.cells()
        )));
    }
    if let Some(&token) = tokens.iter().find(|&&t| t as usize >= step.vocab()) {
        return Err(EngError::BadRequest(format!(
            "{family}: token {token} is outside the vocabulary of {}",
            step.vocab()
        )));
    }
    let chunk = step.chunk_rows();
    let mut logits = Vec::new();
    let mut done = 0;
    while done < tokens.len() {
        let piece = &tokens[done..(done + chunk).min(tokens.len())];
        let last_piece = done + piece.len() == tokens.len();
        step.run_chunk(piece, pos0 + done)?;
        match head {
            HeadRows::None => {}
            HeadRows::Last if !last_piece => {}
            HeadRows::Last => logits.push(step.head_row(piece.len() - 1)?),
            #[cfg(test)]
            HeadRows::All => {
                for row in 0..piece.len() {
                    logits.push(step.head_row(row)?);
                }
            }
        }
        done += piece.len();
    }
    if head == HeadRows::None {
        step.drain()?;
    }
    Ok(logits)
}
