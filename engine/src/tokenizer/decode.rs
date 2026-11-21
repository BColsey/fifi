use super::encode::Tokenizer;
use crate::EngError;
use std::sync::Arc;

pub struct IncrementalDecoder {
    tokenizer: Arc<tokenizers::Tokenizer>,
    vocab_size: u32,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    preserve_inkling_sections: bool,
}

impl IncrementalDecoder {
    pub fn new(tok: &Tokenizer) -> Self {
        Self {
            tokenizer: tok.shared(),
            vocab_size: tok.vocab_size(),
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            preserve_inkling_sections: false,
        }
    }

    pub(crate) fn inkling_chat(tok: &Tokenizer) -> Self {
        Self {
            preserve_inkling_sections: true,
            ..Self::new(tok)
        }
    }

    pub fn push(&mut self, id: u32) -> Result<String, EngError> {
        if id >= self.vocab_size {
            return Err(EngError::Checkpoint(
                "token id outside tokenizer vocab".to_string(),
            ));
        }
        if self.preserve_inkling_sections {
            use super::toolcall::{INKLING_END, INKLING_TEXT, INKLING_THINKING};
            let section = match id {
                200_004 => Some(INKLING_TEXT),
                200_008 => Some(INKLING_THINKING),
                200_010 => Some(INKLING_END),
                _ => None,
            };
            if let Some(section) = section {
                let mut tail = self.finish()?;
                tail.push_str(section);
                return Ok(tail);
            }
        }
        let chunk = tokenizers::step_decode_stream(
            &self.tokenizer,
            id,
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| EngError::Checkpoint(format!("incremental decode failed: {error}")))?;
        Ok(chunk.unwrap_or_default())
    }

    #[cfg(test)]
    pub(super) fn tokenizer_addr(&self) -> *const tokenizers::Tokenizer {
        Arc::as_ptr(&self.tokenizer)
    }

    /// Drains the text held back at a token boundary and resets the decoder.
    ///
    /// An incomplete trailing byte sequence becomes U+FFFD, as in the
    /// reference `errors="replace"` decode. A decode failure, or text that no
    /// longer extends the text already sent, returns `Checkpoint`: the stream
    /// never repeats sent text.
    pub fn finish(&mut self) -> Result<String, EngError> {
        // Take the state first, so every path below leaves the decoder reset.
        let ids = std::mem::take(&mut self.ids);
        let prefix = std::mem::take(&mut self.prefix);
        self.prefix_index = 0;
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|error| EngError::Checkpoint(format!("incremental decode failed: {error}")))?;
        text.strip_prefix(prefix.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                EngError::Checkpoint(
                    "incremental decode no longer extends the text already sent (prefix mismatch)"
                        .to_string(),
                )
            })
    }
}
