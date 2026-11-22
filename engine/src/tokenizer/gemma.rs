use super::TokenizerBundle;
use crate::EngError;
use std::path::Path;

/// Loads the tokenizer of a Gemma 4 model as it is: the files of `dir` when it holds a
/// `tokenizer.json`, else the vocabulary and chat template in the header of the GGUF beside the
/// weights. No revision is pinned: what the bundle read names it.
pub fn load(dir: &Path) -> Result<TokenizerBundle, EngError> {
    TokenizerBundle::load_observed(dir)
}
