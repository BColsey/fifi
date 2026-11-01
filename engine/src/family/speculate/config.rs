//! The speculation settings of one engine entry.
//!
//! The member names follow the TabbyAPI / exllamav3 draft keys: `draft_mode`, `num_draft_tokens`,
//! `dynamic_draft_tokens`, `draft_confidence`, `draft_max_rows`, `hybrid_draft_tokens` and
//! `ngram_match_min`. `mtp_model_dir` names the directory that holds the MTP head. Everything is
//! off unless `draft_mode` says otherwise.

use crate::EngError;
use std::path::PathBuf;

/// Most drafts a window may carry.
pub(crate) const MAX_DRAFT_TOKENS: u32 = 16;
/// Fewest and most rows a window may have, the pending token included.
pub(crate) const MIN_DRAFT_ROWS: u32 = 2;
pub(crate) const MAX_DRAFT_ROWS: u32 = 64;
/// Longest verbatim passage a hybrid draft may copy.
pub(crate) const MAX_HYBRID_TOKENS: u32 = 256;

const DEFAULT_DRAFT_TOKENS: u32 = 2;
const DEFAULT_MAX_ROWS: u32 = 8;
const DEFAULT_CONFIDENCE: f64 = 0.4;
const DEFAULT_NGRAM_MATCH_MIN: u32 = 2;

/// Where the drafts come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DraftMode {
    /// No speculation: one row per step.
    #[default]
    Disabled,
    /// The tokens that followed the latest earlier occurrence of the context's tail.
    Ngram,
    /// The model's own MTP head, fed with the hidden state of the last verified row.
    Mtp,
}

/// The config text of `draft_mode`.
pub(crate) fn draft_mode_name(mode: DraftMode) -> &'static str {
    match mode {
        DraftMode::Disabled => "disabled",
        DraftMode::Ngram => "ngram",
        DraftMode::Mtp => "mtp",
    }
}

/// The members of the `qwen36` object that speculation reads, as the config parser holds them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SpecOptions {
    pub(crate) draft_mode: Option<String>,
    pub(crate) num_draft_tokens: Option<u32>,
    pub(crate) dynamic_draft_tokens: Option<bool>,
    pub(crate) draft_confidence: Option<f64>,
    pub(crate) draft_max_rows: Option<u32>,
    pub(crate) hybrid_draft_tokens: Option<u32>,
    pub(crate) ngram_match_min: Option<u32>,
    pub(crate) mtp_model_dir: Option<PathBuf>,
}

impl SpecOptions {
    /// True when no member is set.
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpecConfig {
    pub(crate) mode: DraftMode,
    /// The most drafts of one window.
    pub(crate) num_draft_tokens: usize,
    /// Shorten the window from the calibrated chance that each draft is accepted.
    pub(crate) dynamic: bool,
    /// The running product of acceptance chances below which a draft is not verified.
    pub(crate) confidence: f32,
    /// The most rows of one window, the pending token included.
    pub(crate) max_rows: usize,
    /// A verbatim match of at least this many tokens drafts that many tokens at once; 0 is off.
    pub(crate) hybrid_tokens: usize,
    /// The shortest tail that an n-gram draft needs to find earlier in the context.
    pub(crate) ngram_match_min: usize,
    pub(crate) mtp_model_dir: Option<PathBuf>,
}

impl Default for SpecConfig {
    /// Speculation off, every other member at its default.
    fn default() -> Self {
        Self {
            mode: DraftMode::Disabled,
            num_draft_tokens: DEFAULT_DRAFT_TOKENS as usize,
            dynamic: false,
            confidence: DEFAULT_CONFIDENCE as f32,
            max_rows: DEFAULT_MAX_ROWS as usize,
            hybrid_tokens: 0,
            ngram_match_min: DEFAULT_NGRAM_MATCH_MIN as usize,
            mtp_model_dir: None,
        }
    }
}

fn bad(message: String) -> EngError {
    EngError::BadRequest(format!("engine config: {message}"))
}

fn ranged(
    key: &str,
    value: Option<u32>,
    default: u32,
    low: u32,
    high: u32,
) -> Result<usize, EngError> {
    let value = value.unwrap_or(default);
    if !(low..=high).contains(&value) {
        return Err(bad(format!(
            "{key} must be between {low} and {high}, found {value}"
        )));
    }
    Ok(value as usize)
}

/// Parses `draft_mode` by exact match; absent is `disabled`.
pub(crate) fn draft_mode_of(text: Option<&str>) -> Result<DraftMode, EngError> {
    match text {
        None | Some("disabled") => Ok(DraftMode::Disabled),
        Some("ngram") => Ok(DraftMode::Ngram),
        Some("mtp") => Ok(DraftMode::Mtp),
        Some(other) => Err(bad(format!(
            "draft_mode must be disabled, ngram or mtp, got {other:?}"
        ))),
    }
}

fn confidence_of(value: Option<f64>) -> Result<f32, EngError> {
    let value = value.unwrap_or(DEFAULT_CONFIDENCE);
    if !(value > 0.0 && value < 1.0) {
        return Err(bad(format!(
            "draft_confidence must be above 0 and below 1, found {value}"
        )));
    }
    Ok(value as f32)
}

impl SpecConfig {
    /// Checks the members. Members of a disabled entry are still range-checked, so a typo shows
    /// before the mode is switched on; `mtp` needs `mtp_model_dir`, and the other modes refuse it.
    pub(crate) fn new(options: &SpecOptions) -> Result<Self, EngError> {
        let mode = draft_mode_of(options.draft_mode.as_deref())?;
        let ngram_match_min = ranged(
            "ngram_match_min",
            options.ngram_match_min,
            DEFAULT_NGRAM_MATCH_MIN,
            1,
            64,
        )?;
        let hybrid_tokens = match options.hybrid_draft_tokens {
            None | Some(0) => 0,
            Some(_) => ranged(
                "hybrid_draft_tokens",
                options.hybrid_draft_tokens,
                0,
                ngram_match_min as u32,
                MAX_HYBRID_TOKENS,
            )?,
        };
        let config = Self {
            mode,
            num_draft_tokens: ranged(
                "num_draft_tokens",
                options.num_draft_tokens,
                DEFAULT_DRAFT_TOKENS,
                1,
                MAX_DRAFT_TOKENS,
            )?,
            dynamic: options.dynamic_draft_tokens.unwrap_or(false),
            confidence: confidence_of(options.draft_confidence)?,
            max_rows: ranged(
                "draft_max_rows",
                options.draft_max_rows,
                DEFAULT_MAX_ROWS,
                MIN_DRAFT_ROWS,
                MAX_DRAFT_ROWS,
            )?,
            hybrid_tokens,
            ngram_match_min,
            mtp_model_dir: options.mtp_model_dir.clone(),
        };
        match (mode, &config.mtp_model_dir) {
            (DraftMode::Mtp, None) => Err(bad("draft_mode mtp needs mtp_model_dir".into())),
            (DraftMode::Disabled | DraftMode::Ngram, Some(_)) => {
                Err(bad("mtp_model_dir needs draft_mode mtp".into()))
            }
            _ => Ok(config),
        }
    }

    /// True when a step may run a window.
    pub(crate) fn enabled(&self) -> bool {
        self.mode != DraftMode::Disabled
    }

    /// The most drafts one window carries: the draft ceiling, held to the row cap.
    pub(crate) fn window_drafts(&self) -> usize {
        self.num_draft_tokens.min(self.max_rows - 1)
    }

    /// The most drafts a hybrid window carries: the passage length, held to the row cap.
    pub(crate) fn hybrid_drafts(&self) -> usize {
        self.hybrid_tokens.min(self.max_rows - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SpecOptions {
        SpecOptions::default()
    }

    #[test]
    fn an_empty_entry_is_disabled_with_the_defaults() {
        let config = SpecConfig::new(&options()).expect("defaults");
        assert_eq!(config, SpecConfig::default());
        assert!(!config.enabled());
        assert_eq!(config.window_drafts(), 2);
        assert_eq!(config.hybrid_drafts(), 0);
    }

    #[test]
    fn the_mode_parses_by_exact_match() {
        for (text, mode) in [
            ("disabled", DraftMode::Disabled),
            ("ngram", DraftMode::Ngram),
            ("mtp", DraftMode::Mtp),
        ] {
            assert_eq!(draft_mode_of(Some(text)).expect(text), mode);
            assert_eq!(draft_mode_name(mode), text);
        }
        for text in ["", "NGRAM", " mtp", "model", "off"] {
            let message = format!("{:?}", draft_mode_of(Some(text)).expect_err(text));
            assert!(
                message.contains("draft_mode must be disabled, ngram or mtp"),
                "{message}"
            );
        }
    }

    #[test]
    fn mtp_needs_its_directory_and_the_others_refuse_one() {
        let mut o = options();
        o.draft_mode = Some("mtp".into());
        assert!(SpecConfig::new(&o).is_err());
        o.mtp_model_dir = Some(PathBuf::from("/m"));
        assert!(SpecConfig::new(&o).expect("mtp").enabled());
        o.draft_mode = Some("ngram".into());
        assert!(SpecConfig::new(&o).is_err());
        o.draft_mode = None;
        assert!(SpecConfig::new(&o).is_err());
    }

    #[test]
    fn each_range_is_enforced_at_both_ends() {
        type Setter = fn(&mut SpecOptions, u32);
        let cases: [(&str, Setter, u32, u32); 4] = [
            (
                "num_draft_tokens",
                |o, v| o.num_draft_tokens = Some(v),
                1,
                MAX_DRAFT_TOKENS,
            ),
            (
                "draft_max_rows",
                |o, v| o.draft_max_rows = Some(v),
                MIN_DRAFT_ROWS,
                MAX_DRAFT_ROWS,
            ),
            ("ngram_match_min", |o, v| o.ngram_match_min = Some(v), 1, 64),
            (
                "hybrid_draft_tokens",
                |o, v| o.hybrid_draft_tokens = Some(v),
                2,
                MAX_HYBRID_TOKENS,
            ),
        ];
        for (key, set, low, high) in cases {
            for ok in [low, high] {
                let mut o = options();
                set(&mut o, ok);
                SpecConfig::new(&o).unwrap_or_else(|e| panic!("{key}={ok}: {e:?}"));
            }
            for refused in [low - 1, high + 1] {
                let mut o = options();
                set(&mut o, refused);
                let message = format!("{:?}", SpecConfig::new(&o).expect_err(key));
                assert!(message.contains(key), "{key}={refused}: {message}");
            }
        }
    }

    #[test]
    fn the_confidence_is_open_at_both_ends() {
        for value in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            let mut o = options();
            o.draft_confidence = Some(value);
            assert!(SpecConfig::new(&o).is_err(), "{value}");
        }
        let mut o = options();
        o.draft_confidence = Some(0.25);
        assert_eq!(SpecConfig::new(&o).expect("0.25").confidence, 0.25);
    }

    #[test]
    fn a_hybrid_passage_is_never_shorter_than_the_ngram_tail() {
        let mut o = options();
        o.ngram_match_min = Some(6);
        o.hybrid_draft_tokens = Some(5);
        assert!(SpecConfig::new(&o).is_err());
        o.hybrid_draft_tokens = Some(0);
        assert_eq!(SpecConfig::new(&o).expect("off").hybrid_tokens, 0);
    }

    #[test]
    fn windows_are_held_to_the_row_cap() {
        let mut o = options();
        o.num_draft_tokens = Some(8);
        o.draft_max_rows = Some(4);
        o.hybrid_draft_tokens = Some(32);
        let config = SpecConfig::new(&o).expect("config");
        assert_eq!(config.window_drafts(), 3);
        assert_eq!(config.hybrid_drafts(), 3);
        o.draft_max_rows = Some(64);
        let config = SpecConfig::new(&o).expect("config");
        assert_eq!((config.window_drafts(), config.hybrid_drafts()), (8, 32));
    }
}
