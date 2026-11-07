//! The one log line of the conv prefill and narrow-block scan switches.

use std::ffi::OsStr;

/// Reports, at the first read of a switch, what its variable held and what the process will run, so a
/// misspelt or out-of-range value is visible in the server log. Nothing is printed when the variable is
/// unset (the default).
pub(crate) fn note(name: &str, raw: Option<&OsStr>, parsed: Option<usize>) {
    let Some(raw) = raw else {
        return;
    };
    match parsed {
        Some(value) => {
            eprintln!("gdn switch {name}={value}: set; it applies to launches of 64 rows or more")
        }
        None => {
            eprintln!("gdn switch {name}={raw:?}: not a value it accepts; the original kernel runs")
        }
    }
}
