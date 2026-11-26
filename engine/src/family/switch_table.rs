//! A table of optional paths that an engine reads from the environment, each with its name and
//! default.
//!
//! `1` turns a path on and `0` turns it off; an unset name takes the default, and any other value
//! keeps the default and is reported once. The whole table is read once per process, on the first
//! question.

/// Declares a fieldless enum with one variant per row, the `TABLE` of its names and defaults, and
/// `on()`, which answers from the environment as the module header describes.
macro_rules! switch_table {
    ($(#[$meta:meta])* $vis:vis enum $switch:ident {
        $($variant:ident = $name:literal, $default:literal;)*
    }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        $vis enum $switch {
            $($variant,)*
        }

        const TABLE: &[($switch, &str, bool)] = &[$(($switch::$variant, $name, $default),)*];

        impl $switch {
            /// The environment name of the path.
            #[cfg(test)]
            fn name(self) -> &'static str {
                TABLE[self as usize].1
            }

            /// Whether the path is on: the environment value when it is `0` or `1`, else the
            /// default.
            $vis fn on(self) -> bool {
                static VALUES: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
                VALUES.get_or_init(|| {
                    TABLE
                        .iter()
                        .map(|&(_, name, default)| {
                            $crate::family::switch_table::read(name, std::env::var_os(name), default)
                        })
                        .collect()
                })[self as usize]
            }
        }
    };
}
pub(crate) use switch_table;

/// One row's value: `1` and `0` as given, an unset name the default, anything else the default
/// with a warning.
pub(crate) fn read(name: &str, value: Option<std::ffi::OsString>, default: bool) -> bool {
    match value.as_deref().and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => default,
        Some("1") => true,
        Some("0") => false,
        _ => {
            eprintln!(
                "{name} takes 0 or 1; keeping the default ({})",
                u8::from(default)
            );
            default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::read;

    #[test]
    fn a_value_other_than_zero_or_one_keeps_the_default() {
        let name = "FIFI_SWITCH_TABLE_TEST";
        assert!(read(name, None, true));
        assert!(!read(name, None, false));
        assert!(read(name, Some("1".into()), false));
        assert!(!read(name, Some("0".into()), true));
        assert!(read(name, Some("yes".into()), true));
        assert!(!read(name, Some("".into()), false));
    }
}
