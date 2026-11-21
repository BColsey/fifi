//! Optional wall-clock accounting for the native text executor.
use std::time::Instant;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn enabled() -> bool {
    std::env::var_os("FIFI_INKLING_COMPONENT_TIMING").is_some_and(|value| value == "1")
}

pub(super) fn measure<T>(total_ns: Option<&mut u128>, work: impl FnOnce() -> T) -> T {
    let Some(total_ns) = total_ns else {
        return work();
    };
    let start = Instant::now();
    let result = work();
    *total_ns += start.elapsed().as_nanos();
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn measured_and_disabled_scopes_return_the_operation_result() {
        let mut elapsed = 0;
        let mut calls = 0;
        let mut operation = || {
            calls += 1;
            Err::<(), _>("kept")
        };
        assert_eq!(super::measure(None, &mut operation), Err("kept"));
        assert_eq!(super::measure(Some(&mut elapsed), operation), Err("kept"));
        assert_eq!(calls, 2);
    }
}
