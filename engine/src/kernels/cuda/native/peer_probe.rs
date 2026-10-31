//! Raw FFI to the device-selection and peer-access entries of
//! `native/peer_probe.cu` (declared in `native/abi_peer.h`). Child module of
//! `native`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants every wrapper here relies on:
//! - The externs take device ordinals by value and at most one out pointer to a
//!   live, writable `c_int`; the adapter keeps no pointer after it returns.
//! - No raw pointer and no unsafe function leaves this module.
//! - Every call runs under the process dispatch guard; the safe callers in
//!   `mod.rs` and `context` enter it. `set_device` and `peer_enable` change the
//!   calling thread's current device, which is why they must never run outside
//!   the guard: another thread's launches would observe the wrong device.

use std::ffi::c_int;

use super::STATUS_OK;

unsafe extern "C" {
    #[link_name = "inference_cuda_set_device"]
    fn set_device_raw(device: c_int) -> c_int;
    #[cfg(test)]
    #[link_name = "inference_cuda_get_device"]
    fn get_device_raw(out_device: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_device_cc"]
    fn device_cc_raw(device: c_int, out_major: *mut c_int, out_minor: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_peer_can_access"]
    fn peer_can_access_raw(device: c_int, peer: c_int, out_can: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_peer_enable"]
    fn peer_enable_raw(device: c_int, peer: c_int) -> c_int;
}

/// Makes `device` the calling thread's current device, or the native status.
pub(in crate::kernels::cuda) fn set_device(device: c_int) -> Result<(), c_int> {
    // SAFETY: the entry takes the ordinal by value and touches no host memory.
    let status = unsafe { set_device_raw(device) };
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(status)
    }
}

/// The calling thread's current device ordinal, or the native status. Test
/// support for the selection cache until a production reader exists.
#[cfg(test)]
pub(in crate::kernels::cuda) fn get_device() -> Result<c_int, c_int> {
    let mut device: c_int = 0;
    // SAFETY: `device` is a live, writable c_int for the whole call; the
    // adapter keeps no pointer after it returns.
    let status = unsafe { get_device_raw(&mut device) };
    if status == STATUS_OK {
        Ok(device)
    } else {
        Err(status)
    }
}

/// The `(major, minor)` compute capability of `device`, or the native status.
pub(in crate::kernels::cuda) fn device_cc(device: c_int) -> Result<(c_int, c_int), c_int> {
    let (mut major, mut minor): (c_int, c_int) = (0, 0);
    // SAFETY: both out values are live, writable c_ints for the whole call.
    let status = unsafe { device_cc_raw(device, &mut major, &mut minor) };
    if status == STATUS_OK {
        Ok((major, minor))
    } else {
        Err(status)
    }
}

/// Whether `device` may directly access `peer`'s memory, or the native status.
pub(in crate::kernels::cuda) fn peer_can_access(device: c_int, peer: c_int) -> Result<bool, c_int> {
    let mut can: c_int = 0;
    // SAFETY: `can` is a live, writable c_int for the whole call.
    let status = unsafe { peer_can_access_raw(device, peer, &mut can) };
    if status == STATUS_OK {
        Ok(can != 0)
    } else {
        Err(status)
    }
}

/// Enables direct access from `device` to `peer`, or the native status.
/// Leaves `device` as the calling thread's current device.
pub(in crate::kernels::cuda) fn peer_enable(device: c_int, peer: c_int) -> Result<(), c_int> {
    // SAFETY: the entry takes both ordinals by value and touches no host memory.
    let status = unsafe { peer_enable_raw(device, peer) };
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    use super::super::STATUS_INVALID_ARGUMENT;
    use super::*;
    use crate::kernels::cuda::testkit::device_present;

    #[test]
    fn a_negative_ordinal_is_an_invalid_argument() {
        assert_eq!(set_device(-1), Err(STATUS_INVALID_ARGUMENT));
        assert_eq!(device_cc(-1), Err(STATUS_INVALID_ARGUMENT));
        assert_eq!(peer_can_access(0, -1), Err(STATUS_INVALID_ARGUMENT));
        assert_eq!(peer_enable(-1, 0), Err(STATUS_INVALID_ARGUMENT));
    }

    #[test]
    fn null_out_pointers_are_invalid_arguments() {
        // SAFETY: the adapter checks every out pointer before any CUDA call.
        let status = unsafe { get_device_raw(std::ptr::null_mut()) };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
        // SAFETY: same adapter-side null check, on the minor out pointer.
        let mut major: c_int = 0;
        let status = unsafe { device_cc_raw(0, &mut major, std::ptr::null_mut()) };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn set_then_get_round_trips_device_zero() {
        if !device_present("set_then_get_round_trips_device_zero") {
            return;
        }
        set_device(0).expect("set device 0");
        assert_eq!(get_device().expect("get device"), 0);
        let (major, minor) = device_cc(0).expect("compute capability");
        assert!(major > 0 && minor >= 0, "cc {major}.{minor}");
    }
}
