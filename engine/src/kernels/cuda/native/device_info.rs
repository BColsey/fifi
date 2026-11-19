//! Raw FFI to the device-property query of `native/device_info.cu` (declared in
//! `native/abi_device.h`). Child module of `native`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants every wrapper here relies on:
//! - The one extern is host only: it reads a device attribute and allocates, copies and
//!   launches nothing.
//! - The out pointer is a live, writable `c_int` for the whole call and the adapter keeps no
//!   pointer after it returns.
//! - No raw pointer and no unsafe function leaves this module.
//! - The call runs under the process dispatch guard on the thread that owns the context;
//!   the caller in `mod.rs` enters the guard.

use std::ffi::c_int;

use super::STATUS_OK;

unsafe extern "C" {
    #[link_name = "inference_cuda_device_integrated"]
    fn device_integrated_raw(out_integrated: *mut c_int) -> c_int;

    // The production caller is the tuner line in the next commit.
    #[allow(dead_code)]
    #[link_name = "inference_cuda_device_attributes"]
    fn device_attributes_raw(
        out_sm_count: *mut c_int,
        out_cc_major: *mut c_int,
        out_cc_minor: *mut c_int,
        out_bus_id: *mut u8,
    ) -> c_int;
}

/// The bus-id buffer size `abi_device.h` fixes for the adapter.
// The production caller is the tuner line in the next commit.
#[allow(dead_code)]
pub(in crate::kernels::cuda) const BUS_ID_BYTES: usize = 16;

/// The static device facts the tuner prices against: multiprocessor count,
/// compute capability and the PCI bus id (for the sysfs link read).
// The production caller is the tuner line in the next commit.
#[allow(dead_code)]
pub(crate) struct DeviceAttributes {
    pub sm_count: u32,
    pub compute_capability: (u32, u32),
    pub bus_id: String,
}

// The production caller is the tuner line in the next commit.
#[allow(dead_code)]
pub(crate) fn device_attributes() -> Result<DeviceAttributes, c_int> {
    let mut sm_count: c_int = 0;
    let mut cc_major: c_int = 0;
    let mut cc_minor: c_int = 0;
    let mut bus_id = [0u8; BUS_ID_BYTES];
    // SAFETY: every out pointer is a live, writable buffer for the whole
    // call; the adapter keeps no pointer after it returns.
    let status = unsafe {
        device_attributes_raw(
            &mut sm_count,
            &mut cc_major,
            &mut cc_minor,
            bus_id.as_mut_ptr(),
        )
    };
    if status == STATUS_OK {
        let end = bus_id
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(BUS_ID_BYTES);
        Ok(DeviceAttributes {
            sm_count: sm_count.max(0) as u32,
            compute_capability: (cc_major.max(0) as u32, cc_minor.max(0) as u32),
            bus_id: String::from_utf8_lossy(&bus_id[..end]).into_owned(),
        })
    } else {
        Err(status)
    }
}

pub(in crate::kernels::cuda) fn device_integrated() -> Result<bool, c_int> {
    let mut value: c_int = 0;
    // SAFETY: `value` is a live, writable c_int for the whole call; the adapter
    // keeps no pointer after it returns and touches no other memory.
    let status = unsafe { device_integrated_raw(&mut value) };
    if status == STATUS_OK {
        Ok(value != 0)
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::super::STATUS_INVALID_ARGUMENT;
    use super::*;
    use crate::kernels::cuda::testkit::device_present;

    #[test]
    fn the_integrated_attribute_matches_the_tegra_release_file() {
        if !device_present("the_integrated_attribute_matches_the_tegra_release_file") {
            return;
        }
        // The oracle is the L4T release file, independent of the attribute: it
        // exists on a Jetson and not on a discrete-GPU host or in the container.
        assert_eq!(
            device_integrated().expect("integrated query"),
            Path::new("/etc/nv_tegra_release").exists()
        );
    }

    #[test]
    fn the_card_reports_its_attributes() {
        if !device_present("the_card_reports_its_attributes") {
            return;
        }
        let attributes = device_attributes().expect("device attributes query");
        assert!(attributes.sm_count > 0, "a card has SMs");
        assert!(attributes.compute_capability.0 >= 5, "a modern card");
        assert!(
            !attributes.bus_id.is_empty(),
            "a discrete card has a bus id"
        );
        eprintln!(
            "device attributes: sm={} cc={}.{} bus={}",
            attributes.sm_count,
            attributes.compute_capability.0,
            attributes.compute_capability.1,
            attributes.bus_id
        );
    }

    #[test]
    fn a_null_out_pointer_is_an_invalid_argument() {
        // SAFETY: the adapter checks the out pointer before any CUDA call and
        // writes nothing through a null pointer.
        let status = unsafe { device_integrated_raw(std::ptr::null_mut()) };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn null_attribute_outs_are_an_invalid_argument() {
        // SAFETY: the adapter checks the out pointers before any CUDA call
        // and writes nothing through a null pointer.
        let status = unsafe {
            device_attributes_raw(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }
}
