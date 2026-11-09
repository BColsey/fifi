//! Raw FFI to `inference_cuda_mapped_alloc` and `inference_cuda_mailbox_caps` of `native/resident/mailbox.cu`
//! (declared in `native/resident/mailbox.h`), and the only code that touches the memory they return. Child
//! module of `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Original code.
//!
//! Safety invariants this file relies on:
//! - A [`RawMapped`] is created once by [`mapped_alloc`] and consumed once by [`mapped_free`]. It has no `Drop`:
//!   the owner in `context` frees it, so that a failed free is reported.
//! - After publication, the device may write while the host reads, so mailbox accesses are atomic: acquire
//!   and release for the 8-byte sequence words, relaxed for the 4-byte payload words. The only references
//!   to a published mailbox are the `&AtomicU32` / `&AtomicU64` of `from_ptr`; no pointer or slice escapes.
//!   Reading a payload only after an acquire load of
//!   the word the device releases it with is the caller's protocol, documented in
//!   `context/mailbox.rs`; so is keeping the 4-byte payload regions and the 8-byte sequence words disjoint, because
//!   atomic accesses of different sizes to the same bytes do not synchronize with each other.
//! - Offsets and lengths are proved inside the allocation and word-aligned before any access (`cudaHostAlloc`
//!   memory is aligned like any host allocation, to at least 16 bytes). The device view ([`RawMapped::device`]) is
//!   never passed to the device free: it only lets `RawRange::new` prove ranges of the mapping for a launch.
//!   Extern declarations and raw pointers stay private.
//! - `UnpublishedMapped` has no device view. It alone may expose a temporary mutable slice
//!   to initialize immutable expert weights before publishing the device view.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the tiered decode step")
)]

use std::ffi::{c_int, c_void};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::{
    DeviceFailure, RawDevice, RawPinned, STAGE_NONE, checked, invalid_argument, pinned_free,
};

unsafe extern "C" {
    #[link_name = "inference_cuda_mapped_alloc"]
    fn mapped_alloc_raw(
        bytes: usize,
        out_host: *mut *mut c_void,
        out_device: *mut *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mailbox_caps"]
    fn mailbox_caps_raw(
        out_can_map_host: *mut c_int,
        out_unified_addressing: *mut c_int,
        out_kernel_exec_timeout: *mut c_int,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Zeroed page-locked host memory mapped into the device address space.
#[derive(Debug)]
#[must_use = "mapped memory must be passed to mapped_free"]
pub(in crate::kernels::cuda) struct RawMapped {
    host: NonNull<c_void>,
    device: RawDevice,
}

/// A fresh mapping whose device pointer has not been exposed to any caller.
/// This separate type permits ordinary host writes only before publication.
pub(in crate::kernels::cuda) struct UnpublishedMapped(RawMapped);

impl UnpublishedMapped {
    pub(in crate::kernels::cuda) fn allocate(bytes: usize) -> Result<Self, DeviceFailure> {
        if bytes > isize::MAX as usize {
            return Err(invalid_argument());
        }
        mapped_alloc(bytes).map(Self)
    }

    pub(in crate::kernels::cuda) fn initialize<E>(
        &mut self,
        fill: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        // SAFETY: this type never exposes its device pointer. The fresh allocation
        // is initialized, exclusive, and bounded by isize::MAX. The slice cannot
        // escape the callback; after publication no host slice can be made.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.0.host.as_ptr().cast::<u8>(), self.0.device.bytes)
        };
        fill(bytes)
    }

    pub(in crate::kernels::cuda) fn publish(self) -> RawMapped {
        self.0
    }
}

// SAFETY: after publication the host pointer is used only for atomic accesses (see the module doc),
// and its owner frees it only when no other handle to it is alive.
unsafe impl Send for RawMapped {}
// SAFETY: as for `Send`: every shared access is atomic.
unsafe impl Sync for RawMapped {}

/// All three from `cudaDeviceGetAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct MailboxCaps {
    pub(in crate::kernels::cuda) can_map_host: bool,
    pub(in crate::kernels::cuda) unified_addressing: bool,
    /// The display watchdog: a kernel that runs longer than the driver's limit is killed.
    pub(in crate::kernels::cuda) kernel_exec_timeout: bool,
}

pub(in crate::kernels::cuda) fn mapped_alloc(bytes: usize) -> Result<RawMapped, DeviceFailure> {
    if bytes == 0 {
        return Err(invalid_argument());
    }
    let (mut host, mut device) = (ptr::null_mut(), ptr::null_mut());
    let mut stage = STAGE_NONE;
    // SAFETY: both pointers and `stage` are live, writable values for the call; the entry writes the
    // pointers only on success.
    let status = unsafe { mapped_alloc_raw(bytes, &mut host, &mut device, &mut stage) };
    checked(status, stage)?;
    let Some(host) = NonNull::new(host) else {
        return Err(invalid_argument());
    };
    let Some(device) = NonNull::new(device) else {
        pinned_free(RawPinned { ptr: host, bytes })?;
        return Err(invalid_argument());
    };
    Ok(RawMapped {
        host,
        device: RawDevice { ptr: device, bytes },
    })
}

/// Waits for the device, then frees the memory (the pinned free; `cudaFreeHost` frees mapped memory too).
/// When the wait fails nothing is freed and the failure is returned.
pub(in crate::kernels::cuda) fn mapped_free(mapped: RawMapped) -> Result<(), DeviceFailure> {
    let bytes = mapped.device.bytes;
    pinned_free(RawPinned {
        ptr: mapped.host,
        bytes,
    })
}

pub(in crate::kernels::cuda) fn mailbox_caps() -> Result<MailboxCaps, DeviceFailure> {
    let (mut can_map, mut unified, mut timeout) = (0, 0, 0);
    let mut stage = STAGE_NONE;
    // SAFETY: the four pointers are live, writable c_ints for the call.
    let status = unsafe { mailbox_caps_raw(&mut can_map, &mut unified, &mut timeout, &mut stage) };
    checked(status, stage)?;
    Ok(MailboxCaps {
        can_map_host: can_map != 0,
        unified_addressing: unified != 0,
        kernel_exec_timeout: timeout != 0,
    })
}

impl RawMapped {
    /// The device's view of the mapping, for `RawRange::new` only.
    pub(in crate::kernels::cuda) fn device(&self) -> &RawDevice {
        &self.device
    }

    fn word_address(&self, offset: usize, words: usize, width: usize) -> Option<*mut u8> {
        let end = offset.checked_add(words.checked_mul(width)?)?;
        if end > self.device.bytes || !offset.is_multiple_of(width) {
            return None;
        }
        // SAFETY: `offset <= end <= bytes`, so the result is inside the allocation or one past its end.
        Some(unsafe { self.host.as_ptr().cast::<u8>().add(offset) })
    }

    /// Payload words the device wrote before it released this word are visible to the loads that
    /// follow.
    pub(in crate::kernels::cuda) fn load_acquire(&self, offset: usize) -> Option<u64> {
        let address = self.word_address(offset, 1, 8)?;
        // SAFETY: the address is inside the live mapping and 8-byte aligned; all access to it is atomic.
        Some(unsafe { AtomicU64::from_ptr(address.cast()) }.load(Ordering::Acquire))
    }

    /// Payload words written before it are visible to the device once it loads this word with
    /// acquire ordering.
    pub(in crate::kernels::cuda) fn store_release(&self, offset: usize, value: u64) -> bool {
        let Some(address) = self.word_address(offset, 1, 8) else {
            return false;
        };
        // SAFETY: as for `load_acquire`.
        unsafe { AtomicU64::from_ptr(address.cast()) }.store(value, Ordering::Release);
        true
    }

    pub(in crate::kernels::cuda) fn read_words(&self, offset: usize, out: &mut [u32]) -> bool {
        self.read_converted(offset, out, |word| word)
    }

    pub(in crate::kernels::cuda) fn read_f32s(&self, offset: usize, out: &mut [f32]) -> bool {
        self.read_converted(offset, out, f32::from_bits)
    }

    pub(in crate::kernels::cuda) fn write_words(&self, offset: usize, src: &[u32]) -> bool {
        self.write_converted(offset, src, |word| word)
    }

    pub(in crate::kernels::cuda) fn write_f32s(&self, offset: usize, src: &[f32]) -> bool {
        self.write_converted(offset, src, f32::to_bits)
    }

    fn read_converted<T>(&self, offset: usize, out: &mut [T], convert: fn(u32) -> T) -> bool {
        let Some(address) = self.word_address(offset, out.len(), 4) else {
            return false;
        };
        for (i, slot) in out.iter_mut().enumerate() {
            // SAFETY: word `i` lies inside the range proved by `word_address`, 4-byte aligned.
            *slot = convert(
                unsafe { AtomicU32::from_ptr(address.cast::<u32>().add(i)) }
                    .load(Ordering::Relaxed),
            );
        }
        true
    }

    fn write_converted<T: Copy>(&self, offset: usize, src: &[T], convert: fn(T) -> u32) -> bool {
        let Some(address) = self.word_address(offset, src.len(), 4) else {
            return false;
        };
        for (i, value) in src.iter().enumerate() {
            // SAFETY: as for `read_converted`.
            unsafe { AtomicU32::from_ptr(address.cast::<u32>().add(i)) }
                .store(convert(*value), Ordering::Relaxed);
        }
        true
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
