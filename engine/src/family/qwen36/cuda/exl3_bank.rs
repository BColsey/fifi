//! The banked EXL3 expert load: the admission arithmetic that decides banking and the slot
//! count (pure, mirrored on `project::cuda`'s demand rule), and the pageable host store that
//! holds every routed expert's bytes in the exact device-bank layout, so a later miss copy
//! moves a member into a slot unchanged.
//!
//! The store is pageable by decision, not pinned: on the 16 GiB hosts this mode targets,
//! 9-10 GiB of page-locked memory would starve the OS; the copy cost of driver-internal
//! staging is a measured follow-up, not a load-time guess.

use crate::{
    EngError,
    family::{
        project::cuda::{Reserve, exl3::Exl3MemberStaging, exl3::exl3_member_layout},
        qwen36::settings::{BankMode, SlotRequest},
    },
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

/// The resolved banked load: the slot count and the figures the start line reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BankAdmission {
    pub(super) slots: usize,
    /// Dense bytes plus the slot banks' bytes: what the device holds after the load.
    pub(super) device_bytes: u64,
    /// The host store's bytes: every routed expert of every layer, in the bank layout.
    pub(super) host_bytes: u64,
}

/// Decides the load shape from the pack figures and the free memory. `Ok(None)` means stay
/// resident: the whole pack fits and the mode is `auto`. `dense` is the walk's device bytes
/// without the routed experts; `routed` is the routed experts' device bytes alone (the same
/// figure the host store holds, because a member's store bytes are its bank bytes).
///
/// The slot arithmetic mirrors `project::cuda::admit`'s demand rule: a load of `tensors`
/// bytes plus the reserve's `kv_cache + work + headroom` must fit `free`. Banking replaces
/// the routed share with `per_member * slots * layers`.
pub(super) fn decide(
    request: SlotRequest,
    mode: BankMode,
    dense: u64,
    routed: u64,
    layers: usize,
    n_expert: usize,
    free: u64,
    reserve: Reserve,
    host: Option<(u64, u64)>,
) -> Result<Option<BankAdmission>, EngError> {
    let hold = reserve
        .kv_cache
        .saturating_add(reserve.work)
        .saturating_add(reserve.headroom);
    let per_member = per_member_bytes(routed, layers, n_expert)?;
    let resident = dense
        .checked_add(routed)
        .ok_or_else(|| refuse("the resident figure overflows u64".into()))?;
    if mode == BankMode::Auto && resident.saturating_add(hold) <= free {
        return Ok(None);
    }
    admit_host(routed, host)?;
    let budget = free.saturating_sub(hold).saturating_sub(dense);
    let slots = match request {
        SlotRequest::Exactly(want) => {
            let demand = per_member
                .checked_mul(want as u64)
                .and_then(|bytes| bytes.checked_mul(layers as u64))
                .ok_or_else(|| refuse("the slot demand overflows u64".into()))?;
            if demand > budget {
                return Err(refuse(format!(
                    "{} slots need {} of expert bytes, the budget after the dense weights and \
                     the reserve is {} (free {}, hold {}, dense {})",
                    want,
                    gib(demand),
                    gib(budget),
                    gib(free),
                    gib(hold),
                    gib(dense),
                )));
            }
            want
        }
        SlotRequest::Fit => {
            let slots = (budget / per_member.max(1)) as usize;
            slots.clamp(1, n_expert)
        }
    };
    let device_bytes = dense.saturating_add(
        per_member
            .saturating_mul(slots as u64)
            .saturating_mul(layers as u64),
    );
    Ok(Some(BankAdmission {
        slots,
        device_bytes,
        host_bytes: routed,
    }))
}

/// The routed-expert bytes of one expert of one layer (the three projections summed). The
/// walk guarantees one rate per projection, so the division is exact; a remainder means a
/// pack the walk did not see, and is refused rather than mis-banked.
fn per_member_bytes(routed: u64, layers: usize, n_expert: usize) -> Result<u64, EngError> {
    if layers == 0 || n_expert == 0 {
        return Err(refuse("the model has no layers or no experts".into()));
    }
    let per_layer = routed / layers as u64;
    if per_layer * layers as u64 != routed {
        return Err(refuse(format!(
            "the routed figure {} does not divide over {layers} layers",
            gib(routed),
        )));
    }
    let per_member = per_layer / n_expert as u64;
    if per_member * n_expert as u64 != per_layer {
        return Err(refuse(
            "the routed figure does not divide evenly over the experts".into(),
        ));
    }
    Ok(per_member)
}

/// The host store's admission: the routed bytes must fit the available RAM under an OS
/// margin of the larger of 2 GiB and 20% of the total. The figures come from
/// `/proc/meminfo` (`probe_host`); a host without them does not get a banked load.
fn admit_host(routed: u64, host: Option<(u64, u64)>) -> Result<(), EngError> {
    let Some((total, available)) = host else {
        return Err(refuse(
            "the host memory figures are unavailable, so the expert store cannot be admitted"
                .into(),
        ));
    };
    let margin = (2 << 30).max(total / 5);
    let budget = available.saturating_sub(margin);
    if routed > budget {
        return Err(refuse(format!(
            "the expert store needs {} of host RAM, the budget is {} (available {}, margin {})",
            gib(routed),
            gib(budget),
            gib(available),
            gib(margin),
        )));
    }
    Ok(())
}

/// Every routed expert of one projection of one layer, in the device-bank byte layout:
/// member `i`'s trellis at `i * stride` and its scales at `i * scale_stride`, exactly the
/// addressing of the device bank, so a slot fill or miss copy is a plain byte move.
pub(super) struct PartStore {
    trellis: Vec<u8>,
    scales: Vec<u8>,
    stride: usize,
    scale_stride: usize,
}

impl PartStore {
    pub(super) fn new(shape: [usize; 2], bits_twice: u8, members: usize) -> Self {
        let (stride, scale_stride) = exl3_member_layout(shape, bits_twice);
        Self {
            trellis: vec![0; stride * members],
            scales: vec![0; scale_stride * members],
            stride,
            scale_stride,
        }
    }

    /// Writes member `member`'s staged bytes at its slot offsets. The length checks are the
    /// store's own rate guard: a member whose staged bytes disagree with the first member's
    /// strides is refused, not mis-laid.
    pub(super) fn put(
        &mut self,
        member: usize,
        staged: &Exl3MemberStaging,
    ) -> Result<(), EngError> {
        if staged.trellis.len() != self.stride || staged.scales.len() != self.scale_stride {
            return Err(EngError::Checkpoint(format!(
                "EXL3 expert bank: member {member} stages {} trellis and {} scale bytes, \
                 the bank layout holds {} and {}",
                staged.trellis.len(),
                staged.scales.len(),
                self.stride,
                self.scale_stride,
            )));
        }
        let trellis_at = member * self.stride;
        self.trellis[trellis_at..trellis_at + self.stride].copy_from_slice(&staged.trellis);
        let scales_at = member * self.scale_stride;
        self.scales[scales_at..scales_at + self.scale_stride].copy_from_slice(&staged.scales);
        Ok(())
    }

    /// Member `member`'s (trellis, scales) bytes, the source of a miss copy.
    pub(super) fn member(&self, member: usize) -> (&[u8], &[u8]) {
        let trellis_at = member * self.stride;
        let scales_at = member * self.scale_stride;
        (
            &self.trellis[trellis_at..trellis_at + self.stride],
            &self.scales[scales_at..scales_at + self.scale_stride],
        )
    }

    pub(super) fn bytes(&self) -> usize {
        self.trellis.len() + self.scales.len()
    }
}

/// Every routed expert of every layer: `layers` entries of the `[gate, up, down]` parts.
pub(crate) struct ExpertStore {
    layers: Vec<[PartStore; 3]>,
}

impl ExpertStore {
    pub(crate) fn with_layers(layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(layers),
        }
    }

    /// Appends one layer's three part stores; the load fills them layer by layer.
    pub(super) fn push_layer(&mut self, parts: [PartStore; 3]) {
        self.layers.push(parts);
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> &[PartStore; 3] {
        &self.layers[layer]
    }

    /// The whole store's bytes, for the start line and the admission's echo.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|parts| parts.iter().map(PartStore::bytes).sum::<usize>() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense ≈ 6 GiB, routed ≈ 10.1 GiB: the q36 EXL3 pack's shape, with the routed figure
    /// built from an exactly-divisible per-member byte count (the guard rejects a figure
    /// that does not divide over the layers and experts).
    fn figures() -> (u64, u64) {
        let per_member = 900_000u64; // ≈ 0.86 MiB per expert per layer (the three parts)
        (6 << 30, 47 * 256 * per_member)
    }

    /// A 16 GiB host with ~14 GiB available: the store fits under the 3.2 GiB margin.
    fn host() -> Option<(u64, u64)> {
        Some((16 << 30, 14 << 30))
    }

    fn reserve(hold: u64) -> Reserve {
        Reserve {
            kv_cache: hold,
            work: 0,
            headroom: 0,
        }
    }

    fn decide_auto(
        dense_routed: (u64, u64),
        free: u64,
        request: SlotRequest,
        host: Option<(u64, u64)>,
    ) -> Result<Option<BankAdmission>, EngError> {
        let (dense, routed) = dense_routed;
        decide(
            request,
            BankMode::Auto,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host,
        )
    }

    #[test]
    fn a_pack_that_fits_stays_resident_in_auto_mode() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30) + (1 << 20);
        assert_eq!(
            decide_auto((dense, routed), free, SlotRequest::Fit, host()).unwrap(),
            None
        );
    }

    #[test]
    fn a_pack_that_does_not_fit_banks_the_fitting_slot_count() {
        let (dense, routed) = figures();
        // Free for 1 GiB of dense + hold, plus exactly 64 slots of experts.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + per_member * 64;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 64);
        assert_eq!(got.host_bytes, routed);
        assert_eq!(got.device_bytes, dense + per_member * 64 * 47);
    }

    #[test]
    fn on_mode_banks_even_when_the_pack_fits() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30);
        let got = decide(
            SlotRequest::Exactly(32),
            BankMode::On,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 32);
        assert_eq!(
            got.device_bytes,
            dense + (routed / 47 / 256) * 32 * 47,
            "linear in the slot count"
        );
    }

    #[test]
    fn an_exact_count_over_budget_is_refused_with_numbers() {
        let (dense, routed) = figures();
        let free = dense + (1 << 30) + (routed / 47 / 256) * 16;
        let error = format!(
            "{:?}",
            decide_auto((dense, routed), free, SlotRequest::Exactly(200), host()).unwrap_err()
        );
        assert!(error.contains("200 slots"), "{error}");
        assert!(error.contains("budget"), "{error}");
    }

    #[test]
    fn a_host_without_room_is_refused_before_any_slot() {
        let (dense, routed) = figures();
        // On mode: banking is requested even though the pack would fit, and the host refuses.
        // Available 8 GiB under a 3.2 GiB margin (20% of 16): 4.8 GiB < ~10.1 GiB of experts.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                Some((16 << 30, 8 << 30)),
            )
            .unwrap_err()
        );
        assert!(error.contains("host RAM"), "{error}");
        // And a host without figures at all.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                None,
            )
            .unwrap_err()
        );
        assert!(error.contains("unavailable"), "{error}");
    }

    #[test]
    fn fit_never_returns_zero_or_above_the_expert_count() {
        let (dense, routed) = figures();
        // Barely past the dense figure: one slot still fits.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + per_member;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 1);
        // And a huge free figure clamps at the full expert count (On mode: Auto would stay
        // resident with room to spare).
        let got = decide(
            SlotRequest::Fit,
            BankMode::On,
            dense,
            routed,
            47,
            256,
            1 << 44,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 256);
    }

    #[test]
    fn the_store_lays_members_at_the_bank_strides() {
        let shape = [64usize, 32];
        let (stride, scale_stride) = exl3_member_layout(shape, 8);
        let mut store = PartStore::new(shape, 8, 4);
        let member = |trellis_byte: u8| Exl3MemberStaging {
            trellis: vec![trellis_byte; stride],
            scales: vec![trellis_byte; scale_stride],
        };
        for i in 0..4u8 {
            store.put(i as usize, &member(i)).unwrap();
        }
        for i in 0..4u8 {
            let (trellis, scales) = store.member(i as usize);
            assert!(trellis.iter().all(|&byte| byte == i));
            assert!(scales.iter().all(|&byte| byte == i));
        }
        // A member staged at another rate is refused, not mis-laid.
        let error = format!(
            "{:?}",
            store
                .put(
                    2,
                    &Exl3MemberStaging {
                        trellis: vec![0; stride - 1],
                        scales: vec![0; scale_stride],
                    },
                )
                .unwrap_err()
        );
        assert!(error.contains("member 2"), "{error}");
    }
}
