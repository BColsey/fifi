//! The per-card execution owner of the multi-device step (v1 eager).
//!
//! [`CardShard`] owns every device resource one card needs, in the drop
//! order the ownership requires (buffers, then the stream, then the
//! context), and [`CardStack`] owns the shards plus the one [`PeerRing`]
//! that spans them. What each shard holds:
//!
//! * the CUDA context, bound to the card's device ordinal;
//! * ONE stream - the coordinator invariant of `multi.rs` (all local
//!   kernels and all ring participation of this card queue on it, in the
//!   global exchange order);
//! * the arena built from `StepPlan::for_card`'s tensor list through the
//!   same planner, so arena tensor `i` IS plan slot `i` at the plan's
//!   offset;
//! * a FULL-SIZE `ResidentState` on the card's own context: the card
//!   touches its head slice at the same offsets a single-device state
//!   would, so no `StateLayout` surgery is owed (the waste is the unused
//!   tail of the delta buffer, accepted for v1);
//! * the card's ring status word, zeroed at construction, ORed by every
//!   ring wait of the session, folded at read-back (`fold_ring_status`).
//!
//! Weights are not owned here yet: they arrive with the shard's weight
//! views (the next slice), loaded through `shard_loads` + `load_sliced`.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device executor, the first production caller"
    )
)]

use crate::EngError;
use crate::family::qwen36::cuda::dims::{StepDims, StepOptions};
use crate::family::qwen36::cuda::shard::ShardTable;
use crate::family::qwen36::cuda::state::ResidentState;
use crate::family::qwen36::cuda::step_plan::{KernelScratch, StepPlan};
use crate::kernels::cuda::peer_ring::PeerRing;
use crate::kernels::cuda::{
    Admission, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream,
};

/// One card's device resources. Field order is drop order: the arena, the
/// state and the status word (all allocated through the context) drop
/// before the stream, and the stream before the context that owns it.
pub(crate) struct CardShard {
    arena: DeviceArena,
    state: ResidentState,
    ring_word: DeviceBuffer,
    stream: DeviceStream,
    context: CudaContext,
}

impl CardShard {
    /// Builds the shard around an existing context: its one stream, the
    /// arena over `plan`'s tensor list, the full-size state, and a zeroed
    /// ring status word. The ring must be built from the bare contexts
    /// BEFORE this takes them: [`PeerRing`] captures each mailbox's base
    /// address, so the contexts may move into shards afterwards.
    pub(crate) fn with_context(
        context: CudaContext,
        plan: &StepPlan,
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        let stream = context.stream()?;
        let arena = context.arena(&plan.arena_tensors())?;
        let state = ResidentState::allocate(&context, &stream, dims, options)?;
        let ring_word = context.allocate(4)?;
        ring_word.write_at(&stream, 0, &0u32.to_le_bytes())?;
        Ok(Self {
            arena,
            state,
            ring_word,
            stream,
            context,
        })
    }

    /// The shard on a freshly admitted `device`.
    pub(crate) fn new(
        device: usize,
        admission: &Admission,
        plan: &StepPlan,
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        let context = CudaContext::with_admission_on(device, admission)?;
        Self::with_context(context, plan, dims, options)
    }

    /// Arena tensor `index` of this card's plan: the same slot, at the
    /// plan's own offset and byte count.
    pub(crate) fn tensor(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        self.arena.tensor(index)
    }

    pub(crate) fn state(&self) -> &ResidentState {
        &self.state
    }

    pub(crate) fn stream(&self) -> &DeviceStream {
        &self.stream
    }

    pub(crate) fn context(&self) -> &CudaContext {
        &self.context
    }

    /// The card's ring status word as the range the ring operations take.
    pub(crate) fn ring_status(&self) -> Result<DeviceRange<'_>, EngError> {
        self.ring_word.range(0, 4)
    }

    /// The status word's buffer, for the read-back that folds it.
    pub(crate) fn ring_word(&self) -> &DeviceBuffer {
        &self.ring_word
    }
}

/// The whole distributed step's owners: one shard per card, the shared
/// ring, the shard table, and every card's plan. The ring is declared
/// first: its mailboxes are allocations of the shard contexts, so it must
/// drop before them.
pub(crate) struct CardStack {
    ring: PeerRing,
    shards: Vec<CardShard>,
    table: ShardTable,
    plans: Vec<StepPlan>,
}

impl CardStack {
    /// Builds the stack for `devices.len()` cards. Card `i` runs on
    /// `devices[i]` (one ordinal per physical card; the same ordinal twice
    /// gives two logical contexts, the rig configuration). The mailbox
    /// covers `dims.shape.hidden` values: every exchange payload of the
    /// plan - embed broadcast, mixer sum, MoE sum - is at most the hidden
    /// row; route picks and gathered K/V rows are narrower.
    pub(crate) fn new(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
    ) -> Result<Self, EngError> {
        if devices.is_empty() {
            return Err(EngError::BadRequest(
                "the card stack needs at least one device".to_owned(),
            ));
        }
        let table = ShardTable::new(&dims.shape, devices.len(), 32)?;
        let scratch = KernelScratch::for_dims(dims, 1, options.kv_mode)?;
        let mut contexts = Vec::with_capacity(devices.len());
        for &device in devices {
            contexts.push(CudaContext::with_admission_on(device, admission)?);
        }
        // The ring first: it captures each mailbox's base address from the
        // bare contexts, which then move into the shards.
        let ring = PeerRing::new(&contexts, dims.shape.hidden)?;
        let mut plans = Vec::with_capacity(devices.len());
        let mut shards = Vec::with_capacity(devices.len());
        for (card, context) in contexts.into_iter().enumerate() {
            let plan = StepPlan::for_card(dims, &table, card, options, &scratch)?;
            shards.push(CardShard::with_context(context, &plan, dims, options)?);
            plans.push(plan);
        }
        Ok(Self {
            ring,
            shards,
            table,
            plans,
        })
    }

    pub(crate) fn ring(&self) -> &PeerRing {
        &self.ring
    }

    pub(crate) fn shards(&self) -> &[CardShard] {
        &self.shards
    }

    pub(crate) fn table(&self) -> &ShardTable {
        &self.table
    }

    pub(crate) fn plans(&self) -> &[StepPlan] {
        &self.plans
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::family::project::cuda::served_admission;
    use crate::family::qwen36::cuda::common_tests::dims;
    use crate::family::qwen36::cuda::dims::KvMode;
    use crate::kernels::cuda::device_present;

    /// Two logical shards on the one physical device: every plan slot lands
    /// in its own arena at the plan's offset and size, both states are
    /// full-size and identical in shape, the ring spans both cards, and
    /// both status words start at zero - the ownership facts the
    /// distributed block tests build on.
    #[test]
    fn two_logical_shards_own_their_plan_arenas_and_states() {
        const NAME: &str = "two_logical_shards_own_their_plan_arenas_and_states";
        if !device_present(NAME) {
            return;
        }
        let d = dims();
        let options = StepOptions {
            kv_mode: KvMode::Int8G32,
            context_tokens: 1024,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: crate::family::project::cuda::WeightLayout::Native,
        };
        let admission = served_admission();
        let stack = CardStack::new(&d, &options, &[0, 0], &admission).expect("two logical shards");
        assert_eq!(stack.shards().len(), 2);
        assert_eq!(stack.plans().len(), 2);
        assert_eq!(stack.ring().cards(), 2);
        assert_eq!(stack.table().cards(), 2);
        for (card, shard) in stack.shards().iter().enumerate() {
            let plan = &stack.plans()[card];
            let tensors = plan.tensors();
            for (index, slot) in tensors.iter().enumerate() {
                let range = shard.tensor(index).expect("arena tensor");
                assert_eq!(
                    range.offset(),
                    plan.plan().offsets[index],
                    "card {card} slot {:?}: offset",
                    slot.slot
                );
                assert_eq!(
                    range.bytes(),
                    slot.bytes,
                    "card {card} slot {:?}: bytes",
                    slot.slot
                );
            }
            // The state is full-size and shaped like every other card's.
            assert!(shard.state().kv_bytes() > 0);
            assert_eq!(
                shard.state().layout().delta.total,
                stack.shards()[1 - card].state().layout().delta.total,
                "card {card}: delta state size"
            );
            assert_eq!(
                shard.state().kv_bytes(),
                stack.shards()[1 - card].state().kv_bytes(),
                "card {card}: kv state size"
            );
        }
        for shard in stack.shards() {
            let mut word = [0u8; 4];
            shard
                .ring_word()
                .read_at(shard.stream(), 0, &mut word)
                .expect("read the status word");
            assert_eq!(u32::from_le_bytes(word), 0);
        }
    }
}
