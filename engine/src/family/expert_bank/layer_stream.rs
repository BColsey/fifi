//! Layer-major expert streaming: the shared half of a prefill that visits one layer at a time
//! and moves each layer's routed experts to the card once per prompt piece.
//!
//! WHAT. A model whose routed experts exceed the card keeps some layers' experts on the card
//! ([`Home::Card`]), the rest in pinned host memory ([`Home::Pinned`]) or on disk read in place
//! ([`Home::Disk`]). [`StreamPlan::choose`] decides this from the bytes it is given. A layer that
//! is not on the card is copied into one of a few staging slots on the card before its visits.
//! [`LayerMajor`] walks the layers in order: while layer `L` is visited for every piece of the
//! prompt, the copy of the next streamed layer runs into the other staging slot, and the slot
//! of `L` is released when its visits are queued.
//!
//! WHY layer-major. A chunk-major prefill visits every layer for chunk 0, then every layer for
//! chunk 1, and so on; a layer whose experts are not on the card is copied again for every
//! chunk (256-480 GB of host-to-card traffic for a 3,661-token prompt against a 46 GB expert
//! set, measured on the EXL3 pack). Visiting the layer for every piece of the prompt before the next
//! layer copies each streamed layer once.
//!
//! WHICH experts. [`UploadRule::choose`] compares the cost of copying the whole layer in one
//! contiguous transfer with the cost of copying only the experts the pieces pick, one transfer
//! each, from the number of rows and the router's top-k. When a subset is cheaper the copy
//! starts from the recipe's prediction (look-ahead) and the experts the router then picks and
//! the prediction missed are copied on demand before the layer's compute.
//!
//! The family supplies two seams and nothing else: a [`LayerRecipe`] (what visiting a layer for
//! a piece does, and what the router picked) and an [`ExpertTransport`] (how bytes of a layer
//! reach a staging slot: a pinned-memory copy, a disk read then a copy, an mmap). Everything
//! about tiers, sizing, order, counters and overlap is here.
//!
//! COUNTERS. Every layer reports the bytes moved, the bytes needed (the picked experts' bytes,
//! when the recipe reports its picks), the copy seconds the transport measured and the seconds
//! the host waited on the copy. From these: PCIe GB/s (moved over copy seconds) and overlap
//! (the share of the copy time hidden behind compute).

// Host code with no family wired yet; the tests and the first consumer's start use it.
#![cfg_attr(not(test), allow(dead_code))]

use std::{collections::VecDeque, fmt, time::Instant};

use crate::EngError;

fn plan_error(detail: String) -> EngError {
    EngError::MemoryPlan(format!("expert layer stream: {detail}"))
}

fn stream_error(detail: String) -> EngError {
    EngError::BadRequest(format!("expert layer stream: {detail}"))
}

/// The routed experts of one layer as stored: the stacks (gate, up, down in any order) that
/// every launch reads at `expert * stack_bytes / experts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerExperts {
    pub(crate) experts: usize,
    /// Bytes of each stack of the whole layer.
    pub(crate) stack_bytes: Vec<u64>,
}

impl LayerExperts {
    pub(crate) fn new(experts: usize, stack_bytes: Vec<u64>) -> Result<Self, EngError> {
        if experts == 0 || stack_bytes.is_empty() {
            return Err(plan_error("a layer needs experts and stacks".into()));
        }
        if let Some(bad) = stack_bytes
            .iter()
            .find(|bytes| **bytes == 0 || **bytes % experts as u64 != 0)
        {
            return Err(plan_error(format!(
                "a stack of {bad} bytes does not divide into {experts} equal experts"
            )));
        }
        Ok(Self {
            experts,
            stack_bytes,
        })
    }

    /// Bytes of the layer's experts.
    pub(crate) fn bytes(&self) -> u64 {
        self.stack_bytes.iter().sum()
    }

    /// Bytes of one expert across its stacks.
    pub(crate) fn record_bytes(&self) -> u64 {
        self.bytes() / self.experts as u64
    }
}

/// Where a layer's experts live between prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Home {
    /// Resident on the card for the whole run: no copy.
    Card,
    /// In pinned host memory: one host-to-card copy per visit set.
    Pinned,
    /// In the model file, read in place into a pinned buffer, then copied.
    Disk,
}

impl Home {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Card => "card",
            Self::Pinned => "pinned",
            Self::Disk => "disk",
        }
    }
}

/// The bytes a plan may place. The caller derives them from what it reads at load: the card's
/// free memory after the trunk, the state and the activations; the host admission's pinnable
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamBudget {
    /// Card bytes available to routed experts: resident layers and the staging slots.
    pub(crate) card_bytes: u64,
    /// Pinned host bytes available to routed experts.
    pub(crate) host_bytes: u64,
    /// Staging slots when anything streams (two: one computes while one loads).
    pub(crate) staging_slots: usize,
}

/// The placement of every layer and the staging the streamed layers need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StreamPlan {
    homes: Vec<Home>,
    staging_slots: usize,
    staging_bytes: u64,
    card_bytes: u64,
    pinned_bytes: u64,
    disk_bytes: u64,
}

impl StreamPlan {
    /// Places `layers`. When everything fits the card, every layer is [`Home::Card`] and no
    /// staging is allocated. Otherwise the staging slots (each the size of the largest layer
    /// that streams) are charged first, the layers that fill the rest of the card's bytes stay
    /// on the card (largest first: each resident byte saves a byte of copy per piece), the
    /// pinned budget takes the next layers in the same order, and the remainder is read from
    /// disk. Refuses, naming both figures, when the staging slots do not fit the card.
    pub(crate) fn choose(layers: &[LayerExperts], budget: StreamBudget) -> Result<Self, EngError> {
        if layers.is_empty() {
            return Err(plan_error("no layers".into()));
        }
        let total: u64 = layers.iter().map(LayerExperts::bytes).sum();
        if total <= budget.card_bytes {
            return Ok(Self {
                homes: vec![Home::Card; layers.len()],
                staging_slots: 0,
                staging_bytes: 0,
                card_bytes: total,
                pinned_bytes: 0,
                disk_bytes: 0,
            });
        }
        let slots = budget.staging_slots.max(2);
        // The staging slot holds the largest layer that can stream. Every layer may stream, so
        // the slot is sized by the largest one; a layer that ends up resident does not need it,
        // and the resident choice below re-checks the sum.
        let order = {
            let mut order: Vec<usize> = (0..layers.len()).collect();
            order.sort_by_key(|&layer| (std::cmp::Reverse(layers[layer].bytes()), layer));
            order
        };
        let mut best: Option<Self> = None;
        // Try the slot size of "the largest streamed layer": drop resident layers from the
        // largest-first order until the staging (sized by what remains streamed) fits.
        for resident_count in (0..=layers.len()).rev() {
            let mut resident = vec![false; layers.len()];
            for &layer in &order[..resident_count] {
                resident[layer] = true;
            }
            let streamed_max = (0..layers.len())
                .filter(|&layer| !resident[layer])
                .map(|layer| layers[layer].bytes())
                .max();
            let staging_bytes = streamed_max.map_or(0, |max| max * slots as u64);
            let card_resident: u64 = (0..layers.len())
                .filter(|&layer| resident[layer])
                .map(|layer| layers[layer].bytes())
                .sum();
            if card_resident + staging_bytes > budget.card_bytes {
                continue;
            }
            best = Some(Self::with_resident(
                layers,
                &resident,
                &order,
                budget,
                slots,
                staging_bytes,
                card_resident,
            ));
            break;
        }
        best.ok_or_else(|| {
            let largest = layers.iter().map(LayerExperts::bytes).max().unwrap_or(0);
            plan_error(format!(
                "the card has {} bytes for routed experts and {slots} staging slots of the \
                 largest layer ({largest} bytes) need {}",
                budget.card_bytes,
                largest * slots as u64
            ))
        })
    }

    fn with_resident(
        layers: &[LayerExperts],
        resident: &[bool],
        order: &[usize],
        budget: StreamBudget,
        slots: usize,
        staging_bytes: u64,
        card_resident: u64,
    ) -> Self {
        let mut homes = vec![Home::Disk; layers.len()];
        let mut pinned_left = budget.host_bytes;
        let (mut pinned_bytes, mut disk_bytes) = (0u64, 0u64);
        for &layer in order {
            if resident[layer] {
                homes[layer] = Home::Card;
                continue;
            }
            let bytes = layers[layer].bytes();
            if bytes <= pinned_left {
                pinned_left -= bytes;
                pinned_bytes += bytes;
                homes[layer] = Home::Pinned;
            } else {
                disk_bytes += bytes;
            }
        }
        Self {
            homes,
            staging_slots: if staging_bytes == 0 { 0 } else { slots },
            staging_bytes,
            card_bytes: card_resident,
            pinned_bytes,
            disk_bytes,
        }
    }

    pub(crate) fn home(&self, layer: usize) -> Home {
        self.homes[layer]
    }

    pub(crate) fn layers(&self) -> usize {
        self.homes.len()
    }

    pub(crate) fn staging_slots(&self) -> usize {
        self.staging_slots
    }

    /// Card bytes of the staging slots together.
    pub(crate) fn staging_bytes(&self) -> u64 {
        self.staging_bytes
    }

    /// Card bytes of the resident layers.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.card_bytes
    }

    pub(crate) fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes
    }

    pub(crate) fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }

    /// The layers that need a copy before their visits, in layer order.
    pub(crate) fn streamed(&self) -> Vec<usize> {
        (0..self.homes.len())
            .filter(|&layer| self.homes[layer] != Home::Card)
            .collect()
    }
}

impl fmt::Display for StreamPlan {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = |home| self.homes.iter().filter(|&&h| h == home).count();
        write!(
            out,
            "expert layers: {} on the card ({} MiB), {} pinned ({} MiB), {} on disk ({} MiB); \
             {} staging slots ({} MiB)",
            count(Home::Card),
            self.card_bytes >> 20,
            count(Home::Pinned),
            self.pinned_bytes >> 20,
            count(Home::Disk),
            self.disk_bytes >> 20,
            self.staging_slots,
            self.staging_bytes >> 20,
        )
    }
}

/// What a transfer carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UploadSet {
    /// Every expert of the layer in one contiguous transfer per stack.
    Whole,
    /// These experts, in the order the recipe expects to need them.
    Subset(Vec<u32>),
}

/// The cost model that picks the whole layer or a subset: one copy of `record_bytes` costs
/// `record_bytes / bandwidth + per_copy_s`, a whole layer costs `bytes / bandwidth` plus one
/// copy of overhead per stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct UploadRule {
    pub(crate) top_k: usize,
    /// Host-to-card bytes per second. Replace the default with the figure the first receipt
    /// measures.
    pub(crate) bandwidth: f64,
    /// Seconds of launch overhead per transfer.
    pub(crate) per_copy_s: f64,
}

impl UploadRule {
    /// The bandwidth (26.1 GB/s) and per-copy overhead (14 us) measured for pinned host memory
    /// to an RTX 3090.
    pub(crate) fn pinned_defaults(top_k: usize) -> Self {
        Self {
            top_k,
            bandwidth: 26.1e9,
            per_copy_s: 14e-6,
        }
    }

    /// The experts `rows` token rows pick in a layer, on average, when picks are uniform:
    /// `E * (1 - (1 - k/E)^rows)`.
    pub(crate) fn expected_unique(&self, experts: usize, rows: usize) -> f64 {
        let e = experts as f64;
        let miss = 1.0 - self.top_k as f64 / e;
        e * (1.0 - miss.powi(i32::try_from(rows).unwrap_or(i32::MAX)))
    }

    /// The set to copy ahead of a layer's visits of `rows` rows. A `prediction` (the recipe's
    /// look-ahead, in need order) makes a subset possible; without one the layer is copied
    /// whole, because the picks are not known before the layer's router has run.
    pub(crate) fn choose(
        &self,
        layer: &LayerExperts,
        rows: usize,
        prediction: Option<&[u32]>,
    ) -> UploadSet {
        let Some(prediction) = prediction.filter(|p| !p.is_empty()) else {
            return UploadSet::Whole;
        };
        let record = layer.record_bytes() as f64;
        let stacks = layer.stack_bytes.len() as f64;
        let whole = layer.bytes() as f64 / self.bandwidth + stacks * self.per_copy_s;
        // The subset is the larger of the prediction and the uniform expectation: a router that
        // picks more than the prediction pays the on-demand copies of the difference.
        let expected = self.expected_unique(layer.experts, rows);
        let count = (prediction.len() as f64)
            .max(expected)
            .min(layer.experts as f64);
        let subset = count * (record / self.bandwidth + stacks * self.per_copy_s);
        if subset < whole {
            UploadSet::Subset(prediction.to_vec())
        } else {
            UploadSet::Whole
        }
    }

    /// Bytes `set` moves for `layer`.
    pub(crate) fn bytes_of(set: &UploadSet, layer: &LayerExperts) -> u64 {
        match set {
            UploadSet::Whole => layer.bytes(),
            UploadSet::Subset(experts) => experts.len() as u64 * layer.record_bytes(),
        }
    }
}

/// The rows of one prefill piece: as many as the card's free bytes allow after the experts,
/// the staging slots and the decode state, in whole multiples of `granule`, between `granule`
/// and `cap`. A wider piece copies each streamed layer once for more rows, so the bytes moved
/// per token fall in proportion; the piece stops growing where the per-row activation bytes
/// (`bytes_per_row`: the stream, the mixer and expert scratch of one row) fill the free bytes.
pub(crate) fn piece_rows(free_bytes: u64, bytes_per_row: u64, granule: usize, cap: usize) -> usize {
    let granule = granule.max(1);
    if bytes_per_row == 0 {
        return cap.max(granule);
    }
    let fit = usize::try_from(free_bytes / bytes_per_row).unwrap_or(usize::MAX);
    (fit.min(cap) / granule * granule).max(granule)
}

/// What a finished transfer reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TransferReport {
    pub(crate) bytes: u64,
    /// Seconds the transfer took, measured by the transport (device events for a card copy).
    pub(crate) seconds: f64,
}

/// How a layer's experts reach a staging slot. The device half implements this for a family's
/// card (pinned copy, disk read then copy); tests implement it with a clock.
pub(crate) trait ExpertTransport {
    type Ticket;

    /// Starts copying `set` of `layer` into staging slot `slot` and returns at once. The slot's
    /// previous compute must be finished by the time the copy writes it: the transport orders
    /// the copy after [`Self::release`] of the same slot.
    fn begin(
        &mut self,
        layer: usize,
        set: &UploadSet,
        slot: usize,
    ) -> Result<Self::Ticket, EngError>;

    /// Blocks until the transfer finishes and reports it.
    fn wait(&mut self, ticket: Self::Ticket) -> Result<TransferReport, EngError>;

    /// The visits that read `slot` are queued: the next copy into it may follow them on the
    /// stream.
    fn release(&mut self, slot: usize) -> Result<(), EngError>;
}

impl<T: ExpertTransport + ?Sized> ExpertTransport for &mut T {
    type Ticket = T::Ticket;

    fn begin(
        &mut self,
        layer: usize,
        set: &UploadSet,
        slot: usize,
    ) -> Result<Self::Ticket, EngError> {
        (**self).begin(layer, set, slot)
    }

    fn wait(&mut self, ticket: Self::Ticket) -> Result<TransferReport, EngError> {
        (**self).wait(ticket)
    }

    fn release(&mut self, slot: usize) -> Result<(), EngError> {
        (**self).release(slot)
    }
}

/// What the recipe knows about one layer's picks, after its router ran for a piece.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Picks {
    /// The distinct experts the piece picked, sorted.
    pub(crate) experts: Vec<u32>,
}

/// The family's side of a layer visit.
pub(crate) trait LayerRecipe {
    fn layers(&self) -> usize;
    fn pieces(&self) -> usize;
    /// Rows of `piece`.
    fn rows(&self, piece: usize) -> usize;
    /// The experts the layer is expected to need, in need order (the look-ahead: a later
    /// router run on the current state, or the previous piece's picks). `None` when the recipe
    /// has no prediction.
    fn predict(&mut self, layer: usize) -> Option<Vec<u32>>;
    /// The routed experts of `layer` for the piece's rows, shared by the family's other recipe
    /// steps. A dense layer or a recipe that does not read its picks back returns `None`.
    fn route(&mut self, layer: usize, piece: usize) -> Result<Option<Picks>, EngError>;
    /// Runs layer `layer` for `piece`: queues every launch of the visit over the staged or
    /// resident experts.
    fn visit(&mut self, layer: usize, piece: usize, slot: Option<usize>) -> Result<(), EngError>;
}

/// The counters of one layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerReport {
    pub(crate) layer: usize,
    pub(crate) home: Home,
    pub(crate) moved_bytes: u64,
    /// The picked experts' bytes, or `None` when the recipe did not report picks.
    pub(crate) needed_bytes: Option<u64>,
    pub(crate) copy_seconds: f64,
    pub(crate) exposed_seconds: f64,
    pub(crate) transfers: usize,
}

/// The counters of one prefill.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct StreamReport {
    pub(crate) layers: Vec<LayerReport>,
}

impl StreamReport {
    pub(crate) fn moved_bytes(&self) -> u64 {
        self.layers.iter().map(|l| l.moved_bytes).sum()
    }

    /// Bytes needed over the layers that reported picks, and the moved bytes of those layers.
    pub(crate) fn needed_and_moved(&self) -> (u64, u64) {
        self.layers
            .iter()
            .filter_map(|l| l.needed_bytes.map(|needed| (needed, l.moved_bytes)))
            .fold((0, 0), |(n, m), (needed, moved)| (n + needed, m + moved))
    }

    pub(crate) fn copy_seconds(&self) -> f64 {
        self.layers.iter().map(|l| l.copy_seconds).sum()
    }

    pub(crate) fn exposed_seconds(&self) -> f64 {
        self.layers.iter().map(|l| l.exposed_seconds).sum()
    }

    /// Bytes moved per prompt row.
    pub(crate) fn moved_per_row(&self, rows: usize) -> f64 {
        self.moved_bytes() as f64 / rows.max(1) as f64
    }

    /// Host-to-card GB/s over the time the copies ran.
    pub(crate) fn pcie_gbps(&self) -> f64 {
        let seconds = self.copy_seconds();
        if seconds > 0.0 {
            self.moved_bytes() as f64 / seconds / 1e9
        } else {
            0.0
        }
    }

    /// The share of the copy time hidden behind compute: `1 - exposed / copy`.
    pub(crate) fn overlap(&self) -> f64 {
        let copy = self.copy_seconds();
        if copy > 0.0 {
            (1.0 - self.exposed_seconds() / copy).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

impl fmt::Display for StreamReport {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (needed, moved_of_reported) = self.needed_and_moved();
        write!(
            out,
            "expert stream: moved {:.2} GB, needed {} (of {:.2} GB moved on reporting layers), \
             {:.1} GB/s, copy {:.2} s, exposed {:.2} s, overlap {:.0}%",
            self.moved_bytes() as f64 / 1e9,
            if self.layers.iter().any(|l| l.needed_bytes.is_some()) {
                format!("{:.2} GB", needed as f64 / 1e9)
            } else {
                "not reported".to_string()
            },
            moved_of_reported as f64 / 1e9,
            self.pcie_gbps(),
            self.copy_seconds(),
            self.exposed_seconds(),
            self.overlap() * 100.0
        )
    }
}

struct Flight<T> {
    layer: usize,
    slot: usize,
    ticket: T,
    set: UploadSet,
}

/// The layer-major driver over a plan, a transport and the family's recipe.
pub(crate) struct LayerMajor<'a, T: ExpertTransport> {
    plan: &'a StreamPlan,
    layers: &'a [LayerExperts],
    rule: UploadRule,
    transport: T,
    streamed: Vec<usize>,
    next: usize,
    free: Vec<usize>,
    flights: VecDeque<Flight<T::Ticket>>,
}

impl<'a, T: ExpertTransport> LayerMajor<'a, T> {
    pub(crate) fn new(
        plan: &'a StreamPlan,
        layers: &'a [LayerExperts],
        rule: UploadRule,
        transport: T,
    ) -> Result<Self, EngError> {
        if plan.layers() != layers.len() {
            return Err(stream_error(format!(
                "the plan places {} layers and the model has {}",
                plan.layers(),
                layers.len()
            )));
        }
        Ok(Self {
            plan,
            layers,
            rule,
            transport,
            streamed: plan.streamed(),
            next: 0,
            free: (0..plan.staging_slots()).rev().collect(),
            flights: VecDeque::new(),
        })
    }

    pub(crate) fn transport(&self) -> &T {
        &self.transport
    }

    /// Starts the copies that fill the free staging slots, in layer order. Call it before the
    /// recipe's own preamble (embedding, host feeds) so the first layers' copies overlap it;
    /// the run loop calls it again before each layer.
    pub(crate) fn warm(&mut self, recipe: &mut impl LayerRecipe) -> Result<(), EngError> {
        while self.next < self.streamed.len() && !self.free.is_empty() {
            let layer = self.streamed[self.next];
            let slot = self.free.pop().expect("a free slot");
            let rows: usize = (0..recipe.pieces()).map(|piece| recipe.rows(piece)).sum();
            let prediction = recipe.predict(layer);
            let set = self
                .rule
                .choose(&self.layers[layer], rows, prediction.as_deref());
            let ticket = self.transport.begin(layer, &set, slot)?;
            self.flights.push_back(Flight {
                layer,
                slot,
                ticket,
                set,
            });
            self.next += 1;
        }
        Ok(())
    }

    /// Visits every layer for every piece.
    pub(crate) fn run(&mut self, recipe: &mut impl LayerRecipe) -> Result<StreamReport, EngError> {
        if recipe.layers() != self.layers.len() {
            return Err(stream_error(format!(
                "the recipe has {} layers and the plan {}",
                recipe.layers(),
                self.layers.len()
            )));
        }
        let mut report = StreamReport::default();
        for layer in 0..self.layers.len() {
            self.warm(recipe)?;
            let mut row = LayerReport {
                layer,
                home: self.plan.home(layer),
                moved_bytes: 0,
                needed_bytes: None,
                copy_seconds: 0.0,
                exposed_seconds: 0.0,
                transfers: 0,
            };
            let mut staged = Vec::new();
            let slot = if self.plan.home(layer) == Home::Card {
                None
            } else {
                let flight = self
                    .flights
                    .pop_front()
                    .ok_or_else(|| stream_error(format!("layer {layer} has no copy in flight")))?;
                if flight.layer != layer {
                    return Err(stream_error(format!(
                        "the copy in flight is layer {}, the walk is at {layer}",
                        flight.layer
                    )));
                }
                let waited = Instant::now();
                let done = self.transport.wait(flight.ticket)?;
                row.exposed_seconds += waited.elapsed().as_secs_f64();
                row.moved_bytes += done.bytes;
                row.copy_seconds += done.seconds;
                row.transfers += 1;
                if let UploadSet::Subset(experts) = &flight.set {
                    staged = experts.clone();
                }
                Some((flight.slot, flight.set))
            };
            let mut picked: Option<Vec<u32>> = None;
            for piece in 0..recipe.pieces() {
                if let Some(picks) = recipe.route(layer, piece)? {
                    let all = picked.get_or_insert_with(Vec::new);
                    all.extend(picks.experts.iter().copied());
                    all.sort_unstable();
                    all.dedup();
                    // A subset copy covers the prediction; the router's other picks are
                    // copied now, before the layer's compute reads them.
                    if let Some((slot, UploadSet::Subset(_))) = &slot {
                        let missing: Vec<u32> = picks
                            .experts
                            .iter()
                            .copied()
                            .filter(|expert| !staged.contains(expert))
                            .collect();
                        if !missing.is_empty() {
                            let set = UploadSet::Subset(missing.clone());
                            let ticket = self.transport.begin(layer, &set, *slot)?;
                            let waited = Instant::now();
                            let done = self.transport.wait(ticket)?;
                            row.exposed_seconds += waited.elapsed().as_secs_f64();
                            row.moved_bytes += done.bytes;
                            row.copy_seconds += done.seconds;
                            row.transfers += 1;
                            staged.extend(missing);
                        }
                    }
                }
                recipe.visit(layer, piece, slot.as_ref().map(|(slot, _)| *slot))?;
            }
            if let Some(all) = &picked {
                row.needed_bytes = Some(all.len() as u64 * self.layers[layer].record_bytes());
            }
            if let Some((slot, _)) = slot {
                self.transport.release(slot)?;
                self.free.push(slot);
            }
            report.layers.push(row);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn layer(experts: usize, mib_per_expert: u64) -> LayerExperts {
        // Three stacks of `mib_per_expert` MiB per expert each.
        LayerExperts::new(experts, vec![experts as u64 * mib_per_expert * MIB; 3]).unwrap()
    }

    fn model(count: usize) -> Vec<LayerExperts> {
        (0..count).map(|_| layer(8, 3)).collect()
    }

    #[test]
    fn a_layer_needs_whole_experts() {
        assert!(LayerExperts::new(8, vec![100, 80]).is_err());
        assert!(LayerExperts::new(0, vec![8]).is_err());
        let l = LayerExperts::new(4, vec![40, 80, 120]).unwrap();
        assert_eq!(l.bytes(), 240);
        assert_eq!(l.record_bytes(), 60);
    }

    #[test]
    fn a_model_that_fits_the_card_streams_nothing() {
        let layers = model(4);
        let total: u64 = layers.iter().map(LayerExperts::bytes).sum();
        let plan = StreamPlan::choose(
            &layers,
            StreamBudget {
                card_bytes: total,
                host_bytes: 0,
                staging_slots: 2,
            },
        )
        .unwrap();
        assert!((0..4).all(|l| plan.home(l) == Home::Card));
        assert_eq!(plan.staging_slots(), 0);
        assert!(plan.streamed().is_empty());
    }

    #[test]
    fn the_staging_slots_are_charged_before_resident_layers() {
        let layers = model(6);
        let one = layers[0].bytes();
        // Room for 2 staging slots and 2 resident layers.
        let plan = StreamPlan::choose(
            &layers,
            StreamBudget {
                card_bytes: 4 * one,
                host_bytes: 2 * one,
                staging_slots: 2,
            },
        )
        .unwrap();
        let on = |home| (0..6).filter(|&l| plan.home(l) == home).count();
        assert_eq!(on(Home::Card), 2);
        assert_eq!(on(Home::Pinned), 2);
        assert_eq!(on(Home::Disk), 2);
        assert_eq!(plan.staging_slots(), 2);
        assert_eq!(plan.staging_bytes(), 2 * one);
        assert!(plan.resident_bytes() + plan.staging_bytes() <= 4 * one);
        assert_eq!(plan.streamed().len(), 4);
    }

    #[test]
    fn no_resident_layers_when_only_staging_fits() {
        let layers = model(5);
        let one = layers[0].bytes();
        let plan = StreamPlan::choose(
            &layers,
            StreamBudget {
                card_bytes: 2 * one,
                host_bytes: u64::MAX,
                staging_slots: 2,
            },
        )
        .unwrap();
        assert!((0..5).all(|l| plan.home(l) == Home::Pinned));
        assert_eq!(plan.resident_bytes(), 0);
    }

    #[test]
    fn a_card_without_room_for_staging_is_refused_with_both_figures() {
        let layers = model(5);
        let one = layers[0].bytes();
        let error = StreamPlan::choose(
            &layers,
            StreamBudget {
                card_bytes: one,
                host_bytes: 0,
                staging_slots: 2,
            },
        )
        .unwrap_err();
        let error = format!("{error:?}");
        assert!(error.contains("staging"), "{error}");
        assert!(error.contains(&(2 * one).to_string()), "{error}");
    }

    #[test]
    fn layers_of_unequal_size_take_the_card_largest_first() {
        let layers = vec![
            layer(8, 1),
            layer(8, 4),
            layer(8, 2),
            layer(8, 4),
            layer(8, 4),
        ];
        let big = layers[1].bytes();
        let plan = StreamPlan::choose(
            &layers,
            StreamBudget {
                // One big layer and two staging slots of the largest streamed layer.
                card_bytes: 3 * big,
                host_bytes: u64::MAX,
                staging_slots: 2,
            },
        )
        .unwrap();
        assert_eq!(plan.home(1), Home::Card);
        assert_eq!(plan.streamed(), vec![0, 2, 3, 4]);
        assert_eq!(plan.staging_bytes(), 2 * big);
    }

    #[test]
    fn the_cost_model_copies_a_layer_whole_when_most_experts_are_picked() {
        let rule = UploadRule::pinned_defaults(10);
        let l = LayerExperts::new(512, vec![512 * 1_000_000; 3]).unwrap();
        let prediction: Vec<u32> = (0..40).collect();
        assert_eq!(rule.choose(&l, 4096, Some(&prediction)), UploadSet::Whole);
        // One row picks ten experts: a subset of the prediction is cheaper.
        assert_eq!(
            rule.choose(&l, 1, Some(&prediction)),
            UploadSet::Subset(prediction.clone())
        );
        // Without a prediction the picks are unknown before the router: whole.
        assert_eq!(rule.choose(&l, 1, None), UploadSet::Whole);
    }

    #[test]
    fn a_wider_piece_moves_fewer_bytes_per_row() {
        let layers = model(5);
        let plan = streamed_plan(&layers);
        let run = |pieces: Vec<usize>| {
            let log = std::cell::RefCell::new(Log::default());
            let rows: usize = pieces.iter().sum();
            let mut recipe = Recipe {
                log: &log,
                layers: 5,
                pieces,
                picks: (0..8).collect(),
                predicted: None,
            };
            drive(
                &layers,
                &plan,
                &mut recipe,
                &log,
                UploadRule::pinned_defaults(2),
            )
            .moved_per_row(rows)
        };
        // The same 4,096 rows as one wide piece or sixteen narrow pieces copied per piece:
        // the layer-major walk copies once either way, a chunk-major walk would copy 16 times.
        assert_eq!(run(vec![4096]), run(vec![256; 16]));
        let streamed = plan.streamed().len() as f64;
        let chunk_major = 16.0 * streamed * layers[0].bytes() as f64 / 4096.0;
        assert!((run(vec![4096]) * 16.0 - chunk_major).abs() < 1e-6);
    }

    #[test]
    fn the_piece_rows_follow_the_free_bytes() {
        assert_eq!(piece_rows(1 << 30, 500_000, 256, 8192), 2048);
        assert_eq!(piece_rows(100 << 30, 500_000, 256, 8192), 8192);
        assert_eq!(piece_rows(1 << 20, 500_000, 256, 8192), 256);
        assert_eq!(piece_rows(1 << 30, 0, 256, 4096), 4096);
    }

    #[test]
    fn expected_unique_matches_the_occupancy_formula() {
        let rule = UploadRule::pinned_defaults(10);
        let one = rule.expected_unique(512, 1);
        assert!((one - 10.0).abs() < 1e-9, "{one}");
        assert!(rule.expected_unique(512, 4096) > 511.9);
    }

    // ---- the driver over fakes ------------------------------------------------------------

    #[derive(Default)]
    struct Log {
        events: Vec<String>,
    }

    struct Fake<'l> {
        log: &'l std::cell::RefCell<Log>,
        layers: Vec<LayerExperts>,
        copy_s: f64,
        /// The layer each staging slot holds.
        owner: Vec<Option<usize>>,
    }

    impl ExpertTransport for Fake<'_> {
        type Ticket = (usize, u64);

        fn begin(
            &mut self,
            layer: usize,
            set: &UploadSet,
            slot: usize,
        ) -> Result<Self::Ticket, EngError> {
            assert!(
                self.owner[slot].is_none_or(|held| held == layer),
                "slot {slot} written before its release"
            );
            self.owner[slot] = Some(layer);
            self.log
                .borrow_mut()
                .events
                .push(format!("begin L{layer} s{slot}"));
            Ok((layer, UploadRule::bytes_of(set, &self.layers[layer])))
        }

        fn wait(&mut self, ticket: Self::Ticket) -> Result<TransferReport, EngError> {
            self.log
                .borrow_mut()
                .events
                .push(format!("wait L{}", ticket.0));
            Ok(TransferReport {
                bytes: ticket.1,
                seconds: self.copy_s,
            })
        }

        fn release(&mut self, slot: usize) -> Result<(), EngError> {
            self.owner[slot] = None;
            self.log
                .borrow_mut()
                .events
                .push(format!("release s{slot}"));
            Ok(())
        }
    }

    struct Recipe<'l> {
        log: &'l std::cell::RefCell<Log>,
        layers: usize,
        pieces: Vec<usize>,
        picks: Vec<u32>,
        predicted: Option<Vec<u32>>,
    }

    impl LayerRecipe for Recipe<'_> {
        fn layers(&self) -> usize {
            self.layers
        }
        fn pieces(&self) -> usize {
            self.pieces.len()
        }
        fn rows(&self, piece: usize) -> usize {
            self.pieces[piece]
        }
        fn predict(&mut self, _layer: usize) -> Option<Vec<u32>> {
            self.predicted.clone()
        }
        fn route(&mut self, _layer: usize, _piece: usize) -> Result<Option<Picks>, EngError> {
            Ok(Some(Picks {
                experts: self.picks.clone(),
            }))
        }
        fn visit(
            &mut self,
            layer: usize,
            piece: usize,
            slot: Option<usize>,
        ) -> Result<(), EngError> {
            self.log
                .borrow_mut()
                .events
                .push(format!("visit L{layer} p{piece} {slot:?}"));
            Ok(())
        }
    }

    fn drive(
        layers: &[LayerExperts],
        plan: &StreamPlan,
        recipe: &mut Recipe<'_>,
        log: &std::cell::RefCell<Log>,
        rule: UploadRule,
    ) -> StreamReport {
        let fake = Fake {
            log,
            layers: layers.to_vec(),
            copy_s: 0.01,
            owner: vec![None; plan.staging_slots().max(1)],
        };
        let mut driver = LayerMajor::new(plan, layers, rule, fake).unwrap();
        driver.run(recipe).unwrap()
    }

    fn streamed_plan(layers: &[LayerExperts]) -> StreamPlan {
        let one = layers[0].bytes();
        StreamPlan::choose(
            layers,
            StreamBudget {
                card_bytes: 2 * one,
                host_bytes: u64::MAX,
                staging_slots: 2,
            },
        )
        .unwrap()
    }

    #[test]
    fn each_streamed_layer_is_copied_once_however_many_pieces_visit_it() {
        let layers = model(5);
        let plan = streamed_plan(&layers);
        let log = std::cell::RefCell::new(Log::default());
        let mut recipe = Recipe {
            log: &log,
            layers: 5,
            pieces: vec![100, 100, 100],
            picks: (0..8).collect(),
            predicted: None,
        };
        let report = drive(
            &layers,
            &plan,
            &mut recipe,
            &log,
            UploadRule::pinned_defaults(2),
        );
        let events = log.borrow().events.clone();
        let begins = events.iter().filter(|e| e.starts_with("begin")).count();
        assert_eq!(begins, 5, "{events:?}");
        let visits = events.iter().filter(|e| e.starts_with("visit")).count();
        assert_eq!(visits, 15);
        assert_eq!(report.moved_bytes(), 5 * layers[0].bytes());
        let (needed, moved) = report.needed_and_moved();
        assert_eq!(needed, moved);
        assert!((report.pcie_gbps() - report.moved_bytes() as f64 / 0.05 / 1e9).abs() < 1e-6);
    }

    #[test]
    fn the_next_layer_is_in_flight_while_the_current_one_is_visited() {
        let layers = model(4);
        let plan = streamed_plan(&layers);
        let log = std::cell::RefCell::new(Log::default());
        let mut recipe = Recipe {
            log: &log,
            layers: 4,
            pieces: vec![64],
            picks: (0..8).collect(),
            predicted: None,
        };
        drive(
            &layers,
            &plan,
            &mut recipe,
            &log,
            UploadRule::pinned_defaults(2),
        );
        let events = log.borrow().events.clone();
        let at = |needle: &str| {
            events
                .iter()
                .position(|e| e.starts_with(needle))
                .unwrap_or_else(|| panic!("{needle} in {events:?}"))
        };
        for layer in 0..3 {
            assert!(
                at(&format!("begin L{}", layer + 1)) < at(&format!("visit L{layer}")),
                "layer {} copy must start before layer {layer} is visited: {events:?}",
                layer + 1
            );
        }
        // Layer 2 reuses layer 0's slot, only after layer 0's release.
        assert!(at("release s") < at("begin L2"), "{events:?}");
    }

    #[test]
    fn resident_layers_are_visited_without_a_copy() {
        let layers = model(6);
        let one = layers[0].bytes();
        let plan = StreamPlan::choose(
            &layers,
            StreamBudget {
                card_bytes: 4 * one,
                host_bytes: u64::MAX,
                staging_slots: 2,
            },
        )
        .unwrap();
        let log = std::cell::RefCell::new(Log::default());
        let mut recipe = Recipe {
            log: &log,
            layers: 6,
            pieces: vec![32, 32],
            picks: vec![1, 2],
            predicted: None,
        };
        let report = drive(
            &layers,
            &plan,
            &mut recipe,
            &log,
            UploadRule::pinned_defaults(2),
        );
        let streamed = plan.streamed().len() as u64;
        assert_eq!(streamed, 4);
        assert_eq!(report.moved_bytes(), streamed * one);
        for row in &report.layers {
            if row.home == Home::Card {
                assert_eq!(row.moved_bytes, 0);
                assert_eq!(row.transfers, 0);
            }
        }
        // Bytes needed (2 of 8 experts) against bytes moved (the whole layer).
        let (needed, moved) = report.needed_and_moved();
        assert!(needed < moved);
        assert!(report.to_string().contains("expert stream: moved"));
    }

    #[test]
    fn a_subset_copy_is_topped_up_with_the_picks_the_prediction_missed() {
        // A big layer, a one-row piece: the rule copies a subset.
        let layers: Vec<LayerExperts> = (0..3)
            .map(|_| LayerExperts::new(512, vec![512 * 1_000_000; 3]).unwrap())
            .collect();
        let plan = streamed_plan(&layers);
        let log = std::cell::RefCell::new(Log::default());
        let mut recipe = Recipe {
            log: &log,
            layers: 3,
            pieces: vec![1],
            picks: vec![3, 4, 5, 6],
            predicted: Some(vec![3, 4, 7]),
        };
        let report = drive(
            &layers,
            &plan,
            &mut recipe,
            &log,
            UploadRule::pinned_defaults(10),
        );
        let record = layers[0].record_bytes();
        for row in &report.layers {
            // 3 predicted + 2 missed (5 and 6)
            assert_eq!(row.moved_bytes, 5 * record, "{row:?}");
            assert_eq!(row.needed_bytes, Some(4 * record));
            assert_eq!(row.transfers, 2);
        }
    }

    #[test]
    fn a_plan_for_another_model_is_refused() {
        let layers = model(4);
        let plan = streamed_plan(&layers);
        let log = std::cell::RefCell::new(Log::default());
        let fake = Fake {
            log: &log,
            layers: layers.clone(),
            copy_s: 0.0,
            owner: vec![None; 2],
        };
        assert!(
            LayerMajor::new(&plan, &layers[..3], UploadRule::pinned_defaults(2), fake).is_err()
        );
    }
}
