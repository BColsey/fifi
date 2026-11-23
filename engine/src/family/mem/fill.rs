//! The receipts of a start's memory plan: how much of the card the start used, how much of the
//! experts it holds resident and where the rest lives, and the least free VRAM the engine saw
//! while it served.
//!
//! A family supplies its own counts and byte sizes; the arithmetic and the words of the line are
//! here once, so every family's start reports the same figures the same way and a headroom
//! constant can be judged against what the engine measured, not what a plan assumed.
//!
//! The API for a family that serves through the shared executors:
//! - at start, build a [`TierReceipt`] from [`TierReceipt::from_layers`] (experts), or
//!   [`TierReceipt::dense_only`] (a model with no routed experts), and print [`TierReceipt::line`];
//! - in the engine, own a [`FreeMarks`] and call [`FreeMarks::record`] with the load-time reading,
//!   then [`FreeMarks::sample`] on decode, [`FreeMarks::sample_prefill`] on each prefill piece and
//!   [`FreeMarks::sample_sessions`] when a session opens. Each takes the read as a closure that
//!   runs only when the throttle lets it, so the hot path pays one clock read.

use std::time::{Duration, Instant};

const MIB: u64 = 1 << 20;

/// One layer's bank: the experts it holds on the card and the bytes of one expert record in the
/// file's type for that layer (a file that mixes types has different sizes per layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTier {
    /// Experts of this layer resident on the card.
    pub(crate) slots: u64,
    /// Bytes of one expert record of this layer.
    pub(crate) slot_bytes: u64,
}

/// The key-value cache a start holds, for the line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheFigure {
    /// Positions the cache holds.
    pub(crate) cells: u64,
    /// Device bytes it takes.
    pub(crate) bytes: u64,
}

/// What a start put where, in the units of the experts it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TierReceipt {
    /// The card's total bytes.
    pub(crate) card_total: u64,
    /// The card's free bytes when the receipt was taken. The card is the whole device: every
    /// process on it counts, not only this engine.
    pub(crate) card_free: u64,
    /// The dense core on the card: every weight that is not a routed expert.
    pub(crate) dense_bytes: u64,
    /// The key-value cache, when the family reports it.
    pub(crate) cache: Option<CacheFigure>,
    /// Routed experts of the model, over all layers.
    pub(crate) experts_total: u64,
    /// Routed experts resident on the card.
    pub(crate) experts_on_card: u64,
    /// Bytes of the resident experts on the card (the banks).
    pub(crate) bytes_on_card: u64,
    /// Bytes of host memory pinned by the engine's own pool (the staging ring and scratch).
    pub(crate) bytes_pinned: u64,
    /// Bytes of experts read from the file on a miss (the page cache and the disk).
    pub(crate) bytes_streamed: u64,
}

impl TierReceipt {
    /// A model whose experts are `experts_per_layer` in each layer, with `layers` holding the
    /// slots and slot bytes each keeps; `pinned` is the engine's pinned pool in bytes. A slot
    /// count above the layer's experts is clamped. Every expert byte is on the card or streamed.
    pub(crate) fn from_layers(
        (card_total, card_free): (u64, u64),
        dense_bytes: u64,
        experts_per_layer: u64,
        layers: &[LayerTier],
        pinned: u64,
    ) -> Self {
        let (mut on_card, mut bytes_on_card, mut streamed) = (0u64, 0u64, 0u64);
        for layer in layers {
            let held = layer.slots.min(experts_per_layer);
            on_card = on_card.saturating_add(held);
            bytes_on_card = bytes_on_card.saturating_add(held.saturating_mul(layer.slot_bytes));
            streamed = streamed
                .saturating_add((experts_per_layer - held).saturating_mul(layer.slot_bytes));
        }
        Self {
            card_total,
            card_free,
            dense_bytes,
            cache: None,
            experts_total: experts_per_layer.saturating_mul(layers.len() as u64),
            experts_on_card: on_card,
            bytes_on_card,
            bytes_pinned: pinned,
            bytes_streamed: streamed,
        }
    }

    /// A model with no routed experts (Bolt) or whose experts are all on the device: only the
    /// dense core and the cache.
    pub(crate) fn dense_only(
        (card_total, card_free): (u64, u64),
        dense_bytes: u64,
        cache: Option<CacheFigure>,
    ) -> Self {
        Self {
            card_total,
            card_free,
            dense_bytes,
            cache,
            experts_total: 0,
            experts_on_card: 0,
            bytes_on_card: 0,
            bytes_pinned: 0,
            bytes_streamed: 0,
        }
    }

    /// The same receipt with the cache the start holds.
    pub(crate) fn with_cache(mut self, cache: CacheFigure) -> Self {
        self.cache = Some(cache);
        self
    }

    /// The card bytes in use: the whole device, every process on it.
    pub(crate) fn card_used(&self) -> u64 {
        self.card_total.saturating_sub(self.card_free)
    }

    /// The share of the card in use, in percent.
    pub(crate) fn card_used_percent(&self) -> f64 {
        percent(self.card_used(), self.card_total)
    }

    /// The share of the routed experts resident on the card, in percent.
    pub(crate) fn resident_percent(&self) -> f64 {
        percent(self.experts_on_card, self.experts_total)
    }

    /// The one line of a start: card use, the dense core, the resident fraction and the bytes of
    /// each tier.
    pub(crate) fn line(&self, label: &str) -> String {
        let experts = if self.experts_total == 0 {
            "no routed experts".to_owned()
        } else {
            format!(
                "experts {} of {} resident ({:.1}%): {} B on the card, {} B pinned on the host, \
                 {} B read from the file on a miss",
                self.experts_on_card,
                self.experts_total,
                self.resident_percent(),
                self.bytes_on_card,
                self.bytes_pinned,
                self.bytes_streamed
            )
        };
        let cache = self.cache.map_or_else(String::new, |cache| {
            format!(", cache {} cells ({} B)", cache.cells, cache.bytes)
        });
        format!(
            "{label}: memory receipt: whole card {} of {} MiB used ({:.1}%, every process), \
             dense core {} B{cache}, {experts}",
            self.card_used() / MIB,
            self.card_total / MIB,
            self.card_used_percent(),
            self.dense_bytes,
        )
    }
}

/// The line a start prints for its receipt: the receipt, or why it is unavailable. A receipt is a
/// report, so a failure to take it is a line and never a failed start.
pub(crate) fn report_line(label: &str, receipt: Result<TierReceipt, crate::EngError>) -> String {
    match receipt {
        Ok(receipt) => receipt.line(label),
        Err(error) => format!("{label}: memory receipt unavailable: {}", error.describe()),
    }
}

/// The receipt line of a unified-memory start (the Mac): the machine's memory, Metal's working
/// set when known and the bytes of the checkpoint. There is no card to subtract from and no pinned
/// pool: the placement line of the start says which experts stream.
pub(crate) fn unified_line(
    label: &str,
    physical: u64,
    working_set: Option<u64>,
    checkpoint: Option<u64>,
) -> String {
    let set = working_set.map_or_else(
        || "Metal working set unknown".to_owned(),
        |bytes| format!("Metal working set {} MiB", bytes / MIB),
    );
    let weights = checkpoint.map_or_else(
        || "checkpoint size unknown".to_owned(),
        |bytes| format!("checkpoint {bytes} B"),
    );
    format!(
        "{label}: memory receipt: unified memory {} MiB, {set}, {weights}",
        physical / MIB
    )
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

/// The least free VRAM an engine has seen at named points of its life. A reading is taken the
/// first time a point is reached and then at most once per [`FreeMarks::INTERVAL`], so a decode
/// loop pays a clock read per token, not a driver query. The least is kept exactly; the log gets
/// a point's first reading and a new low that is lower by more than [`FreeMarks::STEP`].
#[derive(Debug, Default)]
pub(crate) struct FreeMarks {
    seen: Vec<String>,
    /// The true least free bytes of every reading.
    low: Option<u64>,
    /// The least free bytes the log has reported.
    reported: Option<u64>,
    last_read: Option<Instant>,
    largest_prefill: usize,
    most_sessions: usize,
}

impl FreeMarks {
    /// How far below the last reported low a reading must fall to be reported again.
    pub(crate) const STEP: u64 = 32 * MIB;
    /// The least time between two driver reads of a point already seen.
    pub(crate) const INTERVAL: Duration = Duration::from_secs(5);

    /// Takes a reading at `label`; the line to print when it is the label's first or a new low.
    pub(crate) fn record(&mut self, label: &str, free: u64) -> Option<String> {
        let first = !self.seen.iter().any(|seen| seen == label);
        if first {
            self.seen.push(label.to_owned());
        }
        self.low = Some(self.low.map_or(free, |low| low.min(free)));
        let new_low = self
            .reported
            .is_none_or(|reported| free.saturating_add(Self::STEP) <= reported);
        if new_low {
            self.reported = Some(free);
        }
        (first || new_low).then(|| {
            format!(
                "free VRAM after {label}: {} MiB{}",
                free / MIB,
                if new_low && !first { " (new low)" } else { "" }
            )
        })
    }

    /// Whether a point already seen may be read again at `now`.
    fn due(&self, now: Instant) -> bool {
        self.last_read
            .is_none_or(|last| now.duration_since(last) >= Self::INTERVAL)
    }

    /// A throttled reading: the first time `label` is reached it reads at once, later it reads
    /// only when [`FreeMarks::INTERVAL`] has passed since the last read. `read` is the driver
    /// query and runs only when the throttle allows it.
    pub(crate) fn sample(
        &mut self,
        label: &str,
        now: Instant,
        read: impl FnOnce() -> Option<u64>,
    ) -> Option<String> {
        let first = !self.seen.iter().any(|seen| seen == label);
        if !first && !self.due(now) {
            return None;
        }
        let free = read()?;
        self.last_read = Some(now);
        self.record(label, free)
    }

    /// A prefill piece of `rows` rows: read when it is the largest yet (the working set peaks
    /// there), else throttled like any other point.
    pub(crate) fn sample_prefill(
        &mut self,
        rows: usize,
        now: Instant,
        read: impl FnOnce() -> Option<u64>,
    ) -> Option<String> {
        if rows > self.largest_prefill {
            self.largest_prefill = rows;
            let free = read()?;
            self.last_read = Some(now);
            return self.record("the largest prefill piece", free);
        }
        self.sample("a prefill piece", now, read)
    }

    /// A session opened with `live` sessions now: read when it is the most live yet.
    pub(crate) fn sample_sessions(
        &mut self,
        live: usize,
        now: Instant,
        read: impl FnOnce() -> Option<u64>,
    ) -> Option<String> {
        if live <= self.most_sessions {
            return None;
        }
        self.most_sessions = live;
        let free = read()?;
        self.last_read = Some(now);
        self.record(&format!("{live} sessions live"), free)
    }

    /// The least free bytes seen so far.
    pub(crate) fn low(&self) -> Option<u64> {
        self.low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn layers(count: usize, slots: u64, slot_bytes: u64) -> Vec<LayerTier> {
        vec![LayerTier { slots, slot_bytes }; count]
    }

    #[test]
    fn a_tiered_start_reports_its_fraction_and_its_bytes_per_tier() {
        // Q4_K_M on a 24 GiB card: 40 layers of 256 experts, 229 slots resident, 1.77 MiB a slot.
        let receipt = TierReceipt::from_layers(
            (24 * GIB, GIB),
            2 * GIB,
            256,
            &layers(40, 229, 1_769_472),
            32 << 20,
        );
        assert_eq!(receipt.experts_total, 10_240);
        assert_eq!(receipt.experts_on_card, 9_160);
        assert_eq!(receipt.bytes_on_card, 9_160 * 1_769_472);
        assert_eq!(
            receipt.bytes_on_card + receipt.bytes_streamed,
            10_240 * 1_769_472,
            "every expert byte is on the card or streamed"
        );
        assert_eq!(receipt.bytes_pinned, 32 << 20);
        assert!((receipt.resident_percent() - 89.453).abs() < 0.01);
        assert!((receipt.card_used_percent() - 95.833).abs() < 0.01);
        let line = receipt.line("q36 cuda");
        assert!(
            line.contains("whole card 23552 of 24576 MiB used (95.8%, every process)"),
            "{line}"
        );
        assert!(
            line.contains(&format!("dense core {} B", 2 * GIB)),
            "{line}"
        );
        assert!(
            line.contains("experts 9160 of 10240 resident (89.5%)"),
            "{line}"
        );
        assert!(line.contains("33554432 B pinned on the host"), "{line}");
    }

    /// A file that mixes record types has different slot bytes per layer: the bytes are the sum
    /// of each layer's slots times that layer's slot bytes, not the slots times the widest.
    #[test]
    fn a_mixed_quant_file_is_summed_per_layer() {
        let mixed = [
            LayerTier {
                slots: 10,
                slot_bytes: 100,
            },
            LayerTier {
                slots: 10,
                slot_bytes: 300,
            },
            LayerTier {
                slots: 4,
                slot_bytes: 200,
            },
        ];
        let receipt = TierReceipt::from_layers((GIB, GIB / 2), 7, 16, &mixed, 0);
        assert_eq!(receipt.experts_total, 48);
        assert_eq!(receipt.experts_on_card, 24);
        assert_eq!(receipt.bytes_on_card, 10 * 100 + 10 * 300 + 4 * 200);
        assert_eq!(receipt.bytes_streamed, 6 * 100 + 6 * 300 + 12 * 200);
        let widest = 24 * 300;
        assert_ne!(
            receipt.bytes_on_card, widest,
            "not slots x the widest layer"
        );
        assert_eq!(
            receipt.bytes_on_card + receipt.bytes_streamed,
            16 * (100 + 300 + 200)
        );
    }

    #[test]
    fn a_resident_start_holds_every_expert_and_streams_none() {
        let receipt =
            TierReceipt::from_layers((24 * GIB, 2 * GIB), GIB, 256, &layers(40, 256, 1000), 0);
        assert_eq!(receipt.resident_percent(), 100.0);
        assert_eq!(receipt.bytes_streamed, 0);
        assert_eq!(receipt.card_used(), 22 * GIB);
    }

    #[test]
    fn slots_beyond_the_layer_are_clamped_and_an_empty_model_reads_zero() {
        let receipt = TierReceipt::from_layers((GIB, GIB), 0, 4, &layers(2, 9, 100), 0);
        assert_eq!(
            receipt.experts_on_card, 8,
            "a layer holds no more than its experts"
        );
        assert_eq!(receipt.bytes_streamed, 0);
        let empty = TierReceipt::from_layers((0, 0), 0, 0, &[], 0);
        assert_eq!(empty.resident_percent(), 0.0);
        assert_eq!(empty.card_used_percent(), 0.0);
    }

    #[test]
    fn a_model_without_experts_prints_its_dense_core_and_cache() {
        let receipt = TierReceipt::dense_only(
            (24 * GIB, 10 * GIB),
            7 * GIB,
            Some(CacheFigure {
                cells: 65_536,
                bytes: 3 * GIB,
            }),
        );
        let line = receipt.line("bolt cuda");
        assert!(
            line.contains("whole card 14336 of 24576 MiB used"),
            "{line}"
        );
        assert!(line.contains("cache 65536 cells (3221225472 B)"), "{line}");
        assert!(line.contains("no routed experts"), "{line}");
        let tiered = TierReceipt::from_layers((GIB, GIB), 0, 1, &layers(1, 1, 1), 0)
            .with_cache(CacheFigure { cells: 1, bytes: 2 });
        assert_eq!(tiered.cache, Some(CacheFigure { cells: 1, bytes: 2 }));
    }

    #[test]
    fn the_start_prints_the_receipt_or_says_why_it_cannot() {
        let receipt = TierReceipt::from_layers((GIB, GIB / 2), 5, 4, &layers(1, 2, 10), 0);
        let line = report_line("q36 cuda", Ok(receipt));
        assert!(
            line.starts_with("q36 cuda: memory receipt: whole card 512 of 1024 MiB"),
            "{line}"
        );
        let failed = report_line(
            "bolt cuda",
            Err(crate::EngError::MemoryPlan(
                "the card cannot be read".into(),
            )),
        );
        assert!(
            failed.starts_with("bolt cuda: memory receipt unavailable: "),
            "{failed}"
        );
        assert!(failed.contains("the card cannot be read"), "{failed}");
    }

    #[test]
    fn a_unified_memory_start_reports_the_machine_and_the_checkpoint() {
        let line = unified_line("q36 mlx", 36 * GIB, Some(27 * GIB), Some(20 * GIB));
        assert_eq!(
            line,
            format!(
                "q36 mlx: memory receipt: unified memory 36864 MiB, Metal working set 27648 MiB, \
                 checkpoint {} B",
                20 * GIB
            )
        );
        assert!(unified_line("q36 mlx", GIB, None, None).contains("Metal working set unknown"));
    }

    #[test]
    fn a_mark_reports_once_and_again_only_for_a_new_low() {
        let mut marks = FreeMarks::default();
        let line = marks.record("load", 2000 * MIB).expect("first");
        assert!(line.contains("free VRAM after load: 2000 MiB"), "{line}");
        assert_eq!(
            marks.record("load", 2000 * MIB),
            None,
            "same label, no new low"
        );
        assert!(
            marks.record("decode", 1990 * MIB).is_some(),
            "a new label reports"
        );
        assert_eq!(marks.record("decode", 1980 * MIB), None, "within the step");
        let low = marks.record("decode", 1900 * MIB).expect("a new low");
        assert!(low.contains("(new low)"), "{low}");
        assert!(marks.record("prefill", 1950 * MIB).is_some());
        assert_eq!(
            marks.low(),
            Some(1900 * MIB),
            "a higher reading never raises the low"
        );
    }

    /// The low is the true least, not the least that was reported: a dip too small to log is
    /// still the low.
    #[test]
    fn the_low_is_exact_though_the_log_steps_by_32_mib() {
        let mut marks = FreeMarks::default();
        marks.record("load", 2000 * MIB);
        assert_eq!(
            marks.record("decode", 1990 * MIB),
            Some("free VRAM after decode: 1990 MiB".into())
        );
        assert_eq!(
            marks.record("decode", 1985 * MIB),
            None,
            "5 MiB lower is not logged"
        );
        assert_eq!(marks.low(), Some(1985 * MIB), "but it is the low");
    }

    #[test]
    fn a_point_already_seen_is_read_again_only_after_the_interval() {
        use std::cell::Cell;
        let mut marks = FreeMarks::default();
        let start = Instant::now();
        let reads = Cell::new(0u32);
        let read = |free: u64| {
            reads.set(reads.get() + 1);
            Some(free * MIB)
        };
        assert!(
            marks
                .sample("a decode batch", start, || read(2000))
                .is_some()
        );
        // A thousand tokens inside the interval cost no driver read.
        for token in 1..=1000u64 {
            let at = start + Duration::from_millis(token);
            assert_eq!(marks.sample("a decode batch", at, || read(1999)), None);
        }
        assert_eq!(reads.get(), 1, "one query for the whole burst");
        // After the interval it reads again; a dip is kept as the low.
        let later = start + FreeMarks::INTERVAL;
        marks.sample("a decode batch", later, || read(1500));
        assert_eq!(reads.get(), 2);
        assert_eq!(marks.low(), Some(1500 * MIB));
        // A label first reached reads at once, whatever the interval.
        assert!(
            marks
                .sample("another point", later, || read(1400))
                .is_some()
        );
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn the_largest_prefill_and_the_most_sessions_read_when_they_grow() {
        let mut marks = FreeMarks::default();
        let now = Instant::now();
        assert!(
            marks
                .sample_prefill(128, now, || Some(2000 * MIB))
                .is_some()
        );
        // A smaller piece is a point seen for the first time: it reads once, then not again
        // inside the interval.
        assert!(marks.sample_prefill(64, now, || Some(1990 * MIB)).is_some());
        assert_eq!(marks.sample_prefill(64, now, || panic!("no read")), None);
        // A larger piece reads at once and names itself the largest.
        let line = marks
            .sample_prefill(1024, now, || Some(1500 * MIB))
            .expect("larger");
        assert!(line.contains("the largest prefill piece"), "{line}");
        assert!(marks.sample_sessions(1, now, || Some(1490 * MIB)).is_some());
        assert_eq!(marks.sample_sessions(1, now, || panic!("no read")), None);
        let line = marks
            .sample_sessions(3, now, || Some(1300 * MIB))
            .expect("more sessions");
        assert!(line.contains("3 sessions live"), "{line}");
        assert_eq!(marks.low(), Some(1300 * MIB));
    }
}
