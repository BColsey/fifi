#![cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(dead_code, reason = "consumed by the CUDA layer walk")
)]
//! CPU serving of Flint routed experts: configuration, expert access and the hot-set policy.
//!
//! WHAT: [`CpuMoe`] computes routed experts of a Flint MoE layer on the host, from the stored
//! EXL3 bytes the pinned host store already holds, so only the layer's normed activation row
//! crosses the bus (about 8 KB) and the unweighted expert outputs come back. It serves
//! two modes, both default off:
//!
//! - offload (`cpu_moe_offload_layers = N`): every routed expert of the first N MoE layers runs
//!   on the CPU;
//! - split (`cpu_moe_split_experts = N`): each layer keeps `256 - N` experts on the device and
//!   computes the other picks on the CPU instead of copying them in. [`HotTracker`] decides,
//!   slowly, which experts hold the device slots.
//!
//! The two are exclusive, as in the TabbyAPI keys they mirror. The kernel is `kernels::exl3_cpu`;
//! this module owns only policy and the narrow call the layer walk makes.
//!
//! WHY a slow hot set: a swap costs one expert's PCIe copy (about as much as the CPU needs to
//! compute that expert), so an expert should take a device slot only when it will be used
//! several more times; a tier that swaps on every miss moves more bytes than it saves.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::EngError;
use crate::kernels::exl3_cpu::{ExpertCpu, ExpertView, PairJob};
use crate::kernels::nvfp4_cpu::WaitPolicy;

/// The offload count that names every routed layer, as in the TabbyAPI key.
pub(crate) const ALL_LAYERS: usize = 999;

/// Routed experts per layer in the published Flint packs.
pub(crate) const EXPERTS: usize = 256;

/// Worker threads when the options leave `cpu_moe_threads` at 0: the expert kernels are
/// DRAM-bound, so more threads than this do not help.
pub(crate) const DEFAULT_THREADS: usize = 6;

/// Which routed experts the CPU serves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CpuMoeConfig {
    offload_layers: usize,
    split_experts: usize,
    threads: usize,
}

impl CpuMoeConfig {
    /// Validates the three option values. An offload count of [`ALL_LAYERS`] or more means every
    /// routed layer, whatever the model's layer count.
    pub(crate) fn new(
        offload_layers: u32,
        split_experts: u32,
        threads: u32,
    ) -> Result<Self, EngError> {
        let bad = |what: &str| Err(EngError::BadRequest(format!("Flint CPU MoE: {what}")));
        if offload_layers != 0 && split_experts != 0 {
            return bad("cpu_moe_offload_layers and cpu_moe_split_experts are exclusive");
        }
        // The device keeps the experts of one prefill piece; the bank sizing enforces the floor.
        if split_experts as usize >= EXPERTS {
            return bad("cpu_moe_split_experts must be below 256");
        }
        if threads > 256 {
            return bad("cpu_moe_threads must be at most 256");
        }
        Ok(Self {
            offload_layers: offload_layers as usize,
            split_experts: split_experts as usize,
            threads: if threads == 0 {
                DEFAULT_THREADS
            } else {
                threads as usize
            },
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.offload_layers != 0 || self.split_experts != 0
    }

    pub(crate) fn offload_layers(&self) -> usize {
        self.offload_layers
    }

    pub(crate) fn split_experts(&self) -> usize {
        self.split_experts
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    /// Whether every routed expert of the `ordinal`-th routed layer runs on the CPU.
    pub(crate) fn offloads(&self, ordinal: usize) -> bool {
        ordinal < self.offload_layers
    }
}

/// Host-side access to the stored bytes of the routed experts of one layer.
pub(crate) trait ExpertSource {
    fn expert(&self, id: usize) -> Result<ExpertView<'_>, EngError>;
}

/// The CPU expert engine and its counters.
pub(crate) struct CpuMoe {
    config: CpuMoeConfig,
    cpu: ExpertCpu,
    hot: std::sync::Mutex<HotTracker>,
    calls: AtomicU64,
    pairs: AtomicU64,
    nanos: AtomicU64,
}

impl CpuMoe {
    pub(crate) fn new(config: CpuMoeConfig) -> Result<Self, EngError> {
        Ok(Self {
            config,
            cpu: ExpertCpu::new(
                config.threads(),
                WaitPolicy::SpinThenPark(std::time::Duration::from_micros(500)),
                None,
            )?
            .with_published_rounding(true),
            hot: std::sync::Mutex::new(HotTracker::new(0)),
            calls: AtomicU64::new(0),
            pairs: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        })
    }

    pub(crate) fn config(&self) -> CpuMoeConfig {
        self.config
    }

    /// One line for the load log: the configuration and the kernel tier.
    pub(crate) fn describe(&self) -> String {
        let offload = match self.config.offload_layers() {
            layers if layers >= ALL_LAYERS => "all".to_owned(),
            layers => layers.to_string(),
        };
        format!(
            "Flint CPU MoE: offload_layers={offload} split_experts={} threads={} isa={:?}",
            self.config.split_experts(),
            self.config.threads(),
            self.cpu.isa(),
        )
    }

    /// One line with the work done on the CPU since the process started.
    pub(crate) fn summary(&self) -> String {
        let calls = self.calls.load(Ordering::Relaxed);
        let pairs = self.pairs.load(Ordering::Relaxed);
        let nanos = self.nanos.load(Ordering::Relaxed);
        format!(
            "Flint CPU MoE total: calls={calls} pairs={pairs} cpu_ms={:.1} us_per_pair={:.1}",
            nanos as f64 / 1e6,
            nanos as f64 / 1e3 / pairs.max(1) as f64,
        )
    }

    /// Pairs computed on the CPU so far.
    #[cfg(test)]
    fn pairs_done(&self) -> u64 {
        self.pairs.load(Ordering::Relaxed)
    }

    /// Feeds one layer step to the hot-set policy (split mode only) and returns the slot swap it
    /// proposes now.
    pub(crate) fn observe(
        &self,
        ordinal: usize,
        ids: &[i32],
        resident: &[bool; EXPERTS],
    ) -> Option<Swap> {
        if self.config.split_experts() == 0 {
            return None;
        }
        let mut hot = self.hot.lock().unwrap_or_else(|e| e.into_inner());
        hot.note(ordinal, ids);
        hot.end_token(ordinal);
        hot.propose(ordinal, resident)
    }

    /// Computes (row, expert) pairs of one layer, read through `source`: the unweighted expert outputs of activation
    /// row `row` of `xs` land in `out`, `hidden` values per pair, in the order of `pairs`.
    pub(crate) fn compute(
        &self,
        source: &dyn ExpertSource,
        xs: &[f32],
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let jobs = pairs
            .iter()
            .map(|&(row, id)| {
                Ok(PairJob {
                    expert: source.expert(id)?,
                    row,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        self.compute_jobs(xs, &jobs, out)
    }

    /// The compute half of [`Self::compute`] over jobs already built (their expert views read
    /// the pinned host store in place, so they may cross a thread while the store's borrow is
    /// held). Thread-count independent results, as `expert_pairs` guarantees.
    pub(crate) fn compute_jobs(
        &self,
        xs: &[f32],
        jobs: &[PairJob<'_>],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let started = std::time::Instant::now();
        self.cpu.expert_pairs(xs, jobs, out)?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.pairs.fetch_add(jobs.len() as u64, Ordering::Relaxed);
        self.nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }
}

impl CpuMoe {
    /// Indices into `ids` (row-major, `top_k` per row) of the picks the CPU serves for the
    /// `ordinal`-th routed layer. `resident` says which experts hold a device slot; split mode
    /// needs it and sends every other pick to the CPU.
    pub(crate) fn cpu_pairs(
        &self,
        ordinal: usize,
        ids: &[i32],
        resident: Option<&[bool; EXPERTS]>,
    ) -> Vec<usize> {
        if self.config.offloads(ordinal) {
            return (0..ids.len()).collect();
        }
        match resident {
            Some(resident) if self.config.split_experts() != 0 => (0..ids.len())
                .filter(|&p| !resident[ids[p] as usize])
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// The ids the device runs when the CPU serves `cpu_pairs`: each such pick is replaced by a
/// resident expert the same row does not pick (its device output is overwritten by the CPU
/// result), so no pick of the device pass misses its slots and no row repeats an expert.
/// Returns `None` when no resident expert is free for some row.
pub(crate) fn device_ids(
    ids: &[i32],
    top_k: usize,
    cpu_pairs: &[usize],
    resident: &[bool; EXPERTS],
) -> Option<Vec<i32>> {
    let mut out = ids.to_vec();
    for (row, picks) in ids.chunks_exact(top_k).enumerate() {
        let mut taken: Vec<i32> = picks.to_vec();
        let mut fillers = (0..EXPERTS as i32).filter(|&e| resident[e as usize]);
        for rank in 0..top_k {
            if !cpu_pairs.contains(&(row * top_k + rank)) {
                continue;
            }
            let filler = fillers.find(|e| !taken.contains(e))?;
            taken.push(filler);
            out[row * top_k + rank] = filler;
        }
    }
    Some(out)
}

/// The engine the layer walk uses. A Flint engine owns its process, so one slot is enough.
/// `ENABLED` lets the default path (no CPU experts) pay one atomic load per layer.
static ACTIVE: std::sync::Mutex<Option<std::sync::Arc<CpuMoe>>> = std::sync::Mutex::new(None);
static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes `config` the engine of the process: a started [`CpuMoe`] when CPU experts are on,
/// nothing when they are off (which also clears an earlier engine). A second enabled engine
/// is refused while one is installed, so one engine never reads another's configuration.
pub(crate) fn install(config: CpuMoeConfig) -> Result<(), EngError> {
    let mut slot = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    if !config.enabled() {
        *slot = None;
        ENABLED.store(false, Ordering::Release);
        return Ok(());
    }
    if slot.is_some() {
        return Err(EngError::BadRequest(
            "Flint CPU MoE: a CPU expert engine is already installed in this process".into(),
        ));
    }
    *slot = Some(std::sync::Arc::new(CpuMoe::new(config)?));
    ENABLED.store(true, Ordering::Release);
    Ok(())
}

/// The installed engine, or `None` when CPU experts are off.
pub(crate) fn active() -> Option<std::sync::Arc<CpuMoe>> {
    if !ENABLED.load(Ordering::Acquire) {
        return None;
    }
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// One exchange of device slots: `incoming` takes the slot of `outgoing`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Swap {
    pub(crate) incoming: usize,
    pub(crate) outgoing: usize,
}

/// Decayed use counts per expert of one layer, and the rule that turns them into slot swaps.
///
/// Every routed pick adds one; each token boundary multiplies all counts by `DECAY`. A layer
/// proposes a swap only every `SWAP_EVERY` tokens, and only when the busiest expert outside the
/// device beats the quietest one inside by `MARGIN` (a swap costs a PCIe copy, a CPU use does
/// not), so the device set follows the workload slowly.
pub(crate) struct HotTracker {
    counts: Vec<[f32; EXPERTS]>,
    since_swap: Vec<u32>,
}

const DECAY: f32 = 0.995;
const SWAP_EVERY: u32 = 8;
const MARGIN: f32 = 2.0;

impl HotTracker {
    pub(crate) fn new(layers: usize) -> Self {
        Self {
            counts: vec![[0.0; EXPERTS]; layers],
            since_swap: vec![0; layers],
        }
    }

    fn reach(&mut self, layer: usize) {
        if self.counts.len() <= layer {
            self.counts.resize(layer + 1, [0.0; EXPERTS]);
            self.since_swap.resize(layer + 1, 0);
        }
    }

    pub(crate) fn note(&mut self, layer: usize, ids: &[i32]) {
        self.reach(layer);
        for &id in ids {
            self.counts[layer][id as usize] += 1.0;
        }
    }

    /// Ends a token of `layer`: decays its counts and counts it toward the next swap.
    pub(crate) fn end_token(&mut self, layer: usize) {
        self.reach(layer);
        self.counts[layer].iter_mut().for_each(|c| *c *= DECAY);
        self.since_swap[layer] = self.since_swap[layer].saturating_add(1);
    }

    /// The swap `layer` wants now, given which experts hold a device slot, or `None`.
    pub(crate) fn propose(&mut self, layer: usize, resident: &[bool; EXPERTS]) -> Option<Swap> {
        self.reach(layer);
        if self.since_swap[layer] < SWAP_EVERY {
            return None;
        }
        let counts = &self.counts[layer];
        let mut incoming: Option<usize> = None;
        let mut outgoing: Option<usize> = None;
        for (id, &count) in counts.iter().enumerate() {
            if resident[id] {
                if outgoing.is_none_or(|o| count < counts[o]) {
                    outgoing = Some(id);
                }
            } else if incoming.is_none_or(|i| count > counts[i]) {
                incoming = Some(id);
            }
        }
        let (incoming, outgoing) = (incoming?, outgoing?);
        if counts[incoming] < MARGIN * counts[outgoing] + MARGIN {
            return None;
        }
        self.since_swap[layer] = 0;
        Some(Swap { incoming, outgoing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trellis, input scales and output scales of one projection.
    type Packed = (Vec<u8>, Vec<u8>, Vec<u8>);

    /// Experts as owned buffers, standing in for the pinned host store.
    struct Fake {
        experts: Vec<[Packed; 3]>,
    }

    impl Fake {
        const H: usize = 256;
        const F: usize = 128;

        fn new() -> Self {
            let mut state = 0x2545_f491_4f6c_dd1du64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut projection = |k: usize, n: usize| {
                let trellis = (0..(k / 16) * (n / 16) * 128)
                    .map(|_| (next() >> 24) as u8)
                    .collect();
                let scales = |len: usize, next: &mut dyn FnMut() -> u64| -> Vec<u8> {
                    (0..len)
                        .flat_map(|_| {
                            (1.0 + (next() >> 40) as f32 / (1u64 << 25) as f32).to_le_bytes()
                        })
                        .collect()
                };
                (trellis, scales(k, &mut next), scales(n, &mut next))
            };
            let experts = (0..4)
                .map(|_| {
                    [
                        projection(Self::H, Self::F),
                        projection(Self::H, Self::F),
                        projection(Self::F, Self::H),
                    ]
                })
                .collect();
            Self { experts }
        }
    }

    impl ExpertSource for Fake {
        fn expert(&self, id: usize) -> Result<ExpertView<'_>, EngError> {
            use crate::kernels::exl3_cpu::Exl3Projection;
            let [g, u, d] = self
                .experts
                .get(id)
                .ok_or_else(|| EngError::BadRequest("no such fake expert".into()))?;
            Ok(ExpertView {
                gate: Exl3Projection::new([Self::H, Self::F], 64, &g.0, &g.1, &g.2)?,
                up: Exl3Projection::new([Self::H, Self::F], 64, &u.0, &u.1, &u.2)?,
                down: Exl3Projection::new([Self::F, Self::H], 64, &d.0, &d.1, &d.2)?,
            })
        }
    }

    #[test]
    fn compute_returns_each_pair_in_the_order_asked() {
        let fake = Fake::new();
        let moe = CpuMoe::new(CpuMoeConfig::new(1, 0, 2).unwrap()).unwrap();
        let xs: Vec<f32> = (0..2 * Fake::H)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let pairs = [(1usize, 3usize), (0, 2), (1, 0)];
        let mut out = vec![0.0f32; pairs.len() * Fake::H];
        moe.compute(&fake, &xs, &pairs, &mut out).unwrap();
        for (i, &(row, id)) in pairs.iter().enumerate() {
            let mut one = vec![0.0f32; Fake::H];
            moe.compute(&fake, &xs, &[(row, id)], &mut one).unwrap();
            assert_eq!(&out[i * Fake::H..(i + 1) * Fake::H], &one[..]);
            assert!(one.iter().any(|v| *v != 0.0));
        }
        assert_eq!(moe.pairs_done(), 3 + 3);
        assert!(moe.compute(&fake, &xs, &[(2, 0)], &mut one_row()).is_err());
        assert!(moe.compute(&fake, &xs, &[(0, 9)], &mut one_row()).is_err());
    }

    fn one_row() -> Vec<f32> {
        vec![0.0; Fake::H]
    }

    #[test]
    fn cpu_pairs_follow_the_mode_and_device_ids_avoid_repeats() {
        let mut resident = [false; EXPERTS];
        resident[..8].fill(true);
        let ids = [0, 1, 2, 3, 4, 100, 7, 6, 5, 200, 201, 202];
        let offload = CpuMoe::new(CpuMoeConfig::new(2, 0, 2).unwrap()).unwrap();
        assert_eq!(offload.cpu_pairs(1, &ids, Some(&resident)).len(), 12);
        assert!(offload.cpu_pairs(2, &ids, Some(&resident)).is_empty());
        let split = CpuMoe::new(CpuMoeConfig::new(0, 248, 2).unwrap()).unwrap();
        let cpu = split.cpu_pairs(9, &ids, Some(&resident));
        assert_eq!(cpu, vec![5, 9, 10, 11]);
        assert!(split.cpu_pairs(9, &ids, None).is_empty());
        let device = device_ids(&ids, 6, &cpu, &resident).unwrap();
        assert_eq!(&device[..5], &ids[..5], "served picks keep their ids");
        for row in device.as_chunks::<6>().0 {
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 6, "{row:?}");
            assert!(row.iter().all(|&e| resident[e as usize]));
        }
        // The row picks five resident experts and one that is not; no spare resident remains.
        let mut tight = [false; EXPERTS];
        tight[..5].fill(true);
        assert!(device_ids(&[0, 1, 2, 3, 4, 100], 6, &[5], &tight).is_none());
    }

    #[test]
    fn install_refuses_a_second_engine_and_a_default_config_clears() {
        // The statics are process-wide: this is the only test that installs.
        assert!(active().is_none());
        install(CpuMoeConfig::new(1, 0, 2).unwrap()).unwrap();
        assert_eq!(active().unwrap().config().offload_layers(), 1);
        assert!(install(CpuMoeConfig::new(0, 32, 2).unwrap()).is_err());
        assert_eq!(active().unwrap().config().offload_layers(), 1);
        install(CpuMoeConfig::default()).unwrap();
        assert!(active().is_none());
        install(CpuMoeConfig::new(0, 32, 2).unwrap()).unwrap();
        assert_eq!(active().unwrap().config().split_experts(), 32);
        install(CpuMoeConfig::default()).unwrap();
        assert!(active().is_none());
    }

    #[test]
    fn config_validates_exclusivity_ranges_and_defaults() {
        assert!(!CpuMoeConfig::default().enabled());
        let all = CpuMoeConfig::new(999, 0, 0).unwrap();
        assert_eq!(
            (all.offload_layers(), all.threads()),
            (999, DEFAULT_THREADS)
        );
        assert!(all.offloads(46) && all.offloads(998));
        let split = CpuMoeConfig::new(0, 32, 4).unwrap();
        assert_eq!((split.split_experts(), split.threads()), (32, 4));
        assert!(split.enabled() && !split.offloads(0));
        assert!(CpuMoeConfig::new(3, 32, 0).is_err());
        assert!(CpuMoeConfig::new(0, 256, 0).is_err());
        assert!(CpuMoeConfig::new(0, 0, 1000).is_err());
    }

    #[test]
    fn the_hot_set_moves_slowly_and_only_for_a_clear_winner() {
        let mut hot = HotTracker::new(1);
        let mut resident = [false; EXPERTS];
        resident[..4].fill(true);
        // Resident experts 0..4 are used lightly, expert 100 heavily.
        for _ in 0..SWAP_EVERY - 1 {
            hot.note(0, &[0, 1, 100, 100, 100, 100]);
            hot.end_token(0);
            assert_eq!(hot.propose(0, &resident), None, "too early for a swap");
        }
        hot.note(0, &[0, 1, 100, 100, 100, 100]);
        hot.end_token(0);
        let swap = hot.propose(0, &resident).unwrap();
        assert_eq!(swap.incoming, 100);
        assert!(
            [2, 3].contains(&swap.outgoing),
            "a never-used expert leaves"
        );
        // The swap restarts the interval.
        assert_eq!(hot.propose(0, &resident), None);
        // A near tie never swaps.
        let mut even = HotTracker::new(1);
        for _ in 0..SWAP_EVERY {
            even.note(0, &[0, 1, 2, 3, 100]);
            even.note(0, &[0, 1, 2, 3, 101]);
            even.end_token(0);
        }
        assert_eq!(even.propose(0, &resident), None);
    }
}
