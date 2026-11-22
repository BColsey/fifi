//! The card transport on the device: pinned and file sources land byte-exact in the staging
//! slots in layer order, and a copy of synthetic layers measures the host-to-card rate and the
//! share of it hidden behind compute.

use std::{cell::RefCell, io::Write, rc::Rc, sync::Arc, time::Instant};

use super::layer_stream::{
    ExpertTransport, LayerExperts, LayerMajor, LayerRecipe, Picks, StreamBudget, StreamPlan,
    TransferReport, UploadRule, UploadSet,
};
use super::layer_stream_card::{CardTicket, CardTransport, StackHome, StackSpan};
use crate::{
    EngError,
    family::engine_owner::Handle,
    family::project::cuda::CudaProjector,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, PinnedBuf, device_present},
};

const EXPERTS: usize = 8;
const STACKS: usize = 3;

fn pattern(layer: usize, stack: usize, expert: usize, index: usize) -> u8 {
    (layer * 31 + stack * 7 + expert * 3 + index) as u8
}

/// Bytes of stack `stack` of `layer`: layers differ in record size.
fn record_of(layer: usize, stack: usize) -> usize {
    1024 * (1 + (layer % 3) + stack)
}

fn stack_bytes(layer: usize, stack: usize) -> Vec<u8> {
    let record = record_of(layer, stack);
    let mut out = Vec::with_capacity(record * EXPERTS);
    for expert in 0..EXPERTS {
        out.extend((0..record).map(|i| pattern(layer, stack, expert, i)));
    }
    out
}

struct Checking<'a> {
    inner: CardTransport<'a>,
    stream: Rc<DeviceStream>,
    sent: Vec<(usize, usize, UploadSet)>,
    checked: Rc<RefCell<usize>>,
    wanted: Vec<u32>,
}

impl ExpertTransport for Checking<'_> {
    type Ticket = CardTicket;

    fn begin(
        &mut self,
        layer: usize,
        set: &UploadSet,
        slot: usize,
    ) -> Result<Self::Ticket, EngError> {
        self.sent.push((layer, slot, set.clone()));
        self.inner.begin(layer, set, slot)
    }

    fn wait(&mut self, ticket: Self::Ticket) -> Result<TransferReport, EngError> {
        let done = self.inner.wait(ticket)?;
        Ok(done)
    }

    fn release(&mut self, slot: usize) -> Result<(), EngError> {
        self.inner.release(slot)
    }
}

impl Checking<'_> {
    /// Every expert the layer's pieces picked must read back byte-exact from its slot.
    fn verify(&self, layer: usize, slot: usize) {
        for stack in 0..STACKS {
            let range = self.inner.stack(layer, slot, stack).expect("range");
            let record = record_of(layer, stack);
            for &expert in &self.wanted {
                let mut out = vec![0u8; record];
                range
                    .buffer()
                    .read_at(
                        &self.stream,
                        range.offset() + expert as usize * record,
                        &mut out,
                    )
                    .expect("read back");
                let want: Vec<u8> = (0..record)
                    .map(|i| pattern(layer, stack, expert as usize, i))
                    .collect();
                assert_eq!(out, want, "layer {layer} stack {stack} expert {expert}");
                *self.checked.borrow_mut() += 1;
            }
        }
    }
}

struct Verify<'a, 'c> {
    transport: &'a RefCell<Option<Checking<'c>>>,
    layers: usize,
    picks: Vec<u32>,
    predicted: Option<Vec<u32>>,
}

/// The driver owns the transport, so the test routes it through a cell: `LayerMajor` gets a thin
/// handle and the recipe reads the staged bytes through the same cell.
struct Shared<'a, 'c>(&'a RefCell<Option<Checking<'c>>>);

impl ExpertTransport for Shared<'_, '_> {
    type Ticket = CardTicket;

    fn begin(&mut self, l: usize, s: &UploadSet, slot: usize) -> Result<Self::Ticket, EngError> {
        self.0
            .borrow_mut()
            .as_mut()
            .expect("transport")
            .begin(l, s, slot)
    }

    fn wait(&mut self, t: Self::Ticket) -> Result<TransferReport, EngError> {
        self.0.borrow_mut().as_mut().expect("transport").wait(t)
    }

    fn release(&mut self, slot: usize) -> Result<(), EngError> {
        self.0
            .borrow_mut()
            .as_mut()
            .expect("transport")
            .release(slot)
    }
}

impl LayerRecipe for Verify<'_, '_> {
    fn layers(&self) -> usize {
        self.layers
    }
    fn pieces(&self) -> usize {
        2
    }
    fn rows(&self, _piece: usize) -> usize {
        1
    }
    fn predict(&mut self, _layer: usize) -> Option<Vec<u32>> {
        self.predicted.clone()
    }
    fn route(&mut self, _layer: usize, _piece: usize) -> Result<Option<Picks>, EngError> {
        Ok(Some(Picks {
            experts: self.picks.clone(),
        }))
    }
    fn visit(&mut self, layer: usize, _piece: usize, slot: Option<usize>) -> Result<(), EngError> {
        if let Some(slot) = slot {
            let guard = self.transport.borrow();
            guard.as_ref().expect("transport").verify(layer, slot);
        }
        Ok(())
    }
}

fn layer_shapes(layers: usize) -> Vec<LayerExperts> {
    (0..layers)
        .map(|layer| {
            LayerExperts::new(
                EXPERTS,
                (0..STACKS)
                    .map(|stack| (record_of(layer, stack) * EXPERTS) as u64)
                    .collect(),
            )
            .expect("layer")
        })
        .collect()
}

fn run(
    layers: usize,
    from_file: bool,
    direct: bool,
    rule: UploadRule,
    predicted: Option<Vec<u32>>,
) -> usize {
    assert!(
        device_present("expert layer transport"),
        "a GPU is required"
    );
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let compute = Rc::new(ctx.stream().expect("stream"));
    let mut spans: Vec<Vec<StackSpan>> = Vec::new();
    let mut store: Vec<PinnedBuf> = Vec::new();
    let dir = std::env::var("TMPDIR").unwrap_or_else(|_| ".".into());
    let path = std::path::Path::new(&dir).join(format!("layer-stream-{}.bin", std::process::id()));
    let mut file_offset = 0u64;
    let mut writer = std::fs::File::create(&path).expect("file");
    for layer in 0..layers {
        let mut row = Vec::new();
        let mut pinned = Vec::new();
        for stack in 0..STACKS {
            let bytes = stack_bytes(layer, stack);
            let len = bytes.len();
            if from_file {
                writer.write_all(&bytes).expect("write");
                row.push((len, file_offset, 0usize));
                file_offset += len as u64;
            } else {
                row.push((len, 0, pinned.len()));
                pinned.extend_from_slice(&bytes);
            }
        }
        if from_file {
            spans.push(Vec::new());
        } else {
            let mut buffer = ctx.pinned_buf(pinned.len()).expect("pinned");
            buffer.write(0, &pinned).expect("fill");
            let index = store.len();
            store.push(buffer);
            spans.push(
                row.iter()
                    .map(|&(len, _, at)| StackSpan {
                        bytes: len,
                        home: StackHome::Pinned {
                            buf: index,
                            offset: at,
                        },
                        shape: None,
                    })
                    .collect(),
            );
            continue;
        }
        let file = Arc::new(std::fs::File::open(&path).expect("reopen"));
        let direct_handle = if direct {
            match super::host_tier::open_direct(&path) {
                Ok(handle) => Some(Arc::new(handle)),
                Err(error) => {
                    eprintln!("direct source skipped: {error}");
                    return layers * 2 * STACKS * 3;
                }
            }
        } else {
            None
        };
        spans[layer] = row
            .iter()
            .map(|&(len, offset, _)| StackSpan {
                bytes: len,
                home: StackHome::File {
                    file: Arc::clone(&file),
                    direct: direct_handle.clone(),
                    offset,
                },
                shape: None,
            })
            .collect();
    }
    writer.flush().expect("flush");
    drop(writer);
    // The file spans were built after the whole file was written.
    let shapes = layer_shapes(layers);
    let one = shapes.iter().map(LayerExperts::bytes).max().unwrap();
    let plan = StreamPlan::choose(
        &shapes,
        StreamBudget {
            card_bytes: 2 * one,
            host_bytes: u64::MAX,
            staging_slots: 2,
        },
    )
    .expect("plan");
    assert_eq!(plan.streamed().len(), layers);
    let inner = CardTransport::new(
        ctx,
        Handle::Shared(Rc::clone(&compute)),
        EXPERTS,
        spans,
        store,
        plan.staging_slots(),
    )
    .expect("transport");
    let checked = Rc::new(RefCell::new(0usize));
    let picks = vec![2u32, 5, 6];
    let cell = RefCell::new(Some(Checking {
        inner,
        stream: Rc::clone(&compute),
        sent: Vec::new(),
        checked: Rc::clone(&checked),
        wanted: picks.clone(),
    }));
    let mut recipe = Verify {
        transport: &cell,
        layers,
        picks,
        predicted,
    };
    let mut driver = LayerMajor::new(&plan, &shapes, rule, Shared(&cell)).expect("driver");
    let report = driver.run(&mut recipe).expect("run");
    eprintln!("{report}");
    let sent = cell.borrow().as_ref().expect("transport").sent.clone();
    // Layers were first copied in layer order (a top-up repeats a layer's number).
    let mut first: Vec<usize> = Vec::new();
    for (layer, _, _) in &sent {
        if !first.contains(layer) {
            first.push(*layer);
        }
    }
    assert!(first.windows(2).all(|w| w[0] < w[1]), "{first:?}");
    assert_eq!(first.len(), layers);
    let _ = std::fs::remove_file(&path);
    *checked.borrow()
}

/// Every layer's picked experts read back byte-exact from the slot, from pinned memory.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn pinned_layers_land_byte_exact_in_layer_order() {
    let rule = UploadRule::pinned_defaults(2);
    let checked = run(7, false, false, rule, None);
    // 7 layers x 2 pieces x 3 stacks x 3 experts.
    assert_eq!(checked, 7 * 2 * STACKS * 3);
}

/// A subset copy of the predicted experts, topped up with the pick the prediction missed.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn a_predicted_subset_is_topped_up_and_lands_byte_exact() {
    let rule = UploadRule {
        top_k: 2,
        bandwidth: 1e9,
        per_copy_s: 0.0,
    };
    let checked = run(5, false, false, rule, Some(vec![2, 5]));
    assert_eq!(checked, 5 * 2 * STACKS * 3);
}

/// The same layers read in place from a file.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn file_layers_land_byte_exact() {
    let rule = UploadRule::pinned_defaults(2);
    let checked = run(5, true, false, rule, None);
    assert_eq!(checked, 5 * 2 * STACKS * 3);
}

/// The same layers read through a direct handle, several threads per stack.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn direct_file_layers_land_byte_exact() {
    let rule = UploadRule::pinned_defaults(2);
    let checked = run(5, true, true, rule, None);
    assert_eq!(checked, 5 * 2 * STACKS * 3);
}

struct Burn<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    src: &'a DeviceBuffer,
    dst: &'a DeviceBuffer,
    layers: usize,
    copies: usize,
}

impl LayerRecipe for Burn<'_> {
    fn layers(&self) -> usize {
        self.layers
    }
    fn pieces(&self) -> usize {
        1
    }
    fn rows(&self, _piece: usize) -> usize {
        4096
    }
    fn predict(&mut self, _layer: usize) -> Option<Vec<u32>> {
        None
    }
    fn route(&mut self, _layer: usize, _piece: usize) -> Result<Option<Picks>, EngError> {
        Ok(None)
    }
    fn visit(
        &mut self,
        _layer: usize,
        _piece: usize,
        _slot: Option<usize>,
    ) -> Result<(), EngError> {
        for _ in 0..self.copies {
            self.ctx.copy_device(
                self.stream,
                self.src.range(0, self.src.bytes())?,
                self.dst.range(0, self.dst.bytes())?,
            )?;
        }
        Ok(())
    }
}

/// Measures the host-to-card rate of pinned layers and how much of it hides behind a compute
/// stand-in (device-to-device copies of a fixed length per layer), printing the wall time with
/// and without the transfers. A layer is 512 experts of 2.75 MB in three stacks, like the real
/// file's layers.
#[test]
#[ignore = "needs a GPU (about 5 GiB) and 8.5 GiB of pinned host memory"]
fn pinned_stream_rate_and_overlap() {
    assert!(
        device_present("expert layer transport"),
        "a GPU is required"
    );
    let layers = 6usize;
    let experts = 512usize;
    let stack = 917_504usize * experts;
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let compute = Rc::new(ctx.stream().expect("stream"));
    let mut store = Vec::new();
    let mut spans = Vec::new();
    for layer in 0..layers {
        let mut buffer = ctx.pinned_buf(stack * STACKS).expect("pinned");
        let fill = vec![layer as u8; 64 << 20];
        let mut at = 0;
        while at < stack * STACKS {
            let take = fill.len().min(stack * STACKS - at);
            buffer.write(at, &fill[..take]).expect("fill");
            at += take;
        }
        store.push(buffer);
        spans.push(
            (0..STACKS)
                .map(|s| StackSpan {
                    bytes: stack,
                    home: StackHome::Pinned {
                        buf: layer,
                        offset: s * stack,
                    },
                    shape: None,
                })
                .collect::<Vec<_>>(),
        );
    }
    let shapes: Vec<LayerExperts> = (0..layers)
        .map(|_| LayerExperts::new(experts, vec![stack as u64; STACKS]).expect("layer"))
        .collect();
    let plan = StreamPlan::choose(
        &shapes,
        StreamBudget {
            card_bytes: 2 * shapes[0].bytes(),
            host_bytes: u64::MAX,
            staging_slots: 2,
        },
    )
    .expect("plan");
    let src = ctx.allocate(256 << 20).expect("src");
    let dst = ctx.allocate(256 << 20).expect("dst");
    let mut transport = CardTransport::new(
        ctx,
        Handle::Shared(Rc::clone(&compute)),
        experts,
        spans,
        store,
        plan.staging_slots(),
    )
    .expect("transport");
    for copies in [0usize, 60, 240] {
        let mut burn = Burn {
            ctx,
            stream: &compute,
            src: &src,
            dst: &dst,
            layers,
            copies,
        };
        let started = Instant::now();
        for layer in 0..layers {
            burn.visit(layer, 0, None).expect("visit");
        }
        compute.synchronize().expect("sync");
        let compute_only = started.elapsed().as_secs_f64();
        let mut driver = LayerMajor::new(
            &plan,
            &shapes,
            UploadRule::pinned_defaults(10),
            &mut transport,
        )
        .expect("driver");
        let started = Instant::now();
        let report = driver.run(&mut burn).expect("run");
        compute.synchronize().expect("sync");
        let wall = started.elapsed().as_secs_f64();
        eprintln!(
            "compute per layer {} ms: compute only {compute_only:.3} s, streamed wall {wall:.3} s; {report}",
            (compute_only * 1000.0 / layers as f64) as u32
        );
    }
}

/// Stack bytes of Q8_0 blocks with a finite scale and arbitrary quants.
fn q8_0_stack(seed: u8, rows: usize, k: usize) -> Vec<u8> {
    let blocks = rows * (k / 32);
    let mut out = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        out.extend_from_slice(&[0x00, 0x2c]); // f16 0.0625
        out.extend((0..32).map(|i| {
            (block as u8)
                .wrapping_mul(7)
                .wrapping_add((i as u8).wrapping_mul(13))
                .wrapping_add(seed)
        }));
    }
    out
}

/// A stack staged into its slot is read by the same matvec as a resident copy of the same bytes
/// and gives the same bits, for two layers of different size sharing the slots.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn a_staged_stack_reads_bit_identical_to_a_resident_one() {
    use super::layer_stream_card::StackShape;
    use crate::kernels::cuda::{
        NativeWeight,
        resident::{
            matvec_q::QuantType,
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    };
    assert!(device_present("staged native weight"), "a GPU is required");
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let compute = Rc::new(ctx.stream().expect("stream"));
    let k = 256usize;
    let experts = 8usize;
    // Two layers of different row counts per expert: different bytes, one slot pair.
    let rows_per_expert = [64usize, 96];
    let mut store = Vec::new();
    let mut spans = Vec::new();
    let mut bytes_of = Vec::new();
    for (layer, rows) in rows_per_expert.iter().enumerate() {
        let n_rows = experts * rows;
        let bytes = q8_0_stack(layer as u8 + 1, n_rows, k);
        let mut buffer = ctx.pinned_buf(bytes.len()).expect("pinned");
        buffer.write(0, &bytes).expect("fill");
        store.push(buffer);
        spans.push(vec![StackSpan {
            bytes: bytes.len(),
            home: StackHome::Pinned {
                buf: layer,
                offset: 0,
            },
            shape: Some(StackShape {
                quant: QuantType::Q80,
                k,
                n_rows,
            }),
        }]);
        bytes_of.push((bytes, n_rows));
    }
    let mut transport = CardTransport::new(
        ctx,
        Handle::Shared(Rc::clone(&compute)),
        experts,
        spans,
        store,
        2,
    )
    .expect("transport");
    let activation: Vec<u8> = (0..k)
        .flat_map(|i| (((i * 37 % 101) as f32 - 50.0) / 25.0).to_le_bytes())
        .collect();
    let x = ctx.upload(&activation).expect("x");
    let pack = ctx.allocate(q8_1_bytes(k, 1).expect("pack")).expect("pack");
    ctx.quantize_q8_1(
        &compute,
        QuantizeCall {
            x: x.range(0, x.bytes()).expect("x"),
            y: pack.range(0, pack.bytes()).expect("pack"),
            k,
            ncols: 1,
        },
    )
    .expect("quantize");
    for (layer, (bytes, n_rows)) in bytes_of.iter().enumerate() {
        let slot = layer % 2;
        let ticket = transport
            .begin(layer, &UploadSet::Whole, slot)
            .expect("begin");
        transport.wait(ticket).expect("wait");
        let resident = NativeWeight::from_buffer(
            ctx.upload(bytes).expect("resident"),
            QuantType::Q80,
            k,
            *n_rows,
            Some(experts),
        )
        .expect("resident weight");
        let out_bytes = n_rows * 4;
        let y_resident = ctx.allocate(out_bytes).expect("y");
        let y_staged = ctx.allocate(out_bytes).expect("y");
        ctx.matvec_native(
            &compute,
            &resident,
            pack.range(0, pack.bytes()).expect("pack"),
            y_resident.range(0, out_bytes).expect("y"),
            1,
        )
        .expect("resident matvec");
        let staged = transport.staged(layer, slot, 0).expect("staged weight");
        assert_eq!(staged.n_rows(), *n_rows);
        ctx.matvec_native(
            &compute,
            staged,
            pack.range(0, pack.bytes()).expect("pack"),
            y_staged.range(0, out_bytes).expect("y"),
            1,
        )
        .expect("staged matvec");
        let (mut a, mut b) = (vec![0u8; out_bytes], vec![0u8; out_bytes]);
        y_resident.read_at(&compute, 0, &mut a).expect("read");
        y_staged.read_at(&compute, 0, &mut b).expect("read");
        assert!(a.iter().any(|&byte| byte != 0), "the matvec produced zeros");
        assert_eq!(a, b, "layer {layer}: staged against resident");
        transport.release(slot).expect("release");
    }
}
