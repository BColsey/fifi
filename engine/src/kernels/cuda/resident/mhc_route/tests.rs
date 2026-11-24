//! Device tests of the mHC merge wrapper.
//!
//! The merge kernel writes `a` and `b` in place, one thread per element after one read of
//! each, so no element may belong to both streams. One row is two ranges; a chunk's rows are
//! the `[a || b]` halves of `ld_ab`-wide rows, with or without padding. Every admitted layout
//! is checked against the f64 reference of the kernel's two lines, and every layout where a
//! row of `a` shares an element with a row of `b` must be a `BadRequest` before any launch,
//! after which the backend still runs a good call. The suite runs with `--test-threads=1`.

use super::MhcMergeCall;
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

const H: usize = 256;
const F32: usize = 4;
/// `max|d| <= 2^-16 * max(max|ref|, 1)` over the written elements.
const TOL: f64 = 1.0 / 65_536.0;

/// One layout of the two streams: `rows` rows `ld` floats apart, each stream the `half`
/// floats from its `(buffer, offset)`; `cells` are the two storage buffers (buffers 2 and 3
/// hold the branch rows and the route cells).
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    rows: usize,
    ld: usize,
    a: (usize, usize),
    b: (usize, usize),
    cells: [usize; 2],
}

impl Layout {
    /// The floats one stream covers: its first row in full, the rest by stride.
    fn half(&self) -> usize {
        (self.rows - 1) * self.ld + H
    }

    fn span(&self, s: (usize, usize)) -> (usize, usize, usize) {
        (s.0, s.1, self.half())
    }
}

fn admitted() -> Vec<Layout> {
    let rows = 3;
    let (two, three) = (2 * H, 3 * H);
    vec![
        Layout {
            name: "interleaved [a || b]",
            rows,
            ld: two,
            a: (0, 0),
            b: (0, H),
            cells: [rows * two, 16],
        },
        Layout {
            name: "interleaved, b first",
            rows,
            ld: two,
            a: (0, H),
            b: (0, 0),
            cells: [rows * two, 16],
        },
        Layout {
            name: "a, a gap, b",
            rows,
            ld: three,
            a: (0, 0),
            b: (0, H + 64),
            cells: [rows * three, 16],
        },
        Layout {
            name: "two buffers",
            rows,
            ld: two,
            a: (0, 0),
            b: (1, 0),
            cells: [rows * two, rows * two],
        },
        Layout {
            name: "two blocks of one buffer",
            rows,
            ld: two,
            a: (0, 0),
            b: (0, rows * two),
            cells: [2 * rows * two, 16],
        },
        Layout {
            name: "one row, two ranges",
            rows: 1,
            ld: two,
            a: (0, 0),
            b: (0, H),
            cells: [two, 16],
        },
    ]
}

fn sharing() -> Vec<Layout> {
    let rows = 3;
    let two = 2 * H;
    vec![
        Layout {
            name: "the same range",
            rows,
            ld: two,
            a: (0, 0),
            b: (0, 0),
            cells: [rows * two, 16],
        },
        Layout {
            name: "shifted by four floats",
            rows,
            ld: two + 1,
            a: (0, 0),
            b: (0, 4),
            cells: [rows * (two + 1), 16],
        },
        Layout {
            name: "reaching the next row",
            rows,
            ld: two,
            a: (0, 0),
            b: (0, two - 64),
            cells: [(rows - 1) * two + H + two - 64, 16],
        },
        Layout {
            name: "one row, the same range",
            rows: 1,
            ld: two,
            a: (0, 0),
            b: (0, 0),
            cells: [two, 16],
        },
    ]
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("the device context");
        let stream = ctx.stream().expect("the stream");
        Self { ctx, stream }
    }
}

fn range(bufs: &[DeviceBuffer], s: (usize, usize, usize)) -> Result<DeviceRange<'_>, EngError> {
    bufs[s.0].range(s.1 * F32, s.2 * F32)
}

/// Queues the merge of `l` on the zero-filled buffers `bufs` (storages, branch, route).
fn merge(rig: &Rig, bufs: &[DeviceBuffer], l: &Layout) -> Result<(), EngError> {
    rig.ctx.mhc_merge(
        &rig.stream,
        MhcMergeCall {
            hidden: H,
            rows: l.rows,
            ld_ab: l.ld,
            a: range(bufs, l.span(l.a))?,
            b: range(bufs, l.span(l.b))?,
            branch: range(bufs, (2, 0, l.rows * H))?,
            route: range(bufs, (3, 0, l.rows * 6))?,
        },
    )
}

/// Merges random streams in `l` and returns the worst error ratio over the written elements;
/// every element outside both streams must come back bit-identical.
fn check(rig: &Rig, rng: &mut Rng, l: &Layout) -> f64 {
    let mut draw = |n: usize| (0..n).map(|_| rng.normal() as f32).collect::<Vec<f32>>();
    let storage = [draw(l.cells[0]), draw(l.cells[1])];
    let branch = draw(l.rows * H);
    let route: Vec<f32> = (0..l.rows * 6)
        .map(|i| {
            let x = rng.normal() as f32;
            if i % 6 < 2 {
                2.0 / (1.0 + (-x).exp())
            } else {
                (0.5 * x).exp()
            }
        })
        .collect();
    let bufs: Vec<DeviceBuffer> = storage
        .iter()
        .chain([&branch, &route])
        .map(|values| testkit::upload_f32(&rig.ctx, values))
        .collect();
    merge(rig, &bufs, l)
        .unwrap_or_else(|e| panic!("{}: an admitted layout was refused: {e:?}", l.name));
    rig.stream.synchronize().expect("drain");

    let mut want = storage.clone();
    let mut touched = [vec![false; l.cells[0]], vec![false; l.cells[1]]];
    for r in 0..l.rows {
        let cell = &route[r * 6..r * 6 + 6];
        for j in 0..H {
            let ai = l.a.1 + r * l.ld + j;
            let bi = l.b.1 + r * l.ld + j;
            let (a, b, y) = (
                f64::from(storage[l.a.0][ai]),
                f64::from(storage[l.b.0][bi]),
                f64::from(branch[r * H + j]),
            );
            want[l.a.0][ai] =
                (a * f64::from(cell[2]) + b * f64::from(cell[3]) + y * f64::from(cell[0])) as f32;
            want[l.b.0][bi] =
                (a * f64::from(cell[4]) + b * f64::from(cell[5]) + y * f64::from(cell[1])) as f32;
            touched[l.a.0][ai] = true;
            touched[l.b.0][bi] = true;
        }
    }
    let mut worst = 0.0f64;
    for s in 0..2 {
        let bytes = testkit::read_bytes(&rig.ctx, &bufs[s], l.cells[s] * F32);
        let got = testkit::floats(&bytes);
        for i in 0..l.cells[s] {
            if !touched[s][i] {
                assert_eq!(
                    got[i].to_bits(),
                    storage[s][i].to_bits(),
                    "{}: element {s}[{i}] outside both streams changed",
                    l.name
                );
            } else {
                let diff = f64::from(got[i] - want[s][i]).abs();
                worst = worst.max(diff / f64::from(want[s][i].abs().max(1.0)));
            }
        }
    }
    worst
}

fn zero_buffers(rig: &Rig, l: &Layout) -> Vec<DeviceBuffer> {
    [l.cells[0], l.cells[1], l.rows * H, l.rows * 6]
        .iter()
        .map(|&n| testkit::upload_filled(&rig.ctx, 0, n * F32))
        .collect()
}

#[test]
fn mhc_merge_matches_the_reference_for_every_admitted_layout() {
    if !testkit::device_present("mhc_merge_matches_the_reference_for_every_admitted_layout") {
        return;
    }
    let (rig, mut rng) = (Rig::new(), Rng::new(0x7e5c_0001));
    for l in admitted() {
        let ratio = check(&rig, &mut rng, &l);
        println!("mhc_merge {:<30} worst {:.3e}", l.name, ratio);
        assert!(
            ratio.is_finite() && ratio <= TOL,
            "{}: worst {ratio:.3e} > {TOL:.1e}",
            l.name
        );
    }
}

#[test]
fn mhc_merge_refuses_streams_that_share_an_element() {
    if !testkit::device_present("mhc_merge_refuses_streams_that_share_an_element") {
        return;
    }
    let (rig, mut rng) = (Rig::new(), Rng::new(0x7e5c_0002));
    for l in sharing() {
        let bufs = zero_buffers(&rig, &l);
        let result = merge(&rig, &bufs, &l);
        assert!(
            matches!(&result, Err(EngError::BadRequest(m)) if m.contains("a and b")),
            "{}: {result:?}",
            l.name
        );
        drop(bufs);
        // The refusal latched nothing: the backend still runs a good call.
        let good = admitted()[0];
        let ratio = check(&rig, &mut rng, &good);
        assert!(
            ratio.is_finite() && ratio <= TOL,
            "after {}: worst {ratio:.3e}",
            l.name
        );
    }
}
