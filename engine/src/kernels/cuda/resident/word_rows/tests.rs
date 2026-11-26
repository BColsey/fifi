use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream, testkit::device_present,
};

use super::MAX_WORD_ROWS;

fn fill(buffer: &DeviceBuffer, stream: &DeviceStream, words: &[u32]) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.write_at(stream, 0, &bytes).unwrap();
}

fn read(buffer: &DeviceBuffer, stream: &DeviceStream, words: usize) -> Vec<u32> {
    let mut bytes = vec![0u8; words * 4];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// The ranges of `buffer` that start at `starts` words and hold `lengths` words.
fn rows_at<'b>(
    buffer: &'b DeviceBuffer,
    starts: &[usize],
    lengths: &[usize],
) -> Vec<DeviceRange<'b>> {
    starts
        .iter()
        .zip(lengths)
        .map(|(&start, &n)| buffer.range(start * 4, n * 4).unwrap())
        .collect()
}

fn refusal(result: Result<(), crate::EngError>) -> String {
    format!("{:?}", result.unwrap_err())
}

/// Rows of 1, 70, 3 and 256 words with one-word gaps between them: a zero launch clears the rows
/// and leaves the gaps, a copy launch moves the source rows' bits (NaN payloads included) into
/// the rows and leaves the gaps.
#[test]
fn word_rows_zero_and_copy_only_their_rows() {
    if !device_present("word_rows_zero_and_copy_only_their_rows") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let lengths = [1usize, 70, 3, 256];
    let total = lengths.iter().map(|n| n + 1).sum::<usize>();
    let source = ctx.allocate(total * 4).unwrap();
    let target = ctx.allocate(total * 4).unwrap();
    let bits = |i: usize| 0x7fc0_0001_u32.wrapping_add(i as u32 * 2_654_435_761);
    fill(&source, &stream, &(0..total).map(bits).collect::<Vec<_>>());
    fill(&target, &stream, &vec![0xdead_beef; total]);
    let mut starts = Vec::new();
    let mut at = 0;
    for &n in &lengths {
        starts.push(at);
        at += n + 1;
    }
    let rows = |buffer| rows_at(buffer, &starts, &lengths);
    let in_row = |i: usize| {
        starts
            .iter()
            .zip(lengths)
            .any(|(&start, n)| (start..start + n).contains(&i))
    };
    ctx.zero_word_rows(&stream, &rows(&target)).unwrap();
    for (i, word) in read(&target, &stream, total).into_iter().enumerate() {
        assert_eq!(word, if in_row(i) { 0 } else { 0xdead_beef }, "word {i}");
    }
    let pairs: Vec<_> = rows(&source).into_iter().zip(rows(&target)).collect();
    ctx.copy_word_rows(&stream, &pairs).unwrap();
    for (i, word) in read(&target, &stream, total).into_iter().enumerate() {
        assert_eq!(
            word,
            if in_row(i) { bits(i) } else { 0xdead_beef },
            "word {i}"
        );
    }
}

/// A part word, a pair of different sizes, a destination that shares bytes with its source, with
/// another pair's source or with another destination, two zero rows that share bytes, an empty
/// list and one row too many are refused, and a refused call changes nothing.
#[test]
fn word_rows_refuse_bad_lists_before_any_launch() {
    if !device_present("word_rows_refuse_bad_lists_before_any_launch") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let source = ctx.allocate(64).unwrap();
    let target = ctx.allocate(64).unwrap();
    fill(&source, &stream, &[7; 16]);
    fill(&target, &stream, &[9; 16]);
    let s = |at: usize, bytes: usize| source.range(at, bytes).unwrap();
    let t = |at: usize, bytes: usize| target.range(at, bytes).unwrap();
    let error = refusal(ctx.zero_word_rows(&stream, &[t(0, 6)]));
    assert!(error.contains("whole number of words"), "{error}");
    let error = refusal(ctx.zero_word_rows(&stream, &[t(2, 8)]));
    assert!(error.contains("not a multiple of 4"), "{error}");
    let error = refusal(ctx.copy_word_rows(&stream, &[(s(0, 8), t(0, 12))]));
    assert!(error.contains("moves 8 bytes into 12"), "{error}");
    let error = refusal(ctx.copy_word_rows(&stream, &[(s(0, 8), s(4, 8))]));
    assert!(error.contains("overlap"), "{error}");
    let error = refusal(ctx.copy_word_rows(&stream, &[(t(0, 8), t(16, 8)), (s(0, 8), t(4, 8))]));
    assert!(error.contains("overlap"), "{error}");
    // The second pair reads words the first pair writes.
    let error = refusal(ctx.copy_word_rows(&stream, &[(s(0, 8), t(0, 8)), (t(4, 8), s(32, 8))]));
    assert!(error.contains("overlap"), "{error}");
    let error = refusal(ctx.zero_word_rows(&stream, &[t(0, 8), t(4, 8)]));
    assert!(error.contains("overlap"), "{error}");
    let error = refusal(ctx.zero_word_rows(&stream, &[]));
    assert!(error.contains("1 to 64 rows, got 0"), "{error}");
    let many: Vec<_> = (0..=MAX_WORD_ROWS).map(|_| t(0, 4)).collect();
    let error = refusal(ctx.zero_word_rows(&stream, &many));
    assert!(error.contains("got 65"), "{error}");
    assert_eq!(read(&source, &stream, 16), vec![7; 16]);
    assert_eq!(read(&target, &stream, 16), vec![9; 16]);
}
