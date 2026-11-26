//! Independent per-row host work split over scoped threads.

/// The most threads one call uses, and the fewest rows worth a thread of their own.
#[derive(Clone, Copy)]
pub(crate) struct RowSplit {
    pub(crate) max_threads: usize,
    pub(crate) min_rows: usize,
}

impl RowSplit {
    /// Threads for `rows` rows on this host: at most `max_threads` and the host's parallelism,
    /// each with at least `min_rows` rows; one when fewer than two would remain.
    pub(crate) fn threads(self, rows: usize) -> usize {
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(self.max_threads)
            .min(rows / self.min_rows.max(1));
        if threads < 2 { 1 } else { threads }
    }
}

/// `f` applied to every item in order, on scoped threads over consecutive chunks. Results keep
/// the items' order, and a panic in `f` resumes on the caller.
pub(crate) fn map_rows<T: Send, R: Send>(
    items: &mut [T],
    split: RowSplit,
    f: impl Fn(&mut T) -> R + Sync,
) -> Vec<R> {
    let threads = split.threads(items.len());
    if threads == 1 {
        return items.iter_mut().map(f).collect();
    }
    let chunk = items.len().div_ceil(threads);
    let f = &f;
    std::thread::scope(|scope| {
        let parts: Vec<_> = items
            .chunks_mut(chunk)
            .map(|part| scope.spawn(move || part.iter_mut().map(f).collect::<Vec<_>>()))
            .collect();
        parts
            .into_iter()
            .flat_map(|part| {
                part.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_item_order_across_threads_and_mutations_land() {
        let split = RowSplit {
            max_threads: 4,
            min_rows: 3,
        };
        for len in [0, 1, 5, 6, 13, 64] {
            let mut items: Vec<usize> = (0..len).collect();
            let doubled = map_rows(&mut items, split, |x| {
                *x += 100;
                *x * 2
            });
            assert_eq!(doubled, (0..len).map(|x| (x + 100) * 2).collect::<Vec<_>>());
            assert_eq!(items, (100..100 + len).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_split_never_gives_a_thread_fewer_rows_than_its_minimum() {
        let split = RowSplit {
            max_threads: 64,
            min_rows: 16,
        };
        assert_eq!(split.threads(0), 1);
        assert_eq!(split.threads(31), 1);
        let host = std::thread::available_parallelism().map_or(1, usize::from);
        let expect = |n: usize| if n < 2 { 1 } else { n };
        assert_eq!(split.threads(32), expect(2.min(host)));
        assert_eq!(split.threads(512), expect(32.min(host)));
    }
}
