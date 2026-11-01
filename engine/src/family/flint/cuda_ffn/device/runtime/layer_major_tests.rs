use super::*;
use crate::kernels::cuda::plan_arena;

#[derive(Default)]
struct Recording {
    log: Vec<(usize, &'static str, Range<usize>)>,
    fail_at: Option<(usize, &'static str)>,
}

impl Recording {
    fn hook(&mut self, layer: usize, name: &'static str, rows: Range<usize>) -> Result<()> {
        if self.fail_at == Some((layer, name)) {
            return Err(EngError::ModelOutput(format!("{name} {layer} failed")));
        }
        self.log.push((layer, name, rows));
        Ok(())
    }
}

impl LayerPass for Recording {
    fn execute_layer_attention(&mut self, layer: usize, rows: Range<usize>) -> Result<()> {
        self.hook(layer, "attention", rows)
    }

    fn execute_layer_routed_ffn(&mut self, layer: usize, rows: Range<usize>) -> Result<()> {
        self.hook(layer, "ffn", rows)
    }
}

#[test]
fn every_row_finishes_a_layer_before_any_row_starts_the_next() {
    for rows in [1, 7, 8, 9, 63, 64, 65, 4096] {
        let mut pass = Recording::default();
        drive(&mut pass, LAYERS, 0..rows).unwrap();
        assert_eq!(pass.log.len(), 2 * LAYERS);
        for (index, (layer, name, span)) in pass.log.iter().enumerate() {
            assert_eq!(*layer, index / 2);
            assert_eq!(*name, if index % 2 == 0 { "attention" } else { "ffn" });
            assert_eq!(*span, 0..rows, "each hook sees the whole piece");
        }
    }
}

#[test]
fn a_failing_hook_ends_the_walk_without_starting_the_next_layer() {
    for (layer, name) in [(0, "attention"), (5, "ffn"), (48, "attention")] {
        let mut pass = Recording {
            fail_at: Some((layer, name)),
            ..Recording::default()
        };
        let error = drive(&mut pass, LAYERS, 0..64).unwrap_err();
        assert_eq!(
            error,
            EngError::ModelOutput(format!("{name} {layer} failed"))
        );
        let last = pass.log.last().map(|(layer, name, _)| (*layer, *name));
        match (name, layer) {
            ("attention", 0) => assert_eq!(last, None),
            ("attention", _) => assert_eq!(last, Some((layer - 1, "ffn"))),
            _ => assert_eq!(last, Some((layer, "attention"))),
        }
    }
}

#[test]
fn blocks_cover_every_tail_height_once_with_only_the_last_one_short() {
    for rows in [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 4095, 4096] {
        for height in [1, 4, 8] {
            let parts: Vec<_> = blocks(0..rows, height).collect();
            assert_eq!(parts.iter().map(Range::len).sum::<usize>(), rows);
            let mut next = 0;
            for (index, part) in parts.iter().enumerate() {
                assert_eq!(part.start, next);
                assert!(part.len() <= height);
                assert!(
                    part.len() == height || index + 1 == parts.len(),
                    "rows {rows} height {height}: only the last block may be short"
                );
                next = part.end;
            }
        }
    }
    assert_eq!(blocks(5..5, 8).count(), 0);
    assert_eq!(
        blocks(0..3, 0).count(),
        3,
        "a zero height degrades to one row"
    );
}

#[test]
fn the_extra_charge_covers_everything_arming_allocates_beyond_the_pair_faces() {
    for rows in [1, 9, 64, 65, 1000, 4096] {
        let faces = State::layer_major_staging_bytes(rows).unwrap();
        let extra = State::layer_major_extra_bytes(rows, false).unwrap();
        let planes = plan_arena(&plane_tensors(rows).unwrap()).unwrap().total as u64;
        let arena = exl3::PairFaces::bytes(
            rows * TOP_K,
            WIDTH,
            super::super::ROUTED_INTERMEDIATE,
            false,
        )
        .unwrap();
        assert!(
            faces + extra >= planes + arena,
            "rows {rows}: charged {} for {} allocated",
            faces + extra,
            planes + arena
        );
        // The charge is tight: the faces, three planes, the router-input copy
        // and the tile scratch, nothing speculative.
        let one_plane = (rows * WIDTH * 4) as u64;
        assert!(
            faces + extra <= planes + arena + one_plane,
            "rows {rows}: charge exceeds allocation by more than the transient plane"
        );
    }
}

#[test]
fn the_piece_cap_is_the_router_call_limit() {
    assert_eq!(LAYER_MAJOR_PIECE_ROWS, 4096);
    let rows = LAYER_MAJOR_PIECE_ROWS;
    // A full piece's pairs still fit the shared row geometry.
    assert!(rows * TOP_K * WIDTH <= 1 << 32);
}

/// Records the block ends the attention hook would commit, in call order.
#[derive(Default)]
struct CommitOrder(Vec<usize>);

impl LayerPass for CommitOrder {
    fn execute_layer_attention(&mut self, _layer: usize, rows: Range<usize>) -> Result<()> {
        self.0.extend(blocks(rows, 8).map(|block| block.end));
        Ok(())
    }

    fn execute_layer_routed_ffn(&mut self, _layer: usize, _rows: Range<usize>) -> Result<()> {
        Ok(())
    }
}

#[test]
fn attention_commits_block_by_block_inside_each_layer() {
    // The order the executor-level device gates assert on the real pass.
    let mut pass = CommitOrder::default();
    drive(&mut pass, LAYERS, 0..9).unwrap();
    assert_eq!(pass.0, [8, 9].repeat(LAYERS));
    let mut pass = CommitOrder::default();
    drive(&mut pass, 2, 0..65).unwrap();
    assert_eq!(pass.0, [8, 16, 24, 32, 40, 48, 56, 64, 65].repeat(2));
}

#[test]
fn the_jobs_charge_covers_the_batch_scratch_and_a_tall_attention_half() {
    for rows in [9, 64, 1024] {
        let tile = exl3::PairFaces::bytes(
            rows * TOP_K,
            WIDTH,
            super::super::ROUTED_INTERMEDIATE,
            false,
        )
        .unwrap();
        let batch =
            exl3::PairFaces::bytes(rows * TOP_K, WIDTH, super::super::ROUTED_INTERMEDIATE, true)
                .unwrap();
        // The batch scratch is bigger than the tile scratch, and the extra charge says so.
        assert!(batch > tile, "rows {rows}");
        let charged = State::layer_major_extra_bytes(rows, true).unwrap()
            + State::layer_major_staging_bytes(rows).unwrap();
        let planes = plan_arena(&plane_tensors(rows).unwrap()).unwrap().total as u64;
        assert!(charged >= planes + batch, "rows {rows}");
        // The shared-expert scratch of a piece is charged here; the late-health retention of a
        // tall attention half is charged once, by the work reserve of the KV plan.
        let scratch = (rows * 3 * 4096 * 4) as u64;
        assert_eq!(
            State::layer_jobs_extra_bytes(rows).unwrap(),
            scratch,
            "rows {rows}"
        );
    }
}
