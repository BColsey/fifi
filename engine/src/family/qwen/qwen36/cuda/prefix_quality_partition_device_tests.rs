//! The depth-table partition against the piece-model partition of the same prompt, on real text.
//!
//! One Q4_K_M session prefills the first `TOKENS` tokens of a real document twice at cap 2048:
//! once in the pieces the deleted depth table planned (2,048 rows below position 3,072, then
//! 1,024, then 512 from 65,536), once in the pieces the shared loop plans now. Both runs score
//! every `STRIDE`-th next-token position from `FIRST` on, teacher-forced, with the final norm and
//! the head applied to each scored hidden row. The two runs differ only where the partitions
//! differ, so the scored region starts at the first position the partitions do not share. The
//! report gives the signed NLL change (candidate minus reference) with its interval, the top-1
//! agreement of the two runs, each run's top-1 accuracy against the real next token, and the mean
//! candidate-to-reference KL.
use super::*;

const TOKENS: usize = 12_288;
const ARENA: usize = 2048;
/// The first scored position: the table and the model share every piece before it.
const FIRST: usize = 4096;
const STRIDE: usize = 4;
const BLOCKS: usize = 8;

fn fail(message: &str) -> EngError {
    EngError::BadRequest(format!("partition quality: {message}"))
}

/// The rows of the piece starting at `at` under the depth table the shared loop replaced.
fn table_rows(at: usize) -> usize {
    if at < 3 * 1024 {
        2048
    } else if at < 64 * 1024 {
        1024
    } else {
        512
    }
}

fn table_pieces(tokens: usize) -> Vec<(usize, usize)> {
    let mut at = 0;
    let mut pieces = Vec::new();
    while at < tokens {
        let rows = table_rows(at).min(ARENA).min(tokens - at);
        pieces.push((at, rows));
        at += rows;
    }
    pieces
}

struct Scored {
    row: Row,
    correct: bool,
}

struct Summary {
    positions: usize,
    signed_dnll: f64,
    dnll_interval_upper_naive: f64,
    dnll_interval_upper_blocks: f64,
    same_top: f64,
    reference_accuracy: f64,
    candidate_accuracy: f64,
    mean_kl: f64,
}

fn summarize(
    reference: &[Scored],
    candidate: &[Scored],
    width: usize,
) -> Result<Summary, EngError> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(fail("the two runs scored different positions"));
    }
    let n = reference.len();
    let mut diffs = Vec::with_capacity(n);
    let mut kl = 0.0;
    let mut same = 0usize;
    for (r, c) in reference.iter().zip(candidate) {
        let pair = compare(&r.row, &c.row, width)?;
        diffs.push(pair.candidate_nll - pair.reference_nll);
        kl += pair.kl;
        same += usize::from(pair.same_top);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let variance =
        diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let naive = mean + 1.96 * (variance / n as f64).sqrt();
    // Positions of one stretch of text are correlated, so the interval that counts is over
    // contiguous blocks: the mean of each block is one sample.
    let size = n / BLOCKS;
    let block_means: Vec<f64> = (0..BLOCKS)
        .map(|b| diffs[b * size..(b + 1) * size].iter().sum::<f64>() / size as f64)
        .collect();
    let block_mean = block_means.iter().sum::<f64>() / BLOCKS as f64;
    let block_var = block_means
        .iter()
        .map(|m| (m - block_mean).powi(2))
        .sum::<f64>()
        / (BLOCKS as f64 - 1.0);
    // Student t, 7 degrees of freedom, one-sided 97.5 percent.
    let blocks = block_mean + 2.365 * (block_var / BLOCKS as f64).sqrt();
    Ok(Summary {
        positions: n,
        signed_dnll: mean,
        dnll_interval_upper_naive: naive,
        dnll_interval_upper_blocks: blocks,
        same_top: same as f64 / n as f64,
        reference_accuracy: reference.iter().filter(|s| s.correct).count() as f64 / n as f64,
        candidate_accuracy: candidate.iter().filter(|s| s.correct).count() as f64 / n as f64,
        mean_kl: kl / n as f64,
    })
}

fn scored_positions(p0: usize, rows: usize) -> impl Iterator<Item = usize> {
    (p0..p0 + rows).filter(|&p| p >= FIRST && (p - FIRST).is_multiple_of(STRIDE) && p + 1 < TOKENS)
}

/// Runs one piece and scores the strided positions it holds, hidden row by hidden row.
fn run_piece(
    step: &mut QwenResidentStep<'_>,
    sequence: &[u32],
    (p0, rows): (usize, usize),
) -> Result<Vec<Scored>, EngError> {
    let input = sequence
        .get(p0..p0 + rows)
        .ok_or_else(|| fail("incomplete input"))?;
    step.run_chunk(p0, input)?;
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| fail("no admitted arena"))?;
    let views = ChunkViews::new(arena.list(), arena.arena())?;
    let hidden = views.get(ChunkSlot::Hidden)?;
    let scratch = step.view(final_resid(step.res.table.len()))?;
    let row_bytes = step
        .res
        .geometry
        .hidden
        .checked_mul(F32)
        .ok_or_else(|| fail("hidden width overflows"))?;
    if scratch.bytes() != row_bytes {
        return Err(fail("tail scratch does not hold one hidden row"));
    }
    let mut out = Vec::new();
    for position in scored_positions(p0, rows) {
        let offset = hidden
            .offset()
            .checked_add((position - p0) * row_bytes)
            .ok_or_else(|| fail("hidden offset overflows"))?;
        let bytes = read_range(step, hidden.buffer().range(offset, row_bytes)?)?;
        scratch
            .buffer()
            .write_at(&step.stream, scratch.offset(), &bytes)?;
        step.final_norm()?;
        step.lm_head()?;
        let bytes = read_range(step, step.view(Slot::Logits)?)?;
        if bytes.len() != HEAD * F32 {
            return Err(fail("logits do not cover the actual model head"));
        }
        let teacher = sequence[position + 1] as usize;
        let row = row(f32_from_le(&bytes), VOCAB, teacher)?;
        let correct = row.top == teacher;
        out.push(Scored { row, correct });
    }
    Ok(out)
}

type Pieces = Vec<(usize, usize)>;

fn device_phase(ids: &[u32]) -> Result<(Vec<Scored>, Vec<Scored>, Pieces), EngError> {
    let dims = dims();
    let inv = inventory();
    let opts = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let rules = QwenRules::new(vec![248_046, 248_044], VOCAB, HEAD)?;
    let dir = crate::test_paths::required("Q36_CUDA_MODEL_DIR");
    let model = open_qwen3_6_35b_a3b(
        &std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )?;
    let projector = CudaProjector::new()?;
    let weights = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve::default(),
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )?
        .device;
    let budget = projector
        .memory_info()?
        .free
        .checked_sub(1usize << 30)
        .ok_or_else(|| fail("no 1 GiB headroom after loading weights"))?;
    let ctx = projector.context();
    let res = StepResources::new(ctx, dims, &inv, &opts)?;
    let mut step = QwenResidentStep::new(ctx, &res, &weights, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.set_m_sess_cap(ARENA as u32);
    step.admit_prefill(
        PrefillBudget {
            device_free: budget,
            extras: Vec::new(),
        },
        false,
        false,
        false,
    )?;
    if step.chunk_ready() != (ChunkVerdict::Ready { m_sess: ARENA }) {
        return Err(fail("the 2,048-row FP16 arena was not admitted"));
    }
    let sequence = ids.get(..TOKENS).ok_or_else(|| fail("too few tokens"))?;
    let sm = step.prefill_sm_count();
    let model_pieces: Vec<(usize, usize)> =
        crate::family::qwen::qwen36::cuda::prefill_driver::pieces_of(0, TOKENS, ARENA, sm)?
            .into_iter()
            .map(|piece| match piece {
                crate::family::prefill::arena::PrefillStep::Chunk { p0, rows } => (p0, rows),
                crate::family::prefill::arena::PrefillStep::Feed { p0 } => (p0, 1),
            })
            .collect();
    let old_pieces = table_pieces(TOKENS);
    let mut reference = Vec::new();
    step.reset_state()?;
    for piece in &old_pieces {
        reference.extend(run_piece(&mut step, sequence, *piece)?);
    }
    let mut candidate = Vec::new();
    step.reset_state()?;
    for piece in &model_pieces {
        candidate.extend(run_piece(&mut step, sequence, *piece)?);
    }
    step.stream
        .synchronize()
        .inspect_err(|_| step.state.poison())?;
    Ok((reference, candidate, model_pieces))
}

#[test]
fn the_table_planner_ports_the_old_rows() {
    let pieces = table_pieces(6000);
    assert_eq!(
        pieces,
        vec![(0, 2048), (2048, 2048), (4096, 1024), (5120, 880)]
    );
    assert_eq!(table_rows(65_536), 512);
    assert_eq!(scored_positions(4096, 2048).next(), Some(4096));
    assert_eq!(scored_positions(0, 2048).count(), 0);
}

#[test]
fn the_summary_signs_the_change_candidate_minus_reference() {
    let make = |nll: f64, correct: bool| Scored {
        row: row(vec![nll as f32, 0.0], 2, 1).unwrap(),
        correct,
    };
    let reference: Vec<Scored> = (0..80).map(|_| make(1.0, true)).collect();
    let candidate: Vec<Scored> = (0..80).map(|_| make(1.0, true)).collect();
    let summary = summarize(&reference, &candidate, 2).unwrap();
    assert!(summary.signed_dnll.abs() < 1e-12);
    assert_eq!(summary.same_top, 1.0);
    assert_eq!(summary.positions, 80);
    assert!(summarize(&reference, &candidate[..79], 2).is_err());
}

#[test]
#[ignore = "full Q4_K_M model on the 3090 under a lease; Q36_PARTITION_TEXT and Q36_PARTITION_TOKENIZER name the text and tokenizer.json"]
fn the_piece_model_partition_matches_the_depth_table_on_real_text() {
    let text = std::fs::read_to_string(crate::test_paths::required("Q36_PARTITION_TEXT"))
        .expect("the document text");
    let tokenizer =
        tokenizers::Tokenizer::from_file(crate::test_paths::required("Q36_PARTITION_TOKENIZER"))
            .expect("tokenizer.json");
    let encoding = tokenizer.encode(text, false).expect("encode the document");
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    assert!(ids.len() >= TOKENS, "the document has {} tokens", ids.len());
    assert!(ids.iter().take(TOKENS).all(|&id| (id as usize) < VOCAB));
    let (reference, candidate, pieces) = device_phase(&ids).expect("two partitions of one prompt");
    let summary = summarize(&reference, &candidate, VOCAB).expect("summary");
    println!(
        "PARTITION_PIECES model={} first={:?}",
        pieces.len(),
        &pieces[..pieces.len().min(6)]
    );
    println!(
        "PARTITION_GATE positions={} signed_dnll={:.6} upper_naive={:.6} upper_blocks={:.6} \
         top1_same={:.6} accuracy_reference={:.6} accuracy_candidate={:.6} accuracy_loss_pp={:.4} mean_kl={:.8}",
        summary.positions,
        summary.signed_dnll,
        summary.dnll_interval_upper_naive,
        summary.dnll_interval_upper_blocks,
        summary.same_top,
        summary.reference_accuracy,
        summary.candidate_accuracy,
        100.0 * (summary.reference_accuracy - summary.candidate_accuracy),
        summary.mean_kl
    );
    assert!(
        summary.dnll_interval_upper_blocks <= 0.02,
        "delta NLL upper bound"
    );
    assert!(
        100.0 * (summary.reference_accuracy - summary.candidate_accuracy) <= 0.5,
        "top-1 accuracy loss"
    );
}
