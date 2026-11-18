//! Bounded, opt-in timing of the initial P2P chunk's first and last routed layers.
//!
//! Timing events measure intervals in a single stream, including any GPU scheduling
//! and host enqueue gaps. They do not measure PCIe bytes, CPU work or individual experts.
//! The card driver must keep these events alive and only query them after all card
//! streams have completed their successful chunk-close checks.

use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceStream, NativeWeight, TimingEvent},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Pending,
    Started,
    Finished,
}

fn next_phase(current: Phase, start: bool) -> Result<Phase, EngError> {
    match (current, start) {
        (Phase::Pending, true) => Ok(Phase::Started),
        (Phase::Started, false) => Ok(Phase::Finished),
        _ => Err(EngError::BadRequest(
            "P2P expert timing: incomplete or repeated mark".into(),
        )),
    }
}

fn complete_batch(layers: &[Vec<[Phase; 3]>]) -> bool {
    let Some(first) = layers.first() else {
        return false;
    };
    !first.is_empty()
        && layers.iter().all(|cards| {
            cards.len() == first.len()
                && cards
                    .iter()
                    .all(|phases| phases.iter().all(|phase| *phase == Phase::Finished))
        })
}

fn format_record(
    card: usize,
    layer: usize,
    rows: usize,
    role: &str,
    metadata: (&str, &str),
    ms: f32,
) -> Result<String, EngError> {
    if !ms.is_finite() || ms < 0.0 {
        return Err(EngError::BadRequest(
            "P2P expert timing: invalid duration".into(),
        ));
    }
    Ok(format!(
        "card={card} layer={layer} rows={rows} role={role} backing={} quant={} stream_ms={ms:.6}",
        metadata.0, metadata.1
    ))
}

struct Marks {
    start: TimingEvent,
    end: TimingEvent,
    phase: Phase,
    backing: &'static str,
    quant: String,
}

impl Marks {
    fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        Ok(Self {
            start: ctx.event_timing()?,
            end: ctx.event_timing()?,
            phase: Phase::Pending,
            backing: "unknown",
            quant: String::new(),
        })
    }
}

/// Exactly three pairs per card; allocated outside the chunk's layer walk.
pub(super) struct RoutedTiming {
    marks: [Marks; 3],
}

impl RoutedTiming {
    pub(super) fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        Ok(Self {
            marks: [Marks::new(ctx)?, Marks::new(ctx)?, Marks::new(ctx)?],
        })
    }

    pub(super) fn start(
        &mut self,
        index: usize,
        weight: &NativeWeight,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        let mark = self
            .marks
            .get_mut(index)
            .ok_or_else(|| EngError::BadRequest("P2P expert timing: invalid projection".into()))?;
        let next = next_phase(mark.phase, true)?;
        let backing = weight.buffer().backing_label();
        let quant = format!("{:?}", weight.quant());
        stream.record_timing(&mark.start)?;
        mark.phase = next;
        mark.backing = backing;
        mark.quant = quant;
        Ok(())
    }

    pub(super) fn finish(&mut self, index: usize, stream: &DeviceStream) -> Result<(), EngError> {
        let mark = self
            .marks
            .get_mut(index)
            .ok_or_else(|| EngError::BadRequest("P2P expert timing: invalid projection".into()))?;
        let next = next_phase(mark.phase, false)?;
        stream.record_timing(&mark.end)?;
        mark.phase = next;
        Ok(())
    }

    /// Called only after every card has successfully completed chunk close.
    fn records(&self, card: usize, layer: usize, rows: usize) -> Result<Vec<String>, EngError> {
        if self.marks.iter().any(|mark| mark.phase != Phase::Finished) {
            return Err(EngError::BadRequest(
                "P2P expert timing: incomplete marks".into(),
            ));
        }
        let mut records = Vec::with_capacity(3);
        for (role, mark) in ["gate", "up", "down"].iter().zip(&self.marks) {
            let ms = mark.start.elapsed_ms_since(&mark.end)?;
            records.push(format_record(
                card,
                layer,
                rows,
                role,
                (mark.backing, &mark.quant),
                ms,
            )?);
        }
        Ok(records)
    }
}

/// Print no partial timing batch if any card has incomplete or invalid marks.
pub(super) fn report(layers: &[(usize, Vec<RoutedTiming>)], rows: usize) -> Result<(), EngError> {
    let phases: Vec<Vec<[Phase; 3]>> = layers
        .iter()
        .map(|(_, cards)| {
            cards
                .iter()
                .map(|timing| std::array::from_fn(|i| timing.marks[i].phase))
                .collect()
        })
        .collect();
    if !complete_batch(&phases) {
        return Err(EngError::BadRequest(
            "P2P expert timing: incomplete timing batch".into(),
        ));
    }
    let mut records = Vec::new();
    for (layer, cards) in layers {
        for (card, timing) in cards.iter().enumerate() {
            records.extend(timing.records(card, *layer, rows)?);
        }
    }
    eprintln!("q36-p2p-expert-timing: {}", records.join("; "));
    Ok(())
}

/// Opt-in timing control shared by the P2P chunk diagnostic (`Q36_TIERED_TRACE=1`).
pub(super) fn tiered_trace() -> bool {
    std::env::var_os("Q36_TIERED_TRACE").is_some_and(|value| value == "1")
}

#[cfg(test)]
mod tests {
    use super::{Phase, complete_batch, format_record, next_phase};

    #[test]
    fn marks_must_start_then_finish_once() {
        assert_eq!(next_phase(Phase::Pending, true).unwrap(), Phase::Started);
        assert_eq!(next_phase(Phase::Started, false).unwrap(), Phase::Finished);
        assert!(next_phase(Phase::Pending, false).is_err());
        assert!(next_phase(Phase::Started, true).is_err());
        assert!(next_phase(Phase::Finished, true).is_err());
        assert!(next_phase(Phase::Finished, false).is_err());
    }

    #[test]
    fn batch_requires_all_projections_for_each_card_and_layer() {
        let done = [Phase::Finished; 3];
        assert!(!complete_batch(&[]));
        assert!(!complete_batch(&[vec![]]));
        assert!(complete_batch(&[vec![done, done], vec![done, done]]));
        assert!(!complete_batch(&[vec![done, done], vec![done]]));
        assert!(!complete_batch(&[vec![done, [Phase::Started; 3]]]));
    }

    #[test]
    fn record_rejects_invalid_time_and_preserves_unknown_backing() {
        let record = format_record(1, 59, 1024, "gate", ("unknown", "Q4K"), 1.5).unwrap();
        assert_eq!(
            record,
            "card=1 layer=59 rows=1024 role=gate backing=unknown quant=Q4K stream_ms=1.500000"
        );
        assert!(format_record(0, 0, 1, "up", ("device", "Q4K"), f32::NAN).is_err());
        assert!(format_record(0, 0, 1, "up", ("device", "Q4K"), f32::INFINITY).is_err());
        assert!(format_record(0, 0, 1, "up", ("device", "Q4K"), -1.0).is_err());
    }
}
