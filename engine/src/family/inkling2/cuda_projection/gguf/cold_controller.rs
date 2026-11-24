//! One reusable F32 controller triple for visit-major prompt execution.

use super::{ControllerWeights, CudaContext, DEVICE_HEADROOM, DenseDtype, EngError, Resident};

pub(super) struct Staging([Resident; 3]);

impl Staging {
    pub(super) fn admit(
        context: &CudaContext,
        [rank, hidden]: [usize; 2],
    ) -> Result<Option<Self>, EngError> {
        let Some(gate_cols) = hidden.checked_mul(2) else {
            return Ok(None);
        };
        let Some(pair_bytes) = rank
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(size_of::<f32>()))
        else {
            return Ok(None);
        };
        let Some(total) = pair_bytes.checked_mul(4) else {
            return Ok(None);
        };
        if total == 0 {
            return Ok(None);
        }
        let shapes = [[rank, gate_cols], [hidden, rank], [hidden, rank]];
        let sizes = [2 * pair_bytes, pair_bytes, pair_bytes];
        let total = total as u64;
        if !fits(context.memory_info()?.free as u64, total) {
            return Ok(None);
        }
        let allocate = |i: usize| -> Result<Resident, EngError> {
            Ok(Resident {
                codes: context.allocate(sizes[i])?,
                shape: shapes[i],
            })
        };
        let buffers = (|| Ok::<_, EngError>([allocate(0)?, allocate(1)?, allocate(2)?]))();
        match buffers {
            Ok(buffers) if fits(context.memory_info()?.free as u64, 0) => Ok(Some(Self(buffers))),
            Ok(_) | Err(EngError::MemoryPlan(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(super) fn upload(
        &self,
        context: &CudaContext,
        payload: [&[u8]; 3],
    ) -> Result<ControllerWeights<'_>, EngError> {
        if self
            .0
            .iter()
            .zip(payload)
            .any(|(slot, bytes)| slot.codes.bytes() != bytes.len())
        {
            return Err(super::bad(
                "CUDA staged controller extent differs from startup shape",
            ));
        }
        for (buffer, bytes) in self.0.iter().zip(payload) {
            // Blocking upload, then the batch returns only after the projection
            // stream has completed. No staged data is used across visits.
            context.upload_at(&buffer.codes, 0, bytes)?;
        }
        Ok(self.0.each_ref().map(|buffer| (buffer, DenseDtype::F32)))
    }
}

fn fits(free: u64, bytes: u64) -> bool {
    // Like device expert prefill, this workspace uses the reserve above the
    // family floor after experts and persistent projection placement.
    free.checked_sub(bytes)
        .is_some_and(|remaining| remaining >= DEVICE_HEADROOM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_controller_workspace_does_not_spend_the_family_floor() {
        let bytes = 2304 * 64 * 4 * 4;
        assert!(fits(DEVICE_HEADROOM + bytes, bytes));
        assert!(!fits(DEVICE_HEADROOM + bytes - 1, bytes));
        assert!(!fits(DEVICE_HEADROOM - 1, 0));
        assert!(!fits(0, bytes));
    }
}
