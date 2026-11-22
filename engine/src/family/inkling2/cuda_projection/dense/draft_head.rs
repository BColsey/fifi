//! Device-side greedy selection for the two tied DSpark vocabulary projections.

use std::cell::RefCell;

use super::{DenseStep, Resident, bad, float_bytes};
use crate::{
    EngError,
    family::{inkling2::cuda_projection::DEVICE_HEADROOM, speculate::dspark::softcap_logit_bf16},
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceRange, PinnedBuf, plan_arena,
        resident::{
            argmax_rows::{ArgmaxRowsCall, argmax_rows_part_bytes},
            correct_draft_logits_bf16::{CorrectDraftLogitsCall, SOFTCAP_LUT_BYTES},
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes},
            step_params::{MAX_LOGITS, STATUS_NON_FINITE_LOGITS},
        },
    },
};
use half::bf16;

const RESULT_BYTES: usize = 8;

#[repr(usize)]
enum Slot {
    Input,
    Logits,
    Lut,
    ArgmaxPart,
    Result,
}

struct Transfer {
    pinned: PinnedBuf,
    encoded: Vec<u8>,
}

pub(in crate::family::inkling2) struct DraftHead {
    arena: DeviceArena,
    transfer: RefCell<Transfer>,
    cap: f32,
    shape: DenseGemvShape,
}

impl DraftHead {
    pub(in crate::family::inkling2) fn new(
        cuda: &CudaContext,
        [vocab, width]: [usize; 2],
        dtype: DenseDtype,
        cap: f32,
    ) -> Result<Self, EngError> {
        if vocab == 0
            || vocab > MAX_LOGITS
            || !cap.is_finite()
            || cap <= 0.0
            || !bf16::from_f32(cap).is_finite()
        {
            return Err(bad(
                "CUDA draft head has an unsupported vocabulary or logit cap",
            ));
        }
        let shape = DenseGemvShape::packed(2, vocab, width);
        let need = dense_gemv_bytes(shape, dtype)?;
        let tensors = [
            need.x,
            need.y,
            SOFTCAP_LUT_BYTES,
            argmax_rows_part_bytes(1),
            RESULT_BYTES,
        ]
        .map(|bytes| ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        });
        let planned = plan_arena(&tensors)?.total as u64;
        let available = (cuda.memory_info()?.free as u64).saturating_sub(DEVICE_HEADROOM);
        if planned > available {
            return Err(EngError::MemoryPlan(format!(
                "CUDA draft head needs {planned} device bytes, {available} available"
            )));
        }
        let arena = cuda.arena(&tensors)?;
        let lut = cap_lookup(cap);
        cuda.upload_at(
            arena.buffer(),
            arena.tensor(Slot::Lut as usize)?.offset(),
            &lut,
        )?;
        let transfer = Transfer {
            pinned: cuda.pinned_buf(need.x.max(RESULT_BYTES))?,
            encoded: Vec::with_capacity(need.x),
        };
        let free = cuda.memory_info()?.free as u64;
        if free < DEVICE_HEADROOM {
            return Err(EngError::MemoryPlan(format!(
                "CUDA draft head has {free} device bytes left after allocation"
            )));
        }
        Ok(Self {
            arena,
            transfer: RefCell::new(transfer),
            cap,
            shape,
        })
    }

    pub(in crate::family::inkling2) fn choose(
        &self,
        step: &DenseStep,
        cuda: &CudaContext,
        (weight, dtype): (&Resident, DenseDtype),
        rows: [&[f32]; 2],
        cap: f32,
    ) -> Result<u32, EngError> {
        if cap != self.cap {
            return Err(bad("CUDA draft head logit cap changed after admission"));
        }
        let [base, correction] = self.logits()?;
        let result = self.arena.tensor(Slot::Result as usize)?;
        let part = self.arena.tensor(Slot::ArgmaxPart as usize)?;
        let lut = self.arena.tensor(Slot::Lut as usize)?;
        let stream = &step.stream;
        let mut transfer = self.transfer.borrow_mut();
        self.upload_rows(step, cuda, rows, &mut transfer)?;
        let launch = (|| {
            self.project_pair(step, cuda, weight, dtype, [base, correction])?;
            cuda.correct_draft_logits_bf16(
                stream,
                CorrectDraftLogitsCall {
                    len: self.shape.rows,
                    base,
                    correction,
                    softcap_lut: lut,
                },
            )?;
            cuda.argmax_rows(
                stream,
                ArgmaxRowsCall {
                    m: 1,
                    n: self.shape.rows,
                    ld: self.shape.rows,
                    logits: base,
                    part,
                    result,
                },
            )?;
            cuda.copy_to_pinned(stream, result, &mut transfer.pinned, 0)
        })();
        // Retire the input transfer even if a later launch is refused.
        let waited = transfer.pinned.synchronize(stream);
        launch?;
        waited?;
        let mut bytes = [0u8; RESULT_BYTES];
        transfer.pinned.read(0, &mut bytes)?;
        decode_result(bytes, self.shape.rows)
    }

    fn upload_rows(
        &self,
        step: &DenseStep,
        cuda: &CudaContext,
        rows: [&[f32]; 2],
        transfer: &mut Transfer,
    ) -> Result<(), EngError> {
        if rows
            .iter()
            .any(|row| row.len() != self.shape.cols || row.iter().any(|x| !x.is_finite()))
        {
            return Err(bad(
                "CUDA draft head requires two finite hidden rows of the admitted width",
            ));
        }
        transfer.encoded.clear();
        for &value in rows.into_iter().flatten() {
            transfer.encoded.extend_from_slice(&value.to_ne_bytes());
        }
        transfer.pinned.write(0, &transfer.encoded)?;
        cuda.copy_from_pinned(
            &step.stream,
            &mut transfer.pinned,
            0,
            self.arena.tensor(Slot::Input as usize)?,
        )
    }

    fn logits(&self) -> Result<[DeviceRange<'_>; 2], EngError> {
        let all = self.arena.tensor(Slot::Logits as usize)?;
        let bytes = float_bytes(self.shape.rows)?;
        Ok([
            all.buffer().range(all.offset(), bytes)?,
            all.buffer().range(all.offset() + bytes, bytes)?,
        ])
    }

    fn project_pair(
        &self,
        step: &DenseStep,
        cuda: &CudaContext,
        weight: &Resident,
        dtype: DenseDtype,
        outputs: [DeviceRange<'_>; 2],
    ) -> Result<(), EngError> {
        let input = self.arena.tensor(Slot::Input as usize)?;
        let w = weight.codes.range(0, weight.codes.bytes())?;
        if !step.uses_q8_activations(dtype) {
            // Rows of this kernel have independent reductions even when batched.
            return cuda.dense_gemv(
                &step.stream,
                DenseGemvCall {
                    shape: self.shape,
                    dtype,
                    w,
                    x: input,
                    y: self.arena.tensor(Slot::Logits as usize)?,
                },
            );
        }
        // Preserve the existing Q8_1 activation path and reuse its scratch in stream order.
        let one = DenseGemvShape::packed(1, self.shape.rows, self.shape.cols);
        let bytes = float_bytes(one.cols)?;
        for (offset, y) in outputs.into_iter().enumerate() {
            let x = input
                .buffer()
                .range(input.offset() + offset * bytes, bytes)?;
            step.enqueue_gguf_projection(
                cuda,
                &step.stream,
                DenseGemvCall {
                    shape: one,
                    dtype,
                    w,
                    x,
                    y,
                },
            )?;
        }
        Ok(())
    }
}

fn cap_lookup(cap: f32) -> Vec<u8> {
    (0..=u16::MAX)
        .flat_map(|bits| {
            softcap_logit_bf16(bf16::from_bits(bits).to_f32(), cap)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn decode_result(bytes: [u8; RESULT_BYTES], vocab: usize) -> Result<u32, EngError> {
    let [token, status] = bytes.as_chunks::<4>().0 else {
        unreachable!()
    };
    let status = i32::from_le_bytes(*status);
    if status & STATUS_NON_FINITE_LOGITS != 0 {
        return Err(bad("nonfinite corrected draft logit"));
    }
    let token = u32::from_le_bytes(*token);
    if status != 0 || token as usize >= vocab {
        return Err(bad("CUDA draft head returned an invalid token or status"));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path};

    use super::*;
    use crate::{
        family::{
            inkling2::{
                OUTPUT_HEAD,
                cuda_projection::{DensePlan, upload_tensor},
            },
            speculate::dspark::choose_token,
        },
        weights::{GgufLimits, GgufWeights, dspark_gguf::DsparkGguf},
    };

    #[test]
    #[ignore = "requires the published Inkling target and DSpark GGUFs and CUDA"]
    fn published_vocabulary_pair_preserves_projection_bits_and_draft_choice() {
        let path = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
        let sidecar = env::var_os("FIFI_TEST_INKLING_DSPARK_GGUF")
            .expect("set FIFI_TEST_INKLING_DSPARK_GGUF");
        let draft = DsparkGguf::open(Path::new(&sidecar)).unwrap();
        let cap = draft.config().final_logit_softcap;
        let [vocab, width] = [draft.config().vocab, draft.config().width];
        let target = GgufWeights::open_any_type(
            Path::new(&path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 1 << 30,
            },
            None,
        )
        .unwrap();
        let tensor = target.header().tensor(OUTPUT_HEAD).unwrap();
        assert_eq!(tensor.dims().len(), 2);
        assert_eq!(tensor.dims()[0], width as u64);
        assert!(tensor.dims()[1] >= vocab as u64);
        let (kind, weights) = target.read_tensor(OUTPUT_HEAD).unwrap();
        let dtype = DenseDtype::from_gguf(kind).unwrap();
        let need = dense_gemv_bytes(DenseGemvShape::packed(1, vocab, width), dtype).unwrap();

        let cuda = CudaContext::new().unwrap();
        let step =
            DenseStep::new(&cuda, &DensePlan::new([vocab, width], width, 1).unwrap()).unwrap();
        let resident = Resident {
            codes: upload_tensor(&cuda, &weights[..need.w]).unwrap(),
            shape: [vocab, width],
        };
        let head = DraftHead::new(&cuda, resident.shape, dtype, cap).unwrap();
        let inputs: [Vec<f32>; 2] = [19, 31].map(|period| {
            (0..width)
                .map(|i| ((i % period) as f32 - (period / 2) as f32) / period as f32)
                .collect()
        });
        let rows = inputs.each_ref().map(Vec::as_slice);
        let reference = rows.map(|row| {
            let range = resident.codes.range(0, resident.codes.bytes()).unwrap();
            step.project_gguf(&cuda, None, (dtype, range, resident.shape, 1.0), row)
                .unwrap()
        });

        let logits = head.logits().unwrap();
        let mut transfer = head.transfer.borrow_mut();
        head.upload_rows(&step, &cuda, rows, &mut transfer).unwrap();
        head.project_pair(&step, &cuda, &resident, dtype, logits)
            .unwrap();
        transfer.pinned.synchronize(&step.stream).unwrap();
        drop(transfer);
        for (logits, expected) in logits.into_iter().zip(&reference) {
            let mut bytes = vec![0; logits.bytes()];
            logits
                .buffer()
                .read_at(&step.stream, logits.offset(), &mut bytes)
                .unwrap();
            for (raw, &want) in bytes.as_chunks::<4>().0.iter().zip(expected) {
                assert_eq!(u32::from_ne_bytes(*raw), want.to_bits());
            }
        }
        let expected = choose_token(&reference[0], &reference[1], vocab, cap).unwrap();
        assert_eq!(
            head.choose(&step, &cuda, (&resident, dtype), rows, cap)
                .unwrap(),
            expected
        );
        eprintln!(
            "GGUF tied pair matched {} raw logits and token {expected}",
            2 * vocab
        );
    }
}
