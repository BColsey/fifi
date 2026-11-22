//! A state that holds the same sequence in a larger cache.
//!
//! A session starts with the cells its prompt needs and grows as it decodes. The planes are one
//! allocation with a stride per layer, so more cells mean a new allocation: the new state is
//! built at the larger size and the device copies the committed prefix of every layer, plane and
//! head, and the DeltaNet state, from the old planes into the new ones on the owning stream. The
//! old state is released when the caller replaces it, after the copies, which the free orders.

use super::super::dims::{StepDims, StepOptions};
use super::{ResidentState, export::IMPORT_REWRITES_WORDS};
use crate::family::graph_state::book::Health;
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceStream},
};

impl ResidentState {
    /// A state with `options.context_tokens` cells holding this state's tokens, DeltaNet state and
    /// committed cache. The step words restart: the next token writes them from the token count.
    /// Refused for a state that is not healthy and for a size that does not hold its tokens. This
    /// state is left as it was.
    pub(crate) fn grown(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        if self.health() != Health::Valid {
            return Err(EngError::BadRequest(
                "a state that is not valid cannot grow".to_owned(),
            ));
        }
        let tokens = self.book.tokens();
        if options.context_tokens < tokens {
            return Err(EngError::BadRequest(format!(
                "a state of {tokens} positions cannot shrink to {} cells",
                options.context_tokens
            )));
        }
        let old_kv = self.resident_kv()?;
        let mut grown = Self::allocate(ctx, stream, dims, options)?;
        if !grown.layout.same_form(&self.layout) {
            return Err(EngError::BadRequest(
                "a grown state must keep the cache form".to_owned(),
            ));
        }
        grown
            .book
            .begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
        ctx.copy_device(
            stream,
            self.delta.range(0, self.layout.delta.total)?,
            grown.delta.range(0, grown.layout.delta.total)?,
        )?;
        let new_kv = grown.resident_kv()?;
        for layer in 0..self.layout.kv.layers {
            let from = self.layout.kv.prefix_ranges(layer, tokens)?;
            let to = grown.layout.kv.prefix_ranges(layer, tokens)?;
            for (old, new) in from.into_iter().zip(to) {
                if old.bytes == 0 {
                    continue;
                }
                ctx.copy_device(
                    stream,
                    old_kv.range(old.device_offset, old.bytes)?,
                    new_kv.range(new.device_offset, new.bytes)?,
                )?;
            }
        }
        grown.book.finish_overwrite(tokens);
        Ok(grown)
    }
}
