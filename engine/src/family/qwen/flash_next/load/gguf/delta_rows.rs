//! Host-side logical-to-stored row plan for GGUF DeltaNet value heads.

/// Map grouped logical rows to the file's tiled rows. The first `skip` rows
/// are unchanged; the remainder must contain exactly `value_heads` heads.
/// This is the row-level equivalent of `projector::untile`.
pub(super) fn delta_rows(
    skip: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    total_rows: usize,
) -> Result<Vec<usize>, &'static str> {
    if head_dim == 0 || key_heads == 0 || value_heads == 0 || !value_heads.is_multiple_of(key_heads)
    {
        return Err("invalid DeltaNet head dimensions");
    }
    let rows = value_heads
        .checked_mul(head_dim)
        .and_then(|n| skip.checked_add(n))
        .ok_or("DeltaNet row count overflow")?;
    if rows != total_rows {
        return Err("invalid DeltaNet total row count");
    }
    let mut mapping = Vec::new();
    mapping
        .try_reserve_exact(total_rows)
        .map_err(|_| "DeltaNet row plan allocation failed")?;
    mapping.extend(0..skip);
    let per_key = value_heads / key_heads;
    for h in 0..value_heads {
        // Each term is bounded by value_heads; the validated row count
        // bounds the remaining multiplication and addition as well.
        let stored_head = (h % per_key) * key_heads + h / per_key;
        let start = skip + stored_head * head_dim;
        mapping.extend(start..start + head_dim);
    }
    Ok(mapping)
}

/// Read logical, whole-row chunks from a stored row permutation without staging the tensor.
/// The mapping is validated before any source read, including duplicate or out-of-bounds rows.
pub(super) struct DeltaRowReader {
    rows: Vec<usize>,
    row_bytes: usize,
    total_bytes: usize,
}

impl DeltaRowReader {
    pub(super) fn new(
        rows: Vec<usize>,
        row_bytes: usize,
        total_bytes: usize,
    ) -> Result<Self, &'static str> {
        if rows.is_empty()
            || row_bytes == 0
            || rows.len().checked_mul(row_bytes) != Some(total_bytes)
        {
            return Err("invalid DeltaNet row byte geometry");
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(rows.len())
            .map_err(|_| "DeltaNet row validation allocation failed")?;
        seen.resize(rows.len(), false);
        for &row in &rows {
            let slot = seen.get_mut(row).ok_or("DeltaNet row outside tensor")?;
            if *slot {
                return Err("duplicate DeltaNet row");
            }
            *slot = true;
        }
        Ok(Self {
            rows,
            row_bytes,
            total_bytes,
        })
    }

    pub(super) fn read(
        &self,
        offset: usize,
        output: &mut [u8],
        mut source: impl FnMut(usize, &mut [u8]) -> Result<(), crate::EngError>,
    ) -> Result<(), crate::EngError> {
        let invalid =
            || crate::EngError::Checkpoint("Flash-Next GGUF: invalid DeltaNet row read".into());
        if !offset.is_multiple_of(self.row_bytes)
            || !output.len().is_multiple_of(self.row_bytes)
            || offset
                .checked_add(output.len())
                .is_none_or(|end| end > self.total_bytes)
        {
            return Err(invalid());
        }
        let first = offset / self.row_bytes;
        let count = output.len() / self.row_bytes;
        let mut done = 0;
        while done < count {
            let stored = self.rows[first + done];
            let mut run = 1;
            while done + run < count && self.rows[first + done + run] == stored + run {
                run += 1;
            }
            let dst = done * self.row_bytes;
            source(
                stored * self.row_bytes,
                &mut output[dst..dst + run * self.row_bytes],
            )?;
            done += run;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DeltaRowReader, delta_rows};

    #[test]
    fn grouped_to_tiled_with_prefix() {
        assert_eq!(
            delta_rows(2, 2, 2, 6, 14).unwrap(),
            [0, 1, 2, 3, 6, 7, 10, 11, 4, 5, 8, 9, 12, 13]
        );
    }

    #[test]
    fn identity() {
        assert_eq!(
            delta_rows(1, 2, 3, 3, 7).unwrap(),
            (0..7).collect::<Vec<_>>()
        );
        assert_eq!(
            delta_rows(0, 2, 1, 6, 12).unwrap(),
            (0..12).collect::<Vec<_>>()
        );
    }

    #[test]
    fn gathers_whole_row_chunks_and_rejects_invalid_ranges() {
        let reader = DeltaRowReader::new(vec![0, 2, 3, 1], 2, 8).unwrap();
        let source = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut out = [0; 6];
        let mut calls = Vec::new();
        reader
            .read(2, &mut out, |offset, dst| {
                calls.push((offset, dst.len()));
                dst.copy_from_slice(&source[offset..offset + dst.len()]);
                Ok(())
            })
            .unwrap();
        assert_eq!(out, [4, 5, 6, 7, 2, 3]);
        assert_eq!(calls, [(4, 4), (2, 2)]);
        let error = reader
            .read(2, &mut out, |_, _| {
                Err(crate::EngError::Checkpoint("source failure".into()))
            })
            .unwrap_err();
        assert!(
            matches!(error, crate::EngError::Checkpoint(message) if message == "source failure")
        );
        assert!(
            reader
                .read(1, &mut out, |_, _| panic!("unexpected read"))
                .is_err()
        );
        assert!(
            reader
                .read(4, &mut out, |_, _| panic!("unexpected read"))
                .is_err()
        );
        assert!(
            reader
                .read(0, &mut out[..3], |_, _| panic!("unexpected read"))
                .is_err()
        );
        assert!(DeltaRowReader::new(vec![0, 0], 2, 4).is_err());
        assert!(DeltaRowReader::new(vec![0, 2], 2, 4).is_err());
        assert!(DeltaRowReader::new(vec![0, 1], 2, 5).is_err());
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert!(delta_rows(0, 0, 2, 6, 0).is_err());
        assert!(delta_rows(0, 2, 0, 6, 12).is_err());
        assert!(delta_rows(0, 2, 2, 0, 0).is_err());
        assert!(delta_rows(0, 2, 4, 6, 12).is_err());
        assert!(delta_rows(1, 2, 2, 6, 12).is_err());
        assert!(delta_rows(0, usize::MAX, 2, 6, 0).is_err());
        assert!(delta_rows(usize::MAX, 1, 1, 1, 0).is_err());
    }
}
