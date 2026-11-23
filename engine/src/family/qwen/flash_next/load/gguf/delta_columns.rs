//! Host-side grouped-to-tiled DeltaNet column reads, preserving encoded blocks.

pub(super) struct DeltaColumnReader {
    key_heads: usize,
    per_key: usize,
    head_bytes: usize,
    row_bytes: usize,
    total_bytes: usize,
}

impl DeltaColumnReader {
    /// `block_elements` logical elements occupy `block_bytes` encoded bytes.
    /// Each row contains exactly `value_heads * head_dim` elements.
    pub(super) fn new(
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        block_elements: usize,
        block_bytes: usize,
        rows: usize,
        total_bytes: usize,
    ) -> Result<Self, &'static str> {
        if key_heads == 0
            || value_heads == 0
            || head_dim == 0
            || block_elements == 0
            || block_bytes == 0
            || rows == 0
            || value_heads % key_heads != 0
            || head_dim % block_elements != 0
        {
            return Err("invalid DeltaNet column geometry");
        }
        let head_bytes = (head_dim / block_elements)
            .checked_mul(block_bytes)
            .ok_or("DeltaNet column byte overflow")?;
        let row_bytes = value_heads
            .checked_mul(head_bytes)
            .ok_or("DeltaNet column byte overflow")?;
        if row_bytes.checked_mul(rows) != Some(total_bytes) {
            return Err("invalid DeltaNet column tensor length");
        }
        Ok(Self {
            key_heads,
            per_key: value_heads / key_heads,
            head_bytes,
            row_bytes,
            total_bytes,
        })
    }

    /// Read complete logical rows; source offsets and lengths are encoded bytes.
    pub(super) fn read(
        &self,
        offset: usize,
        output: &mut [u8],
        mut source: impl FnMut(usize, &mut [u8]) -> Result<(), crate::EngError>,
    ) -> Result<(), crate::EngError> {
        let invalid =
            || crate::EngError::Checkpoint("Flash-Next GGUF: invalid DeltaNet column read".into());
        if offset % self.row_bytes != 0
            || output.len() % self.row_bytes != 0
            || offset
                .checked_add(output.len())
                .is_none_or(|end| end > self.total_bytes)
        {
            return Err(invalid());
        }
        let heads = self.row_bytes / self.head_bytes;
        for (row, dst) in output.chunks_mut(self.row_bytes).enumerate() {
            let base = offset + row * self.row_bytes;
            let mut h = 0;
            while h < heads {
                let stored = (h % self.per_key) * self.key_heads + h / self.per_key;
                let mut run = 1;
                while h + run < heads {
                    let next = h + run;
                    if (next % self.per_key) * self.key_heads + next / self.per_key != stored + run
                    {
                        break;
                    }
                    run += 1;
                }
                source(
                    base + stored * self.head_bytes,
                    &mut dst[h * self.head_bytes..(h + run) * self.head_bytes],
                )?;
                h += run;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DeltaColumnReader;

    #[test]
    fn grouped_reads_tiled_blocks_across_rows() {
        // Six heads, two blocks per head, two distinctive bytes per block.
        let src: Vec<u8> = (0..48).collect();
        let reader = DeltaColumnReader::new(2, 6, 4, 2, 2, 2, 48).unwrap();
        let mut out = vec![0; 48];
        reader
            .read(0, &mut out, |off, dst| {
                dst.copy_from_slice(&src[off..off + dst.len()]);
                Ok(())
            })
            .unwrap();
        let mut expected = Vec::new();
        for row in 0..2 {
            for h in [0, 2, 4, 1, 3, 5] {
                expected.extend_from_slice(&src[row * 24 + h * 4..row * 24 + h * 4 + 4]);
            }
        }
        assert_eq!(out, expected);
        let mut second = [0; 24];
        reader
            .read(24, &mut second, |off, dst| {
                dst.copy_from_slice(&src[off..off + dst.len()]);
                Ok(())
            })
            .unwrap();
        assert_eq!(&second, &expected[24..]);
    }

    #[test]
    fn identity_and_coalescing() {
        let reader = DeltaColumnReader::new(3, 3, 2, 2, 3, 1, 9).unwrap();
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut out = [0; 9];
        let mut calls = 0;
        reader
            .read(0, &mut out, |off, dst| {
                calls += 1;
                dst.copy_from_slice(&src[off..off + dst.len()]);
                Ok(())
            })
            .unwrap();
        assert_eq!(out, src);
        assert_eq!(calls, 1);
    }

    #[test]
    fn q6_superblock_spanning_two_heads_is_refused() {
        // Pinned Flash-Next geometry: Q6_K's 256-value block shares one scale
        // across two 128-value heads, so copying whole head blocks cannot work.
        let bytes = 2560 * (48 * 128 / 256) * 210;
        assert_eq!(bytes, 12_902_400);
        assert!(DeltaColumnReader::new(16, 48, 128, 256, 210, 2560, bytes).is_err());
    }

    #[test]
    fn invalid_geometry_ranges_and_source_error() {
        for args in [
            (0, 6, 4, 2, 2, 2, 48),
            (4, 6, 4, 2, 2, 2, 48),
            (2, 6, 3, 2, 2, 2, 48),
            (2, 6, 4, 0, 2, 2, 48),
            (2, 6, 4, 2, 0, 2, 48),
            (2, 6, 4, 2, 2, 0, 0),
            (2, 6, 4, 2, 2, 2, 47),
            (1, 2, usize::MAX, 1, 2, 1, 0),
        ] {
            assert!(
                DeltaColumnReader::new(args.0, args.1, args.2, args.3, args.4, args.5, args.6)
                    .is_err()
            );
        }
        let r = DeltaColumnReader::new(2, 6, 4, 2, 2, 2, 48).unwrap();
        assert!(r.read(1, &mut [0; 24], |_, _| panic!()).is_err());
        assert!(r.read(0, &mut [0; 23], |_, _| panic!()).is_err());
        assert!(r.read(48, &mut [0; 24], |_, _| panic!()).is_err());
        let err = r
            .read(0, &mut [0; 24], |_, _| {
                Err(crate::EngError::Checkpoint("source failure".into()))
            })
            .unwrap_err();
        assert!(matches!(err, crate::EngError::Checkpoint(message) if message == "source failure"));
    }
}
