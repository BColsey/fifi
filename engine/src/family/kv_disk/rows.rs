//! Cutting a cache laid out `[heads][capacity][row]` to its valid rows and padding it back, the
//! one shape of a stored attention cache: only the rows a snapshot covers go to disk, and a read
//! gets a cache of the capacity its owner grows to.

use crate::EngError;

fn bad(what: &str) -> EngError {
    EngError::BadRequest(format!("kv disk: {what}"))
}

/// Appends the first `len` rows of every head of `data` (`heads` heads of `cap` rows of
/// `row` bytes) to `out`.
pub(crate) fn cut_rows(
    out: &mut Vec<u8>,
    data: &[u8],
    heads: usize,
    cap: usize,
    len: usize,
    row: usize,
) -> Result<(), EngError> {
    let whole = heads
        .checked_mul(cap)
        .and_then(|n| n.checked_mul(row))
        .filter(|&n| n == data.len() && len <= cap)
        .ok_or_else(|| bad("a cache does not have the shape it is cut by"))?;
    let _ = whole;
    for head in 0..heads {
        let start = head * cap * row;
        out.extend_from_slice(&data[start..start + len * row]);
    }
    Ok(())
}

/// The inverse of [`cut_rows`]: `part` holds `len` rows of every head, and the result holds `cap`
/// rows of every head, the rows past `len` zero.
pub(crate) fn pad_rows(
    part: &[u8],
    heads: usize,
    len: usize,
    cap: usize,
    row: usize,
) -> Result<Vec<u8>, EngError> {
    let cut = len
        .checked_mul(row)
        .ok_or_else(|| bad("a stored cache is too long"))?;
    if len > cap || part.len() != heads * cut {
        return Err(bad(
            "a stored cache does not have the shape it is padded by",
        ));
    }
    let mut padded = vec![0u8; heads * cap * row];
    for head in 0..heads {
        let to = head * cap * row;
        padded[to..to + cut].copy_from_slice(&part[head * cut..(head + 1) * cut]);
    }
    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_cache_pads_back_to_the_valid_rows_with_zero_slack() {
        // Two heads, capacity four, row of three bytes, two valid rows.
        let data: Vec<u8> = (1..=24).collect();
        let mut cut = Vec::new();
        cut_rows(&mut cut, &data, 2, 4, 2, 3).unwrap();
        assert_eq!(cut, [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18]);
        let padded = pad_rows(&cut, 2, 2, 4, 3).unwrap();
        assert_eq!(&padded[..6], &data[..6]);
        assert_eq!(&padded[6..12], &[0; 6]);
        assert_eq!(&padded[12..18], &data[12..18]);
        assert_eq!(&padded[18..], &[0; 6]);
    }

    #[test]
    fn a_cache_of_another_shape_is_refused() {
        assert!(cut_rows(&mut Vec::new(), &[0; 23], 2, 4, 2, 3).is_err());
        assert!(cut_rows(&mut Vec::new(), &[0; 24], 2, 4, 5, 3).is_err());
        assert!(pad_rows(&[0; 11], 2, 2, 4, 3).is_err());
        assert!(pad_rows(&[0; 12], 2, 5, 4, 3).is_err());
    }
}
