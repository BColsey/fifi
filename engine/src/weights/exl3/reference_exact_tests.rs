//! The unrounded codebook value against the rounded one, over every state.

use half::f16;

use super::Codebook;

#[test]
fn rounding_the_exact_mul1_value_once_gives_the_stored_value_for_every_state() {
    let mut differing = 0usize;
    for state in 0..=u16::MAX {
        let exact = Codebook::Mul1.decode_state_exact(state);
        let rounded = Codebook::Mul1.decode_state(state);
        assert_eq!(f16::from_f64(exact).to_bits(), rounded, "state {state}");
        let gap = (exact - f16::from_bits(rounded).to_f64()).abs();
        // Round to nearest: at most half a binary16 step, which is at most 2^-11 of the value.
        assert!(
            gap <= exact.abs() / 2048.0,
            "state {state}: gap {gap} of {exact}"
        );
        differing += usize::from(gap > 0.0);
    }
    // The value depends on the byte sum alone, and most sums are not representable in binary16.
    assert!(differing > 0, "every state is exactly representable");
    eprintln!("{differing} of 65536 states differ from their rounded value");
}

#[test]
fn the_other_codebooks_have_no_unrounded_form_and_return_the_rounded_value() {
    for book in [Codebook::ThreeInst, Codebook::Mcg] {
        for state in [0u16, 1, 77, 4096, u16::MAX] {
            assert_eq!(
                book.decode_state_exact(state),
                f16::from_bits(book.decode_state(state)).to_f64()
            );
        }
    }
}

#[test]
fn an_exact_tile_rounds_to_the_decoded_tile_and_keeps_the_same_permutation() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for words in [16usize, 24, 48, 80] {
        let tile: Vec<u16> = (0..words).map(|_| (next() >> 20) as u16).collect();
        let mut rounded = [0u16; 256];
        let mut exact = [0f64; 256];
        Codebook::Mul1.decode_tile(&tile, &mut rounded).unwrap();
        Codebook::Mul1
            .decode_tile_with(&tile, &mut exact, |s| Codebook::Mul1.decode_state_exact(s))
            .unwrap();
        for (position, (&bits, &value)) in rounded.iter().zip(&exact).enumerate() {
            assert_eq!(
                f16::from_f64(value).to_bits(),
                bits,
                "words {words} position {position}"
            );
        }
    }
}
