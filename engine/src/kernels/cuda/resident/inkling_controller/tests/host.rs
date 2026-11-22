//! Host oracle and synthetic inputs for the CUDA component test.
//! Can also run by itself with `rustc --test`, without the native engine.

pub(crate) const ROLES: usize = 3;

pub(crate) fn exact_tap_fixture(width: usize) -> Vec<f32> {
    let mut words = vec![0.0; 6 * ROLES * width];
    let raw = |role: usize, field: usize, i: usize| (role * 3 + field) * width + i;
    let coeff = |role: usize, field: usize, i: usize| (3 * ROLES + role * 2 + field) * width + i;
    for i in 0..width {
        // One large square followed by squares too small to increment it.
        words[raw(0, 0, i)] = if i == 0 { 4096.0 } else { 1.0 };
        words[raw(0, 2, i)] = 1.0;

        // Signed zero and subnormal x, with squares that underflow to zero.
        words[raw(1, 0, i)] = if i == 0 { -0.0 } else { f32::from_bits(16) };
        words[raw(1, 1, i)] = 1.0;
        words[raw(1, 2, i)] = 1.25;
        words[coeff(1, 0, i)] = -0.0;
        words[coeff(1, 1, i)] = -100.0;

        // Exact f32 tap results for tanh(±100), without relying on host/device
        // agreement at arbitrary coefficients. Exercise both control products.
        words[raw(2, 0, i)] = ((i % 43) as f32 - 17.0) / 13.0;
        words[raw(2, 1, i)] = ((i % 47) as f32 - 19.0) / 11.0;
        words[raw(2, 2, i)] = ((i % 51) as f32 - 21.0) / 7.0;
        words[coeff(2, 0, i)] = if i % 2 == 0 { 100.0 } else { -100.0 };
        words[coeff(2, 1, i)] = if i % 3 == 0 { -100.0 } else { 100.0 };
    }
    words
}

fn controlled(words: &[f32], width: usize, role: usize) -> impl Iterator<Item = f32> + '_ {
    let raw = move |field| (role * 3 + field) * width;
    let coeff = move |field| (3 * ROLES + role * 2 + field) * width;
    (0..width).map(move |i| {
        let input = words[raw(0) + i];
        let message = words[raw(1) + i];
        let gain = words[coeff(0) + i];
        let forget = words[coeff(1) + i];
        (input - forget.tanh() * input) + gain.tanh() * message
    })
}

pub(crate) fn controller_norm(words: &[f32], width: usize, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(ROLES * width);
    for role in 0..ROLES {
        let x: Vec<_> = controlled(words, width, role).collect();
        let sum = x.iter().map(|&x| x * x).sum::<f32>();
        let scale = (sum / width as f32 + eps).sqrt().recip();
        let gamma = (role * 3 + 2) * width;
        out.extend(
            x.into_iter()
                .zip(&words[gamma..gamma + width])
                .map(|(x, g)| (x * scale) * g),
        );
    }
    out
}

#[test]
fn exact_tap_fixture_detects_reduction_reordering() {
    let width = 513;
    let words = exact_tap_fixture(width);
    let x: Vec<_> = controlled(&words, width, 0).collect();
    let sum = x.iter().map(|&x| x * x).sum::<f32>();
    let reversed = x.iter().rev().map(|&x| x * x).sum::<f32>();
    assert_eq!(sum, 4096.0f32.powi(2));
    assert_eq!(reversed, sum + 512.0);
    let scale = |s: f32| (s / width as f32 + 1e-5).sqrt().recip();
    let out = controller_norm(&words, width, 1e-5);
    assert_eq!(out[0].to_bits(), (x[0] * scale(sum)).to_bits());
    assert_ne!(out[0].to_bits(), (x[0] * scale(reversed)).to_bits());
}

#[test]
fn exact_tap_fixture_covers_zero_and_subnormals() {
    let width = 513;
    let words = exact_tap_fixture(width);
    let out = controller_norm(&words, width, 1e-5);
    assert_eq!(out[width].to_bits(), (-0.0f32).to_bits());
    assert!(out[width + 1].is_subnormal());
    let out = controller_norm(&words, width, f32::from_bits(1));
    assert_eq!(out[width].to_bits(), (-0.0f32).to_bits());
    assert!(out[width + 1].is_normal());
}
