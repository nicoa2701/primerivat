/// Computes floor(√n) — the integer square root of n.
///
/// Uses Newton's method for integer arithmetic to avoid floating-point
/// rounding errors near perfect squares (common with f64 for large n).
/// The result r satisfies r² ≤ n < (r+1)².
///
/// # Preconditions
/// - n can be any u128 value; the algorithm is exact throughout.
pub fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // Initial estimate via f64 (exact for n ≤ 2^53 ≈ 9×10^15)
    let mut r = (n as f64).sqrt() as u128;
    // Decrease while r² > n (handles f64 overshoot)
    while r > 0 && r.saturating_mul(r) > n {
        r -= 1;
    }
    // Increase while (r+1)² ≤ n (handles f64 undershoot)
    loop {
        let next = r + 1;
        match next.checked_mul(next) {
            Some(sq) if sq <= n => r = next,
            _ => break,
        }
    }
    r
}

/// Computes floor(∛n) — the integer cube root of n.
///
/// Uses Newton's method for integer arithmetic to avoid floating-point
/// rounding errors near perfect cubes.
/// The result r satisfies r³ ≤ n < (r+1)³.
///
/// # Preconditions
/// - n can be any u128 value; for the intended range n ≤ 10^19,
///   r ≤ 2.15×10^6 and r³ fits comfortably in u128.
pub fn icbrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // Initial estimate via f64
    let mut r = (n as f64).cbrt() as u128;
    // Decrease while r³ > n (handles f64 overshoot)
    while r > 0 && r.saturating_mul(r).saturating_mul(r) > n {
        r -= 1;
    }
    // Increase while (r+1)³ ≤ n (handles f64 undershoot)
    loop {
        let next = r + 1;
        match next.checked_mul(next).and_then(|sq| sq.checked_mul(next)) {
            Some(cube) if cube <= n => r = next,
            _ => break,
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3);
        assert_eq!(isqrt(25), 5);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(1_000_000), 1_000);
        assert_eq!(isqrt(1_000_000_000_000_u128), 1_000_000);
        assert_eq!(isqrt(10_000_000_000_000_000_u128), 100_000_000);
    }

    #[test]
    fn test_icbrt() {
        assert_eq!(icbrt(0), 0);
        assert_eq!(icbrt(1), 1);
        assert_eq!(icbrt(7), 1);
        assert_eq!(icbrt(8), 2);
        assert_eq!(icbrt(9), 2);
        assert_eq!(icbrt(26), 2);
        assert_eq!(icbrt(27), 3);
        assert_eq!(icbrt(1_000), 10);
        assert_eq!(icbrt(1_000_000_000_u128), 1_000);
        assert_eq!(icbrt(1_000_000_000_000_u128), 10_000);
        // floor((10^16)^(1/3)) = floor(10^5.333) = 215_443
        assert_eq!(icbrt(10_000_000_000_000_000_u128), 215_443);
        // floor((10^19)^(1/3)) = floor(10^6.333) = 2_154_434
        assert_eq!(icbrt(10_000_000_000_000_000_000_u128), 2_154_434);
    }
}
