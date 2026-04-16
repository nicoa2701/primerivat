use std::thread;

use crate::sieve::pi_at;

/// Computes S₂ = P₂(x, a) — the count of integers ≤ x with exactly two
/// prime factors both strictly greater than pₐ.
///
/// Formula (Meissel-Lehmer decomposition, paper §2):
///   S₂ = Σ_{k=a+1}^{b} [ π(floor(x / pₖ)) − (k−1) ]
///
/// where b = π(floor(√x)) and the sum runs over primes pₖ with pₐ < pₖ ≤ √x.
/// Each term counts the primes q ≥ pₖ such that pₖ · q ≤ x.
///
/// Using 0-based indexing for `primes` (primes[j] = p_{j+1} in paper notation),
/// the term for index j is:  π(floor(x / primes[j])) − j
///
/// Note: π(floor(x / primes[j])) ≥ j + 1 > j always holds because
/// x / primes[j] ≥ x / z = √x = z, and π(z) = b ≥ j + 1.
///
/// # Preconditions
/// - `primes` contains all primes ≤ floor(√x) in ascending order.
/// - `a = π(floor(x^(1/3)))` (Meissel parameterisation).
/// - `small`, `large`, `z` are the Lucy_Hedgehog tables for `x`.
#[allow(non_snake_case)]
pub fn s2(
    x: u128,
    a: usize,
    z: usize,
    primes: &[u64],
    small: &[u64],
    large: &[u64],
    threads: usize,
) -> u128 {
    let b = primes.len();
    if b <= a {
        return 0;
    }

    let threads = threads.max(1).min(b - a);
    if threads == 1 {
        return s2_range(x, a, b, z, primes, small, large);
    }

    let span = b - a;
    let chunk = span.div_ceil(threads);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);

        for t in 0..threads {
            let start = a + t * chunk;
            if start >= b {
                break;
            }
            let end = (start + chunk).min(b);

            handles.push(scope.spawn(move || s2_range(x, start, end, z, primes, small, large)));
        }

        handles
            .into_iter()
            .map(|h| h.join().expect("S2 worker thread panicked"))
            .sum()
    })
}

#[inline]
fn s2_range(
    x: u128,
    start: usize,
    end: usize,
    z: usize,
    primes: &[u64],
    small: &[u64],
    large: &[u64],
) -> u128 {
    let mut sum = 0u128;

    for j in start..end {
        let p = primes[j] as u128;
        let pi_xp = pi_at(x / p, x, z, small, large) as u128;
        sum += pi_xp - j as u128;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{icbrt, isqrt};
    use crate::sieve::{extract_primes, lucy_hedgehog};

    fn compute_s2(x: u128, threads: usize) -> u128 {
        let z = isqrt(x) as usize;
        let y = icbrt(x);
        let (small, large) = lucy_hedgehog(x);
        let a = pi_at(y, x, z, &small, &large) as usize;
        let primes = extract_primes(&small, z);
        s2(x, a, z, &primes, &small, &large, threads)
    }

    #[test]
    fn test_s2_x10() {
        assert_eq!(compute_s2(10, 1), 1);
        assert_eq!(compute_s2(10, 4), 1);
    }

    #[test]
    fn test_s2_x100() {
        assert_eq!(compute_s2(100, 1), 9);
        assert_eq!(compute_s2(100, 4), 9);
    }

    #[test]
    fn test_s2_x1000() {
        let s1 = compute_s2(1_000, 1);
        let s4 = compute_s2(1_000, 4);
        let s8 = compute_s2(1_000, 8);
        assert_eq!(s1, s4);
        assert_eq!(s1, s8);
    }
}
