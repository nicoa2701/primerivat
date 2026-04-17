pub mod hard;

/// `C` — prefix of primes absorbed by the Meissel `φ_tiny` inclusion-exclusion.
/// The DR engine requires `a = π(α·∛x) > C`; otherwise we fall back to
/// the Lucy–Meissel baseline. Exposed so the test suite can pin the
/// baseline-fallback threshold.
const C: usize = 5;

/// Returns `true` when `prime_pi_dr_meissel_v4(x)` would short-circuit to
/// [`crate::baseline::prime_pi`] instead of running the DR sweep. The
/// guard is `a = π(α·∛x) ≤ C = 5`, which empirically holds for every
/// `x < 13³ = 2197`.
#[doc(hidden)]
pub fn uses_baseline_fallback(x: u128) -> bool {
    use crate::math::{icbrt, isqrt};
    use crate::sieve::sieve_to;

    if x < 2 {
        return true;
    }
    let alpha: f64 = crate::parameters::choose_alpha(x);
    let cbrt_x = icbrt(x);
    let sqrt_x = isqrt(x) as u64;
    let y = ((cbrt_x as f64 * alpha) as u64).clamp(cbrt_x as u64, sqrt_x);
    let (_small_pi, seed_primes) = sieve_to(y);
    seed_primes.len() <= C
}

/// Compute π(x) via the Deléglise-Rivat decomposition
///   π(x) = S1 + S2_hard + a − 1 − P2
///
/// Entry point wired into `crate::deleglise_rivat`. Falls back to the
/// Lucy–Meissel baseline when `a = π(α·∛x) ≤ 5` (small `x`).
pub fn prime_pi_dr_meissel_v4(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::s1_ordinary;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;

    if x < 2 {
        return 0;
    }

    // Hardware-adaptive alpha: α=2.0 only when x≥3e16 AND L3<16Mo AND ≤8 cores.
    // Rationale: α=2.0 halves sieve-window count but doubles easy-region phi table.
    // On i5-9300H (8Mo L3, 4c) → gain; on i5-13450HX (20Mo L3, 10c) → regression.
    // Only α ∈ {1.0, 2.0} verified OK (intermediate values cause bugs, see bug_alpha2_fix.md).
    let alpha: f64 = crate::parameters::choose_alpha(x);

    // ── 1. Sieve up to y = alpha · ∛x ────────────────────────────────────────
    let cbrt_x = icbrt(x);
    let sqrt_x = isqrt(x) as u64;
    let y = ((cbrt_x as f64 * alpha) as u64).clamp(cbrt_x as u64, sqrt_x);
    let z = (x / y as u128) as u64;

    let (_small_pi, seed_primes) = sieve_to(y);
    let a = seed_primes.len();

    // ── 2. Hard-prime cutoff: b_max = π(√y) ──────────────────────────────────
    let sqrty = isqrt(y as u128) as u64;
    let b_max = seed_primes.partition_point(|&p| p <= sqrty);

    // ── Guard: algorithm requires a > C; fall back to baseline for small x ───
    if a <= C {
        return crate::baseline::prime_pi(x);
    }

    // ── 3. S1: ordinary leaves DFS ───────────────────────────────────────────
    let s1 = s1_ordinary(x, y, C, &seed_primes);

    // ── 4+5. S2_hard + P2 in one combined sweep ───────────────────────────────
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    let (s2_hard, p2) = hard::s2_hard_sieve_par(x, y, z, C, b_max, a, &seed_primes, s2_primes);

    // ── 6. π(x) = φ(x, a) + a − 1 − P2  with φ(x,a) = S1 + S2_hard ─────────
    let phi_x_a = (s1 + s2_hard) as u128;
    phi_x_a + a as u128 - 1 - p2
}

/// Timed variant of [`prime_pi_dr_meissel_v4`].
/// Returns (result, [step1_sieve, step2_s1, step3_s2_hard_plus_p2, 0, 0]).
pub fn prime_pi_dr_meissel_v4_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::s1_ordinary;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];

    if x < 2 {
        return (0, times);
    }

    let alpha: f64 = crate::parameters::choose_alpha(x);

    // step1: sieve
    let t0 = Instant::now();
    let cbrt_x = icbrt(x);
    let sqrt_x = isqrt(x) as u64;
    let y = ((cbrt_x as f64 * alpha) as u64).clamp(cbrt_x as u64, sqrt_x);
    let z = (x / y as u128) as u64;
    let (_small_pi, seed_primes) = sieve_to(y);
    let a = seed_primes.len();
    let sqrty = isqrt(y as u128) as u64;
    let b_max = seed_primes.partition_point(|&p| p <= sqrty);
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t0.elapsed();

    if a <= C {
        // Small-x fallback does not surface step timings.
        let result = crate::baseline::prime_pi(x);
        return (result, times);
    }

    // step2: S1 (ordinary DFS)
    let t1 = Instant::now();
    let s1 = s1_ordinary(x, y, C, &seed_primes);
    times[1] = t1.elapsed();

    // step3: S2_hard + P2 combined
    let t2 = Instant::now();
    let (s2_hard, p2) = hard::s2_hard_sieve_par(x, y, z, C, b_max, a, &seed_primes, s2_primes);
    times[2] = t2.elapsed();

    let phi_x_a = (s1 + s2_hard) as u128;
    let result = phi_x_a + a as u128 - 1 - p2;
    (result, times)
}

#[cfg(test)]
mod tests {
    use super::prime_pi_dr_meissel_v4;
    use crate::baseline::prime_pi;

    #[test]
    fn prime_pi_dr_meissel_v4_matches_baseline() {
        for x in [
            2u128, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
            1_000_000_000, 10_000_000_000, 100_000_000_000, 1_000_000_000_000,
        ] {
            assert_eq!(prime_pi_dr_meissel_v4(x), prime_pi(x), "mismatch at x = {x}");
        }
    }

    /// The guard `a = π(α·∛x) ≤ C = 5` inside prime_pi_dr_meissel_v4
    /// routes tiny x to the Lucy–Meissel baseline. This test pins the
    /// exact boundary: 13³ = 2197 is the first x for which the DR sweep
    /// actually runs.
    #[test]
    fn baseline_fallback_triggers_below_2197() {
        use super::uses_baseline_fallback;
        for x in [0u128, 1, 2, 10, 100, 233, 500, 1_000, 2_196] {
            assert!(
                uses_baseline_fallback(x),
                "x = {x} should fall back to baseline"
            );
        }
        for x in [2_197u128, 2_500, 10_000, 1_000_000, 1_000_000_000_000] {
            assert!(
                !uses_baseline_fallback(x),
                "x = {x} should use the DR engine"
            );
        }
    }

    /// Known-good π values for very small x, exercising the baseline
    /// fallback path inside prime_pi_dr_meissel_v4 (a = π(α·∛x) ≤ 5
    /// whenever x < 13³ = 2197) and the boundary just above it.
    #[test]
    fn prime_pi_dr_meissel_v4_small_x_known_values() {
        let cases = [
            (10u128, 4u128),       // primes: 2, 3, 5, 7
            (100, 25),             // baseline path
            (233, 51),             // baseline path, not a power of 10
            (500, 95),             // baseline path
            (1_000, 168),          // baseline path
            (2_196, 327),          // last x routed to baseline (icbrt=12)
            (2_197, 327),          // first x routed to DR (icbrt=13, π(13)=6 > C)
        ];
        for (x, expected) in cases {
            assert_eq!(
                prime_pi_dr_meissel_v4(x),
                expected,
                "π({x}) expected {expected}"
            );
        }
    }
}
