use crate::math::{icbrt, isqrt};
use rayon::prelude::*;

// Minimum iteration count before spawning threads for large[] updates.
// Below this threshold the Rayon overhead exceeds the gain.
const PAR_MIN: usize = 4096;

// ── Half-size sieve array ────────────────────────────────────────────────────
//
// Classic Lucy_Hedgehog stores `small[n] = π(n)` for all n ≤ z (size z+1).
// **Skip-pairs optimisation**: prime 2 is handled implicitly at init time so
// the sieve loop only visits odd primes p = 3, 5, 7, …
//
// `s[j]` = number of *odd* primes in [3, 2j+1]  =  π(2j+1) − 1.
//
// Lookup rule (used via `pi_at`):
//   π(n) = 1 + s[(n−1) >> 1]   for n ≥ 3        (the 1 accounts for prime 2)
//   π(2) = 1                    (special-cased)
//   π(n) = 0                    for n ≤ 1
//
// Size: ⌊(z+1)/2⌋ + 1 entries  →  ~4 MiB for x = 10^12 (vs 8 MiB before).

/// Returns π(n) from the half-size Lucy_Hedgehog tables.
pub fn pi_at(n: u128, x: u128, z: usize, s: &[u64], large: &[u64]) -> u64 {
    if n <= 1 {
        return 0;
    }
    if n == 2 {
        return 1;
    }
    if n as usize <= z {
        1 + s[((n - 1) >> 1) as usize]
    } else {
        large[(x / n) as usize]
    }
}

/// Runs the Lucy_Hedgehog prime-counting sieve **with skip-pairs** and
/// **segmented large-array updates**, then captures φ(x, a) for free.
///
/// # Skip-pairs
/// Prime 2 is pre-applied at initialisation.  The main loop only processes
/// odd primes p = 3, 5, 7, … — halving both the inner trip count and the
/// size of the `s` array.
///
/// # Segmented large update
/// For each prime p the k-loop over `large` is split into two phases:
///
/// 1. k ≤ ⌊z/p⌋  — accesses `large[k·p]` (always a *later* index; no
///    write-after-read hazard because we iterate k upward).
///
/// 2. k > ⌊z/p⌋  — accesses `s[(q−1)>>1]` where q = ⌊x/(k·p)⌋.
///    Processed in blocks of `BLOCK` entries so that the active slice of
///    `large` stays in L2 while the smaller `s` (≤ 4 MiB) stays in L3.
///
/// # φ capture (sieve invariant)
/// After processing exactly a = π(y) primes, the sieve invariant gives:
///
///   `large[1] = a − 1 + φ(x, a)`   →   `φ(x, a) = large[1] − a + 1`
///
/// We snapshot `large[1]` just before the first prime > y is visited.
///
/// # Returns
/// `(s, large, phi_x_a)`:
/// - `s[j]`     = π(2j+1) − 1            (half-size, odd-only)
/// - `large[k]` = π(⌊x/k⌋)               (indexed by k, 1 ≤ k ≤ z)
/// - `phi_x_a`  = φ(x, π(y))
///
/// Pass `y = 0` to skip the φ capture (used internally by `lucy_hedgehog`).
///
/// # Preconditions
/// - `x ≥ 2`
pub fn lucy_hedgehog_with_phi(x: u128, y: u128) -> (Vec<u64>, Vec<u64>, u128) {
    let z = isqrt(x) as usize;

    // ── Use u64 arithmetic in hot paths ─────────────────────────────────────
    //
    // x ≤ 10^19 < u64::MAX (≈ 1.84×10^19): all values of the form ⌊x/k⌋ and
    // ⌊x/p⌋ (with k, p ≤ z) fit in u64.  Using u64 ensures the compiler emits
    // the fast hardware `div` instruction instead of a software __udivti3 call.
    debug_assert!(x <= u64::MAX as u128, "lucy_hedgehog requires x ≤ u64::MAX");
    let x64 = x as u64;

    // ── Initialise arrays — prime 2 pre-applied ─────────────────────────────
    //
    // After removing even composites, survivors in [2, n] = {2} ∪ {odd ≥ 3 in [3,n]}.
    // Count = (n+1)/2 for n ≥ 2.
    //
    // s[j]: odd-survivors count = (total survivors) − 1 (subtracting the even prime 2).
    //   s[0] = 0  (no integers ≥ 2 in [2, 1])
    //   s[j] = j  for j ≥ 1  (j odd integers 3, 5, …, 2j+1)
    let half_z = (z + 1) / 2;
    let mut s: Vec<u64> = (0u64..=(half_z as u64)).collect();

    // large[k] = (⌊x/k⌋ + 1) / 2
    //   = count of {2} ∪ {odd integers in [3, ⌊x/k⌋]}  for x/k ≥ 2
    let mut large: Vec<u64> = {
        let mut v = vec![0u64; z + 1];
        v[1..=z].par_iter_mut().enumerate().for_each(|(i, cell)| {
            *cell = (x64 / (i + 1) as u64 + 1) / 2;
        });
        v
    };

    let mut phi_val: u128 = 0;
    let mut phi_captured = y == 0; // skip when y=0 (called from lucy_hedgehog)

    // ── Main sieve loop — odd primes p = 3, 5, 7, … ────────────────────────
    let mut p = 3usize;
    while p <= z {
        // Prime detection: p is prime iff s[(p-1)/2] > s[(p-1)/2 - 1]
        let idx = p >> 1; // = (p-1)/2  (exact since p is odd)
        if s[idx] <= s[idx - 1] {
            p += 2;
            continue;
        }

        // Capture φ(x, a) just before the first prime > y is processed.
        // Invariant: after processing k total primes (p=2 implicit + odd primes so far),
        // large[1] = k−1 + φ(x, k).  Here k = a = π(y) ≥ 1.
        // Special case a = 0 (y < 2): prime 2 was pre-applied but not counted in π(y),
        // so the invariant doesn't hold; use φ(x, 0) = x directly.
        if !phi_captured && (p as u128) > y {
            let a = pi_at(y, x, z, &s, &large) as u128;
            phi_val = if a == 0 { x } else { large[1] as u128 - a + 1 };
            phi_captured = true;
        }

        // pcnt_large = π(p−1)  = 1 + (count of odd primes below p)
        //            = 1 + s[(p−3)/2]  = 1 + s[idx − 1]
        let pcnt_small = s[idx - 1]; // π_odd(p−1) = count of odd primes < p
        let pcnt_large = 1u64 + pcnt_small; // π(p−1)

        let p64 = p as u64;
        let p2 = p64 * p64; // p² as u64 — safe: p ≤ z, p² ≤ z² ≤ x ≤ u64::MAX
        let max_k = z.min((x64 / p2) as usize);

        // ── Update large[k] for k = 1 .. max_k ──────────────────────────────
        //
        // Phase 1: k ≤ ⌊z/p⌋  →  k·p ≤ z  →  use large[k·p]
        //
        //   large[k·p] has index k·p > k, so it hasn't been touched in this
        //   prime's pass yet — BUT two threads writing to large[kA] and reading
        //   large[kA] as some other thread's large[kB·p] would race.
        //   Fix: snapshot the read values first (sequential), then apply (parallel).
        let kp_thresh = (z / p).min(max_k);
        if kp_thresh >= PAR_MIN {
            // Sequential snapshot: read large[k*p] before any writes happen.
            let deltas: Vec<u64> = (1..=kp_thresh).map(|k| large[k * p] - pcnt_large).collect();
            // Parallel apply: each large[k] is written exactly once.
            large[1..=kp_thresh]
                .par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(cell, &d)| *cell -= d);
        } else {
            for k in 1..=kp_thresh {
                large[k] -= large[k * p] - pcnt_large;
            }
        }

        // Phase 2: k > ⌊z/p⌋  →  k·p > z  →  use s[(q−1)>>1] where q = ⌊x/(k·p)⌋
        let x_div_p = x64 / p64; // ⌊x/p⌋, precomputed once per prime
        let ph2_start = kp_thresh + 1;
        phase2_large_update(&mut large, &s, ph2_start, max_k, x_div_p, pcnt_large);

        // ── Update s[j] for odd i = 2j+1 in [p², z] ─────────────────────────
        //
        // Only odd values need updating because s stores odd-indexed counts only.
        // Iterating j downward ensures s[(i/p − 1)>>1] (at a lower index j/p_adj)
        // has NOT been touched in this prime's pass — pre-p value guaranteed.
        if p2 <= z as u64 {
            let j_min = (p * p - 1) / 2; // i = p² is odd → j = (p²−1)/2
            let j_max = (z - 1) / 2; // largest j with 2j+1 ≤ z (odd)
            s_update_par(&mut s, j_min, j_max, p, pcnt_small);
        }

        p += 2;
    }

    // Edge case: all primes ≤ z are also ≤ y (very small x) — capture at end.
    if !phi_captured {
        let a = pi_at(y, x, z, &s, &large) as u128;
        phi_val = if a == 0 { x } else { large[1] as u128 - a + 1 };
    }

    (s, large, phi_val)
}

/// Convenience wrapper — runs the sieve and discards the φ value.
pub fn lucy_hedgehog(x: u128) -> (Vec<u64>, Vec<u64>) {
    let (s, large, _) = lucy_hedgehog_with_phi(x, 0);
    (s, large)
}

/// Lucy-Hedgehog sieve limited to capturing φ(x, ⌊x^(1/3)⌋).
///
/// Equivalent to `lucy_hedgehog_with_phi(x, icbrt(x))`.  The full Lucy sieve
/// runs to z = √x (required for correctness — see note below), but the loop
/// exits early as soon as φ has been captured at the first prime > y = ∛x.
///
/// Returns `(s, large, phi_x_a)` where:
/// - `s[j]`     = π(2j+1) − 1  for 2j+1 ≤ √x   (fully sieved up to y by the time
///                                                  we return, partially beyond)
/// - `large[k]` = π(⌊x/k⌋)    for 1 ≤ k ≤ √x   (needed internally)
/// - `phi_x_a`  = φ(x, a)      where a = π(y)
///
/// # Why z must equal √x (not y)
/// In phase 2 of the large[] update for prime p ≤ y, the code reads
/// `s[(q−1)>>1]` where q = ⌊x/(k·p)⌋.  When the loop uses kp_thresh = y/p
/// (i.e., large[] has only y entries), k > y/p gives q < x/y = x^(2/3),
/// which can exceed z_s = √x — out of bounds.  The correct split threshold is
/// kp_thresh = z/p = √x/p, which ensures q < √x whenever k > √x/p.
///
/// **Precondition**: `x ≥ 2`.
pub fn lucy_hedgehog_cbrt(x: u128) -> (Vec<u64>, Vec<u64>, u128) {
    let y = icbrt(x);
    lucy_hedgehog_with_phi(x, y)
}

/// Lucy-Hedgehog sieve that stops as soon as φ(x, π(∛x)) is captured.
///
/// Unlike [`lucy_hedgehog_cbrt`] (which runs the full loop to √x), this
/// function breaks out of the main loop immediately after φ is captured —
/// i.e., after processing all primes ≤ ∛x.  For x = 10¹² that means ~1 229
/// iterations instead of ~78 498.
///
/// # What is still valid at early-exit time
/// The capture point is the iteration where the first prime p > y = ∛x is
/// detected.  At that point every prime ≤ y has been processed, so:
/// - `s[j]` for 2j+1 ≤ y  — fully sieved; correct for `pi_at(y, …)` and
///   `extract_primes(s, y)`.
/// - `large[1]`            — fully updated; `phi_x_a` is exact.
///
/// `s[j]` for 2j+1 > y and `large[k]` for k > 1 are **not** finalized and
/// must not be used.
///
/// # Returns
/// `(phi_x_a, a, seed_primes)` where:
/// - `phi_x_a`    = φ(x, a)
/// - `a`          = π(∛x)
/// - `seed_primes`= primes ≤ ∛x in ascending order
///
/// **Precondition**: `x ≥ 2`.
pub fn lucy_phi_early_stop(x: u128) -> (u128, usize, Vec<u64>) {
    let y = icbrt(x);
    let z = isqrt(x) as usize;

    debug_assert!(x <= u64::MAX as u128, "lucy_phi_early_stop requires x ≤ u64::MAX");
    let x64 = x as u64;

    // Allocate full-size arrays (size z = √x) — required for correct phase-1
    // updates of large[] during the primes ≤ y pass.
    let half_z = (z + 1) / 2;
    let mut s: Vec<u64> = (0u64..=(half_z as u64)).collect();

    let mut large: Vec<u64> = {
        let mut v = vec![0u64; z + 1];
        v[1..=z].par_iter_mut().enumerate().for_each(|(i, cell)| {
            *cell = (x64 / (i + 1) as u64 + 1) / 2;
        });
        v
    };

    let mut phi_val: u128 = 0;

    let mut p = 3usize;
    while p <= z {
        let idx = p >> 1;
        if s[idx] <= s[idx - 1] {
            p += 2;
            continue;
        }

        // Capture φ(x, a) at the first prime p > y, then exit.
        if (p as u128) > y {
            let a128 = pi_at(y, x, z, &s, &large) as u128;
            phi_val = if a128 == 0 { x } else { large[1] as u128 - a128 + 1 };
            break;
        }

        let pcnt_small = s[idx - 1];
        let pcnt_large = 1u64 + pcnt_small;
        let p64 = p as u64;
        let p2 = p64 * p64;
        let max_k = z.min((x64 / p2) as usize);

        let kp_thresh = (z / p).min(max_k);
        if kp_thresh >= PAR_MIN {
            let deltas: Vec<u64> = (1..=kp_thresh).map(|k| large[k * p] - pcnt_large).collect();
            large[1..=kp_thresh]
                .par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(cell, &d)| *cell -= d);
        } else {
            for k in 1..=kp_thresh {
                large[k] -= large[k * p] - pcnt_large;
            }
        }

        let x_div_p = x64 / p64;
        let ph2_start = kp_thresh + 1;
        phase2_large_update(&mut large, &s, ph2_start, max_k, x_div_p, pcnt_large);

        if p2 <= z as u64 {
            let j_min = (p * p - 1) / 2;
            let j_max = (z - 1) / 2;
            s_update_par(&mut s, j_min, j_max, p, pcnt_small);
        }

        p += 2;
    }

    // Edge case: all primes ≤ z are also ≤ y.
    if phi_val == 0 {
        let a128 = pi_at(y, x, z, &s, &large) as u128;
        phi_val = if a128 == 0 { x } else { large[1] as u128 - a128 + 1 };
    }

    // Extract outputs that are valid at early-exit time.
    let a = pi_at(y, x, z, &s, &large) as usize;
    let seed_primes = extract_primes(&s, y as usize);
    (phi_val, a, seed_primes)
}

/// Diagnostic variant: same as [`lucy_phi_early_stop`] but prints a breakdown of
/// time spent in Phase 1 (large stride), Phase 2 (large sequential), Phase 3 (s-update).
pub fn lucy_phi_early_stop_profiled(x: u128) -> (u128, usize, Vec<u64>) {
    use std::time::{Duration, Instant};
    let y = icbrt(x);
    let z = isqrt(x) as usize;
    debug_assert!(x <= u64::MAX as u128);
    let x64 = x as u64;

    let half_z = (z + 1) / 2;
    let mut s: Vec<u64> = (0u64..=(half_z as u64)).collect();
    let mut large: Vec<u64> = {
        let mut v = vec![0u64; z + 1];
        v[1..=z].par_iter_mut().enumerate().for_each(|(i, cell)| {
            *cell = (x64 / (i + 1) as u64 + 1) / 2;
        });
        v
    };

    let mut phi_val: u128 = 0;
    let (mut t1, mut t2, mut t3) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut n_p1, mut n_p2, mut n_p3) = (0u64, 0u64, 0u64);

    let mut p = 3usize;
    while p <= z {
        let idx = p >> 1;
        if s[idx] <= s[idx - 1] { p += 2; continue; }
        if (p as u128) > y {
            let a128 = pi_at(y, x, z, &s, &large) as u128;
            phi_val = if a128 == 0 { x } else { large[1] as u128 - a128 + 1 };
            break;
        }
        let pcnt_small = s[idx - 1];
        let pcnt_large = 1u64 + pcnt_small;
        let p64 = p as u64;
        let p2 = p64 * p64;
        let max_k = z.min((x64 / p2) as usize);
        let kp_thresh = (z / p).min(max_k);

        // Phase 1
        let t = Instant::now();
        if kp_thresh >= PAR_MIN {
            let deltas: Vec<u64> = (1..=kp_thresh).map(|k| large[k * p] - pcnt_large).collect();
            large[1..=kp_thresh].par_iter_mut().zip(deltas.par_iter()).for_each(|(cell, &d)| *cell -= d);
        } else {
            for k in 1..=kp_thresh { large[k] -= large[k * p] - pcnt_large; }
        }
        t1 += t.elapsed(); n_p1 += kp_thresh as u64;

        // Phase 2 (q-jump)
        let x_div_p = x64 / p64;
        let ph2_start = kp_thresh + 1;
        let t = Instant::now();
        if ph2_start <= max_k { n_p2 += (max_k - ph2_start + 1) as u64; }
        phase2_large_update(&mut large, &s, ph2_start, max_k, x_div_p, pcnt_large);
        t2 += t.elapsed();

        // Phase 3
        let t = Instant::now();
        if p2 <= z as u64 {
            let j_min = (p * p - 1) / 2;
            let j_max = (z - 1) / 2;
            s_update_par(&mut s, j_min, j_max, p, pcnt_small);
            n_p3 += (j_max - j_min + 1) as u64;
        }
        t3 += t.elapsed();

        p += 2;
    }
    if phi_val == 0 {
        let a128 = pi_at(y, x, z, &s, &large) as u128;
        phi_val = if a128 == 0 { x } else { large[1] as u128 - a128 + 1 };
    }

    let total = t1 + t2 + t3;
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    eprintln!(
        "Lucy phase breakdown  (x={x}):\n  Phase 1 (large stride)  {:7.1} ms  {:8.0}M ops\n  Phase 2 (large seqread) {:7.1} ms  {:8.0}M ops\n  Phase 3 (s-update)      {:7.1} ms  {:8.0}M ops\n  Total Lucy              {:7.1} ms",
        ms(t1), n_p1 as f64 / 1e6,
        ms(t2), n_p2 as f64 / 1e6,
        ms(t3), n_p3 as f64 / 1e6,
        ms(total)
    );

    let a = pi_at(y, x, z, &s, &large) as usize;
    let seed_primes = extract_primes(&s, y as usize);
    (phi_val, a, seed_primes)
}

/// Cache-blocked wave-parallel update of the s-array for a single Lucy prime step.
///
/// Computes `s[j] -= s[(2j+1)/p − 1) >> 1] − pcnt_small` for `j ∈ [j_min, j_max]`
/// using a two-level decomposition:
///
/// **Level 1 — natural waves** (correctness boundary):
/// Wave k covers `[wave_hi/p + 1, wave_hi]`.  For every j in a wave,
/// the read index `(q−1)/2 ≤ wave_hi/p = wave_lo − 1 < wave_lo`, so reads
/// always fall in the already-stable portion `s[..wave_lo]`.
///
/// **Level 2 — cache blocks within each wave** (performance):
/// Each wave is further split into chunks of `S_BLOCK` entries so that the
/// write window (≤ 4 MB) + the corresponding read window (≤ 4/p MB) fit
/// together in the L3 cache (≤ 5.3 MB for p = 3, the worst case).
/// All chunks inside a wave inherit the correctness guarantee:
/// for j ≤ chunk_hi ≤ wave_hi → read index (q−1)/2 ≤ wave_hi/p < wave_lo ≤ chunk_lo
/// → `split_at_mut(chunk_lo)` yields disjoint borrows → race-free parallel iteration.
///
/// Falls back to sequential when a chunk is smaller than `S_PAR_MIN`.
#[inline]
fn s_update_par(s: &mut Vec<u64>, j_min: usize, j_max: usize, p: usize, pcnt_small: u64) {
    let mut wave_hi = j_max;
    loop {
        // ── Natural wave boundary ────────────────────────────────────────────
        let wave_lo_nat = wave_hi / p + 1; // reads for this wave end at wave_lo_nat−1
        let wave_lo = wave_lo_nat.max(j_min);
        if wave_lo > wave_hi {
            break;
        }

        // ── Cache-block sweep within the wave (chunk_hi → wave_lo) ──────────
        let mut chunk_hi = wave_hi;
        loop {
            let chunk_lo = if chunk_hi >= wave_lo + S_BLOCK {
                chunk_hi + 1 - S_BLOCK
            } else {
                wave_lo
            };

            // Invariant: for j ∈ [chunk_lo, chunk_hi],
            //   (q−1)/2 ≤ chunk_hi/p ≤ wave_hi/p = wave_lo_nat−1 < wave_lo ≤ chunk_lo
            // ⇒ reads land in s[..chunk_lo] which split_at_mut exposes as immutable.
            let chunk_len = chunk_hi - chunk_lo + 1;
            if chunk_len >= S_PAR_MIN {
                let (s_ro, s_rw) = s.split_at_mut(chunk_lo);
                s_rw[..chunk_len].par_iter_mut().enumerate().for_each(|(off, cell)| {
                    let j = chunk_lo + off;
                    let q = (2 * j + 1) / p;
                    *cell -= s_ro[(q - 1) >> 1] - pcnt_small;
                });
            } else {
                for j in chunk_lo..=chunk_hi {
                    let q = (2 * j + 1) / p;
                    s[j] -= s[(q - 1) >> 1] - pcnt_small;
                }
            }

            if chunk_lo == wave_lo {
                break;
            }
            chunk_hi = chunk_lo - 1;
        }

        if wave_lo_nat <= j_min {
            break;
        }
        wave_hi = wave_lo_nat - 1;
    }
}

/// Phase 2 of the Lucy large-array update.
///
/// For k in [ph2_start, max_k]:
///   `large[k] -= 1 + s[(floor(x_div_p/k) − 1) >> 1] − pcnt_large`
///
/// Each iteration requires one 64-bit division (~35 cycles). The iterations
/// are independent, so Rayon parallelises them when the range is large enough.
#[inline]
fn phase2_large_update(
    large: &mut [u64],
    s: &[u64],
    ph2_start: usize,
    max_k: usize,
    x_div_p: u64,
    pcnt_large: u64,
) {
    if ph2_start > max_k {
        return;
    }
    let len = max_k - ph2_start + 1;
    if len >= PAR_MIN {
        large[ph2_start..=max_k]
            .par_iter_mut()
            .enumerate()
            .for_each(|(off, cell)| {
                let k = ph2_start + off;
                let q = (x_div_p / k as u64) as usize;
                *cell -= 1 + s[(q - 1) >> 1] - pcnt_large;
            });
    } else {
        for k in ph2_start..=max_k {
            let q = (x_div_p / k as u64) as usize;
            large[k] -= 1 + s[(q - 1) >> 1] - pcnt_large;
        }
    }
}

/// Minimum chunk size to justify spawning Rayon tasks for the s-array update.
const S_PAR_MIN: usize = 8192;

/// Cache block size for s-array updates (entries, not bytes).
/// 512 K entries × 8 B = 4 MB write window.
/// For p = 3 (worst case): 4 MB write + 4/3 MB read = 5.3 MB < 8 MB L3.
const S_BLOCK: usize = 1 << 19; // 524 288 entries

/// Builds a plain Sieve of Eratosthenes up to `limit`.
///
/// Returns `(pi, primes)` where:
/// - `pi[n]` = π(n) for 0 ≤ n ≤ limit  (u32 — fits all practical limits)
/// - `primes` = all primes ≤ limit in ascending order
///
/// Intended for the Meissel-φ backend where only a sieve to ∛x is needed.
pub fn sieve_to(limit: u64) -> (Vec<u32>, Vec<u64>) {
    if limit < 2 {
        return (vec![0u32; (limit + 1) as usize], vec![]);
    }
    let n = limit as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2usize;
    while p * p <= n {
        if is_prime[p] {
            let mut m = p * p;
            while m <= n {
                is_prime[m] = false;
                m += p;
            }
        }
        p += 1;
    }

    let mut pi = vec![0u32; n + 1];
    let mut primes = Vec::new();
    let mut count = 0u32;
    for i in 2..=n {
        if is_prime[i] {
            count += 1;
            primes.push(i as u64);
        }
        pi[i] = count;
    }
    (pi, primes)
}

/// Extracts primes ≤ z from the half-size sieve `s`.
///
/// Prime 2 is always included (it was pre-applied at init time).
/// An odd integer 2j+1 ≥ 3 is prime iff `s[j] > s[j−1]`.
pub fn extract_primes(s: &[u64], z: usize) -> Vec<u64> {
    let mut primes = if z >= 2 { vec![2u64] } else { vec![] };
    let j_max = z.saturating_sub(1) / 2;
    for j in 1..=j_max {
        if s[j] > s[j - 1] {
            primes.push((2 * j + 1) as u64);
        }
    }
    primes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::isqrt;

    fn pi(x: u128) -> u64 {
        let z = isqrt(x) as usize;
        let (s, large) = lucy_hedgehog(x);
        pi_at(x, x, z, &s, &large)
    }

    #[test]
    fn test_pi_small_values() {
        assert_eq!(pi(2), 1);
        assert_eq!(pi(3), 2);
        assert_eq!(pi(4), 2);
        assert_eq!(pi(5), 3);
        assert_eq!(pi(10), 4);
        assert_eq!(pi(100), 25);
        assert_eq!(pi(1_000), 168);
    }

    #[test]
    fn test_extract_primes() {
        let (s, _) = lucy_hedgehog(100);
        let primes = extract_primes(&s, 10);
        assert_eq!(primes, vec![2, 3, 5, 7]);
    }

    #[test]
    fn test_pi_at_intermediate() {
        let x = 100_u128;
        let z = isqrt(x) as usize;
        let (s, large) = lucy_hedgehog(x);
        // π(20) = 8
        assert_eq!(pi_at(20, x, z, &s, &large), 8);
        // π(14) = 6
        assert_eq!(pi_at(14, x, z, &s, &large), 6);
        // π(7) = 4
        assert_eq!(pi_at(7, x, z, &s, &large), 4);
    }
}
