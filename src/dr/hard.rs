/// Per-phase nanosecond accumulators produced by [`s2_hard_sieve_par`] when
/// called from the profiling entry point. Summed across all Rayon bands, so
/// values reflect *CPU time* (not wall time) and should add up to roughly
/// `num_threads × wall_time_of_s2_hard`.
#[derive(Default, Clone, Debug)]
pub struct HardProfile {
    /// Pass 1: counted cross-off for bi ∈ [0, b_limit) maintaining `delta`.
    pub p1_counted_ns: u64,
    /// Pass 1: plain cross-off for bi ∈ [b_limit, b_ext) + bulk cross-off.
    pub p1_plain_bulk_ns: u64,
    /// Pass 1: sieve.fill + total_count calls + seed_in_seg counting.
    pub p1_fill_count_ns: u64,
    /// Pass 2: sieve.fill_presieved_7_11 / sieve.fill + total_count.
    pub p2_fill_ns: u64,
    /// Pass 2: bi ∈ [0, b_limit) — leaves + popcount cursor + counted cross-off.
    pub p2_bi_main_ns: u64,
    /// Pass 2: bi ∈ [b_limit, b_ext) plain cross-off + bulk cross-off.
    pub p2_cross_rest_ns: u64,
    /// Pass 2: final_count + ext-easy leaves + P2 queries + wheel advance.
    pub p2_tail_ns: u64,
}

fn enumerate_hard_leaves(
    x: u128,
    pb: u64,
    max_m: u64,
    primes: &[u64],
    start: usize,  // index in primes of first prime > pb
    m: u64,
    mu: i8,
    out: &mut Vec<(u64, i8)>,
) {
    for i in start..primes.len() {
        let p = primes[i];
        let nm = match (m as u128).checked_mul(p as u128) {
            Some(v) if v <= max_m as u128 => v as u64,
            _ => break,
        };
        let n = (x / (pb as u128 * nm as u128)) as u64;
        out.push((n, -mu)); // μ(nm) = -μ(m)
        enumerate_hard_leaves(x, pb, max_m, primes, i + 1, nm, -mu, out);
    }
}

/// Computes S2_hard = −Σ_{b=c+1}^{b_max} Σ_{m: squarefree, lpf(m)>p_b, p_b·m>y}
///                      μ(m) · φ(x/(p_b·m), b−1)
///
/// via a running sieve vector.  Together with [`crate::phi::s1_ordinary`], this
/// gives φ(x, a) = S1 + S2_hard without any Legendre recursion.
///
/// # Algorithm
/// Sweeps n ∈ [lo_start, z] in SEG-sized windows (ascending).
/// For each window:
/// 1. Fill sieve with p_1,…,p_c pre-sieved (= 2,3,5,7,11).
/// 2. For each hard prime p_b (b = c+1 … b_max):
///    - `phi_vec[b]` = φ(lo−1, b−1) (running accumulator).
///    - For leaves with n = x/(p_b·m) in [lo, lo+SEG): sum −= μ(m)·(phi_vec[b] + popcount(lo,n)).
///    - phi_vec[b] += total set bits in window.
///    - Cross off p_b from sieve (so next b sees p_1,…,p_b crossed off).
///
/// # Parameters
/// - `x`      : the prime-counting argument
/// - `y`      : ∛x (seed-prime limit)
/// - `z`      : x/y = x^{2/3} (sweep upper bound)
/// - `c`      : index of the last "tiny" prime (phi_small_a uses first c primes)
/// - `b_max`  : number of hard primes = π(√y) = π(x^{1/6})
/// - `primes` : all primes ≤ y in ascending order (length = a = π(y))
/// Computes φ(x, a) − φ(x, c) = S2 contribution for b = c+1..=a.
///
/// Covers BOTH hard leaves (b ≤ b_max, squarefree m with lpf(m) > p_b and
/// p_b·m > y) and easy leaves (b > b_max, prime pairs (p_b, p_l) with
/// p_l > p_b and p_b·p_l > y).
///
/// # Parameters
/// - `x`     : argument to π(x)
/// - `y`     : ∛x (cube root, upper bound for "small" primes)
/// - `z`     : x/y ≈ x^{2/3} (upper bound for quotient range)
/// - `c`     : number of tiny primes used by φ_tiny (leaves use φ(·, b-1))
/// - `b_max` : π(√y) = π(x^{1/6}) — hard/easy cutoff index (0-based count)
/// - `a`     : π(y) = total number of primes ≤ y (= len of `primes`)
/// - `primes`: all primes ≤ y in ascending order
///
/// # Returns
/// Σ_{b=c+1}^{a} Σ_{squarefree m, lpf(m)>p_b, p_b·m>y} μ(m)·φ(⌊x/(p_b·m)⌋, b-1)
///
/// Sign convention: `sum -= mu * phi_n`, so hard-leaf terms (μ alternating)
/// and easy-leaf terms (μ(p_l)=−1 → sum += phi_n) are handled uniformly.

/// Parallelised version of [`s2_hard_sieve`] using Rayon.
///
/// # Algorithm
///
/// The sequential version has a dependency on `phi_vec[bi]` that accumulates
/// across segments: each segment adds `running_total` to `phi_vec[bi]`.  We
/// break this with a two-pass approach identical to [`s2_popcount_par`]:
///
/// **Pass 1** (parallel per band): each band independently re-derives its sieve
/// state and sweeps its segments, accumulating `phi_delta[t][bi]` = total
/// increment to `phi_vec[bi]` within band `t`.  No leaf processing in Pass 1.
///
/// **Sequential scan**: prefix-sum `phi_delta` to get `phi_band_inits[t][bi]`,
/// the correct starting value of `phi_vec[bi]` at the beginning of band `t`.
///
/// **Pass 2** (parallel per band): each band starts with the correct `phi_vec`,
/// re-derives sieve state, and processes leaves exactly as the serial version.
///
/// # Expected speedup
/// Both passes are fully parallel with no synchronisation.  Expected ≈ T×
/// speedup on T threads (bottleneck is `cross_off_count` which is pure ALU).
/// Computes S₂_hard + P2 in a single parallel sweep.
///
/// # Why P2 is free here
///
/// After the bi-loop for each segment `[lo, lo+SEG)`, the sieve has had
/// multiples of every prime `p ≤ p_{a−1} ≈ ∛x` removed.  For any `n` in
/// `[lo_start, z] = [∛x, x^{2/3}]`, a composite whose smallest prime factor
/// exceeds `p_{a−1}` would need at least two such factors, making it larger
/// than `p_{a−1}² > x^{2/3} = z`.  Therefore **every surviving bit is a
/// prime**, and calling `fill_prefix_counts` on the post-loop sieve gives an
/// exact π-table over the current window — at the cost of O(SEG/64) extra
/// ops per segment (negligible vs. the bi-loop).
///
/// P2 queries `q_k = ⌊x/p_k⌋` for `p_k ∈ (∛x, √x]` all land in `[√x, z]`,
/// a sub-range already covered by the S₂_hard sweep, so no extra sieve pass
/// is needed.
///
/// # Returns
/// `(s2_hard, p2)` — the two values used in `π(x) = S1 + S2_hard + a − 1 − P2`.
pub fn s2_hard_sieve_par(
    x: u128,
    y: u64,
    z: u64,
    c: usize,
    b_max: usize,
    a: usize,
    primes: &[u64],
    s2_primes: &[u64], // primes in (∛x, √x] for P2 = Σ(π(x/p) − (π(p)−1))
) -> (i128, u128, HardProfile) {
    use crate::segment::{advance_wheel_primes, MonoCount, WheelPrimeData, WheelSieve30, W30_SEG, W30_WORDS, wheel30_next_k};
    use rayon::prelude::*;
    use std::time::Instant;

    // Phi-style wheel-sieve init: primes {2,3,5} are absorbed into the wheel;
    // only primes[3..c] = {7, 11, …} need explicit crossing-off.
    // m is the first coprime-to-30 multiple of p that is ≥ lo (phi-style: start
    // from p itself at lo = 0, i.e. k₀ = 1 so that p is crossed off).
    let phi_tiny_state = |lo: u64| -> Vec<(u64, u64)> {
        primes[3..c]
            .iter()
            .map(|&p| {
                let k0 = if lo == 0 { 1u64 } else { (lo + p - 1) / p };
                let k1 = wheel30_next_k(k0);
                (p, k1 * p)
            })
            .collect()
    };

    if a <= c || z == 0 {
        return (0, 0, HardProfile::default());
    }

    let n_hard = b_max.saturating_sub(c);
    let n_all  = a - c;
    let n_easy = n_all.saturating_sub(n_hard);

    // ── Leaf-case-B threshold ────────────────────────────────────────────────
    // For bi >= b_ext: primes[c+bi] > x^{1/4}, so every easy leaf n = x/(pb*pl)
    // satisfies p_{b-1} ≤ n < p_{b-1}², enabling φ(n, b-1) = π(n) − (b−2).
    // This lets us skip phi_vec maintenance for the bulk of the primes and use
    // the final prime-sieve (same as P2) for the direct π(n) lookup instead.
    let b_ext = {
        let x4: u64 = (x as f64).sqrt().sqrt() as u64 + 2; // x^{1/4} + safety margin
        primes[c..a].partition_point(|&p| p <= x4)
            .max(n_hard)   // must cover all hard leaves
            .min(n_all)    // clamp to valid range
    };
    // n_ext_easy: easy bi values below b_ext that still use phi_vec
    let n_ext_easy = b_ext.saturating_sub(n_hard);

    // ── Build leaf lists for HARD bi only ────────────────────────────────────
    // Easy leaves computed on-the-fly → zero O(a²) allocation.
    let mut hard_leaves: Vec<Vec<(u64, i8)>> = Vec::with_capacity(n_hard);
    for bi in 0..n_hard {
        let b  = bi + c + 1;
        let pb = primes[b - 1];
        let mut leaves: Vec<(u64, i8)> = Vec::new();
        enumerate_hard_leaves(x, pb, y, primes, b, 1u64, 1i8, &mut leaves);
        leaves.sort_unstable_by_key(|&(n, _)| n);
        hard_leaves.push(leaves);
    }

    // ── Compute lo_start ─────────────────────────────────────────────────────
    let n_min_hard: u64 = if n_hard > 0 && b_max > 0 {
        z.saturating_div(primes[b_max - 1])
    } else { z };
    let n_min_easy: u64 = if n_easy > 0 && a >= 2 {
        let pa = primes[a - 1] as u128;
        if pa * pa <= x { (x / (pa * pa)) as u64 } else { 0 }
    } else { z };
    let n_min   = n_min_hard.min(n_min_easy);
    let lo_start = (n_min / W30_SEG as u64) * W30_SEG as u64;

    // ── Band layout ──────────────────────────────────────────────────────────
    let num_segs      = ((z - lo_start) / W30_SEG as u64 + 1) as usize;
    let num_bands     = rayon::current_num_threads().min(num_segs).max(1);
    let segs_per_band = (num_segs + num_bands - 1) / num_bands;

    // ── Initial phi_vec at lo_start ───────────────────────────────────────────
    // Only need b_ext entries: bi >= b_ext use the pi-formula, not phi_vec.
    let initial_phi_vec: Vec<i64> = {
        let mut phi_vec = vec![0i64; b_ext];
        if lo_start > 0 {
            let n_init  = lo_start as usize;
            let n_words = (n_init + 63) / 64;
            let mut bits: Vec<u64> = vec![!0u64; n_words];
            bits[0] &= !1u64;
            // Mask out bits beyond n_init in the last word: the sieve loop
            // stops at m < n_init so bits[n_init..n_words*64] are never cleared
            // but they all start at 1 and would be counted erroneously.
            let overhang = n_init % 64;
            if overhang != 0 {
                bits[n_words - 1] &= (1u64 << overhang) - 1;
            }
            for k in 0..c {
                let p = primes[k] as usize;
                let mut m = p;
                while m < n_init { bits[m / 64] &= !(1u64 << (m % 64)); m += p; }
            }
            if b_ext > 0 {
                // Single popcount for phi_vec[0]; subsequent bi update a running
                // counter by counting only cleared bits (was-set check) instead
                // of re-popcounting the whole bitset each time.
                let mut count: i64 = bits.iter().map(|w| w.count_ones() as i64).sum();
                phi_vec[0] = count;
                for bi in 0..(b_ext - 1) {
                    let pk = primes[c + bi] as usize;
                    let mut m = pk;
                    while m < n_init {
                        let w = m / 64;
                        let mask = 1u64 << (m % 64);
                        if bits[w] & mask != 0 {
                            bits[w] &= !mask;
                            count -= 1;
                        }
                        m += pk;
                    }
                    phi_vec[bi + 1] = count;
                }
            }
        }
        phi_vec
    };

    // ── P2 setup ─────────────────────────────────────────────────────────────
    // π(lo_start − 1) = seed primes strictly below lo_start (s2_primes > y ≥ lo_start).
    let initial_p2_offset: i64 = primes.partition_point(|&p| p < lo_start) as i64;

    // Per-band P2 query ranges: s2_primes sorted ascending (p↑ → q=x/p ↓).
    // Band [band_lo, band_hi) captures queries with q_k ∈ [band_lo, band_hi).
    //   q_k ≥ band_lo  ↔  p_k ≤ x/band_lo  ↔  index < end
    //   q_k < band_hi  ↔  p_k > x/band_hi   ↔  index ≥ start
    let p2_ranges: Vec<(usize, usize)> = (0..num_bands)
        .map(|t| {
            let band_lo = lo_start + (t * segs_per_band) as u64 * W30_SEG as u64;
            let band_hi = (lo_start + ((t + 1) * segs_per_band) as u64 * W30_SEG as u64)
                          .min(z + W30_SEG as u64);
            let end   = if band_lo == 0 { s2_primes.len() } else {
                s2_primes.partition_point(|&p| (x / p as u128) as u64 >= band_lo)
            };
            let start = s2_primes.partition_point(|&p| (x / p as u128) as u64 >= band_hi);
            (start, end)
        })
        .collect();

    // ── Per-prime precomputed wheel data ─────────────────────────────────────────
    // Primes[c..a] are crossed off in every sieve window.  Computing w30res_p,
    // bit_seq, gap_m, delta_group once here avoids repeating it inside every window
    // iteration (critical at large x where n_all ~ 10 000 primes × 20 000 windows).
    let pb_data: Vec<WheelPrimeData> = primes[c..a]
        .iter()
        .map(|&p| WheelPrimeData::new(p))
        .collect();

    // ── Per-prime leaf-active cutoff: lo*(bi) = ⌊x / primes[c+bi]²⌋ ──────────
    // For lo > lo*(bi), p[c+bi] > √(x/lo), so x/(p[c+bi]*m) < lo for all valid m.
    // No leaf for bi appears in window [lo, lo+W30_SEG) or any future window.
    // Stored for bi in 0..b_ext; values are DESCENDING (larger bi → smaller cutoff)
    // because primes[c+bi] is ascending.
    let leaf_cutoff_lo: Vec<u64> = (0..b_ext)
        .map(|bi| {
            let p = primes[c + bi] as u128;
            (x / (p * p)).min(u64::MAX as u128) as u64
        })
        .collect();

    // ── Pass 1 (parallel): phi_delta + p2 prime count per band ───────────────
    // After the full bi-loop, running_total = prime count in [lo, lo+SEG).
    // Accumulate this per band so we can compute per-band π(band_lo − 1) offsets.
    let pass1: Vec<(Vec<i64>, i64, [u64; 3])> = (0..num_bands)
        .into_par_iter()
        .map(|t| {
            let mut ns_counted = 0u64;
            let mut ns_plain   = 0u64;
            let mut ns_fillc   = 0u64;
            // delta only needs b_ext entries: bi >= b_ext use pi-formula, not phi_vec.
            let mut delta    = vec![0i64; b_ext];
            let mut p2_count = 0i64;
            let band_lo      = lo_start + (t * segs_per_band) as u64 * W30_SEG as u64;
            if band_lo > z { return (delta, p2_count, [0u64; 3]); }
            let band_hi      = (lo_start + ((t + 1) * segs_per_band) as u64 * W30_SEG as u64)
                               .min(z + W30_SEG as u64);

            let mut tiny_state = phi_tiny_state(band_lo);
            let mut sieve      = WheelSieve30::new();
            let mut lo         = band_lo;
            // Bucket-sieve: only iterate active bulk primes (p² ≤ lo+W30_SEG).
            // Primes are sorted ascending, so bulk_active_end only ever increases.
            let mut bulk_active_end = {
                let init_hi = band_lo + W30_SEG as u64;
                let mut end = b_ext;
                while end < n_all {
                    let p = primes[c + end] as u64;
                    if p * p > init_hi { break; }
                    end += 1;
                }
                end
            };
            // b_limit: max bi for which leaves are still possible (monotonically ↓).
            // Initialise to the count of bi with leaf_cutoff_lo[bi] >= band_lo.
            // leaf_cutoff_lo is descending, so scan from b_ext-1 downward.
            let mut b_limit = b_ext;
            while b_limit > 0 && band_lo > leaf_cutoff_lo[b_limit - 1] {
                b_limit -= 1;
            }

            while lo < band_hi && lo <= z {
                // Advance b_limit as lo crosses leaf cutoffs (monotonically decreasing).
                while b_limit > 0 && lo > leaf_cutoff_lo[b_limit - 1] {
                    b_limit -= 1;
                }
                let t_fc_a = Instant::now();
                if c == 5 {
                    // Fast path: precomputed {7, 11} template (skips ones-fill
                    // + two wheel-30 cross-off loops per segment).
                    sieve.fill_presieved_7_11(lo);
                } else {
                    sieve.fill(lo, &tiny_state);
                }
                if lo == 0 { sieve.set_bit_for_1(); }
                let mut running_total = sieve.total_count() as i64;
                ns_fillc += t_fc_a.elapsed().as_nanos() as u64;

                let t_cnt = Instant::now();
                // Counted cross-off for bi < b_limit: maintains running_total for delta.
                for bi in 0..b_limit {
                    delta[bi] += running_total;
                    running_total -= sieve.cross_off_count_pd(lo, primes[c + bi], &pb_data[bi]) as i64;
                }
                ns_counted += t_cnt.elapsed().as_nanos() as u64;

                let t_pl = Instant::now();
                // bi in b_limit..b_ext: still need sieve cross-off for the prime sieve
                // (P2 / ext-easy use it), but no leaves → skip delta tracking.
                for bi in b_limit..b_ext {
                    sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                }
                // Bulk cross-off: bucket optimization skips inactive primes (p² > hi).
                // Exception: lo < y → seed_in_seg assumes ALL b_ext..n_all are crossed off.
                let hi = lo + W30_SEG as u64;
                if lo < y {
                    for bi in b_ext..n_all {
                        sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                    }
                } else {
                    while bulk_active_end < n_all {
                        let p = primes[c + bulk_active_end] as u64;
                        if p * p > hi { break; }
                        bulk_active_end += 1;
                    }
                    for bi in b_ext..bulk_active_end {
                        sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                    }
                }
                ns_plain += t_pl.elapsed().as_nanos() as u64;

                let t_fc_b = Instant::now();
                // After all crossings, sieve = prime sieve over [lo, lo+W30_SEG).
                // Use total_count() for the true prime count (running_total only
                // reflects primes > p_{b_ext-1}, not > p_{a-1}).
                let final_count = sieve.total_count() as i64;
                let seed_in_seg: i64 = if lo < y {
                    let j1 = primes.partition_point(|&p| p < lo);
                    let j2 = primes.partition_point(|&p| p < lo + W30_SEG as u64);
                    (j2 - j1) as i64
                } else { 0 };
                p2_count += final_count - if lo == 0 { 1 } else { 0 } + seed_in_seg;
                let next_lo = lo + W30_SEG as u64;
                advance_wheel_primes(&mut tiny_state, next_lo);
                lo = next_lo;
                ns_fillc += t_fc_b.elapsed().as_nanos() as u64;
            }
            (delta, p2_count, [ns_counted, ns_plain, ns_fillc])
        })
        .collect();

    let mut p1_phase_sums = [0u64; 3];
    for (_, _, phases) in &pass1 {
        for i in 0..3 { p1_phase_sums[i] += phases[i]; }
    }
    let (phi_deltas, p2_prime_counts): (Vec<Vec<i64>>, Vec<i64>) =
        pass1.into_iter().map(|(d, p, _)| (d, p)).unzip();

    // ── Sequential prefix scan for phi ────────────────────────────────────────
    // Only b_ext entries per band: bi >= b_ext use pi-formula, not phi_vec.
    let mut phi_band_inits: Vec<Vec<i64>> = vec![vec![0i64; b_ext]; num_bands];
    phi_band_inits[0] = initial_phi_vec;
    for t in 1..num_bands {
        for bi in 0..b_ext {
            phi_band_inits[t][bi] = phi_band_inits[t - 1][bi] + phi_deltas[t - 1][bi];
        }
    }

    // ── Sequential prefix scan for P2 (π offsets per band) ───────────────────
    let mut p2_band_inits = vec![initial_p2_offset; num_bands];
    for t in 1..num_bands {
        p2_band_inits[t] = p2_band_inits[t - 1] + p2_prime_counts[t - 1];
    }

    // ── Hard band_ptrs via binary search ─────────────────────────────────────
    let hard_band_ptrs: Vec<Vec<usize>> = (0..num_bands)
        .map(|t| {
            let band_lo = lo_start + (t * segs_per_band) as u64 * W30_SEG as u64;
            (0..n_hard)
                .map(|bi| hard_leaves[bi].partition_point(|&(n, _)| n < band_lo))
                .collect()
        })
        .collect();

    // ── Pass 2 (parallel): S₂_hard leaves + P2 queries per band ─────────────
    let init_easy = |ei: usize, band_lo: u64| -> (usize, u64) {
        let bi = n_hard + ei;
        let b  = bi + c + 1;
        if b >= a { return (a, u64::MAX); }
        let pb = primes[b - 1];
        let pl_idx = if band_lo == 0 {
            if a > b { a - 1 } else { a }
        } else {
            let max_pl = (x / (pb as u128 * band_lo as u128)) as u64;
            let cnt = primes[b..a].partition_point(|&p| p <= max_pl);
            if cnt == 0 { a } else { b + cnt - 1 }
        };
        let next_n = if pl_idx < a {
            (x / (pb as u128 * primes[pl_idx] as u128)) as u64
        } else { u64::MAX };
        (pl_idx, next_n)
    };

    // far_easy_start: index into easy_ptrs/easy_next_n where ext-easy leaves begin
    let far_easy_start = n_ext_easy; // ei >= far_easy_start use pi-formula

    let band_results: Vec<(i128, u128, [u64; 4])> = (0..num_bands)
        .into_par_iter()
        .map(|t| {
            let mut p_fill_ns: u64 = 0;
            let mut p_bi_ns: u64   = 0;
            let mut p_rest_ns: u64 = 0;
            let mut p_tail_ns: u64 = 0;
            let band_lo = lo_start + (t * segs_per_band) as u64 * W30_SEG as u64;
            if band_lo > z { return (0i128, 0u128, [0u64; 4]); }
            let band_hi = (lo_start + ((t + 1) * segs_per_band) as u64 * W30_SEG as u64)
                          .min(z + W30_SEG as u64);

            // phi_vec is now b_ext entries (bi >= b_ext use pi-formula instead).
            let mut phi_vec   = phi_band_inits[t].clone();
            let mut hard_ptrs = hard_band_ptrs[t].clone();

            let (mut easy_ptrs, mut easy_next_n): (Vec<usize>, Vec<u64>) =
                (0..n_easy).map(|ei| init_easy(ei, band_lo)).unzip();

            // P2 state for this band.
            let (p2_start, p2_end) = p2_ranges[t];
            let mut p2_ptr    = p2_end;   // exclusive; counts down to p2_start
            let mut p2_offset = p2_band_inits[t]; // π(band_lo − 1)
            let mut p2_prefix = [0u32; W30_WORDS + 1];
            let mut p2_sum    = 0u128;

            let mut tiny_state = phi_tiny_state(band_lo);
            let mut sieve      = WheelSieve30::new();
            let mut mono       = MonoCount::new();
            let mut sum: i128  = 0;
            let mut lo         = band_lo;
            // Bucket-sieve: mirror of Pass 1 — only active bulk primes.
            let mut bulk_active_end = {
                let init_hi = band_lo + W30_SEG as u64;
                let mut end = b_ext;
                while end < n_all {
                    let p = primes[c + end] as u64;
                    if p * p > init_hi { break; }
                    end += 1;
                }
                end
            };
            // b_limit (Pass 2): same monotone cutoff as Pass 1.
            let mut b_limit = b_ext;
            while b_limit > 0 && band_lo > leaf_cutoff_lo[b_limit - 1] {
                b_limit -= 1;
            }

            while lo < band_hi && lo <= z {
                // Advance b_limit as lo crosses leaf cutoffs.
                while b_limit > 0 && lo > leaf_cutoff_lo[b_limit - 1] {
                    b_limit -= 1;
                }
                let t_fill = Instant::now();
                if c == 5 {
                    sieve.fill_presieved_7_11(lo);
                } else {
                    sieve.fill(lo, &tiny_state);
                }
                if lo == 0 { sieve.set_bit_for_1(); }
                let mut running_total = sieve.total_count() as i64;
                let hi = lo + W30_SEG as u64;
                p_fill_ns += t_fill.elapsed().as_nanos() as u64;

                let t_bi = Instant::now();
                // ── Inner loop: bi in 0..b_limit (phi_vec maintained) ────────
                // bi ≥ b_limit have no remaining leaves; skip phi tracking.
                // For each bi with a leaf we replace fill_prefix_counts (full
                // W30_WORDS popcount sweep) by a monotonic scan: since leaves
                // of a given bi arrive in ascending n, we only popcount the
                // words between the previous and the current `n`.
                for bi in 0..b_limit {
                    let b  = bi + c + 1;
                    let pb = primes[b - 1];

                    let has_leaf = if bi < n_hard {
                        let ptr = hard_ptrs[bi];
                        ptr < hard_leaves[bi].len() && {
                            let (n, _) = hard_leaves[bi][ptr];
                            n >= lo && n < hi && n <= z
                        }
                    } else {
                        let ei = bi - n_hard; // ei < far_easy_start
                        easy_ptrs[ei] < a && {
                            let n = easy_next_n[ei];
                            n >= lo && n < hi
                        }
                    };

                    if has_leaf {
                        mono.reset();

                        if bi < n_hard {
                            let ptr = &mut hard_ptrs[bi];
                            while *ptr < hard_leaves[bi].len() {
                                let (n, mu) = hard_leaves[bi][*ptr];
                                if n >= hi || n > z { break; }
                                if n >= lo {
                                    let phi_n = phi_vec[bi] + sieve.count_primes_upto_int_m(&mut mono, n, lo) as i64;
                                    sum -= mu as i128 * phi_n as i128;
                                }
                                *ptr += 1;
                            }
                        } else {
                            let ei = bi - n_hard;
                            loop {
                                let pl_idx = easy_ptrs[ei];
                                if pl_idx >= a { break; }
                                let n = easy_next_n[ei];
                                if n >= hi { break; }
                                if n >= lo {
                                    let phi_n = phi_vec[bi] + sieve.count_primes_upto_int_m(&mut mono, n, lo) as i64;
                                    sum += phi_n as i128;
                                }
                                if pl_idx <= b {
                                    easy_ptrs[ei] = a;
                                    break;
                                }
                                let new_idx = pl_idx - 1;
                                easy_ptrs[ei]   = new_idx;
                                easy_next_n[ei] = (x / (pb as u128 * primes[new_idx] as u128)) as u64;
                            }
                        }
                    }

                    phi_vec[bi] += running_total;
                    running_total -= sieve.cross_off_count_pd(lo, pb, &pb_data[bi]) as i64;
                }
                p_bi_ns += t_bi.elapsed().as_nanos() as u64;

                let t_rest = Instant::now();
                // bi in b_limit..b_ext: still need sieve cross-off (for P2/ext-easy),
                // but no leaves → skip phi_vec tracking (count not needed).
                for bi in b_limit..b_ext {
                    sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                }

                // ── Bulk cross-off: bucket skips inactive primes (p² > hi) ──
                // Exception: lo < y → seed_in_seg assumes ALL b_ext..n_all crossed off.
                if lo < y {
                    for bi in b_ext..n_all {
                        sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                    }
                } else {
                    while bulk_active_end < n_all {
                        let p = primes[c + bulk_active_end] as u64;
                        if p * p > hi { break; }
                        bulk_active_end += 1;
                    }
                    for bi in b_ext..bulk_active_end {
                        sieve.cross_off_pd(lo, primes[c + bi], &pb_data[bi]);
                    }
                }

                p_rest_ns += t_rest.elapsed().as_nanos() as u64;

                let t_tail = Instant::now();
                // Sieve is now a prime sieve over [lo, lo+W30_SEG).
                // Compute seg_primes using total_count() (running_total only
                // reflects primes > p_{b_ext-1}, not > p_{a-1}).
                let final_count = sieve.total_count() as i64;
                let seed_in_seg: i64 = if lo < y {
                    let j1 = primes.partition_point(|&p| p < lo);
                    let j2 = primes.partition_point(|&p| p < lo + W30_SEG as u64);
                    (j2 - j1) as i64
                } else { 0 };
                let seg_primes = final_count - if lo == 0 { 1 } else { 0 } + seed_in_seg;

                // ── Ext-easy leaves + P2: both use the final prime sieve ──────
                // p2_prefix is filled lazily (only when the first leaf/query needs it).
                let mut p2_prefix_ready = false;
                let mut seed_below_lo   = 0usize;
                let adj_lo              = if lo == 0 { 1i64 } else { 0i64 };

                // Helper: ensures p2_prefix is filled at most once per window.
                // (Rust closures can't mutably borrow p2_prefix via closure; inline instead.)

                // ── Ext-easy leaves: φ(n, b-1) = π(n) − (b−2) ───────────────
                // Iterate over ALL ext-easy ei; each checks its own easy_next_n.
                for ei in far_easy_start..n_easy {
                    if easy_ptrs[ei] >= a { continue; }
                    if easy_next_n[ei] >= hi { continue; }
                    // At least one leaf for this ei lands in [lo, hi).
                    if !p2_prefix_ready {
                        sieve.fill_prefix_counts(&mut p2_prefix);
                        seed_below_lo = if lo < y {
                            primes.partition_point(|&p| p < lo)
                        } else { 0 };
                        p2_prefix_ready = true;
                    }
                    let bi = n_hard + ei;
                    let b  = bi + c + 1;
                    let pb = primes[b - 1];
                    loop {
                        let pl_idx = easy_ptrs[ei];
                        if pl_idx >= a { break; }
                        let n = easy_next_n[ei];
                        if n >= hi { break; }
                        if n >= lo {
                            let raw = sieve.count_primes_upto_int(&p2_prefix, n, lo) as i64;
                            let seed_in_query: i64 = if lo < y {
                                let j2 = primes.partition_point(|&p| p <= n);
                                (j2 - seed_below_lo) as i64
                            } else { 0 };
                            let pi_n  = p2_offset + raw - adj_lo + seed_in_query;
                            // φ(n, b-1) = π(n) − (b−2) holds only when n ≥ p_{b−1}.
                            // When n < p_{b−1} (π(n) < b−1), φ(n, b−1) = 1.
                            let phi_n = (pi_n - (b as i64 - 2)).max(1);
                            sum += phi_n as i128;
                        }
                        if pl_idx <= b {
                            easy_ptrs[ei] = a;
                            break;
                        }
                        let new_idx = pl_idx - 1;
                        easy_ptrs[ei]   = new_idx;
                        easy_next_n[ei] = (x / (pb as u128 * primes[new_idx] as u128)) as u64;
                    }
                }

                // ── P2 queries ────────────────────────────────────────────────
                if p2_ptr > p2_start {
                    let q_check = (x / s2_primes[p2_ptr - 1] as u128) as u64;
                    if q_check >= lo && q_check < hi {
                        if !p2_prefix_ready {
                            sieve.fill_prefix_counts(&mut p2_prefix);
                            seed_below_lo = if lo < y {
                                primes.partition_point(|&p| p < lo)
                            } else { 0 };
                        }
                        loop {
                            if p2_ptr <= p2_start { break; }
                            let j   = p2_ptr - 1;
                            let q_k = (x / s2_primes[j] as u128) as u64;
                            if q_k >= hi { break; }
                            if q_k < lo  { p2_ptr -= 1; continue; }
                            let raw = sieve.count_primes_upto_int(&p2_prefix, q_k, lo) as i64;
                            let seed_in_query: i64 = if lo < y {
                                let j2 = primes.partition_point(|&p| p <= q_k);
                                (j2 - seed_below_lo) as i64
                            } else { 0 };
                            let pi_qk = p2_offset + raw - adj_lo + seed_in_query;
                            let k = a + j;
                            p2_sum += (pi_qk - k as i64) as u128;
                            p2_ptr -= 1;
                        }
                    }
                }
                p2_offset += seg_primes;

                let next_lo = lo + W30_SEG as u64;
                advance_wheel_primes(&mut tiny_state, next_lo);
                lo = next_lo;
                p_tail_ns += t_tail.elapsed().as_nanos() as u64;
            }
            (sum, p2_sum, [p_fill_ns, p_bi_ns, p_rest_ns, p_tail_ns])
        })
        .collect();

    let s2_total: i128 = band_results.iter().map(|&(s, _, _)| s).sum();
    let p2_total: u128 = band_results.iter().map(|&(_, p, _)| p).sum();
    let mut phase_sums = [0u64; 4];
    for &(_, _, phases) in &band_results {
        for i in 0..4 { phase_sums[i] += phases[i]; }
    }
    let profile = HardProfile {
        p1_counted_ns: p1_phase_sums[0],
        p1_plain_bulk_ns: p1_phase_sums[1],
        p1_fill_count_ns: p1_phase_sums[2],
        p2_fill_ns: phase_sums[0],
        p2_bi_main_ns: phase_sums[1],
        p2_cross_rest_ns: phase_sums[2],
        p2_tail_ns: phase_sums[3],
    };
    (s2_total, p2_total, profile)
}
