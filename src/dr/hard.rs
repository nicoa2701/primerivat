/// Per-phase nanosecond accumulators produced by [`s2_hard_sieve_par`] when
/// called from the profiling entry point. Summed across all Rayon bands, so
/// values reflect *CPU time* (not wall time) and should add up to roughly
/// `num_threads × wall_time_of_s2_hard`.
#[derive(Default, Clone, Debug)]
pub struct HardProfile {
    /// Single-pass sweep: sieve.fill_presieved_7_11 / sieve.fill + total_count.
    pub sweep_fill_ns: u64,

    /// bi ∈ [0, b_limit) main loop (counted cross-off + leaf emit, bundled).
    /// Measured per segment, not per bi, to avoid Instant::now overhead on
    /// the ~3.4 B inner iterations at x=1e17 α=2. Use `n_bi_leaf_hits`
    /// together with `n_leaves_ext_emitted` to weigh the leaf vs xoff share.
    pub sweep_bi_main_ns: u64,

    /// Subset of `sweep_bi_main_ns` spent inside `if has_leaf { … }` (popcount
    /// + hard_ptrs / easy_ptrs walk + fold accumulators). Bracketed only when
    /// a leaf actually fires, so the timer adds ~2 × n_bi_leaf_hits Instant
    /// calls (~0.3 % overhead at x=1e17 α=2). The xoff share is derived as
    /// `sweep_bi_main_ns − sweep_bi_main_leaf_ns` at print time.
    pub sweep_bi_main_leaf_ns: u64,

    // ── bi ∈ [b_limit, n_all) cross-off (split) ───────────────────────────
    /// Plain cross-off bi ∈ [b_limit, b_ext).
    pub rest_plain_ns: u64,
    /// Bulk cross-off bi ∈ [b_ext, bulk_end) (includes bucket advance).
    pub rest_bulk_ns: u64,

    // ── Tail (split) ──────────────────────────────────────────────────────
    /// `fill_prefix_counts` + seed_below_lo bsearch (lazy per segment).
    pub tail_prefix_build_ns: u64,
    /// Ext-easy leaf emission loop.
    pub tail_ext_emit_ns: u64,
    /// P2 query emission loop.
    pub tail_p2_emit_ns: u64,
    /// `final_count` + `seed_in_seg` bsearch + `advance_wheel_primes`.
    pub tail_advance_ns: u64,

    /// Resolution pass: reconciling leaf records with `phi_band_inits` /
    /// `p2_band_inits` to produce the final `sum` / `p2_sum`.
    pub resolve_ns: u64,

    // ── Light counters (no-op past dev, useful to relate time → work). ────
    /// Number of (bi, segment) pairs that triggered leaf emission in the
    /// bi-main loop (hard + easy-with-phi-vec).
    pub n_bi_leaf_hits: u64,
    /// Total non-clamp ext-easy leaves actually emitted into the fold bucket.
    pub n_leaves_ext_emitted: u64,
    /// Clamp leaves pre-counted (skipped by the sweep).
    pub n_leaves_ext_clamped: u64,
    /// Segments that actually triggered a `fill_prefix_counts`.
    pub n_prefix_fills: u64,
    /// Sum across segments of (bulk_active_end - b_ext) at end of band — an
    /// average-ish measure of how many bulk primes are still being crossed
    /// off in the tail of each band.
    pub n_bulk_active_primes_sum: u64,
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

/// Parallelised version of [`s2_hard_sieve`] using Rayon — single-pass
/// deferred-leaf design.
///
/// # Algorithm
///
/// The serial dependency is `phi_vec[bi]` accumulating across segments. Rather
/// than the classic two-pass approach (Pass 1 = build delta, Pass 2 = process
/// leaves with phi_band_inits[t] seeded by a serial prefix scan), we run ONE
/// parallel sweep per band that BOTH accumulates delta AND emits deferred-leaf
/// records against a local snapshot of the in-band phi state:
///
/// **Single pass** (parallel per band): each band sweeps its segments, keeps a
/// band-local `delta[bi]` (= running contribution to `phi_vec[bi]` inside this
/// band) and a band-local `local_p2_offset`. Whenever a leaf / ext-easy / P2
/// query falls in the current segment, it is emitted as a record carrying a
/// snapshot of the local state plus the popcount taken against the current
/// sieve bits — never the final `phi_n` / `pi_n` values, which depend on the
/// still-unknown band init.
///
/// **Sequential scan**: prefix-sum band deltas and band prime counts to get
/// `phi_band_inits[t]` and `p2_band_inits[t]`.
///
/// **Resolution** (parallel per band): iterate the band's leaf records and
/// reconstruct `phi_n = phi_band_inits[t][bi] + local_phi + popcount` (and the
/// analogous formula for ext-easy / P2); accumulate the band's `sum` / `p2_sum`.
///
/// Compared to the prior two-pass design this skips a full sieve re-derivation
/// (fill + cross-off of every prime in `primes[c..a]`), which dominated CPU
/// time at x ≥ 10^14.
///
/// # Returns
/// `(s2_hard, p2)` — the two values used in `π(x) = S1 + S2_hard + a − 1 − P2`.
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
    use crate::segment::{advance_wheel_primes, MonoCount, WheelPrimeData, WheelSieve30, W30_IDX, W30_SEG, W30_WORDS, wheel30_next_k};
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
    // Oversubscribe bands vs. threads. At large x (lo_start = 0) every
    // ext-easy leaf funnels into band 0's low-n tail and caps Rayon scaling
    // at ~3× on 8 threads with 1 band per thread. Finer banding lets Rayon
    // rebalance across the worst-case slice: measured on i5-9300HF at
    //   x=1e15: 3.64s → 2.63s (-28%),
    //   x=1e16: 16.3s → 14.8s (-9%),
    //   x=1e17: 125s → 109s (-13%).
    // Returns diminish past ~16× threads; override via `-b` CLI flag,
    // `parameters::set_band_mult_override`, or `RIVAT3_BAND_MULT` env var.
    let num_bands     = (rayon::current_num_threads() * crate::parameters::band_mult())
        .min(num_segs)
        .max(1);

    // ── Bulk clamp-leaf count (Piste 3, computed early so band_bounds can
    // key the log-scale decision on it). An ext-easy leaf (p_b, p_l) produces
    // φ(n, b-1) = 1 iff n < p_{b-1}, i.e. p_l > x/(p_b·p_{b-1}). Skipping
    // these leaves is handled in the sweep via a pl_idx cap in init_easy.
    let far_easy_start = n_ext_easy; // ei >= far_easy_start use pi-formula
    let total_clamp_count: i64 = (far_easy_start..n_easy)
        .map(|ei| {
            let bi = n_hard + ei;
            let b = bi + c + 1;
            if b >= a || b < 2 { return 0i64; }
            let pb = primes[b - 1] as u128;
            let pbm1 = primes[b - 2] as u128;
            let pl_clamp_threshold = (x / (pb * pbm1)) as u64;
            let nonclamp_cnt = primes[b..a]
                .partition_point(|&p| p <= pl_clamp_threshold);
            let upper_cnt = if lo_start == 0 {
                a - b
            } else {
                let pl_upper = (x / (pb * lo_start as u128)) as u64;
                primes[b..a].partition_point(|&p| p <= pl_upper)
            };
            upper_cnt.saturating_sub(nonclamp_cnt) as i64
        })
        .sum();

    // ── Band boundaries ──────────────────────────────────────────────────────
    // Default: uniform partitioning, which distributes cross-off work evenly
    // and gives the best Rayon scaling whenever cross-off dominates.
    //
    // Opt-in to log-scale ONLY when the α=2 clamp path is active (measured by
    // total_clamp_count > 0). In that regime ext-easy leaves funnel into the
    // first √x of [lo_start, z] and uniform bands stall Rayon scaling. Log
    // bands pack narrow bins near low n and wider bins past √x. Outside of
    // α=2 the log layout makes the last band carry thousands of segments of
    // pure cross-off and REGRESSES wall (-30 % at 1e15 α=1 in testing).
    // Each bound is snapped to a W30_SEG multiple so the sieve stays valid.
    let use_log_scale = total_clamp_count > 0 && num_bands > 1;
    let w_seg = W30_SEG as u64;
    let hi_cap = ((z / w_seg) + 1) * w_seg; // first W30_SEG multiple > z
    let band_bounds: Vec<u64> = {
        let mut bounds = Vec::with_capacity(num_bands + 1);
        bounds.push(lo_start);
        if num_bands == 1 {
            bounds.push(hi_cap);
        } else if !use_log_scale {
            // Uniform (matches pre-Piste-1 behaviour exactly).
            let segs_per_band = (num_segs + num_bands - 1) / num_bands;
            for t in 1..num_bands {
                let b = lo_start + (t * segs_per_band) as u64 * w_seg;
                bounds.push(b.min(hi_cap));
            }
            bounds.push(hi_cap);
        } else {
            // Log-scale from w_seg to hi_cap across num_bands-1 internal
            // boundaries (band 0 spans [0, w_seg)).
            let log_lo = (w_seg as f64).ln();
            let log_hi = (hi_cap as f64).ln();
            let dlog   = (log_hi - log_lo) / (num_bands - 1) as f64;
            let mut prev = 0u64;
            for t in 1..num_bands {
                let target = (log_lo + dlog * (t - 1) as f64).exp() as u64;
                let aligned = (target / w_seg) * w_seg;
                let next_b  = aligned.max(prev + w_seg).min(hi_cap);
                bounds.push(next_b);
                prev = next_b;
            }
            bounds.push(hi_cap);
        }
        bounds
    };

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
            let band_lo = band_bounds[t];
            let band_hi = band_bounds[t + 1].min(z + W30_SEG as u64);
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

    // ── Deferred-leaf resolution: folded accumulators + rare stored records ──
    //
    // To avoid O(N_leaves) memory at large x, per-leaf contributions are folded
    // into band-local scalars / small arrays rather than stored verbatim:
    //
    //   LeafRec folding (safe, no non-linearity):
    //     phi_n = phi_init[bi] + local_phi + popcount
    //     sum  += sign * phi_n
    //   ⇒  leaf_partial += sign * (local_phi + popcount)        // band scalar
    //      bi_contrib[bi] += sign                                // band array[b_ext]
    //   final: sum += leaf_partial + Σ_bi bi_contrib[bi] * phi_init[bi]
    //
    //   P2 folding (safe):
    //     pi_qk = p2_init + local_p2 + raw - adj_lo + seed
    //     p2_sum += pi_qk - k
    //   ⇒  p2_partial += (local_p2 + raw - adj_lo + seed) - k    // band scalar
    //      p2_count   += 1                                        // band scalar
    //   final: p2_sum += p2_init * p2_count + p2_partial
    //
    //   Ext-easy HYBRID (non-linearity from `max(·, 1)`):
    //     Clamp leaves (n < p_{b-1}) are bulk-counted BEFORE the sweep and
    //     added directly to S2 at resolve time (see `total_clamp_count` below);
    //     the sweep skips them entirely by capping the pl iterator at
    //     pl ≤ x/(pb·p_{b-1}). Of the remaining (non-clamp) leaves:
    //       · fold bucket     : v ≥ b-1-band_fold_floor → pi_n ≥ b-1 guaranteed
    //                            → contribution (p2_init + v - (b-2))
    //       · stored records  : t > 0 with Dusart LB too loose → classified at
    //                            resolve-time. In practice always empty.
    #[derive(Clone, Copy)]
    struct ExtEasyRec {
        b_minus_2: i32,
        raw: u32,
        seed_in_query: i32,
        adj_lo: i32,
        local_p2: i64,
    }

    // Per-band safe lower bound on p2_band_inits[t], used to widen the
    // ext-easy fold condition. Dusart (1999): π(n) ≥ n/ln(n) for n ≥ 5393.
    // We use that as a rigorous lower bound for p2_band_inits[t] = π(band_lo[t]−1).
    // Tighter than `initial_p2_offset` for t > 0 (band_lo grows with t), which
    // drastically reduces the number of stored records at large x where
    // lo_start can be 0 (making initial_p2_offset = 0).
    //
    // For band_lo < 5393 the bound is unreliable, so we fall back to 0.
    let pi_lower_bound = |n: u64| -> i64 {
        if n < 5393 { 0 } else {
            ((n as f64) / (n as f64).ln()).floor() as i64
        }
    };
    let p2_init_lb_per_band: Vec<i64> = (0..num_bands)
        .map(|t| {
            let band_lo = band_bounds[t];
            // p2_band_inits[t] = π(band_lo - 1). Use π(band_lo) as a safe
            // lower bound (π is monotone and π(n-1) ≤ π(n)).
            // Actually we want a lower bound on π(band_lo - 1), so use
            // pi_lower_bound(band_lo) - 1 to be safe, then clamp to 0.
            if band_lo == 0 { 0 } else {
                (pi_lower_bound(band_lo) - 1).max(0)
            }
        })
        .collect();

    // (total_clamp_count + far_easy_start are computed earlier — they feed
    // the band_bounds log-scale decision.)

    // ── Single parallel sweep per band: accumulate delta + p2_count AND
    // fold leaf contributions into compact band-local accumulators. ───────────
    type BandSweep = (
        Vec<i64>,           // delta[bi]               (prefix-sum → phi_band_inits)
        i64,                // p2_count                (prefix-sum → p2_band_inits)
        i128,               // leaf_partial
        Vec<i64>,           // bi_contrib[bi]          (size b_ext)
        i128,               // p2_partial
        i64,                // p2_q_count              (number of P2 queries in band)
        i128,               // ext_fold_partial
        i64,                // ext_fold_count
        Vec<ExtEasyRec>,    // ext_stored (fallback when band_fold_floor too loose)
        BandStats,          // fine-grained phase timings + counters
    );

    #[derive(Default, Clone, Copy)]
    struct BandStats {
        fill_ns: u64,
        bi_main_ns: u64,
        bi_main_leaf_ns: u64,
        rest_plain_ns: u64,
        rest_bulk_ns: u64,
        tail_prefix_ns: u64,
        tail_ext_ns: u64,
        tail_p2_ns: u64,
        tail_advance_ns: u64,
        n_bi_leaf_hits: u64,
        n_ext_emitted: u64,
        n_prefix_fills: u64,
        n_bulk_active_sum: u64,
    }

    let band_sweeps: Vec<BandSweep> = (0..num_bands)
        .into_par_iter()
        .map(|t| {
            let mut stats = BandStats::default();

            // delta only needs b_ext entries: bi >= b_ext use pi-formula.
            let mut delta: Vec<i64>       = vec![0i64; b_ext];
            let mut p2_count: i64         = 0;
            // Folded leaf accumulators.
            let mut leaf_partial: i128    = 0;
            let mut bi_contrib: Vec<i64>  = vec![0i64; b_ext];
            // Folded P2 accumulators.
            let mut p2_partial: i128      = 0;
            let mut p2_q_count: i64       = 0;
            // Ext-easy hybrid accumulators.
            let mut ext_fold_partial: i128      = 0;
            let mut ext_fold_count: i64         = 0;
            let mut ext_stored: Vec<ExtEasyRec> = Vec::new();

            let band_lo = band_bounds[t];
            if band_lo > z {
                return (delta, p2_count, leaf_partial, bi_contrib,
                        p2_partial, p2_q_count,
                        ext_fold_partial, ext_fold_count,
                        ext_stored,
                        stats);
            }
            let band_hi = band_bounds[t + 1].min(z + W30_SEG as u64);

            // Per-band easy iterator init. In addition to the band's n-range
            // cap (pl ≤ x/(pb*blo)), we also cap at the NON-CLAMP boundary
            // (pl ≤ x/(pb*p_{b-1})) so the hot sweep never iterates over
            // leaves that would just increment ext_clamped_count. Those are
            // bulk-counted in `total_clamp_count` above.
            let init_easy = |ei: usize, blo: u64| -> (usize, u64) {
                let bi = n_hard + ei;
                let b  = bi + c + 1;
                if b >= a || b < 2 { return (a, u64::MAX); }
                let pb = primes[b - 1];
                let pbm1 = primes[b - 2];
                // Band n-range constraint.
                let band_cnt = if blo == 0 {
                    a - b
                } else {
                    let max_pl = (x / (pb as u128 * blo as u128)) as u64;
                    primes[b..a].partition_point(|&p| p <= max_pl)
                };
                if band_cnt == 0 { return (a, u64::MAX); }
                // Non-clamp constraint (pl ≤ x/(pb*p_{b-1})).
                let pl_clamp_threshold =
                    (x / (pb as u128 * pbm1 as u128)) as u64;
                let nonclamp_cnt = primes[b..a]
                    .partition_point(|&p| p <= pl_clamp_threshold);
                if nonclamp_cnt == 0 { return (a, u64::MAX); }
                // Take the stricter of the two caps. Both are counts of valid
                // pl in primes[b..a]; pl_idx = b + min(cnt) - 1 is the largest
                // valid pl.
                let pl_idx = b + band_cnt.min(nonclamp_cnt) - 1;
                let next_n = (x / (pb as u128 * primes[pl_idx] as u128)) as u64;
                (pl_idx, next_n)
            };

            let (mut easy_ptrs, mut easy_next_n): (Vec<usize>, Vec<u64>) =
                (0..n_easy).map(|ei| init_easy(ei, band_lo)).unzip();

            let mut hard_ptrs: Vec<usize> = (0..n_hard)
                .map(|bi| hard_leaves[bi].partition_point(|&(n, _)| n < band_lo))
                .collect();

            let (p2_start, p2_end) = p2_ranges[t];
            let mut p2_ptr = p2_end; // exclusive, counts down to p2_start

            let mut tiny_state = phi_tiny_state(band_lo);
            let mut sieve      = WheelSieve30::new();
            let mut mono       = MonoCount::new();
            let mut p2_prefix  = [0u32; W30_WORDS + 1];
            let mut lo         = band_lo;
            let mut local_p2_offset: i64 = 0;

            // Effective fold floor for this band: p2_band_inits[t] is at least
            // `initial_p2_offset` (trivial), but for t > 0 we can prove a much
            // tighter bound via the per-band Dusart lower bound (monotone ↑).
            let band_fold_floor: i64 =
                initial_p2_offset.max(p2_init_lb_per_band[t]);

            // Bucket-sieve: only iterate active bulk primes (p² ≤ lo+W30_SEG).
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

            // Per-band persistent cross-off state for bulk primes, keyed by
            // `k = bi - b_ext`. Avoids the per-segment `(lo + p - 1) / p`
            // division: after segment N, `bulk_next_m[k]` holds the next
            // wheel-30 multiple of `primes[c + b_ext + k]` that the cross-off
            // should land on; `bulk_next_j[k]` is the matching wheel index.
            // State is initialised lazily as `bulk_active_end` advances.
            let bulk_cap = n_all.saturating_sub(b_ext);
            let mut bulk_next_m: Vec<u64> = vec![0u64; bulk_cap];
            let mut bulk_next_j: Vec<u8>  = vec![0u8;  bulk_cap];
            let mut bulk_state_valid_end: usize = 0;
            // b_limit: max bi for which leaves are still possible (monotone ↓).
            let mut b_limit = b_ext;
            while b_limit > 0 && band_lo > leaf_cutoff_lo[b_limit - 1] {
                b_limit -= 1;
            }

            while lo < band_hi && lo <= z {
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
                stats.fill_ns += t_fill.elapsed().as_nanos() as u64;

                // ── bi ∈ [0, b_limit): counted cross-off + delta update + leaf emit ──
                // Measured as a single region per segment (not per-bi) because a
                // nested Instant::now() would be called ~b_limit × num_segs times
                // per band and add ~200 s of overhead at x=1e17 α=2. The counter
                // `n_bi_leaf_hits` lets us weigh leaf vs xoff time afterwards.
                let t_bi = Instant::now();
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
                        // Bracketed timer: only fires on leaf hits (sparse —
                        // 7M total at x=1e17 α=2, vs ~3.4 B bi-iterations), so
                        // the per-bi Instant::now warning above does not apply.
                        let t_leaf = Instant::now();
                        stats.n_bi_leaf_hits += 1;
                        mono.reset();
                        // Snapshot local phi BEFORE this segment's running_total update
                        // (matches the pass-2 ordering: phi_vec[bi] += running_total
                        // happened AFTER leaf processing).
                        let snap_phi = delta[bi];

                        if bi < n_hard {
                            let ptr = &mut hard_ptrs[bi];
                            while *ptr < hard_leaves[bi].len() {
                                let (n, mu) = hard_leaves[bi][*ptr];
                                if n >= hi || n > z { break; }
                                if n >= lo {
                                    let popcount =
                                        sieve.count_primes_upto_int_m(&mut mono, n, lo);
                                    // Fold: phi_n = phi_init[bi] + snap_phi + popcount.
                                    // sum += sign * phi_n, sign = -mu.
                                    let sign = -(mu as i64);
                                    leaf_partial += (sign as i128)
                                        * ((snap_phi + popcount as i64) as i128);
                                    bi_contrib[bi] += sign;
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
                                    let popcount =
                                        sieve.count_primes_upto_int_m(&mut mono, n, lo);
                                    // Easy leaves: μ(p_l) = -1 → contribution is +phi_n
                                    // (sign = +1).
                                    leaf_partial +=
                                        (snap_phi + popcount as i64) as i128;
                                    bi_contrib[bi] += 1;
                                }
                                if pl_idx <= b {
                                    easy_ptrs[ei] = a;
                                    break;
                                }
                                let new_idx = pl_idx - 1;
                                easy_ptrs[ei]   = new_idx;
                                easy_next_n[ei] =
                                    (x / (pb as u128 * primes[new_idx] as u128)) as u64;
                            }
                        }
                        stats.bi_main_leaf_ns += t_leaf.elapsed().as_nanos() as u64;
                    }

                    delta[bi] += running_total;
                    running_total -=
                        sieve.cross_off_count_pd_unrolled(lo, pb, &pb_data[bi]) as i64;
                }
                stats.bi_main_ns += t_bi.elapsed().as_nanos() as u64;

                let t_plain = Instant::now();
                // ── bi ∈ [b_limit, b_ext): plain cross-off (no leaves) ──────
                // Kim-style 8-way dispatch with bit positions baked as
                // immediates; replaces the per-bit `bit_seq[j]` lookup +
                // word/bit reconstruction with a single `andb m8, imm8`.
                for bi in b_limit..b_ext {
                    sieve.cross_off_pd_unrolled(lo, primes[c + bi], &pb_data[bi]);
                }
                stats.rest_plain_ns += t_plain.elapsed().as_nanos() as u64;

                let t_bulk = Instant::now();
                // ── Bulk cross-off: bucket skips inactive primes (p² > hi) ──
                let target_end: usize = if lo < y {
                    n_all - b_ext
                } else {
                    while bulk_active_end < n_all {
                        let p = primes[c + bulk_active_end] as u64;
                        if p * p > hi { break; }
                        bulk_active_end += 1;
                    }
                    bulk_active_end - b_ext
                };
                // Initialise persistent state for primes that just became
                // active this segment (paid once per prime per band).
                while bulk_state_valid_end < target_end {
                    let k = bulk_state_valid_end;
                    let p = primes[c + b_ext + k] as u64;
                    let k0 = (lo + p - 1) / p;
                    let k1 = wheel30_next_k(k0);
                    bulk_next_m[k] = k1 * p;
                    bulk_next_j[k] = W30_IDX[(k1 % 30) as usize];
                    bulk_state_valid_end += 1;
                }
                // Cross-off with incremental state: no per-call 64-bit div.
                // NB: `cross_off_pd_from_state_unrolled` (Phase 3) exists in
                // segment.rs and is bit/state-exact, but switching to it
                // here regressed `rest_bulk_xoff` by ~25 % at 1e15 α=1: the
                // bulk primes do 0–3 cross-offs/seg on average, and the
                // 8-way dispatch + byte-view + (m,j)-recovery overhead
                // exceeds the per-iteration savings. Kept rolled here.
                for k in 0..target_end {
                    let p = primes[c + b_ext + k] as u64;
                    let (nm, nj) = sieve.cross_off_pd_from_state(
                        lo, p, &pb_data[b_ext + k],
                        bulk_next_m[k], bulk_next_j[k],
                    );
                    bulk_next_m[k] = nm;
                    bulk_next_j[k] = nj;
                }
                stats.rest_bulk_ns += t_bulk.elapsed().as_nanos() as u64;

                let t_advance_prep = Instant::now();
                // After all cross-offs, sieve = prime sieve over [lo, hi).
                let final_count = sieve.total_count() as i64;
                let seed_in_seg: i64 = if lo < y {
                    let j1 = primes.partition_point(|&p| p < lo);
                    let j2 = primes.partition_point(|&p| p < lo + W30_SEG as u64);
                    (j2 - j1) as i64
                } else { 0 };
                let seg_primes = final_count
                    - if lo == 0 { 1 } else { 0 }
                    + seed_in_seg;
                let adj_lo: i32 = if lo == 0 { 1 } else { 0 };
                stats.tail_advance_ns +=
                    t_advance_prep.elapsed().as_nanos() as u64;

                // Lazy fill of p2_prefix (only if a record actually needs it).
                let mut p2_prefix_ready = false;
                let mut seed_below_lo   = 0usize;

                // Helper closure fragment (used inline): every fill triggers
                // tail_prefix_ns accounting so we can separate that cost from
                // the ext / p2 loop bodies.

                let t_ext = Instant::now();
                // ── Ext-easy leaves (bi >= b_ext) ───────────────────────────
                for ei in far_easy_start..n_easy {
                    if easy_ptrs[ei] >= a { continue; }
                    if easy_next_n[ei] >= hi { continue; }
                    if !p2_prefix_ready {
                        let t_pref = Instant::now();
                        sieve.fill_prefix_counts(&mut p2_prefix);
                        seed_below_lo = if lo < y {
                            primes.partition_point(|&p| p < lo)
                        } else { 0 };
                        stats.tail_prefix_ns +=
                            t_pref.elapsed().as_nanos() as u64;
                        stats.n_prefix_fills += 1;
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
                            let raw = sieve.count_primes_upto_int(&p2_prefix, n, lo);
                            let seed_in_query: i32 = if lo < y {
                                let j2 = primes.partition_point(|&p| p <= n);
                                (j2 - seed_below_lo) as i32
                            } else { 0 };
                            let v = local_p2_offset
                                + raw as i64
                                - adj_lo as i64
                                + seed_in_query as i64;
                            let b_m1 = (b as i64) - 1;
                            // Piste 3 guarantees every emitted leaf satisfies
                            // n ≥ p_{b-1} (clamps bulk-counted before sweep).
                            // At band 0, band_fold_floor = p2_init[0] exactly,
                            // so the fold check always succeeds. At bands t>0
                            // a loose Dusart LB could theoretically force some
                            // leaves into `ext_stored`, but in practice all
                            // ext-easy leaves land in band 0 and this path is
                            // never triggered.
                            if v >= b_m1 - band_fold_floor {
                                ext_fold_partial +=
                                    (v - (b_m1 - 1)) as i128; // V - (b-2)
                                ext_fold_count += 1;
                                stats.n_ext_emitted += 1;
                            } else {
                                ext_stored.push(ExtEasyRec {
                                    b_minus_2: (b as i32) - 2,
                                    raw,
                                    seed_in_query,
                                    adj_lo,
                                    local_p2: local_p2_offset,
                                });
                            }
                        }
                        if pl_idx <= b {
                            easy_ptrs[ei] = a;
                            break;
                        }
                        let new_idx = pl_idx - 1;
                        easy_ptrs[ei]   = new_idx;
                        easy_next_n[ei] =
                            (x / (pb as u128 * primes[new_idx] as u128)) as u64;
                    }
                }

                stats.tail_ext_ns += t_ext.elapsed().as_nanos() as u64;

                let t_p2 = Instant::now();
                // ── P2 queries ─────────────────────────────────────────────
                if p2_ptr > p2_start {
                    let q_check = (x / s2_primes[p2_ptr - 1] as u128) as u64;
                    if q_check >= lo && q_check < hi {
                        if !p2_prefix_ready {
                            let t_pref = Instant::now();
                            sieve.fill_prefix_counts(&mut p2_prefix);
                            seed_below_lo = if lo < y {
                                primes.partition_point(|&p| p < lo)
                            } else { 0 };
                            stats.tail_prefix_ns +=
                                t_pref.elapsed().as_nanos() as u64;
                            stats.n_prefix_fills += 1;
                            // p2_prefix_ready not reassigned: no further use
                            // in this segment (last section before advance).
                        }
                        loop {
                            if p2_ptr <= p2_start { break; }
                            let j   = p2_ptr - 1;
                            let q_k = (x / s2_primes[j] as u128) as u64;
                            if q_k >= hi { break; }
                            if q_k < lo  { p2_ptr -= 1; continue; }
                            let raw = sieve.count_primes_upto_int(&p2_prefix, q_k, lo);
                            let seed_in_query: i32 = if lo < y {
                                let j2 = primes.partition_point(|&p| p <= q_k);
                                (j2 - seed_below_lo) as i32
                            } else { 0 };
                            // Fold P2: pi_qk = p2_init[t] + V, V = local_p2
                            // + raw - adj_lo + seed_in_query; k = a + j.
                            // Σ (pi_qk - k) = p2_init * count + Σ (V - k).
                            let v = local_p2_offset
                                + raw as i64
                                - adj_lo as i64
                                + seed_in_query as i64;
                            let k = (a + j) as i64;
                            p2_partial += (v - k) as i128;
                            p2_q_count += 1;
                            p2_ptr -= 1;
                        }
                    }
                }

                stats.tail_p2_ns += t_p2.elapsed().as_nanos() as u64;

                let t_advance = Instant::now();
                stats.n_bulk_active_sum +=
                    (bulk_active_end.saturating_sub(b_ext)) as u64;
                local_p2_offset += seg_primes;
                p2_count        += seg_primes;

                let next_lo = lo + W30_SEG as u64;
                advance_wheel_primes(&mut tiny_state, next_lo);
                lo = next_lo;
                stats.tail_advance_ns +=
                    t_advance.elapsed().as_nanos() as u64;
            }
            (delta, p2_count, leaf_partial, bi_contrib,
             p2_partial, p2_q_count,
             ext_fold_partial, ext_fold_count,
             ext_stored,
             stats)
        })
        .collect();

    // ── Sequential prefix scan for phi / P2 per band ─────────────────────────
    let mut phi_band_inits: Vec<Vec<i64>> = vec![vec![0i64; b_ext]; num_bands];
    phi_band_inits[0] = initial_phi_vec;
    for t in 1..num_bands {
        for bi in 0..b_ext {
            phi_band_inits[t][bi] =
                phi_band_inits[t - 1][bi] + band_sweeps[t - 1].0[bi];
        }
    }
    let mut p2_band_inits = vec![initial_p2_offset; num_bands];
    for t in 1..num_bands {
        p2_band_inits[t] = p2_band_inits[t - 1] + band_sweeps[t - 1].1;
    }

    // ── Accumulate sweep stats across bands. ─────────────────────────────
    let mut agg = BandStats::default();
    for b in &band_sweeps {
        let s = &b.9;
        agg.fill_ns           += s.fill_ns;
        agg.bi_main_ns        += s.bi_main_ns;
        agg.bi_main_leaf_ns   += s.bi_main_leaf_ns;
        agg.rest_plain_ns     += s.rest_plain_ns;
        agg.rest_bulk_ns      += s.rest_bulk_ns;
        agg.tail_prefix_ns    += s.tail_prefix_ns;
        agg.tail_ext_ns       += s.tail_ext_ns;
        agg.tail_p2_ns        += s.tail_p2_ns;
        agg.tail_advance_ns   += s.tail_advance_ns;
        agg.n_bi_leaf_hits    += s.n_bi_leaf_hits;
        agg.n_ext_emitted     += s.n_ext_emitted;
        agg.n_prefix_fills    += s.n_prefix_fills;
        agg.n_bulk_active_sum += s.n_bulk_active_sum;
    }

    // ── Resolution pass (parallel per band) ──────────────────────────────────
    let t_resolve = Instant::now();
    let resolved: Vec<(i128, u128)> = (0..num_bands)
        .into_par_iter()
        .map(|t| {
            let phi_init = &phi_band_inits[t];
            let p2_init  = p2_band_inits[t];
            let band = &band_sweeps[t];
            let leaf_partial      = band.2;
            let bi_contrib        = &band.3;
            let p2_partial        = band.4;
            let p2_q_count        = band.5;
            let ext_fold_partial  = band.6;
            let ext_fold_count    = band.7;
            let ext_stored        = &band.8;

            // Leaf contribution (fully folded).
            let mut sum: i128 = leaf_partial;
            for bi in 0..bi_contrib.len() {
                sum += (bi_contrib[bi] as i128) * (phi_init[bi] as i128);
            }
            // Ext-easy folded part (max guaranteed not to fire).
            sum += (p2_init as i128) * (ext_fold_count as i128) + ext_fold_partial;
            // Ext-easy may-clamp records (rare, bands > 0 only): apply max(·, 1).
            for rec in ext_stored.iter() {
                let pi_n = p2_init + rec.local_p2
                    + rec.raw as i64
                    - rec.adj_lo as i64
                    + rec.seed_in_query as i64;
                let phi_n = (pi_n - rec.b_minus_2 as i64).max(1);
                sum += phi_n as i128;
            }
            // P2 contribution (fully folded).
            let p2_sum_i: i128 =
                (p2_init as i128) * (p2_q_count as i128) + p2_partial;
            let p2_sum: u128 = if p2_sum_i < 0 { 0 } else { p2_sum_i as u128 };
            (sum, p2_sum)
        })
        .collect();
    let ns_resolve = t_resolve.elapsed().as_nanos() as u64;

    // s2_total = per-band folded sums + bulk clamp count (+1 each).
    let s2_total: i128 =
        resolved.iter().map(|&(s, _)| s).sum::<i128>()
        + total_clamp_count as i128;
    let p2_total: u128 = resolved.iter().map(|&(_, p)| p).sum();

    let profile = HardProfile {
        sweep_fill_ns:             agg.fill_ns,
        sweep_bi_main_ns:          agg.bi_main_ns,
        sweep_bi_main_leaf_ns:     agg.bi_main_leaf_ns,
        rest_plain_ns:             agg.rest_plain_ns,
        rest_bulk_ns:              agg.rest_bulk_ns,
        tail_prefix_build_ns:      agg.tail_prefix_ns,
        tail_ext_emit_ns:          agg.tail_ext_ns,
        tail_p2_emit_ns:           agg.tail_p2_ns,
        tail_advance_ns:           agg.tail_advance_ns,
        resolve_ns:                ns_resolve,
        n_bi_leaf_hits:            agg.n_bi_leaf_hits,
        n_leaves_ext_emitted:      agg.n_ext_emitted,
        n_leaves_ext_clamped:      total_clamp_count.max(0) as u64,
        n_prefix_fills:            agg.n_prefix_fills,
        n_bulk_active_primes_sum:  agg.n_bulk_active_sum,
    };
    (s2_total, p2_total, profile)
}
