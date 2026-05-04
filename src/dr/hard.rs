/// Per-phase nanosecond accumulators produced by [`s2_hard_sieve_par`] when
/// called from the profiling entry point. Summed across all Rayon bands, so
/// values reflect *CPU time* (not wall time) and should add up to roughly
/// `num_threads × wall_time_of_s2_hard`.
#[derive(Default, Clone, Debug)]
pub struct HardProfile {
    /// Single-pass sweep: sieve.fill_presieved / sieve.fill + total_count.
    pub sweep_fill_ns: u64,

    /// bi ∈ [0, b_limit) main loop (counted cross-off + leaf emit, bundled).
    /// Measured per segment, not per bi, to avoid Instant::now overhead on
    /// the ~3.4 B inner iterations at x=1e17 α=2. Use `n_bi_leaf_hits`
    /// together with `n_leaves_ext_emitted` to weigh the leaf vs xoff share.
    pub sweep_bi_main_ns: u64,

    /// Subset of `sweep_bi_main_ns` spent inside `if has_leaf { … }` (popcount
    /// + factor-table walk / easy_ptrs walk + fold accumulators). Bracketed only when
    /// a leaf actually fires, so the timer adds ~2 × n_bi_leaf_hits Instant
    /// calls (~0.3 % overhead at x=1e17 α=2). The xoff share is derived as
    /// `sweep_bi_main_ns − sweep_bi_main_leaf_ns` at print time.
    pub sweep_bi_main_leaf_ns: u64,

    // ── bi ∈ [b_limit, n_all) cross-off (split) ───────────────────────────
    /// Plain cross-off bi ∈ [b_limit, b_ext).
    pub rest_plain_ns: u64,
    /// Bulk cross-off bi ∈ [b_ext, bulk_end) (includes bucket advance).
    pub rest_bulk_ns: u64,
    /// Optional fine-grained rest_bulk timings, populated when
    /// `RIVAT3_REST_BULK_PROFILE=1`.
    pub rest_bulk_detail: RestBulkProfile,

    // ── Tail (split) ──────────────────────────────────────────────────────
    /// `fill_prefix_counts` + seed_below_lo bsearch (lazy per segment).
    pub tail_prefix_build_ns: u64,
    /// Ext-easy leaf emission loop.
    pub tail_ext_emit_ns: u64,
    /// Ext-easy contribution resolved from the in-sieve path (including clamp
    /// leaves). Used by opt-in split-tail probes.
    pub tail_ext_contribution: i128,
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

    /// Per-band breakdown of the same counters/timers, in band order
    /// (band 0 = lowest n). Populated by [`s2_hard_sieve_par`] for
    /// load-balancing diagnostics. Empty `Vec` when not relevant.
    pub per_band: Vec<BandProfile>,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RestBulkProfile {
    pub enabled: bool,
    pub active_scan_ns: u64,
    pub state_init_ns: u64,
    pub xoff_ns: u64,
    pub xoff_calls: u64,
    pub state_inits: u64,
    pub segments: u64,
    pub target_sum: u64,
    pub bin_ns: [u64; REST_BULK_BINS],
    pub bin_calls: [u64; REST_BULK_BINS],
}

pub const REST_BULK_BINS: usize = 5;

/// Per-band profile entry. CPU time and counters captured by one Rayon
/// worker for a single band (segments `[band_lo, band_hi)`).
///
/// Used by `--dr-profile` to diagnose load imbalance in S2_hard. At α=2,
/// ext-easy leaves funnel into the lowest-n bands, capping Rayon scaling.
#[derive(Default, Clone, Debug)]
pub struct BandProfile {
    /// Band index (0 = lowest n).
    pub band_t: usize,
    /// First sieve segment in band (inclusive).
    pub band_lo: u64,
    /// First sieve segment past band (exclusive).
    pub band_hi: u64,
    pub fill_ns: u64,
    pub bi_main_ns: u64,
    pub bi_main_leaf_ns: u64,
    pub rest_plain_ns: u64,
    pub rest_bulk_ns: u64,
    pub tail_prefix_ns: u64,
    pub tail_ext_ns: u64,
    pub tail_p2_ns: u64,
    pub tail_advance_ns: u64,
    pub n_bi_leaf_hits: u64,
    pub n_ext_emitted: u64,
    pub n_prefix_fills: u64,
    pub n_bulk_active_sum: u64,
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
    s2_primes: &crate::prime_bitset::PrimeBitset, // primes in (y, √x] for P2 = Σ(π(x/p) − (π(p)−1))
) -> (i128, u128, HardProfile) {
    use crate::factor_table::FactorTable;
    use crate::bucket_sieve::{BucketChainIter, BucketSieve};
    use crate::segment::{advance_wheel_primes, count_primes_in_segment, MonoCount, WheelPrimeData, WheelSieve30, W30_IDX, W30_SEG, W30_WORDS, wheel30_next_k};
    use rayon::prelude::*;
    use std::time::Instant;

    // Refill `hard_next_n[bi]`/`hard_next_mu[bi]` by walking the descending
    // factor-table cursor `hard_idx[bi]` until the next hard leaf for prime
    // `pb` is reached, or until the cursor crosses `min_idx` (exclusive lower
    // bound = `to_index_floor(pb)` so that the m=pb sentinel is skipped). On
    // exhaustion the cached n is set to `u64::MAX`, which makes `has_leaf`
    // permanently false for that bi until the next band setup. The hot
    // `has_leaf` path then stays a single load + compare.
    #[inline(always)]
    fn refill_hard_next(
        factor: &FactorTable,
        hard_idx: &mut [i64],
        hard_next_n: &mut [u64],
        hard_next_mu: &mut [i8],
        bi: usize,
        pb: u64,
        pb_u16: u16,
        min_idx: i64,
        x: u128,
    ) {
        let mut idx = hard_idx[bi];
        while idx > min_idx {
            let f = factor.mu_lpf(idx as usize);
            // f > pb_u16 alone covers the hard-leaf predicate:
            //   f = 0          → μ(m) = 0,    rejected (0 ≤ pb_u16)
            //   f = lpf − 1    → μ = +1, lpf > pb + 1 ≥ pb (off-by-1 OK
            //                    because primes pb ≥ 13 are odd and lpf is
            //                    an odd prime, so lpf = pb + 1 is impossible)
            //   f = lpf        → μ = −1, lpf > pb
            //   f = U16_MAX    → m prime, m > pb (m = pb is excluded by min_idx)
            if f > pb_u16 {
                let m = FactorTable::to_number(idx as usize);
                let mu = factor.mu(idx as usize);
                let n = (x / (pb as u128 * m as u128)) as u64;
                hard_next_n[bi] = n;
                hard_next_mu[bi] = mu;
                hard_idx[bi] = idx - 1;
                return;
            }
            idx -= 1;
        }
        hard_next_n[bi] = u64::MAX;
        hard_next_mu[bi] = 0;
        hard_idx[bi] = min_idx;
    }

    // Reads /proc/self/status VmRSS in MB (Linux only; returns 0 elsewhere).
    // Used by RIVAT3_MEM_DUMP checkpoints to localize the allocation phase.
    fn proc_rss_mb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
                for line in s.lines() {
                    if let Some(rest) = line.strip_prefix("VmRSS:") {
                        if let Some(kb_str) = rest.split_whitespace().next() {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return (kb as f64) / 1024.0;
                            }
                        }
                    }
                }
            }
            0.0
        }
        #[cfg(not(target_os = "linux"))]
        { 0.0 }
    }
    let mem_dump = std::env::var("RIVAT3_MEM_DUMP").is_ok();
    macro_rules! ck {
        ($label:expr) => {
            if mem_dump {
                eprintln!("[mem-rss @ {}]: {:.0} MB", $label, proc_rss_mb());
            }
        };
    }
    ck!("entry s2_hard_sieve_par");

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
        // Piste D: `b_ext_mult` (≥ 1.0) lets experiments push the frontier
        // ABOVE the canonical x^{1/4}. We never let it go below the natural
        // boundary: pi-formula is only valid for `pb > x^{1/4}` (guarantees
        // n < pb², the leaf-B condition); reducing b_ext under that threshold
        // corrupts π. The natural `partition_point(p ≤ x^{1/4})` is therefore
        // the algorithmic minimum — `b_ext_mult < 1.0` is silently lifted.
        let x4_base = (x as f64).sqrt().sqrt();
        let x4_natural: u64 = x4_base as u64 + 2;
        let mult = crate::parameters::b_ext_mult().max(1.0);
        let x4: u64 = (x4_base * mult) as u64 + 2;
        let natural_partition = primes[c..a].partition_point(|&p| p <= x4_natural);
        primes[c..a].partition_point(|&p| p <= x4)
            .max(natural_partition) // never below the canonical DR boundary
            .max(n_hard)            // must cover all hard leaves
            .min(n_all)             // clamp to valid range
    };
    // n_ext_easy: easy bi values below b_ext that still use phi_vec
    let n_ext_easy = b_ext.saturating_sub(n_hard);

    // ── Compact (μ, lpf) factor table for streaming hard-leaf enumeration ───
    // Replaces the prior `hard_leaves: Vec<Vec<(u64, i8)>>` (sorted leaf
    // lists per hard bi). Each band keeps a per-bi descending cursor into
    // this table and produces the next leaf on demand — same has_leaf hot-
    // path cost (one cached load + compare) but no persistent leaf storage.
    // At x = 1e18 α = 2: ~602 MB → ~830 KB.
    ck!("before factor table build");
    let factor = FactorTable::new(y);
    ck!("after  factor table build");

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

    // P2 walker setup is now derived per work item in `chunk_setup` (see the
    // scheduler block below), so the formerly precomputed `p2_band_setup`
    // array is no longer needed — the closure handles both whole bands and
    // sub-chunks identically.

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
    //   Ext-easy fold (Piste 3 keeps it linear):
    //     Clamp leaves (n < p_{b-1}) are bulk-counted BEFORE the sweep and
    //     added directly to S2 at resolve time (see `total_clamp_count` below);
    //     the sweep skips them entirely by capping the pl iterator at
    //     pl ≤ x/(pb·p_{b-1}). For every emitted (non-clamp) leaf we then
    //     have pi(n) ≥ b-1, so phi(n, b-1) = pi(n) - (b-2) is ≥ 1 with no
    //     clamp needed — contribution is folded as (p2_init + v - (b-2)).

    // Per-segment snapshot used by the 2-pass deferred-tail-ext path. Heavy
    // bands (band 0+1 at α=2 / log-scale regime) skip the in-line ext-easy
    // emission during pass 1 and stash one of these per segment instead;
    // pass 2 then drains them with a nested par_iter on `ei` once the
    // light-band threads are free.  Size ≈ 26 KB / segment (W30_WORDS = 2 185).
    #[derive(Clone)]
    struct DeferredSeg {
        bits:            [u64; W30_WORDS],
        p2_prefix:       [u32; W30_WORDS + 1],
        lo:              u64,
        local_p2_offset: i64,
        adj_lo:          i32,
        seed_below_lo:   usize,
    }

    // (total_clamp_count + far_easy_start are computed earlier — they feed
    // the band_bounds log-scale decision.)

    // ── Single parallel sweep per band: accumulate delta + p2_count AND
    // fold leaf contributions into compact band-local accumulators. ───────────
    type BandSweep = (
        Vec<i64>,           //  0: delta[bi]            (prefix-sum → phi_band_inits)
        i64,                //  1: p2_count             (prefix-sum → p2_band_inits)
        i128,               //  2: leaf_partial
        Vec<i64>,           //  3: bi_contrib[bi]       (size b_ext)
        i128,               //  4: p2_partial
        i64,                //  5: p2_q_count           (number of P2 queries in band)
        i128,               //  6: ext_fold_partial
        i64,                //  7: ext_fold_count
        BandStats,          //  8: fine-grained phase timings + counters
    );

    #[derive(Default, Clone, Copy)]
    struct BandStats {
        fill_ns: u64,
        bi_main_ns: u64,
        bi_main_leaf_ns: u64,
        rest_plain_ns: u64,
        rest_bulk_ns: u64,
        rest_bulk_active_scan_ns: u64,
        rest_bulk_state_init_ns: u64,
        rest_bulk_xoff_ns: u64,
        tail_prefix_ns: u64,
        tail_ext_ns: u64,
        tail_p2_ns: u64,
        tail_advance_ns: u64,
        n_bi_leaf_hits: u64,
        n_ext_emitted: u64,
        n_prefix_fills: u64,
        n_bulk_active_sum: u64,
        n_bulk_xoff_calls: u64,
        n_bulk_state_inits: u64,
        n_bulk_segments: u64,
        n_bulk_target_sum: u64,
        bulk_bin_ns: [u64; REST_BULK_BINS],
        bulk_bin_calls: [u64; REST_BULK_BINS],
    }

    // Heavy-band selection for the 2-pass deferred-tail-ext path. Heuristic
    // (POC stage A): bands 0 + 1 only when the log-scale layout is active
    // (= α=2 clamp regime). Outside α=2, `use_log_scale = false` so the mask
    // is empty and pass 2 is a no-op — code reverts to single-pass behaviour
    // bit-for-bit. Disabled entirely via `--no-deferred-tail-ext`.
    let defer_enabled = !crate::parameters::no_deferred_tail_ext();
    let is_heavy = |t: usize| -> bool {
        defer_enabled && use_log_scale && t < 2
    };

    // Bucket-sieve dispatch for `rest_bulk_xoff` (cible #1). Default OFF
    // since the linear sweep measured 1.93× faster on 9700X at 1e18
    // (`pb_data` linear walk benefits from hardware prefetch enough to
    // out-perform the bucket's per-prime `WheelPrimeData::new(p)` recompute).
    // Bucket kept opt-in for future experiments (compact `pb_data`, alternate
    // recompute strategy, etc.). `--bucket-bulk` / `RIVAT3_BUCKET_BULK=1`.
    let bucket_bulk = crate::parameters::bucket_bulk();
    let rest_bulk_profile = std::env::var("RIVAT3_REST_BULK_PROFILE")
        .ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_lowercase() != "false")
        .unwrap_or(false);
    let tail_ext_split_enabled = std::env::var("RIVAT3_TAIL_EXT_SPLIT")
        .ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_lowercase() != "false")
        .unwrap_or(false);

    // Per-band easy iterator init, hoisted out of the per-band closure so the
    // pass-2 deferred-tail-ext replay can reuse it for heavy bands. Captures
    // `n_hard, c, a, primes, x` from fn scope.  See pass-1 site below for
    // the in-band init that primes `easy_ptrs` / `easy_next_n` for ei < a.
    let init_easy = |ei: usize, blo: u64| -> (usize, u64) {
        let bi = n_hard + ei;
        let b  = bi + c + 1;
        if b >= a || b < 2 { return (a, u64::MAX); }
        let pb   = primes[b - 1];
        let pbm1 = primes[b - 2];
        let band_cnt = if blo == 0 {
            a - b
        } else {
            let max_pl = (x / (pb as u128 * blo as u128)) as u64;
            primes[b..a].partition_point(|&p| p <= max_pl)
        };
        if band_cnt == 0 { return (a, u64::MAX); }
        let pl_clamp_threshold =
            (x / (pb as u128 * pbm1 as u128)) as u64;
        let nonclamp_cnt = primes[b..a]
            .partition_point(|&p| p <= pl_clamp_threshold);
        if nonclamp_cnt == 0 { return (a, u64::MAX); }
        let pl_idx = b + band_cnt.min(nonclamp_cnt) - 1;
        let next_n = (x / (pb as u128 * primes[pl_idx] as u128)) as u64;
        (pl_idx, next_n)
    };

    // ── Memory instrumentation (opt-in via RIVAT3_MEM_DUMP=1) ───────────────
    // Prints accountable bytes of the top allocations right before the parallel
    // sweep starts, so the user can correlate /usr/bin/time -v RSS with the
    // actual structures. Helps decide which compaction target to attack first.
    if std::env::var("RIVAT3_MEM_DUMP").is_ok() {
        let mb = |bytes: u64| (bytes as f64) / (1024.0 * 1024.0);
        let factor_bytes = factor.size_bytes() as u64;
        let bulk_cap = (n_all.saturating_sub(b_ext)) as u64;
        let n_easy_u = n_easy as u64;
        let b_ext_u  = b_ext as u64;
        let num_bands_u = num_bands as u64;
        let n_threads = rayon::current_num_threads() as u64;
        let primes_bytes = (primes.len() as u64) * 8;
        let s2_primes_bytes = s2_primes.size_bytes() as u64;
        let pb_data_bytes = ((a - c) as u64) * 80;
        let phi_band_inits_bytes = num_bands_u * b_ext_u * 8;
        let bandsweep_per = b_ext_u * 16; // delta + bi_contrib (Vec data, ignoring header)
        let bandsweep_total = num_bands_u * bandsweep_per;
        let per_thread_temp =
              n_easy_u * 8     // easy_ptrs
            + n_easy_u * 8     // easy_next_n
            + bulk_cap * 8     // bulk_next_m
            + bulk_cap * 1     // bulk_next_j
            + b_ext_u * 8      // delta
            + b_ext_u * 8;     // bi_contrib
        let per_thread_total = n_threads * per_thread_temp;

        eprintln!("[mem-dump x={x} y={y} a={a} b_ext={b_ext} n_hard={n_hard} \
                   n_easy={n_easy} bulk_cap={bulk_cap} num_bands={num_bands} \
                   threads={n_threads}]");
        eprintln!("  primes (= seed)        : {:>11} entries × 8B  = {:>9.1} MB",
                  primes.len(), mb(primes_bytes));
        eprintln!("  s2_primes (PrimeBitset): {:>11} primes wheel-30 = {:>9.1} MB \
                   (replaces all_primes Vec<u32>)",
                  s2_primes.total(), mb(s2_primes_bytes));
        eprintln!("  factor (μ, lpf) table  : {:>11} slots   × 2B = {:>9.1} MB \
                   (y={y}, replaces hard_leaves)",
                  factor.len(), mb(factor_bytes));
        eprintln!("  pb_data                : {:>11} primes  × 80B = {:>9.1} MB",
                  a - c, mb(pb_data_bytes));
        eprintln!("  phi_band_inits         : {:>11} entries × 8B  = {:>9.1} MB",
                  num_bands * b_ext, mb(phi_band_inits_bytes));
        eprintln!("  BandSweep result Vec   : {} bands × {:.0} KB    = {:>9.1} MB",
                  num_bands, (bandsweep_per as f64) / 1024.0,
                  mb(bandsweep_total));
        eprintln!("  per-thread temp peak   : {} × {:.1} MB = {:>9.1} MB \
                   (transient)", n_threads, mb(per_thread_temp), mb(per_thread_total));
        let accountable = primes_bytes + s2_primes_bytes + factor_bytes
            + pb_data_bytes + phi_band_inits_bytes + bandsweep_total
            + per_thread_total;
        eprintln!("  accountable subtotal   : {:>9.1} MB", mb(accountable));
    }
    ck!("before par_iter (band_sweeps)");

    // ── Phase-0 phi_band_init recompute probe (Design B feasibility) ─────────
    // Opt-in via `RIVAT3_PHI_INIT_PROBE=1`. Measures the wall-time cost of
    // rebuilding phi_vec[0..b_ext] from scratch at sub-band positions 1/4,
    // 1/2, 3/4 within each of the heavy bands {1, 2, 3} (the α=2 / 1e18
    // ext_easy hot-spot identified in session_handoff_2026-05-03). The probe
    // dominates Design B's budget: if recompute > ~5 % of band solo CPU,
    // sub-band chunking can't pay off; otherwise it does.
    if crate::parameters::phi_init_probe() {
        // Replays the same sieve-and-count logic as `initial_phi_vec` above,
        // but parameterised on `lo_target` (an arbitrary chunk_lo within a
        // band) rather than `lo_start`.  Returns the elapsed wall in nanoseconds
        // and a checksum (sum of phi_vec) to keep the loop body alive against
        // dead-code elimination by the optimiser.
        let probe_phi_init = |lo_target: u64| -> (u128, i64) {
            let t0 = Instant::now();
            let mut phi_vec = vec![0i64; b_ext];
            let n_init  = lo_target as usize;
            if n_init > 0 && b_ext > 0 {
                let n_words = (n_init + 63) / 64;
                let mut bits: Vec<u64> = vec![!0u64; n_words];
                bits[0] &= !1u64;
                let overhang = n_init % 64;
                if overhang != 0 {
                    bits[n_words - 1] &= (1u64 << overhang) - 1;
                }
                for k in 0..c {
                    let p = primes[k] as usize;
                    let mut m = p;
                    while m < n_init {
                        bits[m / 64] &= !(1u64 << (m % 64));
                        m += p;
                    }
                }
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
            let elapsed = t0.elapsed().as_nanos();
            let checksum = phi_vec.iter().sum::<i64>();
            (elapsed, checksum)
        };

        eprintln!("┌─ phi_init recompute probe (Phase 0, Design B feasibility) ─");
        eprintln!("│  b_ext = {}, c = {}", b_ext, c);
        eprintln!("│  band  position    chunk_lo       elapsed     checksum");
        for &t in &[1usize, 2, 3] {
            if t + 1 > num_bands { continue; }
            let blo = band_bounds[t];
            let bhi = band_bounds[t + 1].min(z + W30_SEG as u64);
            if bhi <= blo { continue; }
            let span = bhi - blo;
            let snap = |lo: u64| (lo / W30_SEG as u64) * W30_SEG as u64;
            for &(label, num, den) in &[("25%", 1u64, 4u64), ("50%", 1, 2), ("75%", 3, 4)] {
                let target = snap(blo + span * num / den);
                let (ns, sum) = probe_phi_init(target);
                let ms = (ns as f64) / 1_000_000.0;
                eprintln!("│  {:>4}  {:>5}  {:>13}  {:>9.2} ms  {:>13}",
                          t, label, target, ms, sum);
            }
        }
        eprintln!("└─");
    }

    let rest_bulk_bin_for_p = |p: u64| -> usize {
        let w = W30_SEG as u64;
        if p <= w / 4 {
            0
        } else if p <= w / 2 {
            1
        } else if p <= w {
            2
        } else if p <= 2 * w {
            3
        } else {
            4
        }
    };

    // Process one work unit. Phase 2 generalisation: a "chunk" is either a
    // whole band (chunk_lo = band_lo, chunk_hi = band_hi) or a sub-range of a
    // heavy band ([chunk_lo, chunk_hi) ⊆ [band_lo, band_hi)). Internal state
    // (sieve, easy_ptrs, hard_idx, tiny_state, bulk_active_end, p2_walker) is
    // initialised at chunk_lo and the sweep iterates [chunk_lo, chunk_hi).
    // The returned tuple is the same shape as before (a "BandSweep") and is
    // tagged with (band_id, chunk_lo) by the caller for per-band aggregation.
    let process_band = |
        _band_id: usize,
        chunk_lo: u64,
        chunk_hi_in: u64,
        walker_start_n: u64,
        p2_min_rank: usize,
        is_chunk_heavy: bool,
    | -> BandSweep {
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
            // Ext-easy fold accumulators (Piste 3 keeps everything linear).
            let mut ext_fold_partial: i128      = 0;
            let mut ext_fold_count: i64         = 0;
            // Pass-1 deferred snapshots (only populated for heavy bands).
            let mut deferred: Vec<DeferredSeg>  = Vec::new();
            let band_is_heavy = is_chunk_heavy;

            // Aliases preserve the body's local naming. `band_lo`/`band_hi` here
            // refer to the *chunk* boundaries; for whole-band work units they
            // coincide with the band boundaries.
            let band_lo = chunk_lo;
            if band_lo > z {
                return (delta, p2_count, leaf_partial, bi_contrib,
                        p2_partial, p2_q_count,
                        ext_fold_partial, ext_fold_count,
                        stats);
            }
            let band_hi = chunk_hi_in;

            // Easy iterator init. In addition to the band's n-range cap
            // (pl ≤ x/(pb*blo)), we also cap at the NON-CLAMP boundary
            // (pl ≤ x/(pb*p_{b-1})) so the hot sweep never iterates over
            // leaves that would just increment ext_clamped_count. Those are
            // bulk-counted in `total_clamp_count` above. (See `init_easy` at
            // fn scope for the body.)
            let (mut easy_ptrs, mut easy_next_n): (Vec<usize>, Vec<u64>) =
                (0..n_easy).map(|ei| init_easy(ei, band_lo)).unzip();

            // Per-bi descending cursor into the factor table + cached next
            // hard leaf. `hard_min_idx[bi]` = `to_index_floor(pb)` (exclusive
            // lower bound, so m=pb is never visited and the predicate
            // `f > pb_u16` does the right thing for prime m too).
            let mut hard_idx:      Vec<i64> = vec![-1; n_hard];
            let mut hard_next_n:   Vec<u64> = vec![u64::MAX; n_hard];
            let mut hard_next_mu:  Vec<i8>  = vec![0; n_hard];
            let mut hard_min_idx:  Vec<i64> = vec![-1; n_hard];
            for bi in 0..n_hard {
                let pb     = primes[bi + c];
                let pb_u16 = pb as u16;
                hard_min_idx[bi] = FactorTable::to_index_floor(pb);
                let m_top = if band_lo == 0 {
                    y
                } else {
                    ((x / (pb as u128 * band_lo as u128)) as u64).min(y)
                };
                hard_idx[bi] = FactorTable::to_index_floor(m_top);
                refill_hard_next(
                    &factor,
                    &mut hard_idx, &mut hard_next_n, &mut hard_next_mu,
                    bi, pb, pb_u16, hard_min_idx[bi], x,
                );
            }

            // Per-chunk P2 walker: descends through s2_primes from largest p
            // (smallest q = x/p) toward smallest p (largest q). Stops once
            // walker.rank() < p2_min_rank, i.e. once the next prime would
            // have q ≥ chunk_hi (out of this chunk's q-range). The setup
            // values are computed by the caller (cf. `chunk_setup_at`) so the
            // closure is identical for whole-band and sub-chunk work units.
            let mut p2_walker = s2_primes.walker_at(walker_start_n);

            let mut tiny_state = phi_tiny_state(band_lo);
            let mut sieve      = WheelSieve30::new();
            let mut mono       = MonoCount::new();
            let mut p2_prefix  = [0u32; W30_WORDS + 1];
            let mut lo         = band_lo;
            let mut local_p2_offset: i64 = 0;

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
            // Used by the legacy linear path; left untouched (and unused)
            // when the bucket-sieve dispatch is active.
            let bulk_cap = n_all.saturating_sub(b_ext);
            let mut bulk_next_m: Vec<u64> = vec![0u64; bulk_cap];
            let mut bulk_next_j: Vec<u8>  = vec![0u8;  bulk_cap];
            let mut bulk_state_valid_end: usize = 0;

            // Bucket-sieve dispatch for `rest_bulk_xoff`: when active (default),
            // every bulk prime in `[b_ext, n_all)` is pre-distributed into the
            // bucket of its first multiple ≥ band_lo. Each segment's loop
            // drains its bucket chain, crosses off the multiples, and re-inserts
            // each prime into the bucket of its next multiple. Primes whose
            // next multiple lands beyond `band_hi` are silently dropped (the
            // next band re-initialises from scratch).
            //
            // Allocate one extra slot for `band_hi` not snapped exactly to a
            // W30_SEG multiple; cheap (8 B) vs. correctness risk.
            let bucket_num_segs =
                ((band_hi - band_lo) / W30_SEG as u64) as usize + 2;
            let mut bucket_sieve: Option<BucketSieve> = if bucket_bulk {
                let mut bs = BucketSieve::new(bucket_num_segs);
                for k in 0..bulk_cap {
                    let p = primes[c + b_ext + k] as u64;
                    // First wheel-30 multiple of p in [band_lo, ∞).
                    let k0 = (band_lo + p - 1) / p;
                    let k1 = wheel30_next_k(k0);
                    let first_m = match k1.checked_mul(p) {
                        Some(v) => v,
                        None => continue,
                    };
                    if first_m > z { continue; }
                    let target_seg = ((first_m - band_lo) / W30_SEG as u64) as usize;
                    if target_seg >= bucket_num_segs { continue; }
                    let target_seg_lo =
                        band_lo + (target_seg as u64) * W30_SEG as u64;
                    let mi = (first_m - target_seg_lo) as u32;
                    let wi = W30_IDX[(k1 % 30) as usize] as u32;
                    bs.insert(target_seg, p as u32, mi, wi);
                }
                Some(bs)
            } else {
                None
            };
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
                if WheelSieve30::supports_presieved(c) {
                    sieve.fill_presieved(lo, c);
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
                        let n = hard_next_n[bi];
                        n >= lo && n < hi && n <= z
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
                            let pb_u16  = pb as u16;
                            let min_idx = hard_min_idx[bi];
                            loop {
                                let n = hard_next_n[bi];
                                if n >= hi || n > z { break; }
                                if n >= lo {
                                    let mu = hard_next_mu[bi];
                                    let popcount =
                                        sieve.count_primes_upto_int_m(&mut mono, n, lo);
                                    // Fold: phi_n = phi_init[bi] + snap_phi + popcount.
                                    // sum += sign * phi_n, sign = -mu.
                                    let sign = -(mu as i64);
                                    leaf_partial += (sign as i128)
                                        * ((snap_phi + popcount as i64) as i128);
                                    bi_contrib[bi] += sign;
                                }
                                refill_hard_next(
                                    &factor,
                                    &mut hard_idx, &mut hard_next_n, &mut hard_next_mu,
                                    bi, pb, pb_u16, min_idx, x,
                                );
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
                if let Some(bs) = bucket_sieve.as_mut() {
                    // ── Bulk cross-off via bucket sieve (cible #1) ──────────
                    // Each prime in this segment's bucket chain has exactly
                    // its next multiple located inside `[lo, hi)`. We drain
                    // the chain, run the standard wheel-30 cross-off (which
                    // handles 0-3 multiples per segment internally), then
                    // re-insert the prime into its new target segment.
                    let seg_id = ((lo - band_lo) / W30_SEG as u64) as usize;
                    let chain = bs.take_segment(seg_id);
                    if chain.is_some() {
                        let mut iter = BucketChainIter::new(chain);
                        while let Some(sp) = iter.next() {
                            let p  = sp.prime() as u64;
                            let mi = sp.multiple_index() as u64;
                            let wi = sp.wheel_index() as u8;
                            let pd = WheelPrimeData::new(p);
                            let (new_m, new_j) =
                                sieve.cross_off_pd_from_state(lo, p, &pd, lo + mi, wi);
                            // Schedule next cross-off if it lands within this band.
                            if new_m > z { continue; }
                            let new_seg = ((new_m - band_lo) / W30_SEG as u64) as usize;
                            if new_seg >= bucket_num_segs { continue; }
                            let new_seg_lo =
                                band_lo + (new_seg as u64) * W30_SEG as u64;
                            let new_mi = (new_m - new_seg_lo) as u32;
                            bs.insert(new_seg, p as u32, new_mi, new_j as u32);
                        }
                        bs.recycle_chain(iter.into_chain());
                    }
                } else {
                    // ── Linear sweep (production path, the historical bulk loop) ──
                    let t_active_scan = Instant::now();
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
                    if rest_bulk_profile {
                        stats.rest_bulk_active_scan_ns +=
                            t_active_scan.elapsed().as_nanos() as u64;
                        stats.n_bulk_segments += 1;
                        stats.n_bulk_target_sum += target_end as u64;
                    }
                    // Initialise persistent state for primes that just became
                    // active this segment (paid once per prime per band).
                    let t_state_init = Instant::now();
                    let init_start = bulk_state_valid_end;
                    while bulk_state_valid_end < target_end {
                        let k = bulk_state_valid_end;
                        let p = primes[c + b_ext + k] as u64;
                        let k0 = (lo + p - 1) / p;
                        let k1 = wheel30_next_k(k0);
                        bulk_next_m[k] = k1 * p;
                        bulk_next_j[k] = W30_IDX[(k1 % 30) as usize];
                        bulk_state_valid_end += 1;
                    }
                    if rest_bulk_profile {
                        stats.rest_bulk_state_init_ns +=
                            t_state_init.elapsed().as_nanos() as u64;
                        stats.n_bulk_state_inits +=
                            (bulk_state_valid_end - init_start) as u64;
                    }
                    // Cross-off with incremental state: no per-call 64-bit div.
                    // NB: `cross_off_pd_from_state_unrolled` (Phase 3) exists in
                    // segment.rs and is bit/state-exact, but switching to it
                    // here regressed `rest_bulk_xoff` by ~25 % at 1e15 α=1.
                    if rest_bulk_profile {
                        let t_xoff = Instant::now();
                        let mut k = 0usize;
                        while k < target_end {
                            let p0 = primes[c + b_ext + k] as u64;
                            let bin = rest_bulk_bin_for_p(p0);
                            let start = k;
                            let t_bin = Instant::now();
                            while k < target_end {
                                let p = primes[c + b_ext + k] as u64;
                                if rest_bulk_bin_for_p(p) != bin {
                                    break;
                                }
                                let (nm, nj) = sieve.cross_off_pd_from_state(
                                    lo, p, &pb_data[b_ext + k],
                                    bulk_next_m[k], bulk_next_j[k],
                                );
                                bulk_next_m[k] = nm;
                                bulk_next_j[k] = nj;
                                k += 1;
                            }
                            stats.bulk_bin_ns[bin] += t_bin.elapsed().as_nanos() as u64;
                            stats.bulk_bin_calls[bin] += (k - start) as u64;
                        }
                        stats.rest_bulk_xoff_ns += t_xoff.elapsed().as_nanos() as u64;
                        stats.n_bulk_xoff_calls += target_end as u64;
                    } else {
                        for k in 0..target_end {
                            let p = primes[c + b_ext + k] as u64;
                            let (nm, nj) = sieve.cross_off_pd_from_state(
                                lo, p, &pb_data[b_ext + k],
                                bulk_next_m[k], bulk_next_j[k],
                            );
                            bulk_next_m[k] = nm;
                            bulk_next_j[k] = nj;
                        }
                    }
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
                if tail_ext_split_enabled {
                    // Split-tail mode computes the whole ext-easy contribution
                    // after the sweep using `s2_primes.count_le(n)`, so the
                    // in-sieve tail emission is intentionally skipped.
                } else if band_is_heavy {
                    // 2-pass deferred path: capture (sieve.bits, p2_prefix,
                    // local_p2_offset, …) and skip the in-line ext-easy
                    // emission. Pass 2 will drain `deferred` once the light
                    // bands have finished, freeing all Rayon threads for the
                    // nested par_iter on `ei`.
                    sieve.fill_prefix_counts(&mut p2_prefix);
                    seed_below_lo = if lo < y {
                        primes.partition_point(|&p| p < lo)
                    } else { 0 };
                    stats.n_prefix_fills += 1;
                    p2_prefix_ready = true;
                    deferred.push(DeferredSeg {
                        bits:            *sieve.bits_array(),
                        p2_prefix,
                        lo,
                        local_p2_offset,
                        adj_lo,
                        seed_below_lo,
                    });
                } else {
                    // ── Ext-easy leaves (bi >= b_ext) ───────────────────────
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
                        let mut seed_cursor = seed_below_lo;
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
                                    while seed_cursor < primes.len() && primes[seed_cursor] <= n {
                                        seed_cursor += 1;
                                    }
                                    (seed_cursor - seed_below_lo) as i32
                                } else { 0 };
                                let v = local_p2_offset
                                    + raw as i64
                                    - adj_lo as i64
                                    + seed_in_query as i64;
                                let b_m1 = (b as i64) - 1;
                                // Piste 3 (the pl_idx cap at the non-clamp
                                // threshold pl ≤ x/(pb·p_{b-1})) guarantees
                                // every emitted leaf satisfies n ≥ p_{b-1},
                                // hence pi(n) ≥ b-1 and the closed form
                                // φ(n, b-1) = pi(n) - (b-2) is ≥ 1 with no
                                // clamp needed. Fold unconditionally.
                                ext_fold_partial +=
                                    (v - (b_m1 - 1)) as i128; // V - (b-2)
                                ext_fold_count += 1;
                                stats.n_ext_emitted += 1;
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
                }

                stats.tail_ext_ns += t_ext.elapsed().as_nanos() as u64;

                let t_p2 = Instant::now();
                // ── P2 queries ─────────────────────────────────────────────
                if !p2_walker.is_done() && p2_walker.rank() >= p2_min_rank {
                    let q_check = (x / p2_walker.p() as u128) as u64;
                    if q_check >= lo && q_check < hi {
                        let mut seed_cursor = seed_below_lo;
                        if !p2_prefix_ready {
                            let t_pref = Instant::now();
                            sieve.fill_prefix_counts(&mut p2_prefix);
                            seed_below_lo = if lo < y {
                                primes.partition_point(|&p| p < lo)
                            } else { 0 };
                            seed_cursor = seed_below_lo;
                            stats.tail_prefix_ns +=
                                t_pref.elapsed().as_nanos() as u64;
                            stats.n_prefix_fills += 1;
                            // p2_prefix_ready not reassigned: no further use
                            // in this segment (last section before advance).
                        }
                        loop {
                            if p2_walker.is_done() || p2_walker.rank() < p2_min_rank { break; }
                            let p   = p2_walker.p();
                            let j   = p2_walker.rank();
                            let q_k = (x / p as u128) as u64;
                            if q_k >= hi { break; }
                            if q_k < lo  { p2_walker.advance(); continue; }
                            let raw = sieve.count_primes_upto_int(&p2_prefix, q_k, lo);
                            let seed_in_query: i32 = if lo < y {
                                while seed_cursor < primes.len() && primes[seed_cursor] <= q_k {
                                    seed_cursor += 1;
                                }
                                (seed_cursor - seed_below_lo) as i32
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
                            p2_walker.advance();
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

            // ── Inline pass-2 replay (heavy bands only, §8.A) ────────────────
            // Drain `deferred` via nested par_iter on `ei` IMMEDIATELY, while
            // light bands (no tail_ext deferred) are still running their
            // single-pass sweep on other Rayon threads. Heavy bands finish
            // their pass-1 cross-off in ~5 s vs ~14 s for band 127 — by
            // launching pass-2 here, work-stealing routes ei tasks to threads
            // that finished their light bands early, instead of all 8 threads
            // waiting on the slowest band before pass-2 starts.
            if !tail_ext_split_enabled && band_is_heavy && !deferred.is_empty() {
                let (fp, fc, ne, ns) = (far_easy_start..n_easy)
                    .into_par_iter()
                    .map(|ei| {
                        let t_ei = Instant::now();
                        let (mut pl_idx, mut next_n) = init_easy(ei, band_lo);
                        if pl_idx >= a {
                            return (0i128, 0i64, 0u64, 0u64);
                        }
                        let bi = n_hard + ei;
                        let b  = bi + c + 1;
                        let pb = primes[b - 1];
                        let mut local_fp: i128 = 0;
                        let mut local_fc: i64  = 0;
                        let mut local_ne: u64 = 0;
                        let mut seed_cursor = if band_lo < y {
                            primes.partition_point(|&p| p < band_lo)
                        } else { 0 };

                        for seg in deferred.iter() {
                            if pl_idx >= a { break; }
                            let lo_seg = seg.lo;
                            let hi_seg = lo_seg + W30_SEG as u64;
                            if next_n >= hi_seg { continue; }
                            loop {
                                if pl_idx >= a { break; }
                                let n = next_n;
                                if n >= hi_seg { break; }
                                if n >= lo_seg {
                                    let raw = count_primes_in_segment(
                                        &seg.bits, &seg.p2_prefix, n, lo_seg);
                                    let seed_in_query: i32 = if lo_seg < y {
                                        while seed_cursor < primes.len() && primes[seed_cursor] <= n {
                                            seed_cursor += 1;
                                        }
                                        (seed_cursor - seg.seed_below_lo) as i32
                                    } else { 0 };
                                    let v = seg.local_p2_offset
                                        + raw as i64
                                        - seg.adj_lo as i64
                                        + seed_in_query as i64;
                                    let b_m1 = (b as i64) - 1;
                                    // See pass-1 site: Piste 3 guarantees
                                    // pi_n ≥ b-1, so the closed form
                                    // pi_n - (b-2) is ≥ 1 with no clamp.
                                    local_fp += (v - (b_m1 - 1)) as i128;
                                    local_fc += 1;
                                    local_ne += 1;
                                }
                                if pl_idx <= b {
                                    pl_idx = a;
                                    break;
                                }
                                pl_idx -= 1;
                                next_n =
                                    (x / (pb as u128 * primes[pl_idx] as u128)) as u64;
                            }
                        }
                        let ns = t_ei.elapsed().as_nanos() as u64;
                        (local_fp, local_fc, local_ne, ns)
                    })
                    .reduce(
                        || (0i128, 0i64, 0u64, 0u64),
                        |mut acc, b| {
                            acc.0 += b.0;
                            acc.1 += b.1;
                            acc.2 += b.2;
                            acc.3 += b.3;
                            acc
                        },
                    );
                ext_fold_partial    += fp;
                ext_fold_count      += fc;
                stats.n_ext_emitted += ne;
                stats.tail_ext_ns   += ns;
            }
            // `deferred` dropped at end of closure — no longer needed.

            (delta, p2_count, leaf_partial, bi_contrib,
             p2_partial, p2_q_count,
             ext_fold_partial, ext_fold_count,
             stats)
    };

    // ── Phase 1+2 scheduler ────────────────────────────────────────────────
    //
    // Phase 1: dynamic work-pool with hybrid ordering. Top-N ext_easy heavy
    // bands head the queue (their `tail_ext_easy_emit` cost dominates wall at
    // 1e18 α=2); the rest stays in natural `t` order to preserve cache
    // locality on `pb_data` for the consecutive-band sweep.
    //
    // Phase 2: optional sub-band chunking of the heaviest ext_easy bands,
    // gated by `RIVAT3_SUBDIVIDE_HEAVY` (number of bands) and
    // `RIVAT3_HEAVY_CHUNKS` (chunks per band). Default off (= Phase 1).
    // Subdivided bands get K work units each [chunk_lo, chunk_hi). The resolve
    // pass scans sub-chunk deltas in chunk_lo order to seed each chunk from the
    // previous one, so no large from-scratch phi recompute is needed.
    //
    // Opt-out via `RIVAT3_NO_WORKPOOL=1` (reverts to original par_iter).
    // ext_easy_weight: heavier ↦ band whose `tail_ext_easy_emit` is expected
    // to dominate. The dominant cost factor is the ext leaf count, which for
    // a thin band scales like ext_emit ∝ (1/blo - 1/bhi) ≈ span/(blo*bhi).
    // For low-blo bands this is huge; for high-blo (rest_bulk-dominated) bands
    // it's negligible. Band 0 starts at lo_start which may be 0 — but at large
    // x α=2 it is structurally light (most "very low n" leaves are clamped
    // and bulk-counted), so we exclude it from the head explicitly.
    //
    // Earlier attempt: filter `mid < x^(1/4)` was *too* restrictive. At
    // 1e18 α=2, x^(1/4) ≈ 31k but the heavy ext_easy bands sit at mid
    // ≈ 800k-1.8M, so every weight returned 0 → empty `head` → Phase 1
    // ordering was a no-op AND Phase 2 subdivision never triggered.
    let ext_easy_weight = |t: usize| -> u64 {
        let blo = band_bounds[t];
        let bhi = band_bounds[t + 1].min(z + W30_SEG as u64);
        if bhi <= blo { return 0; }
        if blo == 0 { return 0; }
        // 1/blo proxy, scaled to fit u64.
        u64::MAX / blo
    };

    // Tunable: how many heavy ext_easy bands get a head start. 3 matches the
    // 1e18 α=2 hot-spot (bands 1, 2, 3 each at 30-44 s solo).
    const N_HEAD: usize = 3;
    let mut head: Vec<usize> = (0..num_bands).collect();
    head.sort_by(|&a, &b| ext_easy_weight(b).cmp(&ext_easy_weight(a)));
    head.truncate(N_HEAD.min(num_bands));
    head.retain(|&t| ext_easy_weight(t) > 0);

    // Phase 2/3 subdivision config.
    //
    // - Phase 2 (`subdivide_heavy`): top-N ext_easy heavy bands (bands 1, 2, 3
    //   at 1e18 α=2 — the `tail_ext_easy_emit` hot-spot identified in
    //   session 2026-05-03). These are the LOW-blo bands.
    // - Phase 3 (`subdivide_rest_bulk`): top-N rest_bulk heavy bands (bands
    //   248-255 at 1e18 α=2 — the `rest_bulk_xoff` floor uncovered after
    //   Phase 2 cracked ext_easy). These are the HIGH-blo bands near z.
    //
    // Both subdivisions share `heavy_chunks` (K) — each heavy band, regardless
    // of regime, is split into K sub-chunks aligned to W30_SEG. The disjoint
    // nature of ext_easy heavy (low blo) and rest_bulk heavy (high blo) means
    // the two sets don't overlap at 1e18 α=2.
    let n_subdivide_ext  = crate::parameters::subdivide_heavy().min(head.len());
    let n_subdivide_bulk = crate::parameters::subdivide_rest_bulk();
    let chunks_per_heavy = crate::parameters::heavy_chunks();

    // rest_bulk_weight: span-dominated, since rest_bulk_xoff cost ∝ segments
    // (= span / W30_SEG). Excludes band 0 and empty bands.
    let rest_bulk_weight = |t: usize| -> u64 {
        let blo = band_bounds[t];
        let bhi = band_bounds[t + 1].min(z + W30_SEG as u64);
        if bhi <= blo { return 0; }
        if blo == 0 { return 0; }
        bhi - blo
    };

    let mut rest_bulk_heavy: Vec<usize> = (0..num_bands).collect();
    rest_bulk_heavy.sort_by(|&a, &b| rest_bulk_weight(b).cmp(&rest_bulk_weight(a)));
    rest_bulk_heavy.truncate(n_subdivide_bulk.min(num_bands));
    rest_bulk_heavy.retain(|&t| rest_bulk_weight(t) > 0);

    let mut subdivide_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    subdivide_set.extend(head.iter().take(n_subdivide_ext).copied());
    subdivide_set.extend(rest_bulk_heavy.iter().copied());

    // chunk_setup: derive (walker_start_n, p2_min_rank) from a chunk's
    // [chunk_lo, chunk_hi) — replays the formula used by `p2_band_setup`
    // but parametrised so it works for sub-chunks too.
    let chunk_setup = |chunk_lo: u64, chunk_hi: u64| -> (u64, usize) {
        let walker_start_n = if chunk_lo == 0 {
            s2_primes.hi()
        } else {
            ((x / chunk_lo as u128) as u64).min(s2_primes.hi())
        };
        let p2_min_rank = if chunk_hi == 0 {
            s2_primes.total()
        } else {
            s2_primes.count_le((x / chunk_hi as u128) as u64)
        };
        (walker_start_n, p2_min_rank)
    };

    #[derive(Clone, Copy)]
    struct WorkItem {
        band_id: usize,
        chunk_lo: u64,
        chunk_hi: u64,
        walker_start_n: u64,
        p2_min_rank: usize,
        is_heavy: bool,
    }

    let w_seg = W30_SEG as u64;
    let mut work_items: Vec<WorkItem> = Vec::with_capacity(
        num_bands + (n_subdivide_ext + n_subdivide_bulk) * chunks_per_heavy.saturating_sub(1)
    );
    for t in 0..num_bands {
        let band_lo = band_bounds[t];
        let band_hi = band_bounds[t + 1].min(z + W30_SEG as u64);
        if subdivide_set.contains(&t) && chunks_per_heavy > 1 && band_hi > band_lo {
            // Subdivide [band_lo, band_hi) into `chunks_per_heavy` ranges
            // aligned to W30_SEG. Sub-chunks always run single-pass; the
            // deferred-tail-ext path's pass-2 nested par_iter assumes the
            // band is processed as one stateful unit and breaks across
            // chunk boundaries.
            let span = band_hi - band_lo;
            let n_segs = (span + w_seg - 1) / w_seg;
            let segs_per_chunk =
                (n_segs + chunks_per_heavy as u64 - 1) / chunks_per_heavy as u64;
            let mut prev = band_lo;
            for k in 0..chunks_per_heavy {
                let chunk_hi = if k + 1 == chunks_per_heavy {
                    band_hi
                } else {
                    (prev + segs_per_chunk * w_seg).min(band_hi)
                };
                if chunk_hi <= prev { break; }
                let (walker_start_n, p2_min_rank) = chunk_setup(prev, chunk_hi);
                work_items.push(WorkItem {
                    band_id: t,
                    chunk_lo: prev,
                    chunk_hi,
                    walker_start_n,
                    p2_min_rank,
                    is_heavy: false,
                });
                prev = chunk_hi;
            }
        } else {
            let (walker_start_n, p2_min_rank) = chunk_setup(band_lo, band_hi);
            work_items.push(WorkItem {
                band_id: t,
                chunk_lo: band_lo,
                chunk_hi: band_hi,
                walker_start_n,
                p2_min_rank,
                is_heavy: is_heavy(t),
            });
        }
    }
    let n_items = work_items.len();

    // Item processing order. Items are sorted by predicted CPU cost descending,
    // so the longest items start first and the shortest items mop up at the end.
    // This addresses the natural-order failure mode at 1e18 α=2 K=8: the 8
    // rest_bulk heavies (bands 248-255, ~22 s solo each) sat at the *end* of
    // the queue, were picked up at t ≈ 30 s when only a fraction of workers
    // remained free, and stretched wall by another 22 s.
    //
    // Cost proxy combines two regimes:
    //   ext_easy cost ∝ x·span / (lo·hi)     # ≈ ext leaves emitted
    //   rest_bulk cost ∝ span                # linear sweep cost
    // Empirically both contribute additively to band wall; the calibration
    // factor (`* 7`) balances ext_easy and rest_bulk so that whole bands
    // and rest_bulks rank in the right order at 1e18 α=2.
    let predicted_cost = |item: &WorkItem| -> u64 {
        let lo = item.chunk_lo.max(W30_SEG as u64);
        let hi = item.chunk_hi.max(lo + 1);
        let span = hi - lo;
        let ext_easy = (x * span as u128) / (lo as u128 * hi as u128);
        let rest_bulk = span as u128;
        // Calibration: at 1e18 α=2, band 2 (whole, ext_easy) = 44 s, ext_easy
        // proxy = 3.2e11 → 7.3e9/s. Band 252 (rest_bulk) = 22 s, span = 22e9
        // → 1.0e9/s. Scale ext_easy by ×1/7 to align with rest_bulk seconds.
        let cost = ext_easy / 7 + rest_bulk;
        cost.min(u64::MAX as u128) as u64
    };

    let mut item_order: Vec<usize> = (0..n_items).collect();
    item_order.sort_by(|&a, &b| predicted_cost(&work_items[b]).cmp(&predicted_cost(&work_items[a])));

    let process_item = |item: &WorkItem| -> BandSweep {
        process_band(
            item.band_id,
            item.chunk_lo,
            item.chunk_hi,
            item.walker_start_n,
            item.p2_min_rank,
            item.is_heavy,
        )
    };

    let compute_tail_ext_range = |ei_lo: usize, ei_hi: usize| -> i128 {
        let mut range_sum: i128 = 0;
        for ei in ei_lo..ei_hi {
            let bi = n_hard + ei;
            let b = bi + c + 1;
            if b >= a || b < 2 {
                continue;
            }
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
            let clamped = upper_cnt.saturating_sub(nonclamp_cnt) as i128;
            let emit_cnt = nonclamp_cnt.min(upper_cnt);
            let mut sum = clamped;
            let b_minus_2 = (b as i128) - 2;
            for &pl in &primes[b..b + emit_cnt] {
                let n = (x / (pb * pl as u128)) as u64;
                let pi_n = if n <= y {
                    primes.partition_point(|&p| p <= n)
                } else {
                    a + s2_primes.count_le(n)
                } as i128;
                sum += pi_n - b_minus_2;
            }
            range_sum += sum;
        }
        range_sum
    };

    let tail_ext_range_cost = |ei_lo: usize, ei_hi: usize| -> u64 {
        let mut cost: u128 = 0;
        for ei in ei_lo..ei_hi {
            let bi = n_hard + ei;
            let b = bi + c + 1;
            if b >= a || b < 2 {
                continue;
            }
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
            cost += nonclamp_cnt.min(upper_cnt) as u128;
        }
        (cost * 24).min(u64::MAX as u128) as u64
    };

    let compute_split_tail_ext = || -> i128 {
        let t_split = Instant::now();
        let split_ext: i128 = (far_easy_start..n_easy)
            .into_par_iter()
            .map(|ei| compute_tail_ext_range(ei, ei + 1))
            .sum();
        if tail_ext_split_enabled {
            eprintln!(
                "[tail-ext-split] contribution={} elapsed={:.3}s",
                split_ext,
                t_split.elapsed().as_secs_f64(),
            );
        }
        split_ext
    };

    ck!("before par_iter (work_items)");

    // chunk_outputs: (band_id, chunk_lo, sweep), one per work item, in
    // work_items order (NOT processing order).
    let compute_chunk_outputs = || -> Vec<(usize, u64, BandSweep)> {
        if !crate::parameters::no_workpool() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cursor = AtomicUsize::new(0);
        let results: Vec<Mutex<Option<BandSweep>>> =
            (0..n_items).map(|_| Mutex::new(None)).collect();
        let n_workers = rayon::current_num_threads().max(1);

        rayon::scope(|s| {
            let process_item = &process_item;
            let work_items = &work_items;
            let item_order = &item_order;
            let cursor = &cursor;
            let results = &results;
            for _ in 0..n_workers {
                s.spawn(move |_| {
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= n_items { break; }
                        let item_idx = item_order[i];
                        let r = process_item(&work_items[item_idx]);
                        *results[item_idx].lock().unwrap() = Some(r);
                    }
                });
            }
        });

        results.into_iter().enumerate()
            .map(|(i, m)| {
                let sw = m.into_inner().unwrap()
                    .expect("workpool: every item slot must be filled exactly once");
                (work_items[i].band_id, work_items[i].chunk_lo, sw)
            })
            .collect()
        } else {
            (0..n_items).into_par_iter()
                .map(|i| {
                    let sw = process_item(&work_items[i]);
                    (work_items[i].band_id, work_items[i].chunk_lo, sw)
                })
                .collect()
        }
    };

    let (chunk_outputs, split_tail_ext) = if tail_ext_split_enabled {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone, Copy)]
        struct TailItem {
            ei_lo: usize,
            ei_hi: usize,
            cost: u64,
        }

        #[derive(Clone, Copy)]
        enum MixedItem {
            Sweep(usize),
            Tail(usize),
        }

        let tail_chunk = std::env::var("RIVAT3_TAIL_EXT_EI_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(8);
        let mut tail_items: Vec<TailItem> = Vec::new();
        let mut ei = far_easy_start;
        while ei < n_easy {
            let ei_hi = (ei + tail_chunk).min(n_easy);
            let cost = tail_ext_range_cost(ei, ei_hi);
            if cost > 0 {
                tail_items.push(TailItem { ei_lo: ei, ei_hi, cost });
            }
            ei = ei_hi;
        }

        let mut mixed_items: Vec<MixedItem> =
            Vec::with_capacity(n_items + tail_items.len());
        mixed_items.extend((0..n_items).map(MixedItem::Sweep));
        mixed_items.extend((0..tail_items.len()).map(MixedItem::Tail));
        let mut mixed_order: Vec<usize> = (0..mixed_items.len()).collect();
        mixed_order.sort_by(|&a, &b| {
            let cost_a = match mixed_items[a] {
                MixedItem::Sweep(i) => predicted_cost(&work_items[i]),
                MixedItem::Tail(i) => tail_items[i].cost,
            };
            let cost_b = match mixed_items[b] {
                MixedItem::Sweep(i) => predicted_cost(&work_items[i]),
                MixedItem::Tail(i) => tail_items[i].cost,
            };
            cost_b.cmp(&cost_a)
        });

        let t_split = Instant::now();
        let cursor = AtomicUsize::new(0);
        let sweep_results: Vec<Mutex<Option<BandSweep>>> =
            (0..n_items).map(|_| Mutex::new(None)).collect();
        let tail_results: Vec<Mutex<Option<i128>>> =
            (0..tail_items.len()).map(|_| Mutex::new(None)).collect();
        let n_workers = rayon::current_num_threads().max(1);

        rayon::scope(|s| {
            let mixed_items = &mixed_items;
            let mixed_order = &mixed_order;
            let work_items = &work_items;
            let tail_items = &tail_items;
            let process_item = &process_item;
            let compute_tail_ext_range = &compute_tail_ext_range;
            let cursor = &cursor;
            let sweep_results = &sweep_results;
            let tail_results = &tail_results;
            for _ in 0..n_workers {
                s.spawn(move |_| {
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= mixed_order.len() { break; }
                        match mixed_items[mixed_order[i]] {
                            MixedItem::Sweep(item_idx) => {
                                let r = process_item(&work_items[item_idx]);
                                *sweep_results[item_idx].lock().unwrap() = Some(r);
                            }
                            MixedItem::Tail(tail_idx) => {
                                let item = tail_items[tail_idx];
                                let r = compute_tail_ext_range(item.ei_lo, item.ei_hi);
                                *tail_results[tail_idx].lock().unwrap() = Some(r);
                            }
                        }
                    }
                });
            }
        });

        let chunk_outputs: Vec<(usize, u64, BandSweep)> = sweep_results.into_iter().enumerate()
            .map(|(i, m)| {
                let sw = m.into_inner().unwrap()
                    .expect("mixed workpool: every sweep slot must be filled exactly once");
                (work_items[i].band_id, work_items[i].chunk_lo, sw)
            })
            .collect();
        let split_ext: i128 = tail_results.into_iter()
            .map(|m| m.into_inner().unwrap()
                .expect("mixed workpool: every tail slot must be filled exactly once"))
            .sum();
        eprintln!(
            "[tail-ext-split] contribution={} elapsed={:.3}s items={} chunk={}",
            split_ext,
            t_split.elapsed().as_secs_f64(),
            tail_items.len(),
            tail_chunk,
        );
        (chunk_outputs, Some(split_ext))
    } else {
        (compute_chunk_outputs(), None)
    };

    ck!("after  par_iter collected");

    // Group chunks per band and sort each band's chunks by chunk_lo asc.
    let mut chunks_per_band: Vec<Vec<(u64, BandSweep)>> = (0..num_bands).map(|_| Vec::new()).collect();
    for (band_id, chunk_lo, sweep) in chunk_outputs {
        chunks_per_band[band_id].push((chunk_lo, sweep));
    }
    for v in chunks_per_band.iter_mut() {
        v.sort_by_key(|(lo, _)| *lo);
    }

    // Per-band aggregated delta (sum of sub-chunk deltas) and p2_count.
    // Both are commutative across sub-chunks.
    let delta_for_band: Vec<Vec<i64>> = chunks_per_band.iter().map(|chunks| {
        let mut d = vec![0i64; b_ext];
        for (_, sw) in chunks {
            for bi in 0..b_ext { d[bi] += sw.0[bi]; }
        }
        d
    }).collect();
    let p2_count_for_band: Vec<i64> = chunks_per_band.iter().map(|chunks| {
        chunks.iter().map(|(_, sw)| sw.1).sum()
    }).collect();

    // Sequential prefix-sum for phi / P2 init at each band_lo[t].
    let mut phi_band_inits: Vec<Vec<i64>> = vec![vec![0i64; b_ext]; num_bands];
    phi_band_inits[0] = initial_phi_vec;
    for t in 1..num_bands {
        for bi in 0..b_ext {
            phi_band_inits[t][bi] =
                phi_band_inits[t - 1][bi] + delta_for_band[t - 1][bi];
        }
    }
    let mut p2_band_inits = vec![initial_p2_offset; num_bands];
    for t in 1..num_bands {
        p2_band_inits[t] = p2_band_inits[t - 1] + p2_count_for_band[t - 1];
    }

    // Per-band stats aggregation (sum across sub-chunks) + global agg.
    let mut agg = BandStats::default();
    let mut per_band: Vec<BandProfile> = Vec::with_capacity(num_bands);
    for (t, chunks) in chunks_per_band.iter().enumerate() {
        let mut s = BandStats::default();
        for (_, sw) in chunks {
            let cs = &sw.8;
            s.fill_ns           += cs.fill_ns;
            s.bi_main_ns        += cs.bi_main_ns;
            s.bi_main_leaf_ns   += cs.bi_main_leaf_ns;
            s.rest_plain_ns     += cs.rest_plain_ns;
            s.rest_bulk_ns      += cs.rest_bulk_ns;
            s.rest_bulk_active_scan_ns += cs.rest_bulk_active_scan_ns;
            s.rest_bulk_state_init_ns  += cs.rest_bulk_state_init_ns;
            s.rest_bulk_xoff_ns        += cs.rest_bulk_xoff_ns;
            s.tail_prefix_ns    += cs.tail_prefix_ns;
            s.tail_ext_ns       += cs.tail_ext_ns;
            s.tail_p2_ns        += cs.tail_p2_ns;
            s.tail_advance_ns   += cs.tail_advance_ns;
            s.n_bi_leaf_hits    += cs.n_bi_leaf_hits;
            s.n_ext_emitted     += cs.n_ext_emitted;
            s.n_prefix_fills    += cs.n_prefix_fills;
            s.n_bulk_active_sum += cs.n_bulk_active_sum;
            s.n_bulk_xoff_calls += cs.n_bulk_xoff_calls;
            s.n_bulk_state_inits += cs.n_bulk_state_inits;
            s.n_bulk_segments += cs.n_bulk_segments;
            s.n_bulk_target_sum += cs.n_bulk_target_sum;
            for i in 0..REST_BULK_BINS {
                s.bulk_bin_ns[i] += cs.bulk_bin_ns[i];
                s.bulk_bin_calls[i] += cs.bulk_bin_calls[i];
            }
        }
        agg.fill_ns           += s.fill_ns;
        agg.bi_main_ns        += s.bi_main_ns;
        agg.bi_main_leaf_ns   += s.bi_main_leaf_ns;
        agg.rest_plain_ns     += s.rest_plain_ns;
        agg.rest_bulk_ns      += s.rest_bulk_ns;
        agg.rest_bulk_active_scan_ns += s.rest_bulk_active_scan_ns;
        agg.rest_bulk_state_init_ns  += s.rest_bulk_state_init_ns;
        agg.rest_bulk_xoff_ns        += s.rest_bulk_xoff_ns;
        agg.tail_prefix_ns    += s.tail_prefix_ns;
        agg.tail_ext_ns       += s.tail_ext_ns;
        agg.tail_p2_ns        += s.tail_p2_ns;
        agg.tail_advance_ns   += s.tail_advance_ns;
        agg.n_bi_leaf_hits    += s.n_bi_leaf_hits;
        agg.n_ext_emitted     += s.n_ext_emitted;
        agg.n_prefix_fills    += s.n_prefix_fills;
        agg.n_bulk_active_sum += s.n_bulk_active_sum;
        agg.n_bulk_xoff_calls += s.n_bulk_xoff_calls;
        agg.n_bulk_state_inits += s.n_bulk_state_inits;
        agg.n_bulk_segments += s.n_bulk_segments;
        agg.n_bulk_target_sum += s.n_bulk_target_sum;
        for i in 0..REST_BULK_BINS {
            agg.bulk_bin_ns[i] += s.bulk_bin_ns[i];
            agg.bulk_bin_calls[i] += s.bulk_bin_calls[i];
        }

        per_band.push(BandProfile {
            band_t: t,
            band_lo: band_bounds[t],
            band_hi: band_bounds[t + 1],
            fill_ns:           s.fill_ns,
            bi_main_ns:        s.bi_main_ns,
            bi_main_leaf_ns:   s.bi_main_leaf_ns,
            rest_plain_ns:     s.rest_plain_ns,
            rest_bulk_ns:      s.rest_bulk_ns,
            tail_prefix_ns:    s.tail_prefix_ns,
            tail_ext_ns:       s.tail_ext_ns,
            tail_p2_ns:        s.tail_p2_ns,
            tail_advance_ns:   s.tail_advance_ns,
            n_bi_leaf_hits:    s.n_bi_leaf_hits,
            n_ext_emitted:     s.n_ext_emitted,
            n_prefix_fills:    s.n_prefix_fills,
            n_bulk_active_sum: s.n_bulk_active_sum,
        });
    }

    ck!("before resolution pass");

    // Resolution pass: per chunk, combine local accumulators with the
    // appropriate (phi_init, p2_init) reference. For subdivided bands we avoid
    // recomputing phi_init from scratch at chunk_lo; instead, once the band's
    // global init is known, chunks are scanned in chunk_lo order and each
    // chunk's delta seeds the next chunk. This preserves the existing band
    // prefix invariant while making rest_bulk sub-chunks autonomous during
    // their parallel sweep.
    let t_resolve = Instant::now();
    let resolved: Vec<(i128, u128, i128)> = (0..num_bands)
        .into_par_iter()
        .map(|t| {
            let chunks = &chunks_per_band[t];
            let mut total_sum: i128 = 0;
            let mut total_p2: u128 = 0;
            let mut total_ext: i128 = 0;
            let mut phi_init = phi_band_inits[t].clone();
            let mut p2_init = p2_band_inits[t];
            for (_, sw) in chunks {
                let leaf_partial      = sw.2;
                let bi_contrib        = &sw.3;
                let p2_partial        = sw.4;
                let p2_q_count        = sw.5;
                let ext_fold_partial  = sw.6;
                let ext_fold_count    = sw.7;

                let mut sum: i128 = leaf_partial;
                for bi in 0..bi_contrib.len() {
                    sum += (bi_contrib[bi] as i128) * (phi_init[bi] as i128);
                }
                let ext_sum =
                    (p2_init as i128) * (ext_fold_count as i128) + ext_fold_partial;
                sum += ext_sum;
                total_ext += ext_sum;
                total_sum += sum;
                let p2_sum_i: i128 =
                    (p2_init as i128) * (p2_q_count as i128) + p2_partial;
                if p2_sum_i > 0 {
                    total_p2 += p2_sum_i as u128;
                }
                for bi in 0..b_ext {
                    phi_init[bi] += sw.0[bi];
                }
                p2_init += sw.1;
            }
            (total_sum, total_p2, total_ext)
        })
        .collect();
    let ns_resolve = t_resolve.elapsed().as_nanos() as u64;

    // s2_total = per-band folded sums + bulk clamp count (+1 each).
    let fused_s2_total: i128 =
        resolved.iter().map(|&(s, _, _)| s).sum::<i128>()
        + total_clamp_count as i128;
    let p2_total: u128 = resolved.iter().map(|&(_, p, _)| p).sum();
    let fused_tail_ext_contribution: i128 =
        resolved.iter().map(|&(_, _, e)| e).sum::<i128>()
        + total_clamp_count as i128;

    let tail_ext_contribution =
        split_tail_ext.unwrap_or(fused_tail_ext_contribution);
    let s2_total = fused_s2_total - fused_tail_ext_contribution
        + tail_ext_contribution;

    if std::env::var("RIVAT3_TAIL_EXT_SPLIT_CHECK").is_ok() {
        let t_split = Instant::now();
        let split_ext = split_tail_ext.unwrap_or_else(compute_split_tail_ext);
        if tail_ext_split_enabled {
            eprintln!(
                "[tail-ext-split-check] split={} fused=skipped elapsed={:.3}s",
                split_ext,
                t_split.elapsed().as_secs_f64(),
            );
        } else {
            eprintln!(
                "[tail-ext-split-check] fused={} split={} elapsed={:.3}s",
                fused_tail_ext_contribution,
                split_ext,
                t_split.elapsed().as_secs_f64(),
            );
            assert_eq!(
                split_ext,
                fused_tail_ext_contribution,
                "tail_ext split mismatch"
            );
        }
    }

    let profile = HardProfile {
        sweep_fill_ns:             agg.fill_ns,
        sweep_bi_main_ns:          agg.bi_main_ns,
        sweep_bi_main_leaf_ns:     agg.bi_main_leaf_ns,
        rest_plain_ns:             agg.rest_plain_ns,
        rest_bulk_ns:              agg.rest_bulk_ns,
        rest_bulk_detail: RestBulkProfile {
            enabled: rest_bulk_profile,
            active_scan_ns: agg.rest_bulk_active_scan_ns,
            state_init_ns: agg.rest_bulk_state_init_ns,
            xoff_ns: agg.rest_bulk_xoff_ns,
            xoff_calls: agg.n_bulk_xoff_calls,
            state_inits: agg.n_bulk_state_inits,
            segments: agg.n_bulk_segments,
            target_sum: agg.n_bulk_target_sum,
            bin_ns: agg.bulk_bin_ns,
            bin_calls: agg.bulk_bin_calls,
        },
        tail_prefix_build_ns:      agg.tail_prefix_ns,
        tail_ext_emit_ns:          agg.tail_ext_ns,
        tail_ext_contribution,
        tail_p2_emit_ns:           agg.tail_p2_ns,
        tail_advance_ns:           agg.tail_advance_ns,
        resolve_ns:                ns_resolve,
        n_bi_leaf_hits:            agg.n_bi_leaf_hits,
        n_leaves_ext_emitted:      agg.n_ext_emitted,
        n_leaves_ext_clamped:      total_clamp_count.max(0) as u64,
        n_prefix_fills:            agg.n_prefix_fills,
        n_bulk_active_primes_sum:  agg.n_bulk_active_sum,
        per_band,
    };
    ck!("end of s2_hard_sieve_par");
    (s2_total, p2_total, profile)
}
