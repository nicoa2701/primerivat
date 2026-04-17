use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::math::icbrt;
use crate::sieve::{extract_primes, lucy_hedgehog, lucy_hedgehog_with_phi, pi_at};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhiBackend {
    #[default]
    Lucy,
    Reference,
    ReferenceQuotient,
}

#[derive(Default)]
pub struct U128Hasher {
    state: u64,
}

impl U128Hasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        self.state ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.state = self
            .state
            .rotate_left(27)
            .wrapping_mul(0x94d0_49bb_1331_11eb);
    }
}

impl Hasher for U128Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunk = 0u64;
        let mut shift = 0u32;
        for &byte in bytes {
            chunk |= (byte as u64) << shift;
            shift += 8;
            if shift == 64 {
                self.mix(chunk);
                chunk = 0;
                shift = 0;
            }
        }
        if shift != 0 {
            self.mix(chunk);
        }
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.mix(i as u64);
        self.mix((i >> 64) as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
}

type FastMemoMap = HashMap<u128, u128, BuildHasherDefault<U128Hasher>>;

/// Mémo rapide pour φ(n, a) : clé (u128, usize), valeur u128.
/// Utilise le hasher U128Hasher (beaucoup plus rapide que SipHash sur ces clés).
pub type PhiMemoMap = HashMap<(u128, usize), u128, BuildHasherDefault<U128Hasher>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
struct QuotientPlateau {
    start: u128,
    end: u128,
    quotient: u128,
}

#[allow(dead_code)]
fn quotient_plateaus(n: u128, from: u128, to: u128) -> Vec<QuotientPlateau> {
    if from > to || to == 0 {
        return Vec::new();
    }

    let mut plateaus = Vec::new();
    let mut k = from.max(1);
    while k <= to {
        let q = n / k;
        let end = if q == 0 { to } else { (n / q).min(to) };
        plateaus.push(QuotientPlateau {
            start: k,
            end,
            quotient: q,
        });
        k = end.saturating_add(1);
    }

    plateaus
}

struct RecursivePhiEngine<'a> {
    x: u128,
    z: usize,
    primes: &'a [u64],
    small: &'a [u64],
    large: &'a [u64],
    memo: Vec<FastMemoMap>,
}

impl<'a> RecursivePhiEngine<'a> {
    fn new(
        x: u128,
        z: usize,
        primes: &'a [u64],
        small: &'a [u64],
        large: &'a [u64],
        a: usize,
    ) -> Self {
        let memo = (0..=a)
            .map(|depth| {
                let capacity = (a.saturating_sub(depth) + 1).max(8);
                FastMemoMap::with_capacity_and_hasher(
                    capacity,
                    BuildHasherDefault::<U128Hasher>::default(),
                )
            })
            .collect();
        Self {
            x,
            z,
            primes,
            small,
            large,
            memo,
        }
    }

    fn phi(&mut self, n: u128, a: usize) -> u128 {
        if n == 0 {
            return 0;
        }
        if let Some(value) = phi_small_a(n, a, self.primes) {
            return value;
        }

        let pa = self.primes[a - 1] as u128;
        if n < pa {
            return 1;
        }
        let next_leaf = self
            .primes
            .get(a)
            .map(|&p| (p as u128).saturating_mul(p as u128) > n)
            .unwrap_or_else(|| pa.saturating_mul(pa) > n);
        if next_leaf {
            let pi_n = pi_at(n, self.x, self.z, self.small, self.large) as u128;
            return 1 + pi_n - a as u128;
        }

        if let Some(&cached) = self.memo[a].get(&n) {
            return cached;
        }

        let result = self.phi(n, a - 1) - self.phi(n / pa, a - 1);
        self.memo[a].insert(n, result);
        result
    }

    fn phi_quotient_aware(&mut self, n: u128, a: usize) -> u128 {
        if n == 0 {
            return 0;
        }
        if let Some(value) = phi_small_a(n, a, self.primes) {
            return value;
        }

        let pa = self.primes[a - 1] as u128;
        if n < pa {
            return 1;
        }
        let next_leaf = self
            .primes
            .get(a)
            .map(|&p| (p as u128).saturating_mul(p as u128) > n)
            .unwrap_or_else(|| pa.saturating_mul(pa) > n);
        if next_leaf {
            let pi_n = pi_at(n, self.x, self.z, self.small, self.large) as u128;
            return 1 + pi_n - a as u128;
        }

        // Reuse the same memo store while experimenting with the plateau-aware branch.
        if let Some(&cached) = self.memo[a].get(&n) {
            return cached;
        }

        let q = n / pa;
        let lower = n / (q + 1) + 1;
        let upper = n / q;

        let prefix = &self.primes[..a];
        let plateau_start = prefix.partition_point(|&p| (p as u128) < lower);
        let plateau_end = prefix.partition_point(|&p| (p as u128) <= upper);

        let mut sum = 0u128;
        for j in plateau_start..plateau_end {
            sum += self.phi_quotient_aware(q, j);
        }

        let result = self.phi_quotient_aware(n, plateau_start) - sum;
        self.memo[a].insert(n, result);
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiComputation {
    pub y: u128,
    pub z: usize,
    pub a: usize,
    pub phi_x_a: u128,
    pub small: Vec<u64>,
    pub large: Vec<u64>,
    pub primes: Vec<u64>,
    pub backend: PhiBackend,
}

/// Computes the current φ(x, a) package through the Lucy-Hedgehog pipeline.
///
/// This isolates the existing "compute y, run Lucy, recover a and primes"
/// workflow behind a dedicated φ-oriented entry point so future backends can
/// be introduced without keeping that orchestration duplicated in callers.
pub fn phi_computation_with_backend(x: u128, backend: PhiBackend) -> PhiComputation {
    match backend {
        PhiBackend::Lucy => lucy_phi_computation(x),
        PhiBackend::Reference => reference_phi_computation(x),
        PhiBackend::ReferenceQuotient => reference_quotient_phi_computation(x),
    }
}

pub fn default_phi_computation(x: u128) -> PhiComputation {
    phi_computation_with_backend(x, PhiBackend::default())
}

pub fn lucy_phi_computation(x: u128) -> PhiComputation {
    let y = icbrt(x);
    let z = crate::math::isqrt(x) as usize;
    let (small, large, phi_x_a) = lucy_hedgehog_with_phi(x, y);
    let a = pi_at(y, x, z, &small, &large) as usize;
    let primes = extract_primes(&small, z);

    PhiComputation {
        y,
        z,
        a,
        phi_x_a,
        small,
        large,
        primes,
        backend: PhiBackend::Lucy,
    }
}

/// Computes the same φ(x, a) package using the direct recursive reference
/// implementation from this module, while reusing only the standard Lucy sieve
/// tables needed elsewhere by the current callers.
///
/// This backend is intentionally not performance-oriented yet; it serves as a
/// correctness-oriented scaffold for future non-Lucy φ work.
pub fn reference_phi_computation(x: u128) -> PhiComputation {
    let y = icbrt(x);
    let z = crate::math::isqrt(x) as usize;
    let (small, large) = lucy_hedgehog(x);
    let a = pi_at(y, x, z, &small, &large) as usize;
    let primes = extract_primes(&small, z);
    let mut engine = RecursivePhiEngine::new(x, z, &primes, &small, &large, a);
    let phi_x_a = engine.phi(x, a);

    PhiComputation {
        y,
        z,
        a,
        phi_x_a,
        small,
        large,
        primes,
        backend: PhiBackend::Reference,
    }
}

pub fn reference_quotient_phi_computation(x: u128) -> PhiComputation {
    let y = icbrt(x);
    let z = crate::math::isqrt(x) as usize;
    let (small, large) = lucy_hedgehog(x);
    let a = pi_at(y, x, z, &small, &large) as usize;
    let primes = extract_primes(&small, z);
    let mut engine = RecursivePhiEngine::new(x, z, &primes, &small, &large, a);
    let phi_x_a = engine.phi_quotient_aware(x, a);

    PhiComputation {
        y,
        z,
        a,
        phi_x_a,
        small,
        large,
        primes,
        backend: PhiBackend::ReferenceQuotient,
    }
}

#[inline]
fn phi_small_a(n: u128, a: usize, primes: &[u64]) -> Option<u128> {
    if a == 0 {
        return Some(n);
    }
    if a > 5 || a > primes.len() {
        return None;
    }

    let mut total = n as i128;
    let subset_limit = 1usize << a;
    for mask in 1..subset_limit {
        let mut bits = 0usize;
        let mut product = 1u128;
        for (i, &p) in primes.iter().take(a).enumerate() {
            if (mask >> i) & 1 == 1 {
                bits += 1;
                product = product.saturating_mul(p as u128);
            }
        }

        let term = (n / product) as i128;
        if bits % 2 == 1 {
            total -= term;
        } else {
            total += term;
        }
    }

    Some(total as u128)
}

/// Computes φ(x, a) — the count of integers in [1, x] not divisible
/// by any of the first `a` primes p₁, p₂, …, pₐ.
///
/// Uses the Legendre recursion (paper §2.1, eq. 1.3):
///   φ(x, 0) = x
///   φ(x, a) = φ(x, a−1) − φ(floor(x / pₐ), a−1)
///
/// Recursion is cut early (leaf cases) when:
///   • x < pₐ       → only 1 survives: φ = 1
///   • pₐ ≤ x < pₐ² → no composite with smallest prime factor ≥ pₐ
///                    fits in [1, x], so φ(x, a) = 1 + π(x) − a
///
/// Intermediate results are memoised in `memo` to avoid re-computing
/// identical sub-problems (which are common due to the binary recursion).
///
/// # Preconditions
/// - `primes[i]` = (i+1)-th prime (0-indexed: primes[0]=2, primes[1]=3, …)
/// - `a ≤ primes.len()`
/// - `pi_fn(n)` returns π(n) for any n reachable in the recursion
pub fn phi<F>(
    x: u128,
    a: usize,
    primes: &[u64],
    pi_fn: &F,
    memo: &mut HashMap<(u128, usize), u128>,
) -> u128
where
    F: Fn(u128) -> u64,
{
    // Base cases
    if x == 0 {
        return 0;
    }
    if let Some(value) = phi_small_a(x, a, primes) {
        return value;
    }

    // pₐ in the paper is 1-indexed; 0-indexed in `primes` it is primes[a−1].
    let pa = primes[a - 1] as u128;

    // Leaf case A: x < pₐ — every integer ≥ 2 in [1, x] has a prime factor
    // smaller than pₐ (since all primes ≤ x < pₐ are in {p₁,…,p_{a−1}}),
    // so only 1 remains coprime to {p₁,…,pₐ}.
    if x < pa {
        return 1;
    }

    // Leaf case B: pₐ ≤ x < pₐ² — any composite n ≤ x with smallest prime
    // factor ≥ pₐ would satisfy n ≥ pₐ² > x, contradiction. So the survivors
    // in [1, x] are exactly {1} ∪ {primes in (pₐ, x]}, giving:
    //   φ(x, a) = 1 + π(x) − a   (since π(pₐ) = a)
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > x)
        .unwrap_or_else(|| pa.saturating_mul(pa) > x);
    if next_leaf {
        let pi_x = pi_fn(x) as u128;
        // Invariant: pi_x ≥ a because pₐ ≤ x implies π(x) ≥ a.
        return 1 + pi_x - a as u128;
    }

    // Memoisation lookup
    if let Some(&cached) = memo.get(&(x, a)) {
        return cached;
    }

    // Legendre recursion
    let result = phi(x, a - 1, primes, pi_fn, memo) - phi(x / pa, a - 1, primes, pi_fn, memo);

    memo.insert((x, a), result);
    result
}

/// Traverses the Meissel φ recursion tree and collects all values `n` for
/// which `π(n)` will be needed at a leaf, with `n > cbrt_x` (i.e., outside
/// the precomputed small table).
///
/// # Why this is needed
/// The Meissel recursion for φ(x, a) terminates when `p_{a+1}² > n`.  At
/// such a leaf it calls `pi(n)`.  Values `n ≤ cbrt_x` are served by a small
/// sieve table; larger values must be fetched from the BIT sweep.  This
/// function discovers those larger values *before* the BIT sweep runs, so
/// they can be answered in a single combined pass.
///
/// # Complexity
/// With memoisation of visited `(n, a)` pairs, the traversal visits
/// O(x^(2/3) / log²x) unique pairs — the standard D-R bound.
pub fn collect_phi_leaf_queries(
    x: u128,
    a: usize,
    primes: &[u64],
    cbrt_x: u128,
    queries: &mut std::collections::HashSet<u128>,
) {
    let mut visited: HashMap<(u128, usize), ()> = HashMap::new();
    collect_phi_leaf_queries_rec(x, a, primes, cbrt_x, queries, &mut visited);
}

fn collect_phi_leaf_queries_rec(
    n: u128,
    a: usize,
    primes: &[u64],
    cbrt_x: u128,
    queries: &mut std::collections::HashSet<u128>,
    visited: &mut HashMap<(u128, usize), ()>,
) {
    if n == 0 {
        return;
    }
    // phi_small table covers all n ≤ cbrt_x — no π lookup needed there.
    if n <= cbrt_x {
        return;
    }
    // phi_small_a handles a ≤ 5 without any π lookup.
    if a <= 5 || a > primes.len() {
        return;
    }

    let pa = primes[a - 1] as u128;
    if n < pa {
        // φ(n, a) = 1 — no π lookup.
        return;
    }

    // Leaf condition: p_{a+1}² > n.
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > n)
        .unwrap_or_else(|| pa.saturating_mul(pa) > n);
    if next_leaf {
        // Leaf: needs π(n).  Record if n is large.
        if n > cbrt_x {
            queries.insert(n);
        }
        return;
    }

    // Avoid re-visiting the same (n, a) pair.
    if visited.insert((n, a), ()).is_some() {
        return;
    }

    collect_phi_leaf_queries_rec(n, a - 1, primes, cbrt_x, queries, visited);
    collect_phi_leaf_queries_rec(n / pa, a - 1, primes, cbrt_x, queries, visited);
}

/// Precomputes φ(n, k) for all n ≤ cbrt_x and all k ≤ a_full = seed_primes.len().
///
/// Layout: `table[k * (cbrt_x + 1) + n] = φ(n, k)`.
///
/// This allows phi_fast to answer any call with n ≤ cbrt_x in O(1), regardless
/// of the current recursion depth k, eliminating the deep sub-trees that formerly
/// bottlenecked the Meissel φ computation.
///
/// Build cost: O(a_full · cbrt_x) time and space.
/// At x = 10¹², a_full ≈ 1 229, cbrt_x ≈ 10 000 → ~49 MB.
pub fn build_phi_table(cbrt_x: u64, seed_primes: &[u64]) -> Vec<u32> {
    let c = cbrt_x as usize + 1; // row stride = cbrt_x + 1
    let a = seed_primes.len();
    let mut table = vec![0u32; (a + 1) * c];

    // k = 0: φ(n, 0) = n
    for n in 0..c {
        table[n] = n as u32;
    }

    // k = 1..=a: φ(n, k) = φ(n, k-1) − φ(⌊n/p_k⌋, k-1)
    for k in 1..=a {
        let p = seed_primes[k - 1] as usize;
        let prev = (k - 1) * c;
        let curr = k * c;
        for n in 0..c {
            let sub = if n >= p { table[prev + n / p] } else { 0 };
            table[curr + n] = table[prev + n] - sub;
        }
    }
    table
}

/// Precomputes φ(n, a_full) for all n ≤ cbrt_x (single-row version of
/// [`build_phi_table`]).  Only useful when the caller only needs the
/// fully-sieved φ value; prefer [`build_phi_table`] for the Meissel recursion.
pub fn build_phi_small(cbrt_x: u64, seed_primes: &[u64]) -> Vec<u32> {
    let n = cbrt_x as usize;
    let mut table: Vec<u32> = (0u32..=(n as u32)).collect();
    for &p in seed_primes {
        let p = p as usize;
        if p > n {
            break;
        }
        for i in (p..=n).rev() {
            table[i] -= table[i / p];
        }
    }
    table
}

/// Computes φ(x, a) using a precomputed table for small n and a two-tier π lookup.
///
/// # Arguments
/// - `phi_table`: built by [`build_phi_table`]; `phi_table[k*(cbrt_x+1)+n] = φ(n,k)`.
/// - `cbrt_x`: table covers n ≤ cbrt_x — any call with x ≤ cbrt_x is O(1).
/// - `small_pi`: π(n) for n ≤ cbrt_x (from [`crate::sieve::sieve_to`]).
/// - `large_pi`: π(n) for n > cbrt_x needed at leaf nodes (from the BIT sweep).
///
/// With the phi_table, every sub-problem with x ≤ ∛x is answered instantly,
/// dramatically reducing the recursion tree compared to the plain Legendre
/// recursion (which would recurse down to a ≤ 5 even for tiny x values).
pub fn phi_fast(
    x: u128,
    a: usize,
    primes: &[u64],
    phi_table: &[u32],
    cbrt_x: u128,
    small_pi: &[u32],
    large_pi: &std::collections::HashMap<u128, u64>,
    memo: &mut HashMap<(u128, usize), u128>,
) -> u128 {
    if x == 0 {
        return 0;
    }
    // TABLE LOOKUP: φ(x, a) for any x ≤ cbrt_x and any a ≤ a_full.
    // phi_table layout: row k has (cbrt_x+1) entries, phi_table[k*stride + x].
    if x <= cbrt_x {
        let stride = cbrt_x as usize + 1;
        // a is ≤ a_full (seed_primes.len()) in all recursive calls.
        return phi_table[a * stride + x as usize] as u128;
    }

    // Inclusion-exclusion formula for a ≤ 5 (no π call needed).
    if let Some(value) = phi_small_a(x, a, primes) {
        return value;
    }

    let pa = primes[a - 1] as u128;
    if x < pa {
        return 1;
    }
    // Leaf: p_{a+1}² > x  →  φ(x, a) = 1 + π(x) − a.
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > x)
        .unwrap_or_else(|| pa.saturating_mul(pa) > x);
    if next_leaf {
        // x > cbrt_x here, so use large_pi.
        let pi_x = *large_pi
            .get(&x)
            .unwrap_or_else(|| panic!("phi_fast: π({x}) not precomputed")) as u128;
        return 1 + pi_x - a as u128;
    }

    if let Some(&cached) = memo.get(&(x, a)) {
        return cached;
    }
    let result = phi_fast(x, a - 1, primes, phi_table, cbrt_x, small_pi, large_pi, memo)
        - phi_fast(x / pa, a - 1, primes, phi_table, cbrt_x, small_pi, large_pi, memo);
    memo.insert((x, a), result);
    result
}

/// φ(x, a) using dense O(1) medium-π lookups instead of a HashMap.
///
/// This is the v3 variant used by [`crate::dr::prime_pi_dr_v3`].  It differs
/// from [`phi_fast`] only in how leaf nodes resolve π(n):
///
/// | n range            | lookup          | cost   |
/// |--------------------|-----------------|--------|
/// | n ≤ cbrt_x         | `phi_table`     | O(1)   |
/// | cbrt_x < n ≤ x_23  | `medium_pi[n − medium_base]` | O(1) |
///
/// `medium_pi` is built by [`crate::dr::hard::s2_and_medium_pi`] in the same
/// combined sieve sweep that also computes S₂, so there is no extra cost.
///
/// # Panics (debug only)
/// If a leaf query for n falls outside `[medium_base, medium_base + medium_pi.len())`.
pub fn phi_fast_v3(
    x: u128,
    a: usize,
    primes: &[u64],
    phi_table: &[u32],
    cbrt_x: u128,
    medium_pi: &[u32],
    medium_base: u128,
    memo: &mut HashMap<(u128, usize), u128>,
) -> u128 {
    if x == 0 {
        return 0;
    }

    // ── O(1) table lookup for n ≤ cbrt_x ────────────────────────────────────
    if x <= cbrt_x {
        let stride = cbrt_x as usize + 1;
        return phi_table[a * stride + x as usize] as u128;
    }

    // ── Closed-form for a ≤ 5 ───────────────────────────────────────────────
    if let Some(value) = phi_small_a(x, a, primes) {
        return value;
    }

    let pa = primes[a - 1] as u128;
    if x < pa {
        return 1;
    }

    // ── Leaf: p_{a+1}² > x  →  φ(x, a) = 1 + π(x) − a ─────────────────────
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > x)
        .unwrap_or_else(|| pa.saturating_mul(pa) > x);
    if next_leaf {
        // x > cbrt_x here; leaf values are always in (cbrt_x, cbrt_x²].
        debug_assert!(
            x >= medium_base && (x - medium_base) < medium_pi.len() as u128,
            "phi_fast_v3: leaf n={x} out of medium_pi range [{medium_base}, {})",
            medium_base + medium_pi.len() as u128
        );
        let pi_x = medium_pi[(x - medium_base) as usize] as u128;
        return 1 + pi_x - a as u128;
    }

    // ── Memoised interior recursion ──────────────────────────────────────────
    if let Some(&cached) = memo.get(&(x, a)) {
        return cached;
    }
    let result = phi_fast_v3(x, a - 1, primes, phi_table, cbrt_x, medium_pi, medium_base, memo)
        - phi_fast_v3(
            x / pa,
            a - 1,
            primes,
            phi_table,
            cbrt_x,
            medium_pi,
            medium_base,
            memo,
        );
    memo.insert((x, a), result);
    result
}

/// Traverses the Meissel φ recursion tree and collects all `n > cbrt_x`
/// values needed at leaf nodes, using a flat-array bitset for visited
/// tracking instead of a HashMap.
///
/// `visited_flat` is a `Vec<bool>` of length `cbrt_x * flat_stride`
/// indexed by `d * flat_stride + a` for large-n nodes (d = denominator < cbrt_x).
/// `visited_overflow` handles medium-n nodes (d ≥ cbrt_x), keyed by (n, a)
/// to collapse plateau duplicates (multiple d with same ⌊x/d⌋ = n).
pub fn collect_phi_leaf_queries_flat(
    n: u128,
    a: usize,
    d: u128,
    x: u128,
    primes: &[u64],
    cbrt_x: u128,
    queries: &mut std::collections::HashSet<u128>,
    visited_flat: &mut Vec<bool>,
    flat_stride: usize,
    visited_overflow: &mut HashMap<(u128, usize), ()>,
) {
    if n == 0 {
        return;
    }
    if n <= cbrt_x {
        return;
    }
    if a <= 5 || a > primes.len() {
        return;
    }
    let pa = primes[a - 1] as u128;
    if n < pa {
        return;
    }
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > n)
        .unwrap_or_else(|| pa.saturating_mul(pa) > n);
    if next_leaf {
        if n > cbrt_x {
            queries.insert(n);
        }
        return;
    }

    // Mark visited to avoid redundant traversals.
    // Large-n nodes (d < cbrt_x): flat array indexed by d (unique per n).
    // Medium-n nodes (d ≥ cbrt_x): HashMap keyed by (n, a) to collapse plateaus.
    if d < cbrt_x {
        let idx = d as usize * flat_stride + a;
        if visited_flat[idx] {
            return;
        }
        visited_flat[idx] = true;
    } else {
        if visited_overflow.insert((n, a), ()).is_some() {
            return;
        }
    }

    collect_phi_leaf_queries_flat(
        n, a - 1, d, x, primes, cbrt_x, queries,
        visited_flat, flat_stride, visited_overflow,
    );
    collect_phi_leaf_queries_flat(
        n / pa, a - 1, d.saturating_mul(pa), x, primes, cbrt_x, queries,
        visited_flat, flat_stride, visited_overflow,
    );
}

/// φ(x, a) using a flat Vec<u64> memo for large-n nodes (d = ⌊x/n⌋ < cbrt_x).
///
/// # Parameters
/// - `n`, `a` — current sub-problem φ(n, a)
/// - `d` — denominator satisfying `n = ⌊x/d⌋` (passed explicitly for O(1) indexing)
/// - `x`, `primes`, `phi_table`, `cbrt_x` — same semantics as `phi_fast`
/// - `large_pi` — π(n) for leaf nodes with n > cbrt_x
/// - `flat_memo` — `vec![u64::MAX; cbrt_x * flat_stride]`, indexed by `d * flat_stride + a`
/// - `flat_stride` — `primes.len() + 1`
/// - `overflow_memo` — HashMap keyed by (n, a) for medium-n nodes (d ≥ cbrt_x)
pub fn phi_fast_flat(
    n: u128,
    a: usize,
    d: u128,
    x: u128,
    primes: &[u64],
    phi_table: &[u32],
    cbrt_x: u128,
    large_pi: &std::collections::HashMap<u128, u64>,
    flat_memo: &mut Vec<u64>,
    flat_stride: usize,
    overflow_memo: &mut HashMap<(u128, usize), u128>,
) -> u128 {
    if n == 0 {
        return 0;
    }
    // O(1) table lookup for n ≤ cbrt_x.
    if n <= cbrt_x {
        let stride = cbrt_x as usize + 1;
        return phi_table[a * stride + n as usize] as u128;
    }
    // Closed-form for a ≤ 5 (no π lookup needed).
    if let Some(value) = phi_small_a(n, a, primes) {
        return value;
    }
    let pa = primes[a - 1] as u128;
    if n < pa {
        return 1;
    }
    // Leaf check: p_{a+1}² > n.
    let next_leaf = primes
        .get(a)
        .map(|&p| (p as u128).saturating_mul(p as u128) > n)
        .unwrap_or_else(|| pa.saturating_mul(pa) > n);
    if next_leaf {
        let pi_n = *large_pi
            .get(&n)
            .unwrap_or_else(|| panic!("phi_fast_flat: π({n}) not in large_pi")) as u128;
        return 1 + pi_n - a as u128;
    }

    if d < cbrt_x {
        // Large-n node: flat array indexed by d.
        let idx = d as usize * flat_stride + a;
        if flat_memo[idx] != u64::MAX {
            return flat_memo[idx] as u128;
        }
        let result = phi_fast_flat(
            n, a - 1, d, x, primes, phi_table, cbrt_x, large_pi,
            flat_memo, flat_stride, overflow_memo,
        ) - phi_fast_flat(
            n / pa, a - 1, d.saturating_mul(pa), x, primes, phi_table, cbrt_x, large_pi,
            flat_memo, flat_stride, overflow_memo,
        );
        debug_assert!(result <= u64::MAX as u128, "phi_fast_flat: value exceeds u64::MAX");
        flat_memo[idx] = result as u64;
        result
    } else {
        // Medium-n node: HashMap keyed by (n, a) to collapse plateau duplicates.
        if let Some(&cached) = overflow_memo.get(&(n, a)) {
            return cached;
        }
        let result = phi_fast_flat(
            n, a - 1, d, x, primes, phi_table, cbrt_x, large_pi,
            flat_memo, flat_stride, overflow_memo,
        ) - phi_fast_flat(
            n / pa, a - 1, d.saturating_mul(pa), x, primes, phi_table, cbrt_x, large_pi,
            flat_memo, flat_stride, overflow_memo,
        );
        overflow_memo.insert((n, a), result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Minimal π function via binary search in a prime list (for unit tests only).
    fn make_pi(all_primes: &[u64]) -> impl Fn(u128) -> u64 + '_ {
        move |n: u128| all_primes.partition_point(|&p| (p as u128) <= n) as u64
    }

    #[test]
    fn test_phi_base_cases() {
        let primes = vec![2u64, 3, 5, 7, 11, 13, 17, 19];
        let pi = make_pi(&primes);
        let mut memo = HashMap::new();

        // φ(x, 0) = x
        assert_eq!(phi(0, 0, &primes, &pi, &mut memo), 0);
        assert_eq!(phi(1, 0, &primes, &pi, &mut memo), 1);
        assert_eq!(phi(10, 0, &primes, &pi, &mut memo), 10);

        // φ(1, a) = 1 for any a ≥ 1
        assert_eq!(phi(1, 1, &primes, &pi, &mut memo), 1);
        assert_eq!(phi(1, 3, &primes, &pi, &mut memo), 1);
    }

    #[test]
    fn test_phi_known_values() {
        let primes = vec![2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let pi = make_pi(&primes);
        let mut memo = HashMap::new();

        // φ(10, 1): integers ≤ 10 not divisible by 2 → {1,3,5,7,9} = 5
        assert_eq!(phi(10, 1, &primes, &pi, &mut memo), 5);

        // φ(100, 2): integers ≤ 100 not divisible by 2 or 3 = 33
        assert_eq!(phi(100, 2, &primes, &pi, &mut memo), 33);

        // φ(10, 2): integers ≤ 10 not divisible by 2 or 3 → {1,5,7} = 3
        assert_eq!(phi(10, 2, &primes, &pi, &mut memo), 3);
    }

    #[test]
    fn lucy_phi_computation_recovers_consistent_parameters() {
        let computation = lucy_phi_computation(1_000_000);

        assert_eq!(computation.y, 100);
        assert_eq!(computation.a, 25);
        assert_eq!(computation.primes[0], 2);
        assert_eq!(computation.primes[24], 97);
        assert!(computation.z >= computation.y as usize);
        assert!(!computation.small.is_empty());
        assert!(!computation.large.is_empty());
        assert!(computation.phi_x_a > 0);
        assert_eq!(computation.backend, PhiBackend::Lucy);
    }

    #[test]
    fn default_phi_backend_is_lucy() {
        let direct = lucy_phi_computation(100_000);
        let default = default_phi_computation(100_000);
        let explicit = phi_computation_with_backend(100_000, PhiBackend::Lucy);

        assert_eq!(default, direct);
        assert_eq!(explicit, direct);
    }

    #[test]
    fn reference_backend_matches_lucy_phi_value() {
        let lucy = lucy_phi_computation(1_000_000);
        let reference = reference_phi_computation(1_000_000);

        assert_eq!(reference.backend, PhiBackend::Reference);
        assert_eq!(reference.y, lucy.y);
        assert_eq!(reference.z, lucy.z);
        assert_eq!(reference.a, lucy.a);
        assert_eq!(reference.small, lucy.small);
        assert_eq!(reference.large, lucy.large);
        assert_eq!(reference.primes, lucy.primes);
        assert_eq!(reference.phi_x_a, lucy.phi_x_a);
    }

    #[test]
    fn quotient_reference_backend_matches_reference_phi_value() {
        let reference = reference_phi_computation(1_000_000);
        let quotient = reference_quotient_phi_computation(1_000_000);

        assert_eq!(quotient.backend, PhiBackend::ReferenceQuotient);
        assert_eq!(quotient.y, reference.y);
        assert_eq!(quotient.z, reference.z);
        assert_eq!(quotient.a, reference.a);
        assert_eq!(quotient.small, reference.small);
        assert_eq!(quotient.large, reference.large);
        assert_eq!(quotient.primes, reference.primes);
        assert_eq!(quotient.phi_x_a, reference.phi_x_a);
    }

    #[test]
    fn phi_small_a_matches_recursive_reference_up_to_five() {
        let primes = vec![2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let pi = make_pi(&primes);

        for a in 0..=5 {
            for x in [1u128, 10, 100, 1_000, 10_000, 100_000] {
                let mut memo = HashMap::new();
                let recursive = phi(x, a, &primes, &pi, &mut memo);
                let closed = phi_small_a(x, a, &primes).expect("small-a formula should exist");
                assert_eq!(closed, recursive, "mismatch for a={a}, x={x}");
            }
        }
    }

    #[test]
    fn quotient_plateaus_cover_range_without_gaps() {
        let plateaus = quotient_plateaus(100, 1, 20);

        assert_eq!(plateaus.first().unwrap().start, 1);
        assert_eq!(plateaus.last().unwrap().end, 20);

        let mut expected_start = 1u128;
        for plateau in &plateaus {
            assert_eq!(plateau.start, expected_start);
            assert!(plateau.start <= plateau.end);
            for k in plateau.start..=plateau.end {
                assert_eq!(100 / k, plateau.quotient);
            }
            expected_start = plateau.end + 1;
        }
    }

    #[test]
    fn quotient_plateaus_match_known_prefix() {
        let plateaus = quotient_plateaus(20, 1, 10);
        let expected = vec![
            QuotientPlateau {
                start: 1,
                end: 1,
                quotient: 20,
            },
            QuotientPlateau {
                start: 2,
                end: 2,
                quotient: 10,
            },
            QuotientPlateau {
                start: 3,
                end: 3,
                quotient: 6,
            },
            QuotientPlateau {
                start: 4,
                end: 4,
                quotient: 5,
            },
            QuotientPlateau {
                start: 5,
                end: 5,
                quotient: 4,
            },
            QuotientPlateau {
                start: 6,
                end: 6,
                quotient: 3,
            },
            QuotientPlateau {
                start: 7,
                end: 10,
                quotient: 2,
            },
        ];

        assert_eq!(plateaus, expected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhiCache-based φ computation (meissel_v2)
// ─────────────────────────────────────────────────────────────────────────────

/// φ(x, a) avec le cache Walisch et la phi_table pour les petits (n, a).
///
/// Utilisé dans `prime_pi_dr_meissel_v2`. Deux couches de cut-off :
///
/// | Condition                          | Mécanisme       | Coût  |
/// |------------------------------------|-----------------|-------|
/// | n ≤ max_x ET a ≤ max_a (=100)     | PhiCache dense  | O(1)  |
/// | n ≤ cbrt_x, a quelconque           | phi_table       | O(1)  |
/// | n > max_x, feuille B               | large_pi lookup | O(1)  |
/// | nœud intérieur                     | mémo HashMap    | O(1)  |
///
/// Le mémo ne contient que les nœuds avec n ∈ (cbrt_x, max_x] et a > max_a,
/// plus les nœuds avec n > max_x et feuille B non déclenchée.
/// En pratique bien plus petit que le mémo de `phi_fast`.
pub fn phi_tiny(n: u128, a: usize, primes: &[u64]) -> Option<u128> {
    phi_small_a(n, a, primes)
}

/// Computes S1 = Σ_{m squarefree, lpf(m) > p_c, m ≤ y} μ(m) · φ(x/m, c)
///
/// This is the "ordinary leaves" sum in the Deléglise-Rivat formula.
/// Together with `s2_hard_sieve`, it reconstructs φ(x, a) without the
/// Legendre recursion: S1 + S2_hard = φ(x, a).
///
/// # Parameters
/// - `x` : the prime-counting argument
/// - `y` : = ∛x (the sieve limit)
/// - `c` : number of tiny primes used in φ_tiny (must be ≤ 5)
/// - `primes` : all primes ≤ y in ascending order
pub fn s1_ordinary(x: u128, y: u64, c: usize, primes: &[u64]) -> i128 {
    debug_assert!(c <= 5);
    // Base: m = 1, μ(1) = +1
    let base = phi_small_a(x, c, primes).unwrap_or(x) as i128;
    let mut sum = base;
    // DFS over squarefree m = p_{i1}·…·p_{is} with c < i1 < … < is, m ≤ y
    s1_dfs(x, y, c, primes, c, 1u64, -1i128, &mut sum);
    sum
}

/// Recursive DFS helper for [`s1_ordinary`].
///
/// `start` is the index in `primes` of the next prime to consider (0-based).
/// `m` is the current squarefree product (initially 1).
/// `mu` is the Möbius sign to apply for the NEXT prime multiplied in (-1 initially).
fn s1_dfs(
    x: u128,
    y: u64,
    c: usize,
    primes: &[u64],
    start: usize,
    m: u64,
    mu: i128,
    sum: &mut i128,
) {
    for i in start..primes.len() {
        let p = primes[i];
        // nm = m * p; stop if nm > y or overflow
        let nm = match (m as u128).checked_mul(p as u128) {
            Some(v) if v <= y as u128 => v as u64,
            _ => break, // primes are ascending, so all future p' > p also fail
        };
        // Contribution: μ(nm) · φ(x/nm, c)
        *sum += mu * phi_small_a(x / nm as u128, c, primes).unwrap_or(x / nm as u128) as i128;
        // Recurse: next prime starts after i, sign flips
        s1_dfs(x, y, c, primes, i + 1, nm, -mu, sum);
    }
}
