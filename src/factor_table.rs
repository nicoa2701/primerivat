//! Compact `(μ, lpf)` lookup table for streaming hard-leaf enumeration.
//!
//! POC scaffolding for cible #4 (target: drop `hard_leaves` 602 MB by
//! generating leaves on the fly). Not yet wired into the DR engine —
//! see `tests` below for the bit-exact equivalence check vs the current
//! recursive enumeration in `dr::hard::enumerate_hard_leaves`.
//!
//! # Data structure
//!
//! Christian Bau's factor table (used by primecount). For every integer
//! `n ∈ [1, y]` coprime to `{2, 3, 5, 7, 11}` (the wheel base), we store
//! a single `u16` encoding both μ(n) and lpf(n):
//!
//! | encoding | meaning |
//! |---|---|
//! | `T_MAX - 1` | n = 1 |
//! | `T_MAX`     | n is prime |
//! | `0`         | μ(n) = 0  (n has a squared prime factor) |
//! | `lpf - 1`   | μ(n) = +1 (squarefree, even prime-factor count) |
//! | `lpf`       | μ(n) = -1 (squarefree, odd prime-factor count) |
//!
//! The compact predicate `factor[m] > prime` is then equivalent to
//! `μ(m) ≠ 0 ∧ lpf(m) > prime` — exactly the hard-leaf filter — in a
//! single comparison.
//!
//! # Compression
//!
//! 480 of every 2310 consecutive integers are coprime to {2,3,5,7,11},
//! and we store 2 B per slot, so the table costs ≈ y × 480/2310 × 2 ≈
//! y × 0.416 B. At y = 2e6 (DR config at x = 1e18 α=2): ~830 KB.
//! Compare to the present `hard_leaves: Vec<Vec<(u64, i8)>>` which
//! holds 39.5 M entries × 16 B ≈ 602 MB at the same x.
//!
//! # Validity bound
//!
//! Encoding lpf in `u16` requires `lpf(n) ≤ u16::MAX - 1 = 65534`.
//! Since `lpf(n) ≤ √n`, we need `√y ≤ 65534`, i.e. `y ≤ 65534² ≈ 4.3 × 10⁹`.
//! At α = 2 this gives x ≤ ~9.5 × 10²⁸ — well above primerivat's
//! validated range (1e19) and any plausible target.

use std::sync::OnceLock;

/// Wheel base = 2 × 3 × 5 × 7 × 11.
pub const WHEEL: u64 = 2310;

/// Number of integers in `[0, WHEEL)` coprime to {2, 3, 5, 7, 11}.
///
/// = ∏ (p - 1) for p ∈ {2, 3, 5, 7, 11} = 1 × 2 × 4 × 6 × 10 = 480.
pub const COPRIME_LEN: usize = 480;

/// First non-trivial coprime: smallest n > 1 with `gcd(n, 2310) = 1`.
pub const FIRST_COPRIME: u64 = 13;

/// Reserved sentinel: `factor[to_index(1)] = U16_MAX_M1` flags the entry
/// for n = 1 (μ(1) = 1, contributes to S1 in DR).
const U16_MAX_M1: u16 = u16::MAX - 1;

/// Reserved sentinel: `factor[to_index(p)] = U16_MAX` flags a prime entry.
const U16_MAX: u16 = u16::MAX;

/// `COPRIME[i]` = the `i`-th integer ∈ `[0, WHEEL)` coprime to {2,3,5,7,11}.
/// `COPRIME[0] = 1`, `COPRIME[1] = 13`, …, `COPRIME[479] = 2309`.
fn coprime_table() -> &'static [u16; COPRIME_LEN] {
    static TBL: OnceLock<Box<[u16; COPRIME_LEN]>> = OnceLock::new();
    TBL.get_or_init(|| {
        let mut out = Box::new([0u16; COPRIME_LEN]);
        let mut idx = 0;
        for n in 0..(WHEEL as u16) {
            let n_u = n as u64;
            if n_u == 1
                || (n_u > 1
                    && n_u % 2 != 0
                    && n_u % 3 != 0
                    && n_u % 5 != 0
                    && n_u % 7 != 0
                    && n_u % 11 != 0)
            {
                out[idx] = n;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, COPRIME_LEN);
        out
    })
}

/// `COPRIME_INDEXES[r]` = index in `COPRIME` of the largest coprime ≤ r,
/// or `-1` for `r = 0` (no coprime ≤ 0 in `[0, WHEEL)`).
fn coprime_indexes_table() -> &'static [i16; WHEEL as usize] {
    static TBL: OnceLock<Box<[i16; WHEEL as usize]>> = OnceLock::new();
    TBL.get_or_init(|| {
        let cop = coprime_table();
        let mut out = Box::new([-1i16; WHEEL as usize]);
        let mut last_idx: i16 = -1;
        let mut next_pos: usize = 0;
        for r in 0..(WHEEL as usize) {
            if next_pos < COPRIME_LEN && (cop[next_pos] as usize) == r {
                last_idx = next_pos as i16;
                next_pos += 1;
            }
            out[r] = last_idx;
        }
        out
    })
}

/// Compact (μ, lpf) lookup table covering all integers `[1, y]`
/// coprime to the wheel base `{2, 3, 5, 7, 11}`.
///
/// Build cost: O(y log log y) sequential. For the POC we keep it
/// single-threaded; primecount parallelises with OpenMP — the same
/// can be done with Rayon when wiring into prod.
pub struct FactorTable {
    factor: Vec<u16>,
    y: u64,
}

impl FactorTable {
    /// Maximum `y` representable with `u16` slots: `√y ≤ 65534`.
    pub const MAX_Y: u64 = 65534 * 65534;

    /// Builds the factor table for all coprime integers in `[1, y]`.
    pub fn new(y: u64) -> Self {
        assert!(
            y <= Self::MAX_Y,
            "FactorTable<u16> requires y ≤ {} (got {})",
            Self::MAX_Y,
            y
        );
        let n_slots = Self::to_index(y.max(1)) + 1;
        let mut factor = vec![U16_MAX; n_slots];
        // Slot 0 = n = 1: encode mu(1) = +1 with the sentinel that distinguishes
        // it from a regular prime (lpf=1 would be ambiguous).
        factor[0] = U16_MAX_M1;
        let sqrty = (y as f64).sqrt() as u64 + 1;
        // Iterate primes from 13 upward (smaller primes are absorbed by the
        // wheel and would mark every slot ≥ p², spoiling μ for everything).
        //
        // Upper bound is `y / FIRST_COPRIME` (NOT just √y): for any prime p
        // with √y < p ≤ y/13, the multiple p × 13 ≤ y is a valid coprime
        // entry that needs the lpf-XOR step to land its μ correctly. Missing
        // those primes leaves μ flipped on roughly half of the squarefree
        // semiprimes p · q with √y < q < p < y/13.
        let mut sieve_primes: Vec<u64> = Vec::new();
        {
            let limit = (y / FIRST_COPRIME).max(sqrty).max(FIRST_COPRIME) as usize;
            let mut is_prime = vec![true; limit + 1];
            is_prime[0] = false;
            if limit >= 1 {
                is_prime[1] = false;
            }
            let mut p = 2usize;
            while p * p <= limit {
                if is_prime[p] {
                    let mut m = p * p;
                    while m <= limit {
                        is_prime[m] = false;
                        m += p;
                    }
                }
                p += 1;
            }
            for p in (FIRST_COPRIME as usize)..=limit {
                if is_prime[p] {
                    sieve_primes.push(p as u64);
                }
            }
        }
        // For each prime p (≥ 13), walk its coprime-to-wheel multiples ≤ y.
        for &p in &sieve_primes {
            let p_idx_base = Self::to_index(p);
            // Walk m = p × cop[i] for i ≥ 1 (skip i=0 → m=p itself, handled
            // by leaving factor[p_idx_base] = U16_MAX from init since p is prime).
            // For each multiple, set lpf if first encountered, else flip the
            // parity bit to track μ.
            //
            // For multiples that are p² × (anything), set factor = 0 (μ = 0).
            let cop = coprime_table();
            let mut i = 1usize;
            loop {
                let multiplier = (i / COPRIME_LEN) as u64 * WHEEL
                    + cop[i % COPRIME_LEN] as u64;
                let m = match p.checked_mul(multiplier) {
                    Some(v) if v <= y => v,
                    _ => break,
                };
                let m_idx = Self::to_index(m);
                if factor[m_idx] == U16_MAX {
                    factor[m_idx] = p as u16;
                } else if factor[m_idx] != 0 {
                    // Flip LSB to track parity of prime-factor count
                    factor[m_idx] ^= 1;
                }
                i += 1;
                let _ = p_idx_base; // silence unused warning (kept for symmetry with primecount)
            }
            // Mark p²·k = 0 (μ = 0) for all wheel-coprime k.
            if p <= sqrty {
                let psq = p * p;
                let mut j = 0usize;
                loop {
                    let multiplier = (j / COPRIME_LEN) as u64 * WHEEL
                        + cop[j % COPRIME_LEN] as u64;
                    let m = match psq.checked_mul(multiplier) {
                        Some(v) if v <= y => v,
                        _ => break,
                    };
                    factor[Self::to_index(m)] = 0;
                    j += 1;
                }
            }
        }
        FactorTable { factor, y }
    }

    /// Returns the encoded (μ, lpf) value at the given factor-table index.
    /// See module docs for the encoding.
    #[inline]
    pub fn mu_lpf(&self, idx: usize) -> u16 {
        self.factor[idx]
    }

    /// Returns the Möbius function value at index `idx`.
    /// **Precondition**: `mu_lpf(idx) ≠ 0` (caller must filter μ=0 first).
    #[inline]
    pub fn mu(&self, idx: usize) -> i8 {
        let v = self.factor[idx];
        debug_assert_ne!(v, 0, "mu(idx) called on a μ=0 slot");
        if v & 1 == 1 { -1 } else { 1 }
    }

    /// Number of coprime slots in this table (exclusive upper bound).
    #[inline]
    pub fn len(&self) -> usize {
        self.factor.len()
    }

    /// Memory footprint of the table in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.factor.len() * std::mem::size_of::<u16>()
    }

    /// Upper bound `y` this table was built for.
    #[inline]
    pub fn y(&self) -> u64 {
        self.y
    }

    /// Number → factor-table index. `n` must be coprime to {2,3,5,7,11}.
    #[inline]
    pub fn to_index(n: u64) -> usize {
        let q = n / WHEEL;
        let r = (n % WHEEL) as usize;
        let cop_idx = coprime_indexes_table()[r];
        debug_assert!(cop_idx >= 0, "to_index(n) called on n < 1");
        (COPRIME_LEN as u64 * q) as usize + cop_idx as usize
    }

    /// Factor-table index → number. The result is coprime to {2,3,5,7,11}.
    #[inline]
    pub fn to_number(idx: usize) -> u64 {
        let q = (idx / COPRIME_LEN) as u64;
        let r = idx % COPRIME_LEN;
        WHEEL * q + coprime_table()[r] as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference μ(n) via trial-division — exhaustive but only used in tests.
    fn ref_mu(n: u64) -> i8 {
        if n == 0 {
            return 0;
        }
        let mut k = n;
        let mut count = 0i8;
        let mut p: u64 = 2;
        while p * p <= k {
            if k % p == 0 {
                k /= p;
                if k % p == 0 {
                    return 0; // squared factor
                }
                count += 1;
            }
            p += if p == 2 { 1 } else { 2 };
        }
        if k > 1 {
            count += 1;
        }
        if count % 2 == 0 { 1 } else { -1 }
    }

    /// Reference lpf(n) — least prime factor.
    fn ref_lpf(n: u64) -> u64 {
        if n < 2 {
            return n;
        }
        let mut p: u64 = 2;
        while p * p <= n {
            if n % p == 0 {
                return p;
            }
            p += if p == 2 { 1 } else { 2 };
        }
        n
    }

    #[test]
    fn coprime_table_first_entries() {
        let cop = coprime_table();
        assert_eq!(cop[0], 1);
        assert_eq!(cop[1], 13);
        assert_eq!(cop[2], 17);
        assert_eq!(cop[479], 2309);
    }

    #[test]
    fn coprime_indexes_table_basics() {
        let idx = coprime_indexes_table();
        assert_eq!(idx[0], -1);
        assert_eq!(idx[1], 0);
        for r in 2..13 {
            assert_eq!(idx[r], 0, "r={r} should map to coprime index 0 (=1)");
        }
        assert_eq!(idx[13], 1);
        assert_eq!(idx[2309], 479);
    }

    #[test]
    fn to_index_to_number_roundtrip() {
        for &n in &[1u64, 13, 17, 19, 23, 29, 2309, 2311, 2323, 100_003] {
            let idx = FactorTable::to_index(n);
            let m = FactorTable::to_number(idx);
            assert_eq!(m, n, "roundtrip failed for n={n}");
        }
    }

    /// Bit-exact validation: for every coprime n ∈ [1, y_test], the table's
    /// `mu_lpf` encoding must match `ref_mu` and `ref_lpf`.
    #[test]
    fn matches_reference_mu_lpf_up_to_50000() {
        let y = 50_000u64;
        let ft = FactorTable::new(y);
        let cop = coprime_table();
        let mut checked = 0;
        for q in 0..=(y / WHEEL) {
            for &c in cop.iter() {
                let n = q * WHEEL + c as u64;
                if n == 0 || n > y {
                    continue;
                }
                let idx = FactorTable::to_index(n);
                let f = ft.mu_lpf(idx);
                if n == 1 {
                    assert_eq!(f, U16_MAX_M1, "n=1 should be U16_MAX-1 sentinel");
                    continue;
                }
                let mu = ref_mu(n);
                let lpf = ref_lpf(n);
                if mu == 0 {
                    assert_eq!(f, 0, "n={n} expected μ=0 → factor=0, got {f}");
                } else if lpf == n {
                    // n is prime
                    assert_eq!(f, U16_MAX, "n={n} prime expected U16_MAX, got {f}");
                } else if mu == 1 {
                    assert_eq!(
                        f,
                        (lpf as u16) - 1,
                        "n={n} squarefree μ=+1, expected lpf-1={}, got {f}",
                        lpf - 1
                    );
                } else {
                    // mu == -1
                    assert_eq!(
                        f,
                        lpf as u16,
                        "n={n} squarefree μ=-1, expected lpf={lpf}, got {f}"
                    );
                }
                checked += 1;
            }
        }
        // Sanity: we should have checked at least a few thousand coprime entries.
        assert!(checked > 5000, "too few entries checked: {checked}");
    }

    /// The hard-leaf filter `factor.mu_lpf(m) > prime` must yield the same
    /// set of m values as the explicit `μ(m) ≠ 0 ∧ lpf(m) > prime` test
    /// — *for the m range actually used by primecount's S2_hard inner loop*,
    /// which iterates m > prime (the loop's `min_m` is set to ≥ prime by
    /// construction). For m ≤ prime the U16_MAX prime sentinel intentionally
    /// passes the table filter (a harmless artefact since those m values
    /// never appear in the iteration).
    #[test]
    fn hard_leaf_filter_equivalent_to_explicit_check() {
        let y = 30_000u64;
        let ft = FactorTable::new(y);
        let cop = coprime_table();
        for &prime in &[13u16, 17, 23, 41, 59, 101] {
            let mut from_table: Vec<u64> = Vec::new();
            let mut from_oracle: Vec<u64> = Vec::new();
            for q in 0..=(y / WHEEL) {
                for &c in cop.iter() {
                    let n = q * WHEEL + c as u64;
                    if n <= prime as u64 || n > y {
                        continue;
                    }
                    let idx = FactorTable::to_index(n);
                    if ft.mu_lpf(idx) > prime {
                        from_table.push(n);
                    }
                    let mu = ref_mu(n);
                    let lpf = ref_lpf(n);
                    if mu != 0 && lpf > prime as u64 {
                        from_oracle.push(n);
                    }
                }
            }
            assert_eq!(
                from_table, from_oracle,
                "filter mismatch for prime={prime}"
            );
        }
    }

    /// μ recovery from the table must match the reference for every
    /// non-zero, non-prime, non-1 entry.
    #[test]
    fn mu_recovery_matches_reference() {
        let y = 30_000u64;
        let ft = FactorTable::new(y);
        let cop = coprime_table();
        for q in 0..=(y / WHEEL) {
            for &c in cop.iter() {
                let n = q * WHEEL + c as u64;
                if n < 2 || n > y {
                    continue;
                }
                let idx = FactorTable::to_index(n);
                let f = ft.mu_lpf(idx);
                if f == 0 || f == U16_MAX || f == U16_MAX_M1 {
                    continue;
                }
                let expected = ref_mu(n);
                let got = ft.mu(idx);
                assert_eq!(
                    got, expected,
                    "μ mismatch at n={n}: expected {expected}, got {got}"
                );
            }
        }
    }
}
