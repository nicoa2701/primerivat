//! Wheel-30 packed bitset of primes over a contiguous range, with prefix
//! popcount index and a stateful descending walker.
//!
//! Designed to replace `Vec<u32>` storage of primes in the DR engine's
//! `s2_primes` slice. At x = 1e18 (range (y = 2e6, √x = 1e9]):
//!
//! | structure        | bytes  |
//! |------------------|-------:|
//! | `Vec<u32>`       | ~203 MB |
//! | `PrimeBitset`    |  ~34 MB (33 MB bits + ~260 KB prefix index) |
//!
//! # Layout
//!
//! Each `u64` word covers 240 integers (8 wheel-30 sub-windows × 30 numbers).
//! Bit `b` of word `w` represents the integer
//!   `240·w + 30·(b/8) + WHEEL_RESIDUES[b % 8]`
//! where `WHEEL_RESIDUES = [1, 7, 11, 13, 17, 19, 23, 29]`.
//!
//! Numbers not coprime to 30 (incl. the primes 2, 3, 5) are NOT representable.
//! Restricting the API to `lo ≥ 7` is the caller's responsibility; for the DR
//! `s2_primes` use case `lo = y ≥ 1e6`, so this is never a concern.
//!
//! # Access patterns
//!
//! Designed for the P2 sweep in `s2_hard_sieve_par`:
//!   * one [`count_le`](PrimeBitset::count_le) per band start (≤ 1 KB
//!     bsearch cost, called O(num_bands) times),
//!   * a descending [`Walker`] per band that yields `(prime, rank)` pairs
//!     in O(1) amortised per step.

/// The 8 residues coprime to 30 in `[0, 30)` — bit positions within a sub-window.
const WHEEL_RESIDUES: [u8; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

/// Map `r → j` such that `WHEEL_RESIDUES[j] == r`, or `255` if `r` is not
/// coprime to 30.
const WHEEL_IDX: [u8; 30] = [
    255, 0,   255, 255, 255, 255, 255, 1,
    255, 255, 255, 2,   255, 3,   255, 255,
    255, 4,   255, 5,   255, 255, 255, 6,
    255, 255, 255, 255, 255, 7,
];

/// `WHEEL_FLOOR[r]` = bit position (0..7) of the largest residue ≤ `r` in
/// `WHEEL_RESIDUES`, or `255` if `r < 1`.
const WHEEL_FLOOR: [u8; 30] = {
    let mut t = [255u8; 30];
    let mut last = 255u8;
    let mut d = 0;
    while d < 30 {
        if WHEEL_IDX[d] != 255 {
            last = WHEEL_IDX[d];
        }
        t[d] = last;
        d += 1;
    }
    t
};

const NUMBERS_PER_WORD: u64 = 240;
const WORDS_PER_BLOCK: usize = 64; // → 15 360 numbers per prefix block

/// `MASK_LEQ_240[rem]` = bitmask covering all bit positions in a word that
/// represent integers `≤ rem` within that word's 240-integer range.
const MASK_LEQ_240: [u64; 240] = {
    let mut out = [0u64; 240];
    let mut rem = 0;
    while rem < 240 {
        let group = rem / 30;
        let inner = rem - group * 30;
        let floor = WHEEL_FLOOR[inner];
        let nbits = if floor == 255 {
            (group * 8) as u32
        } else {
            (group * 8) as u32 + floor as u32 + 1
        };
        out[rem] = if nbits == 0 {
            0
        } else if nbits >= 64 {
            u64::MAX
        } else {
            (1u64 << nbits) - 1
        };
        rem += 1;
    }
    out
};

/// Wheel-30 packed bitset of primes over `[lo, hi]`, with prefix popcount
/// index and a descending walker. See module docs for layout and intended
/// access pattern.
pub struct PrimeBitset {
    lo: u64,
    hi: u64,
    bits: Vec<u64>,
    /// `prefix[k]` = popcount of `bits[0..k * WORDS_PER_BLOCK]` for
    /// `k ∈ [0, n_blocks]`. `prefix[n_blocks]` = total prime count.
    prefix: Vec<u32>,
    total: usize,
}

impl PrimeBitset {
    /// Build the bitset over `[lo, hi]` (both inclusive). `sieve_primes` must
    /// contain all primes ≤ `√hi` for the underlying segmented sieve.
    ///
    /// Primes 2, 3, 5 are never stored (not coprime to 30).
    pub fn new(lo: u64, hi: u64, sieve_primes: &[u64]) -> Self {
        assert!(lo <= hi, "PrimeBitset::new: lo {lo} > hi {hi}");
        assert!(hi <= u32::MAX as u64, "PrimeBitset::new: hi {hi} > u32::MAX");

        let n_words = (hi / NUMBERS_PER_WORD + 1) as usize;
        let mut bits = vec![0u64; n_words];

        if hi >= 7 {
            use crate::segment::{advance_small_primes, init_small_primes, SegSieve, SEG};
            let sieve_lo = (lo / SEG as u64) * SEG as u64;
            let mut sieve = SegSieve::new();
            let mut state = init_small_primes(sieve_primes, sieve_lo);
            let mut window_lo = sieve_lo;
            while window_lo <= hi {
                sieve.fill(window_lo, &state);
                for p in sieve.iter_primes(window_lo) {
                    if p < lo || p > hi || p < 7 { continue; }
                    let word = (p / NUMBERS_PER_WORD) as usize;
                    let inner = (p % NUMBERS_PER_WORD) as usize;
                    let group = inner / 30;
                    let digit = inner % 30;
                    let bit = 8 * group + WHEEL_IDX[digit] as usize;
                    bits[word] |= 1u64 << bit;
                }
                let next = window_lo + SEG as u64;
                advance_small_primes(&mut state, next);
                window_lo = next;
            }
        }

        // Build the prefix popcount index. `prefix[k+1]` accumulates the
        // popcount of all words in block `k` (= `bits[k * WORDS_PER_BLOCK
        // .. (k+1) * WORDS_PER_BLOCK]`), so a count_le(n) only needs to
        // popcount partial words inside the block containing `n`.
        let n_blocks = (n_words + WORDS_PER_BLOCK - 1) / WORDS_PER_BLOCK;
        let mut prefix = vec![0u32; n_blocks + 1];
        let mut acc: u32 = 0;
        for k in 0..n_blocks {
            let start = k * WORDS_PER_BLOCK;
            let end = (start + WORDS_PER_BLOCK).min(n_words);
            for w in start..end {
                acc += bits[w].count_ones();
            }
            prefix[k + 1] = acc;
        }

        Self { lo, hi, bits, prefix, total: acc as usize }
    }

    /// Number of primes in `[lo, n]`. Returns 0 for `n < lo` and `total()`
    /// for `n ≥ hi` (clamped).
    pub fn count_le(&self, n: u64) -> usize {
        if n < self.lo { return 0; }
        let n = n.min(self.hi);
        let word = (n / NUMBERS_PER_WORD) as usize;
        let block = word / WORDS_PER_BLOCK;
        let mut count = self.prefix[block] as usize;
        let block_start = block * WORDS_PER_BLOCK;
        for w in block_start..word {
            count += self.bits[w].count_ones() as usize;
        }
        let inner = (n % NUMBERS_PER_WORD) as usize;
        count += (self.bits[word] & MASK_LEQ_240[inner]).count_ones() as usize;
        count
    }

    /// Total primes stored in `[lo, hi]`.
    #[inline] pub fn total(&self) -> usize { self.total }
    #[inline] pub fn lo(&self) -> u64 { self.lo }
    #[inline] pub fn hi(&self) -> u64 { self.hi }

    /// Approximate memory footprint (bits + prefix index).
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
            + self.prefix.len() * std::mem::size_of::<u32>()
    }

    /// Initialise a descending walker pointing at the largest prime ≤
    /// `start_n`. Returns an exhausted walker (`is_done() == true`) if no
    /// such prime exists in this bitset.
    pub fn walker_at(&self, start_n: u64) -> Walker<'_> {
        if self.total == 0 || start_n < self.lo {
            return Walker { bitset: self, word_idx: 0, bit_idx: 0, rank: 0, done: true };
        }
        let n = start_n.min(self.hi);
        let word = (n / NUMBERS_PER_WORD) as usize;
        let inner = (n % NUMBERS_PER_WORD) as usize;
        let masked = self.bits[word] & MASK_LEQ_240[inner];
        let (word_idx, word_value) = if masked != 0 {
            (word, masked)
        } else {
            // Walk back word by word until we find a non-zero one.
            let mut w = word;
            loop {
                if w == 0 {
                    return Walker { bitset: self, word_idx: 0, bit_idx: 0, rank: 0, done: true };
                }
                w -= 1;
                let v = self.bits[w];
                if v != 0 { break (w, v); }
            }
        };
        let bit_idx = 63 - word_value.leading_zeros();
        let p = word_bit_to_p(word_idx, bit_idx);
        // count_le(p) includes p itself; subtract 1 for 0-indexed rank.
        let rank = self.count_le(p as u64) - 1;
        Walker { bitset: self, word_idx, bit_idx, rank, done: false }
    }
}

/// Decode `(word, bit) → integer` for the wheel-30 layout described in the
/// module docs.
#[inline]
fn word_bit_to_p(word: usize, bit: u32) -> u32 {
    let group = (bit / 8) as u32;
    let digit = (bit % 8) as usize;
    240 * word as u32 + 30 * group + WHEEL_RESIDUES[digit] as u32
}

/// Stateful descending iterator over the primes of a [`PrimeBitset`],
/// yielding `(prime, rank)` pairs from largest to smallest.
///
/// `rank` is the **0-indexed position** of the current prime within the
/// bitset (so the smallest prime has rank `0`, the largest has rank
/// `total() - 1`). Consumers needing absolute prime indices add
/// `π(lo - 1)` themselves.
pub struct Walker<'a> {
    bitset: &'a PrimeBitset,
    word_idx: usize,
    bit_idx: u32,
    rank: usize,
    done: bool,
}

impl<'a> Walker<'a> {
    #[inline] pub fn is_done(&self) -> bool { self.done }

    /// Current prime under the cursor. Calling on an exhausted walker is
    /// a logic error; the value returned is unspecified.
    #[inline]
    pub fn p(&self) -> u32 {
        debug_assert!(!self.done);
        word_bit_to_p(self.word_idx, self.bit_idx)
    }

    /// 0-indexed rank of the current prime within the bitset.
    #[inline]
    pub fn rank(&self) -> usize {
        debug_assert!(!self.done);
        self.rank
    }

    /// Move to the next smaller prime, or set `done` if exhausted.
    #[inline]
    pub fn advance(&mut self) {
        if self.done { return; }
        let mask = if self.bit_idx == 0 { 0 } else { (1u64 << self.bit_idx) - 1 };
        let masked = self.bitset.bits[self.word_idx] & mask;
        if masked != 0 {
            self.bit_idx = 63 - masked.leading_zeros();
            self.rank -= 1;
            return;
        }
        loop {
            if self.word_idx == 0 {
                self.done = true;
                return;
            }
            self.word_idx -= 1;
            let v = self.bitset.bits[self.word_idx];
            if v != 0 {
                self.bit_idx = 63 - v.leading_zeros();
                self.rank -= 1;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sieve::sieve_to;

    fn ref_primes_in(lo: u64, hi: u64) -> Vec<u64> {
        let (_, all) = sieve_to(hi);
        all.into_iter().filter(|&p| p >= lo && p <= hi && p >= 7).collect()
    }

    #[test]
    fn count_le_matches_reference_small() {
        let lo = 7u64;
        let hi = 5_000u64;
        let (_, sp) = sieve_to(hi);
        let bs = PrimeBitset::new(lo, hi, &sp);
        let refp = ref_primes_in(lo, hi);
        for n in (lo..=hi).step_by(7) {
            let expected = refp.iter().take_while(|&&p| p <= n).count();
            let got = bs.count_le(n);
            assert_eq!(got, expected, "count_le({n}) mismatch");
        }
        assert_eq!(bs.total(), refp.len());
    }

    #[test]
    fn count_le_clamps_below_lo_and_above_hi() {
        let (_, sp) = sieve_to(1000);
        let bs = PrimeBitset::new(100, 1000, &sp);
        let total = bs.total();
        assert_eq!(bs.count_le(0), 0);
        assert_eq!(bs.count_le(99), 0);
        assert_eq!(bs.count_le(1001), total);
        assert_eq!(bs.count_le(u32::MAX as u64), total);
    }

    #[test]
    fn count_le_at_each_prime_increments_by_one() {
        let lo = 7u64;
        let hi = 2_000u64;
        let (_, sp) = sieve_to(hi);
        let bs = PrimeBitset::new(lo, hi, &sp);
        let refp = ref_primes_in(lo, hi);
        for (i, &p) in refp.iter().enumerate() {
            assert_eq!(bs.count_le(p), i + 1, "at p={p}");
            if p > lo { assert_eq!(bs.count_le(p - 1), i, "just below p={p}"); }
        }
    }

    #[test]
    fn walker_descending_matches_reverse_reference() {
        let lo = 7u64;
        let hi = 10_000u64;
        let (_, sp) = sieve_to(hi);
        let bs = PrimeBitset::new(lo, hi, &sp);
        let refp = ref_primes_in(lo, hi);
        let mut w = bs.walker_at(hi);
        for (i, &expected) in refp.iter().enumerate().rev() {
            assert!(!w.is_done(), "walker exhausted early at i={i}");
            assert_eq!(w.p() as u64, expected, "wrong prime at descending pos {i}");
            assert_eq!(w.rank(), i, "wrong rank at descending pos {i}");
            w.advance();
        }
        assert!(w.is_done(), "walker should be done after smallest prime");
    }

    #[test]
    fn walker_starts_at_largest_prime_le_start() {
        let (_, sp) = sieve_to(1000);
        let bs = PrimeBitset::new(7, 1000, &sp);
        // 100 → largest prime ≤ 100 is 97
        let w = bs.walker_at(100);
        assert_eq!(w.p() as u64, 97);
        // exact prime → starts there
        let w = bs.walker_at(97);
        assert_eq!(w.p() as u64, 97);
        // composite right above prime → starts at prime
        let w = bs.walker_at(98);
        assert_eq!(w.p() as u64, 97);
    }

    #[test]
    fn walker_empty_when_no_prime_in_range() {
        let (_, sp) = sieve_to(100);
        // Range with no primes (24, 28] → only 25, 26, 27, 28 (none prime)
        let bs = PrimeBitset::new(24, 28, &sp);
        assert_eq!(bs.total(), 0);
        let w = bs.walker_at(28);
        assert!(w.is_done());
    }

    #[test]
    fn walker_done_when_start_below_lo() {
        let (_, sp) = sieve_to(1000);
        let bs = PrimeBitset::new(100, 1000, &sp);
        let w = bs.walker_at(50);
        assert!(w.is_done());
    }

    #[test]
    fn build_then_walk_large_range() {
        // Cross 1 M to exercise the prefix-block index.
        let lo = 1_000u64;
        let hi = 1_000_000u64;
        let (_, sp) = sieve_to(hi);
        let bs = PrimeBitset::new(lo, hi, &sp);
        let refp = ref_primes_in(lo, hi);
        assert_eq!(bs.total(), refp.len());

        // Spot-check count_le at random points and at some specific primes.
        for &n in &[lo, lo + 1, 100_003, 999_983, 999_999, hi] {
            let expected = refp.iter().take_while(|&&p| p <= n).count();
            assert_eq!(bs.count_le(n), expected, "count_le({n}) at large range");
        }

        // Walk from hi down a few hundred primes and check.
        let mut w = bs.walker_at(hi);
        for &expected in refp.iter().rev().take(500) {
            assert!(!w.is_done());
            assert_eq!(w.p() as u64, expected);
            w.advance();
        }
    }

    #[test]
    fn primes_2_3_5_never_stored() {
        let (_, sp) = sieve_to(100);
        let bs = PrimeBitset::new(0, 100, &sp);
        // count_le(5) excludes the wheel-base primes {2, 3, 5} entirely.
        assert_eq!(bs.count_le(5), 0);
        assert_eq!(bs.count_le(6), 0);
        // 7 is the first stored prime.
        assert_eq!(bs.count_le(7), 1);
    }

    #[test]
    fn walker_rank_consistent_with_count_le() {
        let lo = 7u64;
        let hi = 50_000u64;
        let (_, sp) = sieve_to(hi);
        let bs = PrimeBitset::new(lo, hi, &sp);
        // Sample several start points; for each, walker.rank == count_le(p) - 1.
        for &start in &[10u64, 100, 1_000, 10_000, 49_999, hi] {
            let w = bs.walker_at(start);
            if w.is_done() { continue; }
            assert_eq!(w.rank(), bs.count_le(w.p() as u64) - 1, "rank/count_le inconsistent at start={start}");
        }
    }
}
