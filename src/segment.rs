/// Window size: 2¹⁹ = 524 288 integers per segment.
/// Stored as 8 192 × u64 = 64 KiB — fits in L1 cache.
pub const SEG: usize = 1 << 19;

/// Segmented Eratosthenes sieve over a fixed window `[lo, lo + SEG)`.
///
/// `(bits[w] >> b) & 1 == 1`  ↔  `lo + 64*w + b` is prime.
///
/// Call [`SegSieve::fill`] for each new window.  The caller maintains the
/// per-prime starting multiples between windows (see [`init_small_primes`]).
pub struct SegSieve {
    bits: [u64; SEG / 64],
}

impl SegSieve {
    /// Creates a zeroed (all-composite) sieve.
    pub fn new() -> Self {
        Self {
            bits: [0u64; SEG / 64],
        }
    }

    /// Fills the window `[lo, lo + SEG)` using the supplied small primes.
    ///
    /// `small_primes[i] = (p, m)` where `m` is the first multiple of `p`
    /// that is ≥ `lo` **and** is composite (`m ≠ p`, i.e. `m ≥ 2p` when
    /// `lo ≤ p`, or the first multiple ≥ `lo` otherwise).
    ///
    /// Use [`init_small_primes`] to build this slice for the first window,
    /// then advance each `m` by `p` repeatedly until `m ≥ lo + SEG` before
    /// calling `fill` on the next window.
    pub fn fill(&mut self, lo: u64, small_primes: &[(u64, u64)]) {
        // All positions start as prime candidates.
        self.bits.fill(!0u64);

        // 0 and 1 are not prime.
        if lo == 0 {
            self.bits[0] &= !0b11u64; // clear bits 0 (→ 0) and 1 (→ 1)
        } else if lo == 1 {
            self.bits[0] &= !1u64; // clear bit 0 (→ 1)
        }

        for &(p, mut m) in small_primes {
            debug_assert!(m >= lo, "starting multiple must be ≥ lo");
            debug_assert!(m == 0 || m % p == 0, "m must be a multiple of p");
            while m < lo + SEG as u64 {
                let i = (m - lo) as usize;
                self.bits[i / 64] &= !(1u64 << (i % 64));
                m += p;
            }
        }
    }

    /// Returns `true` if position `lo + i` is currently marked as prime.
    #[inline]
    pub fn is_set(&self, i: usize) -> bool {
        debug_assert!(i < SEG);
        (self.bits[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Sets bit `i` (marks position `lo + i` as coprime / prime).
    ///
    /// Used by the phi sieve to mark integer 1 as coprime after a
    /// `fill(0, …)` call clears it.
    #[inline]
    pub fn set_bit(&mut self, i: usize) {
        debug_assert!(i < SEG);
        self.bits[i / 64] |= 1u64 << (i % 64);
    }

    /// Returns the raw bitset words (for total-count operations, etc.).
    #[inline]
    pub fn bits(&self) -> &[u64; SEG / 64] {
        &self.bits
    }

    /// Fills a prefix-popcount table for fast prime counting within this window.
    ///
    /// After the call, `out[j]` = number of set bits in `self.bits[0..j]`.
    /// - `out[0]`        = 0
    /// - `out[SEG / 64]` = total primes in `[lo, lo + SEG)`
    ///
    /// `out` must have length ≥ `SEG / 64 + 1`.  The caller typically
    /// allocates this once and reuses it across block iterations.
    pub fn fill_prefix_counts(&self, out: &mut [u32]) {
        debug_assert!(out.len() >= SEG / 64 + 1);
        out[0] = 0;
        for j in 0..SEG / 64 {
            out[j + 1] = out[j] + self.bits[j].count_ones();
        }
    }

    /// Counts primes at positions `0..=local` within the current window,
    /// using the table produced by [`fill_prefix_counts`].
    ///
    /// `local` is 0-indexed: 0 corresponds to `lo`, `SEG − 1` to
    /// `lo + SEG − 1`.  Must satisfy `local < SEG`.
    #[inline]
    pub fn count_primes_upto(&self, prefix: &[u32], local: usize) -> u32 {
        debug_assert!(local < SEG);
        let wi = local / 64;
        let bi = local % 64;
        // Select bits 0..=bi: u64::MAX >> (63-bi).
        // Works for bi = 0..=63 without a branch.
        let mask = u64::MAX >> (63 - bi);
        prefix[wi] + (self.bits[wi] & mask).count_ones()
    }

    /// Returns the total number of set bits in the current window.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.bits.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Crosses off all multiples of `p` in the window `[lo, lo + SEG)`.
    ///
    /// Used by the S2_hard sieve to progressively remove hard primes so that
    /// `fill_prefix_counts` reflects φ(·, b−1) for increasing b.
    pub fn cross_off(&mut self, lo: u64, p: u64) {
        // First multiple of p that is ≥ lo.
        let start = {
            let rem = lo % p;
            if rem == 0 { lo } else { lo + p - rem }
        };
        let mut m = start;
        let end = lo + SEG as u64;
        while m < end {
            let i = (m - lo) as usize;
            self.bits[i / 64] &= !(1u64 << (i % 64));
            m += p;
        }
    }

    /// Crosses off all multiples of `p` in `[lo, lo + SEG)` and returns the
    /// number of bits that were actually cleared (i.e. were set before).
    ///
    /// Combines `cross_off` with a count of removed elements, allowing the
    /// caller to maintain a running total without a separate `total_count()`
    /// call (which costs O(SEG/64)).
    #[inline]
    pub fn cross_off_count(&mut self, lo: u64, p: u64) -> u64 {
        let start = {
            let rem = lo % p;
            if rem == 0 { lo } else { lo + p - rem }
        };
        let mut m = start;
        let end = lo + SEG as u64;
        let mut cleared = 0u64;
        while m < end {
            let i = (m - lo) as usize;
            let bit = (self.bits[i / 64] >> (i % 64)) & 1;
            self.bits[i / 64] &= !(1u64 << (i % 64));
            cleared += bit;
            m += p;
        }
        cleared
    }

    /// Iterates over all primes in `[lo, lo + SEG)` in ascending order.
    ///
    /// Uses `trailing_zeros` to skip over zero words in O(1).
    pub fn iter_primes(&self, lo: u64) -> impl Iterator<Item = u64> + '_ {
        self.bits
            .iter()
            .enumerate()
            .flat_map(move |(wi, &word)| WordPrimes {
                word,
                base: lo + (wi as u64) * 64,
            })
    }
}

impl Default for SegSieve {
    fn default() -> Self {
        Self::new()
    }
}

/// Drains set bits from a single 64-bit word, yielding `base + bit_index`.
struct WordPrimes {
    word: u64,
    base: u64,
}

impl Iterator for WordPrimes {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.word == 0 {
            return None;
        }
        let bit = self.word.trailing_zeros() as u64;
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + bit)
    }
}

/// Builds the `(p, m)` pairs needed by [`SegSieve::fill`] for the window
/// starting at `lo`.
///
/// For each prime `p`:
/// - If `lo ≤ p`: the first composite multiple in `[lo, ∞)` is `p²`
///   (all smaller multiples have a smaller prime factor already handled).
/// - If `lo > p`: the first multiple of `p` that is `≥ lo`.
///
/// Returns `Vec<(p, m)>` ready for [`SegSieve::fill`].
pub fn init_small_primes(primes: &[u64], lo: u64) -> Vec<(u64, u64)> {
    primes
        .iter()
        .map(|&p| {
            let m = if lo <= p {
                p * p
            } else {
                let rem = lo % p;
                if rem == 0 { lo } else { lo + p - rem }
            };
            (p, m)
        })
        .collect()
}

/// Advances the `(p, m)` state so each `m` is the first tracked multiple
/// at or after `next_lo`.
pub fn advance_small_primes(small_primes: &mut [(u64, u64)], next_lo: u64) {
    for (p, m) in small_primes.iter_mut() {
        while *m < next_lo {
            *m += *p;
        }
    }
}

// ─── Wheel-30 sieve ──────────────────────────────────────────────────────────
//
// Stores only integers coprime to {2, 3, 5}, i.e. those ≡ 1,7,11,13,17,19,23,29 (mod 30).
// For a window [lo, lo + W30_SEG) with lo ≡ 0 (mod 30):
//
//   group g ∈ [0, W30_GROUPS):  covers integers lo + g*30 + W30_RESIDUES[j], j=0..8
//   bit_idx = g * 8 + j          where j = W30_IDX[(n % 30) as usize]
//   word    = bit_idx / 64,  bit  = bit_idx % 64
//
// Sieve density: 8/30 ≈ 26.7 %, reducing 64 KiB (SegSieve) → ≈ 17 KiB (WheelSieve30).

/// Covering window in integers (multiple of 30).
pub const W30_SEG: usize = (SEG / 30) * 30;       // 524 280
const W30_GROUPS: usize = W30_SEG / 30;            // 17 476
const W30_BITS: usize = W30_GROUPS * 8;            // 139 808
/// u64 words needed to store W30_BITS bits.
pub const W30_WORDS: usize = (W30_BITS + 63) / 64; // 2 185

/// The 8 residues coprime to 30 in [0, 30).
const W30_RESIDUES: [u8; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

/// Kim-style mask table for the popcount hot path. `rem240` indexes directly
/// inside one 240-integer block (= one u64 word), avoiding the old `/ 30`,
/// `% 30`, wheel-position lookup, and bit-index reconstruction sequence.
const W30_MASK_LEQ_240: [u64; 240] = build_w30_mask_leq_240();

const fn w30_j_for_rem(rem: usize) -> usize {
    if rem < 1 {
        0
    } else if rem < 7 {
        1
    } else if rem < 11 {
        2
    } else if rem < 13 {
        3
    } else if rem < 17 {
        4
    } else if rem < 19 {
        5
    } else if rem < 23 {
        6
    } else if rem < 29 {
        7
    } else {
        8
    }
}

const fn w30_mask_leq_240(rem240: usize) -> u64 {
    let group = rem240 / 30;
    let rem = rem240 - group * 30;
    let j = w30_j_for_rem(rem);
    let nbits = group * 8 + j;
    if nbits == 0 {
        0
    } else if nbits >= 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    }
}

const fn build_w30_mask_leq_240() -> [u64; 240] {
    let mut out = [0u64; 240];
    let mut i = 0;
    while i < 240 {
        out[i] = w30_mask_leq_240(i);
        i += 1;
    }
    out
}

/// Map r → j such that W30_RESIDUES[j] == r  (255 if r is not coprime to 30).
pub const W30_IDX: [u8; 30] = [
    255, 0,   255, 255, 255, 255, 255, 1,
    255, 255, 255, 2,   255, 3,   255, 255,
    255, 4,   255, 5,   255, 255, 255, 6,
    255, 255, 255, 255, 255, 7,
];

/// Gap in k (multiple index) from wheel position j to the next position (j+1) % 8.
/// Residues: 1 → 7 → 11 → 13 → 17 → 19 → 23 → 29 → 31 → …
/// Gaps:          6    4    2    4    2    4    6    2  (sum = 30 per cycle)
const WHEEL30_GAPS: [u8; 8] = [6, 4, 2, 4, 2, 4, 6, 2];

/// Increment to add to k (when k % 30 == r) to reach the next coprime-to-30 k.
/// `WHEEL30_NEXT[r]` is 0 when r is already coprime to 30.
pub const WHEEL30_NEXT: [u8; 30] = [
    1, 0, 5, 4, 3, 2, 1, 0,  // r = 0 ..  7
    3, 2, 1, 0, 1, 0, 3, 2,  // r = 8 .. 15
    1, 0, 1, 0, 3, 2, 1, 0,  // r = 16 .. 23
    5, 4, 3, 2, 1, 0,         // r = 24 .. 29
];

/// Returns the smallest k' ≥ k such that k' is coprime to 30.
#[inline]
pub fn wheel30_next_k(k: u64) -> u64 {
    k + WHEEL30_NEXT[(k % 30) as usize] as u64
}

/// Per-prime precomputed constants for wheel-30 crossing-off.
///
/// Compute once with [`WheelPrimeData::new(p)`], then pass to
/// [`WheelSieve30::cross_off_count_pd`] for every window.  This amortises
/// the 40-operation precomputation cost (8 multiplications, 8 divisions,
/// 8 table lookups) over all windows instead of repeating it each call.
#[derive(Clone)]
pub struct WheelPrimeData {
    /// Residue of (k*p) mod 30 at each k-wheel position j (0..8).
    /// Values are 0-29, so u8 suffices.
    pub w30res_p: [u8; 8],
    /// Bit-within-group for m = k*p at k-position j (= W30_IDX[w30res_p[j]]).
    /// Values are 0-7, so u8 suffices.
    pub bit_seq: [u8; 8],
    /// Advance in m to the next coprime-to-30 multiple (= WHEEL30_GAPS[j]*p).
    /// Max value = 6*p; u32 suffices for p ≤ 715_000_000 (well beyond our needs).
    pub gap_m: [u32; 8],
    /// Group advance corresponding to gap_m[j]
    /// (= (w30res_p[j] + gap_m[j] − w30res_p[(j+1)%8]) / 30, always ≥ 0).
    /// Max value = gap_m_max/30; u32 suffices for p ≤ 2_100_000_000.
    pub delta_group: [u32; 8],
}

impl WheelPrimeData {
    /// Precomputes crossing-off constants for prime `p`.
    /// `p` must be coprime to 30 (i.e. p > 5 and odd and not divisible by 3 or 5).
    pub fn new(p: u64) -> Self {
        let w30res_p: [u8; 8] = [
            (W30_RESIDUES[0] as u64 * p % 30) as u8,
            (W30_RESIDUES[1] as u64 * p % 30) as u8,
            (W30_RESIDUES[2] as u64 * p % 30) as u8,
            (W30_RESIDUES[3] as u64 * p % 30) as u8,
            (W30_RESIDUES[4] as u64 * p % 30) as u8,
            (W30_RESIDUES[5] as u64 * p % 30) as u8,
            (W30_RESIDUES[6] as u64 * p % 30) as u8,
            (W30_RESIDUES[7] as u64 * p % 30) as u8,
        ];
        let bit_seq: [u8; 8] = [
            W30_IDX[w30res_p[0] as usize],
            W30_IDX[w30res_p[1] as usize],
            W30_IDX[w30res_p[2] as usize],
            W30_IDX[w30res_p[3] as usize],
            W30_IDX[w30res_p[4] as usize],
            W30_IDX[w30res_p[5] as usize],
            W30_IDX[w30res_p[6] as usize],
            W30_IDX[w30res_p[7] as usize],
        ];
        let gap_m_u64: [u64; 8] = [
            WHEEL30_GAPS[0] as u64 * p,
            WHEEL30_GAPS[1] as u64 * p,
            WHEEL30_GAPS[2] as u64 * p,
            WHEEL30_GAPS[3] as u64 * p,
            WHEEL30_GAPS[4] as u64 * p,
            WHEEL30_GAPS[5] as u64 * p,
            WHEEL30_GAPS[6] as u64 * p,
            WHEEL30_GAPS[7] as u64 * p,
        ];
        let gap_m: [u32; 8] = [
            gap_m_u64[0] as u32, gap_m_u64[1] as u32,
            gap_m_u64[2] as u32, gap_m_u64[3] as u32,
            gap_m_u64[4] as u32, gap_m_u64[5] as u32,
            gap_m_u64[6] as u32, gap_m_u64[7] as u32,
        ];
        let delta_group: [u32; 8] = [
            ((w30res_p[0] as u64 + gap_m_u64[0] - w30res_p[1] as u64) / 30) as u32,
            ((w30res_p[1] as u64 + gap_m_u64[1] - w30res_p[2] as u64) / 30) as u32,
            ((w30res_p[2] as u64 + gap_m_u64[2] - w30res_p[3] as u64) / 30) as u32,
            ((w30res_p[3] as u64 + gap_m_u64[3] - w30res_p[4] as u64) / 30) as u32,
            ((w30res_p[4] as u64 + gap_m_u64[4] - w30res_p[5] as u64) / 30) as u32,
            ((w30res_p[5] as u64 + gap_m_u64[5] - w30res_p[6] as u64) / 30) as u32,
            ((w30res_p[6] as u64 + gap_m_u64[6] - w30res_p[7] as u64) / 30) as u32,
            ((w30res_p[7] as u64 + gap_m_u64[7] - w30res_p[0] as u64) / 30) as u32,
        ];
        Self { w30res_p, bit_seq, gap_m, delta_group }
    }
}

/// Wheel-30 segmented sieve over a window `[lo, lo + W30_SEG)`.
///
/// Only integers coprime to {2, 3, 5} are stored (8 per group of 30).
/// `lo` **must** be a multiple of 30.
pub struct WheelSieve30 {
    bits: [u64; W30_WORDS],
}

/// Monotonic-stop cursor for [`WheelSieve30::count_primes_upto_int_m`].
/// Tracks how many full u64 words of `bits` have already been folded into
/// `sum`, so subsequent queries with a larger stop only popcount the newly
/// traversed words. Reset per bi via [`MonoCount::reset`].
#[derive(Clone, Copy)]
pub struct MonoCount {
    /// Words in `[0, w)` are already added to `sum`.
    w: usize,
    /// Set-bit count in `bits[0..w]`.
    sum: u64,
}

impl MonoCount {
    #[inline]
    pub fn new() -> Self { Self { w: 0, sum: 0 } }
    #[inline]
    pub fn reset(&mut self) { self.w = 0; self.sum = 0; }
}

impl Default for MonoCount {
    fn default() -> Self { Self::new() }
}

/// Pre-sieved pattern for primes {7, 11} over one full wheel-30 period of
/// `lcm(7, 11) = 77` groups (= 2310 integers). Bit `j` of byte `g` is 1 iff
/// the integer `g*30 + W30_RESIDUES[j]` is coprime to both 7 and 11.
///
/// Initialised on first use by [`get_presieve_7_11`]. Using this template
/// lets [`WheelSieve30::fill_presieved_7_11`] skip the per-segment
/// cross-off loops for 7 and 11 (the tiny-prime stage used by the S2_hard
/// sweep), replacing them with a straight byte tile.
const PRESIEVE_BYTES: usize = 7 * 11;       // 77 bytes cover 2310 integers
const PRESIEVE_SPAN:  u64   = (PRESIEVE_BYTES as u64) * 30;  // 2310

static PRESIEVE_7_11: std::sync::OnceLock<[u8; PRESIEVE_BYTES]> = std::sync::OnceLock::new();

fn get_presieve_7_11() -> &'static [u8; PRESIEVE_BYTES] {
    PRESIEVE_7_11.get_or_init(|| {
        let mut t = [0u8; PRESIEVE_BYTES];
        for g in 0..PRESIEVE_BYTES {
            let mut byte = 0u8;
            for j in 0..8 {
                let n = g * 30 + W30_RESIDUES[j] as usize;
                if n % 7 != 0 && n % 11 != 0 {
                    byte |= 1 << j;
                }
            }
            t[g] = byte;
        }
        t
    })
}

impl WheelSieve30 {
    /// Creates a zeroed (all-composite) sieve.
    pub fn new() -> Self {
        Self { bits: [0u64; W30_WORDS] }
    }

    /// Fills the window `[lo, lo + W30_SEG)` using the supplied sieving primes.
    ///
    /// `lo` must be a multiple of 30.
    ///
    /// `sieving_primes[i] = (p, m)` where `m` is the first multiple of `p`
    /// that is ≥ `lo` **and** coprime to 30 (i.e. `m % p == 0` and
    /// `W30_IDX[(m/p % 30) as usize] != 255`).
    pub fn fill(&mut self, lo: u64, sieving_primes: &[(u64, u64)]) {
        debug_assert_eq!(lo % 30, 0, "lo must be a multiple of 30");

        // All coprime-to-30 positions start as prime candidates.
        self.bits.fill(!0u64);
        // Mask unused bits in the last word (W30_BITS % 64 used).
        const USED: usize = W30_BITS % 64; // 32
        if USED > 0 {
            self.bits[W30_WORDS - 1] &= (1u64 << USED) - 1;
        }

        // Integer 1 (at lo = 0) is not prime; clear its bit (bit_idx = 0).
        if lo == 0 {
            self.bits[0] &= !1u64;
        }

        for &(p, m_start) in sieving_primes {
            if m_start >= lo + W30_SEG as u64 {
                continue;
            }
            // m_start is the first coprime-to-30 multiple of p in [lo, ∞).
            // k1 = m_start / p; j0 = wheel position of k1.
            let k1 = m_start / p;
            let j0 = W30_IDX[(k1 % 30) as usize] as usize;

            // Per-prime precomputation (8 values each, fully unrollable):
            //   w30res_p[j] = residue of (k*p) mod 30 when k's wheel pos is j
            //   bit_seq[j]  = bit-within-group for m at k-position j
            //   gap_m[j]    = advance in m to the next coprime-to-30 multiple
            //   delta_group[j] = group advance after that step (always ≥ 0)
            let w30res_p: [u64; 8] = [
                W30_RESIDUES[0] as u64 * p % 30,
                W30_RESIDUES[1] as u64 * p % 30,
                W30_RESIDUES[2] as u64 * p % 30,
                W30_RESIDUES[3] as u64 * p % 30,
                W30_RESIDUES[4] as u64 * p % 30,
                W30_RESIDUES[5] as u64 * p % 30,
                W30_RESIDUES[6] as u64 * p % 30,
                W30_RESIDUES[7] as u64 * p % 30,
            ];
            let bit_seq: [usize; 8] = [
                W30_IDX[w30res_p[0] as usize] as usize,
                W30_IDX[w30res_p[1] as usize] as usize,
                W30_IDX[w30res_p[2] as usize] as usize,
                W30_IDX[w30res_p[3] as usize] as usize,
                W30_IDX[w30res_p[4] as usize] as usize,
                W30_IDX[w30res_p[5] as usize] as usize,
                W30_IDX[w30res_p[6] as usize] as usize,
                W30_IDX[w30res_p[7] as usize] as usize,
            ];
            // gap_m[j] = WHEEL30_GAPS[j] * p
            let gap_m: [u64; 8] = [
                WHEEL30_GAPS[0] as u64 * p,
                WHEEL30_GAPS[1] as u64 * p,
                WHEEL30_GAPS[2] as u64 * p,
                WHEEL30_GAPS[3] as u64 * p,
                WHEEL30_GAPS[4] as u64 * p,
                WHEEL30_GAPS[5] as u64 * p,
                WHEEL30_GAPS[6] as u64 * p,
                WHEEL30_GAPS[7] as u64 * p,
            ];
            // delta_group[j] = (w30res_p[j] + gap_m[j] - w30res_p[(j+1)%8]) / 30
            // This is always an exact non-negative multiple of 30 divided by 30.
            let delta_group: [usize; 8] = [
                ((w30res_p[0] + gap_m[0] - w30res_p[1]) / 30) as usize,
                ((w30res_p[1] + gap_m[1] - w30res_p[2]) / 30) as usize,
                ((w30res_p[2] + gap_m[2] - w30res_p[3]) / 30) as usize,
                ((w30res_p[3] + gap_m[3] - w30res_p[4]) / 30) as usize,
                ((w30res_p[4] + gap_m[4] - w30res_p[5]) / 30) as usize,
                ((w30res_p[5] + gap_m[5] - w30res_p[6]) / 30) as usize,
                ((w30res_p[6] + gap_m[6] - w30res_p[7]) / 30) as usize,
                ((w30res_p[7] + gap_m[7] - w30res_p[0]) / 30) as usize,
            ];

            let local0 = (m_start - lo) as usize;
            let mut group = local0 / 30; // one-time division per prime
            let mut j = j0;
            let mut m = m_start;
            let end = lo + W30_SEG as u64;

            while m < end {
                let bit_idx = group * 8 + bit_seq[j];
                self.bits[bit_idx >> 6] &= !(1u64 << (bit_idx & 63));
                group += delta_group[j];
                m += gap_m[j];
                j = (j + 1) & 7;
            }
        }
    }

    /// Specialised `fill` for the tiny-prime set `{7, 11}`. Replaces the
    /// ones-fill + per-prime wheel-30 cross-off loops with a straight tile
    /// of the pre-computed template, cutting ~32k bit writes per segment.
    ///
    /// Semantically equivalent to `fill(lo, &[(7, m7), (11, m11)])` with the
    /// correct starting multiples, but independent of the segment boundary.
    /// `lo` must be a multiple of 30.
    pub fn fill_presieved_7_11(&mut self, lo: u64) {
        debug_assert_eq!(lo % 30, 0, "lo must be a multiple of 30");
        let template = get_presieve_7_11();

        // Bytes view of self.bits. W30_WORDS * 8 = 17480 bytes.
        let bits_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(
                self.bits.as_mut_ptr() as *mut u8,
                W30_WORDS * 8,
            )
        };

        // Offset within the template for this segment's starting number.
        let off = ((lo % PRESIEVE_SPAN) / 30) as usize;

        // Tile the template over the W30_GROUPS (= 17476) valid bytes.
        let total = W30_GROUPS;
        let mut pos = 0;
        // First chunk (from `off` to end of template).
        let first = (PRESIEVE_BYTES - off).min(total);
        bits_bytes[..first].copy_from_slice(&template[off..off + first]);
        pos += first;
        // Full templates.
        while pos + PRESIEVE_BYTES <= total {
            bits_bytes[pos..pos + PRESIEVE_BYTES].copy_from_slice(template);
            pos += PRESIEVE_BYTES;
        }
        // Tail.
        if pos < total {
            let rem = total - pos;
            bits_bytes[pos..pos + rem].copy_from_slice(&template[..rem]);
        }

        // Zero the padding bytes past W30_GROUPS (they live in the high 32 bits
        // of the final u64 word and must not carry template residue).
        for b in total..W30_WORDS * 8 {
            bits_bytes[b] = 0;
        }

        // Integer 1 is coprime to {7, 11} so the template marks it set; clear
        // it explicitly when `lo == 0` since 1 is not prime.
        if lo == 0 {
            self.bits[0] &= !1u64;
        }
    }

    /// Sets the bit for integer 1 (bit_idx = 0) — used by the φ-sieve to
    /// count 1 as coprime after `fill(0, …)` clears it.
    #[inline]
    pub fn set_bit_for_1(&mut self) {
        self.bits[0] |= 1u64;
    }

    /// Crosses off all coprime-to-30 multiples of `p` in `[lo, lo + W30_SEG)`
    /// and returns the number of bits actually cleared (i.e. were set before).
    ///
    /// `lo` must be a multiple of 30.  `p` must be coprime to 30 (p > 5).
    pub fn cross_off_count(&mut self, lo: u64, p: u64) -> u64 {
        debug_assert_eq!(lo % 30, 0);
        // First coprime-to-30 multiple of p that is ≥ lo.
        let k0 = (lo + p - 1) / p; // ceil(lo / p)
        let k1 = wheel30_next_k(k0);
        let m_start = k1 * p;
        if m_start >= lo + W30_SEG as u64 {
            return 0;
        }

        let w30res_p: [u64; 8] = [
            W30_RESIDUES[0] as u64 * p % 30,
            W30_RESIDUES[1] as u64 * p % 30,
            W30_RESIDUES[2] as u64 * p % 30,
            W30_RESIDUES[3] as u64 * p % 30,
            W30_RESIDUES[4] as u64 * p % 30,
            W30_RESIDUES[5] as u64 * p % 30,
            W30_RESIDUES[6] as u64 * p % 30,
            W30_RESIDUES[7] as u64 * p % 30,
        ];
        let bit_seq: [usize; 8] = [
            W30_IDX[w30res_p[0] as usize] as usize,
            W30_IDX[w30res_p[1] as usize] as usize,
            W30_IDX[w30res_p[2] as usize] as usize,
            W30_IDX[w30res_p[3] as usize] as usize,
            W30_IDX[w30res_p[4] as usize] as usize,
            W30_IDX[w30res_p[5] as usize] as usize,
            W30_IDX[w30res_p[6] as usize] as usize,
            W30_IDX[w30res_p[7] as usize] as usize,
        ];
        let gap_m: [u64; 8] = [
            WHEEL30_GAPS[0] as u64 * p,
            WHEEL30_GAPS[1] as u64 * p,
            WHEEL30_GAPS[2] as u64 * p,
            WHEEL30_GAPS[3] as u64 * p,
            WHEEL30_GAPS[4] as u64 * p,
            WHEEL30_GAPS[5] as u64 * p,
            WHEEL30_GAPS[6] as u64 * p,
            WHEEL30_GAPS[7] as u64 * p,
        ];
        let delta_group: [usize; 8] = [
            ((w30res_p[0] + gap_m[0] - w30res_p[1]) / 30) as usize,
            ((w30res_p[1] + gap_m[1] - w30res_p[2]) / 30) as usize,
            ((w30res_p[2] + gap_m[2] - w30res_p[3]) / 30) as usize,
            ((w30res_p[3] + gap_m[3] - w30res_p[4]) / 30) as usize,
            ((w30res_p[4] + gap_m[4] - w30res_p[5]) / 30) as usize,
            ((w30res_p[5] + gap_m[5] - w30res_p[6]) / 30) as usize,
            ((w30res_p[6] + gap_m[6] - w30res_p[7]) / 30) as usize,
            ((w30res_p[7] + gap_m[7] - w30res_p[0]) / 30) as usize,
        ];

        let j0 = W30_IDX[(k1 % 30) as usize] as usize;
        let local0 = (m_start - lo) as usize;
        let mut group = local0 / 30;
        let mut j = j0;
        let mut m = m_start;
        let end = lo + W30_SEG as u64;
        let mut cleared = 0u64;

        while m < end {
            let bit_idx = group * 8 + bit_seq[j];
            let word = bit_idx >> 6;
            let bit = bit_idx & 63;
            let prev = (self.bits[word] >> bit) & 1;
            self.bits[word] &= !(1u64 << bit);
            cleared += prev;
            group += delta_group[j];
            m += gap_m[j];
            j = (j + 1) & 7;
        }
        cleared
    }

    /// Like [`cross_off_count`] but uses precomputed [`WheelPrimeData`].
    ///
    /// Call this inside the window loop when the same prime `p` is crossed off
    /// across many windows; compute [`WheelPrimeData::new(p)`] once outside.
    #[inline]
    pub fn cross_off_count_pd(&mut self, lo: u64, p: u64, pd: &WheelPrimeData) -> u64 {
        debug_assert_eq!(lo % 30, 0);
        let k0 = (lo + p - 1) / p;
        let k1 = wheel30_next_k(k0);
        let m_start = k1 * p;
        if m_start >= lo + W30_SEG as u64 {
            return 0;
        }
        let j0 = W30_IDX[(k1 % 30) as usize] as usize;
        let local0 = (m_start - lo) as usize;
        let mut group = local0 / 30;
        let mut j = j0;
        let mut m = m_start;
        let end = lo + W30_SEG as u64;
        let mut cleared = 0u64;
        while m < end {
            let bit_idx = group * 8 + pd.bit_seq[j] as usize;
            let word = bit_idx >> 6;
            let bit = bit_idx & 63;
            let prev = (self.bits[word] >> bit) & 1;
            self.bits[word] &= !(1u64 << bit);
            cleared += prev;
            group += pd.delta_group[j] as usize;
            m += pd.gap_m[j] as u64;
            j = (j + 1) & 7;
        }
        cleared
    }

    /// Like [`cross_off_count_pd`] but Kim-style unrolled: dispatches on the
    /// prime's residue group `g = pd.bit_seq[0]` to one of 8 specialised inner
    /// loops with bit positions hardcoded as immediates. Each cleared bit
    /// turns into 4 ops (load byte, extract bit, masked store, accumulate)
    /// vs ~12 ops in the rolled version (table lookups + word/bit reconstruction).
    #[inline]
    pub fn cross_off_count_pd_unrolled(
        &mut self,
        lo: u64,
        p: u64,
        pd: &WheelPrimeData,
    ) -> u64 {
        debug_assert_eq!(lo % 30, 0);
        let k0 = (lo + p - 1) / p;
        let k1 = wheel30_next_k(k0);
        let m_start = k1 * p;
        if m_start >= lo + W30_SEG as u64 {
            return 0;
        }
        let j0 = W30_IDX[(k1 % 30) as usize] as usize;
        let local0 = (m_start - lo) as usize;
        let group_start = local0 / 30;

        let bits_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(self.bits.as_mut_ptr() as *mut u8, W30_WORDS * 8)
        };

        let g = pd.bit_seq[0] as usize;
        let dg = &pd.delta_group;
        match g {
            0 => xoff_count_unrolled_g0(bits_bytes, group_start, j0, dg),
            1 => xoff_count_unrolled_g1(bits_bytes, group_start, j0, dg),
            2 => xoff_count_unrolled_g2(bits_bytes, group_start, j0, dg),
            3 => xoff_count_unrolled_g3(bits_bytes, group_start, j0, dg),
            4 => xoff_count_unrolled_g4(bits_bytes, group_start, j0, dg),
            5 => xoff_count_unrolled_g5(bits_bytes, group_start, j0, dg),
            6 => xoff_count_unrolled_g6(bits_bytes, group_start, j0, dg),
            7 => xoff_count_unrolled_g7(bits_bytes, group_start, j0, dg),
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Like [`cross_off_pd`] but resumes from a stored `(next_m, next_j)`
    /// state instead of recomputing the first multiple with a 64-bit division.
    ///
    /// `next_m` is the next multiple of `p` (coprime to 30) ≥ previous segment's
    /// `hi`; may be `< lo` if the prime was idle for several segments.
    /// `next_j` is the wheel-30 position index for `next_m`.
    ///
    /// Returns the updated `(next_m, next_j)` for the following segment.
    #[inline]
    pub fn cross_off_pd_from_state(
        &mut self,
        lo: u64,
        _p: u64,
        pd: &WheelPrimeData,
        next_m: u64,
        next_j: u8,
    ) -> (u64, u8) {
        debug_assert_eq!(lo % 30, 0);
        let end = lo + W30_SEG as u64;
        let mut m = next_m;
        let mut j = next_j as usize;

        // Advance to first multiple ≥ lo. For primes ≫ W30_SEG this is 0-1
        // iterations; for primes ≪ W30_SEG the while is never entered because
        // the previous segment already left m in [lo_prev, lo_prev+W30_SEG) and
        // the step below naturally continues.
        while m < lo {
            m += pd.gap_m[j] as u64;
            j = (j + 1) & 7;
        }

        if m < end {
            // `m` has residue w30res_p[j] mod 30, so (m - lo) / 30 is exact.
            let mut group = ((m - lo) / 30) as usize;
            while m < end {
                let bit_idx = group * 8 + pd.bit_seq[j] as usize;
                self.bits[bit_idx >> 6] &= !(1u64 << (bit_idx & 63));
                group += pd.delta_group[j] as usize;
                m += pd.gap_m[j] as u64;
                j = (j + 1) & 7;
            }
        }

        (m, j as u8)
    }

    /// Like [`cross_off_count_pd`] but does not count cleared bits.
    ///
    /// Use this for the bulk cross-off pass where the count is not needed,
    /// saving the read-modify-check overhead per bit.
    #[inline]
    pub fn cross_off_pd(&mut self, lo: u64, p: u64, pd: &WheelPrimeData) {
        debug_assert_eq!(lo % 30, 0);
        let k0 = (lo + p - 1) / p;
        let k1 = wheel30_next_k(k0);
        let m_start = k1 * p;
        if m_start >= lo + W30_SEG as u64 {
            return;
        }
        let j0 = W30_IDX[(k1 % 30) as usize] as usize;
        let local0 = (m_start - lo) as usize;
        let mut group = local0 / 30;
        let mut j = j0;
        let mut m = m_start;
        let end = lo + W30_SEG as u64;
        while m < end {
            let bit_idx = group * 8 + pd.bit_seq[j] as usize;
            self.bits[bit_idx >> 6] &= !(1u64 << (bit_idx & 63));
            group += pd.delta_group[j] as usize;
            m += pd.gap_m[j] as u64;
            j = (j + 1) & 7;
        }
    }

    /// Like [`cross_off_pd`] but Kim-style: dispatches on the prime's residue
    /// group (`g = pd.bit_seq[0]`, equivalent to `W30_IDX[p % 30]`) to one of
    /// 8 specialised inner loops with bit positions hardcoded as immediates.
    ///
    /// Each inner loop has the same shape: a ≤ 7-step pre-roll to align on
    /// `j = 0`, an 8-step unrolled main loop (one full wheel-30 cycle of the
    /// prime), then a ≤ 7-step tail. The main loop writes into the
    /// `[u8]` byte view of `self.bits`, replacing the per-iteration
    /// `bit_seq[j]` lookup + word/bit reconstruction with a single `andb m8, imm8`.
    #[inline]
    pub fn cross_off_pd_unrolled(&mut self, lo: u64, p: u64, pd: &WheelPrimeData) {
        debug_assert_eq!(lo % 30, 0);
        let k0 = (lo + p - 1) / p;
        let k1 = wheel30_next_k(k0);
        let m_start = k1 * p;
        if m_start >= lo + W30_SEG as u64 {
            return;
        }
        let j0 = W30_IDX[(k1 % 30) as usize] as usize;
        let local0 = (m_start - lo) as usize;
        let group_start = local0 / 30;

        // Bytes view: 1 byte = 1 group of 30 = 8 wheel positions. The 4 bytes
        // past W30_GROUPS are padding (kept 0 by `fill`); we never write into
        // them because the per-group fns bound writes to W30_GROUPS.
        let bits_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(self.bits.as_mut_ptr() as *mut u8, W30_WORDS * 8)
        };

        // pd.bit_seq[0] = W30_IDX[(1*p) % 30] = group index g ∈ 0..8.
        let g = pd.bit_seq[0] as usize;
        let dg = &pd.delta_group;
        match g {
            0 => xoff_unrolled_g0(bits_bytes, group_start, j0, dg),
            1 => xoff_unrolled_g1(bits_bytes, group_start, j0, dg),
            2 => xoff_unrolled_g2(bits_bytes, group_start, j0, dg),
            3 => xoff_unrolled_g3(bits_bytes, group_start, j0, dg),
            4 => xoff_unrolled_g4(bits_bytes, group_start, j0, dg),
            5 => xoff_unrolled_g5(bits_bytes, group_start, j0, dg),
            6 => xoff_unrolled_g6(bits_bytes, group_start, j0, dg),
            7 => xoff_unrolled_g7(bits_bytes, group_start, j0, dg),
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Returns the total number of set bits.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.bits.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Fills a prefix-popcount table for fast prime counting.
    ///
    /// After the call, `out[j]` = number of set bits in `self.bits[0..j]`.
    /// `out` must have length ≥ `W30_WORDS + 1`.
    pub fn fill_prefix_counts(&self, out: &mut [u32]) {
        debug_assert!(out.len() >= W30_WORDS + 1);
        out[0] = 0;
        for j in 0..W30_WORDS {
            out[j + 1] = out[j] + self.bits[j].count_ones();
        }
    }

    /// Counts primes in `[lo, n]` using the prefix table from [`fill_prefix_counts`].
    ///
    /// `n` must satisfy `lo ≤ n < lo + W30_SEG`.
    #[inline]
    pub fn count_primes_upto_int(&self, prefix: &[u32], n: u64, lo: u64) -> u32 {
        debug_assert!(n >= lo && n < lo + W30_SEG as u64);
        let local = (n - lo) as usize;
        let word = local / 240;
        let mask = W30_MASK_LEQ_240[local % 240];
        prefix[word] + (self.bits[word] & mask).count_ones()
    }

    /// Monotonic variant of [`count_primes_upto_int`]. `n` must satisfy
    /// `lo ≤ n < lo + W30_SEG` and be non-decreasing across successive calls
    /// for the same `stop` cursor.
    #[inline]
    pub fn count_primes_upto_int_m(&self, stop: &mut MonoCount, n: u64, lo: u64) -> u64 {
        debug_assert!(n >= lo && n < lo + W30_SEG as u64);
        let local = (n - lo) as usize;
        let word = local / 240;
        while stop.w < word {
            stop.sum += self.bits[stop.w].count_ones() as u64;
            stop.w += 1;
        }
        stop.sum + (self.bits[word] & W30_MASK_LEQ_240[local % 240]).count_ones() as u64
    }
}

impl Default for WheelSieve30 {
    fn default() -> Self { Self::new() }
}

// Kim-style unrolled cross-off helpers. One specialisation per residue group
// `g = W30_IDX[p % 30] ∈ 0..8`, each with the bit-position sequence baked in
// as immediates. Generated via `impl_xoff_unrolled!`.
//
// Layout in [u8] view: 1 byte covers 8 wheel positions of one group of 30
// integers. `dg[j]` is the byte-index advance from wheel position `j` to
// `(j+1) mod 8` (= `(w30res_p[j] + 30*WHEEL30_GAPS[j]/30 - w30res_p[(j+1)%8])
// / 30`, equivalent to `(p/30)*WHEEL30_GAPS[j] + wheel_corr[g][j]` in Kim).
// One full 8-step cycle advances by `sum(dg) == p` bytes.
macro_rules! impl_xoff_unrolled {
    ($name:ident, $b0:literal, $b1:literal, $b2:literal, $b3:literal,
                  $b4:literal, $b5:literal, $b6:literal, $b7:literal) => {
        #[inline]
        fn $name(bits: &mut [u8], group_start: usize, j0: usize, dg: &[u32; 8]) {
            const BITS: [u8; 8] = [$b0, $b1, $b2, $b3, $b4, $b5, $b6, $b7];
            let d0 = dg[0] as usize;
            let d1 = dg[1] as usize;
            let d2 = dg[2] as usize;
            let d3 = dg[3] as usize;
            let d4 = dg[4] as usize;
            let d5 = dg[5] as usize;
            let d6 = dg[6] as usize;
            let d7 = dg[7] as usize;
            let d_arr = [d0, d1, d2, d3, d4, d5, d6, d7];
            // Max byte offset reached *during* one 8-step cycle (writes happen
            // at g, g+d0, g+d0+d1, …, g+d0+…+d6). Used as the unroll guard.
            let max_offset = d0 + d1 + d2 + d3 + d4 + d5 + d6;
            let group_end = W30_GROUPS;

            let mut g = group_start;

            // Pre-roll: complete the partial cycle from j0..8 step by step.
            for jj in j0..8 {
                if g >= group_end { return; }
                unsafe { *bits.get_unchecked_mut(g) &= !(1u8 << BITS[jj]); }
                g += d_arr[jj];
            }

            // Main unrolled loop: each iteration is one full wheel-30 cycle
            // and advances `g` by exactly `p` bytes. Guard ensures the 7th
            // (last interior) write position `g + max_offset` stays < group_end.
            while g + max_offset < group_end {
                unsafe {
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b0); g += d0;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b1); g += d1;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b2); g += d2;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b3); g += d3;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b4); g += d4;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b5); g += d5;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b6); g += d6;
                    *bits.get_unchecked_mut(g) &= !(1u8 << $b7); g += d7;
                }
            }

            // Tail: from j=0, step by step until we cross group_end.
            for jj in 0..8 {
                if g >= group_end { return; }
                unsafe { *bits.get_unchecked_mut(g) &= !(1u8 << BITS[jj]); }
                g += d_arr[jj];
            }
        }
    };
}

impl_xoff_unrolled!(xoff_unrolled_g0, 0, 1, 2, 3, 4, 5, 6, 7);
impl_xoff_unrolled!(xoff_unrolled_g1, 1, 5, 4, 0, 7, 3, 2, 6);
impl_xoff_unrolled!(xoff_unrolled_g2, 2, 4, 0, 6, 1, 7, 3, 5);
impl_xoff_unrolled!(xoff_unrolled_g3, 3, 0, 6, 5, 2, 1, 7, 4);
impl_xoff_unrolled!(xoff_unrolled_g4, 4, 7, 1, 2, 5, 6, 0, 3);
impl_xoff_unrolled!(xoff_unrolled_g5, 5, 3, 7, 1, 6, 0, 4, 2);
impl_xoff_unrolled!(xoff_unrolled_g6, 6, 2, 3, 7, 0, 4, 5, 1);
impl_xoff_unrolled!(xoff_unrolled_g7, 7, 6, 5, 4, 3, 2, 1, 0);

// Counted variants. Same shape as `impl_xoff_unrolled!` but each byte write
// reads the byte first to extract the previous bit (1 ↔ "was set"), AND-stores
// the cleared byte, and accumulates the cleared count. Kim's `cross_off_count`
// (Sieve.cpp:451–596) uses the same UNSET_BIT macro chained 8 times per cycle.
macro_rules! impl_xoff_count_unrolled {
    ($name:ident, $b0:literal, $b1:literal, $b2:literal, $b3:literal,
                  $b4:literal, $b5:literal, $b6:literal, $b7:literal) => {
        #[inline]
        fn $name(bits: &mut [u8], group_start: usize, j0: usize, dg: &[u32; 8]) -> u64 {
            const BITS: [u8; 8] = [$b0, $b1, $b2, $b3, $b4, $b5, $b6, $b7];
            let d0 = dg[0] as usize;
            let d1 = dg[1] as usize;
            let d2 = dg[2] as usize;
            let d3 = dg[3] as usize;
            let d4 = dg[4] as usize;
            let d5 = dg[5] as usize;
            let d6 = dg[6] as usize;
            let d7 = dg[7] as usize;
            let d_arr = [d0, d1, d2, d3, d4, d5, d6, d7];
            let max_offset = d0 + d1 + d2 + d3 + d4 + d5 + d6;
            let group_end = W30_GROUPS;

            let mut g = group_start;
            let mut cleared: u64 = 0;

            // Per-step: extract the previous bit value (for the cleared
            // counter), then clear it via an in-place `&=` so rustc emits a
            // single `andb m8, imm8` RMW instead of separate load + store.
            macro_rules! unset {
                ($pos:expr, $bit:expr) => {{
                    let byte = unsafe { *bits.get_unchecked($pos) };
                    cleared += ((byte >> $bit) & 1) as u64;
                    unsafe { *bits.get_unchecked_mut($pos) &= !(1u8 << $bit); }
                }};
            }

            // Pre-roll: complete the partial cycle from j0..8 step by step.
            for jj in j0..8 {
                if g >= group_end { return cleared; }
                unset!(g, BITS[jj]);
                g += d_arr[jj];
            }

            // Main unrolled: full 8-step cycle, no per-step bound check
            // (max interior write position = g + max_offset < group_end).
            while g + max_offset < group_end {
                unset!(g, $b0); g += d0;
                unset!(g, $b1); g += d1;
                unset!(g, $b2); g += d2;
                unset!(g, $b3); g += d3;
                unset!(g, $b4); g += d4;
                unset!(g, $b5); g += d5;
                unset!(g, $b6); g += d6;
                unset!(g, $b7); g += d7;
            }

            // Tail: from j=0, step by step until we cross group_end.
            for jj in 0..8 {
                if g >= group_end { return cleared; }
                unset!(g, BITS[jj]);
                g += d_arr[jj];
            }
            cleared
        }
    };
}

impl_xoff_count_unrolled!(xoff_count_unrolled_g0, 0, 1, 2, 3, 4, 5, 6, 7);
impl_xoff_count_unrolled!(xoff_count_unrolled_g1, 1, 5, 4, 0, 7, 3, 2, 6);
impl_xoff_count_unrolled!(xoff_count_unrolled_g2, 2, 4, 0, 6, 1, 7, 3, 5);
impl_xoff_count_unrolled!(xoff_count_unrolled_g3, 3, 0, 6, 5, 2, 1, 7, 4);
impl_xoff_count_unrolled!(xoff_count_unrolled_g4, 4, 7, 1, 2, 5, 6, 0, 3);
impl_xoff_count_unrolled!(xoff_count_unrolled_g5, 5, 3, 7, 1, 6, 0, 4, 2);
impl_xoff_count_unrolled!(xoff_count_unrolled_g6, 6, 2, 3, 7, 0, 4, 5, 1);
impl_xoff_count_unrolled!(xoff_count_unrolled_g7, 7, 6, 5, 4, 3, 2, 1, 0);

/// Advances wheel-sieve state so that each `m` is the first coprime-to-30
/// multiple of `p` at or after `next_lo`.
///
/// Invariant: each `(p, m)` in `state` satisfies `m % p == 0`.
pub fn advance_wheel_primes(state: &mut [(u64, u64)], next_lo: u64) {
    for (p, m) in state.iter_mut() {
        while *m < next_lo {
            *m += *p;
        }
        // m ≥ next_lo; advance k = m/p to the next coprime-to-30 index.
        let k = *m / *p; // exact: m is always a multiple of p
        let adj = WHEEL30_NEXT[(k % 30) as usize] as u64;
        *m = (k + adj) * *p;
    }
}

/// Returns all primes in `[2, limit]` using the provided sieving primes.
///
/// `sieve_primes` must contain all primes ≤ √limit.  Each prime in
/// `sieve_primes` is included in the output if it is ≤ `limit`.
///
/// Intended use: given seed primes ≤ y = ∛x from the Lucy sieve, extend to
/// all primes ≤ √x so that `s2_primes = &all_primes[a..]` covers (y, √x].
pub fn primes_up_to(limit: u64, sieve_primes: &[u64]) -> Vec<u64> {
    if limit < 2 {
        return vec![];
    }

    // Collect sieve primes that are ≤ limit (they are already prime).
    let mut result: Vec<u64> = sieve_primes
        .iter()
        .copied()
        .take_while(|&p| p <= limit)
        .collect();

    // If limit ≤ the largest sieve prime we already have, we're done.
    if sieve_primes.last().copied().unwrap_or(0) >= limit {
        return result;
    }

    // Otherwise sweep [max_covered+1, limit] in SEG-sized windows.
    let start_lo = result.last().copied().map(|p| p + 1).unwrap_or(2);
    let mut sieve = SegSieve::new();
    // Align lo to the SEG boundary at or below start_lo.
    let lo_init = (start_lo / SEG as u64) * SEG as u64;
    let mut lo = lo_init;

    let sp = init_small_primes(sieve_primes, lo);
    let mut state = sp;
    while lo <= limit {
        sieve.fill(lo, &state);
        for p in sieve.iter_primes(lo) {
            if p < start_lo {
                continue;
            }
            if p > limit {
                break;
            }
            result.push(p);
        }
        let next_lo = lo + SEG as u64;
        advance_small_primes(&mut state, next_lo);
        lo = next_lo;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::isqrt;

    /// Reference: all primes ≤ `limit` via a plain Eratosthenes sieve.
    ///
    /// Used only in tests.  The Lucy-Hedgehog `s` array only stores counts up
    /// to √x, so `extract_primes(&s, limit)` would be out-of-bounds for
    /// limit > √x.  A direct sieve is simpler and correct for any limit.
    fn reference_primes(limit: u64) -> Vec<u64> {
        if limit < 2 {
            return vec![];
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
        (2..=limit).filter(|&v| is_prime[v as usize]).collect()
    }

    /// All primes in `[lo, hi)` via the segmented sieve, one window at a time.
    fn seg_primes_in(lo: u64, hi: u64) -> Vec<u64> {
        let sqrt_hi = isqrt(hi as u128) as u64 + 1;
        let sp = reference_primes(sqrt_hi);
        let mut state = init_small_primes(&sp, lo);

        let mut sieve = SegSieve::new();
        let mut result = Vec::new();
        let mut window_lo = lo;

        while window_lo < hi {
            sieve.fill(window_lo, &state);
            for p in sieve.iter_primes(window_lo) {
                if p >= hi {
                    break;
                }
                result.push(p);
            }
            // Advance each starting multiple to the next window.
            let next_lo = window_lo + SEG as u64;
            for (p, m) in state.iter_mut() {
                while *m < next_lo {
                    *m += *p;
                }
            }
            window_lo = next_lo;
        }
        result
    }

    #[test]
    fn matches_reference_single_window_from_two() {
        // SEG integers starting at 2 — exactly one fill call.
        let limit = SEG as u64; // 524 288
        let expected = reference_primes(limit - 1);
        let got = seg_primes_in(2, limit);
        assert_eq!(got, expected);
    }

    #[test]
    fn matches_reference_first_million() {
        let limit = 1_000_000u64;
        let expected = reference_primes(limit);
        let got = seg_primes_in(2, limit + 1);
        assert_eq!(got, expected);
    }

    #[test]
    fn matches_reference_shifted_window() {
        // Window that does not start at 2.
        let lo = 1_000_000u64;
        let hi = lo + 10_000;
        let expected: Vec<u64> = reference_primes(hi)
            .into_iter()
            .filter(|&p| p >= lo)
            .collect();
        let got = seg_primes_in(lo, hi);
        assert_eq!(got, expected);
    }

    #[test]
    fn cross_off_pd_unrolled_matches_cross_off_pd() {
        // Covers all 8 residue groups (p%30 ∈ {1,7,11,13,17,19,23,29}) plus a
        // mix of small/medium/large primes (large = single-bit cross-off per
        // segment), and several `lo` values to exercise non-zero starting `j0`
        // and the pre-roll path.
        let primes_to_test: &[u64] = &[
            // small (cycle through segment many times)
            7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
            // medium
            127, 211, 257, 311, 401, 503, 601, 701, 809, 907,
            // around sqrt(W30_SEG) ≈ 724
            719, 727, 733,
            // large (≤ 1 cross-off per segment)
            12_007, 65_521, 524_287, 524_309,
        ];
        let los: &[u64] = &[
            0,
            30,
            60,
            300,
            W30_SEG as u64,
            (W30_SEG as u64) * 7,
            1_000_000_020,
        ];
        for &lo in los {
            for &p in primes_to_test {
                let pd = WheelPrimeData::new(p);

                let mut s_ref = WheelSieve30::new();
                s_ref.fill(lo, &[]);
                let mut s_new = WheelSieve30::new();
                s_new.bits.copy_from_slice(&s_ref.bits);

                s_ref.cross_off_pd(lo, p, &pd);
                s_new.cross_off_pd_unrolled(lo, p, &pd);

                assert_eq!(
                    s_ref.bits, s_new.bits,
                    "Mismatch lo={} p={} (p%30={}, g={})",
                    lo, p, p % 30, W30_IDX[(p % 30) as usize],
                );
            }
        }
    }

    #[test]
    fn cross_off_count_pd_unrolled_matches_cross_off_count_pd() {
        // Same coverage as cross_off_pd_unrolled_matches_cross_off_pd, plus
        // verifies the returned `cleared` count matches.
        let primes_to_test: &[u64] = &[
            7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
            127, 211, 257, 311, 401, 503, 601, 701, 809, 907,
            719, 727, 733,
            12_007, 65_521, 524_287, 524_309,
        ];
        let los: &[u64] = &[
            0,
            30,
            60,
            300,
            W30_SEG as u64,
            (W30_SEG as u64) * 7,
            1_000_000_020,
        ];
        for &lo in los {
            for &p in primes_to_test {
                let pd = WheelPrimeData::new(p);

                let mut s_ref = WheelSieve30::new();
                s_ref.fill(lo, &[]);
                let mut s_new = WheelSieve30::new();
                s_new.bits.copy_from_slice(&s_ref.bits);

                let cleared_ref = s_ref.cross_off_count_pd(lo, p, &pd);
                let cleared_new = s_new.cross_off_count_pd_unrolled(lo, p, &pd);

                assert_eq!(
                    cleared_ref, cleared_new,
                    "cleared mismatch lo={} p={} (p%30={}, g={})",
                    lo, p, p % 30, W30_IDX[(p % 30) as usize],
                );
                assert_eq!(
                    s_ref.bits, s_new.bits,
                    "bits mismatch lo={} p={} (p%30={}, g={})",
                    lo, p, p % 30, W30_IDX[(p % 30) as usize],
                );
            }
        }
    }

    #[test]
    fn zero_and_one_are_not_prime() {
        let sp = reference_primes(10);
        let state = init_small_primes(&sp, 0);
        let mut sieve = SegSieve::new();
        sieve.fill(0, &state);
        let primes: Vec<u64> = sieve.iter_primes(0).take_while(|&p| p < 10).collect();
        assert_eq!(primes, vec![2, 3, 5, 7]);
    }
}
