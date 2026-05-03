//! Bucket sieve for `rest_bulk_xoff` — port of primesieve's `EratBig`.
//!
//! Designed to replace the linear sweep over `pb_data: Vec<WheelPrimeData>`
//! in [`crate::dr::hard::s2_hard_sieve_par`] for the bulk regime
//! (primes with 0–3 multiples per `W30_SEG` segment). At x = 1e18 the
//! sweep accounts for ~50 % of CPU and is DRAM-bound: each thread touches
//! `~50 K × 80 B = 4 MB` of `pb_data` per band, blowing past the 9700X's
//! 1 MB L2/core. EratBig keeps the working set per segment to a single
//! 8 KB bucket (~1 K sieving primes × 8 B), with primes only present in
//! `buckets[N]` while their next multiple lies in segment `N`.
//!
//! # Layout
//!
//! Each [`SievingPrime`] is 8 B and packs:
//!   * `multiple_index` (23 bits) — offset of the next multiple within the
//!     target segment, in wheel-30 byte units (`< 2²⁰` for `W30_SEG`).
//!   * `wheel_index` (9 bits) — position in the wheel-30 cycle.
//!   * `prime` (32 bits) — the sieving prime itself (≤ √x ≤ 10⁹ at our scale).
//!
//! [`Bucket`] is a fixed-size array of [`SievingPrime`] entries plus a
//! singly-linked tail; total size `≤ BUCKET_BYTES = 8 KB` so an active
//! bucket fits in L2. [`MemoryPool`] is a per-thread freelist that recycles
//! drained buckets to avoid re-allocation churn.
//!
//! [`BucketSieve`] owns a `head` array indexed by segment id; each slot is
//! the head of the bucket chain for that segment. Inserts grow the chain
//! by allocating new buckets through the pool; draining a segment detaches
//! the entire chain and hands the buckets back to the pool when consumed.
//!
//! # Intended access pattern
//!
//! 1. Build: distribute every bulk prime into `buckets[seg_first_multiple]`.
//! 2. Per segment N (in order):
//!    a. `drain = bs.take_segment(N)`
//!    b. For each prime in `drain`, cross off the multiple in `sieve[N]`,
//!       compute the next `(target_seg, multiple_index, wheel_index)`, and
//!       `bs.insert(target_seg, prime, mi, wi)`.
//!    c. Hand the now-empty buckets back via `pool.recycle(drain)`.
//!
//! This commit ships the data structure and tests only. Wiring into
//! `s2_hard_sieve_par` is the next commit.

/// Bucket capacity in bytes — primesieve uses 8 KB, sized to fit a single
/// active bucket per thread inside L2 (≥ 256 KB on every target CPU). Must
/// remain a power of two for fast `is_full` checks if we later switch to
/// the pointer-arithmetic trick from primesieve.
pub const BUCKET_BYTES: usize = 8 << 10;

/// Maximum `multiple_index` storable in a `SievingPrime` (23-bit field).
/// `W30_SEG = 524 280 < 2²⁰`, so the field is sized with comfortable margin.
pub const MAX_MULTIPLE_INDEX: u32 = (1 << 23) - 1;

/// Maximum `wheel_index` storable in a `SievingPrime` (9-bit field).
/// Wheel-30 only uses 8 residues so 9 bits is overkill, but matches
/// primesieve so the layout stays compatible if we later port wheel-210.
pub const MAX_WHEEL_INDEX: u32 = (1 << 9) - 1;

/// Number of [`SievingPrime`] entries per [`Bucket`]. Computed from
/// `BUCKET_BYTES` minus the per-bucket header (`len: u32` padded to 8 B
/// for `Box<Bucket>` alignment + `next: Option<Box<Bucket>>` = 16 B).
///
/// At `BUCKET_BYTES = 8192` this is `(8192 - 16) / 8 = 1022`.
pub const BUCKET_ENTRIES: usize = (BUCKET_BYTES - 16) / 8;

/// Cap on the freelist size to bound memory under bursty workloads.
/// A drained bucket beyond this cap is dropped (re-allocated next time).
/// `1024` ≈ 8 MB freelist, well below the 80 MB pre-bucket footprint.
const POOL_CAPACITY: usize = 1024;

/// 8-byte packed sieving prime: `(multiple_index, wheel_index, prime)`.
/// The `(mi, wi)` pair locates the prime's next multiple inside its
/// owning bucket's segment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SievingPrime {
    /// Low 23 bits = multiple_index, high 9 bits = wheel_index.
    indexes: u32,
    /// Sieving prime itself. `u32` is sufficient up to x ≈ 10¹⁸ (√x ≈ 10⁹).
    prime: u32,
}

impl SievingPrime {
    /// Pack `(prime, multiple_index, wheel_index)` into an 8-byte entry.
    /// All three values must respect their field widths; debug builds
    /// assert this, release builds mask silently.
    #[inline]
    pub fn new(prime: u32, multiple_index: u32, wheel_index: u32) -> Self {
        debug_assert!(multiple_index <= MAX_MULTIPLE_INDEX);
        debug_assert!(wheel_index <= MAX_WHEEL_INDEX);
        SievingPrime {
            indexes: (multiple_index & MAX_MULTIPLE_INDEX) | (wheel_index << 23),
            prime,
        }
    }

    #[inline] pub fn prime(&self) -> u32 { self.prime }
    #[inline] pub fn multiple_index(&self) -> u32 { self.indexes & MAX_MULTIPLE_INDEX }
    #[inline] pub fn wheel_index(&self) -> u32 { self.indexes >> 23 }
}

/// Singly-linked bucket of [`SievingPrime`] entries. Allocated on the heap
/// and recycled through [`MemoryPool`].
pub struct Bucket {
    entries: [SievingPrime; BUCKET_ENTRIES],
    len: u32,
    next: Option<Box<Bucket>>,
}

impl Bucket {
    /// Empty, unchained bucket — public so `MemoryPool` can mint fresh
    /// ones. Callers normally go through [`MemoryPool::acquire`] instead.
    pub fn new_empty() -> Box<Self> {
        // Zero-init via uninit assume-init would shave the memset; for a
        // POC we keep the safe path. The 8 KB memset is amortised by the
        // ~1 K subsequent inserts that reuse the same memory.
        Box::new(Bucket {
            entries: [SievingPrime { indexes: 0, prime: 0 }; BUCKET_ENTRIES],
            len: 0,
            next: None,
        })
    }

    /// Number of entries currently stored in this bucket (excluding the tail).
    #[inline] pub fn len(&self) -> usize { self.len as usize }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn is_full(&self) -> bool { self.len as usize == BUCKET_ENTRIES }

    /// Slice of valid entries in this bucket only — does not traverse the
    /// linked tail. For full-chain iteration use [`BucketChainIter`].
    #[inline]
    pub fn entries(&self) -> &[SievingPrime] {
        &self.entries[..self.len as usize]
    }

    /// Reset to empty without dropping the box. Used by the pool when
    /// recycling.
    fn clear(&mut self) {
        self.len = 0;
        self.next = None;
    }

    /// Append an entry to this bucket. Caller must ensure `!self.is_full()`.
    #[inline]
    fn push_unchecked(&mut self, sp: SievingPrime) {
        debug_assert!(!self.is_full());
        let idx = self.len as usize;
        self.entries[idx] = sp;
        self.len += 1;
    }
}

/// Per-thread freelist of drained buckets. Capped at [`POOL_CAPACITY`]
/// boxes (~8 MB); excess buckets are dropped on recycle.
pub struct MemoryPool {
    freelist: Vec<Box<Bucket>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        MemoryPool { freelist: Vec::new() }
    }

    /// Pull an empty bucket from the freelist or allocate a fresh one.
    pub fn acquire(&mut self) -> Box<Bucket> {
        match self.freelist.pop() {
            Some(mut b) => {
                b.clear();
                b
            }
            None => Bucket::new_empty(),
        }
    }

    /// Hand a single (already-detached) bucket back to the pool. Drops the
    /// box if the freelist is at capacity.
    pub fn recycle_one(&mut self, mut bucket: Box<Bucket>) {
        bucket.clear();
        if self.freelist.len() < POOL_CAPACITY {
            self.freelist.push(bucket);
        }
    }

    /// Hand an entire chain back to the pool, walking through `next`.
    pub fn recycle_chain(&mut self, mut head: Option<Box<Bucket>>) {
        while let Some(mut b) = head {
            head = b.next.take();
            self.recycle_one(b);
        }
    }

    /// Number of buckets currently held by the freelist (debug/test view).
    pub fn freelist_len(&self) -> usize { self.freelist.len() }
}

impl Default for MemoryPool {
    fn default() -> Self { Self::new() }
}

/// Bucket sieve indexed by segment id. Each slot owns the head of a
/// singly-linked bucket chain for that segment.
///
/// Insertion grows the chain by `O(1)` amortised cost (amortised because
/// 1 in `BUCKET_ENTRIES` inserts allocates a new bucket). Drain detaches
/// the chain in `O(1)`; the caller iterates through the entries and
/// returns the buckets to the pool via [`MemoryPool::recycle_chain`].
pub struct BucketSieve {
    /// `head[seg]` = the chain head for segment `seg`, or `None` if no
    /// prime is currently scheduled to land in `seg`.
    head: Vec<Option<Box<Bucket>>>,
    pool: MemoryPool,
}

impl BucketSieve {
    /// New bucket sieve indexed by `[0, num_segments)` with an empty pool.
    pub fn new(num_segments: usize) -> Self {
        let mut head = Vec::with_capacity(num_segments);
        head.resize_with(num_segments, || None);
        BucketSieve { head, pool: MemoryPool::new() }
    }

    /// Number of segment slots in this sieve.
    #[inline] pub fn num_segments(&self) -> usize { self.head.len() }

    /// Direct access to the underlying pool — primarily for stats / tests.
    #[inline] pub fn pool(&self) -> &MemoryPool { &self.pool }
    #[inline] pub fn pool_mut(&mut self) -> &mut MemoryPool { &mut self.pool }

    /// Schedule `prime` to be crossed off in segment `seg` at offset
    /// `multiple_index` (with wheel state `wheel_index`).
    ///
    /// Allocates a new bucket through the pool when the current head is
    /// `None` or full.
    #[inline]
    pub fn insert(&mut self, seg: usize, prime: u32, multiple_index: u32, wheel_index: u32) {
        let sp = SievingPrime::new(prime, multiple_index, wheel_index);
        let slot = &mut self.head[seg];
        let need_new = match slot {
            None => true,
            Some(b) => b.is_full(),
        };
        if need_new {
            let mut fresh = self.pool.acquire();
            // The freshly-acquired bucket becomes the new head; the old
            // head (if any) is pushed onto the chain.
            fresh.next = slot.take();
            fresh.push_unchecked(sp);
            *slot = Some(fresh);
        } else {
            // SAFETY: `need_new = false` ⇒ slot is Some and not full.
            slot.as_mut().unwrap().push_unchecked(sp);
        }
    }

    /// Detach and return the entire chain for segment `seg`, leaving an
    /// empty slot behind. The caller iterates through the returned chain
    /// (via `Bucket::entries` + `Bucket.next`-style walk or the
    /// [`BucketChainIter`] adapter) and returns the buckets via
    /// [`BucketSieve::recycle_chain`] when done.
    pub fn take_segment(&mut self, seg: usize) -> Option<Box<Bucket>> {
        self.head[seg].take()
    }

    /// Convenience: hand a chain back to this sieve's pool.
    pub fn recycle_chain(&mut self, head: Option<Box<Bucket>>) {
        self.pool.recycle_chain(head);
    }

    /// Approximate memory footprint (head array + freelist + live chains).
    /// Heavy: walks every chain. Intended for diagnostics, not hot paths.
    pub fn size_bytes(&self) -> usize {
        let mut bytes = self.head.len() * std::mem::size_of::<Option<Box<Bucket>>>();
        bytes += self.pool.freelist.len() * std::mem::size_of::<Bucket>();
        for slot in &self.head {
            let mut cur = slot.as_deref();
            while let Some(b) = cur {
                bytes += std::mem::size_of::<Bucket>();
                cur = b.next.as_deref();
            }
        }
        bytes
    }
}

/// Iterator that walks every entry of a bucket chain in order, yielding
/// each [`SievingPrime`] by value. Consumes the chain — the caller passes
/// ownership of the head and gets back a "drain" iterator.
///
/// As iteration progresses, each fully-walked bucket is detached from the
/// front of `cur` and pushed onto a `walked` stack. Calling
/// [`BucketChainIter::into_chain`] re-fuses the two so every bucket
/// (walked and unwalked) is returned for recycling.
pub struct BucketChainIter {
    /// Front of the chain we have not yet finished walking. The next entry
    /// to yield (if any) lives at index [`cursor`](Self::cursor) of the
    /// `Box<Bucket>` at the head of `cur`.
    cur: Option<Box<Bucket>>,
    /// Index of the next entry within the head of `cur`.
    cursor: usize,
    /// Stack of buckets already drained, chained via their `next` field.
    /// Top of stack = most recently drained bucket; recycling order
    /// doesn't matter for correctness.
    walked: Option<Box<Bucket>>,
}

impl BucketChainIter {
    pub fn new(chain: Option<Box<Bucket>>) -> Self {
        BucketChainIter { cur: chain, cursor: 0, walked: None }
    }

    /// Recover every bucket (walked + unwalked) for recycling. The
    /// returned chain is "all the boxes we were given on `new()`" — they
    /// may be in a different link order, but the box count is preserved.
    /// Every bucket has `len = 0` after this call, so a partial drain
    /// (calling `into_chain` before the iterator is exhausted) still
    /// yields a chain ready for [`MemoryPool::recycle_chain`].
    pub fn into_chain(mut self) -> Option<Box<Bucket>> {
        match self.cur.take() {
            None => self.walked.take(),
            Some(mut head) => {
                // Walk through every still-attached bucket: clear its len
                // and find the tail so we can splice `walked` after it.
                head.len = 0;
                let mut tail: &mut Bucket = &mut head;
                while tail.next.is_some() {
                    tail = tail.next.as_mut().unwrap();
                    tail.len = 0;
                }
                tail.next = self.walked.take();
                Some(head)
            }
        }
    }
}

impl Iterator for BucketChainIter {
    type Item = SievingPrime;

    fn next(&mut self) -> Option<SievingPrime> {
        loop {
            let head = self.cur.as_deref()?;
            if self.cursor < head.len() {
                let sp = head.entries[self.cursor];
                self.cursor += 1;
                return Some(sp);
            }
            // Front bucket exhausted: detach it, push onto `walked`,
            // advance `cur` to its tail, and reset cursor.
            let mut taken = self.cur.take().unwrap();
            self.cur = taken.next.take();
            taken.len = 0;
            taken.next = self.walked.take();
            self.walked = Some(taken);
            self.cursor = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sieving_prime_pack_unpack_roundtrip() {
        for &(p, mi, wi) in &[
            (7u32, 0u32, 0u32),
            (13, 100, 3),
            (u32::MAX, MAX_MULTIPLE_INDEX, MAX_WHEEL_INDEX),
            (1_000_003, 524_279, 7),
            (104_729, 1, 8),
        ] {
            let sp = SievingPrime::new(p, mi, wi);
            assert_eq!(sp.prime(), p, "prime roundtrip");
            assert_eq!(sp.multiple_index(), mi, "mi roundtrip");
            assert_eq!(sp.wheel_index(), wi, "wi roundtrip");
        }
    }

    #[test]
    fn sieving_prime_size_is_8_bytes() {
        assert_eq!(std::mem::size_of::<SievingPrime>(), 8);
    }

    #[test]
    fn bucket_capacity_matches_8kb_target() {
        // 1022 entries × 8 B = 8176 B; +16 B header → 8192 B (= BUCKET_BYTES).
        assert_eq!(BUCKET_ENTRIES, 1022);
        // The Box<Bucket> heap allocation is exactly `size_of::<Bucket>()`,
        // which we want ≤ BUCKET_BYTES (it can be slightly less due to
        // niche-packing of the Option<Box<_>>, but never more).
        assert!(std::mem::size_of::<Bucket>() <= BUCKET_BYTES,
                "Bucket = {} B exceeds BUCKET_BYTES = {} B",
                std::mem::size_of::<Bucket>(), BUCKET_BYTES);
    }

    #[test]
    fn empty_bucket_sieve_take_returns_none() {
        let mut bs = BucketSieve::new(8);
        for seg in 0..8 {
            assert!(bs.take_segment(seg).is_none());
        }
    }

    #[test]
    fn insert_one_then_drain() {
        let mut bs = BucketSieve::new(4);
        bs.insert(2, 13, 100, 3);
        // Other segments untouched.
        assert!(bs.take_segment(0).is_none());
        assert!(bs.take_segment(1).is_none());
        assert!(bs.take_segment(3).is_none());

        let chain = bs.take_segment(2).expect("seg 2 should have a chain");
        assert!(chain.next.is_none(), "single insert should not chain");
        assert_eq!(chain.len(), 1);
        let sp = chain.entries()[0];
        assert_eq!(sp.prime(), 13);
        assert_eq!(sp.multiple_index(), 100);
        assert_eq!(sp.wheel_index(), 3);
    }

    #[test]
    fn insert_under_capacity_uses_single_bucket() {
        let mut bs = BucketSieve::new(2);
        let n = BUCKET_ENTRIES;
        for i in 0..n {
            bs.insert(0, (i as u32) + 7, (i as u32) % 1024, (i as u32) % 8);
        }
        let chain = bs.take_segment(0).expect("chain expected");
        assert!(chain.next.is_none(), "{} entries should fit in one bucket", n);
        assert_eq!(chain.len(), n);
        for i in 0..n {
            let sp = chain.entries()[i];
            assert_eq!(sp.prime(), (i as u32) + 7, "prime[{i}]");
            assert_eq!(sp.multiple_index(), (i as u32) % 1024, "mi[{i}]");
            assert_eq!(sp.wheel_index(), (i as u32) % 8, "wi[{i}]");
        }
    }

    #[test]
    fn bucket_overflow_chains_and_preserves_all_entries() {
        let mut bs = BucketSieve::new(1);
        let n = BUCKET_ENTRIES * 3 + 17; // forces ≥ 4 buckets
        for i in 0..n {
            bs.insert(0, (i as u32) + 1, (i as u32) % 1024, (i as u32) % 8);
        }
        let chain = bs.take_segment(0).expect("chain expected");

        // Walk the chain and collect every entry.
        let mut got: Vec<SievingPrime> = Vec::new();
        let mut cur: Option<&Bucket> = Some(&chain);
        let mut bucket_count = 0;
        while let Some(b) = cur {
            bucket_count += 1;
            got.extend_from_slice(b.entries());
            cur = b.next.as_deref();
        }
        assert_eq!(bucket_count, (n + BUCKET_ENTRIES - 1) / BUCKET_ENTRIES,
                   "expected {} buckets for {} entries",
                   (n + BUCKET_ENTRIES - 1) / BUCKET_ENTRIES, n);
        assert_eq!(got.len(), n);

        // Inserts go to the *front* of the chain (LIFO across buckets,
        // FIFO within a bucket). So the entries we see when walking the
        // chain head→tail are: last-bucket-of-inserts first, etc. The
        // multiset must still match exactly.
        let mut expected: Vec<SievingPrime> = (0..n)
            .map(|i| SievingPrime::new((i as u32) + 1, (i as u32) % 1024, (i as u32) % 8))
            .collect();
        expected.sort_by_key(|sp| sp.prime());
        let mut got_sorted = got.clone();
        got_sorted.sort_by_key(|sp| sp.prime());
        assert_eq!(got_sorted, expected, "drained set must equal inserted set");
    }

    #[test]
    fn multi_segment_insert_keeps_segments_isolated() {
        let mut bs = BucketSieve::new(16);
        // Round-robin distribute 5 K primes across 16 segments.
        let n = 5_000usize;
        for i in 0..n {
            let seg = i % 16;
            bs.insert(seg, (i as u32) + 2, (i as u32) % 4096, (i as u32) % 8);
        }
        // Each segment should hold exactly n/16 ± 1 entries.
        for seg in 0..16 {
            let chain = bs.take_segment(seg).expect("chain expected");
            let mut count = 0;
            let mut cur: Option<&Bucket> = Some(&chain);
            while let Some(b) = cur {
                for sp in b.entries() {
                    // Every entry in seg `s` was inserted with i ≡ s (mod 16),
                    // and prime = i + 2. Recover i and check the residue.
                    let i = sp.prime() - 2;
                    assert_eq!((i as usize) % 16, seg,
                               "stray entry: prime {} in seg {}", sp.prime(), seg);
                    count += 1;
                }
                cur = b.next.as_deref();
            }
            let lo = n / 16;
            assert!(count == lo || count == lo + 1,
                    "seg {} has {} entries (expected {} or {})", seg, count, lo, lo + 1);
        }
    }

    #[test]
    fn pool_reuses_recycled_buckets() {
        let mut bs = BucketSieve::new(1);
        let n = BUCKET_ENTRIES * 3;
        for i in 0..n {
            bs.insert(0, (i as u32) + 1, (i as u32) % 1024, (i as u32) % 8);
        }
        assert_eq!(bs.pool().freelist_len(), 0,
                   "pool should be empty before recycling");
        let chain = bs.take_segment(0);
        bs.recycle_chain(chain);
        let after = bs.pool().freelist_len();
        assert_eq!(after, 3, "all 3 buckets should be on the freelist");

        // Re-insert: the next 3 buckets must come from the freelist.
        for i in 0..n {
            bs.insert(0, (i as u32) + 1, (i as u32) % 1024, (i as u32) % 8);
        }
        assert_eq!(bs.pool().freelist_len(), 0,
                   "pool drained after re-insert of n=3 buckets");
    }

    #[test]
    fn pool_capacity_caps_freelist_growth() {
        // Force more recycled buckets than POOL_CAPACITY allows.
        let mut bs = BucketSieve::new(1);
        let target_buckets = POOL_CAPACITY + 5;
        let n = BUCKET_ENTRIES * target_buckets;
        for i in 0..n {
            bs.insert(0, (i as u32) + 1, (i as u32) % 1024, (i as u32) % 8);
        }
        let chain = bs.take_segment(0);
        bs.recycle_chain(chain);
        assert_eq!(bs.pool().freelist_len(), POOL_CAPACITY,
                   "freelist must cap at POOL_CAPACITY = {}", POOL_CAPACITY);
    }

    #[test]
    fn bucket_chain_iter_yields_every_entry() {
        let mut bs = BucketSieve::new(1);
        let n = BUCKET_ENTRIES * 2 + 3;
        let mut expected: Vec<SievingPrime> = Vec::with_capacity(n);
        for i in 0..n {
            let p = (i as u32) + 1;
            let mi = (i as u32) % 1024;
            let wi = (i as u32) % 8;
            bs.insert(0, p, mi, wi);
            expected.push(SievingPrime::new(p, mi, wi));
        }
        let chain = bs.take_segment(0);
        let mut iter = BucketChainIter::new(chain);
        let mut got: Vec<SievingPrime> = Vec::with_capacity(n);
        while let Some(sp) = iter.next() {
            got.push(sp);
        }
        assert_eq!(got.len(), n, "iter should yield every entry");

        // Multiset equality (chain order ≠ insert order).
        let mut got_sorted = got.clone();
        got_sorted.sort_by_key(|sp| sp.prime());
        let mut exp_sorted = expected.clone();
        exp_sorted.sort_by_key(|sp| sp.prime());
        assert_eq!(got_sorted, exp_sorted);

        // After full iteration the chain is empty (every bucket cleared)
        // and recyclable.
        let drained = iter.into_chain();
        let mut bucket_count = 0;
        let mut cur: Option<&Bucket> = drained.as_deref();
        while let Some(b) = cur {
            assert_eq!(b.len(), 0, "every bucket should be cleared post-iter");
            bucket_count += 1;
            cur = b.next.as_deref();
        }
        assert_eq!(bucket_count, (n + BUCKET_ENTRIES - 1) / BUCKET_ENTRIES);
    }

    #[test]
    fn into_chain_clears_buckets_on_partial_drain() {
        // Stop iteration mid-stream and verify `into_chain()` still
        // returns a recyclable chain — every box reachable, every len = 0.
        let mut bs = BucketSieve::new(1);
        let n = BUCKET_ENTRIES * 3 + 7;
        for i in 0..n {
            bs.insert(0, (i as u32) + 1, (i as u32) % 1024, (i as u32) % 8);
        }
        let chain = bs.take_segment(0);
        let mut iter = BucketChainIter::new(chain);
        // Walk only a handful of entries — leaves `cur` and possibly `walked`
        // both populated.
        for _ in 0..(BUCKET_ENTRIES + 5) {
            iter.next().expect("iter should yield");
        }

        let drained = iter.into_chain();
        let mut cur: Option<&Bucket> = drained.as_deref();
        let mut bucket_count = 0;
        while let Some(b) = cur {
            assert_eq!(b.len(), 0, "into_chain must clear every bucket's len");
            bucket_count += 1;
            cur = b.next.as_deref();
        }
        assert_eq!(bucket_count, (n + BUCKET_ENTRIES - 1) / BUCKET_ENTRIES,
                   "every original bucket box must remain reachable");

        // And the chain must be safely recyclable through the pool.
        bs.recycle_chain(drained);
        assert_eq!(bs.pool().freelist_len(), bucket_count);
    }

    #[test]
    fn round_trip_10k_primes_across_64_segments() {
        // Stresses insert + drain at moderate scale: 10 K entries spread
        // across 64 segments with a mix of bucket-fill ratios.
        const N: usize = 10_000;
        const SEGS: usize = 64;
        let mut bs = BucketSieve::new(SEGS);

        // Deterministic LCG for reproducible distribution.
        let mut rng = 0x12345_6789u64;
        let mut seg_of: Vec<usize> = Vec::with_capacity(N);
        for i in 0..N {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let seg = (rng >> 32) as usize % SEGS;
            seg_of.push(seg);
            bs.insert(seg, (i as u32) + 7, (i as u32) % 4096, (i as u32) % 8);
        }

        let mut total_drained = 0;
        for seg in 0..SEGS {
            let chain = bs.take_segment(seg);
            let mut iter = BucketChainIter::new(chain);
            while let Some(sp) = iter.next() {
                let i = sp.prime() - 7;
                assert_eq!(seg_of[i as usize], seg,
                           "prime {} drained from seg {} but inserted into seg {}",
                           sp.prime(), seg, seg_of[i as usize]);
                total_drained += 1;
            }
            bs.recycle_chain(iter.into_chain());
        }
        assert_eq!(total_drained, N, "every inserted prime must be drained exactly once");
        assert!(bs.pool().freelist_len() > 0, "pool should hold recycled buckets");
    }
}
