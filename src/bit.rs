/// 1-indexed Fenwick tree (Binary Indexed Tree).
///
/// Supports point updates (`add`) and prefix queries (`prefix_sum`) in O(log n).
///
/// Intended for the Deléglise-Rivat hard-leaf sweep: the BIT is initialised
/// with +1 for every integer in a counting window, then composites are removed
/// one by one with `add(q, -1)` as the sieve sweeps downward.  At any point,
/// `prefix_sum(v)` yields π(v) − 1 inside the current window.
pub struct Bit {
    data: Vec<i32>,
}

impl Bit {
    /// Creates a zeroed BIT for indices 1..=`n`.
    pub fn new(n: usize) -> Self {
        Self {
            data: vec![0i32; n + 1],
        }
    }

    /// Resets all tree entries to zero while keeping the allocation.
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Rebuilds the Fenwick tree from a set of 1-indexed point positions.
    ///
    /// Each position contributes `+1`. The backing allocation is reused.
    /// Returns the number of inserted positions.
    pub fn rebuild_from_positions<I>(&mut self, positions: I) -> usize
    where
        I: IntoIterator<Item = usize>,
    {
        self.clear();

        let mut count = 0usize;
        for i in positions {
            debug_assert!(i >= 1 && i < self.data.len(), "BIT index out of range");
            self.data[i] = self.data[i].wrapping_add(1);
            count += 1;
        }

        for i in 1..self.data.len() {
            let j = i + (i & i.wrapping_neg());
            if j < self.data.len() {
                self.data[j] = self.data[j].wrapping_add(self.data[i]);
            }
        }

        count
    }

    /// Adds `delta` to position `i` (1-indexed). O(log n).
    #[inline]
    pub fn add(&mut self, mut i: usize, delta: i32) {
        debug_assert!(i >= 1 && i < self.data.len(), "BIT index out of range");
        while i < self.data.len() {
            self.data[i] = self.data[i].wrapping_add(delta);
            i += i & i.wrapping_neg();
        }
    }

    /// Returns the prefix sum over 1..=`i`. O(log n).
    #[inline]
    pub fn prefix_sum(&self, mut i: usize) -> i32 {
        debug_assert!(i < self.data.len(), "BIT index out of range");
        let mut s = 0i32;
        while i > 0 {
            s = s.wrapping_add(self.data[i]);
            i -= i & i.wrapping_neg();
        }
        s
    }

    /// Returns the sum over `l..=r`. O(log n).
    #[inline]
    pub fn range_sum(&self, l: usize, r: usize) -> i32 {
        if l > r {
            return 0;
        }
        let right = self.prefix_sum(r);
        let left = if l > 1 { self.prefix_sum(l - 1) } else { 0 };
        right - left
    }

    /// Maximum valid index (= n passed to `new`).
    pub fn capacity(&self) -> usize {
        self.data.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sum_after_point_adds() {
        let mut bit = Bit::new(10);
        bit.add(3, 1);
        bit.add(5, 1);
        bit.add(7, 1);

        assert_eq!(bit.prefix_sum(2), 0);
        assert_eq!(bit.prefix_sum(3), 1);
        assert_eq!(bit.prefix_sum(5), 2);
        assert_eq!(bit.prefix_sum(6), 2);
        assert_eq!(bit.prefix_sum(7), 3);
        assert_eq!(bit.prefix_sum(10), 3);
    }

    #[test]
    fn range_sum_basic() {
        let mut bit = Bit::new(10);
        for i in [2usize, 3, 5, 7] {
            bit.add(i, 1);
        }
        assert_eq!(bit.range_sum(1, 10), 4);
        assert_eq!(bit.range_sum(3, 7), 3); // 3, 5, 7
        assert_eq!(bit.range_sum(4, 6), 1); // 5
        assert_eq!(bit.range_sum(6, 10), 1); // 7
        assert_eq!(bit.range_sum(5, 3), 0); // empty
    }

    #[test]
    fn add_and_subtract() {
        let mut bit = Bit::new(8);
        for i in 1..=8 {
            bit.add(i, 1);
        }
        assert_eq!(bit.prefix_sum(8), 8);
        bit.add(3, -1);
        bit.add(5, -1);
        assert_eq!(bit.prefix_sum(8), 6);
        assert_eq!(bit.prefix_sum(4), 3); // 1, 2, 4
        assert_eq!(bit.prefix_sum(5), 3); // 1, 2, 4 (5 removed)
    }

    #[test]
    fn simulates_prime_counting_window() {
        // Initialise BIT with 1 for every integer in [1, 20],
        // then remove non-primes — prefix_sum(n) should equal π(n).
        const PRIMES_TO_20: &[usize] = &[2, 3, 5, 7, 11, 13, 17, 19];

        let mut bit = Bit::new(20);
        for i in 1..=20usize {
            bit.add(i, 1);
        }
        for i in 1..=20usize {
            if i == 1 || !PRIMES_TO_20.contains(&i) {
                bit.add(i, -1);
            }
        }

        assert_eq!(bit.prefix_sum(1), 0); // π(1)  = 0
        assert_eq!(bit.prefix_sum(2), 1); // π(2)  = 1
        assert_eq!(bit.prefix_sum(10), 4); // π(10) = 4
        assert_eq!(bit.prefix_sum(20), 8); // π(20) = 8
    }

    #[test]
    fn clear_keeps_capacity_and_resets_sums() {
        let mut bit = Bit::new(16);
        bit.add(3, 1);
        bit.add(9, 2);
        assert_eq!(bit.prefix_sum(16), 3);

        let capacity = bit.capacity();
        bit.clear();

        assert_eq!(bit.capacity(), capacity);
        assert_eq!(bit.prefix_sum(16), 0);
        bit.add(5, 1);
        assert_eq!(bit.prefix_sum(16), 1);
    }

    #[test]
    fn rebuild_from_positions_matches_point_updates() {
        let positions = [2usize, 3, 5, 7, 11];

        let mut rebuilt = Bit::new(16);
        let count = rebuilt.rebuild_from_positions(positions);

        let mut incremental = Bit::new(16);
        for pos in positions {
            incremental.add(pos, 1);
        }

        assert_eq!(count, positions.len());
        for i in 1..=16 {
            assert_eq!(rebuilt.prefix_sum(i), incremental.prefix_sum(i));
        }
    }
}
