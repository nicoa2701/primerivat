pub struct PrimeTable {
    primes: Vec<u64>,
}

impl PrimeTable {
    pub fn new(primes: Vec<u64>) -> Self {
        Self { primes }
    }

    pub fn len(&self) -> usize {
        self.primes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primes.is_empty()
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.primes
    }

    pub fn b(&self) -> usize {
        self.primes.len()
    }

    pub fn get(&self, index: usize) -> Option<u64> {
        self.primes.get(index).copied()
    }

    pub fn first_after_a(&self, a: usize) -> Option<u64> {
        self.get(a)
    }
}
