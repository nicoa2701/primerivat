/// Walisch-style dense PhiCache pour φ(n, a) en O(1).
///
/// Chaque niveau a ∈ [0, max_a] stocke un bitset dense sur [1, max_x] :
///   - `bits[b]` : bit j = 1 ssi l'entier b·64+j+1 est copremier à p₁..pₐ
///   - `count[b]` : φ(b·64, a) = # entiers dans [1, b·64] copremiers
///
/// Requête φ(n, a) pour n ≤ max_x et a ≤ max_a :
///   block = (n-1)/64, bit_pos = (n-1)%64
///   mask  = (2 << bit_pos) - 1   (bits 0..bit_pos inclus)
///   result = count[block] + popcount(bits[block] & mask)
///
/// Complexité mémoire : (max_a+1) × ⌈max_x/64⌉ × 12 octets.
/// Bornée à CACHE_BYTES_MAX (16 MB) pour tenir en L3.

/// Nombre maximal de niveaux couverts par le cache.
pub const CACHE_A_MAX: usize = 100;

/// Taille maximale du cache en octets (16 MB).
pub const CACHE_BYTES_MAX: usize = 16 * 1024 * 1024;

/// Un bloc de 64 entiers consécutifs au niveau a.
#[derive(Clone)]
pub struct PhiCacheEntry {
    /// φ(block_start, a) : # entiers dans [1, block·64] copremiers à p₁..pₐ.
    pub count: u32,
    /// Bit j = 1 ssi block·64+j+1 est copremier à p₁..pₐ  (j ∈ 0..63).
    pub bits: u64,
}

pub struct PhiCache {
    /// `levels[a]` : vecteur de PhiCacheEntry sur [1, max_x], niveau a.
    levels: Vec<Vec<PhiCacheEntry>>,
    /// Nombre de niveaux de cache (= min(primes.len(), CACHE_A_MAX)).
    pub max_a: usize,
    /// n maximal couvert : max_x ≥ 1.
    pub max_x: u64,
    /// π(n) pour n ∈ [0, max_x].  Permet aux feuilles B avec n ≤ max_x
    /// de s'affranchir du tableau large_pi.
    pi_small: Vec<u32>,
}

impl PhiCache {
    /// Construit le cache pour x donné et les premiers `primes` = p₁, p₂, …
    ///
    /// `max_x` = min(x^(1/2.3), borne_mémoire) arrondi au multiple de 64.
    pub fn new(x: u128, primes: &[u64]) -> Self {
        let a_max = CACHE_A_MAX.min(primes.len());

        // x^(1/2.3) — la plupart des feuilles de la récursion Mapes
        // ont n ≤ max_x quand a ≤ CACHE_A_MAX.
        let max_x_ideal: u64 = {
            let exp = 1.0_f64 / 2.3_f64;
            let v = (x as f64).powf(exp) as u64;
            v.max(64)
        };

        // Borne mémoire : (a_max+1) niveaux × n_blocks blocs × 12 octets.
        let max_blocks_by_mem = CACHE_BYTES_MAX / ((a_max + 1) * 12).max(1);
        let n_blocks_ideal = ((max_x_ideal as usize) + 63) / 64;
        let n_blocks = n_blocks_ideal.min(max_blocks_by_mem).max(1);
        let max_x = (n_blocks * 64) as u64;

        // Niveau 0 : φ(n, 0) = n → tous les entiers sont copremiers.
        // count[b] = b*64 = φ(b·64, 0), bits = 0xFFFF…FF.
        let level0: Vec<PhiCacheEntry> = (0..n_blocks)
            .map(|b| PhiCacheEntry {
                count: (b as u32) * 64,
                bits: u64::MAX,
            })
            .collect();

        let mut levels = Vec::with_capacity(a_max + 1);
        levels.push(level0);

        // Niveaux 1..=a_max : crible séquentiel.
        for k in 0..a_max {
            let p = primes[k] as usize;
            let mut level = levels[k].clone();

            // Effacer tous les multiples de p dans [1, max_x].
            let mut mult = p; // premier multiple = p lui-même
            while mult <= max_x as usize {
                // L'entier `mult` est en position (mult-1) dans le tableau.
                let idx = mult - 1;
                let block = idx / 64;
                let bit = idx % 64;
                // block < n_blocks est garanti car mult ≤ max_x = n_blocks*64.
                level[block].bits &= !(1u64 << bit);
                mult += p;
            }

            // Recalculer les count[] par scan cumulatif.
            let mut cumulative: u32 = 0;
            for entry in &mut level {
                entry.count = cumulative;
                cumulative += entry.bits.count_ones();
            }

            levels.push(level);
        }

        // Construire le tableau π(n) pour n ∈ [0, max_x] par crible simple.
        // Taille : max_x × 4 bytes ≈ 240 KB à x=1e11, 660 KB à x=1e12.
        let max_x_usize = max_x as usize;
        let mut is_prime = vec![false; max_x_usize + 1];
        for n in 2..=max_x_usize {
            is_prime[n] = true;
        }
        {
            let mut p = 2usize;
            while p * p <= max_x_usize {
                if is_prime[p] {
                    let mut m = p * p;
                    while m <= max_x_usize {
                        is_prime[m] = false;
                        m += p;
                    }
                }
                p += 1;
            }
        }
        let mut pi_small = vec![0u32; max_x_usize + 1];
        let mut pi_count = 0u32;
        for n in 0..=max_x_usize {
            if is_prime[n] {
                pi_count += 1;
            }
            pi_small[n] = pi_count;
        }

        PhiCache { levels, max_a: a_max, max_x, pi_small }
    }

    /// π(n) pour n ≤ max_x, O(1).
    #[inline]
    pub fn pi(&self, n: u64) -> u64 {
        debug_assert!(n <= self.max_x, "n={n} > max_x={}", self.max_x);
        self.pi_small[n as usize] as u64
    }

    /// φ(n, a) via lookup O(1).
    ///
    /// Préconditions : n ≥ 1, n ≤ max_x, a ≤ max_a.
    #[inline]
    pub fn phi(&self, n: u64, a: usize) -> u64 {
        debug_assert!(n >= 1 && n <= self.max_x, "n={n} hors cache [1,{}]", self.max_x);
        debug_assert!(a <= self.max_a, "a={a} > max_a={}", self.max_a);

        let idx = (n - 1) as usize;
        let block = idx / 64;
        let bit_pos = idx % 64;
        // mask = bits 0..=bit_pos : tous les entiers dans le bloc ≤ n.
        let mask: u64 = if bit_pos < 63 {
            (1u64 << (bit_pos + 1)) - 1
        } else {
            u64::MAX
        };
        let entry = &self.levels[a][block];
        entry.count as u64 + (entry.bits & mask).count_ones() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sieve::sieve_to;

    /// Référence naïve : φ(n, a) par crible d'exclusion.
    fn phi_naive(n: u64, a: usize, primes: &[u64]) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut count = 0u64;
        'outer: for k in 1..=n {
            for &p in primes.iter().take(a) {
                if k % p == 0 {
                    continue 'outer;
                }
            }
            count += 1;
        }
        count
    }

    #[test]
    fn phi_cache_small_values() {
        // Tester sur x = 1e8 (relativement petit pour le test)
        let x: u128 = 100_000;
        let (_, primes) = sieve_to(1000);
        let cache = PhiCache::new(x, &primes);

        // Tester quelques valeurs de n et a
        for a in 0..=10.min(cache.max_a) {
            for n in [1u64, 2, 3, 5, 10, 30, 100, 1000, 10000] {
                if n <= cache.max_x {
                    let got = cache.phi(n, a);
                    let expected = phi_naive(n, a, &primes);
                    assert_eq!(got, expected, "phi({n}, {a}) : got {got}, expected {expected}");
                }
            }
        }
    }

    #[test]
    fn phi_cache_level0_is_identity() {
        let x: u128 = 10_000;
        let (_, primes) = sieve_to(100);
        let cache = PhiCache::new(x, &primes);
        // φ(n, 0) = n
        for n in 1..=cache.max_x.min(500) {
            assert_eq!(cache.phi(n, 0), n, "phi({n}, 0) doit valoir {n}");
        }
    }

    #[test]
    fn phi_cache_level1_counts_odds() {
        // φ(n, 1) = ⌈n/2⌉  (entiers impairs dans [1,n])
        let x: u128 = 10_000;
        let (_, primes) = sieve_to(100);
        let cache = PhiCache::new(x, &primes);
        for n in 1..=cache.max_x.min(1000) {
            let expected = (n + 1) / 2;
            let got = cache.phi(n, 1);
            assert_eq!(got, expected, "phi({n}, 1) : got {got}, expected {expected}");
        }
    }
}
