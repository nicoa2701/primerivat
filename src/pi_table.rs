/// Table dense O(1) pour π(n) = #{premiers ≤ n}, n ∈ [0, max_x].
///
/// Construite par crible d'Ératosthène puis somme cumulative.
/// Taille mémoire : (max_x + 1) × 4 octets.
///
/// | x      | max_x = √x  | taille |
/// |--------|------------|--------|
/// | 1e11   | 316 227    |  1.3 MB |
/// | 1e13   | 3 162 277  |  12 MB  |
/// | 1e15   | 31 622 776 | 126 MB  |
///
/// Remplace `collect_phi` + `large_pi` dans `prime_pi_dr_meissel_v3` :
/// tous les nœuds feuilles de la récursion φ(x, a) avec n ≤ max_x
/// obtiennent π(n) en O(1) sans HashMap ni DFS préalable.
pub struct PiTable {
    /// `counts[n]` = π(n) pour n ∈ [0, max_x].
    counts: Vec<u32>,
}

impl PiTable {
    /// Construit la table pour n ∈ [0, max_x] par crible d'Ératosthène.
    pub fn new(max_x: u64) -> Self {
        let n = max_x as usize;

        // Crible d'Ératosthène sur [0, n].
        let mut sieve = vec![true; n + 1];
        if n >= 1 {
            sieve[0] = false;
            sieve[1] = false;
        }
        let mut p = 2usize;
        while p * p <= n {
            if sieve[p] {
                let mut m = p * p;
                while m <= n {
                    sieve[m] = false;
                    m += p;
                }
            }
            p += 1;
        }

        // Somme cumulative : counts[i] = #{premiers ≤ i}.
        let mut counts = vec![0u32; n + 1];
        let mut c = 0u32;
        for i in 0..=n {
            if sieve[i] {
                c += 1;
            }
            counts[i] = c;
        }

        PiTable { counts }
    }

    /// π(n) en O(1). Précondition : n ≤ max_x.
    #[inline]
    pub fn pi(&self, n: u64) -> u64 {
        debug_assert!(
            (n as usize) < self.counts.len(),
            "PiTable::pi({n}) hors borne max_x={}",
            self.max_x()
        );
        self.counts[n as usize] as u64
    }

    /// n maximal couvert par la table.
    #[inline]
    pub fn max_x(&self) -> u64 {
        (self.counts.len().saturating_sub(1)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_table_small() {
        let t = PiTable::new(30);
        // π(0..=1) = 0, π(2) = 1, π(3) = 2, π(5) = 3, π(7) = 4, π(11) = 5, π(30) = 10
        assert_eq!(t.pi(0), 0);
        assert_eq!(t.pi(1), 0);
        assert_eq!(t.pi(2), 1);
        assert_eq!(t.pi(3), 2);
        assert_eq!(t.pi(4), 2);
        assert_eq!(t.pi(5), 3);
        assert_eq!(t.pi(10), 4);
        assert_eq!(t.pi(30), 10);
    }

    #[test]
    fn pi_table_known_values() {
        let t = PiTable::new(1_000_000);
        assert_eq!(t.pi(1_000_000), 78_498); // π(10^6)
        assert_eq!(t.pi(100_000), 9_592);    // π(10^5)
        assert_eq!(t.pi(10_000), 1_229);     // π(10^4)
    }
}
