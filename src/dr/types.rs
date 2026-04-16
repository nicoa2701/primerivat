use crate::parameters::{
    DrDomains, FrontierSet, IndexDomain, Parameters, S1Domain, S1Rule, S2EasyDomain, S2EasyRule,
    S2HardDomain, S2HardRule, S2TrivialDomain, TrivialDomain,
};
use crate::primes::PrimeTable;

use super::DrContributions;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainSet {
    pub domains: DrDomains,
    pub contributions: DrContributions,
    pub result: u128,
}

pub struct DrContext<'a> {
    pub params: Parameters,
    pub x: u128,
    pub y: u128,
    pub z: u128,
    pub z_usize: usize,
    pub a: usize,
    pub b: usize,
    pub phi_x_a: u128,
    pub primes: &'a PrimeTable,
    pub small: &'a [u64],
    pub large: &'a [u64],
}

impl<'a> DrContext<'a> {
    pub fn new(
        params: Parameters,
        phi_x_a: u128,
        primes: &'a PrimeTable,
        small: &'a [u64],
        large: &'a [u64],
    ) -> Self {
        Self {
            x: params.x,
            y: params.y,
            z: params.z,
            z_usize: params.z_usize,
            a: params.a,
            b: params.b,
            phi_x_a,
            params,
            primes,
            small,
            large,
        }
    }

    pub fn first_prime_after_a(&self) -> Option<u128> {
        self.primes.first_after_a(self.a).map(|p| p as u128)
    }

    pub fn smallest_triple_product_above_a(&self) -> Option<u128> {
        self.first_prime_after_a()
            .and_then(|p| p.checked_mul(p))
            .and_then(|p2| p2.checked_mul(self.first_prime_after_a()?))
    }

    pub fn meissel_s3_is_trivial_zero(&self) -> bool {
        self.first_prime_after_a().is_none_or(|p| {
            p.checked_mul(p)
                .and_then(|p2| p2.checked_mul(p))
                .is_none_or(|cube| cube > self.x)
        })
    }

    pub fn active_prime_index_range(&self) -> std::ops::Range<usize> {
        self.a..self.b
    }

    pub fn pi(&self, n: u128) -> u64 {
        crate::sieve::pi_at(n, self.x, self.z_usize, self.small, self.large)
    }

    pub fn prime_at_index(&self, index: usize) -> Option<u128> {
        self.primes.get(index).map(|p| p as u128)
    }

    pub fn s2_term_at(&self, index: usize) -> Option<u128> {
        let p = self.prime_at_index(index)?;
        let pi_x_over_p = self.pi(self.x / p) as u128;
        Some(pi_x_over_p - index as u128)
    }

    pub fn quotient_at(&self, index: usize) -> Option<u128> {
        let p = self.prime_at_index(index)?;
        Some(self.x / p)
    }

    pub fn easy_start(&self) -> usize {
        self.easy_start_with_rule(self.frontier_set().s2_easy)
    }

    pub fn easy_start_with_rule(&self, rule: S2EasyRule) -> usize {
        let mut start = self.b;
        while start > self.a {
            let index = start - 1;
            if self
                .s2_term_at(index)
                .is_some_and(|term| rule.matches(term))
            {
                start -= 1;
            } else {
                break;
            }
        }
        start
    }

    pub fn hard_start(&self, easy_start: usize) -> usize {
        self.hard_start_with_rule(easy_start, self.frontier_set().s2_hard)
    }

    pub fn hard_start_with_rule(&self, easy_start: usize, rule: S2HardRule) -> usize {
        let mut start = easy_start;
        while start > self.a {
            let index = start - 1;
            let Some(term) = self.s2_term_at(index) else {
                break;
            };
            if rule.matches(term) {
                start -= 1;
            } else {
                break;
            }
        }
        start
    }

    pub fn dr_domains(&self) -> DrDomains {
        self.dr_domains_with_frontiers(self.frontier_set())
    }

    pub fn dr_domains_with_frontiers(&self, frontiers: FrontierSet) -> DrDomains {
        let easy_start = self.easy_start_with_rule(frontiers.s2_easy);
        let hard_start = self.hard_start_with_rule(easy_start, frontiers.s2_hard);
        self.params
            .dr_domains_from_frontiers(hard_start, easy_start, frontiers)
    }

    pub fn frontier_set(&self) -> FrontierSet {
        self.params.frontier_set()
    }

    pub fn active_domain(&self) -> IndexDomain {
        self.dr_domains().active
    }

    pub fn s1_domain(&self) -> S1Domain {
        self.dr_domains().s1
    }

    pub fn s1_rule(&self) -> S1Rule {
        self.frontier_set().s1
    }

    pub fn s2_easy_domain(&self) -> S2EasyDomain {
        self.dr_domains().s2_easy
    }

    pub fn s2_easy_rule(&self) -> S2EasyRule {
        self.frontier_set().s2_easy
    }

    pub fn s2_hard_domain(&self) -> S2HardDomain {
        self.dr_domains().s2_hard
    }

    pub fn s2_hard_rule(&self) -> S2HardRule {
        self.frontier_set().s2_hard
    }

    pub fn s2_trivial_domain(&self) -> S2TrivialDomain {
        self.dr_domains().s2_trivial
    }

    pub fn easy_domain(&self) -> IndexDomain {
        self.s2_easy_domain().leaves
    }

    pub fn ordinary_domain(&self) -> IndexDomain {
        self.s1_domain().leaves
    }

    pub fn hard_domain(&self) -> IndexDomain {
        self.s2_hard_domain().leaves
    }

    pub fn trivial_domain(&self) -> TrivialDomain {
        self.s2_trivial_domain().leaves
    }

    pub fn domain_set(&self) -> DomainSet {
        self.domain_set_with_frontiers(self.frontier_set())
    }

    pub fn domain_set_with_frontiers(&self, frontiers: FrontierSet) -> DomainSet {
        let domains = self.dr_domains_with_frontiers(frontiers);
        let sum_range = |range: IndexDomain| -> u128 {
            (range.start..range.end)
                .map(|j| self.s2_term_at(j).expect("domain index must be valid"))
                .sum()
        };
        let contributions = DrContributions {
            trivial: if domains.s2_trivial.leaves.s3_is_zero {
                0
            } else {
                super::trivial::trivial_leaves(self)
            },
            easy: sum_range(domains.s2_easy.leaves),
            ordinary: sum_range(domains.s1.leaves),
            hard: sum_range(domains.s2_hard.leaves),
        };
        let result = self.phi_x_a + self.a as u128 - 1 - contributions.total();

        DomainSet {
            domains,
            contributions,
            result,
        }
    }
}
