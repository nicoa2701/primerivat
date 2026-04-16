pub mod baseline;
pub mod bit;
pub mod dr;
pub mod math;
pub mod parameters;
pub mod phi;
pub mod phi_cache;
pub mod pi_table;
pub mod primes;
pub mod segment;
pub mod sieve;

/// Stable public entry point for prime counting.
pub fn prime_pi(x: u128) -> u128 {
    baseline::prime_pi(x)
}

/// Stable public entry point with configurable parallelism.
pub fn prime_pi_with_threads(x: u128, threads: usize) -> u128 {
    baseline::prime_pi_with_threads(x, threads)
}

/// Deléglise-Rivat prime counting: parallel prefix-popcount for hard leaves.
///
/// Uses `dr::prime_pi_dr_v2` which combines Lucy early-stop with a parallel
/// two-pass prefix-popcount sweep (O(x^(2/3)/log²x), 32 KB per thread).
#[allow(non_snake_case)]
pub fn deleglise_rivat(x: u128) -> u128 {
    dr::prime_pi_dr_v2(x)
}

/// Backward-compatible alias — threads not yet used by the DR engine.
#[allow(non_snake_case)]
pub fn deleglise_rivat_with_threads(x: u128, _threads: usize) -> u128 {
    dr::prime_pi_dr_v2(x)
}
