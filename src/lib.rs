pub mod baseline;
pub mod dr;
pub mod math;
pub mod parameters;
pub mod phi;
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

/// Deléglise-Rivat prime counting — current production engine.
///
/// Delegates to [`dr::prime_pi_dr_meissel_v4`]: hardware-adaptive α,
/// fused S2_hard + P2 sweep with monotonic leaf scan and `{7, 11}`
/// pre-sieve template, and a baseline fallback for small `x`
/// (when `π(α·∛x) ≤ 5`).
#[allow(non_snake_case)]
pub fn deleglise_rivat(x: u128) -> u128 {
    dr::prime_pi_dr_meissel_v4(x)
}

/// Backward-compatible alias — threads are managed by the engine's own
/// Rayon thread pool, so the `_threads` argument is ignored.
#[allow(non_snake_case)]
pub fn deleglise_rivat_with_threads(x: u128, _threads: usize) -> u128 {
    dr::prime_pi_dr_meissel_v4(x)
}
