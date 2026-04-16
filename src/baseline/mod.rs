pub mod s2;
pub mod s3;

use crate::parameters::Parameters;
use crate::phi::default_phi_computation;
use crate::primes::PrimeTable;
use crate::sieve::pi_at;

/// Computes π(x) — the number of primes ≤ x — using the current baseline
/// implementation built around a Lucy-Hedgehog-backed Meissel decomposition.
pub fn prime_pi(x: u128) -> u128 {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    prime_pi_with_threads(x, threads)
}

/// Same baseline implementation with explicit control of the S2 thread count.
#[allow(non_snake_case)]
pub fn prime_pi_with_threads(x: u128, threads: usize) -> u128 {
    if x < 2 {
        return 0;
    }

    let computation = default_phi_computation(x);
    let params = Parameters::from_tables(x, &computation.small, &computation.large);
    let primes = PrimeTable::new(computation.primes);

    debug_assert_eq!(
        primes.len(),
        pi_at(
            params.z,
            x,
            params.z_usize,
            &computation.small,
            &computation.large
        ) as usize
    );

    let s2_val = s2::s2(
        x,
        params.a,
        params.z_usize,
        primes.as_slice(),
        &computation.small,
        &computation.large,
        threads,
    );
    let s3_val = s3::s3(x, params.a, primes.as_slice());

    computation.phi_x_a + params.a as u128 - 1 - s2_val - s3_val
}
