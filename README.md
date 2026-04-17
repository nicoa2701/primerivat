# primerivat

*[Version française](LISEZMOI.md)*

`primerivat` is a Rust implementation of the prime-counting function `π(x)`
based on the **Deléglise–Rivat** algorithm.

The current engine (`prime_pi_dr_meissel_v4`) is active by default and
computes π(x) exactly up to at least `x = 1e17` on commodity hardware.

## Quick start

```bash
cargo build --release
cargo run --release -- 1e13        # π(1e13) = 346 065 536 839
cargo run --release -- 1e15        # π(1e15) = 29 844 570 422 669
```

The binary prints the short git hash (`[dr-meissel4 <hash>]`) so that multiple
builds can be compared side by side.

## Measured performance

Reference machine: i5-9300H (4C/8T, 8 MB L3, DDR4-2666), release build,
multi-threaded, adaptive α.

| x | time | π(x) |
|---|---:|---|
| `1e11` | 0.016 s | 4 118 054 813 |
| `1e13` | 0.30 s  | 346 065 536 839 |
| `1e15` | 7.1 s   | 29 844 570 422 669 |
| `1e16` | 37.7 s  | 279 238 341 033 925 |
| `1e17` | 193 s (α=2) | 2 623 557 157 654 233 |

These numbers are well below the original roadmap targets — further gains are
expected to come from cache/SIMD tuning rather than algorithmic changes.

## Algorithm

The engine implements the classical Meissel–Deléglise–Rivat decomposition:

```
π(x) = φ(x, a) + a − 1 − P2
     = (S1 + S2_hard) + a − 1 − P2
```

with `a = π(y)`, `y = α·∛x`, and:

- **S1** — `Σ μ(m)·φ(x/m, c)` over squarefree `m` with prime factors ≤ `y`,
  computed by a recursive DFS with `c = 5`.
- **S2_hard** — `−Σ_{b=c+1..b_max} Σ_m μ(m)·φ(x/(p_b·m), b−1)`, evaluated
  through three specialized paths.
- **P2** — `Σ_{y < p ≤ √x} (π(x/p) − π(p) + 1)`, fused with S2_hard in a
  single parallel sweep.

### The three S2_hard paths

| `bi` range | `p_b` range | Path | Formula for `φ(n, b−1)` |
|---|---|---|---|
| `0..n_hard` | ≤ √y | hard leaves | `phi_vec` + segmented sieve popcount |
| `n_hard..b_ext` | (√y, x¹ᐟ⁴] | `phi_easy` (`m = p_l`) | `phi_vec` + segmented sieve popcount |
| `b_ext..n_all` | (x¹ᐟ⁴, y] | `ext_easy` (`m = p_l`) | closed form `π(n) − (b−2)` (clamped ≥ 1) |

The `ext_easy` closed form `φ(n, b−1) = π(n) − (b−2)` is valid when
`n ≥ p_{b−1}` and avoids maintaining `phi_vec` for the bulk of primes.

### Adaptive α

`y = α·∛x` with α chosen per magnitude of `x`:

- `x < 3e16` → α = 1.0 (lower overhead for small `x`)
- `x ≥ 3e16` → α = 2.0 (~41% faster at `1e17` thanks to fewer sieve windows)

The auto-selection can be overridden from the CLI with `-a <α>` or
`--alpha <α>`. Accepted range:

- `x ≤ 1e15` → any `α ∈ [1, 2]` (e.g. `1.5`)
- `x > 1e15` → only `α ∈ {1, 2}` (intermediate values are rejected)

> ⚠️ Only `α ∈ {1.0, 2.0}` is safe. Intermediate values (e.g. α = 1.25)
> produce wrong results at specific `x` — root cause not yet identified.

### Fallback for small x

If `a ≤ C = 5`, the driver falls back to the Lucy–Meissel baseline
(`baseline::prime_pi`) since `phi_small_a` cannot be used.

## Parallelism

The S2_hard + P2 sweep is split into disjoint Rayon bands. A two-pass scheme
is used:

1. **Pass 1** — each band computes its local `phi_deltas[bi]` and prime count.
2. **Sequential prefix scan** — turns the local deltas into global offsets.
3. **Pass 2** — processes the leaves and P2 queries per band with the correct
   offsets.

The Rayon stack is raised to 32 MiB at startup to cover the recursion depth
reached at `x ≥ 1e15`.

## Project layout

```text
src/
├── bit.rs          # Fenwick tree (used implicitly via count_primes_upto_int)
├── segment.rs      # WheelSieve30 (wheel-mod-30 segmented sieve), primes_up_to
├── baseline/       # Lucy–Meissel reference; used as small-x fallback
├── dr/
│   ├── mod.rs      # prime_pi_dr_meissel_v4 (active) + legacy variants
│   ├── hard.rs     # s2_hard_sieve_par — parallel S2_hard + P2 + ext_easy sweep
│   ├── easy.rs     # experimental S2_easy partitions (inactive)
│   ├── ordinary.rs # exploratory ordinary-leaf regions
│   ├── trivial.rs
│   └── types.rs    # DrContext, domains, boundary rules
├── phi.rs          # s1_ordinary DFS, phi_small_a (inclusion/exclusion c ≤ 5)
├── sieve.rs        # Lucy sieve (seed_primes)
├── parameters.rs   # shared types
├── lib.rs          # public API
└── main.rs         # CLI
build.rs            # injects GIT_HASH for runtime display
```

## Public API

```rust
rivat3::deleglise_rivat(x)                       // primary entry point
rivat3::deleglise_rivat_with_threads(x, threads) // same, with thread budget
rivat3::prime_pi(x)                              // Lucy–Meissel baseline
rivat3::prime_pi_with_threads(x, threads)
```

## CLI

```bash
# Direct computation (the DR v4 engine manages its own Rayon parallelism)
cargo run --release -- 1e13

# Override the auto-selected α (short or long form)
cargo run --release -- 1e17 -a 2
cargo run --release -- 1e13 --alpha 1

# Profiling
cargo run --release -- --profile 1e11     # baseline phase profile
cargo run --release -- --dr-profile 1e13  # DR phase profile
cargo run --release -- --lucy-profile 1e13

# Multi-x sweep
cargo run --release -- --sweep 1e12
```

## Validation

```bash
cargo test                          # unit tests
target/release/primerivat 1e15      # must print 29 844 570 422 669
```

Reference values checked:

- π(1e13) = 346 065 536 839
- π(1e15) = 29 844 570 422 669
- π(1e16) = 279 238 341 033 925
- π(1e17) = 2 623 557 157 654 233

## Implementation notes

1. **Sweep direction** — ascending, from `lo_start` up to `z`. `phi_vec[bi]`
   holds `φ(lo − 1, b − 1)` and is updated at the end of each window.
2. **Bucket sieve** — in the bulk cross-off for primes `≥ x¹ᐟ⁴`, primes with
   `p² > hi` are skipped: any composite in `[lo, hi)` has a factor `≤ √hi`.
3. **Seed correction** — for `lo < y`, seed primes in `[lo, hi)` are crossed
   off as multiples of themselves and re-added via `seed_in_seg` /
   `seed_in_query`.
4. **`ext_easy` clamp** — `φ(n, b−1)` is clamped to `≥ 1` when `n < p_{b−1}`.
5. **Small-x guard** — `if a ≤ C { return baseline::prime_pi(x) }`.

## Reference

Deléglise & Rivat, *Computing π(x): the Meissel, Lehmer, Lagarias, Miller,
Odlyzko method*, Math. Comp. 65 (1996).

A self-contained mathematical description of the algorithm is available in:

- [ALGORITHM.md](ALGORITHM.md) — English
- [ALGORITHME.md](ALGORITHME.md) — français

## License

BSD 2-Clause License — see [LICENSE](LICENSE).
