# primerivat

*[Version française](LISEZMOI.md)*

`primerivat` is a Rust implementation of the prime-counting function `π(x)`
based on the **Deléglise–Rivat** algorithm.

The current engine (`prime_pi_dr_meissel_v4`) is active by default and
computes π(x) exactly up to at least `x = 1e18` on commodity hardware
(verified cross-CPU through commit `fe5a03f`).

## Quick start

```bash
cargo build --release
cargo run --release -- 1e13        # π(1e13) = 346 065 536 839
cargo run --release -- 1e15        # π(1e15) = 29 844 570 422 669
```

The binary prints a startup tag with the short git hash, detected L3, and
the selected α (e.g. `[primerivat fe5a03f | L3=8Mo α=2]`), making it easy
to compare multiple builds side by side.

## Measured performance

Release build, multi-threaded, adaptive α. Three reference machines:

- **i5-9300H** — 4C/8T, 8 MB L3, DDR4-2666 (Coffee Lake, 2019)
- **i5-13450HX** — 6P+4E (10 phys / 16T), 20 MB L3, DDR5 (Raptor Lake, 2023)
- **Ryzen 7 9700X** — 8C/16T, 32 MB L3, DDR5 (Zen 5, 2024)

| x | i5-9300H | i5-13450HX | Ryzen 9700X | π(x) |
|---|---:|---:|---:|---|
| `1e11` | 9 ms | 9 ms | 4 ms | 4 118 054 813 |
| `1e12` | 30 ms | 31 ms | 10 ms | 37 607 912 018 |
| `1e13` | 117 ms | 56 ms | 42 ms | 346 065 536 839 |
| `1e14` | 466 ms | 220 ms | 163 ms | 3 204 941 750 802 |
| `1e15` | 1.85 s | 0.81 s | 0.62 s | 29 844 570 422 669 |
| `1e16` | 8.74 s | 3.08 s | 2.29 s | 279 238 341 033 925 |
| `1e17` | 42.1 s α=2 | 14.4 s α=1 | 10.6 s α=1 | 2 623 557 157 654 233 |
| `3e17` | 135 s α=2 | 39.9 s α=1 | 20.3 s α=2 | 7 650 011 911 220 803 |
| `1e18` | 313 s α=2 | 97.0 s α=1 | 49.2 s α=2 | 24 739 954 287 740 860 |

All columns measured at commit `8a6d89b` (post-2-tier α rule), batch
single-trial sweep `1e11 1e12 … 1e18`. The 9300H thermal-throttles after
~10 s sustained load, so 1e17 here (42.1 s, α=2) is mid-throttle; the
post-Étape-A cool single-run snapshot at the same magnitude is 38.9 s.
1e18 on 9300H (313 s α=2) is run separately as it takes ~5+ min — that
result is both 36 % below the pre-Phase-1+2A snapshot of 488 s and 75 %
below the pre-cascade `9e9162a` baseline of 1266 s.

The 9700X column shows the new α=2 regime kicking in from 3e17 onward
(adaptive rule, see below): 20.3 s at 3e17 (vs ~26 s at α=1) and 49.2 s
at 1e18 (vs ~72 s at α=1).

Cumulative speedup vs. the pre-cascade baseline at commit `9e9162a`
across all three CPUs (the cascade gains are universal — Phase 1+2A,
popcount LUT-240, single-pass merge, fold accumulators, band-split 16×,
2-pass deferred-tail-ext for α=2 regime):

| x | i5-9300H | i5-13450HX | Ryzen 9700X |
|---|---:|---:|---:|
| 1e15 | 5.51 → 1.85 s (**−66 %**) | — → 0.81 s | — → 0.62 s |
| 1e17 α=2 | 160 → 42 s (**−74 %**) | — | — |
| 1e17 α=1 | — | 44.8 → 14.4 s (**−68 %**) | — → 10.6 s |
| 1e18 α=2 | 1266 → 313 s (**−75 %**) | — | — → 49.2 s |
| 1e18 α=1 | — | 301 → 97 s (**−68 %**) | — |

The cumulative gains come from the multi-commit S2_hard refactor cascade
(single-pass deferred-leaf design, fold accumulators, log-scale band
layout for α=2, clamp-leaf bulk pre-count, `{7, 11}` pre-sieve tile,
popcount via 240-bit LUT, Kim-style 8-way unrolled cross-off in
`bi_main_xoff` and `rest_plain_xoff`, 2-pass deferred-tail-ext, and the
2-tier adaptive α rule).

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
| `0..n_hard` | ≤ √y | hard leaves | `phi_vec` + monotonic popcount cursor |
| `n_hard..b_ext` | (√y, x¹ᐟ⁴] | `phi_easy` (`m = p_l`) | `phi_vec` + monotonic popcount cursor |
| `b_ext..n_all` | (x¹ᐟ⁴, y] | `ext_easy` (`m = p_l`) | closed form `π(n) − (b−2)` (clamped ≥ 1) |

The `ext_easy` closed form `φ(n, b−1) = π(n) − (b−2)` is valid when
`n ≥ p_{b−1}` and avoids maintaining `phi_vec` for the bulk of primes.

### Adaptive α

`y = α·∛x` with α chosen per magnitude of `x` **and hardware**, in two
hardware tiers:

- **9300H tier** (L3 < 16 MB **and** ≤ 8 physical cores, e.g. cache-
  constrained laptops): α = 2.0 from `x ≥ 3e16` (~41 % faster at `1e17`).
- **9700X tier** (≥ 8 physical cores **and** pure SMT, i.e.
  `logical == 2 × physical`, e.g. Ryzen 7 9700X 8C/16T or any Threadripper
  with HT/SMT enabled): α = 2.0 from `x ≥ 3e17` (~24 % faster at `1e18`,
  ~28 % faster at `5e17`).
- All other CPUs (e.g. Intel hybrid P+E parts where `logical < 2 ×
  physical`, or any `x` below the relevant threshold): α = 1.0.

The two thresholds reflect that the SMT-symmetric desktop tier (16 logical
threads, large L3) absorbs α=1's larger sieve windows up to ~1e17, after
which α=2's algorithmic CPU savings (−42 % at 1e18) dominate the slightly
worse Rayon balance.

The auto-selection can be overridden from the CLI with `-a <α>` or
`--alpha <α>`. Accepted range:

- `x ≤ 1e15` → any `α ∈ [1, 2]` (e.g. `1.5`)
- `x > 1e15` → only `α ∈ {1, 2}`

Any value outside these bounds is logged as a warning and ignored — the
engine falls back to the auto-selected α.

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
├── bit.rs          # Fenwick tree (legacy S2 variants only; prod uses prefix popcount)
├── segment.rs      # WheelSieve30 (wheel-mod-30 segmented sieve) + MonoCount cursor
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

# Batch: multiple x in a single invocation (prints a summary table)
cargo run --release -- 1e11 1e12 1e13 1e14

# Override the auto-selected α (short or long form)
cargo run --release -- 1e17 -a 2
cargo run --release -- 1e13 --alpha 1

# Tune Rayon banding (advanced; -b 16 is the empirical plateau on 9300H)
cargo run --release -- 1e17 -a 2 -b 16

# Lift the b_ext bulk frontier (Piste D debug knob; K=1.0 is optimal,
# K<1.0 is rejected because the leaf-B condition would break)
cargo run --release -- 1e17 -a 2 -B 1.5

# Profiling
cargo run --release -- --dr-profile 1e13  # DR v4 step timings + per-band breakdown
cargo run --release -- --lucy-profile 1e13

# Decade sweep up to x_max (default 1e12)
cargo run --release -- --sweep 1e14
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
- π(1e18) = 24 739 954 287 740 860

## Implementation notes

1. **Sweep direction** — ascending, from `lo_start` up to `z`. `phi_vec[bi]`
   holds `φ(lo − 1, b − 1)` and is updated at the end of each window.
2. **Monotonic leaf scan** — within a given `bi`, leaves arrive in ascending
   `n`. A `MonoCount` cursor caches the running popcount frontier so each
   leaf query only popcounts the u64 words newly traversed since the previous
   leaf. Replaces the `fill_prefix_counts` full-sieve popcount sweep
   (~2 185 words per call) that used to run once per bi with leaves.
3. **Pre-sieve `{7, 11}` template** — `fill_presieved_7_11` tiles a
   precomputed 77-byte bitmap (covering `lcm(7, 11) · 30 = 2 310` integers)
   into the sieve at each segment start, replacing the ones-fill + two
   wheel-30 cross-off loops. Sequential byte-copy vectorises well, avoiding
   the scattered bit writes of the generic path.
4. **Kim-style 8-way unrolled cross-off** — `cross_off_pd_unrolled` (used
   by `rest_plain_xoff`) and `cross_off_count_pd_unrolled` (used by
   `bi_main_xoff`) dispatch on the prime's residue group `g = p % 30 ∈ {1,
   7, 11, 13, 17, 19, 23, 29}` and execute 8 specialised inner loops with
   bit positions baked as immediates (`andb m8, imm8`). 2.0× to 3.4×
   speedup over the rolled variant on those phases.
5. **Popcount via 240-bit LUT** — `count_primes_upto_int{,_m}` rewires the
   inner popcount step to use a 1920-byte `W30_MASK_LEQ_240[u64]` table
   (240 = 8 wheel residues × 30, exactly one full u64 word), reducing each
   call to two div/mod + one LUT load. Universal across α regimes.
6. **Bucket sieve** — in the bulk cross-off for primes `≥ x¹ᐟ⁴`, primes with
   `p² > hi` are skipped: any composite in `[lo, hi)` has a factor `≤ √hi`.
7. **Seed correction** — for `lo < y`, seed primes in `[lo, hi)` are crossed
   off as multiples of themselves and re-added via `seed_in_seg` /
   `seed_in_query`.
8. **`ext_easy` clamp** — `φ(n, b−1)` is clamped to `≥ 1` when `n < p_{b−1}`.
9. **Small-x guard** — `if a ≤ C { return baseline::prime_pi(x) }`.

## Reference

Deléglise & Rivat, *Computing π(x): the Meissel, Lehmer, Lagarias, Miller,
Odlyzko method*, Math. Comp. 65 (1996).

A self-contained mathematical description of the algorithm is available in:

- [ALGORITHM.md](ALGORITHM.md) — English
- [ALGORITHME.md](ALGORITHME.md) — français

## License

BSD 2-Clause License — see [LICENSE](LICENSE).
