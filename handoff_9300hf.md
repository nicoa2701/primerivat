# Reprise sur i5-9300HF — note de passation

Pendant à `handoff_13450hx.md`. Note pour la **prochaine session dédiée**
sur le **i5-9300HF** (4c/8t, 32 KB L1d, 256 KB L2, 8 MB L3, AVX2 only —
pas d'AVX-512). Écrite le 2026-04-26 à la fin d'une session d'analyse
+ optims légères + comparaison à primecount.

---

## État au 2026-04-26 (3 commits non encore poussés)

```
9772571 cpu_detect: wrap __cpuid calls in unsafe for older Rust toolchains
8349c91 S2_hard: replace 8-elt partition_point with 30-byte lookup in popcount hot path
85426ab profile: split bi_main timer into leaf-emit vs xoff+bookkeeping
```

`8349c91` est le seul gain de perf : **−12 % wall à 1e15 / −8 % à 1e17**
sur 9300HF, universel α=1/α=2. Lookup 30-byte vs partition_point 8-elt
dans `count_primes_upto_int{,_m}` (1.7 G calls à 1e17).

**Baseline post-A2 actuel (en attendant le gros refacto) :**

| x | wall (Win) | wall (WSL Linux) | régime |
|---|---:|---:|---|
| 1e15 | 2.42 s | 2.24 s | α=1 |
| 1e17 | 54.2 s | 51.0 s | α=2 |
| primecount 1e17 (WSL, même CPU) | — | 7.68 s | DR (`-d`) |

**Écart implémentation pur vs Kim : 6.6× à 1e17** (même OS, même
machine — l'OS ne vaut que ~6 % wall). Efficacité Rayon identique 78 %.

---

## Pistes testées et fermées dans la session

| piste | résultat | conclusion |
|---|---|---|
| **A1** réduire SEG | +26 % wall à 1e17 | sieve déjà en L1, plus de segs = plus d'overhead par seg (fill, prefix_build, bulk setup) |
| **bucket sieve** outer-loop skip | +1-2 % wall (sous bruit) | `cross_off_pd_from_state` est `#[inline]`, no-fire visit ~5 ns |
| **pre-sieve C=6** (template 1001 B) | neutre dans le bruit | gain trop petit pour 1 prime |
| **pre-sieve C=7** (template 17017 B) | +5.5 % wall | template 17 KB + sieve 17 KB > 32 KB L1d → cache miss |
| **POC cross-off déroulé groupe 0** byte-view | −2.5 % à 1e15 / neutre 1e17 | concept validé mais 1/8 des primes seulement, non shippable |

**Insight clé** : tout template pre-sieve doit garder
`template_bytes + sieve_bytes (17 480) ≤ 32 KB L1d`. C'est exactement
pourquoi Kim chaîne 7 petits templates AND'd (≤ 4757 B chacun) plutôt
qu'un seul gros.

---

## Source primecount de Kim disponible localement

**Chemin Windows** (ce que Claude doit lire) :
```
c:/Users/Kbda9/projet/3rivat3/primecount/src/
```

Fichiers clés à relire :
- `Sieve.cpp` lignes 222–596 — le **switch sur 64 cases** pour `cross_off`
  et `cross_off_count`. C'est la référence directe pour Phase 1+2 du
  refacto cross-off déroulé.
- `Sieve.hpp` — interface, struct PrimeState (uint32_t multiple + uint8_t wheel_index), Counter.
- `Sieve_pre_sieve.hpp` — chaîne de 7 templates AND'd (primes 13, puis paires 17-19, 23-29, …, 67-71). Pour le futur multi-template AND.
- `Sieve_count_simd.hpp` — POPCNT64 / AVX-512 / ARM SVE pour le tail counting (sur 9300HF c'est juste POPCNT64 puisque pas d'AVX-512).
- `BitSieve240.hpp` — tables `unset_smaller`, `unset_larger`, `set_bit`, `unset_bit` indexées par n%240.
- `Sieve_arrays.hpp` — `pre_sieved_*` constantes, `wheel_init`, `wheel_offsets`, `wheel_corr`, `mod30_prime_residues`. **Toutes les tables nécessaires y sont définies, à reprendre tel quel.**

**Chemin du binaire** (utilisé pour les bench croisés en WSL) :
```
~/primecount/primecount     # ELF Linux dans le rootfs WSL
```
Compilé depuis le même tree, build dir = `c:/Users/Kbda9/projet/3rivat3/primecount/build/`.

---

## Profile détaillé à 1e17 α=2 (post-A2, WSL)

CPU 319 s / wall 51 s = 6.26× sur 8 threads = **78 % efficient**.

| poste | CPU | s | type | levier |
|---|---:|---:|---|---|
| **bi_main_xoff** | **27.6 %** | **88** | counted cross-off (small primes 13–960) | cross-off déroulé switch-64 |
| **tail_ext_emit** | 29.4 % | 94 | popcount-based leaf emit | counter array always-current |
| rest_bulk_xoff | 21.0 % | 67 | state-resume cross-off (mid-large primes) | cross-off déroulé switch-64 |
| rest_plain_xoff | 12.9 % | 41 | plain cross-off (mid primes) | cross-off déroulé switch-64 |
| tail_advance | 5.2 % | 17 | total_count + bsearches | minor |
| bi_main_leaf | 3.4 % | 11 | popcount + leaf-fold | déjà optimisé par A2 |
| reste | < 1 % | — | fill, p2, sweep | non |

Cross-off total = **62 % CPU**. C'est le levier #1.

---

## Plan séance dédiée prochaine (≈ 5–8 h focus)

**Objectif** : se rapprocher de primecount à **3-4× Kim** (≈ 25-30 s à
1e17 sur 9300HF) via le cross-off déroulé Kim-style.

### Phase 1 : `cross_off_pd` (rest_plain_xoff, 11.6 % CPU)

Le plus simple : pas de count, pas de state. 8 specialized functions
(une par residue group p%30 ∈ {1,7,11,…,29}). Inspiré du switch sur 64
cases dans `primecount/src/Sieve.cpp::cross_off`.

Implémentation Rust :
1. Voir le bitset comme `&mut [u8]` via `from_raw_parts_mut(bits as *mut u8, W30_WORDS * 8)`
2. Dispatcher sur `p % 30` (lookup table de 30 entrées → group_index 0..8)
3. Chaque groupe a son inner loop déroulé par 8 avec **bit positions immédiates** (`& !(1u8 << 0)`, ..., `& !(1u8 << 7)`)
4. `delta_group[]` chargé en 8 stack locals (registres)
5. Pre-roll (≤ 7 partial steps) pour aligner à j=0, puis unrolled cycles, puis tail

**Gain attendu Phase 1** : 1.5-2× sur rest_plain_xoff = ~3-4 % wall.

POC du groupe 0 (réverté) a sauvé 2.5 % à 1e15. Extrapolation 8× pour
tous les groupes → 8-15 % théorique sur le poste, mais en pratique
4-7 % à cause des primes mid-sized avec peu d'iters.

### Phase 2 : `cross_off_count_pd` (bi_main_xoff, **27.6 % CPU — biggest**)

Plus complexe : il faut maintenir un compteur. Deux options :

**Option A — counter array Kim-style** (préféré). Au lieu de notre
`prefix_counts` reconstruit (`fill_prefix_counts` à chaque tail emit),
on maintient un `counter[i]` qui contient le nombre de bits set dans
l'intervalle `[i * counter_dist, (i+1) * counter_dist)`. À chaque
cross-off avec count : `counter[m >> log2_dist] -= is_bit`.

Le `count(stop)` lit `counter[stop >> log2_dist]` + popcount du résidu.
Pas de rebuild. Voir `primecount/src/Sieve.cpp::cross_off_count` et
`Sieve_count_stop.hpp`.

Refonte de `count_primes_upto_int_m` aussi (il opère via `MonoCount` qui
n'aurait plus de raison d'être — remplaçable par counter array lookup).

**Option B — garder `prefix_counts`**, ajouter juste le déroulage par 8
(pas le compteur incrémental). Plus simple mais on n'élimine pas les 33 %
CPU de `tail_ext_emit`. Demi-gain.

**Gain attendu Phase 2 (Option A)** :
- bi_main_xoff cross-off déroulé : 1.5-2× → 6-7 % wall
- tail_ext_emit avec counter array (plus de prefix rebuild) : 1.3-1.5× → 3-4 % wall
- **Cumulé : ~10 % wall**

### Phase 3 : `cross_off_pd_from_state` (rest_bulk_xoff, 20.4 % CPU)

Variante avec state persistant (next_m, next_j). Adapter le pattern
8-groupes en gérant la résumption. Plus subtil parce que le state est
mémorisé entre segments et il faut reprendre à la bonne position.

**Gain attendu Phase 3** : 1.3-1.5× → 4-5 % wall.

### Cumul total attendu : **15-20 % wall** → 1e17 wall ≈ 41-43 s sur 9300HF (vs 7.68 s Kim).

---

## Détails techniques importants pour la prochaine session

**Layout actuel** :
- `WheelSieve30.bits: [u64; W30_WORDS]` où `W30_WORDS = 2185`, total 17 480 octets
- `W30_GROUPS = 17 476` valid bytes ; les 4 derniers sont padding (à laisser 0)
- Chaque byte = 8 bits = 8 wheel positions {1, 7, 11, 13, 17, 19, 23, 29}
- `bit_idx = group * 8 + j`, `word = bit_idx >> 6`, `bit = bit_idx & 63`

**Layout Kim** :
- `Vec<uint8_t>` direct, accédé par `sieve[m]` où `m` = group index
- Bit positions 0..7 dans le byte = constantes immédiates
- Pas de division par 64, pas de mask 63 → écriture en 1 instruction `andb`

**Conversion sûre** : `unsafe { core::slice::from_raw_parts_mut(self.bits.as_mut_ptr() as *mut u8, W30_WORDS * 8) }`.
Déjà utilisée dans `fill_presieved_7_11`. Vérifier alignement (u64 → u8
toujours OK).

**bit_seq par residue group** (à dériver, pas hardcoder pour éviter erreur) :
- group 0 (p%30=1) : `[0, 1, 2, 3, 4, 5, 6, 7]` (identité)
- group 1 (p%30=7) : `[1, 5, 4, 0, 7, 3, 2, 6]`
- group 2 (p%30=11) : `[2, 4, 0, 6, 1, 7, 3, 5]`
- group 3 (p%30=13) : `[3, 0, 6, 5, 2, 1, 7, 4]`
- group 4 (p%30=17) : `[4, 7, 1, 2, 5, 6, 0, 3]`
- group 5 (p%30=19) : `[5, 3, 7, 1, 6, 0, 4, 2]`
- group 6 (p%30=23) : `[6, 2, 3, 7, 0, 4, 5, 1]`
- group 7 (p%30=29) : `[7, 6, 5, 4, 3, 2, 1, 0]`

Vérifiables via `pd.bit_seq` au runtime pour chaque p donné. Pour la
session POC j'ai vérifié group 0 et group 7.

**Garde-fou non-régression** : tester à chaque phase :
- `cargo test --release`
- `target/release/primerivat 1e13 1e15` (sanity π)
- `--dr-profile 1e15` (α=1) et `--dr-profile 1e17` (α=2) sur Win + WSL
- ext_emitted, bi_leaf_hits, ext_clamped, prefix_fills, bulk_active_sum **doivent rester identiques** (sinon bug de comptage)
- bench croisé sur 13450HX (α=1) avant push
  - notre 13450HX baseline post-cascade : 140 s à 1e18 α=1
  - régression > 2 % à α=1 ⇒ debug ou conditionnement par α

---

## Outils

```bash
cargo build --release
cargo test --release
target/release/primerivat --dr-profile 1e15           # α=1, fast
target/release/primerivat --dr-profile 1e17           # α=2, ~50 s
target/release/primerivat --dr-profile 1e17 --alpha 1 # forcer α=1 (proxy 13450HX)
target/release/primerivat -b 16 ...                   # band mult override

# Comparer à primecount (même machine, WSL)
~/primecount/primecount -d 1e17 --time
```

---

## Mémoire principale

`~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_s2_hard_refactor.md`

Mise à jour le 2026-04-26 avec : POC group 0 réverté, leçon cache L1
template, source primecount sur disque local.
