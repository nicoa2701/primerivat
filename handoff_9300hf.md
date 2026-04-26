# Reprise sur i5-9300HF — note de passation

Pendant à `handoff_13450hx.md`. Note pour la **prochaine session dédiée**
sur le **i5-9300HF** (4c/8t, 32 KB L1d, 256 KB L2, 8 MB L3, AVX2 only —
pas d'AVX-512). Écrite le 2026-04-26, mise à jour en fin de session
**Phase 1 + 2A** Kim-style cross-off déroulé.

---

## État au 2026-04-26 (8 commits cumulés, dernière révision `e2528cf`)

```
e2528cf segment: Kim-style 8-way unrolled cross_off_count for bi_main (Phase 2A)
66ce0b1 segment: Kim-style 8-way unrolled cross-off for rest_plain_xoff (Phase 1)
d7ecaa4 docs: refresh 9300HF handoff for post-7bb4099 state
47a8e57 segment: drop orphan count_monotonic
7bb4099 segment: use 240-block masks for wheel counts
25f6f9e docs: add per-CPU handoff notes for follow-up sessions
9772571 cpu_detect: wrap __cpuid calls in unsafe for older Rust toolchains
8349c91 S2_hard: replace 8-elt partition_point with 30-byte lookup in popcount hot path
```

Trois groupes de gains empilés cette session (universels α=1/α=2) :
- **Popcount LUTs** (`8349c91` + `7bb4099` + `47a8e57`) — `count_primes_upto_int{,_m}`
  réduit à 2 div/mod + 1 LUT load via la table `W30_MASK_LEQ_240[u64]` (1920 B en L1d).
  Cumulé ~−18 % wall à 1e15 / ~−14 % à 1e17 vs pré-cascade.
- **`66ce0b1` Phase 1** — `cross_off_pd_unrolled` Kim-style pour `rest_plain_xoff`.
  8 fonctions spécialisées par groupe `g = W30_IDX[p%30]`, dispatchées via
  `match pd.bit_seq[0]`. Bit positions baked en immédiats → `andb m8, imm8`
  chaîné. Pré-roll j0..8 + main loop déroulée 8-step (avance `p` bytes/cycle)
  + tail.
- **`e2528cf` Phase 2A** — `cross_off_count_pd_unrolled` même structure pour
  `bi_main_xoff`, plus accumulateur `cleared`. **Détail critique** : le macro
  per-step doit faire `*bits[g] &= !mask` après lecture explicite (pas
  `*bits[g] = byte & !mask`) pour conserver le RMW natif et éviter la
  régression +46 % observée avec le pattern store-via-register.

**Baseline 9300HF cool post-`e2528cf` (Win, défaut `-b 16`, médiane 2 cool trials) :**

| x | wall | régime | vs post-`7bb4099` | vs `e2528cf` baseline mesuré direct |
|---|---:|---|---:|---:|
| 1e15 | **1.847 s** | α=1 | −15 % | (baseline mesurée 2.193 s : −15.8 %) |
| 1e17 | **46.486 s** | α=2 | −7 % | (baseline mesurée 53.672 s : −13.4 %) |

**Speedups par poste cool-machine** (vs baseline `d7ecaa4` mesurée direct dans la même session) :

| poste | regime | baseline | HEAD (P1+2A) | gain |
|---|---|---:|---:|---:|
| `bi_main_xoff` CPU | 1e15 α=1 | 5959 ms | 2566 ms | **2.32×** |
| `rest_plain_xoff` CPU | 1e15 α=1 | 2778 ms | 1343 ms | **2.07×** |
| `bi_main_xoff` CPU | 1e17 α=2 | 89 396 ms | 40 605 ms | **2.20×** |
| `rest_plain_xoff` CPU | 1e17 α=2 | 37 224 ms | 31 875 ms | 1.17× (primes plus larges, moins de cycles déroulés) |

**Comparaison vs primecount Kim WSL** (handoff de référence, à re-mesurer post-2A) :

| x | nous Win cool | primecount WSL | facteur estimé |
|---|---:|---:|---:|
| 1e15 | 1.847 s | 0.407 s | **~4.5×** |
| 1e17 | 46.486 s | 7.68 s | **~6.05×** |

> Mesure WSL post-`e2528cf` à faire pour figer le facteur (Win→WSL ~6-7 %).
> Avant cette session : 5.5× / 6.6× ; après Phase 1+2A : ~4.5× / ~6.05×.
> Phase 3 (rest_bulk) reste à faire pour viser 3-4× Kim.

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

## Profile détaillé à 1e17 α=2 (post-`e2528cf`, Win cool, single trial)

CPU 263 s / wall 46.5 s = 5.66× sur 8 threads = **71 % efficient**.

> Le drop d'efficacité Rayon (78 → 71 %) post-2A est attendu : on a éliminé
> ~64 s de CPU sur `bi_main_xoff`, mais le wall ne descend que ~7 s parce
> que les autres bands sont saturées par `rest_bulk_xoff` (toujours rolled)
> et `tail_ext_emit`. Phase 3 (rest_bulk déroulé) devrait reéquilibrer.

| poste | CPU | s | type | levier restant |
|---|---:|---:|---|---|
| **tail_ext_emit** | **37.3 %** | **98** | popcount-based leaf emit | déjà optimal (LUT 240) |
| **rest_bulk_xoff** | 24.3 % | 64 | state-resume cross-off (mid-large primes) | **Phase 3** : cross-off déroulé switch-64 |
| **bi_main_xoff** | 15.4 % | 41 | counted cross-off (Kim-déroulé Phase 2A) | minor (déroulé fait) |
| **rest_plain_xoff** | 12.1 % | 32 | plain cross-off (Kim-déroulé Phase 1) | minor (déroulé fait) |
| tail_advance | 6.2 % | 16 | total_count + bsearches | minor |
| bi_main_leaf | 4.1 % | 11 | popcount + leaf-fold | déjà optimisé |
| reste | < 1 % | — | fill, p2, sweep | non |

Cross-off total = **51.8 % CPU** (vs 62 % pré-Phase-1+2A). Le levier #1 restant
est `rest_bulk_xoff` qui passe de 21 % → 24 % en relatif (mais l'absolu est
similaire 67→64 s, juste la base CPU a baissé).

---

## Phase 1 + 2A : DONE (cette session)

**Phase 1 (`66ce0b1`) — `cross_off_pd_unrolled`** pour `rest_plain_xoff`.
8 fonctions spécialisées par groupe via macro `impl_xoff_unrolled!`.
Asm vérifiée : main loop = 8× `andb m8, imm8` chaînés sur 8 adresses
pré-calculées + 3 adds + cmp + jb (= identique au pattern Kim Sieve.cpp).

**Phase 2A (`e2528cf`) — `cross_off_count_pd_unrolled`** pour `bi_main_xoff`.
Même structure + `cleared` accumulator. Macro `impl_xoff_count_unrolled!`
(piège évité : utiliser `*bits[g] &= !mask` après lecture explicite, pas
`*bits[g] = byte & !mask` qui régressait +46 %).

**Counter array Kim DELIBÉRÉMENT non adopté** : notre `prefix_counts` est
déjà 0.1 % CPU (build seul, lookups O(1)) et `tail_ext_emit` fait des
queries non-monotones (pl_idx décroissant per ei → n peut décroître). Le
counter array Kim suppose un curseur monotone. Bénéfice marginal vs coût
sur cross_off_count.

Tests : 47 lib + 6 main passent ; 2 nouveaux tests bit-exact + cleared-exact
(`cross_off_pd_unrolled_matches_cross_off_pd`,
`cross_off_count_pd_unrolled_matches_cross_off_count_pd`) couvrent les
8 groupes × 36 primes × 7 lo values. π et tous les counters S2_hard
inchangés (bi_leaf_hits, ext_emitted, ext_clamped, prefix_fills,
bulk_active_sum).

---

## Phase 3 (à faire prochaine session, ≈ 2-3 h) : `cross_off_pd_from_state` (rest_bulk_xoff, 24.3 % CPU)

Variante avec **state persistant** (`next_m`, `next_j`) entre segments.
Pattern 8-groupes identique à Phase 1+2A, mais la fonction doit :

1. **Catch-up** : avancer `m` de `next_m` jusqu'à `lo` (prime peut être
   resté inactif sur plusieurs segments). Code actuel :
   `while m < lo { m += gap_m[j]; j = (j+1) & 7; }`. Le pré-roll Kim-style
   doit gérer ça avant d'attaquer le main loop.
2. **Sortie de boucle qui retourne `(m, j)`** au caller (pas juste `return`)
   pour que `bulk_next_m[k]` / `bulk_next_j[k]` soient mis à jour pour le
   segment suivant.
3. Dispatch sur `g` mais préserver `j` à l'entrée et à la sortie : la
   signature est `fn(self, lo, p, pd, next_m, next_j) -> (next_m, next_j)`
   (cf [src/segment.rs:754](src/segment.rs#L754)).

**Gain attendu Phase 3** : 1.3-1.5× sur rest_bulk_xoff = **~4-5 % wall** au
total. Modeste car les primes bulk font typiquement 1-3 cross-offs/seg
(pas de boucle déroulée qui amortit), donc l'optim profite surtout du
byte-write + dispatch direct.

Call site : [src/dr/hard.rs:766](src/dr/hard.rs#L766) — boucle `for k in 0..target_end`.

### Cumul Phase 1+2A+3 attendu : **20-25 % wall** → 1e17 wall ≈ 35-37 s sur 9300HF (vs 7.68 s Kim, soit ~4.6×).

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
- `cargo test --release` (les tests `*_unrolled_matches_*` cassent vite si bit_seq mal dérivé)
- `target/release/primerivat 1e13 1e15` (sanity π)
- `--dr-profile 1e15` (α=1) et `--dr-profile 1e17` (α=2) sur Win + WSL
- ext_emitted, bi_leaf_hits, ext_clamped, prefix_fills, bulk_active_sum **doivent rester identiques** (sinon bug de comptage)
- bench croisé sur 13450HX (α=1) avant push
  - notre 13450HX baseline post-cascade : 140 s à 1e18 α=1
  - régression > 2 % à α=1 ⇒ debug ou conditionnement par α

**Piège codegen (à reproduire pour Phase 3)** : pour les variantes count,
le macro per-step doit faire `*bits.get_unchecked_mut(g) &= !mask` après
une lecture explicite séparée. **Ne pas** écrire `*bits.get_unchecked_mut(g)
= byte & !mask` (store via reg) : rustc/LLVM perd le `andb m8, imm8` et
on régresse +46 % sur le poste. Vérifiable avec `cargo rustc --release
--lib -- --emit=asm` puis `grep "andb.*\$-2.*\(.*\)"` (chaîne `andb` de
8 dans la main loop).

**Cadence bench 9300HF** : la machine throttle après ~2 runs consécutifs
de 50 s. Drift typique +30-65 % wall sur trial 2-3. Bencher **1 run, puis
pause** avec attention au refroidissement avant chaque trial. Comparer
trial 1 vs trial 1 (cool) ; les CPU shares restent stables même throttlés
et donnent un signal d'appoint.

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
