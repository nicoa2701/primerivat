# Reprise sur i5-9300HF — note de passation

Pendant à `handoff_13450hx.md`. Note pour la **prochaine session dédiée**
sur le **i5-9300HF** (4c/8t, 32 KB L1d, 256 KB L2, 8 MB L3, AVX2 only —
pas d'AVX-512). Écrite le 2026-04-26, mise à jour en fin de session
**Phase 1 + 2A** Kim-style cross-off déroulé wirées + **Phase 3** explorée
puis revertée pour cause de régression sur le bulk.

---

## État au 2026-04-26 (9 commits cumulés, dernière révision `f43d1b5`)

```
f43d1b5 segment: Phase 3 explored — cross_off_pd_from_state_unrolled (not wired)
a8aa681 docs: refresh 9300HF handoff for post-Phase-1+2A state
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
> que les autres bands sont saturées par `rest_bulk_xoff` et `tail_ext_emit`.
> Tentative Phase 3 (rest_bulk déroulé) revertée — voir section dédiée.

| poste | CPU | s | type | levier restant |
|---|---:|---:|---|---|
| **tail_ext_emit** | **37.3 %** | **98** | popcount-based leaf emit | déjà optimal (LUT 240) |
| **rest_bulk_xoff** | 24.3 % | 64 | state-resume cross-off (mid-large primes) | rolled, déroulé regressait −25 % (cf section Phase 3) |
| **bi_main_xoff** | 15.4 % | 41 | counted cross-off (Kim-déroulé Phase 2A) | minor (déroulé fait) |
| **rest_plain_xoff** | 12.1 % | 32 | plain cross-off (Kim-déroulé Phase 1) | minor (déroulé fait) |
| tail_advance | 6.2 % | 16 | total_count + bsearches | minor |
| bi_main_leaf | 4.1 % | 11 | popcount + leaf-fold | déjà optimisé |
| reste | < 1 % | — | fill, p2, sweep | non |

Cross-off total = **51.8 % CPU** (vs 62 % pré-Phase-1+2A). Le poste de tête
est désormais `tail_ext_emit` (37 %) — déjà optimal. Le 2e levier serait
`rest_bulk_xoff`, mais la tentative Phase 3 telle quelle régresse.

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

## Phase 3 : explorée + revertée (`f43d1b5`, 2026-04-26)

**Implémentation complète et bit-exact** : `cross_off_pd_from_state_unrolled`
+ macro `impl_xoff_state_unrolled!` + 8 helpers `xoff_state_unrolled_g{0..7}`
dans [src/segment.rs](src/segment.rs). Test multi-segment chained
(`cross_off_pd_from_state_unrolled_matches_cross_off_pd_from_state`) qui
vérifie `(bits, m, j)` à chaque pas sur 36 primes × 5 segments — couvre
catch-up, pré-roll exit, tail exit, large primes.

**Wirage reverté** ([src/dr/hard.rs:769](src/dr/hard.rs#L769)) à cause
d'une **régression de ~25 % sur `rest_bulk_xoff`** (37.7 % share vs 30.1 %
post-Phase-2A à 1e15 α=1 ; mesure thermal-indépendante car CPU shares
sont normalisées au CPU total). Conséquence : wall total ~+8 % vs
Phase 1+2A.

**Cause racine** : pour les primes bulk (b_ext..bulk_end, p > x^{1/4}),
chaque appel ne fait que **0–3 cross-offs en moyenne**. Avec
`bulk_active_sum = 112 M` paires (k, segment) à 1e15, l'overhead par
appel (dispatch 8-way `match` + `from_raw_parts_mut` byte-view +
formule de recovery `(m, j) = lo + g*30 + w30res_p[j]`) ajoute ~50 ns/call
soit ~5 s de coût pur, supérieur au gain par-itération du déroulage.

**Code conservé** dans `segment.rs` comme matériel de référence pour une
future tentative à seuil conditionnel (cf section follow-ups).

### Lesson pour Phase 3 v2

Le pattern Kim-style est **pénalisant pour les call-sites où le nombre
moyen d'itérations par appel est faible**. Pour les primes bulk c'est
~1-3 ; pour les Phases 1 et 2A c'est plusieurs centaines (segment
balayé via la main loop déroulée). Toute future Phase 3 doit donc soit :
- **Splitter par seuil** : `p < W30_SEG * 8 / 30 ≈ 140 000` → unrolled
  (au moins ~1 cycle complet par segment), `p ≥ 140 000` → rolled.
  Implémenter via un `partition_point` sur les primes bulk + 2 boucles.
- **Réduire l'overhead par appel** : éviter `from_raw_parts_mut` à chaque
  appel (le faire une fois par segment au-dessus de la boucle), passer
  juste `bits_bytes` aux 8 helpers ; éliminer la recovery `(m, j)` en
  retournant `(g, j)` au caller qui maintiendrait directement `(g, j)`
  comme state au lieu de `(m, j)`.

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

## Follow-ups ouverts pour la prochaine session (par ordre de priorité)

1. **Multi-template AND pre-sieve** (style Kim `Sieve_pre_sieve.hpp`) —
   chaîne de 7 petits templates AND'd dans le sieve, chacun ≤ 4757 B
   pour rester sous L1d (32 KB) avec le sieve courant (17 480 B).
   Couvre primes 13, puis paires {17,19}, {23,29}, …, {67,71}. Gain
   estimé **5–10 % wall**, ~300 LOC. Garde-fou : tester que
   `template_bytes + sieve_bytes ≤ 32 KB` strictement (la tentative
   C=7 mono-template = 17017 B avait régressé +5.5 % par cache miss).

2. **Phase 3 v2 avec seuil conditionnel** (cf section ci-dessus). Effort
   faible (~30 LOC : `partition_point` + 2 boucles dans hard.rs). Gain
   estimé modeste (1–2 % wall) car la moitié des primes bulk reste
   rolled. Tenter seulement si la Mesure WSL post-2A confirme qu'on est
   loin de Kim.

3. **Re-bench WSL post-`a8aa681`** pour figer le facteur Kim définitif
   (Win→WSL ~6-7 % wall). Estimation actuelle 4.5× / 6.05× à confirmer.

4. **Piste D** — étendre le frontier `b_ext` vers `b_limit` pour basculer
   plus de primes en chemin bulk (avec son scaling de coût). Effort
   élevé, gain plausible 5–10 %.

5. **Piste C** — sub-task Rayon dans `bi_main` pour récupérer les ~10
   points de % d'efficacité Rayon perdus à α=2 (78 → 71 % post-2A).

6. **README refresh** : la perf table publiée est figée à `9e9162a`
   (avant cascade). Update au state cool actuel.

---

## Mémoire principale

`~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_s2_hard_refactor.md`

Mise à jour le 2026-04-26 avec : Phase 1+2A landed (66ce0b1, e2528cf),
chiffres cool définitifs (1e15 −15.8 %, 1e17 −13.4 %), Phase 3 explored+reverted
(f43d1b5) avec rationale du seuil conditionnel pour Phase 3 v2.
