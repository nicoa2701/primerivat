# Session dédiée — bucket sieve pour `rest_bulk_xoff`

> **But de la session** : porter le bucket sieve de primesieve (`EratBig`)
> dans `s2_hard_sieve_par` pour faire chuter `rest_bulk_xoff` (50 % du
> CPU à 1e18, et qui grossit avec x). Cible : −25 à −33 % wall global
> sur 9700X à 1e18.
> **Critère de pass** : wall ≤ 36 s @ 1e18 sur 9700X (vs 47.9 s
> baseline post-PrimeBitset), ou ratio ≤ 5× vs primecount-d.

Écrit le 2026-05-03 après la cascade mémoire (FactorTable + PrimeBitset,
commits `1aeb41d`, `c911d02`, `7e99cba`). État working tree : clean,
3 commits ahead of `origin/main`.

---

## 1. État de référence (post-`7e99cba`)

Apples-to-apples sur 1e18, **Ryzen 7 9700X** (8C/16T / Zen 5 / 32 Mo L3) :

| algo @ 1e18 | wall | RSS | notes |
|---|---:|---:|---|
| primerivat post-PrimeBitset wirage | **47.9 s** | **146 Mo** | référence pour cette session |
| primecount **`-d`** (DR forcé) | **7.5 s** | **80 Mo** | apples-to-apples |
| ratio actuel | **6.4× wall** | **1.83× RSS** | |

Cascade mémoire complète (4 sessions) : RSS 7.34 Go → 146 Mo (-98 %),
gap mémoire vs primecount-d : 103× → 1.83×. Les leviers mémoire sont
épuisés. Le projet pivote sur la perf.

## 2. Profil CPU — pourquoi `rest_bulk_xoff`

Décomposition CPU à 1e16/1e17/1e18 (α=1 sur 13450HX local) :

| phase | 1e16 | 1e17 | 1e18 | tendance |
|---|---:|---:|---:|---|
| `bi_main_xoff` | 23 % | 17 % | 14 % | ↓ |
| `rest_plain_xoff` | 17 % | 19 % | 19 % | ≈ |
| **`rest_bulk_xoff`** | **37 %** | **46 %** | **50 %** | **↑↑** |
| `tail_*` autres | 19 % | 16 % | 15 % | ≈ |
| total cross-off | 77 % | 82 % | **83 %** | ↑ |

**`rest_bulk_xoff` = 50 % du CPU à 1e18 et grossit ~10× par décade**.
À 1e19 il sera à ~55 %. Tout autre levier perf perd en ROI face à lui.

À 7.8 ns/step (mesure) on est cache-bound DRAM, pas CPU-bound. Le
coupable :
- `pb_data: Vec<WheelPrimeData>` = **80 octets/prime** × ~50 K bulk
  primes × num_bands = **~4 Mo par band** → dépasse L2 9700X (1 Mo/core)
- 16 threads en parallèle → contention L3
- Chaque segment vérifie TOUS les primes même ceux avec 0 multiple

## 3. Pourquoi bucket sieve marche ici

Mécanique primesieve `EratBig` :
- `SievingPrime` = 8 octets compacts (uint32 indexes packed + uint32 prime/30)
- `Bucket` = ~8 Ko, ~1000 SievingPrime entries, chaîné pour overflow
- `buckets_[segment_id]` → bucket chain pour ce segment
- Un prime n'est dans `buckets[N]` que quand son prochain multiple est en N
- À chaque cross-off : on calcule le segment cible (N+k via shift) et on
  ré-insère le prime dans `buckets[N+k]`

**Working set par segment** : ~100-200 Ko (un bucket actif), pas 4 Mo.
Tient en L2 par thread.

**Pourquoi parfait pour notre cas** : nos bulk primes font 0-3 multiples
par segment (commentaire existant dans `s2_hard_sieve_par`). C'est
exactement le régime sweet-spot d'EratBig.

## 4. Inner loop EratBig (référence — à porter)

```cpp
// primesieve src/EratBig.cpp lines 220-239
for (; prime != end; prime++) {
  std::size_t mi = prime->getMultipleIndex();
  std::size_t wi = prime->getWheelIndex();
  std::size_t p  = prime->getSievingPrime();

  sieve[mi] &= wheel210[wi].unsetBit;
  mi += wheel210[wi].nextMultipleFactor * p;
  mi += wheel210[wi].correct;
  wi  = wheel210[wi].next;

  segment = mi >> log2SieveSize;
  mi &= moduloSieveSize;

  if (Bucket::isFull(buckets[segment]))
    memoryPool.addBucket(buckets[segment]);
  buckets[segment]++->set(p, mi, wi);
}
```

À adapter pour wheel-30 (notre layout existant, pas wheel-210). Le
shift `>> log2SieveSize` impose **segment size en puissance de 2** —
notre `W30_SEG = 524 280` est juste sous 2^19 = 524 288, faut soit
adapter à 524 288 soit remplacer le shift par `mi / W30_SEG` (plus
lent mais propre).

## 5. Plan de la session — 3 commits

### Commit 1 : `bucket_sieve.rs` standalone + tests (~1.5 h)

Nouveau module `src/bucket_sieve.rs` :
- `SievingPrime { indexes: u32, prime: u32 }` (8B compact)
- `Bucket { entries: [SievingPrime; BUCKET_SIZE], len: u32, next: Option<Box<Bucket>> }`
- `MemoryPool` thread-local (freelist + bump allocator)
- `BucketSieve` : `Vec<*mut Bucket>` indexé par segment_id
- API minimale : `new(num_segments)`, `insert(seg, prime, mi, wi)`,
  `take_segment_buckets(seg)` (retourne chain pour iteration)

Tests :
- Insert + drain bit-exact
- Bucket overflow (forcer plusieurs buckets / segment)
- MemoryPool reuse via freelist
- Round-trip 10K primes

Style : reprendre la convention `prime_bitset.rs` (constants locales,
docs explicites, ~10 tests).

### Commit 2 : Wirage dans `s2_hard_sieve_par` derrière flag (~3 h)

Refactor de `rest_bulk_xoff` (lignes ~870-905 actuelles) :
- AVANT le par_iter : pré-distribuer chaque bulk prime dans son
  bucket de premier multiple (via une passe segmentée d'init)
- DANS la closure : remplacer le for k in 0..target_end loop par
  un drain de `buckets[seg_id]` avec re-insertion segment-cible
- Conserver le path actuel derrière `--legacy-bulk` (env var ou CLI
  flag) pour A/B comparaison

Validation bit-exact : `cargo test --release` + π exact à 1e13/15/17/18.

### Commit 3 : Bench cross-CPU + activation par défaut (~1 h)

- 9700X : `--dr-profile 1e16/17/18`, RSS @ 1e18 via `RIVAT3_MEM_DUMP=1`
- 13450HX (local) : même bench
- 9300HF (si dispo) : sanity check (cache-constrained = oracle dur)
- Si pass critère : retirer le path legacy (`--legacy-bulk`), simplifier

## 6. Garde-fous

- `cargo test --release` : 72 tests minimum (66 lib + 6 bin actuellement)
- π exact aux 4 magnitudes de référence : 1e13, 1e15, 1e17, 1e18
- Pas de régression > 2 % wall sur 9300HF (cache-constrained, oracle dur)
- Mémoire : `pb_data` peut maintenant DIMINUER (on n'a plus besoin de
  WheelPrimeData persistante pour les bulk primes — ils vivent dans les
  buckets). Possible bonus mémoire ~−2 Mo / band.
- Si bucket sieve lent : **ne pas merger**, garder le path legacy. Le
  diagnostic est juste, mais la transposition Rust pourrait avoir des
  pièges (alignement, false sharing entre threads, etc.).

## 7. Fichiers primesieve à relire

```
c:/Users/Kbda9/projet/primesieve/src/
  EratBig.cpp + EratBig.hpp        # bucket sieve, inner loop critique
  Bucket.hpp                        # struct SievingPrime + Bucket
  MemoryPool.cpp + MemoryPool.hpp   # allocator + freelist pattern
  Erat.hpp                          # dispatch EratSmall→Medium→Big
c:/Users/Kbda9/projet/primesieve/include/primesieve/
  config.hpp                        # cache-size constants
```

Pour le multi-template pre-sieve (commit 2 follow-up éventuel) :
```
c:/Users/Kbda9/projet/primesieve/src/
  PreSieve.cpp + PreSieve.hpp + PreSieveTables.hpp
  PreSieve_x86_sse2.hpp / _x86_avx512.hpp
c:/Users/Kbda9/projet/3rivat3/primecount/src/
  Sieve_pre_sieve.hpp               # version primecount, +simple
```

Pour le contexte algorithmique (la S2_hard structure que primecount
utilise — déjà mirroré chez nous) :
```
c:/Users/Kbda9/projet/3rivat3/primecount/src/
  Sieve.cpp                         # cross_off + count, scalaire
  deleglise-rivat/S2_hard.cpp       # structure de la boucle
```

## 8. Référence d'architecture (pas la source de vérité perf)

`c:/Users/Kbda9/projet/a12/src/sieve.rs` : crible wheel-30 byte-packed
multi-template AND **pour π(n) ≤ 1e11**. Bon pour comprendre le layout
wheel-30 byte-packed et le multi-template AND en Rust idiomatique.
**MAIS pas optimisé** — ne pas prendre comme étalon perf.

## 9. Roadmap long terme — fermer le gap primecount-DR

> Source de vérité : ce tableau remplace la roadmap dans
> `~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_long_term_goal_close_primecount_gap.md`.
> Mémoire à mettre à jour en fin de session si bucket sieve passe.

### Objectif

**Ratio cible long-terme : ≤ 2× wall** vs primecount-DR (= même classe
d'implémentation, modulo Rust/C++ et choix de structures). Atteindre
exactement 1× est improbable sans répliquer toutes les micro-optims Kim.

### Trajectoire mesurée (9700X, 1e18, α=2 auto)

| date | commit clé | wall | RSS | ratio wall | ratio RSS |
|---|---|---:|---:|---:|---:|
| 2026-05-02 | `d308a01` (baseline) | 49 s | 7.34 Go | 6.5× | 103× |
| 2026-05-03 | `56c37f8` (u32 all_primes + ext_stored fix) | 48-50 s | 925 Mo | 6.5× | 13× |
| 2026-05-03 | `1aeb41d` (FactorTable wirage) | **46.4 s** | **308 Mo** | **6.2×** | **3.95×** |
| 2026-05-03 | `7e99cba` (PrimeBitset wirage) | 47.9 s | **146 Mo** | 6.4× | **1.83×** |
| **cible session bucket sieve** | TBD | **≤ 36 s** | ~140 Mo | **≤ 5×** | 1.75× |

**Cascade mémoire complète** (4 sessions) : RSS −98 %, gap mémoire
103× → **1.83×** (sous 2× pour la 1ʳᵉ fois). Le projet pivote sur
la perf à partir de cette session.

### Roadmap leviers (révisée 2026-05-03 après lecture primecount + primesieve)

| # | levier | gain estimé | effort | confiance | statut |
|---|---|---:|---:|---:|---:|
| **1** | **Bucket sieve `rest_bulk_xoff`** (primesieve EratBig) | **−25 à −33 % wall** | ~500 LOC, 1-2 sessions | élevée | **CETTE SESSION** |
| 2 | Multi-template AND pre-sieve étendu (primes 13..71) | −5 % wall | ~300 LOC, 1 session | élevée | planifié |
| 3 | `fast_div64` libdivide pour `x/(pb·m)` | −5 % wall | ~80 LOC, 0.5 session | élevée | planifié |
| 4 | `pb_data` packing 80B → 32B | −5 à −10 % wall | ~150 LOC | moyenne | **caduc si bucket sieve passe** (les bulk primes vivent dans les buckets, pas dans `pb_data`) |
| 5 | AVX-512 popcount sur `count_primes_upto_int` | −3 % wall | ~150 LOC | moyenne (Zen 5 OK, 13450HX désactivé) | planifié |
| 6 | `LoadBalancerS2`-style dynamique vs bands statiques | −5 à −10 % wall | ~150 LOC | moyenne (imbalance s'améliore avec x) | planifié |
| 7 | `phi_tiny` O(1) Kim-style (table `pp` à a=8) | −5 % wall | ~600 LOC, 2-3 sessions | basse (S1 déjà petit) | optionnel |
| ~~SIMD cross-off AVX2~~ | **invalidé** : primecount cross-off est scalaire 8-way comme nous, pas SIMD | — | — | — | — |
| ~~Compress `all_primes` PiTable~~ | **fait** (`7e99cba`, −171 Mo RSS) | — | — | — | — |
| ~~Stream `hard_leaves` via FactorTable<u16>~~ | **fait** (`1aeb41d`, −617 Mo RSS, -5 % wall) | — | — | — | — |

### Cumul si tous les leviers planifiés passent

Ratio actuel **6.4×** → optimiste **~3-3.5×** (post leviers 1-6 cumulés
en multiplicateurs : 6.4 × 0.7 × 0.95 × 0.95 × 0.97 × 0.93 ≈ 3.6).
Pour descendre sous 2× il faudra probablement aussi répliquer
`phi_tiny` Kim et quelques peephole micro-optims (counter array
tuning, prefetch hints, etc.).

### Comment chaque session doit s'inscrire

1. **Établir le baseline** apples-to-apples au début (nouveau ratio
   sur 9700X via `--dr-profile 1e18` + `RIVAT3_MEM_DUMP=1`).
2. **Choisir un levier** dans la roadmap selon le ROI / effort restant.
3. **Mesurer le delta** sur les 3 CPUs de référence (9700X / 13450HX /
   9300HF si dispo) avant de merger.
4. **Mettre à jour la mémoire** (`project_long_term_goal_close_primecount_gap.md`
   et `project_primecount_gap_9700x.md`) avec le nouveau ratio.
5. **Mettre à jour ce fichier** ou créer un nouveau `session_<slug>.md`
   pour la session suivante.

### À ne pas confondre

- **DR vs DR** (apples-to-apples, ce projet) : ratio actuel 6.4×, cible 2×.
- **DR vs Gourdon** : primecount default est ~2× plus rapide algorithmiquement
  car Gourdon < DR à grande magnitude. Hors scope de primerivat (qui reste
  DR pur). Ne pas confondre les deux ratios.
