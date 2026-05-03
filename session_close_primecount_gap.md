# Session dédiée — fermer le gap primecount-DR

> **But de la session** : poser un premier jalon mesurable qui rapproche
> primerivat de primecount-DR sur le ratio apples-to-apples. État de
> départ : 6.5× au 1e18 sur 9700X (49 s vs 7.5 s `primecount -d`).
> Critère de pass : ratio ≤ 5× ou gain wall ≥ 15 % sur 1e18 9700X.

Écrit le 2026-05-02 après la session "2-tier α" (commits `8a6d89b` →
`d308a01`).

---

## 1. État de référence (commit `d308a01`)

Apples-to-apples sur 1e18, 9700X (8C/16T / Zen 5 / 32 Mo L3) :

| algo | wall | ratio vs primerivat |
|---|---:|---:|
| primerivat (DR, α=2 auto) | 49 s | 1.0× (référence) |
| primecount **`-d`** (DR forcé) | **7.5 s** | **6.5×** |
| primecount default (Gourdon) | 3.234 s | (mix algo + implé, ~13×) |

Sur 9300HF, ratios DR-vs-DR antérieurs : 4.5× à 1e15, 6.05× à 1e17 (à
re-mesurer post-cascade).

## 2. Observation mémoire critique — 1e19

À x=1e19 sur le 9700X :

| algo | RSS pic |
|---|---:|
| primecount Kim | **145 Mo** |
| primerivat | **5.94 Go** |
| facteur | **~41×** |

C'est plus important que le facteur perf (6.5×). À cette échelle :
- 5.94 Go ≈ 19 % de la RAM serveur — risque de pression mémoire / swap
- L'écart mémoire suggère que primerivat **pré-alloue** des structures que
  primecount **streame** ou compresse
- Suspects principaux à analyser :
  - `seed_primes: Vec<u64>` à 1e19, `π(√1e19) = π(3.16e9) ≈ 1.46e8`
    primes × 8 octets = **1.17 Go** rien que pour les primes seed
  - `prefix_counts` re-rempli à chaque segment — devrait être OK si bien
    réutilisé
  - `phi_vec[bi]` × `b_max` × bands — taille `b_max ≈ a ≈ 5e8` à 1e19,
    si stocké en u64 : 4 Go par band — **VRAISEMBLABLEMENT le coupable**
- Note : x=1e19 dépasse la plage validée (1e18). Pas de garantie de
  correctness.

L'écart mémoire impacte aussi la perf : moins de cache hits, plus de
pression sur le memory controller, surtout sur Zen 5 où la BW DDR5 est
le facteur limitant à grande magnitude.

## 3. Roadmap leviers (ré-ordonnée avec mémoire)

| # | levier | ROI perf | ROI mémoire | effort |
|---|---|---:|---:|---:|
| **A** | **SIMD cross-off AVX2/AVX-512** sur `WheelSieve30` | **−30 à −50 %** | neutre | gros (~500 LOC + asm) |
| **B** | **Audit + compaction `phi_vec`** (passer u64 → u32 si possible, voir si scope plus court) | 5-15 % | **−50 à −80 %** | moyen (~200 LOC) |
| **C** | **phi_tiny O(1)** Kim-style (`pp = ∏primes[1..a] = 9 699 690`) | −5 à −10 % | léger gain | gros (~600 LOC, repense S1) |
| **D** | **Multi-template AND pre-sieve** (Kim `Sieve_pre_sieve.hpp`) | −5 à −10 % | neutre | moyen (~300 LOC) |
| E | §8.B `rest_bulk_xoff` intra-band parallel | −15 % α=2 | neutre | moyen (~150 LOC) |
| F | §9 heuristique `is_heavy` généralisée | 0-3 % | neutre | léger (~150 LOC) |

**Levier B est nouveau** depuis l'observation mémoire — devient probablement
priorité 1 ou 2 selon la mesure RSS à 1e17/1e18 (plus pratique que 1e19
qui dépasse la plage validée).

## 4. Plan de la session — option par défaut : Levier B (audit mémoire)

Plus accessible que SIMD (Levier A), gros ROI mémoire potentiel, gain perf
collatéral via cache locality. Plan en 4 phases :

### Phase B.0 — Mesure préalable (~30 min)

```bash
# Sur 9700X, ajouter la mesure RSS au tag de démarrage si pas déjà fait
/usr/bin/time -v ~/primerivat/target/release/primerivat 1e17 2>&1 | grep "Maximum resident"
/usr/bin/time -v ~/primerivat/target/release/primerivat 1e18 2>&1 | grep "Maximum resident"

# Comparaison primecount
/usr/bin/time -v ~/primecount/primecount -d 1e17 2>&1 | grep "Maximum resident"
/usr/bin/time -v ~/primecount/primecount -d 1e18 2>&1 | grep "Maximum resident"
```

Établit le ratio mémoire sur la plage validée (1e17, 1e18) — probable
~30-50× selon l'extrapolation depuis 1e19.

### Phase B.1 — Audit code (~1 h)

Identifier les 3-4 plus gros allocateurs via :
1. `cargo run --release --features dhat-heap -- 1e17` (si possible)
2. Lecture statique de `s2_hard_sieve_par` et alentours pour repérer les
   `Vec<u64>` × `b_max` ou similaires
3. Comparer avec primecount/src/Sieve.cpp et S2_hard.cpp pour voir leurs
   structures équivalentes

Rendu : un tableau "structure → taille @ 1e18 → potentiel de compression".

### Phase B.2 — Implémenter une réduction (~2 h)

Cibler la plus grosse poche. Hypothèse de travail (à vérifier) :
- `phi_vec` peut probablement passer u64 → u32 sans débordement si les
  contributions par segment sont bornées par `W30_SEG / log(p_b)`
- Ou alors `phi_vec` n'a pas besoin d'être maintenu pour tous les bi
  simultanément — primecount le streame

### Phase B.3 — Bench + valider (~30 min)

```bash
# Sur 9700X
/usr/bin/time -v ~/primerivat/target/release/primerivat 1e18    # mémoire + wall
~/primerivat/target/release/primerivat --dr-profile 1e18        # validation
cargo test --release                                             # bit-exact
```

Critères pass :
- Mémoire 1e18 : ≥ −30 %
- Wall 1e18 : neutre ou meilleur (cache locality bonus)
- π(1e18) bit-exact = 24 739 954 287 740 860
- 49 + 6 tests passent

## 5. Plan alternatif — option agressive : Levier A (SIMD cross-off)

Si on a 4-6 h dédiées et qu'on veut viser le gros gain perf direct.

### Plan en 3 étapes

1. **POC vectoriel** sur `cross_off_pd_unrolled` un seul groupe (g=0,
   p%30=1) avec AVX2 (`_mm256_and_si256` sur 32 bytes / 256 bits par cycle).
   Comparer asm + bench microbenchmark.
2. **Étendre aux 8 groupes** via `match` dispatching, garder rolled
   fallback pour primes très grands.
3. **Wirage** dans `bi_main_xoff` et `rest_plain_xoff` + tests bit-exact.

Risque : l'expérience Phase 3 (cross_off_pd_from_state_unrolled, reverted)
montre que le déroulé pénalise les call-sites à faible nombre d'itérations.
SIMD aurait probablement le même problème pour les primes bulk
(`p > x^{1/4}`) — prévoir un seuil conditionnel d'emblée.

Effort réel attendu : 1-2 sessions. La première pose le POC + bench, la
seconde wire dans la prod.

## 6. Garde-fous

À chaque session :
- `cargo test --release` — 49 + 6 tests doivent passer
- π exact à 1e13, 1e15, 1e17, 1e18 (les 4 magnitudes de référence)
- Bench cross-CPU avant merge :
  - 9700X (auto-α : 1e17 α=1, 1e18 α=2) — la cible Zen 5
  - 9300HF (1e17 α=2 cool, ~38-42 s baseline) — vérif tier 9300H
  - 13450HX (1e17 α=1, ~14 s baseline) — vérif tier asym
- Pas de régression > 2 % wall sur aucun des 3 CPUs

## 7. Fichiers Kim primecount à relire

```
~/primecount/src/                              # Linux WSL ou serveur
c:/Users/Kbda9/projet/3rivat3/primecount/src/  # Windows
```

Pour Levier A (SIMD cross-off) : `Sieve.cpp` lignes 222-596 (switch 64
cases pour `cross_off`/`cross_off_count`), `Sieve_count_simd.hpp`.

Pour Levier B (mémoire) : `Sieve.hpp` (struct PrimeState, Counter),
`S2_hard.cpp` (organisation des buffers), `phi.cpp` (phi_tiny pour la
relation S1 / mémoire).

Pour Levier C (phi_tiny) : `phi.cpp`, `Sieve_arrays.hpp` (tables
`pre_sieved_*`).

Pour Levier D (multi-template AND) : `Sieve_pre_sieve.hpp`, chaîne de 7
templates.

## 8. Suivi mémoire (à mettre à jour après la session)

Après chaque session, mettre à jour
`~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_long_term_goal_close_primecount_gap.md`
avec le nouveau ratio post-session (perf et mémoire), pour suivre la
trajectoire vers la cible ≤ 2×.
