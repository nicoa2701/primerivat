# Handoff post-session 2026-05-03 — primerivat

> **Lecture obligatoire en début de prochaine session.**
> Cette session a tenté 4 leviers, tous échoués/neutres. Le ratio wall reste
> **6.4×** vs primecount-DR sur 9700X 1e18 α=2. Mais le profile cross-magnitude
> a livré le diagnostic actionnable pour la suite.

---

## 1. Référence chiffrée actuelle (post-revert)

Working tree **clean**, commit HEAD = `1a85b95`.

| | wall | RSS | ratio vs primecount-d |
|---|---:|---:|---:|
| 9700X 1e18 α=2 | **47-49 s** (médiane 47.88 s, min 46.30 s) | 146 Mo | **6.4×** wall, 1.83× RSS |
| 13450HX 1e18 α=2 | ~67 s (variance ±35 % sur batterie / thermique) | — | — |

Critère long-terme : ≤ 2× wall. Cumulé optimiste sur leviers restants : ~3.6×.

## 2. Insight CENTRAL : profile cross-magnitude (le seul résultat productif de la session)

`--dr-profile` à α=auto sur 9700X (decomp `S2_hard`) :

| phase / x | 1e15 | 1e16 | 1e17 | 1e18 |
|---|---:|---:|---:|---:|
| `bi_main_xoff` | 26.2 % | 22.9 % | 18.4 % | 12.1 % |
| `rest_plain_xoff` | 14.0 % | 17.0 % | 18.8 % | 14.5 % |
| `rest_bulk_xoff` | 29.3 % | 35.4 % | 43.1 % | 29.9 % |
| **`tail_ext_easy_emit`** | 13.7 % | 11.4 % | 9.6 % | **36.7 %** ← saut α=1→α=2 |
| `tail_advance` | 13.0 % | 11.0 % | 8.4 % | 4.7 % |
| **imbalance Rayon** | 31.4× | 19.7× | 8.7× | **18.8×** |
| ext_emitted | 39 M | 162 M | 674 M | **7.24 G** |

**Conclusion** : le saut tail_ext_easy 9.6 % → 36.7 % entre 1e17 (α=1) et 1e18 (α=2) explique le gap. **7.2 G ext-easy leaves émises à 1e18** concentrées sur 3 bands à basse n (band 1+2+3 = 102 s sur 221 s total tail_ext_easy CPU). Eff Rayon globale ~5 %. **Là où il faut frapper.**

## 3. Pourquoi les 4 leviers tentés cette session ont échoué (HONNÊTEMENT)

### Bucket sieve `rest_bulk_xoff`
- Hypothèse fausse : `pb_data ~70 Mo > L3` ⇒ bucket sieve réduit working set
- Réalité : pb_data ~12 MB déjà dans L3 ; recompute `WheelPrimeData::new(p)` (~250 cycles) > prefetcher matériel qui absorbait déjà le DRAM cost
- Mesure : 9700X 1e18 = **1.93× pire** (94.8 s vs 49.1 s)
- État : module conservé derrière `--bucket-bulk` opt-in

### `fast_div64` (libdivide / pré-calcul `x/pb`)
- Hypothèse fausse : u128/u128 = ~80 cycles ⇒ u64/u64 ~3× plus rapide
- Réalité : LLVM optimise déjà `(pb as u128 * m as u128)` via `__udivti3` fast-path runtime `if high == 0` ⇒ ~30 cycles déjà
- Mesure : 9700X 1e18 α=2 médiane neutre (-0.5 %, run 3 post = run 1 baseline)
- Diagnostic complémentaire : la div est diluée par popcount + binsearch dominants dans le hot loop

### `LoadBalancerS2` Phase A (multi-heavy via `ext_estimate ≥ 4× mean`)
- Hypothèse fausse : "marquer plus de bands heavy déférerait plus de tail_ext_easy"
- Réalité : 3 heavy bands lancent chacune un `nested par_iter` sur 40K ei en parallèle = ~48 sub-tasks pour 16 threads ⇒ contention massive + overhead Rayon nesting
- Mesure : 9700X 1e18 α=2 = **+9.4 % régression** (47.88 → 53.33 s)

### `LoadBalancerS2` Phase A bis (top-1 heavy seul)
- Hypothèse plus modeste : "défer juste le band max (band 2 à 1376M emit)"
- Réalité : à 1e18 α=2, les bands hautes-n (252-255) tournent ~30 s chacune sur `rest_bulk_xoff` ⇒ pas de thread idle pour le drain pass-2 nested du band 2
- Mesure : 9700X 1e18 α=2 = **+2.6 % régression** (47.88 → 49.14 s)
- **Insight clé** : le commentaire du code "pass-2 routes ei tasks to threads that finished their light bands early" est faux à 1e18 α=2

### Conclusion mécanique
Le mécanisme `defer + pass-2 nested par_iter` est **structurellement cassé** à 1e18 α=2. Vrai work-stealing à la Kim (mutex + queue + segments adaptatifs) requis pour fixer `tail_ext_easy_emit`.

## 4. Deux chemins viables pour la prochaine session

### Option 1 : Refactor `c` extensible + multi-template AND pre-sieve
**Cible** : étendre `c = 5` (wheel-30 + phi-tiny pour {2,3,5,7,11}) à `c = 6/7/8` (intégrer 13/17/19 dans phi-tiny et le sieve init).
**Apport** : permet le pre-sieve à la primecount Kim (templates byte-tiled + AND), gain ~5 % S2_hard mesuré chez Kim.
**Effort** : ~400 LOC. Touches multiples :
- `phi.rs` : étendre la table phi-tiny
- `factor_table.rs` : ajuster les bornes / encoding
- `dr/hard.rs` : décaler `b = bi + c + 1`, leaves bounds
- `segment.rs` : étendre `fill_presieved_7_11` à `fill_presieved_to_K(K)`

**Sources primecount à relire** :
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/Sieve_pre_sieve.hpp` (mécanisme + 7 templates)
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/Sieve_arrays.hpp` (pre_sieved tables, à inspecter pour le format byte-tiled)
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/phi.cpp` (phi_tiny avec table `pp = ∏ primes[1..a]`)

**Confiance** : élevée (mécanique éprouvée), risque moyen (refactor en plusieurs fichiers).

### Option 2 : Vrai work-stealing global à la Kim LoadBalancerS2
**Cible** : `tail_ext_easy_emit` 36.7 % à 1e18 + imbalance 18.8× ⇒ eff Rayon 5 % → 30-50 %.
**Apport** : remplace `(0..num_bands).into_par_iter().map(closure)` par mutex + queue + chunks alloués dynamiquement, segment_size adaptatif (x^1/4 → L1 → L2 → sqrt(high)).
**Effort** : ~400 LOC. Refactor profond Rayon → manuel (threads explicites + Mutex + condvar).
**Gain attendu** : −10 à −20 % wall à 1e18 α=2 (mesure 9700X).

**Sources primecount à relire** :
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/LoadBalancerS2.cpp`
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/LoadBalancerS2.hpp`
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/deleglise-rivat/S2_hard.cpp` (boucle de consommation des chunks)

**Confiance** : moyenne (mécanique éprouvée chez Kim mais transposition Rust non triviale, threads manuels au lieu de Rayon natif).
**Risque** : élevé (refactor profond du flux par_iter ; tests bit-exact à valider à chaque étape).

### Option 3 (fallback) : leviers plus petits
- AVX-512 popcount Zen 5 : ~150 LOC, gain estimé −3 %, ROI faible mais ciblé
- phi_tiny O(1) Kim-style : ~600 LOC, S1 deja petit, gain ≤ 5 %

## 5. Recommandation

**Option 1** (`c` refactor) si on veut un gain prévisible (~5 %) avec mécanique éprouvée. La voie naturelle après cette session ratée : on attaque directement la cause racine `tail_ext_easy` via le **bon** mécanisme côté algo, pas côté distribution.

**Option 2** si on veut un gros gain potentiel (−10 à −20 %) et qu'on accepte le risque d'un refactor Rayon → manuel.

L'option 1 est plus dans la lignée de ce qui a marché historiquement sur primerivat (PiTable compress, FactorTable<u16>, PrimeBitset — tous des refactors de structures à mécanique claire).

## 6. À lancer en début de prochaine session

1. **Vérifier l'état** : `git status` (doit être clean), `git log --oneline -5`
2. **Bench baseline 9700X 1e18 α=2** (3 runs) : confirmer ~47-49 s
3. **Demander à l'utilisateur** quelle option (1, 2 ou 3)
4. **Lire les sources primecount listées** ci-dessus selon l'option
5. **Présenter un plan détaillé** avant de coder (l'utilisateur préfère discuter avant)

## 7. Garde-fous absolus à chaque session

- Bit-exact : 79 tests `cargo test --release` doivent passer
- π exact aux 4 magnitudes : 1e13, 1e15, 1e17, 1e18
- Bench cross-CPU avant merge : 9700X (cible) + 13450HX local (avec AC)
- Bruit 13450HX = ±5-10 % minimum ; 9700X bruit ≤ 2 % ; trancher sur 9700X
- Si neutre : ne pas merger (cf. mémoire `feedback_bench_power`)

## 8. Pointeurs de structure de code (gain de temps)

- `src/dr/hard.rs:183` — `s2_hard_sieve_par()` (entry point S2_hard, ~1300 LOC)
- `src/dr/hard.rs:611` — heuristic heavy band (actuellement `t < 2`)
- `src/dr/hard.rs:704` — `(0..num_bands).into_par_iter().map(|t| ...)` (le top-level Rayon à éventuellement refactor)
- `src/dr/hard.rs:1057-1077` — branche heavy : skip pass-1 inline, push deferred
- `src/dr/hard.rs:1205-1275` — pass-2 inline replay (nested par_iter)
- `src/segment.rs:340` — struct `WheelPrimeData` (80 B/prime)
- `src/segment.rs:581` — `fill_presieved_7_11` (template byte-tiled actuel)
- `src/segment.rs:716` — `cross_off_count_pd` (rolled, hot path leaves)
- `src/segment.rs:751` — `cross_off_count_pd_unrolled` (unrolled g, hot path bi_main + plain)
- `src/segment.rs:796` — `cross_off_pd_from_state` (rolled, hot path rest_bulk linear sweep)
- `src/factor_table.rs` — `FactorTable<u16>` (μ + lpf, hard-leaves stream)
- `src/phi.rs` — `phi_tiny`, S1 DFS (point d'entrée pour refactor `c`)

---

*Fichier généré 2026-05-03 en clôture de la session "fast_div64 + LoadBalancerS2 phase A/A bis", après 4 leviers échoués/neutres. Prochaine session : choisir entre option 1 (refactor `c`) et option 2 (vrai work-stealing).*
