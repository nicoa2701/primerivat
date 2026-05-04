# Handoff post-session 2026-05-04 — primerivat

> **Lecture obligatoire en début de prochaine session.**
> Cette session a tenté Option 2 du handoff précédent (work-stealing global à la
> Kim). Trois phases livrées : Phase 1 cost-sort scheduling (gain -5 %), Phase 2
> sub-band chunking ext_easy (gain neutre vs Phase 1), Phase 3 sub-band chunking
> rest_bulk (BUG, désactivé). Ratio wall **6.5× → 6.2×**. Plancher rest_bulk
> mono-thread reste le bottleneck.

---

## 1. Référence chiffrée actuelle (post-session)

Working tree **modifié** (non-commité) :
- `M src/dr/hard.rs` — ~250 LOC ajoutées (compute_phi_init_at, WorkItem struct, work-pool, cost-sort, subdivision)
- `M src/parameters.rs` — env vars `RIVAT3_NO_WORKPOOL`, `RIVAT3_PHI_INIT_PROBE`, `RIVAT3_SUBDIVIDE_HEAVY`, `RIVAT3_HEAVY_CHUNKS`, `RIVAT3_SUBDIVIDE_REST_BULK`

Commit HEAD = `1a85b95` (inchangé).

| | wall step3 | wall total | ratio vs primecount-d |
|---|---:|---:|---:|
| 9700X 1e18 α=2 baseline (no env vars, workpool cost-sort actif) | 45.46 s | 47.90 s | **6.06×** wall |
| 9700X 1e18 α=2 avec RIVAT3_SUBDIVIDE_HEAVY=3 RIVAT3_HEAVY_CHUNKS=8 | 45.24 s | 47.67 s | **6.04×** wall |
| 9700X 1e18 α=2 RIVAT3_NO_WORKPOOL=1 (= ancien comportement) | 47.60 s | 50.05 s | **6.34×** wall |

Gain net session : **-2.36 s wall** (47.60 → 45.24) = **-5 %**.

Critère long-terme : ≤ 2× wall. Cumulé optimiste sur leviers restants : ~3.5×.

## 2. Ce qui a fonctionné

### Phase 1 — Cost-sort scheduling (gain -5 %)

Replace `(0..num_bands).into_par_iter().map(...)` with :
- Build `Vec<WorkItem>` (1 par bande pour baseline) avec metadata (band_id, chunk_lo, chunk_hi, walker_start_n, p2_min_rank, is_heavy)
- Workers via `rayon::scope + AtomicUsize cursor`, picknt items dans l'ordre `item_order` (sort desc par predicted_cost)
- predicted_cost = `(x * span / (lo * hi)) / 7 + span` (combine ext_easy ∝ x*span/(lo*hi) et rest_bulk ∝ span avec calibration empirique)

**Bénéfice mesuré** : -2.1 s wall vs `into_par_iter` natural order. Le sort desc place les bandes lourdes en tête, démarre le travail long-running tôt.

**Coût** : +30 s CPU (+5 %) à cause de cache thrash partiel sur rest_bulk_xoff (les 8 bandes hautes démarrent simultanément accèdent ~32 MB de pb_data).

### Phase 2 — Sub-band chunking ext_easy (gain marginal)

Subdivide top-3 ext_easy heavy bands (1, 2, 3 à 1e18 α=2) en K sub-chunks via
`RIVAT3_SUBDIVIDE_HEAVY=3 RIVAT3_HEAVY_CHUNKS=8`. Chaque sub-chunk :
- Recompute `phi_init_at_chunk_lo` from scratch via `compute_phi_init_at(chunk_lo)` (extracté du block `initial_phi_vec`)
- p2_init via `primes.partition_point(|&p| p < chunk_lo) + s2_primes.count_le(chunk_lo - 1)` (fix pour chunk_lo > y)
- `is_heavy: false` (single-pass — défère pass-2 nested par_iter cassé sur sub-chunks)
- Process_band paramétré accepte (band_id, chunk_lo, chunk_hi, walker_start_n, p2_min_rank, is_chunk_heavy)
- Resolve par chunk : si `chunk_lo == band_lo` use prefix-sum phi_band_inits[t] ; sinon compute from scratch

**Bit-exact validé** : 1e13, 1e15 α=2, 1e16 α=2, 1e17 α=2, 1e18 α=2 (résultat 24 739 954 287 740 860).

**Bénéfice mesuré** : +0.2 s wall sur baseline cost-sort (DANS LE BRUIT). La subdivision parallélise band 2 (44 s solo) en 8 sub-chunks de ~5.5 s, mais le wall reste plafonné par rest_bulk heavies (band 255 = 28 s solo).

**Phase 0 instrumentation** (`RIVAT3_PHI_INIT_PROBE=1`) : conservée, mesure le coût recompute phi_init au sein des bandes lourdes. 9700X 1e18 α=2 confirmé : 0.8-2.5 ms par chunk_lo, **négligeable** vs solo CPU des bandes (44 s).

## 3. Ce qui a échoué (et pourquoi)

### Tentative 1 : Phase 1 sort desc plain (réverté)
- Sort tous les items par predicted_cost desc
- 9700X 1e18 α=2 : **+40 % wall (45 → 62 s)**
- Cause : 8 rest_bulk heavies + 8 sub-chunks ext_easy démarrent simultanément, accèdent 8 régions distinctes de pb_data → cache thrash L3 massif.
- Fix : limit head ordering à 3 ext_easy heavies, le reste en ordre naturel (Tentative 2).

### Tentative 2 : Phase 1 hybrid head ordering (réverté)
- Top-3 ext_easy heavies en tête + reste en ordre naturel
- 9700X 1e18 α=2 : neutre (~46 s, +1.6 % vs no_workpool baseline)
- Cause : les 3 bandes lourdes étaient déjà en early positions du natural order. Le head ordering n'apporte rien.
- Bug critique du head : `ext_easy_weight` filtrait `mid >= x^(1/4)` qui rejetait toutes les bandes ext_easy heavy à 1e18 α=2 (mid > 31k). Head **toujours vide**. Phase 1 et Phase 2 tournaient en mode no-op (no head, no subdivision).
- Fix : viré le filter x4, retenir uniquement `blo == 0` exclusion.

### Tentative 3 : Phase 1 cost-sort + head ordering (réverté)
- Après fix x4, head non-vide. Wall régresse à 50 s (vs 45 s avec head vide).
- Cause : avec head non-vide, item_order place [band 1, 2, 3] en tête, pousse les rest_bulk heavies (bands 248-255) à la **queue de queue**. Workers picknt rest_bulks tard, finissent à t=50 s.
- Fix final : remplacer head ordering par sort par predicted_cost desc (placement uniforme).

### Tentative 4 : Phase 3 sub-band chunking rest_bulk (BUG, désactivé)
- `RIVAT3_SUBDIVIDE_REST_BULK=N` ajoute les top-N rest_bulk heavies au subdivide_set
- 1e15 α=2 minimal repro (1 band en K=2) : **π faux + 19.9s wall** (vs 0.5s normal)
- Cause **non-déterministe** (π différent entre runs) → race condition ou bug de cohérence
- π wrong + 40× slowdown → sub-chunks émettent leaves incorrectes, delta/bi_contrib/leaf_partial non-commutatifs
- **Investigation incomplète** : 2-3 hypothèses débuggées (p2_init formula pour chunk_lo > y → fixé, mais bug persiste)
- Hypothèse résiduelle : interaction subtile avec `local_p2_offset`, `bulk_active_end` ou `b_limit` quand chunk_lo dépasse `x^(1/4)`

## 4. Comparaison primecount (relue cette session)

primecount `S2_hard.cpp` traite **chaque chunk indépendamment** :
```cpp
Vector<int64_t> phi = phi_vector(low, max_b, primes, pi);  // recomputed per chunk
Sieve sieve(low, segment_size, max_b);
// segmented sieve loop ... iterates b in [min_b, max_b] for ALL b, no separation
```

**Différences architecturales clés** :
1. **`max_b` adapté par chunk** : pour high-blo, `max_b = π(min(√(x/low), √z, y))`. Diminue avec low.
   primerivat utilise `b_ext` fixe (= π(x^(1/4))). Pour high-blo où √(x/low) < x^(1/4), notre b_ext est trop large.
2. **Pas de séparation ext_easy / rest_bulk** : un seul flux `b in [min_b, max_b]`, leaves émises directement.
   primerivat sépare en bi_main (≤ b_limit), rest_plain (b_limit..b_ext), rest_bulk (b_ext..bulk_active_end). Cette séparation est mécaniquement utile (cross-off tight, leaf emit séparé) mais **complique** le subdivision.
3. **Pas de delta-prefix-sum chain entre bands** : chaque chunk est autonome (phi_vector recomputé).
   primerivat combine ces deux mondes : whole bands via prefix-sum, sub-chunks via from-scratch. C'est cette dualité qui semble briser quelque chose pour Phase 3.

**Conclusion** : Phase 3 propre nécessiterait probablement un refactor plus profond pour aligner primerivat avec l'architecture primecount (chaque chunk indépendant, max_b adapté, pas de bulk_regime séparé). C'est une session dédiée.

## 5. Cibles probables pour la prochaine session

### Option A : Debug Phase 3 (continuation de cette session)
- Reproduire le bug à 1e15 K=2 minimal (déjà fait)
- Ajouter eprintln traces dans process_band pour high-blo sub-chunks
- Comparer delta/bi_contrib output entre 1 whole band vs sa subdivision en K=2
- Trouver où la commutativité casse
- **Confiance** : moyenne (bug subtle mais isolé)
- **Gain potentiel si Phase 3 marche** : -5 à -10 % wall additionnel (= total -10 à -15 % vs pré-session)

### Option B : Refactor primecount-style (rebasculer sur Option 1 du handoff précédent)
- `c` extensible (5 → 6/7/8) + multi-template AND pre-sieve (Kim `Sieve_pre_sieve.hpp`)
- ~400 LOC, mécanique éprouvée, gain attendu -5 % wall sur S2_hard
- **Confiance** : élevée (mécanique éprouvée chez Kim)
- **Gain attendu** : -5 % wall

### Option C : AVX-512 popcount Zen 5 (fallback)
- ~150 LOC, gain estimé -3 %
- ROI faible mais ciblé sur 9700X spécifiquement

### Recommandation
**Option B** si on veut un gain prévisible avec mécanique éprouvée. C'est le pivot du handoff précédent qui n'a pas été essayé cette session.

**Option A** si on veut épuiser Phase 3 d'abord (le bug est ciblé, pas une impasse architecturale, et le levier est mesurément le plus prometteur — band 255 plafonne à 28 s solo, subdiviser casse ce plancher).

## 6. À lancer en début de prochaine session

1. **Lire ce fichier en entier**
2. **Vérifier git status** : working tree doit montrer `M src/dr/hard.rs` et `M src/parameters.rs` (non-commités)
3. **Bench baseline 9700X 1e18 α=2** (3 runs) : confirmer ~45 s avec workpool cost-sort par défaut
4. **Demander à l'utilisateur** quelle option (A, B, C)
5. **Si Option A** : revisiter Phase 3 avec debug ciblé (instrumenter process_band)
6. **Si Option B** : repartir du handoff `session_handoff_2026-05-03.md`, lire les sources primecount listées
7. **Si Option C** : implémenter directement, pas de discussion préalable

## 7. Garde-fous absolus à chaque session

- Bit-exact : 79+6 tests `cargo test --release` doivent passer
- π exact aux 4 magnitudes : 1e13, 1e15 α=2, 1e16 α=2, 1e17 α=2, 1e18 α=2
- Bench cross-CPU avant merge : 9700X (cible) + 13450HX local (avec AC)
- Bruit 13450HX = ±5-10 % minimum ; 9700X bruit ≤ 2 % ; trancher sur 9700X
- Si neutre : ne pas merger (cf. mémoire `feedback_bench_power`)

## 8. Pointeurs de structure (gain de temps)

- [src/dr/hard.rs:482-530](src/dr/hard.rs#L482) — `compute_phi_init_at` closure (utilisable pour Phase 3 debug ou refactor primecount-style)
- [src/dr/hard.rs:805-820](src/dr/hard.rs#L805) — `process_band` signature paramétrée par chunk
- [src/dr/hard.rs:1438-1495](src/dr/hard.rs#L1438) — Phase 2/3 subdivide_set + work_items builder
- [src/dr/hard.rs:1530-1560](src/dr/hard.rs#L1530) — predicted_cost + item_order sort
- [src/dr/hard.rs:1605-1700](src/dr/hard.rs#L1605) — resolve par chunk (clone phi_band_inits ou compute_phi_init_at)
- [src/parameters.rs](src/parameters.rs) — env vars (PHI_INIT_PROBE, NO_WORKPOOL, SUBDIVIDE_HEAVY, HEAVY_CHUNKS, SUBDIVIDE_REST_BULK)

### Sources primecount à relire pour Option B
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/Sieve_pre_sieve.hpp` (mécanisme + 7 templates)
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/Sieve_arrays.hpp` (pre_sieved tables, format byte-tiled)
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/phi.cpp` (phi_tiny avec table `pp = ∏ primes[1..a]`)

### Sources primecount à relire pour Option A (Phase 3 debug)
- `c:/Users/Kbda9/projet/3rivat3/primecount/src/deleglise-rivat/S2_hard.cpp` (boucle de consommation des chunks)
- Note : primecount n'a PAS de bulk regime séparé. C'est un indice que notre b_ext-fixe + bulk_active_end-dynamique introduit une incohérence quand chunk_lo dépasse certains seuils.

---

*Fichier généré 2026-05-04 en clôture de la session "Phase 1+2+3 work-pool +
sub-band chunking", après gain net -5 % wall (Phase 1+2 OK, Phase 3 bug).
Prochaine session : choisir entre debug Phase 3 (Option A) ou refactor `c`
extensible (Option B).*
