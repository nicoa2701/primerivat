# 2-pass S2_hard restructure — design doc

> **STATUS au 2026-04-29 fin de journée — Étape A shipped (commit `ddc1786`)**
>
> POC V2-ciblée implémenté : heavy bands `t ∈ {0,1}` quand `use_log_scale`
> est actif (= α=2 régime) skip le `tail_ext_emit` en pass 1 et stashent un
> `DeferredSeg` par segment. Pass 2 = nested par_iter sur heavy bands × `ei`,
> exécuté après le `band_sweeps.collect()` quand les light bands ont fini.
>
> **Bench cool 9300HF, 1e17 α=2 : 46.5 s → 38.9 s (−16.3 %).** π(x) bit-exact
> sur tout le range testé (10^6 à 10^17). À α=1 le path est bypassé
> (`ext_clamped=0` ⇒ `use_log_scale=false` ⇒ `is_heavy=false`), neutralité
> prouvée par codepath.
>
> **Critère Étape B (`≥ −10 %` à 1e17 α=2 ET ≤ +2 % à α=1) : PASSED.**
>
> **Pour reprendre dans une session dédiée**, deux pistes ouvertes en §8/§9
> ci-dessous (auto-portantes). La §1-§7 documente le design d'origine et
> reste valable comme référence pour comprendre le code post-`ddc1786`.

---

Ce document trace le plan d'attaque pour la **dernière piste perf
significative** du projet : restructurer `s2_hard_sieve_par` en 2 passes
pour libérer le bottleneck **bands 0+1 / tail_ext_emit** identifié à
x=1e17 α=2 sur i5-9300HF.

Écrit le 2026-04-29 en fin de session, après que les pistes Multi-
template AND, Piste C (nested Rayon), Piste D (b_ext_mult), oversubscribe
-b 32, et phi-recursive c>8 aient toutes été fermées par bench cool.

État du repo au moment de l'écriture : `main` propre, dernier commit
`a1cf99a`, perf cool 9300HF 1e17 α=2 = 46.0 s, efficacité Rayon 71 %.

État du repo après Étape A (2026-04-29 soir) : `main` à `ddc1786`, 38.9 s
à 1e17 α=2 cool, 82 % d'efficacité Rayon.

---

## 1. Pourquoi le 2-pass est la dernière piste viable

### Le bottleneck mesuré (cf. `handoff_9300hf.md` §État 2026-04-29)

À x=1e17 α=2, le profile per-band montre :

```
band  1 (lo ∈ [524k, 1M]) : 26.1 s CPU (99.9 % tail_ext_emit, 526 M leaves)
band  0 (lo ∈ [0, 524k]) : 20.4 s CPU (99.9 % tail_ext_emit, 275 M leaves)
band 126 (high n)         : 17.1 s CPU (rest_bulk + tail_ext mid)
band 127                  : 16.5 s CPU
…
```

Les bands 0+1 cumulent **46.5 s CPU** (= 17.5 % du CPU total 263 s),
quasi-exclusivement via `tail_ext_emit`. L'imbalance ratio max/mean
= 13.2× — le thread qui hérite de band 1 tourne 26 s tandis que les
autres 7 threads finissent leur 16 bands chacun en ~5-7 s par band et
attendent.

Calcul du plafond accessible :

- Wall actuel : 47 s (CPU 263 s / 8 threads idéal = 33 s ; gap = 14 s
  perdus en attente)
- Wall théorique parfait : 33 s (CPU/8) → **−30 % wall** si on
  recouvre l'imbalance complètement

### Pourquoi les approches simples ont échoué

- **Splitter bands 0+1 plus fin** : bands sont déjà à 1 W30_SEG = 524 280
  entiers chacun. Minimum incompressible (le sieve fonctionne par
  segment de W30_SEG bits).
- **Nested Rayon** sur la boucle `for ei in far_easy_start..n_easy`
  dans tail_ext_emit : **+41 % wall**. Outer par_iter sur 128 bands
  sature les 8 threads ; nested spawns trouvent zéro thread idle.
- **Oversubscribe -b 32** : double num_bands à 256 mais ne split que
  les bands déjà rapides au-dessus de band 1. Les bands 0+1 restent
  les bottlenecks.

→ **Seule restructuration viable** : décorréler le tail_ext_emit
des bands 0+1 du par_iter outer concurrent, pour qu'un autre stage
parallèle puisse l'attaquer SANS contention.

---

## 2. Structure actuelle du sweep (référence)

`s2_hard_sieve_par` ([src/dr/hard.rs:178-1090](src/dr/hard.rs#L178)) fait,
pour chaque band t en parallèle :

```text
for each segment lo in [band_lo, band_hi):                         # séquentiel
    sieve.fill_presieved_7_11(lo)                                  # 0.1 % CPU
    for bi in 0..b_limit:                                          # ~17 % CPU
        if has_leaf: emit_hard_leaf_records(bi, ...)
        cross_off_count_pd_unrolled(p_b, ...) for prime p_b
    for bi in b_limit..b_ext: cross_off_pd_unrolled(p_b, ...)      # ~12 % CPU (rest_plain)
    for k in bulk_active: cross_off_pd_from_state(...)              # ~24 % CPU (rest_bulk)
    fill_prefix_counts(&mut p2_prefix)                              # ~0.1 % CPU
    for ei in far_easy_start..n_easy:                              # ~35 % CPU (tail_ext)
        while easy_ptrs[ei] valid:
            raw = sieve.count_primes_upto_int(p2_prefix, n, lo)
            v = local_p2_offset + raw + corrections
            if v >= threshold: ext_fold_partial += ...              # rapide
            else: ext_stored.push(...)                              # rare
            advance easy_ptrs[ei]
    advance bulk state, advance tiny_state                          # ~6 % CPU (tail_advance)
```

Dépendances séquentielles intra-band :
- `phi_vec[bi]` accumule à travers segments
- `delta[bi]` (band-local) idem
- `bulk_next_m[k] / bulk_next_j[k]` (band-local) idem
- `easy_ptrs[ei] / easy_next_n[ei]` (band-local) idem
- `local_p2_offset` accumule

→ **Sub-paralléliser les segments d'un band** est bloqué par toutes
ces dépendances. Le 2-pass vise plutôt à **décorréler le stage qui
n'a PAS de dépendance entre segments** : le `tail_ext_emit` consomme
juste `sieve` (à un état spécifique par segment) + `p2_prefix` + des
corrections, tous calculables séparément.

---

## 3. Variantes envisageables

### V1 — Defer ALL tail_ext_emit queries → 2nd pass global

**Idée** : pendant le sweep, ne pas faire `count_primes_upto_int` ;
juste enregistrer chaque triple `(ei_or_records, lo_segment, n_query)`
dans un buffer. Après le sweep, `par_iter` global sur tous les
records, distribué sur les 8 threads sans contention extérieure.

**Bloqueur** : à 1e17 α=2, **1.7 milliards de leaves émises**. Au
minimum 24 B/record (3 × u64 dans un encoding compact) → **41 GB de
RAM**. Inacceptable.

Si on essaie de réduire à 16 B (e.g. delta-encoding)… toujours **27 GB**.

→ V1 invalide pour 1e17 α=2 sans une compression drastique non-évidente.

### V2 — Store intermediate sieve states + replay tail_ext_emit en pass 2

**Idée** :
- Pass 1 : sweep complet **sans** `tail_ext_emit`. À chaque segment,
  capturer un snapshot du sieve à l'état "post-cross-off"
  (dans une zone d'espace mémoire pré-allouée par band).
- Pass 2 : `par_iter` sur les bands "lourds" (heuristique : bands où
  `n_ext_emitted > threshold` ; en pratique band 0 et 1 à α=2). Pour
  chaque band, refaire le `tail_ext_emit` sur les snapshots stockés.

**Mémoire** : sieve = 17 480 B/segment.
- À 1e17 α=2 : 191 k segments × 17 480 B = **3.3 GB** pour stocker
  TOUS les sieves. Trop pour la RAM machine (8 GB libre).
- Si on stocke seulement les segments des bands "lourds" (= bands 0+1
  à α=2 = 2 segments) : 2 × 17 480 = 35 KB. Trivial.

**Mais** : la PROFITABILITÉ de la pass 2 dépend de combien de threads
on peut consacrer au tail_ext de bands 0+1. Si on a juste band 0 et 1
à traiter, on peut avoir 8 threads en pass 2, mais ces deux bands ne
contiennent que 2 segments → la sub-paralélisation interne se ramène
au problème de la Piste C nested Rayon.

→ V2 marche **uniquement si la pass 2 sub-parallélise sur ei**, ce qui
ramène au scénario Piste C avec un avantage clé : **plus de
contention outer** (les autres bands sont déjà finis). Donc le par_iter
nested fonctionnera. **C'est la voie la plus prometteuse**.

**Coût implémentation** : ~250 LOC.
- Pass 1 : sortir `tail_ext_emit` du single-pass actuel. Capture des
  sieve+p2_prefix+seed_below_lo+local_p2_offset par segment dans la
  zone "deferred ext-emit" pour bands sélectionnés.
- Sélection : heuristique `if t < 2 || (heuristique band lourd)`.
- Pass 2 : `bands_with_deferred.into_par_iter().for_each(|band| { for
  segment in band.deferred_segments { par_iter sur ei pour
  tail_ext_emit } })` — par_iter NESTED, mais cette fois le pool
  outer est libre.

**Risques** :
- Quelle heuristique pour identifier les bands "lourds" ? Compter
  `ext_emitted` projeté avant le sweep ? Or fixer t < 2 à α=2 ?
- Le nested par_iter en pass 2 a-t-il un overhead non négligeable
  même sans contention extérieure ? À mesurer.
- L'état du sieve à chaque segment est un snapshot ; il faut soit
  le copier (35 KB par segment × 2 segments = 70 KB, OK) soit le
  recalculer (recompute coûte ~5 % du sweep total).

### V3 — Pass 1 = cross-off only par band, pass 2 = global re-sweep avec leaves

**Idée** :
- Pass 1 : tous les bands en parallèle font UNIQUEMENT cross-off
  (bi_main, rest_plain, rest_bulk). Pas de leaf emission. Capturent
  delta[bi] et bulk state.
- Sequential prefix scan sur deltas (déjà fait actuellement).
- Pass 2 : tous les bands en parallèle re-font la sweep avec
  l'état initial correct, et émettent les leaves. Cette fois, les
  bands lourds (0+1 à α=2) ne sont plus seuls car la pass 1 a déjà
  fini → tous les threads sont libres pour pass 2.

**Problème** : pass 2 fait à nouveau le cross-off (plus de la moitié
du CPU). On double presque le CPU total → **−15 % wall théorique gain
mangé par +50 % CPU** = wall augmente. Non viable.

→ V3 fail à moins qu'on stocke les sieves de pass 1 (cf. V2).

### V4 — Decouple bands 0+1 avant le par_iter

**Idée** : faire un design custom où band 0 et band 1 sont traités
SÉPARÉMENT (séquentiellement, avec sub-parallélisation dédiée), et
les bands 2..127 dans le par_iter outer existant (taille 126).

```rust
let (heavy_bands_results, light_bands_results) = rayon::join(
    || process_heavy_bands(0..2),       // 2 bands × tail_ext_par
    || (2..num_bands).into_par_iter()...
);
```

Le `process_heavy_bands` fait les 2 bands séquentiellement mais avec
nested par_iter sur ei dans le tail_ext_emit. Pendant ce temps, les
8 threads rayon sont occupés par les 126 bands light.

**Problème** : si "heavy" prend 26+20 = 46 s sequential et "light"
prend `(126/8) × 5-7 s = 80-110 s` total… nope, 126 bands × 5-7 s
chacun ≈ 750 s total CPU ÷ 8 threads ≈ 93 s wall. C'est BIEN PIRE
que les 47 s actuels.

Wait, en réalité les 126 bands light ne prennent PAS 5-7 s chacun à
1e17 α=2. Selon le profil per-band, bands 120-127 prennent 8-17 s
chacun. Les bands 2..119 prennent moins. Total light ≈ ~210 s CPU /
8 threads = 26 s wall. Heavy = 46 s sequential. **Total wall = 46 s,
identique au baseline**. Pas de gain.

Pour gagner : il faut que heavy soit AUSSI parallélisé. C'est donc
V2 + dispatcher.

### V5 — Sieve state checkpoint à mid-band

**Idée** : ajouter à `s2_hard_sieve_par` une option pour stocker
le sieve state à un POINT FIXE de chaque band (e.g. 50 % du chemin),
puis pass 2 démarre depuis ce checkpoint avec 2× plus de bands
effectifs.

**Bloqueur** : la cross-off est en SEG-segments incrémentaux, pas
en demi-band. Découper un band en milieu nécessite de stocker l'état
du sieve à ce point + tous les bulk_next_m/j + delta[bi]. C'est
faisable mais l'overhead annule probablement le gain.

→ V5 = V2 raffiné. Pas plus efficace que V2.

---

## 4. Recommandation : V2 ciblée sur bands 0+1 à α=2

### Plan d'implémentation (ordre suggéré)

**Étape A — POC standalone (~50 LOC, 1 jour)**

Ajouter un mode `--profile-deferred-tail-ext` qui :
1. Détecte les "heavy bands" (par exemple t ∈ {0, 1} à α=2) via un
   threshold simple sur `n_ext_emit_predicted` calculé pré-sweep.
2. Dans le sweep, pour ces bands, COPIE le sieve final + p2_prefix +
   variables locales (local_p2_offset, seed_below_lo, adj_lo) à
   chaque segment dans un Vec. Ne fait PAS le tail_ext_emit pendant
   le sweep.
3. Après le sweep (= fin du `band_sweeps` collect), `par_iter` sur les
   heavy bands. Pour chaque band heavy, replay le tail_ext_emit en
   utilisant les snapshots stockés. Les threads sont libres car
   light bands ont fini.

Mesurer wall à 1e17 α=2 cool. **Critère de succès : −10 % wall**
(vs 46 s baseline → 41 s).

**Étape B — décider** : si gain ≥ 10 % à 1e17 α=2 ET neutre (≤ +2 %)
à 1e15 α=1, on poursuit Étape C. Sinon, on documente et on ferme.

**Étape C — généralisation propre (~150 LOC)**

Heuristique automatique : sélectionner les heavy bands sur la base
de `band.n_ext_emit_predicted > k × mean_ext_emit_per_band`. k à
tuner, peut-être 5 ou 10.

CLI flag `--no-deferred-tail-ext` pour fallback (debugging).

Tests : tous les π(10^k) doivent rester exact ; ajouter un test
explicite que defered match non-defered bit-pour-bit.

**Étape D — bench cross-CPU**

Vérifier que sur 13450HX (toujours α=1, peu de leaves dans bands
bas-n), la heuristique sélectionne 0 heavy band et le code est neutre.

---

## 5. Estimations RAM précises (V2 ciblée)

Pour band 0 et 1 à 1e17 α=2 :
- 1 segment chacun × 17 480 B (sieve) + 4 × (W30_WORDS+1) B (p2_prefix)
  + ~30 B variables = **~21 KB par segment heavy**
- 2 heavy bands × 1 segment chacun = **~42 KB total**

Marginal vs RAM totale. ✓

Si on étend à 4 heavy bands (e.g. 0, 1, 2, 3 à α=2) : 84 KB. ✓

À 1e15 α=1 où aucun band n'est "heavy" : 0 byte stocké, code branche
vers le single-pass actuel. ✓

---

## 6. Risques et failure modes

1. **Nested par_iter en pass 2 a quand même un overhead** : à mesurer
   en POC. Si l'overhead est ≥ 10 % du gain, V2 fail.

2. **Heuristique de sélection mal calibrée** : si on sélectionne trop
   de bands "heavy" à α=1, le pass 2 prend du temps utile aux light
   bands. Mitigation : threshold conservateur basé sur ratio (e.g.
   `band.ext_emit > 5 × mean`).

3. **L'imbalance vraie ne vient pas que de tail_ext_emit** : à α=2 on
   a aussi rest_bulk_xoff dominant (24 % CPU) sur bands 120-127. Si
   on libère bands 0+1, le wall pourrait être limité par le bulk
   d'autres bands. Borne théorique du gain à mesurer.

4. **L'état captured doit être bit-exact** : `seed_below_lo`,
   `adj_lo`, `local_p2_offset`, et l'état du sieve doivent matcher
   exactement ce qu'aurait vu le tail_ext_emit en single-pass. Test
   bit-exact obligatoire.

5. **Recompute alternative** : au lieu de stocker, on peut juste
   refaire le sweep cross-off sur les heavy bands en pass 2.
   Coût additionnel = ~24 % CPU × 2 bands / 128 bands = +0.4 % CPU
   total. Trade-off RAM 42 KB vs +0.4 % CPU. La copie est plus simple.

---

## 7. Si le 2-pass V2 fail aussi

> **Cette section est obsolète depuis le 2026-04-29 soir** : le 2-pass V2
> n'a pas failli, il a livré −16.3 % à 1e17 α=2 cool (commit `ddc1786`).
> Plan B (`phi_tiny` O(1) Kim) reste valide comme refactor indépendant si
> on veut viser le facteur 5× vs primecount, mais il n'est plus
> conditionné à un échec du 2-pass.

Plan B : **`phi_tiny` O(1) Kim-style refactor** (gain estimé 5-10 %
wall, indépendant des bottlenecks Rayon). Effort ~600 LOC, repenser
S1, table de taille `pp = ∏primes[1..a] = 9 699 690` à a=8. C'est un
projet à part entière. Voir aussi le rapport agent du 2026-04-29 sur
primecount Sieve_pre_sieve.hpp et phi.cpp.

Plan C : accepter que **71 % efficacité Rayon à 1e17 α=2 est le
plafond** et focaliser sur d'autres axes (ex: README/docs/tests
qualité code).

---

## 8. Piste prioritaire post-Étape-A — band 127 rest_bulk_xoff

**Contexte (après commit `ddc1786`)** : à 1e17 α=2 cool, le bottleneck
n'est plus tail_ext_emit des bands 0+1 (déplacé vers le pass 2 nested
par_iter qui scale bien) mais **band 127 single-thread**. Profile post-
deferred :

```
band 127 (lo ∈ [97 830 123 720, 107 722 238 760]) : 14.4 s wall
  ├── rest_bulk_xoff      6.6 s   (46 %)  ← le nouveau dominant
  ├── tail_ext_emit mid   1.4 s   (10 %)
  ├── bi_main_xoff        ~3 s
  └── reste               ~3 s
band 126                  : 13.5 s wall
band 124-125              : 11-13 s wall
```

Les bands 120-127 sont **séquentielles sur leur thread Rayon** — chacune
prend ~13-14 s. Le `band_sweeps.collect()` doit attendre la dernière band
finie avant que pass 2 démarre. Ces ~14 s sont donc une borne stricte
sur le wall total : même si pass 2 prenait 0 s, on ne descendrait pas
sous 14 s + overhead = ~16 s. Actuellement on est à 38.9 s, soit
~10 s "cachés" (pass 2 effectif post-pass 1).

**Décomposition wall actuelle (1e17 α=2 cool, post-Étape A)** :

```
total wall                  38.9 s
  ├── pass 1 wall          ~16 s   (band 127 = bottleneck single-thread)
  ├── pass 2 wall          ~22 s   (heavies tail_ext spread sur 8 threads)
  └── resolve              < 1 s
```

**Ratio CPU/wall total** : 256 s / 38.9 s = 6.58× sur 8 threads = 82 %.
Pour passer à 90 % il faut combler ~5 s de wall ; pour 95 % il faut
combler ~7 s (reste = `band_127_wall_residuel + scheduling overhead`).

### 8.1 Sous-pistes à tester (par ROI estimé)

#### 8.A Overlap pass-1 / pass-2 via `rayon::join`

**Idée** : démarrer pass 2 sur les heavy bands AVANT que pass 1 ait
fini les light bands. Comme heavy bands finissent leur cross-off très
vite (~5 s sur leur thread, vs 14 s pour band 127), leurs `DeferredSeg`
sont prêts bien avant la fin du `collect()`.

**Pattern Rayon** :
```rust
let (light_results, heavy_results) = rayon::join(
    || (2..num_bands).into_par_iter().map(...).collect(), // pass 1 light
    || {
        // Heavy bands : faire leur pass 1 d'abord (capture seulement),
        // PUIS lancer pass 2 nested sur le pool. Tout ça pendant que
        // light bands tournent encore.
        let heavy_pass1 = (0..2).into_par_iter().map(...).collect();
        let pass2 = (0..2).into_par_iter()
            .map(|t| pass2_replay(&heavy_pass1[t]))
            .collect();
        (heavy_pass1, pass2)
    }
);
```

**Wall idéal** : `max(pass1_light_wall, pass1_heavy_wall + pass2_wall)`.
Avec pass1_light = 14 s (band 127), pass1_heavy = 5 s, pass2 = 7 s →
max(14, 12) = 14 s. **Gain projeté : 38.9 → ~20 s = −48 % wall** vs
post-Étape A, soit **~−57 %** vs baseline 46 s.

Mais : le pass 2 nested utilise le MEME pool Rayon que le pass 1 light.
Si on sature le pool avec 8 threads sur pass 1 light, le nested pass 2
trouvera 0 thread idle (= retour au problème Piste C). Il faut donc :

1. Soit séparer en deux pools (rayon::ThreadPoolBuilder, mais 2 pools
   se concurrencent pour les 8 cœurs physiques).
2. Soit ré-architecturer pass 2 pour qu'il REMPLACE le travail des heavy
   bands dans le par_iter du pass 1 (= heavy bands deviennent des "tâches
   différées" à exécuter par n'importe quel thread idle).

**Implémentation suggérée (option 2)** : utiliser `rayon::scope` ou un
crossbeam channel. Pendant le par_iter outer sur 128 bands, chaque heavy
band, après avoir fini son cross-off, push ses `DeferredSeg` dans un
channel. Un consumer pool prend des chunks (heavy_band_id, ei_range)
et les traite. Tous les threads peuvent participer à l'un ou l'autre.

**Effort** : ~80-150 LOC selon le pattern Rayon retenu. Risques :
- Rayon's lazy work-stealing peut faire le job sans channel explicite,
  mais il faut tester.
- L'ordre de finalisation des `BandStats` doit rester déterministe pour
  les tests bit-exact.

**Garde-fous** : tests `cargo test --release` doivent passer ; π(1e15)
α=2 et π(1e17) α=2 doivent matcher post-`ddc1786` exactement.

#### 8.B Paralléliser band 127 rest_bulk_xoff intra-band

**Idée** : la boucle ligne ~840 de `hard.rs` (current commit `ddc1786`,
search for `for k in 0..target_end { ... cross_off_pd_from_state ...`)
itère sur les primes bulk d'une band. Ces primes sont indépendantes
entre elles (chaque prime cross-off ses multiples sans interaction
avec les autres). On peut splitter cette boucle en `chunks(N)` et
lancer un par_iter dessus.

**Bloqueur** : la boucle est INSIDE le segment loop, qui lui-même est
INSIDE la band closure exécutée par un seul thread Rayon. Pour
paralléliser, il faut sortir le bulk cross-off de la band closure ET
le synchroniser avec le segment cursor (chaque prime maintient
`bulk_next_m[k]` / `bulk_next_j[k]` qui DÉPEND du segment courant).

Ça implique de restructurer la band closure en :
1. Phase A par segment : sieve fill + presieve (séquentiel)
2. Phase B par segment : par_iter sur les primes bulk (parallèle)
3. Phase C par segment : tail_ext + tail_p2 + tail_advance (séquentiel)

Avec 6.6 s de rest_bulk pour band 127 sur un seul thread, idéalement
sur 8 threads : 0.83 s. Gain projeté : ~5 s wall sur band 127 → wall
total ~33 s = **−15 %** vs post-Étape A. Modeste vs option 8.A.

**Bloqueur supplémentaire** : nested par_iter (band-level outer sature
toujours le pool). Sauf à n'activer ce parallélisme que sur band 127
spécifiquement (et 126, 125…), via un mode "heavy-bulk band". Mais
alors c'est exactement le pattern de l'option 8.A appliqué au bulk
au lieu du tail_ext.

**Effort** : ~150 LOC. **Verdict** : option 8.A domine 8.B, à attaquer
en premier. 8.B reste ouvert si 8.A ne suffit pas.

### 8.2 Procédure de bench pour la session

```bash
# Baseline post-ddc1786 à confirmer (cool, single trial)
./target/release/primerivat --dr-profile 1e17 --alpha 2

# Cible : ~33-35 s avec 8.A, ~33 s avec 8.B
# Critère pass : ≥ −10 % vs 38.9 s = ≤ 35 s
# Critère fail : ≥ +2 % à α=1 (sanity 1e15)

# Neutralité α=1 :
./target/release/primerivat --dr-profile 1e15

# A/B avec --no-deferred-tail-ext (vérifie que la nouvelle archi
# preserve aussi le path "no defer") :
./target/release/primerivat --dr-profile 1e17 --alpha 2 --no-deferred-tail-ext
```

**Mémoire 9300HF** : 1 trial cool, puis pause refroidissement. Le
fichier mémoire `feedback_bench_pace.md` formalise cette règle.

---

## 9. Piste secondaire — Étape C : heuristique généralisée

> Cette piste est **annexe** à 8 : la heuristique actuelle
> (`t < 2 && use_log_scale`) capture déjà le bottleneck dominant à
> 1e17 α=2. La généraliser ne libère pas de wall supplémentaire tant
> que le plafond band 127 n'est pas levé. Faire **après** §8 ou en
> parallèle si la session a du temps.

**Contexte** : la heuristique Étape A est codée en dur :

```rust
let is_heavy = |t: usize| -> bool {
    defer_enabled && use_log_scale && t < 2
};
```

C'est correct pour x ∈ [3e16, 1e18] α=2 sur 9300HF (les bands 0+1
concentrent 99 % du tail_ext_emit dans ce régime). Mais :

- À x très grand (10^19+), 3 ou 4 bands pourraient devenir "heavy".
- À x intermédiaire ou avec une autre topologie de bands (CPU avec plus
  de threads → plus de bands → distribution différente du tail_ext),
  la coupure `t < 2` peut être trop stricte ou trop large.

### 9.1 Heuristique proposée

Calculer pré-sweep la projection `n_ext_emit_predicted[t]` pour chaque
band, puis sélectionner les bands où ce nombre dépasse `k × mean`. La
projection peut s'inspirer de `total_clamp_count` qui est déjà calculé :

```rust
// Pré-sweep, pour chaque band t, compter les leaves ext-easy non-clampées
// projetées :
let n_ext_emit_predicted: Vec<i64> = (0..num_bands).map(|t| {
    let band_lo = band_bounds[t];
    let band_hi = band_bounds[t + 1].min(z + W30_SEG as u64);
    (far_easy_start..n_easy)
        .map(|ei| {
            // Pour chaque ei, compter les pl tels que x/(pb*pl) ∈ [band_lo, band_hi)
            // ET pl > x/(pb*p_{b-1}) (non-clamp).
            // C'est un partition_point sur primes[b..a].
            let bi = n_hard + ei;
            let b = bi + c + 1;
            if b >= a || b < 2 { return 0i64; }
            let pb = primes[b - 1] as u128;
            let pbm1 = primes[b - 2] as u128;
            let pl_max_band = if band_lo == 0 { primes[a-1] }
                              else { (x / (pb * band_lo as u128)) as u64 };
            let pl_min_band = if band_hi == 0 { primes[b] }
                              else { (x / (pb * (band_hi - 1) as u128)) as u64 + 1 };
            let pl_clamp = (x / (pb * pbm1)) as u64;
            let pl_max_eff = pl_max_band.min(pl_clamp);
            if pl_max_eff < pl_min_band { return 0i64; }
            let cnt_max = primes[b..a].partition_point(|&p| p <= pl_max_eff);
            let cnt_min = primes[b..a].partition_point(|&p| p < pl_min_band);
            (cnt_max - cnt_min) as i64
        })
        .sum()
}).collect();

let mean_pred: f64 = n_ext_emit_predicted.iter().sum::<i64>() as f64
    / num_bands as f64;
let k_threshold: f64 = 5.0; // ou 10.0 — à tuner par bench

let is_heavy = |t: usize| -> bool {
    defer_enabled && use_log_scale &&
    (n_ext_emit_predicted[t] as f64) > k_threshold * mean_pred
};
```

**Coût pré-sweep** : à 1e17 α=2, ~70 k ei × 128 bands × 2 partition_points
= ~18 M opérations × ~50 ns = ~0.9 s séquentiel. Doit être parallélisé
via `into_par_iter` sur les bands. Cible : < 100 ms wall.

**Calibrage `k`** : tester k ∈ {3, 5, 10} cool. À k=5 sur 9300HF 1e17 α=2,
la heuristique doit re-sélectionner exactement bands 0+1 (= 100M et 14M
ext_emitted, vs mean ~6M = 17× et 2.4× le mean → seul band 0 passe).
Hmm, k=5 ne suffirait pas pour band 1 (2.4× < 5). Tester k=2 d'abord.

### 9.2 Garde-fous

- **Test bit-exact** : après refactor, `cargo test --release` ; π à
  1e10/1e13/1e15/1e17 doit matcher post-`ddc1786`.
- **Test α=1 1e15** : `n_ext_emit_predicted` doit être nul ou très
  faible partout (puisque ext-easy non-clampé est rare à α=1) ; donc
  `is_heavy(t) = false` pour tout t. Path bypassé.
- **Test 13450HX 1e18** : régime α=1 avec 12 threads, bands différentes ;
  vérifier que la sélection reste vide (use_log_scale=false suffit déjà).

### 9.3 Effort estimé et ROI

- **Effort** : ~150 LOC (fonction de prédiction + threshold + tests).
- **ROI direct** : ~0-3 % wall à 1e17 α=2 (la heuristique courante
  capture déjà 99 % du bottleneck).
- **ROI indirect** : robustesse à des régimes/CPU futurs où la coupure
  `t < 2` deviendrait inadaptée. Évite le hard-coding.

---

## Annexe : commandes utiles

```bash
# Build + test
cargo build --release
cargo test --release

# Profile cool single-trial (laisser refroidir entre)
./target/release/primerivat --dr-profile 1e15           # α=1, ~1.9 s
./target/release/primerivat --dr-profile 1e17 --alpha 2 # α=2, post-ddc1786 ~38.9 s

# A/B avec / sans le 2-pass (Étape A) :
./target/release/primerivat --dr-profile 1e17 --alpha 2                          # deferred ON
./target/release/primerivat --dr-profile 1e17 --alpha 2 --no-deferred-tail-ext   # deferred OFF
# Env var équivalente : RIVAT3_NO_DEFERRED_TAIL_EXT=1 ./target/release/primerivat ...

# Per-band breakdown sous --dr-profile (shipped 75c054c)
# Top-10 bands + imbalance ratio dans le footer
# Note : à α=2 post-ddc1786, le tail_ext_ns reporté pour bands 0+1 est
# le CPU TIME cumulé des ei workers de pass 2 (pas le wall de la band).

# Comparer à primecount (WSL)
~/primecount/primecount -d 1e17 --time
```

Source primecount Kim disponible localement :
```
c:/Users/Kbda9/projet/3rivat3/primecount/src/   # Windows
~/primecount/                                    # WSL (mêmes sources)
```

Fichiers utiles pour les sessions §8/§9 :
- `hard.rs` (commit `ddc1786`) — pass 1 + pass 2 nested par_iter, voir
  les blocks `is_heavy(t)`, `DeferredSeg`, et la pass-2 replay.
- `segment.rs` — `count_primes_in_segment` et `bits_array()` (utilisés
  par pass 2).
- `parameters.rs` — `no_deferred_tail_ext()` et l'override.
- `Sieve.cpp` lignes 222-596 (cross_off_count switch-64-cases) pour la
  référence Kim si on veut paralléliser rest_bulk_xoff (§8.B).
- `S2_hard.cpp` (référence Kim) pour comparer la décomposition par phase.

---

## Mémoire principale

`~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_s2_hard_refactor.md`

Mise à jour 2026-04-29 (soir) avec : Étape A 2-pass POC shipped
(`ddc1786`), gain confirmé −16.3 % à 1e17 α=2 cool, neutralité α=1
prouvée par codepath. Les 3 commits avant ont closé Multi-template AND
C=8 + Piste C nested Rayon + Piste D b_ext_mult + phi-recursive
(`75c054c`, `fe5a03f`, `a1cf99a`, `7c4856a`).
