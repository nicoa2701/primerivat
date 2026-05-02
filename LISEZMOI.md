# primerivat

*[English version](README.md)*

`primerivat` est une implémentation Rust de la fonction de comptage des
nombres premiers `π(x)` basée sur l'algorithme de **Deléglise–Rivat**.

Le moteur actuel (`prime_pi_dr_meissel_v4`) est actif par défaut et calcule
π(x) exactement jusqu'à au moins `x = 1e18` sur du matériel grand public
(vérifié cross-CPU au commit `9e9162a`).

## Démarrage rapide

```bash
cargo build --release
cargo run --release -- 1e13        # π(1e13) = 346 065 536 839
cargo run --release -- 1e15        # π(1e15) = 29 844 570 422 669
```

Le binaire affiche un tag de démarrage avec le hash git court, le L3
détecté, et l'α sélectionné (ex. `[primerivat 9e9162a | L3=8Mo α=2]`),
permettant de comparer plusieurs builds côte à côte.

## Performances mesurées

Build release, multi-thread, α adaptatif. Trois machines de référence :

- **i5-9300H** — 4C/8T, 8 Mo L3, DDR4-2666 (Coffee Lake, 2019)
- **i5-13450HX** — 6P+4E (10 phys / 16T), 20 Mo L3, DDR5 (Raptor Lake, 2023)
- **Ryzen 7 9700X** — 8C/16T, 32 Mo L3, DDR5 (Zen 5, 2024)

| x | i5-9300H | i5-13450HX | Ryzen 9700X | π(x) |
|---|---:|---:|---:|---|
| `1e11` | 9 ms | 9 ms | 4 ms | 4 118 054 813 |
| `1e12` | 30 ms | 31 ms | 10 ms | 37 607 912 018 |
| `1e13` | 117 ms | 56 ms | 42 ms | 346 065 536 839 |
| `1e14` | 466 ms | 220 ms | 163 ms | 3 204 941 750 802 |
| `1e15` | 1,85 s | 0,81 s | 0,62 s | 29 844 570 422 669 |
| `1e16` | 8,74 s | 3,08 s | 2,29 s | 279 238 341 033 925 |
| `1e17` | 42,1 s α=2 | 14,4 s α=1 | 10,6 s α=1 | 2 623 557 157 654 233 |
| `3e17` | 135 s α=2 | 39,9 s α=1 | 20,3 s α=2 | 7 650 011 911 220 803 |
| `1e18` | 313 s α=2 | 97,0 s α=1 | 49,2 s α=2 | 24 739 954 287 740 860 |

Toutes les colonnes sont mesurées au commit `8a6d89b` (post-règle α
2-tier), sweep batch single-trial `1e11 1e12 … 1e18`. Le 9300H thermal-
throttle après ~10 s de charge soutenue, donc 1e17 ici (42,1 s α=2) est
mid-throttle ; le snapshot cool single-run post-Étape-A à la même
magnitude est de 38,9 s. 1e18 sur 9300H (313 s α=2) est lancé séparément
(~5+ min) — résultat 36 % en dessous du snapshot pré-Phase-1+2A de 488 s
et 75 % en dessous de la baseline `9e9162a` à 1266 s.

La colonne 9700X montre le nouveau régime α=2 qui se déclenche à partir
de 3e17 (règle adaptative ci-dessous) : 20,3 s à 3e17 (vs ~26 s en α=1)
et 49,2 s à 1e18 (vs ~72 s en α=1).

Gain cumulé vs la baseline pré-cascade au commit `9e9162a` sur les trois
CPUs (les gains du cascade sont universels — Phase 1+2A, popcount LUT-
240, single-pass merge, fold accumulators, band-split 16×, 2-pass
deferred-tail-ext pour le régime α=2) :

| x | i5-9300H | i5-13450HX | Ryzen 9700X |
|---|---:|---:|---:|
| 1e15 | 5,51 → 1,85 s (**−66 %**) | — → 0,81 s | — → 0,62 s |
| 1e17 α=2 | 160 → 42 s (**−74 %**) | — | — |
| 1e17 α=1 | — | 44,8 → 14,4 s (**−68 %**) | — → 10,6 s |
| 1e18 α=2 | 1266 → 313 s (**−75 %**) | — | — → 49,2 s |
| 1e18 α=1 | — | 301 → 97 s (**−68 %**) | — |

Les gains cumulés viennent du cascade S2_hard multi-commits (single-pass
deferred-leaf design, fold accumulators, band layout log-scale pour
α=2, clamp-leaf bulk pre-count, template pré-sieve `{7, 11}`, popcount
via LUT 240 bits, cross-off Kim-style 8-way déroulé dans `bi_main_xoff`
et `rest_plain_xoff`, 2-pass deferred-tail-ext, et la règle α adaptative
2-tier).

## Algorithme

Le moteur implémente la décomposition classique Meissel–Deléglise–Rivat :

```
π(x) = φ(x, a) + a − 1 − P2
     = (S1 + S2_hard) + a − 1 − P2
```

avec `a = π(y)`, `y = α·∛x`, et :

- **S1** — `Σ μ(m)·φ(x/m, c)` sur les `m` squarefree dont les facteurs premiers
  sont ≤ `y`, calculé par un DFS récursif avec `c = 5`.
- **S2_hard** — `−Σ_{b=c+1..b_max} Σ_m μ(m)·φ(x/(p_b·m), b−1)`, évalué via
  trois chemins spécialisés.
- **P2** — `Σ_{y < p ≤ √x} (π(x/p) − π(p) + 1)`, fusionné avec S2_hard dans
  un seul sweep parallèle.

### Les trois chemins de S2_hard

| Plage `bi` | Plage `p_b` | Chemin | Formule pour `φ(n, b−1)` |
|---|---|---|---|
| `0..n_hard` | ≤ √y | hard leaves | `phi_vec` + curseur popcount monotone |
| `n_hard..b_ext` | (√y, x¹ᐟ⁴] | `phi_easy` (`m = p_l`) | `phi_vec` + curseur popcount monotone |
| `b_ext..n_all` | (x¹ᐟ⁴, y] | `ext_easy` (`m = p_l`) | forme fermée `π(n) − (b−2)` (clamp ≥ 1) |

La forme fermée `ext_easy` `φ(n, b−1) = π(n) − (b−2)` est valide lorsque
`n ≥ p_{b−1}` et évite de maintenir `phi_vec` pour le gros des primes.

### α adaptatif

`y = α·∛x` avec α choisi selon l'ordre de grandeur de `x` **et le matériel**,
en deux tiers hardware :

- **Tier 9300H** (L3 < 16 Mo **et** ≤ 8 cœurs physiques, ex. laptops
  contraints par le cache) : α = 2,0 dès `x ≥ 3e16` (~41 % plus rapide
  à `1e17`).
- **Tier 9700X** (≥ 8 cœurs physiques **et** SMT pur, c.-à-d.
  `logical == 2 × physical`, ex. Ryzen 7 9700X 8C/16T ou tout
  Threadripper avec HT/SMT activé) : α = 2,0 dès `x ≥ 3e17` (~24 % plus
  rapide à `1e18`, ~28 % plus rapide à `5e17`).
- Tous les autres CPUs (ex. parts hybrides P+E Intel où `logical < 2 ×
  physical`, ou tout `x` sous le seuil) : α = 1,0.

Les deux seuils reflètent le fait que le tier desktop SMT-symétrique
(16 threads logiques, gros L3) absorbe les fenêtres de crible plus
larges d'α=1 jusqu'à ~1e17, après quoi les économies CPU algorithmiques
d'α=2 (−42 % à 1e18) dominent l'équilibre Rayon légèrement moins bon.

La sélection automatique peut être surchargée depuis la CLI via `-a <α>` ou
`--alpha <α>`. Plage acceptée :

- `x ≤ 1e15` → tout `α ∈ [1, 2]` (ex. `1,5`)
- `x > 1e15` → seulement `α ∈ {1, 2}`

Toute valeur hors bornes déclenche un avertissement et est ignorée : le
moteur retombe sur l'α sélectionné automatiquement.

> ⚠️ Seul `α ∈ {1.0, 2.0}` est sûr. Les valeurs intermédiaires (ex. α = 1,25)
> produisent des résultats faux pour certains `x` — cause non identifiée.

### Fallback pour les petits x

Si `a ≤ C = 5`, le driver bascule sur la baseline Lucy–Meissel
(`baseline::prime_pi`) car `phi_small_a` ne peut pas être utilisé.

## Parallélisme

Le sweep S2_hard + P2 est découpé en bandes Rayon disjointes. Schéma en deux
passes :

1. **Pass 1** — chaque bande calcule ses `phi_deltas[bi]` locaux et son
   nombre de primes.
2. **Prefix scan séquentiel** — transforme les deltas locaux en offsets
   globaux.
3. **Pass 2** — traite les leaves et les requêtes P2 par bande avec les bons
   offsets.

La stack Rayon est portée à 32 MiB au démarrage pour couvrir la profondeur de
récursion atteinte à `x ≥ 1e15`.

## Organisation du projet

```text
src/
├── bit.rs          # arbre de Fenwick (legacy S2 seulement ; la prod utilise prefix popcount)
├── segment.rs      # WheelSieve30 (crible segmenté wheel-mod-30) + curseur MonoCount
├── baseline/       # référence Lucy–Meissel ; utilisée comme fallback petits x
├── dr/
│   ├── mod.rs      # prime_pi_dr_meissel_v4 (actif) + variantes legacy
│   ├── hard.rs     # s2_hard_sieve_par — sweep parallèle S2_hard + P2 + ext_easy
│   ├── easy.rs     # partitions S2_easy expérimentales (inactives)
│   ├── ordinary.rs # régions de leaves ordinary exploratoires
│   ├── trivial.rs
│   └── types.rs    # DrContext, domaines, règles de frontière
├── phi.rs          # DFS s1_ordinary, phi_small_a (inclusion/exclusion c ≤ 5)
├── sieve.rs        # crible Lucy (seed_primes)
├── parameters.rs   # types partagés
├── lib.rs          # API publique
└── main.rs         # CLI
build.rs            # injecte GIT_HASH pour l'affichage à l'exécution
```

## API publique

```rust
rivat3::deleglise_rivat(x)                       // point d'entrée principal
rivat3::deleglise_rivat_with_threads(x, threads) // idem, avec budget de threads
rivat3::prime_pi(x)                              // baseline Lucy–Meissel
rivat3::prime_pi_with_threads(x, threads)
```

## CLI

```bash
# Calcul direct (le moteur DR v4 gère son propre pool Rayon)
cargo run --release -- 1e13

# Batch : plusieurs x dans un seul appel (tableau récapitulatif à la fin)
cargo run --release -- 1e11 1e12 1e13 1e14

# Surcharger l'α sélectionné automatiquement (forme courte ou longue)
cargo run --release -- 1e17 -a 2
cargo run --release -- 1e13 --alpha 1

# Profilage
cargo run --release -- --dr-profile 1e13  # timings par étape du moteur v4
cargo run --release -- --lucy-profile 1e13

# Sweep par puissance de 10 jusqu'à x_max (défaut 1e12)
cargo run --release -- --sweep 1e14
```

## Validation

```bash
cargo test                          # tests unitaires
target/release/primerivat 1e15      # doit afficher 29 844 570 422 669
```

Valeurs de référence vérifiées :

- π(1e13) = 346 065 536 839
- π(1e15) = 29 844 570 422 669
- π(1e16) = 279 238 341 033 925
- π(1e17) = 2 623 557 157 654 233
- π(1e18) = 24 739 954 287 740 860

## Notes d'implémentation

1. **Sens du sweep** — ascendant, de `lo_start` vers `z`. `phi_vec[bi]` stocke
   `φ(lo − 1, b − 1)` et est mis à jour à la fin de chaque fenêtre.
2. **Scan monotone des leaves** — pour un `bi` donné, les leaves arrivent par
   `n` croissant. Un curseur `MonoCount` garde le popcount cumulé en cours et
   chaque requête ne popcount que les mots u64 traversés depuis la leaf
   précédente. Remplace le `fill_prefix_counts` (balayage complet de ~2 185
   mots) qui s'exécutait une fois par bi avec leaves.
3. **Template pré-sieve `{7, 11}`** — `fill_presieved_7_11` tile un bitmap
   précalculé de 77 octets (couvrant `lcm(7, 11) · 30 = 2 310` entiers) dans
   le crible au début de chaque segment, en remplacement du fill-à-1 et des
   deux boucles de cross-off wheel-30. Le byte-copy séquentiel vectorise
   bien et évite les bit writes scattered du chemin générique.
4. **Crible bucket** — dans le cross-off de masse pour les primes `≥ x¹ᐟ⁴`,
   les primes tels que `p² > hi` sont skippés : toute composite dans
   `[lo, hi)` admet un facteur `≤ √hi`.
5. **Correction seed** — pour `lo < y`, les primes seed dans `[lo, hi)` sont
   crossés comme multiples d'eux-mêmes puis réinjectés via `seed_in_seg` /
   `seed_in_query`.
6. **Clamp `ext_easy`** — `φ(n, b−1)` est clampé à `≥ 1` quand
   `n < p_{b−1}`.
7. **Garde petits x** — `if a ≤ C { return baseline::prime_pi(x) }`.

## Référence

Deléglise & Rivat, *Computing π(x): the Meissel, Lehmer, Lagarias, Miller,
Odlyzko method*, Math. Comp. 65 (1996).

Une description mathématique autonome de l'algorithme est disponible dans :

- [ALGORITHM.md](ALGORITHM.md) — English
- [ALGORITHME.md](ALGORITHME.md) — français

## Licence

Licence BSD 2-Clause — voir [LICENSE](LICENSE).
