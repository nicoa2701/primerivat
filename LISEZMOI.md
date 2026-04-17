# primerivat

*[English version](README.md)*

`primerivat` est une implémentation Rust de la fonction de comptage des
nombres premiers `π(x)` basée sur l'algorithme de **Deléglise–Rivat**.

Le moteur actuel (`prime_pi_dr_meissel_v4`) est actif par défaut et calcule
π(x) exactement jusqu'à au moins `x = 1e17` sur du matériel grand public.

## Démarrage rapide

```bash
cargo build --release
cargo run --release -- 1e13        # π(1e13) = 346 065 536 839
cargo run --release -- 1e15        # π(1e15) = 29 844 570 422 669
```

Le binaire affiche le hash git court (`[dr-meissel4 <hash>]`) pour permettre
de comparer plusieurs builds côte à côte.

## Performances mesurées

Build release, multi-thread, α adaptatif. Deux machines de référence :

- **i5-9300H** — 4C/8T, 8 Mo L3, DDR4-2666 (Coffee Lake, 2019)
- **i5-13450HX** — 6P+4E / 12T, 20 Mo L3, DDR5 (Raptor Lake, 2023)

| x | i5-9300H | i5-13450HX | π(x) |
|---|---:|---:|---|
| `1e11` | 0,014 s | 0,012 s | 4 118 054 813 |
| `1e13` | 0,24 s  | 0,11 s  | 346 065 536 839 |
| `1e14` | 1,08 s  | 0,47 s  | 3 204 941 750 802 |
| `1e15` | 5,61 s  | 1,98 s  | 29 844 570 422 669 |
| `1e16` | 30,5 s  | 9,20 s  | 279 238 341 033 925 |
| `1e17` | 183 s (α=2) | 48,3 s (α=1) | 2 623 557 157 654 233 |
| `1e18` | —       | 303 s   | 24 739 954 287 740 860 |

Les chiffres i5-9300H reflètent le HEAD actuel (scan monotone des leaves +
pré-sieve `{7, 11}`). Ceux du i5-13450HX datent du commit précédent
(`faf8a77`) et n'ont pas encore été re-mesurés avec le pré-sieve. Gain
cumulé vs la baseline d'avant la session : **~−23 % de runtime entre 1e13
et 1e15**, réparti sur trois changements (init `phi_vec` fusionné, curseur
monotone en remplacement de `fill_prefix_counts`, template pré-sieve
`{7, 11}`).

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

`y = α·∛x` avec α choisi selon l'ordre de grandeur de `x` **et le matériel** :

- `x < 3e16` → α = 1,0 (overhead plus faible pour les petits `x`)
- `x ≥ 3e16` **et L3 < 16 Mo et ≤ 8 cœurs physiques** → α = 2,0
  (~41 % plus rapide à `1e17` sur les CPU contraints par le cache, grâce à
  moins de fenêtres de crible)
- `x ≥ 3e16` sur un CPU plus large (L3 ≥ 16 Mo, ex. desktop / HX) → α = 1,0
  (le L3 plus gros absorbe les fenêtres supplémentaires, α = 1 devient plus
  rapide)

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
├── bit.rs          # arbre de Fenwick (utilisé implicitement via count_primes_upto_int)
├── segment.rs      # WheelSieve30 (crible segmenté wheel-mod-30), primes_up_to
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
# Calcul direct (le moteur DR v4 gère son propre parallélisme Rayon)
cargo run --release -- 1e13

# Surcharger l'α sélectionné automatiquement (forme courte ou longue)
cargo run --release -- 1e17 -a 2
cargo run --release -- 1e13 --alpha 1

# Profilage
cargo run --release -- --profile 1e11     # profil de phase baseline
cargo run --release -- --dr-profile 1e13  # profil de phase DR
cargo run --release -- --lucy-profile 1e13

# Sweep multi-x
cargo run --release -- --sweep 1e12
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
