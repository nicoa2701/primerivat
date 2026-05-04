# Handoff session 2026-05-04 - primerivat

> Lecture obligatoire au debut de la prochaine session.
> Etat de cloture: working tree clean avant mise a jour de ce fichier; ce fichier
> doit etre le seul changement a committer si on veut figer la pause.

---

## 1. Etat Git

Derniers commits importants de la session:

- `df036e1 perf(s2-hard): add opt-in rest-bulk kernel dispatch`
- `2da4580 perf(s2-hard): allow late deferred tail-ext scheduling`
- `506003e perf(s2-hard): tune deferred tail-ext band count`
- `6eeccfc profile(s2-hard): report work-item wall time`
- `a5f7337 profile(s2-hard): add work-item breakdown`
- `139aae5 fix(s2-hard): retain clamp-only split tail-ext items`
- `04d51ac perf(s2-hard): schedule split tail-ext as work items`
- `03d5eff perf(s2-hard): overlap split tail-ext with sweep`
- `9f3822c perf(s2-hard): add opt-in split tail-ext mode`
- `d284486 probe(s2-hard): add split tail-ext check`
- `232a852 probe(dr): add parallel segmented P2 check`
- `32b8e26 probe(dr): add standalone streaming P2 check`
- `4b3ae8c profile(s2-hard): add rest bulk fine breakdown`

Apres demande utilisateur en fin de pause, le chemin par defaut a ete cale sur
le meilleur temps de cette session:

- `RIVAT3_SUBDIVIDE_HEAVY` default: `3`
- `RIVAT3_SUBDIVIDE_REST_BULK` default: `20`
- `RIVAT3_HEAVY_CHUNKS` default: `4`
- `RIVAT3_DEFERRED_TAIL_EXT_BANDS` default: `4`
- `RIVAT3_DEFERRED_TAIL_EXT_ORDER` reste off par defaut
- `RIVAT3_REST_BULK_KERNEL` reste `scalar` par defaut
- `RIVAT3_TAIL_EXT_SPLIT` reste off par defaut

Les autres changements restent opt-in/profiling.

Tests locaux passes apres les changements:

```text
cargo test --release
86 tests OK
```

---

## 2. Baseline 9700X observee

Cible de bench: Ubuntu 9700X, `x = 1e18`, commande:

```bash
./target/release/primerivat --dr-profile 1000000000000000000
```

Baselines recentes varient selon run autour de:

- total: `37.0s` a `39.0s`
- `step3 S2_hard`: `34.6s` a `36.4s`
- resultat correct:
  `pi(1e18) = 24 739 954 287 740 860`

Le bruit de scheduling reste non negligeable; trancher sur plusieurs runs quand
une variation est inferieure a environ 2%.

Meilleur profil retenu comme defaut: combinaison equivalente a
`SUBDIVIDE_HEAVY=3`, `SUBDIVIDE_REST_BULK=20`, `HEAVY_CHUNKS=4`,
`DEFERRED_TAIL_EXT_BANDS=4`, sans `late`. Meilleur run observe:

- total: `37.004s`
- step3: `34.585s`

---

## 3. Piste tail_ext split

Objectif: sortir `tail_ext` du sweep pour eviter les bandes basses lourdes.

Knobs ajoutes:

- `RIVAT3_TAIL_EXT_SPLIT=1`
- `RIVAT3_TAIL_EXT_SPLIT_CHECK=1`
- `RIVAT3_TAIL_EXT_EI_CHUNK=N`

Resultat:

- Mathematiquement correct apres fix `139aae5`.
- Contribution correcte a 1e18:
  `3253087321603626`
- Mais trop lent: le split devient la branche critique.
- Exemples 9700X:
  - gros split overlappe: environ `39s`, split autour de `36s`
  - work-items split: correct mais toujours environ `39s`

Conclusion: garder comme probe/correctness, ne pas utiliser comme piste perf
principale pour l'instant.

---

## 4. Profil work-items

Knobs ajoutes:

- `RIVAT3_WORK_ITEM_PROFILE=1`
- affiche maintenant `wall ms` et `total ms` par work-item.

Commande utile:

```bash
RIVAT3_WORK_ITEM_PROFILE=1 \
  ./target/release/primerivat --dr-profile 1000000000000000000
```

Ce profil a montre:

- Les bandes basses `1/2/3` etaient bien des queues wall quand elles sont
  lancees trop tot ou sans deferred favorable.
- Les bandes hautes `245..255` restent la queue quand on ne subdivise pas
  `rest_bulk`.

---

## 5. Deferred tail_ext bands / late scheduling

Knobs ajoutes:

- `RIVAT3_DEFERRED_TAIL_EXT_BANDS=N`
- `RIVAT3_DEFERRED_TAIL_EXT_ORDER=late`

Resultat:

- `RIVAT3_DEFERRED_TAIL_EXT_BANDS=4` inclut les bandes `1/2/3`.
- `late` reduit fortement le wall des bandes basses:
  - band 2: environ `17s wall` -> environ `2.3s wall`
  - band 1/3: environ `10s wall` -> environ `2.3-2.5s wall`
- Mais le total ne gagne pas stablement car la queue se deplace vers les
  bandes hautes/rest_bulk.

Conclusion: utile pour diagnostic; pas une solution seule.

---

## 6. Subdivision rest_bulk

Knobs existants/testes:

- `RIVAT3_SUBDIVIDE_REST_BULK=N`
- `RIVAT3_HEAVY_CHUNKS=K`
- souvent combine avec:
  - `RIVAT3_DEFERRED_TAIL_EXT_BANDS=4`
  - `RIVAT3_DEFERRED_TAIL_EXT_ORDER=late`

Mesures notables 9700X 1e18:

- `REST_BULK=10, CHUNKS=4`: environ `38.3s`
- `REST_BULK=20, CHUNKS=4, late`: environ `38.5s`
- `REST_BULK=32, CHUNKS=4, late`: environ `38.9s`
- `REST_BULK=16, CHUNKS=6, late`: environ `38.8s`
- `REST_BULK=16, CHUNKS=8, late`: environ `40.4s`

Profil:

- Trop peu de subdivision laisse une queue vers `238/239`.
- Trop de subdivision augmente l'overhead et regresse.

Conclusion: le scheduling seul ne donne pas de gain stable. Stopper cette piste
tant qu'on n'a pas optimise le noyau `rest_bulk_xoff`.

---

## 7. Rest_bulk fine profile et kernel dispatch

Knobs ajoutes:

- `RIVAT3_REST_BULK_PROFILE=1`
- `RIVAT3_REST_BULK_KERNEL=scalar|large|wide|unrolled`

Commandes utiles:

```bash
RIVAT3_REST_BULK_PROFILE=1 \
RIVAT3_REST_BULK_KERNEL=large \
  ./target/release/primerivat --dr-profile 1000000000000000000
```

```bash
RIVAT3_REST_BULK_PROFILE=1 \
RIVAT3_REST_BULK_KERNEL=wide \
  ./target/release/primerivat --dr-profile 1000000000000000000
```

```bash
RIVAT3_REST_BULK_PROFILE=1 \
RIVAT3_REST_BULK_KERNEL=unrolled \
  ./target/release/primerivat --dr-profile 1000000000000000000
```

Resultats 9700X:

- `large`: total `44.290s`, `rest_bulk xoff 271.8s`
- `wide`: total `48.749s`, `rest_bulk xoff 349.2s`
- `unrolled`: total `59.887s`, `rest_bulk xoff 533.8s`

Conclusion nette:

- `cross_off_pd_from_state_unrolled` est mauvais pour `rest_bulk`.
- Le kernel actuel `scalar` reste le meilleur.
- Les gros volumes sont dans:
  - `p<=W/4`: 8.3B appels
  - `p<=W/2`: 9.4B appels
  - `p<=W`: 13.2B appels
- Optimiser seulement `p>W` ne peut presque rien donner.

---

## 8. Prochaine tranche recommandee

Ne pas continuer le scheduling pour l'instant.

Piste recommandee:

1. Inspecter `WheelSieve30::cross_off_pd_from_state`.
2. Ajouter un kernel opt-in `state8`:
   - derouler une roue de 8 pas,
   - garder l'ecriture `u64` actuelle,
   - eviter la vue `u8` du kernel Kim unrolled,
   - cibler les bins volumineux `p<=W/4`, `p<=W/2`, `p<=W`.
3. Tester via:

```bash
RIVAT3_REST_BULK_PROFILE=1 \
RIVAT3_REST_BULK_KERNEL=state8 \
  ./target/release/primerivat --dr-profile 1000000000000000000
```

Si `state8` regresse aussi, il faudra chercher plus bas niveau:

- reduire les dependances de boucle (`m`, `group`, `j`),
- specialiser par `p` bin,
- ou revoir la representation des multiples actifs.

---

## 9. Rappels pratiques

- L'utilisateur prefere faire les gros tests sur le 9700X Ubuntu pour eviter le
  throttling thermique local.
- Ne plus inclure `cargo build --release` dans les commandes de bench sauf si
  le code vient d'etre modifie.
- Toujours garder les pistes risquées opt-in.
- Ne pas promouvoir une option si le gain est dans le bruit.

---

## 10. Commandes de reprise

Verifier etat:

```bash
git status --short
git log --oneline -5
```

Baseline simple:

```bash
./target/release/primerivat --dr-profile 1000000000000000000
```

Profil rest_bulk scalar:

```bash
RIVAT3_REST_BULK_PROFILE=1 \
  ./target/release/primerivat --dr-profile 1000000000000000000
```
