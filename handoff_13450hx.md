# Reprise sur i5-13450HX — note de passation

Écrit depuis le i5-9300HF le 2026-04-20, après clôture du cascade S2_hard
(6 commits, de `07f1741` à `b3089b4`). Le but de cette session : attaquer
les leviers α=1 qui bénéficient aux **deux** CPUs, sur la machine où
l'itération est la plus rapide.

---

## Où on en est

**Branche** : `main`, working tree propre au moment de l'écriture. Deux
commits locaux en avance sur `origin/main` (`25746de` profile finer-grained,
`b3089b4` Piste 1 log-scale). À jour par `git pull` après `git push` depuis
l'autre machine, sinon vérifie `git log --oneline -10`.

**Ce qui vient d'être fait (cascade S2_hard, branche DR-v4) :**
1. `07f1741` — single-pass Pass 1/Pass 2 merge
2. `c38b297` — fold deferred-leaf records dans band-local accumulators
3. `9775efb` + `7e7b17d` — band-split 16× + flag `-b/--band-mult`
4. `f10da60` — **Piste 3** : skip bulk-clamp ext-easy leaves (α=2 only)
5. `25746de` — HardProfile finer-grained (8 phases + 5 counters)
6. `b3089b4` — **Piste 1** : log-scale bands à α=2 pour Rayon rebalancing

**Ce que cette CPU voit concrètement** : 139 s vs 140 s à 1e18 pré/post
cascade, parce que le 13450HX reste α=1 sur toute la plage utile (20 Mo
de L3 suffisent). Les Pistes 1 et 3 gardent explicitement α=2
(`total_clamp_count > 0` et `ext_clamped > 0` respectivement) et ne se
déclenchent donc jamais ici. C'est normal, pas un bug.

---

## La question centrale pour cette session

Le cross-off pèse **68–81 % du CPU** dans S2_hard, indépendamment de α.
C'est le seul chemin commun aux deux CPUs. Tous les leviers restants
passent par là.

## Leviers ouverts, par ordre de priorité

1. **SIMD cross-off** (AVX2 / AVX-512) sur `WheelSieve30` — le gros morceau.
   Gain estimé 5-15 % wall, gros refactor. Cible : `rest_bulk` (30 % du
   CPU à 1e17 α=2, probablement similaire à α=1 sur cette CPU — à
   confirmer profile).
2. **Tuning cadence bulk** — plus local que SIMD, regarde si la
   granularité du bulk-batch peut être retunée sans vectoriser.
3. **Wheel-210** (absorbe le prime 7) — ~5 % de cross-offs en moins,
   refactor lourd de `WheelSieve30`.
4. **`tail_ext_emit` gather** — à 61 ns/leaf (mesuré 9300HF α=2), SIMD
   gather sur `prefix_counts` est la seule piste. À α=1 le nombre de
   leaves est plus faible, donc impact moindre — à quantifier profile.

## Premier réflexe en arrivant

Avant de coder quoi que ce soit, établir la baseline **spécifique à
cette CPU** :

```bash
cargo build --release
target/release/primerivat --dr-profile 1e17
target/release/primerivat --dr-profile 1e18
```

Ça donnera la répartition CPU de référence en α=1 sur 12 threads /
20 Mo L3. Le profile sur 9300HF α=2 (à titre de comparaison, pour voir
si la structure est la même) :

| phase | 1e17 α=2 / 9300HF |
|---|---:|
| bi main (leaf+xoff) | 25.4 % |
| rest plain xoff | 11.9 % |
| **rest bulk xoff** | **30.7 %** |
| tail ext-easy emit | 27.3 % |
| tail advance | 4.3 % |
| sweep+resolve+P2 | < 1 % |

À α=1 on s'attend à `ext_clamped=0` (compteur) et `rest_bulk` plus
petit (moins de primes bulk). L'équilibre phase par phase sera
probablement différent — c'est ce profile qui devra orienter la suite,
pas celui du 9300HF.

## Garde-fou : validation croisée

Toute optim α=1 touche aussi le chemin α=2 (mêmes boucles cross-off
dans `WheelSieve30`). Protocole :
- 46 tests unitaires : `cargo test --release`
- π(10^k) pour k = 6…18 en auto-α
- Une fois sur cette CPU, une fois sur le 9300HF (forcé `--alpha 2` à
  1e17/1e18 pour exercer le chemin chaud) avant de merger.

## Fermeture définitive

- Piste 2 (Dusart affinée) : empiriquement inutile avec le layout de
  bandes actuel — bucket stored toujours vide. Ne pas rouvrir.

## Quirk Windows

`build.rs` observe `.git/refs/heads` (dossier) ; Windows ne propage pas
les mtime des fichiers enfants, donc `GIT_HASH` affiché peut retarder
d'un commit tant qu'aucun source ne change. Bénin. Fix propre : watcher
`.git/refs/heads/main` directement. Pas urgent.

## Suivi à jour

La mémoire principale est
`~/.claude/projects/c--Users-Kbda9-projet-primerivat/memory/project_s2_hard_refactor.md`.
Elle a les chiffres complets du cascade et les observations CPU-specificity.
Ne pas perdre de vue : **cette session est censée produire un gain qui
apparaît sur les deux CPUs**, sinon c'est une autre optim α=1-only.
