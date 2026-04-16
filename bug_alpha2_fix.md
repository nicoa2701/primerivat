# Bug α=2 : formule φ(n, b−1) invalide pour petit n

## Symptôme

Avec `ALPHA = 2.0`, `π(10⁶)` donnait **74 727** au lieu de **78 498** (déficit de 3 771).  
Avec `ALPHA = 1.0`, le résultat était correct.

---

## Diagnostic

### Vérifications brute-force

| Quantité | Valeur attendue | Valeur calculée | Statut |
|----------|-----------------|-----------------|--------|
| φ(x, a) | 102 753 | 98 982 | ✗ |
| S1 | 66 089 | 66 089 | ✓ |
| P2 | 24 300 | 24 300 | ✓ |
| S2 = φ(x,a) − S1 | **36 664** | **32 893** | ✗ |

La formule `φ(x, a) = S1 + S2` identifie S2 comme la seule source d'erreur.

### Décomposition de S2

| Composant | Attendu (BF) | Calculé | Statut |
|-----------|-------------|---------|--------|
| hard | +9 665 | +9 665 | ✓ |
| phi_easy | 18 847 | 18 847 | ✓ |
| ext_easy | **8 152** | **4 381** | ✗ |
| **Total** | **36 664** | **32 893** | ✗ |

La somme brute-force directe sur les 820 paires `Σ_{c<l<b≤a} φ(⌊x/(p_l·p_b)⌋, l−1)` confirme que la valeur attendue est bien 36 664 — c'est la formule qui est correcte, et le composant `ext_easy` qui est faux.

---

## Cause racine

### La formule utilisée dans ext_easy

```rust
let phi_n = pi_n - (b as i64 - 2);
```

Cette formule implémente : `φ(n, b−1) = π(n) − (b−2)`

### Condition de validité

La formule `φ(n, k) = π(n) − k + 1` est valable **uniquement quand `p_k ≤ n`**, c'est-à-dire quand le k-ème nombre premier est inférieur ou égal à n.

**Démonstration :** quand `n < p_{k+1}²`, tout composé `m ≤ n` a forcément un facteur premier `≤ √n < p_{k+1}`, donc parmi `{p_1, …, p_k}` → éliminé. Les survivants sont : **1** et les premiers `p_{k+1}, …, p_{π(n)}`. Soit `φ(n, k) = 1 + π(n) − k`.

Mais si `n < p_k` : alors `π(n) < k`, et la formule donne `π(n) − k + 1 ≤ 0`. La vraie valeur est `φ(n, k) = 1` (seul 1 est copremier avec tous les premiers `≤ n` qui sont tous parmi `{p_1,…,p_k}`).

### Pourquoi α=1 ne déclenchait pas le bug

Avec **α=1** : `y = ∛x`, `p_a = x^{1/3}`.  
Chaque feuille ext_easy a `n = x/(p_b · p_l)`. Le minimum est atteint pour `p_l = p_a` :

```
n_min = x / (p_b · p_a) ≥ x / p_a² = x / x^{2/3} = x^{1/3} = p_a ≥ p_{b−1}
```

La condition `p_{b−1} ≤ n` est **toujours satisfaite**. ✓

### Pourquoi α=2 casse la condition

Avec **α=2** : `y = 2·∛x`, `p_a = 2·x^{1/3}`.  
Le minimum devient :

```
n_min = x / (p_b · p_a) = x / (p_b · 2·x^{1/3})
```

Pour les grands `b` (où `p_b` est proche de `p_a`), `n_min` peut être bien inférieur à `p_{b−1}`.

**Exemple concret** avec x = 10⁶ (α=2 → y=200, a=46) :

| bi | b | p_b | p_{b−1} | n_min = x/(p_b·p_a) | Condition |
|----|---|-----|---------|---------------------|-----------|
| 15 | 21 | 73 | 71 | 10⁶/(73·199) = 68 | 71 > 68 ✗ |
| 19 | 25 | 97 | 89 | 10⁶/(97·199) = 51 | 89 > 51 ✗ |
| 39 | 45 | 197 | 193 | 10⁶/(197·199) = 25 | 193 > 25 ✗ |

Pour la paire (bi=39, p_l=199) : `n = 25`, `π(25) = 9`.  
- Formule : `9 − 43 = −34` ← **faux**  
- Valeur correcte : `φ(25, 44) = 1` ← car `p_{44} = 193 > 25`

---

## Le correctif

**Fichier :** [`src/dr/hard.rs`](src/dr/hard.rs), dans la boucle `ext_easy` de `s2_hard_sieve_par`.

```rust
// Avant (invalide quand n < p_{b−1}) :
let phi_n = pi_n - (b as i64 - 2);

// Après :
// φ(n, b-1) = π(n) − (b−2) tient seulement si n ≥ p_{b−1}.
// Quand n < p_{b−1} (π(n) < b−1), φ(n, b−1) = 1.
let phi_n = (pi_n - (b as i64 - 2)).max(1);
```

**Justification :** quand `π(n) < b−1`, toutes les valeurs `1 < m ≤ n` ont un facteur premier parmi `{p_1,…,p_{b−1}}` (puisque tous les premiers `≤ n` sont dans cet ensemble). Seul 1 est copremier → `φ(n, b−1) = 1`.

---

## Résultats après correction

| x | Résultat | Statut |
|---|----------|--------|
| 10⁶ | 78 498 | ✓ |
| 10⁷ | 664 579 | ✓ |
| 10¹³ | 346 065 536 839 | ✓ |
| 10¹⁵ | 29 844 570 422 669 | ✓ |

Tous les tests passent : `cargo test` → ok.
