# primerivat — Note mathématique complète

Cette note décrit l'algorithme implémenté dans `prime_pi_dr_meissel_v4`
(`src/dr/mod.rs`). L'objectif est de calculer exactement

$$
\pi(x) = \lvert\{p \le x : p \text{ premier}\}\rvert
$$

pour $x$ jusqu'à $\sim 10^{18}$, en temps $\mathcal{O}(x^{2/3}/\log^2 x)$.

---

## 1. Fonction de Legendre $\varphi$

On note $p_1 = 2 < p_2 = 3 < p_3 = 5 < \dots$ la suite des nombres premiers.

$$
\varphi(x, a) = \lvert\{n \le x : n = 1 \text{ ou aucun facteur premier de } n
                \text{ n'est parmi } p_1,\dots,p_a\}\rvert
$$

C'est le nombre d'entiers dans $[1, x]$ qui sont **coprimes** aux $a$
premiers premiers.

**Propriétés élémentaires :**

1. $\varphi(x, 0) = \lfloor x \rfloor$
2. $\varphi(x, a) = \varphi(x, a-1) - \varphi(\lfloor x/p_a \rfloor, a-1)$
   (inclusion-exclusion sur la divisibilité par $p_a$)
3. Si $p_a \le \sqrt{x}$, alors $\varphi(x, a) \ge 1$ (le chiffre $1$ compte toujours)
4. Pour $p_a > x$ : $\varphi(x, a) = 1$ (seul $1$ reste)

**Cas fermé utile.** Pour $p_a \le x < p_{a+1}^2$, on a :

$$
\varphi(x, a) = 1 + \pi(x) - a
$$

**Preuve.** Sous cette hypothèse, tout entier composé $m \le x$ admet un
facteur premier $\le \sqrt{x} < p_{a+1}$, donc parmi $\{p_1, \dots, p_a\}$,
donc il est éliminé. Les survivants sont $\{1\} \cup \{\text{premiers dans } (p_a, x]\}$, d'où le comptage. $\square$

**Corollaire.** Si $p_a > x$ (donc $\pi(x) < a$), on a aussi $\varphi(x, a) = 1$
(seul $1$ reste). C'est le cas limite géré par `.max(1)` dans le code.

---

## 2. Formule de Meissel

Choisissons $a = \pi(y)$ avec $y = \alpha \cdot x^{1/3}$ pour un $\alpha \ge 1$.
La formule de Meissel s'écrit :

$$
\pi(x) = \varphi(x, a) + a - 1 - P_2(x, a)
$$

avec

$$
P_2(x, a) = \lvert\{(p, q) : p, q \text{ premiers}, y < p \le q, pq \le x\}\rvert.
$$

Chaque paire $(p, q)$ est un nombre composé $pq \le x$ dont les deux facteurs
sont strictement plus grands que $y$. $P_2$ se réécrit :

$$
P_2(x, a) = \sum_{y < p \le \sqrt{x}} \bigl(\pi(x/p) - \pi(p) + 1\bigr).
$$

**Pourquoi $y = \alpha x^{1/3}$ ?** Avec ce choix, un produit $pqr$ de trois
premiers tous $> y$ satisferait $pqr > y^3 = \alpha^3 x \ge x$. Donc pas de $P_3$,
$P_4$, etc. La formule s'arrête à $P_2$.

---

## 3. Décomposition $\varphi(x, a) = S_1 + S_2$

On applique récursivement la relation $\varphi(x, a) = \varphi(x, a-1)
- \varphi(\lfloor x/p_a \rfloor, a-1)$ jusqu'à atteindre un petit indice
$c$ (dans ce code, $c = 5$, donc on bascule sur les premiers $\{2,3,5,7,11\}$).
Le résultat est une somme sur les **produits squarefree** $m$ de premiers
$> p_c$ et $\le y$ :

$$
\varphi(x, a) = \sum_{\substack{m \text{ squarefree} \\ p \mid m \Rightarrow p_c < p \le y}}
                \mu(m) \cdot \varphi(\lfloor x/m \rfloor, c).
$$

**Split $S_1 + S_2$.** On partitionne les $m$ selon que leur **plus grand
facteur premier** $P^+(m)$ se trouve dans la zone « hard » ou non :

- $S_1$ : $m$ tel que tous les facteurs premiers sont $\le y$ ET
  $\lfloor x/m \rfloor > z = x/y$ (hm en pratique : $m$ de petite taille).
- $S_2$ : le complément.

Dans l'implémentation, on regroupe plutôt par **plus grand facteur** $p_b$ :

$$
S_2 = -\sum_{b=c+1}^{a} \sum_{m'} \mu(m') \cdot \varphi\bigl(\lfloor x/(p_b m') \rfloor, b-1\bigr)
$$

où $m'$ parcourt les squarefrees dont tous les facteurs premiers sont dans
$(p_{b-1}, y]$ (en fait $> p_b$ pour éviter le double-comptage — avec le signe
donné par $\mu(m p_b) = -\mu(m)$ qui est absorbé dans le signe global).

Le $S_1$ traite les $m$ « petits » via un DFS récursif sur les squarefrees
produits de premiers $\le y$ (cf. `phi::s1_ordinary`).

---

## 4. Classification des leaves dans $S_2$

On indexe $p_b$ par $bi = b - c - 1$ (donc $bi \ge 0$).

| Plage de $bi$              | $p_b$ dans              | Nom          | Technique pour $\varphi(n, b-1)$       |
|---------------------------|------------------------|--------------|--------------------------------------|
| $[0, n_{\text{hard}})$    | $\le \sqrt{y}$         | **hard**     | $\varphi_{vec}[bi]$ + sieve popcount |
| $[n_{\text{hard}}, b_{\text{ext}})$ | $(\sqrt{y}, x^{1/4}]$  | **phi_easy** | $\varphi_{vec}[bi]$ + sieve popcount |
| $[b_{\text{ext}}, n_{\text{all}})$ | $(x^{1/4}, y]$         | **ext_easy** | formule fermée $\pi(n) - (b-2)$ (avec clamp) |

**Pourquoi trois chemins ?**

### 4.1 hard leaves ($p_b \le \sqrt{y}$)

Quand $p_b$ est petit, on peut avoir $m' = p_{l_1} p_{l_2} \dots$ multi-premier
(le produit total reste $\le y$). On énumère tous ces produits squarefree via
une DFS bornée (`enumerate_hard_leaves`), puis on évalue $\varphi(n, b-1)$
au passage du sweep.

### 4.2 phi_easy leaves ($\sqrt{y} < p_b \le x^{1/4}$)

Ici, tout $m' = p_{l_1} p_{l_2}$ à deux facteurs satisfait $m' > y$ (car
$p_{l_1}, p_{l_2} > p_b > \sqrt{y}$). Donc seul $m' = p_l$ (un seul premier,
$l > b$) est valide. C'est pourquoi on parle de « easy » : un seul $l$ par
$(b, \text{pair})$.

On évalue toujours via $\varphi_{vec}[bi]$ (maintenu par le sweep) + un
popcount du sieve entre $lo$ et $n$.

### 4.3 ext_easy leaves ($p_b > x^{1/4}$)

Même structure que phi_easy (un seul $p_l$). Mais ici on a la borne
géométrique :

$$
n = \frac{x}{p_b p_l} < \frac{x}{p_b^2} < \frac{x}{x^{1/2}} = \sqrt{x}.
$$

De plus, pour tout couple valide, $p_{b-1} \le n < p_b^2$ **quand $\alpha = 1$**
(voir §5). Dans ce cas, on peut remplacer la coûteuse formule avec
$\varphi_{vec}$ par la formule fermée :

$$
\varphi(n, b-1) = \pi(n) - (b - 2).
$$

Cela évite de maintenir $\varphi_{vec}$ pour le gros des primes (ceux au-dessus
de $x^{1/4}$ sont nombreux).

---

## 5. Validité de $\varphi(n, b-1) = \pi(n) - (b-2)$

On applique la formule du §1 avec $a \leftarrow b-1$ :

$$
\varphi(n, b-1) = \pi(n) - (b-1) + 1 = \pi(n) - (b-2)
$$

**valide lorsque** $p_{b-1} \le n < p_b^2$.

### 5.1 Borne supérieure $n < p_b^2$

Pour ext_easy : $p_b > x^{1/4}$ donc $p_b^4 > x$ et $x/p_b^3 < p_b$. Comme
$p_l > p_b$, on a $x/(p_b p_l) < x/p_b^2 = p_b^2 / (p_b^2/p_b) \dots$
Plus simplement : $n = x/(p_b p_l) \le x/(p_b \cdot p_b) = x/p_b^2$, et on veut
$n < p_b^2$, soit $x/p_b^2 < p_b^2$, soit $x < p_b^4$, toujours vrai. $\square$

### 5.2 Borne inférieure $p_{b-1} \le n$

$n_{\min} = x/(p_b p_a)$ (atteint à $p_l = p_a \approx y$). Condition :

$$
p_{b-1} \le \frac{x}{p_b p_a} \iff p_{b-1} \cdot p_b \cdot p_a \le x.
$$

Prenons le cas défavorable $p_{b-1} \approx p_b \approx p_a \approx y =
\alpha x^{1/3}$ :

$$
p_{b-1} p_b p_a \approx y^3 = \alpha^3 x.
$$

**Cette condition tient pour $\alpha^3 \le 1$, donc $\alpha \le 1$.**
Pour $\alpha > 1$, il existe des leaves où $p_{b-1} > n$, et la formule
fermée est invalide.

### 5.3 Correction `.max(1)` (fix du commit `3017aa2`)

Quand $p_{b-1} > n$ (équivalent à $\pi(n) < b-1$), la formule fermée donne
$\pi(n) - (b-2) \le 0$, ce qui est absurde. La vraie valeur est :

$$
\varphi(n, b-1) = 1
$$

(car tous les premiers $\le n$ sont déjà dans $\{p_1, \dots, p_{b-1}\}$ ;
seul $1$ survit). D'où le clamp `(pi_n - (b as i64 - 2)).max(1)` dans
`hard::s2_hard_sieve_par`.

---

## 6. Rôle du paramètre $\alpha$

Plus $\alpha$ grand, plus $y$ grand :
- **Moins de fenêtres de sieve** dans le sweep (taille $z = x/y$ diminue avec
  $\alpha$) : gain environ $1/\alpha$.
- **Plus de primes seed** ($a = \pi(y)$ grandit avec $\alpha$) : coût par
  fenêtre augmente, et allocation seed_primes plus grande.
- **Plus de leaves ext_easy** : la zone $(x^{1/4}, y]$ s'élargit.

Le compromis optimal dépend de $x$. Mesures sur i5-9300H :

| $x$ | Meilleur $\alpha$ |
|-----|-------------------|
| $\le 10^{15}$ | $\alpha = 1$ (overhead seed_primes dominerait avec $\alpha = 2$) |
| $10^{16}$     | $\approx$ égalité |
| $\ge 10^{17}$ | $\alpha = 2$ (gain $\approx 41\%$) |

D'où le choix **adaptatif** : $\alpha = 1$ si $x < 3 \cdot 10^{16}$,
sinon $\alpha = 2$.

**Attention :** $\alpha = 1.25$ donne des résultats **faux** à certains $x$
(observé à $9.93 \cdot 10^{15}$, $9.95 \cdot 10^{15}$, $10^{16}$). Bug non
compris à ce jour — n'utiliser que $\alpha \in \{1, 2\}$.

---

## 7. Algorithme global en pseudocode

```
fn prime_pi(x):
    alpha <- 1.0 si x < 3e16 sinon 2.0
    y <- alpha * cbrt(x)
    z <- x / y
    seed_primes <- primes jusqu'a y                      # crible Lucy
    a <- len(seed_primes)
    c <- 5                                               # {2,3,5,7,11} absorbés

    if a <= c:
        return baseline_prime_pi(x)                      # fallback petits x

    # S1 : DFS sur squarefree m, facteurs ≤ y
    s1 <- 0
    DFS m = 1, p_index = c:
        s1 += mu(m) * phi_small_a(x/m, c, primes)
        # récurse sur m*p pour p > p_index, m*p ≤ y

    # S2_hard + P2 : sweep parallèle sur [lo_start, z]
    (s2_hard, p2) <- 0, 0
    for chaque bande [band_lo, band_hi) en parallèle:
        pour chaque fenêtre [lo, lo+SEG):
            sieve <- WheelSieve30(lo, lo+SEG)
            pour bi in 0..b_ext:
                si leaves présents (hard ou phi_easy):
                    phi_n <- phi_vec[bi] + count_primes(sieve, lo, n)
                    s2_hard += (-mu(m)) * phi_n            # hard
                    s2_hard += phi_n                       # phi_easy
                sieve.cross_off(primes[c+bi])             # mise à jour phi_vec
                phi_vec[bi] += running_total
            bulk sieve : cross off primes restants (p² ≤ hi)
            pour bi in b_ext..n_all (ext_easy):
                pour chaque leaf n in [lo, hi):
                    pi_n <- p2_offset + primes_count(sieve, lo, n)
                    s2_hard += max(1, pi_n - (b-2))       # clamp !
            P2 queries in window:
                p2 += pi(x/p) - rank(p)  pour chaque prime p dans s2_primes
                       dont x/p ∈ [lo, hi)
            p2_offset += primes dans la fenêtre

    phi_x_a <- s1 + s2_hard
    return phi_x_a + a - 1 - p2
```

---

## 8. Complexité

### 8.1 Espace

- $seed\_primes$ : $\pi(y) \approx y/\ln y \approx \alpha x^{1/3}/\ln x$ entiers
- $all\_primes$ : $\pi(\sqrt{x}) \approx \sqrt{x}/\ln x$ entiers → **dominant**
  à grand $x$.
- $\varphi_{vec}$ par bande : $b_{\text{ext}}$ entiers $\approx \pi(x^{1/4})
  = \mathcal{O}(x^{1/4}/\ln x)$.
- $hard\_leaves$ : total borné par $\sum_b |\{m : p | m \Rightarrow p_b < p,
  m \le y\}|$. Pour $p_b \le \sqrt{y}$, $\mathcal{O}(y)$ par prime, donc
  globalement $\mathcal{O}(\pi(\sqrt{y}) \cdot y) = \mathcal{O}(x^{2/3}/\ln^2 x)$.

Donc **espace total $\mathcal{O}(\sqrt{x}/\ln x + x^{2/3}/\ln^2 x) =
\mathcal{O}(x^{2/3}/\ln^2 x)$** à grand $x$.

### 8.2 Temps

- $S_1$ DFS : nombre de squarefree $m \le y$ à facteurs dans $(p_c, y]$,
  borné par $y \cdot \prod (1 + 1/p) = \mathcal{O}(y \ln y)$. Chaque nœud
  fait un lookup $\varphi\_small\_a$ en $\mathcal{O}(2^c) = \mathcal{O}(1)$
  (c=5, donc 32 termes).
  **Total $S_1$ : $\mathcal{O}(x^{1/3} \ln x)$**, négligeable.

- Sweep $S_2 + P_2$ : $z/SEG$ fenêtres, chacune cible :
  - Remplissage du sieve : $\mathcal{O}(SEG)$
  - Cross-off des primes actifs (pour $\varphi_{vec}$) : $b_{\text{ext}}$
    primes, chacune visite $SEG/p$ multiples, total
    $\mathcal{O}(SEG \ln \ln x)$.
  - Bulk cross-off (primes $> x^{1/4}$, $p^2 \le hi$) : chaque prime $p$
    participe à $\lfloor hi/p \rfloor$ fenêtres, chacune visitant $SEG/p$
    multiples.
  - Leaves & queries : $\mathcal{O}(1)$ par leaf, total $\mathcal{O}(x^{2/3}/\ln^2 x)$.

  **Total sweep : $\mathcal{O}(x^{2/3}/\ln^2 x \cdot \ln \ln x)$**
  (log-log du sieve d'Ératosthène).

### 8.3 Comparaison

- Meissel pur : $\mathcal{O}(x^{2/3}/\ln x)$
- Deléglise-Rivat : $\mathcal{O}(x^{2/3}/\ln^2 x)$ → **gain logarithmique**
- Baseline Lucy-Hedgehog : $\mathcal{O}(x^{3/4})$ → **nettement plus lent**
  à grand $x$

---

## 9. Références

- M. Deléglise & J. Rivat, *Computing $\pi(x)$: the Meissel, Lehmer, Lagarias,
  Miller, Odlyzko method*, Math. Comp. **65** (1996), 235–245.
- D. Hedgehog, « Prime counting » discussions, Project Euler forums
  (implémentation rapide de la formule de Lucy).
- Kim Walisch, *primecount*, https://github.com/kimwalisch/primecount
  (implémentation C++ de référence, algorithme Deléglise-Rivat).

---

## 10. Valeurs de référence (validation)

| $x$      | $\pi(x)$ attendu        |
|----------|-------------------------|
| $10^{10}$ | 455 052 511            |
| $10^{11}$ | 4 118 054 813          |
| $10^{12}$ | 37 607 912 018         |
| $10^{13}$ | 346 065 536 839        |
| $10^{14}$ | 3 204 941 750 802      |
| $10^{15}$ | 29 844 570 422 669     |
| $10^{16}$ | 279 238 341 033 925    |
| $10^{17}$ | 2 623 557 157 654 233  |
| $10^{18}$ | 24 739 954 287 740 860 |
