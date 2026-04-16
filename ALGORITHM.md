# primerivat — Mathematical note

This note describes the algorithm implemented in `prime_pi_dr_meissel_v4`
(`src/dr/mod.rs`). The goal is to compute exactly

$$
\pi(x) = \lvert\{p \le x : p \text{ prime}\}\rvert
$$

for $x$ up to $\sim 10^{18}$, in time $\mathcal{O}(x^{2/3}/\log^2 x)$.

---

## 1. Legendre function $\varphi$

Let $p_1 = 2 < p_2 = 3 < p_3 = 5 < \dots$ denote the sequence of prime numbers.

$$
\varphi(x, a) = \lvert\{n \le x : n = 1 \text{ or no prime factor of } n
                \text{ lies in } p_1,\dots,p_a\}\rvert
$$

This is the count of integers in $[1, x]$ that are **coprime** to the first
$a$ primes.

**Elementary properties:**

1. $\varphi(x, 0) = \lfloor x \rfloor$
2. $\varphi(x, a) = \varphi(x, a-1) - \varphi(\lfloor x/p_a \rfloor, a-1)$
   (inclusion–exclusion on divisibility by $p_a$)
3. If $p_a \le \sqrt{x}$, then $\varphi(x, a) \ge 1$ (the integer $1$ always
   survives)
4. For $p_a > x$: $\varphi(x, a) = 1$ (only $1$ remains)

**Useful closed form.** For $p_a \le x < p_{a+1}^2$,

$$
\varphi(x, a) = 1 + \pi(x) - a
$$

**Proof.** Under this hypothesis, every composite integer $m \le x$ has a
prime factor $\le \sqrt{x} < p_{a+1}$, hence one among
$\{p_1, \dots, p_a\}$, and is therefore eliminated. The survivors are
$\{1\} \cup \{\text{primes in } (p_a, x]\}$, which gives the count. $\square$

**Corollary.** If $p_a > x$ (so $\pi(x) < a$), then $\varphi(x, a) = 1$ as
well (only $1$ survives). This boundary case is handled by the `.max(1)`
clamp in the code.

---

## 2. Meissel's formula

Choose $a = \pi(y)$ with $y = \alpha \cdot x^{1/3}$ for some $\alpha \ge 1$.
Meissel's formula reads:

$$
\pi(x) = \varphi(x, a) + a - 1 - P_2(x, a)
$$

with

$$
P_2(x, a) = \lvert\{(p, q) : p, q \text{ prime}, y < p \le q, pq \le x\}\rvert.
$$

Each pair $(p, q)$ is a composite number $pq \le x$ whose two factors are
both strictly larger than $y$. $P_2$ can be rewritten as

$$
P_2(x, a) = \sum_{y < p \le \sqrt{x}} \bigl(\pi(x/p) - \pi(p) + 1\bigr).
$$

**Why $y = \alpha x^{1/3}$?** With this choice, a product $pqr$ of three
primes all $> y$ would satisfy $pqr > y^3 = \alpha^3 x \ge x$. So no $P_3$,
$P_4$, etc. — the formula stops at $P_2$.

---

## 3. Decomposition $\varphi(x, a) = S_1 + S_2$

Applying the relation $\varphi(x, a) = \varphi(x, a-1)
- \varphi(\lfloor x/p_a \rfloor, a-1)$ recursively until reaching a small
index $c$ (here $c = 5$, so we fall back on the primes $\{2,3,5,7,11\}$)
yields a sum over the **squarefree products** $m$ whose prime factors lie in
$(p_c, y]$:

$$
\varphi(x, a) = \sum_{\substack{m \text{ squarefree} \\ p \mid m \Rightarrow p_c < p \le y}}
                \mu(m) \cdot \varphi(\lfloor x/m \rfloor, c).
$$

**Split $S_1 + S_2$.** The $m$'s are partitioned by whether their **largest
prime factor** $P^+(m)$ lies in the "hard" zone or not:

- $S_1$: all prime factors of $m$ are $\le y$ AND
  $\lfloor x/m \rfloor > z = x/y$ (in practice, small $m$).
- $S_2$: the complement.

In the implementation, the grouping is by **largest prime factor** $p_b$:

$$
S_2 = -\sum_{b=c+1}^{a} \sum_{m'} \mu(m') \cdot \varphi\bigl(\lfloor x/(p_b m') \rfloor, b-1\bigr)
$$

where $m'$ ranges over squarefrees whose prime factors all lie in
$(p_{b-1}, y]$ (actually $> p_b$ to avoid double-counting — the sign from
$\mu(m p_b) = -\mu(m)$ is absorbed in the global sign).

$S_1$ handles the "small" $m$'s via a recursive DFS over squarefree
products of primes $\le y$ (see `phi::s1_ordinary`).

---

## 4. Classification of leaves in $S_2$

Index $p_b$ by $bi = b - c - 1$ (so $bi \ge 0$).

| Range of $bi$              | $p_b$ in               | Name         | Technique for $\varphi(n, b-1)$      |
|---------------------------|------------------------|--------------|--------------------------------------|
| $[0, n_{\text{hard}})$    | $\le \sqrt{y}$         | **hard**     | $\varphi_{vec}[bi]$ + sieve popcount |
| $[n_{\text{hard}}, b_{\text{ext}})$ | $(\sqrt{y}, x^{1/4}]$  | **phi_easy** | $\varphi_{vec}[bi]$ + sieve popcount |
| $[b_{\text{ext}}, n_{\text{all}})$ | $(x^{1/4}, y]$         | **ext_easy** | closed form $\pi(n) - (b-2)$ (with clamp) |

**Why three paths?**

### 4.1 hard leaves ($p_b \le \sqrt{y}$)

When $p_b$ is small, $m' = p_{l_1} p_{l_2} \dots$ can be multi-prime (the
total product still $\le y$). All such squarefree products are enumerated
via a bounded DFS (`enumerate_hard_leaves`), and $\varphi(n, b-1)$ is
evaluated along the sweep.

### 4.2 phi_easy leaves ($\sqrt{y} < p_b \le x^{1/4}$)

Here, any $m' = p_{l_1} p_{l_2}$ with two factors satisfies $m' > y$
(because $p_{l_1}, p_{l_2} > p_b > \sqrt{y}$). So only $m' = p_l$ (a single
prime, $l > b$) is valid. Hence "easy": one $l$ per $(b, \text{pair})$.

Evaluation still goes through $\varphi_{vec}[bi]$ (maintained by the sweep)
plus a popcount on the sieve between $lo$ and $n$.

### 4.3 ext_easy leaves ($p_b > x^{1/4}$)

Same structure as phi_easy (a single $p_l$). But here a geometric bound
holds:

$$
n = \frac{x}{p_b p_l} < \frac{x}{p_b^2} < \frac{x}{x^{1/2}} = \sqrt{x}.
$$

Moreover, for every valid pair, $p_{b-1} \le n < p_b^2$ **when $\alpha = 1$**
(see §5). In that case, the expensive formula with $\varphi_{vec}$ can be
replaced by the closed form

$$
\varphi(n, b-1) = \pi(n) - (b - 2).
$$

This avoids maintaining $\varphi_{vec}$ for the bulk of primes (those above
$x^{1/4}$, which are numerous).

---

## 5. Validity of $\varphi(n, b-1) = \pi(n) - (b-2)$

Applying the formula of §1 with $a \leftarrow b-1$:

$$
\varphi(n, b-1) = \pi(n) - (b-1) + 1 = \pi(n) - (b-2)
$$

**valid when** $p_{b-1} \le n < p_b^2$.

### 5.1 Upper bound $n < p_b^2$

For ext_easy: $p_b > x^{1/4}$ so $p_b^4 > x$ and $x/p_b^3 < p_b$. Since
$p_l > p_b$, we have $n = x/(p_b p_l) \le x/(p_b \cdot p_b) = x/p_b^2$; we
require $n < p_b^2$, i.e. $x/p_b^2 < p_b^2$, i.e. $x < p_b^4$, which always
holds. $\square$

### 5.2 Lower bound $p_{b-1} \le n$

$n_{\min} = x/(p_b p_a)$ (attained at $p_l = p_a \approx y$). Condition:

$$
p_{b-1} \le \frac{x}{p_b p_a} \iff p_{b-1} \cdot p_b \cdot p_a \le x.
$$

Take the worst case $p_{b-1} \approx p_b \approx p_a \approx y =
\alpha x^{1/3}$:

$$
p_{b-1} p_b p_a \approx y^3 = \alpha^3 x.
$$

**This holds for $\alpha^3 \le 1$, i.e. $\alpha \le 1$.** For $\alpha > 1$,
some leaves have $p_{b-1} > n$, and the closed form is invalid.

### 5.3 The `.max(1)` fix

When $p_{b-1} > n$ (equivalently $\pi(n) < b-1$), the closed form yields
$\pi(n) - (b-2) \le 0$, which is nonsensical. The true value is

$$
\varphi(n, b-1) = 1
$$

(all primes $\le n$ already lie in $\{p_1, \dots, p_{b-1}\}$, so only $1$
survives). Hence the clamp `(pi_n - (b as i64 - 2)).max(1)` in
`hard::s2_hard_sieve_par`.

---

## 6. Role of parameter $\alpha$

Larger $\alpha$ means larger $y$:
- **Fewer sieve windows** in the sweep (the length $z = x/y$ shrinks with
  $\alpha$): gain roughly $1/\alpha$.
- **More seed primes** ($a = \pi(y)$ grows with $\alpha$): per-window cost
  increases, and the seed_primes allocation is larger.
- **More ext_easy leaves**: the range $(x^{1/4}, y]$ widens.

The optimal trade-off depends on $x$. Measurements on i5-9300H:

| $x$           | Best $\alpha$                                                     |
|---------------|--------------------------------------------------------------------|
| $\le 10^{15}$ | $\alpha = 1$ (seed_primes overhead would dominate with $\alpha = 2$) |
| $10^{16}$     | roughly break-even                                                 |
| $\ge 10^{17}$ | $\alpha = 2$ (gain $\approx 41\%$)                                 |

Hence the **adaptive choice**: $\alpha = 1$ if $x < 3 \cdot 10^{16}$,
otherwise $\alpha = 2$.

**Warning:** $\alpha = 1.25$ yields **wrong** results at some $x$ (observed
at $9.93 \cdot 10^{15}$, $9.95 \cdot 10^{15}$, $10^{16}$). Root cause not
understood — use only $\alpha \in \{1, 2\}$.

---

## 7. Global algorithm in pseudocode

```
fn prime_pi(x):
    alpha <- 1.0 if x < 3e16 else 2.0
    y <- alpha * cbrt(x)
    z <- x / y
    seed_primes <- primes up to y                       # Lucy sieve
    a <- len(seed_primes)
    c <- 5                                              # {2,3,5,7,11} absorbed

    if a <= c:
        return baseline_prime_pi(x)                     # small-x fallback

    # S1: DFS over squarefree m, factors <= y
    s1 <- 0
    DFS m = 1, p_index = c:
        s1 += mu(m) * phi_small_a(x/m, c, primes)
        # recurse on m*p for p > p_index, m*p <= y

    # S2_hard + P2: parallel sweep over [lo_start, z]
    (s2_hard, p2) <- 0, 0
    for each band [band_lo, band_hi) in parallel:
        for each window [lo, lo+SEG):
            sieve <- WheelSieve30(lo, lo+SEG)
            for bi in 0..b_ext:
                if leaves are present (hard or phi_easy):
                    phi_n <- phi_vec[bi] + count_primes(sieve, lo, n)
                    s2_hard += (-mu(m)) * phi_n         # hard
                    s2_hard += phi_n                    # phi_easy
                sieve.cross_off(primes[c+bi])           # update phi_vec
                phi_vec[bi] += running_total
            bulk sieve: cross off remaining primes (p² <= hi)
            for bi in b_ext..n_all (ext_easy):
                for each leaf n in [lo, hi):
                    pi_n <- p2_offset + primes_count(sieve, lo, n)
                    s2_hard += max(1, pi_n - (b-2))     # clamp!
            P2 queries in window:
                p2 += pi(x/p) - rank(p)  for each prime p in s2_primes
                       with x/p in [lo, hi)
            p2_offset += primes in the window

    phi_x_a <- s1 + s2_hard
    return phi_x_a + a - 1 - p2
```

---

## 8. Complexity

### 8.1 Space

- $seed\_primes$: $\pi(y) \approx y/\ln y \approx \alpha x^{1/3}/\ln x$
  integers.
- $all\_primes$: $\pi(\sqrt{x}) \approx \sqrt{x}/\ln x$ integers →
  **dominant** for large $x$.
- $\varphi_{vec}$ per band: $b_{\text{ext}}$ integers $\approx \pi(x^{1/4})
  = \mathcal{O}(x^{1/4}/\ln x)$.
- $hard\_leaves$: total bounded by $\sum_b |\{m : p | m \Rightarrow p_b < p,
  m \le y\}|$. For $p_b \le \sqrt{y}$, $\mathcal{O}(y)$ per prime, hence
  globally $\mathcal{O}(\pi(\sqrt{y}) \cdot y) = \mathcal{O}(x^{2/3}/\ln^2 x)$.

So **total space $\mathcal{O}(\sqrt{x}/\ln x + x^{2/3}/\ln^2 x) =
\mathcal{O}(x^{2/3}/\ln^2 x)$** for large $x$.

### 8.2 Time

- $S_1$ DFS: number of squarefree $m \le y$ with factors in $(p_c, y]$,
  bounded by $y \cdot \prod (1 + 1/p) = \mathcal{O}(y \ln y)$. Each node
  does an $\mathcal{O}(2^c) = \mathcal{O}(1)$ $\varphi\_small\_a$ lookup
  ($c = 5$, so 32 terms).
  **Total $S_1$: $\mathcal{O}(x^{1/3} \ln x)$**, negligible.

- Sweep $S_2 + P_2$: $z/SEG$ windows, each handling:
  - Sieve fill: $\mathcal{O}(SEG)$
  - Cross-off of the active primes (for $\varphi_{vec}$): $b_{\text{ext}}$
    primes, each visiting $SEG/p$ multiples, total
    $\mathcal{O}(SEG \ln \ln x)$.
  - Bulk cross-off (primes $> x^{1/4}$, $p^2 \le hi$): each prime $p$
    participates in $\lfloor hi/p \rfloor$ windows, each visiting $SEG/p$
    multiples.
  - Leaves & queries: $\mathcal{O}(1)$ per leaf, total
    $\mathcal{O}(x^{2/3}/\ln^2 x)$.

  **Total sweep: $\mathcal{O}(x^{2/3}/\ln^2 x \cdot \ln \ln x)$**
  (log-log from the Eratosthenes sieve).

### 8.3 Comparison

- Pure Meissel: $\mathcal{O}(x^{2/3}/\ln x)$
- Deléglise–Rivat: $\mathcal{O}(x^{2/3}/\ln^2 x)$ → **logarithmic gain**
- Lucy–Hedgehog baseline: $\mathcal{O}(x^{3/4})$ → **markedly slower** at
  large $x$

---

## 9. References

- M. Deléglise & J. Rivat, *Computing $\pi(x)$: the Meissel, Lehmer, Lagarias,
  Miller, Odlyzko method*, Math. Comp. **65** (1996), 235–245.
- Lucy_Hedgehog, "Prime counting" discussions, Project Euler forums (fast
  implementation of Lucy's formula).
- Kim Walisch, *primecount*, https://github.com/kimwalisch/primecount
  (reference C++ implementation of the Deléglise–Rivat algorithm).

---

## 10. Reference values (validation)

| $x$      | expected $\pi(x)$       |
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
