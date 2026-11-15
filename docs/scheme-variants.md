# Partition scheme variants: SSF, Gaussian SS, LKO and mBecke

The [Becke derivative page](crate::doc::becke_derivatives) derives the
gradient and Hessian for the classic pair product. The other four schemes
keep the same quotient rule, translation correction and streaming API, but
change the pair geometry or the cell function. This page states those
differences and their algorithmic cost; it does **not** repeat the Becke
derivation.

All schemes share the normalized weight

$$
w_g = q_g\ \frac{P_{A_g}}{Z_g},
\qquad
Z_g = \sum_M P_M,
$$

and the **[partition derivative API](crate::doc::partition_derivatives)**
contracts (`dw`, `ddw`, `dc`, `ddc`, atom selection, generating-atom
correction). Only the unnormalized cell product $P_M$ changes.

## 1. Cost at a glance

| Scheme | Cell factor | Pair work per point | Derivative backend |
|---|---|---|---|
| classic Becke | full $\mu$ polynomial | $O(n^2)$ | gradient: fixed lanes; Hessian: lane or pointwise |
| SSF | compact support, exactly 0/1 outside $\lvert\nu\rvert \le 0.64$ | $O(n\log n)$ + $O(n_\mathrm{surv}\ n)$ | sparse active-pair passes |
| Gaussian SS | compact asymmetric support, exactly 0/1 outside $[-0.86, 0.852]$ | $O(n\log n)$ + $O(n_\mathrm{surv}\ n)$ | sparse active-pair passes |
| LKO | saturating $R_{AB}$, clamped $\mu$ | $O(\lvert S\rvert\ \lvert T\rvert)$ | masked sparse lane gradient above the cut (dense lanes below); pointwise Hessian (dense below the crossover, sparse near-set above) |
| mBecke | classic product, nearest-first order | $O(n^2)$ | classic backend when the clamp is inert |

$n$ is the number of atoms, $n_\mathrm{surv}$ the survivors of the
compact-support candidate scan, and $S$, $T$ the LKO near sets defined in
§3.

## 2. Stratmann–Scuseria–Frisch (SSF)

The SSF switch (Stratmann, Scuseria and Frisch, 1996; PySCF
`gen_grid.stratmann`) replaces the cubic polynomial by a piecewise rational
function. With $A = 0.64$ and $m = \nu/A$:

$$
f(\nu) =
\begin{cases}
\dfrac{1}{16}m\left(35 + m^2\left(-35 + m^2(21-5m^2)\right)\right),
& |\nu| < A,\cr
\operatorname{sign}(\nu), & |\nu| \ge A.
\end{cases}
$$

The cell factors are $s = \tfrac12(1-f)$, so they are **exactly zero** where
$f=+1$ and **exactly one** where $f=-1$. Multiplying by an exact one is a
no-op; only pairs with $|\nu| < A$ (the *active* pairs) matter.

Two consequences drive the kernel:

1. **Only a closer atom can kill a product.** For a partner $j$ with
   $d_j \ge d_M$, $\mu \le 0$ and, for every allowed size adjustment
   $|a| \le 1/2$, $\nu \le 1/2 < A$. A product can therefore only be zeroed
   by a partner closer to the point than $M$.
2. **A distance bound certifies the kill.** Let $a_\mathrm{max}$ be the
   largest $|a_{AB}|$ in the molecule and $\mu_\mathrm{kill}$ the root of
   $\mu - a_\mathrm{max}(1-\mu^2) = A$. With

    $$
    R_\mathrm{cut} = \frac{1+\mu_\mathrm{kill}}{1-\mu_\mathrm{kill}},
    \qquad
    d_M \ge R_\mathrm{cut}\ d_j
    \quad\Longrightarrow\quad P_M = 0,
    $$

   where $d_j$ is the nearest centre distance. $R_\mathrm{cut}$ is between
   about $4.56$ (no adjustment) and $9.58$ ($a_\mathrm{max}=1/2$); with the
   Bragg radii it is about $6.25$.

The value kernel therefore evaluates all point–centre distances, takes the
candidate prefix $d_i \le R_\mathrm{cut} d_\mathrm{min}$, sorts it, and
evaluates products only for the survivors. The derivative kernels use the
same survivors and add pair passes over the *active* pairs; every other pair
has a locally constant factor and contributes neither a first nor a second
derivative. The per-point work is

$$
O(n\log n) + O(n_\mathrm{surv}\ n)
$$

instead of $O(n^2)$. For small molecules the sort and bookkeeping dominate,
so a dense `f64x8` fallback is used below a size threshold that depends on
the compiled instruction set. Both paths agree to round-off, and the scheme is validated against the dense
cell-product reference.

**Derivative geometry.** For an active pair,

$$
\kappa = \frac{\partial\nu}{\partial\mu} = 1 - 2a\mu,
\qquad
\ell = \frac{\partial\log s}{\partial\mu}
= -\frac{1}{2}\frac{f^{\prime}(\nu)\kappa}{s},
$$

$$
\ell_2 = \frac{\partial^2\log s}{\partial\mu^2}
= \frac{-\tfrac12\left(f^{\prime\prime}(\nu)\kappa^2 + f^{\prime}(\nu)\kappa^{\prime}\right)}{s}
\quad - \ell^2.
$$

The pair derivative $\partial\mu/\partial\mathbf R$ and the second derivative
$\partial^2\mu$ are the same quotient-rule blocks as in classic Becke; only
the switch derivatives $f^{\prime}$ and $f^{\prime\prime}$ change. The SSF Hessian is assembled
sparsely: per survivor it stores the log-derivative row and reuses it for the
rank-one cross term, and a selected row costs its own row plus a small number
of pair re-evaluations rather than a full $O(n^3)$ sweep.

### 2.1 Gaussian 16 SS

Gaussian 16 `SSWeights` uses the same cell product with a different,
asymmetric switch.  With $x=\nu/0.86$,

$$
F_1(x)=\frac{3003}{2048}\thinspace x\left(
1-2x^2+3x^4-\frac{20}{7}x^6+\frac{5}{3}x^8
-\frac{6}{11}x^{10}+\frac{1}{13}x^{12}\right),
\qquad f(\nu)=2F_1(x),
$$

and the cell factor is $s=\tfrac12(1-f)=\tfrac12-F_1(x)$.  Gaussian clips
the switch at $\nu\le -0.86$ (factor exactly one) and at
$\nu\ge 0.852$ (factor exactly zero); the positive bound is slightly inside
the polynomial endpoint.  The asymmetric clip does not change the kernel
structure: only a closer partner can zero a product, and the candidate bound
uses $\mu_\mathrm{kill}=0.852$ instead of $A$.  Without a size adjustment
$R_\mathrm{cut}\approx 12.5$; it grows with $|a_{AB}|$.

The dense fallback evaluates the partner factor in its own direction because
the two sides of the asymmetric switch are not complementary.  The sparse
path and the analytic gradient/Hessian use the same active-pair
log-derivative formulas as §2; only $f^{\prime}$ and $f^{\prime\prime}$
come from the degree-13 polynomial.  The switch is $C^1$ with vanishing
derivatives at the negative bound; the positive bound is a tiny downward
clip (the polynomial is already below $2\times10^{-13}$ there).  The
preset [`crate::partition::PartitionConfig::gaussian_ss`] uses equality
radii, which is the exact Gaussian mapping.

## 3. Laqua–Kussmann–Ochsenfeld (LKO)

The LKO partition (Laqua, Kussmann and Ochsenfeld, 2018; PySCF
`gen_grid.becke_lko`) replaces the interatomic distance by a saturating
length:

$$
\operatorname{sat}(r)
= 5\left(1-\exp\left\lbrack -\sum_{m=1}^{12}\frac{(r/5)^m}{m}\right\rbrack \right),
$$

$$
\mu_{AB} = \frac{d_A - d_B}{\operatorname{sat}(R_{AB})},
\qquad
g = \operatorname{clamp}(\mu_{AB}, -1, 1),
\qquad
\nu = g + a_{AB}(1-g^2),
$$

followed by the classic switch $s = \tfrac12(1-f(\nu))$. The size
adjustment $a_{AB}$ is the Treutler adjustment. The clamp is part of the
definition; it makes the cell factors locally constant for pairs with
$|\mu| \ge 1$.

### 3.1 Exact locality

$\operatorname{sat}(r) < 5$ for every finite $r$, so
$1/\operatorname{sat}(R) > 0.2$. For a point with nearest-centre distance
$d_\mathrm{min}$:

* If $d_i > d_\mathrm{min} + 5$, the pair with the nearest centre has
  $\mu > 1$, hence $g = 1$, $f(1) = 1$ and $s_i = 0$ exactly. Only atoms in

    $$
    S = \lbrace i : d_i \le d_\mathrm{min} + 5 \rbrace
    $$

  can have a nonzero product.
* A partner $j$ of a live atom changes the product only if its factor differs
  from one. For $d_j - d_i \ge 5$, $\mu \le -1$, $g=-1$, $f(-1)=-1$ and
  $s=1$ exactly. Only partners in

    $$
    T = \lbrace j : d_j \le d_\mathrm{min} + 10 \rbrace
    $$

  matter; $S \subseteq T$.
* Pairs with both endpoints outside $S$ cannot change a live product, so the
  pair work is at most $\lvert S\rvert\ \lvert T\rvert$, not
  $n(n-1)/2$.

The implementation widens both cuts by a small floating-point margin so the
clamp is guaranteed despite the rounding of $1/\operatorname{sat}$ and of
$d_i - d_j$. The weight kernel evaluates eight adjacent points at once with
per-lane $S/T$ masks and a union candidate list; it is bitwise identical to
the per-point reference. A dense `f64x8` fallback is used below the same
kind of size threshold as SSF.

### 3.2 Derivative geometry

With $f = d_i - d_j$ and $g = \operatorname{sat}(R_{ij})$,
$\mu = f/g$. At a fixed grid point,

$$
\frac{\partial\mu}{\partial\mathbf R_i}
= \operatorname{fac}\ \hat{\mathbf r}\_i - \mu k\ \hat{\mathbf u}\_{ij},
\qquad
\frac{\partial\mu}{\partial\mathbf R_j}
= -\operatorname{fac}\ \hat{\mathbf r}\_j + \mu k\ \hat{\mathbf u}\_{ij},
$$

where $\operatorname{fac} = 1/\operatorname{sat}$ and
$k = \operatorname{sat}^{\prime}/\operatorname{sat}$. The second derivative follows
from the quotient rule,

$$
\partial^2\mu
= \operatorname{fac}\left\lbrack
f_{XY} - k(f_X q_Y + q_X f_Y)
\quad + (k^2-k^{\prime})f q_X q_Y - k f\ \partial^2 R_{XY}
\right\rbrack ,
$$

with $q_i = \hat{\mathbf u}\_{ij}$, $q_j = -\hat{\mathbf u}\_{ij}$,
$k^{\prime} = \operatorname{sat}^{\prime\prime}/\operatorname{sat} - k^2$, and $f_{XY}$ the
projection matrix of the differentiated roles. The same quotient helper as
classic Becke is used, so the implementation shares its arithmetic structure.

Clamp handling: when $|\mu| \ge 1$ both cell factors are locally constant
(0 and 1), so the pair contributes **no** derivative. The gradient does not
need an explicit mask because $f^{\prime}(\pm1) = 0$; the Hessian explicitly skips
clamped pairs, because it differentiates the unclamped $\mu$.

The LKO gradient uses eight-point lanes: a dense kernel below the crossover
and a masked union-list kernel above it, which reuses the weight kernel's
$S/T$ lane masks. The Hessian is pointwise (no lane variant): a dense kernel for
small molecules and, above a size threshold (chosen empirically), a sparse
near-set kernel
that visits only $T\times T$ pairs and stores one support row per survivor.
The two Hessian kernels agree to signed zero, and the weights are bitwise
consistent between the value and derivative paths. The saturated derivatives
are finite-difference checked to about $5\cdot10^{-11}$ for the gradient and
$10^{-14}$ for the Hessian.

## 4. mBecke (ORCA 2022)

ORCA's mBecke scheme sorts the atoms nearest-first at each point and
multiplies the classic cell factors in that order. It adds:

* a clamp $p$: $\nu \ge p$ sets $P_i = 0$ and stops; $\nu \le -p$ skips the
  pair (the factor is exactly one);
* a relative early exit when the running remainder is negligible;
* a nearest-atom shortcut when $p < 1$.

The crate's default clamp is $p = 1.0$ (`MBECKE_CLAMP`). At this value and
for $|a_{AB}| \le 1/2$, the clamp cannot fire away from the boundary
$\nu = \pm1$: the product is algebraically identical to the classic Becke
product, only the multiplication order and the early-exit truncation differ.
The blocked value kernel therefore evaluates the classic eight-point pair
product; the scalar single-point API keeps ORCA's exact nearest-first order.
The value paths agree to round-off; the scalar path preserves ORCA's
evaluation order.

The exact kernel is inherently $O(n^2)$ scalar work: the triangle inequality
gives $|\mu| \le 1$, so factors are exactly one only at a measure-zero
collinear configuration and there is no zero/one structure to exploit.

**Derivative support.** mBecke at the default clamp is differentiated by the
classic Becke backend, because the classic product is the same function up to
floating-point evaluation order. The evaluator checks the eligibility guard

$$
p \ge 1
\quad\text{and}\quad
\max_{A,B}\lvert a_{AB}\rvert \le \tfrac12
$$

before dispatching. A clamp $p < 1$, or a radii/adjustment choice for which
the guard fails, returns `Error::UnsupportedDerivative`: ORCA's
clamp/nearest-atom branches are discontinuous. The finite-difference tests
compare against the production (classic-product) value path; the scalar
ORCA-order API differs by a few ulp.

## 5. Efficiency

The schemes differ by more than a constant: SSF and LKO reduce the pair work
from $O(n^2)$ to $O(n)$ for large molecules, while classic Becke and mBecke
keep the pair product. Measured throughput, the molecule set, and the build
recipe are in the [performance page](crate::doc::performance).

## 6. Validation summary

All five schemes are checked offline:

* value paths: bitwise regression tests, the PySCF construction, and a direct
  dense cell-product reference for the sparse
  SSF/Gaussian SS/LKO kernels;
* the Gaussian SS switch against its published polynomial, and the
  SSF/Gaussian dense-versus-sparse products against each other;
* gradients: central differences of the weights, with the point translated
  by its generating atom;
* Hessians: central differences of the analytic gradient, symmetry and
  translation-sum rules;
* selected rows: bitwise equality with the corresponding rows of the
  all-atom result;
* combined runs: a Hessian request that also emits `dw` matches the
  gradient-only kernel.

Tolerances are summarised on the
test suite and in the
[Becke derivative page](crate::doc::becke_derivatives) §6.

## 7. References

* A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).
* H. Stratmann, G. E. Scuseria, M. J. Frisch, *Chem. Phys. Lett.* **257**,
  213 (1996).
* H. Laqua, J. Kussmann, C. Ochsenfeld, *J. Chem. Phys.* **149**, 204111
  (2018).
* PySCF 2.x `pyscf.dft.gen_grid`: `stratmann`, `becke_lko`.
* ORCA 6.1 manual, mBecke (2022) grid partition.
