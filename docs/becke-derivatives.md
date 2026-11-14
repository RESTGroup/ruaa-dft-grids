# Classic Becke partition derivatives

This page derives the analytic gradient and Hessian of the **classic Becke**
partition weights implemented in [`crate::partition`]. It is the
Becke-specific companion to the scheme-independent
[partition derivative API](crate::doc::partition_derivatives); read that page
first for tensor layouts, selection and the generating-atom correction.
Formula differences for the other schemes are collected in
[scheme variants](crate::doc::scheme_variants).

Scope: exact classic Becke only (`PartitionScheme::Becke`,
`becke_early_exit == false`). The ORCA early-exit truncation is
discontinuous and is rejected by the derivative evaluator.

## 1. Notation

The derivation is for the exact classic scheme:

```rust
use ruaa_dft_grids::partition::PartitionScheme;

let scheme = PartitionScheme::Becke; // with becke_early_exit == false
assert_eq!(format!("{scheme:?}"), "Becke");
```

| Symbol | Meaning |
|---|---|
| $g$ | grid point, coordinate $\mathbf r_g$ |
| $A_g$ (`parent`) | atom that generated point $g$ |
| $\mathbf R_A$ | nuclear coordinates |
| $d_A = \lVert \mathbf r_g - \mathbf R_A\rVert$ | point–atom distance |
| $\hat{\mathbf r}\_A = (\mathbf R_A - \mathbf r_g)/d_A$ | $\partial d_A/\partial \mathbf R_A$ |
| $R_{AB}$ | interatomic distance |
| $\hat{\mathbf u}\_{AB} = (\mathbf R_A - \mathbf R_B)/R_{AB}$ | interatomic unit vector |
| $a_{AB}$ | Becke size-adjustment factor |
| $\mu_{AB} = (d_A - d_B)/R_{AB}$ | ellipsoidal coordinate difference |
| $\nu_{AB} = \mu_{AB} + a_{AB}(1-\mu_{AB}^2)$ | adjusted coordinate |
| $f(\nu)$ | $p$ composed `hardness` times |
| $s_{AB} = \tfrac12(1-f)$ | cell factor |
| $P_M = \prod_{N\ne M} s_{MN}$ | running product |
| $Z = \sum_M P_M$ | normalization |
| $\bar P = P_{A_g}$ | numerator of the generating atom |
| $q_g$ | fixed quadrature weight |

The grid weight is

$$
w_g = q_g\ \frac{\bar P_g}{Z_g},
\qquad
\bar P_g = P_{A_g}.
$$

The crate returns derivatives of $w_g$ (with $q_g$ folded in) for the
streamed gradient/Hessian API, and unit-quadrature values from the
`*_at` single-point helpers.

## 2. Code layout and molecule tables

The implementation lives in:

```text
src/partition/
├── becke/
│   ├── mod.rs        module docs + DerivTables (1/R, unit vectors, projections)
│   ├── switch.rs     p(x), f, f', f'', hardness-3 unrolled forms
│   ├── weights.rs    classic pair-product weight kernels
│   ├── gradient.rs   analytic dw (fixed-lane, selected rows)
│   ├── hessian.rs    analytic d2w (scalar pointwise)
│   ├── hessian_lane.rs analytic d2w (fixed-lane policy)
│   └── deriv.rs      BeckeDeriv: the DerivScheme adapter
├── deriv/
│   ├── mod.rs        PartitionDerivEvaluator (streaming, one-pass fill)
│   ├── source.rs     point access for plan and materialised-grid evaluation
│   ├── scheme.rs     DerivScheme / DerivScratch extension point
│   ├── workspace.rs  one workspace for block and point scratch
│   ├── contract.rs   DerivContract, GradientContract, HessianContract
│   └── output.rs     DerivRequest, DerivOutput
├── kernel.rs         weight-kernel scratch / distance tile
└── ...
```

The public API (types, tensor shapes, streaming and selection conventions) is
documented in the [partition derivative API](crate::doc::partition_derivatives).
Molecule-only derivative data is precomputed once in `DerivTables`:

| Field | Formula | Size |
|---|---|---|
| `inv_r` | $1/R_{ij}$ | $n^2$ |
| `u_ab` | $\hat{\mathbf u}\_{ij}$ | $3n^2$ |
| `pu_ab` | $(\mathbf I - \hat{\mathbf u}\hat{\mathbf u}^\mathsf{T})/R_{ij}$ | $9n^2$ |

These are the only $O(n^2)$ tables. `PartitionTables` (the $a_{AB}$ and
$R_{AB}$ weight tables) is reused unchanged.

The Hessian kernel also emits the selected gradient when the caller asks for
`dw`/`dc` through the same `DerivRequest`. It reads the already-computed
$\partial P$ and $\partial Z$ rows, so the extra cost is only the writes and
the contraction; there is no second pair sweep. The dedicated blocked
gradient kernel remains the fast path when no Hessian output is requested.

## 3. The switch function and its derivatives

The composition layer is

$$
p(x) = \frac{3}{2}x - \frac{1}{2}x^3,
\qquad
p^{\prime}(x) = \frac{3}{2}(1-x^2),
\qquad
p^{\prime\prime}(x) = -3x.
$$

With $f_0 = \nu$ and $f_{k+1} = p(f_k)$,

$$
f^{\prime}\_{k+1} = p^{\prime}(f_k) f^{\prime}\_k,
\qquad
f^{\prime\prime}\_{k+1} = p^{\prime\prime}(f_k) (f^{\prime}\_k)^2 + p^{\prime}(f_k) f^{\prime\prime}\_k.
$$

For hardness 3 the kernel uses an unrolled form that performs the same
sequence of operations. The first layer's second derivative is **not** zero:
$f^{\prime\prime}\_1 = p^{\prime\prime}(\nu)$, and forgetting it produces a wrong Hessian while the
gradient still passes. The unrolled form is

```text
dd1 = p''(f0)
dd2 = p''(f1) df1^2 + p'(f1) dd1
dd3 = p''(f2) df2^2 + p'(f2) dd2
```

with `df0 = 1`. Generic hardness uses the same recursion.

The $\mu$-derivatives are

$$
\nu^{\prime} = 1 - 2a\mu,
\qquad
\nu^{\prime\prime} = -2a,
$$

and for $s_{AB} = \tfrac12(1-f)$, $s_{BA} = \tfrac12(1+f)$:

$$
\frac{\partial s}{\partial \mu}
= \mp\frac{1}{2} f^{\prime} \nu^{\prime},
\qquad
\frac{\partial^2 s}{\partial \mu^2}
= \mp\frac{1}{2}\left(f^{\prime\prime}(\nu^{\prime})^2 + f^{\prime}\nu^{\prime\prime}\right).
$$

The logarithmic derivatives used by the kernels are

$$
D_{AB} = \frac{1}{s_{AB}^{\text{safe}}}\frac{\partial s_{AB}}{\partial \mu},
\qquad
D^{(2)}\_{AB}
= \frac{1}{s_{AB}^{\text{safe}}}
\frac{\partial^2 s_{AB}}{\partial \mu^2} - D_{AB}^2,
$$

with $s^{\text{safe}} = \max(s, 10^{-14})$ (`INVTOL`). The clamp matters: the
product $P_M$ is not divided by, so only the bounded $1/s$ is regularised.
This is a standard numerical-stability choice for the pair product.

## 4. Gradient

### 4.1 Pair roles

For a pair $i > j$ define $\mu = (d_i - d_j)/R_{ij}$. Differentiating with
respect to the two endpoints,

$$
\frac{\partial \mu}{\partial \mathbf R_i}
= \frac{\hat{\mathbf r}\_i - \mu\ \hat{\mathbf u}\_{ij}}{R_{ij}},
\qquad
\frac{\partial \mu}{\partial \mathbf R_j}
= \frac{-\hat{\mathbf r}\_j + \mu\ \hat{\mathbf u}\_{ij}}{R_{ij}}.
$$

The running product uses the logarithmic form

$$
\partial P_M = P_M \sum_{N\ne M} D_{MN}\ \partial\mu_{MN},
$$

so no $P_M$ appears in a denominator. For every pair the kernel forms

$$
C_Z = P_i D_{ij} + P_j D_{ji},
\qquad
C_{\bar P} = \delta_{iA_g} P_i D_{ij} + \delta_{jA_g} P_j D_{ji},
$$

and accumulates

$$
\frac{\partial Z}{\partial \mathbf R_i}
\mathrel{+}= C_Z \frac{\partial\mu}{\partial \mathbf R_i},
\qquad
\frac{\partial \bar P}{\partial \mathbf R_i}
\mathrel{+}= C_{\bar P}\frac{\partial\mu}{\partial \mathbf R_i},
$$

and likewise for $\mathbf R_j$. **A subtle but essential point:** for the
$\bar P$ numerator, the same coefficient $C_{\bar P}$ multiplies both role
derivatives. Using only the $i = A_g$ term for row $i$ (and only the
$j = A_g$ term for row $j$) is wrong; the partner's parent term also
contributes to row $i$ through $\partial\mu/\partial\mathbf R_i$. This is
caught by the finite-difference test.

### 4.2 Quotient rule

With $q = \bar P/Z$ and the partial derivatives taken at fixed
$\mathbf r_g$,

$$
\frac{\partial w}{\partial R_{At}}
= q_g\left\lbrack
\frac{1}{Z}\frac{\partial \bar P}{\partial R_{At}}
\quad - \frac{\bar P}{Z^2}\frac{\partial Z}{\partial R_{At}}
\right\rbrack
= \frac{q_g}{Z}\left(
\frac{\partial \bar P}{\partial R_{At}}
\quad - q\frac{\partial Z}{\partial R_{At}}
\right).
$$

### 4.3 Translational invariance and the generating atom

The grid point is not an independent variable:
$\mathbf r_g = \mathbf R_{A_g} + \mathbf r_g^\mathrm{local}$. The weight is
invariant under a common translation of $(\mathbf r_g, \mathbf R_A)$, hence

$$
\frac{\partial w}{\partial \mathbf r_g}
\quad + \sum_A \frac{\partial w}{\partial \mathbf R_A} = 0,
\qquad
\frac{\mathrm d w}{\mathrm d \mathbf R_{A_g}}
= -\sum_{A\ne A_g}\frac{\partial w}{\partial \mathbf R_A}.
$$

The kernel already has $\partial w/\partial \mathbf R_{A_g}$; it subtracts
the full sum of the partial rows. The sum is accumulated directly as
`dp_sum`/`dz_sum` during the pair loop, so unselected rows are not needed:

$$
\sum_A \partial w_{At} = \frac{q_g}{Z}\left(
\sum_A \partial \bar P_{At} - q \sum_A \partial Z_{At}\right).
$$

### 4.4 Fixed-lane kernel and selection

The gradient kernel processes eight points per `f64x8` lane with the same
`[center * nlane + lane]` tile as the weight kernel. It then runs a second
pair pass for the derivatives (the products $P_M$ must be complete before
the logarithmic derivatives can be contracted). Selected rows write tiles of
size $n_\mathrm{sel} \times 3 \times n_\mathrm{lane}$; the two sums are
$3 \times n_\mathrm{lane}$. This is why `atoms = [A]` saves memory and
writes even though the pair loop itself is still $O(n^2)$ per point. Mixed
generating atoms in one lane use lane masks; the common homogeneous lane
takes a scalar parent branch.

Complexity per point: $O(n^2)$ time, $O(n^2)$ (all selected) or
$O(n_\mathrm{sel}\ n)$ output/scratch. The gradient has no $O(n^3)$ term.

## 5. Hessian

### 5.1 Second derivatives of the running product

Differentiating $\partial P_M = P_M\ \partial\log P_M$ once more,

$$
\partial^2 P_M
= P_M\left(\partial^2\log P_M
\quad + \partial\log P_M \otimes \partial\log P_M\right).
$$

The first term is pairwise additive; the second is a rank-one cross term that
spans different pairs and cannot be accumulated inside the pair loop. This is
the structural difference from the gradient:

* the pair loop builds $\partial\log P_M$ for **all** $M$ and accumulates the
  pairwise $\partial^2 Z$ and $\partial^2\bar P$ and their sums over the
  first index;
* the cross term is added afterwards, when $\partial\log P_M$ is complete.

### 5.2 Pairwise part

For a pair $i > j$, let $\ell_{M,C,t} = \partial\log P_M/\partial R_{Ct}$.
The pair contributes to four blocks. With

$$
\mathcal C_1 = P_i D_{ij} + P_j D_{ji},
\qquad
\mathcal C_2 = P_i D^{(2)}\_{ij} + P_j D^{(2)}\_{ji},
$$

the $(A,B)$ block is

$$
\partial^2 Z_{At,Bs} \mathrel{+}=
\mathcal C_2\ (\partial_A\mu)\_t(\partial_B\mu)\_s
\quad + \mathcal C_1\ \partial^2_{At,Bs}\mu,
$$

and analogously for $\bar P$ with the parent-selected coefficients
$\delta_{iA_g}P_i D_{ij} + \delta_{jA_g}P_j D_{ji}$ and their
second-derivative counterparts. The second derivative of $\mu$ uses the
quotient rule

$$
\frac{\partial^2 (f/g)}{\partial X\partial Y}
= \frac{f_{XY}g - (f_X g_Y + g_X f_Y) - f g_{XY}}{g^2}
\quad + \frac{2f g_X g_Y}{g^3},
$$

with $f = d_i - d_j$, $g = R_{ij}$, and the four role tables:

| role | $f_X$ | $f_Y$ | $f_{XY}$ | $g_X$ | $g_Y$ | $g_{XY}$ |
|---|---|---|---|---|---|---|
| AA | $\hat{\mathbf r}\_i$ | $\hat{\mathbf r}\_i$ | $\mathrm{Proj}(\hat{\mathbf r}\_i)/d_i$ | $\hat{\mathbf u}$ | $\hat{\mathbf u}$ | $\mathrm{Proj}(\hat{\mathbf u})/R$ |
| AB | $\hat{\mathbf r}\_i$ | $-\hat{\mathbf r}\_j$ | 0 | $\hat{\mathbf u}$ | $-\hat{\mathbf u}$ | $-\mathrm{Proj}(\hat{\mathbf u})/R$ |
| BB | $-\hat{\mathbf r}\_j$ | $-\hat{\mathbf r}\_j$ | $-\mathrm{Proj}(\hat{\mathbf r}\_j)/d_j$ | $-\hat{\mathbf u}$ | $-\hat{\mathbf u}$ | $\mathrm{Proj}(\hat{\mathbf u})/R$ |

`BA` is the $(t,s)$ transpose of `AB`.
$\mathrm{Proj}(\hat{\mathbf r}\_M)/d_M$ and
$\mathrm{Proj}(\hat{\mathbf u})/R$ are precomputed per atom / per pair
(`prm`, `pu_ab`).

The $\partial\log P_M$ array is $n \times 3n$. Its diagonal blocks accumulate
over pairs; every off-diagonal block $(M,C)$ is written exactly once by the
pair $(\max(M,C), \min(M,C))$, so only the diagonal blocks need zeroing. The
same "assign off-diagonal, accumulate diagonal" discipline applies to the
pairwise $\partial^2 Z$ and $\partial^2\bar P$ rows.

### 5.3 Cross term

$$
C_{At,Bs} = \sum_M P_M\ \ell_{M,At}\ \ell_{M,Bs},
\qquad
C^{\bar P}\_{At,Bs} = \bar P\ \ell_{A_g,At}\ \ell_{A_g,Bs}.
$$

$C$ is symmetric; computing the upper triangle and mirroring halves the
dominant $O(n^3)$ work. The $\bar P$ cross term has only the generating-atom
row.

The fixed-lane kernel materialises $C$ in $12 \times 12$ tiles of `f64x8`
accumulators (cache-sized) with the sum over $M$ innermost. Every entry
uses the canonical orientation $(\min(r,c), \max(r,c))$, so the mirrored
lower-triangle value is bit-identical to the value a selected-row-only run
would compute. For a full all-atom Hessian only the upper-triangular tiles
are evaluated; for a partial selection only the $3 n_\mathrm{sel}$ requested
rows are materialised. In both cases `dlogP` is re-read once per tile
(instead of once per output row), and a subset of rows remains bitwise
identical to the same rows of the full Hessian.

### 5.4 Quotient rule and translation

With $q = \bar P/Z$ and $\mathrm dq = (\partial\bar P - q\ \partial Z)/Z$:

$$
\partial^2 w_{At,Bs} = q_g\left\lbrace
\frac{1}{Z}\left\lbrack
\partial^2\bar P_{At,Bs}
\quad - \mathrm dq_{Bs}\ \partial Z_{At}
\quad - q\ \partial^2 Z_{At,Bs}
\right\rbrack
\quad - \frac{1}{Z}\mathrm dq_{At}\ \partial Z_{Bs}
\right \rbrace.
$$

The translation correction for the second derivative has three cases, because
each of the two indices may be the generating atom:

$$
\partial^2 w_{A_g t, Bs} = -\sum_{A\ne A_g}\partial^2 w_{At,Bs},
$$

$$
\partial^2 w_{At, A_g s} = -\sum_{B\ne A_g}\partial^2 w_{At,Bs},
\qquad
\partial^2 w_{A_g t, A_g s}
= \sum_{A\ne A_g}\sum_{B\ne A_g}\partial^2 w_{At,Bs}.
$$

### 5.5 Selected rows without materialising all rows

A naive implementation computes all $\partial^2 w$ rows and then applies the
sums above. That is $O(n^3)$ even when only one row is requested. The kernel
uses the linearity of the quotient rule instead. Define

$$
S_t = \sum_A \partial Z_{At},
\qquad
Q_t = \sum_A \mathrm dq_{At},
\qquad
\mathrm dS_{t,Bs} = \sum_A \partial^2 Z_{At,Bs},
\qquad
\mathrm dT_{t,Bs} = \sum_A \partial^2 \bar P_{At,Bs}.
$$

Then the full correction row is

$$
\mathrm{fullA}\_{B,t,s}
= \frac{\mathrm dT_{t,Bs} - \mathrm dq_{Bs}S_t - q\ \mathrm dS_{t,Bs}}{Z}
\quad - \frac{Q_t\ \partial Z_{Bs}}{Z}.
$$

`dS`/`dT` split into pairwise and cross parts. The pairwise parts are
accumulated per pair into a $3 \times n \times 3$ array (cheap: only the two
rows $i$ and $j$ contribute to the sum over $A$). The cross parts are

$$
\mathrm dS^{\text{cross}}\_{t,Bs}
= \sum_M P_M \Big(\sum_A \ell_{M,At}\Big)\ell_{M,Bs},
\qquad
\mathrm dT^{\text{cross}}\_{t,Bs}
= \bar P \Big(\sum_A \ell_{A_g,At}\Big)\ell_{A_g,Bs}.
$$

So the correction needs only `dlogP`, its column sums, and two $9n$ arrays —
never the unselected rows. A selected row costs the rank-one row
$O(n\cdot 3n)$ plus a quotient sweep, and the all-atom case is exactly the
dense Hessian.

Memory per point (all selected): `dlogP` ($3n^2$) + selected pairwise
$\partial^2 Z$/$\partial^2\bar P$ ($2\cdot9n^2$) + output ($9n^2$)
$\approx 30n^2$ doubles. For $n = 168$ this is about 7 MB per thread; use
small streaming ranges. For $n_\mathrm{sel} = 4$ it drops to under 1 MB,
while the time is $O(n^2 + n_\mathrm{sel}\ n\cdot3n)$ per point.

The same point pass already has $\partial P$, $\partial Z$, $q$, $Z$ and the
summed translation coefficients. When `DerivRequest::dw` or `dc` is set, the
kernel writes

$$
\mathrm dw_{At} = q_g\left\lbrack
\frac{\partial \bar P_{At} - q\ \partial Z_{At}}{Z}
\quad - \delta_{A,A_g}
\frac{\partial \bar P^{\text{sum}}\_t - q\ \partial Z^{\text{sum}}\_t}{Z}
\right\rbrack ,
$$

which is exactly the expression of the blocked gradient kernel. A combined
Hessian+gradient run therefore does not redo the pair sweep; it only pays for
the extra output writes and contractions.

## 6. Validation specifics

The finite-difference method (perturb the centres, translate the point with
its generating atom), step sizes and the generic invariant list are in the
[partition derivative API](crate::doc::partition_derivatives) §8. The
Becke-specific tests and tolerances are:

| Test | Reference | Tolerance |
|---|---|---|
| `gradient_matches_total_finite_difference` | central difference of the weight, perturbing centres and translating the point with its parent | $2\cdot10^{-6}$ absolute (water, benzene) |
| `hessian_matches_gradient_finite_difference` | central difference of the analytic gradient | $2\cdot10^{-5}$ absolute (water, benzene) |
| `hessian_matches_second_difference_of_weight` | four-point second difference of the weight | $2\cdot10^{-4}$ absolute (water) |
| `hessian_is_symmetric_and_translation_invariant` | $\partial^2 w_{At,Bs} = \partial^2 w_{Bs,At}$ and $\sum_A \partial^2 w = 0$ | $10^{-10}$ |
| `gradient_sums_to_zero_over_atoms` | $\sum_A \partial w = 0$ | $10^{-11}$ |
| `selection_is_exactly_a_row_subset` | selected rows vs the same rows of the all-atom result | bitwise |
| `streaming_contract_matches_full_blocks` | $\sum_g c_g\ \mathrm{ddw}$ from `ddc` vs the visitor accumulation | $10^{-13}$ |
| `gradient_batch_matches_point_api` | block kernel vs `gradient_at` | $10^{-13}$ relative (after $q$) |
| `gradient_weights_match_weight_evaluator` | derivative-path $\bar P/Z$ vs `PartitionEvaluator` | bitwise |
| `sorted_point_order_is_per_point_invariant` | box-sorted grid vs source order | bitwise |
| `parallel_matches_serial` | `fill` vs `fill_serial` for `dw`/weights and `ddc` | bitwise / $10^{-12}$ |
| `combined_hessian_emits_bitwise_gradient` | fused Hessian+gradient `dw` vs gradient-only | $10^{-12}$ relative |
| `contract_sets_and_windows_are_independent` | different `nset` and a contract window vs single-set runs | $10^{-13}$ |
| `raw_ddw_matches_visitor` | batch `ddw` vs concatenated visitor blocks | bitwise |
| `custom_scheme_drives_evaluator` | a toy non-Becke `DerivScheme` through the full one-pass machinery | exact |
| `empty_selection_is_valid` | empty row selection with a non-empty range | no panic, empty outputs |
| `single_atom_has_trivial_derivatives` | one atom: weight 1, all derivatives 0 | $10^{-15}$ / $10^{-12}$ |

Becke-specific value-path guard: the derivative path uses precomputed $1/R$
while the value path divides by $R$. The weight output is forced to be
bitwise equal and the derivative differs only at the $10^{-16}$ level. The
ORCA bitwise test and the PySCF asset tests additionally guard the
weight path, including the hardness-3 block specialisation.

## 7. Limitations specific to Becke

* Derivatives are exact for the **untruncated** classic Becke weight. The
  ORCA early-exit value path is deliberately not differentiated because the
  truncated product is discontinuous when a pair factor crosses $10^{-14}$.
* A post-build `PostPrune` is a selection, not part of the derivative. Fixed
  `KeepIndices`/`KeepMask` subsets differentiate as ordinary point subsets.
  Threshold cuts (`TotalWeightCut`, `PartitionCut`) are geometry-dependent
  and not differentiable at the cutoff; a consumer that re-evaluates them at
  displaced centres must decide how to handle the boundary.
* The Hessian scratch is $O(n_\mathrm{atom}^2)$ per thread, inherent to an
  exact Becke Hessian. It is not an
  $n_\mathrm{atom} \times n_\mathrm{grid}$ object, but an all-atom Hessian on
  a very large molecule needs a small streamed range, atom selection, or a
  contracted output.
* The grid-shift term $w_g\ \partial F/\partial\mathbf r_g$ (the coordinate
  part of the DFT grid-shift derivative) is a consumer responsibility; this
  crate provides `dw` and `ddw` of the weights.

## 8. References

* A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).
* O. Treutler, R. Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995).
* PySCF 2.x `pyscf.dft.gen_grid`: `get_partition`, `arg_group_grids`,
  `VXCgen_grid`.
* The [partition derivative API](crate::doc::partition_derivatives) for the
  generic tensor layouts and the
  [performance page](crate::doc::performance) for measured throughput.
