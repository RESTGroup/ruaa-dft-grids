# Numerical algorithms

This page is the formula-level background for the grids that `ruaa_dft_grids`
builds. It is written for users who already know how a Kohn–Sham integration
grid is built and want to know exactly which
construction this crate implements. The API-level walkthrough is in the
[user guide](crate::doc::guide); the numerical data tables are internal and
are reached through the public accessor functions documented on this page.

## 1. What a molecular grid is

A molecular integration grid is a finite quadrature of the exchange–correlation
integral:

$$
\int F(\mathbf r)\ \mathrm d^3 r
\quad\approx\quad
\sum&#95;{g} w&#95;g\ F(\mathbf r&#95;g).
$$

The points are generated atom by atom. Atom $A$ carries a spherical grid

$$
\mathbf r&#95;{A,i} = \mathbf R&#95;A + r&#95;i\ \hat{\mathbf u}&#95;j,
$$

where $r&#95;i$ is a radial node and $\hat{\mathbf u}&#95;j$ a direction on the unit
sphere. The **quadrature weight** of the point is

$$
q&#95;{A,ij} = w&#95;i^{(\mathrm{rad})}\ w&#95;j^{(\mathrm{ang})},
$$

and the **total grid weight** used in the integral is

$$
w&#95;g = q&#95;g\ P&#95;g,
\qquad
P&#95;g = \frac{\bar P&#95;g}{Z&#95;g},
\qquad
Z&#95;g = \sum&#95;{M} P&#95;{M g}.
$$

Here $P&#95;{M g}$ is the unnormalized partition cell function of atom $M$ at
point $g$, $\bar P&#95;g$ is the cell function of the *generating atom* (the atom
whose spherical grid produced the point), and $P&#95;g$ is the normalized
partition fraction. The partition is the subject of §5.

### 1.1 Units and weight convention

* All coordinates and lengths in the public API are in **Bohr**.
* The angular weights returned by [`crate::angular::angular_grid`] sum to
  **one** (the PySCF convention). The factor $4\pi$ is carried by the radial
  weight, so

    $$
    w&#95;i^{(\mathrm{rad})} = 4\pi\ r&#95;i^2\ \Delta r&#95;i,
    \qquad
    \sum&#95;i w&#95;i^{(\mathrm{rad})} \approx 4\pi \int r^2\ \mathrm d r .
    $$

* The product $w&#95;i^{(\mathrm{rad})} w&#95;j^{(\mathrm{ang})}$ is therefore the
  volume element, and the placement of $4\pi$ is invisible to the
  integrator. `radial::Radial::radial_rd` is the raw $(r,\Delta r)$ kernel
  without the $4\pi$ or $r^2$ factors; `radial::Radial::grid_xi` applies the
  canonical normalization.
* The public unpartitioned weight is stored as `quad_weight`; the
  normalized partition fraction is stored as `partition_fraction`; the
  total is `weight = quad_weight * partition_fraction`.

## 2. Radial quadrature

A radial rule is a change of variable from $x \in \lbrack -1, 1\rbrack $ (or a Chebyshev
angle) to $r \in \lbrack 0, \infty)$ plus a quadrature rule in $x$. The crate keeps
all node mappings in [`crate::radial::kernels`] and the backend-independent
selector in [`crate::radial::Radial`].

### 2.1 GC-corrected Chebyshev map

The `Radial::OrcaOptM3` variant implements the Gauss–Chebyshev-corrected
construction (the name is historical). For
$i = 1,\ldots,n&#95;r$ and $\theta&#95;i = i\pi/(n&#95;r+1)$, with $s=\sin\theta&#95;i$ and
$c=\cos\theta&#95;i$:

$$
t&#95;i = \frac{2i}{n&#95;r+1}
 \quad - s c\left(0.4244131815783876\ s^2 + \frac{2}{\pi}\right),
$$

$$
r&#95;i = \frac{\xi}{\ln 2}\ln\frac{2}{t&#95;i},
\qquad
w&#95;i = r&#95;i^2\
\frac{16\xi}{3}\
\frac{s^4}{t&#95;i\ln 2\ (n&#95;r+1)}.
$$

The weights already include the $r^2$ measure. The kernel stores shells
outside-in; the crate returns ascending $r$. The published plain M3 rule
(Treutler–Ahlrichs) is `Radial::OrcaM3`; the Krack–Köster reference kernel is
an internal raw kernel with no public selector.

### 2.2 PySCF radial methods

The PySCF 2.x methods are ported directly from `pyscf.dft.radi` and exposed
through the same selector:

| `Radial` variant | PySCF function | Character |
|---|---|---|
| `TreutlerAhlrichs` | `radi.treutler_ahlrichs` | M4; PySCF default |
| `Delley` | `radi.delley` | log2 / Gauss–Legendre |
| `MuraKnowles` | `radi.mura_knowles` | log3 |
| `GaussChebyshev` | `radi.gauss_chebyshev` | M3 |
| `Becke` | `radi.becke` | Becke mapping, Bragg-scaled |

Each method chooses its own element-dependent length scale $\xi$ (see
[`crate::radial::Radial::xi_for_element`]). The lower-level kernels take
$\xi$ explicitly and return $(r, \Delta r)$; `Radial::grid_xi` multiplies by
$4\pi$ and $r^2$ to produce the canonical weight.

### 2.3 Euler–Maclaurin squared ratio

The `Radial::EulerMaclaurin` map is the squared-ratio Euler–Maclaurin
construction. With $n = n&#95;\mathrm{rad}$ and $i = 0,\ldots,n-1$:

$$
r&#95;i = \xi\left(\frac{i+1}{n-i}\right)^{2},
\qquad
x&#95;i = \frac{i+1}{n+1},
\qquad
\Delta r&#95;i = \frac{2\xi\ x&#95;i}{(1-x&#95;i)^3(n+1)}.
$$

The caller supplies the per-element scale $A(Z)$; a preset with molecular
cutoffs also applies the
molecular radial cutoff and the five-batch angular ladder described below.
`Radial::EulerMaclaurin` exposes the same map to every
builders.

### 2.4 Shell counts

A data-driven preset supplies the nominal radial count, the radial scale and
an optional cutoff.  It may also supply a per-element or per-row count; the
`GridPreset` data model is described in the [user guide](crate::doc::guide).
The built-in PySCF path uses the level table `RAD_GRIDS[level][period]` from
`pyscf.dft.gen_grid`; see [`crate::pyscf::presets`].

## 3. Angular quadrature

Angular weights always sum to one. The main rule is the Lebedev rule of the
requested size ([`crate::angular`]); the crate also carries the special
rules that are useful for different constructions.

| Rule | Where | Notes |
|---|---|---|
| Lebedev, $n$-point | `angular::angular_grid` | Oh-orbit generator; weights sum to 1 |
| 12-point icosahedral | `angular::angular_grid(12)` | Exact through degree 5 |
| 18-point octahedral | `angular::octahedral_18` | Exact through degree 3 |
| Gauss–Legendre $\times$ uniform $\varphi$ | `angular::product_angular_grid` | Product/polar rules |

A size outside the supported set is rejected during validation.  For a
product rule with $n&#95;\theta$ Gauss–Legendre rings and
$2n&#95;\theta$ azimuthal points per ring, the normalized weight per point is
$w&#95;\mathrm{GL}/(4 n&#95;\theta)$ in the crate's sum-to-one convention.

## 4. Angular pruning

Three mechanisms are available:

* [`crate::prune::Prune`] — the PySCF `gen_grid` schemes plus `None`;
* [`crate::prune::LadderPrune`] — a five-batch ladder with caller-supplied
  sizes and element-row boundaries;
* [`crate::preset::AngularLayout::Shells`] — an explicit size for every
  active radial shell.

[`crate::preset::AngularLayout::Fixed`] is the unpruned case.  A pruning
policy consumes the shell radii (and, for the ladder, the dimensionless
$r/\xi$ values) and returns one angular size per active shell.

### 4.1 PySCF schemes

PySCF's pruning schemes are not functions of one common ratio:

* `nwchem` and `sgx` use $r/R_{\mathrm{Bragg}}(Z)$ and the periodic row;
* `sg1` uses $r/R_{\mathrm{SG1}}(Z)$;
* `treutler` uses the shell-index fraction, independent of radius and element.

The schemes are implemented in [`crate::prune`] (see
[`crate::prune::Prune`]).

### 4.2 Five-batch ladder

A [`crate::prune::LadderPrune`] maps $r/\xi$ to one of five sizes
$(a&#95;1,a&#95;2,a&#95;3,a&#95;4,a&#95;3)$ using four row-specific boundaries;
the fifth batch steps back down.  The ladder, the unpruned fallback and the
element-row boundaries are data supplied by the preset.  Elements above a
configured `max_pruned_z` use the unpruned base or keep the ladder according
to [`crate::prune::HeavyFallback`].

## 5. Partition weights

Without a partition, every atom's spherical grid would integrate the whole
molecule. The partition assigns each point to its generating atom and
renormalizes so that the fractions sum to one:

$$
\sum&#95;M P&#95;{M g} = 1,
\qquad
P&#95;{M g} = \frac{\bar P&#95;{M g}}{Z&#95;g},
\qquad
Z&#95;g = \sum&#95;N \bar P&#95;{N g}.
$$

The unnormalized cell functions $\bar P&#95;{M g}$ differ between schemes.

### 5.1 Classic Becke

With $\mu&#95;{AB} = (d&#95;A - d&#95;B)/R&#95;{AB}$ and the size adjustment
$\nu&#95;{AB} = \mu&#95;{AB} + a&#95;{AB}(1-\mu&#95;{AB}^2)$,

$$
p(x) = \frac{3}{2}x - \frac{1}{2}x^3,
\qquad
f(\nu) = p\circ p\circ p\ (\nu),
\qquad
s&#95;{AB} = \frac{1}{2}\left(1-f(\nu&#95;{AB})\right),
$$

$$
\bar P&#95;A = \prod&#95;{B\ne A} s&#95;{AB}.
$$

The adjustment factor $a&#95;{AB}$ comes from the radii table and the chosen
`RadiiAdjust` policy; the implementations are in [`crate::partition`]. The normalized weights are $P&#95;A = \bar P&#95;A/Z$.

### 5.2 Stratmann–Scuseria–Frisch (SSF)

The SSF switch replaces $p\circ p\circ p$ with a piecewise rational function
that is **exactly** $\pm1$ outside $|\nu| \le A = 0.64$:

$$
f(\nu) =
\begin{cases}
\dfrac{1}{16}\ m&#95;a\left(35 + m&#95;a^2\left(-35 + m&#95;a^2(21-5m&#95;a^2)\right)\right),
 & m&#95;a = \nu/A,\ |\nu| < A,\cr
\operatorname{sign}(\nu), & |\nu| \ge A.
\end{cases}
$$

A cell factor is exactly zero or one outside the support, which makes the
pair product sparse. Formula differences and the sparse kernel are described
in the [scheme variants](crate::doc::scheme_variants) page.

### 5.3 SS switch

The SS variant keeps the same product but replaces the SSF switch by
the degree-13 polynomial

$$
F_1(x)=\frac{3003}{2048}\,x\left(
1-2x^2+3x^4-\frac{20}{7}x^6+\frac{5}{3}x^8
-\frac{6}{11}x^{10}+\frac{1}{13}x^{12}\right),
\qquad x=\frac{\nu}{0.86},
$$

with cell factor $s=\tfrac12-F_1(x)$.  The positive-support clip is
$\nu\le -0.86$ (factor one) and $\nu\ge 0.852$ (factor zero).  Equality
radii give the exact `SSWeights` mapping; the optional `IOp(79)` cutoff and
the neighbour-shell truncation are not modelled.  The sparse kernel and the
analytic gradient/Hessian are shared with SSF; only the switch derivatives
differ.  See the [scheme variants](crate::doc::scheme_variants) page.

### 5.4 mBecke

The mBecke (2022) scheme evaluates the same classic cell functions
nearest-atom-first with an early exit and a clamp. At the crate's default
clamp the product is algebraically identical to the classic Becke product;
the differences are the multiplication order and a round-off-level early
exit. See [`crate::partition`] and the
[scheme variants](crate::doc::scheme_variants) page.

### 5.5 LKO

The Laqua–Kussmann–Ochsenfeld partition replaces $R&#95;{AB}$ by a saturating
length:

$$
\mathrm{sat}(r) = 5\left(1-\exp\left\lbrack -\sum&#95;{m=1}^{12}\frac{(r/5)^m}{m}\right\rbrack \right),
\qquad
\mu&#95;{AB} = \frac{d&#95;A-d&#95;B}{\mathrm{sat}(R&#95;{AB})}.
$$

The clamped $\mu$ then enters the classic size adjustment and switch. Since
$\mathrm{sat}(r) < 5$, atoms farther than a fixed cutoff cannot contribute
nonzero factors; this is the basis of the $O(n&#95;\mathrm{atom})$ weight kernel.
The gradient uses eight-point lanes (dense below the crossover, masked
union-list above); the Hessian is pointwise, with a dense small-molecule
kernel and a sparse near-set kernel above a size threshold (chosen
empirically). See the
[scheme variants](crate::doc::scheme_variants) page.

### 5.6 Default choices

The crate has no hidden program default.  `PartitionConfig::pyscf()` is
classic Becke with PySCF's Bragg/Treutler radii;
`PartitionConfig::gaussian_ss()` and `PartitionConfig::gaussian_becke()` are
the published SS and BWeights formulas with equality radii.  A caller can
supply a custom radii table.

A program's point cutoff is a **post-build filter** on the normalized
partition fraction, not part of the partition formula; use
[`crate::moleculegrid::PostPrune::PartitionCut`] with that program's cutoff
value.

## 6. Molecular assembly and ordering

The structural pipeline is

```text
GridSpec -> GridPlan -> PartitionEvaluator -> BatchBuffer
```

* [`crate::grid::GridSpec`] is the full configuration: topology plus
  partition.
* [`crate::grid::GridPlan`] is the point topology: flat local coordinates,
  parent atoms and unpartitioned quadrature weights. It is
  geometry-independent for PySCF and for presets without molecular
  cutoffs (the default) makes it geometry-dependent.
* [`crate::partition::PartitionEvaluator`] fills only a requested point
  range, so the crate never stores an $n&#95;\mathrm{atom}\times n&#95;\mathrm{grid}$
  object.
* [`crate::moleculegrid::MolecularGrid`] is the convenience full-grid
  output. Post-build pruning and ordering are explicit operations on it.

Point order is never changed implicitly: the value tests
and the range evaluator both depend on the atom-contiguous order. A
consumer order is applied with [`crate::order::GridOrder`].

## 7. References

* A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).
* O. Treutler, R. Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995).
* M. Krack, A. M. Köster, *J. Chem. Phys.* **108**, 3226 (1998).
* H. Stratmann, G. E. Scuseria, M. J. Frisch, *Chem. Phys. Lett.* **257**,
  213 (1996).
* H. Laqua, J. Kussmann, C. Ochsenfeld, *J. Chem. Phys.* **149**, 204111
  (2018).
* PySCF 2.x `pyscf.dft.radi` and `pyscf.dft.gen_grid`.
* ORCA 6.1 manual, "Numerical Integration".
* Gaussian 16 manual, `Integral` keyword page.
