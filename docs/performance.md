# Performance notes

This page collects measured throughput for the weight and derivative kernels.
They are engineering measurements, not guarantees: the absolute numbers shift
with the CPU, the compiler, the molecule and the point distribution. Use the
ratios and the build guidance, then measure your own workload.

## 1. Build target

The kernels are correct on the baseline target, but the blocked and lane
paths are designed for a target CPU. The crate deliberately does **not** set
a target in `.cargo/config.toml`; the caller keeps control of portability.

```bash
# Fastest for the lane kernels; binary is host-specific.
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Portable AVX2 binary (x86-64-v3).
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

`f64x8` is one AVX-512 register under `native`/`x86-64-v4` and two AVX2
registers under `x86-64-v3`. The baseline build uses the same source and the
same arithmetic; it is slower but remains correct.

Rayon parallelism is orthogonal: `build_molecular_grid` and the derivative
`fill` (with `DerivPoints`) use the ambient Rayon pool. Set
`RAYON_NUM_THREADS` or install a pool when a fixed thread count matters.
`fill` is optimised for ranges of a few thousand points; very small
ranges pay the per-call Rayon setup cost (see §4.4).
`fill_serial` and `fill_serial_visit` are the
serial grid-native forms (no Rayon region).

## 2. Measurement setup

The tables below were measured on:

* AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 48 KiB L1d, 1 MiB L2,
  128 MiB L3, pinned to CPUs 0–15;
* rustc 1.99.0, 16 Rayon threads;
* the grid and partition settings listed below, unless a table says
  otherwise.

The input `.xyz` geometries used by the benchmark examples are **not
bundled** with this crate; pass your own files to the example commands in
§5.  The tables are engineering measurements, not portable guarantees.

Unless stated otherwise, the grid is **PySCF level 3**:

| Setting | Value |
|---|---|
| radial rule | Treutler–Ahlrichs (`pyscf.dft.radi.treutler_ahlrichs`) |
| angular rule and pruning | NWChem pruning (`pyscf.dft.gen_grid.nwchem_prune`) |
| partition radii | PySCF Bragg (`RadiiConfig::pyscf()`) |
| size adjustment | Treutler |
| timing scope | partition evaluation only; grid and plan construction excluded |

Each table states its own point count, selection and evaluation batch; the
tables are not all on the same workload.

The four schemes in the weight table (classic Becke, SSF, mBecke and LKO) use `RadiiConfig::pyscf()` (PySCF Bragg
radii plus the Treutler adjustment), so mBecke and classic Becke evaluate the
same pair product at the default clamp and their timings are expected to
match.  ORCA's production mBecke path uses the ORCA Bragg radii and is a
separate preset; the radii table changes the geometry of the cells but not the
complexity class.

The molecules are the organic benchmark geometries distributed with
GPU4PySCF (`gpu4pyscf/benchmarks/molecules/organic`).  The PySCF level-3
topology is **geometry-independent**: it depends only on the nuclear charges,
so the point count is fixed by the molecular formula and is reproduced by any
geometry with that composition.  The partition timings do depend weakly on
the geometry through the pair distances; a different conformer changes the
last digits, not the ratios.

| Molecule | Formula | Atoms | Level-3 points |
|---|---|---:|---:|
| water | H<sub>2</sub>O | 3 | 33,698 |
| benzene | C<sub>6</sub>H<sub>6</sub> | 12 | 143,556 |
| vitamin C | C<sub>6</sub>H<sub>8</sub>O<sub>6</sub> | 20 | 247,664 |
| inosine | C<sub>10</sub>H<sub>12</sub>N<sub>4</sub>O<sub>5</sub> | 31 | 384,894 |
| bisphenol A | C<sub>15</sub>H<sub>16</sub>O<sub>2</sub> | 33 | 396,862 |
| Mg porphyrin | C<sub>20</sub>H<sub>12</sub>MgN<sub>4</sub> | 37 | 474,244 |
| penicillin V | C<sub>16</sub>H<sub>18</sub>N<sub>2</sub>O<sub>5</sub>S | 42 | 519,526 |
| ochratoxin A | C<sub>20</sub>H<sub>18</sub>ClNO<sub>5</sub> | 45 | 562,096 |
| cetirizine | C<sub>21</sub>H<sub>25</sub>ClN<sub>2</sub>O<sub>3</sub> | 52 | 630,608 |
| raffinose | C<sub>18</sub>H<sub>32</sub>O<sub>16</sub> | 66 | 793,292 |
| sphingomyelin | lipid (formula varies) | 84 | 970,552 |
| azadirachtin | C<sub>35</sub>H<sub>44</sub>O<sub>16</sub> | 95 | 1,150,994 |
| taxol (paclitaxel) | C<sub>47</sub>H<sub>51</sub>NO<sub>14</sub> | 113 | 1,374,804 |
| valinomycin | C<sub>54</sub>H<sub>90</sub>N<sub>6</sub>O<sub>18</sub> | 168 | 1,981,980 |

Timings are nanoseconds per grid point, minimum of several interleaved
repetitions unless a row is marked "one pass". Boost and background load move
individual runs by up to 5–30%; the ratios are more stable than the absolute
values. The 168-atom rows are expensive; they were measured once.

## 3. Partition weights

Full grid, batch 4096 points, 16 threads, native build, ns per grid point.
`ratio` is the scheme time divided by classic Becke; a value below 1 is faster
than Becke. This is the original four-scheme full-grid measurement run; the weight
kernels have not changed since, while the derivative tables below were
measured after the sparse LKO work.

| Molecule | Atoms | Points | Becke | SSF | mBecke | LKO | LKO / Becke | SSF / Becke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| water (H<sub>2</sub>O) | 3 | 33,698 | 10.3 | 9.6 | 11.1 | 10.7 | 1.04 | 0.93 |
| benzene (C<sub>6</sub>H<sub>6</sub>) | 12 | 143,556 | 14.1 | 16.7 | 14.2 | 14.2 | 1.00 | 1.18 |
| vitamin C (C<sub>6</sub>H<sub>8</sub>O<sub>6</sub>) | 20 | 247,664 | 18.3 | 26.9 | 18.1 | 17.5 | 0.96 | 1.47 |
| inosine (C<sub>10</sub>H<sub>12</sub>N<sub>4</sub>O<sub>5</sub>) | 31 | 384,894 | 27.0 | 42.7 | 26.6 | 26.0 | 0.96 | 1.58 |
| bisphenol A (C<sub>15</sub>H<sub>16</sub>O<sub>2</sub>) | 33 | 396,862 | 28.4 | 43.7 | 29.3 | 27.7 | 0.97 | 1.54 |
| penicillin V (C<sub>16</sub>H<sub>18</sub>N<sub>2</sub>O<sub>5</sub>S) | 42 | 519,526 | 40.8 | 50.6 | 39.6 | 36.7 | 0.90 | 1.24 |
| cetirizine (C<sub>21</sub>H<sub>25</sub>ClN<sub>2</sub>O<sub>3</sub>) | 52 | 630,608 | 54.4 | 64.7 | 54.6 | 52.4 | 0.96 | 1.19 |
| raffinose (C<sub>18</sub>H<sub>32</sub>O<sub>16</sub>) | 66 | 793,292 | 82.8 | 71.7 | 83.1 | 60.3 | 0.73 | 0.87 |
| sphingomyelin (lipid) | 84 | 970,552 | 138.7 | 76.1 | 137.4 | 49.9 | 0.36 | 0.55 |
| azadirachtin (C<sub>35</sub>H<sub>44</sub>O<sub>16</sub>) | 95 | 1,150,994 | 167.2 | 104.4 | 172.7 | 101.3 | 0.61 | 0.62 |
| taxol (C<sub>47</sub>H<sub>51</sub>NO<sub>14</sub>) | 113 | 1,374,804 | 236.5 | 115.0 | 240.5 | 119.9 | 0.51 | 0.49 |
| valinomycin (C<sub>54</sub>H<sub>90</sub>N<sub>6</sub>O<sub>18</sub>) | 168 | 1,981,980 | 562.4 | 155.4 | 567.2 | 150.6 | 0.27 | 0.28 |

Reading the table:

* SSF and LKO are slower than classic Becke for small molecules because the
  sparse bookkeeping (distance pass, sort or masks) dominates; both switch to
  a dense `f64x8` fallback below a size threshold.
* Above roughly 60 atoms the compact-support/sparse kernels win, and the
  advantage grows with molecule size. The pair work changes from $O(n^2)$ to
  $O(n\log n)$ (SSF) or $O(\lvert S\rvert\lvert T\rvert)$ (LKO).
* mBecke follows classic Becke at the default clamp: it dispatches to the
  classic pair product. Its exact nearest-first kernel is $O(n^2)$ scalar
  work and is used only when the clamp is not inert.

The sparse SSF kernel is also faster than the dense SSF
implementation; for valinomycin (168 atoms) the measured speedup was about
$7.4\times$. The LKO screened kernel is bitwise identical to the per-point
reference and the SSF dense/sparse paths agree to round-off.

## 4. Partition derivatives

### 4.1 Setup

All derivative numbers are ns per grid point, 16 threads, native build,
PySCF level 3, partition evaluation only.

* **Gradient, four selected rows:** the whole grid, 1024-point `fill`
  ranges.
* **Gradient, all rows:** 50,000 points (a full grid would write a
  $3n_\mathrm{atom}\times n_\mathrm{grid}$ array).
* **Hessian, four selected rows:** the whole grid, 4096-point ranges.

mBecke derivatives use the classic Becke backend at the default clamp, so
their timings are the Becke column.

### 4.2 Gradient

Whole grid, four selected rows:

| Molecule | Atoms | Points | Becke | SSF | LKO | LKO / SSF |
|---|---:|---:|---:|---:|---:|---:|
| vitamin C | 20 | 247,664 | 69 | 129 | 101 | 0.78 |
| inosine | 31 | 384,894 | 165 | 181 | 173 | 0.96 |
| bisphenol A | 33 | 396,862 | 177 | 175 | 122 | 0.70 |
| penicillin V | 42 | 519,526 | 209 | 225 | 119 | 0.53 |
| valinomycin (one pass) | 168 | 1,981,980 | 3,268 | 679 | 492 | 0.72 |

All rows, 50,000 points:

| Molecule | Atoms | Becke | SSF | LKO |
|---|---:|---:|---:|---:|
| vitamin C | 20 | 125 | 150 | 140 |
| valinomycin | 168 | 3,804 | 1,165 | 1,502 |

The LKO gradient uses the dense eight-point lane kernel below 32 atoms and a
masked union-list kernel above it; the two agree to signed zero. With a small
selection the sparse kernel is faster than SSF from about 33 atoms; with all
rows selected the selected-row accumulation dominates and LKO is slower than
SSF at 168 atoms. The classic Becke gradient uses its lane kernel at all
sizes; SSF uses the sparse active-pair kernel.

### 4.3 Hessian

Whole grid, four selected rows, 4096-point ranges:

| Molecule | Atoms | Points | Becke | SSF | LKO | LKO / SSF |
|---|---:|---:|---:|---:|---:|---:|
| vitamin C | 20 | 247,664 | 615 | 239 | 638 | 2.67 |
| inosine | 31 | 384,894 | 1,626 | 330 | 994 | 3.01 |
| bisphenol A | 33 | 396,862 | 1,790 | 314 | 1,094 | 3.49 |
| Mg porphyrin | 37 | 474,244 | 2,140 | 369 | 1,036 | 2.81 |
| penicillin V | 42 | 519,526 | 2,692 | 344 | 1,024 | 2.98 |
| ochratoxin A | 45 | 562,096 | 3,119 | 374 | 1,070 | 2.87 |
| cetirizine | 52 | 630,608 | 4,362 | 449 | 1,531 | 3.41 |
| valinomycin (one pass) | 168 | 1,981,980 | 61,390 | 1,411 | 3,202 | 2.27 |

mBecke uses the classic Becke Hessian backend at the default clamp, so its
numbers are the Becke column (the 168-atom entry is the pointwise path).

The LKO Hessian is pointwise (no lane variant) and dispatches between a dense
kernel below about 36 atoms and a sparse near-set kernel above it. The sparse
kernel visits only $T\times T$ pairs, keeps one support row per survivor
instead of the dense $3n_\mathrm{atom}\times3n_\mathrm{atom}$
log-derivative table, skips pairs whose two cell products are exactly zero,
and builds the summed-over-first-index correction only when the generating
atom is selected. It is bitwise identical to the dense reference up to signed
zero. Relative to the dense LKO kernel the sparse path is about
$1.3$–$2.2\times$ faster on the 20–42-atom full grids and about
$10\times$ faster per point at 168 atoms (33,385 → 3,202 ns/point on the full
grid). LKO overtakes classic Becke from about 31 atoms and remains about
$2.3$–$3.5\times$ slower than the SSF sparse Hessian.

The all-row Hessian is output-bound (a dense
$3n_\mathrm{atom}\times3n_\mathrm{atom}$ block per point); the
contracted `ddc` path and small selections are where the sparse structure
pays off most.

### 4.4 Backends and kernel crossovers

The cutoffs below are empirical and depend on the compiled ISA; they are
validated by the bitwise and finite-difference tests, not by the cutoffs
themselves.

| Kernel | Small-molecule path | Larger path |
|---|---|---|
| LKO weights | dense lanes | sparse masked lanes, $n \ge 40$ (AVX-512; 32 AVX2, 24 baseline) |
| LKO gradient | dense lanes | masked union-list lanes, $n \ge 32$ |
| LKO Hessian | dense pointwise | sparse near-set, $n \ge 36$ |
| classic Becke Hessian | lane (scratch ≤ 4 MiB) | pointwise scalar |
| SSF weights and derivatives | dense lanes below the cut | sparse candidate scan |

The classic Becke lane/pointwise gate is a scratch-footprint test, so it also
accounts for the number of selected rows. Measured with all rows selected:

| Molecule | Atoms | Lane | Pointwise |
|---|---:|---:|---:|
| penicillin V | 42 | 6,498 | 8,825 |
| cetirizine | 52 | 14,063 | 16,014 |
| tamoxifen | 57 | 19,514 | 18,129 |

On the reference CPU the lane path wins through about 52 atoms with all rows,
and through about 57 atoms with four rows. The
`DerivOptions::hessian_backend` option (`Lane` or `Pointwise`) forces a path
for testing.

Classic Becke derivatives, build comparison. This table uses a different
workload (50,000 gradient points; 128 Hessian points) and is only meant to
compare build targets:

| Case | Default build | Native build |
|---|---:|---:|
| valinomycin (C<sub>54</sub>H<sub>90</sub>N<sub>6</sub>O<sub>18</sub>) gradient, all 168 rows | 8,650–10,100 | 3,900–4,600 |
| vitamin C (C<sub>6</sub>H<sub>8</sub>O<sub>6</sub>) Hessian, 20 rows | 3,700–4,800 | 1,300–2,700 |
| valinomycin Hessian, 4 rows | 79,000–104,000 | 64,000–82,000 |
| valinomycin Hessian, all 168 rows, 20 points | 515,000–636,000 | 465,000–625,000 |
| vitamin C combined `dw + dc + ddc` | 3,900–4,900 | 1,300–2,600 |

`fill` is parallel per call: a 64-point Hessian range is split into
four-point chunks and the per-call Rayon setup becomes comparable to the
per-point work. Single-threaded runs differ by only a few percent between
64-point and 4096-point ranges, so this is a small-range scheduling cost, not
a scratch or algorithmic one. Prefer ranges of a few thousand points when
calling `fill` in a loop.

## 5. Complexity summary

| Operation | Classic Becke | SSF | LKO | mBecke |
|---|---|---|---|---|
| weight, pair work | $O(n^2)$ | $O(n\log n + n_\mathrm{surv} n)$ | $O(\lvert S\rvert\lvert T\rvert)$ | $O(n^2)$ |
| gradient | $O(n^2)$ per point | active pairs only | masked sparse lanes above the cut (dense lanes below) | classic backend |
| Hessian | $O(n^3)$ cross term, $O(n^2)$ memory | sparse active-pair rows | pointwise; sparse near-set pairs above the cut | classic backend |
| default small-molecule path | dense lanes | dense SSF lanes below the cut | dense LKO lanes / dense Hessian below the cut | exact nearest-first if needed |

The $O(n^3)$ Becke Hessian is the rank-one cross term; the selected-row
assembly avoids materialising all rows, so a small selection costs
$O(n^2 + n_\mathrm{sel} n^2)$ rather than $O(n^3)$ output.

## 6. Reproducing a measurement

The examples accept geometry files in XYZ format (Ångström, converted to
Bohr by the example).  The benchmark geometries are the `NNN_Name.xyz` files
of the GPU4PySCF organic set, for example `020_Vitamin_C.xyz`,
`066_Raffinose.xyz` and `168_Valinomycin.xyz`.  Any geometry with the same
formula gives the same PySCF level-3 point count; the timings change only at
the few-percent level between conformers.

```text
# Weights, full grid (repeat with becke, stratmann, gaussian_ss or mbecke).
cargo run --release --example bench_weights -- 168_Valinomycin.xyz 3 4096 lko

# Gradient, four selected rows, full grid.
cargo run --release --example bench_deriv -- 168_Valinomycin.xyz 3 grad 1024 4 10000000 0 auto lko

# Gradient, all rows, 50,000 points.
cargo run --release --example bench_deriv -- 168_Valinomycin.xyz 3 grad 1024 168 50000 0 auto lko

# Hessian, four selected rows, full grid (10000000 is clamped to it).
cargo run --release --example bench_deriv -- 020_Vitamin_C.xyz 3 hess 4096 4 10000000 0 auto lko
```

The argument order for `bench_deriv` is
`<xyz> <level> <mode> [batch] [nsel] [points] [block] [backend] [scheme]`.
`bench_weights` takes `<xyz> <level> <batch> [scheme]`.  Pass `becke`,
`stratmann`, `gaussian_ss`, `mbecke` or `lko` as the scheme.  Use a fixed thread count and
pin the process when comparing runs:

```bash
taskset -c 0-15 env RAYON_NUM_THREADS=16 \
    cargo run --release --example bench_weights -- 168_Valinomycin.xyz 3 4096
```

The numbers in this page are indicative. Re-measure on the target CPU with
the target build; a single-thread comparison isolates the algorithm from
Rayon scheduling.
