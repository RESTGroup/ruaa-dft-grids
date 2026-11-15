# Development sketch

This is a working sketch, not an API contract. It records how the project is
put together, where it is safe to extend, and what kind of refactor is worth
doing. Update it when an architectural decision changes; record the decision
itself in [`decisions/`](decisions/).

## 1. Current shape

The repository contains one Rust package, `.` (`ruaa-dft-grids`), plus
repository-only tooling: `agents/` for process material and `python/` for the
dependency-free preset reader.  The in-tree Python/C binding crates were
removed from the package (decision 0004) and live in the maintainer's local
workspace until they are ported to the current API; `cargo test` builds only
the crate.

### Layers

```text
data / leaf      params, lebedev, angular directions, atomgrid, error
constructors     radial, prune, radii, pyscf
orchestration    grid (GridSpec/GridPlan/BatchBuffer), moleculegrid, order
partition        config, value kernels, streaming derivatives
tooling          python preset reader and canonical fixture (repository-only)
```

The main consumer path is:

```text
GridSpec -> GridPlan -> PartitionEvaluator -> BatchBuffer
                     \-> MolecularGrid (materialised convenience output)
```

`GridPlan` is the geometry-independent topology: flat local coordinates,
quadrature weights, and a parent-atom index. `PartitionEvaluator` fills a
caller-selected point range into a reusable `BatchBuffer`. The derivative path
is separate and streams gradient/Hessian blocks through
`PartitionDerivEvaluator`.

### Where the extension pressure is

* `src/grid.rs` owns `GridSpec`, the three backend spec structs, `GridPlan`,
  and `BatchBuffer`. It imports the backend builders.
* `src/radial/mod.rs` is the unified radial selector. A new radial map is one
  enum variant plus the node/increment kernel and its element-scale rule.
* `src/prune/mod.rs` is the pruning selector.
* `src/radii.rs` already has a `Custom` table, so a new named table is usually
  data plus one variant.
* `src/partition/mod.rs` owns `PartitionScheme`, `PartitionConfig`, the
  molecule tables, and the value evaluator. `src/partition/deriv/` owns the
  streaming derivative layer and exposes `DerivScheme`, a real trait-based
  extension point for a future partition algorithm.
* `src/partition/simd.rs` and `src/partition/kernel.rs` contain the fixed-lane
  CPU kernels. They are private and are the least portable part of the crate.

The current design is an enum-and-match design, not a trait-object design. That
is deliberate: it keeps the hot paths monomorphic, keeps rustdoc readable, and
lets the compiler prove exhaustiveness inside the crate. The cost is that
adding a variant is a source-breaking change for downstream code that matches
the enum exhaustively. Public enums that are expected to grow should be
`#[non_exhaustive]` before 0.1; see [`rules/api-stability.md`](rules/api-stability.md).

## 2. Invariants

Treat these as the specification unless a deliberate, documented change says
otherwise.

1. **Units.** Coordinates and lengths are in Bohr. The Python layer converts at
   its boundary; the Rust core does not guess units.
2. **Weight split.** Angular weights sum to one; the radial weight carries the
   `4*pi*r^2*dr` measure. The quadrature weight is `q = w_rad * w_ang`, and the
   total weight is `q * P` with the normalized partition fraction `P`.
3. **Build is not prune.** Builders return every point. `PostPrune` and
   `GridOrder` are explicit post-build operations; the build never filters or
   reorders points implicitly.
4. **The plan is geometry-independent** unless the spec says otherwise. The
   Gaussian molecular-cutoff mode is the documented exception; it is reported
   through `GridSpec::geometry_dependent`.
5. **Streaming.** The evaluator never materialises a dense
   `n_atom x n_grid` object. A change that needs one must provide a separate
   opt-in API rather than growing the default path.
6. **Numerics are frozen during refactors.** A refactor must not change
   arithmetic, table values, lane width, loop order, or evaluation order. The
   bitwise ORCA tests and the checked-in goldens are the guard.
7. **Errors are typed.** Construction, evaluation, ordering, and unsupported
   combinations return `Error`; do not turn a recoverable input error into a
   panic.
8. **The core is offline-testable.** `cargo test` must not require ORCA, PySCF,
   Gaussian, network access, or a local scratch directory.

## 3. Extension seams

The detailed recipe is in [`skills/extend/SKILL.md`](skills/extend/SKILL.md).
This table
is the summary.

| Change | Main edit points | Test expectation | Main risk |
|---|---|---|---|
| Radial map | `src/radial/mod.rs` (`RadialMap`, `RadialScalePolicy`, `RadialSpec`), `src/radial/kernels.rs` | node/weight values, quadrature moments, explicit or element scale | a new map with a different weight convention |
| Pruning scheme | `src/prune/mod.rs` (`PruneContext`, `Pruning`), a policy module | region boundaries, per-shell sizes, fallback warnings | angular sizes not supported by the target backend |
| Radii table | `src/radii.rs`, `src/params.rs` | coverage, `Z` range, adjustment interaction | mixing tables from different programs |
| Partition scheme | `SchemeOps` impl + `ops_for` in `src/partition/mod.rs`, new kernel module, `src/partition/deriv/` | partition sums, finite-difference gradients/Hessians | derivative backend; the value seam is still in-crate, not a public trait |
| Backend preset | `src/grid.rs` (`GridSpec`, spec struct, `From`), backend module | reference probe + round-trip through `build_molecular_grid` | geometry-dependent topology and partition defaults |
| Angular rule | `src/angular.rs`, `src/lebedev.rs` | exact direction set, sum-to-one weights, moments | orientation and weight normalization |
| Ordering policy | `src/order.rs`, `src/moleculegrid.rs` | permutation invariants, derivative parity | parent-atom attribution after permutation |

Seams worth knowing:

* **`DerivScheme`** is the supported extension point for a new partition
  derivative algorithm. Implement the trait and install it with
  `PartitionDerivEvaluator::with_scheme`; the evaluator keeps the streaming,
  selection, contraction, and parallel-reduction logic. The built-in schemes
  in `src/partition/{becke,stratmann,lko}/deriv.rs` are the reference
  implementations.
* **`SchemeOps`** is the private value-path equivalent. Implement it in
  `src/partition/mod.rs`, add one arm in `ops_for`, and the point, blocked,
  parallel and switch paths all dispatch through the new scheme. `Scratch`
  only needs a new variant if the kernel needs a new scratch type.
* **`RadialMap` + `RadialScalePolicy`** separate the node mapping from the
  element-to-xi rule. Use `RadialSpec` for a custom map; `RadialScalePolicy::Explicit`
  covers a map with no element rule.
* **`Pruning` + `PruneContext`** is the common pruning contract. `Prune`
  implements it; a new policy can be used through `&dyn Pruning` without
  editing the enum.
* **`RadiiTable::Custom`** already covers a caller-supplied table. A new
  named table is only worth adding when it has a clear provenance and a
  documented element range.

### What not to do

* Do not add a generic scalar parameter (`f32`/`f64`) to the core. The public
  API is `f64`; a lower-precision accelerator path should be a separate
  backend with its own conversions and parity tests.
* Do not add a public `dyn Backend` trait just to avoid a `match`. The current
  enum is cheap and gives better errors and documentation. Add a trait only
  when there are at least two independent backend implementations to justify
  it, and keep the public spec enum as the stable surface.
* Do not put a new program's parameter tables in `params.rs` without a
  redistribution decision. A table that cannot be published must stay in a
  separate package, not behind a private module in the core crate.
* Do not change a tolerance, delete a golden row, or refresh a hash to make a
  failing test pass. Investigate the construction first.

## 4. Non-CPU strategy

The current accelerator is the CPU: `f64`, fixed eight-lane `f64x8` kernels,
Rayon over point ranges, and compile-time target features. There is no runtime
device dispatch and no GPU dependency. That is a reasonable 0.1 position.

The expensive part is partition evaluation and its derivatives, not grid
generation. Radial/angular construction is small and easy to move. The
partition path is where a device backend matters.

### Recommended path

**Phase A — keep the CPU reference and expose data, not internals.**

* Keep the CPU implementation as the reference and the default.
* Profile before changing anything. The existing `examples/bench_*` programs
  and `docs/performance.md` are the starting point.
* When an accelerator prototype needs it, add a read-only public view of the
  molecule pair topology (distances, radii, adjustment factors) rather than
  making `PartitionTables` public wholesale.
* Keep `GridPlan` as the device-neutral topology. Its flat arrays and
  `atom_range` indexing are already a good transfer format.

**Phase B — separate crate or feature-gated backend.**

* Put the first non-CPU implementation in a separate crate (for example a
  workspace member such as `ruaa-dft-grids-accelerator`) or behind a
  non-default feature. Do not make the core crate depend on a device runtime.
* Give the accelerator its own buffer and error types if its memory model
  differs. Do not force device buffers into `BatchBuffer`.
* Validate bitwise or within a tight tolerance against the CPU path on the same
  case set. The CPU path remains the oracle.
* Keep the public Rust API unchanged for CPU users. A caller who does not ask
  for the accelerator should not pay for it in build time or dependencies.

**Phase C — optimize and specialize.**

* Mixed precision, fused kernels, and device-specific layouts belong here, not
  in Phase A.
* Record measurements in `docs/performance.md` with the hardware, compiler,
  thread count, and workload.
* Only after the accelerated path is correct and measured should it become a
  default for any workload.

### What not to do now

* Do not add a `Device` generic parameter to `GridSpec`, `PartitionConfig`, or
  `PartitionEvaluator`.
* Do not add a runtime `Backend` enum with unimplemented variants. A public
  variant that only returns an `Unsupported…` error is a maintenance and API
  cost; the removed `Engine::Neighbor` is the precedent.
* Do not make Rayon or `lightweight-simd` part of the public type signatures.
  They are implementation details; feature-gating them later should be an
  additive change.

## 5. Refactor stance

The current shape is publishable for 0.1. A big-bang rewrite would put the
validated numerics and the public paths at risk for no immediate user benefit.
The recommended order is:

1. **Before 0.1: small API hardening only.**
   * Audit public enums and add `#[non_exhaustive]` where future variants are
     expected. The backend, scheme, radial, pruning, radii, adjustment and
     post-prune selectors are non-exhaustive, and genuinely closed enums say
     so in their documentation.
   * Remove or implement dead public variants instead of publishing them.
   * Keep the `Cargo.toml` package metadata and `exclude` list current.
   * The offline suite and the optional PySCF cross-check are the validation
     surface; no validation tree is bundled in the crate.
2. **After 0.1: mechanical splits when a file blocks work.**
   * `src/partition/mod.rs` now has the private `SchemeOps` seam but still
     mixes configuration, tables, evaluator, and tests; it can be split
     behind `pub use` shims.
   * `src/partition/deriv/mod.rs` can be split into evaluation, streaming, and
     Hessian modules without changing the public API.
   * `src/params.rs` can be split by namespace with re-exports.
   * Every split must preserve public paths and leave the goldens unchanged.
3. **When a fourth program backend is scheduled: introduce one backend
   boundary.**
   * A sealed `BackendSpec` (or equivalent) with `atomic_plan`, `partition`,
     and `geometry_dependent` would stop `src/grid.rs` from importing every
     backend.
   * Move spec structs and builders to a `backend/` module, keep `grid`
     re-exports for path compatibility, and add a single `GridSpec` variant.
   * This is a medium-sized refactor. Do it when the second independent
     backend is real, not speculatively.
4. **Never mix a refactor with an arithmetic change.** Land the split with
   identical outputs, then change the algorithm in a separate commit with its
   own validation.

## 6. Roadmap

This is a sequencing sketch, not a promise of dates.

### 0.1.0 — first public core release

* Close the release gates in
  [`decisions/0003-release-gates.md`](decisions/0003-release-gates.md):
  CI, package audit, history review, provenance/licence review, metadata, API
  audit.
* Keep the numerical surface as it is. New schemes are not a release blocker.
* Publish the core crate. The Python preset reader is repository tooling,
  not a release artifact.

### 0.1.x — maintenance

* Bug fixes, additional validation cases, documentation corrections.
* Small additive APIs that do not change existing construction or weights.
* Keep the offline test suite and the goldens green.

### 0.2 — extension and coverage

* Add the next public schemes/radii/radial maps/pruning policies, each with a
  reference or an analytic test.
* Decide whether the backend boundary is needed for the next program backend.
* Consider optional `parallel`/`simd` features if downstream users need a
  serial or minimal build.
* Add a changelog and a public compatibility page if they are not in 0.1.

### 0.3 — accelerator work

* Prototype the non-CPU path as described in section 4.
* Keep the CPU path canonical; publish parity and performance results.
* Do not change the public semantics of `GridPlan`, `BatchBuffer`, or the
  derivative contracts for a device implementation.

### 1.0 — stability

* Freeze the public API only after the extension seams and the accelerator
  story have settled.
* Commit to the semver policy and the MSRV policy at that point.

## 7. Definition of done for a change

A change is complete when all of the following hold:

* The new behaviour has a test that fails without the change.
* The offline suite passes: `cargo test`.
* Formatting and lints pass: `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`.
* Documentation builds without warnings:
  `RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps`.
* Any public API change has a doc comment, an example where useful, and a note
  in the changelog or release record.
* Any numerical change has a validation reference or a finite-difference /
  analytic check, and the golden diff is reviewed.
* The package still excludes repository-process material:
  `cargo package --list` shows no `agents/`, local configuration, or scratch
  paths.
* `agents/development.md` or the relevant rule is updated if the change alters
  the architecture or the extension procedure.

## 8. Open questions

These are deliberately unresolved; record the answer as a decision when it is
made.

* Should `rayon` and `lightweight-simd` become optional features, and which
  should be on by default?
* Should the public `params` tables become accessor-only, or stay public data?
* Is a sealed backend trait worth introducing for the next program backend, or
  is another `GridSpec` variant sufficient?
* Should the value-path `SchemeOps` become a public extension trait, or is an
  in-crate scheme plus a custom `GridPlan` sufficient?
* Should `RadialSpec` and `Pruning` be accepted directly by the backend specs,
  or remain tools for the custom-plan path?
* Does the first non-CPU target need `f32`, or is `f64` acceptable on the
  target hardware?
* Is the accelerator a separate crate, a feature, or a private downstream
  integration?
* When is the public API frozen: at 1.0, or earlier for the consumer-facing
  subset?
