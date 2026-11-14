# 0002: Extension and accelerator strategy

- **Status:** proposed
- **Date:** 2026-11-13
- **Deciders:** maintainers

## Context

The crate is near a 0.1 release. Its numerical core is validated at the point
and weight level, including bitwise ORCA cases. The likely future work is:

* more partition schemes, radii tables, radial maps, and pruning policies;
* possibly another program backend;
* a non-CPU (accelerator) realization and optimization.

The current design is enum-and-match dispatch with monomorphic CPU kernels. The
partition hot path uses `f64`, fixed eight-lane `f64x8` helpers, Rayon, and
compile-time target features. `GridPlan` is a flat, geometry-independent
topology; `PartitionEvaluator` and `PartitionDerivEvaluator` stream ranges.
`DerivScheme` is an existing trait-based extension point for derivative
kernels.

A large refactor before 0.1 would put the validated numerics at risk for no
immediate user benefit. A device abstraction added now would be speculative:
the target hardware, precision, and transfer model are not decided.

## Decision

1. **Do not perform a big-bang refactor before 0.1.** Keep the current module
   layout and public paths. Limit pre-0.1 changes to small API hardening.
2. **Harden public enum growth.** Audit public enums that dispatch over
   extensible sets and add `#[non_exhaustive]` where future variants are
   expected. The backend, scheme, radial, pruning, radii, adjustment and
   post-prune selectors are non-exhaustive. Remove or implement dead public
   variants rather than leaving a placeholder that always errors.
3. **Extend the existing seams first.** A new radial map implements
   `RadialMap` and pairs with a `RadialScalePolicy` in `RadialSpec`; a new
   pruning policy implements `Pruning` and consumes `PruneContext`; a new
   partition scheme adds a `SchemeOps` implementation plus `DerivScheme` for
   its derivatives; a new backend adds a `GridSpec` variant and a C ABI
   unknown-backend branch. These changes do not require a new architecture.
4. **Defer the backend boundary until a new program backend is real.** When a
   fourth independent backend is scheduled, introduce one sealed backend
   specification and move the per-backend builders behind it, with `pub use`
   shims for public paths. Do not do this speculatively.
5. **Do not add a device generic or a runtime device enum to the core.** A
   non-CPU path starts as a separate crate or a non-default feature. The CPU
   path remains the reference implementation and the default.
6. **Expose data, not internals, for an accelerator.** When a prototype needs
   the molecule pair topology, add a read-only public view rather than making
   the private partition tables public wholesale. Keep `GridPlan` as the
   device-neutral topology.
7. **Keep `f64` as the public numeric type.** A lower-precision accelerator
   path is a separate implementation with explicit conversion and parity
   tests; it is not a generic parameter threaded through every type.
8. **Mechanical splits are allowed after 0.1 when a file blocks work.** Split
   behind re-exports, preserve public paths, and do not change arithmetic,
   table values, loop order, or evaluation order in the same change.

## Consequences

* The 0.1 release is lower risk and can ship without a speculative abstraction.
* Adding a backend or scheme variant is no longer a source-breaking change for
  downstream exhaustive matches; the extensible selectors are
  `#[non_exhaustive]`.
* A future accelerator may duplicate some host-side table setup. That is
  acceptable until the interface is justified by a working prototype.
* The first accelerator integration will need a small public data view or a
  crate-internal feature; it should not require rewriting the CPU kernels.
* If a new backend arrives sooner than expected, the backend-boundary
  refactor moves up; it should still be separate from the numerical change.

## Open questions

* Separate crate versus feature flag for the accelerator.
* Whether the first accelerator target needs `f32` or can use `f64`.
* Whether the public `params` tables should become accessor-only.
* Whether `rayon` and `lightweight-simd` should become optional features.
* When the consumer-facing API is frozen.
