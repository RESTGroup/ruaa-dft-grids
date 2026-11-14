# Test targets

`cargo test` builds one integration-test binary per top-level `*.rs` file.
The published crate ships no external-program reference probes; the targets
below check mathematical invariants, the built-in PySCF construction and the
public API.

| target | covers |
|---|---|
| [`batch_api.rs`](batch_api.rs) | plan/range API, batch buffers, partition-independent topology |
| [`error_api.rs`](error_api.rs) | typed errors on the construction path |
| [`grid_coverage.rs`](grid_coverage.rs) | every radial map, angular layout, pruning and partition scheme through the public API |
| [`grid_deriv.rs`](grid_deriv.rs) | grid-native partition derivatives |
| [`order_api.rs`](order_api.rs) | permutations and ordering invariants |
| [`p1_regressions.rs`](p1_regressions.rs) | normalization at a symmetric cage, molecular-cutoff presets, custom atomic grids |
| [`partition_deriv.rs`](partition_deriv.rs) | analytic Becke/SSF/SS/LKO/mBecke gradients and Hessians against finite differences |
| [`partition_weights.rs`](partition_weights.rs) | partition weights at caller-supplied points and the table path |
| [`post_prune.rs`](post_prune.rs) | `PostPrune` policies and retained/pruned splits |
| [`user_workflows.rs`](user_workflows.rs) | consumer workflows: PySCF and data-driven presets -> `MolecularGrid`, weight derivatives, sorting, prune/sort/derive; mirrors [`../docs/user-guide.md`](../docs/user-guide.md) |

The unit tests in `src/` cover the radial kernels, angular rules, pruning
schemes, partition kernels and the preset data model.

Cross-code reference probes are maintained outside the published package; the
crate's tests are offline and self-contained.
