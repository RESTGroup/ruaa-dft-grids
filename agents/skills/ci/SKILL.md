---
name: ci
description: Add or maintain the continuous-integration gate for the Rust core and the repository tooling.
---

# Continuous integration

Use this skill when adding a CI workflow or changing what CI checks. The core
workflow is [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml);
this skill is its design and local mirror. Update both together.

## Local gate

Run these from the repository root before proposing a CI change:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps
python -m unittest discover -s python/tests -v
cargo package --list
```

`cargo test` is the offline core path. It must not gain a dependency on ORCA,
PySCF, Gaussian, `strace`, network access, or a maintainer work directory.
The Python preset tooling is dependency-free (Python 3.11+); it runs as its own
job so a Python failure does not hide a core failure.

## Minimum CI jobs

| Job | Command | Notes |
|---|---|---|
| format | `cargo fmt --check` | fast, no build |
| lint | `cargo clippy --all-targets --locked -- -D warnings` | root package |
| test | `cargo test --locked` | offline, default features |
| docs | docs build with `-D warnings` | includes the KaTeX header |
| package | `cargo package --list --locked` | checks the published file set |
| python-preset | `python -m unittest discover -s python/tests -v` | Python 3.11+, no Rust |
| msrv | `cargo check --all-targets --locked` on 1.87 | required for the declared MSRV |
| package-verify | `cargo package` + packaged tests | builds the tarball |
| dry-run | `cargo publish --dry-run --locked` | manual `workflow_dispatch` only |
| release-plz-pr | `release-plz release-pr` | manual release-PR workflow |
| release-plz | `release-plz release` | manual publish workflow |

Optional jobs, once verified on a maintainer machine:

* a SIMD job with `RUSTFLAGS="-C target-cpu=x86-64-v3"` for the lane kernels;
* a single-thread job with `RAYON_NUM_THREADS=1` for derivative reductions;
* a macOS and Windows matrix after the Ubuntu jobs are stable.

Do not make an external quantum-chemistry program a required CI dependency.
Reference generation is a maintainer-side action; the checked-in probes are
what CI reads.

## Workflow guidance

* Pin the Rust toolchain with `dtolnay/rust-toolchain` or an equivalent action,
  and keep the MSRV job separate from the stable job.
* Cache `~/.cargo` and `target`; the workspace has large build directories and
  a cold build is the dominant cost.
* Run jobs on `ubuntu-latest` first. Add the OS matrix only when the code and
  the dependencies are known to build there.
* Keep the package job read-only. Do not run `cargo publish` from a pull
  request or from a workflow that untrusted code can modify.
* The Python preset job needs only a Python interpreter and the standard
  library; it does not build a wheel or a Rust extension.

## Release workflows

Release automation lives in
[`.github/workflows/release-plz-pr.yml`](../../../.github/workflows/release-plz-pr.yml)
and
[`.github/workflows/release-plz.yml`](../../../.github/workflows/release-plz.yml).
Both are `workflow_dispatch` only; do not add `push` or `pull_request`
triggers.  The publish job is the only job that needs
`CARGO_REGISTRY_TOKEN`, and it releases only `ruaa-dft-grids`; the Python
preset reader is repository tooling.  See
[`../release/SKILL.md`](../release/SKILL.md) for the release procedure.

`release-plz.toml` is repository process material and must stay out of the
crate package.

## Package check

The published crate must not contain repository-process material:

```bash
cargo package --list | grep -E '^(agents/|python/|release-plz\.toml|\.github/|AGENTS|CLAUDE|.*\.local)'
```

The command should print nothing. The `exclude` list in `Cargo.toml` is the
source of truth; update it when a new repository-only directory is added.

## When CI fails

1. Reproduce the exact command locally with the same toolchain.
2. Distinguish a code failure from a toolchain/feature failure.
3. Fix the code or the workflow, not the assertion. Do not add
   `continue-on-error` to hide a required job.
4. If the failure is platform-specific, keep the job non-blocking only while a
   tracking issue exists, and say so in the workflow comment.

## Completion checklist

* [ ] Every command in the local gate passes locally.
* [ ] The workflow does not require an external QC program.
* [ ] The package check excludes `agents/`, `python/`, `release-plz.toml` and
      local configuration.
* [ ] The MSRV job is only enabled after it passes on the declared version.
* [ ] No secrets are required for the core test jobs.
* [ ] The release workflows remain manual-dispatch only.
