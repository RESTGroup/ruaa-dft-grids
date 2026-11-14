---
name: release
description: Prepare, verify, and publish a crates.io release; the publish command itself is human-only.
---

# Release

Use this skill to prepare a release candidate. An agent may run every
preparation and verification step; only a human runs `cargo publish`.

## Preconditions

* The release gates in
  [`../../decisions/0003-release-gates.md`](../../decisions/0003-release-gates.md)
  are satisfied.
* The working tree is clean and the branch is the intended release branch.
* The version in `Cargo.toml` is correct and the lockfile is committed if the
  repository tracks it.
* The repository URL is filled in (`Cargo.toml`, `pyproject.toml`).
* The history that will be published has passed the provenance review. A
  clean working tree does not clean old commits; see
  [`../../rules/provenance.md`](../../rules/provenance.md).
* The licence and the redistribution terms for bundled reference data and
  parameter tables have been reviewed and recorded.

## Verification

Run from the repository root:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --workspace
RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps
cargo package --list
```

Then inspect the package list and the packaged archive:

```bash
cargo package --list | grep -E '^(agents/|\.github/|AGENTS|CLAUDE|.*\.local)'   # must be empty
cargo package --list | wc -l
```

The package must not contain `agents/`, `AGENTS.md`, local configuration,
scratch directories, raw probe output, build artifacts, or a stale lockfile.
The validation probes under `validation/` are intentionally included because
the comparison tests need them.

Finally, run the registry dry run:

```bash
cargo publish --dry-run
```

`--dry-run` packages and verifies without uploading. It may update the local
index; that is expected.

## Version and tag

1. Update the version in `Cargo.toml` and the binding crates if they share the
   release.
2. Add or update the changelog with the user-visible changes and any breaking
   API change.
3. Commit the release preparation with the repository's commit convention.
4. Create an annotated tag, for example `v0.1.0`.
5. A human runs `cargo publish` from the clean, tagged commit.

Do not publish the core crate and the binding crates in one step unless all of
them have passed their own packaging checks. The core crate is the first
release candidate; the Python and C bindings can follow separately.

## Bindings (separate release)

Both binding crates set `publish = false`; they are not part of the crates.io
core release. If they are released:

* **Python.** Build wheels for the supported platforms with the toolchain in
  `docs/python-interface.md`, run the Python tests under
  `crates/ruaa-dft-grids-python/tests/`, and verify the package metadata and
  `pyproject.toml` URLs. Publish to PyPI only after the wheel matrix passes.
* **C.** Build the shared and static libraries, regenerate and diff the header
  with cbindgen, run `crates/ruaa-dft-grids-capi/tests/smoke.sh`, and package
  the header, library and pkg-config file together. Bump the ABI version if
  the C surface changes.
* Do not tag a binding release until its own CI job is green and its metadata
  is complete. The binding version may lag the core crate, but it must record
  which core version it wraps.

## After publishing

* Verify the docs.rs build and the crate page.
* Verify the tag points at the published commit.
* Record the release in the changelog or release notes.
* If a release is broken, prefer a follow-up patch over a yank; yank only when
  the published artifact is unusable or unsafe.
* Do not reuse a version number after a yank.

## Rollback

`cargo yank --version <version>` prevents new resolutions but does not delete
the artifact. Publish a fixed patch version and document the reason. A yank is
a human decision.

## Completion checklist

* [ ] All local gates pass on the release commit.
* [ ] `cargo package --list` contains no repository-process material.
* [ ] `cargo publish --dry-run` succeeds.
* [ ] The repository URL and metadata are correct.
* [ ] The history and data-licence review are recorded.
* [ ] A human has explicitly approved the publish.
