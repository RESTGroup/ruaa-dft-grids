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
* The repository URL is publicly reachable (a private repository returns 404
  to anonymous users).
* The history decision has been recorded. A clean working tree does not clean
  old commits; see [`../../rules/provenance.md`](../../rules/provenance.md).
* The licence and the redistribution terms for bundled reference data and
  parameter tables have been reviewed and recorded.

## Verification

Run from the repository root:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps
cargo package --list
```

Then inspect the package list and the packaged archive:

```bash
cargo package --list | grep -E '^(agents/|python/|\.github/|AGENTS|CLAUDE|.*\.local)'   # must be empty
cargo package --list | wc -l
```

The package must not contain `agents/`, `python/`, `AGENTS.md`, local
configuration, scratch directories, raw probe output, build artifacts, or a
stale lockfile.  It does contain the tests, examples, docs and the canonical
preset fixture; program-specific reference data stays in the local workspace.

Finally, run the registry dry run:

```bash
cargo publish --dry-run
```

`--dry-run` packages and verifies without uploading. It may update the local
index; that is expected.

## Version and tag

1. Update the version in `Cargo.toml`.
2. Add or update the changelog with the user-visible changes and any breaking
   API change.
3. Commit the release preparation with the repository's commit convention.
4. Create an annotated tag, for example `v0.1.0`.
5. A human runs `cargo publish` from the clean, tagged commit.

The core crate is the only release artifact.  The Python preset reader in
`python/` and the agent material in `agents/` are repository tooling; they are
excluded from the package and are validated by the CI preset-tooling job.

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
