---
name: release
description: Prepare, verify, and publish the Rust crate with release-plz; the publish workflow is human-triggered.
---

# Release

Use this skill to prepare a release candidate.  An agent may run every
preparation and verification step; only a human dispatches the publish
workflow in GitHub Actions.  There is no local publish path.  The Python
preset reader in `python/` is repository tooling and is not part of the
release (decision
[`0005`](../../decisions/0005-defer-python-release.md)).

## Preconditions

* The release gates in
  [`../../decisions/0003-release-gates.md`](../../decisions/0003-release-gates.md)
  are satisfied.
* The working tree is clean and the branch is the intended release branch.
* The version in `Cargo.toml` is correct and the lockfile is committed if the
  repository tracks it.
* The repository URL is publicly reachable (a private repository returns 404
  to anonymous users).
* The history decision has been recorded.  A clean working tree does not
  clean old commits; see
  [`../../rules/provenance.md`](../../rules/provenance.md).
* The licence and the redistribution terms for bundled reference data and
  parameter tables have been reviewed and recorded.
* The `CARGO_REGISTRY_TOKEN` repository secret exists.  A new crates.io token
  needs the `publish-new` and `publish-update` scopes.  This is required for
  the first publish; crates.io trusted publishing cannot publish a new crate.
* The release-PR workflow can create pull requests.  Either enable the
  repository setting that lets GitHub Actions create pull requests, or set a
  `RELEASE_PLZ_TOKEN` secret with a fine-grained PAT that can write repository
  contents and pull requests.  The workflow uses that token in preference to
  `GITHUB_TOKEN` when it is present.

## Release scope

The `0.1.x` release artifact is the Rust crate `ruaa-dft-grids` only.  The
Python preset reader under `python/` and the agent material under `agents/` are
repository tooling; they are excluded from the crate package and are not
published as a Python distribution.

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
cargo package --list | grep -E '^(agents/|python/|release-plz\.toml|\.github/|AGENTS|CLAUDE|.*\.local)'   # must be empty
cargo package --list | wc -l
```

The package must not contain `agents/`, `python/`, `release-plz.toml`,
`AGENTS.md`, local configuration, scratch directories, raw probe output, build
artifacts, or a stale lockfile.  It does contain the tests, examples, docs and
the canonical preset fixture; program-specific reference data stays in the
local workspace.

Finally, run the registry dry run:

```bash
cargo publish --dry-run
```

`--dry-run` packages and verifies without uploading.  It may update the local
index; that is expected.

## Release automation

Two workflows use [release-plz](https://release-plz.dev/) and are defined in
[`.github/workflows/`](../../../.github/workflows/):

| Workflow | File | Purpose |
|---|---|---|
| Release-plz PR | `release-plz-pr.yml` | opens or updates the release pull request with the next version |
| Release-plz | `release-plz.yml` | publishes the unpublished Cargo package, creates the version tag and opens the GitHub release |

Both workflows are `workflow_dispatch` only.  A push or a merged pull request
never publishes automatically.  [`release-plz.toml`](../../../release-plz.toml)
enables the release only for the `ruaa-dft-grids` package; Python is not a
Cargo package and cannot be picked up by release-plz.

The built-in `GITHUB_TOKEN` must be allowed to create pull requests for the
release-PR workflow.  If an organization policy blocks that setting, set a
`RELEASE_PLZ_TOKEN` secret with a fine-grained PAT that can write repository
contents and pull requests; the workflow uses it in preference to
`GITHUB_TOKEN`.

The changelog is curated, so `changelog_update` is disabled.  The release PR
carries the version bump; add the dated changelog section to that PR before
merging it.  `release-plz` still reads the changelog when it creates the
GitHub release.

## First release (0.1.0)

The version and the dated changelog section are already prepared, so the first
release does not need a release PR.

1. Merge the final `0.1.0` tree after all gates pass.
2. Make the repository public.
3. Confirm the `CARGO_REGISTRY_TOKEN` secret and the release-PR token or
   repository setting.
4. Dispatch **Release-plz** on `main`.  It runs `cargo publish`, creates the
   `v0.1.0` tag and opens the GitHub release for `ruaa-dft-grids`.
5. Verify the crates.io page, the docs.rs build, the tag commit and the GitHub
   release.

If the dispatch fails before publishing, fix the cause and dispatch again.  Do
not create the tag by hand while the workflow is the release path.

## Later releases

1. Dispatch **Release-plz PR**.  It opens a pull request that bumps
   `Cargo.toml`; add the dated changelog section to the same pull request.
2. Review the version bump, the changelog and the CI result, then merge.
3. Dispatch **Release-plz** on `main`.  It publishes the unpublished version,
   creates the tag and opens the GitHub release.

A human dispatches the release workflow; that dispatch is the publish
approval.  Do not add a push trigger to the release workflow.

After the first successful publish, crates.io trusted publishing can replace
the stored token.  That requires `id-token: write` on the release job and a
trusted-publisher configuration on crates.io; it is a separate change.

## CI-only publish

Do not run `cargo publish`, `cargo release`, or any other upload command from
a local checkout.  The only publish path is the manual **Release-plz** GitHub
workflow.  Local commands stop at `cargo publish --dry-run`, which packages
and verifies without uploading.

## After publishing

* Verify the docs.rs build and the crate page.
* Verify the tag points at the published commit.
* Record the release in the changelog or release notes.
* Confirm that no Python distribution was uploaded.
* If a release is broken, prefer a follow-up patch over a yank; yank only when
  the published artifact is unusable or unsafe.
* Do not reuse a version number after a yank.

## Rollback

`cargo yank --version <version>` prevents new resolutions but does not delete
the artifact.  Publish a fixed patch version and document the reason.  A yank
is a human decision.

## Completion checklist

* [ ] All local gates pass on the release commit.
* [ ] `cargo package --list` contains no repository-process material.
* [ ] `cargo publish --dry-run` succeeds.
* [ ] The repository URL and metadata are correct.
* [ ] The history and data-licence review are recorded.
* [ ] The repository is public and the crates.io token secret is configured.
* [ ] The release-PR token or GitHub Actions pull-request setting is
      configured.
* [ ] No local upload command was run; publishing stays in GitHub Actions.
* [ ] A human has explicitly dispatched the release workflow.
