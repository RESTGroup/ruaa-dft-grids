# Provenance and tracked-content hygiene

The normative rule is in [`../../AGENTS.md`](../../AGENTS.md). This page is the
review procedure that keeps the working tree and the published history inside
that rule.

## Scope

The rule applies to everything a reader of the repository can see:

* source, comments, docstrings, examples, tests, and golden data;
* `README.md`, `docs/`, `tests/`, `examples/`, `agents/`, and configuration files;
* commit messages, tag messages, and release notes;
* generated artifacts that are checked in, including schemas and headers.

It does **not** apply to the maintainer's local notes, raw program output, or
scratch directories, because those are not part of the repository. They must
stay outside the tracked tree.

## What is allowed

* Factual statements of a program's behaviour and defaults ("ORCA 6.1 uses
  adaptive pruning"; "the radial nodes are ordered outside-in").
* Formulas, parameter values, keywords, and data tables that may be
  redistributed.
* Neutral validation claims that point at the checked-in references, such as
  "validated against the bundled ORCA 6.1.1 reference probes".
* PySCF and other open-source citations, including function and module names.

## What is not allowed

* Narratives of how a value was obtained by inspecting a closed-source binary:
  disassembly, decoding, binary addresses or offsets, internal symbol names,
  proof campaigns, or residual-investigation stories.
* References to maintainer-local reports, probe trees, scratch directories, or
  absolute filesystem paths.
* Benchmark fragments in code comments or commit messages: timings, host or
  CPU names, measured crossover points, or tuning brackets. Curated
  measurements belong in [`../../docs/performance.md`](../../docs/performance.md).
* Secrets, licence keys, hostnames, and machine-specific configuration.
* A "TODO: replace with the real value" note that hides where a value came
  from. Either the value is publishable with a factual description, or it does
  not belong in the repository.

## Review procedure

Before committing:

1. Run `git diff --check` and read every hunk in `git diff`. Do not rely on a
   summary or a generated commit message.
2. Search the changed files for absolute path prefixes, home directories,
   scratch-directory names, and the categories listed under "What is not
   allowed". Remove or rewrite each hit.
3. Check generated files (schemas, headers, reference probes) as carefully as
   source files. A generated comment can leak the same material.
4. Read the commit subject and body. The history is part of the repository; a
   clean tree with a revealing commit message is not clean.
5. If a value came from an unpublished source, do not transfer the value with
   its derivation. Decide separately whether the value itself may be
   redistributed, and record that decision in [`../decisions/`](../decisions/).

Before a release:

1. Run `cargo package --list` and review the file list. The package must not
   contain `agents/`, local configuration, scratch directories, raw probe
   output, or build artifacts.
2. Run `git log --stat` over the history that will be published. If the
   repository will be published with its current history, old commits must
   satisfy the same rule as the current tree. A history rewrite or a fresh
   repository is a maintainer decision; do not perform one unilaterally.
3. Run the content search over the packaged tree, not only over the working
   tree.
4. Record the provenance/licence review for bundled data in a decision record.
   A release gate is not satisfied by an informal memory.

## Transferring material from the local workspace

The local workspace is allowed to contain full derivation notes, raw output,
and measurement worklogs. Only a small subset is suitable for the repository:

* Transfer process knowledge (how to run a check, what a release needs) after
  rewriting it in neutral, repository-relative terms.
* Transfer a data table only when its redistribution is clear and its
  description is factual.
* Never transfer a narrative, a local report reference, a binary address, or a
  raw measurement log.
* If the same fact already exists in `docs/` or a docstring, link to it instead
  of importing the workspace version.

When in doubt, leave the material out. A missing process note is cheaper than
a provenance leak in a published repository.
