# ruaa_dft_grids LLM Agents Guidance

Rust generator for molecular DFT numerical-integration grids: atomic grids
(radial kernels, Lebedev angular rules, pruning), molecular grid assembly,
and partition weights (classic Becke, Stratmann/SSF, SS, mBecke, LKO)
including analytic gradients and Hessians.  The published crate is
program-agnostic: it ships the open-source PySCF level presets and a
data-driven `GridPreset` API.  ORCA/Gaussian parameter data, reference
probes and construction checks live in a separate local workspace outside
this repository.  The dependency-free Python preset reader under `python/` is
repository tooling; the `0.1.x` release publishes only the Rust crate.

## Important notices to AI code agent

- **Local configuration**: This file (`AGENTS.md` / `CLAUDE.md`) is always read first. If a per-developer, gitignored guidance file exists at the repository root, read it as well: it supplements this file, and where explicitly stated, overrides it.
- **Repository agent material**: `agents/README.md` indexes the repository's operational skills, review rules, decisions, and development sketch. Read it for validation refresh, CI, release, extension, or benchmarking work. The root rules here remain normative.
- **Prohibit local paths**: This project must not present any local filesystem path in *git tracked* code, comments, docs or scripts. Git-untracked names and per-developer files count as local paths too. Paths that exist only on a maintainer machine (probe trees, local reports, bench harnesses) belong to the local development workspace, not this repository.
- **No program data in the published crate**: ORCA/Gaussian parameter tables,
  reference goldens, probe scripts and program-specific construction code
  belong outside this repository.  The crate may document a program's
  *behavior* factually (for example a published partition formula) but must
  not embed or ship that program's parameter data.
- **No reverse-engineering traces**: Code may state a program's *behavior* factually ("ORCA 6.1 defaults to adaptive pruning", "ORCA orders radial nodes outside-in"), but tracked content must stay free of how that behavior was obtained: no "decoded"/"disassembly" narratives, binary addresses or offsets, internal symbol names, derivation or proof campaigns ("proved against N measured runs"), residual-investigation stories, or references to maintainer-local notes or reports. User-facing ORCA keywords (`AngularGrid`, `GridPruning`, `DEFGRID`, `COSX`, `SimpleGrid`, `IntAcc`, `HGridReduced`) are fine. PySCF is Apache-2.0 open source: citing its published functions and code is welcome. Detailed provenance and derivation notes live in the maintainer's local workspace; never paste them back into tracked files.
- **No benchmark fragments in comments**: code comments and docstrings must not carry fragmented efficiency measurements — timings, measured crossover tables, host/CPU names, measured brackets behind tuning constants. Keep at most a one-line empirical rationale ("cut chosen empirically; a wrong call costs little"). Consolidated measurements belong in curated docs (`docs/` if published) or the local workspace benchmark notes. A curated `docs/` page embedded by `src/doc/mod.rs` is documentation, not a code docstring: the performance page may include host details and consolidated measurements when it states that the input geometries and harness are not bundled. Code-level docstrings remain subject to this rule.
- **Code comments concise; docstrings scientific**: inline (non-docstring) comments should be concise and descriptive, 1–2 lines, at most 4 — code should explain itself. Docstrings are for API callers; as a scientific program, a docstring *may* be lengthy when it documents algorithms and formulas, but keep it organized (overview, construction, conventions, special cases) and free of reverse-engineering evidence, benchmark fragments, and refactor histories.
- **Math in Markdown**: rustdoc parses doc comments and the embedded
  `docs/*.md` pages as Markdown before KaTeX runs.  Inside `$...$` and
  `$$...$$`, write a subscript underscore as `\_` when it does not immediately
  follow an ASCII letter or digit (`\hat{\mathbf u}\_{AB}`, but `w_g`), and
  never spell a LaTeX command as backslash plus ASCII punctuation (`\,`,
  `\{`, `\\`) other than the `\_` above; use the named forms (`\thinspace`,
  `\lbrace`, `\rbrace`, `\cr`, `\ast`).  `tests/math_markdown.rs` enforces
  the rule.
- **Keep validation claims neutral and brief**: statements like "checked
  against reference probes maintained outside this package" are welcome; the
  underlying evidence chains are not.  Do not claim bundled program goldens
  that the crate does not contain.
- **Golden data is data**: the local workspace's test files may hold
  measured/golden numbers — they are the validation.  Keep their surrounding
  comments factual and brief; do not delete golden tables as "clutter".
- **Git commit rule**: use the subject form `<crate-or-module>: <summary>` (for example `ruaa_dft_grids (partition): ...`). If the environment provides a commit-attribution skill, follow it for the co-author trailers.
- **Not auto commit unless human requested**: do not run `git commit` on your own, even with permission to do so.
- **Release only through CI**: do not run `cargo publish`, `cargo release`, or any upload command from a local checkout. The manual release workflow in GitHub Actions is the only publish path; local verification stops at `cargo publish --dry-run`.
- **Write in English**: the user may interact in any language; communicate in the user's language, but code, comments, documentation, and commit messages are always English.
- **API not fixed**: the crate is at an early stage (0.1.x). Do not assume any API is fixed. While working on higher-level code, lower-level APIs may be proposed or edited when poorly designed — notify human developers in the summary.
- **Large files**: if you find large or bulky files — especially ones that are or would be `.gitignore`d — do not act on your own: ask the maintainer.

## Expected agent usage

This guidance is written for a "human proposes, agent implements / reviews"
workflow, not for fully autonomous operation.

## Fidelity / provenance quick reference

- OK in tracked content: formulas, redistributable parameter values,
  program-behavior statements, and brief validation claims pointing to
  `tests/`.
- Local-notes material: how constants/rules were obtained, validation
  histories, residual-anomaly investigations, binary-level derivation details,
  internal symbol mappings, performance measurement tables.
- When you must write about provenance in a PR description, say "checked
  against the reference construction maintained outside this package" and
  leave the derivation to the local notes.
