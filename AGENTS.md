# AGENTS.md

Guidance for AI assistants and human contributors working on jMetalPy.

## Project

jMetalPy is the Python implementation of the [jMetal](https://github.com/jMetal/jMetal) framework
for multi-objective optimization with metaheuristics. Source layout: `src/jmetal/` (`algorithm/`,
`core/`, `operator/`, `problem/`, `util/`, `lab/`), tests mirror that structure under `tests/`,
runnable examples live in `examples/`, Sphinx docs in `docs/`. Target: Python 3.11+.

## Scope

- Use agents for refactors, tests, docs, examples, and small-to-medium utilities.
- Avoid delegating API changes, licensing, or research conclusions without maintainer sign-off.
- Keep changes scoped; prefer incremental PRs over broad rewrites. Large mechanical changes
  (e.g. repo-wide reformatting) are fine when explicitly requested, but call out the blast radius
  before running them.

## Coding Standards

**ALWAYS follow the rules in [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md).** Short version:

- English-only identifiers, comments, and docstrings (Google style: Args/Returns/Raises)
- Type new/modified code as a style convention — there's no static type checker in CI (tried
  mypy, dropped it: findings were mostly it failing to follow deliberate duck-typing/generics
  in 6+ year old internal code, not real bugs; see `CODING_GUIDELINES.md` §5)
- `@dataclass(slots=True, frozen=True)` only for stateless data (config, DTOs) — never for
  `Solution` and other containers algorithms mutate in place
- One `return` per function except guard clauses; keep functions small; avoid nested conditionals
- Exceptions are the default error-handling mechanism; the `Ok[T] | Err` pattern is reserved for
  I/O-boundary code only
- Tests: pytest, AAA structure, `test_should_<behavior>` naming (not given-when-then — that's a
  Java/BDD convention, not idiomatic here), group related cases in `class Test<Subject>:`,
  prefer `@pytest.mark.parametrize` over near-duplicate tests

## Git Conventions

**ALWAYS follow the rules in [`GIT_GUIDELINES.md`](GIT_GUIDELINES.md).** Short version:

- Format: `<type>[(scope)][!]: <short imperative description>`
- Allowed types: `feat`, `fix`, `perf`, `test`, `refactor`, `style`, `docs`, `build`, `ci`,
  `chore`, `revert`
- One logical change per commit — never mix production code, tests, tooling, and docs in the
  same commit
- Tests must pass and `ruff check` must be clean before committing

## Tooling and Commands

- Environment: `pip install -e ".[dev]"` (or the project's `jmetalpy` conda env) installs
  everything needed for development, including ruff, pytest, build, and twine.
- Use the `Makefile` targets over ad hoc commands: `make lint` (ruff, blocking), `make format`
  (ruff), `make test`, `make test-coverage`, `make package` (build + twine check).
- CI mirrors this as three independent, parallel GitHub Actions workflows in
  `.github/workflows/`: `lint`, `test`, `build`.
- Summarize command output instead of dumping raw logs.

## Safety and Privacy

- Do not share credentials, private datasets, or unpublished results.
- Installing declared project dependencies (via the extras in `pyproject.toml`) to do the
  requested task is expected and doesn't need per-package confirmation; ask before adding a new
  dependency that isn't already declared, or before any install unrelated to the task at hand.
- Environment/tooling setup (conda envs, venvs) may reasonably touch locations outside the repo
  (e.g. `~/miniconda3`, `/tmp`); avoid modifying other files outside the repository workspace.
- Avoid destructive commands (removing tracked files, rewriting history, force-push). Deleting or
  replacing a file as an explicit, agreed part of a change is fine; when in doubt, ask first.

## Interaction Style

- Be concise; reference paths with backticks and line numbers (e.g., `src/jmetal/core/solution.py:42`).
- Explain rationale for non-obvious choices; add code comments only when the *why* isn't obvious
  from the code itself.
- Never revert user changes unless explicitly asked.

## Pre-Submission Checklist

- [ ] Code complies with `CODING_GUIDELINES.md`; commit messages comply with `GIT_GUIDELINES.md`.
- [ ] `make lint` and `make test` pass.
- [ ] No secrets, credentials, or private data added; no destructive actions taken.
- [ ] Summary of changes and affected files prepared for the reviewer.

---

Last updated: 2026-07-09
