# AGENTS

Guidance for humans and AI assistants collaborating on jMetalPy.

## Scope
- Use agents for refactors, tests, docs, experiment scripts, and small utilities.
- Avoid delegating API changes, licensing, or research conclusions without maintainer sign-off.
- Keep changes scoped; prefer incremental PRs over broad rewrites.

## Safety and Privacy
- Do not share credentials, private datasets, or unpublished results.
- Assume no outbound network; request approval before downloads or package installs.
- Avoid destructive commands (e.g., removing files, rewriting history). When in doubt, ask.
- Work only inside the repository workspace.

## Coding Expectations
- Follow `CODING_GUIDELINES.md` (typing, docstrings, Python 3.11+, testing).
- English-only identifiers and comments; use ASCII unless the file already uses Unicode.
- Prefer clarity over cleverness; keep functions small and focused.

## Tooling and Commands
- Prefer `rg` for searches; summarize command output instead of dumping logs.
- Run targeted `pytest`, `ruff`, and `mypy` for affected areas when feasible; note any tests not run.
- Use `apply_patch` or minimal diffs; avoid auto-generated bulk changes unless requested.

## Interaction Style
- Be concise, reference paths with backticks (e.g., `src/module.py:42`), and summarize results.
- Explain rationale for non-obvious choices; add brief comments only when code needs context.
- Never revert user changes unless explicitly asked.

## Pre-Submission Checklist
- [ ] Changes comply with `CODING_GUIDELINES.md`.
- [ ] Tests/lint/type checks run where relevant, with results noted.
- [ ] No secrets or external data added; no destructive actions taken.
- [ ] Summary of changes and affected files prepared for the reviewer.
