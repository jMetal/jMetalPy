# Git and Commit Guidelines

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Message format

```
<type>[(scope)][!]: <short imperative description>

[optional body]

[optional footer(s)]
```

- Subject line: imperative mood ("add", not "added"/"adds"), no trailing period, ≤ 72 characters.
- `scope` is optional and names the affected area, e.g. `operator`, `algorithm`, `problem`, `util`, `lab`, `docs`.
- `!` after the type/scope marks a breaking change (see below).
- Identifiers and messages must be in **English**, per [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

## Allowed types

| Type       | When to use                                                         |
| ---------- | ------------------------------------------------------------------- |
| `feat`     | A new feature, algorithm, operator, problem, or public method       |
| `fix`      | A bug fix                                                           |
| `perf`     | A change that improves performance without changing behavior        |
| `test`     | Adding or correcting tests (no production code)                     |
| `refactor` | Code change that is neither a fix nor a new feature                 |
| `style`    | Formatting only (whitespace, `black`/`ruff` fixes); no logic change |
| `docs`     | Documentation only (`README.md`, `AGENTS.md`, docstrings, etc.)     |
| `build`    | Packaging or dependency changes (`pyproject.toml`, `setup.cfg`)     |
| `ci`       | Changes to CI configuration (`.github/workflows/*.yml`)             |
| `chore`    | Tooling, `.gitignore`, and other changes that don't fit above       |
| `revert`   | Reverts a previous commit                                           |

## Breaking changes

Mark commits that change or remove a public API (e.g. `Algorithm`, `Problem`, `Solution`, `Operator` signatures) with `!`:

```
feat(operator)!: change PolynomialMutation constructor argument order
```

Explain the impact and migration in the body/footer:

```
BREAKING CHANGE: `PolynomialMutation` now takes `distribution_index` before
`probability`. Update call sites accordingly.
```

## Body and footers

- Add a body when the *why* isn't obvious from the subject line alone (wrap at ~72 chars).
- Reference issues with a footer, e.g. `Closes #187`, `Refs #42`.
- Trailers such as `Co-Authored-By:` are allowed and go last.

## Atomic commits

Each commit must represent **one single logical change**. Guidelines:

- If the commit message needs "and" to describe what it does, split it into two commits.
- Before committing, run the checks relevant to the change:
  - `make test` (or `pytest tests/ -x`) — tests pass
  - `make lint` — `ruff` is clean
- Never mix production code changes with test changes in the same commit.
- Never mix code changes with documentation changes in the same commit.

## Merge commits

Prefer squash-merging pull requests so `main` keeps one Conventional Commit per logical change.
If a merge commit is unavoidable, the default `Merge pull request #N from ...` message is
acceptable as an exception to the `<type>: ...` format.

## Examples

```bash
# Good
git commit -m "feat(algorithm): add SMS-EMOA implementation"
git commit -m "fix(operator): correct PMX crossover infinite loop on repeated genes"
git commit -m "test(util): add tests for DistanceBasedArchive"
git commit -m "perf(util): vectorize non-dominated sorting with numpy"
git commit -m "docs: add hyperparameter tuning section to README"
git commit -m "ci: run tests on Python 3.11 and 3.12"

# Bad — too broad, mixes concerns
git commit -m "add stuff and fix tests and update readme"

# Bad — no type, not English
git commit -m "cambios"
```
