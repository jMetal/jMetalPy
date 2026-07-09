# Python Coding and Testing Guidelines (Python 3.11+)

All source code, identifiers, and comments **must be written in English**.

## 1. General Rules

- Target version: **Python 3.11+**
- Build backend: **setuptools**
- Code and comments: **English only**
- One responsibility per function, one module per concept

## 2. Typing and Structure

- Use `|` for unions, `TypeAlias`, `Literal`, `Final`, `TypedDict`, and `Self`
- Annotate parameters, return types, and key variables on new/modified code
- Use `@dataclass(slots=True, frozen=True)` for stateless, immutable data — config
  objects, DTOs, parameter bundles. **Do not** apply it to `Solution` and other
  mutable containers that algorithms modify in place (crossover, mutation, repair);
  freezing them would force copy-on-write in hot evolutionary loops
- Prefer `Enum` for discrete choices
- Public functions should be typed on new/modified code. This is a style
  convention, not tool-enforced — the project doesn't run a static type
  checker (see [Enforcement](#enforcement)); most of the codebase predates
  strict typing and retrofitting it wholesale isn't worth the churn
- Cognitive complexity ≤ 10 (enforced by ruff on new/modified code)

## 3. Function Rules

- One `return` per function (except parameter validation guards)
- Guard clauses only for invalid inputs
- Keep functions ≤ 20 lines when possible
- Avoid nested conditionals (“pyramid of ifs”)

## 4. Error and Resource Handling

- Always use context managers (`with`)
- The codebase's default is exceptions: raise specific exceptions (`ValueError`,
  `TypeError`, etc.) for invalid input and unrecoverable errors — this is what
  existing algorithms, operators, and problems already do
- The `Ok[T] | Err` result pattern is reserved for I/O-boundary code (file
  parsing, CLI argument handling) where a caller is expected to branch on
  failure without exceptions; do not introduce it inside the algorithm/operator
  core, which relies on exceptions throughout
- Avoid global exception handling except at the entry point

## 5. Style, Tooling, and Documentation

- **ruff**: linting, formatting, import order, and complexity — the blocking gate
  in CI (`make lint`, `make format`).
- No static type checker (mypy) runs in CI. It was tried and dropped: most of
  the codebase is 6+ years old and predates strict typing, and chasing its
  findings to zero had a poor cost/benefit ratio — the bulk were mypy failing
  to follow deliberate duck-typing and generics, not real bugs, concentrated
  in internal plumbing rather than the public API surface users actually call.
  Nothing stops a contributor from running `mypy` locally out of personal
  preference, but it isn't part of the project's enforced workflow.
- Use **Google-style docstrings** with Args / Returns / Raises
- Clarity over cleverness — no "smart" one-liners

## 6. Unit Testing

- Framework: **pytest**
- Follow **AAA pattern** (Arrange–Act–Assert)
- Name tests `test_should_<behavior>` (e.g. `test_should_raise_error_on_negative_probability`);
  put given/when/then detail in a one-line docstring, not in the function name.
  `given_when_then`-style names are a Java/BDD import, not idiomatic pytest, and are not
  required — this is descriptive, not enforced by tooling
- Group related scenarios under `class Test<Subject>:` rather than flat module-level functions
- Prefer `@pytest.mark.parametrize` over hand-written near-duplicate tests for scenario variants
- Each test should focus on a single behavior
- Use fixtures for setup
- Always include both success and failure paths
- Exception checks use `pytest.raises`
- Use plain `assert` (pytest rewrites it for rich failure diffs); no matcher library

## 7. AI-Aware Design

- Always include type hints and docstrings
- Keep functions pure and self-contained
- Prefer descriptive variable names
- Provide minimal working examples and tests

---

## Enforcement

- **ruff** (style, imports, complexity) — blocking in CI
- **pytest** (tests) — blocking in CI
