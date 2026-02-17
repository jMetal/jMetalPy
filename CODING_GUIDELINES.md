# Python Coding and Testing Guidelines (Python 3.11+)

All source code, identifiers, and comments **must be written in English**.

## 1. General Rules

- Target version: **Python 3.11+**
- Build backend: **setuptools**
- Code and comments: **English only**
- One responsibility per function, one module per concept

## 2. Typing and Structure

- Use `|` for unions, `TypeAlias`, `Literal`, `Final`, `TypedDict`, and `Self`
- Annotate parameters, return types, and key variables
- Use `@dataclass(slots=True, frozen=True)` for data containers
- Prefer `Enum` for discrete choices
- No untyped public functions
- Cognitive complexity ≤ 10

## 3. Function Rules

- One `return` per function (except parameter validation guards)
- Guard clauses only for invalid inputs
- Keep functions ≤ 20 lines when possible
- Avoid nested conditionals (“pyramid of ifs”)

## 4. Error and Resource Handling

- Always use context managers (`with`)
- Use `Ok[T] | Err` pattern for recoverable results
- Raise specific exceptions for unrecoverable errors
- Avoid global exception handling except at the entry point

## 5. Style, Tooling, and Documentation

- Use **ruff** for linting, style, and complexity enforcement
- Use **mypy** for type checking
- Use **Google-style docstrings** with Args / Returns / Raises
- Clarity over cleverness — no “smart” one-liners

## 6. Unit Testing

- Framework: **pytest**
- Follow **AAA pattern** (Arrange–Act–Assert)
- Use **given-when-then** naming convention: test_given_valid_input_when_process_then_returns_ok()
- Each test should focus on a single behavior
- Use fixtures for setup
- Always include both success and failure paths
- Exception checks use `pytest.raises`

## 7. AI-Aware Design

- Always include type hints and docstrings
- Keep functions pure and self-contained
- Prefer descriptive variable names
- Provide minimal working examples and tests

---

## Enforcement

All rules are automatically verified through:
- **ruff** (style, imports, complexity, docstrings, comments)
- **mypy** (typing)
- **pytest** (tests)
