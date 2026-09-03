# Contributing to jMetalPy

Thanks for your interest in contributing! This file covers the practical setup; the actual
standards live in a few dedicated documents so they don't drift out of sync with this one:

- [AGENTS.md](AGENTS.md) — project layout, scope, and conventions (written for AI assistants and
  human contributors alike).
- [CODING_GUIDELINES.md](CODING_GUIDELINES.md) — style, typing, docstrings, and test conventions.
- [GIT_GUIDELINES.md](GIT_GUIDELINES.md) — commit message format (Conventional Commits) and
  atomic-commit practices.

## Getting set up

```console
$ git clone https://github.com/jMetal/jMetalPy.git
$ cd jMetalPy
$ pip install -e ".[dev]"
```

## Before opening a pull request

```console
$ make lint    # ruff check
$ make test    # pytest
```

Both must pass. Follow [GIT_GUIDELINES.md](GIT_GUIDELINES.md) for commit messages — one logical
change per commit, Conventional Commits format, English only.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/jMetal/jMetalPy/issues).
