# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Bayesian Learning Rule as optax `GradientTransformation`s. Built with JAX, optax, equinox, and jaxtyping.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests in parallel: uv run pytest -v -n auto
make test-fast            # Fast unit tests (skips slow + integration; matches PR CI)
make test-slow            # Only the slow + integration tests
make test-cov             # Run tests with coverage report (parallel)
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/optax_bayes
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/optax_bayes/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/optax_bayes  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/optax_bayes/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Test Speed Tiers

CI (PRs and pushes to `main`) runs only unmarked (fast) tests:
`-m "not slow and not integration"`. Slow and integration tests never run
automatically — trigger them on demand with the "Extended Tests" workflow:
`gh workflow run tests-extended.yml` (heavy lane) or
`gh workflow run tests-extended.yml -f suite=full` (entire suite), or run
`make test-slow` / `make test` locally. When adding tests:

- Unmarked (default): unit tests, < ~1 s each.
- `@pytest.mark.slow`: individually expensive tests (> ~1.5 s — heavy
  numerics, long scan training loops, Monte Carlo moment checks).
- `@pytest.mark.integration`: end-to-end workflows (e.g. analytic posterior
  recovery on conjugate models). Usually combined with `slow`.

## Optional Dependencies

`gaussx` is an optional extra (`pip install "optax_bayes[gaussx]"`), required
only by the full-rank / low-rank transforms and their posterior/sampling
helpers. Never import gaussx at module top level in `src/` — go through
`optax_bayes._src._optional.require_gaussx` so the diagonal/IVON core stays
importable without it. The `dev` dependency group installs gaussx, so the
full test suite always exercises both paths.

## Architecture

### Package structure

All implementation lives in `src/optax_bayes/`. The public API is re-exported through `src/optax_bayes/__init__.py`.

### Key directories

| Path | Purpose |
|------|---------|
| `src/optax_bayes/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Convert and execute: `jupytext --to notebook foo.py` then `jupyter nbconvert --execute --inplace foo.ipynb`
3. Delete the `.py` — the executed `.ipynb` is the committed source of truth
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Coding Conventions

- Google-style docstrings
- `dataclasses` or `attrs` for data containers
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
