# Contributing to optax_bayes

See the full contributor guide at [docs/contributing.md](docs/contributing.md) for:

- Label taxonomy (`type:*`, `area:*`, `layer:*`, `wave:*`, `priority:*`)
- Two-layer epic model (**Wave → Theme → Issue**)
- The optax_bayes three-layer stack (L0 primitives → L1 optax transforms → L2 wrappers)
- Issue format conventions (`.github/ISSUE_TEMPLATE/`)
- Relationships syntax (`Parent:`, `Blocked by:`, `Blocks:`, `Related:`)
- Drafting a wave backlog in `.plans/`
- Pre-commit checklist and quality gates

Bootstrap the optax_bayes label set on a fresh clone:

```bash
make gh-labels
```

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) spec — enforced on PR titles by CI.
