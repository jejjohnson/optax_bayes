# Contributing to optax_bayes

This page documents the label taxonomy, epic model, and issue conventions used in this project.

---

## Label Taxonomy

Every issue carries exactly one `type:*`, one or more `area:*`, at most one `layer:*`, one `wave:*`, and one `priority:*`.

| Scope | Labels |
|---|---|
| **Type** | `type:epic-wave`, `type:epic-theme`, `type:feature`, `type:design`, `type:chore`, `type:docs`, `type:bug`, `type:research` |
| **Area** | `area:algorithmic` (BLR math / Hessian estimators), `area:engineering`, `area:testing`, `area:docs`, `area:integration` (gaussx / efax / equinox), `area:code` |
| **Layer** | `layer:0-primitives`, `layer:1-components`, `layer:2-models` — see the **three-layer stack** below |
| **Wave** | `wave:0-bootstrap`, `wave:1-diagonal`, `wave:2-fullrank`, `wave:3-lowrank`, `wave:4-advanced` — matches the project roadmap |
| **Priority** | `priority:p0` (blocker), `priority:p1` (high), `priority:p2` (normal) |

Bootstrap the full set on a fresh clone:

```bash
make gh-labels
```

The script lives at [`.github/scripts/create-labels.sh`](https://github.com/jejjohnson/optax_bayes/blob/main/.github/scripts/create-labels.sh). Edit the hard-coded `create …` entries to customise labels, then re-run — the script is idempotent.

---

## Three-Layer Stack

optax_bayes follows the three-layer architecture described in the design docs:

| Layer | Label | Contents | Source paths |
|---|---|---|---|
| **L0 Primitives** | `layer:0-primitives` | Pure JAX BLR update equations, Hessian estimators, natural ↔ mean conversions | `src/optax_bayes/_src/primitives.py`, `hessians.py`, `conversions.py` |
| **L1 Components** | `layer:1-components` | optax `GradientTransformation`s: `blr_diagonal`, `blr_full_rank`, `blr_low_rank` | `src/optax_bayes/_src/diagonal.py`, `full_rank.py`, `low_rank.py`, `types.py` |
| **L2 Models** | `layer:2-models` | `_for_loss` wrappers, `get_posterior_*` extraction, schedule composition | `src/optax_bayes/_src/wrappers.py`, `posterior.py` |

The labeler at [`.github/labeler.yml`](https://github.com/jejjohnson/optax_bayes/blob/main/.github/labeler.yml) auto-applies the right `layer:*` label based on which source file a PR touches.

---

## Two-Layer Epic Model

Work is organised as **Wave → Theme → Issue**:

```
[EPIC] Wave N: <title>      (L1)   release-scoped mega-epic, owns a milestone
  ├── [EPIC] <theme>        (L2)   parallel-safe group of issues
  │     ├── feature / design / chore / research / bug issue
  │     └── …
  └── [EPIC] <theme>        (L2)
        └── …
```

- **Wave epic** (`type:epic-wave`) maps one-to-one with a milestone (`vX.Y-<slug>`). Groups several Theme epics that can run in parallel.
- **Theme epic** (`type:epic-theme`) groups concrete issues that ship together as a coherent slice. Themes inside a wave are parallel-safe unless the body says otherwise.
- **Issue** (feature / design / chore / research / bug) is the leaf — a single substantial deliverable that rolls up to a Theme epic.

Stragglers are discouraged: any `type:feature` issue should attach to a Theme epic. If no suitable theme exists, create one first.

### The optax_bayes roadmap

| Wave | Milestone | Focus |
|---|---|---|
| 0 | `v0.0-bootstrap` | Repo bootstrap — package identity, deps, labels, docs skeleton |
| 1 | `v0.1-diagonal` | BLR-Diagonal (self-contained; the primary entry point per D5) |
| 2 | `v0.2-fullrank` | BLR-FullRank with gaussx backend for structured precision |
| 3 | `v0.3-lowrank` | BLR-LowRank via gaussx Woodbury / `LowRankUpdate` |
| 4+ | `v0.4-advanced` | Hutchinson Hessian, schedule composition, K-FAC, ecosystem showcases |

---

## Issue Format

Issue templates live in [`.github/ISSUE_TEMPLATE/`](https://github.com/jejjohnson/optax_bayes/blob/main/.github/ISSUE_TEMPLATE/):

| Template | When to use |
|---|---|
| `Epic — Wave (L1)` | Opening a new release-scoped wave |
| `Epic — Theme (L2)` | Grouping related issues inside a wave |
| `Feature / Enhancement` | One substantial deliverable |
| `Design / ADR` | Resolve an open design question for a new API |
| `Bug report` | Something isn't working |
| `Research / Comparative Analysis` | Study prior art (external repo, paper) and produce a prioritized roadmap of follow-up issues |

### Optional sections on Feature + Design

Both templates include two **optional** sections for context-heavy issues:

- **Design Snapshot** — paste API sketches, code examples, or excerpts from private / external design docs so the issue is self-contained. Rename to fit the issue type (`Demo To Implement`, `Demo Snippet To Include`, `Config Snippet`, `Reference Trace`).
- **Mathematical Notes** — equations, sign conventions, numerical considerations. Prefer unicode math in prose (σ², E₁, ∑, Λ⁻¹, O(d³)) over LaTeX blocks. Use ` ```text ` code fences for multi-line equations so pseudo-math isn't mangled by syntax highlighting:

    ```text
    s_next   = (1 − ρ) · s   + ρ · (s₀   − h)
    η_next   = (1 − ρ) · η   + ρ · (η₀ + g − h ⊙ m)
    ```

Both exist so that an implementer (human or AI agent) can work on an issue without opening other repos or chats.

---

## Relationships

Use an explicit `## Relationships` block at the bottom of each issue / epic body:

```markdown
## Relationships
- Parent: #<theme-epic>
- Blocked by: #
- Blocks: #
- Related: #
```

GitHub's task-list feature links bidirectionally from the parent, so checklist items in a Theme epic body auto-show in the referenced issues.

---

## Drafting a wave backlog

For large planning exercises (new wave, new release, large refactor), draft the whole backlog as one markdown file **before** opening GitHub issues. A template lives at [`docs/templates/wave-backlog.md`](https://github.com/jejjohnson/optax_bayes/blob/main/docs/templates/wave-backlog.md).

Why:

- **Review the whole wave in one scroll** instead of clicking through 15 half-drafted issues
- **Share context once** at the top (Shared Context · Design Snapshot · Intended Package Layout · Runtime Boundary) rather than duplicating across every child issue
- **Stable draft IDs** (`OBX-01`, `OBX-02`, …) let child issues reference each other before GitHub issue numbers exist

Suggested prefix for optax_bayes drafts: `OBX`.

Workflow:

1. Copy [`docs/templates/wave-backlog.md`](https://github.com/jejjohnson/optax_bayes/blob/main/docs/templates/wave-backlog.md) into `.plans/` (gitignored). Rename to describe the wave — e.g. `.plans/wave-1-diagonal-backlog.md`.
2. Number drafts sequentially: `OBX-01`, `OBX-02`, …
3. Fill in shared context at the top, then draft each issue body. Reach for the [`api/`](https://github.com/jejjohnson/optax_bayes/tree/main/docs/api) and the math treatment when filling the Design Snapshot / Mathematical Notes sections.
4. When the file is ready, open each draft as a real GitHub issue using the matching `.github/ISSUE_TEMPLATE/`.
5. Record GitHub issue numbers next to draft IDs, or replace draft IDs throughout. Update cross-references.
6. Either delete the backlog file or archive it in `.plans/archive/`.

---

## Pre-commit checklist

Run these locally before opening a PR:

```bash
make format       # ruff format . + ruff check --fix .   (applies changes)
make lint         # ruff check .                         (CI-style check)
make typecheck    # ty check
make test         # pytest
```

Note that `make format` **mutates files** — it formats and applies autofixes. `make lint` is the CI-parity read-only check. Run `make format` first, commit the result, then run `make lint` / `make test` to verify.

Pre-commit hooks run ruff on every commit. Run `make precommit` to apply them to all files manually.

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) specification — enforced on PR titles by [`.github/workflows/conventional-commits.yml`](https://github.com/jejjohnson/optax_bayes/blob/main/.github/workflows/conventional-commits.yml).
