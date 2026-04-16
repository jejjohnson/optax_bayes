---
applyTo: "docs/**/*.ipynb,docs/**/*.py,docs/**/*.md,notebooks/**/*.ipynb"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **executed `.ipynb` files**. The committed `.ipynb` carries both source cells and rendered cell outputs (including matplotlib figures as embedded PNGs). `mkdocs-jupyter` renders them with `execute: false`, so the committed outputs are what users see in the docs.

Every notebook is **Google Colab compatible** — the first cell detects Colab and `pip install`s the right dependencies so users can click "Open in Colab" and run end-to-end without touching the host environment.

## Directory Layout

```
docs/
├── notebooks/
│   ├── demo_foo.ipynb
│   ├── tutorial_bar.ipynb
│   └── ...
└── api/
    └── ...
```

No separate `images/` directory — figures live inside the `.ipynb` cell outputs.

## Authoring Workflow

**Develop in jupytext percent format**, then convert and execute for the final commit. Plain `.py` diffs are vastly easier to review than raw `.ipynb` JSON, so do the substantive editing on the `.py` side.

1. Create `docs/notebooks/foo.py` in jupytext percent format (header below).
2. Iterate — edit, smoke-run via `uv run --group docs python docs/notebooks/foo.py`, repeat.
3. When satisfied, convert to `.ipynb`:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/foo.py
   ```

4. Execute in place so cell outputs are embedded:

   ```bash
   uv run --group docs jupyter nbconvert --to notebook \
     --execute docs/notebooks/foo.ipynb \
     --inplace \
     --ExecutePreprocessor.timeout=180
   ```

5. Delete the `.py` — the `.ipynb` is the committed source of truth.
6. Commit the `.ipynb`. `mkdocs-jupyter` picks it up automatically.

## Jupytext Header (dev only)

While developing in `.py`, start the file with:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Cell Markers (dev only)

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import jax.numpy as jnp
```

## First Markdown Cell — Title + Colab Badge

Every notebook opens with a `#`-level title and a Colab badge pointing at its `main`-branch URL. Replace `OWNER/REPO` below with the actual GitHub owner and repository name (e.g. from `repo_url` in `mkdocs.yml`):

```markdown
# Demo — Feature Overview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/docs/notebooks/demo_foo.ipynb)

<one-paragraph summary>

**What you'll learn:**

1. ...
2. ...
3. ...
```

## First Code Cell — Colab Detection + Install

Detect Colab, install the package only when needed (substitute `OWNER/REPO` with the actual GitHub owner and repository name):

```python
import subprocess
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "optax_bayes @ git+https://github.com/OWNER/REPO@main",
        ],
        check=True,
    )
```

Local / CI users with the environment set up skip the install and go straight to imports.

## Imports + Warnings

```python
import warnings

warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax.numpy as jnp
import matplotlib.pyplot as plt
# ... other imports
```

- Suppress the IProgress warning from ipywidgets so the output is clean.

## Reproducibility — `watermark`

After imports, print a version readout. The cell uses `get_ipython()` and an `importlib.util.find_spec` check so a plain `python foo.py` smoke run during dev — and a local `nbconvert --execute` on a machine without `watermark` installed — both no-op cleanly instead of raising `UsageError: Line magic function %load_ext not found`:

```python
import importlib.util

try:
    from IPython import get_ipython

    ipython = get_ipython()
except ImportError:
    ipython = None

if ipython is not None and importlib.util.find_spec("watermark") is not None:
    ipython.run_line_magic("load_ext", "watermark")
    ipython.run_line_magic(
        "watermark",
        "-v -m -p jax,jaxlib,optax,optax_bayes",
    )
else:
    print("watermark extension not installed; skipping reproducibility readout.")
```

The `[docs]` dependency group pulls in `watermark`, so under the documented authoring workflow (`uv run --group docs jupyter nbconvert --execute ...`) the readout actually prints. Users reproducing the notebook later see exactly which package versions generated the committed outputs.

## Notebook Structure

1. **Title + Colab badge** (markdown)
2. **Background** (markdown) — motivation, math, what the user will learn
3. **Setup** — Colab detection + install
4. **Imports + config + watermark**
5. **Problem setup** — data, grids, initial conditions
6. **Core demonstration(s)** — alternating markdown and code
7. **Visualizations** — figures are embedded directly as cell outputs
8. **Summary / takeaways**

## Matplotlib Style

**Defaults only** — no `plt.style.use` and no `rcParams` tweaks:

- `C0`, `C1`, `C2` (matplotlib defaults) for main series.
- `"k--"` for truth / reference lines.
- `figsize=(12, 5)` for single plots, `(18, 5)` for 1×3 comparison grids.
- `ax.scatter(...)` for data points.
- `ax.fill_between(..., alpha=0.2)` for uncertainty bands.

## Markdown Paragraph Wrapping

**Each paragraph in a `# %% [markdown]` block must be a single long line.** Do not soft-wrap paragraph text across multiple `#` lines. jupytext preserves source newlines as soft breaks, which mkdocs-jupyter renders as awkward visual breaks.

Right:

```python
# %% [markdown]
# This notebook demonstrates the key features of the package. We walk through all the main patterns using a simple example so the only thing that differs is how the parameters are configured.
```

Wrong:

```python
# %% [markdown]
# This notebook demonstrates the key features of the package. We walk
# through all the main patterns using a simple example so the only
# thing that differs is how the parameters are configured.
```

Lines that *must* stay on their own (do not join):

- Headings: `# # Title`, `# ## Section`
- Display math: `# $$...$$` block (one expression per line)
- Table rows: `# | col | col |`
- List item heads: `# - item` or `# 1. item`
- Code-fence delimiters: `# ``` ` and contents inside the fence
- Blockquotes: `# > quote`

## Math in Markdown Cells

Inline: `$\|x - x'\|^2$`.
Display:

```markdown
$$f(x) = \sum_{i=1}^{N} w_i \phi_i(x)$$
```

MathJax is configured in `mkdocs.yml` — both inline and display math render in the docs.

## Timing & Benchmarks

For performance comparisons, use a JIT-warm-then-measure pattern:

```python
import time
import jax

def time_fn(fn, warmup=2, repeats=5):
    """JIT-compile, warm up, then time *repeats* calls (seconds)."""
    jitted = jax.jit(fn)
    for _ in range(warmup):
        jitted().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        jitted().block_until_ready()
    return (time.perf_counter() - t0) / repeats
```

Key rules:
- **Always warm up** JIT-compiled functions before timing
- **Use `block_until_ready()`** for JAX async dispatch
- **Collect results into a dict/list**, then produce a summary figure or table at the end

## Checklist for New Notebooks

- [ ] Authored in jupytext `.py` percent format during development
- [ ] First markdown cell: `#`-level title + Colab badge
- [ ] Second markdown cell: background + math + "What you'll learn"
- [ ] Setup cell: Colab detection + package install via `subprocess`
- [ ] `warnings.filterwarnings(..., IProgress, ...)`
- [ ] `%watermark` version readout
- [ ] Matplotlib defaults only (no `style.use`, no `rcParams`)
- [ ] Converted to `.ipynb` and executed in place
- [ ] `.py` deleted; `.ipynb` with embedded outputs committed
- [ ] Listed in `mkdocs.yml` nav
