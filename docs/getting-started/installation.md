# Installation

## When to use this page

Use this page when setting up mechestim for the first time.

## Install as a dependency

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```

## Install for development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
```

## ✅ Verify installation

```bash
uv run python -c "import mechestim as me; print(me.__version__)"
```

## 🔍 What you'll see

```
0.2.0+np2.2.6
```

The version string includes the installed NumPy version suffix. If you see a version number, mechestim is installed correctly.

## ⚠️ Common pitfalls

**Symptom:** `ImportError: numpy version mismatch`

**Fix:** mechestim supports NumPy >=2.0.0,<2.3.0 (default install uses NumPy 2.2). Using `uv` handles this automatically. If you installed manually, check your NumPy version:

```bash
uv run python -c "import numpy; print(numpy.__version__)"
```

## 📎 Related pages

- [Your First Budget](./first-budget.md) — run your first FLOP-counted computation
