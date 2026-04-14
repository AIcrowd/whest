---
sidebar_position: 1
sidebar_label: Installation
---
# Installation

## When to use this page

Use this page when setting up whest for the first time.

## Install as a dependency

```bash
uv add git+https://github.com/AIcrowd/whest.git
```

## Install for development

```bash
git clone https://github.com/AIcrowd/whest.git
cd whest
uv sync --all-extras
```

## ✅ Verify installation

```bash
uv run python -c "import whest as we; print(we.__version__)"
```

## 🔍 What you'll see

```
0.2.0+np2.2.6
```

The version string includes the installed NumPy version suffix. If you see a version number, whest is installed correctly.

## ⚠️ Common pitfalls

**Symptom:** `ImportError: numpy version mismatch`

**Fix:** whest supports NumPy >=2.0.0,<2.3.0 (default install uses NumPy 2.2). Using `uv` handles this automatically. If you installed manually, check your NumPy version:

```bash
uv run python -c "import numpy; print(numpy.__version__)"
```

## 📎 Related pages

- [Your First Budget](./first-budget.md) — run your first FLOP-counted computation
