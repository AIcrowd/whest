---
sidebar_position: 1
sidebar_label: For AI Agents
---
# For AI Agents

This page is for AI coding assistants (Claude, Cursor, Copilot, etc.) helping
users write whest code. It explains what resources are available, how to
access them, and the key things you must know before generating code.

---

## Quick orientation

**whest is NOT NumPy.** It wraps a subset of NumPy with analytical FLOP
counting. Every arithmetic operation is charged against a budget. Code that
works with NumPy may fail or behave differently with whest:

- All counted operations require an active `BudgetContext`
- 35 operations are blocked entirely (I/O, config, state)
- `sort`, `argsort`, `trace`, `random.*` sampling ops are now **counted** (not free)
- Costs are analytical (from tensor shapes), not measured at runtime

## Machine-readable resources

| Resource | Format | Use case |
|----------|--------|----------|
| [llms.txt](/llms.txt) | Markdown | Start here. Curated index of all doc pages with one-line descriptions. Under 4K tokens. |
| [llms-full.txt](/llms-full.txt) | Markdown | Complete docs in one file. Use if your context window is large enough (~115KB). |
| [ops.json](../ops.json) | JSON | Machine-readable manifest of all 508 operations. Query programmatically for name, category, cost formula, status. |
| [FLOP Cost Cheat Sheet](./cheat-sheet.md) | Markdown | Dense reference of every operation's cost. Optimized for agent context windows. |
| [Operation Audit](./operation-audit.md) | Markdown | 7-column searchable table: operation, whest ref, NumPy ref, category, cost, status, notes. |

### How to use llms.txt

If you're an agent encountering whest for the first time:

1. Fetch `llms.txt` — this gives you the doc map in ~300 words
2. Identify which page answers your question from the section descriptions
3. Fetch that specific page

**URL patterns:** `llms.txt` links to `.md` variants of each page (raw
markdown for agents). Every page is also available as rendered HTML — just
drop the trailing `/index.md` from the URL:

| Agent URL (raw markdown) | Human URL (rendered HTML) |
|--------------------------|---------------------------|
| `.../getting-started/installation/index.md` | `.../getting-started/installation/` |
| `.../how-to/use-einsum/index.md` | `.../how-to/use-einsum/` |
| `.../api/linalg/index.md` | `.../api/linalg/` |

If you have a large context window, fetch `llms-full.txt` instead to get
everything in one request.

### How to use ops.json

`ops.json` contains a JSON object with an `operations` array. Each entry has:

```json
{
  "name": "einsum",
  "module": "numpy",
  "whest_ref": "we.einsum",
  "numpy_ref": "np.einsum",
  "category": "counted_custom",
  "cost_formula": "product of all index dims (FMA=1)",
  "cost_formula_latex": "$\\prod_i d_i$",
  "free": false,
  "blocked": false,
  "status": "supported",
  "notes": "Supports SymmetricTensor inputs and repeated-operand detection for automatic cost reduction"
}
```

Use this to:

- Check if an operation is supported: filter by `"blocked": false`
- Get the cost formula for a specific operation: look up by `name`
- List all free operations: filter by `"free": true`
- Map between NumPy and whest calls: use `numpy_ref` and `whest_ref`

## Five rules for generating whest code

**1. A global default budget is active automatically — use `BudgetContext` for control.**

A global default budget auto-activates when whest is imported, so quick
scripts work without any setup. For precise budget control and namespacing,
use an explicit `BudgetContext`. Both forms are valid:

```python
# Quick work — global default handles budget tracking automatically
result = we.einsum('ij,jk->ik', A, B)

# Recommended for budget control and namespacing
with we.BudgetContext(flop_budget=10**8) as budget:
    result = we.einsum('ij,jk->ik', A, B)

# Decorator form for functions
@we.BudgetContext(flop_budget=10**8)
def my_forward_pass(x):
    return we.einsum('ij,j->i', W, x)
```

**2. Know what's free and what's counted.**

Free (0 FLOPs): `zeros`, `ones`, `reshape`, `transpose`, `copy`,
`random.seed`, `random.get_state`, `random.set_state`, `random.default_rng`.

Custom cost (numel FLOPs): `array`, `linspace`, `arange`, `concatenate`, `where`.
These are NOT free — each charges `numel(output)` FLOPs against the budget.

Counted: `einsum`, `dot`, `matmul`, `exp`, `log`, `add`, `multiply`, `sum`,
`mean`, all `linalg.*`, all `fft.*`, `sort`, `argsort`, `trace`,
`unique`, set ops (`in1d`, `isin`, etc.), `histogram`, `random.*` sampling.

Blocked: `save`, `load`, `geterr`, `seterr`, and 28 others. These raise
`AttributeError`.

When in doubt, check `ops.json` or the [cheat sheet](./cheat-sheet.md).

**3. Use `we.flops.*` to estimate costs before running.**

```python
cost = we.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
cost = we.flops.svd_cost(m=256, n=256, k=10)
```

These are pure functions — no `BudgetContext` needed.

**4. Use `we.einsum` as the primary computation primitive.**

Most linear algebra can be expressed as einsum. The cost is simply the
product of all index dimensions — each FMA (fused multiply-add) counts
as 1 operation.
`'ij,jk->ik'` with shapes `(m, k)` and `(k, n)` costs `m * k * n` FLOPs.

**5. Exploit symmetry for cost savings.**

- Use `symmetric_axes` for symmetric outputs:
  `we.einsum('ki,kj->ij', X, X, symmetric_axes=[(0, 1)])`
- Wrap known-symmetric matrices with `we.as_symmetric(data, symmetric_axes=(0, 1))`
  for automatic savings in downstream ops

## Common mistakes agents make

| Mistake | What happens | Fix |
|---------|-------------|-----|
| Using `np.einsum` instead of `we.einsum` | FLOPs not counted, budget not checked | Always use `we.*` for operations you want tracked |
| Skipping `BudgetContext` entirely | No error (global default handles it), but budget is harder to track and namespace | Use an explicit `BudgetContext` for any work you want to measure or label |
| Assuming `array`, `linspace`, `concatenate`, `where` are free | Underestimates budget usage — each charges `numel(output)` FLOPs | These are custom-cost ops, not free; check the cheat sheet |
| Assuming `sort` is free | Underestimates budget usage | `sort` costs `n*ceil(log2(n))` per slice — check the cheat sheet |
| Using `we.save()` or `we.load()` | `AttributeError` — blocked | Use `numpy` directly for I/O |
| Nesting two explicit `BudgetContext` blocks | `RuntimeError` | Use a single explicit context; nesting with the global default is fine |

## 📎 Related pages

- [FLOP Cost Cheat Sheet](./cheat-sheet.md) — every operation's cost at a glance
- [Operation Audit](./operation-audit.md) — full 508-operation inventory
- [Exploit Symmetry](../how-to/exploit-symmetry.md) — detailed symmetry guide
- [Common Errors](../troubleshooting/common-errors.md) — error messages and fixes
