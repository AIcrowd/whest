# For AI Agents

This page is for AI coding assistants (Claude, Cursor, Copilot, etc.) helping
users write mechestim code. It explains what resources are available, how to
access them, and the key things you must know before generating code.

---

## Quick orientation

**mechestim is NOT NumPy.** It wraps a subset of NumPy with analytical FLOP
counting. Every arithmetic operation is charged against a budget. Code that
works with NumPy may fail or behave differently with mechestim:

- All counted operations require an active `BudgetContext`
- 32 operations are blocked entirely (I/O, config, state)
- `sort`, `argsort`, `trace` are free (0 FLOPs) — this is surprising
- Costs are analytical (from tensor shapes), not measured at runtime

## Machine-readable resources

| Resource | Format | Use case |
|----------|--------|----------|
| [llms.txt](../../llms.txt) | Markdown | Start here. Curated index of all doc pages with one-line descriptions. Under 4K tokens. |
| [llms-full.txt](../../llms-full.txt) | Markdown | Complete docs in one file. Use if your context window is large enough (~115KB). |
| [ops.json](../ops.json) | JSON | Machine-readable manifest of all 482 operations. Query programmatically for name, category, cost formula, status. |
| [FLOP Cost Cheat Sheet](./cheat-sheet.md) | Markdown | Dense reference of every operation's cost. Optimized for agent context windows. |
| [Operation Audit](./operation-audit.md) | Markdown | 7-column searchable table: operation, mechestim ref, NumPy ref, category, cost, status, notes. |

### How to use llms.txt

If you're an agent encountering mechestim for the first time:

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
  "mechestim_ref": "me.einsum",
  "numpy_ref": "np.einsum",
  "category": "counted_custom",
  "cost_formula": "product of all index dims * op_factor",
  "cost_formula_latex": "$\\text{op\\_factor} \\cdot \\prod_i d_i$",
  "free": false,
  "blocked": false,
  "status": "supported",
  "notes": "Supports symmetric tensors via input_symmetries for automatic cost reduction"
}
```

Use this to:

- Check if an operation is supported: filter by `"blocked": false`
- Get the cost formula for a specific operation: look up by `name`
- List all free operations: filter by `"free": true`
- Map between NumPy and mechestim calls: use `numpy_ref` and `mechestim_ref`

## Five rules for generating mechestim code

**1. A global default budget is active automatically — use `BudgetContext` for control.**

A global default budget auto-activates when mechestim is imported, so quick
scripts work without any setup. For precise budget control and namespacing,
use an explicit `BudgetContext`. Both forms are valid:

```python
# Quick work — global default handles budget tracking automatically
result = me.einsum('ij,jk->ik', A, B)

# Recommended for budget control and namespacing
with me.BudgetContext(flop_budget=10**8) as budget:
    result = me.einsum('ij,jk->ik', A, B)

# Decorator form for functions
@me.BudgetContext(flop_budget=10**8)
def my_forward_pass(x):
    return me.einsum('ij,j->i', W, x)
```

**2. Know what's free and what's counted.**

Free (0 FLOPs): `zeros`, `ones`, `array`, `reshape`, `transpose`,
`concatenate`, `sort`, `argsort`, all `random.*` functions.

Counted: `einsum`, `dot`, `matmul`, `exp`, `log`, `add`, `multiply`, `sum`,
`mean`, all `linalg.*`, all `fft.*`.

Blocked: `save`, `load`, `geterr`, `seterr`, and 28 others. These raise
`AttributeError`.

When in doubt, check `ops.json` or the [cheat sheet](./cheat-sheet.md).

**3. Use `me.flops.*` to estimate costs before running.**

```python
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
cost = me.flops.svd_cost(m=256, n=256, k=10)
```

These are pure functions — no `BudgetContext` needed.

**4. Use `me.einsum` as the primary computation primitive.**

Most linear algebra can be expressed as einsum. The cost follows the opt_einsum
convention: `product_of_all_index_dims * op_factor`, where `op_factor` is 2
when there is an inner product (summed indices) and 1 otherwise.
`'ij,jk->ik'` with shapes `(m, k)` and `(k, n)` costs `2 * m * k * n` FLOPs.

**5. Exploit symmetry for cost savings.**

- Use `symmetric_axes` for symmetric outputs:
  `me.einsum('ki,kj->ij', X, X, symmetric_axes=[(0, 1)])`
- Wrap known-symmetric matrices with `me.as_symmetric(data, symmetric_axes=(0, 1))`
  for automatic savings in downstream ops

## Common mistakes agents make

| Mistake | What happens | Fix |
|---------|-------------|-----|
| Using `np.einsum` instead of `me.einsum` | FLOPs not counted, budget not checked | Always use `me.*` for operations you want tracked |
| Skipping `BudgetContext` entirely | No error (global default handles it), but budget is harder to track and namespace | Use an explicit `BudgetContext` for any work you want to measure or label |
| Assuming `sort` costs FLOPs | Overestimates budget usage | `sort` is free — check the cheat sheet |
| Using `me.save()` or `me.load()` | `AttributeError` — blocked | Use `numpy` directly for I/O |
| Nesting two explicit `BudgetContext` blocks | `RuntimeError` | Use a single explicit context; nesting with the global default is fine |

## 📎 Related pages

- [FLOP Cost Cheat Sheet](./cheat-sheet.md) — every operation's cost at a glance
- [Operation Audit](./operation-audit.md) — full 482-operation inventory
- [Exploit Symmetry](../how-to/exploit-symmetry.md) — detailed symmetry guide
- [Common Errors](../troubleshooting/common-errors.md) — error messages and fixes
