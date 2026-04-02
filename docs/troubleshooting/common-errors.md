# Common Errors

## When to use this page

Use this page when you encounter an error from mechestim and need to understand what went wrong.

---

## BudgetExhaustedError

**Symptom:**

```
mechestim.errors.BudgetExhaustedError: einsum would cost 16,777,216 FLOPs but only 1,000,000 remain
```

**Why:** The operation you called would exceed the remaining FLOP budget. The operation did **not** execute.

**Fix:** Increase `flop_budget` in your `BudgetContext`, or reduce the cost of your computation. Use `budget.summary()` to see which operations are consuming the most FLOPs.

---

## NoBudgetContextError

**Symptom:**

```
mechestim.errors.NoBudgetContextError: No active BudgetContext. Wrap your code in `with mechestim.BudgetContext(...):`
```

**Why:** You called a counted operation (like `me.einsum`, `me.exp`, etc.) outside a `BudgetContext`.

**Fix:** Wrap your computation in a `BudgetContext`:

```python
with me.BudgetContext(flop_budget=10_000_000) as budget:
    # your code here
```

---

## AttributeError: module 'mechestim' has no attribute '...'

**Symptom:**

```
AttributeError: module 'mechestim' has no attribute 'fft'. mechestim does not support this operation.
```

**Why:** The NumPy function you're trying to use is not in mechestim's allowlist.

**Fix:** Check [Operation Categories](../concepts/operation-categories.md) for supported operations, or see the [Operation Audit](../reference/operation-audit.md) for the complete list.

---

## RuntimeError: Cannot nest BudgetContexts

**Symptom:**

```
RuntimeError: Cannot nest BudgetContexts
```

**Why:** You opened a `BudgetContext` inside another one. Only one can be active per thread.

**Fix:** Restructure your code to use a single `BudgetContext`.

---

## SymmetryError

**Symptom:**

```
mechestim.errors.SymmetryError: Tensor not symmetric along dims (0, 1): max deviation = 0.5
```

**Why:** You passed the same array object to multiple einsum operands, but the array values don't satisfy the symmetry that mechestim detected.

**Fix:** This usually indicates a bug — the same Python object is expected to have identical values. Check that you haven't mutated the array between creating it and calling einsum.

---

## 📎 Related pages

- [Debug Budget Overruns](../how-to/debug-budget-overruns.md) — diagnose which operations are expensive
- [Error Reference (API)](../api/errors.md) — full error class documentation
