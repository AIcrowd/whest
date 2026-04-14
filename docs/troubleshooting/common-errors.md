# Common Errors

## When to use this page

Use this page when you encounter an error from whest and need to understand what went wrong.

---

## BudgetExhaustedError

**Symptom:**

```
whest.errors.BudgetExhaustedError: einsum would cost 16,777,216 FLOPs but only 1,000,000 remain
```

**Why:** The operation you called would exceed the remaining FLOP budget. The operation did **not** execute.

**Fix:** Increase `flop_budget` in your `BudgetContext`, or reduce the cost of your computation. Use `budget.summary()` to see which operations are consuming the most FLOPs.

---

## NoBudgetContextError

**Symptom:**

```
whest.errors.NoBudgetContextError: No active BudgetContext. Wrap your code in `with whest.BudgetContext(...):`
```

**Why:** A counted operation (like `we.einsum`, `we.exp`, etc.) was called with no active budget session.

**In the core library:** This error is unlikely in normal use. whest automatically activates a global default budget, so operations run freely without any explicit setup. If you do see this error in the core library, it may indicate the global default was somehow torn down — restarting your session should resolve it.

**In the client-server model:** The server requires an open session. If your code runs on the server without a `BudgetContext`, this error will fire. Fix it by wrapping your computation:

```python
with we.BudgetContext(flop_budget=10_000_000) as budget:
    # your code here
```

---

## AttributeError: module 'whest' has no attribute '...'

**Symptom:**

```
AttributeError: module 'whest' has no attribute 'save'. whest does not support this operation.
```

**Why:** The NumPy function you're trying to use is blocked by whest (I/O, config, and system-level functions are not part of the competition API).

**Fix:** Check [Operation Categories](../concepts/operation-categories.md) for supported operations, or see the [Operation Audit](../reference/operation-audit.md) for the complete list.

---

## RuntimeError: Cannot nest BudgetContexts

**Symptom:**

```
RuntimeError: Cannot nest BudgetContexts
```

**Why:** You opened a `BudgetContext` inside another explicit `BudgetContext`. This error can only arise when two explicit contexts overlap.

**Note:** If the global default budget is active (the normal case in the core library), opening an explicit `BudgetContext` is fine — it temporarily replaces the global default for the duration of the `with` block, then restores it on exit. This is not nesting and does not raise an error.

**Fix:** If you see this error, you have two explicit `BudgetContext` managers open at the same time. Restructure your code so only one explicit context is active at a time, or rely on the global default for the outer scope.

---

## SymmetryError

**Symptom:**

```
whest.errors.SymmetryError: Tensor not symmetric along axes (0, 1): max deviation = 0.5
```

**Why:** You called `we.as_symmetric()` to declare that a tensor is symmetric along certain axes, but the actual data does not satisfy the symmetry within tolerance.

**Fix:** Verify that the tensor truly has the claimed symmetry (e.g., `A[i,j] == A[j,i]`). If it's approximately symmetric, you may need to symmetrise it first: `A = (A + A.T) / 2`.

---

## TimeExhaustedError

**Symptom:**

```
whest.errors.TimeExhaustedError: matmul: wall-clock time 5.003s exceeds limit 5.000s
```

**Why:** Your computation exceeded the wall-clock time limit set via `wall_time_limit_s` on the `BudgetContext`. The deadline is checked cooperatively — before each operation starts and after each operation completes — so the reported elapsed time may slightly exceed the limit by the duration of the last numpy call.

**Fix:**

1. **Optimise your code** — use `budget.summary()` to see which operations take the most wall-clock time and FLOP budget. Reducing the number of operations or using smaller matrices will help.
2. **Increase the time limit** — if you control the `BudgetContext`, pass a larger `wall_time_limit_s`.
3. **Check for unintended Python overhead** — the `Untracked time` in the summary shows time spent outside numpy calls (Python loops, data preparation). If this is large, your bottleneck may be Python code rather than linear algebra.

**Note:** In competition evaluation, the container also enforces a hard kernel-level time limit via cgroups. The in-library `TimeExhaustedError` gives you a clean error message with diagnostic info (which operation, how long you ran, what the limit was), whereas hitting the container limit results in a kill with no diagnostics.

---

## 📎 Related pages

- [Debug Budget Overruns](../how-to/debug-budget-overruns.md) — diagnose which operations are expensive
- [Error Reference (API)](../api/errors.md) — full error class documentation
