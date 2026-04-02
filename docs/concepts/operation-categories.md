# Operation Categories

## When to use this page

Use this page to understand which operations cost FLOPs, which are free, and which are unsupported.

## Three categories

Every NumPy function falls into one of three categories in mechestim:

### 🟢 Free operations (0 FLOPs)

Operations that involve no arithmetic computation — just memory allocation, reshaping, or data movement.

**Examples:** `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `empty`, `reshape`, `transpose`, `concatenate`, `stack`, `split`, `squeeze`, `expand_dims`, `ravel`, `take`, `where`, `copy`, `astype`, `asarray`, `array_equal`

**Random operations** are also free: `me.random.randn`, `me.random.normal`, `me.random.seed`, etc.

### 🟡 Counted operations (cost > 0)

Operations that perform arithmetic. Cost is computed analytically from tensor shapes.

| Sub-category | Cost formula | Examples |
|-------------|-------------|----------|
| Unary | numel(output) | `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `ceil`, `floor` |
| Binary | numel(output) | `add`, `multiply`, `maximum`, `divide`, `power`, `subtract` |
| Reduction | numel(input) | `sum`, `mean`, `max`, `min`, `std`, `var`, `argmax`, `nansum` |
| Einsum | product of all index dims | `me.einsum(...)` |
| Dot/Matmul | equivalent einsum | `me.dot(A, B)`, `A @ B` |
| SVD | m × n × k | `me.linalg.svd(A, k=10)` |

### 🔴 Unsupported operations

Operations not in the mechestim allowlist. Calling them raises an `AttributeError` with a message explaining what's available.

```python
me.fft.fft(x)
# AttributeError: module 'mechestim.fft' has no attribute 'fft'.
# mechestim.fft currently supports: (none)
```

## 📎 Related pages

- [Operation Audit](../reference/operation-audit.md) — complete list of every operation and its category
- [FLOP Counting Model](./flop-counting-model.md) — how costs are calculated
- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — what changes when moving from NumPy
