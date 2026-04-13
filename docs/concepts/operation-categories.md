# Operation Categories

## When to use this page

Use this page to understand which operations cost FLOPs, which are free, and which are unsupported.

## Three categories

Every NumPy function falls into one of three categories in mechestim:

### 🟢 Free operations (0 FLOPs)

Operations that involve no arithmetic computation — just memory allocation, reshaping, or data movement.

**Examples:** `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `empty`, `reshape`, `transpose`, `concatenate`, `stack`, `split`, `squeeze`, `expand_dims`, `ravel`, `take`, `where`, `copy`, `astype`, `asarray`

### 🟡 Counted operations (cost > 0)

Operations that perform arithmetic. Cost is computed analytically from tensor shapes.

| Sub-category | Cost formula | Examples |
|-------------|-------------|----------|
| Unary | numel(output) | `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `ceil`, `floor` |
| Binary | numel(output) | `add`, `multiply`, `maximum`, `divide`, `power`, `subtract` |
| Reduction | numel(input) | `sum`, `mean`, `max`, `min`, `std`, `var`, `argmax`, `nansum` |
| Einsum | product of all index dims | `me.einsum(...)` |
| Dot/Matmul | equivalent einsum | `me.dot(A, B)`, `A @ B` |
| Linalg | per-operation formula | `me.linalg.solve`, `me.linalg.eigh`, `me.linalg.cholesky` |
| FFT | 5 N log N | `me.fft.fft`, `me.fft.rfft`, `me.fft.fft2` |
| SVD | m × n × k | `me.linalg.svd(A, k=10)` |
| Sort/Search | n log n per slice | `sort`, `argsort`, `unique`, `searchsorted` |
| Random | numel(output) | `me.random.randn`, `me.random.normal`, `me.random.uniform` |
| Stats | flat per-element (varies) | `me.stats.norm.pdf`, `me.stats.expon.cdf`, `me.stats.cauchy.ppf` |

When inputs are `SymmetricTensor`, many operations automatically get reduced costs. See [Exploit Symmetry](../how-to/exploit-symmetry.md).

### 🔴 Blacklisted operations

Operations not relevant to numerical computation. Calling them raises an `AttributeError`. These are I/O, configuration, datetime, and display functions that have no meaningful FLOP cost.

```python
me.save(array, "file.npy")
# AttributeError: 'save' is blacklisted in mechestim (I/O operation).
```

**Blacklisted categories:** I/O (`save`, `load`, `loadtxt`, `savetxt`, `savez`, `genfromtxt`), configuration (`seterr`, `geterr`, `setbufsize`), datetime (`busday_count`, `is_busday`), display (`array2string`, `array_repr`), functional (`apply_along_axis`, `piecewise`, `frompyfunc`).

See [Operation Audit](../reference/operation-audit.md) for the complete list.

## 📎 Related pages

- [Operation Audit](../reference/operation-audit.md) — complete list of every operation and its category
- [FLOP Counting Model](./flop-counting-model.md) — how costs are calculated
- [Migrate from NumPy](../how-to/migrate-from-numpy.md) — what changes when moving from NumPy
