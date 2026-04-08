# Subgraph Symmetry Detection

A contributor-level walkthrough of the algorithm used to detect permutation
symmetry in einsum intermediates and propagate it through the contraction path.

## The problem

A multi-operand einsum like `'ij,ai,bj->ab'` is decomposed by opt_einsum into
a sequence of pairwise contractions. At each step the optimizer must evaluate
candidate pairs — and it needs to know, for each candidate intermediate, whether
the result is symmetric, so it can score it with a reduced cost.

When operands are `SymmetricTensor`s, their per-operand symmetry is known
upfront. But there is a second source of symmetry: when the same Python object
appears at multiple operand positions, the output can be symmetric in index
labels contributed by those repeated operands — even if the operands are dense.

The naive approach is to rerun a detection procedure at every step for every
candidate subset. This is too expensive for large contractions. We want:

1. **Correctness** — detect all exploitable symmetry without false positives.
2. **Memoization** — compute each intermediate's symmetry at most once.
3. **Laziness** — only evaluate subsets that the path optimizer actually visits.

Subgraph symmetry detection achieves all three.

## The bipartite graph

The core data structure is a bipartite graph over the einsum expression.

**Left vertices (U):** One U-vertex per equivalence class of axes within each
operand. For a dense operand with subscript `"ai"`, each axis is its own class,
producing two U-vertices. For a `SymmetricTensor` with subscript `"ij"` and
declared symmetry `S₂{i,j}`, both axes are in the same class, producing one
U-vertex.

**Right vertices (labels):** One right vertex per unique index label. Labels are
partitioned into:
- **V (free labels):** appear in the final output subscript or in operands
  outside the current subset (they "cross the cut").
- **W (summed labels):** contracted entirely within the current subset.

**Incidence:** An edge from U-vertex `u` to label `c` has weight equal to the
multiplicity of `c` in the axes belonging to class `u`.

**Identical-operand groups:** Operands that are the same Python object are
grouped. These groups are the source of induced symmetry.

### Worked example

Consider `'ij,ai,bj->ab'` with operands `T, A, B` where `T` is a dense tensor:

```
Subscripts:  ij,  ai,  bj  →  ab
Operands:    T    A    B
```

U-vertices (dense operands, one class per axis):

- `(T, 0)` — label set `{i}`
- `(T, 1)` — label set `{j}`
- `(A, 0)` — label set `{a}`
- `(A, 1)` — label set `{i}`
- `(B, 0)` — label set `{b}`
- `(B, 1)` — label set `{j}`

Free labels at the top level: `{a, b}` (appear in output `->ab`).
Summed labels at the top level: `{i, j}` (contracted out).

No identical operands in this example — `T`, `A`, and `B` are distinct Python
objects.

Now consider the subset `{A, B}` (positions 1 and 2):

- U-vertices in subset: `(A, 0)`, `(A, 1)`, `(B, 0)`, `(B, 1)`
- Labels in subset: `{a, i, b, j}`
- Labels outside subset (in T): `{i, j}`
- Crossing labels (in subset AND in outside): `{i, j}`
- V at this step = `{a, b} ∪ {i, j}` = `{a, b, i, j}` (all four — `{i,j}` cross the cut)
- W at this step = `{}` (nothing is summed entirely within `{A, B}`)

The incidence matrix M at this subset (rows = U-vertices, columns = V∪W):

```
         a  i  b  j
(A, 0):  1  0  0  0
(A, 1):  0  1  0  0
(B, 0):  0  0  1  0
(B, 1):  0  0  0  1
```

## The subset-keyed oracle

The key invariant is the **pure-in-subset property**: the symmetry of an
intermediate tensor depends only on the set of original operands it was formed
from, not on the order in which they were contracted. This is because:

- The bipartite graph structure is fixed for the full einsum.
- The induced subgraph on a subset `S` is fully determined by which operands
  are in `S`.
- Symmetry is a property of the final intermediate, not its contraction history.

This property makes the subset key canonical. The oracle stores results in a
`dict[frozenset[int], IndexSymmetry | None]` and returns cached results on
subsequent calls with the same subset.

```python
# One oracle per contract_path call
oracle = SubgraphSymmetryOracle(
    operands=list(operands),
    subscript_parts=input_parts,
    per_op_syms=index_symmetries,
    output_chars=output_str,
)

# Lazy evaluation — only computed on first access per subset
sym = oracle.sym(frozenset({0, 1}))  # symmetry of intermediate from ops 0 and 1
```

## Per-index pair detection (Step 2a)

Given the induced subgraph M for a subset, Step 2a detects which pairs of free
labels (V-labels) are symmetry-equivalent.

**Algorithm:**

1. Fix a canonical column encoding: for each label `c`, let `col(c)` be the
   tuple of incidence values down the rows of M.

2. For each permutation `σ` of the identical-operand groups:
   a. Lift `σ` to a row permutation on M (permute rows belonging to swapped operands).
   b. Compute `σ·col(c)` for each label.
   c. Validate that W-column multisets are preserved (summed labels are inert).
   d. For each pair `(i, j)` of V-labels: if `col(i) == σ·col(j)` and
      `col(j) == σ·col(i)` and all other V-labels are fixed by `σ`, record the
      pair `(i, j)`.

3. Run union-find over the collected pairs to produce connected components.
   Each component of size ≥ 2 becomes one S_k symmetry group.

**Complexity:** `O(m! · poly(n))` per subset, where `m` is the size of the
largest identical-operand group and `n` is the number of labels. In practice
`m` is small (2 or 3 for typical einsums).

## Hybrid block path (Step 2b)

Step 2a detects per-index symmetry. Step 2b extends this to block symmetry,
where each operand contributes a tuple of free indices rather than a single
index.

For each pair `(i, j)` of operands in the same identical-operand group:

1. Compute `free_i` = labels that appear in operand `i`'s subscript, survive
   to the output of the current subset, and do NOT appear in operand `j`'s
   subscript or any other operand in the subset. These are the labels that
   "belong uniquely" to operand `i` at this step.
2. Compute `free_j` similarly for operand `j`.
3. If `|free_i| == |free_j| > 0`, build the label swap `σ: free_i ↔ free_j`
   pairing them positionally in subscript order. For an outer product
   `einsum('abc,def->abcdef', X, X)` this produces
   `σ = {a↔d, b↔e, c↔f}`.
4. Validate `σ` via the same incidence-matrix column-equality machinery used
   in Step 2a: lift the operand swap `tilde_sigma = {i: j, j: i}` to a row
   permutation of `M_S`, compute `σ(M_S)`, and check that every V-column `L`
   satisfies `M_S[:, L] == σ(M_S)[:, σ(L)]` — i.e., the column-for-column
   match is performed under the *label relabeling* rather than requiring
   every column to stay put. The W-column multiset must also be preserved.

If valid, record a block symmetry group. Block size 1 collapses to a per-index
pair (already handled equivalently by Step 2a). Block size ≥ 2 produces a
true block group like `frozenset({('a','b','c'), ('d','e','f')})` whose
semantics are "these two 3-tuples can swap as a unit, but you cannot swap
only one axis without the others."

**Follow-up (`sigma → pi` extension, `TODO(sigma-to-pi)`):** The natural
unification of Steps 2a and 2b is to extend Step 2a to derive the full label
permutation `π` induced by each operand permutation `σ`, rather than iterating
pairs. This subsumes the block path entirely. Deferred to a follow-up iteration.

## Complexity bound

The oracle evaluates each subset at most once. For a contract with `N` operands
and groups of sizes `k₁, k₂, …`:

- Number of subsets visited: at most `2^N` (usually much less — path algorithms
  visit only O(N²) subsets in practice for greedy and branch-bounded search).
- Per-subset cost: `O(m! · poly(n))` where `m = max(kᵢ)`.
- Total: `O(2^N · m! · poly(n))`, dominated by the path algorithm itself.

For the common case of a single pair of identical operands (`m = 2`):
per-subset cost is `O(poly(n))`.

## What we deleted and why

The previous implementation used three separate mechanisms:

1. `_detect_induced_output_symmetry` — a dense scan over operand pairs at the
   top level only, missing intermediates.
2. `propagate_symmetry` — a function that restricted per-operand groups through
   contraction steps, called eagerly at every step.
3. `induced_output_symmetry` kwarg on `contract_path` — passed the top-level
   detection result into the path algorithms.

**Problems with the old approach:**

- **Top-level only.** Detection ran once on the full operand list, not on
  intermediates. Symmetry detected at the top level was often consumed by the
  first contraction, giving zero savings on subsequent steps.
- **Over-eager propagation.** `propagate_symmetry` was called on every
  candidate pair during path search, with no caching. For large contractions
  this was O(N² · poly(n)) per step.
- **Silent drop.** When detection produced no result, the code silently fell
  back to dense costs with no diagnostic.

**What replaced it:**

- `SubgraphSymmetryOracle` — one object per `contract_path` call, subset-keyed,
  lazy, cached.
- `symmetry_oracle` kwarg on `contract_path` — plumbed through `_PATH_OPTIONS`
  so all algorithms receive it.
- `tests/test_no_silent_symmetry_drop.py` — enforces that the oracle is
  consumed and no silent fallback occurs.

The `induced_output_symmetry` kwarg, `propagate_symmetry`, and
`_detect_induced_output_symmetry` are all deleted. No replacement text is
needed — the oracle subsumes all three.
