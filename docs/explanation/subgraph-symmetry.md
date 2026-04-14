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

**Left vertices (U):** One U-vertex per axis of each operand. For a dense
operand with subscript `"ai"`, each axis produces its own U-vertex (two total).
For a `SymmetricTensor` with subscript `"ij"` and declared symmetry `S₂{i,j}`,
both axes still produce separate U-vertices — per-operand symmetry does not
affect the graph topology. Instead, per-operand symmetry is handled entirely
by the expanded sigma-loop (see below), which uses the declared symmetry
generators as an additional source of row permutations.

**Right vertices (labels):** One right vertex per unique index label. Labels are
partitioned into:
- **V (free labels):** appear in the final output subscript or in operands
  outside the current subset (they "cross the cut").
- **W (summed labels):** contracted entirely within the current subset.

**Incidence:** An edge from U-vertex `u` to label `c` has weight equal to the
multiplicity of `c` in the axes belonging to U-vertex `u`.

**Identical-operand groups:** Operands that are the same Python object are
grouped. These groups are the source of induced symmetry.

### Worked example

Consider `'ij,ai,bj->ab'` with operands `T, A, B` where `T` is a dense tensor:

```
Subscripts:  ij,  ai,  bj  →  ab
Operands:    T    A    B
```

U-vertices (one per axis):

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

#### Full bipartite graph

```
   U (axes)                         Labels
   ─────────────────               ──────
                                   V (free):
   (A, 0) ───────────────────────── a
   (B, 0) ───────────────────────── b

                                   W (summed):
   (T, 0) ──────────┐
                    ├────────────── i
   (A, 1) ──────────┘

   (T, 1) ──────────┐
                    ├────────────── j
   (B, 1) ──────────┘
```

Now consider the subset `{A, B}` (positions 1 and 2):

- U-vertices in subset: `(A, 0)`, `(A, 1)`, `(B, 0)`, `(B, 1)`
- Labels in subset: `{a, i, b, j}`
- Labels outside subset (in T): `{i, j}`
- Crossing labels (in subset AND in outside): `{i, j}`
- V at this step = `{a, b} ∪ {i, j}` = `{a, b, i, j}` (all four — `{i,j}` cross the cut)
- W at this step = `{}` (nothing is summed entirely within `{A, B}`)

#### Induced subgraph for subset {A, B}

When we restrict to subset {A, B}, labels `i` and `j` cross the cut (they also
appear in T, outside the subset), so they move from W to V:

```
   U (subset {A, B} only)           Labels
   ──────────────────────           ──────
                                    V (all free):
   (A, 0) ───────────────────────── a
   (A, 1) ───────────────────────── i
   (B, 0) ───────────────────────── b
   (B, 1) ───────────────────────── j

                                    W: (empty)
```

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
`dict[frozenset[int], SubsetSymmetry]` and returns cached results on
subsequent calls with the same subset.

```python
# One oracle per contract_path call
oracle = SubgraphSymmetryOracle(
    operands=list(operands),
    subscript_parts=input_parts,
    per_op_groups=perm_groups,
    output_chars=output_str,
)

# Lazy evaluation — only computed on first access per subset
result = oracle.sym(frozenset({0, 1}))  # SubsetSymmetry for intermediate from ops 0 and 1
result.output  # V-side (output tensor) symmetry
result.inner   # W-side (inner summation) symmetry
```

## π-based detection

### Goal

For a fixed subset `S` with incidence matrix `M`, we want the full group
of **automorphisms of the labelled bipartite graph** — pairs `(σ, π)`
where `σ` permutes identical-operand rows and `π` permutes label columns,
such that applying `π` to the columns of `σ(M)` recovers `M`:

```
π(σ(M)) = M          (equivalently, σ(M) = π⁻¹(M))
```

Every such `π` is a symmetry of the intermediate tensor built from `S`.
Restricted to V labels it contributes to the output (V-side) symmetry;
restricted to W labels it contributes to the inner (W-side) symmetry.
The V/W partition is part of the labelled structure, so legitimate
automorphisms must preserve it — `π(V) ⊆ V` and `π(W) ⊆ W` — and any
`π` with a cycle crossing V↔W is rejected.

The algorithm iterates over **generators** of the row-permutation group
(per-operand symmetry generators and identical-operand swap generators)
and, for each one, recovers the unique `π` (up to fingerprint collisions)
satisfying the equation above. The collected `π`'s become generators of the
detected `PermutationGroup` on labels, closed via Dimino's algorithm.

Given M for a subset, detection proceeds as follows:

### Column fingerprints

For each label `c`, compute its column fingerprint `col(c)` — the tuple of
incidence values down the rows of M. Labels with identical fingerprints are
*candidates* for symmetry equivalence. The fingerprint-to-label mapping is
used by the sigma-loop (below) to derive pi in O(1) per label via hash lookup.

!!! note "Removed fast path"
    Earlier versions had a standalone fast path that detected S_k whenever
    labels shared a fingerprint, without running the sigma-loop. This was
    incorrect for non-S_k groups (see the C3 bug note below) and has been
    removed. Fingerprints are now used only for pi derivation inside the
    sigma-loop — they are not a standalone detection mechanism.

### σ loop: derive π from generators

The sigma-loop iterates over **generators** of the row-permutation group on M,
drawn from two sources:

- **Source A — per-operand internal symmetry generators.** For each operand
  that carries a declared `PermutationGroup`, each generator of that group is
  lifted to a row permutation on M (permuting only the rows belonging to that
  operand). This captures symmetry that was previously handled by orbit-based
  axis merging.

- **Source B — identical-operand swap generators.** For each group of k
  identical operands (same Python object), the k-1 adjacent transpositions
  `(op_i, op_{i+1})` are used as generators. Each such swap is lifted to a row
  permutation that exchanges the rows of the two operands.

For each generator `σ`:

1. Lift `σ` to a row permutation on M.
2. Compute `σ(M)`'s column fingerprints: `σ·col(c)` for each label `c`.
3. **Derive the induced label permutation π directly.** For each label `ℓ`,
   `π(ℓ)` is the label whose M-column matches `σ(M)`'s column for `ℓ` — a
   hash-table lookup in `O(1)`. When multiple labels share a fingerprint
   (collision), pick the lex-first unused candidate. If any label has no
   match, reject this σ.
4. **Validate π:** `π(V) ⊆ V` and `π(W) ⊆ W`. Any cycle crossing V↔W
   invalidates the σ.
5. **Collect π as a `Permutation` object** restricted to V labels (and
   separately to W labels). Non-identity permutations become generators
   of the detected `PermutationGroup`.

The σ-loop collects all non-identity π restrictions as `Permutation` objects.
These generators are passed to Dimino's algorithm (via `PermutationGroup(...)`)
to close the group and build the exact symmetry group on V (and separately
on W).

### Worked example: `einsum('ab,cd->abcd', X, X)`

Consider two identical dense matrices `X`. The bipartite graph has one
U-vertex per axis (four total), one label per column, and no W-labels
(nothing is contracted):

```
   U (axes)                         Labels
   ─────────────────               ──────
   operand X₀:                     V (all free):
     X₀ · a ─────────────────────── a
     X₀ · b ─────────────────────── b

   operand X₁ (= X₀):
     X₁ · c ─────────────────────── c
     X₁ · d ─────────────────────── d

                                    W: (empty)
```

X₀ and X₁ are the same Python object (grouped above), forming one
identical-operand group of size 2.

The incidence matrix M is the 4×4 identity:

```
        a  b  c  d
u0_a  [ 1  0  0  0 ]
u0_b  [ 0  1  0  0 ]
u1_c  [ 0  0  1  0 ]
u1_d  [ 0  0  0  1 ]
```

**Fingerprints:** All four fingerprints are distinct — no label collisions.

**σ loop:** The only nontrivial σ swaps operands 0 and 1, permuting rows
(0↔2, 1↔3). We ask: *what relabeling π of the columns turns `σ(M)` back
into `M`?*

```
  M (original rows)          σ(M) (rows 0↔2, 1↔3)
  ┌──────────────────┐       ┌──────────────────┐
  │ u0_a  1 0 0 0    │  σ    │ u1_c  0 0 1 0    │
  │ u0_b  0 1 0 0    │ ───>  │ u1_d  0 0 0 1    │
  │ u1_c  0 0 1 0    │       │ u0_a  1 0 0 0    │
  │ u1_d  0 0 0 1    │       │ u0_b  0 1 0 0    │
  └──────────────────┘       └──────────────────┘
         columns: a b c d           columns: a b c d

  Match σ(M) columns back to M columns — i.e., for each column ℓ of
  σ(M), find the M-column that equals it; that label is π(ℓ):
    σ(M)[:,a] = col_of[c]  →  π(a) = c
    σ(M)[:,b] = col_of[d]  →  π(b) = d
    σ(M)[:,c] = col_of[a]  →  π(c) = a
    σ(M)[:,d] = col_of[b]  →  π(d) = b

  Induced π = (a c)(b d).
```

**Recover M from σ(M).** We now show concretely that `π(σ(M)) = M`.
Rename each column `ℓ` of `σ(M)` by `π(ℓ)` (so column `a` → `c`, `b` → `d`,
`c` → `a`, `d` → `b`), then re-sort the columns alphabetically:

```
     σ(M), cols renamed by π:       after sorting cols (a,b,c,d):
              c  d  a  b                         a  b  c  d
     u1_c  [ 0  0  1  0 ]             u1_c  [ 1  0  0  0 ]
     u1_d  [ 0  0  0  1 ]             u1_d  [ 0  1  0  0 ]
     u0_a  [ 1  0  0  0 ]             u0_a  [ 0  0  1  0 ]
     u0_b  [ 0  1  0  0 ]             u0_b  [ 0  0  0  1 ]
```

Entry-by-entry the result matches M — both are the 4×4 identity. The
row labels have been carried along by `σ` (which sent `u0_a ↔ u1_c`
and `u0_b ↔ u1_d`), so every incidence `(u, ℓ)` of M maps under
`(σ, π)` to another incidence of M. The labelled bipartite graph is
unchanged — `(σ, π)` is an automorphism, and `π(σ(M)) = M` holds in
the graph-automorphism sense.

Equivalently, we can verify the recovery edge by edge. M has four
edges; apply `(σ, π)` to each and check the image is still an edge
of M:

```
  (u0_a, a) ─(σ,π)→ (u1_c, c)   ✓ in M
  (u0_b, b) ─(σ,π)→ (u1_d, d)   ✓ in M
  (u1_c, c) ─(σ,π)→ (u0_a, a)   ✓ in M
  (u1_d, d) ─(σ,π)→ (u0_b, b)   ✓ in M
```

All four edges are preserved → `(σ, π)` is an automorphism of the
labelled bipartite graph, confirming the matrix-level recovery.

So π = (a c)(b d). Two disjoint 2-cycles from one σ, all in V (W is empty).
Classify: number of cycles = 2 = block size; cycle length = 2 = number of
blocks. Group by operand: block₁ = (a, b), block₂ = (c, d). The result is
block S₂: `{(a,b), (c,d)}`.

### Worked example: `einsum('ia,ib->ab', X, X)` — per-index V symmetry

This computes XᵀX, which is symmetric. The detection derives this from the
bipartite graph alone.

U-vertices (one per axis):

- `(X₀, 0)` — label `i` (subscript `ia`, axis 0)
- `(X₀, 1)` — label `a` (subscript `ia`, axis 1)
- `(X₁, 0)` — label `i` (subscript `ib`, axis 0)
- `(X₁, 1)` — label `b` (subscript `ib`, axis 1)

Labels: V = `{a, b}` (output), W = `{i}` (contracted). X₀ and X₁ are the
same Python object — one identical-operand group of size 2.

```
   U (axes)                         Labels
   ─────────────────               ──────
   operand X₀:                     V (free):
     X₀ · a ─────────────────────── a

   operand X₁ (= X₀):
     X₁ · b ─────────────────────── b

                                   W (summed):
     X₀ · i ────────┐
                    ├────────────── i
     X₁ · i ────────┘
```

The incidence matrix M (rows = U-vertices, columns = V∪W):

```
         a  b  i
X₀ · i [ 0  0  1 ]
X₀ · a [ 1  0  0 ]
X₁ · i [ 0  0  1 ]
X₁ · b [ 0  1  0 ]
```

**Fingerprints:** `col(a) = (0,1,0,0)`, `col(b) = (0,0,0,1)`, `col(i) = (1,0,1,0)`.
All three are distinct — no label collisions.

**σ loop:** The only nontrivial σ swaps operands 0 and 1, permuting rows
(0↔2, 1↔3):

```
  M (original rows)        σ(M) (rows 0↔2, 1↔3)
  ┌────────────────┐       ┌────────────────┐
  │ X₀·i  0 0 1   │  σ    │ X₁·i  0 0 1   │
  │ X₀·a  1 0 0   │ ───>  │ X₁·b  0 1 0   │
  │ X₁·i  0 0 1   │       │ X₀·i  0 0 1   │
  │ X₁·b  0 1 0   │       │ X₀·a  1 0 0   │
  └────────────────┘       └────────────────┘
       columns: a b i           columns: a b i

  Match σ(M) columns back to M columns:
    σ·col(a) = (0,0,0,1) = col(b)  →  π(a) = b
    σ·col(b) = (0,1,0,0) = col(a)  →  π(b) = a
    σ·col(i) = (1,0,1,0) = col(i)  →  π(i) = i

  Induced π = (a b),  i fixed
```

Validate: π(V) = {b, a} ⊆ V ✓, π(W) = {i} ⊆ W ✓.

**Recovery check.** Apply `(σ, π)` to each edge of M and verify it
lands back on an edge of M:

```
  (X₀·i, i) ─(σ,π)→ (X₁·i, i)   ✓ in M   (i is a π-fixed point)
  (X₀·a, a) ─(σ,π)→ (X₁·b, b)   ✓ in M
  (X₁·i, i) ─(σ,π)→ (X₀·i, i)   ✓ in M
  (X₁·b, b) ─(σ,π)→ (X₀·a, a)   ✓ in M
```

All four edges are preserved, so `π(σ(M)) = M`. Notice that `i`
stays in W on both sides of the map — this is what makes the
`π(W) ⊆ W` validation pass and is why the W/V partition check is
needed: if a hypothetical π had pulled `i` into V (or an output
label into W), the implied bipartite-graph action would no longer
be an automorphism of the *labelled* graph.

Cycle structure on V: single 2-cycle (a b) → **per-index S₂{a, b}**.
The oracle reports this as the output symmetry — XᵀX is symmetric in its
two indices, which is exactly what we expect.

### Cautionary note: the C3 axis-merging bug

`einsum('ijk,jki->ik', T, T)` with C3 symmetry declared on T was falsely
detected as having S2{i,k} output symmetry. The root cause was orbit-based
axis merging in the bipartite graph construction.

C3 acting on {i,j,k} has a single orbit {i,j,k}, so all three axes were
merged into one U-vertex per operand. With merged vertices, labels `i` and
`k` received identical column fingerprints, and the fingerprint fast path
promoted them to S2{i,k}. However, C3 contains only 3-cycles — no
transpositions — so the 2-cycle (i k) is not a valid automorphism.

The fix has two parts:

1. **Removed orbit-based merging.** Each axis now gets its own U-vertex
   regardless of declared symmetry. This ensures that the graph topology
   faithfully reflects the actual incidence structure.
2. **Expanded the sigma-loop.** Per-operand symmetry generators are fed
   directly into the sigma-loop (Source A above) alongside identical-operand
   swap generators (Source B). Dimino's algorithm closes the group from the
   discovered pi generators, producing C3 (not S2) for this case.

### V-side and W-side

V-side groups are symmetries of the output tensor — they reduce the number of
unique output elements that need to be computed. W-side groups are symmetries
among the contracted (summed) labels — they reduce the number of unique
summation terms. Both contribute multiplicatively to the cost reduction:

```
cost = dense_cost × (unique_output / total_output) × (unique_inner / total_inner)
```

opt_einsum decomposes contractions into pairwise steps and finds the
optimal path by accumulating costs. At each pairwise step, if the oracle
detects a W-symmetry and **all** the W-group's labels are present as
contracted indices in that specific step, the inner savings are applied.
If any of the W-group's labels were contracted at an earlier step (and
are no longer present), the inner reduction is skipped.

For example, in `einsum('abij,abkl->ijkl', T, T)` where T is symmetric
in `(a,b)`, both `a` and `b` are contracted in a single step → the inner
reduction applies. In contrast, for a multi-step path where `a` and `b`
are contracted in separate steps, each step only sees one of those
labels — the inner reduction is correctly skipped.

This behaviour is controlled by `we.configure(use_inner_symmetry=True/False)`.
In the contraction path table, `[W✓: ...]` indicates the inner reduction was
applied, while `[W: ...]` indicates it was detected but not applied at that
step.

The oracle returns a `SubsetSymmetry` dataclass with `.output` (V-side) and
`.inner` (W-side) fields.

## Complexity bound

The oracle evaluates each subset at most once. For a contract with `N` operands
and groups of sizes `k₁, k₂, …`:

- Number of subsets visited: at most `2^N` (usually much less — path algorithms
  visit only O(N²) subsets in practice for greedy and branch-bounded search).
- Per-subset cost: `O(m! · poly(n))` where `m = max(kᵢ)`.
- Total: `O(2^N · m! · poly(n))`, dominated by the path algorithm itself.

For the common case of a single pair of identical operands (`m = 2`):
per-subset cost is `O(poly(n))`.

---

## Exact group detection and Burnside counting

The σ-loop collects all valid π permutations as `Permutation`
objects (generators) and builds a `PermutationGroup` directly.

These generators define a `PermutationGroup` on the label set. When the
generated group equals S_k (checked via `order == k!`), the existing
`C(n+k-1, k)` formula applies as before. When the group is a proper subgroup
(e.g., C₃ from `einsum('ij,jk,ki->', A, A, A)`), Burnside's lemma gives the
exact unique element count.

### Worked example: tr(A³)

For `einsum('ij,jk,ki->', A, A, A)` with the same n×n matrix A:

1. The σ-loop tries all 6 permutations of {operand 0, 1, 2}
2. Only 3 produce valid π's: identity, (i→j→k), (i→k→j)
3. These are the generators of C₃ (cyclic group of order 3)
4. Burnside counting on W-labels {i,j,k} with dimension n:
   - Identity: n³ fixed tuples (every tuple is fixed)
   - 3-cycle (ijk): n fixed tuples (only i=j=k tuples)
   - 3-cycle (ikj): n fixed tuples
   - Total: (n³ + 2n) / 3

For n=10: 340 unique elements instead of 220 (S₃), giving a more accurate
(higher) FLOP estimate.
