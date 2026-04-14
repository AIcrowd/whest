# Symmetry Explorer: Unified Variable-Centric Redesign

## Problem

The symmetry explorer's custom mode has three issues:

1. **Symmetry is per-operand-slot, not per-variable.** If X appears twice with different symmetry settings, the meaning is ambiguous.
2. **Custom generators are silently collapsed to S_k.** The cycle notation UI is cosmetic — `handleAnalyze()` line 262-265 returns `'symmetric'` for all custom generators.
3. **Generated Python code has a `we`/`me` alias bug.** The import says `import whest as me` but all references use `we.`.

## Design

### Architecture: Variable Dictionary + Einsum Expression

Replace the current per-slot model with a two-part interface:

- **Variable Dictionary:** Define each tensor once with its name, rank, and symmetry.
- **Expression Panel:** Write an einsum expression referencing variables by name.

Preset examples become pre-filled states of the same unified interface. Clicking a preset populates the variable dictionary and expression. Users can then modify anything freely.

### Variable Card

Each variable is shown as an expanded card with these UI elements:

| Element | Behavior |
|---------|----------|
| **Name** | Editable text input, max 8 chars, monospace font |
| **Rank stepper** | `−` / `+` buttons, range 1–8. Changes the number of axis chips. |
| **Symmetry type** | Radio-style toggle buttons: `dense`, `S_k`, `C_k`, `D_k`, `custom` |
| **Axis chips** | One per axis (0 to rank-1). Toggle to select which axes participate in the symmetry group. Hidden when symmetry is `dense`. |
| **Cycle notation input** | Text field, visible only when symmetry is `custom`. User enters generators like `(0 1)(2 3), (0 2)(1 3)`. Indices are 0-based within the selected axes. |
| **Summary badge** | Computed: group name + order (e.g. "C₃(1,2,3) — order: 3"). Updates live. |
| **Remove button** | `×` in top-right corner. Disabled when only one variable remains. |

A dashed `+ Add Variable` card appears at the end of the variable row.

**Constraints:**
- Named groups (S_k, C_k, D_k) require at least 2 axes selected.
- Custom generators require at least 1 generator entered.
- k in S_k/C_k/D_k equals the number of selected axes, not the rank.

### Expression Panel

Two input fields side by side:

1. **Subscripts + output:** e.g. `aijk,ab→ijkb`
2. **Operand names:** e.g. `T, W`

Wrapped in `einsum('...', ...)` chrome so it reads naturally.

### Real-Time Validation

All validation runs on every keystroke/change. The Analyze button is disabled until all checks pass. Checks:

1. **Subscript length matches operand rank.** Each comma-separated subscript must have exactly as many characters as the corresponding operand's rank.
2. **All output labels exist in at least one input subscript.**
3. **All operand names reference defined variables.**
4. **No duplicate labels within a single subscript.**
5. **Subscripts contain only lowercase letters.**
6. **Custom generator cycle indices are within range** of the number of selected axes.
7. **At least one operand** in the expression.

Errors shown inline below the expression, highlighted in red, as the user types. Example: `⚠ Subscript "aij" has 3 labels but T has rank 4`.

### Real-Time Python Code Preview

A syntax-highlighted code block with a Copy button, updating live as variables and expression change. Uses the existing `PythonHighlight` component (regex-based tokenizer with spans for keywords, strings, numbers, comments, function calls).

**Code structure:**

```python
import whest as we

n = 5

def my_symmetrize(shape, group):
    """Random tensor with given symmetry (Reynolds averaged)."""
    data = we.random.randn(*shape)
    data = sum(we.transpose(data, g.array_form) for g in group.elements()) / group.order()
    return we.as_symmetric(data, symmetry=group)

# --- Variables ---
T = my_symmetrize((n,n,n,n), we.PermutationGroup.cyclic(3, axes=(1,2,3)))
W = we.random.randn(n, n)

# --- Expression ---
path, info = we.einsum_path('aijk,ab->ijkb', T, W)
print(info)
```

**Rules:**
- `my_symmetrize` helper is only included when at least one variable has symmetry. Dense-only expressions omit it.
- Dense variables use `we.random.randn(n, n, ...)` directly.
- Named groups use factory methods: `we.PermutationGroup.symmetric(k, axes=(...))`, `.cyclic(...)`, `.dihedral(...)`.
- Custom generators use: `we.PermutationGroup(we.Permutation(we.Cycle(0, 1)(2, 3)), ..., axes=(...))`.
- The import alias is `we` (fixing the current `me` bug).
- The dimension `n` is controlled by the existing dimension slider.

### Visual Symmetry Indicators (Color System)

Each variable gets a unique color from a palette, assigned in definition order. This color is used everywhere the variable appears — providing traceability from definition through to the algorithm visualization.

Symmetry types get a distinguishing icon (shown alongside the variable's color):

| Symmetry | Icon |
|----------|------|
| dense | none |
| S_k | ◆ diamond |
| C_k | ↻ cycle arrow |
| D_k | ⬢ hexagon |
| custom | ⚙ gear |

Default variable color palette (can be reused if >6 variables): `#4a7cff`, `#ffb74d`, `#bb86fc`, `#ec4899`, `#22c55e`, `#94a3b8`.

These indicators appear on:
- **Variable cards:** colored border + symmetry icon badge with group name
- **Expression bar:** operand names colored to match their variable card
- **Bipartite graph:** U-nodes colored by their variable; symmetric-axis U-nodes get thicker border + symmetry icon; edges inherit variable color
- **Incidence matrix:** row labels colored by variable
- **V/W labels:** labels touched by a symmetry group are tinted with the variable's color

### Preset Integration

The preset card grid remains at the top for one-click selection. Clicking a preset:

1. Populates the variable dictionary with the right variables, ranks, and symmetries.
2. Fills the expression field.
3. Highlights the preset card as "active."

Editing any variable or expression field clears the active preset indicator (visual cue that it's now custom).

**Preset → variable mapping examples:**

| Preset | Variables | Expression |
|--------|-----------|------------|
| Gram matrix | X: rank 2, dense | `einsum('ia,ib→ab', X, X)` |
| Triple outer (S3) | X: rank 2, dense | `einsum('ia,ib,ic→abc', X, X, X)` |
| Undirected 4-cycle | S: rank 2, S₂(0,1) | `einsum('ij,jk,kl,li→ijkl', S, S, S, S)` |
| Declared C₃ | T: rank 4, C₃(1,2,3); W: rank 2, dense | `einsum('aijk,ab→ijkb', T, W)` |
| Declared D₄ | T: rank 5, D₄(1,2,3,4); W: rank 2, dense | `einsum('aijkl,ab→ijklb', T, W)` |
| Tr(A·A) | A: rank 2, dense | `einsum('ij,ji→', A, A)` |
| A·A (no symmetry) | A: rank 2, dense | `einsum('ij,jk→ik', A, A)` |
| A·B·A (mixed) | A: rank 2, dense; B: rank 2, dense | `einsum('ij,jk,kl→il', A, B, A)` |

### Bug Fixes (Included in This Work)

1. **`we`/`me` alias bug:** Generated Python uses `we.` but imports as `me`. Fix: use `we` consistently throughout `generatePythonCode()` and `generateCustomPythonCode()`.

2. **Custom generators collapsed to S_k:** `handleAnalyze()` line 262-265 returns `'symmetric'` for custom generators. Fix: parse cycle notation and pass actual generator permutations to the algorithm engine.

3. **Algorithm engine doesn't use custom groups:** The JS `buildBipartite()` only handles string symmetry types (`'symmetric'`, `'cyclic'`, `'dihedral'`) and `{type, axes}` objects. Fix: extend to accept custom generator arrays and build the correct equivalence classes from them.

### Files Changed

| File | Change |
|------|--------|
| `src/data/examples.js` | Add variable-dictionary metadata to each preset (rank, symmetry type, axes) |
| `src/components/ExampleChooser.jsx` | Rewrite custom builder: variable cards, expression panel, real-time validation, real-time Python preview, preset population |
| `src/engine/algorithm.js` | Support custom generator permutations in `buildBipartite()` and `findDeclaredGroupForLabels()` |
| `src/components/BipartiteGraph.jsx` | Color U-nodes and edges by variable symmetry color |
| `src/components/MatrixView.jsx` | Color row labels by variable symmetry color |
| `src/styles.css` | New styles for variable cards, axis chips, symmetry badges, expression panel |

### What Does Not Change

- The 7-step pipeline architecture (bipartite → matrix → σ-loop → group → Burnside → cost)
- The algorithm engine's core logic (σ-loop, π derivation, Dimino's algorithm)
- The SigmaLoop, GroupView, BurnsideView, CostView components (except receiving color info)
- The dimension slider
- The permutation.js library
