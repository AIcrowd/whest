# Symmetry Explorer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the symmetry explorer's custom mode into a unified variable-dictionary + einsum-expression interface with real-time validation, live Python preview, and color-coded symmetry indicators flowing through all visualizations.

**Architecture:** The ExampleChooser component is rewritten as a two-panel interface (variable cards + expression). Preset examples become pre-filled states of this same interface. The algorithm engine is extended to support custom generator permutations. Color metadata flows from variable definitions through all visualization components.

**Tech Stack:** React 19.2, Vite, vanilla CSS, KaTeX (existing). No new dependencies.

**Spec:** `.aicrowd/superpowers/specs/2026-04-14-symmetry-explorer-redesign-design.md`

---

## File Structure

All paths relative to `docs/visualization/symmetry-explorer/`.

| File | Role | Change |
|------|------|--------|
| `src/data/examples.js` | Preset example definitions | Add `variables` field (name, rank, symmetry, axes, generators) to each preset |
| `src/engine/algorithm.js` | Core algorithm engine | Extend `buildBipartite()` to support custom generators; fix `findDeclaredGroupForLabels()` |
| `src/engine/cycleParser.js` | **New.** Parse cycle notation strings | Standalone parser: `"(0 1)(2 3)"` → `[[0,1],[2,3]]` |
| `src/engine/validation.js` | **New.** Real-time einsum validation | Pure functions: validate subscripts, output, operand names, ranks, generators |
| `src/engine/pythonCodegen.js` | **New.** Python code generation | Pure function: variables + expression → Python string with `my_symmetrize` helper |
| `src/engine/colorPalette.js` | **New.** Variable color assignment | Assigns colors from palette, provides symmetry icons |
| `src/components/ExampleChooser.jsx` | Main input UI | Full rewrite: variable cards, expression panel, real-time validation, preset population |
| `src/components/BipartiteGraph.jsx` | Bipartite graph SVG | Accept `variableColors` prop, color U-nodes and edges by variable |
| `src/components/MatrixView.jsx` | Incidence matrix | Accept `variableColors` prop, color row labels by variable |
| `src/App.jsx` | Orchestrator | Thread `variableColors` from ExampleChooser through to visualization components |
| `src/styles.css` | Styles | New styles for variable cards, axis chips, symmetry badges, expression panel |

---

## Task 1: Cycle Notation Parser

**Files:**
- Create: `src/engine/cycleParser.js`

This is a dependency for the custom generator support in both the algorithm engine and the UI validation.

- [ ] **Step 1: Create the cycle parser module**

```javascript
// src/engine/cycleParser.js

/**
 * Parse a comma-separated list of generators in cycle notation.
 * Input:  "(0 1)(2 3), (0 2)(1 3)"
 * Output: [[[0,1],[2,3]], [[0,2],[1,3]]]
 * Each generator is a list of cycles, each cycle is a list of integers.
 * Returns { generators, error } where error is null on success.
 */
export function parseCycleNotation(input) {
  const trimmed = input.trim();
  if (!trimmed) return { generators: null, error: 'Empty input' };

  // Split by comma that is NOT inside parentheses
  const genStrings = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i <= trimmed.length; i++) {
    if (i === trimmed.length || (trimmed[i] === ',' && depth === 0)) {
      const part = trimmed.slice(start, i).trim();
      if (part) genStrings.push(part);
      start = i + 1;
    } else if (trimmed[i] === '(') depth++;
    else if (trimmed[i] === ')') depth--;
  }

  if (genStrings.length === 0) return { generators: null, error: 'No generators found' };

  const generators = [];
  for (const gs of genStrings) {
    const cycles = [];
    const cycleRe = /\(([^)]+)\)/g;
    let match;
    while ((match = cycleRe.exec(gs)) !== null) {
      const nums = match[1].trim().split(/[\s,]+/).map(Number);
      if (nums.some(isNaN)) return { generators: null, error: `Invalid number in cycle: "${match[0]}"` };
      if (nums.length < 2) return { generators: null, error: `Cycle must have at least 2 elements: "${match[0]}"` };
      if (new Set(nums).size !== nums.length) return { generators: null, error: `Duplicate elements in cycle: "${match[0]}"` };
      cycles.push(nums);
    }
    if (cycles.length === 0) return { generators: null, error: `No cycles found in generator: "${gs}"` };
    generators.push(cycles);
  }

  return { generators, error: null };
}

/**
 * Convert parsed cycles to array-form permutation of given size.
 * cycles: [[0,1],[2,3]], size: 4 → [1,0,3,2]
 */
export function cyclesToArrayForm(cycles, size) {
  const arr = Array.from({ length: size }, (_, i) => i);
  for (const cycle of cycles) {
    for (let i = 0; i < cycle.length; i++) {
      arr[cycle[i]] = cycle[(i + 1) % cycle.length];
    }
  }
  return arr;
}

/**
 * Get all unique indices mentioned in a list of generators.
 * generators: [[[0,1],[2,3]], [[0,2],[1,3]]] → [0,1,2,3]
 */
export function generatorIndices(generators) {
  const indices = new Set();
  for (const gen of generators) {
    for (const cycle of gen) {
      for (const idx of cycle) indices.add(idx);
    }
  }
  return [...indices].sort((a, b) => a - b);
}
```

- [ ] **Step 2: Verify the parser works by running the dev server**

Run: `cd docs/visualization/symmetry-explorer && npx vite --open`

Open browser console and test manually (the module is imported by later tasks). This step is just to make sure the file has no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/cycleParser.js
git commit -m "feat(explorer): add cycle notation parser for custom generators"
```

---

## Task 2: Validation Engine

**Files:**
- Create: `src/engine/validation.js`

Pure validation functions used by the UI for real-time feedback.

- [ ] **Step 1: Create the validation module**

```javascript
// src/engine/validation.js
import { parseCycleNotation, generatorIndices } from './cycleParser.js';

/**
 * Validate the full state: variables + expression.
 * Returns { valid: boolean, errors: string[] }
 *
 * variables: [{ name, rank, symmetry, symAxes, generators }]
 * subscripts: string (comma-separated, e.g. "aijk,ab")
 * output: string (e.g. "ijkb")
 * operandNames: string (comma-separated, e.g. "T, W")
 */
export function validateAll(variables, subscripts, output, operandNames) {
  const errors = [];

  // Parse expression parts
  const subs = subscripts.split(',').map(s => s.trim()).filter(Boolean);
  const ops = operandNames.split(',').map(s => s.trim()).filter(Boolean);
  const outLabels = output.trim();

  // --- Variable-level checks ---
  const varNames = new Set();
  for (let i = 0; i < variables.length; i++) {
    const v = variables[i];
    if (!v.name.trim()) {
      errors.push(`Variable ${i + 1}: name is empty`);
      continue;
    }
    varNames.add(v.name.trim());

    if (v.rank < 1) errors.push(`Variable "${v.name}": rank must be at least 1`);

    if (v.symmetry !== 'none' && v.symmetry !== 'custom') {
      if (!v.symAxes || v.symAxes.length < 2) {
        errors.push(`Variable "${v.name}": ${v.symmetry} symmetry requires at least 2 axes selected`);
      }
      if (v.symAxes && v.symAxes.some(a => a >= v.rank)) {
        errors.push(`Variable "${v.name}": axis index out of range (rank is ${v.rank})`);
      }
    }

    if (v.symmetry === 'custom') {
      if (!v.generators || !v.generators.trim()) {
        errors.push(`Variable "${v.name}": enter at least one generator`);
      } else {
        const { error } = parseCycleNotation(v.generators);
        if (error) errors.push(`Variable "${v.name}": ${error}`);
        else {
          // Check cycle indices are within selected axes range
          const { generators } = parseCycleNotation(v.generators);
          const indices = generatorIndices(generators);
          const numAxes = v.symAxes ? v.symAxes.length : v.rank;
          const maxIdx = Math.max(...indices);
          if (maxIdx >= numAxes) {
            errors.push(`Variable "${v.name}": cycle index ${maxIdx} exceeds number of selected axes (${numAxes})`);
          }
        }
      }
    }
  }

  // --- Expression-level checks ---
  if (ops.length === 0) {
    errors.push('Expression needs at least one operand');
  }

  if (subs.length !== ops.length) {
    errors.push(`${subs.length} subscript(s) but ${ops.length} operand(s)`);
  }

  // Check operand names reference defined variables
  for (const op of ops) {
    if (op && !varNames.has(op)) {
      errors.push(`Operand "${op}" is not a defined variable`);
    }
  }

  // Check subscript lengths match operand ranks
  for (let i = 0; i < Math.min(subs.length, ops.length); i++) {
    const sub = subs[i];
    const opName = ops[i];
    if (!sub || !opName) continue;

    // Subscripts must be lowercase letters only
    if (!/^[a-z]+$/.test(sub)) {
      errors.push(`Subscript "${sub}" must be lowercase letters only`);
      continue;
    }

    // No duplicate labels within one subscript
    if (new Set(sub).size !== sub.length) {
      const dupes = [...sub].filter((c, j) => sub.indexOf(c) !== j);
      errors.push(`Subscript "${sub}" has duplicate label "${dupes[0]}"`);
    }

    // Find the variable for this operand
    const v = variables.find(v => v.name.trim() === opName);
    if (v && sub.length !== v.rank) {
      errors.push(`Subscript "${sub}" has ${sub.length} labels but ${opName} has rank ${v.rank}`);
    }
  }

  // Check output labels exist in inputs
  if (outLabels && !/^[a-z]*$/.test(outLabels)) {
    errors.push('Output must be lowercase letters only');
  } else {
    const allInputLabels = new Set(subs.join(''));
    for (const ch of outLabels) {
      if (!allInputLabels.has(ch)) {
        errors.push(`Output label "${ch}" not found in any input subscript`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/validation.js
git commit -m "feat(explorer): add real-time validation engine for einsum expressions"
```

---

## Task 3: Python Code Generator

**Files:**
- Create: `src/engine/pythonCodegen.js`

Pure function that generates the Python preview string.

- [ ] **Step 1: Create the code generator module**

```javascript
// src/engine/pythonCodegen.js

/**
 * Generate Python code from variable definitions and expression.
 *
 * variables: [{ name, rank, symmetry, symAxes, generators }]
 * subscripts: "aijk,ab"
 * output: "ijkb"
 * operandNames: "T, W"
 * dimensionN: number
 *
 * Returns a string of valid Python code.
 */
export function generatePython(variables, subscripts, output, operandNames, dimensionN) {
  const n = dimensionN;
  const lines = [];
  lines.push('import whest as we');
  lines.push('');
  lines.push(`n = ${n}`);

  // Check if any variable has symmetry — if so, include my_symmetrize
  const hasAnySym = variables.some(v => v.symmetry !== 'none');
  if (hasAnySym) {
    lines.push('');
    lines.push('def my_symmetrize(shape, group):');
    lines.push('    """Random tensor with given symmetry (Reynolds averaged)."""');
    lines.push('    data = we.random.randn(*shape)');
    lines.push('    data = sum(we.transpose(data, g.array_form) for g in group.elements()) / group.order()');
    lines.push('    return we.as_symmetric(data, symmetry=group)');
  }

  lines.push('');
  lines.push('# --- Variables ---');

  // Deduplicate: only emit each variable name once
  const defined = new Set();
  for (const v of variables) {
    const name = v.name.trim() || 'X';
    if (defined.has(name)) continue;
    defined.add(name);

    const shape = Array(v.rank).fill('n').join(', ');

    if (v.symmetry === 'none') {
      lines.push(`${name} = we.random.randn(${shape})`);
    } else if (v.symmetry === 'custom') {
      // Parse generators and build PermutationGroup constructor
      const gens = v.generators.trim();
      const axes = v.symAxes ? v.symAxes.join(', ') : [...Array(v.rank).keys()].join(', ');
      const genParts = parseGensForPython(gens);
      if (genParts.length === 1) {
        lines.push(`${name} = my_symmetrize((${shape}), we.PermutationGroup(`);
        lines.push(`    ${genParts[0]},`);
        lines.push(`    axes=(${axes})`);
        lines.push('))');
      } else {
        lines.push(`${name} = my_symmetrize((${shape}), we.PermutationGroup(`);
        for (const gp of genParts) {
          lines.push(`    ${gp},`);
        }
        lines.push(`    axes=(${axes})`);
        lines.push('))');
      }
    } else {
      // Named group: symmetric, cyclic, dihedral
      const groupFn = v.symmetry === 'symmetric' ? 'symmetric'
        : v.symmetry === 'cyclic' ? 'cyclic' : 'dihedral';
      const axes = v.symAxes ? v.symAxes.join(', ') : [...Array(v.rank).keys()].join(', ');
      const k = v.symAxes ? v.symAxes.length : v.rank;
      lines.push(`${name} = my_symmetrize((${shape}), we.PermutationGroup.${groupFn}(${k}, axes=(${axes})))`);
    }
  }

  lines.push('');
  lines.push('# --- Expression ---');

  const subs = subscripts.trim();
  const out = output.trim();
  const ops = operandNames.trim();
  lines.push(`path, info = we.einsum_path('${subs}->${out}', ${ops})`);
  lines.push('print(info)');

  return lines.join('\n');
}

/**
 * Convert cycle notation generators into Python expression strings.
 * "(0 1)(2 3), (0 2)(1 3)" → [
 *   "we.Permutation(we.Cycle(0, 1)(2, 3))",
 *   "we.Permutation(we.Cycle(0, 2)(1, 3))"
 * ]
 */
function parseGensForPython(input) {
  // Split generators by comma outside parens
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i <= input.length; i++) {
    if (i === input.length || (input[i] === ',' && depth === 0)) {
      const part = input.slice(start, i).trim();
      if (part) parts.push(part);
      start = i + 1;
    } else if (input[i] === '(') depth++;
    else if (input[i] === ')') depth--;
  }

  return parts.map(part => {
    // Convert "(0 1)(2 3)" → "we.Cycle(0, 1)(2, 3)"
    const pyCycles = part.replace(/\(([^)]+)\)/g, (_, inner) => {
      const nums = inner.trim().split(/[\s,]+/).join(', ');
      return `(${nums})`;
    });
    return `we.Permutation(we.Cycle${pyCycles})`;
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/pythonCodegen.js
git commit -m "feat(explorer): add Python code generator with my_symmetrize helper"
```

---

## Task 4: Color Palette Module

**Files:**
- Create: `src/engine/colorPalette.js`

Assigns a unique color per variable and provides symmetry icons.

- [ ] **Step 1: Create the color palette module**

```javascript
// src/engine/colorPalette.js

const PALETTE = [
  '#4a7cff', '#ffb74d', '#bb86fc', '#ec4899', '#22c55e', '#94a3b8',
  '#ef4444', '#06b6d4', '#f59e0b', '#8b5cf6',
];

const SYMMETRY_ICONS = {
  none: '',
  symmetric: '\u25C6',  // ◆
  cyclic: '\u27F3',      // ⟳
  dihedral: '\u2B22',    // ⬢
  custom: '\u2699',      // ⚙
};

/**
 * Build a color map from variable definitions.
 * Returns { [variableName]: { color, icon, symmetryLabel } }
 */
export function buildVariableColors(variables) {
  const colors = {};
  const seen = new Set();
  let idx = 0;

  for (const v of variables) {
    const name = v.name.trim();
    if (!name || seen.has(name)) continue;
    seen.add(name);

    const color = PALETTE[idx % PALETTE.length];
    idx++;

    let symmetryLabel = 'dense';
    if (v.symmetry === 'symmetric') {
      const k = v.symAxes ? v.symAxes.length : v.rank;
      symmetryLabel = `S${k}`;
    } else if (v.symmetry === 'cyclic') {
      const k = v.symAxes ? v.symAxes.length : v.rank;
      symmetryLabel = `C${k}`;
    } else if (v.symmetry === 'dihedral') {
      const k = v.symAxes ? v.symAxes.length : v.rank;
      symmetryLabel = `D${k}`;
    } else if (v.symmetry === 'custom') {
      symmetryLabel = 'custom';
    }

    colors[name] = {
      color,
      icon: SYMMETRY_ICONS[v.symmetry] || '',
      symmetryLabel,
      symmetry: v.symmetry,
    };
  }

  return colors;
}

export { PALETTE, SYMMETRY_ICONS };
```

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/colorPalette.js
git commit -m "feat(explorer): add variable color palette and symmetry icons"
```

---

## Task 5: Update Example Data with Variable Metadata

**Files:**
- Modify: `src/data/examples.js`

Add a `variables` array to each preset example so presets can populate the variable dictionary.

- [ ] **Step 1: Rewrite examples.js with variable metadata**

Each preset gets a `variables` field: `[{ name, rank, symmetry, symAxes, generators }]` and an `expression` field with `{ subscripts, output, operandNames }` strings.

```javascript
// src/data/examples.js

export const EXAMPLES = [
  {
    id: 'gram',
    name: 'Gram matrix',
    formula: "einsum('ia,ib→ab', X, X)",
    description: 'XᵀX is symmetric — detects S2 on output labels {a,b}',
    expectedGroup: 'S2{a,b}',
    color: '#4a7cff',
    // Variable dictionary metadata
    variables: [
      { name: 'X', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ia,ib', output: 'ab', operandNames: 'X, X' },
  },
  {
    id: 'triple-outer',
    name: 'Triple outer (S3)',
    formula: "einsum('ia,ib,ic→abc', X, X, X)",
    description: '3 identical operands → full S3 on output. Shows how S3 needs all 3! = 6 permutations',
    expectedGroup: 'S3{a,b,c}',
    color: '#23B761',
    variables: [
      { name: 'X', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ia,ib,ic', output: 'abc', operandNames: 'X, X, X' },
  },
  {
    id: 'outer',
    name: 'Outer product',
    formula: "einsum('ab,cd→abcd', X, X)",
    description: 'Detects block symmetry — swapping (a,b)↔(c,d)',
    expectedGroup: 'S2{a,c}×S2{b,d}',
    color: '#3ddc84',
    variables: [
      { name: 'X', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ab,cd', output: 'abcd', operandNames: 'X, X' },
  },
  {
    id: 'triangle',
    name: 'Directed triangle',
    formula: "einsum('ij,jk,ki→ijk', A, A, A)",
    description: 'Cyclic chain — only rotations are valid (not reflections), so C3 not S3',
    expectedGroup: 'C3{i,j,k}',
    color: '#ffb74d',
    variables: [
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ij,jk,ki', output: 'ijk', operandNames: 'A, A, A' },
  },
  {
    id: 'four-cycle',
    name: 'Undirected 4-cycle',
    formula: "einsum('ij,jk,kl,li→ijkl', S, S, S, S)",
    description: 'S symmetric ⇒ axes collapse, enabling reflections. C4 + reflections = D4',
    expectedGroup: 'D4{i,j,k,l}',
    color: '#bb86fc',
    variables: [
      { name: 'S', rank: 2, symmetry: 'symmetric', symAxes: [0, 1], generators: '' },
    ],
    expression: { subscripts: 'ij,jk,kl,li', output: 'ijkl', operandNames: 'S, S, S, S' },
  },
  {
    id: 'trace-product',
    name: 'Tr(A·A)',
    formula: "einsum('ij,ji→', A, A)",
    description: 'No free labels — symmetry is on W (summed) side. S2{i,j} reduces the contraction cost',
    expectedGroup: 'W: S2{i,j}',
    color: '#94A3B8',
    variables: [
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ij,ji', output: '', operandNames: 'A, A' },
  },
  {
    id: 'declared-c3',
    name: 'Declared C₃ (contraction)',
    formula: "einsum('aijk,ab→ijkb', T, W)",
    description: 'T has C₃ on axes {i,j,k}, contracted with W on index a. Non-identical operands → σ-loop empty, but declared C₃ is preserved (not promoted to S₃)',
    expectedGroup: 'C3{i,j,k}',
    color: '#F59E0B',
    variables: [
      { name: 'T', rank: 4, symmetry: 'cyclic', symAxes: [1, 2, 3], generators: '' },
      { name: 'W', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'aijk,ab', output: 'ijkb', operandNames: 'T, W' },
  },
  {
    id: 'declared-d4',
    name: 'Declared D₄ (contraction)',
    formula: "einsum('aijkl,ab→ijklb', T, W)",
    description: 'T has D₄ on axes {i,j,k,l}, contracted with W on index a. Detects D₄ — without the fix, wrongly promoted to S₄',
    expectedGroup: 'D4{i,j,k,l}',
    color: '#EC4899',
    variables: [
      { name: 'T', rank: 5, symmetry: 'dihedral', symAxes: [1, 2, 3, 4], generators: '' },
      { name: 'W', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'aijkl,ab', output: 'ijklb', operandNames: 'T, W' },
  },
  {
    id: 'matrix-chain',
    name: 'A·A (no symmetry)',
    formula: "einsum('ij,jk→ik', A, A)",
    description: 'Identical operands but different subscript structure → σ-loop finds no valid π',
    expectedGroup: 'trivial',
    color: '#D1D5DB',
    variables: [
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ij,jk', output: 'ik', operandNames: 'A, A' },
  },
  {
    id: 'mixed-chain',
    name: 'A·B·A (mixed)',
    formula: "einsum('ij,jk,kl→il', A, B, A)",
    description: 'A appears twice but B breaks the chain — no identical group forms',
    expectedGroup: 'trivial',
    color: '#E5E7EB',
    variables: [
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
      { name: 'B', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ],
    expression: { subscripts: 'ij,jk,kl', output: 'il', operandNames: 'A, B, A' },
  },
];
```

Note: The old fields (`subscripts`, `output`, `operandNames`, `perOpSymmetry`) are removed. The `expression` object replaces them. App.jsx will be updated to derive the old fields from the new format for the algorithm engine.

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/data/examples.js
git commit -m "feat(explorer): add variable-dictionary metadata to preset examples"
```

---

## Task 6: Extend Algorithm Engine for Custom Generators

**Files:**
- Modify: `src/engine/algorithm.js`

Two changes: (1) accept custom generator arrays in `buildBipartite()`, (2) fix `findDeclaredGroupForLabels()` to handle them.

- [ ] **Step 1: Update buildBipartite to support custom generators**

In `buildBipartite()` (around line 30-44), add a branch for when `opSym` is an object with `type: 'custom'` and a `generators` field. Custom generators should merge the specified axes into one equivalence class (same as named groups — the detailed group structure is handled later in `buildGroup`/`findDeclaredGroupForLabels`).

Find this block in `src/engine/algorithm.js`:

```javascript
    } else if (opSym && typeof opSym === 'object' && opSym.axes) {
      // Partial symmetry: only the specified axes collapse into one class
      const symAxes = opSym.axes;
      if (symAxes.length >= 2) {
        const target = symAxes[0];
        for (let j = 1; j < symAxes.length; j++) {
          classOf[symAxes[j]] = target;
        }
      }
    }
```

Replace with:

```javascript
    } else if (opSym && typeof opSym === 'object' && opSym.axes) {
      // Partial symmetry (named or custom): specified axes collapse into one class
      const symAxes = opSym.axes;
      if (symAxes.length >= 2) {
        const target = symAxes[0];
        for (let j = 1; j < symAxes.length; j++) {
          classOf[symAxes[j]] = target;
        }
      }
    }
```

(The code is actually the same — the existing branch already handles `{type, axes}` objects correctly for bipartite graph construction. The key fix is in `findDeclaredGroupForLabels` below.)

- [ ] **Step 2: Update findDeclaredGroupForLabels for custom generators**

Find this function (around line 348-384) and add handling for `opSym.type === 'custom'` with `opSym.generators`:

```javascript
function findDeclaredGroupForLabels(example, equivLabels) {
  const { subscripts, perOpSymmetry } = example;
  if (!perOpSymmetry) return null;
  const equivSet = new Set(equivLabels);

  for (let opIdx = 0; opIdx < subscripts.length; opIdx++) {
    const opSym = Array.isArray(perOpSymmetry) ? perOpSymmetry[opIdx] : perOpSymmetry;
    if (!opSym) continue;

    const sub = subscripts[opIdx];
    let coveredLabels;
    let symType;
    if (opSym === 'symmetric' || opSym === 'cyclic' || opSym === 'dihedral') {
      coveredLabels = new Set(sub);
      symType = opSym;
    } else if (opSym && typeof opSym === 'object' && opSym.axes) {
      coveredLabels = new Set(opSym.axes.map(ax => sub[ax]));
      symType = opSym.type || 'symmetric';
    } else {
      continue;
    }

    if (coveredLabels.size !== equivSet.size) continue;
    let match = true;
    for (const l of equivSet) {
      if (!coveredLabels.has(l)) { match = false; break; }
    }
    if (!match) continue;

    // For custom generators, build Permutation objects from the cycle arrays
    if (symType === 'custom' && opSym.generators) {
      const { Permutation } = await import('./permutation.js');
      // opSym.generators is an array of cycle arrays: [[[0,1],[2,3]], [[0,2],[1,3]]]
      const gens = opSym.generators.map(cycles => {
        const arr = Array.from({ length: equivLabels.length }, (_, i) => i);
        for (const cycle of cycles) {
          for (let i = 0; i < cycle.length; i++) {
            arr[cycle[i]] = cycle[(i + 1) % cycle.length];
          }
        }
        return new Permutation(arr);
      }).filter(p => !p.isIdentity);
      if (gens.length > 0) return { generators: gens };
    }

    const gens = declaredSymGenerators(symType, equivLabels.length);
    if (gens.length > 0) return { generators: gens };
  }
  return null;
}
```

Wait — this uses `await import` which won't work in a synchronous function. Since `Permutation` is already imported at the top of algorithm.js, just use it directly:

Find the import line at the top of `algorithm.js`:

```javascript
import { Permutation, dimino, burnsideCount } from './permutation.js';
```

This is already imported. So in `findDeclaredGroupForLabels`, just before `const gens = declaredSymGenerators(symType, equivLabels.length);`, add the custom branch:

```javascript
    if (symType === 'custom' && opSym.generators) {
      const gens = opSym.generators.map(cycles => {
        const arr = Array.from({ length: equivLabels.length }, (_, i) => i);
        for (const cycle of cycles) {
          for (let i = 0; i < cycle.length; i++) {
            arr[cycle[i]] = cycle[(i + 1) % cycle.length];
          }
        }
        return new Permutation(arr);
      }).filter(p => !p.isIdentity);
      if (gens.length > 0) return { generators: gens };
      return null;
    }

    const gens = declaredSymGenerators(symType, equivLabels.length);
```

- [ ] **Step 3: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/engine/algorithm.js
git commit -m "fix(explorer): support custom generator permutations in algorithm engine"
```

---

## Task 7: Rewrite ExampleChooser Component

**Files:**
- Modify: `src/components/ExampleChooser.jsx` (full rewrite)

This is the largest task. The component is rewritten to implement the variable-dictionary + expression interface.

- [ ] **Step 1: Rewrite ExampleChooser.jsx**

The full rewrite replaces the 608-line file. Key sections:

1. **State:** `variables` array, `subscriptsStr`, `outputStr`, `operandNamesStr`, `activePresetIdx`
2. **Preset loading:** `loadPreset(idx)` populates state from `EXAMPLES[idx].variables` and `.expression`
3. **Variable card rendering:** name input, rank stepper, symmetry toggle, axis chips, generator input, summary badge
4. **Expression panel:** subscripts input, output input, operand names input
5. **Real-time validation:** calls `validateAll()` on every state change, shows errors inline
6. **Real-time Python preview:** calls `generatePython()` on every state change
7. **Analyze handler:** builds the `example` object from state and calls `onCustomExample()`

The component should:
- Call `loadPreset(0)` on mount (default to Gram matrix)
- Clear `activePresetIdx` when any variable or expression field is edited
- Disable the Analyze button when validation fails
- Show the Python code preview with syntax highlighting and copy button at all times

Write the complete component. It should import from the new modules:

```javascript
import { useState, useCallback, useMemo, useEffect } from 'react';
import { validateAll } from '../engine/validation.js';
import { generatePython } from '../engine/pythonCodegen.js';
import { buildVariableColors, SYMMETRY_ICONS } from '../engine/colorPalette.js';
import { parseCycleNotation, cyclesToArrayForm, generatorIndices } from '../engine/cycleParser.js';
```

The `onCustomExample` callback should produce an object compatible with the algorithm engine:

```javascript
{
  id: activePresetIdx >= 0 ? EXAMPLES[activePresetIdx].id : 'custom',
  name: activePresetIdx >= 0 ? EXAMPLES[activePresetIdx].name : 'Custom',
  formula: `einsum('${subs}->${out}', ${ops})`,
  subscripts: subsArr,          // ['aijk', 'ab']
  output: out,                  // 'ijkb'
  operandNames: opsArr,         // ['T', 'W']
  perOpSymmetry: perOpSymmetry, // [{ type, axes, generators? }, null, ...]
  description: '...',
  expectedGroup: '',
  color: '#7C3AED',
}
```

The `perOpSymmetry` array must map each operand slot (not each variable) to its symmetry spec:
- `null` for dense
- `'symmetric'` when S_k on all axes of that operand
- `{ type: 'symmetric'|'cyclic'|'dihedral', axes: [...] }` for named groups on a subset
- `{ type: 'custom', axes: [...], generators: [[[0,1],[2,3]], ...] }` for custom generators

The component should also expose `variableColors` (from `buildVariableColors`) to the parent via a callback or by including it in the custom example object.

The `PythonHighlight` component and `highlightPython` function should be kept from the current file (lines 510-608) — they work fine and just need the syntax highlighting and copy button they already have.

- [ ] **Step 2: Verify the dev server starts and the UI renders**

Run: `cd docs/visualization/symmetry-explorer && npx vite`

Open browser, verify:
- Preset grid displays
- Clicking a preset populates variable cards + expression
- Variable cards show name, rank, symmetry type, axis chips
- Expression panel shows subscripts + output + operand names
- Python code preview updates in real-time
- Validation errors appear/disappear as expected
- Analyze button works and triggers the pipeline

- [ ] **Step 3: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/components/ExampleChooser.jsx
git commit -m "feat(explorer): rewrite ExampleChooser with variable-dictionary interface"
```

---

## Task 8: Update App.jsx to Thread Colors and New Data Format

**Files:**
- Modify: `src/App.jsx`

App needs to: (1) pass `variableColors` to visualization components, (2) handle the new example format where `expression` replaces top-level `subscripts`/`output`/`operandNames`.

- [ ] **Step 1: Update App.jsx**

Key changes:

1. Import `parseCycleNotation` from `../engine/cycleParser.js`. The `example` object from presets now has `expression.subscripts` etc. instead of top-level fields. Normalize this before passing to the pipeline. Add a `normalizeExample` function:

```javascript
function normalizeExample(example) {
  if (example.subscripts) return example; // already normalized (custom or legacy)
  // Convert new format to algorithm-compatible format
  const { expression, variables } = example;
  const subsArr = expression.subscripts.split(',').map(s => s.trim());
  const opsArr = expression.operandNames.split(',').map(s => s.trim());
  // Build perOpSymmetry from variables
  const perOpSymmetry = opsArr.map(opName => {
    const v = variables.find(v => v.name === opName);
    if (!v || v.symmetry === 'none') return null;
    if (v.symmetry === 'symmetric' && v.symAxes && v.symAxes.length === v.rank) return 'symmetric';
    if (v.symmetry === 'custom') {
      const { generators } = parseCycleNotation(v.generators);
      return { type: 'custom', axes: v.symAxes || [...Array(v.rank).keys()], generators };
    }
    return { type: v.symmetry, axes: v.symAxes || [...Array(v.rank).keys()] };
  });
  const hasAnySym = perOpSymmetry.some(s => s !== null);
  return {
    ...example,
    subscripts: subsArr,
    output: expression.output,
    operandNames: opsArr,
    perOpSymmetry: hasAnySym ? perOpSymmetry : null,
  };
}
```

2. Add `variableColors` state, passed down from ExampleChooser and threaded to BipartiteGraph and MatrixView.

3. The ExampleChooser `onCustomExample` callback now receives both the normalized example and the variable colors. Update the callback:

```javascript
const handleCustomExample = useCallback((ex, colors) => {
  setCustomExample(ex);
  setVariableColors(colors);
  setExampleIdx(CUSTOM_IDX);
}, []);
```

4. Pass `variableColors` to `<BipartiteGraph>` and `<MatrixView>`.

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/App.jsx
git commit -m "feat(explorer): thread variable colors through App, normalize new example format"
```

---

## Task 9: Color Bipartite Graph by Variable

**Files:**
- Modify: `src/components/BipartiteGraph.jsx`

Accept `variableColors` prop and use it to color U-node borders, edges, and labels.

- [ ] **Step 1: Update BipartiteGraph.jsx**

1. Add `variableColors` to the destructured props.
2. In the U-node rendering (around line 186-207), use the variable's color for the pill stroke and text fill:

```javascript
const opName = example.operandNames?.[u.opIdx];
const vc = variableColors?.[opName];
const nodeColor = vc?.color || '#FA9E33';
const nodeIcon = vc?.icon || '';
```

Use `nodeColor` for stroke and text fill on the pill. If the variable has symmetry (icon is non-empty), render a small icon indicator next to the pill.

3. In the operand group box rendering (around line 111-128), use the variable color for the box fill and stroke.

4. In the edge rendering (around line 166-184), tint edges with the source U-node's variable color instead of the default gray.

5. Fall back to existing colors when `variableColors` is not provided (backward compatibility).

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/components/BipartiteGraph.jsx
git commit -m "feat(explorer): color bipartite graph nodes and edges by variable"
```

---

## Task 10: Color Matrix View by Variable

**Files:**
- Modify: `src/components/MatrixView.jsx`

Accept `variableColors` prop and color row labels.

- [ ] **Step 1: Update MatrixView.jsx**

1. Add `variableColors` to destructured props.
2. In the row label rendering (around line 32-35), use the variable color:

```javascript
const opName = example.operandNames?.[u.opIdx];
const vc = variableColors?.[opName];
const labelColor = vc?.color;
```

Apply as inline style on the `.op-tag` span.

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/components/MatrixView.jsx
git commit -m "feat(explorer): color matrix row labels by variable"
```

---

## Task 11: Add CSS Styles for New Components

**Files:**
- Modify: `src/styles.css`

Add styles for variable cards, axis chips, symmetry badges, and expression panel.

- [ ] **Step 1: Add new CSS rules**

Append to `src/styles.css`:

```css
/* ── Variable Cards ── */
.var-cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }

.var-card {
  flex: 1; min-width: 200px; max-width: 320px;
  border-radius: 10px; padding: 14px;
  background: rgba(0,0,0,0.08); border: 1.5px solid rgba(148,163,184,0.2);
  transition: border-color 0.15s;
}

.var-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }

.var-name-input {
  font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: bold;
  background: rgba(0,0,0,0.15); border: 1px solid rgba(148,163,184,0.15);
  border-radius: 4px; padding: 2px 8px; width: 80px; color: inherit;
}

.rank-stepper { display: flex; align-items: center; gap: 4px; }
.rank-stepper button {
  width: 22px; height: 22px; border-radius: 4px;
  background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.15);
  color: #94a3b8; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;
}
.rank-stepper span { font-family: 'IBM Plex Mono', monospace; font-size: 14px; width: 20px; text-align: center; }

/* Symmetry type toggle */
.sym-toggles { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 10px; }
.sym-toggle {
  padding: 4px 10px; border-radius: 4px; font-size: 11px;
  background: rgba(148,163,184,0.06); border: 1px solid rgba(148,163,184,0.15);
  cursor: pointer; color: #94a3b8; transition: all 0.15s;
}
.sym-toggle.active { border-color: currentColor; }

/* Axis chips */
.axis-chips { display: flex; gap: 4px; margin-bottom: 8px; }
.axis-chip {
  padding: 3px 10px; border-radius: 4px; font-size: 11px;
  background: rgba(148,163,184,0.06); border: 1px solid rgba(148,163,184,0.12);
  cursor: pointer; color: #64748b; transition: all 0.15s;
}
.axis-chip.selected { border-color: currentColor; }

/* Summary badge */
.sym-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 4px; font-size: 11px;
}
.sym-order { font-size: 10px; color: #64748b; }

/* Add variable card */
.var-card-add {
  flex: 1; min-width: 200px; max-width: 320px;
  border: 1.5px dashed rgba(148,163,184,0.2); border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  color: #64748b; font-size: 13px; cursor: pointer; padding: 14px;
  transition: border-color 0.15s;
}
.var-card-add:hover { border-color: rgba(148,163,184,0.4); }

/* Generator input */
.gen-input {
  font-family: 'IBM Plex Mono', monospace; font-size: 13px;
  padding: 6px 10px; background: rgba(0,0,0,0.2);
  border: 1px solid rgba(148,163,184,0.15); border-radius: 4px;
  width: 100%; color: inherit; margin-top: 4px;
}

/* Expression panel */
.expr-panel { margin: 16px 0; }
.expr-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.expr-chrome { color: #64748b; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
.expr-input {
  flex: 1; font-family: 'IBM Plex Mono', monospace; font-size: 14px;
  padding: 6px 10px; background: rgba(0,0,0,0.2);
  border: 1px solid rgba(148,163,184,0.15); border-radius: 4px; color: inherit;
}
.expr-input.has-error { border-color: rgba(239,68,68,0.5); }

/* Validation errors */
.validation-errors { margin-top: 8px; }
.validation-error {
  color: #ef4444; font-size: 12px; padding: 2px 0;
  display: flex; align-items: center; gap: 6px;
}
.validation-error::before { content: '⚠'; }
```

- [ ] **Step 2: Commit**

```bash
git add docs/visualization/symmetry-explorer/src/styles.css
git commit -m "feat(explorer): add CSS styles for variable cards, expression panel, validation"
```

---

## Task 12: Integration Test — Full Walkthrough

**Files:** None (manual verification)

- [ ] **Step 1: Start the dev server**

Run: `cd docs/visualization/symmetry-explorer && npx vite`

- [ ] **Step 2: Test preset loading**

Click each preset card and verify:
- Variable cards populate correctly (name, rank, symmetry, axes)
- Expression panel fills in
- Python preview updates with correct code
- Pipeline runs and shows all 7 steps

- [ ] **Step 3: Test custom editing**

Starting from Gram matrix preset:
- Change X's symmetry to C₂ → verify Python code updates, active preset clears
- Add a new variable W (rank 3, dense) → verify card appears
- Type a new expression using W → verify validation catches rank mismatch
- Fix expression → verify Analyze works

- [ ] **Step 4: Test custom generators**

- Create a rank-4 variable R with "custom" symmetry
- Select all 4 axes
- Type generators: `(0 1)(2 3), (0 2)(1 3)`
- Verify: summary badge shows order 4, Python code shows `we.PermutationGroup(we.Permutation(we.Cycle(...)), ...)`
- Write an expression using R and analyze → verify the algorithm uses the custom group

- [ ] **Step 5: Test validation**

- Type a subscript with wrong length → verify red error appears
- Type an output label not in inputs → verify error
- Reference an undefined variable → verify error
- Verify Analyze button is disabled while errors exist

- [ ] **Step 6: Test color propagation**

- With Declared C₃ preset: verify T's color appears on bipartite graph U-nodes, matrix row labels
- With custom expression using 2 different variables: verify each has a distinct color

- [ ] **Step 7: Commit final state**

```bash
git add -A docs/visualization/symmetry-explorer/
git commit -m "feat(explorer): complete symmetry explorer redesign with unified variable interface"
```
