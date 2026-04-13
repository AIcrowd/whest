export const EXAMPLES = [
  // ── Classic: S2 on output ──
  {
    id: 'gram',
    name: 'Gram matrix',
    formula: "einsum('ia,ib→ab', X, X)",
    subscripts: ['ia', 'ib'],
    output: 'ab',
    operandNames: ['X', 'X'],
    perOpSymmetry: null,
    description: 'XᵀX is symmetric — detects S2 on output labels {a,b}',
    expectedGroup: 'S2{a,b}',
    color: '#4a7cff',
  },

  // ── S3 build-up: 3 identical operands ──
  {
    id: 'triple-outer',
    name: 'Triple outer (S3)',
    formula: "einsum('ia,ib,ic→abc', X, X, X)",
    subscripts: ['ia', 'ib', 'ic'],
    output: 'abc',
    operandNames: ['X', 'X', 'X'],
    perOpSymmetry: null,
    description: '3 identical operands → full S3 on output. Shows how S3 needs all 3! = 6 permutations',
    expectedGroup: 'S3{a,b,c}',
    color: '#23B761',
  },

  // ── Block symmetry ──
  {
    id: 'outer',
    name: 'Outer product',
    formula: "einsum('ab,cd→abcd', X, X)",
    subscripts: ['ab', 'cd'],
    output: 'abcd',
    operandNames: ['X', 'X'],
    perOpSymmetry: null,
    description: 'Detects block symmetry — swapping (a,b)↔(c,d)',
    expectedGroup: 'S2{a,c}×S2{b,d}',
    color: '#3ddc84',
  },

  // ── C3: cyclic only, no reflections ──
  {
    id: 'triangle',
    name: 'Directed triangle',
    formula: "einsum('ij,jk,ki→ijk', A, A, A)",
    subscripts: ['ij', 'jk', 'ki'],
    output: 'ijk',
    operandNames: ['A', 'A', 'A'],
    perOpSymmetry: null,
    description: 'Cyclic chain — only rotations are valid (not reflections), so C3 not S3',
    expectedGroup: 'C3{i,j,k}',
    color: '#ffb74d',
  },

  // ── D4: per-operand symmetry enables reflections ──
  {
    id: 'four-cycle',
    name: 'Undirected 4-cycle',
    formula: "einsum('ij,jk,kl,li→ijkl', S, S, S, S)",
    subscripts: ['ij', 'jk', 'kl', 'li'],
    output: 'ijkl',
    operandNames: ['S', 'S', 'S', 'S'],
    perOpSymmetry: 'symmetric',
    description: 'S symmetric ⇒ axes collapse, enabling reflections. C4 + reflections = D4',
    expectedGroup: 'D4{i,j,k,l}',
    color: '#bb86fc',
  },

  // ── W-side symmetry: contracted indices ──
  {
    id: 'trace-product',
    name: 'Tr(A·A)',
    formula: "einsum('ij,ji→', A, A)",
    subscripts: ['ij', 'ji'],
    output: '',
    operandNames: ['A', 'A'],
    perOpSymmetry: null,
    description: 'No free labels — symmetry is on W (summed) side. S2{i,j} reduces the contraction cost',
    expectedGroup: 'W: S2{i,j}',
    color: '#94A3B8',
  },

  // ── Declared C3: non-identical operands with contraction ──
  {
    id: 'declared-c3',
    name: 'Declared C₃ (contraction)',
    formula: "einsum('aijk,ab→ijkb', T, W)",
    subscripts: ['aijk', 'ab'],
    output: 'ijkb',
    operandNames: ['T', 'W'],
    perOpSymmetry: [{ type: 'cyclic', axes: [1, 2, 3] }, null],
    description: 'T has C₃ on axes {i,j,k}, contracted with W on index a. Non-identical operands → σ-loop empty, but declared C₃ is preserved (not promoted to S₃)',
    expectedGroup: 'C3{i,j,k}',
    color: '#F59E0B',
  },

  // ── Declared D4: non-identical operands with contraction ──
  {
    id: 'declared-d4',
    name: 'Declared D₄ (contraction)',
    formula: "einsum('aijkl,ab→ijklb', T, W)",
    subscripts: ['aijkl', 'ab'],
    output: 'ijklb',
    operandNames: ['T', 'W'],
    perOpSymmetry: [{ type: 'dihedral', axes: [1, 2, 3, 4] }, null],
    description: 'T has D₄ on axes {i,j,k,l}, contracted with W on index a. Detects D₄ — without the fix, wrongly promoted to S₄',
    expectedGroup: 'D4{i,j,k,l}',
    color: '#EC4899',
  },

  // ── No symmetry despite identical operands ──
  {
    id: 'matrix-chain',
    name: 'A·A (no symmetry)',
    formula: "einsum('ij,jk→ik', A, A)",
    subscripts: ['ij', 'jk'],
    output: 'ik',
    operandNames: ['A', 'A'],
    perOpSymmetry: null,
    description: 'Identical operands but different subscript structure → σ-loop finds no valid π',
    expectedGroup: 'trivial',
    color: '#D1D5DB',
  },

  // ── Mixed operands: no symmetry ──
  {
    id: 'mixed-chain',
    name: 'A·B·A (mixed)',
    formula: "einsum('ij,jk,kl→il', A, B, A)",
    subscripts: ['ij', 'jk', 'kl'],
    output: 'il',
    operandNames: ['A', 'B', 'A'],
    perOpSymmetry: null,
    description: 'A appears twice but B breaks the chain — no identical group forms',
    expectedGroup: 'trivial',
    color: '#E5E7EB',
  },
];
