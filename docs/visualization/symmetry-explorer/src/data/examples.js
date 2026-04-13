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

  // ── Declared cyclic C3: Reynolds operator ──
  {
    id: 'cyclic-tensor',
    name: 'Cyclic 3-tensor (C₃)',
    formula: "T[i,j,k] = T[j,k,i]",
    subscripts: ['ijk'],
    output: 'ijk',
    operandNames: ['T'],
    perOpSymmetry: 'cyclic',
    description: 'Declared C₃ symmetry via PermutationGroup — T is invariant under cyclic rotation but NOT transposition',
    expectedGroup: 'C3{i,j,k}',
    color: '#F59E0B',
    declared: true,
  },

  // ── Declared block swap ──
  {
    id: 'block-swap',
    name: 'Block swap (ij)↔(kl)',
    formula: "T[i,j,k,l] = T[k,l,i,j]",
    subscripts: ['ijkl'],
    output: 'ijkl',
    operandNames: ['T'],
    perOpSymmetry: 'block-swap',
    description: 'Block swap via Cycle(0,2)(1,3) — axes swap in pairs, like 2-electron integrals (ij|kl)=(kl|ij)',
    expectedGroup: 'S2{blocks}',
    color: '#EC4899',
    declared: true,
  },

  // ── Declared dihedral D4 ──
  {
    id: 'dihedral-grid',
    name: 'Dihedral D₄ grid',
    formula: "T[i,j,k,l] — D₄ symmetry",
    subscripts: ['ijkl'],
    output: 'ijkl',
    operandNames: ['T'],
    perOpSymmetry: 'dihedral',
    description: 'Dihedral D₄ on a 4-tensor — rotations + reflections of a square. 8 of 24 permutations valid',
    expectedGroup: 'D4{i,j,k,l}',
    color: '#8B5CF6',
    declared: true,
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
    expectedGroup: 'trivial',
    color: '#94A3B8',
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
