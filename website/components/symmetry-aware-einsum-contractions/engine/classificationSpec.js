/**
 * Component Classification Spec
 *
 * Single source of truth for the case-classification decision tree.
 *
 * Both the engine (decomposeAndClassify) and the UI visualization
 * (components/ComponentView DecisionTree) consume this spec. Adding a
 * new case or changing a predicate happens in exactly one place here,
 * so the engine and the rendered tree cannot drift apart.
 *
 * Facts required to classify a component:
 *   order        number of elements in the component's restricted group
 *   vCount       number of V-labels (free/output) in the component
 *   wCount       number of W-labels (summed/contracted) in the component
 *   hasCrossGen  true if any generator maps a V-label to a W-label
 *   isFullSym    true if the group is Sym(Lₐ), i.e. order === |Lₐ|!
 *   labelCount   total labels in the component (|Lₐ|)
 *
 * classifyComponent(facts) -> { caseType, leaf, path: ['q0', 'q1', 'a'] }
 */

export const CLASSIFICATION_LEAVES = {
  trivial: {
    id: 'trivial',
    caseType: 'trivial',
    label: 'Direct count (trivial)',
    shortLabel: '∅',
    methodLabel: 'Direct count',
    humanName: 'No symmetry — each tuple stands alone',
    description: 'No nontrivial symmetry — every assignment is distinct.',
    // When Gₐ = {e}, the quotient Iₐ/Gₐ degenerates to Iₐ itself.
    latex: String.raw`|I_a / G_a| = |I_a| \quad \text{when } G_a = \{e\}`,
    glossary: '$I_a$: the set of all tuples that label this component (size $\\prod_{\\ell} n_\\ell$). $G_a$: the detected symmetry group (here just $\\{e\\}$, the identity). With no symmetry, each tuple is its own orbit, so the count is just $|I_a|$.',
  },
  a: {
    id: 'a',
    caseType: 'A',
    label: 'Case A: V-only',
    shortLabel: 'A',
    methodLabel: 'Burnside mults · Cartesian outputs',
    humanName: 'Products fold; every output bin still written',
    description: 'All labels are free (output). Symmetry reduces unique multiplications, but every output bin must still be written.',
    latex: String.raw`\rho_a = \prod_{\ell \in V_a} n_\ell`,
    glossary: '$\\rho_a$: accumulation cost (distinct output bins for this component). $V_a$: free (output) labels in the component. $n_\\ell$: dimension size of label $\\ell$. Multiplications still fold via Burnside on $G_a$, but since $W_a$ (summed labels) is empty, no output tuples collapse — every output bin is written, so accumulation is just the Cartesian product of free-label sizes.',
  },
  b: {
    id: 'b',
    caseType: 'B',
    label: 'Case B: W-only',
    shortLabel: 'B',
    methodLabel: 'Burnside on Gₐ',
    humanName: 'Orbits collapse multiplications and outputs together',
    description: 'All labels are summed. Orbits collapse both multiplications and accumulations equally.',
    latex: String.raw`\rho_a = |I_a / G_a| \text{ (Burnside)}`,
    glossary: '$\\rho_a$: accumulation cost. $I_a$: tuple space for this component. $G_a$: detected group. Since $V_a$ (free labels) is empty, each orbit writes exactly one output bin — so multiplications and accumulations have the same count, computed via Burnside (average of fixed-point sets over $G_a$).',
  },
  c: {
    id: 'c',
    caseType: 'C',
    label: 'Case C: Correlated',
    shortLabel: 'C',
    methodLabel: 'Burnside mults · Orbit-walk outputs',
    humanName: 'Mixed labels, no crossing — walk orbits and project',
    description: 'V and W labels are both present and generators act on both sides simultaneously, but no generator crosses the V/W boundary. No product formula exists — enumerate orbits exactly.',
    latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    glossary: '$\\rho_a$: accumulation cost. $I_a/G_a$: the set of $G_a$-orbits on $I_a$. $O$: one orbit. $\\pi_{V_a}(O)$: the orbit restricted to the free labels. Multiplications still fold via Burnside on $G_a$. For outputs, each orbit may touch several distinct bins, so $\\rho_a$ sums the distinct $V$-projections across all orbits — explicit enumeration required.',
  },
  d: {
    id: 'd',
    caseType: 'D',
    label: 'Case D: Cross (Young)',
    shortLabel: 'D',
    methodLabel: 'Burnside mults · Hₐ-Burnside outputs',
    humanName: 'Full symmetric group — V-fixing subgroup gives a shortcut',
    description: 'Cross-boundary generators with the full symmetric group. The V-stabilizer Hₐ gives an analytic Burnside count for accumulation cost.',
    latex: String.raw`\rho_a = |I_a / H_a|, \quad H_a = \mathrm{Stab}_{G_a}(V_a)`,
    glossary: '$\\rho_a$: accumulation cost. $I_a$: tuple space. $H_a$: the $V$-stabilizer of $G_a$, i.e. the subgroup whose elements leave every free label fixed pointwise. Multiplications fold via Burnside on $G_a$. Because $G_a$ is the full symmetric group on $L_a$ (Young structure), Burnside on $H_a$ gives the output-bin count analytically — no explicit orbit walk needed.',
  },
  e: {
    id: 'e',
    caseType: 'E',
    label: 'Case E: Cross (general)',
    shortLabel: 'E',
    methodLabel: 'Burnside mults · Orbit-walk outputs',
    humanName: 'Partial group — no shortcut, walk orbits explicitly',
    description: 'Cross-boundary generators but not the full symmetric group. Value coincidences can merge output bins unpredictably — must enumerate orbits.',
    latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    glossary: '$\\rho_a$: accumulation cost. $I_a/G_a$: $G_a$-orbits on $I_a$. $\\pi_{V_a}(O)$: projection of orbit $O$ onto the free labels $V_a$. Multiplications fold via Burnside on $G_a$. Since $G_a$ is not the full symmetric group, there is no Young-style shortcut for outputs — orbits must be enumerated and their $V$-projections counted individually.',
  },
};

/**
 * Decision tree, top-down. Each question has:
 *   id         stable id (used in paths, tests, and the UI tree)
 *   short      compact text shown in the tree node (e.g. "W₀ = ∅ ?")
 *   long       full text shown in the tooltip
 *   latex      optional formula for the tooltip
 *   test       (facts) -> boolean; true follows onTrue, false follows onFalse
 *   onTrue     id of the next step (either another question id or a leaf id)
 *   onFalse    id of the next step
 *
 * Every `onTrue` / `onFalse` must reference either a question id present in
 * this list or a leaf id present in CLASSIFICATION_LEAVES.
 */
export const CLASSIFICATION_QUESTIONS = [
  {
    id: 'q0',
    short: 'Has symmetry?',
    long: 'Check: nontrivial symmetry?',
    description: 'Does this component have any detected symmetry beyond the identity? If not, there is no quotienting to do and the count stays direct.',
    test: (facts) => facts.order > 1,
    onTrue: 'q1',
    onFalse: 'trivial',
  },
  {
    id: 'q1',
    short: 'W₀ = ∅ ?',
    long: 'Check: W-labels present?',
    description: 'Does this component contain any summed (contracted) labels? If not, the symmetry only acts on output indices.',
    test: (facts) => facts.wCount === 0,
    onTrue: 'a',
    onFalse: 'q2',
  },
  {
    id: 'q2',
    short: 'V₀ = ∅ ?',
    long: 'Check: V-labels present?',
    description: 'Does this component contain any free (output) labels? If not, the symmetry only acts on summed indices.',
    test: (facts) => facts.vCount === 0,
    onTrue: 'b',
    onFalse: 'q3',
  },
  {
    id: 'q3',
    short: 'Cross V/W gens?',
    long: 'Check: cross-boundary generators?',
    description: 'Does any generator of Gₐ map a V-label to a W-label or vice versa? If no, V and W actions are correlated but partition-preserving.',
    test: (facts) => facts.hasCrossGen,
    onTrue: 'q4',
    onFalse: 'c',
  },
  {
    id: 'q4',
    short: 'Gₐ = Sym(Lₐ) ?',
    long: 'Check: full symmetric group?',
    description: 'Is Gₐ the full symmetric group on all labels in this component? That is, |Gₐ| = |Lₐ|! (factorial). If yes, the Young-tableau formula gives an analytic shortcut.',
    latex: String.raw`|G_a| = |L_a|! \implies \rho_a = |I_a / H_a|`,
    test: (facts) => facts.isFullSym,
    onTrue: 'd',
    onFalse: 'e',
  },
];

const QUESTION_BY_ID = Object.fromEntries(
  CLASSIFICATION_QUESTIONS.map((q) => [q.id, q]),
);

function isLeafId(id) {
  return Object.prototype.hasOwnProperty.call(CLASSIFICATION_LEAVES, id);
}

/**
 * Walk the decision tree for a component, returning its caseType, leaf id,
 * and the full path of visited nodes (questions + final leaf).
 *
 * Throws if the spec points to an unknown id or if the walk exceeds the
 * number of questions (which would indicate a cycle).
 */
export function classifyComponent(facts) {
  const path = [];
  let currentId = CLASSIFICATION_QUESTIONS[0]?.id;
  let safety = CLASSIFICATION_QUESTIONS.length + 1;

  while (currentId && safety > 0) {
    if (isLeafId(currentId)) {
      path.push(currentId);
      const leaf = CLASSIFICATION_LEAVES[currentId];
      return { caseType: leaf.caseType, leaf: leaf.id, path };
    }

    const question = QUESTION_BY_ID[currentId];
    if (!question) {
      throw new Error(`Classification spec references unknown id: ${currentId}`);
    }
    path.push(question.id);
    currentId = question.test(facts) ? question.onTrue : question.onFalse;
    safety -= 1;
  }

  throw new Error('Classification walk exceeded question count (cycle in spec?)');
}

/**
 * Helper: list every distinct leaf id the spec can land on.
 * Used by alignment tests and by the tree visualizer.
 */
export function enumerateReachableLeaves() {
  const reachable = new Set();
  const stack = [CLASSIFICATION_QUESTIONS[0]?.id];
  const visited = new Set();
  while (stack.length > 0) {
    const id = stack.pop();
    if (!id || visited.has(id)) continue;
    visited.add(id);
    if (isLeafId(id)) {
      reachable.add(id);
      continue;
    }
    const q = QUESTION_BY_ID[id];
    if (!q) continue;
    stack.push(q.onTrue, q.onFalse);
  }
  return [...reachable];
}
