const p = (text) => ({ kind: 'paragraph', text });

const preamble = {
  title: 'Counting symmetry-aware einsums',
  deck: 'Multiply once; accumulate wherever the orbit projects. This interactive paper counts exact direct events as representative-product multiplication chains plus $O \\to Q$ accumulation updates.',
  slots: {
    einsumIntroBeforeSummed: [
      p('A direct einsum visits assignments of index labels. Labels that appear on inputs but not on the output are '),
    ],
    einsumIntroBetweenSummedAndFree: [
      p(': the evaluator loops over them and accumulates. Labels on the output are '),
    ],
    einsumIntroAfterFree: [
      p(': they index the result. Together they form $L = V \\sqcup W$. A dense direct implementation visits every assignment in the full label grid $X = \\prod_{\\ell \\in L}[n_\\ell]$ before any symmetry is used.'),
    ],
    mentalFrameworkIntroBeforeRepSet: [
      p('A symmetry-aware evaluator keeps the same skeleton. Symmetry changes three data structures: '),
    ],
    mentalFrameworkIntroBetweenRepSetAndOuts: [
      p(' (one representative per product-equivalence class), '),
    ],
    mentalFrameworkIntroBetweenOutsAndCoeff: [
      p(' (stored output representatives reached by that class), and '),
    ],
    mentalFrameworkIntroAfterCoeff: [
      p(' (the number of full assignments contributing to each stored representative). Products are rows; updates are filled cells.'),
    ],
    calloutBodyBeforeGroup: [
      p('Declared operand symmetries and repeated operand identities certify relabelings that preserve each pre-summation product. Those certified relabelings form the pointwise product-symmetry group '),
    ],
    calloutBodyBetweenGroupAndOrbits: [
      p(', which acts on the full assignment space $X$. Its '),
    ],
    calloutBodyAfterOrbits: [
      p(' of full label assignments give the representative products. If the action were free, the representative-product count would shrink by $|G_{\\text{pt}}|$; in ordinary tensor-index actions, diagonal assignments create fixed points, so Burnside\'s lemma gives the exact orbit count.'),
    ],
    calloutFooter: [
      p('The explorer detects this structural pointwise group $G_{\\text{pt}}$ and derives the output representative action $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ from it. It reports $M$ (product-orbit representatives), $\\mu = (k-1)M$ (multiplication-chain events), and $\\alpha$ (accumulation updates from product-orbit representatives into stored output representatives).'),
    ],
    handoffBeforeSectionLink: [
      p('Every figure below recomputes the same chain: $G_{\\text{pt}}$ partitions full assignments into product rows $O$, $H$ partitions stored outputs into columns $Q$, and $\\mathrm{Total}=\\mu+\\alpha$ adds multiplication-chain events to filled $O \\to Q$ cells. Start with '),
    ],
    handoffAfterSectionLink: [
      p(' below to pick or build a contraction.'),
    ],
  },
};

export default preamble;
