// website/components/symmetry-aware-einsum-contractions/engine/glossary.js
//
// Centralized glossary for cross-V/W-aware classification prose.
// Consumed by FormulaPopover, GlossaryList, and inline tooltips in the
// explorer narrative and decision-ladder leaves.

export const GLOSSARY = [
  {
    term: 'pointwise symmetry',
    definition: 'A label permutation π such that for every tuple t of index values, summand(t) = summand(π⁻¹ t). The invariance is required at every individual summand (pointwise on the tuple space), not merely on the total. Compression uses this group: reusing one orbit representative is only valid when every tuple in the orbit has the same summand value.',
  },
  {
    term: 'formal symmetry',
    definition: 'A V-preserving label permutation π = (π_V, π_W) ∈ Sym(V) × Sym(W) under which the output tensor R, viewed as a polynomial in the entries of the operand tensors, is invariant: R[π_V·ω] = R[ω] as polynomials, after relabelling the summed indices by π_W. "Formal" here has its standard mathematical meaning — invariance at the level of the expression as a formal polynomial, not at the level of its values on any specific operand. The V-preserving restriction of any pointwise symmetry projects to a formal symmetry; the converse does not hold in general (dummy W-renamings in {e} × S(W) are formal but not pointwise).',
  },
  {
    term: '$G_{\\text{pt}}\\big|_V$',
    definition: 'The induced permutation group on V by $G_{\\text{pt}}$: for each V/W-preserving element $\\pi \\in G_{\\text{pt}}$, record its action on V-positions and deduplicate. The resulting subgroup of $\\mathrm{Sym}(V)$ acts as an output-tensor symmetry, i.e. $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\sigma \\in G_{\\text{pt}}\\big|_V$.',
  },
  {
    term: '$S(W)$',
    definition: 'The full symmetric group on the summed (contracted) labels W. Every permutation of W is a formal symmetry because the sum over W is bound-variable iteration — renaming dummies does not change the total.',
  },
  {
    term: 'Source A',
    definition: "σ-loop generators from declared axis symmetries on individual operands. For T declared symmetric on axes (0,1), Source A emits the label swap linking those axes.",
  },
  {
    term: 'Source B',
    definition: 'σ-loop generators from identical-operand swaps. The engine groups operands by name (the JS engine in `algorithm.js` uses `nameToPositions[name]` to bucket positions sharing an operand identifier), and when two positions share that name, swapping them induces a label permutation matching axes across the two subscripts.',
  },
  {
    term: 'component',
    definition: "A connected piece of the einsum's label-interaction graph (labels linked by the σ-loop's generators). Each component has its own labels L_c, V_c, W_c, and restricted symmetry group G_c. The classification tree runs per-component; components combine to give the einsum's total cost.",
  },
  {
    term: 'cross-V/W element',
    definition: 'A group element that maps at least one V-label to a W-label (or vice versa). Such elements exist when declared axis symmetries span both V and W axes on an operand, or when identical-operand swaps pair V with W across subscripts.',
  },
  {
    term: 'Young subgroup',
    definition: 'A subgroup of a symmetric group that is a product of smaller symmetric groups on a partition. For L = V ⊔ W, both Sym(V) × Sym(W) and {e} × Sym(W) are Young subgroups of Sym(L). When G = Sym(L), the pointwise V-stabilizer is the Young subgroup Sym(W).',
  },
  {
    term: 'pointwise V-stabilizer',
    definition: 'The subgroup of G whose elements fix every V-label individually. When G = Sym(L), this is Sym(W). The Young regime uses α = |X / pointwise-V-stabilizer|, a multinomial closed form.',
  },
  {
    term: 'F-check',
    definition: 'The direct-product test on a materialized group G: verify that no element crosses V/W, then check |G| = |G_V| · |G_W|. Passing the F-check (and the meaningfulness guard on both projection sizes > 1) means G factors as G_V × G_W acting factor-wise on V and W.',
  },
  {
    term: 'meaningfulness guard',
    definition: 'An additional predicate in the directProduct recognizer requiring both projection sizes |G_V| > 1 AND |G_W| > 1. Prevents directProduct from firing on trivial-projection cases where simpler regimes (allVisible, allSummed) carry the same content.',
  },
];

/**
 * Look up a glossary entry by term. Returns undefined if not found.
 */
export function lookup(term) {
  return GLOSSARY.find((g) => g.term === term);
}
