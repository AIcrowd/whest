// website/components/symmetry-aware-einsum-contractions/engine/glossary.js
//
// Centralized glossary for cross-V/W-aware classification prose.
// Consumed by FormulaPopover, GlossaryList, and inline tooltips in the
// explorer narrative and decision-ladder leaves.

export const GLOSSARY = [
  {
    term: 'per-tuple symmetry',
    definition: 'A label permutation π such that for every tuple t of index values, summand(t) = summand(π⁻¹ t). Compression uses this group: reusing one orbit representative is only valid when every tuple in the orbit has the same summand value.',
  },
  {
    term: 'expression-level symmetry',
    definition: 'A label permutation π such that the total sum is unchanged under renaming all labels by π, even if individual terms get reshuffled. Always a superset of per-tuple symmetries; includes dummy-rename permutations of summed labels.',
  },
  {
    term: '$V_{\\text{sub}}$',
    definition: 'The per-tuple group $G_{\\text{pt}}$ projected onto the free (output) labels V. Each $π ∈ G_{\\text{pt}}$ restricted to V-labels; the set of such restrictions is a subgroup of $\\mathrm{Sym}(V)$.',
  },
  {
    term: '$S(W)$',
    definition: 'The full symmetric group on the summed (contracted) labels W. Every permutation of W is always an expression-level symmetry because the sum over W is bound-variable iteration — renaming dummies does not change the total.',
  },
  {
    term: 'Source A',
    definition: "σ-loop generators from declared axis symmetries on individual operands. For T declared symmetric on axes (0,1), Source A emits the label swap linking those axes.",
  },
  {
    term: 'Source B',
    definition: 'σ-loop generators from identical-operand swaps. When two operands share Python identity, swapping their positions induces a label permutation matching axes across the two subscripts.',
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
