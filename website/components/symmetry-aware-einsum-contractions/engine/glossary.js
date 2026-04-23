// website/components/symmetry-aware-einsum-contractions/engine/glossary.js
//
// Centralized glossary for cross-V/W-aware classification prose.
// Consumed by FormulaPopover, GlossaryList, and inline tooltips in the
// explorer narrative and decision-ladder leaves.

export const GLOSSARY = [
  {
    term: 'pointwise symmetry',
    definition: 'A label relabeling π that preserves every pre-summation scalar product under the declared operand equality symmetries and repeated-operand identities. This is the group used for direct computation: one representative per orbit is valid only when every assignment in the orbit has the same summand product.',
  },
  {
    term: 'formal symmetry',
    definition: 'A symmetry of the completed expression after summed labels have become bound variables. Formal symmetry may include output relabelings inherited from pointwise symmetry and same-domain dummy renamings of summed labels. It explains expression equality after summation; it is not generally valid for reducing pre-summation products or accumulation updates.',
  },
  {
    term: '$G_{\\text{pt}}\\big|_V$',
    definition: 'The induced permutation group on V by $G_{\\text{pt}}$: for each V/W-preserving element $\\pi \\in G_{\\text{pt}}$, record its action on V-positions and deduplicate. The resulting subgroup of $\\mathrm{Sym}(V)$ acts as an output-tensor symmetry, i.e. $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\sigma \\in G_{\\text{pt}}\\big|_V$.',
  },
  {
    term: '$S(W)$',
    definition: 'The symmetric group on a same-domain block of summed labels. With heterogeneous sizes, the valid dummy-renaming factor is ∏_d S(W_d), where each W_d contains summed labels with the same domain/size. Full S(W) is valid only when all summed labels share a common domain.',
  },
  {
    term: 'representative products M',
    definition: 'M is the number of product orbits under G_pt. In components, M = ∏_a M_a. It counts how many distinct product values the direct symmetry-aware evaluator must form before accounting for the k-operand multiplication chain length.',
  },
  {
    term: 'multiplication cost μ',
    definition: 'μ is the multiplication-chain event count derived from representative products: μ = (k - 1)M for k operand tensors. μ is not the product-orbit count itself.',
  },
  {
    term: 'accumulation cost α',
    definition: 'α is the direct output-bin update count. It is an orbit-projection count: sum over product orbits O of the number of distinct visible/output projections touched by O. It is not output storage and not generally equal to M.',
  },
  {
    term: 'wreath product',
    definition: 'The row-permutation group the σ-loop enumerates: $\\prod_i (H_i \\wr S_{m_i})$, where $i$ ranges over identical-operand groups (operands sharing a name), $H_i$ is each operand\'s declared axis symmetry, and $m_i$ is the number of copies. Elements factor as $(h_0, \\ldots, h_{m-1}; \\sigma)$ with each $h_j \\in H_i$ and $\\sigma \\in S_{m_i}$.',
  },
  {
    term: 'row-witnessed',
    definition: 'A symmetry is row-witnessed if it arises as `derivePi(σ)` for some wreath element $\\sigma \\in \\prod_i (H_i \\wr S_{m_i})$ and survives the identity-filter. $G_{\\text{pt}}|_V$ is row-witnessed.',
  },
  {
    term: 'row-unwitnessed',
    definition: 'A symmetry is row-unwitnessed if no wreath element produces it via `derivePi`. The $S(W)$ factor of $G_{\\text{f}}$ is entirely row-unwitnessed — bound-variable renamings of summed labels that leave the total invariant without appearing at any row level.',
  },
  {
    term: 'base-group generator',
    definition: 'An element of the wreath\'s base factor $H_i^{m_i}$ — an axis permutation acting within a single copy of operand $i$, with the top-group component held as identity. Produced from an operand\'s declared axis symmetry.',
  },
  {
    term: 'top-group transposition',
    definition: 'An adjacent copy-transposition in the wreath\'s top factor $S_{m_i}$: swaps adjacent copies of an identical-operand group with base-group components held as identity. Produced from identical-operand swaps.',
  },
  {
    term: 'component',
    definition: "A support-connected block of labels induced by the detected generators. Each component has labels L_a, output labels V_a, summed labels W_a, and restricted group G_a. The decomposition is safe for the displayed product formula; algebraically independent factors that remain inside a support-connected block are handled by the regime ladder.",
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
    term: 'Factorization check',
    definition: 'The direct-product recognizer checks that no group element crosses V/W and that |G| = |G_V| · |G_W|. Passing means the action factors over visible and summed labels, so the direct-product α formula is exact.',
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
