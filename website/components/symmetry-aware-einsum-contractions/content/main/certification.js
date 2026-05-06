const p = (text) => ({ kind: 'paragraph', text });

const certification = {
  title: 'Certify the Pointwise Symmetry Group',
  deck: 'Which candidate relabelings preserve the summand itself?',
  slots: {
    intro: [
      p('Candidate row moves come from the wreath product of declared axis symmetries and permutations of identical operand occurrences. If a repeated-operand family has $m_i$ copies and declared internal group $H_i$, its row candidates contribute $H_i \\wr S_{m_i}$; the product over families gives the row-move space $G_{\\mathrm{wreath}}$ enumerated by the $\\sigma$-loop.'),
      p('For each row move $\\sigma$, the explorer asks whether some label relabeling $\\pi$ restores the incidence fingerprints and preserves label domains. An accepted witness $(\\sigma,\\pi)$ proves that every full assignment and its relabeled assignment produce the same scalar product under the declared equality symmetries and commutative scalar multiplication.'),
      p('The accepted relabelings are closed under composition to obtain the detected pointwise group $G_{\\text{pt}}$. This is the product-side group: its orbits in the full assignment grid give the representative products. The output-side representatives are derived from the same group by keeping the elements that preserve the visible labels and restricting them to those labels, $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V_{\\mathrm{free}})|_{V_{\\mathrm{free}}}$. It is a detected structural subgroup, not a claim that all possible numerical or algebraic symmetries have been inferred.'),
    ],
    produces: [
      p('A detected product-side group $G_{\\text{pt}}$, represented by accepted witnesses and the generated label action on the full assignment grid; and, by restriction to visible-label stabilizers, the output representative action $H$ used later for accumulation updates.'),
    ],
  },
};

export default certification;
