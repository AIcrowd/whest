const p = (text) => ({ kind: 'paragraph', text });

const productSymmetry = {
  title: 'Product Symmetry',
  deck: 'What can be multiplied once?',
  slots: {
    intro: [
      p('The first quotient is product-side reuse: when do two full assignments produce the same pre-summation product? The detected pointwise group $G_{\\mathrm{pt}}$ acts on the full assignment grid $X$, and each orbit $O \\in X/G_{\\mathrm{pt}}$ has one representative product. The representative-product count is $M = |X/G_{\\mathrm{pt}}|$. Burnside is the right mental model for $M$: if every non-identity relabeling moved every assignment, the count would shrink by $|G_{\\mathrm{pt}}|$; tensor-index grids have diagonals and fixed points, so the exact count averages fixed assignments instead.'),
      p('The bipartite graph and incidence matrix remain the structural audit trail. They record which operand-axis slots carry which labels and which same-domain relabelings are eligible to test. Candidate row moves are not proof yet; certification in Section 6 will decide which moves actually generate $G_{\\mathrm{pt}}$.'),
    ],
    produces: [
      p('The product rows $O = X/G_{\\mathrm{pt}}$ and the structural candidate space used to certify them: labels, axis classes, incidence fingerprints, declared slot actions, and same-domain constraints. Section 3 projects those rows onto stored output columns.'),
    ],
  },
};

export default productSymmetry;
