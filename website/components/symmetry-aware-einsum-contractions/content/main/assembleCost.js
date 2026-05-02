const p = (text) => ({ kind: 'paragraph', text });
const l = (text) => ({ kind: 'label', text });

const assembleCost = {
  title: 'Assemble the Direct Cost Model',
  deck: 'What is the final direct-event cost of the symmetry-aware computation?',
  slots: {
    intro: [
      p('All the pieces are now in place. The preceding sections have produced a detected pointwise group and a support-connected component decomposition of its label action. For each independent component, $M_a$ counts representative products and $\\alpha_a$ counts filled local $O \\to Q$ cells; globally, representative products multiply across components, and accumulation reach multiplies across independent incidence relations.'),
      p('The final number is $\\mathrm{Total} = \\mu + \\alpha = (k - 1)\\prod_a M_a + \\prod_a \\alpha_a$. This is a direct indexed scalar-event count: multiplication-chain events plus output updates, not a contraction-path or BLAS runtime model.'),
    ],
    appendixNoteTitle: [
      l('Appendix note'),
      p('Is this the full symmetry of the final expression?'),
    ],
    appendixNoteBody: [
      p('The direct cost above uses two actions derived from the same detected pointwise symmetry: $G_{\\text{pt}}$ on product assignments and $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ on stored output representatives. The completed expression can have a larger formal label-renaming group, $G_{\\text{f}} = H \\times \\prod_d S(W_d)$, where each $W_d$ is a same-domain block of summed labels. That larger group explains expression-level equality after summation. Its dummy-label factor must not replace the product-orbit/output-representative relation used for $\\alpha$.'),
    ],
    produces: [
      p('The final direct-event count and the boundary between this model and the appendix proof layer.'),
    ],
  },
};

export default assembleCost;
