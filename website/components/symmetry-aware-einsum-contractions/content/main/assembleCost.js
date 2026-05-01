const p = (text) => ({ kind: 'paragraph', text });
const l = (text) => ({ kind: 'label', text });

const assembleCost = {
  title: 'Assemble the Direct Cost Model',
  deck: 'What is the final direct-event cost of the symmetry-aware computation?',
  slots: {
    appendixNoteTitle: [
      l('Appendix note'),
      p('Is this the full symmetry of the final expression?'),
    ],
    appendixNoteBody: [
      p('The direct cost above uses two actions derived from the same detected pointwise symmetry: $G_{\\text{pt}}$ on product assignments and $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ on stored output representatives. The completed expression can have a larger formal label-renaming group, $G_{\\text{f}} = H \\times \\prod_d S(W_d)$, where each $W_d$ is a same-domain block of summed labels. That larger group explains expression-level equality after summation. Its dummy-label factor must not replace the product-orbit/output-representative relation used for $\\alpha$.'),
    ],
  },
};

export default assembleCost;
