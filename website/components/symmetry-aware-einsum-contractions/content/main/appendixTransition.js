const p = (text) => ({ kind: 'paragraph', text });

const appendixTransition = {
  title: 'Appendix Transition',
  deck: 'Where the interactive argument ends and the formal proofs begin.',
  slots: {
    intro: [
      p('The main article has built the interactive argument: products are orbits, updates are incidences. The appendix makes the construction audit-ready. It gives the formal certification method for $G_{\\text{pt}}$, theorem-style statements for the classification cases, the typed partition-counting theorem, the boundary between pointwise and formal symmetry, and the complete scope statement.'),
    ],
    produces: [
      p('Appendix A: product-side certification (#appendix-section-1). Appendix B: classification-tree cases (#appendix-section-7). Appendix C: typed partition theorem (#appendix-section-6). Appendix D: completed-expression formal symmetry (#appendix-section-4). Appendix E: scope, assumptions, and exactness (#appendix-section-8).'),
    ],
  },
};

export default appendixTransition;
