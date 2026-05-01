const p = (text) => ({ kind: 'paragraph', text });

const appendixTransition = {
  title: 'Appendix Transition',
  deck: 'Where the interactive argument ends and the formal proofs begin.',
  slots: {
    intro: [
      p('The main article has built the interactive argument: products are orbits, updates are incidences. The appendix makes the construction audit-ready. It gives the formal certification method for $G_{\\text{pt}}$, theorem-style statements for the classification cases, the typed partition-counting theorem, the boundary between pointwise and formal symmetry, and the complete scope statement.'),
    ],
    produces: [
      p('Five appendix sections: product-side certification; classification cases; typed partition counting; formal symmetry after summation; scope and non-goals.'),
    ],
  },
};

export default appendixTransition;
