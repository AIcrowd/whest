const p = (text) => ({ kind: 'paragraph', text });

const section4 = {
  title: 'Count Product Orbits and Output Projections',
  deck: 'How does the pointwise group become multiplication and accumulation cost?',
  slots: {
    intro: [
      p('The group $G_{\\text{pt}}$ acts on the full assignment grid $X$, not only on output labels. The number of representative products is the orbit count $M = |X/G_{\\text{pt}}|$. In components this becomes $M = \\prod_a M_a$, with each $M_a$ computed by size-aware Burnside or exact orbit enumeration.'),
      p('Accumulation is the part that is easy to over-compress. The output is stored by the output representative action $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ inherited from the detected pointwise group, so the direct update count is $\\alpha = \\#\\{(O, Q) \\in X/G_{\\text{pt}} \\times Y/H : \\pi_{V_{\\mathrm{free}}}(O) \\cap Q \\neq \\varnothing\\}$ — pairs of a product orbit and a stored output representative reached by projecting that orbit.'),
    ],
    produces: [
      p('Per-component product counts $M_a$ and update counts $\\alpha_a$, then global $M=\\prod_a M_a$ and $\\alpha=\\prod_a\\alpha_a$ when the support-connected components factor independently.'),
    ],
  },
};

export default section4;
