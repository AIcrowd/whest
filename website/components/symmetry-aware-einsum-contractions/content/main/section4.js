const p = (text) => ({ kind: 'paragraph', text });

const section4 = {
  title: 'Count Product Orbits and Output Projections',
  deck: 'How does the pointwise group become multiplication and accumulation cost?',
  slots: {
    intro: [
      p('The group $G_{\\text{pt}}$ acts on the full assignment grid $X$, not only on output labels. The number of representative products is the orbit count $M = |X/G_{\\text{pt}}|$. In components this becomes $M = \\prod_a M_a$, with each $M_a$ computed by size-aware Burnside or exact orbit enumeration.'),
      p('Accumulation is the part that is easy to over-compress. The output is stored by the output representative action $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ inherited from the detected pointwise group, so the direct update count is $\\alpha = \\#\\{(O, Q) \\in X/G_{\\text{pt}} \\times Y/H : \\pi_{V_{\\mathrm{free}}}(O) \\cap Q \\neq \\varnothing\\}$ — pairs of a product orbit and a stored output representative reached by projecting that orbit.'),
      p('A product-orbit representative can contain many full index assignments; their projections to the visible labels may land in one stored output representative or several. $\\alpha$ counts one update per stored output representative reached. The subtlety is that projection is not always a function from product orbits to output representatives: sometimes one product orbit reaches several. That is exactly why accumulation counting is harder than multiplication counting.'),
      p('A product orbit may contain many full assignments. When those assignments are projected to the output labels, they may reach one stored output representative or several. Enumerating every concrete assignment is correct but can be wasteful.'),
      p('Counting product orbits alone is therefore not enough: a single product orbit can update multiple stored output representatives, so the accumulation count needs an extra reach factor on top of the orbit count.'),
    ],
    produces: [
      p('Per-component product counts $M_a$ and update counts $\\alpha_a$, then global $M=\\prod_a M_a$ and $\\alpha=\\prod_a\\alpha_a$ when the support-connected components factor independently.'),
    ],
  },
};

export default section4;
