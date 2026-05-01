const p = (text) => ({ kind: 'paragraph', text });

const rowsCols = {
  title: 'Rows and Columns',
  deck: 'How does $G_{\\text{pt}}$ become rows, and $H$ become columns?',
  slots: {
    intro: [
      p('The detected pointwise group $G_{\\text{pt}}$ acts on the full assignment grid $X$, not only on output labels. The number of representative products is the orbit count $M = |X/G_{\\text{pt}}|$. In components this becomes $M = \\prod_a M_a$, with each $M_a$ computed by size-aware Burnside or exact orbit enumeration.'),
      p('Accumulation is the part that is easy to over-compress. The output is stored by the output representative action $H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V$ inherited from the detected pointwise group, so the direct update count is $\\alpha = \\#\\{(O, Q) \\in X/G_{\\text{pt}} \\times Y/H : \\pi_{V_{\\mathrm{free}}}(O) \\cap Q \\neq \\varnothing\\}$ — pairs of a product orbit and a stored output representative reached by projecting that orbit.'),
      p('A product orbit can contain many full assignments. After projection, those assignments may reach one stored output representative or several, so $\\alpha$ counts one update per reached column, not one update per product row. Enumerating every concrete assignment is correct but wasteful; the remaining sections explain when this $O \\to Q$ reach relation factors, when it has shortcuts, and when typed partitions count it exactly.'),
    ],
    produces: [
      p('The row quotient $X/G_{\\text{pt}}$, the column quotient $Y/H$, and the reason accumulation needs a reach relation between them. Section 5 shows when that relation factors into independent components.'),
    ],
  },
};

export default rowsCols;
