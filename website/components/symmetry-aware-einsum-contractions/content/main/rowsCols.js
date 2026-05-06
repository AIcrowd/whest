const p = (text, column = null, options = {}) => ({ kind: 'paragraph', text, column, ...options });
const h = (text, column = null) => ({ kind: 'heading', text, column });
const eq = (math, column = null, compact = false, label = null) => ({
  kind: 'equation',
  math,
  column,
  compact,
  label,
});

const rowsCols = {
  title: 'Rows and Columns',
  deck: 'How does $G_{\\text{pt}}$ become rows, and $H$ become columns?',
  slots: {
    intro: [
      p('Read one dense assignment first. A tuple such as $(i=0,j=1,k=0)$ is one choice of values for every label, not a whole row. The $O \\to Q$ matrix starts by grouping those single assignments into product rows, then asks which stored output columns each row reaches.', 'full', { lead: true, align: 'left' }),
      h('Rows: product orbits', 1),
      p('Start with the full assignment space $X$. The schematic shows only examples from $X$; at $n=5$, Cross $S_2$ has $5^3=125$ such assignments. The detected pointwise group $G_{\\text{pt}}$ relabels assignments that produce the same representative product, so one matrix row is one orbit $O$.', 1, { align: 'left' }),
      eq('\\displaystyle O \\in X/G_{\\text{pt}}', 1, false, 'Row quotient'),
      p('Counting rows gives $M$: the number of representative products the evaluator multiplies once.', 1, { align: 'left' }),
      h('Columns: stored output representatives', 2),
      p('The output side keeps only the visible labels. Not every product-side symmetry survives there: $H$ is the certified part of $G_{\\text{pt}}$ that preserves the visible set and restricts to those labels.', 2, { align: 'left' }),
      eq('\\displaystyle H = \\mathrm{Stab}_{G_{\\text{pt}}}(V)|_V,\\qquad Q \\in Y/H', 2, false, 'Column quotient'),
      p('A column is one stored representative $Q$. If $H$ is trivial, every output assignment is its own column; if $H$ is nontrivial, several assignments share one stored representative.', 2, { align: 'left' }),
      h('Projection: filled cells', 'full'),
      p('Projection connects the two quotients. A product orbit can reach one stored output representative, or several, after its members are projected to visible labels.', 'full', { align: 'left' }),
      eq('\\displaystyle \\alpha = \\#\\{(O,Q) \\in X/G_{\\text{pt}} \\times Y/H : \\pi_V(O) \\cap Q \\neq \\varnothing\\}', 'full', true, 'Accumulation count'),
      p('Thus $M$ counts product-orbit rows, while $\\alpha$ counts filled $O \\to Q$ cells. When every row reaches exactly one column, $\\alpha = M$; when projection branches, $\\alpha$ needs its own counting method.', 'full', { align: 'left' }),
    ],
    produces: [
      p('Rows are product orbits $X/G_{\\text{pt}}$. Columns are stored output representatives $Y/H$. The accumulation count $\\alpha$ is the reach relation between them: one product row may fill one column or several. Section 5 shows when this relation factors into independent components.'),
    ],
  },
};

export default rowsCols;
