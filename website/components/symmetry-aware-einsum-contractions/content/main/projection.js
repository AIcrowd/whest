const p = (text) => ({ kind: 'paragraph', text });

const projection = {
  title: 'Projection',
  deck: 'The $O \\to Q$ incidence relation.',
  slots: {
    intro: [
      p('Products are rows. Updates are filled cells. At this point the product-side symmetry has already grouped full assignments into product orbits. Each product orbit $O \\in X/G_{\\mathrm{pt}}$ represents a set of assignments that all produce the same scalar product before accumulation. That is why one orbit becomes one row of the $O \\to Q$ matrix: the row asks where this one representative product must be added.'),
      p('Projection answers that question by dropping the summed labels and keeping only the visible/output labels: $\\pi_V: X \\to Y$, where $Y = \\prod_{v \\in V}[n_v]$. The stored output representatives are the columns $Q \\in Y/H$. A cell $(O,Q)$ is filled when at least one member of the product orbit lands in that stored output representative, meaning $\\pi_V(O) \\cap Q \\neq \\varnothing$. The accumulation count $\\alpha$ is the number of filled cells:'),
      p('For example, take the Cross S2 case, with $A$ declared symmetric in $(i,j)$. The labels $i$ and $k$ are visible because they appear in the output; $j$ is summed because it appears only inside the summation. So projection keeps $(i,k)$ and drops $j$.'),
      p('One representative product, but two stored-output updates. Product reuse reduced multiplication work, but accumulation still has to follow every output destination the orbit reaches.'),
    ],
    produces: [
      p('The $O \\to Q$ incidence matrix: rows are product orbits, columns are stored output representatives, filled cells are accumulation updates. Section 4 separates the row quotient from the column quotient so update count is not mistaken for product count.'),
    ],
  },
};

export default projection;
