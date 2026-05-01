const p = (text) => ({ kind: 'paragraph', text });

const projection = {
  title: 'Projection',
  deck: 'The $O \\to Q$ incidence relation.',
  slots: {
    intro: [
      p('Products are rows. Updates are filled cells. Projection drops the summed labels and keeps the visible labels: $\\pi_V: X \\to Y$, where $Y = \\prod_{v \\in V}[n_v]$. For the default microscope, $(i,j,k) \\mapsto (i,k)$.'),
      p('Take a whole product orbit $O$, project all of its members, and canonicalize the projected output assignments under the stored-output action $H$. The stored output representatives are the columns $Q \\in Y/H$. A cell $(O,Q)$ is filled when $\\pi_V(O) \\cap Q \\neq \\varnothing$. The accumulation count is $\\alpha = \\#\\{(O,Q) \\in X/G_{\\mathrm{pt}} \\times Y/H : \\pi_V(O) \\cap Q \\neq \\varnothing\\}$.'),
      p('For Cross S2, the product orbit $O = \\{(0,1,0),(1,0,0)\\}$ has one representative product $A[0,1]B[0] = A[1,0]B[0]$. But its members project to two different output entries: $(0,1,0) \\mapsto R[0,0]$ and $(1,0,0) \\mapsto R[1,0]$. So this row has two filled cells: one representative product, two stored-output updates. This is the central phenomenon: multiply once; accumulate wherever the orbit projects.'),
    ],
    produces: [
      p('The $O \\to Q$ incidence matrix: rows are product orbits, columns are stored output representatives, filled cells are accumulation updates. The count $\\alpha$ is the number of filled cells.'),
    ],
  },
};

export default projection;
