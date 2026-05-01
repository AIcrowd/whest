const p = (text) => ({ kind: 'paragraph', text });

const einsumGlance = {
  title: 'Einsum at a Glance',
  deck: 'What computation are we counting?',
  slots: {
    intro: [
      p('An einsum is a compact notation for an indexed computation. Labels that appear in the output are **visible labels**: they index the result. Labels that appear on inputs but not in the output are **summed labels**: the evaluator loops over them and accumulates. Together they form $L = V \\sqcup W$, where $V = V_{\\mathrm{free}}$ are visible/output labels and $W = W_{\\mathrm{summed}}$ are summed labels.'),
      p('For the default microscope preset, $R[i,k] = \\sum_j A[i,j]\\,B[k]$, with $A$ declared symmetric in $(i,j)$. Here $V = \\{i,k\\}$, $W = \\{j\\}$, and $L = \\{i,j,k\\}$. The full assignment space is $X = \\prod_{\\ell \\in L}[n_\\ell]$, so a dense direct evaluator visits assignments of $(i,j,k)$. Before symmetry is used, every full assignment forms one product and performs one output update.'),
      p('Cross S2 is a microscope, not a benchmark. It is intentionally too small to be a performance story: a different algorithm could algebraically factor the expression. We use it because it isolates the projection phenomenon in one row, making product orbits and accumulation updates easy to see side by side.'),
    ],
    produces: [
      p('A normalized direct-index contraction instance: label set $L = V \\sqcup W$, assignment grid $X$, operand slots, label domains, and declared equality symmetries. Reusing the same operand name means the same tensor object appears more than once.'),
    ],
  },
};

export default einsumGlance;
