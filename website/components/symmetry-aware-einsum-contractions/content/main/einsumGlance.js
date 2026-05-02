const p = (text) => ({ kind: 'paragraph', text });

const einsumGlance = {
  title: 'Specify the Contraction',
  deck: 'What computation are we counting?',
  slots: {
    intro: [
      p('The first step is to fix the mathematical object, not to detect symmetry. We specify the ordered operand occurrences, one subscript for each occurrence, the output labels, the label sizes, and the equality symmetries declared on individual inputs.'),
      p('This produces a normalized contraction instance: a label set $L = V_{\\mathrm{free}} \\sqcup W_{\\mathrm{summed}}$, where $V_{\\mathrm{free}}$ are visible/output labels and $W_{\\mathrm{summed}}$ are summed labels; an assignment grid $X = \\prod_{\\ell\\in L}[n_\\ell]$; and declared slot actions for each operand. Reusing the same operand name means the same tensor object appears more than once. Distinct names are treated as distinct objects even if their shapes or numerical values happen to match.'),
      p('At this stage $V_{\\mathrm{free}}$ only says which labels survive syntactically; the stored output representatives are derived later from the detected pointwise symmetry.'),
    ],
    produces: [
      p('A normalized direct-index contraction instance: label set $L = V \\sqcup W$, assignment grid $X$, operand slots, label domains, and declared equality symmetries. Section 2 uses the same instance to find which products can be represented once.'),
    ],
  },
};

export default einsumGlance;
