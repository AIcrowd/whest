const p = (text) => ({ kind: 'paragraph', text });

const countingShortcuts = {
  title: 'Counting Shortcuts',
  deck: 'Why does the classification tree exist?',
  slots: {
    intro: [
      p('The definition of $\\alpha$ is simple: $\\alpha = \\#\\{(O,Q) : \\pi_V(O) \\cap Q \\neq \\varnothing\\}$. The literal algorithm is also simple — enumerate product orbits, project members, canonicalize under $H$, count filled cells. This is exact, but it may require materializing product orbits and projecting orbit members for every contraction.'),
      p('The classification tree asks a single question: what is the cheapest exact way to count the filled cells of the local $O \\to Q$ matrix? It runs component by component. Each leaf returns $\\alpha_a$, the same local incidence count, using the strongest valid shortcut. The cases fall into three regimes: the matrix collapses to a simple count; the matrix branches but has a special closed form; or the matrix branches generically and requires typed partition counting.'),
    ],
    produces: [
      p('A per-component regime assignment and the corresponding $\\alpha_a$ count: trivial, all-visible, all-summed, functional projection, single visible label, full symmetric multiset, typed partition count, corrected brute force, or unavailable.'),
    ],
  },
};

export default countingShortcuts;
