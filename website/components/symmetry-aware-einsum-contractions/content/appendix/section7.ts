import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

// V3.1 Appendix B — Classification-tree cases.
//
// Each B.x sub-case corresponds to a leaf in the main page's DecisionLadder.
// The shared object every case counts is the local accumulation count
//
//   alpha_a = #{ (O, Q) in X_a/G_a x Y_a/H_a : pi_{V_a}(O) ∩ Q ≠ ∅ }.
//
// Slots are flat per V3.1 narrative authoring: each `case<k>` slot is a
// stand-alone {Condition, Claim, Intuition} block. The `intro` slot
// orients the reader; `closingNote` reinforces that every B.x reduces to
// the same alpha_a object, which is the unifying contract of the section.
const section7 = {
  title: 'Classification-tree cases',
  deck: 'For each independent component $a$, the local accumulation count $\\alpha_a$ admits one of nine closed forms or fallbacks. Each case below counts the same object — pairs $(O, Q)$ where the projection of a product orbit hits a stored output representative — using the structure that is available.',
  slots: {
    intro: [
      p('For an independent component $a$, the local accumulation count is'),
      p('$\\alpha_a = \\#\\{(O,Q)\\in X_a/G_a\\times Y_a/H_a:\\pi_{V_a}(O)\\cap Q\\neq\\varnothing\\}.$'),
      p('Each case below counts this same object. The DecisionLadder on the main page picks the first case whose condition the component satisfies; the more specialised cases sit higher in the tree because they admit cheaper closed forms, while the general fallbacks (B.7 typed partition counting and B.8 corrected brute force) handle every other configuration. B.9 is the explicit unavailable state — the engine returns no number rather than a wrong one when both fallbacks exceed budget.'),
    ],
    case1Label: [
      l('B.1 — No detected product symmetry'),
    ],
    case1: [
      p('Condition. $G_a = \\{e\\}$.'),
      p('Claim. $\\alpha_a = |X_a| = \\prod_{\\ell\\in L_a} n_\\ell$, and $M_a = |X_a|$.'),
      p('Intuition. With trivial group, every full assignment is its own product orbit. Its projection lands in exactly one stored output representative, so the count collapses to the raw assignment-grid size.'),
    ],
    case2Label: [
      l('B.2 — All labels visible'),
    ],
    case2: [
      p('Condition. $W_a = \\varnothing$.'),
      p('Claim. $\\alpha_a = M_a = |X_a / G_a|$.'),
      p('Intuition. With no summed labels, $V_a = L_a$ and projection is the identity. Every group element preserves $V_a$, so the output action is the restriction of $G_a$ to all component labels, and product orbits coincide with stored output representatives.'),
    ],
    case3Label: [
      l('B.3 — All labels summed'),
    ],
    case3: [
      p('Condition. $V_a = \\varnothing$.'),
      p('Claim. $\\alpha_a = M_a = |X_a / G_a|$.'),
      p('Intuition. With no visible labels, the output assignment space is a singleton. Every product orbit updates that scalar representative exactly once, so $\\alpha_a$ collapses to the number of product orbits.'),
    ],
    case4Label: [
      l('B.4 — Functional projection under setwise preservation of $V_a$'),
    ],
    case4: [
      p('Condition. $g(V_a) = V_a$ for every $g \\in G_a$.'),
      p('Claim. $\\alpha_a = M_a$ — each product orbit reaches exactly one stored output representative.'),
      p('Intuition. Setwise preservation of $V_a$ means projection descends to a well-defined map $X_a/G_a \\to Y_a/H_a$. This is the corrected general condition: the group does not need to look like a visible-side product times a summed-side product; setwise preservation alone is enough.'),
    ],
    case5Label: [
      l('B.5 — Single visible label'),
    ],
    case5: [
      p('Condition. $|V_a| = 1$. Let $V_a = \\{v\\}$, $\\Omega = G_a \\cdot v$ the orbit of $v$ under $G_a$ (with common domain size $n_\\Omega$), and $R_a = L_a \\setminus \\Omega$.'),
      p('Claim. $\\alpha_a = \\dfrac{n_\\Omega}{|G_a|} \\sum_{g\\in G_a} \\left(\\prod_{c\\in\\mathrm{cycles}(g|_{R_a})} n_c\\right)\\!\\left(n_\\Omega^{c_\\Omega(g)} - (n_\\Omega-1)^{c_\\Omega(g)}\\right),$ where $c_\\Omega(g)$ is the number of cycles of $g$ contained in $\\Omega$.'),
      p('Intuition. With one visible label, the output representative action is trivial on that component. The formula counts, by weighted Burnside and inclusion-exclusion, those product orbits whose visible-label orbit reaches a distinguished output coordinate.'),
    ],
    case6Label: [
      l('B.6 — Full symmetric multiset shortcut'),
    ],
    case6: [
      p('Condition. $G_a = S(L_a)$ (the full symmetric group on component labels), all labels in $L_a$ share one domain size $n$, both $V_a$ and $W_a$ are nonempty, and $|V_a| \\ge 2$.'),
      p('Claim. $\\alpha_a = \\dbinom{n + |V_a| - 1}{|V_a|}\\,\\dbinom{n + |W_a| - 1}{|W_a|}.$'),
      p('Intuition. Full symmetry removes order. Visible assignments modulo the output action are multisets of size $|V_a|$, summed assignments contribute multisets of size $|W_a|$, and the count factors as the product of the two multiset counts.'),
    ],
    case7Label: [
      l('B.7 — Typed partition count'),
    ],
    case7: [
      p('Condition. Simpler closed forms do not apply, and typed equality patterns can be enumerated within the exact-counting budget.'),
      p('Claim. $\\alpha_a = \\sum_{\\tilde p \\in P_{\\mathrm{typed}}(L_a)/G_a} \\dfrac{\\prod_s (n_s)_{b_s(\\tilde p)}}{|\\overline{G}_{\\tilde p}|}\\,|A_{\\tilde p}/H_a|.$'),
      p('Intuition. Group full assignments by typed equality pattern. The falling factorial counts injective concrete labelings; dividing by the induced block-action image $|\\overline{G}_{\\tilde p}|$ removes block-permutation overcounting; $|A_{\\tilde p}/H_a|$ counts the stored output representatives reached by each pattern family. Definitions and proof appear in Appendix C.'),
    ],
    case8Label: [
      l('B.8 — Corrected brute-force orbit count'),
    ],
    case8: [
      p('Condition. Closed forms do not apply, typed partition counting is unavailable or over budget, and tuple-orbit enumeration is still feasible.'),
      p('Claim. $\\alpha_a = \\sum_{O \\in X_a/G_a} |\\pi_{V_a}(O) / H_a|.$'),
      p('Intuition. Evaluate the definition of $\\alpha_a$ directly. Projected outputs must be canonicalised under $H_a$; otherwise the count is wrong whenever output representatives are nontrivial.'),
    ],
    case9Label: [
      l('B.9 — Unavailable case'),
    ],
    case9: [
      p('Condition. No closed form applies, typed partition counting is unavailable or over budget, and corrected brute force exceeds the interactive budget.'),
      p('Claim. The page reports the local $\\alpha_a$ as unavailable.'),
      p('Intuition. This is part of the exactness contract: the explorer returns exact counts or explicitly withholds them. An unavailable cell is a feature, not a failure — it preserves the guarantee that any reported number is correct.'),
    ],
    closingNote: [
      p('All nine cases compute the same $\\alpha_a$. The classification tree is a routing strategy, not a definition: it picks the cheapest case whose preconditions hold, falling back to the general counting machinery when structure is absent. When the engine has to fall through to B.9, it withholds the number rather than guess, and the main page surfaces an "unavailable" cell with a link back to this section.'),
    ],
  },
} satisfies SectionCopy;

export default section7;
