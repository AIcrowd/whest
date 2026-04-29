// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

import { notationLatex } from '../lib/notationSystem.js';

// Shape descriptions are semantic rather than formula-owning under the
// output-orbit narrative: the ONE accumulation metric is alpha = #{(O, Q) :
// projection_V(O) intersects Q}. The regime ladder picks the cheapest exact
// way to count this same number. Shape entries describe when each branch of
// the ladder is reached, not standalone formulas.

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'No detected product symmetry',
    shortLabel: '∅',
    description: 'Each full assignment is its own product orbit, so each update goes to exactly one stored output representative.',
    latex: String.raw`\alpha = M = |X| = \prod_{\ell \in ${notationLatex('l_labels')}} ${notationLatex('n_label')} \quad (|${notationLatex('g_detected')}|=1)`,
    when: 'Detected group G is trivial.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of updates from product-orbit representatives into stored output representatives.' },
      { term: 'M', definition: 'the product-orbit count $|X/G|$.' },
      { term: 'L', definition: 'the full label set of the component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: 'G', definition: 'the detected symmetry group of the component; here $|G| = 1$.' },
      { term: '|G| = 1', definition: 'means every assignment is its own singleton orbit, so $\\alpha = M$ collapses to $|X| = \\prod_\\ell n_\\ell$.' },
    ],
    themeRole: 'caseTrivial',
  },
  allVisible: {
    id: 'allVisible',
    label: 'All labels are visible',
    shortLabel: 'V',
    description: 'There is no summation. Product representatives and stored output representatives are the same quotient.',
    latex: String.raw`\alpha = M = |X / ${notationLatex('g_detected')}|`,
    when: 'W = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count.' },
      { term: 'M', definition: 'the product-orbit count $|X/G|$.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: 'W = \\varnothing', definition: 'no summed labels in this component.' },
      { term: notationLatex('n_label'), definition: 'the size of label $\\ell$.' },
      {
        term: `H = G|_${notationLatex('v_free')}`,
        definition: 'with $V = L$ every $g \\in G$ trivially preserves $V$, so $H = G$ and product orbits and stored output representatives coincide. $\\alpha = M$.',
      },
    ],
    themeRole: 'caseAllVisible',
  },
  allSummed: {
    id: 'allSummed',
    label: 'All labels are summed',
    shortLabel: 'W',
    description: 'The output is one scalar representative. Each product orbit updates it once.',
    latex: String.raw`\alpha = M = |X / ${notationLatex('g_detected')}|`,
    when: 'V = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — equals the product-orbit count $M$ since the output has one stored representative.' },
      { term: 'M', definition: 'the product-orbit count $|X/G|$.' },
      { term: `X = [n]^${notationLatex('l_labels')}`, definition: 'the full assignment space for this component.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `X / ${notationLatex('g_detected')}`, definition: 'the set of $G$-orbits on $X$; by Burnside, $|X/G| = \\tfrac{1}{|G|}\\sum_g |\\mathrm{Fix}(g)|$.' },
      { term: 'V = \\varnothing', definition: 'no visible labels, so $Y = \\{*\\}$ and every product orbit projects to the single stored representative.' },
    ],
    themeRole: 'caseAllSummed',
  },
  mixed: {
    id: 'mixed',
    label: 'Visible and summed labels both appear',
    shortLabel: 'V+W',
    description: `Both $${notationLatex('v_free')}$ and $${notationLatex('w_summed')}$ are present. Projection may have one destination per product orbit, or it may branch across stored output representatives — the regime ladder picks the cheapest exact counter.`,
    latex: String.raw`\alpha = \#\{(${notationLatex('orbit_o')}, Q): ${notationLatex('orbit_o')} \in X/${notationLatex('g_detected')},\ Q \in Y/H,\ \pi_{${notationLatex('v_free')}}(${notationLatex('orbit_o')}) \cap Q \neq \varnothing\}`,
    when: `$${notationLatex('v_free')}$, $${notationLatex('w_summed')}$ both nonempty.`,
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — pairs $(O, Q)$ where $O$ is a product orbit and $Q$ is a stored output representative reached by projecting $O$ to the visible labels.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: notationLatex('w_summed'), definition: 'the summed (contracted) labels.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `H = \\mathrm{Stab}_${notationLatex('g_detected')}(${notationLatex('v_free')})|_${notationLatex('v_free')}`, definition: 'the output representative action, induced by elements of $G$ that preserve $V$ as a set, restricted to $V$.' },
      { term: `Y = [n]^${notationLatex('v_free')}`, definition: 'the output assignment space.' },
      { term: 'Y/H', definition: 'the set of stored output representatives.' },
      { term: notationLatex('orbit_o'), definition: 'a $G$-orbit of full assignments.' },
      { term: `\\pi_{${notationLatex('v_free')}}(${notationLatex('orbit_o')})`, definition: 'its projection onto the visible labels — a subset of $Y$.' },
    ],
    themeRole: 'caseMixed',
  },
};
