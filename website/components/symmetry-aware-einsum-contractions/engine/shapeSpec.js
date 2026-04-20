// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js

import { notationColor, notationLatex } from '../lib/notationSystem.js';

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'Direct count — no symmetry to exploit, so every cell is distinct work.',
    latex: String.raw`\alpha = \prod_{\ell \in ${notationLatex('l_labels')}} ${notationLatex('n_label')} \quad (|${notationLatex('g_detected')}|=1)`,
    when: 'Detected group G is trivial.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of distinct output-bin updates.' },
      { term: 'L', definition: 'the full label set of the component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: 'G', definition: 'the detected symmetry group of the component; here $|G| = 1$.' },
      { term: '|G| = 1', definition: 'means every assignment is its own singleton orbit, so $\\alpha$ collapses to $|X| = \\prod_\\ell n_\\ell$.' },
    ],
    color: notationColor('w_summed'),
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'Cartesian product — W is empty, so each free-label tuple is its own output bin.',
    latex: String.raw`\alpha = \prod_{\ell \in ${notationLatex('v_free')}} ${notationLatex('n_label')}`,
    when: 'W = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of distinct output-bin updates.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: 'W = \\varnothing', definition: 'no summed labels in this component.' },
      { term: notationLatex('n_label'), definition: 'the size of label $\\ell$. Every output bin is written exactly once, so $\\alpha$ is the Cartesian product of free-label sizes.' },
      {
        term: `${notationLatex('v_free')}\\text{-symmetry}`,
        definition: `Symmetry on $${notationLatex('v_free')}$ does NOT reduce $\\alpha$ here — the output tensor is dense, so every $${notationLatex('v_free')}$-tuple is its own bin even when two tuples are related by a symmetry. $${notationLatex('v_free')}$-symmetry DOES reduce the multiplication count $\\mu$: you compute each orbit's product once and reuse it across the orbit's positions. See the Multiplication Cost card.`,
      },
    ],
    color: notationColor('v_free'),
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'Size-aware Burnside — V is empty, so every orbit maps to the single scalar output.',
    latex: String.raw`\alpha = |X / ${notationLatex('g_detected')}| = \tfrac{1}{|${notationLatex('g_detected')}|} \sum_{g \in ${notationLatex('g_detected')}} \prod_{c \in \mathrm{cycles}(g)} ${notationLatex('n_cycle')}`,
    when: 'V = ∅.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — equals the number of $G$-orbits here, since $V = \\varnothing$ means each orbit writes the single scalar output once.' },
      { term: `X = [n]^${notationLatex('l_labels')}`, definition: 'the full assignment space for this component.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `X / ${notationLatex('g_detected')}`, definition: 'the set of $G$-orbits on $X$; by Burnside, $|X/G| = \\tfrac{1}{|G|}\\sum_g |\\mathrm{Fix}(g)|$.' },
      { term: 'g', definition: 'an element of $G$; we sum over all of them.' },
      { term: '\\mathrm{cycles}(g)', definition: 'the disjoint cycles of $g$ viewed as a label permutation.' },
      { term: notationLatex('n_cycle'), definition: 'the common label-size inside cycle $c$ of $g$ (forced equal by the action).' },
    ],
    color: notationColor('w_summed_component'),
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: `Mixed shape — both $${notationLatex('v_free')}$ and $${notationLatex('w_summed')}$ are nonempty; dispatch to the regime ladder.`,
    latex: String.raw`\alpha = \sum_{${notationLatex('orbit_o')} \in X/${notationLatex('g_detected')}} |${notationLatex('projection_pi_v_free')}|`,
    when: `$${notationLatex('v_free')}$, $${notationLatex('w_summed')}$ both nonempty.`,
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — total distinct output-bin updates across all orbits.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: notationLatex('w_summed'), definition: 'the summed (contracted) labels.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `X = [n]^${notationLatex('l_labels')}`, definition: 'the full assignment space.' },
      { term: notationLatex('orbit_o'), definition: 'a $G$-orbit of full assignments in $X$.' },
      { term: notationLatex('projection_pi_v_free'), definition: "its projection onto the free labels — the distinct output bins that orbit touches." },
    ],
    color: '#0F172A', // slate-900 (gateway)
  },
};
