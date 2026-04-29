// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

import { notationLatex } from '../lib/notationSystem.js';

/**
 * Color palette: each shape and regime has a stable color used by
 * CaseBadge, DecisionLadder, and RegimeTrace. The palette is chosen so
 * sibling regimes contrast while staying in the site's visual family.
 *
 * `glossary` is an array of `{term, definition}` pairs — rendered as a
 * definition list beside the formula in every tooltip (see GlossaryList).
 *
 * Only α (accumulation) formulas are carried per case; μ (multiplications)
 * is the universal Burnside orbit count across every case, so showing it
 * per-tooltip adds no signal. All symbols appearing in the α formula (and
 * in the description prose) must be explained in the glossary.
 */
export const REGIME_SPEC = {
  functionalProjection: {
    id: 'functionalProjection',
    label: 'One destination per product orbit',
    shortLabel: 'A = M',
    when: 'Every detected symmetry preserves the output-label set V as a set, so projection descends to a single stored output representative per product orbit.',
    latex: String.raw`\alpha = M = |X / ${notationLatex('g_detected')}|`,
    description: 'Every detected symmetry preserves the output-label set, so each product orbit reaches one stored output representative. The accumulation count equals the product-orbit count.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — number of updates from product-orbit representatives into stored output representatives.' },
      { term: 'M', definition: 'the product-orbit count $|X/G|$.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on the component.' },
      { term: 'V', definition: 'the visible/output label set of the component.' },
      { term: `${notationLatex('g_detected')}\\cdot V = V`, definition: 'every $g \\in G$ maps $V$ into $V$ as a set, so the projection $\\pi_V$ descends to the quotient.' },
    ],
    themeRole: 'caseFunctionalProjection',
  },
  singleton: {
    id: 'singleton',
    label: 'Single output label',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latex: String.raw`\alpha = \frac{${notationLatex('n_omega')}}{|${notationLatex('g_detected')}|}\sum_{g}\left(\prod_{c \in ${notationLatex('r_complement')}} ${notationLatex('n_cycle')}\right)\left(${notationLatex('n_omega')}^{${notationLatex('c_omega_cycles')}} - (${notationLatex('n_omega')}-1)^{${notationLatex('c_omega_cycles')}}\right)`,
    description: 'With one output axis, the output representative action H is trivial; the singleton formula counts product-orbit destinations directly via weighted Burnside (inclusion–exclusion on the free label\'s orbit).',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — pairs $(O, Q)$ where $O$ is a product orbit and $Q$ is a stored output representative reached by projecting $O$.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on the component.' },
      { term: `\\Omega = ${notationLatex('g_detected')} \\cdot v`, definition: 'the $G$-orbit of the single free label $v$.' },
      { term: notationLatex('n_omega'), definition: 'the common size of labels in $\\Omega$ (forced equal by the action).' },
      { term: `${notationLatex('r_complement')} = ${notationLatex('l_labels')} \\setminus \\Omega`, definition: 'the other labels; $g$ acts on $R$ independently.' },
      { term: 'g', definition: 'an element of $G$; we sum over all of them.' },
      { term: notationLatex('c_omega_cycles'), definition: 'the number of cycles of $g$ restricted to $\\Omega$.' },
      { term: '\\prod_{c \\in R} n_c', definition: 'multiply one factor $n_c$ for each cycle $c$ of $g$ inside $R$.' },
      { term: notationLatex('n_cycle'), definition: 'the common size of labels in cycle $c$.' },
    ],
    themeRole: 'caseSingleton',
  },
  young: {
    id: 'young',
    label: 'Full symmetric multiset formula',
    shortLabel: 'Young',
    when: 'Cross-V/W elements present, G is the full symmetric group on the component labels, all component label sizes agree, and |V| ≥ 2.',
    latex: String.raw`\alpha = \binom{n_L + |${notationLatex('v_free')}| - 1}{|${notationLatex('v_free')}|}\binom{n_L + |${notationLatex('w_summed')}| - 1}{|${notationLatex('w_summed')}|}`,
    description: 'Full symmetry turns visible and summed assignments into multisets. The visible factor counts stored output representatives directly; the summed factor counts the multisets that combine independently with each visible multiset to form a product orbit.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — pairs $(O, Q)$ where $O$ is a product orbit and $Q$ is a stored output representative reached by $O$.' },
      { term: 'n_L', definition: 'the shared dimension size of every label in the component.' },
      { term: `|${notationLatex('v_free')}|`, definition: 'number of free (output) labels in the component.' },
      { term: `|${notationLatex('w_summed')}|`, definition: 'number of summed (contracted) labels in the component.' },
      { term: `\\binom{n_L + |${notationLatex('v_free')}| - 1}{|${notationLatex('v_free')}|}`, definition: 'number of size-$|V|$ multisets from $[n_L]$ — equivalently $|Y/H|$, the count of stored output representatives.' },
      { term: `\\binom{n_L + |${notationLatex('w_summed')}| - 1}{|${notationLatex('w_summed')}|}`, definition: 'number of size-$|W|$ multisets from $[n_L]$ — the summed-side multisets that combine independently with each visible multiset.' },
      {
        term: `\\mathrm{Sym}(${notationLatex('w_summed')})`,
        definition: `the symmetric group on the summed labels. When $G = \\mathrm{Sym}(${notationLatex('l_labels')})$, the pointwise $${notationLatex('v_free')}$-stabilizer is the Young subgroup $\\mathrm{Sym}(${notationLatex('w_summed')})$ — the same combinatorics that produces the multiset count on the summed side.`,
      },
    ],
    themeRole: 'caseYoung',
  },
  partitionCount: {
    id: 'partitionCount',
    label: 'Typed partition count',
    shortLabel: 'Partition',
    when: 'Heterogeneous label dimensions or non-full symmetry — counts equality-pattern orbits and the stored output representatives each pattern can reach.',
    latex: String.raw`\alpha = \sum_{\tilde{x}\in P_{\mathrm{typed}}(${notationLatex('l_labels')})/${notationLatex('g_detected')}} \frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}\,|A_{\tilde{x}}/H|`,
    description: 'Counts equality-pattern orbits and the stored output representatives each pattern can reach. Blocks may merge only positions with the same domain class; each domain contributes its own falling factorial.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — pairs $(O, Q)$ where $O$ is a product orbit and $Q$ is a stored output representative.' },
      { term: '\\tilde{x}', definition: 'a typed equality pattern — a partition of label positions where blocks may only merge positions with the same domain class.' },
      { term: `P_{\\mathrm{typed}}(${notationLatex('l_labels')})/${notationLatex('g_detected')}`, definition: 'typed equality patterns up to the action of $G$.' },
      { term: '(n_s)_{b_s}', definition: 'falling factorial $n_s (n_s - 1) \\cdots (n_s - b_s + 1)$, where $b_s$ is the number of $\\tilde{x}$-blocks with domain class $s$.' },
      { term: '\\overline{G}_{\\tilde{x}}', definition: 'the induced action of $\\mathrm{Stab}_G(\\tilde{x})$ on the blocks of $\\tilde{x}$ — the IMAGE in $\\mathrm{Sym}(\\mathrm{blocks})$, not the raw stabilizer order.' },
      { term: '|A_{\\tilde{x}}/H|', definition: 'number of stored output representatives reached by any product orbit above $\\tilde{x}$.' },
      { term: 'H', definition: '$\\mathrm{Stab}_G(V)|_V$ — the output representative action induced by $G$.' },
    ],
    themeRole: 'casePartitionCount',
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Corrected brute-force orbit count',
    shortLabel: 'Brute',
    when: 'Terminal leaf — fires when no closed form applies, gated by $|X| \\cdot |G| \\leq 1{,}500{,}000$ to bound page latency.',
    latex: String.raw`\alpha = \sum_{${notationLatex('orbit_o')} \in ${notationLatex('x_space')} / ${notationLatex('g_detected')}} |\pi_V(${notationLatex('orbit_o')})/H|`,
    description: 'Enumerates product orbits and canonicalizes projected outputs under the derived output action H. Used only when no analytic regime applies and the pair-touch budget is small enough for the interactive page.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — pairs $(O, Q)$ summed across orbits.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `X = [n]^${notationLatex('l_labels')}`, definition: 'the full assignment space for the component.' },
      { term: `${notationLatex('x_space')} / ${notationLatex('g_detected')}`, definition: 'the set of $G$-orbits on $X$.' },
      { term: notationLatex('orbit_o'), definition: `a single $${notationLatex('g_detected')}$-orbit in $${notationLatex('x_space')} / ${notationLatex('g_detected')}$.` },
      { term: `\\pi_V(${notationLatex('orbit_o')})/H`, definition: 'the projection of $O$ onto $V$, canonicalized under the output representative action $H = \\mathrm{Stab}_G(V)|_V$.' },
      { term: 'runtime', definition: 'this method costs $O(|X| \\cdot |G|)$ — exactly one hash insert per (tuple, $g$) pair. Capped by the budget below.' },
      { term: 'budget', definition: 'The cap on |X| · |G| pair-touches. Below the cap this regime is exact; above the cap the UI reports the component count as unavailable rather than guessing.' },
    ],
    themeRole: 'caseBruteForceOrbit',
  },
};

// Order mirrors the ladder priority in MIXED_REGIMES.
export const REGIME_PRIORITY = [
  'singleton',
  'young',
  'partitionCount',
  'bruteForceOrbit',
];
