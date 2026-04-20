// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

import { notationColor, notationLatex } from '../lib/notationSystem.js';

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
  singleton: {
    id: 'singleton',
    label: 'Singleton (|V|=1)',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latex: String.raw`\alpha = \frac{${notationLatex('n_omega')}}{|${notationLatex('g_detected')}|}\sum_{g}\left(\prod_{c \in ${notationLatex('r_complement')}} ${notationLatex('n_cycle')}\right)\left(${notationLatex('n_omega')}^{c_\Omega(g)} - (${notationLatex('n_omega')}-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted inclusion–exclusion — one free label lets Burnside close in a product-minus-product form.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on the component.' },
      { term: `\\Omega = ${notationLatex('g_detected')} \\cdot v`, definition: 'the $G$-orbit of the single free label $v$.' },
      { term: notationLatex('n_omega'), definition: 'the common size of labels in $\\Omega$ (forced equal by the action).' },
      { term: `${notationLatex('r_complement')} = ${notationLatex('l_labels')} \\setminus \\Omega`, definition: 'the other labels; $g$ acts on $R$ independently.' },
      { term: 'g', definition: 'an element of $G$; we sum over all of them.' },
      { term: 'c_\\Omega(g)', definition: 'the number of cycles of $g$ restricted to $\\Omega$.' },
      { term: '\\prod_{c \\in R} n_c', definition: 'multiply one factor $n_c$ for each cycle $c$ of $g$ inside $R$.' },
      { term: notationLatex('n_cycle'), definition: 'the common size of labels in cycle $c$.' },
    ],
    color: notationColor('g_detected'),
  },
  directProduct: {
    id: 'directProduct',
    label: 'Direct Product',
    shortLabel: 'G_V_free × G_W_summed',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latex: String.raw`\alpha = \left(\prod_{\ell \in ${notationLatex('v_free')}} ${notationLatex('n_label')}\right) \cdot |${notationLatex('x_w_summed')} / ${notationLatex('g_w_factor')}|`,
    description: 'Factor V and W — generators split cleanly, so Burnside runs on each side independently.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: notationLatex('w_summed'), definition: 'the summed (contracted) labels.' },
      { term: notationLatex('n_label'), definition: 'the size of label $\\ell$ (its dimension).' },
      { term: notationLatex('g_v_factor'), definition: `the subgroup of $${notationLatex('g_detected')}$ moving only $${notationLatex('v_free')}$-labels.` },
      { term: notationLatex('g_w_factor'), definition: `the subgroup moving only $${notationLatex('w_summed')}$-labels.` },
      { term: `${notationLatex('x_w_summed')} = [n]^{${notationLatex('w_summed')}}`, definition: 'the assignment space for summed labels alone.' },
      { term: `${notationLatex('x_w_summed')} / ${notationLatex('g_w_factor')}`, definition: `the $${notationLatex('g_w_factor')}$-orbits on $${notationLatex('x_w_summed')}$, counted by size-aware Burnside.` },
      { term: `${notationLatex('g_detected')} = ${notationLatex('g_v_factor')} \\times ${notationLatex('g_w_factor')}`, definition: `orbits factor: $${notationLatex('v_free')}$ contributes $\\prod_{\\ell \\in ${notationLatex('v_free')}} ${notationLatex('n_label')}$ (no reduction) and $${notationLatex('w_summed')}$ contributes Burnside on $${notationLatex('g_w_factor')}$.` },
    ],
    color: notationColor('pi_relabeling'),
  },
  young: {
    id: 'young',
    label: 'Young subgroup',
    shortLabel: 'Young',
    when: 'Cross-V/W elements present AND G = Sym(L_c) AND |V_c|≥2.',
    latex: String.raw`\alpha = n_L^{|${notationLatex('v_free')}|} \cdot \binom{n_L + |${notationLatex('w_summed')}| - 1}{|${notationLatex('w_summed')}|}`,
    description: 'When the detected group is the full symmetric group on all labels in the component, the pointwise V-stabilizer is a Young subgroup (specifically, Sym(W) — permutations that fix every V-label individually). α can be computed directly via Burnside on Sym(W) — a multinomial closed form, no orbit enumeration needed.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: 'n_L', definition: 'the shared dimension size of every label in the component.' },
      { term: `|${notationLatex('v_free')}|`, definition: 'number of free (output) labels in the component.' },
      { term: `|${notationLatex('w_summed')}|`, definition: 'number of summed (contracted) labels in the component.' },
      { term: 'Young subgroup', definition: 'a subgroup of a symmetric group that is a product of smaller symmetric groups on a partition of the underlying set.' },
      {
        term: `\\mathrm{Stab}_{${notationLatex('g_detected')}}(${notationLatex('v_free')})`,
        definition: `the pointwise $${notationLatex('v_free')}$-stabilizer — elements that fix every $${notationLatex('v_free')}$-label individually. When $G = \\mathrm{Sym}(L)$, this is the Young subgroup $\\mathrm{Sym}(${notationLatex('w_summed')})$.`,
      },
    ],
    color: '#23B761',
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit',
    shortLabel: 'Brute',
    when: 'Terminal leaf — fires when no closed form applies, gated by $|X| \\cdot |G| \\leq 1{,}500{,}000$ to bound page latency.',
    latex: String.raw`\alpha = \sum_{${notationLatex('orbit_o')} \in ${notationLatex('x_space')} / ${notationLatex('g_detected')}} |${notationLatex('projection_pi_v_free')}|`,
    description: 'Walks each orbit by applying every $g \\in G$ and projecting onto $V$. Declines (αₐ reported as Unavailable) when $|X| \\cdot |G|$ would exceed the latency budget — the refusal is calibrated, not a bug.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates, summed across orbits.' },
      { term: notationLatex('v_free'), definition: 'the free (output) labels.' },
      { term: notationLatex('g_detected'), definition: 'the detected symmetry group acting on $X$.' },
      { term: `X = [n]^${notationLatex('l_labels')}`, definition: 'the full assignment space for the component.' },
      { term: `${notationLatex('x_space')} / ${notationLatex('g_detected')}`, definition: 'the set of $G$-orbits on $X$.' },
      { term: notationLatex('orbit_o'), definition: `a single $${notationLatex('g_detected')}$-orbit in $${notationLatex('x_space')} / ${notationLatex('g_detected')}$.` },
      { term: notationLatex('projection_pi_v_free'), definition: 'the projection of $O$ onto the free labels — the distinct output bins this orbit touches.' },
      { term: 'runtime', definition: 'this method costs $O(|X| \\cdot |G|)$ — exactly one hash insert per (tuple, $g$) pair. Capped by the budget below.' },
      { term: 'budget', definition: 'the cap on $|X| \\cdot |G|$ — counted in $(\\text{tuple}, g)$ $\\textit{pair-touches}$ (each ≈ one hash-map op), set at $1{,}500{,}000$. A calibration, not a constant: roughly what a JS main thread handles in a few hundred ms without visibly hitching the UI. Below the cap the regime fires; above it, it declines and αₐ is reported as Unavailable. The cap bounds demo latency; it does not reflect the einsum\'s structural cost.' },
    ],
    color: notationColor('alpha_total'),
  },
};

// Order mirrors the ladder priority in MIXED_REGIMES.
export const REGIME_PRIORITY = [
  'singleton',
  'directProduct',
  'young',
  'bruteForceOrbit',
];
