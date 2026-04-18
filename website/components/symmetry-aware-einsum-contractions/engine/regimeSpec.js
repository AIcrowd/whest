// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

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
    latex: String.raw`\alpha = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted inclusion–exclusion — one free label lets Burnside close in a product-minus-product form.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: 'G', definition: 'the detected symmetry group acting on the component.' },
      { term: '\\Omega = G \\cdot v', definition: 'the $G$-orbit of the single free label $v$.' },
      { term: 'n_\\Omega', definition: 'the common size of labels in $\\Omega$ (forced equal by the action).' },
      { term: 'R = L \\setminus \\Omega', definition: 'the other labels; $g$ acts on $R$ independently.' },
      { term: 'g', definition: 'an element of $G$; we sum over all of them.' },
      { term: 'c_\\Omega(g)', definition: 'the number of cycles of $g$ restricted to $\\Omega$.' },
      { term: 'c_R(g)', definition: 'the number of cycles of $g$ restricted to $R$.' },
      { term: 'n_c', definition: 'the common size of labels in cycle $c$.' },
    ],
    color: '#8B5CF6', // violet
  },
  directProduct: {
    id: 'directProduct',
    label: 'Direct product G_V × G_W',
    shortLabel: 'G_V × G_W',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latex: String.raw`\alpha = \left(\prod_{\ell \in V} n_\ell\right) \cdot |X_W / G_W|`,
    description: 'Factor V and W — generators split cleanly, so Burnside runs on each side independently.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'W', definition: 'the summed (contracted) labels.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: 'G_V', definition: 'the subgroup of $G$ moving only $V$-labels.' },
      { term: 'G_W', definition: 'the subgroup moving only $W$-labels.' },
      { term: 'X_W = [n]^W', definition: 'the assignment space for summed labels alone.' },
      { term: 'X_W / G_W', definition: 'the $G_W$-orbits on $X_W$, counted by size-aware Burnside.' },
      { term: 'G = G_V \\times G_W', definition: 'orbits factor: $V$ contributes $\\prod_{\\ell \\in V} n_\\ell$ (no reduction) and $W$ contributes Burnside on $G_W$.' },
    ],
    color: '#4A7CFF', // blue
  },
  young: {
    id: 'young',
    label: 'Young subgroup (full Sym, cross V/W)',
    shortLabel: 'Young',
    when: 'Cross-V/W elements present AND G = Sym(L_c) AND |V_c|≥2.',
    latex: String.raw`\alpha = n_L^{|V|} \cdot \binom{n_L + |W| - 1}{|W|}`,
    description: 'When the detected group is the full symmetric group on all labels in the component, the pointwise V-stabilizer is a Young subgroup (specifically, Sym(W) — permutations that fix every V-label individually). α can be computed directly via Burnside on Sym(W) — a multinomial closed form, no orbit enumeration needed.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates.' },
      { term: 'n_L', definition: 'the shared dimension size of every label in the component.' },
      { term: '|V|', definition: 'number of free (output) labels in the component.' },
      { term: '|W|', definition: 'number of summed (contracted) labels in the component.' },
      { term: 'Young subgroup', definition: 'a subgroup of a symmetric group that is a product of smaller symmetric groups on a partition of the underlying set.' },
      { term: '\\mathrm{Stab}_G(V)', definition: 'the pointwise V-stabilizer — elements that fix every V-label individually. When G = Sym(L), this is the Young subgroup Sym(W).' },
    ],
    color: '#23B761',
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit',
    shortLabel: 'Brute',
    when: 'Terminal leaf — fires when no closed form applies, gated by $|X| \\cdot |G| \\leq 1{,}500{,}000$ to bound page latency.',
    latex: String.raw`\alpha = \sum_{O \in X / G} |\pi_V(O)|`,
    description: 'Walks each orbit by applying every $g \\in G$ and projecting onto $V$. Declines (αₐ reported as Unavailable) when $|X| \\cdot |G|$ would exceed the latency budget — the refusal is calibrated, not a bug.',
    glossary: [
      { term: '\\alpha', definition: 'the accumulation count — distinct output-bin updates, summed across orbits.' },
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'G', definition: 'the detected symmetry group acting on $X$.' },
      { term: 'X = [n]^L', definition: 'the full assignment space for the component.' },
      { term: 'X / G', definition: 'the set of $G$-orbits on $X$.' },
      { term: 'O', definition: 'a single $G$-orbit in $X / G$.' },
      { term: '\\pi_V(O)', definition: 'the projection of $O$ onto the free labels — the distinct output bins this orbit touches.' },
      { term: 'runtime', definition: 'this method costs $O(|X| \\cdot |G|)$ — exactly one hash insert per (tuple, $g$) pair. Capped by the budget below.' },
      { term: 'budget', definition: 'the cap on $|X| \\cdot |G|$ — counted in $(\\text{tuple}, g)$ $\\textit{pair-touches}$ (each ≈ one hash-map op), set at $1{,}500{,}000$. A calibration, not a constant: roughly what a JS main thread handles in a few hundred ms without visibly hitching the UI. Below the cap the regime fires; above it, it declines and αₐ is reported as Unavailable. The cap bounds demo latency; it does not reflect the einsum\'s structural cost.' },
    ],
    color: '#F0524D', // red
  },
};

// Order mirrors the ladder priority in MIXED_REGIMES.
export const REGIME_PRIORITY = [
  'singleton',
  'directProduct',
  'young',
  'bruteForceOrbit',
];
