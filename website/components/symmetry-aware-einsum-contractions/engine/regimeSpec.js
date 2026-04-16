// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

/**
 * Color palette: each shape and regime has a stable color used by
 * CaseBadge, DecisionLadder, and RegimeTrace. The palette is chosen so
 * sibling regimes contrast while staying in the site's visual family.
 *
 * `glossary` is an array of `{term, definition}` pairs — rendered as a
 * definition list beside the formula in every tooltip (see GlossaryList).
 */
export const REGIME_SPEC = {
  singleton: {
    id: 'singleton',
    label: 'Singleton (|V|=1)',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latex: String.raw`A = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted Burnside inclusion–exclusion on the free label\'s G-orbit.',
    glossary: [
      { term: '\\Omega = G \\cdot v', definition: 'the $G$-orbit of the single free label $v$.' },
      { term: 'n_\\Omega', definition: 'the common size of labels in $\\Omega$ (forced equal by the action).' },
      { term: 'R = L \\setminus \\Omega', definition: 'the other labels; $g$ acts on $R$ independently.' },
      { term: 'c_\\Omega(g)', definition: 'the number of cycles of $g$ restricted to $\\Omega$.' },
      { term: 'c_R(g)', definition: 'the number of cycles of $g$ restricted to $R$.' },
      { term: 'n_c', definition: 'the common size of labels in cycle $c$.' },
    ],
    color: '#8B5CF6', // violet
  },
  fullSymmetric: {
    id: 'fullSymmetric',
    label: 'Full-symmetric S_L',
    shortLabel: 'S_L',
    when: 'G = Sym(L) with uniform label sizes.',
    latex: String.raw`A = n^m \binom{r + n - 1}{n - 1}`,
    description: 'Young-style shortcut for the maximal group.',
    glossary: [
      { term: 'n', definition: 'the common label size (uniform across all of $L$).' },
      { term: 'm = |V|', definition: 'the number of free labels.' },
      { term: 'r = |W|', definition: 'the number of summed labels.' },
      { term: '\\binom{r+n-1}{n-1}', definition: 'the number of multisets of size $r$ drawn from $n$ values.' },
    ],
    color: '#23B761', // green
  },
  alternating: {
    id: 'alternating',
    label: 'Alternating A_L',
    shortLabel: 'A_L',
    when: 'G = A_L with uniform label sizes.',
    latex: String.raw`A = n^m\binom{r+n-1}{n-1} + \mathbf{1}_{\{n \ge N,\, r \ge 2\}}\binom{n-m}{r}\, n^{\underline{m}}`,
    description: 'Full-symmetric count plus an injective-coloring correction.',
    glossary: [
      { term: 'n, m, r', definition: 'as in the full-symmetric case.' },
      { term: 'N = |L|', definition: 'the total label count.' },
      { term: 'n^{\\underline{m}} = n(n-1)\\cdots(n-m+1)', definition: 'the falling factorial.' },
      { term: '\\mathbf{1}_{\\{n \\ge N,\\, r \\ge 2\\}}', definition: 'the indicator fires only when injective $V$-colorings are possible and $|W| \\ge 2$ — reflecting cosets that $S_L$ would merge but $A_L$ distinguishes.' },
    ],
    color: '#0EA5E9', // sky
  },
  directProduct: {
    id: 'directProduct',
    label: 'Direct product G_V × G_W',
    shortLabel: 'G_V × G_W',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latex: String.raw`A = \left(\prod_{\ell \in V} n_\ell\right) \cdot |[n]^W / G_W|`,
    description: 'V and W decouple — visible contributes Π, hidden via Burnside.',
    glossary: [
      { term: 'G_V', definition: 'the subgroup of $G$ moving only $V$-labels.' },
      { term: 'G_W', definition: 'the subgroup moving only $W$-labels.' },
      { term: 'G = G_V \\times G_W', definition: 'orbits factor: $V$ contributes $\\prod_{\\ell \\in V} n_\\ell$ (no reduction) and $W$ contributes size-aware Burnside on $G_W$.' },
    ],
    color: '#4A7CFF', // blue
  },
  wreath: {
    id: 'wreath',
    label: 'Wreath H ≀ S_b',
    shortLabel: 'H ≀ S_b',
    when: 'b equal-size blocks with per-block H and S_b block-permuting; V is u whole blocks.',
    latex: String.raw`A = n^{su} \binom{h + t_H(n) - 1}{t_H(n) - 1}`,
    description: 'Block-structured closed form.',
    glossary: [
      { term: 's', definition: 'the block size.' },
      { term: 'b', definition: 'the number of blocks.' },
      { term: 'u', definition: 'the number of visible blocks — $V$ covers exactly $u$ whole blocks.' },
      { term: 'h = b - u', definition: 'the number of hidden blocks.' },
      { term: 't_H(n) = |[n]^s / H|', definition: 'the number of within-block equivalence classes under the base group $H$.' },
    ],
    color: '#D946EF', // fuchsia
  },
  diagonalSimultaneous: {
    id: 'diagonalSimultaneous',
    label: 'Diagonal S_m',
    shortLabel: 'Diag S_m',
    when: '|V| = |W| = m with a paired S_m acting the same way on both.',
    latex: String.raw`A = \sum_{c_1+\cdots+c_{n_V}=m}\binom{m}{c_1,\ldots,c_{n_V}}\prod_{a=1}^{n_V}\binom{c_a + n_W - 1}{n_W - 1}`,
    description: 'Composition-sum closed form over visible multiplicities.',
    glossary: [
      { term: 'm = |V| = |W|', definition: 'each side has the same cardinality.' },
      { term: 'n_V', definition: 'the common size of each $V$-label.' },
      { term: 'n_W', definition: 'the common size of each $W$-label.' },
      { term: '(c_1, \\ldots, c_{n_V})', definition: 'a composition of $m$ — how many $V$-slots take each of the $n_V$ values.' },
      { term: '\\binom{m}{c_1,\\ldots,c_{n_V}}', definition: 'the multinomial counting orderings.' },
      { term: '\\binom{c_a + n_W - 1}{n_W - 1}', definition: 'multisets of $W$-values on the $c_a$ hidden slots paired to value $a$.' },
    ],
    color: '#14B8A6', // teal
  },
  vSetwiseStable: {
    id: 'vSetwiseStable',
    label: 'V setwise-stable',
    shortLabel: 'Setwise',
    when: 'Every g ∈ G preserves V as a set (but may not fix pointwise).',
    latex: String.raw`A = \sum_{[u] \in [n]^V / H} |H \cdot u| \cdot |[n]^W / G_u|`,
    description: 'Orbit-fiber reduction — iterate H-orbits on V, Burnside on stabilizer per W.',
    glossary: [
      { term: 'H = \\rho_V(G)', definition: 'the action of $G$ induced on $V$ (well-defined because $V$ is $G$-stable).' },
      { term: '[u]', definition: 'a visible-orbit representative.' },
      { term: '|H \\cdot u|', definition: 'the size of that orbit.' },
      { term: 'G_u = \\operatorname{Stab}_G(u)', definition: 'the elements fixing $u$ pointwise.' },
      { term: '|[n]^W / G_u|', definition: 'hidden-orbit count under the stabilizer (Burnside on $W$).' },
    ],
    color: '#FA9E33', // orange
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit projection',
    shortLabel: 'Brute',
    when: 'No closed form fired; Π n_ℓ · |G| ≤ budget.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    description: 'Always correct; the exact fallback.',
    glossary: [
      { term: 'X = [n]^L', definition: 'the full assignment space.' },
      { term: 'X / G', definition: 'the $G$-orbits on $X$.' },
      { term: '\\pi_V', definition: 'projection that drops all summed coordinates.' },
      { term: 'O(|X| \\cdot |G|)', definition: 'each orbit contributes one distinct output bin per distinct projection it touches — gated by the brute-force budget.' },
    ],
    color: '#F0524D', // red
  },
};

// Order mirrors the ladder priority in MIXED_REGIMES.
export const REGIME_PRIORITY = [
  'singleton',
  'fullSymmetric',
  'alternating',
  'directProduct',
  'wreath',
  'diagonalSimultaneous',
  'vSetwiseStable',
  'bruteForceOrbit',
];
