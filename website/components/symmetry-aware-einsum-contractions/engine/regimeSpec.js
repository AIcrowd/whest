// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

export const REGIME_SPEC = {
  singleton: {
    id: 'singleton',
    label: 'Singleton (|V|=1)',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latex: String.raw`A = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted Burnside inclusion–exclusion on the free label\'s G-orbit.',
  },
  fullSymmetric: {
    id: 'fullSymmetric',
    label: 'Full-symmetric S_L',
    shortLabel: 'S_L',
    when: 'G = Sym(L) with uniform label sizes.',
    latex: String.raw`A = n^m \binom{r + n - 1}{n - 1}`,
    description: 'Young-style shortcut for the maximal group.',
  },
  alternating: {
    id: 'alternating',
    label: 'Alternating A_L',
    shortLabel: 'A_L',
    when: 'G = A_L with uniform label sizes.',
    latex: String.raw`A = n^m\binom{r+n-1}{n-1} + \mathbf{1}\binom{n-m}{r}n^{\underline{m}}`,
    description: 'Full-symmetric count plus an injective-coloring correction.',
  },
  directProduct: {
    id: 'directProduct',
    label: 'Direct product G_V × G_W',
    shortLabel: 'G_V × G_W',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latex: String.raw`A = \left(\prod_{\ell \in V} n_\ell\right) \cdot |[n]^W / G_W|`,
    description: 'V and W decouple — visible contributes Π, hidden via Burnside.',
  },
  wreath: {
    id: 'wreath',
    label: 'Wreath H ≀ S_b',
    shortLabel: 'H ≀ S_b',
    when: 'b equal-size blocks with per-block H and S_b block-permuting; V is u whole blocks.',
    latex: String.raw`A = n^{su} \binom{h + t_H(n) - 1}{t_H(n) - 1}`,
    description: 'Block-structured closed form.',
  },
  diagonalSimultaneous: {
    id: 'diagonalSimultaneous',
    label: 'Diagonal S_m',
    shortLabel: 'Diag S_m',
    when: '|V| = |W| = m with a paired S_m acting the same way on both.',
    latex: String.raw`A = \sum_{c_1+\cdots+c_{n_V}=m}\binom{m}{c_1,\ldots,c_{n_V}}\prod_{a}\binom{c_a + n_W - 1}{n_W - 1}`,
    description: 'Composition-sum closed form over visible multiplicities.',
  },
  vSetwiseStable: {
    id: 'vSetwiseStable',
    label: 'V setwise-stable',
    shortLabel: 'Setwise',
    when: 'Every g ∈ G preserves V as a set (but may not fix pointwise).',
    latex: String.raw`A = \sum_{[u] \in [n]^V / H} |H \cdot u| \cdot |[n]^W / G_u|`,
    description: 'Orbit-fiber reduction — iterate H-orbits on V, Burnside on stabilizer per W.',
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit projection',
    shortLabel: 'Brute',
    when: 'No closed form fired; Π n_ℓ · |G| ≤ budget.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    description: 'Always correct; the exact fallback.',
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
