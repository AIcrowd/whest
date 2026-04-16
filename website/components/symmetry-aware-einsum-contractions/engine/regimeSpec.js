// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js

/**
 * Color palette: each shape and regime has a stable color used by
 * CaseBadge, DecisionLadder, and RegimeTrace. The palette is chosen so
 * sibling regimes contrast while staying in the site's visual family.
 *
 * `glossary` text explains the symbols in `latex` — rendered alongside
 * the formula in every tooltip.
 */
export const REGIME_SPEC = {
  singleton: {
    id: 'singleton',
    label: 'Singleton (|V|=1)',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latex: String.raw`A = \frac{n_\Omega}{|G|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted Burnside inclusion–exclusion on the free label\'s G-orbit.',
    glossary: '$n_\\Omega$: common size of labels in the visible orbit $\\Omega = G\\cdot v$. $R = L\\setminus\\Omega$: the other labels. $c_\\Omega(g)$, $c_R(g)$: the number of cycles of $g$ restricted to $\\Omega$ and $R$. $n_c$: common size within one such cycle (forced equal by the G-action).',
    color: '#8B5CF6', // violet
  },
  fullSymmetric: {
    id: 'fullSymmetric',
    label: 'Full-symmetric S_L',
    shortLabel: 'S_L',
    when: 'G = Sym(L) with uniform label sizes.',
    latex: String.raw`A = n^m \binom{r + n - 1}{n - 1}`,
    description: 'Young-style shortcut for the maximal group.',
    glossary: '$n$: common label size (uniform across all of $L$). $m = |V|$: free-label count. $r = |W|$: summed-label count. $\\binom{r+n-1}{n-1}$: number of multisets of size $r$ drawn from $n$ values.',
    color: '#23B761', // green
  },
  alternating: {
    id: 'alternating',
    label: 'Alternating A_L',
    shortLabel: 'A_L',
    when: 'G = A_L with uniform label sizes.',
    latex: String.raw`A = n^m\binom{r+n-1}{n-1} + \mathbf{1}_{\{n \ge N,\, r \ge 2\}}\binom{n-m}{r}\, n^{\underline{m}}`,
    description: 'Full-symmetric count plus an injective-coloring correction.',
    glossary: '$n, m, r$: as in full-symmetric. $N = |L|$: total label count. $n^{\\underline{m}} = n(n-1)\\cdots(n-m+1)$: falling factorial. The indicator fires only when injective $V$-colorings are possible ($n \\ge N$) and $|W| \\ge 2$ — reflecting that $A_L$ distinguishes two cosets $S_L$ would merge.',
    color: '#0EA5E9', // sky
  },
  directProduct: {
    id: 'directProduct',
    label: 'Direct product G_V × G_W',
    shortLabel: 'G_V × G_W',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latex: String.raw`A = \left(\prod_{\ell \in V} n_\ell\right) \cdot |[n]^W / G_W|`,
    description: 'V and W decouple — visible contributes Π, hidden via Burnside.',
    glossary: '$G_V$: subgroup of $G$ moving only $V$-labels. $G_W$: subgroup moving only $W$-labels. When $G = G_V \\times G_W$, the orbits factor: the visible side contributes $\\prod_{\\ell \\in V} n_\\ell$ (no reduction) and the hidden side contributes size-aware Burnside on $G_W$.',
    color: '#4A7CFF', // blue
  },
  wreath: {
    id: 'wreath',
    label: 'Wreath H ≀ S_b',
    shortLabel: 'H ≀ S_b',
    when: 'b equal-size blocks with per-block H and S_b block-permuting; V is u whole blocks.',
    latex: String.raw`A = n^{su} \binom{h + t_H(n) - 1}{t_H(n) - 1}`,
    description: 'Block-structured closed form.',
    glossary: '$s$: block size. $b$: number of blocks. $u$: number of visible blocks ($V$ covers exactly $u$ whole blocks). $h = b - u$: hidden blocks. $t_H(n) = |[n]^s / H|$: the number of within-block equivalence classes under the base group $H$.',
    color: '#D946EF', // fuchsia
  },
  diagonalSimultaneous: {
    id: 'diagonalSimultaneous',
    label: 'Diagonal S_m',
    shortLabel: 'Diag S_m',
    when: '|V| = |W| = m with a paired S_m acting the same way on both.',
    latex: String.raw`A = \sum_{c_1+\cdots+c_{n_V}=m}\binom{m}{c_1,\ldots,c_{n_V}}\prod_{a=1}^{n_V}\binom{c_a + n_W - 1}{n_W - 1}`,
    description: 'Composition-sum closed form over visible multiplicities.',
    glossary: '$m = |V| = |W|$. $n_V$: common size of each $V$-label. $n_W$: common size of each $W$-label. $(c_1,\\ldots,c_{n_V})$: a composition of $m$ — how many $V$-slots take each of the $n_V$ values. The multinomial counts orderings; the inner $\\binom{c_a + n_W - 1}{n_W - 1}$ counts multisets of $W$-values on the $c_a$ hidden slots paired to value $a$.',
    color: '#14B8A6', // teal
  },
  vSetwiseStable: {
    id: 'vSetwiseStable',
    label: 'V setwise-stable',
    shortLabel: 'Setwise',
    when: 'Every g ∈ G preserves V as a set (but may not fix pointwise).',
    latex: String.raw`A = \sum_{[u] \in [n]^V / H} |H \cdot u| \cdot |[n]^W / G_u|`,
    description: 'Orbit-fiber reduction — iterate H-orbits on V, Burnside on stabilizer per W.',
    glossary: '$H = \\rho_V(G)$: the action of $G$ induced on $V$ (well-defined because $V$ is $G$-stable). $[u]$: a visible-orbit representative. $|H \\cdot u|$: the size of that orbit. $G_u = \\operatorname{Stab}_G(u)$: the elements fixing $u$ pointwise. $|[n]^W / G_u|$: hidden-orbit count under the stabilizer (Burnside on $W$).',
    color: '#FA9E33', // orange
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit projection',
    shortLabel: 'Brute',
    when: 'No closed form fired; Π n_ℓ · |G| ≤ budget.',
    latex: String.raw`A = \sum_{O \in X/G} |\pi_V(O)|`,
    description: 'Always correct; the exact fallback.',
    glossary: '$X = [n]^L$: the full assignment space. $X/G$: the $G$-orbits on $X$. $\\pi_V$: projection that drops all summed coordinates. Each orbit contributes one distinct output bin per distinct projection it touches. Exact but $O(|X|\\cdot|G|)$ — gated by the brute-force budget.',
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
