// website/components/symmetry-aware-einsum-contractions/engine/regimeSpec.js
import { MATH_COLOR_HEX } from '../lib/mathColors.js';

const V = String.raw`\textcolor{${MATH_COLOR_HEX.V}}{V}`;
const W = String.raw`\textcolor{${MATH_COLOR_HEX.W}}{W}`;
const G = String.raw`\textcolor{${MATH_COLOR_HEX.G}}{G}`;
const MU = String.raw`\textcolor{${MATH_COLOR_HEX.mu}}{\mu}`;
const ALPHA = String.raw`\textcolor{${MATH_COLOR_HEX.alpha}}{\alpha}`;

/**
 * Color palette: each shape and regime has a stable color used by
 * CaseBadge, DecisionLadder, and RegimeTrace. The palette is chosen so
 * sibling regimes contrast while staying in the site's visual family.
 *
 * `glossary` is an array of `{term, definition}` pairs — rendered as a
 * definition list beside the formula in every tooltip (see GlossaryList).
 *
 * `latexMult` / `latexAcc` are Distill-style per-count formulas: the first
 * shows how μ (multiplications) is computed, the second shows how α
 * (accumulations) is computed. `latex` remains as a back-compat alias for
 * the α formula (the one the classic tooltip displays).
 */
export const REGIME_SPEC = {
  singleton: {
    id: 'singleton',
    label: 'Singleton (|V|=1)',
    shortLabel: '|V|=1',
    when: 'Exactly one free label.',
    latexMult: String.raw`${MU} = |X / ${G}|`,
    latexAcc: String.raw`${ALPHA} = \frac{n_\Omega}{|${G}|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    latex: String.raw`${ALPHA} = \frac{n_\Omega}{|${G}|}\sum_{g}\left(\prod_{c \in R} n_c\right)\left(n_\Omega^{c_\Omega(g)} - (n_\Omega-1)^{c_\Omega(g)}\right)`,
    description: 'Weighted inclusion–exclusion — one free label lets Burnside close in a product-minus-product form.',
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
  directProduct: {
    id: 'directProduct',
    label: 'Direct product G_V × G_W',
    shortLabel: 'G_V × G_W',
    when: 'Every generator moves only V-labels OR only W-labels.',
    latexMult: String.raw`${MU} = |X_{${V}} / ${G}_{${V}}| \cdot |X_{${W}} / ${G}_{${W}}|`,
    latexAcc:  String.raw`${ALPHA} = \left(\prod_{\ell \in ${V}} n_\ell\right) \cdot |X_{${W}} / ${G}_{${W}}|`,
    latex: String.raw`${ALPHA} = \left(\prod_{\ell \in ${V}} n_\ell\right) \cdot |X_{${W}} / ${G}_{${W}}|`,
    description: 'Factor V and W — generators split cleanly, so Burnside runs on each side independently.',
    glossary: [
      { term: 'G_V', definition: 'the subgroup of $G$ moving only $V$-labels.' },
      { term: 'G_W', definition: 'the subgroup moving only $W$-labels.' },
      { term: 'G = G_V \\times G_W', definition: 'orbits factor: $V$ contributes $\\prod_{\\ell \\in V} n_\\ell$ (no reduction) and $W$ contributes size-aware Burnside on $G_W$.' },
    ],
    color: '#4A7CFF', // blue
  },
  bruteForceOrbit: {
    id: 'bruteForceOrbit',
    label: 'Brute-force orbit',
    shortLabel: 'Brute',
    when: 'No closed form fired; Π n_ℓ · |G| ≤ budget.',
    latexMult: String.raw`${MU} = |X / ${G}|`,
    latexAcc:  String.raw`${ALPHA} = \sum_{O \in X / ${G}} |\pi_{${V}}(O)|`,
    latex: String.raw`${ALPHA} = \sum_{O \in X / ${G}} |\pi_{${V}}(O)|`,
    description: 'Enumerate X/G — no closed-form shortcut applies; walk every orbit and project.',
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
  'directProduct',
  'bruteForceOrbit',
];
