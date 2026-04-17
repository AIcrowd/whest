// website/components/symmetry-aware-einsum-contractions/engine/shapeSpec.js
import { MATH_COLOR_HEX } from '../lib/mathColors.js';

// Pre-built `\textcolor{...}{token}` fragments. Using these (rather than
// raw V/W/G/μ/α in the latex strings) keeps every formula on the page
// colour-consistent with its prose — one symbol, one colour, everywhere.
const V = String.raw`\textcolor{${MATH_COLOR_HEX.V}}{V}`;
const W = String.raw`\textcolor{${MATH_COLOR_HEX.W}}{W}`;
const G = String.raw`\textcolor{${MATH_COLOR_HEX.G}}{G}`;
const MU = String.raw`\textcolor{${MATH_COLOR_HEX.mu}}{\mu}`;
const ALPHA = String.raw`\textcolor{${MATH_COLOR_HEX.alpha}}{\alpha}`;

export const SHAPE_SPEC = {
  trivial: {
    id: 'trivial',
    label: 'Trivial',
    shortLabel: '∅',
    description: 'Direct count — no symmetry to exploit, so every cell is distinct work.',
    latexMult: String.raw`${MU} = \prod_{\ell \in L} n_\ell`,
    latexAcc:  String.raw`${ALPHA} = \prod_{\ell \in L} n_\ell`,
    latex: String.raw`${ALPHA} = \prod_{\ell \in L} n_\ell \quad (|${G}|=1)`,
    when: 'Detected group G is trivial.',
    glossary: [
      { term: 'L', definition: 'the full label set of the component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$ (its dimension).' },
      { term: '|G| = 1', definition: 'means every assignment is its own singleton orbit, so $\\alpha$ collapses to $|X| = \\prod_\\ell n_\\ell$.' },
    ],
    color: '#94A3B8', // slate
  },
  allVisible: {
    id: 'allVisible',
    label: 'All-visible',
    shortLabel: 'V',
    description: 'Cartesian product — W is empty, so each free-label tuple is its own output bin.',
    latexMult: String.raw`${MU} = |X / ${G}| \;=\; \tfrac{1}{|${G}|} \sum_{g} \prod_{c} n_c`,
    latexAcc:  String.raw`${ALPHA} = \prod_{\ell \in ${V}} n_\ell`,
    latex: String.raw`${ALPHA} = \prod_{\ell \in ${V}} n_\ell`,
    when: 'W = ∅.',
    glossary: [
      { term: 'V', definition: 'the free (output) labels.' },
      { term: 'W = \\varnothing', definition: 'no summed labels in this component.' },
      { term: 'n_\\ell', definition: 'the size of label $\\ell$. Every output bin is written exactly once, so $\\alpha$ is the Cartesian product of free-label sizes.' },
    ],
    color: '#4A7CFF', // blue
  },
  allSummed: {
    id: 'allSummed',
    label: 'All-summed',
    shortLabel: 'W',
    description: 'Size-aware Burnside — V is empty, so every orbit maps to the single scalar output.',
    latexMult: String.raw`${MU} = |X / ${G}| \;=\; \tfrac{1}{|${G}|} \sum_{g} \prod_{c \in \mathrm{cycles}(g)} n_c`,
    latexAcc:  String.raw`${ALPHA} = ${MU}`,
    latex: String.raw`${ALPHA} = |X / ${G}| = \tfrac{1}{|${G}|} \sum_{g \in ${G}} \prod_{c \in \mathrm{cycles}(g)} n_c`,
    when: 'V = ∅.',
    glossary: [
      { term: 'X = [n]^L', definition: 'the full assignment space for this component.' },
      { term: 'X / G', definition: 'the $G$-orbits on $X$. Each orbit writes exactly one output bin, so $\\alpha$ equals the orbit count.' },
      { term: 'n_c', definition: 'the common label-size inside cycle $c$ of $g$ (forced equal by the action).' },
    ],
    color: '#64748B', // darker slate
  },
  mixed: {
    id: 'mixed',
    label: 'Mixed',
    shortLabel: 'V+W',
    description: 'Mixed shape — V and W both nonempty; dispatch to the regime ladder.',
    latexMult: String.raw`${MU} = |X / ${G}|`,
    latexAcc:  String.raw`${ALPHA} = \sum_{O \in X/${G}} |\pi_{${V}}(O)|`,
    latex: String.raw`${ALPHA} = \sum_{O \in X/${G}} |\pi_{${V}}(O)|`,
    when: 'V, W both nonempty.',
    glossary: [
      { term: 'O', definition: 'a $G$-orbit of full assignments.' },
      { term: '\\pi_V(O)', definition: 'its projection onto the free labels — the distinct output bins that orbit touches.' },
      { term: '\\alpha', definition: 'the sum of projection sizes across all orbits — equivalent to counting (multiplication-orbit, output-bin) pairs.' },
    ],
    color: '#0F172A', // slate-900 (gateway)
  },
};
