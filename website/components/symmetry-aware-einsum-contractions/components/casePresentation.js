import { CASE_META } from '../engine/componentDecomposition.js';

export const CASE_PRESENTATION = {
  A: {
    label: CASE_META.A.label,
    shortLabel: 'A',
    tooltip: {
      title: CASE_META.A.label,
      body: 'All labels are free (output). Symmetry reduces unique multiplications, but every output bin must still be written.',
      latex: String.raw`\rho_a = \prod_{\ell \in V_a} n_\ell`,
    },
  },
  B: {
    label: CASE_META.B.label,
    shortLabel: 'B',
    tooltip: {
      title: CASE_META.B.label,
      body: 'All labels are summed. Orbits collapse both multiplications and accumulations equally.',
      latex: String.raw`\rho_a = |I_a / G_a| \text{ (Burnside)}`,
    },
  },
  C: {
    label: CASE_META.C.label,
    shortLabel: 'C',
    tooltip: {
      title: CASE_META.C.label,
      body: 'V and W labels are both present but no generator crosses the boundary. Needs orbit enumeration.',
      latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    },
  },
  D: {
    label: CASE_META.D.label,
    shortLabel: 'D',
    tooltip: {
      title: CASE_META.D.label,
      body: 'Cross-boundary generators with the full symmetric group. The V-stabilizer Hₐ gives an analytic Burnside count.',
      latex: String.raw`\rho_a = |I_a / H_a|, \quad H_a = \mathrm{Stab}_{G_a}(V_a)`,
    },
  },
  E: {
    label: CASE_META.E.label,
    shortLabel: 'E',
    tooltip: {
      title: CASE_META.E.label,
      body: 'Cross-boundary generators but not the full symmetric group. Must enumerate orbits.',
      latex: String.raw`\rho_a = \sum_{O \in I_a/G_a} |\pi_{V_a}(O)|`,
    },
  },
};

export function getCasePresentation(caseType) {
  const key = String(caseType ?? '?');
  return CASE_PRESENTATION[key] ?? {
    label: `Case ${key}`,
    shortLabel: key,
    tooltip: null,
  };
}
