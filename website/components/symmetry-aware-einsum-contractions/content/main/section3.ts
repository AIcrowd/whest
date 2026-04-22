import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section3 = {
  title: 'Certify the Pointwise Symmetry Group',
  deck: 'Which candidate relabelings preserve the summand itself?',
  slots: {
    intro: [
      p('Candidate row moves come from the wreath product of declared axis symmetries and permutations of identical operand occurrences. If a repeated-operand family has $m_i$ copies and declared internal group $H_i$, its row candidates contribute $H_i \\wr S_{m_i}$; the product over families gives the candidate space $G_{\\mathrm{wreath}}$ that the σ-loop enumerates.'),
      p('For each row move $\\sigma_{\\mathrm{row\\;move}}$, the explorer asks whether some label relabeling $\\pi_{\\mathrm{relabeling}}$ restores the incidence pattern. An accepted pair $(\\sigma, \\pi)$ is the lifted witness used by this model for the detected pointwise action under the declared equality symmetries. The accepted relabelings are then closed under composition to obtain the detected pointwise group $G_{\\mathrm{pt}}$ used by the cost model.'),
    ],
    produces: [
      p('A detected pointwise group $G_{\\mathrm{pt}}$, represented by accepted $(\\sigma, \\pi)$ pairs and the generated label action.'),
    ],
  },
} satisfies SectionCopy;

export default section3;
