import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const c = (text: string) => ({ kind: 'caption', text } as const);

const section6 = {
  title: 'The valid leftover optimization is output storage',
  deck: 'After accumulation has been decided, $G_{\\mathrm{out}}$ can still reduce mirrored output cells. The dummy-label factor $S(W_{\\mathrm{summed}})$ cannot.',
  slots: {
    intro: [
      p('The failure of $G_{\\text{f}}$ as an accumulation group does not mean the additional structure is useless. It means the optimization must be placed on the correct axis.'),
      p('Accumulation is governed by $G_{\\text{pt}}$. Output storage is governed by $G_{\\mathrm{out}}$. Once an output value has been computed, all output cells in the same $G_{\\mathrm{out}}$-orbit contain the same value. A storage layout may therefore keep one representative per output orbit and recover the mirrored entries by symmetry.'),
      p('The dummy-label group $S(W_{\\mathrm{summed}})$ contributes nothing to output storage, because its labels do not index output cells. They have already been summed away.'),
    ],
    footnote: [
      p('All rows are computed at $n = 3$. Savings here are storage-only savings after accumulation has already been accounted for.'),
    ],
    tableNote: [
      c('The table below records the storage-only savings available for the presets at $n = 3$. The column $\\alpha_{\\text{engine}}$ is the accumulation representative count used by the main page. The column $\\alpha_{\\text{storage}}$ is the number of output-storage representatives after grouping output cells into $G_{\\mathrm{out}}$-orbits. These are different quantities; $\\alpha_{\\text{storage}}$ is not a replacement for the accumulation cost.'),
    ],
    scopeLabel: [
      c('SCOPE'),
    ],
    footer: [
      p('The $\\alpha$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with $G_{\\text{pt}}$ as the summand-value equivalence relation.'),
      p('Output-tensor storage collapse, algebraic restructuring such as factoring $R = v v^\\top$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page.'),
    ],
  },
} satisfies SectionCopy;

export default section6;
