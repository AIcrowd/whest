import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const c = (text: string) => ({ kind: 'caption', text } as const);

const section6 = {
  title: 'The valid leftover optimization is output storage',
  deck: 'After accumulation has been decided, $G_{\\mathrm{out}}$ can still reduce mirrored output cells. Dummy-label renamings cannot.',
  slots: {
    intro: [
      p('The failure of $G_{\\text{f}}$ as an accumulation group does not mean the additional structure is useless. It means the optimization must be placed on the correct axis.'),
      p('Direct accumulation is governed by $G_{\\text{pt}}$. Output storage is governed by $G_{\\mathrm{out}}$. Once an output value has been computed, all output cells in the same $G_{\\mathrm{out}}$-orbit contain the same value. A storage layout may therefore keep one representative per output orbit and recover mirrored entries by symmetry.'),
      p('The dummy-label factor contributes nothing to output storage, because dummy labels do not index output cells. They have already been summed away.'),
    ],
    footnote: [
      p('Rows are computed from the current preset definitions at display size $n = 3$, including any preset-specific label-size overrides. Savings here are storage-only savings after accumulation has already been accounted for.'),
    ],
    tableNote: [
      c('The table above records storage-only savings available for the presets at display size n = 3. The column $\\alpha_{\\text{engine}}$ is the direct accumulation count used by the main page. The column $\\alpha_{\\text{storage}}$ is the number of update representatives after grouping touched output cells into $G_{\\mathrm{out}}$-orbits. These are different quantities; $\\alpha_{\\text{storage}}$ is not a replacement for the accumulation cost.'),
    ],
    scopeLabel: [
      c('SCOPE'),
    ],
    footer: [
      p('The $\\alpha$ shown on the main page counts direct output-bin updates in the enumerate-and-accumulate evaluation model with $G_{\\text{pt}}$ as the summand-value equivalence relation.'),
      p('Output-tensor storage collapse, algebraic restructuring such as factoring $R = v v^\\top$, contraction re-ordering, and memory-traffic optimization all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page.'),
    ],
  },
} satisfies SectionCopy;

export default section6;
