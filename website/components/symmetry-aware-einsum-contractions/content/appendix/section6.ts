import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const c = (text: string) => ({ kind: 'caption', text } as const);

const section6 = {
  title: 'What $G_{\\mathrm{out}}$ buys you in the current evaluator',
  deck: 'For the direct dense-output evaluator counted on the main page, $G_{\\mathrm{out}}$ can collapse storage but does not change $\\alpha$. A stronger symmetry-aware accumulation model would be a different evaluator.',
  slots: {
    intro: [
      p('The failure of $G_{\\text{f}}$ as an accumulation group does not make the visible output symmetry useless. It means the optimization has to be attached to the right evaluator model.'),
      p('**Model 1 — current page.** The main page counts direct accumulation into explicit dense output bins. In that evaluator, $\\alpha$ is governed by $G_{\\text{pt}}$: each representative product still updates every explicit output cell in its visible projection.'),
      p('**Model 2 — symmetry-aware storage only.** After those updates have been decided, $G_{\\mathrm{out}}$ can collapse mirrored output cells to one representative per output orbit. This changes how the finished output tensor is stored, but it does not by itself change the stream of direct updates.'),
      p('**Model 3 — symmetry-aware storage plus symmetry-aware accumulation.** One could instead design a different evaluator that accumulates directly into $G_{\\mathrm{out}}$-orbit representatives, with the required multiplicity or coefficient aggregation. In that stronger model, output symmetry could reduce update events as well. That is not the evaluator counted on the main page.'),
      p('The dummy-label factor contributes nothing to output storage, because dummy labels do not index output cells. They have already been summed away.'),
    ],
    footnote: [
      p('Rows are computed from the current preset definitions at display size $n = 3$, including any preset-specific label-size overrides. Savings here are storage-only savings after the current accumulation model has already been accounted for.'),
    ],
    tableNote: [
      c('The table above records storage-side quantities for the presets at display size n = 3. The column $\\alpha_{\\text{engine}}$ is the direct accumulation count used by the main page. The column $\\alpha_{\\text{storage}}$ is the number of output-orbit representatives touched if one keeps one stored representative per $G_{\\mathrm{out}}$-orbit after those updates have already been decided. It is a storage quantity in the current evaluator, not a replacement for the accumulation cost and not the update count of a stronger symmetry-aware accumulation model.'),
    ],
    scopeLabel: [
      c('SCOPE'),
    ],
    footer: [
      p('The $\\alpha$ shown on the main page counts direct output-bin updates in the enumerate-and-accumulate evaluator with $G_{\\text{pt}}$ as the summand-value equivalence relation.'),
      p('Symmetry-aware storage, algebraic restructuring such as factoring $R = v v^\\top$, contraction re-ordering, memory-traffic optimization, and any evaluator with symmetry-aware storage plus symmetry-aware accumulation all sit outside that model and require their own cost definitions.'),
    ],
  },
} satisfies SectionCopy;

export default section6;
