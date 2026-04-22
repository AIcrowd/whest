import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section4 = {
  title: 'Count Product Orbits and Output Projections',
  deck: 'How does the group action become multiplication and accumulation cost?',
  slots: {
    intro: [
      p('The cost model acts on full label assignments, not only on output labels. For each component, the multiplication count $m_{\\mathrm{component}}$ is the number of representative product orbits under the restricted pointwise group, computed by Burnside or by exact orbit enumeration.'),
      p('Accumulation is subtler. A product orbit may project to one output bin, several output bins, or no visible labels at all. Therefore the per-component accumulation count $\\alpha_{\\mathrm{component}}$ is an orbit-projection count: one update for each distinct free-label projection touched by each product orbit. This is the step that prevents a free-label symmetry from being mistaken for an automatic reduction in output updates.'),
    ],
    produces: [
      p('Per-component quantities $m_{\\mathrm{component}}$ and $\\alpha_{\\mathrm{component}}$, with each component routed through the cheapest applicable exact formula or fallback enumeration.'),
    ],
  },
} satisfies SectionCopy;

export default section4;
