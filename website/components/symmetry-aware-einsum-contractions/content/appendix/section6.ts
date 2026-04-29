import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const section6 = {
  title: 'Partition-counting theorem for branching projections',
  deck: 'When projection branches, equality patterns give an exact count without enumerating every tuple.',
  slots: {
    intro: [
      p('The main page\'s hard case is a mixed component where some detected symmetry moves visible labels into summed labels or summed labels into visible labels. Then projection from product orbits to stored output representatives is a relation, not a function.'),
      p('Partition counting computes this relation exactly by grouping full assignments by typed equality pattern. A typed equality pattern records which same-domain label positions are equal. Blocks from different domains cannot merge.'),
      p('For a component with product-side group $G$, visible labels $V$, output action $H = \\mathrm{Stab}_G(V)|_V$, and typed equality patterns $\\tilde{x}$, the accumulation count is'),
      p('$\\alpha = \\sum_{\\tilde{x}\\in P_{\\mathrm{typed}}(L)/G} \\frac{\\prod_s (n_s)_{b_s(\\tilde{x})}}{|\\overline{G}_{\\tilde{x}}|}\\,|A_{\\tilde{x}}/H|.$'),
      p('$P_{\\mathrm{typed}}(L)$ is the set of equality patterns that only merge same-domain labels. $b_s(\\tilde{x})$ is the number of blocks of domain class $s$. $(n_s)_{b_s}$ is the falling factorial, counting injective labelings for those blocks. $\\overline{G}_{\\tilde{x}}$ is the induced action of the stabilizer of $\\tilde{x}$ on its blocks — the image on blocks, not the raw stabilizer order. $A_{\\tilde{x}}$ is the set of maps from output positions to input equality blocks obtained by moving through the product orbit, and $|A_{\\tilde{x}}/H|$ counts the stored output representatives reached by any product orbit above that pattern.'),
    ],
    footer: [
      p('The formula computes the same $\\alpha$ as corrected brute-force orbit enumeration. In the current engine policy, the typed partition counter runs whenever its equality-pattern budget passes; otherwise the engine falls back to corrected brute force when the tuple-enumeration budget still passes.'),
      p('Algebraic restructuring such as factoring $R = v v^\\top$, contraction re-ordering, and memory-traffic optimization sit outside this model and require their own cost definitions.'),
    ],
  },
} satisfies SectionCopy;

export default section6;
