import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

// V3.1 Appendix E — Scope, assumptions, and non-goals.
//
// Compressed to four continuous-prose sentences (one per E.1–E.4 slot).
// The modal renders all four sentences inline as a single paragraph;
// the run-in `*Label` slots are gone because the new layout has no
// run-in headings. The `intro` slot stays (sr-only in the modal) so
// orienting copy is available to assistive tech and the slot-shape
// test contract is preserved.
const section8 = {
  title: 'Scope, assumptions, and non-goals',
  deck: 'What the reported number is, what the model assumes, what it does not include, and what "exactness" means under those constraints. The central claim: $\\text{exact direct cost} = \\mu + \\alpha$, where $\\mu$ counts product-representative multiplication chains and $\\alpha$ counts projection-induced accumulation updates.',
  slots: {
    intro: [
      p('Every reported number lives inside a model; this section states its scope and assumptions.'),
    ],
    costModel: [
      p('The reported quantity is a direct indexed scalar-event count $\\text{Total} = \\mu + \\alpha$, where $\\mu$ counts multiplication-chain events for representative products and $\\alpha$ counts accumulation updates into stored output representatives.'),
    ],
    includedAssumptions: [
      p('The model assumes exact commutative scalar arithmetic, declared equality symmetries only, repeated operand names denoting the same tensor object, and explicit-index einsum syntax with compatible label domains under relabeling.'),
    ],
    excludedPhenomena: [
      p('It does not model wall-clock runtime, memory traffic, BLAS/kernel selection, contraction-path optimization, sparsity, antisymmetry, conjugation, or approximate numerical symmetry.'),
    ],
    exactnessContract: [
      p('Under these assumptions the count is the exact direct cost; outside them, the number is a structural direct-event count, not a claim about fastest possible execution.'),
    ],
  },
} satisfies SectionCopy;

export default section8;
