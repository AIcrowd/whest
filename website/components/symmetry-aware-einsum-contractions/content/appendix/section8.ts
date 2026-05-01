import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

// V3.1 Appendix E — Scope, assumptions, and non-goals.
//
// This section makes the explorer's contract explicit: what the reported
// number is (a direct indexed scalar-event count Total = μ + α), what the
// model assumes, what it deliberately does NOT include, and what
// "exactness" means under those constraints.
//
// Slot layout mirrors the V3.1 narrative's E.1–E.4 sub-sections, with
// label-style kickers on the two list-bearing slots (E.2, E.3) so the
// modal can render them as headed lists, matching how section7 uses
// case<k>Label kickers ahead of body slots.
const section8 = {
  title: 'Scope, assumptions, and non-goals',
  deck: 'What the reported number is, what the model assumes, what it does not include, and what "exactness" means under those constraints. The central claim: $\\text{exact direct cost} = \\mu + \\alpha$, where $\\mu$ counts product-representative multiplication chains and $\\alpha$ counts projection-induced accumulation updates.',
  slots: {
    intro: [
      p('Every number this explorer reports lives inside a precise model. This appendix states that model: the cost being counted, the assumptions the count relies on, the phenomena the count deliberately excludes, and the exactness contract that ties them together. Read this when a reported figure surprises you — the surprise is almost always a scope question, not a counting bug.'),
    ],
    costModelLabel: [
      l('E.1 — Cost model'),
    ],
    costModel: [
      p('The reported quantity is a direct indexed scalar-event count: $\\text{Total} = \\mu + \\alpha$. Here $\\mu = (k-1)\\,M$ counts multiplication-chain events for representative products, and $\\alpha$ counts accumulation updates into stored output representatives.'),
    ],
    includedAssumptionsLabel: [
      l('E.2 — Included assumptions'),
    ],
    includedAssumptionsLead: [
      p('The model assumes:'),
    ],
    includedAssumptions: [
      p('exact commutative scalar arithmetic;'),
      p('declared equality symmetries only;'),
      p('repeated operand names denote the same tensor object;'),
      p('explicit-index einsum syntax with explicit outputs;'),
      p('compatible label domains under relabeling;'),
      p('output storage by the inherited output action $H$;'),
      p('exact counting or explicit unavailability when exact counting exceeds the interactive budget.'),
    ],
    excludedPhenomenaLabel: [
      l('E.3 — Excluded phenomena'),
    ],
    excludedPhenomenaLead: [
      p('The model does NOT include:'),
    ],
    excludedPhenomena: [
      p('wall-clock runtime;'),
      p('memory traffic;'),
      p('BLAS/kernel selection;'),
      p('contraction-path optimization;'),
      p('sparsity;'),
      p('antisymmetry or sign representations;'),
      p('conjugation;'),
      p('approximate numerical symmetry;'),
      p('algebraic refactoring;'),
      p('hardware scheduling;'),
      p('general library FLOP accounting outside this direct evaluator.'),
    ],
    exactnessContractLabel: [
      l('E.4 — Exactness contract'),
    ],
    exactnessContract: [
      p('Under the stated direct orbit-stored evaluator model, the count is exact. Outside that model, it should be read as a structural direct-event count, not as a claim about fastest possible execution.'),
      p('The central claim is'),
      p('$\\text{exact direct cost} = \\text{product-representative multiplication chains} + \\text{projection-induced accumulation updates}.$'),
      p('The first term is an orbit count. The second term is an incidence count.'),
    ],
  },
} satisfies SectionCopy;

export default section8;
