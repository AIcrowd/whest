import type { SectionCopy } from '../schema.ts';

const p = (text: string) => ({ kind: 'paragraph', text } as const);

const preamble = {
  title: 'What this explorer is counting',
  deck: 'The direct indexed computation, the representative products, and the output-bin updates it induces.',
  slots: {
    einsumIntroBeforeSummed: [
      p('Every index label that appears on an input but not on the output is '),
    ],
    einsumIntroBetweenSummedAndFree: [
      p('; labels on the output are '),
    ],
    einsumIntroAfterFree: [
      p('. A dense implementation pays for every cell of the full input grid — so even modestly sized examples explode quickly.'),
    ],
    mentalFrameworkIntroBeforeRepSet: [
      p('Every contraction has the same shape. Symmetry only changes the content of three things: '),
    ],
    mentalFrameworkIntroBetweenRepSetAndOuts: [
      p(', '),
    ],
    mentalFrameworkIntroBetweenOutsAndCoeff: [
      p(', and '),
    ],
    mentalFrameworkIntroAfterCoeff: [
      p('. The rest of the explorer is about counting them without enumerating the full grid.'),
    ],
    calloutBodyBeforeGroup: [
      p('If several operands are identical or individually symmetric (e.g. $A_{ij} = A_{ji}$), the formula is invariant under certain permutations of the labels. Those permutations form a '),
    ],
    calloutBodyBetweenGroupAndOrbits: [
      p(', and whole '),
    ],
    calloutBodyAfterOrbits: [
      p(' of products collapse to a single distinct computation — the dense $n^{5}$ drops to $n^{5}/|G|$ in the best case (a free action), and to a Burnside count $(1/|G|)\\sum_g |\\mathrm{Fix}(g)|$ in general.'),
    ],
    calloutFooter: [
      p('The explorer finds $G$ automatically, then counts the distinct products ($\\mu$) and distinct output-bin updates ($\\alpha$) — the two numbers driving the code on the right.'),
    ],
    handoffBeforeSectionLink: [
      p('The rest of this page shows how the explorer detects the symmetry group $G$ from a contraction and computes $\\mu$ and $\\alpha$ automatically. Start with '),
    ],
    handoffAfterSectionLink: [
      p(' below to pick or build a contraction.'),
    ],
  },
} satisfies SectionCopy;

export default preamble;
