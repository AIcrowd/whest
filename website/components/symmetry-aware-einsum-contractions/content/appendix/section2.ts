import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section2 = {
  title: 'Summation creates a second kind of symmetry',
  deck: 'Once the $W_{\\mathrm{summed}}$-labels have been summed over, they become dummy variables. Renaming them preserves the completed expression, even when it does not preserve individual summands.',
  slots: {
    intro: [
      p('The pointwise group is not the only symmetry visible after the expression has been fully assembled. The labels in $W_{\\mathrm{summed}}$ are bound summation variables. After the sum has been taken, a permutation of $W_{\\mathrm{summed}}$ is an α-renaming of dummy variables: it changes the names used inside the summation, but not the value of the completed expression.'),
      p('This gives a second group, $S(W_{\\mathrm{summed}})$, the symmetric group on the summed labels. Its elements are formal symmetries of the completed expression. They are not, in general, pointwise symmetries of the summands.'),
    ],
    takeaway: [
      p('$S(W_{\\mathrm{summed}})$ is a post-summation symmetry. It preserves the completed sum, not necessarily the individual indexed products used to compute that sum.'),
    ],
    runningExampleLabelPrefix: [
      l('Running example —'),
    ],
    runningExamplePresetLabel: [
      l('bilinear trace'),
    ],
    runningExampleLead: [
      p('For the bilinear trace preset, the expression has output labels $V_{\\mathrm{free}} = \\{i,j\\}$ and summed labels $W_{\\mathrm{summed}} = \\{k,l\\}$.'),
      p('The swap $(k\\;l)$ preserves the double sum as a formal expression. Under that swap, the summand $A[i,k]A[j,l]$ is carried to $A[i,l]A[j,k]$, so the transformed expression can be written as'),
      p('Rename dummy variables back by swapping $k$ and $l$ inside the summation indices, and the completed double sum is unchanged. That is why $S(W_{\\mathrm{summed}})$ is a symmetry of the finished expression even though it does not guarantee pointwise equality term by term.'),
    ],
  },
} satisfies SectionCopy;

export default section2;
