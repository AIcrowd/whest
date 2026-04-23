import type { SectionCopy } from '../schema';

const p = (text: string) => ({ kind: 'paragraph', text } as const);
const l = (text: string) => ({ kind: 'label', text } as const);

const section2 = {
  title: 'Summation creates a second kind of symmetry',
  deck: 'After the summed labels have been bound by summation, same-domain dummy variables may be renamed without changing the completed expression.',
  slots: {
    intro: [
      p('The pointwise group is not the only symmetry visible after the expression has been fully assembled. Labels in $W$ are bound summation variables. After the sum has been taken, renaming dummy variables within the same domain changes notation but not the value of the completed expression.'),
      p('When all summed labels share one domain, this dummy-renaming factor is $S(W)$. With heterogeneous label sizes, it is instead $\\prod_d S(W_d)$, where the $W_d$ are same-domain blocks of summed labels. A permutation that maps a size-2 label to a size-3 label is not a valid action on the assignment grid.'),
      p('These dummy renamings are formal symmetries of the completed expression. They are not, in general, pointwise symmetries of the summands.'),
    ],
    takeaway: [
      p('$\\prod_d S(W_d)$ is a post-summation symmetry. It preserves the completed sum, not necessarily the individual indexed products used to compute that sum.'),
    ],
    runningExampleLabelPrefix: [
      l('Running example —'),
    ],
    runningExamplePresetLabel: [
      l('bilinear trace'),
    ],
    runningExampleLead: [
      p('For the bilinear trace preset, the expression has output labels $V = \\{i,j\\}$ and summed labels $W = \\{k,l\\}$. At equal sizes, the swap $(k\\;l)$ is a valid dummy renaming.'),
      p('Under that swap, the summand $A[i,k]A[j,l]$ is carried to $A[i,l]A[j,k]$. The completed double sum is unchanged because $k$ and $l$ are bound variables and can be renamed back inside the summation.'),
      p('That is why dummy renaming is a symmetry of the finished expression even when it does not guarantee pointwise equality term by term.'),
    ],
  },
} satisfies SectionCopy;

export default section2;
