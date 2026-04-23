const p = (text) => ({ kind: 'paragraph', text });

const preamble = {
  title: 'What this explorer counts',
  deck: 'A direct computation count: representative products plus the output-bin updates they induce.',
  slots: {
    einsumIntroBeforeSummed: [
      p('Every index label that appears on an input but not on the output is '),
    ],
    einsumIntroBetweenSummedAndFree: [
      p('; labels on the output are '),
    ],
    einsumIntroAfterFree: [
      p('. A dense direct implementation visits every assignment in the full label grid before any symmetry is used.'),
    ],
    mentalFrameworkIntroBeforeRepSet: [
      p('Every direct evaluator has the same skeleton. Symmetry changes only three data structures: '),
    ],
    mentalFrameworkIntroBetweenRepSetAndOuts: [
      p(', '),
    ],
    mentalFrameworkIntroBetweenOutsAndCoeff: [
      p(', and '),
    ],
    mentalFrameworkIntroAfterCoeff: [
      p('. The rest of the explorer is about certifying these sets and counting them without enumerating the dense grid.'),
    ],
    calloutBodyBeforeGroup: [
      p('Some relabelings preserve each pre-summation product under the declared operand symmetries and repeated operand identities. Those accepted relabelings form the '),
    ],
    calloutBodyBetweenGroupAndOrbits: [
      p(', and its '),
    ],
    calloutBodyAfterOrbits: [
      p(' of full label assignments give the representative products. If the action were free, the representative-product count would shrink by $|G_{\\text{pt}}|$; in ordinary tensor-index actions, diagonal assignments create fixed points, so Burnside’s lemma gives the exact orbit count.'),
    ],
    calloutFooter: [
      p('The explorer detects this structural pointwise group, then reports $M$ (representative products), $\\mu = (k-1)M$ (multiplication-chain events), and $\\alpha$ (output-bin updates).'),
    ],
    handoffBeforeSectionLink: [
      p('The rest of this page explains which relabelings are certified, why $\\alpha$ is an orbit-projection count, and why post-summation dummy renamings belong in the appendix rather than in the direct accumulation cost. Start with '),
    ],
    handoffAfterSectionLink: [
      p(' below to pick or build a contraction.'),
    ],
  },
};

export default preamble;
