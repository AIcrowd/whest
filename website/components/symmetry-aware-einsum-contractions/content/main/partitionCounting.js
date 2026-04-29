const p = (text) => ({ kind: 'paragraph', text });

const partitionCounting = {
  title: 'Partition Counting',
  deck: 'A different way to count $\\alpha$ — by typed equality patterns.',
  slots: {
    intro: [
      p('The accumulation count $\\alpha$ is an edge count: pairs $(O, Q)$ where one product orbit $O$ projects to one stored output representative $Q$. Section 4 enumerated those edges directly. Partition counting reframes the same $\\alpha$ by grouping assignments into typed equality patterns — partitions of label positions where blocks share a coordinate value — and counting how many product orbits sit above each pattern, then how many output representatives each pattern can reach.'),
      p('For uniform-size labels this reduces to ordinary set partitions over the label positions. The implementation uses the typed variant so heterogeneous label sizes work cleanly: a partition is valid only when every block stays within a single domain class. The count factors neatly across blocks; the engine sums the per-pattern contributions to recover $\\alpha$ exactly. The demo below decomposes the live preset by typed equality pattern and shows each pattern\'s contribution.'),
    ],
    produces: [
      p('A live decomposition of $\\alpha$ by typed equality pattern, with the engine\'s per-pattern coefficients, block structure, and projection reach.'),
    ],
  },
};

export default partitionCounting;
