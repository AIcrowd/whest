const p = (text) => ({ kind: 'paragraph', text });

const partitionCounting = {
  title: 'Typed Partition Counting',
  deck: 'Compressing the accumulation count $\\alpha$ by typed equality patterns.',
  slots: {
    intro: [
      p('When projection branches generically, the $O \\to Q$ matrix is exact but expensive to build. Typed partition counting compresses the same count by grouping assignments according to their equality pattern — a partition of label positions where each block shares a coordinate value. For each typed equality pattern, we count concrete labelings, quotient by induced block symmetry, count stored output representatives reached after projection, and add the contribution to $\\alpha$.'),
      p('To see why this matters, consider Cross C3: $\\mathrm{einsum}(\\texttt{\'abc->ab\'}, T)$ with $T$ declared cyclic on $(a,b,c)$. At $n=3$, the pattern $ab|c$ (where $a$ and $b$ share one coordinate value and $c$ differs) has $3 \\cdot 2 = 6$ concrete labelings and output reach $3$, giving contribution $6 \\cdot 3 = 18$. This 18 is one pattern-orbit contribution, not the full $\\alpha$ for Cross C3 at $n=3$; the full count also includes the all-equal pattern and the all-distinct pattern, for $\\alpha = 27 = 3 + 18 + 6$. The point is that one pattern chip represents a whole family of filled $O \\to Q$ cells without enumerating every tuple one by one.'),
    ],
    produces: [
      p('A live decomposition of $\\alpha$ by typed equality pattern, with the engine\'s per-pattern coefficients, block structure, and projection reach. Here $\\bar{G}_{\\tilde{p}}$ is the induced image of the stabilizer of the pattern acting on equality blocks — not the raw stabilizer size.'),
    ],
  },
};

export default partitionCounting;
