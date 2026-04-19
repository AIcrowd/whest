import { useEffect, useMemo } from 'react';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';

// V/W color palette — same hexes the rest of the page uses so the worked
// examples inside the modal read against the same colour convention.
const COLOR_V = '#4A7CFF';
const COLOR_W = '#64748B';

const vStyle = { color: COLOR_V, fontWeight: 600 };
const wStyle = { color: COLOR_W, fontWeight: 600 };

/**
 * Appendix modal that walks the reader through the expression-level
 * counting symmetry: define the two groups (G_pt, G_expr), define the two
 * building blocks of G_expr (V_sub and S(W)), assemble them, explain why
 * we do not use G_expr for compression, and disclose the output-tensor
 * savings we still leave on the table.
 */
export default function ExpressionLevelModal({ isOpen, onClose, analysis, group }) {
  const vLabels = group?.vLabels ?? [];
  const wLabels = group?.wLabels ?? [];
  const expressionGroup = analysis?.expressionGroup ?? null;
  const exprAlpha = useMemo(
    () => computeExpressionAlphaTotal({ analysis }),
    [analysis],
  );

  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sectionNumber = 'text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground';
  const sectionTitle = 'font-heading text-base font-semibold text-gray-900';

  return (
    <div
      className="fixed inset-0 z-[9998] flex items-start justify-center overflow-y-auto bg-black/40 px-4 py-10"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="relative w-full max-w-3xl rounded-lg border border-gray-200 bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="expr-modal-heading"
      >
        {/* Header */}
        <div className="flex items-start justify-between border-b border-gray-200 px-6 py-4">
          <div>
            <div className={sectionNumber}>Appendix</div>
            <h2 id="expr-modal-heading" className="font-heading text-lg font-semibold text-gray-900">
              Counting symmetry: <Latex math="G_{\text{expr}} = V_{\text{sub}} \times S(W)" />
            </h2>
            <p className="mt-1 text-sm leading-6 text-gray-600">
              <InlineMathText>
                {`The main page reports a single detected symmetry group $G$ and uses it to drive every cost number. Inside this appendix we refer to it as $G_{\\text{pt}}$ — the $\\textit{per-tuple}$ group — to distinguish it from the larger counting symmetry $G_{\\text{expr}}$ discussed here. The two sections below define both groups precisely before the rest of the appendix builds $G_{\\text{expr}}$ from its components.`}
              </InlineMathText>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ml-4 shrink-0 rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-900"
            aria-label="Close"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path
                fillRule="evenodd"
                d="M4.22 4.22a.75.75 0 011.06 0L10 8.94l4.72-4.72a.75.75 0 111.06 1.06L11.06 10l4.72 4.72a.75.75 0 11-1.06 1.06L10 11.06l-4.72 4.72a.75.75 0 01-1.06-1.06L8.94 10 4.22 5.28a.75.75 0 010-1.06z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="space-y-10 px-6 py-6">
          {/* §0 — Definitions upfront */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§0 · Terms used in this appendix</div>
              <h3 className={sectionTitle}>Compression, counting, and the two groups</h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Compression">
                {`The main page's goal. Given an einsum, report the minimum number of distinct scalar multiplications $\\mu$ and the minimum number of distinct accumulations $\\alpha$ needed to produce the output tensor, exploiting symmetry to reuse products and share work across tuples. Cost formulas like "$\\mu = |X / G|$" are compression claims.`}
              </NarrativeCallout>
              <NarrativeCallout label="Counting symmetry" tone="algorithm">
                {`A structural invariance of the $\\textit{total sum}$ — a label permutation under which the einsum $\\sum_t \\text{summand}(t)$ evaluates to the same scalar total, even though individual summands may reshuffle. Counting symmetries describe what the expression admits as a symbolic object; they do not, on their own, license arithmetic reuse.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="G_pt — the per-tuple group">
                {`The group of label permutations under which $\\text{summand}(t)$ equals $\\text{summand}(\\pi^{-1} t)$ for every tuple $t$, not just on the total sum. $G_{\\text{pt}}$ is the largest group that licenses compression in the enumerate-and-accumulate evaluation model. The engine computes it via σ-loop Sources A and B, and every reported $\\mu$ and $\\alpha$ is taken with respect to this group.`}
              </NarrativeCallout>
              <NarrativeCallout label="G_expr — the counting group">
                {`The full expression-level group: all label permutations under which $\\sum_t \\text{summand}(t)$ is invariant, including dummy relabellings of summed indices. $G_{\\text{expr}} \\supseteq G_{\\text{pt}}$. In the bilinear-trace case below, $G_{\\text{expr}}$ has four elements and $G_{\\text{pt}}$ has two.`}
              </NarrativeCallout>
            </div>
          </section>

          {/* §1 — The distinction (Frobenius worked example) */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§1 · The distinction made concrete</div>
              <h3 className={sectionTitle}>
                Frobenius trace at <Latex math="n = 2" />
              </h3>
            </div>

            <div className="mb-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Take $R = \\sum_{i,j} A[i,j] \\cdot A[i,j]$ on a $2 \\times 2$ generic matrix $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$. The four summands are:`}
              </InlineMathText>
            </div>

            <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
              <div>
                (<span style={vStyle}>i</span>=0, <span style={vStyle}>j</span>=0):{' '}
                A[<span style={vStyle}>0</span>,<span style={vStyle}>0</span>] ·
                A[<span style={vStyle}>0</span>,<span style={vStyle}>0</span>] = 1 · 1 = 1
              </div>
              <div>
                (<span style={vStyle}>i</span>=0, <span style={vStyle}>j</span>=1):{' '}
                A[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>] ·
                A[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>] = 2 · 2 = 4
              </div>
              <div>
                (<span style={vStyle}>i</span>=1, <span style={vStyle}>j</span>=0):{' '}
                A[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>] ·
                A[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>] = 3 · 3 = 9
              </div>
              <div>
                (<span style={vStyle}>i</span>=1, <span style={vStyle}>j</span>=1):{' '}
                A[<span style={vStyle}>1</span>,<span style={vStyle}>1</span>] ·
                A[<span style={vStyle}>1</span>,<span style={vStyle}>1</span>] = 4 · 4 = 16
              </div>
              <div className="mt-2 border-t border-border/60 pt-2">
                R = 1 + 4 + 9 + 16 = <strong>30</strong>
              </div>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Apply the permutation $(i\\;j)$: for each tuple $(i,j)$ replace it with $(j,i)$. The four summands become`}
              </InlineMathText>
            </p>

            <div className="mt-2 rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
              <div>
                (0, 0) → (0, 0): A[0,0]² = 1 <span className="text-muted-foreground">(unchanged)</span>
              </div>
              <div>
                (0, 1) → (1, 0): A[1,0]² = <strong className="text-red-700">9</strong>{' '}
                <span className="text-muted-foreground">(was 4)</span>
              </div>
              <div>
                (1, 0) → (0, 1): A[0,1]² = <strong className="text-red-700">4</strong>{' '}
                <span className="text-muted-foreground">(was 9)</span>
              </div>
              <div>
                (1, 1) → (1, 1): A[1,1]² = 16 <span className="text-muted-foreground">(unchanged)</span>
              </div>
              <div className="mt-2 border-t border-border/60 pt-2">
                R' = 1 + 9 + 4 + 16 = <strong>30</strong>
              </div>
            </div>

            <div className="mt-4 rounded-md border-l-4 border-amber-500 bg-amber-50 px-5 py-3 text-sm leading-7 text-amber-900">
              <InlineMathText>
                {`The totals agree: $R = R' = 30$. However, the individual summands at positions 2 and 3 have exchanged values ($4 \\leftrightarrow 9$). The permutation $(i\\;j)$ preserves the sum by reshuffling — not by leaving each term invariant. Hence $(i\\;j)$ is a counting symmetry of this expression but is not an element of $G_{\\text{pt}}$; compression that treated it as such would require $A[0,1]^2 = A[1,0]^2$, which fails for a non-symmetric $A$.`}
              </InlineMathText>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="Takeaway" tone="accent">
                {`For this einsum $G_{\\text{pt}} = \\{e\\}$ (the identity alone is per-tuple on a generic $A$), while $G_{\\text{expr}}$ contains the extra element $(i\\;j)$. The remainder of the appendix explains where that extra element comes from and why it is harmless for counting but incorrect for compression.`}
              </NarrativeCallout>
            </div>
          </section>

          {/* §2 — V_sub */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§2 · Building block 1</div>
              <h3 className={sectionTitle}>
                <Latex math="V_{\text{sub}}" /> — the V-restriction of <Latex math="G_{\text{pt}}" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$V_{\\text{sub}}$ is the image of $G_{\\text{pt}}$ under restriction to the V-labels. Concretely: for each $V/W$-preserving element $\\pi \\in G_{\\text{pt}}$, record its action on V-positions and discard its action on W-positions; deduplicate the resulting permutations. The output is a subgroup of $\\mathrm{Sym}(V)$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Interpretation on the output tensor" tone="algorithm">
                {`For every $\\sigma \\in V_{\\text{sub}}$, the output tensor satisfies $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\omega \\in [n]^V$. This is a genuine symmetry of the computed output, not a symbolic invariance of the sum.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace.</p>
              <p className="mt-1">
                <InlineMathText>
                  {`Consider $\\mathtt{ik{,}jl\\to ij}$, that is $R[i,j] = \\sum_{k,l} A[i,k] \\cdot A[j,l]$. Here $V = \\{i, j\\}$ and $W = \\{k, l\\}$. The detected per-tuple group is $G_{\\text{pt}} = \\{e,\\;(i\\;j)(k\\;l)\\}$: swapping the two identical $A$'s exchanges V-labels $i \\leftrightarrow j$ together with W-labels $k \\leftrightarrow l$. Restricting the non-identity element to V yields $(i\\;j)$, so $V_{\\text{sub}} = \\{e,\\;(i\\;j)\\} \\cong S_2$ acting on $\\{i,j\\}$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`The output tensor is genuinely symmetric under this action: $R[i,j] = R[j,i]$ for every $(i,j)$ in this einsum, because $R[i,j] = (\\sum_k A[i,k])(\\sum_l A[j,l])$ is a product of two scalars.`}
                </InlineMathText>
              </p>
            </div>
          </section>

          {/* §3 — S(W) */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§3 · Building block 2</div>
              <h3 className={sectionTitle}>
                <Latex math="S(W)" /> — dummy relabellings of summed indices
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$S(W)$ is the full symmetric group on the summed labels $W$: every permutation of $W$, of which there are $|W|!$ in total, regardless of operand structure or declared symmetries.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why every permutation of W is a counting symmetry" tone="algorithm">
                {`W-labels are bound summation indices. Relabelling them consistently across every operand occurrence yields the identity $\\sum_{k} f(k) = \\sum_{k'} f(k')$ on the total. This invariance is syntactic — it holds regardless of $f$ — and it provides no term-level identity, since the individual summand values at $k=0$ and $k=1$ need not coincide.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace (continued).</p>
              <p className="mt-1">
                <InlineMathText>
                  {`With $W = \\{k,l\\}$, $S(W) = \\{e,\\;(k\\;l)\\}$. The permutation $(k\\;l)$ is a counting symmetry because $\\sum_{k,l} A[i,k]\\,A[j,l] = \\sum_{k,l} A[i,l]\\,A[j,k]$ — the two double sums iterate over the same set of index pairs and differ only in which variable is named $k$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`However, $(k\\;l)$ is not per-tuple. Fixing $(i,j) = (0,1)$ and comparing the individual summands at $(k,l) = (0,1)$ and its image $(1,0)$ gives $A[0,0] \\cdot A[1,1]$ versus $A[0,1] \\cdot A[1,0]$. These expressions differ for a generic $A$, so applying $(k\\;l)$ to Burnside's orbit formula would produce a compression claim that is numerically incorrect.`}
                </InlineMathText>
              </p>
            </div>
          </section>

          {/* §4 — Assemble G_expr */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§4 · Assembly</div>
              <h3 className={sectionTitle}>
                <Latex math="G_{\text{expr}} = V_{\text{sub}} \times S(W)" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Construction">
                {`Each pair $(\\sigma_V,\\;\\sigma_W)$ with $\\sigma_V \\in V_{\\text{sub}}$ and $\\sigma_W \\in S(W)$ lifts to a single label permutation on $V \\cup W$ that acts as $\\sigma_V$ on V-positions and as $\\sigma_W$ on W-positions. The set of all such lifts forms $G_{\\text{expr}}$, a subgroup of $\\mathrm{Sym}(V \\cup W)$ of order $|V_{\\text{sub}}| \\cdot |W|!$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Cost of construction" tone="algorithm">
                {`No Dimino closure is required. $V_{\\text{sub}}$ is already materialized from $G_{\\text{pt}}$; $S(W)$ is an immediate $|W|!$ enumeration of permutations of the summed labels. $G_{\\text{expr}}$ is then the on-the-fly Cartesian product.`}
              </NarrativeCallout>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              The widget below enumerates these pairs for the currently selected preset. Each row in the rightmost column corresponds to the pair{' '}
              <span className="font-mono text-[12px]">V<sub>sub</sub> row i × S(W) row j</span>.
              Hovering a row in any column highlights the corresponding rows in the other two columns.
            </p>
            <div className="mt-3">
              <VSubSwConstruction
                expressionGroup={expressionGroup}
                vLabels={vLabels}
                wLabels={wLabels}
              />
            </div>
          </section>

          {/* §5 — Why not for compression */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§5 · Why G_expr is not used for compression</div>
              <h3 className={sectionTitle}>
                <Latex math="\alpha" /> under <Latex math="G_{\text{expr}}" />
              </h3>
            </div>

            {exprAlpha !== null ? (
              <>
                <div className="rounded-md border-l-4 border-amber-500 bg-amber-50 px-5 py-4 text-sm leading-7 text-amber-900">
                  <p>
                    <InlineMathText>
                      {`Applying Burnside's orbit-counting formula to $G_{\\text{expr}}$ in place of $G_{\\text{pt}}$ would yield $\\alpha =$`}
                    </InlineMathText>{' '}
                    <strong>{exprAlpha}</strong>.
                  </p>
                  <p className="mt-2 text-[13px] text-amber-800">
                    <InlineMathText>
                      {`This value is incorrect. Dummy-rename orbits under $S(W)$ contain tuples whose summand values differ, so selecting one representative per orbit and multiplying by the orbit size produces a compression claim that does not match the true output for a generic operand. The main-page cost card therefore reports $\\alpha$ with respect to $G_{\\text{pt}}$ only.`}
                    </InlineMathText>
                  </p>
                </div>
                <div className="mt-4">
                  <NarrativeCallout label="Why G_pt, and not G_expr" tone="accent">
                    {`$G_{\\text{pt}}$ is the largest group under which every orbit's summand values are equal; Burnside's "one representative per orbit" principle is faithful there. Any larger group collapses orbits whose representatives Burnside would implicitly assume to be equal but which are not — yielding numerically wrong compressed outputs.`}
                  </NarrativeCallout>
                </div>
              </>
            ) : (
              <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
                <InlineMathText>
                  {`For this einsum $G_{\\text{expr}}$ coincides with $G_{\\text{pt}}$ (either $|W| \\leq 1$ or the residual $V_{\\text{sub}}$ factor is trivial), so the two counts agree and no contrast is available to display.`}
                </InlineMathText>
              </div>
            )}
          </section>

          {/* §6 — Leftover savings */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§6 · A real opportunity this engine does not claim</div>
              <h3 className={sectionTitle}>
                Output-tensor symmetry from <Latex math="V_{\text{sub}}" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="The open optimization">
                {`For every $\\sigma \\in V_{\\text{sub}}$, the identity $R[\\sigma\\,\\omega] = R[\\omega]$ holds on the output tensor. In a computational model equipped with symmetry-aware output storage — where a single physical slot represents all cells in a $V_{\\text{sub}}$-orbit — writes to mirrored cells collapse automatically, reducing the accumulation count.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why α does not fold this in" tone="algorithm">
                {`The reported $\\alpha$ counts distinct accumulation operations in the enumerate-and-accumulate evaluation, using $G_{\\text{pt}}$ as the equivalence relation on summand values. Post-accumulation storage collapse is an independent optimization axis. Folding it into $\\alpha$ without changing the underlying computational model would conflate two distinct cost reductions and obscure the source of each.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Quantifying the gap — bilinear trace at n = 2.</p>
              <p className="mt-1">
                <InlineMathText>
                  {`The main-page cost card reports $\\alpha = 14$. Under $V_{\\text{sub}}$-aware storage, the output cells $R[0,1]$ and $R[1,0]$ occupy a single physical slot; the 4 accumulations each would have received reduce to a single stream of 4 accumulations into that slot plus no additional copy, for a total of 10 accumulations instead of 14.`}
                </InlineMathText>
              </p>
              <p className="mt-2">
                <InlineMathText>
                  {`Generalising: per $V_{\\text{sub}}$-orbit of size $s$, the savings are $(s-1) \\cdot (\\text{accumulations per bin})$ operations. This is a real opportunity, and it lives in $V_{\\text{sub}}$; the $S(W)$ factor of $G_{\\text{expr}}$ contributes nothing at the storage level because $S(W)$ acts on summation variables, not on output cells.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="Recommended framing for downstream audiences" tone="accent">
                {`Report $\\alpha$ as "distinct accumulation operations under the enumerate-and-accumulate evaluation, using $G_{\\text{pt}}$ for per-summand equivalence." State explicitly that $V_{\\text{sub}}$-level output-tensor symmetry, algebraic restructuring (e.g. $R = v\\,v^\\top$), and contraction re-ordering are complementary optimizations not folded into this figure.`}
              </NarrativeCallout>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
