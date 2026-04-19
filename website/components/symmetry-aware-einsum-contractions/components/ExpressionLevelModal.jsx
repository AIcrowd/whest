import { useEffect, useMemo } from 'react';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';

/**
 * Appendix-style modal that walks the reader through the expression-level
 * counting symmetry: define the two building blocks (V_sub, S(W)), assemble
 * them into G_expr, show why we don't use G_expr for compression, and
 * disclose the savings we DON'T claim.
 *
 * Narrative shape mirrors the main page (Interpretation → Approach →
 * worked widget → "What this produces" callouts).
 *
 * Props:
 *   isOpen
 *   onClose
 *   analysis  full analyzeExample(...) result (may be null)
 *   group     symmetry output of buildGroup (may be null)
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
              Counting symmetry:{' '}
              <span className="font-mono">
                G<sub>expr</sub>
              </span>{' '}
              ={' '}
              <span className="font-mono">
                V<sub>sub</sub>
              </span>{' '}
              × <span className="font-mono">S(W)</span>
            </h2>
            <p className="mt-1 text-sm leading-6 text-gray-600">
              <InlineMathText>
                The page's main flow computes $G_{'\\text{pt}'}$ — the per-tuple group — and uses it
                to drive every cost number. There's a <em>larger</em> counting symmetry
                ($G_{'\\text{expr}'}$) that the sum admits but that we deliberately do not use for
                compression. This appendix defines its two building blocks, shows how we assemble
                them, and is honest about the op savings it still leaves on the table.
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

        {/* Body — 5 numbered sections, narrative style matching the main page */}
        <div className="space-y-10 px-6 py-6">
          {/* §1 — The distinction */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§1 · The distinction</div>
              <h3 className={sectionTitle}>Per-tuple vs expression-level</h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Interpretation">
                A label permutation π can be a "symmetry" in two different senses. The
                <strong> per-tuple</strong> sense preserves each individual summand value; the
                <strong> expression-level</strong> sense only preserves the total sum.
                Compression needs the stronger per-tuple notion; expression-level alone is a
                counting fact about the sum.
              </NarrativeCallout>
              <NarrativeCallout label="Working example" tone="algorithm">
                {`Take the Frobenius trace $\\mathtt{ij{,}ij\\to}$ — that is $\\sum_{i,j} A_{ij}^2$. Swapping $(i\\;j)$ reshuffles $A_{01}^2 \\leftrightarrow A_{10}^2$: totals agree by commutativity, but individual terms differ for a non-symmetric $A$.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 overflow-x-auto rounded-md border border-border/60 bg-muted/20 px-5 py-4">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border/60">
                    <th className="text-left px-3 py-2 font-semibold text-muted-foreground text-[11px] uppercase tracking-wide">
                      Permutation
                    </th>
                    <th className="text-left px-3 py-2 font-semibold text-muted-foreground text-[11px] uppercase tracking-wide">
                      Per-tuple (each summand)
                    </th>
                    <th className="text-left px-3 py-2 font-semibold text-muted-foreground text-[11px] uppercase tracking-wide">
                      Expression-level (total sum)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="px-3 py-2 font-mono">
                      <Latex math="(i\;j)" />
                    </td>
                    <td className="px-3 py-2 text-red-600">
                      Fails &mdash; <Latex math="A_{01}^2 \neq A_{10}^2" /> in general
                    </td>
                    <td className="px-3 py-2 text-emerald-700">
                      Passes &mdash; same total by commutativity
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="Takeaway" tone="accent">
                {`$G_{\\text{pt}}$ powers every cost formula on this page. $G_{\\text{expr}}$ is strictly larger on most einsums; the rest of this appendix defines the two pieces it adds and shows why adding them to compression would be wrong.`}
              </NarrativeCallout>
            </div>
          </section>

          {/* §2 — V_sub */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§2 · Building block 1</div>
              <h3 className={sectionTitle}>
                <Latex math="V_{\text{sub}}" /> — the V-restriction of{' '}
                <Latex math="G_{\text{pt}}" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$V_{\\text{sub}}$ is what $G_{\\text{pt}}$ looks like when we forget the W-labels and keep only its action on the output labels V. Concretely: for every V/W-preserving element $\\pi \\in G_{\\text{pt}}$, record its restriction to the V-positions; dedupe. The result is a subgroup of $\\mathrm{Sym}(V)$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why it makes sense on the output" tone="algorithm">
                {`Every $\\sigma \\in V_{\\text{sub}}$ lifts to a symmetry of the output tensor itself: $R[\\sigma\\,\\omega] = R[\\omega]$. This is a real structural property of the final tensor, not just a relabeling of the sum.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace.</p>
              <p className="mt-1">
                <InlineMathText>
                  {`$G_{\\text{pt}} = \\{e,\\;(i\\;j)(k\\;l)\\}$. V = $\\{i,j\\}$, W = $\\{k,l\\}$. Restricting the one non-identity element to V gives $(i\\;j)$. So $V_{\\text{sub}} = \\{e,\\;(i\\;j)\\} \\cong S_2$ on $\\{i,j\\}$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`Note the output really is symmetric: $R[i,j] = R[j,i]$. That's $V_{\\text{sub}}$ showing up in the output tensor.`}
                </InlineMathText>
              </p>
            </div>
          </section>

          {/* §3 — S(W) */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§3 · Building block 2</div>
              <h3 className={sectionTitle}>
                <Latex math="S(W)" /> — dummy renames of summed labels
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$S(W)$ is the full symmetric group on the summed labels W — every permutation of W, period. It has $|W|!$ elements regardless of what operands or declared symmetries the einsum has.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why it's free" tone="algorithm">
                {`W-labels are bound summation indices. Renaming them consistently across the whole expression can't change the sum: $\\sum_k f(k) = \\sum_{k'} f(k')$. That's it — no operand structure required.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Concretely, for bilinear trace.</p>
              <p className="mt-1">
                <InlineMathText>
                  {`W = $\\{k,l\\}$, so $S(W) = \\{e,\\;(k\\;l)\\}$ — the 2-element group that renames the two summed indices. $(k\\;l)$ is always an expression-level symmetry because $\\sum_{k,l} f(k,l) = \\sum_{k,l} f(l,k)$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`But $(k\\;l)$ is NOT per-tuple: the individual summand at $(k,l)=(0,1)$ becomes the one at $(1,0)$, which is a different number for a general operand. Dummy-rename works on totals, not on terms.`}
                </InlineMathText>
              </p>
            </div>
          </section>

          {/* §4 — G_expr construction */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§4 · Assemble</div>
              <h3 className={sectionTitle}>
                <Latex math="G_{\text{expr}} = V_{\text{sub}} \times S(W)" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Construction">
                {`Cartesian product. Every pair $(\\sigma_V,\\sigma_W)$ with $\\sigma_V \\in V_{\\text{sub}}$ and $\\sigma_W \\in S(W)$ lifts to a single permutation of all labels that applies $\\sigma_V$ on V-positions and $\\sigma_W$ on W-positions. Together they form $G_{\\text{expr}}$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Free to compute" tone="algorithm">
                {`No Dimino needed. $V_{\\text{sub}}$ is already materialized inside $G_{\\text{pt}}$; $S(W)$ is just $|W|!$ permutations. We enumerate the Cartesian product on the fly.`}
              </NarrativeCallout>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              Each row in the rightmost column below is{' '}
              <span className="font-mono text-[12px]">V<sub>sub</sub> row <em>i</em> × S(W) row <em>j</em></span>.
              Hover a row in any column to highlight its counterparts.
            </p>
            <div className="mt-3">
              <VSubSwConstruction
                expressionGroup={expressionGroup}
                vLabels={vLabels}
                wLabels={wLabels}
              />
            </div>
          </section>

          {/* §5 — Why we don't use it for compression */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§5 · Why we don't use it for compression</div>
              <h3 className={sectionTitle}>
                What <Latex math="\alpha" /> would be under <Latex math="G_{\text{expr}}" />
              </h3>
            </div>

            {exprAlpha !== null ? (
              <>
                <div className="rounded-md border-l-4 border-amber-500 bg-amber-50 px-5 py-4 text-sm leading-7 text-amber-900">
                  <p>
                    <InlineMathText>
                      Naive Burnside under $G_{'\\text{expr}'}$ would claim $\alpha =$
                    </InlineMathText>{' '}
                    <strong>{exprAlpha}</strong>.
                  </p>
                  <p className="mt-2 text-[13px] text-amber-800">
                    {`This is wrong — not slightly over-counted, actually wrong. Dummy-rename orbits under $S(W)$ contain tuples whose summand values differ, so "pick one representative and multiply by orbit size" produces the wrong numerical output for a non-symmetric operand. The engine therefore reports the honest $G_{\\text{pt}}$-based $\\alpha$ in the cost card.`}
                  </p>
                </div>
                <div className="mt-4">
                  <NarrativeCallout label="Why G_pt and not G_expr" tone="accent">
                    {`$G_{\\text{pt}}$ is the largest group under which every orbit's summand values really are equal — so Burnside's "one representative per orbit" is faithful. Any larger group (including $G_{\\text{expr}}$) collapses orbits that Burnside would silently assume are equal, but that aren't, giving wrong numbers.`}
                  </NarrativeCallout>
                </div>
              </>
            ) : (
              <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
                <InlineMathText>
                  $G_{'\\text{expr}'}$ coincides with $G_{'\\text{pt}'}$ for this contraction
                  (|W| ≤ 1 or no residual V-sub factor), so the two counts agree and there is no
                  contrast to show.
                </InlineMathText>
              </div>
            )}
          </section>

          {/* §6 — Honest disclosure */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§6 · What we still leave on the table</div>
              <h3 className={sectionTitle}>
                <Latex math="V_{\text{sub}}" /> gives a real output-tensor symmetry —
                and we don't fold it into <Latex math="\alpha" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="The open optimization">
                {`For every $\\sigma \\in V_{\\text{sub}}$, $R[\\sigma\\,\\omega] = R[\\omega]$. In principle that lets you compute one representative bin per $V_{\\text{sub}}$-orbit and $\\textbf{copy}$ the rest instead of re-accumulating.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why our α doesn't count it" tone="algorithm">
                {`Our $\\alpha$ is "distinct accumulation operations in the enumerate-and-accumulate evaluation, exploiting per-tuple symmetries." Post-accumulation cell-copy is a separate optimization level; conflating them would mix operation types and hide what each step buys.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked gap — bilinear trace at n = 2.</p>
              <p className="mt-1">
                Engine reports α = <strong>14</strong>. With V<sub>sub</sub>-level copy the{' '}
                <Latex math="\{(0,1),(1,0)\}" /> orbit needs only 4 accumulations + 1 copy (vs 8
                accumulations), bringing total ops to 11. The real gap is{' '}
                <span className="font-mono">3 ops</span> per invocation. More generally, per{' '}
                <Latex math="V_{\text{sub}}" />-orbit of size s:{' '}
                <Latex math="(s-1)\cdot(\text{accumulations per bin}) - (s-1)\,\text{copies}" />.
              </p>
              <p className="mt-3 text-muted-foreground text-[13px]">
                Related out-of-scope optimizations: algebraic restructuring (e.g. recognizing{' '}
                <Latex math="R = v\,v^\top" /> and computing v first), contraction re-ordering
                (opt_einsum), and mixed-precision accumulation. Each is orthogonal to the orbit
                compression this page measures.
              </p>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="For academic presentation" tone="accent">
                {`Scope $\\alpha$ as "distinct accumulation operations under enumerate-and-accumulate, exploiting per-tuple symmetries." Call out $V_{\\text{sub}}$ output-tensor copy, algebraic factoring, and contraction re-ordering as separate downstream optimizations not folded in.`}
              </NarrativeCallout>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
