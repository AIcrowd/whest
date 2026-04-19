import { useEffect, useMemo } from 'react';
import Latex from './Latex.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';

/**
 * Appendix-style modal that surfaces the expression-level counting symmetry
 * (G_expr = V_sub × S(W)) without cluttering the main article flow. Opens
 * from a button in Section 5 (Price Savings).
 *
 * Four panels:
 *   1. Intuition — per-tuple vs expression-level (Frobenius contrast).
 *   2. V_sub × S(W) construction — the existing interactive widget.
 *   3. α under G_expr — the "naive Burnside would say..." contrast.
 *   4. Leftover savings — V_sub output-tensor symmetry the engine's α does
 *      NOT capture (an honest disclosure so cost numbers are not misread
 *      as an absolute lower bound on compute).
 *
 * Props:
 *   isOpen   boolean
 *   onClose  () -> void
 *   analysis the full analyzeExample(...) result (may be null)
 *   group    symmetry/group output of buildGroup (may be null)
 */
export default function ExpressionLevelModal({ isOpen, onClose, analysis, group }) {
  const vLabels = group?.vLabels ?? [];
  const wLabels = group?.wLabels ?? [];
  const expressionGroup = analysis?.expressionGroup ?? null;
  const exprAlpha = useMemo(
    () => computeExpressionAlphaTotal({ analysis }),
    [analysis],
  );

  // Esc closes the modal.
  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

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
        <div className="flex items-start justify-between border-b border-gray-200 px-6 py-4">
          <div>
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
              The expression-level group is larger than the per-tuple group used for compression.
              Here's why, how it's built, and what it does <em>not</em> give us for free.
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

        <div className="space-y-8 px-6 py-6">
          {/* Panel 1: Intuition — Frobenius contrast */}
          <section>
            <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
              1. Intuition: per-tuple vs expression-level
            </h3>
            <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground space-y-3">
              <p>
                Consider the Frobenius trace <Latex math="\mathtt{ij{,}ij \to}" /> (i.e.{' '}
                <Latex math="\sum_{i,j} A_{ij}^2" />). Swapping labels{' '}
                <Latex math="i \leftrightarrow j" /> gives <Latex math="\mathtt{ji{,}ji \to}" />{' '}
                which equals <Latex math="\sum_{i,j} A_{ji}^2" />.
              </p>
              <p>
                These two expressions compute the same total sum by commutativity of addition.
                That makes <Latex math="(i\;j)" /> an{' '}
                <strong>expression-level symmetry</strong> — the sum is unchanged.
              </p>
              <p>
                But look at individual terms: the summand at <Latex math="(i,j) = (0,1)" /> is{' '}
                <Latex math="A_{01}^2" />, while after swapping it maps to <Latex math="A_{10}^2" />.
                For a general matrix these differ (<Latex math="A_{01} \neq A_{10}" />), so the term
                is <em>reshuffled</em>, not preserved. This is <em>not</em> a{' '}
                <strong>per-tuple symmetry</strong>.
              </p>

              <div className="overflow-x-auto mt-2">
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
                    <tr className="border-b border-border/40">
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

              <p className="text-muted-foreground text-[13px]">
                <strong>Why it matters:</strong> Compression needs the <em>stronger</em> notion
                (per-tuple). Using expression-level symmetry instead would give a wrong number for
                any non-symmetric operand. <Latex math="G_{\text{pt}}" /> drives cost formulas;{' '}
                <Latex math="G_{\text{expr}}" /> tells the counting story.
              </p>
            </div>
          </section>

          {/* Panel 2: V_sub × S(W) construction */}
          <section>
            <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
              2. Construction: <Latex math="V_{\text{sub}} \times S(W)" />
            </h3>
            <p className="mb-3 text-sm leading-7 text-foreground">
              <Latex math="G_{\text{expr}}" /> is built directly from{' '}
              <Latex math="G_{\text{pt}}" />: take the V-restriction{' '}
              <Latex math="V_{\text{sub}}" /> and pair it with every permutation of the summed
              labels <Latex math="S(W)" />. No Dimino needed. Each product row on the right is{' '}
              <span className="font-mono text-[12px]">V-sub row <em>i</em> × S(W) row <em>j</em></span>.
            </p>
            <VSubSwConstruction
              expressionGroup={expressionGroup}
              vLabels={vLabels}
              wLabels={wLabels}
            />
          </section>

          {/* Panel 3: α under G_expr (the naive-Burnside contrast) */}
          <section>
            <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
              3. What α would be under <Latex math="G_{\text{expr}}" />
            </h3>
            {exprAlpha !== null ? (
              <div className="rounded border-l-2 border-amber-500 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                <p>
                  Naive Burnside under the expression-level group would claim{' '}
                  <Latex math="\alpha" /> = <strong>{exprAlpha}</strong>.
                </p>
                <p className="mt-2 text-xs text-amber-800">
                  This <strong>over-compresses</strong> because dummy-rename orbits contain tuples
                  with different summand values. Reporting this as the real accumulation count
                  would give numerically wrong output — not just a different estimate. The engine
                  uses <Latex math="G_{\text{pt}}" /> and reports the correct, honest α in the
                  cost card.
                </p>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                <Latex math="G_{\text{expr}}" /> coincides with <Latex math="G_{\text{pt}}" /> for
                this contraction (no dummy-rename or V-only factor left over), so there is no
                naive-Burnside contrast to show.
              </p>
            )}
          </section>

          {/* Panel 4: Leftover savings (honest disclosure) */}
          <section>
            <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
              4. What this engine does <em>not</em> claim
            </h3>
            <div className="rounded-md border border-gray-200 bg-gray-50 px-5 py-4 text-sm leading-7 text-gray-800 space-y-3">
              <p>
                For every <Latex math="\sigma \in V_{\text{sub}}" /> we have{' '}
                <Latex math="R[\sigma\,\omega] = R[\omega]" /> — the output tensor has a real
                symmetry. In principle this lets you compute one representative bin of each{' '}
                <Latex math="V_{\text{sub}}" />-orbit and <strong>copy</strong> the rest, instead
                of re-accumulating into every mirrored bin.
              </p>
              <p>
                Our reported <Latex math="\alpha" /> counts accumulation operations under the{' '}
                <em>enumerate-and-accumulate</em> evaluation model. It does <em>not</em> fold in
                that post-accumulation cell-copy optimization. The op gap is, per{' '}
                <Latex math="V_{\text{sub}}" />-orbit of size{' '}
                <Latex math="s" />:{' '}
                <Latex math="(s-1)\cdot(\text{accumulations per bin}) - (s-1)\,\text{copies}" />.
              </p>
              <p>
                <strong>Worked example.</strong> Bilinear trace at <Latex math="n=2" /> gives
                engine <Latex math="\alpha = 14" />. With{' '}
                <Latex math="V_{\text{sub}}" />-level copy, the{' '}
                <Latex math="\{(0,1),(1,0)\}" /> orbit needs only 4 accumulations + 1 copy
                (vs 8 accumulations), bringing total ops to 11. A real gap of 3 ops per invocation.
              </p>
              <p>
                <strong>Why we stop at <Latex math="G_{\text{pt}}" />.</strong> The engine's scope
                is per-summand orbit compression within a fixed evaluation order. Output-tensor
                symmetry (this panel), algebraic factoring (
                <Latex math="R = v\,v^\top" />
                -style), and contraction re-ordering (opt_einsum) are orthogonal optimizations
                that require different algorithms; folding them into <Latex math="\alpha" /> here
                would mix op types and obscure what each optimization buys you.
              </p>
              <p className="text-[13px] text-gray-600">
                When you present these numbers: scope <Latex math="\alpha" /> as
                &ldquo;distinct accumulation operations under enumerate-and-accumulate, exploiting
                per-tuple symmetries,&rdquo; and call out V-sub output-tensor copy as a separate
                downstream optimization.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
