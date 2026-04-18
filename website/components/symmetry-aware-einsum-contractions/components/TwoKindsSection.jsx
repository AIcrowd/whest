import ExplorerSectionCard, { SectionEyebrow, AnchorLink } from './ExplorerSectionCard.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import SigmaLoop from './SigmaLoop.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import { EXPLORER_ACTS } from './explorerNarrative.js';

/**
 * Act 3 — Two Kinds of Symmetry.
 *
 * Three widgets:
 *   1. Intuition — per-tuple vs expression-level via Frobenius trace example.
 *   2. σ-loop visualization (relocated from Act 4/Proof).
 *   3. V-sub × S(W) construction widget.
 *
 * Props: { analysis, graph, matrixData, example, sigmaResults, variableColors,
 *          group, onSelectedPairChange, selectedSigmaPairIndex }
 */
export default function TwoKindsSection({
  analysis,
  graph,
  matrixData,
  example,
  sigmaResults,
  variableColors,
  group,
  bridge,
  onSelectedPairChange,
  selectedSigmaPairIndex,
}) {
  const act = EXPLORER_ACTS[2]; // 'two-kinds'
  const vLabels = group?.vLabels ?? [];
  const wLabels = group?.wLabels ?? [];

  return (
    <ExplorerSectionCard
      eyebrow={<SectionEyebrow n={3} anchorId={act.id} />}
      title={act.heading}
      description={act.question}
      className="border-gray-200 bg-white"
      contentClassName="pt-5"
    >
      <div className="grid gap-4 md:grid-cols-2">
        <NarrativeCallout label="Interpretation">{act.interpretation}</NarrativeCallout>
        <NarrativeCallout label="Approach" tone="algorithm">{act.algorithm}</NarrativeCallout>
      </div>

      {(bridge ?? act.bridge) && (
        <p className="mt-4 text-sm leading-7 text-foreground">
          <InlineMathText>{bridge ?? act.bridge}</InlineMathText>
        </p>
      )}

      {/* Widget 1: Intuition — Frobenius trace example */}
      <div id="two-kinds-intuition" className="mt-6 scroll-mt-24">
        <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
          <AnchorLink anchorId="two-kinds-intuition" labelText="Intuition: per-tuple vs expression-level">
            Intuition: per-tuple vs expression-level
          </AnchorLink>
        </h3>

        <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground space-y-3">
          <p>
            Consider the Frobenius trace{' '}
            <Latex math="\mathtt{ij{,}ij \to}" />{' '}
            (i.e. <Latex math="\sum_{i,j} A_{ij}^2" />).
            Swapping labels <Latex math="i \leftrightarrow j" /> gives{' '}
            <Latex math="\mathtt{ji{,}ji \to}" />{' '}
            which equals <Latex math="\sum_{i,j} A_{ji}^2" />.
          </p>
          <p>
            These two expressions compute the same total sum by commutativity of
            addition. That makes <Latex math="(i\;j)" /> an{' '}
            <strong>expression-level symmetry</strong> — the sum is unchanged.
          </p>
          <p>
            But look at individual terms: the summand at{' '}
            <Latex math="(i,j) = (0,1)" /> is <Latex math="A_{01}^2" />, while
            after swapping it maps to <Latex math="A_{10}^2" />. For a general
            matrix these differ (<Latex math="A_{01} \neq A_{10}" />), so the
            term is{' '}
            <em>reshuffled</em>, not preserved. This is <em>not</em> a{' '}
            <strong>per-tuple symmetry</strong>.
          </p>

          {/* Comparison table */}
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
                    Fails &mdash;{' '}
                    <Latex math="A_{01}^2 \neq A_{10}^2" /> in general
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
            (per-tuple). Using expression-level symmetry instead inflates the apparent
            group size and over-counts reuse opportunities.{' '}
            <Latex math="G_{\text{pt}}" /> drives cost formulas;{' '}
            <Latex math="G_{\text{expr}}" /> tells the counting story.
          </p>
        </div>
      </div>

      {/* Widget 2: σ-loop visualization (relocated from Proof) */}
      {sigmaResults && graph && matrixData && (
        <div id="two-kinds-sigma-loop" className="mt-6 scroll-mt-24">
          <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
            <AnchorLink anchorId="two-kinds-sigma-loop" labelText="σ-Loop Sources A and B">
              σ-Loop &amp; π Detection (Sources A and B)
            </AnchorLink>
          </h3>
          <p className="text-sm leading-7 text-foreground mb-4">
            The σ-loop enumerates candidate row permutations and tests whether a
            recovering label permutation π exists. Valid (σ, π) pairs become
            generators for <Latex math="G_{\text{pt}}" /> via Source A
            (declared axis symmetries) and Source B (identical-operand swaps).
          </p>
          <SigmaLoop
            results={sigmaResults}
            graph={graph}
            matrixData={matrixData}
            example={example}
            variableColors={variableColors}
            group={group}
            onSelectedPairChange={onSelectedPairChange}
          />
        </div>
      )}

      {/* Widget 3: V-sub × S(W) construction */}
      <div id="two-kinds-vsub-sw" className="mt-6 scroll-mt-24">
        <h3 className="font-heading text-base font-semibold text-gray-900 mb-3">
          <AnchorLink anchorId="two-kinds-vsub-sw" labelText="V-sub times S(W) construction">
            <Latex math="G_{\text{expr}} = V_{\text{sub}} \times S(W)" /> Construction
          </AnchorLink>
        </h3>
        <p className="text-sm leading-7 text-foreground mb-4">
          <Latex math="G_{\text{expr}}" /> is computed directly: take the V-restriction
          of <Latex math="G_{\text{pt}}" /> (<Latex math="V_{\text{sub}}" />) and pair it
          with every permutation of the summed labels (<Latex math="S(W)" />). No
          Dimino needed. Hover a row in either left column to highlight the
          corresponding product entries.
        </p>
        <VSubSwConstruction
          expressionGroup={analysis?.expressionGroup}
          vLabels={vLabels}
          wLabels={wLabels}
        />
      </div>

      <div className="mt-4">
        <NarrativeCallout label="What this produces" tone="accent">{act.produces}</NarrativeCallout>
      </div>
    </ExplorerSectionCard>
  );
}
