import { Fragment, useMemo } from 'react';

// V/W color palette — same hexes the rest of Act 4 uses (Interaction Graph
// legend, IncidenceMatrix v/w columns, TotalCostView hero glossary).
const COLOR_V = '#4A7CFF'; // free / output
const COLOR_W = '#64748B'; // summed / contracted (slate-600 — darker than the
                           // canonical #94A3B8 so the labels stay readable
                           // when wrapped in cycle parens on a white card)

// Tokenize a cycle-notation / mapping / label-list string and recolor every
// label that matches the component's V or W role. Pass-through for parens,
// whitespace, commas, arrows, and any non-label punctuation. Identity 'e' is
// left uncolored on purpose — it's a group element, not a label.
function ColoredLabels({ text, vSet, wSet }) {
  if (!text || typeof text !== 'string') return text ?? null;
  const tokens = text.split(/([(),\s→·\u27e8\u27e9])/);
  return (
    <>
      {tokens.map((tok, i) => {
        if (!tok) return null;
        if (vSet.has(tok)) {
          return (
            <span key={i} style={{ color: COLOR_V, fontWeight: 600 }}>
              {tok}
            </span>
          );
        }
        if (wSet.has(tok)) {
          return (
            <span key={i} style={{ color: COLOR_W, fontWeight: 600 }}>
              {tok}
            </span>
          );
        }
        return <Fragment key={i}>{tok}</Fragment>;
      })}
    </>
  );
}

function permutationKeyFromPi(pi, orderedLabels) {
  if (!pi || !orderedLabels?.length) return null;
  const indexByLabel = new Map(orderedLabels.map((label, index) => [label, index]));
  const arr = orderedLabels.map((label) => indexByLabel.get(pi[label]));
  if (arr.some((value) => value === undefined)) return null;
  return arr.join(',');
}

function fmtPiMapping(pi, labels) {
  if (!pi || !labels?.length) return '—';
  return labels.map((label) => `${label} → ${pi[label]}`).join(', ');
}

function newElementsFromClosure(candidate) {
  if (!candidate) return [];
  const beforeKeys = new Set((candidate.beforeElements || []).map((element) => element.key()));
  return (candidate.afterElements || []).filter((element) => !beforeKeys.has(element.key()));
}

function PermCards({ elements, labels, vSet, wSet, highlight = false, emptyText = null }) {
  if (!elements.length) {
    return emptyText ? <div className="text-sm text-muted-foreground">{emptyText}</div> : null;
  }

  return (
    <div className="perm-list">
      {elements.map((element, idx) => (
        <div
          key={`${element.key()}-${idx}`}
          className={`perm-card ${element.isIdentity ? 'identity' : ''} ${highlight ? 'generator dimino-new' : ''}`}
        >
          <code className="perm-notation">
            <ColoredLabels text={element.cycleNotation(labels)} vSet={vSet} wSet={wSet} />
          </code>
          <span className="perm-structure">{element.isIdentity ? 'identity' : `${element.cyclicForm().length} moved cycle${element.cyclicForm().length === 1 ? '' : 's'}`}</span>
        </div>
      ))}
    </div>
  );
}

function ProofSection({ title, children }) {
  return (
    <div className="rounded-lg border border-border bg-surface-raised p-3">
      <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">{title}</div>
      <div className="mt-3">{children}</div>
    </div>
  );
}

export default function DiminoView({ group, sigmaResults = [], selectedPairIndex = null }) {
  const orderedLabels = group?.allLabels || [];
  const vSet = useMemo(() => new Set(group?.vLabels ?? []), [group?.vLabels]);
  const wSet = useMemo(() => new Set(group?.wLabels ?? []), [group?.wLabels]);
  const validPairs = useMemo(() => sigmaResults.filter((result) => !result.skipped), [sigmaResults]);
  const selectedPair = useMemo(() => {
    if (selectedPairIndex === null || selectedPairIndex < 0 || selectedPairIndex >= validPairs.length) {
      return validPairs[0] ?? null;
    }
    return validPairs[selectedPairIndex] ?? validPairs[0] ?? null;
  }, [selectedPairIndex, validPairs]);
  const selectedPermutationKey = selectedPair?.pi ? permutationKeyFromPi(selectedPair.pi, orderedLabels) : null;
  const candidatePermutations = group?.generatorSelection?.candidatePermutations || [];
  const matchedCandidate = useMemo(
    () => candidatePermutations.find((entry) => entry.permutationKey === selectedPermutationKey) ?? null,
    [candidatePermutations, selectedPermutationKey],
  );
  const candidate = matchedCandidate ?? candidatePermutations[0] ?? null;
  const closureNewElements = useMemo(() => newElementsFromClosure(candidate), [candidate]);
  const decisionKeepsCandidate = Boolean(candidate?.kept);
  const hasMergedProvenance = (candidate?.sourcePiIds?.length || 0) > 1;
  const usingPairFallback = selectedPairIndex === null || selectedPairIndex < 0 || selectedPairIndex >= validPairs.length;
  const usingCandidateFallback = Boolean(selectedPair && !matchedCandidate && candidate && candidate === candidatePermutations[0]);

  if (!selectedPair || !candidate) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h3 className="font-heading text-base font-semibold text-gray-900">Generator Construction</h3>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          Select a valid `(σ, π)` pair on the left to test the induced label permutation it induces.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div>
        {usingPairFallback ? (
          <div className="rounded-lg border border-dashed border-border bg-surface-raised px-3 py-2 text-xs leading-5 text-muted-foreground">
            No exact pair is selected yet, so this panel is showing the first valid `(σ, π)` pair as a stable fallback.
          </div>
        ) : null}
        {usingCandidateFallback ? (
          <div className="mt-3 rounded-lg border border-dashed border-border bg-surface-raised px-3 py-2 text-xs leading-5 text-muted-foreground">
            The selected pair does not map to a unique induced label permutation, so this panel is showing the first one as a stable fallback.
          </div>
        ) : null}
        <p className="mt-2 max-w-[62ch] text-sm leading-6 text-muted-foreground">
          Each valid π induces a label permutation on the active labels. We keep it only if adding it enlarges the subgroup generated so far.
        </p>
        {group ? (
          <div className="mt-3 rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs leading-5 text-muted-foreground">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-foreground">
              <code className="rounded bg-stone-100 px-1.5 py-0.5 font-mono text-[12px] font-semibold tracking-wide text-foreground">
                <ColoredLabels text={group.fullGroupName} vSet={vSet} wSet={wSet} />
              </code>
              <span className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                order&nbsp;<span className="font-semibold text-foreground">{group.fullOrder}</span>
              </span>
            </div>
            <div className="mt-1.5">
              {`${group.fullDegree} active labels: `}
              <ColoredLabels text={group.allLabels.join(', ')} vSet={vSet} wSet={wSet} />
              {group.fullGenerators?.length ? ` · ${group.fullGenerators.length} generator${group.fullGenerators.length === 1 ? '' : 's'}` : ''}
            </div>
          </div>
        ) : null}
      </div>

      {!selectedPair ? (
        <div className="mt-4 rounded-lg border border-dashed border-border bg-surface-raised p-4 text-sm leading-6 text-muted-foreground">
          Select a valid (σ, π) pair on the left to inspect the single candidate-construction step it induces.
        </div>
      ) : !candidate ? (
        <div className="mt-4 rounded-lg border border-dashed border-border bg-surface-raised p-4 text-sm leading-6 text-muted-foreground">
          This selection does not produce a non-identity induced label permutation for generator construction.
        </div>
      ) : (
        <div className="mt-4 space-y-4">
          <ProofSection title="Current candidate from π">
            <div className="space-y-3">
              <div className="perm-card generator dimino-new">
                <code className="perm-notation">
                  <ColoredLabels text={candidate.cycleNotation} vSet={vSet} wSet={wSet} />
                </code>
                <span className="perm-structure">candidate induced by the selected valid π</span>
              </div>
              <div className="text-sm leading-6 text-muted-foreground">
                {'Selected π mapping: '}
                <ColoredLabels text={fmtPiMapping(selectedPair.pi, orderedLabels)} vSet={vSet} wSet={wSet} />
              </div>
            </div>
          </ProofSection>

          <ProofSection title="Previous subgroup">
            <div className="space-y-3">
              <div className="text-sm leading-6 text-muted-foreground">
                {'Before testing '}
                <ColoredLabels text={candidate.cycleNotation} vSet={vSet} wSet={wSet} />
                {`, the subgroup already had order ${candidate.growthFrom}.`}
              </div>
              <PermCards
                elements={candidate.beforeElements || []}
                labels={orderedLabels}
                vSet={vSet}
                wSet={wSet}
                emptyText="The construction starts from the identity subgroup."
              />
            </div>
          </ProofSection>

          <ProofSection title="Closure test">
            <div className="space-y-3">
              <div className="text-sm leading-6 text-muted-foreground">
                Close the previous subgroup together with the candidate under composition and check whether any new elements appear.
              </div>
              <PermCards
                elements={closureNewElements}
                labels={orderedLabels}
                vSet={vSet}
                wSet={wSet}
                highlight
                emptyText="Closure adds no new elements, so the candidate is already generated by the previous subgroup."
              />
              <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                <span>{`order before ${candidate.growthFrom}`}</span>
                <span>{`new from closure ${closureNewElements.length}`}</span>
                <span>{`order after ${candidate.growthTo}`}</span>
              </div>
            </div>
          </ProofSection>

          <ProofSection title="Decision">
            <div className={`rounded-lg border px-3 py-3 text-sm leading-6 ${decisionKeepsCandidate ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : 'border-amber-200 bg-amber-50 text-amber-900'}`}>
              {decisionKeepsCandidate ? 'Keep ' : 'Discard '}
              <ColoredLabels text={candidate.cycleNotation} vSet={vSet} wSet={wSet} />
              {decisionKeepsCandidate
                ? `: the subgroup grows from order ${candidate.growthFrom} to ${candidate.growthTo}.`
                : `: closure stays at order ${candidate.growthTo}, so this candidate is redundant.`}
            </div>
          </ProofSection>

          <ProofSection title="Subgroup after test">
            <div className="space-y-3">
              <PermCards
                elements={candidate.afterElements || []}
                labels={orderedLabels}
                vSet={vSet}
                wSet={wSet}
              />
              <div className="text-sm leading-6 text-muted-foreground">
                {decisionKeepsCandidate
                  ? 'This enlarged subgroup becomes the starting point for the next candidate.'
                  : 'The subgroup is unchanged, so the next candidate is tested against the same generated set.'}
              </div>
            </div>
          </ProofSection>

          {hasMergedProvenance ? (
            <div className="rounded-lg border border-border bg-white p-3 text-xs leading-6 text-muted-foreground">
              {`Provenance note: ${candidate.sourcePiIds.length} valid π mappings collapse to this same induced label permutation, so they share this closure decision.`}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
