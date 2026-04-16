import { useMemo } from 'react';

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

function PermCards({ elements, labels, highlight = false, emptyText = null }) {
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
          <code className="perm-notation">{element.cycleNotation(labels)}</code>
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
  const selectedPair = useMemo(() => {
    const activePairs = sigmaResults.filter((result) => !result.skipped);
    if (selectedPairIndex === null || selectedPairIndex < 0 || selectedPairIndex >= activePairs.length) {
      return null;
    }
    return activePairs[selectedPairIndex] ?? null;
  }, [selectedPairIndex, sigmaResults]);
  const selectedPermutationKey = selectedPair?.pi ? permutationKeyFromPi(selectedPair.pi, orderedLabels) : null;
  const candidate = useMemo(
    () => group?.generatorSelection?.candidatePermutations?.find((entry) => entry.permutationKey === selectedPermutationKey) ?? null,
    [group?.generatorSelection?.candidatePermutations, selectedPermutationKey],
  );
  const closureNewElements = useMemo(() => newElementsFromClosure(candidate), [candidate]);
  const decisionKeepsCandidate = Boolean(candidate?.kept);
  const hasMergedProvenance = (candidate?.sourcePiIds?.length || 0) > 1;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div>
        <h3 className="font-heading text-base font-semibold text-gray-900">Generator Construction</h3>
        <p className="mt-2 max-w-[62ch] text-sm leading-6 text-muted-foreground">
          Each valid π induces a candidate permutation. We keep it only if adding it enlarges the subgroup generated so far.
        </p>
      </div>

      {!selectedPair ? (
        <div className="mt-4 rounded-lg border border-dashed border-border bg-surface-raised p-4 text-sm leading-6 text-muted-foreground">
          Select a valid (σ, π) pair on the left to inspect the single candidate-construction step it induces.
        </div>
      ) : !candidate ? (
        <div className="mt-4 rounded-lg border border-dashed border-border bg-surface-raised p-4 text-sm leading-6 text-muted-foreground">
          This selection does not produce a non-identity candidate permutation for generator construction.
        </div>
      ) : (
        <div className="mt-4 space-y-4">
          <ProofSection title="Current candidate from π">
            <div className="space-y-3">
              <div className="perm-card generator dimino-new">
                <code className="perm-notation">{candidate.cycleNotation}</code>
                <span className="perm-structure">candidate induced by the selected valid π</span>
              </div>
              <div className="text-sm leading-6 text-muted-foreground">
                {`Selected π mapping: ${fmtPiMapping(selectedPair.pi, orderedLabels)}`}
              </div>
            </div>
          </ProofSection>

          <ProofSection title="Previous subgroup">
            <div className="space-y-3">
              <div className="text-sm leading-6 text-muted-foreground">
                {`Before testing ${candidate.cycleNotation}, the subgroup already had order ${candidate.growthFrom}.`}
              </div>
              <PermCards
                elements={candidate.beforeElements || []}
                labels={orderedLabels}
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
              {decisionKeepsCandidate
                ? `Keep ${candidate.cycleNotation}: the subgroup grows from order ${candidate.growthFrom} to ${candidate.growthTo}.`
                : `Discard ${candidate.cycleNotation}: closure stays at order ${candidate.growthTo}, so this candidate is redundant.`}
            </div>
          </ProofSection>

          <ProofSection title="Subgroup after test">
            <div className="space-y-3">
              <PermCards elements={candidate.afterElements || []} labels={orderedLabels} />
              <div className="text-sm leading-6 text-muted-foreground">
                {decisionKeepsCandidate
                  ? 'This enlarged subgroup becomes the starting point for the next candidate.'
                  : 'The subgroup is unchanged, so the next candidate is tested against the same generated set.'}
              </div>
            </div>
          </ProofSection>

          {hasMergedProvenance ? (
            <div className="rounded-lg border border-border bg-white p-3 text-xs leading-6 text-muted-foreground">
              {`Provenance note: ${candidate.sourcePiIds.length} valid π mappings collapse to this same candidate permutation, so they share this closure decision.`}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
