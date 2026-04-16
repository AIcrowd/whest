import { useEffect, useMemo, useState } from 'react';
import { buildDiminoStages } from '../engine/permutation.js';

function fmtPiMapping(pi, labels) {
  return labels.map((label) => `${label}→${pi[label]}`).join(', ');
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

export default function DiminoView({ group, sigmaResults = [], selectedPairIndex = null }) {
  const stages = useMemo(() => buildDiminoStages(group.fullGenerators || []), [group.fullGenerators]);
  const validPiCards = useMemo(() => group.generatorSelection?.validPiCards || [], [group.generatorSelection]);
  const selectedSigmaPair = useMemo(() => {
    const activePairs = sigmaResults.filter((result) => !result.skipped);
    if (selectedPairIndex === null || selectedPairIndex < 0 || selectedPairIndex >= activePairs.length) {
      return null;
    }

    return activePairs[selectedPairIndex] ?? null;
  }, [selectedPairIndex, sigmaResults]);
  const selectedGeneratorCards = useMemo(() => {
    const firstMatchByKey = new Map(validPiCards.map((card) => [card.permutationKey, card]));
    return (group.fullGenerators || []).map((generator, idx) => ({
      generator,
      index: idx,
      key: generator.key(),
      source: firstMatchByKey.get(generator.key()) || null,
    }));
  }, [group.fullGenerators, validPiCards]);
  const [activeRoundIdx, setActiveRoundIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setActiveRoundIdx(0);
    setIsPlaying(false);
  }, [stages.length]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    if (activeRoundIdx >= stages.length - 1) {
      setIsPlaying(false);
      return undefined;
    }

    const timer = window.setTimeout(() => {
      setActiveRoundIdx((current) => Math.min(current + 1, stages.length - 1));
    }, 1400);

    return () => window.clearTimeout(timer);
  }, [activeRoundIdx, isPlaying, stages.length]);

  const stage = stages[activeRoundIdx];
  const isInitialRound = activeRoundIdx === 0;
  const canGoPrev = activeRoundIdx > 0;
  const canGoNext = activeRoundIdx < stages.length - 1;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-heading text-base font-semibold text-gray-900">Dimino: build the subgroup round by round</h3>
          <p className="mt-2 max-w-[62ch] text-sm leading-6 text-muted-foreground">
            Dimino starts from the identity subgroup. In each round, we add one chosen generator and close under composition with the subgroup already built.
          </p>
        </div>
        <div className="rounded-full border border-border bg-surface-raised px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          round {activeRoundIdx} / {Math.max(stages.length - 1, 0)}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-border bg-surface-raised p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Selected σ-pair</div>
        {selectedSigmaPair ? (
          <div className="mt-2 space-y-2">
            <div className="text-sm font-semibold text-foreground">
              {`Pair ${selectedPairIndex + 1}: ${selectedSigmaPair.isValid ? 'valid input for Dimino' : 'not used by Dimino'}`}
            </div>
            <div className="text-sm leading-6 text-muted-foreground">
              {selectedSigmaPair.pi
                ? `π mapping: ${fmtPiMapping(selectedSigmaPair.pi, group.allLabels)}`
                : selectedSigmaPair.reason || 'No π mapping is available for this pair.'}
            </div>
          </div>
        ) : (
          <div className="mt-2 text-sm leading-6 text-muted-foreground">
            No σ-pair is selected yet. Pick a pair in the σ-loop to inspect how it feeds the Dimino generator set.
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="rounded-full border border-border bg-white px-3 py-1.5 text-xs font-semibold text-foreground transition-colors hover:border-coral hover:text-coral disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => setActiveRoundIdx((current) => Math.max(current - 1, 0))}
          disabled={!canGoPrev}
        >
          Previous round
        </button>
        <button
          type="button"
          className="rounded-full border border-coral bg-coral px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-coral-dark"
          onClick={() => setIsPlaying((current) => !current)}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          className="rounded-full border border-border bg-white px-3 py-1.5 text-xs font-semibold text-foreground transition-colors hover:border-coral hover:text-coral disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => setActiveRoundIdx((current) => Math.min(current + 1, stages.length - 1))}
          disabled={!canGoNext}
        >
          Next round
        </button>
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          {stages.map((candidate) => (
            <button
              key={`dimino-round-${candidate.roundNumber}`}
              type="button"
              className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                candidate.roundNumber === activeRoundIdx
                  ? 'bg-coral text-white'
                  : 'border border-border bg-white text-muted-foreground hover:border-coral hover:text-coral'
              }`}
              onClick={() => {
                setActiveRoundIdx(candidate.roundNumber);
                setIsPlaying(false);
              }}
            >
              {`Round ${candidate.roundNumber}`}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-border bg-surface-raised p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Input generators</div>
        <div className="mt-2 text-sm leading-6 text-muted-foreground">
          Dimino runs on this selected generator set, not on every valid π.
        </div>
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {selectedGeneratorCards.map((card) => {
            const isActive = stage.generatorIndex === card.index;
            const isSeen = card.index < (stage.activeGenerators?.length ?? 0);
            return (
              <div
                key={`dimino-generator-${card.key}-${card.index}`}
                className={`rounded-lg border px-3 py-2 transition-colors ${
                  isActive
                    ? 'border-coral bg-coral-light'
                    : isSeen
                      ? 'border-border bg-white'
                      : 'border-border/70 bg-white/70'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    {`Generator ${card.index + 1}`}
                  </div>
                  <div className={`text-[11px] font-semibold ${isActive ? 'text-coral' : 'text-muted-foreground'}`}>
                    {isActive ? 'adding this generator now' : isSeen ? 'already included' : 'not added yet'}
                  </div>
                </div>
                <div className="mt-2 font-mono text-sm text-foreground">{card.generator.cycleNotation(group.allLabels)}</div>
                <div className="mt-2 text-xs text-muted-foreground">
                  {card.source ? `from π: ${fmtPiMapping(card.source.pi, group.allLabels)}` : 'from the deduplicated candidate-permutation bank'}
                </div>
              </div>
            );
          })}
          {selectedGeneratorCards.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-white px-3 py-3 text-sm text-muted-foreground">
              No non-identity generators were selected. The group stays at the identity subgroup.
            </div>
          ) : null}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-border bg-surface-raised p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Current round transformation</div>
        <div className="mt-2 text-sm font-semibold text-foreground">
          {isInitialRound ? 'Round 0: start from identity' : `Round ${stage.roundNumber}: add Generator ${stage.roundNumber}`}
        </div>
        <div className="mt-2 text-sm leading-6 text-foreground">
          {isInitialRound
            ? 'No generator yet. The subgroup starts at the identity element only.'
            : `Take the subgroup from the previous round, add Generator ${stage.roundNumber}, then close under composition.`}
        </div>

        <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_48px_minmax(0,220px)_48px_minmax(0,1fr)] xl:items-start">
          <div className="min-w-0">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Subgroup before</div>
            <PermCards elements={stage.beforeElements} labels={group.allLabels} />
          </div>

          <div className="hidden xl:flex h-full items-start justify-center pt-10 text-lg font-semibold text-muted-foreground">+</div>

          <div className="rounded-lg border border-border bg-white p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              {isInitialRound ? 'Start state' : `Add Generator ${stage.roundNumber}`}
            </div>
            <div className="mt-2">
              {stage.generator ? (
                <div className="perm-card generator dimino-new">
                  <code className="perm-notation">{stage.generator.cycleNotation(group.allLabels)}</code>
                  <span className="perm-structure">chosen generator</span>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">Identity subgroup only.</div>
              )}
            </div>
            <div className="mt-3 text-xs text-muted-foreground">
              {isInitialRound
                ? 'There is no added generator in Round 0.'
                : 'This is the new generator introduced at this round.'}
            </div>
          </div>

          <div className="hidden xl:flex h-full items-start justify-center pt-10 text-lg font-semibold text-muted-foreground">=</div>

          <div className="min-w-0">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Subgroup after</div>
            <PermCards elements={stage.afterElements} labels={group.allLabels} />
          </div>
        </div>

        <div className="mt-4 rounded-lg border border-border bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Closure under composition</div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            These new elements appear because the added generator composes with the subgroup already generated.
          </p>
          <div className="mt-3">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">New elements from closure</div>
            <PermCards
              elements={stage.newElements}
              labels={group.allLabels}
              highlight
              emptyText={isInitialRound ? 'Identity is the starting subgroup.' : 'This generator adds no new elements beyond the subgroup already built.'}
            />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            <span>{`order before ${stage.beforeSize}`}</span>
            <span>{`new elements ${stage.newCount}`}</span>
            <span>{`order after ${stage.afterSize}`}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
