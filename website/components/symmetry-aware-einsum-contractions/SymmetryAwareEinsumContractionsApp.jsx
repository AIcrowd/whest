import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import { analyzeExample } from './engine/pipeline.js';
import { buildMentalModelCode, pickDefaultOrbitRow } from './engine/teachingModel.js';
import StickyBar from './components/StickyBar.jsx';
import ExplorerSectionCard from './components/ExplorerSectionCard.jsx';
import { EXPLORER_ACTS, buildAnalysisCheckpoint } from './components/explorerNarrative.js';
import NarrativeCallout from './components/NarrativeCallout.jsx';
import ExampleChooser from './components/ExampleChooser.jsx';
import PresetSidebar from './components/PresetSidebar.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import GroupView from './components/GroupView.jsx';
import PythonCodeBlock from './components/PythonCodeBlock.jsx';
import ComponentCostView from './components/ComponentCostView.jsx';
import TotalCostView from './components/TotalCostView.jsx';
import { mergeObservedActEntries, pickTopVisibleAct } from './lib/activeAct.js';
import { getPresetControlSelection } from './lib/presetSelection.js';
import { reduceMentalModelVisibility } from './lib/mentalModelState.js';
import './styles.css';

const CUSTOM_IDX = -1;

/**
 * Convert the new preset format (with `variables` and `expression` fields)
 * to the algorithm-compatible format (with top-level `subscripts`, `output`,
 * `operandNames`, `perOpSymmetry`).
 */
function normalizeExample(example) {
  // Already normalized (from onCustomExample callback)
  if (Array.isArray(example.subscripts)) return example;
  // Convert new preset format
  const { expression, variables } = example;
  if (!expression) return example;
  const subsArr = expression.subscripts.split(',').map(s => s.trim());
  const opsArr = expression.operandNames.split(',').map(s => s.trim());
  const perOpSymmetry = opsArr.map(opName => {
    const v = variables.find(v => v.name === opName);
    if (!v || v.symmetry === 'none') return null;
    const axes = v.symAxes || [...Array(v.rank).keys()];
    if (v.symmetry === 'symmetric' && axes.length === v.rank) return 'symmetric';
    if (v.symmetry === 'custom' && v.generators) {
      const { generators } = parseCycleNotation(v.generators);
      return { type: 'custom', axes, generators };
    }
    return { type: v.symmetry, axes };
  });
  const hasAnySym = perOpSymmetry.some(s => s !== null);
  return {
    ...example,
    subscripts: subsArr,
    output: expression.output,
    operandNames: opsArr,
    perOpSymmetry: hasAnySym ? perOpSymmetry : null,
  };
}

export default function SymmetryAwareEinsumContractionsApp() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [customExample, setCustomExample] = useState(null);
  const [dimensionN, setDimensionN] = useState(5);
  const [selectedOrbitIdx, setSelectedOrbitIdx] = useState(-1);
  const [activeActId, setActiveActId] = useState(EXPLORER_ACTS[0].id);
  const [isDirty, setIsDirty] = useState(false);
  const [showMentalModel, setShowMentalModel] = useState(false);
  const observedEntriesRef = useRef(new Map());

  // Resolve the active example: preset or custom
  const isCustom = exampleIdx === CUSTOM_IDX;
  const example = isCustom ? customExample : EXAMPLES[exampleIdx];
  const selectedPresetIdx = getPresetControlSelection(exampleIdx, isDirty);

  // Derive variable colors from the example's variables (works for both presets and custom)
  const variableColors = useMemo(() => {
    if (example?.variables) return buildVariableColors(example.variables);
    return {};
  }, [example]);

  // Normalize the example for algorithm consumption
  const normalizedExample = useMemo(() => example ? normalizeExample(example) : null, [example]);

  // Handle preset selection
  const handleSelect = useCallback((idx) => {
    setShowMentalModel((isOpen) => reduceMentalModelVisibility(isOpen, 'selectPreset'));
    setExampleIdx(idx);
    setIsDirty(false);
    setSelectedOrbitIdx(-1);
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setShowMentalModel((isOpen) => reduceMentalModelVisibility(isOpen, 'customExample'));
    setCustomExample(ex);
    setExampleIdx(CUSTOM_IDX);
    setSelectedOrbitIdx(-1);
  }, []);

  const handleCustomMode = useCallback(() => {
    setShowMentalModel((isOpen) => reduceMentalModelVisibility(isOpen, 'customMode'));
    setExampleIdx(CUSTOM_IDX);
    setSelectedOrbitIdx(-1);
  }, []);

  const analysis = useMemo(() => {
    if (!example) return null;
    try {
      return analyzeExample(example, dimensionN);
    } catch (err) {
      console.error('Pipeline error:', err);
      return null;
    }
  }, [example, dimensionN]);

  const {
    graph,
    matrixData,
    sigmaResults,
    symmetry: group,
    componentData,
    costModel: cost,
  } = analysis || {};

  const resolvedSelectedOrbitIdx = useMemo(() => {
    const orbitRows = cost?.orbitRows ?? [];
    if (orbitRows.length === 0) return -1;
    if (selectedOrbitIdx >= 0 && selectedOrbitIdx < orbitRows.length) return selectedOrbitIdx;
    return pickDefaultOrbitRow(orbitRows);
  }, [cost, selectedOrbitIdx]);

  const mentalModelCode = useMemo(
    () => buildMentalModelCode(cost?.orbitRows?.[resolvedSelectedOrbitIdx] ?? null),
    [cost, resolvedSelectedOrbitIdx],
  );

  // Check if per-op symmetry is active for any operand
  const hasPerOpSym = normalizedExample && (
    normalizedExample.perOpSymmetry === 'symmetric' ||
    (Array.isArray(normalizedExample.perOpSymmetry) && normalizedExample.perOpSymmetry.some(s => s === 'symmetric' || (s && typeof s === 'object')))
  );

  useEffect(() => {
    const sections = EXPLORER_ACTS
      .map(({ id }) => document.getElementById(id))
      .filter(Boolean);

    observedEntriesRef.current = new Map();

    if (sections.length === 0) return undefined;

    const observer = new IntersectionObserver((entries) => {
      observedEntriesRef.current = mergeObservedActEntries(observedEntriesRef.current, entries);
      setActiveActId((current) => pickTopVisibleAct(Array.from(observedEntriesRef.current.values()), current));
    }, {
      rootMargin: '-18% 0px -55% 0px',
      threshold: [0, 0.2, 0.5],
    });

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [analysis, example]);

  const checkpointItems = buildAnalysisCheckpoint({ example: normalizedExample, group });

  return (
    <div className="min-h-screen bg-background">
      <StickyBar
        example={example}
        group={group}
        dimensionN={dimensionN}
        onDimensionChange={setDimensionN}
        activeActId={activeActId}
      />

      <div className="mx-auto flex max-w-[1400px] flex-col gap-6 px-6 pb-16 pt-6 md:px-10">
        <ExplorerSectionCard
          eyebrow="Interactive walkthrough"
          title="Symmetry Aware Einsum Contractions"
          description={<><em>Symmetry detection for einsum contractions</em>, explained as a four-act explorer.</>}
          className="border-border/70 shadow-sm"
          contentClassName="pt-5"
        >
          {example && group ? (
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <Badge className="bg-coral text-white hover:bg-coral">einsum</Badge>
                <code className="truncate rounded-md bg-muted px-3 py-1.5 font-mono text-sm text-foreground">
                  {example.formula}
                </code>
                <Badge variant="outline" className="border-coral/30 bg-coral-light text-coral">
                  {group.fullGroupName || 'trivial'}
                </Badge>
              </div>
              <p className="max-w-2xl text-sm text-muted-foreground">
                Start in Act 1 to choose a preset or build a custom contraction, then follow the graph, proof, and savings views in order.
              </p>
            </div>
          ) : (
            <p className="max-w-3xl text-sm text-muted-foreground">
              Choose a preset or define your own contraction to walk through structure detection, group construction, and symmetry-aware pricing.
            </p>
          )}
        </ExplorerSectionCard>

        <div className="flex flex-col gap-8 md:flex-row md:items-start">
          <main className="min-w-0 flex-1">
            <section id="setup" className="section scroll-mt-24">
              <ExampleChooser
                examples={EXAMPLES}
                onSelect={handleSelect}
                selectedPresetIdx={selectedPresetIdx}
                dimensionN={dimensionN}
                onCustom={handleCustomMode}
                onCustomExample={handleCustomExample}
                onDirtyChange={setIsDirty}
                act={EXPLORER_ACTS[0]}
                checkpointItems={checkpointItems}
              />
            </section>

            {/* Only render pipeline sections when we have results */}
            {analysis && example && (
              <>
                <section id="structure" className="section scroll-mt-24 border-t border-gray-200 pt-10 pb-6">
                  <ActHeader
                    number={2}
                    heading={EXPLORER_ACTS[1].heading}
                    question={EXPLORER_ACTS[1].question}
                  />
                  <NarrativeCallout label="Why this matters">{EXPLORER_ACTS[1].why}</NarrativeCallout>
                  <p className="section-desc mt-4">
                    Left vertices (U) are operand axis-classes. Right vertices are index labels,
                    partitioned into <span className="pill pill-v">V free</span> and{' '}
                    <span className="pill pill-w">W summed</span>.
                    {hasPerOpSym && (
                      <> Per-operand symmetry <em>collapses</em> each operand&apos;s axes into a single U-vertex.</>
                    )}
                  </p>
                  <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
                    <div>
                      <h3 className="mb-2 font-accent text-sm font-semibold text-gray-900">Bipartite Graph</h3>
                      <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} />
                    </div>
                    <div>
                      <h3 className="mb-2 font-accent text-sm font-semibold text-gray-900">Incidence Matrix M</h3>
                      <MatrixView matrixData={matrixData} graph={graph} example={normalizedExample} variableColors={variableColors} />
                    </div>
                  </div>
                  <p className="mt-4 text-sm text-gray-600">{EXPLORER_ACTS[1].bridge}</p>
                  <div className="mt-4">
                    <NarrativeCallout label="Takeaway" tone="accent">{EXPLORER_ACTS[1].takeaway}</NarrativeCallout>
                  </div>
                </section>

                <section id="proof" className="section scroll-mt-24 border-t border-gray-200 pt-10 pb-6">
                  <ActHeader
                    number={3}
                    heading={EXPLORER_ACTS[2].heading}
                    question={EXPLORER_ACTS[2].question}
                  />
                  <NarrativeCallout label="Why this matters">{EXPLORER_ACTS[2].why}</NarrativeCallout>
                  <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
                    <div>
                      <h3 className="mb-2 font-accent text-sm font-semibold text-gray-900">σ-Loop & π Detection</h3>
                      <SigmaLoop
                        results={sigmaResults}
                        graph={graph}
                        matrixData={matrixData}
                        example={normalizedExample}
                        variableColors={variableColors}
                      />
                    </div>
                    <div>
                      <h3 className="mb-2 font-accent text-sm font-semibold text-gray-900">Group Construction</h3>
                      <GroupView group={group} />
                    </div>
                  </div>
                  <p className="mt-4 text-sm text-gray-600">{EXPLORER_ACTS[2].bridge}</p>
                  <div className="mt-4">
                    <NarrativeCallout label="Takeaway" tone="accent">{EXPLORER_ACTS[2].takeaway}</NarrativeCallout>
                  </div>
                </section>

                <section id="savings" className="section scroll-mt-24 border-t border-gray-200 pt-10 pb-6">
                  <ActHeader
                    number={4}
                    heading={EXPLORER_ACTS[3].heading}
                    question={EXPLORER_ACTS[3].question}
                  />
                  <div className="mt-6 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="max-w-3xl">
                      <NarrativeCallout label="Why this matters">{EXPLORER_ACTS[3].why}</NarrativeCallout>
                    </div>
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full bg-coral-light text-coral hover:bg-coral-light/80 lg:w-auto lg:self-start"
                      aria-label="Open mental framework"
                      onClick={() => setShowMentalModel(true)}
                    >
                      Open mental framework
                    </Button>
                  </div>

                  {showMentalModel && (
                    <div
                      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm"
                      onClick={() => setShowMentalModel(false)}
                    >
                      <div
                        className="max-h-[85vh] w-[min(960px,92vw)] overflow-y-auto rounded-xl bg-white shadow-2xl"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="mental-framework-modal-title"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <div className="flex items-center justify-between gap-4 border-b border-border/70 px-5 py-4">
                          <h2 id="mental-framework-modal-title" className="sr-only">Mental Framework</h2>
                          <div className="text-sm font-medium text-foreground">Mental Framework</div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            aria-label="Close mental framework"
                            onClick={() => setShowMentalModel(false)}
                          >
                            Close
                          </Button>
                        </div>
                        <div className="px-5 pb-5 pt-5">
                          <PythonCodeBlock
                            code={mentalModelCode}
                            title="Mental Framework"
                            description="Read this as the mental model for the rest of Act 4: first count one symmetry-unique multiplication representative, then count every distinct output-bin update it causes."
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="mt-6">
                    <ComponentCostView
                      componentData={componentData}
                      costModel={cost}
                      dimensionN={dimensionN}
                      allLabels={group.allLabels}
                      vLabels={group.vLabels}
                      selectedOrbitIdx={resolvedSelectedOrbitIdx}
                      onSelectOrbit={setSelectedOrbitIdx}
                    />
                  </div>

                  <div className="mt-6">
                    <TotalCostView
                      costModel={cost}
                      componentData={componentData}
                      dimensionN={dimensionN}
                      numTerms={normalizedExample?.subscripts?.length ?? 1}
                    />
                  </div>

                  <p className="mt-4 text-sm text-gray-600">{EXPLORER_ACTS[3].bridge}</p>
                  <div className="mt-4">
                    <NarrativeCallout label="Takeaway" tone="accent">{EXPLORER_ACTS[3].takeaway}</NarrativeCallout>
                  </div>
                </section>
              </>
            )}

            {/* Show prompt when custom is selected but no expression analyzed yet */}
            {isCustom && !analysis && (
              <div className="custom-prompt">
                Define your variables and einsum expression above, then click <strong>Analyze</strong> to explore the symmetry detection algorithm.
              </div>
            )}
          </main>
          <PresetSidebar
            examples={EXAMPLES}
            selectedPresetIdx={selectedPresetIdx}
            onSelect={handleSelect}
            onCustom={handleCustomMode}
          />
        </div>
      </div>
    </div>
  );
}

function ActHeader({ number, heading, question }) {
  return (
    <div className="mb-8 flex items-start gap-4">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-coral text-sm font-mono font-bold text-white">
        {number}
      </span>
      <div className="flex-1">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-coral">Act {number}</p>
        <h2 className="font-accent text-xl font-bold tracking-tight text-gray-900">{heading}</h2>
        <p className="mt-1 text-sm italic text-gray-600">{question}</p>
      </div>
    </div>
  );
}
