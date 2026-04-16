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
import RoleBadge from './components/RoleBadge.jsx';
import PythonCodeBlock from './components/PythonCodeBlock.jsx';
import ComponentCostView, { COMPONENT_STORY_TEXT } from './components/ComponentCostView.jsx';
import TotalCostView from './components/TotalCostView.jsx';
import { mergeObservedActEntries, pickTopVisibleAct } from './lib/activeAct.js';
import { getPresetControlSelection } from './lib/presetSelection.js';
import { reduceMentalModelVisibility } from './lib/mentalModelState.js';
import './styles.css';

const CUSTOM_IDX = -1;
const DEFAULT_EXAMPLE_ID = 'triple-outer';
const DEFAULT_EXAMPLE_IDX = Math.max(0, EXAMPLES.findIndex((example) => example.id === DEFAULT_EXAMPLE_ID));

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
  const [exampleIdx, setExampleIdx] = useState(DEFAULT_EXAMPLE_IDX);
  const [customExample, setCustomExample] = useState(null);
  const [previewExample, setPreviewExample] = useState(EXAMPLES[DEFAULT_EXAMPLE_IDX]);
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
    setPreviewExample(EXAMPLES[idx] ?? null);
    setIsDirty(false);
    setSelectedOrbitIdx(-1);
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setShowMentalModel((isOpen) => reduceMentalModelVisibility(isOpen, 'customExample'));
    setCustomExample(ex);
    setPreviewExample(ex);
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
        example={previewExample ?? example}
        group={group}
        activeActId={activeActId}
      />

      <div className="w-full pb-20 pt-8">
        <div className="flex flex-col pb-6">
          <ExplorerSectionCard
          eyebrow="Interactive walkthrough"
          title="Symmetry Aware Einsum Contractions"
          description={<><em>Symmetry detection for einsum contractions</em>, explained as a five-act explorer.</>}
          className="border-border/70 shadow-sm"
          contentClassName="pt-6"
        >
          </ExplorerSectionCard>

        </div>

        <div className="mt-8 flex items-start gap-8">
          <PresetSidebar
            examples={EXAMPLES}
            selectedPresetIdx={selectedPresetIdx}
            onSelect={handleSelect}
            onCustom={handleCustomMode}
          />
          <main className="min-w-0 flex-1">
            <div className="mx-auto flex max-w-[1460px] flex-col px-6 md:px-8 lg:px-10">
            <section id={EXPLORER_ACTS[0].id} className="mb-12 scroll-mt-24">
              <ExplorerSectionCard
                eyebrow="Act 1"
                title={EXPLORER_ACTS[0].heading}
                description={EXPLORER_ACTS[0].question}
                className="border-gray-200 bg-white"
                contentClassName="pt-5"
              >
                <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[0].interpretation}</NarrativeCallout>
                <NarrativeCallout label="Algorithm" tone="algorithm">{EXPLORER_ACTS[0].algorithm}</NarrativeCallout>
                <p className="mt-4 text-sm leading-7 text-foreground">
                  This act specifies declared input symmetry only. It fixes the operands and labels before any detected contraction symmetry is considered.
                </p>
                <ExampleChooser
                  examples={EXAMPLES}
                  onSelect={handleSelect}
                  selectedPresetIdx={selectedPresetIdx}
                  dimensionN={dimensionN}
                  onDimensionChange={setDimensionN}
                  onCustom={handleCustomMode}
                  onCustomExample={handleCustomExample}
                  onPreviewChange={setPreviewExample}
                  onDirtyChange={setIsDirty}
                  checkpointItems={checkpointItems}
                />
                <div className="mt-4">
                  <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[0].produces}</NarrativeCallout>
                </div>
              </ExplorerSectionCard>
            </section>

            {/* Only render pipeline sections when we have results */}
            {analysis && example && (
              <>
                <section id={EXPLORER_ACTS[1].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="Act 2"
                    title={EXPLORER_ACTS[1].heading}
                    description={EXPLORER_ACTS[1].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[1].interpretation}</NarrativeCallout>
                    <NarrativeCallout label="Algorithm" tone="algorithm">{EXPLORER_ACTS[1].algorithm}</NarrativeCallout>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      Declared input symmetry changes the encoding before any detected contraction symmetry is computed.
                    </p>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      {EXPLORER_ACTS[1].bridge}
                    </p>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      Left vertices (U) are operand axis-classes. Right vertices are index labels,
                      partitioned into <RoleBadge role="v">V free</RoleBadge> and
                      <RoleBadge role="w">W summed</RoleBadge>.
                      {hasPerOpSym && (
                        <> Per-operand symmetry <em>collapses</em> each operand&apos;s axes into a single U-vertex.</>
                      )}
                    </p>
                    <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
                      <div>
                        <h3 className="mb-2 font-heading text-base font-semibold text-gray-900">Bipartite Graph</h3>
                        <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} />
                      </div>
                      <div>
                        <h3 className="mb-2 font-heading text-base font-semibold text-gray-900">Incidence Matrix M</h3>
                        <MatrixView matrixData={matrixData} graph={graph} example={normalizedExample} variableColors={variableColors} />
                      </div>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[1].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[2].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="Act 3"
                    title={EXPLORER_ACTS[2].heading}
                    description={EXPLORER_ACTS[2].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[2].interpretation}</NarrativeCallout>
                    <NarrativeCallout label="Algorithm" tone="algorithm">{EXPLORER_ACTS[2].algorithm}</NarrativeCallout>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      Detected contraction symmetry is a property of the whole einsum expression, not just one input tensor.
                    </p>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      {EXPLORER_ACTS[2].bridge}
                    </p>
                    <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <div>
                        <h3 className="mb-2 font-heading text-base font-semibold text-gray-900">σ-Loop & π Detection</h3>
                        <SigmaLoop
                          results={sigmaResults}
                          graph={graph}
                          matrixData={matrixData}
                          example={normalizedExample}
                          variableColors={variableColors}
                        />
                      </div>
                      <div>
                        <h3 className="mb-2 font-heading text-base font-semibold text-gray-900">Group Construction</h3>
                        <GroupView group={group} />
                      </div>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[2].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[3].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="Act 4"
                    title={EXPLORER_ACTS[3].heading}
                    description={EXPLORER_ACTS[3].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                    action={
                      <Button
                        type="button"
                        size="lg"
                        className="gap-2 font-semibold shadow-sm"
                        aria-label="Open mental framework"
                        onClick={() => setShowMentalModel(true)}
                      >
                        Open Mental Framework
                      </Button>
                    }
                  >
                    <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[3].interpretation}</NarrativeCallout>
                    <NarrativeCallout label="Algorithm" tone="algorithm">{EXPLORER_ACTS[3].algorithm}</NarrativeCallout>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      The detected global action may have been influenced by declared input symmetry, but it is the object being decomposed.
                    </p>
                    <p className="mt-4 text-sm leading-7 text-foreground">
                      {EXPLORER_ACTS[3].bridge}
                    </p>
                    <div className="grid gap-4 md:grid-cols-2">
                      <NarrativeCallout label="Component Story">{COMPONENT_STORY_TEXT}</NarrativeCallout>
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
                      showComponentStory={false}
                    />
                  </div>

                  <div className="mt-4">
                    <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[3].produces}</NarrativeCallout>
                  </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[4].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="Act 5"
                    title={EXPLORER_ACTS[4].heading}
                    description={EXPLORER_ACTS[4].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <NarrativeCallout label="Why this matters">{EXPLORER_ACTS[4].why}</NarrativeCallout>

                    <div className="mt-6">
                      <TotalCostView
                        costModel={cost}
                        componentData={componentData}
                        dimensionN={dimensionN}
                        numTerms={normalizedExample?.subscripts?.length ?? 1}
                      />
                    </div>

                    <p className="mt-4 text-sm text-gray-600">{EXPLORER_ACTS[4].bridge}</p>
                    <div className="mt-4">
                      <NarrativeCallout label="Takeaway" tone="accent">{EXPLORER_ACTS[4].takeaway}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>
              </>
            )}

            {/* Show prompt when custom is selected but no expression analyzed yet */}
            {isCustom && !analysis && (
              <div className="rounded-lg border border-dashed border-border bg-background px-5 py-10 text-center text-sm text-muted-foreground">
                Define your variables and einsum expression above, then click <strong className="font-semibold text-coral">Analyze</strong> to explore the symmetry detection algorithm.
              </div>
            )}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
