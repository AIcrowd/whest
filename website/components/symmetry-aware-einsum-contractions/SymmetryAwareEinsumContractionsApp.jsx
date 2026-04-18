import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Badge } from '@/components/ui/badge';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import { analyzeExample } from './engine/pipeline.js';
import { pickDefaultOrbitRow } from './engine/teachingModel.js';
import StickyBar from './components/StickyBar.jsx';
import ExplorerSectionCard from './components/ExplorerSectionCard.jsx';
import { EXPLORER_ACTS } from './components/explorerNarrative.js';
import NarrativeCallout from './components/NarrativeCallout.jsx';
import AlgorithmAtAGlance from './components/AlgorithmAtAGlance.jsx';
import ExampleChooser from './components/ExampleChooser.jsx';
import PresetSidebar from './components/PresetSidebar.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import DiminoView from './components/DiminoView.jsx';
import RoleBadge from './components/RoleBadge.jsx';
import ComponentCostView from './components/ComponentCostView.jsx';
import TotalCostView from './components/TotalCostView.jsx';
import { mergeObservedActEntries, pickTopVisibleAct } from './lib/activeAct.js';
import { getPresetControlSelection } from './lib/presetSelection.js';
import { useKeyboardShortcuts } from './lib/useKeyboardShortcuts.js';
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
  const [defaultSize, setDefaultSize] = useState(5);
  const [clusterSizes, setClusterSizes] = useState({}); // { [clusterId]: size }
  // Back-compat alias — many child components still take a single `dimensionN` prop.
  const dimensionN = defaultSize;
  const [selectedOrbitIdx, setSelectedOrbitIdx] = useState(-1);
  const [selectedSigmaPairIndex, setSelectedSigmaPairIndex] = useState(null);
  const [activeActId, setActiveActId] = useState(EXPLORER_ACTS[0].id);
  const [isDirty, setIsDirty] = useState(false);
  // Cross-highlight payload emitted by the Act-4 Interaction Graph on hover.
  // `labels` → halo those characters in the StickyBar einsum equation;
  // `leafKeys` → spotlight matching leaves in the DecisionLadder.
  const [graphHover, setGraphHover] = useState(null);
  const handleGraphHover = useCallback((payload) => setGraphHover(payload), []);
  const hoveredLabelSet = useMemo(
    () => (graphHover?.labels?.length ? new Set(graphHover.labels) : null),
    [graphHover],
  );
  const spotlightLeafSet = useMemo(
    () => (graphHover?.leafKeys?.length ? new Set(graphHover.leafKeys) : null),
    [graphHover],
  );
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
    setExampleIdx(idx);
    setPreviewExample(EXAMPLES[idx] ?? null);
    setIsDirty(false);
    setSelectedOrbitIdx(-1);
    setSelectedSigmaPairIndex(null);
    setClusterSizes({});
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setCustomExample(ex);
    setPreviewExample(ex);
    setExampleIdx(CUSTOM_IDX);
    setSelectedOrbitIdx(-1);
    setSelectedSigmaPairIndex(null);
    setClusterSizes({});
  }, []);

  const handleCustomMode = useCallback(() => {
    setExampleIdx(CUSTOM_IDX);
    setSelectedOrbitIdx(-1);
    setSelectedSigmaPairIndex(null);
    setClusterSizes({});
  }, []);

  const analysis = useMemo(() => {
    if (!example) return null;
    try {
      const normalized = normalizeExample(example);
      const examplePreset = normalized.labelSizes || {};
      const mergedLabelSizes = { ...examplePreset, ...clusterSizes };
      return analyzeExample({ ...normalized, labelSizes: mergedLabelSizes }, defaultSize);
    } catch (err) {
      console.error('Pipeline error:', err);
      return null;
    }
  }, [example, clusterSizes, defaultSize]);

  const {
    graph,
    matrixData,
    sigmaResults,
    symmetry: group,
    componentData,
    costModel: cost,
    componentCosts,
    clusters: analysisClusters,
  } = analysis || {};

  // Seed clusterSizes from analysis when clusters appear
  useEffect(() => {
    if (!analysis?.clusters) return;
    setClusterSizes((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const c of analysis.clusters) {
        if (next[c.id] == null) { next[c.id] = c.size; changed = true; }
      }
      return changed ? next : prev;
    });
  }, [analysis?.clusters]);

  const resolvedSelectedOrbitIdx = useMemo(() => {
    const orbitRows = cost?.orbitRows ?? [];
    if (orbitRows.length === 0) return -1;
    if (selectedOrbitIdx >= 0 && selectedOrbitIdx < orbitRows.length) return selectedOrbitIdx;
    return pickDefaultOrbitRow(orbitRows);
  }, [cost, selectedOrbitIdx]);

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

  useKeyboardShortcuts({
    ArrowLeft: () => {
      if (selectedPresetIdx == null) return;
      const target = Math.max(0, (selectedPresetIdx ?? 0) - 1);
      handleSelect(target);
    },
    ArrowRight: () => {
      if (selectedPresetIdx == null) return;
      const target = Math.min(EXAMPLES.length - 1, (selectedPresetIdx ?? 0) + 1);
      handleSelect(target);
    },
    r: () => handleSelect(Math.floor(Math.random() * EXAMPLES.length)),
  });

  return (
    <div className="min-h-screen bg-background">
      <StickyBar
        example={previewExample ?? example}
        group={group}
        activeActId={activeActId}
        hoveredLabels={hoveredLabelSet}
      />

      <div className="w-full pb-20 pt-8">
        <div className="mx-auto flex w-full max-w-[1460px] flex-col gap-2 px-6 pb-6 md:px-8 lg:px-10">
          <h1 className="text-lg font-semibold leading-tight text-foreground">
            Symmetry Aware Einsum Contractions
          </h1>
          <p className="text-sm leading-6 text-muted-foreground">
            <em>Symmetry detection for einsum contractions</em>, explained in five sections.
          </p>
        </div>

        <div className="mx-auto w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <AlgorithmAtAGlance />
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
                eyebrow="§ 1"
                title={EXPLORER_ACTS[0].heading}
                description={EXPLORER_ACTS[0].question}
                className="border-gray-200 bg-white"
                contentClassName="pt-5"
              >
                <div className="grid gap-4 md:grid-cols-2">
                  <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[0].interpretation}</NarrativeCallout>
                  <NarrativeCallout label="Approach" tone="algorithm">{EXPLORER_ACTS[0].algorithm}</NarrativeCallout>
                </div>
                <div className="mt-6">
                  <ExampleChooser
                    examples={EXAMPLES}
                    onSelect={handleSelect}
                    selectedPresetIdx={selectedPresetIdx}
                    dimensionN={dimensionN}
                    onDimensionChange={setDefaultSize}
                    onCustom={handleCustomMode}
                    onCustomExample={handleCustomExample}
                    onPreviewChange={setPreviewExample}
                    onDirtyChange={setIsDirty}
                  />
                </div>
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
                    eyebrow="§ 2"
                    title={EXPLORER_ACTS[1].heading}
                    description={EXPLORER_ACTS[1].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <div className="grid gap-4 md:grid-cols-2">
                      <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[1].interpretation}</NarrativeCallout>
                      <NarrativeCallout label="Approach" tone="algorithm">{EXPLORER_ACTS[1].algorithm}</NarrativeCallout>
                    </div>
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
                      <div className="grid grid-rows-[auto_1fr] gap-2">
                        <h3 className="font-heading text-base font-semibold text-gray-900">Bipartite Graph</h3>
                        <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} />
                      </div>
                      <div className="grid grid-rows-[auto_1fr] gap-2">
                        <h3 className="font-heading text-base font-semibold text-gray-900">Incidence Matrix M</h3>
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
                    eyebrow="§ 3"
                    title={EXPLORER_ACTS[2].heading}
                    description={EXPLORER_ACTS[2].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <div className="grid gap-4 md:grid-cols-2">
                      <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[2].interpretation}</NarrativeCallout>
                      <NarrativeCallout label="Approach" tone="algorithm">{EXPLORER_ACTS[2].algorithm}</NarrativeCallout>
                    </div>
                    {EXPLORER_ACTS[2].bridge && (
                      <p className="mt-4 text-sm leading-7 text-foreground">
                        {EXPLORER_ACTS[2].bridge}
                      </p>
                    )}
                    <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <div className="grid grid-rows-[auto_1fr] gap-2">
                        <h3 className="font-heading text-base font-semibold text-gray-900">σ-Loop & π Detection</h3>
                        <SigmaLoop
                          results={sigmaResults}
                          graph={graph}
                          matrixData={matrixData}
                          example={normalizedExample}
                          variableColors={variableColors}
                          group={group}
                          onSelectedPairChange={setSelectedSigmaPairIndex}
                        />
                      </div>
                      <div className="grid grid-rows-[auto_1fr] gap-2">
                        <h3 className="font-heading text-base font-semibold text-gray-900">Generator Construction</h3>
                        <DiminoView
                          group={group}
                          sigmaResults={sigmaResults}
                          selectedPairIndex={selectedSigmaPairIndex}
                        />
                      </div>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[2].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[3].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="§ 4"
                    title={EXPLORER_ACTS[3].heading}
                    description={EXPLORER_ACTS[3].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <div className="grid gap-4 md:grid-cols-2">
                      <NarrativeCallout label="Interpretation">{EXPLORER_ACTS[3].interpretation}</NarrativeCallout>
                      <NarrativeCallout label="Approach" tone="algorithm">{EXPLORER_ACTS[3].algorithm}</NarrativeCallout>
                    </div>
                    {EXPLORER_ACTS[3].bridge && (
                      <p className="mt-4 text-sm leading-7 text-foreground">
                        {EXPLORER_ACTS[3].bridge}
                      </p>
                    )}

                  <div className="mt-6">
                    <ComponentCostView
                      componentData={componentData}
                      costModel={cost}
                      dimensionN={dimensionN}
                      numTerms={normalizedExample?.subscripts?.length ?? 1}
                      allLabels={group.allLabels}
                      vLabels={group.vLabels}
                      fullGenerators={group.fullGenerators}
                      selectedOrbitIdx={resolvedSelectedOrbitIdx}
                      onSelectOrbit={setSelectedOrbitIdx}
                      onGraphHover={handleGraphHover}
                      spotlightLeafIds={spotlightLeafSet}
                    />
                  </div>

                  <div className="mt-4">
                    <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[3].produces}</NarrativeCallout>
                  </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[4].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow="§ 5"
                    title={EXPLORER_ACTS[4].heading}
                    description={EXPLORER_ACTS[4].question}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <p className="text-sm leading-7 text-foreground">
                      {EXPLORER_ACTS[4].why}
                    </p>

                    <div className="mt-6">
                      <TotalCostView
                        componentCosts={componentCosts}
                        componentData={componentData}
                        dimensionN={dimensionN}
                        numTerms={normalizedExample?.subscripts?.length ?? 1}
                      />
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
