import { useState, useMemo, useCallback, useEffect, useRef, useSyncExternalStore } from 'react';
import { Badge } from '@/components/ui/badge';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import { analyzeExample } from './engine/pipeline.js';
import { pickDefaultOrbitRow } from './engine/teachingModel.js';
import StickyBar from './components/StickyBar.jsx';
import ExpressionLevelModal from './components/ExpressionLevelModal.jsx';
import ExplorerSectionCard, { SectionEyebrow, AnchorLink } from './components/ExplorerSectionCard.jsx';
import ExplorerSubsectionHeader from './components/ExplorerSubsectionHeader.jsx';
import { EXPLORER_ACTS } from './components/explorerNarrative.js';
import NarrativeCallout from './components/NarrativeCallout.jsx';
import SectionIntroProse from './components/SectionIntroProse.jsx';
import InlineMathText from './components/InlineMathText.jsx';
import Latex from './components/Latex.jsx';
import AlgorithmAtAGlance from './components/AlgorithmAtAGlance.jsx';
import ExampleChooser from './components/ExampleChooser.jsx';
import ExplorerThemeDock from './components/ExplorerThemeDock.jsx';
import PresetSidebar from './components/PresetSidebar.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import DiminoView from './components/DiminoView.jsx';
import WreathStructureView from './components/WreathStructureView.jsx';
import ComponentCostView from './components/ComponentCostView.jsx';
import TotalCostView from './components/TotalCostView.jsx';
import { mergeObservedActEntries, pickTopVisibleAct } from './lib/activeAct.js';
import {
  getExplorerThemeCssVariables,
  EXPLORER_THEME_RECOMMENDED_ID,
  getActiveExplorerThemeId,
  getExplorerThemePreset,
  resetActiveExplorerTheme,
  setActiveExplorerTheme,
  subscribeActiveExplorerTheme,
} from './lib/explorerTheme.js';
import { getPresetControlSelection } from './lib/presetSelection.js';
import {
  notationLatex,
} from './lib/notationSystem.js';
import { useKeyboardShortcuts } from './lib/useKeyboardShortcuts.js';
import './styles.css';

const CUSTOM_IDX = -1;
const DEFAULT_EXAMPLE_ID = 'triple-outer';
const DEFAULT_EXAMPLE_IDX = Math.max(0, EXAMPLES.findIndex((example) => example.id === DEFAULT_EXAMPLE_ID));
const DEFAULT_DIMENSION_N = 5;

function presetDefaultSize(example) {
  const sizes = Object.values(example?.labelSizes ?? {}).filter(
    (value) => Number.isFinite(value) && value > 0,
  );
  if (sizes.length === 0) return DEFAULT_DIMENSION_N;
  return Math.max(...sizes);
}

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
  const [defaultSize, setDefaultSize] = useState(() => presetDefaultSize(EXAMPLES[DEFAULT_EXAMPLE_IDX]));
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
  const [exprModalOpen, setExprModalOpen] = useState(false);
  const [isThemeDockVisible, setIsThemeDockVisible] = useState(false);
  const explorerThemeId = useSyncExternalStore(
    subscribeActiveExplorerTheme,
    getActiveExplorerThemeId,
    () => EXPLORER_THEME_RECOMMENDED_ID,
  );
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
  const theme = useMemo(() => getExplorerThemePreset(explorerThemeId), [explorerThemeId]);
  const themeCssVars = useMemo(() => getExplorerThemeCssVariables(theme), [theme]);

  // Derive variable colors from the example's variables (works for both presets and custom)
  const variableColors = useMemo(() => {
    if (example?.variables) return buildVariableColors(example.variables, theme.id);
    return {};
  }, [example, theme.id]);

  // Normalize the example for algorithm consumption
  const normalizedExample = useMemo(() => example ? normalizeExample(example) : null, [example]);
  const preambleExample = useMemo(() => {
    if (!normalizedExample) return null;
    return {
      ...normalizedExample,
      labelSizes: {
        ...(normalizedExample.labelSizes || {}),
        ...clusterSizes,
      },
    };
  }, [normalizedExample, clusterSizes]);

  // Handle preset selection
  const handleSelect = useCallback((idx) => {
    setExampleIdx(idx);
    setPreviewExample(EXAMPLES[idx] ?? null);
    if (EXAMPLES[idx]) setDefaultSize(presetDefaultSize(EXAMPLES[idx]));
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

  useEffect(() => {
    resetActiveExplorerTheme();
    return () => resetActiveExplorerTheme();
  }, []);

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

  useKeyboardShortcuts([
    {
      key: 'ArrowLeft',
      handler: () => {
        if (selectedPresetIdx == null) return;
        const target = Math.max(0, (selectedPresetIdx ?? 0) - 1);
        handleSelect(target);
      },
    },
    {
      key: 'ArrowRight',
      handler: () => {
        if (selectedPresetIdx == null) return;
        const target = Math.min(EXAMPLES.length - 1, (selectedPresetIdx ?? 0) + 1);
        handleSelect(target);
      },
    },
    {
      key: 'r',
      handler: () => handleSelect(Math.floor(Math.random() * EXAMPLES.length)),
    },
    {
      key: 'E',
      modifiers: { ctrlKey: true, shiftKey: true },
      handler: () => setIsThemeDockVisible((visible) => !visible),
    },
  ]);

  return (
    <div className="symmetry-aware-einsum-explorer min-h-screen bg-background" style={themeCssVars}>
      <StickyBar
        example={previewExample ?? example}
        group={group}
        activeActId={activeActId}
        hoveredLabels={hoveredLabelSet}
        dimensionN={dimensionN}
      />
      {isThemeDockVisible ? (
        <ExplorerThemeDock explorerThemeId={explorerThemeId} onChange={setActiveExplorerTheme} />
      ) : null}

      <div className="w-full pb-20 pt-10">
        {/* Editorial masthead — matches the docs home page register
            (app/(home)/page.tsx). Uppercase kicker with leading rule +
            Newsreader display-serif headline ending in a coral period +
            Source Serif 4 italic lede. The same typographic rhythm the
            reader sees at aicrowd.github.io/whest/ carries directly
            into the explorer so the two pages feel like one product. */}
        <header className="mx-auto flex w-full max-w-[1460px] flex-col px-6 pb-10 md:px-8 lg:px-10">
          <div
            className="mb-5 font-sans text-[10px] font-semibold uppercase text-gray-400"
            style={{ letterSpacing: '0.2em' }}
          >
            <span aria-hidden className="mr-2 inline-block h-px w-8 align-middle bg-gray-300" />
            An interactive walkthrough
          </div>

          <h1
            className="m-0 font-semibold text-gray-900 dark:text-gray-100"
            style={{
              fontFamily: 'var(--font-display-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 72",
              fontSize: 'clamp(36px, 5vw, 52px)',
              letterSpacing: '-0.02em',
              lineHeight: 1.05,
            }}
          >
            Symmetry-aware einsum contractions<span style={{ color: 'var(--coral)' }}>.</span>
          </h1>

          <p
            className="mt-5 max-w-[var(--prose-max)] text-[17px] italic text-gray-600 dark:text-gray-300"
            style={{
              fontFamily: 'var(--font-paper-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 18",
              lineHeight: 1.6,
            }}
          >
            Given a tensor contraction written in einsum notation, this explorer constructs
            candidate relabelings, accepts the lifted relabelings used by the cost model, and
            then counts the product orbits and output-bin updates required by the resulting
            symmetry-aware direct computation. After analysis, the visualizations update as
            the contraction, declared symmetries, and label sizes change.
          </p>
        </header>

        <div className="mx-auto w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <AlgorithmAtAGlance example={preambleExample} />
        </div>

        <div className="mx-auto mb-8 mt-[-0.25rem] w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <div className="border-y border-stone-200 bg-white px-5 py-4">
            <div className="font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-coral">
              Scope of the calculation
            </div>
            <p
              className="mt-2 font-serif text-[15px] leading-7 text-stone-700"
              style={{ textAlign: 'justify' }}
            >
              <InlineMathText>
                {`This page counts the direct indexed computation under the detected pointwise permutation symmetries used by the model. A detected symmetry may move free labels; when it does, it can reduce the number of distinct products while still requiring separate updates to each output component touched by the orbit. For that reason the cost model uses orbit projections, not a naive division by the group order.`}
              </InlineMathText>
            </p>
            <p className="mt-2 text-[12.5px] leading-6 text-stone-600">
              Assumptions: scalar commutative exact arithmetic; declared equality
              symmetries only; shared sizes along permuted label orbits; operand identity
              determined by the selected expression. Contraction-path optimizations are outside this
              calculation.
            </p>
          </div>
        </div>

        <div className="mx-auto mt-8 w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <div className="flex items-start gap-8">
            <PresetSidebar
              examples={EXAMPLES}
              selectedPresetIdx={selectedPresetIdx}
              onSelect={handleSelect}
              onCustom={handleCustomMode}
            />
            <main className="min-w-0 flex-1">
              <div className="flex flex-col">
            <section id={EXPLORER_ACTS[0].id} className="mb-12 scroll-mt-24">
              <ExplorerSectionCard
                eyebrow={<SectionEyebrow n={1} anchorId={EXPLORER_ACTS[0].id} />}
                title={EXPLORER_ACTS[0].heading}
                description={<InlineMathText>{EXPLORER_ACTS[0].question}</InlineMathText>}
                className="border-gray-200 bg-white"
                contentClassName="pt-5"
              >
                <SectionIntroProse paragraphs={EXPLORER_ACTS[0].introParagraphs} />
                <div className="mt-6">
                  <ExampleChooser
                    examples={EXAMPLES}
                    onSelect={handleSelect}
                    selectedPresetIdx={selectedPresetIdx}
                    dimensionN={dimensionN}
                    explorerThemeId={explorerThemeId}
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
                    eyebrow={<SectionEyebrow n={2} anchorId={EXPLORER_ACTS[1].id} />}
                    title={EXPLORER_ACTS[1].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[1].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse paragraphs={EXPLORER_ACTS[1].introParagraphs} />
                    <div className="editorial-two-col-divider-md mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
                      <div id="bipartite-graph" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-24">
                        <ExplorerSubsectionHeader anchorId="bipartite-graph" labelText="Bipartite Graph">
                          Bipartite Graph
                        </ExplorerSubsectionHeader>
                        <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} />
                      </div>
                      <div id="incidence-matrix" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-24">
                        <ExplorerSubsectionHeader anchorId="incidence-matrix" labelText="Incidence Matrix">
                          Incidence Matrix M
                        </ExplorerSubsectionHeader>
                        <MatrixView matrixData={matrixData} graph={graph} example={normalizedExample} variableColors={variableColors} />
                      </div>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout tone="preamble">
                        <p className="text-[14px] leading-7 text-foreground" style={{ textAlign: 'justify' }}>
                          <InlineMathText>
                            {`The graph and incidence matrix describe which relabelings are structurally plausible. A relabeling becomes a detected symmetry only in the next section, after it is lifted through operand identity and declared slot symmetries and accepted by the $${notationLatex('sigma_row_move')}$-loop used by this model.`}
                          </InlineMathText>
                        </p>
                      </NarrativeCallout>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[1].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[2].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={3} anchorId={EXPLORER_ACTS[2].id} />}
                    title={EXPLORER_ACTS[2].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[2].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse paragraphs={EXPLORER_ACTS[2].introParagraphs} />
                    {/* Wreath structure renders full-width — the enumeration target the σ-loop walks over. */}
                    <div id="wreath-structure" className="mt-6 flex flex-col gap-2 scroll-mt-24">
                      <ExplorerSubsectionHeader anchorId="wreath-structure" labelText="Wreath structure">
                        Wreath structure
                      </ExplorerSubsectionHeader>
                      <WreathStructureView
                        analysis={analysis}
                        example={normalizedExample}
                      />
                    </div>
                    {/* σ-Loop (enumerates the wreath) + Generator Construction (closes valid π's). */}
                    <div className="editorial-two-col-divider-lg mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <div id="sigma-loop" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-24">
                        <ExplorerSubsectionHeader anchorId="sigma-loop" labelText="σ-Loop & π Detection">
                          <InlineMathText>
                            {`$${notationLatex('sigma_row_move')}$-Loop & $${notationLatex('pi_relabeling')}$ Detection`}
                          </InlineMathText>
                        </ExplorerSubsectionHeader>
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
                      <div id="generator-construction" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-24">
                        <ExplorerSubsectionHeader anchorId="generator-construction" labelText="Generator Construction">
                          Generator Construction
                        </ExplorerSubsectionHeader>
                        <DiminoView
                          group={group}
                          sigmaResults={sigmaResults}
                          selectedPairIndex={selectedSigmaPairIndex}
                        />
                      </div>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout tone="preamble">
                        <p className="text-[14px] leading-7 text-foreground" style={{ textAlign: 'justify' }}>
                          <InlineMathText>
                            {`The accepted objects are lifted pairs: a row move $${notationLatex('sigma_row_move')}$ together with a label relabeling $${notationLatex('pi_relabeling')}$. The generated group is built from those accepted relabelings. The expression-level formal group discussed in the appendix is deliberately kept separate from this pointwise group and is not used for multiplication compression.`}
                          </InlineMathText>
                        </p>
                      </NarrativeCallout>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[2].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                <section id={EXPLORER_ACTS[3].id} className="mb-12 scroll-mt-24">
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={4} anchorId={EXPLORER_ACTS[3].id} />}
                    title={EXPLORER_ACTS[3].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[3].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse paragraphs={EXPLORER_ACTS[3].introParagraphs} />
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
                    eyebrow={<SectionEyebrow n={5} anchorId={EXPLORER_ACTS[4].id} />}
                    title={EXPLORER_ACTS[4].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[4].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <TotalCostView
                      componentCosts={componentCosts}
                      componentData={componentData}
                      dimensionN={dimensionN}
                      numTerms={normalizedExample?.subscripts?.length ?? 1}
                      explorerThemeId={explorerThemeId}
                    />

                    <button
                      type="button"
                      onClick={() => setExprModalOpen(true)}
                      className="-mx-4 -mb-4 mt-8 block w-auto cursor-pointer border-t border-stone-200/70 bg-gray-50 px-4 py-4 text-left transition-colors hover:bg-stone-100/70 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-coral"
                    >
                      <span className="block font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
                        Appendix note
                      </span>
                      <span className="mt-1.5 block font-serif text-[15px] leading-7 text-stone-900">
                        Is this the full symmetry of the final expression?
                      </span>
                      <span className="mt-1.5 block text-[12.5px] leading-6 text-stone-700">
                        The cost above uses <Latex math={notationLatex('g_pointwise')} /> for accumulation. The fully summed expression can have a larger label-renaming formal symmetry, <Latex math={`${notationLatex('g_formal')} = ${notationLatex('g_output')} \\times ${notationLatex('s_w_summed')}`} />. That larger group cannot reduce the accumulation count, but its visible output part <Latex math={notationLatex('g_output')} /> can still reduce output storage.
                      </span>
                    </button>
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

      <ExpressionLevelModal
        isOpen={exprModalOpen}
        onClose={() => setExprModalOpen(false)}
        analysis={analysis}
        group={group}
        example={example}
        onSelectPreset={handleSelect}
      />
    </div>
  );
}
