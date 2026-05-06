import { useState, useMemo, useCallback, useEffect, useRef, useSyncExternalStore, useTransition } from 'react';
import { Badge } from '@/components/ui/badge';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import { analyzeExample } from './engine/pipeline.js';
import { pickDefaultOrbitRow } from './engine/teachingModel.js';
import { restrictStabilizerToPositions } from './engine/outputOrbit.js';
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
import AnalysisLoadingBoundary from './components/AnalysisLoadingBoundary.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import DiminoView from './components/DiminoView.jsx';
import WreathStructureView from './components/WreathStructureView.jsx';
import ComponentCostView from './components/ComponentCostView.jsx';
import BranchingDemo from './components/BranchingDemo.jsx';
import DecisionLadder from './components/DecisionLadder.jsx';
import { LabelInteractionGraph } from './components/ComponentView.jsx';
import TypedPartitionDemo from './components/TypedPartitionDemo.jsx';
import TwoQuotientSchematic from './components/TwoQuotientSchematic.jsx';
import TotalCostView from './components/TotalCostView.jsx';
import NaiveAlphaCostMeter from './components/NaiveAlphaCostMeter.jsx';
import TuplePatternMeter from './components/TuplePatternMeter.jsx';
import CertificationSummaryStrip from './components/CertificationSummaryStrip.jsx';
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
import { selectSection1PreambleExample } from './lib/section1ExampleView.js';
import { useKeyboardShortcuts } from './lib/useKeyboardShortcuts.js';
import { setActiveAlphaMethodBus } from './lib/alphaMethodBus.js';
import './styles.css';

const CUSTOM_IDX = -1;
const DEFAULT_EXAMPLE_ID = 'cross-s2';
const DEFAULT_EXAMPLE_IDX = Math.max(0, EXAMPLES.findIndex((example) => example.id === DEFAULT_EXAMPLE_ID));
const DEFAULT_DIMENSION_N = 5;
const APPENDIX_ROOT_HASH = '#appendix';
const APPENDIX_SECTION_HASH_PREFIX = '#appendix-section-';
const APPENDIX_RETURN_HASH = '#assemble-cost';
const APPENDIX_MAP = [
  { letter: 'A', title: 'Product-side certification', hash: '#appendix-section-1' },
  { letter: 'B', title: 'Classification-tree cases', hash: '#appendix-section-7' },
  { letter: 'C', title: 'Typed partition theorem', hash: '#appendix-section-6' },
  { letter: 'D', title: 'Completed-expression formal symmetry', hash: '#appendix-section-4' },
  { letter: 'E', title: 'Scope, assumptions, and exactness', hash: '#appendix-section-8' },
];
const PROJECTION_ALPHA_FORMULA = String.raw`\alpha = \#\{(O,Q) \in X/G_{\mathrm{pt}} \times Y/H : ${notationLatex('projection_pi_v_free')} \cap Q \neq \varnothing\}`;
const CROSS_S2_CONTRACTION_FORMULA = String.raw`R[i,k] = \sum_j A[i,j]B[k]`;
const CROSS_S2_OPERANDS_FORMULA = String.raw`A = \begin{bmatrix}1 & 2 \\ 2 & 4\end{bmatrix},\qquad B = \begin{bmatrix}5 \\ 7\end{bmatrix}`;
const CROSS_S2_ORBIT_FORMULA = String.raw`O = \{(0,1,0), (1,0,0)\}`;
const CROSS_S2_PRODUCT_EQUALITY_FORMULA = String.raw`\begin{aligned}A[0,1]B[0] &= 2 \cdot 5 = 10\\ A[1,0]B[0] &= 2 \cdot 5 = 10\end{aligned}`;
const CROSS_S2_PROJECTION_DESTINATIONS_FORMULA = String.raw`(0,1,0) \mapsto R[0,0],\qquad (1,0,0) \mapsto R[1,0]`;
const PROJECTION_MATRIX_BRIDGE = 'The matrix below is just this incidence test drawn out: scan a product-orbit row, mark every stored-output column it reaches, and count the marks.';

function ProjectionIntroProse({ paragraphs }) {
  const leftParagraphs = paragraphs.slice(0, 2);
  const rightParagraphs = paragraphs.slice(2);
  const paragraphClassName = 'font-serif text-[17px] leading-[1.75] text-gray-700';
  const formulaClassName = 'projection-example-math overflow-x-auto whitespace-nowrap text-[18px]';
  const renderParagraph = (paragraph, index) => (
    <p
      key={index}
      className={paragraphClassName}
      style={{ textAlign: 'justify' }}
    >
      <InlineMathText>{paragraph}</InlineMathText>
    </p>
  );
  const renderDisplayFormula = (math, className = formulaClassName) => (
    <div className={className}>
      <div className="min-w-max">
        <Latex math={math} display />
      </div>
    </div>
  );

  return (
    <div className="editorial-two-col-divider-md grid gap-x-8 gap-y-4 md:grid-cols-2">
      <div className="space-y-4 md:px-4">
        {leftParagraphs.map(renderParagraph)}
        {renderDisplayFormula(PROJECTION_ALPHA_FORMULA, `mt-3 ${formulaClassName}`)}
        <p className={paragraphClassName} style={{ textAlign: 'justify' }}>
          <InlineMathText>{PROJECTION_MATRIX_BRIDGE}</InlineMathText>
        </p>
      </div>
      <div className="space-y-3 md:px-4">
        <p className="text-[15px] font-semibold leading-7 text-gray-900">
          Worked example — <span className="font-semibold">Cross S2</span>
        </p>
        {renderParagraph(rightParagraphs[0], 0)}
        {renderDisplayFormula(CROSS_S2_CONTRACTION_FORMULA)}
        <p className={paragraphClassName} style={{ textAlign: 'justify' }}>
          <InlineMathText>{'Let $A$ and $B$ be concrete operands:'}</InlineMathText>
        </p>
        {renderDisplayFormula(CROSS_S2_OPERANDS_FORMULA)}
        <p className={paragraphClassName} style={{ textAlign: 'justify' }}>
          <InlineMathText>{'Now look at the product orbit'}</InlineMathText>
        </p>
        {renderDisplayFormula(CROSS_S2_ORBIT_FORMULA)}
        <p className={paragraphClassName} style={{ textAlign: 'justify' }}>
          <InlineMathText>{'The two assignments give the same scalar product:'}</InlineMathText>
        </p>
        {renderDisplayFormula(CROSS_S2_PRODUCT_EQUALITY_FORMULA)}
        <p className={paragraphClassName} style={{ textAlign: 'justify' }}>
          <InlineMathText>{'So they can share one representative product. But projection keeps only $(i,k)$, so the same two assignments land in different output entries:'}</InlineMathText>
        </p>
        {renderDisplayFormula(CROSS_S2_PROJECTION_DESTINATIONS_FORMULA)}
        {renderParagraph(rightParagraphs[1], 1)}
      </div>
    </div>
  );
}

function isAppendixHash(hash = '') {
  return hash === APPENDIX_ROOT_HASH || hash.startsWith(APPENDIX_SECTION_HASH_PREFIX);
}

function scrollToHashTarget(hash) {
  if (typeof document === 'undefined') return;
  if (hash === APPENDIX_ROOT_HASH) {
    document.getElementById('expr-modal-heading')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  if (!hash?.startsWith('#')) return;
  document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function presetDefaultSize(example) {
  const sizes = Object.values(example?.labelSizes ?? {}).filter(
    (value) => Number.isFinite(value) && value > 0,
  );
  if (sizes.length === 0) return DEFAULT_DIMENSION_N;
  return Math.max(...sizes);
}

function buildAnalysisCacheKey(example, dimensionN, clusterSizes = {}) {
  if (!example) return '';
  const normalized = normalizeExample(example);
  const mergedLabelSizes = { ...(normalized.labelSizes || {}), ...clusterSizes };
  return JSON.stringify({
    id: normalized.id ?? null,
    subscripts: normalized.subscripts ?? normalized.expression?.subscripts ?? null,
    output: normalized.output ?? normalized.expression?.output ?? null,
    operandNames: normalized.operandNames ?? normalized.expression?.operandNames ?? null,
    perOpSymmetry: normalized.perOpSymmetry ?? null,
    variables: normalized.variables ?? null,
    labelSizes: mergedLabelSizes,
    dimensionN,
  });
}

function computeAnalysis(example, dimensionN, clusterSizes = {}) {
  const normalized = normalizeExample(example);
  const examplePreset = normalized.labelSizes || {};
  const mergedLabelSizes = { ...examplePreset, ...clusterSizes };
  return analyzeExample({ ...normalized, labelSizes: mergedLabelSizes }, dimensionN);
}

function initialAnalysisState(example, dimensionN) {
  try {
    const cacheKey = buildAnalysisCacheKey(example, dimensionN, {});
    return {
      analysis: computeAnalysis(example, dimensionN, {}),
      example,
      dimensionN,
      status: 'ready',
      pendingSelection: null,
      error: null,
      cacheKey,
    };
  } catch (err) {
    console.error('Pipeline error:', err);
    return {
      analysis: null,
      example,
      dimensionN,
      status: 'error',
      pendingSelection: null,
      error: err,
      cacheKey: '',
    };
  }
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
  const [analysisState, setAnalysisState] = useState(() => (
    initialAnalysisState(EXAMPLES[DEFAULT_EXAMPLE_IDX], presetDefaultSize(EXAMPLES[DEFAULT_EXAMPLE_IDX]))
  ));
  const [isCommitPending, startCommitTransition] = useTransition();
  const analysisCacheRef = useRef(new Map());
  const selectionTimerRef = useRef(null);
  const analysisTimerRef = useRef(null);
  const analysisRequestIdRef = useRef(0);
  const [clusterSizes, setClusterSizes] = useState({}); // { [clusterId]: size }
  // Back-compat alias — many child components still take a single `dimensionN` prop.
  const dimensionN = defaultSize;
  const [selectedOrbitIdx, setSelectedOrbitIdx] = useState(-1);
  const [selectedSigmaPairIndex, setSelectedSigmaPairIndex] = useState(null);
  const [activeActId, setActiveActId] = useState(EXPLORER_ACTS[0].id);
  const [isDirty, setIsDirty] = useState(false);
  // Cross-highlight payload emitted by the Certification section Interaction Graph on hover.
  // `labels` → halo those characters in the StickyBar einsum equation;
  // `leafKeys` → spotlight matching leaves in the DecisionLadder.
  const [graphHover, setGraphHover] = useState(null);
  // Bidirectional label-hover bus: StickyBar can also emit hovered labels
  // (Behavior 1 — C01). When the strip fires onHoveredLabelsChange, we merge
  // it into the shared hoveredLabelSet. The graphHover payload from the
  // Interaction Graph takes precedence when both are active.
  const [stripHoveredLabels, setStripHoveredLabels] = useState(null);
  // α-method hover state (Behavior 3 — C01): StickyBar emits method id on
  // badge hover; classification tree reads this to spotlight the matching leaf.
  const [activeAlphaMethodHover, setActiveAlphaMethodHover] = useState(null);
  // Hover-component bus (Gap 3 — V3.1 §C20): LabelInteractionGraph emits
  // activeComponentId on hull hover; future C29 classification tree subscribes.
  const [activeComponentId, setActiveComponentId] = useState(null);
  const [exprModalOpen, setExprModalOpen] = useState(false);
  const [isThemeDockVisible, setIsThemeDockVisible] = useState(false);
  const appendixReturnHashRef = useRef(APPENDIX_RETURN_HASH);
  const explorerThemeId = useSyncExternalStore(
    subscribeActiveExplorerTheme,
    getActiveExplorerThemeId,
    () => EXPLORER_THEME_RECOMMENDED_ID,
  );
  const handleGraphHover = useCallback((payload) => setGraphHover(payload), []);
  // Merge graph hover + strip hover — graph takes precedence
  const hoveredLabelSet = useMemo(() => {
    if (graphHover?.labels?.length) return new Set(graphHover.labels);
    if (stripHoveredLabels instanceof Set && stripHoveredLabels.size > 0) return stripHoveredLabels;
    return null;
  }, [graphHover, stripHoveredLabels]);
  const spotlightLeafSet = useMemo(
    () => (graphHover?.leafKeys?.length ? new Set(graphHover.leafKeys) : null),
    [graphHover],
  );
  // Sync activeAlphaMethodHover → alphaMethodBus so DecisionLadder (rendered
  // inside ComponentCostView without a direct prop chain) can subscribe to it.
  useEffect(() => {
    setActiveAlphaMethodBus(activeAlphaMethodHover);
  }, [activeAlphaMethodHover]);
  const observedEntriesRef = useRef(new Map());

  // Resolve the active example: preset or custom
  const isCustom = exampleIdx === CUSTOM_IDX;
  const example = isCustom ? customExample : EXAMPLES[exampleIdx];
  const selectedPresetIdx = getPresetControlSelection(exampleIdx, isDirty);
  const pendingSelection = analysisState.pendingSelection;
  const pendingPresetIdx = pendingSelection?.kind === 'preset' || pendingSelection?.kind === 'custom'
    ? pendingSelection.idx
    : null;
  const analysisUpdating = analysisState.status === 'updating' || isCommitPending;
  const localAnalysisLoading = analysisUpdating && Boolean(pendingSelection);
  const theme = useMemo(() => getExplorerThemePreset(explorerThemeId), [explorerThemeId]);
  const themeCssVars = useMemo(() => getExplorerThemeCssVariables(theme), [theme]);

  // Derive variable colors from the example's variables (works for both presets and custom)
  const variableColors = useMemo(() => {
    const sourceExample = analysisState.example ?? example;
    if (sourceExample?.variables) return buildVariableColors(sourceExample.variables, theme.id);
    return {};
  }, [analysisState.example, example, theme.id]);

  // Normalize the example for algorithm consumption
  const normalizedExample = useMemo(() => example ? normalizeExample(example) : null, [example]);
  const normalizedAnalysisExample = useMemo(
    () => (analysisState.example ? normalizeExample(analysisState.example) : null),
    [analysisState.example],
  );

  // Handle preset selection
  const queuePresetCommit = useCallback((idx) => {
    if (selectionTimerRef.current) clearTimeout(selectionTimerRef.current);
    selectionTimerRef.current = setTimeout(() => {
      selectionTimerRef.current = null;
      const nextExample = EXAMPLES[idx];
      startCommitTransition(() => {
        setExampleIdx(idx);
        setPreviewExample(nextExample ?? null);
        if (nextExample) setDefaultSize(presetDefaultSize(nextExample));
        setIsDirty(false);
        setSelectedOrbitIdx(-1);
        setSelectedSigmaPairIndex(null);
        setClusterSizes({});
      });
    }, 80);
  }, [startCommitTransition]);

  // Handle preset selection
  const handleSelect = useCallback((idx) => {
    const nextExample = EXAMPLES[idx] ?? null;
    if (!nextExample) return;
    setAnalysisState((prev) => ({
      ...prev,
      status: 'updating',
      pendingSelection: {
        kind: 'preset',
        idx,
        name: nextExample.name ?? 'selected preset',
      },
      error: null,
    }));
    queuePresetCommit(idx);
  }, [queuePresetCommit]);

  const commitCustomExample = useCallback((ex) => {
    if (selectionTimerRef.current) clearTimeout(selectionTimerRef.current);
    selectionTimerRef.current = setTimeout(() => {
      selectionTimerRef.current = null;
      startCommitTransition(() => {
        setCustomExample(ex);
        setPreviewExample(ex);
        setExampleIdx(CUSTOM_IDX);
        setSelectedOrbitIdx(-1);
        setSelectedSigmaPairIndex(null);
        setClusterSizes({});
      });
    }, 80);
  }, [startCommitTransition]);

  const commitCustomMode = useCallback(() => {
    if (selectionTimerRef.current) clearTimeout(selectionTimerRef.current);
    selectionTimerRef.current = setTimeout(() => {
      selectionTimerRef.current = null;
      startCommitTransition(() => {
        setExampleIdx(CUSTOM_IDX);
        setSelectedOrbitIdx(-1);
        setSelectedSigmaPairIndex(null);
        setClusterSizes({});
      });
    }, 80);
  }, [startCommitTransition]);

  const handleDimensionChange = useCallback((nextSize) => {
    setDefaultSize(nextSize);
    setAnalysisState((prev) => ({
      ...prev,
      status: 'updating',
      pendingSelection: {
        kind: 'dimension',
        name: `n = ${nextSize}`,
      },
      error: null,
    }));
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setAnalysisState((prev) => ({
      ...prev,
      status: 'updating',
      pendingSelection: {
        kind: 'custom',
        idx: CUSTOM_IDX,
        name: ex?.name ?? 'Custom',
      },
      error: null,
    }));
    commitCustomExample(ex);
  }, [commitCustomExample]);

  const handleCustomMode = useCallback(() => {
    setAnalysisState((prev) => ({
      ...prev,
      status: 'updating',
      pendingSelection: {
        kind: 'custom',
        idx: CUSTOM_IDX,
        name: 'Custom',
      },
      error: null,
    }));
    commitCustomMode();
  }, [commitCustomMode]);

  const analysisInputKey = useMemo(
    () => buildAnalysisCacheKey(example, defaultSize, clusterSizes),
    [example, defaultSize, clusterSizes],
  );

  useEffect(() => {
    if (!example) return undefined;
    if (analysisState.cacheKey === analysisInputKey) return undefined;

    if (analysisTimerRef.current) clearTimeout(analysisTimerRef.current);
    const requestId = analysisRequestIdRef.current + 1;
    analysisRequestIdRef.current = requestId;

    setAnalysisState((prev) => ({
      ...prev,
      status: prev.analysis ? 'updating' : 'ready',
      pendingSelection: prev.pendingSelection,
      error: null,
    }));

    analysisTimerRef.current = setTimeout(() => {
      try {
        let nextAnalysis = analysisCacheRef.current.get(analysisInputKey);
        if (!nextAnalysis) {
          nextAnalysis = computeAnalysis(example, defaultSize, clusterSizes);
          analysisCacheRef.current.set(analysisInputKey, nextAnalysis);
        }
        if (analysisRequestIdRef.current !== requestId) return;
        startCommitTransition(() => {
          setAnalysisState({
            analysis: nextAnalysis,
            example,
            dimensionN: defaultSize,
            status: 'ready',
            pendingSelection: null,
            error: null,
            cacheKey: analysisInputKey,
          });
        });
      } catch (err) {
        if (analysisRequestIdRef.current !== requestId) return;
        console.error('Pipeline error:', err);
        setAnalysisState((prev) => ({
          ...prev,
          status: 'error',
          pendingSelection: null,
          error: err,
        }));
      }
    }, 0);

    return () => {
      if (analysisTimerRef.current) {
        clearTimeout(analysisTimerRef.current);
        analysisTimerRef.current = null;
      }
    };
  }, [analysisInputKey, analysisState.cacheKey, clusterSizes, defaultSize, example, exampleIdx, isCustom, startCommitTransition]);

  useEffect(() => () => {
    if (selectionTimerRef.current) clearTimeout(selectionTimerRef.current);
    if (analysisTimerRef.current) clearTimeout(analysisTimerRef.current);
  }, []);

  const analysis = analysisState.analysis;
  const analysisExample = analysisState.example;
  const analysisDimensionN = analysisState.dimensionN ?? dimensionN;

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

  const preambleExample = useMemo(() => {
    return selectSection1PreambleExample({
      example: normalizedExample,
      previewExample,
      isDirty,
      analysisClusters,
      defaultSize,
    });
  }, [normalizedExample, previewExample, isDirty, analysisClusters, defaultSize]);

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

  const outputActionSize = useMemo(() => {
    const labels = group?.allLabels ?? [];
    const vLabels = group?.vLabels ?? [];
    const elements = group?.fullElements ?? [];
    if (elements.length === 0) return 1;

    const positionByLabel = new Map(labels.map((label, idx) => [label, idx]));
    const visiblePositions = vLabels.map((label) => positionByLabel.get(label));
    if (visiblePositions.some((position) => !Number.isInteger(position))) return 1;

    return restrictStabilizerToPositions(elements, visiblePositions).length;
  }, [group]);

  const twoQuotientCurrent = useMemo(() => {
    const orbitRows = cost?.orbitRows ?? [];
    const outputKeys = new Set();
    for (const row of orbitRows) {
      for (const output of row?.outputs ?? []) {
        outputKeys.add(output?.outKey ?? JSON.stringify(output?.outTuple ?? output));
      }
    }
    const branchRows = orbitRows.filter((row) => (
      (row?.outputCount ?? row?.outputs?.length ?? 0) > 1
    )).length;
    const labelOrder = group?.allLabels ?? [];
    const visibleLabels = group?.vLabels ?? [];
    const denseAssignmentCount = Number.isFinite(analysisDimensionN) && labelOrder.length > 0
      ? Math.pow(analysisDimensionN, labelOrder.length)
      : null;
    return {
      presetName: analysisExample?.name ?? 'custom contraction',
      orbitRows,
      labelOrder,
      visibleLabels,
      denseAssignmentCount,
      rowCount: cost?.orbitCount ?? orbitRows.length,
      columnCount: outputKeys.size,
      alpha: cost?.reductionCostExact,
      branchRows,
      hSize: outputActionSize,
      dimensionN: analysisDimensionN,
    };
  }, [analysisDimensionN, analysisExample?.name, cost, group, outputActionSize]);

  const activeLeafIds = useMemo(() => {
    return (componentData?.components ?? [])
      .flatMap((component) => [component.accumulation?.regimeId, component.shape])
      .filter(Boolean);
  }, [componentData]);

  const liveReasonsByLeaf = useMemo(() => {
    const map = new Map();
    for (const component of componentData?.components ?? []) {
      const trace = component.accumulation?.trace ?? [];
      for (const step of trace) {
        if (!step?.regimeId || !step?.reason) continue;
        const list = map.get(step.regimeId) ?? [];
        if (!list.includes(step.reason)) list.push(step.reason);
        map.set(step.regimeId, list);
      }
    }
    return map;
  }, [componentData]);

  useEffect(() => {
    resetActiveExplorerTheme();
    return () => resetActiveExplorerTheme();
  }, []);

  const openAppendix = useCallback((hash = APPENDIX_ROOT_HASH) => {
    if (typeof window !== 'undefined') {
      const currentHash = window.location.hash;
      if (currentHash && !isAppendixHash(currentHash)) {
        appendixReturnHashRef.current = currentHash;
      }
      window.history.replaceState(null, '', hash);
    }
    setExprModalOpen(true);
    if (typeof window !== 'undefined') {
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          scrollToHashTarget(hash);
        });
      });
    }
  }, []);

  const closeAppendix = useCallback(() => {
    setExprModalOpen(false);
    if (typeof window === 'undefined') return;
    if (!isAppendixHash(window.location.hash)) return;
    const fallbackHash = appendixReturnHashRef.current || APPENDIX_RETURN_HASH;
    window.history.replaceState(null, '', fallbackHash);
    if (fallbackHash.startsWith('#')) {
      document.getElementById(fallbackHash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const syncAppendixFromHash = () => {
      const hash = window.location.hash;
      if (isAppendixHash(hash)) {
        setExprModalOpen(true);
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => {
            scrollToHashTarget(hash);
          });
        });
        return;
      }
      if (hash) appendixReturnHashRef.current = hash;
      setExprModalOpen(false);
    };

    syncAppendixFromHash();
    window.addEventListener('hashchange', syncAppendixFromHash);
    return () => window.removeEventListener('hashchange', syncAppendixFromHash);
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
        onDimensionNChange={handleDimensionChange}
        onHoveredLabelsChange={setStripHoveredLabels}
        activeAlphaMethod={activeAlphaMethodHover}
        onActiveAlphaMethodHoverChange={setActiveAlphaMethodHover}
      />
      {isThemeDockVisible ? (
        <ExplorerThemeDock explorerThemeId={explorerThemeId} onChange={setActiveExplorerTheme} />
      ) : null}

      <div className="w-full pb-20 pt-10">
        {/* Editorial masthead — matches the docs home page register
            (app/(home)/page.tsx). Uppercase kicker with leading rule +
            Newsreader display-serif headline ending in a coral period +
            Source Serif 4 italic lede. The same typographic rhythm the
            reader sees at aicrowd.github.io/flopscope/ carries directly
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
	            Counting symmetry-aware einsums<span style={{ color: 'var(--coral)' }}>.</span>
          </h1>

          <p
            className="mt-5 max-w-[var(--prose-max)] text-[17px] italic text-gray-600 dark:text-gray-300"
            style={{
              fontFamily: 'var(--font-paper-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 18",
              lineHeight: 1.6,
            }}
          >
            <InlineMathText>{String.raw`Symmetry can turn many label assignments into one representative product, but reuse is only half the cost. The product still has to be accumulated wherever its orbit reaches. This article separates the two direct events: multiplication-chain work $\mu$ and filled $O \to Q$ updates $\alpha$, so $\mathrm{Total}=\mu+\alpha$.`}</InlineMathText>
          </p>
        </header>

        <div className="mx-auto w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <AlgorithmAtAGlance
            example={preambleExample}
            hoveredLabels={hoveredLabelSet}
            onHoveredLabelsChange={setStripHoveredLabels}
          />
        </div>

        {/* Passing-disclaimer style for the "scope of the calculation"
            note: softer single bottom border, no top border, gray run-in
            label inline with the prose, and the assumptions list flowing
            as a final sentence. Mirrors the Appendix E "passing
            disclaimer" treatment so reviewers see one consistent tone for
            scope notes across both the main page and the modal. */}
        <div className="mx-auto mb-8 mt-[-0.25rem] w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
          <div className="border-b border-stone-200 bg-white py-3">
            <p
              className="font-serif text-[14px] leading-[1.7] text-stone-700"
              style={{ textAlign: 'justify' }}
            >
              <span className="font-semibold text-stone-900">
                Scope of the calculation.
              </span>{' '}
              <InlineMathText>
                {`This page reports a direct indexed scalar-event count: multiplication-chain events for representative products plus accumulation updates into stored output representatives. It is not flopscope's general FMA FLOP convention, not wall-clock time, and not a contraction-path or memory-traffic model. The output is assumed to be stored by the output symmetry inherited from the detected pointwise group, so one product-orbit representative may update several stored output representatives when projection branches.`}
              </InlineMathText>
              {' '}
              <span className="text-stone-600">
                Assumptions: exact commutative scalar arithmetic; declared equality
                symmetries only; repeated operand names denote the same tensor object;
                the accepted explicit-index einsum language uses lowercase
                single-character labels with explicit outputs and forbids
                ellipsis, broadcasting, repeated labels within one input, and
                duplicate output labels; label permutations preserve label
                domains/sizes; no antisymmetry/signs, conjugation, sparsity,
                approximate numerical symmetry, or contraction-path optimization.
              </span>
            </p>
          </div>
        </div>

        <div className="mx-auto mt-8 w-full max-w-[1460px] px-6 md:px-8 lg:px-10">
	          <div className="flex flex-col items-stretch gap-6 lg:flex-row lg:items-start lg:gap-8">
            <PresetSidebar
              examples={EXAMPLES}
              selectedPresetIdx={selectedPresetIdx}
              pendingPresetIdx={pendingPresetIdx}
              isPreparing={analysisUpdating}
              onSelect={handleSelect}
              onCustom={handleCustomMode}
            />
            <main className="min-w-0 flex-1">
              <div className="flex flex-col">
                {localAnalysisLoading ? (
                  <div
                    role="status"
                    aria-live="polite"
                    className="sr-only"
                    data-analysis-updating="true"
                  >
                    Preparing {pendingSelection.name}. Interactive components are loading.
                  </div>
                ) : null}

            {/* §1 Einsum at a Glance — id: einsum-glance, backward alias: setup */}
            <section id={EXPLORER_ACTS[0].id} className="mb-12 scroll-mt-sticky">
              {/* Backward-compat anchor alias */}
              <span id="setup" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
              <ExplorerSectionCard
                eyebrow={<SectionEyebrow n={1} anchorId={EXPLORER_ACTS[0].id} />}
                title={EXPLORER_ACTS[0].heading}
                description={<InlineMathText>{EXPLORER_ACTS[0].question}</InlineMathText>}
                className="border-gray-200 bg-white"
                contentClassName="pt-5"
              >
                <SectionIntroProse paragraphs={EXPLORER_ACTS[0].introParagraphs} balancedColumns />
                <div className="mt-6">
                  <ExampleChooser
                    examples={EXAMPLES}
                    onSelect={handleSelect}
                    selectedPresetIdx={selectedPresetIdx}
                    pendingPresetIdx={pendingPresetIdx}
                    isPreparing={analysisUpdating}
                    dimensionN={dimensionN}
                    explorerThemeId={explorerThemeId}
                    onDimensionChange={handleDimensionChange}
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
            {analysis && analysisExample && (
              <>
                {/* §2 Product Symmetry — id: product-symmetry, backward alias: structure */}
                <section id={EXPLORER_ACTS[1].id} className="mb-12 scroll-mt-sticky">
                  {/* Backward-compat anchor alias */}
                  <span id="structure" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={2} anchorId={EXPLORER_ACTS[1].id} />}
                    title={EXPLORER_ACTS[1].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[1].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
	                  >
	                    <SectionIntroProse paragraphs={EXPLORER_ACTS[1].introParagraphs} />
	                    <div className="editorial-two-col-divider-md mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
	                      <div id="bipartite-graph" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-sticky">
	                        <ExplorerSubsectionHeader anchorId="bipartite-graph" labelText="Bipartite Graph">
                          Bipartite Graph
                        </ExplorerSubsectionHeader>
                        <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="bipartite graph" variant="graph" minHeight={320}>
                          <BipartiteGraph graph={graph} example={normalizedAnalysisExample} variableColors={variableColors} />
                        </AnalysisLoadingBoundary>
                      </div>
                      <div id="incidence-matrix" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-sticky">
                        <ExplorerSubsectionHeader anchorId="incidence-matrix" labelText="Incidence Matrix">
                          Incidence Matrix M
                        </ExplorerSubsectionHeader>
                        <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="incidence matrix" variant="matrix" minHeight={320}>
                          <MatrixView matrixData={matrixData} graph={graph} example={normalizedAnalysisExample} variableColors={variableColors} />
                        </AnalysisLoadingBoundary>
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

                {/* §3 Projection — id: projection, backward alias: decompose */}
                {/* Hero: O→Q incidence matrix via ComponentCostView (contains BranchingDemo) */}
                <section id={EXPLORER_ACTS[2].id} className="mb-12 scroll-mt-sticky">
                  {/* Backward-compat anchor alias */}
                  <span id="decompose" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={3} anchorId={EXPLORER_ACTS[2].id} />}
                    title={EXPLORER_ACTS[2].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[2].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <ProjectionIntroProse paragraphs={EXPLORER_ACTS[2].introParagraphs} />
	                    <div className="mt-6">
	                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="O → Q matrix" variant="matrix" minHeight={520}>
	                        <BranchingDemo
	                        componentData={componentData}
	                        costModel={cost}
	                        dimensionN={analysisDimensionN}
	                        selectedOrbitIdx={resolvedSelectedOrbitIdx}
	                        onSelectOrbit={setSelectedOrbitIdx}
	                        onHover={handleGraphHover}
	                        hoveredLabels={hoveredLabelSet}
	                        expressionInfo={normalizedAnalysisExample ? {
	                          subscripts: normalizedAnalysisExample.subscripts ?? [],
                          output: normalizedAnalysisExample.output ?? '',
	                          operandNames: (normalizedAnalysisExample.expression?.operandNames ?? '')
                            .split(',')
	                            .map((s) => s.trim())
	                            .filter(Boolean),
	                        } : null}
	                        />
	                      </AnalysisLoadingBoundary>
	                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[2].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §4 Rows and Columns — id: rows-cols, no backward alias */}
                <section id={EXPLORER_ACTS[3].id} className="mb-12 scroll-mt-sticky">
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={4} anchorId={EXPLORER_ACTS[3].id} />}
                    title={EXPLORER_ACTS[3].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[3].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse
                      blocks={EXPLORER_ACTS[3].introBlocks}
                      className="mx-auto max-w-[1180px]"
                    />
                    {/* §4 Two-quotient schematic — C16 */}
                    <div className="mt-6">
                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="rows and columns schematic" variant="cards" minHeight={360}>
                        <TwoQuotientSchematic current={twoQuotientCurrent} />
                      </AnalysisLoadingBoundary>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[3].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §5 Component Factorization — id: component-factorization, no backward alias */}
                <section id={EXPLORER_ACTS[4].id} className="mb-12 scroll-mt-sticky">
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={5} anchorId={EXPLORER_ACTS[4].id} />}
                    title={EXPLORER_ACTS[4].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[4].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
	                  >
	                    <SectionIntroProse paragraphs={EXPLORER_ACTS[4].introParagraphs} />
	                    <div className="mt-6">
	                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="label interaction graph" variant="graph" minHeight={420}>
	                        <LabelInteractionGraph
	                        allLabels={group?.allLabels ?? []}
	                        vLabels={group?.vLabels ?? []}
	                        interactionGraph={componentData?.interactionGraph}
	                        components={componentData?.components ?? null}
	                        fullGenerators={group?.fullGenerators ?? []}
	                        onHover={handleGraphHover}
	                        activeComponentId={activeComponentId}
	                        onActiveComponentHoverChange={setActiveComponentId}
	                        />
	                      </AnalysisLoadingBoundary>
	                    </div>
	                    <div className="mt-6">
	                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="component accounting" variant="cards" minHeight={420}>
	                        <ComponentCostView
	                        componentData={componentData}
	                        costModel={cost}
	                        dimensionN={analysisDimensionN}
	                        numTerms={normalizedAnalysisExample?.subscripts?.length ?? 1}
	                        allLabels={group.allLabels}
	                        vLabels={group.vLabels}
	                        fullGenerators={group.fullGenerators}
	                        selectedOrbitIdx={resolvedSelectedOrbitIdx}
	                        onSelectOrbit={setSelectedOrbitIdx}
	                        onGraphHover={handleGraphHover}
	                        spotlightLeafIds={spotlightLeafSet}
	                        hoveredLabels={hoveredLabelSet}
	                        expressionInfo={normalizedAnalysisExample ? {
	                          subscripts: normalizedAnalysisExample.subscripts ?? [],
	                          output: normalizedAnalysisExample.output ?? '',
	                          operandNames: (normalizedAnalysisExample.expression?.operandNames ?? '')
	                            .split(',')
	                            .map((s) => s.trim())
	                            .filter(Boolean),
	                        } : null}
	                        showBranchingDemo={false}
	                        showCostCards={false}
	                        showDecisionLadder={false}
	                        onActiveComponentHoverChange={setActiveComponentId}
	                        onActiveAlphaMethodHoverChange={setActiveAlphaMethodHover}
	                        onDimensionNChange={handleDimensionChange}
	                        />
	                      </AnalysisLoadingBoundary>
	                    </div>
	                    <div className="mt-4">
	                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[4].produces}</NarrativeCallout>
	                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §6 Certification — id: certification, backward alias: proof */}
                {/* Sigma-loop / wreath structure / certification UI */}
                <section id={EXPLORER_ACTS[5].id} className="mb-12 scroll-mt-sticky">
                  {/* Backward-compat anchor alias */}
                  <span id="proof" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={6} anchorId={EXPLORER_ACTS[5].id} />}
                    title={EXPLORER_ACTS[5].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[5].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse paragraphs={EXPLORER_ACTS[5].introParagraphs} />
                    {/* Wreath structure renders full-width — the enumeration target the σ-loop walks over. */}
                    <div id="wreath-structure" className="mt-6 flex flex-col gap-2 scroll-mt-sticky">
                      <ExplorerSubsectionHeader anchorId="wreath-structure" labelText="Wreath structure">
                        Wreath structure
                      </ExplorerSubsectionHeader>
                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="wreath structure" variant="cards" minHeight={300}>
                        <WreathStructureView
                          analysis={analysis}
                          example={normalizedAnalysisExample}
                        />
                      </AnalysisLoadingBoundary>
                    </div>
                    {/* σ-Loop (enumerates the wreath) + Generator Construction (closes valid π's). */}
                    <div className="editorial-two-col-divider-lg mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <div id="sigma-loop" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-sticky">
                        <ExplorerSubsectionHeader anchorId="sigma-loop" labelText="σ-Loop & π Detection">
                          <InlineMathText>
                            {`$${notationLatex('sigma_row_move')}$-Loop & $${notationLatex('pi_relabeling')}$ Detection`}
                          </InlineMathText>
                        </ExplorerSubsectionHeader>
                        <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="sigma loop" variant="cards" minHeight={520}>
                          <SigmaLoop
                          results={sigmaResults}
                          graph={graph}
                          matrixData={matrixData}
                          example={normalizedAnalysisExample}
                          variableColors={variableColors}
                          group={group}
                          onSelectedPairChange={setSelectedSigmaPairIndex}
                          onSwitchToDirectedTriangle={() => {
                            // C25 — Witness gallery CTA: jump to the canonical
                            // Directed triangle preset which has both an accepted
                            // (rotation) and a rejected (reflection) σ. We resolve
                            // the index by id so the call is robust to preset reorder.
                            const idx = EXAMPLES.findIndex((ex) => ex.id === 'triangle');
                            if (idx >= 0) handleSelect(idx);
                          }}
                          />
                        </AnalysisLoadingBoundary>
                      </div>
                      <div id="generator-construction" className="grid grid-rows-[auto_1fr] gap-2 scroll-mt-sticky">
                        <ExplorerSubsectionHeader anchorId="generator-construction" labelText="Generator Construction">
                          Generator Construction
                        </ExplorerSubsectionHeader>
                        <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="generator construction" variant="cards" minHeight={520}>
                          <DiminoView
                            group={group}
                            sigmaResults={sigmaResults}
                            selectedPairIndex={selectedSigmaPairIndex}
                          />
                        </AnalysisLoadingBoundary>
                      </div>
                    </div>
	                    {/* C27 — Certification Summary Strip: 7-metric bridge that
	                        sits at the end of the certification section, after
	                        the SigmaLoop / Generator-Construction block and before
	                        the next section's narrative callouts.
                        Each pill is hover-aware: writing to upstream highlight
                        targets (wreath / audit / generator-closure / column-action)
                        and the existing activeComponentId bus. */}
                    <div className="mt-6">
                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="certification summary" variant="compact" minHeight={170}>
                        <CertificationSummaryStrip
                          candidateMoves={(sigmaResults ?? []).filter((r) => !r.skipped).length}
                          accepted={(sigmaResults ?? []).filter((r) => !r.skipped && r.isValid).length}
                          identityOnly={(sigmaResults ?? []).filter((r) => r.skipped).length}
                          rejected={(sigmaResults ?? []).filter((r) => !r.skipped && !r.isValid).length}
                          gPtSize={group?.fullElements?.length ?? 1}
                          hSize={outputActionSize}
                          componentsCount={(componentData?.components ?? []).length}
                          setActiveComponentId={setActiveComponentId}
                        />
                      </AnalysisLoadingBoundary>
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
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[5].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §7 Counting Shortcuts — id: counting-shortcuts, no backward alias */}
                <section id={EXPLORER_ACTS[6].id} className="mb-12 scroll-mt-sticky">
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={7} anchorId={EXPLORER_ACTS[6].id} />}
                    title={EXPLORER_ACTS[6].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[6].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
	                  >
	                    <SectionIntroProse paragraphs={EXPLORER_ACTS[6].introParagraphs} />
	                    <div id="classification-tree" className="mt-6 bg-white p-4 scroll-mt-sticky">
	                      <ExplorerSubsectionHeader anchorId="classification-tree" labelText="Classification Tree">
	                        Shortcut Decision Ladder
	                      </ExplorerSubsectionHeader>
	                      <p className="explorer-support-prose mt-2">
	                        Each component follows this yes/no spine until the cheapest exact accumulation counter applies. The highlighted leaves show where the active example lands.
	                      </p>
	                      <div className="mt-4">
	                        <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="shortcut decision ladder" variant="cards" minHeight={360}>
	                          <DecisionLadder
	                          activeLeafIds={activeLeafIds}
	                          spotlightLeafIds={spotlightLeafSet}
	                          liveReasonsByLeaf={liveReasonsByLeaf}
	                          />
	                        </AnalysisLoadingBoundary>
	                      </div>
	                    </div>
	                    <div className="mt-6">
	                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="naive alpha meter" variant="compact" minHeight={220}>
	                        <NaiveAlphaCostMeter
	                        dimensionN={analysisDimensionN}
                        allLabels={group?.allLabels ?? []}
                        groupSize={group?.fullElements?.length ?? 1}
                        hSize={outputActionSize}
	                        />
	                      </AnalysisLoadingBoundary>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[6].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §8 Typed Partition Counting — id: typed-partition, backward alias: partition-counting */}
                <section id={EXPLORER_ACTS[7].id} className="mb-12 scroll-mt-sticky">
                  {/* Backward-compat anchor alias */}
                  <span id="partition-counting" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={8} anchorId={EXPLORER_ACTS[7].id} />}
                    title={EXPLORER_ACTS[7].heading}
                    description={<InlineMathText>{EXPLORER_ACTS[7].question}</InlineMathText>}
                    className="border-gray-200 bg-white"
                    contentClassName="pt-5"
                  >
                    <SectionIntroProse paragraphs={EXPLORER_ACTS[7].introParagraphs} />
                    <div className="mt-6">
                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="tuple pattern meter" variant="compact" minHeight={220}>
                        <TuplePatternMeter
                          dimensionN={analysisDimensionN}
                          allLabels={group?.allLabels ?? []}
                          groupSize={group?.fullElements?.length ?? 1}
                          componentData={componentData}
                        />
                      </AnalysisLoadingBoundary>
                    </div>
                    <div className="mt-6">
                      <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="typed partition counter" variant="cards" minHeight={420}>
                        <TypedPartitionDemo
                          componentData={componentData}
                          costModel={cost}
                        />
                      </AnalysisLoadingBoundary>
                    </div>
                    <div className="mt-4">
                      <NarrativeCallout label="What this produces" tone="accent">{EXPLORER_ACTS[7].produces}</NarrativeCallout>
                    </div>
                  </ExplorerSectionCard>
                </section>

                {/* §9 Assemble the Cost — id: assemble-cost, backward alias: cost-savings */}
                <section id={EXPLORER_ACTS[8].id} className="mb-12 scroll-mt-sticky">
                  {/* Backward-compat anchor alias */}
                  <span id="cost-savings" aria-hidden="true" style={{ position: 'absolute', height: 0 }} />
                  <ExplorerSectionCard
                    eyebrow={<SectionEyebrow n={9} anchorId={EXPLORER_ACTS[8].id} />}
                    title={EXPLORER_ACTS[8].heading}
	                    description={<InlineMathText>{EXPLORER_ACTS[8].question}</InlineMathText>}
	                    className="border-gray-200 bg-white"
	                    contentClassName="pt-5"
	                  >
	                    <AnalysisLoadingBoundary isLoading={localAnalysisLoading} label="total cost view" variant="compact" minHeight={360}>
	                      <TotalCostView
	                      componentCosts={componentCosts}
                      componentData={componentData}
                      dimensionN={analysisDimensionN}
                      numTerms={normalizedAnalysisExample?.subscripts?.length ?? 1}
                      explorerThemeId={explorerThemeId}
                      presetName={analysisExample?.name ?? null}
	                      />
	                    </AnalysisLoadingBoundary>

                    {/* Single appendix-note block. Per user feedback "we
                        can just have a SINGLE appendix note block": the
                        previous §9 button-styled note + separate §10
                        Appendix-Transition section have been merged into
                        one inline gray-50 strip that:
                          (a) keeps the G_pt vs G_f question + prose,
                          (b) adds the five A–E letter-cards as quick
                              links to specific appendix sections,
                          (c) carries the appendix-transition anchor id
                              so existing #appendix-transition links
                              continue to resolve.
                        The whole strip is now click-anywhere — clicking
                        any non-letter-card region of the block opens the
                        appendix at the top. The five letter-card buttons
                        keep their own click targets (deep-link to specific
                        sub-sections) by stopping click propagation. The
                        outer wrapper uses role="button" + tabIndex + Enter/
                        Space handler so keyboard users get the same
                        single-action affordance. */}
                    <div
                      id={EXPLORER_ACTS[9].id}
                      role="button"
                      tabIndex={0}
                      aria-label="Open the appendix"
                      onClick={() => openAppendix()}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          openAppendix();
                        }
                      }}
                      className="group -mx-4 -mb-4 mt-8 cursor-pointer border-t border-stone-200/70 bg-gray-50 px-4 py-4 transition-colors scroll-mt-sticky hover:bg-stone-100/70 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-coral"
                    >
                      <span className="block font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
                        Appendix note
                      </span>
                      <span
                        className="mt-1.5 block font-serif text-[15px] leading-7 text-stone-900 transition-colors group-hover:text-coral"
                      >
                        Is this the full symmetry of the final expression?
                      </span>
                      <p className="mt-1.5 text-[12.5px] leading-6 text-stone-700">
                        The cost above uses <Latex math={notationLatex('g_pointwise')} /> on product assignments and <Latex math={notationLatex('h_output')} /> on stored output representatives, where <Latex math={String.raw`H = \mathrm{Stab}_{G_{\text{pt}}}(V)|_V`} />. The completed expression can have a larger formal symmetry <Latex math={String.raw`G_{\text{f}} = H \times \prod_d S(W_d)`} />. Its dummy-label factor acts after summation and must not be used to remove pre-summation product or update events.
                      </p>
                      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5" aria-label="Appendix A-E map">
                        {APPENDIX_MAP.map((item) => (
                          <button
                            key={item.letter}
                            type="button"
                            onClick={(e) => { e.stopPropagation(); openAppendix(item.hash); }}
                            className="min-h-[88px] cursor-pointer rounded-md border border-stone-200 bg-white p-3 text-left transition-colors hover:border-coral/50 hover:bg-coral-light/30 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-coral"
                          >
                            <span className="block font-sans text-[10px] font-semibold uppercase tracking-[0.18em] text-coral">
                              Appendix {item.letter}
                            </span>
                            <span className="mt-2 block text-[13px] font-semibold leading-5 text-gray-900">
                              {item.title}
                            </span>
                          </button>
                        ))}
                      </div>
                      <span className="sr-only">{EXPLORER_ACTS[9].heading}</span>
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

      <ExpressionLevelModal
        isOpen={exprModalOpen}
        onClose={closeAppendix}
        analysis={analysis}
        group={group}
        example={analysisExample}
        onSelectPreset={handleSelect}
      />
    </div>
  );
}
