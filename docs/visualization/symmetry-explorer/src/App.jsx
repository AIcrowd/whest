import { useState, useMemo, useCallback } from 'react';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import {
  buildBipartite, buildIncidenceMatrix, runSigmaLoop,
  buildGroup, computeBurnside, computeCostReduction,
} from './engine/algorithm.js';
import PresetSidebar from './components/PresetSidebar.jsx';
import ExampleChooser from './components/ExampleChooser.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import GroupView from './components/GroupView.jsx';
import BurnsideView from './components/BurnsideView.jsx';
import CostView from './components/CostView.jsx';
import './styles.css';

const STEPS = [
  { id: 'example', num: 1, title: 'Choose Example' },
  { id: 'graph', num: 2, title: 'Bipartite Graph' },
  { id: 'matrix', num: 3, title: 'Incidence Matrix M' },
  { id: 'sigma', num: 4, title: 'σ-Loop & π Detection' },
  { id: 'group', num: 5, title: 'Group Construction' },
  { id: 'burnside', num: 6, title: 'Burnside Counting' },
  { id: 'cost', num: 7, title: 'Cost Reduction' },
];

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

export default function App() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [customExample, setCustomExample] = useState(null);
  const [dimensionN, setDimensionN] = useState(5);
  const [isDirty, setIsDirty] = useState(false);
  // Resolve the active example: preset or custom
  const isCustom = exampleIdx === CUSTOM_IDX;
  const example = isCustom ? customExample : EXAMPLES[exampleIdx];

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
    setIsDirty(false);
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setCustomExample(ex);
    setExampleIdx(CUSTOM_IDX);
    setIsDirty(false);
  }, []);

  // Handle config edits (before Analyze is clicked)
  const handleDirty = useCallback(() => {
    setIsDirty(true);
  }, []);

  // Run the full algorithm pipeline
  const pipeline = useMemo(() => {
    if (!normalizedExample) return null;
    try {
      const graph = buildBipartite(normalizedExample);
      const matrixData = buildIncidenceMatrix(graph);
      const sigmaResults = runSigmaLoop(graph, matrixData, normalizedExample);
      const group = buildGroup(sigmaResults, graph, normalizedExample);
      return { graph, matrixData, sigmaResults, group };
    } catch (err) {
      console.error('Pipeline error:', err);
      return null;
    }
  }, [normalizedExample]);

  const { graph, matrixData, sigmaResults, group } = pipeline || {};

  const burnside = useMemo(
    () => group ? computeBurnside(group, dimensionN) : null,
    [group, dimensionN]
  );

  const numTerms = normalizedExample?.subscripts?.length || 2;
  const cost = useMemo(
    () => (burnside && group) ? computeCostReduction(burnside, group, numTerms) : null,
    [burnside, group, numTerms]
  );

  // Check if per-op symmetry is active for any operand
  const hasPerOpSym = normalizedExample && (
    normalizedExample.perOpSymmetry === 'symmetric' ||
    (Array.isArray(normalizedExample.perOpSymmetry) && normalizedExample.perOpSymmetry.some(s => s === 'symmetric' || (s && typeof s === 'object')))
  );

  // Handle switching to custom mode from sidebar
  const handleCustomMode = useCallback(() => {
    setExampleIdx(CUSTOM_IDX);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Subgraph Symmetry Explorer</h1>
        <p className="subtitle">Interactive walkthrough of <em>einsum symmetry detection</em></p>
      </header>

      <nav className="step-nav">
        {STEPS.map(s => (
          <a key={s.id} href={`#${s.id}`} className="step-link">
            <span className="step-num">{s.num}</span>
            <span className="step-title">{s.title}</span>
          </a>
        ))}
      </nav>

      <div className="app-body">
        <PresetSidebar
          examples={EXAMPLES}
          selected={exampleIdx}
          onSelect={handleSelect}
          onCustom={handleCustomMode}
        />

        <main className="main">
          {/* -- Prominent einsum banner -- */}
          {example && group && (
            <div className="einsum-banner">
              <span className="einsum-label">einsum</span>
              <code className="einsum-expr">{example.formula}</code>
              <span className="einsum-group-tag">
                {group.vGroupName !== 'trivial'
                  ? group.vGroupName
                  : group.wGroupName !== 'trivial'
                    ? `W: ${group.wGroupName}`
                    : 'trivial'}
              </span>
            </div>
          )}

          <section id="example" className="section">
            <SectionHeader num={1} title="Configure Example" />
            <ExampleChooser
              example={example}
              dimensionN={dimensionN}
              onDimensionChange={setDimensionN}
              onCustomExample={handleCustomExample}
              onDirty={handleDirty}
            />
          </section>

          {/* Only render pipeline sections when we have results */}
          {pipeline && example && (
            <div className={isDirty ? 'pipeline-stale' : undefined}>
              <section id="graph" className="section">
                <SectionHeader num={2} title="Bipartite Graph" />
                <p className="section-desc">
                  Left vertices (U) are operand axis-classes. Right vertices are index labels,
                  partitioned into <span className="pill pill-v">V free</span> and{' '}
                  <span className="pill pill-w">W summed</span>.
                  {hasPerOpSym && (
                    <> Per-operand symmetry <em>collapses</em> each operand's axes into a single U-vertex.</>
                  )}
                </p>
                <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} group={group} />
              </section>

              <section id="matrix" className="section">
                <SectionHeader num={3} title="Incidence Matrix M" />
                <p className="section-desc">
                  Each cell M[u, l] is the multiplicity of label l in axis-class u.
                  Column fingerprints (reading down each column) identify structurally equivalent labels.
                </p>
                <MatrixView matrixData={matrixData} graph={graph} example={normalizedExample} variableColors={variableColors} />
              </section>

              <section id="sigma" className="section">
                <SectionHeader num={4} title="σ-Loop & π Detection" />
                <p className="section-desc">
                  When identical operands are swapped (σ), the incidence matrix M gets its
                  rows shuffled into σ(M). We search for a column relabeling π that
                  recovers the original: π(σ(M)) = M. If such a π exists and
                  respects the V/W partition, it reveals a symmetry of the einsum expression.
                </p>
                <SigmaLoop
                  results={sigmaResults}
                  graph={graph}
                  matrixData={matrixData}
                  example={normalizedExample}
                  variableColors={variableColors}
                />
              </section>

              <section id="group" className="section">
                <SectionHeader num={5} title="Group Construction" />
                <p className="section-desc">
                  Collected π's are generators. Dimino's algorithm enumerates all group elements
                  by composing generators until closure.
                </p>
                <GroupView group={group} sigmaResults={sigmaResults} graph={graph} example={normalizedExample} />
              </section>

              <section id="burnside" className="section">
                <SectionHeader num={6} title="Burnside Counting" />
                <p className="section-desc">
                  Each group element fixes a different number of tuples based on its cycle structure.
                  Burnside's lemma counts the unique orbits.
                </p>
                <BurnsideView
                  burnside={burnside}
                  group={group}
                  dimensionN={dimensionN}
                />
              </section>

              <section id="cost" className="section">
                <SectionHeader num={7} title="Cost Reduction" />
                <p className="section-desc">
                  The FLOP cost is reduced by the ratio of unique to total elements.
                </p>
                <CostView cost={cost} numOperands={normalizedExample?.subscripts?.length || 0} />
              </section>
            </div>
          )}

          {/* Show prompt when custom is selected but no expression analyzed yet */}
          {isCustom && !pipeline && (
            <div className="custom-prompt">
              Define your variables and einsum expression above, then click <strong>Analyze</strong> to explore the symmetry detection algorithm.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function SectionHeader({ num, title }) {
  return (
    <div className="section-header">
      <span className="section-num">{num}</span>
      <h2>{title}</h2>
    </div>
  );
}
