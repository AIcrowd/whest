import { useState, useMemo, useCallback } from 'react';
import { EXAMPLES } from './data/examples.js';
import { parseCycleNotation } from './engine/cycleParser.js';
import { buildVariableColors } from './engine/colorPalette.js';
import { analyzeExample } from './engine/pipeline.js';
import { pickDefaultOrbitRow } from './engine/teachingModel.js';
import ExampleChooser from './components/ExampleChooser.jsx';
import BipartiteGraph from './components/BipartiteGraph.jsx';
import MatrixView from './components/MatrixView.jsx';
import SigmaLoop from './components/SigmaLoop.jsx';
import GroupView from './components/GroupView.jsx';
import BurnsideView from './components/BurnsideView.jsx';
import CostView from './components/CostView.jsx';
import PseudocodeRail from './components/PseudocodeRail.jsx';
import OrbitInspector from './components/OrbitInspector.jsx';
import './styles.css';

const STEPS = [
  { id: 'example', num: 1, title: 'Choose Example' },
  { id: 'graph', num: 2, title: 'Bipartite Graph' },
  { id: 'matrix', num: 3, title: 'Incidence Matrix M' },
  { id: 'sigma', num: 4, title: 'σ-Loop & π Detection' },
  { id: 'group', num: 5, title: 'Group Construction' },
  { id: 'framework', num: 6, title: 'Mental Framework' },
  { id: 'burnside', num: 7, title: 'Burnside For Evaluation Cost' },
  { id: 'cost', num: 8, title: 'Reduction Cost' },
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
  const [selectedOrbitIdx, setSelectedOrbitIdx] = useState(-1);
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
  }, []);

  // Handle custom example submission
  const handleCustomExample = useCallback((ex) => {
    setCustomExample(ex);
    setExampleIdx(CUSTOM_IDX);
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
    burnside,
    costModel: cost,
  } = analysis || {};

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

      <main className="main main-full">
          {/* -- Prominent einsum banner -- */}
          {example && group && (
            <div className="einsum-banner">
              <span className="einsum-label">einsum</span>
              <code className="einsum-expr">{example.formula}</code>
              <span className="einsum-group-tag">
                {group.fullGroupName || 'trivial'}
              </span>
            </div>
          )}

          <section id="example" className="section">
            <SectionHeader num={1} title="Choose an Example" />
            <ExampleChooser
              examples={EXAMPLES}
              selected={exampleIdx}
              onSelect={handleSelect}
              example={example}
              dimensionN={dimensionN}
              onDimensionChange={setDimensionN}
              onCustomExample={handleCustomExample}
            />
          </section>

          {/* Only render pipeline sections when we have results */}
          {analysis && example && (
            <>
              <section id="graph" className="section">
                <SectionHeader num={2} title="Bipartite Graph" />
                <p className="section-desc">
                  Left vertices (U) are operand axis-classes. Right vertices are index labels,
                  partitioned into <span className="pill pill-v">V free</span> and{' '}
                  <span className="pill pill-w">W summed</span>.
                  {hasPerOpSym && (
                    <> Per-operand symmetry <em>collapses</em> each operand&apos;s axes into a single U-vertex.</>
                  )}
                </p>
                <BipartiteGraph graph={graph} example={normalizedExample} variableColors={variableColors} />
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
                  recovers the original: π(σ(M)) = M. Valid π mappings are kept on the
                  full label set, so cross V/W symmetries are surfaced instead of discarded.
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
                  Collected π&apos;s are generators for one full symmetry group on the active labels.
                  Dimino&apos;s algorithm enumerates all group elements by composing generators until closure.
                </p>
                <GroupView group={group} />
              </section>

              <section id="framework" className="section">
                <SectionHeader num={6} title="Mental Framework: Compute Once, Reduce Many" />
                <p className="section-desc">
                  This is the mental model for everything that follows. The group-theory machinery
                  only tells us what <code>RepSet</code> and <code>Outs(rep)</code> are for this
                  contraction step.
                </p>
                <div className="mental-model-grid">
                  <PseudocodeRail activeStepId="framework" selectedOrbitRow={cost?.orbitRows?.[resolvedSelectedOrbitIdx] ?? null} />
                  <OrbitInspector
                    orbitRows={cost?.orbitRows ?? []}
                    selectedOrbitIdx={resolvedSelectedOrbitIdx}
                    onSelectOrbit={setSelectedOrbitIdx}
                    kicker="Orbit Example"
                    title="Selected orbit driving the code comments"
                    description="Pick one representative orbit here. The code comments on the left update to show the same rep, its projected outputs, and one concrete coeff value."
                  />
                </div>
              </section>

              <section id="burnside" className="section">
                <SectionHeader num={7} title="Burnside For Evaluation Cost" />
                <p className="section-desc">
                  Burnside&apos;s lemma counts full tuple orbits under the full group. In the mental
                  framework above, this is the line that loops over <code>RepSet</code> and
                  increments <code>evaluation_cost</code>.
                </p>
                <BurnsideView
                  burnside={burnside}
                  group={group}
                  dimensionN={dimensionN}
                />
              </section>

              <section id="cost" className="section">
                <SectionHeader num={8} title="Reduction Cost" />
                <p className="section-desc">
                  Reduction cost counts how many output bins the symmetry-compressed representatives
                  still update. In the mental framework above, this is the inner loop over
                  <code>Outs(rep)</code>, not the Burnside count itself.
                </p>
                <CostView costModel={cost} />
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
