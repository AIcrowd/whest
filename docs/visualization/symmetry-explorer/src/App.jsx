import { useState, useMemo } from 'react';
import { EXAMPLES } from './data/examples.js';
import {
  buildBipartite, buildIncidenceMatrix, runSigmaLoop,
  buildGroup, computeBurnside, computeCostReduction,
} from './engine/algorithm.js';
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

export default function App() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [dimensionN, setDimensionN] = useState(5);

  const example = EXAMPLES[exampleIdx];

  // Run the full algorithm pipeline (steps 2-5 are structural, 6-7 depend on n)
  const pipeline = useMemo(() => {
    const graph = buildBipartite(example);
    const matrixData = buildIncidenceMatrix(graph);
    const sigmaResults = runSigmaLoop(graph, matrixData);
    const group = buildGroup(sigmaResults, graph);
    return { graph, matrixData, sigmaResults, group };
  }, [exampleIdx]);

  const { graph, matrixData, sigmaResults, group } = pipeline;

  const burnside = useMemo(
    () => computeBurnside(group, dimensionN),
    [group, dimensionN]
  );

  const cost = useMemo(
    () => computeCostReduction(burnside, group),
    [burnside, group]
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>Subgraph Symmetry Explorer</h1>
        <p className="subtitle">Interactive walkthrough of einsum symmetry detection</p>
      </header>

      <nav className="step-nav">
        {STEPS.map(s => (
          <a key={s.id} href={`#${s.id}`} className="step-link">
            <span className="step-num">{s.num}</span>
            <span className="step-title">{s.title}</span>
          </a>
        ))}
      </nav>

      <main className="main">
        <section id="example" className="section">
          <SectionHeader num={1} title="Choose an Example" />
          <ExampleChooser
            examples={EXAMPLES}
            selected={exampleIdx}
            onSelect={setExampleIdx}
            dimensionN={dimensionN}
            onDimensionChange={setDimensionN}
          />
        </section>

        <section id="graph" className="section">
          <SectionHeader num={2} title="Bipartite Graph" />
          <p className="section-desc">
            Left vertices (U) are operand axis-classes. Right vertices are index labels,
            partitioned into <span className="pill pill-v">V free</span> and{' '}
            <span className="pill pill-w">W summed</span>.
            {example.perOpSymmetry === 'symmetric' && (
              <> Per-operand S₂ symmetry <em>collapses</em> each operand's axes into a single U-vertex.</>
            )}
          </p>
          <BipartiteGraph graph={graph} example={example} />
        </section>

        <section id="matrix" className="section">
          <SectionHeader num={3} title="Incidence Matrix M" />
          <p className="section-desc">
            Each cell M[u, ℓ] is the multiplicity of label ℓ in axis-class u.
            Column fingerprints (reading down each column) identify structurally equivalent labels.
          </p>
          <MatrixView matrixData={matrixData} graph={graph} example={example} />
        </section>

        <section id="sigma" className="section">
          <SectionHeader num={4} title="σ-Loop & π Detection" />
          <p className="section-desc">
            For each permutation σ of identical operands, shuffle M's rows, match column
            fingerprints back to M, and derive the induced label permutation π.
            Valid π's satisfy π(V) ⊆ V and π(W) ⊆ W.
          </p>
          <SigmaLoop
            results={sigmaResults}
            graph={graph}
            matrixData={matrixData}
            example={example}
          />
        </section>

        <section id="group" className="section">
          <SectionHeader num={5} title="Group Construction" />
          <p className="section-desc">
            Collected π's are generators. Dimino's algorithm enumerates all group elements
            by composing generators until closure.
          </p>
          <GroupView group={group} />
        </section>

        <section id="burnside" className="section">
          <SectionHeader num={6} title="Burnside Counting" />
          <p className="section-desc">
            unique = (1/|G|) × Σ<sub>g∈G</sub> Π<sub>cycles c of g</sub> n<sub>c</sub>.
            Each group element fixes a different number of tuples based on its cycle structure.
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
          <CostView cost={cost} />
        </section>
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
