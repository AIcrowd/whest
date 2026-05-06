import BipartiteGraph from './BipartiteGraph.jsx';
import { LabelInteractionGraph } from './ComponentView.jsx';

export default {
  title: 'Section4/BipartiteGraph',
  component: BipartiteGraph,
  parameters: { layout: 'padded' },
};

// ---------------------------------------------------------------------------
// Shared example and variableColors used across stories
// ---------------------------------------------------------------------------

// Cross-S2 style: "ij,jk->ik"
// Operand A has axes i,j  →  2 U-vertices: u0={i}, u1={j}
// Operand B has axes j,k  →  2 U-vertices: u2={j}, u3={k}
// freeLabels  = {i, k}  (appear in output "ik")
// summedLabels = {j}    (contracted)
// incidence[uIdx][label] = multiplicity

// ---------------------------------------------------------------------------
// Story 1: Empty — no labels, no edges
// ---------------------------------------------------------------------------
export const Empty = {
  args: {
    graph: {
      uVertices: [],
      incidence: [],
      freeLabels: new Set(),
      summedLabels: new Set(),
      identicalGroups: [],
    },
    example: {
      subscripts: [],
      output: '',
      operandNames: [],
    },
    variableColors: {},
    highlightedLabels: new Set(),
  },
};

// ---------------------------------------------------------------------------
// Story 2: Populated — Cross-S2 "ij,jk->ik"
//   U-vertices (left): A:{i}, A:{j}, B:{j}, B:{k}
//   V free (right):    i, k
//   W summed (right):  j
//   Edges: A:{i}->i, A:{j}->j, B:{j}->j, B:{k}->k
// ---------------------------------------------------------------------------
export const Populated = {
  args: {
    graph: {
      uVertices: [
        { opIdx: 0, classId: 0, labels: new Set(['i']) },
        { opIdx: 0, classId: 1, labels: new Set(['j']) },
        { opIdx: 1, classId: 0, labels: new Set(['j']) },
        { opIdx: 1, classId: 1, labels: new Set(['k']) },
      ],
      incidence: [
        { i: 1 },        // u0: A:{i}  -> i
        { j: 1 },        // u1: A:{j}  -> j
        { j: 1 },        // u2: B:{j}  -> j
        { k: 1 },        // u3: B:{k}  -> k
      ],
      freeLabels: new Set(['i', 'k']),
      summedLabels: new Set(['j']),
      identicalGroups: [],
    },
    example: {
      subscripts: ['ij', 'jk'],
      output: 'ik',
      operandNames: ['A', 'B'],
    },
    variableColors: {},
    highlightedLabels: new Set(),
  },
};

// ---------------------------------------------------------------------------
// Story 3: Disconnected — two independent components
//   Component 1: A:{i} -- i  and  A:{a} -- a
//   Component 2: B:{j} -- j  and  B:{b} -- b
//   i,j are free (output); a,b are summed (contracted)
//   No edge crosses between the two components — useful for testing
//   correctness of Label Interaction Graph decomposition (C20 sibling).
// ---------------------------------------------------------------------------
export const Disconnected = {
  args: {
    graph: {
      uVertices: [
        { opIdx: 0, classId: 0, labels: new Set(['i']) },
        { opIdx: 0, classId: 1, labels: new Set(['a']) },
        { opIdx: 1, classId: 0, labels: new Set(['j']) },
        { opIdx: 1, classId: 1, labels: new Set(['b']) },
      ],
      incidence: [
        { i: 1 },        // u0: A:{i}  -> i  (component 1)
        { a: 1 },        // u1: A:{a}  -> a  (component 1)
        { j: 1 },        // u2: B:{j}  -> j  (component 2)
        { b: 1 },        // u3: B:{b}  -> b  (component 2)
      ],
      freeLabels: new Set(['i', 'j']),
      summedLabels: new Set(['a', 'b']),
      identicalGroups: [],
    },
    example: {
      subscripts: ['ia', 'jb'],
      output: 'ij',
      operandNames: ['A', 'B'],
    },
    variableColors: {},
    highlightedLabels: new Set(),
  },
};

// ---------------------------------------------------------------------------
// Story 4 (C20 Gap 2): LabelInteractionGraph — toggle "certified factors"
//   Shows the certified-independent-factors view (default, V3.1 §C20).
//   Exercises Gap 2: the 2-button toggle and Gap 1: hull visual encoding.
// ---------------------------------------------------------------------------
export const LIGCertifiedFactors = {
  render: () => (
    <div style={{ padding: 24, maxWidth: 280 }}>
      <div style={{ marginBottom: 8, fontSize: 12, color: '#5D5F60' }}>
        LabelInteractionGraph — certified factors view (default)
      </div>
      <LabelInteractionGraph
        allLabels={['i', 'j', 'k', 'l']}
        vLabels={['i', 'k']}
        interactionGraph={{
          edges: [
            [0, 1, 0], // i–j via σ0
            [2, 3, 0], // k–l via σ0 (same generator, disjoint cycle → multi-cycle-glued)
          ],
          components: [[0, 1, 2, 3]],
        }}
        components={null}
        fullGenerators={null}
      />
    </div>
  ),
  name: 'LIG — certified factors (gap 2)',
};

// ---------------------------------------------------------------------------
// Story 5 (C20 Gap 3): LabelInteractionGraph — hover-component bus
//   Passes activeComponentId to show coral outline on hull.
// ---------------------------------------------------------------------------
export const LIGHoveredComponent = {
  render: () => (
    <div style={{ padding: 24, maxWidth: 280 }}>
      <div style={{ marginBottom: 8, fontSize: 12, color: '#5D5F60' }}>
        LabelInteractionGraph — hull highlighted via activeComponentId prop
      </div>
      <LabelInteractionGraph
        allLabels={['i', 'j', 'k', 'l']}
        vLabels={['i', 'k']}
        interactionGraph={{
          edges: [
            [0, 1, 0],
            [2, 3, 0],
          ],
          components: [[0, 1, 2, 3]],
        }}
        components={null}
        fullGenerators={null}
        activeComponentId="i,j,k,l"
        onActiveComponentHoverChange={() => {}}
      />
    </div>
  ),
  name: 'LIG — hovered component (gap 3)',
};

// ---------------------------------------------------------------------------
// Story 6 (C20 Gap 1): LabelInteractionGraph — generator support mode
//   Shows the "generator supports" toggle with dashed inner stroke for
//   multi-cycle-glued factor (σ=(i j)(k l) — labels i,k have no direct edge).
// ---------------------------------------------------------------------------
export const LIGGeneratorSupports = {
  render: () => (
    <div style={{ padding: 24, maxWidth: 280 }}>
      <div style={{ marginBottom: 8, fontSize: 12, color: '#5D5F60' }}>
        LabelInteractionGraph — generator-support mode, click "generator supports"
      </div>
      <LabelInteractionGraph
        allLabels={['i', 'j', 'k', 'l']}
        vLabels={['i', 'k']}
        interactionGraph={{
          edges: [
            [0, 1, 0], // i–j (cycle 1 of σ=(i j)(k l))
            [2, 3, 0], // k–l (cycle 2 of σ=(i j)(k l))
          ],
          components: [[0, 1, 2, 3]], // one factor via generator-support union-find
        }}
        components={null}
        fullGenerators={null}
      />
    </div>
  ),
  name: 'LIG — generator supports (gap 1)',
};
