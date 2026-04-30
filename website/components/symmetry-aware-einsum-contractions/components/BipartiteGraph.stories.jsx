import BipartiteGraph from './BipartiteGraph.jsx';

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
