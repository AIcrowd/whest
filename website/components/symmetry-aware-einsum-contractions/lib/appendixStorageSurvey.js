import { EXAMPLES } from '../data/examples.js';
import { analyzeExample } from '../engine/pipeline.js';

export const APPENDIX_STORAGE_SURVEY_DIMENSION = 3;

const APPENDIX_SAVINGS_TABLE_ORDER = [
  'four-cycle',
  'bilinear-trace-3',
  'direct-s3-s2',
  'young-s4-v3w1',
  'triple-outer',
  'declared-c3',
  'triangle',
  'outer',
  'bilinear-trace',
  'direct-s2-c3',
  'four-A-grid',
  'young-s4-v2w2',
  'direct-s2-s2',
  'young-s3',
  'mixed-chain',
  'matrix-chain',
  'cross-c3-partial',
  'cross-s2',
  'cyclic-cross',
  'cross-s3',
  'frobenius',
  'trace-product',
];

function tupleKey(tuple) {
  return tuple.join('|');
}

function factorial(n) {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

function cycleLengths(perm) {
  const visited = new Array(perm.arr.length).fill(false);
  const lengths = [];
  for (let i = 0; i < perm.arr.length; i += 1) {
    if (visited[i] || perm.arr[i] === i) continue;
    let cursor = i;
    let len = 0;
    while (!visited[cursor]) {
      visited[cursor] = true;
      cursor = perm.arr[cursor];
      len += 1;
    }
    if (len > 0) lengths.push(len);
  }
  return lengths;
}

function movedSupportSize(elements, degree) {
  const moved = new Set();
  for (const element of elements) {
    for (let i = 0; i < element.arr.length; i += 1) {
      if (element.arr[i] !== i) moved.add(i);
    }
  }
  return moved.size || degree;
}

function hasFullCycle(elements, effectiveDegree) {
  return elements.some((element) => {
    const lengths = cycleLengths(element);
    return lengths.length === 1 && lengths[0] === effectiveDegree;
  });
}

function formatOutputGroupLatex(vLabels, vSubElements) {
  const degree = vLabels.length;
  const order = vSubElements.length;

  if (order <= 1 || degree === 0) return String.raw`\{e\}`;

  const effectiveDegree = movedSupportSize(vSubElements, degree);
  if (effectiveDegree === degree && order === factorial(degree)) {
    return `S_${degree}`;
  }
  if (effectiveDegree === degree && degree >= 3 && order === degree && hasFullCycle(vSubElements, effectiveDegree)) {
    return `C_${degree}`;
  }

  return `\\text{order-}${order}`;
}

function countOutputOrbits(outputs, vLabels, vSubElements) {
  const seen = new Set();
  let total = 0;

  for (const { outTuple } of outputs) {
    const tuple = vLabels.map((label) => outTuple[label]);
    const key = tupleKey(tuple);
    if (seen.has(key)) continue;

    total += 1;
    for (const element of vSubElements) {
      const moved = new Array(tuple.length);
      for (let i = 0; i < tuple.length; i += 1) {
        moved[element.arr[i]] = tuple[i];
      }
      seen.add(tupleKey(moved));
    }
  }

  return total;
}

export function computeAppendixStorageRepresentatives(analysis) {
  const vLabels = analysis?.symmetry?.vLabels ?? [];
  const vSubElements = analysis?.expressionGroup?.vSub?.length
    ? analysis.expressionGroup.vSub
    : [{ arr: Array.from({ length: vLabels.length }, (_, idx) => idx) }];
  const orbitRows = analysis?.costModel?.orbitRows ?? [];

  return orbitRows.reduce(
    (sum, row) => sum + countOutputOrbits(row.outputs ?? [], vLabels, vSubElements),
    0,
  );
}

function formatSavingsPercent(ae, as) {
  if (ae <= 0 || ae <= as) return '0';
  return (((ae - as) * 100) / ae).toFixed(1);
}

function compareSavingsRows(a, b) {
  const aIndex = APPENDIX_SAVINGS_TABLE_ORDER.indexOf(a.id);
  const bIndex = APPENDIX_SAVINGS_TABLE_ORDER.indexOf(b.id);
  if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
  if (aIndex !== -1) return -1;
  if (bIndex !== -1) return 1;
  return a.id.localeCompare(b.id);
}

export function buildAppendixSavingsTableRows({
  examples = EXAMPLES,
  dimensionN = APPENDIX_STORAGE_SURVEY_DIMENSION,
} = {}) {
  return examples
    .map((preset) => {
      const analysis = analyzeExample(preset, dimensionN);
      const vLabels = analysis?.symmetry?.vLabels ?? [];
      const vSubElements = analysis?.expressionGroup?.vSub?.length
        ? analysis.expressionGroup.vSub
        : [{ arr: Array.from({ length: vLabels.length }, (_, idx) => idx) }];
      const ae = analysis?.componentCosts?.alpha ?? 0;
      const as = computeAppendixStorageRepresentatives(analysis);
      const saving = Math.max(ae - as, 0);

      return {
        id: preset.id,
        v: vLabels.length ? vLabels.join(',') : '\\varnothing',
        vSub: formatOutputGroupLatex(vLabels, vSubElements),
        ae,
        as,
        saving,
        pct: formatSavingsPercent(ae, as),
      };
    })
    .sort(compareSavingsRows);
}
