function formatTupleInline(tuple) {
  if (!tuple || typeof tuple !== 'object') return '(...)';
  const parts = Object.entries(tuple).map(([label, value]) => `${label}=${value}`);
  return `(${parts.join(', ')})`;
}

function formatOutputsInline(outputs) {
  if (!Array.isArray(outputs) || outputs.length === 0) return '[...]';
  return `[${outputs.map((output) => formatTupleInline(output.outTuple)).join(', ')}]`;
}

function firstOutputCoeff(outputs) {
  if (!Array.isArray(outputs) || outputs.length === 0) {
    return {
      outTuple: '(...)',
      coeff: '?',
    };
  }

  return {
    outTuple: formatTupleInline(outputs[0].outTuple),
    coeff: outputs[0].coeff,
  };
}

export function buildMentalModelLines(selectedOrbitRow = null) {
  const exampleRep = formatTupleInline(selectedOrbitRow?.repTuple);
  const exampleOuts = formatOutputsInline(selectedOrbitRow?.outputs);
  const exampleCoeff = firstOutputCoeff(selectedOrbitRow?.outputs);

  return [
    { id: 'intro-1', number: 1, code: '# sigma row moves induce valid pi relabelings on the active labels.' },
    { id: 'intro-2', number: 2, code: '# Those pi relabelings generate the full symmetry group G for this step.' },
    { id: 'intro-3', number: 3, code: '# multiplication_cost counts one product evaluation per G-orbit representative.' },
    { id: 'init-eval', number: 4, code: 'multiplication_cost = 0' },
    { id: 'intro-4', number: 5, code: '# accumulation_cost counts one accumulation per distinct projected output bin.' },
    { id: 'init-reduce', number: 6, code: 'accumulation_cost = 0' },
    { id: 'blank-1', number: 7, code: '' },
    { id: 'rep-comment-1', number: 8, code: '# RepSet = one representative full tuple from each orbit of G.' },
    { id: 'rep-comment-2', number: 9, code: '# Burnside counts RepSet without enumerating every dense tuple.' },
    { id: 'rep-comment-3', number: 10, code: `# Example rep = ${exampleRep}` },
    { id: 'rep-loop', number: 11, code: 'for rep in RepSet:' },
    { id: 'base-val', number: 12, code: '    base_val = product_of_operand_entries_at(rep)' },
    { id: 'eval-inc', number: 13, code: '    multiplication_cost += max(num_terms - 1, 0)' },
    { id: 'blank-2', number: 14, code: '' },
    { id: 'reduce-comment-1', number: 15, code: '    # project_V keeps only the output labels V from a full tuple.' },
    { id: 'reduce-comment-2', number: 16, code: `    # Example Outs(rep) = ${exampleOuts}` },
    { id: 'reduce-comment-3', number: 17, code: `    # Example coeff(rep, ${exampleCoeff.outTuple}) = ${exampleCoeff.coeff}` },
    { id: 'out-loop', number: 18, code: '    for out in Outs(rep):' },
    { id: 'reduce-update', number: 19, code: '        R[out] += coeff(rep, out) * base_val' },
    { id: 'reduce-inc', number: 20, code: '        accumulation_cost += 1' },
  ];
}

export function buildMentalModelCode(selectedOrbitRow = null) {
  return buildMentalModelLines(selectedOrbitRow)
    .map(({ code }) => code)
    .join('\n');
}

export const PSEUDOCODE_LINES = buildMentalModelLines();

const ALL_LINES = PSEUDOCODE_LINES.map((line) => line.number);

const STEP_TO_LINES = {
  framework: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
  'component-cost': [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
  'total-cost': [4, 6, 13, 20],
};

export function getFocusedLines(stepId) {
  return STEP_TO_LINES[stepId] ?? ALL_LINES;
}

const KEYWORD_TOKENS = new Set(['for', 'in']);
const STATE_TOKENS = new Set([
  'multiplication_cost',
  'accumulation_cost',
  'base_val',
  'out',
  'rep',
  'num_terms',
  'R',
]);

export function tokenizePseudocodeLine(code) {
  if (code.trimStart().startsWith('#')) {
    return [{ text: code, kind: 'comment' }];
  }

  const parts = code.match(/\s+|[A-Za-z_][A-Za-z0-9_]*|\d+|./g) ?? [];

  return parts.map((part, idx) => {
    if (/^\s+$/.test(part)) return { text: part, kind: 'plain' };
    if (/^\d+$/.test(part)) return { text: part, kind: 'number' };
    if (KEYWORD_TOKENS.has(part)) return { text: part, kind: 'keyword' };
    if (STATE_TOKENS.has(part)) return { text: part, kind: 'state' };

    if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(part)) {
      const nextNonSpace = parts.slice(idx + 1).find((token) => !/^\s+$/.test(token));
      if (nextNonSpace === '(') return { text: part, kind: 'function' };
    }

    return { text: part, kind: 'plain' };
  });
}

export function pickDefaultOrbitRow(orbitRows) {
  if (!Array.isArray(orbitRows) || orbitRows.length === 0) return -1;

  let bestIdx = 0;
  for (let idx = 1; idx < orbitRows.length; idx += 1) {
    const row = orbitRows[idx] ?? {};
    const best = orbitRows[bestIdx] ?? {};
    const rowScore = [
      row.outputCount ?? 0,
      row.orbitSize ?? 0,
      row.outputs?.length ?? 0,
      row.orbitTuples?.length ?? 0,
    ];
    const bestScore = [
      best.outputCount ?? 0,
      best.orbitSize ?? 0,
      best.outputs?.length ?? 0,
      best.orbitTuples?.length ?? 0,
    ];

    if (
      rowScore[0] > bestScore[0] ||
      (rowScore[0] === bestScore[0] && rowScore[1] > bestScore[1]) ||
      (rowScore[0] === bestScore[0] && rowScore[1] === bestScore[1] && rowScore[2] > bestScore[2]) ||
      (rowScore[0] === bestScore[0] &&
        rowScore[1] === bestScore[1] &&
        rowScore[2] === bestScore[2] &&
        rowScore[3] > bestScore[3])
    ) {
      bestIdx = idx;
    }
  }

  return bestIdx;
}
