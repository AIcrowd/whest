import { notationColor, notationLatex } from './notationSystem.js';
import { variableSymmetryLabel } from './symmetryLabel.js';

function normalizeSection1Example(example) {
  if (!example) return null;
  const variables = Array.isArray(example.variables) ? example.variables : [];

  const subscripts = Array.isArray(example.subscripts)
    ? example.subscripts
    : typeof example.expression?.subscripts === 'string'
      ? example.expression.subscripts.split(',').map((part) => part.trim()).filter(Boolean)
      : [];

  const operandNames = Array.isArray(example.operandNames)
    ? example.operandNames
    : typeof example.expression?.operandNames === 'string'
      ? example.expression.operandNames.split(',').map((part) => part.trim()).filter(Boolean)
      : [];

  const output = typeof example.output === 'string'
    ? example.output.trim()
    : typeof example.expression?.output === 'string'
      ? example.expression.output.trim()
      : '';

  return {
    ...example,
    subscripts,
    operandNames: operandNames.length > 0
      ? operandNames
      : subscripts.map((_, index) => variables[index]?.name || `T${index + 1}`),
    output,
    variables,
  };
}

function escapeTexttt(value) {
  return value
    .replace(/\\/g, String.raw`\textbackslash{}`)
    .replace(/([{}_$%&#])/g, String.raw`\\$1`)
    .replace(/\^/g, String.raw`\^{} `)
    .replace(/~/g, String.raw`\~{} `);
}

function uniqueChars(text) {
  return [...new Set((text || '').split('').filter(Boolean))];
}

function roleColor(label, freeSet, palette = {}) {
  return freeSet.has(label)
    ? (palette.freeLabelColor ?? notationColor('v_free'))
    : (palette.summedLabelColor ?? notationColor('w_summed'));
}

function coloredLabel(label, freeSet, palette = {}) {
  return String.raw`\textcolor{${roleColor(label, freeSet, palette)}}{${label}}`;
}

function coloredIndexList(labels, freeSet, palette = {}) {
  return labels.map((label) => coloredLabel(label, freeSet, palette)).join(',');
}

function formatOperandFactor(name, subscript, freeSet, palette = {}) {
  const base = name || 'T';
  const labels = uniqueChars(subscript);
  return `${base}[${coloredIndexList(labels, freeSet, palette)}]`;
}

function formatOutputTensor(output, freeSet, palette = {}) {
  const labels = uniqueChars(output);
  if (labels.length === 0) return 'R';
  return `R[${coloredIndexList(labels, freeSet, palette)}]`;
}

function formatDeclaredSymmetrySummary(variables) {
  if (!Array.isArray(variables) || variables.length === 0) return 'dense';
  return variables
    .map((variable) => `${variable.name || 'T'}: ${variableSymmetryLabel(variable)}`)
    .join(' · ');
}

export function buildSection1ExampleView(example, palette = {}) {
  const normalized = normalizeSection1Example(example);
  if (!normalized) return null;

  const { subscripts, operandNames, output, variables } = normalized;
  const allLabels = uniqueChars(subscripts.join(''));
  const freeLabels = uniqueChars(output);
  const freeSet = new Set(freeLabels);
  const summedLabels = allLabels.filter((label) => !freeSet.has(label));
  const operandList = operandNames.join(', ');
  const exactEinsumText = operandList.length > 0
    ? `einsum('${subscripts.join(',')}->${output}', ${operandList})`
    : `einsum('${subscripts.join(',')}->${output}')`;
  const exactEinsumLatex = String.raw`\texttt{${escapeTexttt(exactEinsumText)}}`;

  const expandedLeft = formatOutputTensor(output, freeSet, palette);
  const sumPart = summedLabels.length > 0
    ? String.raw`\sum_{${coloredIndexList(summedLabels, freeSet, palette)}}\;`
    : '';
  const product = subscripts
    .map((subscript, idx) => formatOperandFactor(operandNames[idx], subscript, freeSet, palette))
    .join(String.raw`\,\cdot\,`);

  return {
    exactEinsumText,
    exactEinsumLatex,
    expandedEquationLatex: `${expandedLeft} \\;=\\; ${sumPart}${product}`,
    operandCount: subscripts.length,
    labelCount: allLabels.length,
    operandSummary: operandList || '—',
    outputSummary: output || 'scalar',
    vFreeSummary: freeLabels.length ? freeLabels.join(', ') : '∅',
    wSummedSummary: summedLabels.length ? summedLabels.join(', ') : '∅',
    declaredSymmetrySummary: formatDeclaredSymmetrySummary(variables),
    vFreeNotation: notationLatex('v_free'),
    wSummedNotation: notationLatex('w_summed'),
  };
}
