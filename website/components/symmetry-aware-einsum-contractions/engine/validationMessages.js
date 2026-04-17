/**
 * Friendly message + fix builders for every validation rule in the
 * symmetry-aware einsum explorer.
 *
 * Each builder returns a structured error of the shape
 *   {
 *     code:    stable machine-readable identifier,
 *     field:   the form field that, when touched, should reveal this error,
 *     message: human-readable prose parametrised by the current config,
 *     fix?:    { label, apply } — an optional single-click remedy.
 *   }
 *
 * `fix.apply(state)` is a pure transform over the ExampleChooser form
 * state `{ variables, subscriptsStr, outputStr, operandNamesStr }`.
 */

export const ERROR_CODES = Object.freeze({
  NAME_EMPTY: 'name-empty',
  RANK_TOO_SMALL: 'rank-too-small',
  NAMED_SYM_AXES_TOO_FEW: 'named-sym-axes-too-few',
  NAMED_SYM_AXIS_OOR: 'named-sym-axis-oor',
  CUSTOM_GENERATORS_EMPTY: 'custom-generators-empty',
  CUSTOM_GENERATORS_PARSE: 'custom-generators-parse',
  CUSTOM_GENERATOR_AXIS_OOR: 'custom-generator-axis-oor',
  NO_OPERANDS: 'no-operands',
  SUBSCRIPTS_OPERANDS_COUNT_MISMATCH: 'subscripts-operands-count-mismatch',
  OPERAND_UNDEFINED: 'operand-undefined',
  SUBSCRIPT_NON_LOWERCASE: 'subscript-non-lowercase',
  SUBSCRIPT_DUPLICATE_LABEL: 'subscript-duplicate-label',
  SUBSCRIPT_LENGTH_MISMATCH: 'subscript-length-mismatch',
  OUTPUT_NON_LOWERCASE: 'output-non-lowercase',
  OUTPUT_LABEL_MISSING: 'output-label-missing',
});

export function varField(i, kind) {
  return `var-${i}-${kind}`;
}

// ── Private helpers ──────────────────────────────────────────────────

function cycleExample(axesCount) {
  if (axesCount >= 4) return '(0 1)(2 3)';
  if (axesCount === 3) return '(0 1 2)';
  return '(0 1)';
}

function indexRange(axesCount) {
  if (axesCount <= 1) return '0';
  return `0–${axesCount - 1}`;
}

function usedLabelsInExpression(subscriptsStr, outputStr) {
  const used = new Set();
  for (const ch of `${subscriptsStr}${outputStr}`) {
    if (/[a-z]/.test(ch)) used.add(ch);
  }
  return used;
}

function nextFreshLabel(used) {
  for (let code = 97; code <= 122; code += 1) {
    const ch = String.fromCharCode(code);
    if (!used.has(ch)) return ch;
  }
  return 'a';
}

function replaceNthSubscript(subscriptsStr, n, newValue) {
  const parts = subscriptsStr.split(',');
  if (n < 0 || n >= parts.length) return subscriptsStr;
  const original = parts[n];
  const prefix = original.match(/^\s*/)[0];
  const suffix = original.match(/\s*$/)[0];
  parts[n] = `${prefix}${newValue}${suffix}`;
  return parts.join(',');
}

function updateVariable(state, i, patch) {
  return {
    ...state,
    variables: state.variables.map((v, idx) => (idx === i ? { ...v, ...patch } : v)),
  };
}

// ── Variable-level builders ──────────────────────────────────────────

export function nameEmptyError(i) {
  return {
    code: ERROR_CODES.NAME_EMPTY,
    field: varField(i, 'name'),
    message:
      'A variable needs a name — the operand symbol used in the expression, like A or T.',
  };
}

export function rankTooSmallError(i, { name, rank }) {
  const current = rank == null ? 'unset' : String(rank);
  return {
    code: ERROR_CODES.RANK_TOO_SMALL,
    field: varField(i, 'rank'),
    message: `Variable "${name}" needs rank ≥ 1 (currently ${current}).`,
    fix: {
      label: 'Set rank to 1',
      apply: (state) => updateVariable(state, i, { rank: 1 }),
    },
  };
}

export function namedSymAxesTooFewError(i, { name, rank, symmetry }) {
  const base = {
    code: ERROR_CODES.NAMED_SYM_AXES_TOO_FEW,
    field: varField(i, 'axes'),
    message: `Variable "${name}"'s ${symmetry} symmetry needs at least 2 axes — select more axes above.`,
  };
  if (rank >= 2) {
    return {
      ...base,
      fix: {
        label: 'Select all axes',
        apply: (state) =>
          updateVariable(state, i, { symAxes: [...Array(rank).keys()] }),
      },
    };
  }
  return base;
}

export function namedSymAxisOorError(i, { name, badIdx, rank }) {
  return {
    code: ERROR_CODES.NAMED_SYM_AXIS_OOR,
    field: varField(i, 'axes'),
    message: `Variable "${name}" has rank ${rank}, so axis ${badIdx} doesn't exist — valid axes are ${indexRange(rank)}.`,
    fix: {
      label: `Remove axis ${badIdx}`,
      apply: (state) => {
        const v = state.variables[i];
        const cleaned = (v.symAxes || []).filter((a) => a !== badIdx);
        return updateVariable(state, i, { symAxes: cleaned });
      },
    },
  };
}

// ── Custom-generator builders ────────────────────────────────────────

export function customGeneratorsEmptyError(i, { name, axesCount }) {
  let message;
  if (axesCount === 2) {
    message = `Variable "${name}": add a cycle like (0 1) to swap the two selected axes.`;
  } else if (axesCount === 3) {
    message = `Variable "${name}": add a cycle like (0 1) to swap two axes, or (0 1 2) to rotate all three.`;
  } else {
    message = `Variable "${name}": add at least one cycle, e.g. ${cycleExample(axesCount)}. Separate multiple generators with commas: (0 1), (1 2).`;
  }
  return {
    code: ERROR_CODES.CUSTOM_GENERATORS_EMPTY,
    field: varField(i, 'generators'),
    message,
  };
}

export function customGeneratorsParseError(i, { name, axesCount, rawError }) {
  const example = cycleExample(axesCount);
  const range = indexRange(axesCount);
  const prefix = `Variable "${name}": `;

  let message;
  if (rawError.startsWith('Empty cycle')) {
    message = `${prefix}empty cycle () — write cycles like ${example}.`;
  } else if (rawError.startsWith('Cycle must contain at least 2 elements')) {
    const m = rawError.match(/\(([^)]*)\)/);
    const inner = m ? m[1] : '';
    message = `${prefix}the cycle (${inner}) is too short — cycles need ≥ 2 elements. Try something like ${example}.`;
  } else if (rawError.startsWith('Invalid element')) {
    const m = rawError.match(/"([^"]*)"/);
    const bad = m ? m[1] : 'that token';
    message = `${prefix}"${bad}" isn't a valid index — use whole numbers ${range}, e.g. ${example}.`;
  } else if (rawError.startsWith('Duplicate index')) {
    const m = rawError.match(/index (\d+) within cycle \(([^)]*)\)/);
    message = m
      ? `${prefix}axis ${m[1]} appears twice in (${m[2]}) — each cycle can visit an axis at most once.`
      : `${prefix}an axis appears twice inside one cycle — each cycle can visit an axis at most once.`;
  } else if (rawError.startsWith('Unexpected characters outside parentheses')) {
    const m = rawError.match(/"([^"]*)"/);
    const leftover = m ? m[1] : '';
    message = `${prefix}"${leftover}" is outside any cycle — every cycle must be wrapped in parentheses, e.g. ${example}.`;
  } else if (rawError.startsWith('No cycles found')) {
    message = `${prefix}no cycles detected — try ${example}.`;
  } else if (/^Index \d+ appears in more than one cycle/.test(rawError)) {
    const m = rawError.match(/Index (\d+)/);
    const idx = m ? m[1] : 'an axis';
    message = `${prefix}axis ${idx} appears in more than one cycle of the same generator — within one generator cycles must be disjoint. Split them with a comma, e.g. (0 1), (0 2).`;
  } else {
    message = `${prefix}${rawError}`;
  }

  return {
    code: ERROR_CODES.CUSTOM_GENERATORS_PARSE,
    field: varField(i, 'generators'),
    message,
  };
}

export function customGeneratorAxisOorError(i, { name, badIdx, axesCount }) {
  const axisWord = axesCount === 1 ? 'axis' : 'axes';
  return {
    code: ERROR_CODES.CUSTOM_GENERATOR_AXIS_OOR,
    field: varField(i, 'generators'),
    message: `Variable "${name}": index ${badIdx} is outside your ${axesCount} selected ${axisWord} — valid indices are ${indexRange(axesCount)}.`,
  };
}

// ── Expression-level builders ────────────────────────────────────────

export function noOperandsError() {
  return {
    code: ERROR_CODES.NO_OPERANDS,
    field: 'operands',
    message: "The expression needs at least one operand — like A in einsum('ij,j->i', A, b).",
  };
}

export function subscriptsOperandsCountMismatchError({ subsCount, opsCount }) {
  const plural = (n, s) => `${n} ${s}${n === 1 ? '' : 's'}`;
  return {
    code: ERROR_CODES.SUBSCRIPTS_OPERANDS_COUNT_MISMATCH,
    field: 'subscripts',
    message: `The expression has ${plural(subsCount, 'subscript')} but ${plural(opsCount, 'operand')} — add or remove one so the counts match.`,
  };
}

export function operandUndefinedError({ name }) {
  return {
    code: ERROR_CODES.OPERAND_UNDEFINED,
    field: 'operands',
    message: `Operand "${name}" isn't a defined variable — either rename the operand, or add a variable called "${name}".`,
  };
}

export function subscriptNonLowercaseError({ opIdx, sub, name }) {
  const cleaned = sub.toLowerCase().replace(/[^a-z]/g, '');
  const base = {
    code: ERROR_CODES.SUBSCRIPT_NON_LOWERCASE,
    field: 'subscripts',
    message: `Subscript "${sub}" for operand "${name}" must use lowercase letters a–z only.`,
  };
  if (cleaned !== sub && cleaned.length > 0) {
    return {
      ...base,
      fix: {
        label: `Clean to "${cleaned}"`,
        apply: (state) => ({
          ...state,
          subscriptsStr: replaceNthSubscript(state.subscriptsStr, opIdx, cleaned),
        }),
      },
    };
  }
  return base;
}

export function subscriptDuplicateLabelError({ sub, name, ch }) {
  return {
    code: ERROR_CODES.SUBSCRIPT_DUPLICATE_LABEL,
    field: 'subscripts',
    message: `Subscript "${sub}" for operand "${name}" uses "${ch}" twice — each axis needs its own letter.`,
  };
}

export function subscriptLengthMismatchError({
  opIdx,
  sub,
  name,
  rank,
  subscriptsStr,
  outputStr,
}) {
  const diff = rank - sub.length;
  const haveWord = `${sub.length} label${sub.length === 1 ? '' : 's'}`;
  if (diff > 0) {
    const tempUsed = new Set(usedLabelsInExpression(subscriptsStr, outputStr));
    let padded = sub;
    for (let k = 0; k < diff; k += 1) {
      const fresh = nextFreshLabel(tempUsed);
      padded += fresh;
      tempUsed.add(fresh);
    }
    const moreWord = `${diff} more label${diff === 1 ? '' : 's'}`;
    return {
      code: ERROR_CODES.SUBSCRIPT_LENGTH_MISMATCH,
      field: 'subscripts',
      message: `Variable "${name}" has rank ${rank} but subscript "${sub}" has ${haveWord} — add ${moreWord} (e.g. "${padded}"), or change "${name}"'s rank to ${sub.length}.`,
      fix: {
        label: `Pad to "${padded}"`,
        apply: (state) => ({
          ...state,
          subscriptsStr: replaceNthSubscript(state.subscriptsStr, opIdx, padded),
        }),
      },
    };
  }
  const extraWord = `${-diff} label${-diff === 1 ? '' : 's'}`;
  return {
    code: ERROR_CODES.SUBSCRIPT_LENGTH_MISMATCH,
    field: 'subscripts',
    message: `Variable "${name}" has rank ${rank} but subscript "${sub}" has ${haveWord} — drop ${extraWord}, or change "${name}"'s rank to ${sub.length}.`,
  };
}

export function outputNonLowercaseError({ outputStr }) {
  const cleaned = outputStr.toLowerCase().replace(/[^a-z]/g, '');
  const base = {
    code: ERROR_CODES.OUTPUT_NON_LOWERCASE,
    field: 'output',
    message: 'The output subscript must use lowercase letters a–z only.',
  };
  if (cleaned !== outputStr) {
    return {
      ...base,
      fix: {
        label: `Clean to "${cleaned}"`,
        apply: (state) => ({ ...state, outputStr: cleaned }),
      },
    };
  }
  return base;
}

export function outputLabelMissingError({ ch }) {
  return {
    code: ERROR_CODES.OUTPUT_LABEL_MISSING,
    field: 'output',
    message: `Output label "${ch}" doesn't appear in any input subscript — there's no source to index from. Remove "${ch}" from the output, or add it to one of the inputs.`,
    fix: {
      label: `Remove "${ch}"`,
      apply: (state) => ({
        ...state,
        outputStr: state.outputStr.split('').filter((c) => c !== ch).join(''),
      }),
    },
  };
}
