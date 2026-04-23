/**
 * Python code generator for the symmetry explorer.
 *
 * Produces a runnable snippet that uses the `whest` library (imported as `we`)
 * to construct tensors with the requested symmetries and call `we.einsum_path`.
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tupleExpr(values) {
  if (values.length === 1) {
    return `(${values[0]},)`;
  }
  return `(${values.join(', ')})`;
}

/**
 * Parse cycle notation such as "(0 1)(2 3), (0 2)(1 3)" into local generator
 * literals for SymmetryGroup.from_generators(...).
 *
 * The cycle labels refer to the selected tensor axes in `symAxes`, not to the
 * local positions within the symmetry group. We therefore map the cycle labels
 * through the `symAxes` order and emit permutation arrays on
 * `range(len(symAxes))`.
 */
function parseGeneratorLiterals(input, symAxes) {
  if (!input || !input.trim()) return [];

  const generators = [];
  let depth = 0;
  let current = '';
  for (const ch of input) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ',' && depth === 0) {
      generators.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim()) generators.push(current.trim());

  const axisToLocal = new Map(symAxes.map((axis, idx) => [axis, idx]));

  return generators.map((gen) => {
    const arr = Array.from({ length: symAxes.length }, (_, idx) => idx);
    const cycleRe = /\(([^)]+)\)/g;
    let matched = false;
    let m;

    while ((m = cycleRe.exec(gen)) !== null) {
      matched = true;
      const cycleAxes = m[1].trim().split(/\s+/).map((value) => Number.parseInt(value, 10));
      const cycle = cycleAxes.map((axis) => axisToLocal.get(axis));
      if (cycle.some((idx) => idx === undefined)) {
        return null;
      }
      for (let i = 0; i < cycle.length; i += 1) {
        arr[cycle[i]] = cycle[(i + 1) % cycle.length];
      }
    }

    if (!matched) {
      return null;
    }
    return `[${arr.join(', ')}]`;
  }).filter(Boolean);
}

/**
 * Build the Python expression that creates the symmetry group for a variable.
 */
function buildGroupExpr(variable) {
  const { symmetry, symAxes, generators } = variable;
  const axesTuple = tupleExpr(symAxes);

  switch (symmetry) {
    case 'symmetric':
      return `we.SymmetryGroup.symmetric(axes=${axesTuple})`;
    case 'cyclic':
      return `we.SymmetryGroup.cyclic(axes=${axesTuple})`;
    case 'dihedral':
      return `we.SymmetryGroup.dihedral(axes=${axesTuple})`;
    case 'custom': {
      const gens = parseGeneratorLiterals(generators || '', symAxes);
      return `we.SymmetryGroup.from_generators([${gens.join(', ')}], axes=${axesTuple})`;
    }
    default:
      return null;
  }
}

/**
 * Build the shape tuple string for a variable, e.g. "(n, n, n)".
 */
function shapeExpr(rank) {
  return tupleExpr(new Array(rank).fill('n'));
}

function indentLines(lines, spaces = 4) {
  const prefix = ' '.repeat(spaces);
  return lines.map((line) => (line ? `${prefix}${line}` : line));
}

function buildGroupLines(variable) {
  const { symmetry, symAxes, generators } = variable;
  const axesTuple = tupleExpr(symAxes);

  switch (symmetry) {
    case 'symmetric':
      return [
        'we.SymmetryGroup.symmetric(',
        `    axes=${axesTuple},`,
        ')',
      ];
    case 'cyclic':
      return [
        'we.SymmetryGroup.cyclic(',
        `    axes=${axesTuple},`,
        ')',
      ];
    case 'dihedral':
      return [
        'we.SymmetryGroup.dihedral(',
        `    axes=${axesTuple},`,
        ')',
      ];
    case 'custom': {
      const gens = parseGeneratorLiterals(generators || '', symAxes);
      const lines = ['we.SymmetryGroup.from_generators('];
      lines.push(`    [${gens.join(', ')}],`);
      lines.push(`    axes=${axesTuple},`);
      lines.push(')');
      return lines;
    }
    default:
      return [];
  }
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Generate a complete, runnable Python snippet for the current explorer state.
 *
 * @param {Array<{name:string, rank:number, symmetry:string, symAxes:number[], generators?:string}>} variables
 * @param {string} subscripts  e.g. "aijk,ab"
 * @param {string} output      e.g. "ijkb"
 * @param {string} operandNames e.g. "T, W"
 * @param {number} dimensionN
 * @returns {string} Python source code
 */
export function generatePython(variables, subscripts, output, operandNames, dimensionN) {
  const lines = [];

  // --- Import ---------------------------------------------------------------
  lines.push('import whest as we');
  lines.push('');

  // --- Dimension ------------------------------------------------------------
  lines.push(`n = ${dimensionN}`);
  lines.push('');

  // --- Variables ------------------------------------------------------------
  lines.push('# --- Variables ---');

  const seen = new Set();
  for (const v of variables) {
    if (seen.has(v.name)) continue;
    seen.add(v.name);

    const shape = shapeExpr(v.rank);
    if (v.symmetry === 'none') {
      lines.push(`${v.name} = we.random.randn(${new Array(v.rank).fill('n').join(', ')})`);
    } else {
      lines.push(`${v.name} = we.random.symmetric(`);
      lines.push(`    ${shape},`);
      lines.push(...indentLines(buildGroupLines(v)));
      lines.push(')');
    }
  }

  lines.push('');

  // --- Expression -----------------------------------------------------------
  lines.push('# --- Expression ---');
  const names = operandNames.split(',').map((s) => s.trim()).join(', ');
  const einsumStr = `'${subscripts}->${output}'`;
  lines.push('path, info = we.einsum_path(');
  lines.push(`    ${einsumStr},`);
  lines.push(`    ${names},`);
  lines.push(')');
  lines.push('print(info)');
  lines.push('');

  return lines.join('\n');
}
