/**
 * Python code generator for the symmetry explorer.
 *
 * Produces a runnable snippet that uses the `whest` library (imported as `we`)
 * to construct tensors with the requested symmetries and call `we.einsum_path`.
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Parse a cycle-notation string such as "(0 1)(2 3), (0 2)(1 3)" into an
 * array of Python expression strings, one per generator.
 *
 * Each generator is a product of disjoint cycles written as `(a b c ...)`.
 * Generators are separated by commas that are *outside* parentheses.
 *
 * A single generator like `(0 1)(2 3)` becomes:
 *   "we.Permutation(we.Cycle(0, 1)(2, 3))"
 */
function parseGensForPython(input) {
  if (!input || !input.trim()) return [];

  // Split on commas that sit outside parentheses.
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

  return generators.map((gen) => {
    // gen looks like "(0 1)(2 3)" — possibly multiple cycles chained.
    // Convert each (a b c) into we.Cycle(a, b, c), then chain with ().
    const cycles = [];
    const cycleRe = /\(([^)]+)\)/g;
    let m;
    while ((m = cycleRe.exec(gen)) !== null) {
      const elements = m[1].trim().split(/\s+/).join(', ');
      cycles.push(`we.Cycle(${elements})`);
    }
    if (cycles.length === 0) return null;
    // Chain cycles: we.Cycle(0, 1)(2, 3)
    const chained = cycles[0] + cycles.slice(1).map((c) => {
      // Extract the args portion from "we.Cycle(args)"
      const inner = c.slice('we.Cycle'.length);
      return inner; // already "(a, b)"
    }).join('');
    return `we.Permutation(${chained})`;
  }).filter(Boolean);
}

/**
 * Build the Python expression that creates the symmetry group for a variable.
 */
function buildGroupExpr(variable) {
  const { symmetry, symAxes, generators } = variable;
  const k = symAxes.length;
  const axesTuple = `(${symAxes.join(',')})`;

  switch (symmetry) {
    case 'symmetric':
      return `we.PermutationGroup.symmetric(${k}, axes=${axesTuple})`;
    case 'cyclic':
      return `we.PermutationGroup.cyclic(${k}, axes=${axesTuple})`;
    case 'dihedral':
      return `we.PermutationGroup.dihedral(${k}, axes=${axesTuple})`;
    case 'custom': {
      const gens = parseGensForPython(generators || '');
      return `we.PermutationGroup(${gens.join(', ')}, axes=${axesTuple})`;
    }
    default:
      return null;
  }
}

/**
 * Build the shape tuple string for a variable, e.g. "(n, n, n)".
 */
function shapeExpr(rank) {
  return `(${new Array(rank).fill('n').join(', ')})`;
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

  // --- Helper (only when needed) --------------------------------------------
  const needsHelper = variables.some((v) => v.symmetry !== 'none');
  if (needsHelper) {
    lines.push('def my_symmetrize(shape, group):');
    lines.push('    """Random tensor with given symmetry (Reynolds averaged)."""');
    lines.push('    data = we.random.randn(*shape)');
    lines.push('    data = sum(we.transpose(data, g.array_form) for g in group.elements()) / group.order()');
    lines.push('    return we.as_symmetric(data, symmetry=group)');
    lines.push('');
  }

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
      const group = buildGroupExpr(v);
      lines.push(`${v.name} = my_symmetrize(${shape}, ${group})`);
    }
  }

  lines.push('');

  // --- Expression -----------------------------------------------------------
  lines.push('# --- Expression ---');
  const names = operandNames.split(',').map((s) => s.trim()).join(', ');
  const einsumStr = `'${subscripts}->${output}'`;
  lines.push(`path, info = we.einsum_path(${einsumStr}, ${names})`);
  lines.push('print(info)');
  lines.push('');

  return lines.join('\n');
}
