import { parseCycleNotation, generatorIndices } from './cycleParser.js';

// ── Friendly hints for the custom-generators field ───────────────────
// Each hint is parametrised by the variable's selected axes count, so
// suggestions always refer to indices the user can actually use.

function cycleExample(axesCount) {
  if (axesCount >= 4) return '(0 1)(2 3)';
  if (axesCount === 3) return '(0 1 2)';
  return '(0 1)';
}

function indexRange(axesCount) {
  if (axesCount <= 1) return '0';
  return `0–${axesCount - 1}`;
}

function emptyGeneratorHint(varName, axesCount) {
  if (axesCount === 2) {
    return `Variable "${varName}": add a cycle like (0 1) to swap the two selected axes.`;
  }
  if (axesCount === 3) {
    return `Variable "${varName}": add a cycle like (0 1) to swap two axes, or (0 1 2) to rotate all three.`;
  }
  return `Variable "${varName}": add at least one cycle, e.g. ${cycleExample(axesCount)}. Separate multiple generators with commas: (0 1), (1 2).`;
}

function outOfRangeHint(varName, idx, axesCount) {
  return `Variable "${varName}": index ${idx} is outside your ${axesCount} selected ${axesCount === 1 ? 'axis' : 'axes'} — valid indices are ${indexRange(axesCount)}.`;
}

function prettifyParseError(varName, axesCount, raw) {
  const prefix = `Variable "${varName}": `;
  const example = cycleExample(axesCount);
  const range = indexRange(axesCount);

  if (raw.startsWith('Empty cycle')) {
    return `${prefix}empty cycle () — write cycles like ${example}.`;
  }
  if (raw.startsWith('Cycle must contain at least 2 elements')) {
    const m = raw.match(/\(([^)]*)\)/);
    const inner = m ? m[1] : '';
    return `${prefix}the cycle (${inner}) is too short — cycles need ≥ 2 elements. Try something like ${example}.`;
  }
  if (raw.startsWith('Invalid element')) {
    const m = raw.match(/"([^"]*)"/);
    const bad = m ? m[1] : 'that token';
    return `${prefix}"${bad}" isn't a valid index — use whole numbers ${range}, e.g. ${example}.`;
  }
  if (raw.startsWith('Duplicate index')) {
    const m = raw.match(/index (\d+) within cycle \(([^)]*)\)/);
    if (m) {
      return `${prefix}axis ${m[1]} appears twice in (${m[2]}) — each cycle can visit an axis at most once.`;
    }
    return `${prefix}an axis appears twice inside one cycle — each cycle can visit an axis at most once.`;
  }
  if (raw.startsWith('Unexpected characters outside parentheses')) {
    const m = raw.match(/"([^"]*)"/);
    const leftover = m ? m[1] : '';
    return `${prefix}"${leftover}" is outside any cycle — every cycle must be wrapped in parentheses, e.g. ${example}.`;
  }
  if (raw.startsWith('No cycles found')) {
    return `${prefix}no cycles detected — try ${example}.`;
  }
  if (/^Index \d+ appears in more than one cycle/.test(raw)) {
    const m = raw.match(/Index (\d+)/);
    const idx = m ? m[1] : 'an axis';
    return `${prefix}axis ${idx} appears in more than one cycle of the same generator — within one generator cycles must be disjoint. Split them with a comma, e.g. (0 1), (0 2).`;
  }
  // Fallback: never drop information.
  return `${prefix}${raw}`;
}

/**
 * Validate all variable definitions and the einsum expression.
 *
 * @param {Array<{name:string, rank:number, symmetry:string, symAxes:number[], generators:string}>} variables
 * @param {string} subscripts  – comma-separated subscript strings, e.g. "aijk,ab"
 * @param {string} output      – output subscript, e.g. "ijkb"
 * @param {string} operandNames – comma-separated operand names, e.g. "T, W"
 * @returns {{ valid: boolean, errors: string[] }}
 */
export function validateAll(variables, subscripts, output, operandNames) {
  const errors = [];

  // ── Variable-level checks ──────────────────────────────────────────

  const varMap = new Map();
  for (const v of variables) {
    // 1. Name must be non-empty
    if (!v.name || v.name.trim() === '') {
      errors.push('Variable name must be non-empty.');
      continue;
    }

    varMap.set(v.name.trim(), v);

    // 2. Rank must be >= 1
    if (v.rank == null || v.rank < 1) {
      errors.push(`Variable "${v.name}": rank must be >= 1.`);
    }

    const isNamed = ['symmetric', 'cyclic', 'dihedral'].includes(v.symmetry);

    // 3. Named symmetries require symAxes with at least 2 elements
    if (isNamed) {
      if (!Array.isArray(v.symAxes) || v.symAxes.length < 2) {
        errors.push(
          `Variable "${v.name}": ${v.symmetry} symmetry requires at least 2 symmetry axes.`
        );
      } else {
        // 4. Named symmetry axis indices must be < rank
        for (const idx of v.symAxes) {
          if (idx >= v.rank) {
            errors.push(
              `Variable "${v.name}": symmetry axis index ${idx} is out of range (rank ${v.rank}).`
            );
          }
        }
      }
    }

    // 5. Custom symmetry requires non-empty generators string
    if (v.symmetry === 'custom') {
      const axesCount = Array.isArray(v.symAxes) ? v.symAxes.length : v.rank;
      if (!v.generators || v.generators.trim() === '') {
        errors.push(emptyGeneratorHint(v.name, axesCount));
      } else {
        // 6. Custom generators must parse successfully
        const { generators: parsedGenerators, error: parseError } =
          parseCycleNotation(v.generators);

        if (parseError) {
          errors.push(prettifyParseError(v.name, axesCount, parseError));
        } else {
          // 7. Custom generator cycle indices must be within range of selected axes count
          for (const idx of generatorIndices(parsedGenerators)) {
            if (idx >= axesCount) {
              errors.push(outOfRangeHint(v.name, idx, axesCount));
            }
          }
        }
      }
    }
  }

  // ── Expression-level checks ────────────────────────────────────────

  const opNames = operandNames
    ? operandNames.split(',').map((s) => s.trim()).filter(Boolean)
    : [];
  const subs = subscripts
    ? subscripts.split(',').map((s) => s.trim())
    : [];

  // 8. At least one operand
  if (opNames.length === 0) {
    errors.push('Expression must have at least one operand.');
  }

  // 9. Number of subscripts must equal number of operands
  if (subs.length !== opNames.length) {
    errors.push(
      `Number of subscripts (${subs.length}) must equal number of operands (${opNames.length}).`
    );
  }

  // 10–13: per-operand checks
  const allInputLabels = new Set();

  for (let i = 0; i < opNames.length; i++) {
    const name = opNames[i];

    // 10. All operand names must reference defined variable names
    if (!varMap.has(name)) {
      errors.push(`Operand "${name}" does not reference a defined variable.`);
    }

    if (i < subs.length) {
      const sub = subs[i];

      // 11. Each subscript must be lowercase letters only
      if (!/^[a-z]+$/.test(sub)) {
        errors.push(
          `Subscript "${sub}" for operand "${name}" must contain only lowercase letters.`
        );
      }

      // 12. No duplicate labels within a single subscript
      const seen = new Set();
      for (const ch of sub) {
        if (seen.has(ch)) {
          errors.push(
            `Subscript "${sub}" for operand "${name}" has duplicate label "${ch}".`
          );
          break;
        }
        seen.add(ch);
      }

      // 13. Each subscript length must match the referenced variable's rank
      const v = varMap.get(name);
      if (v && sub.length !== v.rank) {
        errors.push(
          `Subscript "${sub}" length (${sub.length}) does not match variable "${name}" rank (${v.rank}).`
        );
      }

      for (const ch of sub) {
        allInputLabels.add(ch);
      }
    }
  }

  // 14. Output must be lowercase letters only
  if (output && !/^[a-z]*$/.test(output)) {
    errors.push('Output subscript must contain only lowercase letters.');
  }

  // 15. All output labels must exist in at least one input subscript
  if (output) {
    for (const ch of output) {
      if (/[a-z]/.test(ch) && !allInputLabels.has(ch)) {
        errors.push(
          `Output label "${ch}" does not appear in any input subscript.`
        );
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
