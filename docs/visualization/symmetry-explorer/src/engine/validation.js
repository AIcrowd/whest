import { parseCycleNotation, generatorIndices } from './cycleParser.js';

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
      if (!v.generators || v.generators.trim() === '') {
        errors.push(
          `Variable "${v.name}": custom symmetry requires a non-empty generators string.`
        );
      } else {
        // 6. Custom generators must parse successfully
        let parsed;
        try {
          parsed = parseCycleNotation(v.generators);
        } catch (e) {
          errors.push(
            `Variable "${v.name}": failed to parse generators – ${e.message}`
          );
          parsed = null;
        }

        // 7. Custom generator cycle indices must be within range of selected axes count
        if (parsed && parsed.generators) {
          const axesCount = Array.isArray(v.symAxes) ? v.symAxes.length : v.rank;
          const indices = generatorIndices(parsed.generators);
          for (const idx of indices) {
            if (idx >= axesCount) {
              errors.push(
                `Variable "${v.name}": generator cycle index ${idx} is out of range (${axesCount} axes selected).`
              );
            }
          }
        } else if (parsed && parsed.error) {
          errors.push(
            `Variable "${v.name}": ${parsed.error}`
          );
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
