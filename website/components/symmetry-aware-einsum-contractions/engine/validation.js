import { parseCycleNotation, generatorIndices } from './cycleParser.js';
import {
  nameEmptyError,
  rankTooSmallError,
  namedSymAxesTooFewError,
  namedSymAxisOorError,
  customGeneratorsEmptyError,
  customGeneratorsParseError,
  customGeneratorAxisOorError,
  noOperandsError,
  subscriptsOperandsCountMismatchError,
  operandUndefinedError,
  subscriptNonLowercaseError,
  subscriptDuplicateLabelError,
  subscriptLengthMismatchError,
  outputNonLowercaseError,
  outputDuplicateLabelError,
  outputLabelMissingError,
} from './validationMessages.js';

/**
 * Validate all variable definitions and the einsum expression.
 *
 * Returns an array of structured errors, each of the shape
 *   { code, field, message, fix? }
 * (see validationMessages.js). The legacy consumer pattern of
 * `err.includes('...')` still works on `err.message`; anything new
 * should switch to `err.code` or `err.field`.
 *
 * @param {Array<{name:string, rank:number, symmetry:string, symAxes:number[], generators:string}>} variables
 * @param {string} subscripts  – comma-separated subscript strings, e.g. "aijk,ab"
 * @param {string} output      – output subscript, e.g. "ijkb"
 * @param {string} operandNames – comma-separated operand names, e.g. "T, W"
 * @returns {{ valid: boolean, errors: Array<{code:string, field:string, message:string, fix?: {label:string, apply:Function}}> }}
 */
export function validateAll(variables, subscripts, output, operandNames) {
  const errors = [];

  // ── Variable-level checks ──────────────────────────────────────────

  const varMap = new Map();
  for (let i = 0; i < variables.length; i += 1) {
    const v = variables[i];

    // 1. Name must be non-empty
    if (!v.name || v.name.trim() === '') {
      errors.push(nameEmptyError(i));
      continue;
    }

    varMap.set(v.name.trim(), v);

    // 2. Rank must be >= 1
    if (v.rank == null || v.rank < 1) {
      errors.push(rankTooSmallError(i, { name: v.name, rank: v.rank }));
    }

    const isNamed = ['symmetric', 'cyclic', 'dihedral'].includes(v.symmetry);

    // 3. Named symmetries require symAxes with at least 2 elements
    if (isNamed) {
      if (!Array.isArray(v.symAxes) || v.symAxes.length < 2) {
        errors.push(
          namedSymAxesTooFewError(i, {
            name: v.name,
            rank: v.rank,
            symmetry: v.symmetry,
          }),
        );
      } else {
        // 4. Named symmetry axis indices must be in [0, rank)
        for (const idx of v.symAxes) {
          if (idx < 0 || idx >= v.rank) {
            errors.push(
              namedSymAxisOorError(i, {
                name: v.name,
                badIdx: idx,
                rank: v.rank,
              }),
            );
          }
        }
      }
    }

    // 5–7. Custom symmetry
    if (v.symmetry === 'custom') {
      const axesExplicitlySelected = Array.isArray(v.symAxes);
      const axesCount = axesExplicitlySelected ? v.symAxes.length : v.rank;
      if (!v.generators || v.generators.trim() === '') {
        errors.push(customGeneratorsEmptyError(i, { name: v.name, axesCount }));
      } else {
        const { generators: parsedGenerators, error: parseError } =
          parseCycleNotation(v.generators);

        if (parseError) {
          errors.push(
            customGeneratorsParseError(i, {
              name: v.name,
              axesCount,
              rawError: parseError,
            }),
          );
        } else {
          for (const idx of generatorIndices(parsedGenerators)) {
            if (idx < 0 || idx >= axesCount) {
              errors.push(
                customGeneratorAxisOorError(i, {
                  name: v.name,
                  badIdx: idx,
                  axesCount,
                  axesExplicitlySelected,
                }),
              );
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
    errors.push(noOperandsError());
  }

  // 9. Number of subscripts must equal number of operands
  if (subs.length !== opNames.length) {
    errors.push(
      subscriptsOperandsCountMismatchError({
        subsCount: subs.length,
        opsCount: opNames.length,
      }),
    );
  }

  // 10–13: per-operand checks
  const allInputLabels = new Set();

  for (let i = 0; i < opNames.length; i += 1) {
    const name = opNames[i];

    // 10. All operand names must reference defined variable names
    if (!varMap.has(name)) {
      errors.push(operandUndefinedError({ name }));
    }

    if (i < subs.length) {
      const sub = subs[i];

      // 11. Each subscript must be lowercase letters only
      if (!/^[a-z]+$/.test(sub)) {
        errors.push(subscriptNonLowercaseError({ opIdx: i, sub, name }));
      }

      // 12. No duplicate labels within a single subscript
      const seen = new Set();
      for (const ch of sub) {
        if (seen.has(ch)) {
          errors.push(subscriptDuplicateLabelError({ sub, name, ch }));
          break;
        }
        seen.add(ch);
      }

      // 13. Each subscript length must match the referenced variable's rank
      const v = varMap.get(name);
      if (v && sub.length !== v.rank) {
        errors.push(
          subscriptLengthMismatchError({
            opIdx: i,
            sub,
            name,
            rank: v.rank,
            subscriptsStr: subscripts || '',
            outputStr: output || '',
          }),
        );
      }

      for (const ch of sub) {
        allInputLabels.add(ch);
      }
    }
  }

  // 14. Output must be lowercase letters only
  if (output && !/^[a-z]*$/.test(output)) {
    errors.push(outputNonLowercaseError({ outputStr: output }));
  }

  // 15. Explicit output labels must be unique
  if (output) {
    const seenOutput = new Set();
    for (const ch of output) {
      if (!/[a-z]/.test(ch)) continue;
      if (seenOutput.has(ch)) {
        errors.push(outputDuplicateLabelError({ ch }));
        break;
      }
      seenOutput.add(ch);
    }
  }

  // 16. All output labels must exist in at least one input subscript
  if (output) {
    for (const ch of output) {
      if (/[a-z]/.test(ch) && !allInputLabels.has(ch)) {
        errors.push(outputLabelMissingError({ ch }));
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
