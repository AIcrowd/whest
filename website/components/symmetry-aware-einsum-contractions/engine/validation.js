import { parseCycleNotation, cyclesToArrayForm, generatorIndices } from './cycleParser.js';
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
  ellipsisUnsupportedError,
  incompatibleDomainMoveError,
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

  // V3.1 §3: ellipsis / broadcasting is unsupported. Detect a "." anywhere
  // in subscripts or output before the lowercase-only checks complain — the
  // dedicated message is more actionable than "use lowercase letters only".
  if (subscripts && subscripts.indexOf('.') !== -1) {
    errors.push(ellipsisUnsupportedError({ where: 'subscripts' }));
  }
  if (output && output.indexOf('.') !== -1) {
    errors.push(ellipsisUnsupportedError({ where: 'output' }));
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

      // 11. Each subscript must be lowercase letters only.
      // When the offending characters are only "." we let the dedicated
      // ellipsisUnsupportedError carry the message — emit the lowercase
      // complaint only if non-dot, non-letter characters are present, or if
      // letters are mixed with dots in a way the strip-dots fix wouldn't
      // help.
      const isLowercaseOnly = /^[a-z]+$/.test(sub);
      const isPureEllipsis = sub.length > 0 && /^\.+$/.test(sub);
      if (!isLowercaseOnly && !isPureEllipsis) {
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

  // 14. Output must be lowercase letters only.
  // Same suppression rule as subscripts: when the only offender is the
  // literal ellipsis we let ellipsisUnsupportedError carry the message.
  if (output && !/^[a-z]*$/.test(output) && !/^\.+$/.test(output)) {
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

  // V3.1 §3: a custom generator's cycle must keep axes within the same label
  // "domain" — i.e. it can't permute a *summed* label onto a *free* (output-
  // facing) label. The engine's wreath construction needs every cycle to act
  // within one label-class so the resulting orbit lives consistently in
  // either Σ-summed or Π-output space.
  const outputLabelSet = new Set();
  if (output) {
    for (const ch of output) {
      if (/[a-z]/.test(ch)) outputLabelSet.add(ch);
    }
  }
  for (let varIdx = 0; varIdx < variables.length; varIdx += 1) {
    const variable = variables[varIdx];
    if (variable.symmetry !== 'custom') continue;
    if (!variable.generators || variable.generators.trim() === '') continue;
    const { generators: parsedGens, error: parseErr } = parseCycleNotation(variable.generators);
    if (parseErr) continue;

    // Find the first operand using this variable so we can read its labels.
    const opIdx = opNames.findIndex((n) => n === variable.name.trim());
    if (opIdx < 0 || opIdx >= subs.length) continue;
    const sub = subs[opIdx];
    const axes = Array.isArray(variable.symAxes)
      ? variable.symAxes
      : [...Array(variable.rank).keys()];
    // Only run the domain check when the subscript is well-formed enough.
    if (sub.length !== variable.rank) continue;

    const axisLabels = axes.map((a) => sub[a]).filter((ch) => /[a-z]/.test(ch));
    if (axisLabels.length !== axes.length) continue;

    for (const cycles of parsedGens) {
      const perm = cyclesToArrayForm(cycles, axes.length);
      for (let pos = 0; pos < perm.length; pos += 1) {
        const target = perm[pos];
        if (target === pos) continue; // fixed point — no domain crossing
        const fromLabel = axisLabels[pos];
        const toLabel = axisLabels[target];
        if (fromLabel == null || toLabel == null) continue;
        if (fromLabel === toLabel) continue;
        // Cycle moves between two different labels — that's only admissible
        // when both labels live in the same "summed-vs-free" class.
        const fromIsFree = outputLabelSet.has(fromLabel);
        const toIsFree = outputLabelSet.has(toLabel);
        if (fromIsFree !== toIsFree) {
          errors.push(
            incompatibleDomainMoveError(varIdx, {
              name: variable.name,
              fromLabel,
              toLabel,
            }),
          );
          break;
        }
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
