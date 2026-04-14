/**
 * Cycle notation parser for symmetry group generators.
 *
 * Accepts comma-separated generators written in disjoint-cycle notation,
 * e.g. "(0 1)(2 3), (0 2)(1 3)", and converts them into structured data
 * suitable for the symmetry explorer engine.
 */

/**
 * Split a generator string on commas that are NOT inside parentheses.
 * Returns trimmed, non-empty segments.
 */
function splitGenerators(input) {
  const segments = [];
  let depth = 0;
  let start = 0;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (ch === "(") depth++;
    else if (ch === ")") depth--;
    else if (ch === "," && depth === 0) {
      segments.push(input.slice(start, i).trim());
      start = i + 1;
    }
  }

  const last = input.slice(start).trim();
  if (last.length > 0) segments.push(last);
  return segments;
}

/**
 * Parse a single generator string such as "(0 1)(2 3)" into an array of
 * cycles, where each cycle is an array of integer indices.
 *
 * Returns `{ cycles, error }`.
 */
function parseGenerator(text) {
  const cycles = [];
  // Match each (...) group
  const cycleRe = /\(([^)]*)\)/g;
  let match;
  let consumed = "";

  while ((match = cycleRe.exec(text)) !== null) {
    consumed += match[0];
    const inner = match[1].trim();
    if (inner.length === 0) {
      return { cycles: null, error: "Empty cycle ()" };
    }

    const tokens = inner.split(/\s+/);
    if (tokens.length < 2) {
      return {
        cycles: null,
        error: `Cycle must contain at least 2 elements, got (${inner})`,
      };
    }

    const indices = [];
    for (const tok of tokens) {
      if (!/^\d+$/.test(tok)) {
        return {
          cycles: null,
          error: `Invalid element "${tok}" — only non-negative integers are allowed`,
        };
      }
      indices.push(Number(tok));
    }

    // Check for duplicates within this cycle
    const seen = new Set();
    for (const idx of indices) {
      if (seen.has(idx)) {
        return {
          cycles: null,
          error: `Duplicate index ${idx} within cycle (${inner})`,
        };
      }
      seen.add(idx);
    }

    cycles.push(indices);
  }

  // Make sure the entire string was consumed (aside from whitespace)
  const remaining = text.replace(/\(([^)]*)\)/g, "").trim();
  if (remaining.length > 0) {
    return {
      cycles: null,
      error: `Unexpected characters outside parentheses: "${remaining}"`,
    };
  }

  if (cycles.length === 0) {
    return { cycles: null, error: "No cycles found — use e.g. (0 1)" };
  }

  // Check for duplicates across cycles within the same generator
  const allIndices = new Set();
  for (const cycle of cycles) {
    for (const idx of cycle) {
      if (allIndices.has(idx)) {
        return {
          cycles: null,
          error: `Index ${idx} appears in more than one cycle within the same generator`,
        };
      }
      allIndices.add(idx);
    }
  }

  return { cycles, error: null };
}

/**
 * Parse a comma-separated list of generators in cycle notation.
 *
 * @param {string} input — e.g. "(0 1)(2 3), (0 2)(1 3)"
 * @returns {{ generators: number[][][] | null, error: string | null }}
 */
export function parseCycleNotation(input) {
  if (typeof input !== "string" || input.trim().length === 0) {
    return { generators: null, error: "Input must be a non-empty string" };
  }

  const segments = splitGenerators(input);
  const generators = [];

  for (const seg of segments) {
    const { cycles, error } = parseGenerator(seg);
    if (error) {
      return { generators: null, error };
    }
    generators.push(cycles);
  }

  return { generators, error: null };
}

/**
 * Convert an array of disjoint cycles into array-form permutation.
 *
 * Each cycle [a, b, c] means a->b, b->c, c->a. Cycles are applied
 * left-to-right (first cycle first).
 *
 * @param {number[][]} cycles — e.g. [[0,1],[2,3]]
 * @param {number} size — length of the permutation array
 * @returns {number[]} — array-form permutation where position i maps to arr[i]
 */
export function cyclesToArrayForm(cycles, size) {
  // Start with identity
  const perm = Array.from({ length: size }, (_, i) => i);

  for (const cycle of cycles) {
    const len = cycle.length;
    for (let i = 0; i < len; i++) {
      const from = cycle[i];
      const to = cycle[(i + 1) % len];
      perm[from] = to;
    }
  }

  return perm;
}

/**
 * Collect all unique indices mentioned across all generators, sorted ascending.
 *
 * @param {number[][][]} generators — array of generators, each an array of cycles
 * @returns {number[]} — sorted unique indices
 */
export function generatorIndices(generators) {
  const seen = new Set();
  for (const gen of generators) {
    for (const cycle of gen) {
      for (const idx of cycle) {
        seen.add(idx);
      }
    }
  }
  return Array.from(seen).sort((a, b) => a - b);
}
