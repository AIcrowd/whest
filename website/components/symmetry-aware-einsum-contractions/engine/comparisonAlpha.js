// website/components/symmetry-aware-einsum-contractions/engine/comparisonAlpha.js
//
// Pedagogical helper: compute what α would be if orbit compression used
// G_expr (the expression-level counting symmetry V-sub × S(W)) instead of
// G_pt (the per-tuple group). G_expr over-compresses because dummy-rename
// orbits can contain tuples with different summand values.
//
// This result is NEVER used for the real cost display — only shown in the
// optional comparison toggle in TotalCostView.

function applyPerm(tuple, perm) {
  const out = new Array(tuple.length);
  for (let i = 0; i < tuple.length; i += 1) out[perm.arr[i]] = tuple[i];
  return out;
}

function allAssignments(sizes) {
  const out = [];
  const cur = new Array(sizes.length).fill(0);
  function rec(idx) {
    if (idx === sizes.length) { out.push([...cur]); return; }
    for (let v = 0; v < sizes[idx]; v += 1) { cur[idx] = v; rec(idx + 1); }
  }
  if (sizes.length === 0) out.push([]); else rec(0);
  return out;
}

/**
 * Compute α that would result from applying orbit-based compression under
 * G_expr instead of G_pt. Purely didactic — never used for the real cost
 * display.
 *
 * Performs orbit enumeration at the GLOBAL level (all einsum labels together)
 * under G_expr, projecting each orbit to the free (V) labels and counting
 * distinct output bins touched. This matches the naive approach that conflates
 * counting symmetry with per-tuple symmetry.
 *
 * Returns null if:
 *   - expressionGroup is missing / empty, or
 *   - G_expr and G_pt have the same order (no comparison to show).
 */
export function computeExpressionAlphaTotal({ analysis }) {
  const expressionGroup = analysis?.expressionGroup;
  const perTupleGroup = analysis?.symmetry;
  if (!expressionGroup?.elements?.length) return null;

  // Quick check: if G_expr has the same order as G_pt, they're identical in
  // effect and there's no meaningful comparison to show.
  const gExprOrder = expressionGroup.elements.length;
  const gPtOrder = perTupleGroup?.fullElements?.length ?? 1;
  if (gExprOrder === gPtOrder) return null;

  const globalLabels = perTupleGroup?.allLabels ?? [];
  const vLabels = perTupleGroup?.vLabels ?? [];
  if (!globalLabels.length) return null;

  // Build the global sizes array and the V-positions for projection.
  // Use the clusters from analysis for size-aware cases, falling back to n=1.
  const clusters = analysis?.clusters ?? [];
  const sizeByLabel = new Map();
  for (const c of clusters) {
    for (const label of c.labels) sizeByLabel.set(label, c.size);
  }
  const sizes = globalLabels.map((l) => sizeByLabel.get(l) ?? 1);
  const vPos = vLabels.map((l) => globalLabels.indexOf(l));

  // Global orbit enumeration under G_expr on [n]^|allLabels|.
  const elements = expressionGroup.elements;
  const keyFn = (t) => t.join('|');
  const remaining = new Map();
  for (const a of allAssignments(sizes)) remaining.set(keyFn(a), a);

  let totalAlpha = 0;
  while (remaining.size > 0) {
    const [, rep] = remaining.entries().next().value;
    const orbit = new Map();
    for (const g of elements) {
      const moved = applyPerm(rep, g);
      const k = keyFn(moved);
      if (!orbit.has(k)) orbit.set(k, moved);
      remaining.delete(k);
    }
    const projected = new Set();
    for (const m of orbit.values()) {
      projected.add(vPos.map((p) => m[p]).join('|'));
    }
    totalAlpha += projected.size;
  }

  return totalAlpha;
}
