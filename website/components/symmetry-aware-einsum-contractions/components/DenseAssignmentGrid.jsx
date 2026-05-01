/**
 * DenseAssignmentGrid — V3.1 §6 — C06 (NEW)
 *
 * The first concrete object readers see, before product orbits are introduced.
 * Renders the FULL assignment space (no symmetry collapse yet) for a small
 * einsum like Cross S2 ('ij,k → ik'). Each assignment (i,j,k) is a clickable
 * cell in a 2D faceted grid: rows of (i,j) cells, one panel per k value.
 *
 * Selecting a cell shows the V3.1 detail block:
 *
 *     full assignment: (i=0, j=1, k=0)
 *     product: A[0,1] · B[0]
 *     output: R[0,0]
 *
 * Toggles:
 *   - "show products" — each cell labels the product term inline
 *   - "show outputs"  — each cell labels the output destination inline
 *
 * The grid is capped at n ≤ 4 — past that we render the cap message instead
 * of the dense grid (the cardinality grows as n^|labels|, so even n=5 with 3
 * labels is 125 cells).
 *
 * All colours via the design-system token map. No raw hex outside TOKEN.
 */

import { useCallback, useMemo, useState } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors colors_and_type.css tier 1 + 3A)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:      'var(--coral)',        // #F0524D
  coralLight: 'var(--coral-light)',  // #FEF2F1
  gray900:    'var(--gray-900)',     // #292C2D
  gray700:    'var(--gray-700)',     // #3F4142
  gray600:    'var(--gray-600)',     // #5D5F60
  gray500:    'var(--gray-500)',     // #888B8D
  gray400:    'var(--gray-400)',     // #AAACAD
  gray300:    'var(--gray-300)',     // #C4C7C8
  gray200:    'var(--gray-200)',     // #D9DCDC
  gray100:    'var(--gray-100)',     // #F1F3F5
  gray50:     'var(--gray-50)',      // #F8F9F9
  white:      'var(--white)',        // #FFFFFF
  einV:       'var(--ein-v)',        // #4A7CFF (visible/free label)
};

const RENDER_CAP = 4;

/* ─────────────────────────────────────────────────────────────────────────────
   Orbit colour palette — V3.1 §8 (C08 Product-Orbit Lens)
   Subtle, distinguishable outlines so the dense grid stays legible. We cycle
   through a small palette of CSS variables (already pinned in styles.css /
   colors_and_type.css). Token-only — no raw hex.
   ───────────────────────────────────────────────────────────────────────────── */
const ORBIT_PALETTE = [
  'var(--ein-v)',          // blue (visible)
  'var(--ein-w)',          // contracted
  'var(--coral)',          // coral
  'var(--accent-purple, #7E57C2)', // fallback purple
  'var(--accent-teal, #26A69A)',   // fallback teal
  'var(--accent-amber, #F59E0B)',  // fallback amber
];

/** Cycle through the orbit palette by orbit-id (numeric or string-hashed). */
export function orbitColor(orbitId) {
  if (orbitId == null) return TOKEN.gray300;
  let n;
  if (typeof orbitId === 'number') {
    n = orbitId;
  } else {
    let h = 0;
    const s = String(orbitId);
    for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) | 0;
    n = Math.abs(h);
  }
  return ORBIT_PALETTE[n % ORBIT_PALETTE.length];
}

/* ─────────────────────────────────────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────────────────────────────────────── */

/**
 * Generate every assignment as an object keyed by label name.
 * Labels are taken in the supplied order and each ranges over [0, n).
 *   labels=['i','j','k'], n=2 → [{i:0,j:0,k:0}, {i:0,j:0,k:1}, ...]
 */
function enumerateAssignments(labels, n) {
  if (!labels || labels.length === 0) return [];
  const out = [];
  const total = n ** labels.length;
  for (let idx = 0; idx < total; idx += 1) {
    const a = {};
    let r = idx;
    for (let li = labels.length - 1; li >= 0; li -= 1) {
      a[labels[li]] = r % n;
      r = Math.floor(r / n);
    }
    out.push(a);
  }
  return out;
}

/**
 * Render the product expression for an assignment.
 *   subscripts=['ij','k'], operandNames=['A','B'], a={i:0,j:1,k:0}
 *     → "A[0,1] · B[0]"
 */
function productExpr(subscripts, operandNames, assignment) {
  return subscripts
    .map((sub, i) => {
      const name = operandNames[i] ?? `T${i}`;
      const idxs = sub.split('').map((lbl) => assignment[lbl]).join(',');
      return `${name}[${idxs}]`;
    })
    .join(' · ');
}

/**
 * Render the output destination for an assignment.
 *   output='ik', a={i:0,j:1,k:0} → "R[0,0]"
 */
function outputExpr(output, assignment, outputName = 'R') {
  if (!output) return `${outputName}[]`;
  const idxs = output.split('').map((lbl) => assignment[lbl]).join(',');
  return `${outputName}[${idxs}]`;
}

/**
 * Render the assignment as `(i=0, j=1, k=0)`.
 */
function assignmentTuple(labels, assignment) {
  return `(${labels.map((l) => `${l}=${assignment[l]}`).join(', ')})`;
}

/**
 * Stable key for an assignment tuple under a given label order, e.g.
 *   labels=['i','j','k'], a={i:0,j:1,k:2} → "0,1,2".
 * Mirrors the Map key shape `orbitsByAssignment` is expected to use.
 */
export function assignmentTupleKey(labels, assignment) {
  return labels.map((l) => assignment[l]).join(',');
}

/* ─────────────────────────────────────────────────────────────────────────────
   Component
   ───────────────────────────────────────────────────────────────────────────── */

function DenseAssignmentGrid({
  /** Effective dimension n; each label ranges over [0, n). */
  dimensionN,
  /** All label names in the einsum (V ∪ W), in deterministic order. */
  allLabels = [],
  /** Per-operand subscript strings, e.g. ['ij','k']. */
  subscripts = [],
  /** Operand display names in matching order, e.g. ['A','B']. */
  operandNames = [],
  /** Output subscript string, e.g. 'ik'. */
  output = '',
  /**
   * V3.1 §8 — C08 Product-Orbit Lens
   * Map from assignment-tuple-key (see `assignmentTupleKey`) to orbit-id
   * (string or number). When provided, cells expose `data-orbit-id` and
   * the "Show all orbits" toggle paints subtle orbit outlines.
   */
  orbitsByAssignment = null,
  /**
   * Fired when a cell is clicked AND has an orbit id, to lock the orbit
   * selection in the parent (which renders the side-card).
   */
  onOrbitSelect,
}) {
  const n = Number.isFinite(dimensionN) ? dimensionN : 2;
  const labels = Array.isArray(allLabels) ? allLabels : [];

  const [showProducts, setShowProducts] = useState(false);
  const [showOutputs, setShowOutputs] = useState(false);
  const [showAllOrbits, setShowAllOrbits] = useState(false);
  const [hoveredOrbitId, setHoveredOrbitId] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(0);

  const hasOrbits = orbitsByAssignment != null
    && typeof orbitsByAssignment.get === 'function';

  // Resolve the orbit-id for an assignment object.
  const orbitIdFor = useCallback((assignment) => {
    if (!hasOrbits) return null;
    return orbitsByAssignment.get(assignmentTupleKey(labels, assignment)) ?? null;
  }, [hasOrbits, orbitsByAssignment, labels]);

  // Pick the facet label (k for triples, the last label otherwise).
  // First two labels span the (rows, cols) of each panel; remaining labels
  // facet the panels.
  const rowLabel = labels[0] ?? null;
  const colLabel = labels[1] ?? null;
  const facetLabels = labels.slice(2); // 0..many facets

  const tooLarge = n > RENDER_CAP;

  // Enumerate all assignments once.
  const assignments = useMemo(() => {
    if (tooLarge || labels.length === 0) return [];
    return enumerateAssignments(labels, n);
  }, [tooLarge, labels, n]);

  // Group by facet-label values.
  const facetPanels = useMemo(() => {
    if (tooLarge || labels.length === 0) return [];
    if (facetLabels.length === 0) {
      return [{ key: '__single__', label: null, items: assignments.map((a, idx) => ({ a, idx })) }];
    }
    const groups = new Map();
    for (let idx = 0; idx < assignments.length; idx += 1) {
      const a = assignments[idx];
      const key = facetLabels.map((l) => `${l}=${a[l]}`).join(',');
      if (!groups.has(key)) groups.set(key, { key, label: key, items: [] });
      groups.get(key).items.push({ a, idx });
    }
    return Array.from(groups.values());
  }, [tooLarge, assignments, facetLabels, labels.length]);

  // Selected detail.
  const selected = assignments[selectedIdx] ?? null;

  const handleSelect = useCallback((idx) => {
    setSelectedIdx(idx);
    if (typeof onOrbitSelect === 'function') {
      const a = assignments[idx];
      const id = a ? orbitIdFor(a) : null;
      if (id != null) onOrbitSelect(id);
    }
  }, [assignments, orbitIdFor, onOrbitSelect]);

  const handleKey = useCallback((e, idx) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleSelect(idx);
    }
  }, [handleSelect]);

  const handleHoverOrbit = useCallback((id) => {
    setHoveredOrbitId(id);
  }, []);

  // ─── Cap message ───────────────────────────────────────────────────────────
  if (tooLarge) {
    return (
      <div
        data-testid="dense-assignment-grid"
        data-cap-state="too-large"
        className="rounded-md border px-4 py-6 font-mono text-[12px]"
        style={{
          borderColor: TOKEN.gray200,
          background: TOKEN.gray50,
          color: TOKEN.gray600,
        }}
      >
        Grid hidden — n={n} is too large for visual rendering. Reduce n to see the dense grid.
      </div>
    );
  }

  // ─── Empty state — no labels (e.g. before analysis) ────────────────────────
  if (labels.length === 0 || !rowLabel) {
    return (
      <div
        data-testid="dense-assignment-grid"
        data-cap-state="empty"
        className="rounded-md border px-4 py-6 font-mono text-[12px]"
        style={{
          borderColor: TOKEN.gray200,
          background: TOKEN.gray50,
          color: TOKEN.gray500,
        }}
      >
        no assignments yet — pick a preset or analyze a custom expression
      </div>
    );
  }

  return (
    <div
      data-testid="dense-assignment-grid"
      data-cap-state="ok"
      className="w-full"
    >
      {/* Header row: title + toggles */}
      <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <div
          className="font-mono text-[12px] tracking-[0.04em]"
          style={{ color: TOKEN.gray700 }}
        >
          dense assignment grid
          <span className="ml-2" style={{ color: TOKEN.gray500 }}>
            n = <strong style={{ color: TOKEN.gray700 }}>{n}</strong>
            <span className="ml-1">{`· {0..${n - 1}}`}</span>
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3 font-mono text-[11px]">
          <label className="flex cursor-pointer items-center gap-1.5" style={{ color: TOKEN.gray600 }}>
            <input
              type="checkbox"
              checked={showProducts}
              onChange={(e) => setShowProducts(e.target.checked)}
              data-testid="dense-assignment-grid-toggle-products"
              className="h-3 w-3"
            />
            <span>show products</span>
          </label>
          <label className="flex cursor-pointer items-center gap-1.5" style={{ color: TOKEN.gray600 }}>
            <input
              type="checkbox"
              checked={showOutputs}
              onChange={(e) => setShowOutputs(e.target.checked)}
              data-testid="dense-assignment-grid-toggle-outputs"
              className="h-3 w-3"
            />
            <span>show outputs</span>
          </label>
          {hasOrbits && (
            <label
              className="flex cursor-pointer items-center gap-1.5"
              style={{ color: TOKEN.gray600 }}
              aria-label="Show all orbits"
            >
              <input
                type="checkbox"
                checked={showAllOrbits}
                onChange={(e) => setShowAllOrbits(e.target.checked)}
                data-testid="dense-assignment-grid-toggle-all-orbits"
                aria-label="Show all orbits"
                className="h-3 w-3"
              />
              <span>show all orbits</span>
            </label>
          )}
        </div>
      </div>

      {/* Faceted grid: one panel per facet-label value */}
      <div
        data-testid="dense-assignment-grid-panels"
        className="flex flex-wrap gap-4"
      >
        {facetPanels.map((panel) => (
          <div
            key={panel.key}
            data-testid="dense-assignment-grid-panel"
            data-facet-key={panel.key}
            className="flex flex-col gap-1.5"
          >
            {panel.label && (
              <div
                className="font-mono text-[10px] tracking-[0.04em]"
                style={{ color: TOKEN.gray500 }}
              >
                facet: {panel.label}
              </div>
            )}
            {/* Column headers (col-label values) — only rendered for the first row */}
            <div
              className="grid gap-1"
              style={{
                gridTemplateColumns: colLabel !== null
                  ? `auto repeat(${n}, minmax(56px, auto))`
                  : 'auto minmax(56px, auto)',
              }}
            >
              {/* Top-left empty cell */}
              <div />
              {/* col headers */}
              {colLabel !== null
                ? Array.from({ length: n }, (_, c) => (
                    <div
                      key={`col-${c}`}
                      className="text-center font-mono text-[10px]"
                      style={{ color: TOKEN.gray500 }}
                    >
                      {colLabel}={c}
                    </div>
                  ))
                : (
                  <div className="text-center font-mono text-[10px]" style={{ color: TOKEN.gray500 }}>
                    {/* singleton: no col-axis */}
                  </div>
                )
              }
              {/* rows */}
              {Array.from({ length: n }, (_, r) => {
                const cellsThisRow = colLabel !== null
                  ? Array.from({ length: n }, (_, c) =>
                      panel.items.find(({ a }) => a[rowLabel] === r && a[colLabel] === c) ?? null,
                    )
                  : [panel.items.find(({ a }) => a[rowLabel] === r) ?? null];

                return (
                  <ROWFragment
                    key={`row-${r}`}
                    rowIdx={r}
                    rowLabel={rowLabel}
                    cells={cellsThisRow}
                    selectedIdx={selectedIdx}
                    showProducts={showProducts}
                    showOutputs={showOutputs}
                    showAllOrbits={showAllOrbits && hasOrbits}
                    subscripts={subscripts}
                    operandNames={operandNames}
                    output={output}
                    onSelect={handleSelect}
                    onKey={handleKey}
                    onHoverOrbit={handleHoverOrbit}
                    hoveredOrbitId={hoveredOrbitId}
                    orbitIdFor={orbitIdFor}
                    labels={labels}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Selected detail block — V3.1 §6 verbatim labels */}
      <div
        data-testid="dense-assignment-grid-detail"
        className="mt-4 rounded-md border px-3 py-2 font-mono text-[12px] leading-6"
        style={{
          borderColor: TOKEN.gray200,
          background: TOKEN.gray50,
          color: TOKEN.gray700,
        }}
      >
        {selected ? (
          <>
            <div>
              <span style={{ color: TOKEN.gray500 }}>full assignment: </span>
              <span style={{ color: TOKEN.gray900 }}>{assignmentTuple(labels, selected)}</span>
            </div>
            <div>
              <span style={{ color: TOKEN.gray500 }}>product: </span>
              <span style={{ color: TOKEN.gray900 }}>
                {productExpr(subscripts, operandNames, selected)}
              </span>
            </div>
            <div>
              <span style={{ color: TOKEN.gray500 }}>output: </span>
              <span style={{ color: TOKEN.gray900 }}>{outputExpr(output, selected)}</span>
            </div>
          </>
        ) : (
          <span style={{ color: TOKEN.gray500 }}>select a cell to see the assignment</span>
        )}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Row sub-component — kept inline so the grid file stays self-contained.
   ───────────────────────────────────────────────────────────────────────────── */
function ROWFragment({
  rowIdx,
  rowLabel,
  cells,
  selectedIdx,
  showProducts,
  showOutputs,
  showAllOrbits,
  subscripts,
  operandNames,
  output,
  onSelect,
  onKey,
  onHoverOrbit,
  hoveredOrbitId,
  orbitIdFor,
  labels,
}) {
  return (
    <>
      <div
        className="flex items-center justify-end pr-1 font-mono text-[10px]"
        style={{ color: TOKEN.gray500 }}
      >
        {rowLabel}={rowIdx}
      </div>
      {cells.map((entry, c) => {
        if (!entry) {
          return <div key={`empty-${c}`} aria-hidden="true" />;
        }
        const { a, idx } = entry;
        const isSelected = idx === selectedIdx;
        const tupleLabel = assignmentTuple(labels, a);
        const productLabel = productExpr(subscripts, operandNames, a);
        const outputLabel = outputExpr(output, a);
        const ariaLabel = `assignment ${tupleLabel} — product ${productLabel}; output ${outputLabel}`;
        // V3.1 §8 — orbit-overlay decoration. Per cell:
        //   - data-orbit-id attribute (when an orbit-id is known)
        //   - subtle outline color when "show all orbits" is on, OR when
        //     this cell's orbit is the currently-hovered one (so the entire
        //     orbit lights up).
        const orbitId = typeof orbitIdFor === 'function' ? orbitIdFor(a) : null;
        const isHoveredOrbit = orbitId != null && hoveredOrbitId === orbitId;
        const showOrbitOutline = orbitId != null && (showAllOrbits || isHoveredOrbit);
        const orbitBorderColor = showOrbitOutline ? orbitColor(orbitId) : null;
        const borderColor = isSelected
          ? TOKEN.coral
          : (orbitBorderColor ?? TOKEN.gray200);
        return (
          <div
            key={`cell-${idx}`}
            role="button"
            tabIndex={0}
            aria-label={ariaLabel}
            aria-pressed={isSelected}
            data-testid="dense-assignment-grid-cell"
            data-assignment-idx={idx}
            data-selected={isSelected ? 'true' : 'false'}
            data-orbit-id={orbitId != null ? String(orbitId) : undefined}
            data-orbit-hovered={isHoveredOrbit ? 'true' : undefined}
            onClick={() => onSelect(idx)}
            onKeyDown={(e) => onKey(e, idx)}
            onMouseEnter={() => onHoverOrbit?.(orbitId)}
            onMouseLeave={() => onHoverOrbit?.(null)}
            onFocus={() => onHoverOrbit?.(orbitId)}
            onBlur={() => onHoverOrbit?.(null)}
            className="cursor-pointer rounded border px-1.5 py-1 text-center font-mono text-[10px] leading-tight transition-colors"
            style={{
              borderColor,
              borderWidth: isHoveredOrbit && !isSelected ? 2 : 1,
              background: isSelected ? TOKEN.coralLight : TOKEN.white,
              color: isSelected ? TOKEN.gray900 : TOKEN.gray600,
              outline: 'none',
            }}
          >
            {/* default content: tuple of the FIRST two label values (the local
                row,col within the panel). When toggles are on, replace with
                product / output expressions. */}
            {showProducts && (
              <div style={{ color: TOKEN.gray700 }}>{productLabel}</div>
            )}
            {showOutputs && (
              <div style={{ color: TOKEN.einV }}>{outputLabel}</div>
            )}
            {!showProducts && !showOutputs && (
              <div>
                {labels.map((l) => `${a[l]}`).join(',')}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

export default DenseAssignmentGrid;
