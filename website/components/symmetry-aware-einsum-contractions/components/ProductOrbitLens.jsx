/**
 * ProductOrbitLens — V3.1 §8 — C08 (NEW)
 *
 * The side-card companion to the DenseAssignmentGrid orbit overlays.
 * When an orbit is locked (via DenseAssignmentGrid's onOrbitSelect), this card
 * shows the V3.1 §8 product-orbit fields:
 *
 *     Representative      (rep tuple)
 *     Members             (every tuple in the orbit)
 *     Orbit size          (|Orbit|)
 *     Reason              ("Product equality" — explains WHY these tuples are
 *                          identified, e.g. AB(i,j,k)=AB(j,i,k) for Cross S2)
 *     Fixed point / Non-fixed orbit label
 *
 * Token-only colours; no raw hex. Pure presentational — receives a single
 * orbit row plus an explanatory `reason` string.
 */

import { orbitColor } from './DenseAssignmentGrid.jsx';

const TOKEN = {
  gray900: 'var(--gray-900)',
  gray700: 'var(--gray-700)',
  gray600: 'var(--gray-600)',
  gray500: 'var(--gray-500)',
  gray300: 'var(--gray-300)',
  gray200: 'var(--gray-200)',
  gray100: 'var(--gray-100)',
  gray50:  'var(--gray-50)',
  white:   'var(--white)',
};

function formatTuple(labels, tuple) {
  if (!tuple) return '—';
  return `(${labels.map((l) => `${l}=${tuple[l]}`).join(', ')})`;
}

/**
 * Build a default "product equality" reason sentence for the locked orbit.
 * Uses the orbit's representative tuple to instantiate one concrete product
 * equality the orbit certifies, e.g.
 *
 *   A[0,1] · B[0]  =  A[1,0] · B[0]
 *
 * The caller can override this with an explicit `reason` prop.
 */
function defaultReason({ orbit, labels, subscripts, operandNames }) {
  if (!orbit || !Array.isArray(orbit.orbitTuples) || orbit.orbitTuples.length === 0) {
    return 'Tuples in this orbit produce equal products under the einsum symmetry.';
  }
  if (orbit.orbitSize <= 1) {
    return 'Fixed point — this tuple is its own orbit under the symmetry group.';
  }
  // Render the rep's product expression and one orbit member's product, to
  // make the equality concrete.
  const renderProduct = (a) =>
    subscripts
      .map((sub, i) => {
        const name = operandNames[i] ?? `T${i}`;
        const idxs = sub.split('').map((lbl) => a[lbl]).join(',');
        return `${name}[${idxs}]`;
      })
      .join(' · ');
  const repP = renderProduct(orbit.repTuple);
  // Pick the first non-rep member.
  const other = orbit.orbitTuples.find(
    (t) => labels.some((l) => t[l] !== orbit.repTuple[l]),
  );
  if (!other) {
    return 'Fixed point — this tuple is its own orbit under the symmetry group.';
  }
  const otherP = renderProduct(other);
  return `Product equality: ${repP} = ${otherP}.`;
}

export default function ProductOrbitLens({
  /** The locked orbit row (shape from costModel.orbitRows). */
  orbit = null,
  /** Numeric or string id used to colour-key this orbit. */
  orbitId = null,
  /** Label order used to format the rep / member tuples. */
  labels = [],
  /** Per-operand subscripts (for the product equality). */
  subscripts = [],
  /** Operand display names (for the product equality). */
  operandNames = [],
  /** Override for the auto-derived reason sentence. */
  reason = null,
}) {
  if (!orbit) {
    return (
      <div
        data-testid="product-orbit-lens"
        data-orbit-state="empty"
        className="rounded-md border px-3 py-3 font-mono text-[12px]"
        style={{
          borderColor: TOKEN.gray200,
          background: TOKEN.gray50,
          color: TOKEN.gray500,
        }}
      >
        Click a cell in the dense grid to lock its product orbit.
      </div>
    );
  }

  const orbitSize = orbit.orbitSize ?? (orbit.orbitTuples?.length ?? 0);
  const isFixedPoint = orbitSize <= 1;
  const swatch = orbitColor(orbitId);
  const rep = formatTuple(labels, orbit.repTuple);
  const why = reason ?? defaultReason({ orbit, labels, subscripts, operandNames });

  return (
    <div
      data-testid="product-orbit-lens"
      data-orbit-state="locked"
      data-orbit-id={orbitId != null ? String(orbitId) : undefined}
      data-fixed-point={isFixedPoint ? 'true' : 'false'}
      className="rounded-md border px-4 py-3 font-mono text-[12px] leading-6"
      style={{
        borderColor: TOKEN.gray200,
        background: TOKEN.white,
        color: TOKEN.gray700,
      }}
    >
      <div className="mb-2 flex items-center gap-2">
        <span
          aria-hidden="true"
          className="inline-block h-2.5 w-2.5 rounded-sm"
          style={{ background: swatch, border: `1px solid ${swatch}` }}
        />
        <span
          className="text-[10px] uppercase tracking-[0.16em]"
          style={{ color: TOKEN.gray500 }}
        >
          Product orbit lens
        </span>
        <span
          data-testid="product-orbit-lens-fixed-label"
          className="ml-auto rounded px-1.5 py-0.5 text-[10px] uppercase tracking-[0.08em]"
          style={{
            background: isFixedPoint ? TOKEN.gray100 : TOKEN.gray50,
            color: TOKEN.gray700,
            border: `1px solid ${TOKEN.gray200}`,
          }}
        >
          {isFixedPoint ? 'Fixed point' : 'Non-fixed orbit'}
        </span>
      </div>

      <dl className="space-y-1">
        <div className="flex items-baseline gap-2">
          <dt className="w-24 shrink-0 text-[10px] uppercase tracking-[0.08em]" style={{ color: TOKEN.gray500 }}>
            Representative
          </dt>
          <dd style={{ color: TOKEN.gray900 }}>{rep}</dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="w-24 shrink-0 text-[10px] uppercase tracking-[0.08em]" style={{ color: TOKEN.gray500 }}>
            Orbit size
          </dt>
          <dd style={{ color: TOKEN.gray900 }}>{orbitSize}</dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="w-24 shrink-0 text-[10px] uppercase tracking-[0.08em]" style={{ color: TOKEN.gray500 }}>
            Members
          </dt>
          <dd
            data-testid="product-orbit-lens-members"
            className="flex flex-wrap gap-1"
            style={{ color: TOKEN.gray900 }}
          >
            {(orbit.orbitTuples ?? []).map((tuple, i) => (
              <code
                key={`pol-member-${i}`}
                className="rounded border px-1.5 py-0.5"
                style={{
                  borderColor: TOKEN.gray200,
                  background: TOKEN.gray50,
                  color: TOKEN.gray700,
                }}
              >
                {formatTuple(labels, tuple)}
              </code>
            ))}
          </dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="w-24 shrink-0 text-[10px] uppercase tracking-[0.08em]" style={{ color: TOKEN.gray500 }}>
            Reason
          </dt>
          <dd style={{ color: TOKEN.gray900 }}>{why}</dd>
        </div>
      </dl>
    </div>
  );
}
