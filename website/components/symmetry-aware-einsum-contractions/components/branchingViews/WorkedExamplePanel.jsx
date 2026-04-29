import Latex from '../Latex.jsx';
import { labelledTuple, tupleKey } from './orbitRepMatrixLayout.js';

// Token palette anchored to design-system colors_and_type.css.
const COLOR = {
  empty: '#F8F9F9',
  border: '#D9DCDC',
  coral: '#F0524D',
  coralMuted: 'rgba(240,82,77,0.85)',
  coralLight: '#FEF2F1',
  divider: '#ECEFEF',
};

// Find members of an orbit whose visible-side projection matches repTuple.
// Without H-canonicalization plumbed through (Task 9 work), this is a simple
// exact-match on V labels — adequate for live presets where H is trivial.
function membersProjectingTo(orbit, repTuple, componentInfo) {
  if (!orbit?.orbitTuples || !componentInfo) return [];
  const { vLabels } = componentInfo;
  if (!vLabels) return [];
  return orbit.orbitTuples.filter((m) => vLabels.every((l) => m[l] === repTuple?.[l]));
}

// Canonical-form LaTeX for the einsum equation.
function canonicalEquationLatex(expressionInfo) {
  if (!expressionInfo) return '';
  const { subscripts = [], output = '', operandNames = [] } = expressionInfo;
  if (!subscripts.length || !output) return '';
  const inOps = new Set();
  subscripts.forEach((s) => [...s].forEach((c) => inOps.add(c)));
  const summed = [...inOps].filter((c) => !output.includes(c)).sort();
  const opTerms = subscripts.map((s, i) => {
    const name = operandNames[i] ?? operandNames[0] ?? 'T';
    return `${name}[${[...s].join(',')}]`;
  });
  const sumPrefix = summed.length ? `\\sum_{${summed.join(',')}}\\,` : '';
  return `R[${[...output].join(',')}] \\;=\\; ${sumPrefix}${opTerms.join(' \\cdot ')}`;
}

export default function WorkedExamplePanel({
  hover, pin,
  orbitRows = [], reps = [], cells = [],
  expressionInfo = null, componentInfo = null,
  onClearPin = () => {},
}) {
  const focus = pin || hover;

  // Empty state.
  if (!focus) {
    return (
      <div
        data-testid="worked-example-panel"
        className="rounded-lg border p-4 text-center text-[12px] italic"
        style={{ background: COLOR.empty, borderColor: COLOR.border, color: '#9AA0A0' }}
      >
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-400 not-italic mb-2">
          Worked example
        </div>
        Hover any cell on the left to see the (O, Q) projection and contribution.
      </div>
    );
  }

  const row = orbitRows[focus.row];
  const rep = reps[focus.col];
  const coeff = cells[focus.row]?.[focus.col] ?? null;
  const filled = coeff !== null;

  if (!row || !rep) {
    // Defensive — shouldn't happen but guards against stale state.
    return (
      <div
        data-testid="worked-example-panel"
        className="rounded-lg border p-4 text-center text-[12px] italic"
        style={{ background: COLOR.empty, borderColor: COLOR.border, color: '#9AA0A0' }}
      >
        Worked example unavailable
      </div>
    );
  }

  const allLabels = expressionInfo?.subscripts
    ? [...new Set(expressionInfo.subscripts.flatMap((s) => [...s]))]
    : Object.keys(row.repTuple ?? {});
  const vLabels = componentInfo?.vLabels ?? Object.keys(rep.tuple ?? {});

  const contributing = membersProjectingTo(row, rep.tuple, componentInfo);
  const otherReached = (row.outputs ?? [])
    .filter((o) => tupleKey(o.outTuple) !== tupleKey(rep.tuple))
    .map((o) => ({ outTuple: o.outTuple, members: membersProjectingTo(row, o.outTuple, componentInfo) }));
  const branchCount = row.outputs?.length ?? 1;
  const operandName = expressionInfo?.operandNames?.[0] ?? 'T';

  return (
    <div
      data-testid="worked-example-panel"
      className="rounded-lg border bg-white p-4"
      style={{ borderColor: COLOR.border }}
    >
      {/* Eyebrow + clear-pin button */}
      <div className="flex items-baseline gap-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: COLOR.coral }}>
          Worked example · O → Q
        </div>
        {pin && (
          <button
            type="button"
            data-action="clear-pin"
            onClick={onClearPin}
            className="ml-auto text-[10px] font-medium text-gray-600 hover:text-gray-900 transition-colors"
          >
            × clear pin
          </button>
        )}
      </div>

      {/* Tuples header */}
      <div className="mt-2 font-mono text-[11px] text-gray-900 leading-7">
        <div>
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mr-2 font-sans">orbit O</span>
          <strong>{labelledTuple(row.repTuple)}</strong>
          <span className="ml-2 text-gray-500 font-sans">
            · {row.orbitSize ?? '?'} member{row.orbitSize === 1 ? '' : 's'}
            {branchCount > 1 ? ` · branches to ${branchCount} reps` : ''}
          </span>
        </div>
        <div>
          <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mr-2 font-sans">stored rep Q</span>
          <strong>{labelledTuple(rep.tuple)}</strong>
        </div>
      </div>

      {/* Mini row preview */}
      <div data-testid="worked-example-row-preview" className="mt-3 rounded p-2.5" style={{ background: COLOR.empty }}>
        <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mb-2">
          orbit O's row in the O → Q matrix
        </div>
        <div className="flex gap-[2px] h-5">
          {reps.map((r, c) => {
            const coef = cells[focus.row][c];
            const isThis = c === focus.col;
            const isOther = !isThis && coef !== null;
            return (
              <div
                key={c}
                className="flex-1 rounded-sm"
                style={{
                  background: isThis ? COLOR.coral : (isOther ? COLOR.coralLight : '#FFFFFF'),
                  border: isOther ? '1px solid rgba(240,82,77,0.45)' : '1px solid #D9DCDC',
                }}
              />
            );
          })}
        </div>
      </div>

      {/* π_V projection sketch */}
      {(row.orbitTuples?.length ?? 0) > 0 && (
        <div data-testid="worked-example-projection" className="mt-3 rounded p-2.5 font-mono text-[10.5px] leading-7" style={{ background: COLOR.empty }}>
          <div className="grid grid-cols-[1fr_auto_1fr] gap-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 font-sans mb-1">
            <span>members of O</span>
            <span>π<sub>V</sub></span>
            <span>destination</span>
          </div>
          {row.orbitTuples.map((m, i) => {
            const memberV = {};
            for (const l of vLabels) memberV[l] = m[l];
            const matchesThis = vLabels.every((l) => m[l] === rep.tuple[l]);
            return (
              <div key={i} className="grid grid-cols-[1fr_auto_1fr] gap-2 items-center">
                <span>{labelledTuple(m)}</span>
                <span className="text-gray-400 text-center">→</span>
                <span style={{ color: matchesThis ? COLOR.coral : COLOR.coralMuted, fontWeight: matchesThis ? 600 : 500 }}>
                  R[{vLabels.map((l) => memberV[l]).join(', ')}] {matchesThis ? '← THIS Q ●' : '← OTHER Q ●'}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Einsum equation */}
      {expressionInfo && canonicalEquationLatex(expressionInfo) && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-gray-400 mb-1.5">einsum equation</div>
          <div className="pl-2">
            <Latex math={canonicalEquationLatex(expressionInfo)} display />
          </div>
        </div>
      )}

      {/* Ledger: this Q's contributions */}
      {filled && expressionInfo && contributing.length > 0 && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] mb-1.5" style={{ color: COLOR.coral }}>
            contribution to R[{vLabels.map((l) => rep.tuple[l]).join(', ')}] · this Q
          </div>
          <div className="pl-2 font-mono text-[11px] leading-7 text-gray-900">
            {contributing.map((m, i) => (
              <div key={i}>
                If ({allLabels.join(', ')}) = ({allLabels.map((l) => m[l]).join(', ')}), then R[{vLabels.map((l) => m[l]).join(', ')}] += {operandName}[{allLabels.map((l) => m[l]).join(', ')}]
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ledger: other Qs the orbit reaches (dimmed) */}
      {otherReached.length > 0 && expressionInfo && (
        <div className="mt-3 pt-2 border-t" style={{ borderColor: COLOR.divider }}>
          <div className="italic text-[10px] font-semibold tracking-[0.16em] uppercase mb-1.5" style={{ color: COLOR.coralMuted }}>
            contribution to other Qs · the orbit's other reached bins
          </div>
          {otherReached.map((entry, ei) => (
            <div key={ei} className="pl-2 font-mono text-[11px] leading-7 text-gray-700 mt-2 first:mt-0">
              <div className="text-[10px] uppercase tracking-[0.16em] text-gray-400 mb-1 font-sans">
                R[{vLabels.map((l) => entry.outTuple[l]).join(', ')}]
              </div>
              {entry.members.map((m, i) => (
                <div key={i}>
                  If ({allLabels.join(', ')}) = ({allLabels.map((l) => m[l]).join(', ')}), then R[{vLabels.map((l) => m[l]).join(', ')}] += {operandName}[{allLabels.map((l) => m[l]).join(', ')}]
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* Branching note */}
      <div className="mt-3 pt-2 border-t font-serif text-[12.5px] italic leading-7 text-gray-700" style={{ borderColor: COLOR.divider }}>
        {branchCount > 1 ? (
          <>
            <strong className="not-italic text-gray-900">Branching:</strong> this single product orbit O fills <strong className="not-italic text-gray-900">{branchCount} cells in its row</strong>. α for this orbit alone is {branchCount} — one update per filled cell, regardless of member count.
          </>
        ) : filled ? (
          <>
            <strong className="not-italic text-gray-900">{contributing.length}</strong> of {row.orbitSize} member{row.orbitSize === 1 ? '' : 's'} of this orbit project{contributing.length === 1 ? 's' : ''} to this output bin. Each filled cell adds 1 to α.
          </>
        ) : (
          <>No member of this orbit projects to this Q. The cell stays empty and contributes nothing to α.</>
        )}
      </div>
    </div>
  );
}
