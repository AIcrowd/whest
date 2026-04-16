import Latex from './Latex.jsx';
import RoleBadge from './RoleBadge.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';

function formatTuple(tuple) {
  return Object.entries(tuple ?? {})
    .map(([label, value]) => `${label}=${value}`)
    .join(', ');
}

function formatRepresentative(tuple) {
  return `(${formatTuple(tuple)})`;
}

export default function OrbitInspector({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit,
  title = 'Explore one representative orbit',
  kicker = 'Orbit Inspector',
  description = 'Each orbit is one evaluation representative. The projected outputs show why reduction cost can stay high even when evaluation cost drops.',
  showHeader = true,
  formulaMath = null,
  dimensionN = null,
  componentContext = null,
}) {
  if (!Array.isArray(orbitRows) || orbitRows.length === 0) {
    return (
      <div className="orbit-inspector orbit-inspector-empty">
        <div className="orbit-inspector-header">
          <span className="orbit-inspector-kicker">{kicker}</span>
          <h3>Representative orbits will appear here</h3>
          <p>Once the engine has orbit rows, this panel shows one representative, its full orbit, and the output bins it updates.</p>
        </div>
      </div>
    );
  }

  const selectedRow = orbitRows[selectedOrbitIdx] ?? orbitRows[0];
  const resolvedIdx = orbitRows[selectedOrbitIdx] ? selectedOrbitIdx : 0;
  const totalOrbitTuples = orbitRows.reduce((sum, row) => sum + (row.orbitSize ?? 0), 0);
  const totalProjectedOutputs = orbitRows.reduce((sum, row) => sum + (row.outputCount ?? 0), 0);
  const selectedRepresentative = formatRepresentative(selectedRow.repTuple);
  const orderedLabels = componentContext?.labels ?? [];
  const vLabels = new Set(componentContext?.va ?? []);

  return (
    <div className="orbit-inspector">
      {showHeader ? (
        <div className="orbit-inspector-header">
          <span className="orbit-inspector-kicker">{kicker}</span>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      ) : null}

      <div className="rounded-lg border border-border/70 bg-muted/20 px-4 py-3">
        <div className="grid gap-3 sm:grid-cols-4">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Total orbits</div>
            <div className="font-mono text-base font-bold text-foreground">{orbitRows.length}</div>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Orbit tuples examined</div>
            <div className="font-mono text-base font-bold text-foreground">{totalOrbitTuples}</div>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Projected outputs</div>
            <div className="font-mono text-base font-bold text-foreground">{totalProjectedOutputs}</div>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">N</div>
            <div className="font-mono text-base font-bold text-foreground">{dimensionN ?? '—'}</div>
          </div>
        </div>
        {componentContext || formulaMath ? (
          <div className="mt-3 border-t border-border/70 pt-3">
            <div className="flex flex-wrap items-start gap-4 lg:flex-nowrap">
              {componentContext ? (
                <div className="min-w-0 flex-1">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Labels</div>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5">
                    {orderedLabels.map((label) => (
                      <RoleBadge key={`orbit-context-${label}`} role={vLabels.has(label) ? 'v' : 'w'}>
                        {label}
                      </RoleBadge>
                    ))}
                  </div>
                </div>
              ) : null}
              {componentContext ? (
                <div className="min-w-[9rem]">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Symmetry</div>
                  <div className="mt-1">
                    <SymmetryBadge value={componentContext.groupName || 'trivial'} />
                  </div>
                </div>
              ) : null}
              {formulaMath ? (
                <div className="min-w-0 flex-[1.3]">
                  <div className="mt-5 text-sm text-foreground lg:mt-0">
                    <Latex math={formulaMath} />
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      <div className="orbit-inspector-controls space-y-2">
        <label className="orbit-select-label" htmlFor="orbit-selector">
          Choose an orbit representative
        </label>
        <select
          id="orbit-selector"
          className="orbit-select"
          value={resolvedIdx}
          onChange={(event) => onSelectOrbit?.(Number(event.target.value))}
        >
          {orbitRows.map((row, idx) => (
            <option key={`orbit-option-${idx}`} value={idx}>
              {`${formatRepresentative(row.repTuple)} · size ${row.orbitSize} · outputs ${row.outputCount} · orbit ${idx + 1}`}
            </option>
          ))}
        </select>
      </div>

      <div className="orbit-detail-card orbit-detail-card-standalone">
        <div className="orbit-detail-summary">
          <div>
            <span className="orbit-detail-label">Representative</span>
            <code>{selectedRepresentative}</code>
          </div>
          <div>
            <span className="orbit-detail-label">Orbit size</span>
            <strong>{selectedRow.orbitSize}</strong>
          </div>
          <div>
            <span className="orbit-detail-label">Distinct outputs</span>
            <strong>{selectedRow.outputCount}</strong>
          </div>
          <div>
            <span className="orbit-detail-label">This orbit contributes</span>
            <strong>1 evaluation, {selectedRow.outputCount} reductions</strong>
          </div>
        </div>

        <div className="orbit-detail-columns">
          <div>
            <h4>Orbit tuples</h4>
            <div className="orbit-token-list">
              {(selectedRow.orbitTuples ?? []).map((tuple, idx) => (
                <code key={`orbit-tuple-${idx}`} className="orbit-token">
                  ({formatTuple(tuple)})
                </code>
              ))}
            </div>
          </div>

          <div>
            <h4>Projected outputs</h4>
            <div className="orbit-token-list">
              {(selectedRow.outputs ?? []).map((output, idx) => (
                <div key={`orbit-output-${idx}`} className="orbit-output-row">
                  <code className="orbit-token">({formatTuple(output.outTuple)})</code>
                  <span className="orbit-output-coeff">x{output.coeff}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
