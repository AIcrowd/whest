function formatTuple(tuple) {
  return Object.entries(tuple ?? {})
    .map(([label, value]) => `${label}=${value}`)
    .join(', ');
}

export default function OrbitInspector({
  orbitRows = [],
  selectedOrbitIdx = -1,
  onSelectOrbit,
  title = 'Explore one representative orbit',
  kicker = 'Orbit Inspector',
  description = 'Each orbit is one evaluation representative. The projected outputs show why reduction cost can stay high even when evaluation cost drops.',
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

  return (
    <div className="orbit-inspector">
      <div className="orbit-inspector-header">
        <span className="orbit-inspector-kicker">{kicker}</span>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className="orbit-inspector-controls">
        <label className="orbit-select-label" htmlFor="orbit-selector">
          Orbit representative
        </label>
        <select
          id="orbit-selector"
          className="orbit-select"
          value={resolvedIdx}
          onChange={(event) => onSelectOrbit?.(Number(event.target.value))}
        >
          {orbitRows.map((row, idx) => (
            <option key={`orbit-option-${idx}`} value={idx}>
              Orbit {idx + 1} · size {row.orbitSize} · outputs {row.outputCount}
            </option>
          ))}
        </select>
      </div>

      <div className="orbit-detail-card orbit-detail-card-standalone">
        <div className="orbit-detail-summary">
          <div>
            <span className="orbit-detail-label">Representative</span>
            <code>({formatTuple(selectedRow.repTuple)})</code>
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
