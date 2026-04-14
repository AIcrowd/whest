export default function MatrixView({ matrixData, graph, example, variableColors }) {
  const { matrix, labels, colFingerprints, fpToLabels } = matrixData;
  const { uVertices, freeLabels } = graph;

  // Group fingerprints by value to highlight equivalent columns
  const fpColors = {};
  let colorIdx = 0;
  const palette = ['#4a7cff', '#3ddc84', '#ffb74d', '#bb86fc', '#ff5252'];
  for (const [fp, lblSet] of Object.entries(fpToLabels)) {
    if (lblSet.size >= 2) {
      fpColors[fp] = palette[colorIdx++ % palette.length];
    }
  }

  return (
    <div className="matrix-view">
      <div className="matrix-wrapper">
        <table className="matrix-table">
          <thead>
            <tr>
              <th></th>
              {labels.map(lbl => (
                <th key={lbl} className={freeLabels.has(lbl) ? 'col-v' : 'col-w'}>
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => {
              const u = uVertices[rIdx];
              const lblStr = [...u.labels].sort().join(',');
              return (
                <tr key={rIdx}>
                  <td className="row-label">
                    {(() => {
                      const opName = example.operandNames?.[u.opIdx] || `Op${u.opIdx}`;
                      const vc = variableColors?.[opName];
                      return (
                        <>
                          <span className="op-tag" style={vc ? { color: vc.color, borderColor: `${vc.color}33` } : {}}>
                            {opName}
                          </span>
                          ·{lblStr}
                        </>
                      );
                    })()}
                  </td>
                  {row.map((val, cIdx) => (
                    <td key={cIdx}
                      className={`matrix-cell ${val > 0 ? 'cell-active' : ''}`}>
                      {val}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="fingerprints">
        <h4>Column Fingerprints</h4>
        <div className="fp-grid">
          {labels.map(lbl => {
            const fp = colFingerprints[lbl];
            const eqColor = fpColors[fp];
            return (
              <div key={lbl} className="fp-item" style={eqColor ? { borderColor: eqColor } : {}}>
                <span className="fp-label">{lbl}</span>
                <code className="fp-value">({fp.replace(/,/g, ', ')})</code>
                {eqColor && <span className="fp-eq" style={{ background: eqColor }}>≡</span>}
              </div>
            );
          })}
        </div>
        {Object.keys(fpColors).length > 0 ? (
          <p className="fp-note">Labels sharing a fingerprint are structurally equivalent → fast-path S<sub>k</sub> detection.</p>
        ) : (
          <p className="fp-note">All fingerprints are distinct — no fast-path equivalences. Detection requires the σ-loop.</p>
        )}
      </div>
    </div>
  );
}
