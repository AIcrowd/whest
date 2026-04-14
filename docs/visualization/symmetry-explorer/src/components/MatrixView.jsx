import IncidenceMatrix from './IncidenceMatrix.jsx';

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
        <IncidenceMatrix
          matrix={matrix}
          colLabels={labels}
          uVertices={uVertices}
          example={example}
          freeLabels={freeLabels}
          label="M"
        />
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
          <p className="fp-note">Labels sharing a fingerprint are structurally equivalent — the σ-loop uses these to derive π.</p>
        ) : (
          <p className="fp-note">All fingerprints are distinct — no equivalences among labels.</p>
        )}
      </div>
    </div>
  );
}
