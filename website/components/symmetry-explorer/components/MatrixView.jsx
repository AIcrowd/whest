import IncidenceMatrix from './IncidenceMatrix.jsx';

export default function MatrixView({ matrixData, graph, example, variableColors }) {
  const { matrix, labels, fpToLabels } = matrixData;
  const { uVertices, freeLabels } = graph;

  // Check if any fingerprints are equivalent
  const hasEquivalences = Object.values(fpToLabels).some(s => s.size >= 2);

  return (
    <div className="matrix-view">
      <div className="matrix-wrapper">
        <IncidenceMatrix
          matrix={matrix}
          colLabels={labels}
          uVertices={uVertices}
          example={example}
          freeLabels={freeLabels}
          variableColors={variableColors}
          label="M"
        />
      </div>

      <div className="fingerprints">
        {hasEquivalences ? (
          <p className="fp-note">Labels sharing a fingerprint are structurally equivalent — the σ-loop uses these to derive π.</p>
        ) : (
          <p className="fp-note">All fingerprints are distinct — no equivalences among labels.</p>
        )}
      </div>
    </div>
  );
}
