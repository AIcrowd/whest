import { useState } from 'react';

export default function SigmaLoop({ results, graph, matrixData, example }) {
  const [expandedIdx, setExpandedIdx] = useState(null);
  const { allLabels, uVertices } = graph;
  const labels = matrixData.labels;

  const nonTrivial = results.filter(r => !r.skipped);
  const accepted = nonTrivial.filter(r => r.isValid);
  const rejected = nonTrivial.filter(r => !r.isValid);

  function sigmaStr(sigma) {
    const entries = Object.entries(sigma).filter(([k, v]) => Number(k) !== v);
    if (entries.length === 0) return 'identity';
    // Build cycle notation for sigma on operand indices
    const visited = new Set();
    const cycles = [];
    for (const [k] of entries) {
      const kn = Number(k);
      if (visited.has(kn)) continue;
      const cycle = [];
      let cur = kn;
      while (!visited.has(cur)) {
        visited.add(cur);
        cycle.push(cur);
        cur = sigma[cur] ?? cur;
      }
      if (cycle.length > 1) cycles.push(cycle);
    }
    return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'identity';
  }

  function piStr(pi) {
    if (!pi) return '—';
    const entries = Object.entries(pi).filter(([k, v]) => k !== v);
    if (entries.length === 0) return 'identity';
    const visited = new Set();
    const cycles = [];
    for (const [k] of entries) {
      if (visited.has(k)) continue;
      const cycle = [];
      let cur = k;
      while (!visited.has(cur)) {
        visited.add(cur);
        cycle.push(cur);
        cur = pi[cur];
      }
      if (cycle.length > 1) cycles.push(cycle);
    }
    return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'identity';
  }

  return (
    <div className="sigma-loop">
      <div className="sigma-summary">
        <div className="sigma-stat">
          <span className="stat-num">{results.length}</span>
          <span className="stat-label">total σ's</span>
        </div>
        <div className="sigma-stat">
          <span className="stat-num">{results.filter(r => r.skipped).length}</span>
          <span className="stat-label">identity (skipped)</span>
        </div>
        <div className="sigma-stat accepted">
          <span className="stat-num">{accepted.length}</span>
          <span className="stat-label">valid π found</span>
        </div>
        <div className="sigma-stat rejected">
          <span className="stat-num">{rejected.length}</span>
          <span className="stat-label">rejected</span>
        </div>
      </div>

      <div className="sigma-list">
        {nonTrivial.map((r, i) => {
          const expanded = expandedIdx === i;
          return (
            <div key={i}
              className={`sigma-card ${r.isValid ? 'valid' : 'invalid'} ${expanded ? 'expanded' : ''}`}
              onClick={() => setExpandedIdx(expanded ? null : i)}
            >
              <div className="sigma-card-header">
                <code className="sigma-label">σ = {sigmaStr(r.sigma)}</code>
                {r.isValid ? (
                  <span className="sigma-result valid">
                    ✓ π = {piStr(r.pi)}
                  </span>
                ) : (
                  <span className="sigma-result invalid">✗ {r.reason}</span>
                )}
              </div>

              {expanded && r.sigmaMatrix && (
                <div className="sigma-detail">
                  <div className="sigma-matrices">
                    <div className="mini-matrix-block">
                      <h5>σ(M)</h5>
                      <table className="mini-matrix">
                        <thead>
                          <tr>
                            <th></th>
                            {labels.map(l => <th key={l}>{l}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {r.sigmaMatrix.map((row, ri) => (
                            <tr key={ri}>
                              <td className="row-label mini">
                                Op{uVertices[r.sigmaRowPerm[ri]].opIdx}
                              </td>
                              {row.map((v, ci) => (
                                <td key={ci} className={`matrix-cell mini ${v > 0 ? 'cell-active' : ''}`}>
                                  {v}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {r.isValid && r.pi && (
                      <div className="pi-mapping">
                        <h5>π mapping</h5>
                        <div className="pi-arrows">
                          {Object.entries(r.pi).map(([from, to]) => (
                            <div key={from} className={`pi-arrow ${from !== to ? 'moved' : 'fixed'}`}>
                              <span>{from}</span>
                              <span className="arrow">→</span>
                              <span>{to}</span>
                            </div>
                          ))}
                        </div>
                        <div className="pi-check">π(σ(M)) = M ✓</div>
                      </div>
                    )}

                    {!r.isValid && r.sigmaColOf && (
                      <div className="fp-mismatch">
                        <h5>Fingerprint comparison</h5>
                        {labels.map(lbl => {
                          const origFp = matrixData.colFingerprints[lbl];
                          const sigmaFp = r.sigmaColOf[lbl];
                          const matches = origFp === sigmaFp ||
                            Object.values(matrixData.colFingerprints).includes(sigmaFp);
                          return (
                            <div key={lbl} className={`fp-compare ${matches ? '' : 'no-match'}`}>
                              <span>{lbl}:</span>
                              <code>({sigmaFp.replace(/,/g,', ')})</code>
                              {!matches && <span className="no-match-tag">no match</span>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
