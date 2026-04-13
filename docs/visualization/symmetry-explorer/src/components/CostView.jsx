import { useState } from 'react';
import Latex from './Latex.jsx';

export default function CostView({ cost, numOperands = 0 }) {
  const {
    denseCost, reducedCost, ratio, speedup,
    uniqueCount, totalCount, groupName, groupOrder,
    wHasSymmetry, wReducedCost, wRatio, wSpeedup,
    wUniqueCount, wTotalCount, wGroupName,
    combinedRatio, combinedReducedCost,
  } = cost;

  const isBinary = numOperands === 2;
  // For binary einsums, W-side always applies — show expanded by default
  const [showW, setShowW] = useState(isBinary);

  const vTrivial = groupOrder <= 1;
  const pctSaved = ((1 - ratio) * 100).toFixed(1);

  // No symmetry at all (neither V nor W)
  if (vTrivial && !wHasSymmetry) {
    return (
      <div className="cost-view">
        <div className="burnside-trivial">
          <div className="trivial-icon">1.0×</div>
          <div className="trivial-text">
            <strong>No cost reduction</strong>
            <span>Dense cost: {denseCost.toLocaleString()} FLOPs — no symmetry to exploit.</span>
          </div>
        </div>
      </div>
    );
  }

  const combinedPctSaved = ((1 - combinedRatio) * 100).toFixed(1);

  return (
    <div className="cost-view">
      {/* V-side reduction */}
      {!vTrivial && (
        <div className="cost-formula-card">
          <h4 style={{ margin: '0 0 8px' }}>
            <span className="pill pill-v">V</span> Output symmetry reduction
          </h4>
          <div className="burnside-compact" style={{ marginBottom: 12 }}>
            <code className="compact-ratio">V:{uniqueCount.toLocaleString()}/{totalCount.toLocaleString()}</code>
          </div>
          <div className="cost-values">
            <div className="cost-item">
              <span className="cost-label">Dense cost</span>
              <span className="cost-val">{denseCost.toLocaleString()}</span>
            </div>
            <div className="cost-item highlight">
              <span className="cost-label">V-reduced cost</span>
              <span className="cost-val">{reducedCost.toLocaleString()}</span>
            </div>
          </div>
          <div className="speedup-card" style={{ marginTop: 12 }}>
            <div className="speedup-number">{speedup.toFixed(1)}×</div>
            <div className="speedup-label">via {groupName}</div>
            <div className="speedup-detail">{pctSaved}% saved</div>
          </div>
        </div>
      )}

      {/* V trivial message */}
      {vTrivial && (
        <div className="cost-formula-card">
          <h4 style={{ margin: '0 0 8px' }}>
            <span className="pill pill-v">V</span> Output symmetry
          </h4>
          <p style={{ margin: '4px 0', color: '#666' }}>
            {totalCount <= 1
              ? 'Scalar output — no output elements to reduce.'
              : 'Trivial group — no output symmetry savings.'}
          </p>
        </div>
      )}

      {/* W-side toggle + collapsible section */}
      {wHasSymmetry && (
        <div className={`cost-w-section ${showW ? 'cost-w-expanded' : 'cost-w-collapsed'}`} style={{ marginTop: 16 }}>
          <button
            className="cost-w-toggle"
            onClick={() => setShowW(s => !s)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: 'none',
              border: isBinary ? '1px solid #e0e0e0' : '1px dashed #ccc',
              borderRadius: 8,
              padding: '8px 14px', cursor: 'pointer', width: '100%',
              color: isBinary ? '#555' : '#888', fontSize: '0.9em',
            }}
          >
            <span style={{ fontSize: '1.1em' }}>{showW ? '▾' : '▸'}</span>
            <span className="pill pill-w" style={{ fontSize: '0.75em' }}>W</span>
            <span>{isBinary ? 'W-side savings (binary einsum — always applies)' : 'Show W-side savings'}</span>
            {!isBinary && (
              <span style={{ marginLeft: 'auto', fontSize: '0.8em', fontStyle: 'italic' }}>
                may not apply in multi-step paths
              </span>
            )}
          </button>

          {showW && (
            <div className="cost-formula-card" style={{
              marginTop: 8,
              opacity: isBinary ? 1 : 0.85,
              borderStyle: isBinary ? 'solid' : 'dashed',
            }}>
              <h4 style={{ margin: '0 0 8px', color: '#888' }}>
                <span className="pill pill-w">W</span> Inner-sum symmetry reduction
                <span style={{ fontSize: '0.7em', marginLeft: 8, fontWeight: 'normal', color: '#aaa' }}>
                  (single-step estimate)
                </span>
              </h4>
              <div className="burnside-compact" style={{ marginBottom: 12 }}>
                <code className="compact-ratio" style={{ opacity: 0.7 }}>
                  W:{wUniqueCount.toLocaleString()}/{wTotalCount.toLocaleString()}
                </code>
              </div>
              <div className="cost-values">
                <div className="cost-item">
                  <span className="cost-label">Dense cost</span>
                  <span className="cost-val">{denseCost.toLocaleString()}</span>
                </div>
                <div className="cost-item" style={{ borderColor: '#ffb74d' }}>
                  <span className="cost-label">V+W reduced cost</span>
                  <span className="cost-val" style={{ color: '#e68a00' }}>
                    {combinedReducedCost.toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="speedup-card" style={{ marginTop: 12, opacity: 0.8 }}>
                <div className="speedup-number" style={{ color: '#e68a00' }}>{wSpeedup.toFixed(1)}×</div>
                <div className="speedup-label">via {wGroupName}</div>
                <div className="speedup-detail">{combinedPctSaved}% saved (V+W combined)</div>
              </div>
              {isBinary ? (
                <p style={{
                  margin: '10px 0 0', fontSize: '0.82em', color: '#4a7',
                  padding: '6px 10px', background: '#f0faf4', borderRadius: 4, borderLeft: '3px solid #4a7',
                }}>
                  ✓ Binary einsum — all summed labels are contracted in this single step,
                  so W-side savings always apply.
                </p>
              ) : (
                <p style={{
                  margin: '10px 0 0', fontSize: '0.82em', color: '#999',
                  padding: '6px 10px', background: '#fff8f0', borderRadius: 4, borderLeft: '3px solid #ffb74d',
                }}>
                  ⚠ This einsum has {numOperands} operands, so the optimizer may decompose it into
                  multiple pairwise steps. The summed labels ({cost.wGroupName.match(/\{(.+)\}/)?.[1] || 'W'}) might
                  be split across different steps, reducing or eliminating W-side savings. The estimate
                  above assumes single-step contraction.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Cost bars */}
      <div className="cost-bars">
        <div className="cost-bar-row">
          <span className="bar-name">Dense</span>
          <div className="cost-bar-track">
            <div className="cost-bar-fill dense" style={{ width: '100%' }}></div>
          </div>
          <span className="bar-val">{denseCost.toLocaleString()}</span>
        </div>
        {!vTrivial && (
          <div className="cost-bar-row">
            <span className="bar-name">V only</span>
            <div className="cost-bar-track">
              <div className="cost-bar-fill reduced" style={{ width: `${ratio * 100}%` }}></div>
            </div>
            <span className="bar-val">{reducedCost.toLocaleString()}</span>
          </div>
        )}
        {wHasSymmetry && showW && (
          <div className="cost-bar-row" style={{ opacity: isBinary ? 1 : 0.6 }}>
            <span className="bar-name" style={{ fontStyle: isBinary ? 'normal' : 'italic' }}>V + W</span>
            <div className="cost-bar-track" style={{ borderStyle: isBinary ? 'solid' : 'dashed' }}>
              <div className="cost-bar-fill" style={{
                width: `${combinedRatio * 100}%`,
                background: '#ffb74d',
              }}></div>
            </div>
            <span className="bar-val" style={{ fontStyle: isBinary ? 'normal' : 'italic' }}>{combinedReducedCost.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}
