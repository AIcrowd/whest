import Latex from './Latex.jsx';

export default function CostView({ cost }) {
  const {
    denseCost, reducedCost, ratio, speedup,
    uniqueCount, totalCount, groupName, groupOrder,
    wHasSymmetry, wReducedCost, wRatio, wSpeedup,
    wUniqueCount, wTotalCount, wGroupName,
    combinedRatio, combinedReducedCost,
  } = cost;

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

  const wPctSaved = wHasSymmetry ? ((1 - wRatio) * 100).toFixed(1) : '0.0';
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

      {/* W-side reduction */}
      {wHasSymmetry && (
        <div className="cost-formula-card" style={{ marginTop: 16 }}>
          <h4 style={{ margin: '0 0 8px' }}>
            <span className="pill pill-w">W</span> Inner-sum symmetry reduction
          </h4>
          <div className="burnside-compact" style={{ marginBottom: 12 }}>
            <code className="compact-ratio">W:{wUniqueCount.toLocaleString()}/{wTotalCount.toLocaleString()}</code>
          </div>
          <div className="cost-values">
            <div className="cost-item">
              <span className="cost-label">Dense cost</span>
              <span className="cost-val">{denseCost.toLocaleString()}</span>
            </div>
            <div className="cost-item highlight">
              <span className="cost-label">W-reduced cost</span>
              <span className="cost-val">{combinedReducedCost.toLocaleString()}</span>
            </div>
          </div>
          <div className="speedup-card" style={{ marginTop: 12 }}>
            <div className="speedup-number">{wSpeedup.toFixed(1)}×</div>
            <div className="speedup-label">via {wGroupName}</div>
            <div className="speedup-detail">{combinedPctSaved}% saved (V+W combined)</div>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '0.85em', color: '#888' }}>
            ⚠ W-side savings apply when all summed labels are contracted in one step.
            In multi-step paths, labels may be split across steps.
          </p>
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
        {wHasSymmetry && (
          <div className="cost-bar-row">
            <span className="bar-name">V + W</span>
            <div className="cost-bar-track">
              <div className="cost-bar-fill" style={{ width: `${combinedRatio * 100}%`, background: '#ffb74d' }}></div>
            </div>
            <span className="bar-val">{combinedReducedCost.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}
