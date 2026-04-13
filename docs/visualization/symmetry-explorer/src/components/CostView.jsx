import Latex from './Latex.jsx';

export default function CostView({ cost }) {
  const {
    denseCost, reducedCost, ratio, speedup,
    uniqueCount, totalCount, groupName, groupOrder,
  } = cost;

  const isTrivial = groupOrder <= 1;
  const pctSaved = ((1 - ratio) * 100).toFixed(1);

  if (isTrivial) {
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

  const formulaLatex = String.raw`\text{cost} = \text{dense\_cost} \times \frac{${uniqueCount.toLocaleString()}}{${totalCount.toLocaleString()}}`;

  return (
    <div className="cost-view">
      <div className="cost-formula-card">
        <div className="burnside-compact" style={{ marginBottom: 12 }}>
          <code className="compact-ratio">V:{uniqueCount.toLocaleString()}/{totalCount.toLocaleString()}</code>
        </div>
        <div className="burnside-formula" style={{ marginBottom: 16 }}>
          <Latex math={formulaLatex} display />
        </div>
        <div className="cost-values">
          <div className="cost-item">
            <span className="cost-label">Dense cost</span>
            <span className="cost-val">{denseCost.toLocaleString()}</span>
          </div>
          <div className="cost-item highlight">
            <span className="cost-label">Reduced cost</span>
            <span className="cost-val">{reducedCost.toLocaleString()}</span>
          </div>
        </div>
      </div>

      <div className="speedup-card">
        <div className="speedup-number">{speedup.toFixed(1)}×</div>
        <div className="speedup-label">speedup via {groupName}</div>
        <div className="speedup-detail">{pctSaved}% of FLOPs saved</div>
      </div>

      <div className="cost-bars">
        <div className="cost-bar-row">
          <span className="bar-name">Dense</span>
          <div className="cost-bar-track">
            <div className="cost-bar-fill dense" style={{ width: '100%' }}></div>
          </div>
          <span className="bar-val">{denseCost.toLocaleString()}</span>
        </div>
        <div className="cost-bar-row">
          <span className="bar-name">Symmetric</span>
          <div className="cost-bar-track">
            <div className="cost-bar-fill reduced" style={{ width: `${ratio * 100}%` }}></div>
          </div>
          <span className="bar-val">{reducedCost.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}
