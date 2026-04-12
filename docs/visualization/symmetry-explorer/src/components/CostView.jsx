export default function CostView({ cost }) {
  const {
    denseCost, reducedCost, ratio, speedup,
    uniqueCount, totalCount, groupName, groupOrder,
  } = cost;

  const pctSaved = ((1 - ratio) * 100).toFixed(1);

  return (
    <div className="cost-view">
      <div className="cost-formula-card">
        <div className="cost-equation">
          <span className="cost-term">cost</span>
          <span className="cost-eq">=</span>
          <span className="cost-term dense">dense_cost</span>
          <span className="cost-op">×</span>
          <span className="cost-fraction">
            <span className="frac-num">{uniqueCount.toLocaleString()}</span>
            <span className="frac-bar"></span>
            <span className="frac-den">{totalCount.toLocaleString()}</span>
          </span>
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
