export default function CostView({ costModel }) {
  if (!costModel) return null;

  const {
    orbitCount = 0,
    evaluationCost = 0,
    reductionCost = 0,
  } = costModel;

  const totalCost = evaluationCost + reductionCost;

  return (
    <div className="cost-view">
      <div className="cost-formula-card">
        <h4 style={{ margin: '0 0 8px' }}>Evaluation vs Reduction</h4>
        <p className="group-side-note" style={{ marginTop: 0 }}>
          Evaluation cost counts one unique product formation per full orbit representative.
          Reduction cost counts one output update for each distinct output bin hit by that orbit.
        </p>

        <div className="cost-values">
          <div className="cost-item">
            <span className="cost-label">Orbit representatives</span>
            <span className="cost-val">{orbitCount.toLocaleString()}</span>
          </div>
          <div className="cost-item">
            <span className="cost-label">Evaluation cost</span>
            <span className="cost-val">{evaluationCost.toLocaleString()}</span>
          </div>
          <div className="cost-item highlight">
            <span className="cost-label">Reduction cost</span>
            <span className="cost-val">{reductionCost.toLocaleString()}</span>
          </div>
        </div>

        <div className="speedup-card" style={{ marginTop: 12 }}>
          <div className="speedup-number">{totalCost.toLocaleString()}</div>
          <div className="speedup-label">total teaching-model cost</div>
          <div className="speedup-detail">evaluation + reduction under the exact small-n model</div>
        </div>
      </div>
    </div>
  );
}
