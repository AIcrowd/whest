export default function GroupView({ group }) {
  const { vLabels, vGenerators, vElements, vGroupName, vOrder, vDegree } = group;

  return (
    <div className="group-view">
      <div className="group-summary-row">
        <div className="group-badge">
          <span className="group-name">{vGroupName}</span>
          <span className="group-order">order {vOrder}</span>
        </div>
        <div className="group-meta">
          <span>Degree: {vDegree} labels ({vLabels.join(', ')})</span>
          <span>Generators: {vGenerators.length}</span>
        </div>
      </div>

      {vGenerators.length > 0 && (
        <div className="group-generators">
          <h4>Generators</h4>
          <div className="perm-list">
            {vGenerators.map((g, i) => (
              <code key={i} className="perm-card generator">
                {g.cycleNotation(vLabels)}
              </code>
            ))}
          </div>
        </div>
      )}

      <div className="group-elements">
        <h4>All {vOrder} Elements <span className="dim-text">(via Dimino's algorithm)</span></h4>
        <div className="perm-list">
          {vElements.map((g, i) => {
            const cycles = g.cyclicForm();
            const isId = g.isIdentity;
            return (
              <div key={i} className={`perm-card ${isId ? 'identity' : ''}`}>
                <code className="perm-notation">
                  {g.cycleNotation(vLabels)}
                </code>
                <span className="perm-structure">
                  {isId ? '—' : cycles.map(c => `${c.length}-cycle`).join(' + ')}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
