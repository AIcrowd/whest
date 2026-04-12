export default function BurnsideView({ burnside, group, dimensionN }) {
  const { perElement, totalFixed, uniqueCount, totalCount } = burnside;
  const { vLabels, vOrder, vGroupName } = group;

  const pct = ((uniqueCount / totalCount) * 100).toFixed(1);
  const saved = (100 - parseFloat(pct)).toFixed(1);

  return (
    <div className="burnside-view">
      <div className="burnside-formula">
        unique = (1 / |{vGroupName}|) × Σ<sub>g∈G</sub> Π<sub>cycles</sub> n<sub>c</sub>
        {' '}= {totalFixed} / {vOrder} = <strong>{uniqueCount.toLocaleString()}</strong>
        <span className="dim-text"> (of {totalCount.toLocaleString()} total, n={dimensionN})</span>
      </div>

      <div className="burnside-table-wrap">
        <table className="burnside-table">
          <thead>
            <tr>
              <th>g</th>
              <th>Cycle decomposition</th>
              <th>Π n<sub>c</sub></th>
              <th>Fixed points</th>
            </tr>
          </thead>
          <tbody>
            {perElement.map((row, i) => {
              const notation = row.element.cycleNotation(vLabels);
              const cycleDesc = row.cycles.map(c =>
                c.cycle.length === 1
                  ? `(${vLabels[c.cycle[0]]})`
                  : `(${c.cycle.map(j => vLabels[j]).join(' ')})`
              ).join(' ');
              const formula = row.cycles.map(c => `${c.size}`).join(' × ');
              return (
                <tr key={i}>
                  <td><code>{notation}</code></td>
                  <td className="cycle-desc"><code>{cycleDesc}</code></td>
                  <td className="formula-cell"><code>{formula}</code></td>
                  <td className="fixed-count">{row.fixedCount.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3} className="total-label">Total ÷ |G| = {totalFixed.toLocaleString()} ÷ {vOrder}</td>
              <td className="total-value">{uniqueCount.toLocaleString()}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="burnside-bar">
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${pct}%` }}>
            <span className="bar-label">{uniqueCount.toLocaleString()} unique ({pct}%)</span>
          </div>
        </div>
        <div className="bar-legend">
          <span>{totalCount.toLocaleString()} total elements</span>
          <span className="savings">{saved}% redundant</span>
        </div>
      </div>
    </div>
  );
}
