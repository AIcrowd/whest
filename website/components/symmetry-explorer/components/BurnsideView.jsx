import Latex from './Latex.jsx';

export default function BurnsideView({ burnside, group, dimensionN }) {
  const {
    fullPerElement = [],
    fullTotalFixed = 0,
    orbitCount = 0,
    totalTupleCount = 0,
  } = burnside;

  const pct = totalTupleCount > 0 ? ((orbitCount / totalTupleCount) * 100).toFixed(1) : '100.0';
  const groupNameLatex = (group.fullGroupName || 'G').replace(/([{}])/g, '\\$1');
  const formulaLatex = String.raw`\text{evaluation reps} = \frac{1}{|${groupNameLatex}|} \sum_{g \in G} \mathrm{Fix}(g) = \frac{${fullTotalFixed}}{${group.fullOrder}} = \mathbf{${orbitCount.toLocaleString()}}`;

  return (
    <div className="burnside-view">
      <div className="burnside-compact">
        <code className="compact-ratio">
          eval reps: {orbitCount.toLocaleString()}/{totalTupleCount.toLocaleString()}
        </code>
      </div>

      <p className="group-side-note">
        Burnside is counting full tuple orbits under <strong>{group.fullGroupName}</strong>. That gives
        the number of symmetry-unique representatives we need to evaluate.
      </p>

      <div className="burnside-formula">
        <Latex math={formulaLatex} display />
        <div style={{ textAlign: 'center', marginTop: 4 }}>
          <Latex math={String.raw`\text{(with } n = ${dimensionN}\text{ and } |L| = ${group.fullDegree}\text{ labels)}`} />
        </div>
      </div>

      <div className="burnside-table-wrap">
        <table className="burnside-table">
          <thead>
            <tr>
              <th><Latex math="g" /></th>
              <th>Cycle decomposition on active labels</th>
              <th>Fixed tuples</th>
            </tr>
          </thead>
          <tbody>
            {fullPerElement.map((row, idx) => {
              const notation = row.element.cycleNotation(group.allLabels);
              const cycleDesc = row.cycles.map((cycle) => (
                cycle.cycle.length === 1
                  ? `(${group.allLabels[cycle.cycle[0]]})`
                  : `(${cycle.cycle.map((labelIdx) => group.allLabels[labelIdx]).join(' ')})`
              )).join(' ');

              return (
                <tr key={`burnside-row-${idx}`}>
                  <td><code>{notation}</code></td>
                  <td className="cycle-desc"><code>{cycleDesc || 'identity'}</code></td>
                  <td className="fixed-count">{row.fixedCount.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={2} className="total-label">
                <Latex math={String.raw`\text{Total fixed tuples} \div |G| = ${fullTotalFixed.toLocaleString()} \div ${group.fullOrder}`} />
              </td>
              <td className="total-value">{orbitCount.toLocaleString()}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="burnside-bar">
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${pct}%` }}>
            <span className="bar-label">{orbitCount.toLocaleString()} evaluation reps ({pct}%)</span>
          </div>
        </div>
        <div className="bar-legend">
          <span>{totalTupleCount.toLocaleString()} dense tuples</span>
          <span className="savings">{(100 - parseFloat(pct)).toFixed(1)}% merged into orbits</span>
        </div>
      </div>
    </div>
  );
}
