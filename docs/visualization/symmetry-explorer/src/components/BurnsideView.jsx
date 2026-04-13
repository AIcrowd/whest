import Latex from './Latex.jsx';

export default function BurnsideView({ burnside, group, dimensionN }) {
  const { perElement, totalFixed, uniqueCount, totalCount } = burnside;
  const { vLabels, vOrder, vGroupName } = group;

  const isTrivial = vOrder <= 1;
  const noVLabels = !vLabels || vLabels.length === 0;
  const wHasSym = group.wOrder > 1;
  const pct = totalCount > 0 ? ((uniqueCount / totalCount) * 100).toFixed(1) : '100.0';
  const saved = (100 - parseFloat(pct)).toFixed(1);

  // Trivial V-side group or no output labels
  if (isTrivial || noVLabels) {
    return (
      <div className="burnside-view">
        <div className="burnside-compact">
          <code className="compact-ratio compact-ratio-trivial">
            {noVLabels ? 'scalar output' : `V:${totalCount.toLocaleString()}/${totalCount.toLocaleString()}`}
          </code>
        </div>
        <div className="burnside-trivial">
          <div className="trivial-icon">—</div>
          <div className="trivial-text">
            <strong>{noVLabels ? 'Scalar output — no V-side to reduce' : 'No V-side symmetry detected'}</strong>
            <span>
              {noVLabels
                ? `The output is a scalar (no free labels). ${wHasSym ? `W-side ${group.wGroupName} symmetry may reduce the inner contraction cost.` : ''}`
                : `The V-side group is trivial (order 1). All ${totalCount.toLocaleString()} output elements are unique — no Burnside reduction.`}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Escape braces for LaTeX
  const groupNameLatex = vGroupName.replace(/([{}])/g, '\\$1');

  const formulaLatex = String.raw`\text{unique} = \frac{1}{|${groupNameLatex}|} \sum_{g \in G} \prod_{\text{cycles } c} n_c = \frac{${totalFixed}}{${vOrder}} = \mathbf{${uniqueCount.toLocaleString()}}`;
  const contextLatex = String.raw`\text{(of ${totalCount.toLocaleString()} total, } n = ${dimensionN}\text{)}`;

  return (
    <div className="burnside-view">
      <div className="burnside-compact">
        <code className="compact-ratio">V:{uniqueCount.toLocaleString()}/{totalCount.toLocaleString()}</code>
      </div>

      <div className="burnside-formula">
        <Latex math={formulaLatex} display />
        <div style={{ textAlign: 'center', marginTop: 4 }}>
          <Latex math={contextLatex} />
        </div>
      </div>

      <div className="burnside-table-wrap">
        <table className="burnside-table">
          <thead>
            <tr>
              <th><Latex math="g" /></th>
              <th>Cycle decomposition</th>
              <th><Latex math={String.raw`\prod n_c`} /></th>
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
              const formulaParts = row.cycles.map(c => `${c.size}`);
              const formulaStr = formulaParts.join(String.raw` \times `);
              return (
                <tr key={i}>
                  <td><code>{notation}</code></td>
                  <td className="cycle-desc"><code>{cycleDesc}</code></td>
                  <td className="formula-cell"><Latex math={formulaStr} /></td>
                  <td className="fixed-count">{row.fixedCount.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3} className="total-label">
                <Latex math={String.raw`\text{Total} \div |G| = ${totalFixed.toLocaleString()} \div ${vOrder}`} />
              </td>
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
