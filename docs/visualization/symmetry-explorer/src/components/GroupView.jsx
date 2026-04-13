export default function GroupView({ group, sigmaResults, graph, example }) {
  const { vLabels, wLabels } = group;
  const operandNames = example?.operandNames || [];
  const subscripts = example?.subscripts || [];

  const hasW = wLabels && wLabels.length > 0;

  return (
    <div className="group-view">
      {/* ── V-side group ── */}
      <GroupSideSection
        title="V-side (output / free labels)"
        labels={vLabels}
        generators={group.vGenerators || []}
        elements={group.vElements || []}
        groupName={group.vGroupName}
        order={group.vOrder}
        degree={group.vDegree}
        pill="V"
        pillClass="pill-v"
        note="Used for Burnside counting → FLOP savings"
      />

      {/* ── W-side group ── */}
      {hasW && (
        <GroupSideSection
          title="W-side (contracted / summed labels)"
          labels={wLabels}
          generators={group.wGenerators || []}
          elements={group.wElements || []}
          groupName={group.wGroupName || 'trivial'}
          order={group.wOrder || 1}
          degree={group.wDegree || wLabels.length}
          pill="W"
          pillClass="pill-w"
          note="Inner-sum symmetry — may reduce contraction cost when all summed labels are contracted in one step (not always applicable across multi-step paths)"
        />
      )}
    </div>
  );
}

/* ── Group side section component ── */

function GroupSideSection({
  title, labels, generators, elements, groupName, order, degree,
  pill, pillClass, note
}) {
  const hasSym = order > 1;

  return (
    <div className="group-side-section">
      <h4 className="group-side-title">
        <span className={`pill ${pillClass}`}>{pill}</span> {title}
      </h4>

      <div className="group-summary-row">
        <div className="group-badge">
          <span className="group-name">{groupName}</span>
          <span className="group-order">order {order}</span>
        </div>
        <div className="group-meta">
          <span>{degree} labels ({labels.join(', ')})</span>
          <span>{generators.length} generator{generators.length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      {note && <p className="group-side-note">{note}</p>}

      {hasSym && (
        <div className="group-elements">
          <div className="perm-list">
            {elements.map((g, i) => {
              const cycles = g.cyclicForm();
              const isId = g.isIdentity;
              const isGen = generators.some(gen => gen.key() === g.key());
              return (
                <div key={i} className={`perm-card ${isId ? 'identity' : ''} ${isGen ? 'generator' : ''}`}>
                  <code className="perm-notation">{g.cycleNotation(labels)}</code>
                  <span className="perm-structure">
                    {isId ? 'identity' : isGen ? 'generator' : cycles.map(c => `${c.length}-cycle`).join(' + ')}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!hasSym && (
        <div className="group-elements">
          <div className="perm-list">
            <div className="perm-card identity">
              <code className="perm-notation">e</code>
              <span className="perm-structure">identity only</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Helpers ── */

function buildSigmaDesc(sigma, operandNames, subscripts) {
  const entries = Object.entries(sigma).filter(([k, v]) => Number(k) !== v);
  if (entries.length === 0) return 'identity';

  const visited = new Set();
  const cycles = [];
  for (const [k] of entries) {
    const kn = Number(k);
    if (visited.has(kn)) continue;
    const cycle = [];
    let cur = kn;
    while (!visited.has(cur)) {
      visited.add(cur);
      const name = operandNames[cur] || `Op${cur}`;
      const sub = subscripts[cur] || '';
      cycle.push({ idx: cur, name, sub });
      cur = sigma[cur] ?? cur;
    }
    if (cycle.length > 1) cycles.push(cycle);
  }

  const opTag = (c) => (
    <span key={c.idx} className="sigma-op-tag">
      <span className="sigma-desc-op">{c.name}</span>
      <sub className="sigma-desc-sub">{c.sub}</sub>
    </span>
  );

  return (
    <span className="sigma-desc">
      {cycles.map((cycle, ci) => (
        <span key={ci} className="sigma-desc-cycle">
          {ci > 0 && <span className="sigma-desc-sep">, </span>}
          {cycle.length === 2 ? (
            // 2-cycle: swap with ↔
            <>{opTag(cycle[0])}<span className="sigma-desc-arrow"> ↔ </span>{opTag(cycle[1])}</>
          ) : (
            // k-cycle: rotation with → and wrap
            <>
              {cycle.map((c, i) => (
                <span key={c.idx}>
                  {i > 0 && <span className="sigma-desc-arrow"> → </span>}
                  {opTag(c)}
                </span>
              ))}
              <span className="sigma-desc-arrow"> ⟲</span>
            </>
          )}
        </span>
      ))}
    </span>
  );
}

function fmtPi(pi) {
  if (!pi) return '—';
  const entries = Object.entries(pi).filter(([k, v]) => k !== v);
  if (entries.length === 0) return 'e';
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
  return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'e';
}

function fmtRestricted(pi, vLabels) {
  const entries = vLabels.filter(l => pi[l] !== l);
  if (entries.length === 0) return 'e';
  const visited = new Set();
  const cycles = [];
  for (const k of entries) {
    if (visited.has(k)) continue;
    const cycle = [];
    let cur = k;
    while (!visited.has(cur) && vLabels.includes(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = pi[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.map(c => '(' + c.join(' ') + ')').join('') || 'e';
}

function arrToCycleNotation(arr, labels) {
  const visited = new Set();
  const cycles = [];
  for (let i = 0; i < arr.length; i++) {
    if (visited.has(i) || arr[i] === i) continue;
    const cycle = [];
    let cur = i;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(labels[cur]);
      cur = arr[cur];
    }
    if (cycle.length > 1) cycles.push(cycle);
  }
  return cycles.length > 0 ? cycles.map(c => '(' + c.join(' ') + ')').join('') : 'e';
}
