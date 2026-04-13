import { useState, useMemo } from 'react';

export default function GroupView({ group, sigmaResults, graph, example }) {
  const { vLabels, wLabels, vGenerators, vGeneratorsAll, vElements, vGroupName, vOrder, vDegree } = group;
  const [showDimino, setShowDimino] = useState(false);
  const hasW = wLabels && wLabels.length > 0;
  const operandNames = example?.operandNames || [];
  const subscripts = example?.subscripts || [];

  // Build provenance: which σ-π pair produced each generator
  const generatorProvenance = useMemo(() => {
    if (!sigmaResults) return [];
    const validPairs = sigmaResults.filter(r => r.isValid && !r.skipped && !r.piIsIdentity);
    const seen = new Set();
    const provenance = [];

    const vIdx = {};
    vLabels.forEach((l, i) => { vIdx[l] = i; });

    for (const r of validPairs) {
      const pi = r.pi;
      const arr = vLabels.map(l => vIdx[pi[l] ?? l]);
      const key = arr.join(',');
      if (seen.has(key)) continue;
      if (arr.every((v, i) => v === i)) continue;
      seen.add(key);

      // Build operand-level description of σ
      const sigmaDesc = buildSigmaDesc(r.sigma, operandNames, subscripts);
      // Full π on all labels
      const piStr = fmtPi(r.pi);
      // π restricted to V
      const piVStr = fmtRestricted(r.pi, vLabels);
      // Individual label mappings for the "what moves" display
      const labelMoves = Object.entries(r.pi)
        .filter(([k, v]) => k !== v)
        .map(([from, to]) => ({ from, to, isV: vLabels.includes(from) }));

      // Check if this π|V is one of the essential (minimal) generators
      const isEssential = vGenerators.some(g => {
        const gArr = Array.from({ length: vLabels.length }, (_, i) => g.apply(i));
        return gArr.join(',') === arr.join(',');
      });

      provenance.push({ sigma: r.sigma, sigmaDesc, piStr, piVStr, labelMoves, isEssential });
    }
    return provenance;
  }, [sigmaResults, vLabels, vGenerators, operandNames, subscripts]);

  // Simulate Dimino's algorithm step by step
  const diminoSteps = useMemo(() => {
    if (vGenerators.length === 0) return [];
    const steps = [];
    const n = vGenerators[0].size;
    const identity = { array: Array.from({ length: n }, (_, i) => i), isIdentity: true };
    const elements = [identity];
    const seen = new Set([identity.array.join(',')]);

    steps.push({ label: 'Start with identity', newElements: [identity], totalSoFar: 1 });

    for (let gi = 0; gi < vGenerators.length; gi++) {
      const gen = vGenerators[gi];
      const genArr = Array.from({ length: n }, (_, i) => gen.apply(i));
      const genKey = genArr.join(',');
      if (seen.has(genKey)) continue;

      seen.add(genKey);
      const genElem = { array: genArr, isIdentity: false };
      elements.push(genElem);
      steps.push({
        label: `Add generator: ${gen.cycleNotation(vLabels)}`,
        newElements: [genElem],
        totalSoFar: elements.length,
      });

      let frontier = [genElem];
      let round = 0;
      while (frontier.length > 0 && round < 50) {
        round++;
        const nextFrontier = [];
        for (const elem of frontier) {
          for (let gj = 0; gj <= gi; gj++) {
            const g2 = vGenerators[gj];
            const g2Arr = Array.from({ length: n }, (_, j) => g2.apply(j));
            // elem ∘ g2
            const prodArr = elem.array.map(i => g2Arr[i]);
            const prodKey = prodArr.join(',');
            if (!seen.has(prodKey)) {
              seen.add(prodKey);
              const p = { array: prodArr, isIdentity: false };
              elements.push(p);
              nextFrontier.push(p);
            }
            // g2 ∘ elem
            const prod2Arr = g2Arr.map(i => elem.array[i]);
            const prod2Key = prod2Arr.join(',');
            if (!seen.has(prod2Key)) {
              seen.add(prod2Key);
              const p = { array: prod2Arr, isIdentity: false };
              elements.push(p);
              nextFrontier.push(p);
            }
          }
        }
        if (nextFrontier.length > 0) {
          steps.push({
            label: `Closure: compose with generators`,
            newElements: nextFrontier,
            totalSoFar: elements.length,
          });
        }
        frontier = nextFrontier;
      }
    }
    return steps;
  }, [vGenerators, vLabels]);

  return (
    <div className="group-view">
      {/* Group badge */}
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

      {/* ── Provenance ── */}
      {generatorProvenance.length > 0 && (
        <div className="gen-provenance">
          <h4>How generators are found</h4>
          <p className="gen-provenance-desc">
            Each valid operand swap from Step 4 forces index labels to move.
            {hasW
              ? ' Only the movements within the free labels V become generators (summed labels W are dropped).'
              : ' Since all labels are free (W is empty), the full label permutation is the generator.'}
          </p>

          <table className="gen-table">
            <thead>
              <tr>
                <th>Swap operands</th>
                <th>Labels move</th>
                <th>{hasW ? 'Generator (V only)' : 'Generator'}</th>
              </tr>
            </thead>
            <tbody>
              {generatorProvenance.map((p, i) => (
                <tr key={i} className={p.isEssential ? '' : 'gen-row-redundant'}>
                  <td className="gen-swap-cell">{p.sigmaDesc}</td>
                  <td>
                    <span className="gen-moves-inline">
                      {p.labelMoves.map(({ from, to, isV }, j) => (
                        <span key={j} className={`gen-move ${isV ? 'gen-move-v' : 'gen-move-w'}`}>
                          {from}→{to}
                        </span>
                      ))}
                    </span>
                  </td>
                  <td className="gen-result-cell">
                    <code>{p.piVStr}</code>
                    {!p.isEssential && <span className="gen-redundant-tag">redundant</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Dimino build-up ── */}
      {vGenerators.length > 0 && (
        <div className="dimino-section">
          <h4>
            Dimino's algorithm
            <button className="dimino-toggle" onClick={() => setShowDimino(s => !s)}>
              {showDimino ? 'Hide steps' : 'Show steps'}
            </button>
          </h4>

          {showDimino && (
            <div className="dimino-steps">
              {diminoSteps.map((step, i) => (
                <div key={i} className="dimino-step">
                  <div className="dimino-step-header">
                    <span className="dimino-step-num">{i + 1}</span>
                    <span className="dimino-step-label">{step.label}</span>
                    <span className="dimino-step-count">{step.totalSoFar} elements</span>
                  </div>
                  <div className="dimino-step-elements">
                    {step.newElements.map((elem, j) => (
                      <code key={j} className={`perm-card ${elem.isIdentity ? 'identity' : ''} dimino-new`}>
                        {elem.isIdentity ? 'e' : arrToCycleNotation(elem.array, vLabels)}
                      </code>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="group-elements">
            <h4>All {vOrder} elements</h4>
            <div className="perm-list">
              {vElements.map((g, i) => {
                const cycles = g.cyclicForm();
                const isId = g.isIdentity;
                const isGen = vGenerators.some(gen => gen.key() === g.key());
                return (
                  <div key={i} className={`perm-card ${isId ? 'identity' : ''} ${isGen ? 'generator' : ''}`}>
                    <code className="perm-notation">{g.cycleNotation(vLabels)}</code>
                    <span className="perm-structure">
                      {isId ? 'identity' : isGen ? 'generator' : cycles.map(c => `${c.length}-cycle`).join(' + ')}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {vGenerators.length === 0 && (
        <div className="group-elements">
          <h4>All {vOrder} elements</h4>
          <div className="perm-list">
            <div className="perm-card identity">
              <code className="perm-notation">e</code>
              <span className="perm-structure">identity</span>
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
