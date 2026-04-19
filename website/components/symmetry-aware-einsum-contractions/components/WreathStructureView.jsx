import { useMemo } from 'react';
import Latex from './Latex.jsx';

/**
 * WreathStructureView — visualizes the σ-loop's wreath structure
 * ∏_i (H_i ≀ S_{m_i}) and the 3-way π-dedup classification of its
 * elements for the current preset.
 *
 * Three bands, stacked:
 *   A. wreath factorisation formula + operand legend.
 *   B. per-element table (small wreath) or aggregated summary (large).
 *   C. count summary + bridge to the modal's formal argument.
 *
 * This revision implements Bands A and B; Band C is added in Task 9.
 */

const BAND_B_THRESHOLD = 24;

export default function WreathStructureView({ analysis, example, onOpenModalSection }) {
  const symmetry = analysis?.symmetry;
  const wreathElements = symmetry?.wreathElements;
  const opNames = example?.expression?.operandNames?.split(',').map((s) => s.trim()) || [];
  const variables = example?.variables || [];

  const factors = useMemo(() => {
    if (!symmetry) return [];
    const { identicalGroups = [] } = symmetry;
    return identicalGroups.map((group) => {
      const firstPos = group[0];
      const name = opNames[firstPos];
      const variable = variables.find((v) => v.name === name);
      const rank = example?.subscripts?.[firstPos]?.length
        ?? example?.expression?.subscripts?.split(',')[firstPos]?.trim().length
        ?? 0;
      const m = group.length;
      return {
        name,
        rank,
        m,
        symmetryLabel: describeSymmetry(variable, rank),
        hOrder: hGroupOrder(variable, rank),
      };
    });
  }, [symmetry, opNames, variables, example]);

  const totalOrder = factors.reduce((acc, f) => acc * (f.hOrder ** f.m) * factorial(f.m), 1);

  if (!wreathElements || wreathElements.length === 0) return null;

  const smallWreath = wreathElements.length <= BAND_B_THRESHOLD;

  const formulaLatex = factors.length === 0
    ? 'G_{\\text{wreath}} = \\{e\\}'
    : `G_{\\text{wreath}} = ${factors.map((f) => `(${f.symmetryLabel} \\wr S_{${f.m}})`).join(' \\times ')} = ${totalOrder}`;

  return (
    <div className="rounded-md border border-border/60 bg-muted/10 px-4 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground mb-2">
        Wreath structure
      </div>
      <div className="mb-3">
        <Latex math={formulaLatex} display />
      </div>
      <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        {factors.map((f, i) => (
          <div key={i} className="rounded border border-border/40 bg-white px-2 py-1 font-mono">
            <span className="font-semibold text-foreground">{f.name}</span>
            <span className="ml-2">rank {f.rank}</span>
            <span className="ml-2">H = {f.symmetryLabel}</span>
            <span className="ml-2">m = {f.m}</span>
          </div>
        ))}
      </div>
      {/* Band B — per-element table (small wreath) or aggregated summary (large wreath, Task 8) */}
      {smallWreath ? (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr className="border-b border-border/60 text-left text-muted-foreground">
                <th className="px-2 py-2 font-semibold">#</th>
                <th className="px-2 py-2 font-semibold">Factor decomposition</th>
                <th className="px-2 py-2 font-semibold">Matrix effect</th>
                <th className="px-2 py-2 font-semibold">derivePi(σ)</th>
                <th className="px-2 py-2 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {wreathElements.map((e, idx) => (
                <WreathElementRow key={e.id} element={e} index={idx} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        // Large wreath — aggregated summary lands in Task 8
        null
      )}
      {/* Band C (summary + bridge) added in Task 9 */}
    </div>
  );
}

function WreathElementRow({ element, index }) {
  const classLabel = {
    'valid': '✓ valid',
    'matrix-preserving': '✗ matrix-preserving',
    'rejected': '✗ rejected',
  }[element.classification];

  const rowColorClass = {
    'valid': '',
    'matrix-preserving': 'text-muted-foreground italic',
    'rejected': 'text-amber-600/80 italic',
  }[element.classification];

  const matrixEffect = element.matrixPreserving
    ? <span className="text-muted-foreground">M_σ = M</span>
    : <span>M_σ ≠ M</span>;

  const piStr = element.derivePiResult === null
    ? '—'
    : piCycleNotation(element.derivePiResult);

  const factorStr = element.factorization
    ? element.factorization.map((f) => {
        const baseStr = f.baseTuple.map((h) => (h.isIdentity ? 'e' : h.cycleNotation())).join(',');
        const topStr = f.topPerm.map((v, idx) => (v === idx ? null : `${idx}→${v}`)).filter(Boolean).join(',') || 'id';
        return `(${baseStr}; ${topStr})`;
      }).join(' × ')
    : '';

  return (
    <tr className={`border-b border-border/30 ${rowColorClass}`}>
      <td className="px-2 py-1 font-mono text-[11px]">{index}</td>
      <td className="px-2 py-1 font-mono text-[11px]">{factorStr}</td>
      <td className="px-2 py-1">{matrixEffect}</td>
      <td className="px-2 py-1 font-mono text-[11px]">{piStr}</td>
      <td className="px-2 py-1">{classLabel}</td>
    </tr>
  );
}

function factorial(n) { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; }

function describeSymmetry(variable, rank) {
  if (!variable || variable.symmetry === 'none') return '\\{e\\}';
  const axes = variable.symAxes || Array.from({ length: rank }, (_, i) => i);
  const k = axes.length;
  if (variable.symmetry === 'symmetric') return `S_${k}`;
  if (variable.symmetry === 'cyclic') return `C_${k}`;
  if (variable.symmetry === 'dihedral') return `D_${k}`;
  if (variable.symmetry === 'custom') return `\\text{custom}_${k}`;
  return '\\{e\\}';
}

function hGroupOrder(variable, rank) {
  if (!variable || variable.symmetry === 'none') return 1;
  const axes = variable.symAxes || Array.from({ length: rank }, (_, i) => i);
  const k = axes.length;
  if (variable.symmetry === 'symmetric') return factorial(k);
  if (variable.symmetry === 'cyclic') return k;
  if (variable.symmetry === 'dihedral') return k >= 3 ? 2 * k : k;
  // 'custom' — order not known from declaration alone; engine carries it
  // but we compute it lazily in the widget. For the Band-A formula we
  // approximate as 1 to avoid a false value. Band B shows element-by-
  // element truth via wreathElements.length.
  return 1;
}

// derivePi returns a plain label→label object, not a Permutation. These
// helpers give it isIdentity / cycleNotation semantics without upgrading
// the engine's data shape.
function piIsIdentity(pi) {
  if (!pi) return false;
  for (const [k, v] of Object.entries(pi)) {
    if (k !== v) return false;
  }
  return true;
}

function piCycleNotation(pi) {
  if (!pi) return '—';
  if (piIsIdentity(pi)) return 'identity';
  const visited = new Set();
  const cycles = [];
  const keys = Object.keys(pi).sort();
  for (const label of keys) {
    if (visited.has(label)) continue;
    const cycle = [];
    let cur = label;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = pi[cur];
    }
    if (cycle.length >= 2) cycles.push(`(${cycle.join(' ')})`);
  }
  return cycles.length ? cycles.join('') : 'identity';
}
