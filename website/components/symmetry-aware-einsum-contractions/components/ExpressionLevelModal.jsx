import { useEffect, useMemo } from 'react';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import AppendixSection from './AppendixSection.jsx';
import AppendixTheoremBlock from './AppendixTheoremBlock.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';
import { EXAMPLES } from '../data/examples.js';
import { variableSymmetryLabel } from '../lib/symmetryLabel.js';
import { notationColor } from '../lib/notationSystem.js';

// Lookup map keyed by preset id so §7's savings table can pull the raw
// einsum and per-operand symmetry declarations straight from the source
// of truth, rather than duplicating that data into the row array below.
const EXAMPLES_BY_ID = new Map(EXAMPLES.map((ex) => [ex.id, ex]));

/**
 * Per-preset operand listing: distinct operand names in first-appearance
 * order, annotated with their repeat count (drives top-group transpositions
 * in the wreath σ-loop) and declared axis symmetry (base-group generators).
 * Used to render the "Operand sym" column in §7's savings table.
 *
 * The `sym` field uses `variableSymmetryLabel` — the same short-form
 * vocabulary (`dense`, `S3`, `C4`, `D2`, `custom (N gens)`) the main-page
 * builder and preset sidebar use, so the reader does not have to learn a
 * second notation inside the appendix.
 */
function describeOperands(preset) {
  if (!preset) return [];
  const opNames = (preset.expression?.operandNames ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const counts = new Map();
  for (const n of opNames) counts.set(n, (counts.get(n) ?? 0) + 1);
  const seen = new Set();
  const rows = [];
  for (const n of opNames) {
    if (seen.has(n)) continue;
    seen.add(n);
    const v = preset.variables?.find((x) => x.name === n);
    rows.push({
      name: n,
      count: counts.get(n),
      sym: variableSymmetryLabel(v),
    });
  }
  return rows;
}

/**
 * Long-form, human-readable description of a single operand's declared axis
 * symmetry. Richer than `variableSymmetryLabel` (which returns the short
 * badge string `S3`, `C4`, `dense`, etc.) — this also names the axes
 * involved and spells out the custom generators, so the appendix tooltip
 * can show the reader the full construction without them needing to open
 * the builder.
 *
 * Examples:
 *   - dense
 *   - S3 on axes (0, 1, 2)
 *   - C3 on axes (1, 2, 3)
 *   - custom ⟨(0 1), (0 2), (3 4)⟩
 */
function detailedSymmetryDescription(variable) {
  if (!variable || variable.symmetry === 'none') return 'dense';
  if (variable.symmetry === 'custom') {
    return `custom ⟨${(variable.generators || '').trim() || '∅'}⟩`;
  }
  const axes = variable.symAxes ?? [];
  const axesStr = axes.length ? `axes (${axes.join(', ')})` : 'all axes';
  const prefix =
    variable.symmetry === 'symmetric' ? 'S' :
    variable.symmetry === 'cyclic' ? 'C' :
    variable.symmetry === 'dihedral' ? 'D' : '';
  const k = axes.length || variable.rank;
  return `${prefix}${k} on ${axesStr}`;
}

/**
 * Content for the hover tooltip on each row's Einsum cell. Renders:
 *   - Line 1: the full `einsum(...)` formula exactly as the page's main-page
 *     preset list shows it (operand names included).
 *   - Line 2+: one row per distinct operand with its rank and a full
 *     symmetry descriptor (axes / generators spelled out).
 *
 * Designed to sit inside a `group-hover:block` wrapper so the reveal is
 * pure CSS — no state, no Portal, works across every row identically.
 */
function EinsumConstructionTooltip({ preset }) {
  if (!preset) return null;
  const opNames = (preset.expression?.operandNames ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const counts = new Map();
  for (const n of opNames) counts.set(n, (counts.get(n) ?? 0) + 1);
  const seen = new Set();
  const distinctOps = [];
  for (const n of opNames) {
    if (seen.has(n)) continue;
    seen.add(n);
    const v = preset.variables?.find((x) => x.name === n);
    distinctOps.push({ name: n, count: counts.get(n), variable: v });
  }
  return (
    <>
      <div className="mb-2 font-mono text-[12px] font-semibold text-gray-900 break-all">
        {preset.formula}
      </div>
      <div className="border-t border-gray-200 pt-2">
        <div className="mb-1 text-[10.5px] font-semibold uppercase tracking-wider text-gray-500">
          Operands
        </div>
        <div className="space-y-1">
          {distinctOps.map((op) => (
            <div key={op.name} className="flex flex-wrap items-baseline gap-x-1.5 text-[11.5px] leading-5">
              <span className="font-mono font-semibold text-gray-900">{op.name}</span>
              {op.count > 1 && (
                <span className="font-mono text-[10.5px] text-gray-500">×{op.count}</span>
              )}
              <span className="text-gray-400">·</span>
              <span className="text-gray-700">rank {op.variable?.rank ?? '?'}</span>
              <span className="text-gray-400">·</span>
              <span className="font-mono text-gray-800">{detailedSymmetryDescription(op.variable)}</span>
            </div>
          ))}
        </div>
      </div>
      {preset.description && (
        <div className="mt-2 border-t border-gray-200 pt-2 text-[11px] leading-5 text-gray-600">
          {preset.description}
        </div>
      )}
    </>
  );
}

// V/W color palette — same hexes the rest of the page uses so the worked
// examples inside the modal read against the same colour convention.
const COLOR_V = notationColor('v_free');
const COLOR_W = notationColor('w_summed');

const vStyle = { color: COLOR_V, fontWeight: 600 };
const wStyle = { color: COLOR_W, fontWeight: 600 };

/**
 * Per-preset storage-α ledger used by §7.
 *
 * Baseline: `analyzeExample(preset, 3)` followed by
 * `α_storage = Σ over G_pt-orbits O of |π_V(O) / G_pt|_V|`, i.e. the number
 * of distinct (G_pt|_V)-output-slot writes each orbit contributes under a
 * symmetry-aware output store. For presets with trivial $G_{\text{pt}}\big|_{V_{\mathrm{free}}}$,
 * α_storage equals α_engine (no mirrored cells to collapse).
 *
 * Label-size overrides. A few presets declare non-uniform `labelSizes` on
 * their definition in `data/examples.js` — currently `triple-outer`
 * (`{ i: 6, 'a,b,c': 3 }`) and `outer` (`{ 'a,c': 4, 'b,d': 3 }`). The
 * engine applies those overrides to every label's effective size, and the
 * rows below apply them *consistently* to BOTH `ae` and `as`: otherwise
 * the two columns describe different operand shapes and the reported
 * saving becomes meaningless. (An earlier version of this table mixed
 * override-aware `ae` with uniform-n `as` for those two presets; the
 * SymPy audit at `.claude/worktrees/.../audit/check_savings_table.py`
 * caught the discrepancy.)
 *
 * Rows are sorted by savings percentage descending so the nontrivial cases
 * read first and the zero-savings "nothing to mirror" block sits at the end.
 *
 * If the engine's α definition or the preset list changes, regenerate by
 * running a small survey of EXAMPLES through `analyzeExample(preset, 3)`
 * and the audit's `alpha_storage` reference.
 */
const SAVINGS_TABLE_ROWS = [
  { id: 'four-cycle',       v: 'i,j,k,l',        vSub: '\\text{order-}8', ae: 81,  as: 21,  saving: 60,  pct: '74.1' },
  { id: 'bilinear-trace-3', v: 'i,j,m',          vSub: 'S_3',             ae: 516, as: 165, saving: 351, pct: '68.0' },
  { id: 'direct-s3-s2',     v: 'a,b,c',          vSub: 'S_3',             ae: 162, as: 60,  saving: 102, pct: '63.0' },
  { id: 'young-s4-v3w1',    v: 'a,b,c',          vSub: 'S_3',             ae: 81,  as: 30,  saving: 51,  pct: '63.0' },
  { id: 'triple-outer',     v: 'a,b,c',          vSub: 'S_3',             ae: 162, as: 60,  saving: 102, pct: '63.0' },
  { id: 'declared-c3',      v: 'b,i,j,k',        vSub: '\\text{order-}3', ae: 243, as: 99,  saving: 144, pct: '59.3' },
  { id: 'triangle',         v: 'i,j,k',          vSub: 'C_3',             ae: 27,  as: 11,  saving: 16,  pct: '59.3' },
  { id: 'outer',            v: 'a,b,c,d',        vSub: '\\text{order-}2', ae: 144, as: 78,  saving: 66,  pct: '45.8' },
  { id: 'bilinear-trace',   v: 'i,j',            vSub: 'S_2',             ae: 72,  as: 45,  saving: 27,  pct: '37.5' },
  { id: 'direct-s2-c3',     v: 'a,b',            vSub: 'S_2',             ae: 99,  as: 66,  saving: 33,  pct: '33.3' },
  { id: 'four-A-grid',      v: 'a,b',            vSub: 'S_2',             ae: 54,  as: 36,  saving: 18,  pct: '33.3' },
  { id: 'young-s4-v2w2',    v: 'a,b',            vSub: 'S_2',             ae: 54,  as: 36,  saving: 18,  pct: '33.3' },
  { id: 'direct-s2-s2',     v: 'a,b',            vSub: 'S_2',             ae: 54,  as: 36,  saving: 18,  pct: '33.3' },
  { id: 'young-s3',         v: 'a,b',            vSub: 'S_2',             ae: 27,  as: 18,  saving: 9,   pct: '33.3' },
  { id: 'mixed-chain',      v: 'i,l',            vSub: '\\{e\\}',         ae: 81,  as: 81,  saving: 0,   pct: '0' },
  { id: 'matrix-chain',     v: 'i,k',            vSub: '\\{e\\}',         ae: 27,  as: 27,  saving: 0,   pct: '0' },
  { id: 'cross-c3-partial', v: 'a,b',            vSub: '\\{e\\}',         ae: 27,  as: 27,  saving: 0,   pct: '0' },
  { id: 'cross-s2',         v: 'i,k',            vSub: '\\{e\\}',         ae: 27,  as: 27,  saving: 0,   pct: '0' },
  { id: 'cyclic-cross',     v: 'i',              vSub: '\\{e\\}',         ae: 21,  as: 21,  saving: 0,   pct: '0' },
  { id: 'cross-s3',         v: 'i',              vSub: '\\{e\\}',         ae: 18,  as: 18,  saving: 0,   pct: '0' },
  { id: 'frobenius',        v: '\\varnothing',   vSub: '\\{e\\}',         ae: 9,   as: 9,   saving: 0,   pct: '0' },
  { id: 'trace-product',    v: '\\varnothing',   vSub: '\\{e\\}',         ae: 6,   as: 6,   saving: 0,   pct: '0' },
];

/**
 * Appendix modal that walks the reader through the distinction between the
 * two symmetry groups of an einsum:
 *
 *   · the pointwise symmetry group G_pt  — holds at each summand
 *   · the formal symmetry group     G_f  — holds only on the total sum
 *
 * and builds G_f from its two components: the induced permutation group
 * G_pt|_{V_free} and the symmetric group S(W_summed). The modal then contrasts the α
 * that naive Burnside on G_f would claim with the correct G_pt-based α,
 * and discloses the output-tensor savings still available via
 * G_pt|_V-aware storage.
 */
export default function ExpressionLevelModal({ isOpen, onClose, analysis, group }) {
  const vLabels = group?.vLabels ?? [];
  const wLabels = group?.wLabels ?? [];
  const expressionGroup = analysis?.expressionGroup ?? null;
  const exprAlpha = useMemo(
    () => computeExpressionAlphaTotal({ analysis }),
    [analysis],
  );

  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9998] flex items-start justify-center overflow-y-auto bg-black/40 px-4 py-10"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="relative w-full max-w-5xl rounded-lg border border-gray-200 bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="expr-modal-heading"
      >
        <div className="relative px-8 pt-8 pb-6">
          <div className="text-center">
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gray-500">
              Appendix
            </div>
            <h2
              id="expr-modal-heading"
              className="mt-3 font-heading text-[32px] font-semibold leading-tight text-gray-900"
            >
              Expression-level symmetry and output storage
            </h2>
            <p className="mx-auto mt-4 max-w-[72ch] font-serif text-[17px] leading-[1.75] text-gray-700">
              <InlineMathText>
                {`The main page reports the detected pointwise symmetry group that licenses orbit compression. This appendix distinguishes that group from the larger formal symmetry group, explains why Burnside on the formal group overcounts, and shows the storage-aware savings still left on the table.`}
              </InlineMathText>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="absolute right-6 top-6 rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-900"
            aria-label="Close"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path
                fillRule="evenodd"
                d="M4.22 4.22a.75.75 0 011.06 0L10 8.94l4.72-4.72a.75.75 0 111.06 1.06L11.06 10l4.72 4.72a.75.75 0 11-1.06 1.06L10 11.06l-4.72 4.72a.75.75 0 01-1.06-1.06L8.94 10 4.22 5.28a.75.75 0 010-1.06z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        <div className="px-8 pb-10">
          {/* §1 — Definitions */}
          <AppendixSection
            n={1}
            label="The distinction"
            title="The distinction"
            deck="The main page reports the detected symmetry group used for cost compression; this appendix separates that pointwise group from the larger formal symmetry group."
          >

            <div className="max-w-[74ch] space-y-4 font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`The main page's goal is operational: given an einsum $\\sum_t \\text{summand}(t)$, it reports the minimum number of distinct scalar multiplications $\\mu$ and accumulations $\\alpha$ needed to produce the output tensor by reusing work across tuples.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`$G_{\\text{pt}}$ is the pointwise symmetry group acting on individual summands, while $G_{\\text{f}}$ records formal invariance of the total expression after dummy-index renaming.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`Concretely, $G_{\\text{pt}} \\subseteq \\mathrm{Sym}(L)$ consists of label permutations $\\pi$ for which $\\text{summand}(t) = \\text{summand}(\\pi^{-1} t)$ for every tuple $t \\in [n]^L$. It is the largest group that genuinely licenses Burnside-style orbit compression in the enumerate-and-accumulate model, and every main-page cost is computed with respect to it.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`By contrast, $G_{\\text{f}} \\subseteq \\mathrm{Sym}(V_{\\mathrm{free}}) \\times \\mathrm{Sym}(W_{\\mathrm{summed}})$ captures invariance of the output tensor viewed as a formal polynomial. Its extra $W_{\\mathrm{summed}}$ permutations are dummy-index renamings: they preserve the total sum after aggregation, but they do not identify equal summands tuple-by-tuple.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`These groups are related through $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the $V_{\\mathrm{free}}$-action induced by the $V_{\\mathrm{free}}$-preserving elements of the detected group. That induced action becomes the $V_{\\mathrm{free}}$-factor of $G_{\\text{f}}$, while the row-unwitnessed dummy renamings supply the $S(W_{\\mathrm{summed}})$ factor.`}
                </InlineMathText>
              </p>
            </div>
          </AppendixSection>

          {/* §2 — 4-preset gallery + detection boundary */}
          <AppendixSection
            n={2}
            label="The row-level detection boundary"
            title="When can the σ-loop see a symmetry, and when can't it?"
          >

            <div className="mb-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`The two symmetry groups we just defined — the pointwise $G_{\\text{pt}}$ and the formal $G_{\\text{f}}$ — differ on every einsum where the σ-loop's wreath elements either collapse to identity $\\pi$ under dedup or get rejected by $\\texttt{derivePi}$. This section makes the mechanism visible on four concrete presets, then states the exact rule for when $G_{\\text{pt}} = \\{e\\}$ and why.`}
              </InlineMathText>
            </div>

            <div className="mb-4 overflow-x-auto">
              <table className="w-full text-[12px] border-collapse">
                <thead>
                  <tr className="border-b border-border/60 text-left text-muted-foreground">
                    <th className="px-2 py-2 font-semibold">Preset</th>
                    <th className="px-2 py-2 font-semibold"><Latex math="L, V, W" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="H_A, m_A" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="|G_{\text{wreath}}|" /></th>
                    <th className="px-2 py-2 font-semibold">valid</th>
                    <th className="px-2 py-2 font-semibold">matrix-preserving</th>
                    <th className="px-2 py-2 font-semibold">rejected</th>
                    <th className="px-2 py-2 font-semibold"><Latex math="|G_{\text{pt}}|" /></th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2 font-mono">frobenius</td>
                    <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                    <td className="px-2 py-2 font-mono">2</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">0</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">2</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">0</td>
                    <td className="px-2 py-2 font-mono">1</td>
                  </tr>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2 font-mono">trace-product</td>
                    <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                    <td className="px-2 py-2 font-mono">2</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">1</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">1</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">0</td>
                    <td className="px-2 py-2 font-mono">2</td>
                  </tr>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2 font-mono">triangle</td>
                    <td className="px-2 py-2"><Latex math="\{i,j,k\}, \{i,j,k\}, \varnothing" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 3" /></td>
                    <td className="px-2 py-2 font-mono">6</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">2</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">1</td>
                    <td className="px-2 py-2 font-mono text-amber-600/80">3</td>
                    <td className="px-2 py-2 font-mono">3</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-2 font-mono">young-s3</td>
                    <td className="px-2 py-2"><Latex math="\{a,b,c\}, \{a,b\}, \{c\}" /></td>
                    <td className="px-2 py-2"><Latex math="S_3, 1" /></td>
                    <td className="px-2 py-2 font-mono">6</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">5</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">1</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">0</td>
                    <td className="px-2 py-2 font-mono">6</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <ul className="mb-4 list-none space-y-1.5 text-[12px] leading-5 text-foreground">
              <li><span className="font-mono font-semibold">frobenius</span> — <em>class 1 only</em>: wreath non-trivial, π-dedup collapses everything to identity.</li>
              <li><span className="font-mono font-semibold">trace-product</span> — <em>classes 1 + 2</em>: the two A's have different subscripts, so the operand swap induces <InlineMathText>{`$\\pi = (i\\;j)$`}</InlineMathText>; G_pt gains it.</li>
              <li><span className="font-mono font-semibold">triangle</span> — <em>all three classes visible</em>: the three adjacent copy-transpositions are rejected by derivePi (their column fingerprints don't match). Smallest preset exhibiting the rejected class.</li>
              <li><span className="font-mono font-semibold">young-s3</span> — <em>class 2 dominates</em>: only base-group generators (declared <InlineMathText>{`$S_3$`}</InlineMathText> on T's axes); every non-identity wreath element is valid.</li>
            </ul>

            <div className="mb-4 rounded-md border-l-4 border-border/60 bg-muted/20 px-4 py-3 text-[12px] leading-6 text-foreground">
              <p className="font-semibold mb-1">Frobenius</p>
              <p>
                <InlineMathText>
                  {`$M$ is invariant elementwise under the operand swap (both A's have subscripts $(i, j)$); $\\texttt{derivePi}(\\text{swap})$ returns identity. $G_{\\text{pt}} = \\{e\\}$. The $(i\\;j)$ dummy-renaming is **row-unwitnessed** and lives in $S(W_{\\mathrm{summed}})$.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mb-4 rounded-md border-l-4 border-border/60 bg-muted/20 px-4 py-3 text-[12px] leading-6 text-foreground">
              <p className="font-semibold mb-1">Triangle</p>
              <p>
                <InlineMathText>
                  {`Each adjacent copy-transposition changes the set of U-vertex rows, but the column fingerprints don't form a bijection with the original's (a missing cyclic rotation would be required); $\\texttt{derivePi}$ returns $\\texttt{null}$. Only the two 3-cycles produce valid $\\pi$'s. $G_{\\text{pt}} = C_3$.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mb-4 rounded-md border border-primary/40 bg-primary/5 px-5 py-4 text-[13px] leading-7 text-foreground">
              <AppendixTheoremBlock
                kind="Proposition"
                lead="Frobenius-class presets."
              >
                {`Let an einsum satisfy: (1) No operand has declared axis symmetry ($H_i = \\{e\\}$ for every identical-operand group i); (2) Every identical-operand group of size $\\geq 2$ consists of copies with identical subscript strings. Then for generic operands, $G_{\\text{pt}} = \\{e\\}$ and $G_{\\text{f}} = S(W_{\\mathrm{summed}})$ — every formal symmetry beyond identity is row-unwitnessed and lives in the $S(W_{\\mathrm{summed}})$ factor of the §5 construction.`}
              </AppendixTheoremBlock>
              <p className="mb-3">
                <InlineMathText>
                  {`*Proof.* The wreath $\\prod_i (H_i \\wr S_{m_i})$ reduces under (1) to $\\prod_i S_{m_i}$ (pure copy-permutations). A copy-permutation $\\sigma$ within group $i$ sends U-vertex $(i, j, \\text{axis } k)$ to $(i, \\sigma_i(j), k)$. Under (2), every copy in group $i$ has the same axis-to-label mapping, so the permuted U-vertex carries the same label as the original. Hence $M_\\sigma = M$ elementwise. By Theorem (b) below, $\\texttt{derivePi}(\\sigma) = \\text{identity}$. No non-identity $\\pi$ contributions exist, so $G_{\\text{pt}} = \\{e\\}$. Then $G_{\\text{f}} = G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}} \\times S(W_{\\mathrm{summed}}) = \\{e\\} \\times S(W_{\\mathrm{summed}}) = S(W_{\\mathrm{summed}})$. ∎`}
                </InlineMathText>
              </p>
              <AppendixTheoremBlock
                kind="Theorem"
                lead="Row-level detection boundary."
              >
                {`Let $G_{\\text{wreath}}$ denote the σ-loop's row-permutation group — namely $\\prod_i (H_i \\wr S_{m_i})$ per the wreath-equivalence result.`}
              </AppendixTheoremBlock>
              <p className="mb-2 pl-4">
                <em>
                  <InlineMathText>
                    {`(a) The σ-loop iterates over every $\\sigma \\in G_{\\text{wreath}}$ and applies $\\texttt{derivePi}$. A given $\\sigma$ yields a valid label permutation iff $\\sigma$'s column fingerprints bijectively match the original's; some wreath elements pass (e.g. triangle's two 3-cycles), others fail (e.g. triangle's three adjacent transpositions) and are rejected.`}
                  </InlineMathText>
                </em>
              </p>
              <p className="mb-2 pl-4">
                <em>
                  <InlineMathText>
                    {`(b) For any $\\sigma \\in G_{\\text{wreath}}$ that passes $\\texttt{derivePi}$, the derived $\\pi$ is the identity iff $M_\\sigma = M$ elementwise — i.e. $\\sigma$ preserves the incidence matrix row-by-row with no column relabelling needed.`}
                  </InlineMathText>
                </em>
              </p>
              <p className="mb-3 pl-4">
                <em>
                  <InlineMathText>
                    {`(c) The non-identity valid $\\pi$'s — $\\{ \\texttt{derivePi}(\\sigma) : \\sigma \\in G_{\\text{wreath}}, \\texttt{derivePi}(\\sigma) \\neq \\texttt{null}, \\texttt{derivePi}(\\sigma) \\neq e \\}$ — form a subgroup of $\\mathrm{Sym}(L)$ which is exactly $G_{\\text{pt}}$. (The engine closes them via Dimino for robustness, but the set is already composition-closed.)`}
                  </InlineMathText>
                </em>
              </p>
              <AppendixTheoremBlock kind="Corollary">
                {`If every $\\sigma \\in G_{\\text{wreath}}$ preserves $M$ elementwise, then $G_{\\text{pt}} = \\{e\\}$.`}
              </AppendixTheoremBlock>
            </div>
          </AppendixSection>

          {/* §3 — Induced permutation group on V */}
          <AppendixSection
            n={3}
            label="First component"
            title={`The induced permutation group G_pt|V_free`}
          >

            <p className="mb-3 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Under the wreath framing of Section 3, $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ is the image under $\\texttt{derivePi}$ of the $V_{\\mathrm{free}}$-preserving wreath elements, restricted to $V_{\\mathrm{free}}$. These are the $V_{\\mathrm{free}}$-actions that are **row-witnessed** — discoverable by inspecting the row-permutation structure of the bipartite graph.`}
              </InlineMathText>
            </p>

            <div className="max-w-[74ch] space-y-4 font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`$G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ is the image of $G_{\\text{pt}}$ under restriction to the $V_{\\mathrm{free}}$ labels. Concretely, let $\\mathrm{Stab}_{G_{\\text{pt}}}(V_{\\mathrm{free}})$ be the subgroup whose elements preserve $V_{\\mathrm{free}}$ setwise, then deduplicate their actions on the free positions to obtain a subgroup of $\\mathrm{Sym}(V_{\\mathrm{free}})$.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`This induced group is written $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$. For every $\\sigma$ in it, the output tensor satisfies $R[\\sigma\\,\\omega] = R[\\omega]$ for all $\\omega \\in [n]^{V_{\\mathrm{free}}}$, so the symmetry is genuine on the computed output tensor itself and not merely on the formal sum.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace at <Latex math="n = 2" />.</p>
              <p className="mt-2">
                <InlineMathText>
                  {`The einsum $\\mathtt{ik{,}jl\\to ij}$ computes $R[i,j] = \\sum_{k,l} A[i,k] \\cdot A[j,l]$ with $V_{\\mathrm{free}} = \\{i, j\\}$ and $W_{\\mathrm{summed}} = \\{k, l\\}$. The σ-loop's top-group transposition emits the permutation that swaps the two identical $A$ operands, exchanging $i \\leftrightarrow j$ together with $k \\leftrightarrow l$. The detected pointwise group is therefore $G_{\\text{pt}} = \\{e,\\;(i\\;j)(k\\;l)\\}$. Restricting each element to $V_{\\mathrm{free}}$ yields $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}} = \\{e,\\;(i\\;j)\\}$, a copy of $S_2$ acting on $\\{i,j\\}$.`}
                </InlineMathText>
              </p>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Using the same $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$ as in Section 2, each output cell expands as a sum of four products:`}
              </InlineMathText>
            </p>

            <div className="mt-2 rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
              <div>
                R[<span style={vStyle}>0</span>,<span style={vStyle}>0</span>]
                {' = '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]
                {' = '}
                1 + 2 + 2 + 4 = <strong>9</strong>
              </div>
              <div className="mt-1">
                R[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>]
                {' = '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                {' = '}
                3 + 4 + 6 + 8 = <strong>21</strong>
              </div>
              <div className="mt-1">
                R[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>]
                {' = '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]
                {' = '}
                3 + 6 + 4 + 8 = <strong>21</strong>
              </div>
              <div className="mt-1">
                R[<span style={vStyle}>1</span>,<span style={vStyle}>1</span>]
                {' = '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]
                {' + '}
                A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                {' = '}
                9 + 12 + 12 + 16 = <strong>49</strong>
              </div>
            </div>

            <div className="mt-4 rounded-md border-l-4 border-emerald-500 bg-emerald-50 px-5 py-3 text-sm leading-7 text-emerald-900">
              <InlineMathText>
                {`The two off-diagonal cells agree: $R[0,1] = R[1,0] = 21$, and the agreement is term-by-term — each product in one expansion is the commuted twin of a product in the other. The equality therefore holds on the computed output tensor itself, not merely on a total: $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\sigma \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$.`}
              </InlineMathText>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-4 py-3 text-[13px] leading-6 text-muted-foreground">
              <InlineMathText>
                {`Algebraically, $R[i,j] = (\\sum_k A[i,k])(\\sum_l A[j,l]) = v_i\\,v_j$ with $v = \\mathrm{rowsum}(A)$; the outer product $v\\,v^\\top$ is symmetric by construction, so $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}} = \\{e,(i\\;j)\\}$ acts trivially on $R$ for every $A$.`}
              </InlineMathText>
            </div>
          </AppendixSection>

          {/* §4 — S(W_summed) */}
          <AppendixSection
            n={4}
            label="Second component"
            title="The symmetric group on summed labels"
          >

            <p className="mb-3 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Unlike $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the $S(W_{\\mathrm{summed}})$ factor is **row-unwitnessed**: no $\\sigma$ in the wreath produces these $W_{\\mathrm{summed}}$-permutations as its induced $\\pi$. They are formal symmetries by virtue of being permutations of bound summation indices — renaming dummies leaves any sum invariant. This is the only factor of $G_{\\text{f}}$ that is strictly non-row-visible.`}
              </InlineMathText>
            </p>

            <div className="max-w-[74ch] space-y-4 font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`$S(W_{\\mathrm{summed}})$ is the full symmetric group on the summed labels $W_{\\mathrm{summed}}$. Its size is $|W_{\\mathrm{summed}}|!$, and unlike $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ it depends only on how many summed labels there are, not on operand structure or declared symmetries.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`Every permutation of $W_{\\mathrm{summed}}$ is a formal symmetry because summed labels are dummy variables: relabelling them consistently across all operands yields the same total sum. That invariance is syntactic rather than pointwise, since individual summands at two permuted tuples need not agree.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace (continued).</p>
              <p className="mt-1">
                <InlineMathText>
                  {`With $W_{\\mathrm{summed}} = \\{k,l\\}$, $S(W_{\\mathrm{summed}}) = \\{e,\\;(k\\;l)\\}$. The permutation $(k\\;l)$ is a formal symmetry because $\\sum_{k,l} A[i,k]\\,A[j,l] = \\sum_{k,l} A[i,l]\\,A[j,k]$ — the two double sums iterate over the same set of index pairs and differ only in which variable is named $k$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`However, $(k\\;l)$ is not pointwise. Fix $(i,j) = (0,1)$ and compare the individual summands at $(k,l) = (0,1)$ and its image $(1,0)$: one is $A[0,0] \\cdot A[1,1]$ and the other is $A[0,1] \\cdot A[1,0]$. These expressions differ for a generic $A$, so applying $(k\\;l)$ to Burnside's orbit formula would yield a compression claim that does not match the true output.`}
                </InlineMathText>
              </p>
            </div>
          </AppendixSection>

          {/* §5 — Assemble G_f */}
          <AppendixSection
            n={5}
            label="How the formal group is built"
            title="How the formal group is built"
          >
            <div className="mb-3 text-sm text-muted-foreground">
                <Latex math="= (\text{row-witnessed V-action}) \times (\text{row-unwitnessed W-permutation})" />
            </div>

            <div className="max-w-[74ch] space-y-4 font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`Each pair $(\\sigma_{V_{\\mathrm{free}}},\\;\\sigma_{W_{\\mathrm{summed}}})$ with $\\sigma_{V_{\\mathrm{free}}} \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ and $\\sigma_{W_{\\mathrm{summed}}} \\in S(W_{\\mathrm{summed}})$ lifts to a single label permutation on $V_{\\mathrm{free}} \\cup W_{\\mathrm{summed}}$ that acts separately on the free and summed positions.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`The set of all such lifts is $G_{\\text{f}}$, with order $|G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}| \\cdot |W_{\\mathrm{summed}}|!$. This is the decomposition promised by the appendix: every $V_{\\mathrm{free}}$-preserving polynomial symmetry splits into a row-witnessed action on output labels and a row-unwitnessed permutation of dummy labels.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`No additional closure step is needed at this stage. $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ has already been determined from the detected pointwise group, $S(W_{\\mathrm{summed}})$ depends only on the number of summed labels, and $G_{\\text{f}}$ is their direct product.`}
                </InlineMathText>
              </p>
            </div>

            <p className="mt-5 text-sm leading-7 text-foreground">
              The widget below enumerates these pairs for the currently selected preset. Each row in the rightmost column corresponds to the pair{' '}
              <span className="font-mono text-[12px]">G<sub>pt</sub>|<sub>V_free</sub> row i × S(W_summed) row j</span>.
              Hovering a row in any column highlights the corresponding rows in the other two columns.
            </p>
            <div className="mt-3">
              <VSubSwConstruction
                expressionGroup={expressionGroup}
                vLabels={vLabels}
                wLabels={wLabels}
              />
            </div>
          </AppendixSection>

          {/* §6 — Why G_f is not used for compression */}
          <AppendixSection
            n={6}
            label="Why Burnside on the formal group overcounts"
            title="Why Burnside on the formal group overcounts"
          >

            {exprAlpha !== null ? (
              <>
                <div className="rounded-md border-l-4 border-amber-500 bg-amber-50 px-5 py-4 text-sm leading-7 text-amber-900">
                  <p>
                    <InlineMathText>
                      {`Applying Burnside's orbit-counting formula to $G_{\\text{f}}$ in place of $G_{\\text{pt}}$ would yield $\\alpha =$`}
                    </InlineMathText>{' '}
                    <strong>{exprAlpha}</strong>.
                  </p>
                  <p className="mt-2 text-[13px] text-amber-800">
                    <InlineMathText>
                      {`This value is not correct as a compression count. Orbits under $S(W_{\\mathrm{summed}})$ contain tuples whose summand values differ — because $S(W_{\\mathrm{summed}})$ is row-unwitnessed, its orbits don't correspond to any row-level symmetry of the summand, only to the bound-variable renaming of the total (Section 4 above). The main-page cost card therefore reports $\\alpha$ with respect to $G_{\\text{pt}}$ only.`}
                    </InlineMathText>
                  </p>
                </div>
                <div className="mt-4 max-w-[74ch] font-serif text-[17px] leading-[1.75] text-gray-700">
                  <p>
                    <InlineMathText>
                      {`$G_{\\text{pt}}$ is the largest group under which every orbit's summand values are genuinely equal, so Burnside's "one representative per orbit" principle is faithful there. Any larger group collapses tuples whose representatives are only formally related, producing a compression claim that does not match the actual output tensor.`}
                    </InlineMathText>
                  </p>
                </div>
              </>
            ) : (
              <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
                <InlineMathText>
                  {`For this einsum $G_{\\text{f}}$ coincides with $G_{\\text{pt}}$ (either $|W_{\\mathrm{summed}}| \\leq 1$ or the induced permutation group on $V_{\\mathrm{free}}$ is trivial), so the two counts agree and no contrast is available to display.`}
                </InlineMathText>
              </div>
            )}
          </AppendixSection>

          {/* §7 — Leftover savings via G_pt|_V-aware storage */}
          <AppendixSection
            n={7}
            label="Storage-aware savings"
            title="Storage-aware savings"
          >

            {/*
              Bridging paragraph. Without this the reader hits
              "G_pt|V-aware storage" immediately after six sections spent
              arguing that "G_f's extras beyond G_pt admit no pointwise
              compression", and the apparent contradiction lands before
              the axis-pivot is named. Three beats:
                1. Close the accumulation-count story from §6.
                2. Name the new axis (output-tensor storage).
                3. Reframe G_pt|V's role: it appeared in §5 as the
                   V-factor of G_f, but equivalently it is a *subgroup
                   of G_pt itself*. That dual role is the reason R is
                   genuinely pointwise-symmetric under G_pt|V — and
                   therefore why storage collapse is a legitimate
                   (not formal-only) win here.
            */}
            <p className="mb-4 text-sm leading-7 text-gray-700">
              <InlineMathText>
                {`Section 6 closed the accumulation-count story: $G_{\\text{f}}$'s extra elements beyond $G_{\\text{pt}}$ admit no pointwise compression of $\\alpha$ under the enumerate-and-accumulate evaluation model. Before closing the appendix we pivot to a *different* optimization axis — output-tensor storage — where savings *are* available. The group governing this axis is $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$: it appeared in Section 5 as the $V_{\\mathrm{free}}$-factor of $G_{\\text{f}}$, but equivalently it is a subgroup of $G_{\\text{pt}}$ itself — the $V_{\\mathrm{free}}$-action that $G_{\\text{pt}}$'s $V_{\\mathrm{free}}/W_{\\mathrm{summed}}$-preserving elements already induce. That dual role is precisely why the output tensor is genuinely *pointwise*-symmetric under it, and therefore why the savings discussed below are not formal-only.`}
              </InlineMathText>
            </p>

            <div className="max-w-[74ch] space-y-4 font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`For every $\\sigma \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the identity $R[\\sigma\\,\\omega] = R[\\omega]$ holds on the output tensor cell by cell. For generic operands at large enough $n$, this inclusion is tight: $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ is the complete structural $V_{\\mathrm{free}}$ symmetry of $R$.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`At small $n$, the index space can degenerate and create coincidental free-label symmetries that disappear once larger label sizes separate the relevant orbits. A symmetry-aware output store can exploit the persistent $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ orbits by mapping each orbit of output cells to one physical slot and thereby reducing mirrored writes.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  {`The reported $\\alpha$ does not include that win because it counts distinct accumulation operations in the enumerate-and-accumulate model using $G_{\\text{pt}}$ as the summand-value equivalence relation. Post-accumulation storage collapse is a separate optimization axis, so folding it into $\\alpha$ would blur two different sources of savings.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Magnitude of the gap, across every preset in the explorer.</p>
              <p className="mt-2">
                <InlineMathText>
                  {`Per $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$-orbit of size $s$, the savings are $(s-1) \\cdot (\\text{accumulations per bin})$ operations, so the total gap scales with the order and orbit structure of $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$. Presets with trivial $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ carry no output-tensor mirroring — there is nothing to collapse, and $\\alpha_{\\text{storage}} = \\alpha_{\\text{engine}}$.`}
                </InlineMathText>
              </p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-[12px] border-collapse">
                  <thead>
                    <tr className="border-b border-border/60 text-left text-[12px] text-muted-foreground">
                      <th className="px-2 py-2 font-semibold">Preset</th>
                      <th className="px-2 py-2 font-semibold">Einsum</th>
                      <th className="px-2 py-2 font-semibold">Operand sym.</th>
                      <th className="px-2 py-2 font-semibold">V</th>
                      <th className="px-2 py-2 font-semibold"><Latex math="G_{\text{pt}}\big|_{V_{\mathrm{free}}}" /></th>
                      <th className="px-2 py-2 font-semibold text-right"><Latex math="\alpha_{\text{engine}}" /></th>
                      <th className="px-2 py-2 font-semibold text-right"><Latex math="\alpha_{\text{storage}}" /></th>
                      <th className="px-2 py-2 font-semibold text-right">Saving</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SAVINGS_TABLE_ROWS.map((r, idx) => {
                      const isLast = idx === SAVINGS_TABLE_ROWS.length - 1;
                      const hasSaving = r.saving > 0;
                      const preset = EXAMPLES_BY_ID.get(r.id);
                      const subs = preset?.expression?.subscripts ?? '';
                      const output = preset?.expression?.output ?? '';
                      const operands = describeOperands(preset);
                      return (
                        <tr
                          key={r.id}
                          className={`${isLast ? '' : 'border-b border-border/40'} ${hasSaving ? '' : 'text-muted-foreground'}`}
                        >
                          <td className="px-2 py-2 font-mono whitespace-nowrap">{r.id}</td>
                          <td className="px-2 py-2 font-mono whitespace-nowrap">
                            {/*
                              Pure-CSS hover reveal: the inner `group` wraps both
                              the cell's visible einsum text and an absolutely
                              positioned tooltip. `group-hover:block` flips the
                              tooltip in when the cursor enters the wrapper; a
                              dotted underline on the einsum surface signals the
                              affordance. z-50 keeps it above sibling rows, and
                              `pointer-events-none` prevents the panel from
                              trapping the cursor inside the cell (important so
                              moving to the next row dismisses the current
                              tooltip cleanly).
                            */}
                            <div className="group relative inline-block">
                              <span className="cursor-help border-b border-dotted border-gray-300">
                                {subs.replace(/,/g, ', ')}
                                <span className="mx-1">→</span>
                                {output || <Latex math="\varnothing" />}
                              </span>
                              <div className="pointer-events-none absolute left-0 top-full z-50 mt-1.5 hidden w-[380px] whitespace-normal rounded-lg border border-gray-300 bg-white p-3 text-[12px] leading-5 text-gray-700 shadow-xl group-hover:block">
                                <EinsumConstructionTooltip preset={preset} />
                              </div>
                            </div>
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap">
                            {operands.map((o, i) => (
                              <span key={o.name}>
                                {i > 0 && <span className="text-muted-foreground">; </span>}
                                <span className="font-mono font-semibold">{o.name}</span>
                                {o.count > 1 && (
                                  <span className="font-mono text-muted-foreground">×{o.count}</span>
                                )}
                                <span className="text-muted-foreground">: </span>
                                <span className={o.sym === 'dense' ? 'text-muted-foreground italic' : 'font-mono'}>
                                  {o.sym}
                                </span>
                              </span>
                            ))}
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap">
                            <Latex math={r.v === '\\varnothing' ? '\\varnothing' : `\\{${r.v}\\}`} />
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap"><Latex math={r.vSub} /></td>
                          <td className="px-2 py-2 text-right font-mono">{r.ae}</td>
                          <td className="px-2 py-2 text-right font-mono">{r.as}</td>
                          <td className={`px-2 py-2 text-right font-mono whitespace-nowrap ${hasSaving ? 'text-emerald-700' : ''}`}>
                            {hasSaving ? `${r.saving} (${r.pct}%)` : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="mt-2 text-[11px] italic text-muted-foreground">
                  All entries computed at <Latex math="n = 3" />; sorted by % saving, descending. <span className="font-mono not-italic">×k</span> on an operand indicates it appears <Latex math="k" /> times in the expression (driving top-group transpositions in the wreath σ-loop). Hover any einsum to see the full construction with per-operand ranks, declared axes, and generators.
                </p>
              </div>
              <p className="mt-3 text-[13px] text-muted-foreground">
                <InlineMathText>
                  {`The $S(W_{\\mathrm{summed}})$ factor of $G_{\\text{f}}$ contributes nothing at the storage level because $S(W_{\\mathrm{summed}})$ acts on summation variables rather than output cells; there is no output-tensor symmetry to exploit on the $W_{\\mathrm{summed}}$-side beyond what $G_{\\text{pt}}$ already captures through its orbit structure on tuples.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4 max-w-[74ch] font-serif text-[17px] leading-[1.75] text-gray-700">
              <p>
                <InlineMathText>
                  {`The $\\alpha$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with $G_{\\text{pt}}$ as the equivalence relation on summand values. Output-tensor storage collapse, algebraic restructuring such as factoring $R = v\\,v^\\top$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured here.`}
                </InlineMathText>
              </p>
            </div>
          </AppendixSection>
        </div>
      </div>
    </div>
  );
}
