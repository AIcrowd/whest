import { useEffect, useMemo } from 'react';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import NarrativeCallout from './NarrativeCallout.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';
import { EXAMPLES } from '../data/examples.js';

// Lookup map keyed by preset id so §6's savings table can pull the raw
// einsum and per-operand symmetry declarations straight from the source
// of truth, rather than duplicating that data into the row array below.
const EXAMPLES_BY_ID = new Map(EXAMPLES.map((ex) => [ex.id, ex]));

/**
 * Compact human-readable descriptor for a single operand's declared axis
 * symmetry. Returns:
 *   - 'none'            — no declared symmetry
 *   - 'symmetric'       — full symmetric across all axes
 *   - 'sym(0,1,2)'      — symmetric restricted to listed axes
 *   - 'cyclic'          — full cyclic across all axes
 *   - 'cyclic(1,2,3)'   — cyclic restricted to listed axes
 *   - '⟨(0 1), (2 3)⟩'  — user-declared generators
 */
function symDescriptor(variable) {
  if (!variable || variable.symmetry === 'none') return 'none';
  const axes = variable.symAxes;
  const axesStr = axes?.join(',') ?? '';
  const fullRank = axes && axes.length === variable.rank;
  if (variable.symmetry === 'symmetric') return fullRank ? 'symmetric' : `sym(${axesStr})`;
  if (variable.symmetry === 'cyclic') return fullRank ? 'cyclic' : `cyclic(${axesStr})`;
  if (variable.symmetry === 'custom') return `⟨${(variable.generators || '').trim()}⟩`;
  return variable.symmetry;
}

/**
 * Per-preset operand listing: distinct operand names in first-appearance
 * order, annotated with their repeat count (drives Source B of the σ-loop)
 * and declared axis symmetry (Source A). Used to render the "Operand sym"
 * column in §6's savings table.
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
      sym: symDescriptor(v),
    });
  }
  return rows;
}

// V/W color palette — same hexes the rest of the page uses so the worked
// examples inside the modal read against the same colour convention.
const COLOR_V = '#4A7CFF';
const COLOR_W = '#64748B';

const vStyle = { color: COLOR_V, fontWeight: 600 };
const wStyle = { color: COLOR_W, fontWeight: 600 };

/**
 * Per-preset storage-α ledger used by §6.
 *
 * All measurements taken at n = 3 by `analyzeExample(preset, 3)` followed by
 * `α_storage = Σ over G_pt-orbits O of |π_V(O) / G_pt|_V|`, i.e. the number
 * of distinct (G_pt|_V)-output-slot writes each orbit contributes under a
 * symmetry-aware output store. For presets with trivial $G_{\text{pt}}\big|_V$,
 * α_storage equals α_engine (no mirrored cells to collapse).
 *
 * Rows are sorted by savings percentage descending so the nontrivial cases
 * read first and the zero-savings "nothing to mirror" block sits at the end.
 *
 * If the engine's α definition or the preset list changes, regenerate by
 * running a small survey of EXAMPLES through analyzeExample at n = 3.
 */
const SAVINGS_TABLE_ROWS = [
  { id: 'triple-outer',     v: 'a,b,c',          vSub: 'S_3',             ae: 162, as: 30,  saving: 132, pct: '81.5' },
  { id: 'four-cycle',       v: 'i,j,k,l',        vSub: '\\text{order-}8', ae: 81,  as: 21,  saving: 60,  pct: '74.1' },
  { id: 'outer',            v: 'a,b,c,d',        vSub: '\\text{order-}2', ae: 144, as: 45,  saving: 99,  pct: '68.8' },
  { id: 'bilinear-trace-3', v: 'i,j,m',          vSub: 'S_3',             ae: 516, as: 165, saving: 351, pct: '68.0' },
  { id: 'direct-s3-s2',     v: 'a,b,c',          vSub: 'S_3',             ae: 162, as: 60,  saving: 102, pct: '63.0' },
  { id: 'young-s4-v3w1',    v: 'a,b,c',          vSub: 'S_3',             ae: 81,  as: 30,  saving: 51,  pct: '63.0' },
  { id: 'declared-c3',      v: 'b,i,j,k',        vSub: '\\text{order-}3', ae: 243, as: 99,  saving: 144, pct: '59.3' },
  { id: 'triangle',         v: 'i,j,k',          vSub: 'C_3',             ae: 27,  as: 11,  saving: 16,  pct: '59.3' },
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
 * G_pt|_V and the symmetric group S(W). The modal then contrasts the α
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

  const sectionNumber = 'text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground';
  const sectionTitle = 'font-heading text-base font-semibold text-gray-900';

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
        {/* Header */}
        <div className="flex items-start justify-between border-b border-gray-200 px-6 py-4">
          <div>
            <div className={sectionNumber}>Appendix</div>
            <h2 id="expr-modal-heading" className="font-heading text-lg font-semibold text-gray-900">
              The formal symmetry group: <Latex math="G_{\text{f}} = G_{\text{pt}}\big|_V \times S(W)" />
            </h2>
            <p className="mt-1 text-sm leading-6 text-gray-600">
              <InlineMathText>
                {`Throughout this appendix we write an einsum in the generic form $R = \\sum_t \\text{summand}(t)$, where $t$ ranges over tuples of label values in $[n]^L$ and $\\text{summand}(t) = \\prod_k T_k[s_k(t)]$ is the product of operand values at that tuple. The main page reports a single detected symmetry group $G$ and uses it to drive every cost number. Inside this appendix we refer to that group as $G_{\\text{pt}}$ — the $\\textit{pointwise symmetry group}$ — to distinguish it from the larger $\\textit{formal symmetry group}$ $G_{\\text{f}}$ discussed here. The first section below defines both groups precisely; the sections that follow construct $G_{\\text{f}}$ from its two components.`}
              </InlineMathText>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ml-4 shrink-0 rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-900"
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

        {/* Body */}
        <div className="space-y-10 px-6 py-6">
          {/* §0 — Definitions */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§0 · Preliminaries</div>
              <h3 className={sectionTitle}>Definitions</h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Compression">
                {`The main page's goal. Given an einsum $\\sum_t \\text{summand}(t)$, report the minimum number of distinct scalar multiplications $\\mu$ and the minimum number of distinct accumulations $\\alpha$ needed to produce the output tensor, using symmetry to reuse products and share work across tuples.`}
              </NarrativeCallout>
              <NarrativeCallout label="Pointwise symmetry group" tone="algorithm">
                {`$G_{\\text{pt}}$ — the subgroup of $\\mathrm{Sym}(L)$ consisting of label permutations $\\pi$ for which $\\text{summand}(t) = \\text{summand}(\\pi^{-1} t)$ holds for every tuple $t \\in [n]^L$. The invariance is required at every individual summand ("pointwise" on the tuple space), not merely on the total. $G_{\\text{pt}}$ is the largest group that licenses Burnside-style orbit compression in the enumerate-and-accumulate evaluation model, and every $\\mu$ and $\\alpha$ on the main page is computed with respect to it.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Formal symmetry group">
                {`$G_{\\text{f}}$ — the subgroup of $\\mathrm{Sym}(L)$ consisting of label permutations $\\pi$ for which $\\sum_t \\text{summand}(t) = \\sum_t \\text{summand}(\\pi^{-1} t)$; that is, the total sum is invariant although individual summands may be reshuffled. "Formal" here has its standard mathematical meaning — invariance at the level of the expression treated as a formal sum, not at the level of its values.`}
              </NarrativeCallout>
              <NarrativeCallout label="Relationship" tone="algorithm">
                {`$G_{\\text{pt}} \\subseteq G_{\\text{f}}$ always. The additional elements of $G_{\\text{f}} \\setminus G_{\\text{pt}}$ come from two sources: dummy-variable renamings of the summed labels, and V-only reshuffles that only hold after aggregation. For bilinear-trace (§2 below) $G_{\\text{pt}}$ has 2 elements and $G_{\\text{f}}$ has 4.`}
              </NarrativeCallout>
            </div>
          </section>

          {/* §1 — Frobenius worked example */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§1 · Motivating example</div>
              <h3 className={sectionTitle}>
                The Frobenius trace at <Latex math="n = 2" />
              </h3>
            </div>

            <div className="mb-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Take $R = \\sum_{i,j} A[i,j] \\cdot A[i,j]$ on a generic $2 \\times 2$ matrix $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$. The four summands are:`}
              </InlineMathText>
            </div>

            <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
              <div>
                (<span style={vStyle}>i</span>=0, <span style={vStyle}>j</span>=0):{' '}
                A[<span style={vStyle}>0</span>,<span style={vStyle}>0</span>] ·
                A[<span style={vStyle}>0</span>,<span style={vStyle}>0</span>] = 1 · 1 = 1
              </div>
              <div>
                (<span style={vStyle}>i</span>=0, <span style={vStyle}>j</span>=1):{' '}
                A[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>] ·
                A[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>] = 2 · 2 = 4
              </div>
              <div>
                (<span style={vStyle}>i</span>=1, <span style={vStyle}>j</span>=0):{' '}
                A[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>] ·
                A[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>] = 3 · 3 = 9
              </div>
              <div>
                (<span style={vStyle}>i</span>=1, <span style={vStyle}>j</span>=1):{' '}
                A[<span style={vStyle}>1</span>,<span style={vStyle}>1</span>] ·
                A[<span style={vStyle}>1</span>,<span style={vStyle}>1</span>] = 4 · 4 = 16
              </div>
              <div className="mt-2 border-t border-border/60 pt-2">
                R = 1 + 4 + 9 + 16 = <strong>30</strong>
              </div>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Applying the permutation $(i\\;j)$ — replace each tuple $(i,j)$ with $(j,i)$ — yields:`}
              </InlineMathText>
            </p>

            <div className="mt-2 rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
              <div>
                (0, 0) → (0, 0): A[0,0]² = 1 <span className="text-muted-foreground">(unchanged)</span>
              </div>
              <div>
                (0, 1) → (1, 0): A[1,0]² = <strong className="text-red-700">9</strong>{' '}
                <span className="text-muted-foreground">(was 4)</span>
              </div>
              <div>
                (1, 0) → (0, 1): A[0,1]² = <strong className="text-red-700">4</strong>{' '}
                <span className="text-muted-foreground">(was 9)</span>
              </div>
              <div>
                (1, 1) → (1, 1): A[1,1]² = 16 <span className="text-muted-foreground">(unchanged)</span>
              </div>
              <div className="mt-2 border-t border-border/60 pt-2">
                R' = 1 + 9 + 4 + 16 = <strong>30</strong>
              </div>
            </div>

            <div className="mt-4 rounded-md border-l-4 border-amber-500 bg-amber-50 px-5 py-3 text-sm leading-7 text-amber-900">
              <InlineMathText>
                {`The totals agree: $R = R' = 30$. However, the individual summands at positions 2 and 3 have exchanged values ($4 \\leftrightarrow 9$). The permutation $(i\\;j)$ preserves the sum through reshuffling rather than through term-by-term equality. Hence $(i\\;j) \\in G_{\\text{f}}$ — a formal symmetry — but $(i\\;j) \\notin G_{\\text{pt}}$; a compression scheme that treated it as pointwise would require $A[0,1]^2 = A[1,0]^2$, which fails for a generic $A$.`}
              </InlineMathText>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="Takeaway" tone="accent">
                {`For this einsum $G_{\\text{pt}} = \\{e\\}$ on a generic $A$ (only the identity is pointwise), while $G_{\\text{f}}$ contains the additional element $(i\\;j)$. The remainder of the appendix explains where $G_{\\text{f}}$'s extra elements come from and why they admit no arithmetic reuse.`}
              </NarrativeCallout>
            </div>
          </section>

          {/* §2 — Induced permutation group on V */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§2 · First component</div>
              <h3 className={sectionTitle}>
                The induced permutation group <Latex math="G_{\text{pt}}\big|_V" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$G_{\\text{pt}}\\big|_V$ is the image of $G_{\\text{pt}}$ under restriction to the V-labels. Concretely, let $\\mathrm{Stab}_{G_{\\text{pt}}}(V)$ be the subgroup of $G_{\\text{pt}}$ whose elements preserve $V$ setwise (they permute V-labels among themselves and W-labels among themselves, without crossing). For each $\\pi \\in \\mathrm{Stab}_{G_{\\text{pt}}}(V)$, record its action on V-positions and deduplicate; the resulting set is a subgroup of $\\mathrm{Sym}(V)$, called the $\\textit{induced permutation group on V}$ and written $G_{\\text{pt}}\\big|_V$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Interpretation on the output tensor" tone="algorithm">
                {`For every $\\sigma \\in G_{\\text{pt}}\\big|_V$, the output tensor satisfies $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\omega \\in [n]^V$. This is a genuine symmetry of the computed output tensor itself, not a symbolic invariance of the sum — the cells $R[\\omega]$ and $R[\\sigma\\,\\omega]$ carry identical values.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace at <Latex math="n = 2" />.</p>
              <p className="mt-2">
                <InlineMathText>
                  {`The einsum $\\mathtt{ik{,}jl\\to ij}$ computes $R[i,j] = \\sum_{k,l} A[i,k] \\cdot A[j,l]$ with $V = \\{i, j\\}$ and $W = \\{k, l\\}$. The σ-loop's Source B emits the permutation that swaps the two identical $A$ operands, exchanging $i \\leftrightarrow j$ together with $k \\leftrightarrow l$. The detected pointwise group is therefore $G_{\\text{pt}} = \\{e,\\;(i\\;j)(k\\;l)\\}$. Restricting each element to V yields $G_{\\text{pt}}\\big|_V = \\{e,\\;(i\\;j)\\}$, a copy of $S_2$ acting on $\\{i,j\\}$.`}
                </InlineMathText>
              </p>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              <InlineMathText>
                {`Using the same $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$ as in §1, each output cell expands as a sum of four products:`}
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
                {`The two off-diagonal cells agree: $R[0,1] = R[1,0] = 21$, and the agreement is term-by-term — each product in one expansion is the commuted twin of a product in the other. The equality therefore holds on the computed output tensor itself, not merely on a total: $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\sigma \\in G_{\\text{pt}}\\big|_V$.`}
              </InlineMathText>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-4 py-3 text-[13px] leading-6 text-muted-foreground">
              <InlineMathText>
                {`Algebraically, $R[i,j] = (\\sum_k A[i,k])(\\sum_l A[j,l]) = v_i\\,v_j$ with $v = \\mathrm{rowsum}(A)$; the outer product $v\\,v^\\top$ is symmetric by construction, so $G_{\\text{pt}}\\big|_V = \\{e,(i\\;j)\\}$ acts trivially on $R$ for every $A$.`}
              </InlineMathText>
            </div>
          </section>

          {/* §3 — S(W) */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§3 · Second component</div>
              <h3 className={sectionTitle}>
                The symmetric group on summed labels <Latex math="S(W)" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Definition">
                {`$S(W)$ is the full symmetric group on the summed labels $W$: every permutation of $W$, of which there are $|W|!$ in total. Unlike $G_{\\text{pt}}\\big|_V$, the group $S(W)$ depends only on the cardinality of $W$; it is independent of operand structure or declared symmetries.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why every permutation of W is a formal symmetry" tone="algorithm">
                {`W-labels are bound summation indices. Relabelling them consistently across every operand occurrence yields the identity $\\sum_{k} f(k) = \\sum_{k'} f(k')$ on the total, independent of $f$. The invariance is syntactic — it holds at the level of the formal sum — and it provides no term-level identity, since individual summand values at $k=0$ and $k=1$ need not coincide.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Worked example — bilinear trace (continued).</p>
              <p className="mt-1">
                <InlineMathText>
                  {`With $W = \\{k,l\\}$, $S(W) = \\{e,\\;(k\\;l)\\}$. The permutation $(k\\;l)$ is a formal symmetry because $\\sum_{k,l} A[i,k]\\,A[j,l] = \\sum_{k,l} A[i,l]\\,A[j,k]$ — the two double sums iterate over the same set of index pairs and differ only in which variable is named $k$.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-muted-foreground text-[13px]">
                <InlineMathText>
                  {`However, $(k\\;l)$ is not pointwise. Fix $(i,j) = (0,1)$ and compare the individual summands at $(k,l) = (0,1)$ and its image $(1,0)$: one is $A[0,0] \\cdot A[1,1]$ and the other is $A[0,1] \\cdot A[1,0]$. These expressions differ for a generic $A$, so applying $(k\\;l)$ to Burnside's orbit formula would yield a compression claim that does not match the true output.`}
                </InlineMathText>
              </p>
            </div>
          </section>

          {/* §4 — Assemble G_f */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§4 · Construction of G_f</div>
              <h3 className={sectionTitle}>
                The formal symmetry group:{' '}
                <Latex math="G_{\text{f}} = G_{\text{pt}}\big|_V \times S(W)" />
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="Construction">
                {`Each pair $(\\sigma_V,\\;\\sigma_W)$ with $\\sigma_V \\in G_{\\text{pt}}\\big|_V$ and $\\sigma_W \\in S(W)$ lifts to a single label permutation on $V \\cup W$ that acts as $\\sigma_V$ on V-positions and as $\\sigma_W$ on W-positions. The set of all such lifts forms $G_{\\text{f}}$, a subgroup of $\\mathrm{Sym}(V \\cup W)$ of order $|G_{\\text{pt}}\\big|_V| \\cdot |W|!$.`}
              </NarrativeCallout>
              <NarrativeCallout label="Cost of construction" tone="algorithm">
                {`No Dimino closure is required. $G_{\\text{pt}}\\big|_V$ is already determined by $G_{\\text{pt}}$, and $S(W)$ is an immediate $|W|!$ enumeration of permutations of the summed labels. $G_{\\text{f}}$ is then the on-the-fly Cartesian product.`}
              </NarrativeCallout>
            </div>

            <p className="mt-4 text-sm leading-7 text-foreground">
              The widget below enumerates these pairs for the currently selected preset. Each row in the rightmost column corresponds to the pair{' '}
              <span className="font-mono text-[12px]">G<sub>pt</sub>|<sub>V</sub> row i × S(W) row j</span>.
              Hovering a row in any column highlights the corresponding rows in the other two columns.
            </p>
            <div className="mt-3">
              <VSubSwConstruction
                expressionGroup={expressionGroup}
                vLabels={vLabels}
                wLabels={wLabels}
              />
            </div>
          </section>

          {/* §5 — Why G_f is not used for compression */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§5 · Consequences for compression</div>
              <h3 className={sectionTitle}>
                The accumulation count <Latex math="\alpha" /> under <Latex math="G_{\text{f}}" />
              </h3>
            </div>

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
                      {`This value is not correct as a compression count. Orbits under $S(W)$ contain tuples whose summand values differ (§3 above), so selecting one representative per orbit and multiplying by the orbit size produces a claim that does not match the true output for a generic operand. The main-page cost card therefore reports $\\alpha$ with respect to $G_{\\text{pt}}$ only.`}
                    </InlineMathText>
                  </p>
                </div>
                <div className="mt-4">
                  <NarrativeCallout label="Why G_pt, and not G_f" tone="accent">
                    {`$G_{\\text{pt}}$ is the largest group under which every orbit's summand values are equal; Burnside's "one representative per orbit" principle is faithful there. Any larger group collapses orbits whose representatives Burnside would implicitly assume to be equal when they are not, yielding compression claims that do not match the true output.`}
                  </NarrativeCallout>
                </div>
              </>
            ) : (
              <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm text-muted-foreground">
                <InlineMathText>
                  {`For this einsum $G_{\\text{f}}$ coincides with $G_{\\text{pt}}$ (either $|W| \\leq 1$ or the induced permutation group on V is trivial), so the two counts agree and no contrast is available to display.`}
                </InlineMathText>
              </div>
            )}
          </section>

          {/* §6 — Leftover savings via G_pt|_V-aware storage */}
          <section>
            <div className="mb-2">
              <div className={sectionNumber}>§6 · Remark on output-tensor symmetry</div>
              <h3 className={sectionTitle}>
                Savings accessible under <Latex math="G_{\text{pt}}\big|_V" />-aware storage
              </h3>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <NarrativeCallout label="The open optimization">
                {`For every $\\sigma \\in G_{\\text{pt}}\\big|_V$, the identity $R[\\sigma\\,\\omega] = R[\\omega]$ holds on the output tensor. For generic operands this inclusion is tight: $G_{\\text{pt}}\\big|_V$ is the complete structural V-symmetry of $R$, in the sense that any further $\\sigma \\in \\mathrm{Sym}(V)$ with $R[\\sigma\\,\\omega] = R[\\omega]$ would require value-level structure in the operands (rank-deficiency, sparsity) outside this engine's scope. A computational model with symmetry-aware output storage — where a single physical slot represents all cells in a $G_{\\text{pt}}\\big|_V$-orbit — collapses writes to mirrored cells automatically and reduces the accumulation count.`}
              </NarrativeCallout>
              <NarrativeCallout label="Why α does not fold this in" tone="algorithm">
                {`The reported $\\alpha$ counts distinct accumulation operations in the enumerate-and-accumulate evaluation, using $G_{\\text{pt}}$ as the equivalence relation on summand values. Post-accumulation storage collapse is an independent optimization axis. Folding it into $\\alpha$ without changing the underlying computational model would conflate two distinct cost reductions and obscure the source of each.`}
              </NarrativeCallout>
            </div>

            <div className="mt-4 rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
              <p className="font-semibold">Magnitude of the gap, across every preset in the explorer.</p>
              <p className="mt-2">
                <InlineMathText>
                  {`Per $G_{\\text{pt}}\\big|_V$-orbit of size $s$, the savings are $(s-1) \\cdot (\\text{accumulations per bin})$ operations, so the total gap scales with the order and orbit structure of $G_{\\text{pt}}\\big|_V$. Presets with trivial $G_{\\text{pt}}\\big|_V$ carry no output-tensor mirroring — there is nothing to collapse, and $\\alpha_{\\text{storage}} = \\alpha_{\\text{engine}}$.`}
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
                      <th className="px-2 py-2 font-semibold"><Latex math="G_{\text{pt}}\big|_V" /></th>
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
                            {subs.replace(/,/g, ', ')}
                            <span className="mx-1">→</span>
                            {output || <Latex math="\varnothing" />}
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
                                <span className={o.sym === 'none' ? 'text-muted-foreground italic' : 'font-mono'}>
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
                  All entries computed at <Latex math="n = 3" />; sorted by % saving, descending. <span className="font-mono not-italic">×k</span> on an operand indicates it appears <Latex math="k" /> times in the expression (driving Source B of the σ-loop).
                </p>
              </div>
              <p className="mt-3 text-[13px] text-muted-foreground">
                <InlineMathText>
                  {`The $S(W)$ factor of $G_{\\text{f}}$ contributes nothing at the storage level because $S(W)$ acts on summation variables rather than output cells; there is no output-tensor symmetry to exploit on the W-side beyond what $G_{\\text{pt}}$ already captures through its orbit structure on tuples.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-4">
              <NarrativeCallout label="Scope of the reported α" tone="accent">
                {`The $\\alpha$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model, with $G_{\\text{pt}}$ as the equivalence relation on summand values. Three optimization axes lie outside this scope: $G_{\\text{pt}}\\big|_V$-level output-tensor storage (discussed above), algebraic restructuring such as factoring $R = v\\,v^\\top$, and contraction re-ordering. Each can reduce the total operation count further than $\\alpha$ reports, and each requires algorithmic machinery distinct from the pointwise orbit compression this page measures.`}
              </NarrativeCallout>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
