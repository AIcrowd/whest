import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import AppendixSection from './AppendixSection.jsx';
import AppendixFormalBlock from './AppendixFormalBlock.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';
import { FormulaHighlighted, SymmetryChip } from './StickyBar.jsx';
import { computeExpressionAlphaTotal } from '../engine/comparisonAlpha.js';
import { EXAMPLES } from '../data/examples.js';
import { variableSymmetryLabel } from '../lib/symmetryLabel.js';
import { notationColor, notationLatex } from '../lib/notationSystem.js';

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

function tooltipChipSymmetryLabel(variable, fallback) {
  if (!variable) return fallback;
  if (variable.symmetry !== 'custom') return fallback;
  const generators = (variable.generators ?? '').trim();
  return generators ? `⟨${generators}⟩` : 'custom';
}

function appendixGroupLabel(value) {
  const text = String(value ?? '').trim();
  if (!text) return 'trivial';
  return text
    .replace(/\\text\{([^}]*)\}/g, '$1')
    .replace(/\\varnothing/g, 'varnothing')
    .replace(/\\\{e\\\}/g, 'e')
    .replace(/\s+/g, ' ')
    .replace(/([A-Z])_(\d+)/g, '$1$2')
    .replace(/[{}\\]/g, '')
    .trim();
}

/**
 * Portaled hover card reused across appendix preset-name and einsum
 * surfaces. It intentionally mirrors the top sticky-bar vocabulary:
 * highlighted formula, per-operand symmetry chips, and an output-symmetry
 * badge on the same row.
 */
function EinsumConstructionTooltip({ preset, groupLabel }) {
  if (!preset) return null;
  const variablesByName = new Map((preset.variables ?? []).map((variable) => [variable.name, variable]));
  const operandItems = describeOperands(preset).map((operand) => {
    const variable = variablesByName.get(operand.name);
    return {
      ...operand,
      variable,
      chipSymmetry: tooltipChipSymmetryLabel(variable, operand.sym),
    };
  });

  return (
    <div className="space-y-3">
      <div className="inline-flex max-w-full self-start rounded-xl border border-stone-200 bg-white px-3 py-2.5 shadow-sm">
        <div className="font-mono text-[12px] leading-6 text-stone-900">
          <FormulaHighlighted example={preset} hoveredLabels={null} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[12px] leading-6 text-stone-700">
        {operandItems.map((operand, idx) => (
          <span key={`${operand.name}-${idx}`} className="contents">
            {idx > 0 && <span className="text-stone-400">,</span>}
            <SymmetryChip name={operand.name} symmetry={operand.chipSymmetry} />
          </span>
        ))}
        <span className="text-stone-500">→</span>
        <SymmetryBadge value={groupLabel} className="h-6 px-2.5 text-[11px] leading-5 shadow-none" />
      </div>
      {operandItems.length > 0 ? (
        <div className="border-t border-gray-200 pt-2 text-[11.5px] leading-5 text-gray-700">
          {operandItems.map((operand) => {
            const variable = operand.variable;
            return (
              <div key={`detail-${operand.name}`} className="flex flex-wrap items-baseline gap-x-1.5">
                <span className="font-mono font-semibold text-gray-900">{operand.name}</span>
                <span className="text-gray-400">·</span>
                <span className="text-gray-700">rank {variable?.rank ?? '?'}</span>
                <span className="text-gray-400">·</span>
                <span className="font-mono text-gray-800">{detailedSymmetryDescription(variable)}</span>
              </div>
            );
          })}
        </div>
      ) : null}
      {preset.description && (
        <div className="mt-2 border-t border-gray-200 pt-2 text-[11px] leading-5 text-gray-600">
          {preset.description}
        </div>
      )}
    </div>
  );
}

function vStyle() {
  return { color: notationColor('v_free'), fontWeight: 600 };
}

function wStyle() {
  return { color: notationColor('w_summed'), fontWeight: 600 };
}

const APPENDIX_PROSE_CLASS = 'font-serif text-[17px] leading-[1.75] text-gray-700';
const APPENDIX_PROSE_JUSTIFIED_CLASS = `${APPENDIX_PROSE_CLASS} text-justify`;
const APPENDIX_FORMAL_PROSE_CLASS = 'font-serif text-[17px] leading-[1.85] text-gray-800';
const APPENDIX_KICKER_CLASS = 'text-[11px] font-semibold uppercase tracking-[0.16em] text-gray-500';
const APPENDIX_FOOTNOTE_CLASS = 'text-[11px] italic text-muted-foreground';

function AppendixTwoColBlock({
  left,
  right,
  useLg = false,
  className = '',
}) {
  return (
        <div
          className={[
            useLg
              ? 'editorial-two-col-divider-lg grid gap-y-4 gap-x-8 lg:grid-cols-2'
              : 'editorial-two-col-divider-md grid gap-y-4 gap-x-8 md:grid-cols-2',
            className,
          ].join(' ')}
        >
      <div className="min-w-0">{left}</div>
      <div className="min-w-0">{right}</div>
    </div>
  );
}

const appendixRailClass = 'mx-auto w-full max-w-[var(--content-max)] px-6 md:px-8 lg:px-10';

function AppendixHoverTrigger({ preset, groupLabel, ariaLabel, className, children }) {
  const [open, setOpen] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0, flipped: false });
  const triggerRef = useRef(null);
  const hideTimerRef = useRef(null);

  const clearHideTimer = () => {
    if (hideTimerRef.current) {
      window.clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  };

  const updateTooltipPosition = () => {
    if (!triggerRef.current || typeof document === 'undefined') return;
    const rect = triggerRef.current.getBoundingClientRect();
    const tooltipW = 380;
    const tooltipH = 220;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const x = Math.max(
      tooltipW / 2 + 16,
      Math.min(rect.left + rect.width / 2, vw - tooltipW / 2 - 16),
    );
    const roomAbove = rect.top;
    const roomBelow = vh - rect.bottom;
    const flipped = roomBelow < tooltipH + 16 && roomAbove > roomBelow;
    const y = flipped ? rect.top - 10 : rect.bottom + 10;
    setTooltipPos({ x, y, flipped });
  };

  const openTooltip = () => {
    clearHideTimer();
    updateTooltipPosition();
    setOpen(true);
  };

  const scheduleClose = () => {
    clearHideTimer();
    hideTimerRef.current = window.setTimeout(() => setOpen(false), 80);
  };

  useEffect(() => () => clearHideTimer(), []);

  useEffect(() => {
    if (!open) return undefined;

    const dismiss = () => {
      clearHideTimer();
      setOpen(false);
    };
    const dismissOnEscape = (event) => {
      if (event.key === 'Escape') dismiss();
    };
    const dismissIfOutside = (event) => {
      if (!triggerRef.current) return dismiss();
      if (event.target instanceof Node && triggerRef.current.contains(event.target)) return;
      dismiss();
    };

    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    window.addEventListener('keydown', dismissOnEscape);
    window.addEventListener('pointerdown', dismissIfOutside);
    window.addEventListener('blur', dismiss);

    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('keydown', dismissOnEscape);
      window.removeEventListener('pointerdown', dismissIfOutside);
      window.removeEventListener('blur', dismiss);
    };
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={ariaLabel}
        onMouseEnter={openTooltip}
        onMouseLeave={scheduleClose}
        onFocus={openTooltip}
        onBlur={scheduleClose}
        className={`appearance-none border-0 bg-transparent p-0 align-baseline ${className}`}
      >
        {children}
      </button>
      {open && typeof document !== 'undefined'
        ? createPortal(
            <div
              className="pointer-events-none fixed z-[9999] w-[380px] max-w-[calc(100vw-2rem)] whitespace-normal rounded-lg border border-gray-300 bg-white p-3 text-[12px] leading-5 text-gray-700 shadow-xl"
              style={{
                left: tooltipPos.x,
                top: tooltipPos.y,
                transform: tooltipPos.flipped
                  ? 'translateX(-50%) translateY(-100%)'
                  : 'translateX(-50%)',
              }}
            >
              <EinsumConstructionTooltip preset={preset} groupLabel={groupLabel} />
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

function AppendixEinsumHoverCell({ subs, output, preset, groupLabel }) {
  return (
    <AppendixHoverTrigger
      preset={preset}
      groupLabel={groupLabel}
      ariaLabel={`Show einsum construction for ${preset?.id ?? 'preset'}`}
      className="inline-flex items-center cursor-help border-b border-dotted border-gray-300 text-left"
    >
      {subs.replace(/,/g, ', ')}
      <span className="mx-1">→</span>
      {output || <Latex math="\varnothing" />}
    </AppendixHoverTrigger>
  );
}

function AppendixPresetHoverLabel({ preset, groupLabel, children = null, className = 'font-mono font-semibold cursor-help border-b border-dotted border-gray-300 text-left' }) {
  return (
    <AppendixHoverTrigger
      preset={preset}
      groupLabel={groupLabel}
      ariaLabel={`Show appendix preset details for ${preset?.id ?? 'preset'}`}
      className={className}
    >
      {children ?? preset?.id ?? 'preset'}
    </AppendixHoverTrigger>
  );
}

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
        className="relative w-full max-w-[var(--content-max)] rounded-lg border border-gray-200 bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="expr-modal-heading"
      >
        <div className={`relative pt-8 pb-6 ${appendixRailClass}`}>
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
            <p className={`mx-auto mt-4 max-w-[72ch] ${APPENDIX_PROSE_JUSTIFIED_CLASS}`} style={{ textAlign: 'justify' }}>
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

        <div className={`pb-10 ${appendixRailClass}`}>
          {/* §1 — Definitions */}
          <AppendixSection
            n={1}
            label="The distinction"
            title="The distinction"
          >
            <AppendixTwoColBlock
              left={
                <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`} style={{ textAlign: 'justify' }}>
                  <p>
                    <InlineMathText>
                      {`The main page reports the detected pointwise symmetry group that licenses orbit compression. This appendix asks why that group is the right one for accumulation, and why a larger formal symmetry group appears once the full expression is viewed after summation.`
}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`The main page's goal is operational: given an einsum $\\sum_t \\text{summand}(t)$, it reports the minimum number of distinct scalar multiplications $\\mu$ and accumulations $\\alpha$ needed to produce the output tensor by reusing work across tuples.`}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`Concretely, $${notationLatex('g_pointwise')} \\subseteq \\mathrm{Sym}(L)$ consists of label permutations $\\pi$ for which $\\text{summand}(t) = \\text{summand}(\\pi^{-1} t)$ for every tuple $t \\in [n]^L$. It is the largest group that genuinely licenses Burnside-style orbit compression in the enumerate-and-accumulate model, and every main-page cost is computed with respect to it.`}
                    </InlineMathText>
                  </p>
                </div>
              }
              right={
                <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`} style={{ textAlign: 'justify' }}>
                  <p>
                    <InlineMathText>
                      {`By contrast, $${notationLatex('g_formal')} \\subseteq \\mathrm{Sym}(V_{\\mathrm{free}}) \\times \\mathrm{Sym}(W_{\\mathrm{summed}})$ captures invariance of the output tensor viewed as a formal polynomial after the summed labels have been aggregated out.`}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`The two groups need not agree. Formal dummy-index renamings preserve the total expression after summation, but they do not identify equal summands tuple-by-tuple and therefore do not automatically license pointwise compression.`}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`The appendix question is therefore where that boundary appears in the structure of the contraction itself. Section 2 answers it at the row level, by showing which candidate row moves induce genuine relabelings of the contraction and which do not.`}
                    </InlineMathText>
                  </p>
                </div>
              }
            />
          </AppendixSection>

          {/* §2 — 4-preset gallery + detection boundary */}
          <AppendixSection
            n={2}
            label="The row-level detection boundary"
            title="When can the σ-loop see a symmetry, and when can't it?"
          >

            <AppendixTwoColBlock
              left={
                <div className="space-y-4">
                  <div className="text-sm leading-7 text-foreground">
                    <InlineMathText>
                      {`The distinction from Section 1 becomes operational at the row level. The σ-loop ranges over the wreath-product row symmetries $${notationLatex('g_wreath')}$. For a candidate row move $${notationLatex('sigma_row_move')}$, let $M_{\\sigma}$ denote the incidence matrix obtained by permuting the rows of $${notationLatex('m_incidence')}$ by $${notationLatex('sigma_row_move')}$. We then ask whether the columns of $M_{\\sigma}$ can be matched bijectively with the columns of $${notationLatex('m_incidence')}$ by equality of column fingerprints. When they can, $${notationLatex('sigma_row_move')}$ induces a label permutation $\\pi_{\\sigma}$; when they cannot, the row move contributes no detected pointwise symmetry. The gallery below shows the three possible outcomes: identity after matching, a non-identity induced relabeling, or rejection.`}
                    </InlineMathText>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-[12px] border-collapse">
                <thead>
                  <tr className="border-b border-border/60 text-left text-muted-foreground">
                    <th className="px-2 py-2 font-semibold">Preset</th>
                    <th className="px-2 py-2 font-semibold"><Latex math="L, V, W" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="H_A, m_A" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="|G_{\text{wreath}}|" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="\pi_\sigma \neq \mathrm{id}" /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math="\pi_\sigma = \mathrm{id}" /></th>
                    <th className="px-2 py-2 font-semibold">rejected</th>
                    <th className="px-2 py-2 font-semibold"><Latex math="|G_{\text{pt}}|" /></th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2">
                      <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('frobenius')} groupLabel="trivial" />
                    </td>
                    <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                    <td className="px-2 py-2 font-mono">2</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">0</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">2</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">0</td>
                    <td className="px-2 py-2 font-mono">1</td>
                  </tr>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2">
                      <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('trace-product')} groupLabel="W: S2{i,j}" />
                    </td>
                    <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                    <td className="px-2 py-2 font-mono">2</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">1</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">1</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">0</td>
                    <td className="px-2 py-2 font-mono">2</td>
                  </tr>
                  <tr className="border-b border-border/40">
                    <td className="px-2 py-2">
                      <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('triangle')} groupLabel="C3{i,j,k}" />
                    </td>
                    <td className="px-2 py-2"><Latex math="\{i,j,k\}, \{i,j,k\}, \varnothing" /></td>
                    <td className="px-2 py-2"><Latex math="\{e\}, 3" /></td>
                    <td className="px-2 py-2 font-mono">6</td>
                    <td className="px-2 py-2 font-mono text-emerald-700">2</td>
                    <td className="px-2 py-2 font-mono text-muted-foreground">1</td>
                    <td className="px-2 py-2 font-mono text-amber-600/80">3</td>
                    <td className="px-2 py-2 font-mono">3</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-2">
                      <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('young-s3')} groupLabel="S3{i,j,k}" />
                    </td>
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
                </div>
              }
              right={
                <>
                  <ul className="space-y-1.5 text-[12px] leading-5 text-foreground">
                    <li><AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('frobenius')} groupLabel="trivial">frobenius</AppendixPresetHoverLabel> — <em>identity only</em>: every row move preserves the incidence pattern elementwise, so no non-identity induced relabeling survives.</li>
                    <li><AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('trace-product')} groupLabel="W: S2{i,j}">trace-product</AppendixPresetHoverLabel> — <em>identity + non-identity</em>: the two A&apos;s have different subscripts, so the operand swap induces <InlineMathText>{`$\\pi = (i\\;j)$`}</InlineMathText>; the detected group gains it.</li>
                    <li><AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('triangle')} groupLabel="C3{i,j,k}">triangle</AppendixPresetHoverLabel> — <em>all three outcomes visible</em>: the adjacent copy-transpositions are not admissible because their permuted column fingerprints do not match the original label-columns.</li>
                    <li><AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('young-s3')} groupLabel="S3{i,j,k}">young-s3</AppendixPresetHoverLabel> — <em>non-identity dominates</em>: the declared <InlineMathText>{`$S_3$`}</InlineMathText> action on T&apos;s axes yields admissible non-identity relabelings throughout.</li>
                  </ul>

                  <div className="mt-4 rounded-md border-l-4 border-border/60 bg-muted/20 px-4 py-3 text-[12px] leading-6 text-foreground">
                    <p className="mb-1 font-semibold">
                      <AppendixPresetHoverLabel
                        preset={EXAMPLES_BY_ID.get('frobenius')}
                        groupLabel="trivial"
                        className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                      >
                        Frobenius
                      </AppendixPresetHoverLabel>
                    </p>
                    <p>
                      <InlineMathText>
                        {`$${notationLatex('m_incidence')}$ is invariant elementwise under the operand swap (both A's have subscripts $(i, j)$), so the induced relabeling is the identity. No non-identity pointwise symmetry is detected, and the dummy renaming $(i\\;j)$ remains **row-unwitnessed** inside $S(W_{\\mathrm{summed}})$.`}
                      </InlineMathText>
                    </p>
                  </div>

                  <div className="mt-4 rounded-md border-l-4 border-border/60 bg-muted/20 px-4 py-3 text-[12px] leading-6 text-foreground">
                    <p className="mb-1 font-semibold">
                      <AppendixPresetHoverLabel
                        preset={EXAMPLES_BY_ID.get('triangle')}
                        groupLabel="C3{i,j,k}"
                        className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                      >
                        Triangle
                      </AppendixPresetHoverLabel>
                    </p>
                    <p>
                      <InlineMathText>
                        {`Each adjacent copy-transposition permutes the rows, but the columns of $M_{\\sigma}$ cannot be matched bijectively back to those of $${notationLatex('m_incidence')}$. That row move is therefore not admissible. Only the two 3-cycles induce non-identity relabelings $\\pi_{\\sigma}$, so $G_{\\text{pt}} = C_3$.`}
                      </InlineMathText>
                    </p>
                  </div>
                </>
              }
            >
            </AppendixTwoColBlock>

            <div className="mt-8">
              <div className={`space-y-5 ${APPENDIX_PROSE_CLASS}`}>
                <AppendixFormalBlock>
                  <div className="space-y-5">
                    <div className="space-y-2">
                      <p className={APPENDIX_KICKER_CLASS}>
                        Setup
                      </p>
                      <p className={APPENDIX_FORMAL_PROSE_CLASS}>
                        <InlineMathText>
                          {`Write $${notationLatex('g_wreath')} = \\prod_i (H_i \\wr S_{m_i})$ for the row-permutation group explored by the σ-loop. For each $${notationLatex('sigma_row_move')} \\in ${notationLatex('g_wreath')}$, the row-permuted matrix $M_{\\sigma}$ is compared against $${notationLatex('m_incidence')}$ at the level of column fingerprints. If those fingerprints determine a bijection of label-columns, we call $${notationLatex('sigma_row_move')}$ admissible and write $\\pi_{\\sigma}$ for the induced permutation of labels.`}
                        </InlineMathText>
                      </p>
                    </div>

                    <div className="h-px bg-gray-200" />

                    <div className="space-y-2">
                      <p className={APPENDIX_KICKER_CLASS}>
                        Proposition
                      </p>
                      <p className={APPENDIX_FORMAL_PROSE_CLASS}>
                        <span className="font-semibold text-gray-900">Frobenius-class presets.</span>{' '}
                        <InlineMathText>
                          {`Suppose no operand has declared axis symmetry, and every identical-operand group of size at least $2$ consists of copies with the same subscript string. Then $G_{\\text{pt}} = \\{e\\}$. Every row move in $${notationLatex('g_wreath')}$ preserves $${notationLatex('m_incidence')}$ elementwise, so every induced relabeling $\\pi_{\\sigma}$ is the identity; the only remaining formal symmetry lies in the dummy-label factor $S(W_{\\mathrm{summed}})$.`}
                        </InlineMathText>
                      </p>
                    </div>

                    <div className="h-px bg-gray-200" />

                    <div className="space-y-3">
                      <div className="space-y-2">
                        <p className={APPENDIX_KICKER_CLASS}>
                          Detection Principle
                        </p>
                        <p className={APPENDIX_FORMAL_PROSE_CLASS}>
                          The detection boundary is therefore as follows.
                        </p>
                      </div>
                      <ol className={`space-y-3 pl-5 ${APPENDIX_FORMAL_PROSE_CLASS} marker:font-semibold marker:text-gray-600`}>
                        <li>
                          <InlineMathText>
                            {`The σ-loop examines every row move $${notationLatex('sigma_row_move')} \\in ${notationLatex('g_wreath')}$.`}
                          </InlineMathText>
                        </li>
                        <li>
                          <InlineMathText>
                            {`If $${notationLatex('sigma_row_move')}$ is admissible and $\\pi_{\\sigma} \\neq \\mathrm{id}$, then $\\pi_{\\sigma}$ is a genuine pointwise symmetry and belongs to $G_{\\text{pt}}$.`}
                          </InlineMathText>
                        </li>
                        <li>
                          <InlineMathText>
                            {`If $${notationLatex('sigma_row_move')}$ is admissible but $\\pi_{\\sigma} = \\mathrm{id}$, then the row move preserves the incidence pattern without producing a new label symmetry.`}
                          </InlineMathText>
                        </li>
                        <li>
                          <InlineMathText>
                            {`If $${notationLatex('sigma_row_move')}$ is not admissible, then the row move does not correspond to any relabeling of the contraction and is rejected.`}
                          </InlineMathText>
                        </li>
                      </ol>
                    </div>
                  </div>
                </AppendixFormalBlock>
                <p className={`mt-5 ${APPENDIX_PROSE_CLASS}`}>
                  <InlineMathText>
                    {`This is the boundary used by the rest of the appendix. Section 3 extracts the visible-label action $${notationLatex('g_pointwise_restricted_v')}$ from the admissible relabelings $\\pi_{\\sigma}$, Section 4 isolates the row-unwitnessed dummy renamings $${notationLatex('s_w_summed')}$, and Section 5 combines the two into the formal symmetry group $${notationLatex('g_formal')}$.`}
                  </InlineMathText>
                </p>
              </div>
            </div>
          </AppendixSection>

          {/* §3 — Induced permutation group on V */}
          <AppendixSection
            n={3}
            label="First component"
            title={`The induced permutation group G_pt|V_free`}
          >

            <AppendixTwoColBlock
              useLg
              left={
                <div className="space-y-4">
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`Among the admissible induced relabelings $\\pi_{\\sigma}$ from Section 2, some preserve $${notationLatex('v_free')}$ setwise. Restricting those relabelings to $${notationLatex('v_free')}$ yields $${notationLatex('g_pointwise_restricted_v')}$. This is the row-witnessed visible-label action inherited from the detected pointwise symmetry group.`}
                    </InlineMathText>
                  </p>
                  <div className={`space-y-4 ${APPENDIX_PROSE_CLASS}`}>
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
                    <p className="font-semibold">
                      Worked example —{' '}
                      <AppendixPresetHoverLabel
                        preset={EXAMPLES_BY_ID.get('bilinear-trace')}
                        groupLabel={EXAMPLES_BY_ID.get('bilinear-trace')?.expectedGroup}
                        className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                      >
                        bilinear trace
                      </AppendixPresetHoverLabel>{' '}
                      at <Latex math="n = 2" />.
                    </p>
                    <p>
                      <InlineMathText>
                        {`The einsum $\\mathtt{ik{,}jl\\to ij}$ computes $R[i,j] = \\sum_{k,l} A[i,k] \\cdot A[j,l]$ with $V_{\\mathrm{free}} = \\{i, j\\}$ and $W_{\\mathrm{summed}} = \\{k, l\\}$. The σ-loop's top-group transposition emits the permutation that swaps the two identical $A$ operands, exchanging $i \\leftrightarrow j$ together with $k \\leftrightarrow l$. The detected pointwise group is therefore $G_{\\text{pt}} = \\{e,\\;(i\\;j)(k\\;l)\\}$. Restricting each element to $V_{\\mathrm{free}}$ yields $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}} = \\{e,\\;(i\\;j)\\}$, a copy of $S_2$ acting on $\\{i,j\\}$.`}
                      </InlineMathText>
                    </p>
                  </div>
                </div>
              }
              right={
                <div className="space-y-4">
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`Using the same $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$ as in Section 2, each output cell expands as a sum of four products:`}
                    </InlineMathText>
                  </p>
                  <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 font-mono text-[13px] leading-relaxed text-foreground">
                    <div>
                      R[<span style={vStyle()}>0</span>,<span style={vStyle()}>0</span>]
                      {' = '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]
                      {' = '}
                      1 + 2 + 2 + 4 = <strong>9</strong>
                    </div>
                    <div className="mt-1">
                      R[<span style={vStyle()}>0</span>,<span style={vStyle()}>1</span>]
                      {' = '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]
                      {' = '}
                      3 + 4 + 6 + 8 = <strong>21</strong>
                    </div>
                    <div className="mt-1">
                      R[<span style={vStyle()}>1</span>,<span style={vStyle()}>0</span>]
                      {' = '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]
                      {' = '}
                      3 + 6 + 4 + 8 = <strong>21</strong>
                    </div>
                    <div className="mt-1">
                      R[<span style={vStyle()}>1</span>,<span style={vStyle()}>1</span>]
                      {' = '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]
                      {' + '}
                      A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]
                      {' = '}
                      9 + 12 + 12 + 16 = <strong>49</strong>
                    </div>
                  </div>
                  <div className="rounded-md border-l-4 border-emerald-500 bg-emerald-50 px-5 py-3 text-sm leading-7 text-emerald-900">
                    <InlineMathText>
                      {`The two off-diagonal cells agree: $R[0,1] = R[1,0] = 21$, and the agreement is term-by-term — each product in one expansion is the commuted twin of a product in the other. The equality therefore holds on the computed output tensor itself, not merely on a total: $R[\\sigma\\,\\omega] = R[\\omega]$ for every $\\sigma \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$.`}
                    </InlineMathText>
                  </div>
                  <div className="rounded-md border border-border/60 bg-muted/20 px-4 py-3 text-[13px] leading-6 text-muted-foreground">
                    <InlineMathText>
                      {`Algebraically, $R[i,j] = (\\sum_k A[i,k])(\\sum_l A[j,l]) = v_i\\,v_j$ with $v = \\mathrm{rowsum}(A)$; the outer product $v\\,v^\\top$ is symmetric by construction, so $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}} = \\{e,(i\\;j)\\}$ acts trivially on $R$ for every $A$.`}
                    </InlineMathText>
                  </div>
                </div>
              }
            />
          </AppendixSection>

          {/* §4 — S(W_summed) */}
          <AppendixSection
            n={4}
            label="Second component"
            title="The symmetric group on summed labels"
          >

            <AppendixTwoColBlock
              left={
                <div className="space-y-4">
                  <p className="mb-3 text-sm leading-7 text-foreground">
                    <InlineMathText>
                      {`Section 3 isolated the visible-label action that remains pointwise on the output tensor. The complementary factor comes from the summed labels alone. Unlike $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the $S(W_{\\mathrm{summed}})$ factor is **row-unwitnessed**: no $\\sigma$ in the wreath produces these $W_{\\mathrm{summed}}$-permutations as its induced $\\pi$. They are formal symmetries by virtue of being permutations of bound summation indices — renaming dummies leaves any sum invariant. This is the only factor of $G_{\\text{f}}$ that is strictly non-row-visible.`}
                    </InlineMathText>
                  </p>
                  <div className={`space-y-4 ${APPENDIX_PROSE_CLASS}`}>
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
                </div>
              }
              right={
                <div className="space-y-4">
                  <div className="rounded-md border border-border/60 bg-muted/20 px-5 py-4 text-sm leading-7 text-foreground">
                    <p className="font-semibold">
                      Worked example —{' '}
                      <AppendixPresetHoverLabel
                        preset={EXAMPLES_BY_ID.get('bilinear-trace')}
                        groupLabel={EXAMPLES_BY_ID.get('bilinear-trace')?.expectedGroup}
                        className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                      >
                        bilinear trace
                      </AppendixPresetHoverLabel>{' '}
                      (continued).
                    </p>
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
                </div>
              }
            />
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

            <AppendixTwoColBlock
              useLg
              left={
                <div className={`space-y-4 ${APPENDIX_PROSE_CLASS}`}>
                  <p>
                    <InlineMathText>
                      {`Sections 3 and 4 have now isolated the two ingredients of the formal symmetry story: the row-witnessed action on $V_{\\mathrm{free}}$ and the row-unwitnessed permutations of $W_{\\mathrm{summed}}$. We now combine them.`}
                    </InlineMathText>
                  </p>
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
                      {`No additional closure step is needed at this stage. $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ has already been determined from the detected pointwise group, $S(W_{\\mathrm{summed}})$ depends only on how many summed labels there are, and $G_{\\text{f}}$ is their direct product.`}
                    </InlineMathText>
                  </p>
                </div>
              }
              right={
                <div className="space-y-4">
                  <div className="rounded-md border border-gray-100 bg-muted/20 px-4 py-3 text-sm leading-7 text-foreground">
                    <p>
                      The widget below enumerates the direct-product pairings for the selected preset.
                    </p>
                    <p className="mt-2 text-[12.5px] text-muted-foreground">
                      Rightmost-column rows pair one <InlineMathText>{`$G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$`}</InlineMathText>{' '}
                      orbit and one <InlineMathText>{`$S(W_{\\mathrm{summed}})$`}</InlineMathText>{' '}
                      orbit.
                      Hover in any column to highlight corresponding rows.
                    </p>
                  </div>
                  <div className="rounded-md border border-gray-200 bg-white p-2">
                    <VSubSwConstruction
                      expressionGroup={expressionGroup}
                      vLabels={vLabels}
                      wLabels={wLabels}
                    />
                  </div>
                </div>
              }
            />

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
                <div className={`mt-4 max-w-[74ch] ${APPENDIX_PROSE_CLASS}`}>
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
            <AppendixTwoColBlock
              left={
                <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`}>
                  <p>
                    <InlineMathText>
                      {`Section 6 closed the accumulation-count story: $G_{\\text{f}}$'s extra elements beyond $G_{\\text{pt}}$ admit no pointwise compression of $\\alpha$ under the enumerate-and-accumulate evaluation model. Before closing the appendix we pivot to a *different* optimization axis — output-tensor storage — where savings *are* available. The group governing this axis is $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$: it appeared in Section 5 as the $V_{\\mathrm{free}}$-factor of $G_{\\text{f}}$, but equivalently it is a subgroup of $G_{\\text{pt}}$ itself — the $V_{\\mathrm{free}}$-action that $G_{\\text{pt}}$'s $V_{\\mathrm{free}}/W_{\\mathrm{summed}}$-preserving elements already induce. That dual role is precisely why the output tensor is genuinely *pointwise*-symmetric under it, and therefore why the savings discussed below are not formal-only.`}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`For every $\\sigma \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the identity $R[\\sigma\\,\\omega] = R[\\omega]$ holds on the output tensor cell by cell. For generic operands at large enough $n$, this inclusion is tight: $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ is the complete structural $V_{\\mathrm{free}}$ symmetry of $R$.`}
                    </InlineMathText>
                  </p>
                  <p>
                    <InlineMathText>
                      {`The table below records the additional savings available when output storage also respects the visible-label symmetry induced by $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$.`}
                    </InlineMathText>
                  </p>
                </div>
              }
              right={
                <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`}>
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
                  <p>
                    <InlineMathText>
                      {`Per $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$-orbit of size $s$, the savings are $(s-1) \\cdot (\\text{accumulations per bin})$ operations, so the total gap scales with the order and orbit structure of $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$. Presets with trivial $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ carry no output-tensor mirroring — there is nothing to collapse, and $\\alpha_{\\text{storage}} = \\alpha_{\\text{engine}}$.`}
                    </InlineMathText>
                  </p>
                  <p className={APPENDIX_FOOTNOTE_CLASS}>
                    <InlineMathText>
                      {`The $S(W_{\\mathrm{summed}})$ factor of $G_{\\text{f}}$ contributes nothing at the storage level because it acts on summation variables rather than output cells; there is no output-tensor symmetry to exploit on the $W_{\\mathrm{summed}}$ side beyond what $G_{\\text{pt}}$ already captures through its orbit structure on tuples.`}
                    </InlineMathText>
                  </p>
                </div>
              }
            />

            <div className="mt-5 overflow-x-auto">
              <table className="w-full border-collapse text-[12px]">
                <thead className="border-b border-gray-200">
                  <tr className="text-left text-[12px] text-muted-foreground">
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
                <tbody className="[&_tr]:border-b [&_tr]:border-gray-100">
                  {SAVINGS_TABLE_ROWS.map((r) => {
                    const hasSaving = r.saving > 0;
                    const preset = EXAMPLES_BY_ID.get(r.id);
                    const subs = preset?.expression?.subscripts ?? '';
                    const output = preset?.expression?.output ?? '';
                    const operands = describeOperands(preset);
                    const groupLabel = preset?.expectedGroup ?? appendixGroupLabel(r.vSub);
                    return (
                      <tr key={r.id} className={hasSaving ? '' : 'text-muted-foreground'}>
                        <td className="px-2 py-2 whitespace-nowrap">
                          <AppendixPresetHoverLabel preset={preset} groupLabel={groupLabel} />
                        </td>
                        <td className="px-2 py-2 font-mono whitespace-nowrap">
                          <AppendixEinsumHoverCell
                            subs={subs}
                            output={output}
                            preset={preset}
                            groupLabel={groupLabel}
                          />
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
              <p className={`mt-2 ${APPENDIX_FOOTNOTE_CLASS}`}>
                All entries computed at <Latex math="n = 3" />; sorted by % saving, descending. <span className="font-mono not-italic">×k</span> on an operand indicates it appears <Latex math="k" /> times in the expression (driving top-group transpositions in the wreath σ-loop). Hover any preset name or einsum to see the full construction with per-operand ranks, declared axes, and generators.
              </p>
            </div>

            <div className={`mt-4 space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`}>
              <p>
                <InlineMathText>
                  {`The $S(W_{\\mathrm{summed}})$ factor of $G_{\\text{f}}$ contributes nothing at the storage level because $S(W_{\\mathrm{summed}})$ acts on summation variables rather than output cells; there is no output-tensor symmetry to exploit on the $W_{\\mathrm{summed}}$-side beyond what $G_{\\text{pt}}$ already captures through its orbit structure on tuples.`}
                </InlineMathText>
              </p>
              <p>
                <InlineMathText>
                  All entries in the table above are computed at{' '}
                  <Latex math="n = 3" />, sorted by savings, descending.
                </InlineMathText>
              </p>
            </div>

            <div className="mt-6 border-t border-stone-200/70 bg-gray-50 px-5 py-4 md:px-6">
              <div className="font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
                Scope
              </div>
              <p className="mt-1.5 text-[12.5px] leading-6 text-stone-700">
                <InlineMathText>
                  {`The $\\alpha$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with $G_{\\text{pt}}$ as the summand-value equivalence relation.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-[12.5px] leading-6 text-stone-700">
                <InlineMathText>
                  {`Output-tensor storage collapse, algebraic restructuring such as factoring $R = v\\,v^\\top$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured here.`}
                </InlineMathText>
              </p>
            </div>
          </AppendixSection>
        </div>
      </div>
    </div>
  );
}
