import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import AppendixSection from './AppendixSection.jsx';
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
const BURNSIDE_GAP_PRESET_IDS = ['bilinear-trace', 'young-s3', 'young-s4-v2w2'];
const BURNSIDE_GAP_PRESETS = BURNSIDE_GAP_PRESET_IDS
  .map((id) => {
    const preset = EXAMPLES_BY_ID.get(id);
    const idx = EXAMPLES.findIndex((example) => example.id === id);
    return preset && idx >= 0 ? { id, idx, preset } : null;
  })
  .filter(Boolean);

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

const APPENDIX_PROSE_CLASS = 'font-serif text-[17px] leading-[1.75] text-gray-900';
const APPENDIX_PROSE_JUSTIFIED_CLASS = `${APPENDIX_PROSE_CLASS} text-justify`;
const APPENDIX_FORMAL_PROSE_CLASS = 'font-serif text-[17px] leading-[1.85] text-gray-800';
const APPENDIX_APP_TEXT_CLASS = 'text-[13px] leading-[1.55] text-gray-700';
const APPENDIX_APP_TEXT_STRONG_CLASS = 'text-[13px] leading-[1.55] text-gray-900';
const APPENDIX_SMALL_TEXT_CLASS = 'text-[12px] leading-5 text-gray-600';
const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-[13px] leading-relaxed text-gray-900';
const APPENDIX_KICKER_CLASS = 'text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-400';
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

function AppendixWorkedExample({
  preset,
  groupLabel,
  title,
  intro = null,
  children,
}) {
  return (
    <div className="space-y-4">
      <p className="text-[15px] font-semibold leading-7 text-gray-900">
        Worked example —{' '}
        <AppendixPresetHoverLabel
          preset={preset}
          groupLabel={groupLabel}
          className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
        >
          {title ?? preset?.label ?? preset?.name ?? preset?.id ?? 'preset'}
        </AppendixPresetHoverLabel>
      </p>
      {intro ? (
        <div className={APPENDIX_PROSE_CLASS}>
          {intro}
        </div>
      ) : null}
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function WorkedExampleEquation({ assignment, numeric }) {
  return (
    <div className="space-y-1">
      <div className="text-gray-900">{assignment}</div>
      <div className="pl-[5.5ch] text-gray-700">{numeric}</div>
    </div>
  );
}

function WorkedExampleEquationLedger({ children }) {
  return (
    <div className={APPENDIX_MONO_LEDGER_CLASS}>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function WorkedExampleNote({ tone = 'neutral', children }) {
  const contentClass =
    tone === 'success'
      ? APPENDIX_PROSE_CLASS.replace('text-gray-900', 'text-gray-800')
      : APPENDIX_PROSE_CLASS.replace('text-gray-900', 'text-gray-700');
  return (
    <div className={contentClass}>
      {children}
    </div>
  );
}

function AppendixPartHeader({ part, title, deck = null, className = '' }) {
  return (
    <div className={['space-y-3', className].join(' ')}>
      <p className={APPENDIX_KICKER_CLASS}>{part}</p>
      <h3 className="font-heading text-[28px] font-semibold leading-tight text-gray-900">
        {title}
      </h3>
      {deck ? (
        <p className="max-w-[78ch] font-serif text-[17px] leading-[1.75] text-gray-700">
          {deck}
        </p>
      ) : null}
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
export default function ExpressionLevelModal({ isOpen, onClose, analysis, group, example = null, onSelectPreset = null }) {
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
          <div className="flex flex-col">
            <div
              className="mb-5 font-sans text-[10px] font-semibold uppercase text-gray-400"
              style={{ letterSpacing: '0.2em' }}
            >
              <span aria-hidden className="mr-2 inline-block h-px w-8 align-middle bg-gray-300" />
              Appendix
            </div>

            <h2
              id="expr-modal-heading"
              className="m-0 font-semibold text-gray-900"
              style={{
                fontFamily: 'var(--font-display-serif), Georgia, serif',
                fontVariationSettings: "'opsz' 72",
                fontSize: 'clamp(36px, 5vw, 52px)',
                letterSpacing: '-0.02em',
                lineHeight: 1.05,
              }}
            >
              Expression-level symmetry and Symmetry-aware storage
              <span style={{ color: 'var(--coral)' }}>.</span>
            </h2>

            <p
              className="mt-5 max-w-[min(100%,980px)] text-[17px] italic text-gray-600"
              style={{
                fontFamily: 'var(--font-paper-serif), Georgia, serif',
                fontVariationSettings: "'opsz' 18",
                lineHeight: 1.6,
              }}
            >
              This appendix has two parts. The first follows the
              expression-level symmetry story from the row-level boundary to
              the assembled formal group. The second turns to symmetry-aware
              storage, which sits outside the main page&apos;s accumulation
              count and exploits a different optimization axis.
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
          <AppendixPartHeader
            part="Part I"
            title="Expression-level symmetry"
            deck="This first part follows the expression-level argument from the row-level admissibility test to the larger formal symmetry group built after summation."
          />

          <AppendixSection
            n={1}
            label="Where the boundary appears"
            title="Where does the boundary appear?"
            className="pt-10"
          >
            <div className="space-y-8">
              <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`} style={{ textAlign: 'justify' }}>
                <p>
                  <InlineMathText>
                    {`The main page works with $${notationLatex('g_pointwise')}$ because that is the group under which summands are genuinely equal tuple by tuple. A larger group, $${notationLatex('g_formal')}$, appears only after the full expression is viewed post-summation: it preserves the resulting formal polynomial, but not necessarily the individual summands that produced it.`}
                  </InlineMathText>
                </p>
                <p>
                  <InlineMathText>
                    {`The distinction first becomes visible at the row level. Candidate row moves either induce genuine relabelings of the contraction or fail that test altogether, and that boundary determines what can later act on the output tensor and what survives only formally after summation.`}
                  </InlineMathText>
                </p>
              </div>

              <AppendixTwoColBlock
                left={
                  <div className="space-y-5">
                    <p className={APPENDIX_PROSE_CLASS}>
                      <InlineMathText>
                        {`The $\\sigma$-loop tests a candidate row move $${notationLatex('sigma_row_move')} \\in ${notationLatex('g_wreath')}$ by permuting the rows of $${notationLatex('m_incidence')}$. Writing $M_{\\sigma}$ for the row-permuted incidence matrix, we call $${notationLatex('sigma_row_move')}$ admissible when the column fingerprints of $M_{\\sigma}$ match those of $${notationLatex('m_incidence')}$ bijectively; that matching induces the label permutation $\\pi_{\\sigma}$. The ledger below records the outcome for four representative presets.`}
                      </InlineMathText>
                    </p>

                    <div className="overflow-x-auto">
                      <table className={`w-full border-collapse ${APPENDIX_SMALL_TEXT_CLASS} text-gray-600`}>
                        <thead className="border-y border-gray-200/90">
                          <tr className="text-left text-gray-500">
                            <th className="px-2 py-2 font-semibold text-gray-600">Preset</th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="L, V, W" /></th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="H_A, m_A" /></th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="|G_{\text{wreath}}|" /></th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="\pi_\sigma \neq \mathrm{id}" /></th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="\pi_\sigma = \mathrm{id}" /></th>
                            <th className="px-2 py-2 font-semibold text-gray-600">rejected</th>
                            <th className="px-2 py-2 font-semibold text-gray-600"><Latex math="|G_{\text{pt}}|" /></th>
                          </tr>
                        </thead>
                        <tbody className="[&_tr]:border-b [&_tr]:border-gray-100/90">
                          <tr>
                            <td className="px-2 py-2">
                              <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('frobenius')} groupLabel="trivial" />
                            </td>
                            <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                            <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">2</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-emerald-700">0</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">2</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">1</td>
                          </tr>
                          <tr>
                            <td className="px-2 py-2">
                              <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('trace-product')} groupLabel="W: S2{i,j}" />
                            </td>
                            <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                            <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">2</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-emerald-700">1</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">2</td>
                          </tr>
                          <tr>
                            <td className="px-2 py-2">
                              <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('triangle')} groupLabel="C3{i,j,k}" />
                            </td>
                            <td className="px-2 py-2"><Latex math="\{i,j,k\}, \{i,j,k\}, \varnothing" /></td>
                            <td className="px-2 py-2"><Latex math="\{e\}, 3" /></td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">6</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-emerald-700">2</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-amber-600">3</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">3</td>
                          </tr>
                          <tr>
                            <td className="px-2 py-2">
                              <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('young-s3')} groupLabel="S3{i,j,k}" />
                            </td>
                            <td className="px-2 py-2"><Latex math="\{a,b,c\}, \{a,b\}, \{c\}" /></td>
                            <td className="px-2 py-2"><Latex math="S_3, 1" /></td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">6</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-emerald-700">5</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                            <td className="px-2 py-2 font-mono tabular-nums text-gray-800">6</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    <div className="space-y-3 border-t border-gray-200 pt-4">
                      <p className={APPENDIX_APP_TEXT_CLASS}>
                        <span className="font-semibold text-gray-900">Frobenius.</span>{' '}
                        <InlineMathText>
                          {`The operand swap is admissible because both copies of $A$ carry the same subscript string $(i,j)$, but the induced relabeling remains $\\pi_{\\sigma} = \\mathrm{id}$. No non-identity pointwise symmetry is detected, and the leftover dummy renaming survives only in $${notationLatex('s_w_summed')}$.`}
                        </InlineMathText>
                      </p>
                      <p className={APPENDIX_APP_TEXT_CLASS}>
                        <span className="font-semibold text-gray-900">Triangle.</span>{' '}
                        <InlineMathText>
                          {`The two 3-cycles are admissible and induce non-identity relabelings $\\pi_{\\sigma}$, while the adjacent copy-transpositions fail the fingerprint match and are rejected. The same preset therefore shows both sides of the boundary at once.`}
                        </InlineMathText>
                      </p>
                    </div>
                  </div>
                }
                right={
                  <div className="space-y-5">
                    <div className="border-y border-gray-200 py-5">
                      <p className={APPENDIX_KICKER_CLASS}>Formal takeaway</p>
                      <p className={`mt-4 ${APPENDIX_FORMAL_PROSE_CLASS}`}>
                        <InlineMathText>
                          {`A row move $${notationLatex('sigma_row_move')} \\in ${notationLatex('g_wreath')}$ is admissible when the column fingerprints of $M_{\\sigma}$ determine a bijection back to the columns of $${notationLatex('m_incidence')}$. That bijection induces the label permutation $\\pi_{\\sigma}$.`}
                        </InlineMathText>
                      </p>
                    </div>

                    <p className={APPENDIX_PROSE_CLASS}>
                      <InlineMathText>
                        {`If every admissible $\\pi_{\\sigma}$ is the identity, as in Frobenius-class presets, then $G_{\\text{pt}} = \\{e\\}$ even though the dummy-label factor $${notationLatex('s_w_summed')}$ may still survive as a larger formal symmetry.`}
                      </InlineMathText>
                    </p>

                    <div className="border-y border-gray-200">
                      <div className="grid gap-x-5 gap-y-1 py-3 sm:grid-cols-[112px_minmax(0,1fr)]">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-700">
                          Record
                        </p>
                        <p className={APPENDIX_APP_TEXT_CLASS}>
                          <InlineMathText>
                            {`Record every admissible $\\pi_{\\sigma} \\neq \\mathrm{id}$ in $G_{\\text{pt}}$.`}
                          </InlineMathText>
                        </p>
                      </div>
                      <div className="grid gap-x-5 gap-y-1 border-t border-gray-200 py-3 sm:grid-cols-[112px_minmax(0,1fr)]">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-500">
                          Identity-only
                        </p>
                        <p className={APPENDIX_APP_TEXT_CLASS}>
                          <InlineMathText>
                            {`Treat admissible moves with $\\pi_{\\sigma} = \\mathrm{id}$ as identity-only witnesses of the incidence pattern.`}
                          </InlineMathText>
                        </p>
                      </div>
                      <div className="grid gap-x-5 gap-y-1 border-t border-gray-200 py-3 sm:grid-cols-[112px_minmax(0,1fr)]">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-amber-700">
                          Reject
                        </p>
                        <p className={APPENDIX_APP_TEXT_CLASS}>
                          <InlineMathText>
                            {`Reject every non-admissible $${notationLatex('sigma_row_move')}$, since no relabeling of the contraction realizes it.`}
                          </InlineMathText>
                        </p>
                      </div>
                    </div>

                    <p className={`${APPENDIX_APP_TEXT_CLASS} text-gray-600`}>
                      <InlineMathText>
                        {`This boundary drives the rest of Part I. Chapter 2 places the visible-label action $${notationLatex('g_pointwise_restricted_v')}$ beside the row-unwitnessed dummy renamings $${notationLatex('s_w_summed')}$, and Chapter 3 combines the two into the formal symmetry group $${notationLatex('g_formal')}$.`}
                      </InlineMathText>
                    </p>
                  </div>
                }
              />
            </div>
          </AppendixSection>

          <AppendixSection
            n={2}
            label="Pointwise versus formal symmetry"
            title="Pointwise symmetry versus formal symmetry"
          >
            <div className={`space-y-4 ${APPENDIX_PROSE_CLASS}`}>
              <p>
                <InlineMathText>
                  {`Chapter 1 isolates the row moves that genuinely relabel the contraction. What remains now splits in two: a visible action that still moves output cells, and a formal-only action that survives only after the summed labels have been aggregated away.`}
                </InlineMathText>
              </p>
            </div>

            <div className="editorial-two-col-divider-lg mt-2 grid gap-y-4 gap-x-8 lg:grid-cols-2">
              <div className="order-1 space-y-4 lg:col-start-1 lg:row-start-1">
                <p className={APPENDIX_PROSE_CLASS}>
                  <span className="font-semibold text-gray-900">Visible action on the output tensor.</span>{' '}
                  <InlineMathText>
                    {`Among the admissible induced relabelings $\\pi_{\\sigma}$ from Chapter 1, some preserve $${notationLatex('v_free')}$ setwise. Restricting them to $${notationLatex('v_free')}$ produces $${notationLatex('g_pointwise_restricted_v')}$, the action inherited from the detected pointwise symmetry group.`}
                  </InlineMathText>
                </p>
              </div>
              <div className="order-5 space-y-4 lg:col-start-2 lg:row-start-1">
                <p className={APPENDIX_PROSE_CLASS}>
                  <span className="font-semibold text-gray-900">Formal symmetry on the summed labels.</span>{' '}
                  <InlineMathText>
                    {`The complementary factor comes from the summed labels alone. Once the visible $V_{\\mathrm{free}}$ action is separated, permutations of $W_{\\mathrm{summed}}$ act only after aggregation. They preserve the full sum after aggregation but do not give pointwise equal summands.`}
                  </InlineMathText>
                </p>
              </div>

              <div className="order-2 border-y border-gray-200 py-5 lg:col-start-1 lg:row-start-2">
                <p className={APPENDIX_KICKER_CLASS}>Formal takeaway</p>
                <p className={`mt-4 ${APPENDIX_FORMAL_PROSE_CLASS}`}>
                  <InlineMathText>
                    {`$${notationLatex('g_pointwise_restricted_v')}$ is obtained by restricting the $${notationLatex('v_free')}$-preserving part of $G_{\\text{pt}}$ to the free labels alone.`}
                  </InlineMathText>
                </p>
              </div>
              <div className="order-6 border-y border-gray-200 py-5 lg:col-start-2 lg:row-start-2">
                <p className={APPENDIX_KICKER_CLASS}>Formal takeaway</p>
                <p className={`mt-4 ${APPENDIX_FORMAL_PROSE_CLASS}`}>
                  <InlineMathText>
                    {`$${notationLatex('s_w_summed')}$ is the full symmetric group on the summed labels; its elements rename bound summation variables and preserve the formal expression after summation.`}
                  </InlineMathText>
                </p>
              </div>

              <div className="order-3 lg:col-start-1 lg:row-start-3">
                <p className={APPENDIX_PROSE_CLASS}>
                  <InlineMathText>
                    {`This is the part of $G_{\\text{pt}}$ that still acts on the computed output tensor: for every $\\sigma \\in ${notationLatex('g_pointwise_restricted_v')}$, the identity $R[\\sigma\\,\\omega] = R[\\omega]$ holds on output cells themselves.`}
                  </InlineMathText>
                </p>
              </div>
              <div className="order-7 lg:col-start-2 lg:row-start-3">
                <p className={APPENDIX_PROSE_CLASS}>
                  <InlineMathText>
                    {`That invariance is weaker than pointwise symmetry: it preserves the total sum but does not force equality of individual summands at permuted tuples.`}
                  </InlineMathText>
                </p>
              </div>

              <div className="order-4 lg:col-start-1 lg:row-start-4">
                <AppendixWorkedExample
                  preset={EXAMPLES_BY_ID.get('bilinear-trace')}
                  title="bilinear trace"
                  groupLabel={EXAMPLES_BY_ID.get('bilinear-trace')?.expectedGroup}
                  intro={
                    <>
                      <p>
                        <InlineMathText>
                          {`For bilinear trace at $n = 2$, the top-group transposition exchanges $i \\leftrightarrow j$ together with $k \\leftrightarrow l$, so restricting the detected pointwise symmetry to $${notationLatex('v_free')}$ yields $(i\\;j)$.`}
                        </InlineMathText>
                      </p>
                      <p>
                        <InlineMathText>
                          {`To see the visible symmetry on $R$, compare the two off-diagonal output cells:`}
                        </InlineMathText>
                      </p>
                    </>
                  }
                >
                  <WorkedExampleEquationLedger>
                    <WorkedExampleEquation
                      assignment={
                        <>
                          R[<span style={vStyle}>0</span>,<span style={vStyle}>1</span>] =
                          A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>] + A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>] + A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>] + A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                        </>
                      }
                      numeric={
                        <>
                          = 3 + 4 + 6 + 8 = <strong>21</strong>
                        </>
                      }
                    />
                    <WorkedExampleEquation
                      assignment={
                        <>
                          R[<span style={vStyle}>1</span>,<span style={vStyle}>0</span>] =
                          A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>] + A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>] + A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>] + A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]
                        </>
                      }
                      numeric={
                        <>
                          = 3 + 6 + 4 + 8 = <strong>21</strong>
                        </>
                      }
                    />
                  </WorkedExampleEquationLedger>
                  <WorkedExampleNote tone="success">
                    <InlineMathText>
                      {`The equality $R[0,1] = R[1,0]$ is therefore genuine on the computed output tensor, not merely on the formal sum.`}
                    </InlineMathText>
                  </WorkedExampleNote>
                  <WorkedExampleNote>
                    <InlineMathText>
                      {`Algebraically, $R[i,j] = v_i\\,v_j$ with $v = \\mathrm{rowsum}(A)$, so the induced transposition on $${notationLatex('v_free')}$ acts visibly on $R$.`}
                    </InlineMathText>
                  </WorkedExampleNote>
                </AppendixWorkedExample>
              </div>
              <div className="order-8 lg:col-start-2 lg:row-start-4">
                <AppendixWorkedExample
                  preset={EXAMPLES_BY_ID.get('bilinear-trace')}
                  title="bilinear trace"
                  groupLabel={EXAMPLES_BY_ID.get('bilinear-trace')?.expectedGroup}
                  intro={
                    <>
                      <p>
                        <InlineMathText>
                          {`With $W_{\\mathrm{summed}} = \\{k,l\\}$, the dummy swap $(k\\;l)$ preserves the double sum but sends individual summands to different products.`}
                        </InlineMathText>
                      </p>
                      <p>
                        <InlineMathText>
                          {`Compare one pair of permuted summands:`}
                        </InlineMathText>
                      </p>
                    </>
                  }
                >
                  <WorkedExampleEquationLedger>
                    <WorkedExampleEquation
                      assignment={
                        <>
                          (k,l) = (<span style={wStyle}>0</span>,<span style={wStyle}>1</span>) :
                          A[<span style={vStyle}>0</span>,<span style={wStyle}>0</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>1</span>]
                        </>
                      }
                      numeric={
                        <>
                          = 1 · 4 = <strong>4</strong>
                        </>
                      }
                    />
                    <WorkedExampleEquation
                      assignment={
                        <>
                          (k,l) = (<span style={wStyle}>1</span>,<span style={wStyle}>0</span>) :
                          A[<span style={vStyle}>0</span>,<span style={wStyle}>1</span>]·A[<span style={vStyle}>1</span>,<span style={wStyle}>0</span>]
                        </>
                      }
                      numeric={
                        <>
                          = 2 · 3 = <strong>6</strong>
                        </>
                      }
                    />
                  </WorkedExampleEquationLedger>
                  <WorkedExampleNote>
                    <InlineMathText>
                      {`The whole double sum is unchanged by renaming dummy variables, but the products are not equal term-by-term; formal symmetry does not imply pointwise equality.`}
                    </InlineMathText>
                  </WorkedExampleNote>
                </AppendixWorkedExample>
              </div>
            </div>
          </AppendixSection>

          <AppendixSection
            n={3}
            label="Assembling the formal group"
            title="Assembling the formal group"
          >
            <div className={`space-y-4 ${APPENDIX_PROSE_CLASS}`}>
              <p>
                <InlineMathText>
                  {`Chapter 2 isolates the two surviving ingredients: the visible action on $V_{\\mathrm{free}}$ and the formal-only permutations of $W_{\\mathrm{summed}}$. Putting them together yields the larger formal symmetry group $${notationLatex('g_formal')}$ — but the same larger group does not license pointwise compression of the accumulation count.`}
                </InlineMathText>
              </p>
            </div>

            <AppendixTwoColBlock
              useLg
              className="mt-2"
              left={
                <div className="space-y-4">
                  <div className={APPENDIX_PROSE_CLASS}>
                    <p>
                      <InlineMathText>
                        {`Each pair $(\\sigma_{V_{\\mathrm{free}}},\\;\\sigma_{W_{\\mathrm{summed}}})$ with $\\sigma_{V_{\\mathrm{free}}} \\in G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ and $\\sigma_{W_{\\mathrm{summed}}} \\in S(W_{\\mathrm{summed}})$ lifts to a single label permutation on $V_{\\mathrm{free}} \\cup W_{\\mathrm{summed}}$ that acts separately on the free and summed positions. The set of all such lifts is $G_{\\text{f}}$, with order $|G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}| \\cdot |W_{\\mathrm{summed}}|!$.`}
                      </InlineMathText>
                    </p>
                    <p className="mt-4">
                      <InlineMathText>
                        {`No extra closure step is needed here. $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ has already been determined from the detected pointwise group, $S(W_{\\mathrm{summed}})$ depends only on the number of summed labels, and $G_{\\text{f}}$ is their direct product.`}
                      </InlineMathText>
                    </p>
                  </div>
                  <p className={APPENDIX_SMALL_TEXT_CLASS}>
                    The widget below enumerates these direct-product pairings for
                    the selected preset. Hover in any column to highlight the
                    corresponding rows.
                  </p>
                  <VSubSwConstruction
                    expressionGroup={expressionGroup}
                    vLabels={vLabels}
                    wLabels={wLabels}
                  />
                </div>
              }
              right={
                <div className="space-y-4">
                  {exprAlpha !== null ? (
                    <>
                      <p className={APPENDIX_PROSE_CLASS}>
                        <InlineMathText>
                          {`If Burnside's orbit-counting formula is applied to $G_{\\text{f}}$ instead of $G_{\\text{pt}}$, it counts formal orbits rather than genuine equal-summand orbits. The resulting compression claim is therefore too optimistic.`}
                        </InlineMathText>
                      </p>
                      <div className="rounded-md border border-amber-200 bg-amber-50/80 px-5 py-4">
                        <p className={APPENDIX_APP_TEXT_STRONG_CLASS.replace('text-gray-900', 'text-amber-900')}>
                          <InlineMathText>
                            {`Applying Burnside to $G_{\\text{f}}$ would yield $\\alpha =$`}
                          </InlineMathText>{' '}
                          <strong>{exprAlpha}</strong>.
                        </p>
                        <p className={`mt-2 ${APPENDIX_APP_TEXT_CLASS.replace('text-gray-700', 'text-amber-800')}`}>
                          <InlineMathText>
                            {`That value is not a faithful compression count. Orbits under $S(W_{\\mathrm{summed}})$ contain tuples whose summand values differ, because the $W_{\\mathrm{summed}}$ factor only preserves the post-summation expression, not the tuple-level summands themselves.`}
                          </InlineMathText>
                        </p>
                      </div>
                      <p className={APPENDIX_PROSE_CLASS}>
                        <InlineMathText>
                          {`$G_{\\text{pt}}$ is the largest group under which every orbit's summand values are genuinely equal, so Burnside's "one representative per orbit" principle is faithful there and only there.`}
                        </InlineMathText>
                      </p>
                    </>
                  ) : (
                    <div className="rounded-md border border-gray-200 bg-gray-50 px-5 py-4">
                      <div className="space-y-3">
                        {example ? (
                          <div className="space-y-2">
                            <p className={APPENDIX_KICKER_CLASS}>Selected einsum</p>
                            <div className="inline-flex max-w-full rounded-xl border border-stone-200 bg-white px-3 py-2.5 shadow-sm">
                              <div className="font-mono text-[12px] leading-6 text-stone-900">
                                <FormulaHighlighted example={example} hoveredLabels={null} />
                              </div>
                            </div>
                          </div>
                        ) : null}
                        <p className={APPENDIX_APP_TEXT_CLASS}>
                          <InlineMathText>
                            {`For this einsum $G_{\\text{f}}$ coincides with $G_{\\text{pt}}$ — either $|W_{\\mathrm{summed}}| \\leq 1$ or the induced permutation group on $V_{\\mathrm{free}}$ is trivial — so there is no Burnside overcount to display.`}
                          </InlineMathText>
                        </p>
                        {onSelectPreset && BURNSIDE_GAP_PRESETS.length ? (
                          <div className="border-t border-gray-200 pt-3">
                            <p className={APPENDIX_SMALL_TEXT_CLASS}>
                              To see the impact, jump to one of these presets:
                            </p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {BURNSIDE_GAP_PRESETS.map((suggestedPreset) => (
                                <button
                                  key={suggestedPreset.id}
                                  type="button"
                                  onClick={() => onSelectPreset?.(suggestedPreset.idx)}
                                  className="inline-flex items-center rounded-full border border-stone-200 bg-white px-3 py-1.5 text-[12px] font-medium text-stone-700 shadow-sm transition-colors hover:border-stone-300 hover:bg-stone-50"
                                >
                                  {suggestedPreset.preset.name}
                                </button>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  )}
                </div>
              }
            />

            <p className={`mt-6 ${APPENDIX_APP_TEXT_CLASS} text-gray-600`}>
              <InlineMathText>
                {`That closes the expression-level story. Part II turns to storage, where the visible free-label action still matters, but now as a post-accumulation output-layout symmetry rather than a summand-compression rule.`}
              </InlineMathText>
            </p>
          </AppendixSection>

          <AppendixPartHeader
            part="Part II"
            title="Symmetry-aware storage"
            deck="The last appendix item changes optimization axis. It is no longer about compressing summands at expression level, but about collapsing mirrored output cells after accumulation has already happened."
            className="border-t border-gray-200 pt-10"
          />

          <section className="pt-8">
            <div className="space-y-5">
              <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`}>
                <p>
                  <InlineMathText>
                    {`The governing group for storage is $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$, the visible free-label action already extracted in Part I. It appeared there as the $V_{\\mathrm{free}}$ factor of $${notationLatex('g_formal')}$, but it is equally a subgroup of $G_{\\text{pt}}$ itself. That is why the output tensor is genuinely pointwise-symmetric under it, and why storage collapse here is legitimate rather than merely formal.`}
                  </InlineMathText>
                </p>
                <p>
                  <InlineMathText>
                    {`The table below records the additional savings available when output storage respects those $G_{\\text{pt}}\\big|_{V_{\\mathrm{free}}}$ orbits. This is a separate optimization axis from the main page's $\\alpha$: it reduces mirrored writes to output cells after the accumulations have already been determined.`}
                  </InlineMathText>
                </p>
              </div>

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
                  All entries computed at <Latex math="n = 3" />; sorted by %
                  saving, descending. <span className="font-mono not-italic">×k</span>{' '}
                  on an operand indicates it appears <Latex math="k" /> times
                  in the expression. Hover any preset name or einsum to see the
                  full construction with per-operand ranks, declared axes, and
                  generators.
                </p>
              </div>

              <div className={`space-y-4 ${APPENDIX_PROSE_JUSTIFIED_CLASS}`}>
                <p>
                  <InlineMathText>
                    {`The formal-only factor $${notationLatex('s_w_summed')}$ contributes nothing at the storage level because it acts on summation variables rather than output cells. The only storage symmetries available are the visible free-label symmetries already inherited from $G_{\\text{pt}}$.`}
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
                    {`Output-tensor storage collapse, algebraic restructuring such as factoring $R = v\\,v^\\top$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page.`}
                  </InlineMathText>
                </p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
