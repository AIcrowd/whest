import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Latex from './Latex.jsx';
import InlineMathText from './InlineMathText.jsx';
import VSubSwConstruction from './VSubSwConstruction.jsx';
import AppendixSection from './AppendixSection.jsx';
import EditorialCallout from './EditorialCallout.jsx';
import SectionReferenceLink from './SectionReferenceLink.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';
import { FormulaHighlighted, SymmetryChip } from './StickyBar.jsx';
import { computeExpressionAlphaComparison } from '../engine/comparisonAlpha.js';
import { EXAMPLES } from '../data/examples.js';
import { formatGeneratorNotation, variableSymmetryLabel } from '../lib/symmetryLabel.js';
import { notationColor, notationColoredLatex, notationLatex } from '../lib/notationSystem.js';

// Lookup map keyed by preset id so §7's savings table can pull the raw
// einsum and per-operand symmetry declarations straight from the source
// of truth, rather than duplicating that data into the row array below.
const EXAMPLES_BY_ID = new Map(EXAMPLES.map((ex) => [ex.id, ex]));
const BURNSIDE_GAP_PRESET_IDS = ['bilinear-trace', 'direct-s2-c3', 'mixed-chain'];
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
 * vocabulary (`dense`, `S3`, `C4`, `D2`, `⟨(0 1), (2 3)⟩`) the main-page
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
    return `custom ${formatGeneratorNotation(variable.generators) ?? '⟨∅⟩'}`;
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
  return formatGeneratorNotation(variable.generators) ?? 'custom';
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
const APPENDIX_ARTICLE_LANE_CLASS = 'max-w-[78ch] space-y-4 [&_p]:text-justify';
const APPENDIX_SUPPORT_SHELF_CLASS = 'rounded-xl border border-gray-200 bg-white p-4 md:p-5';
const APPENDIX_SHORT_V_LATEX = () => notationColoredLatex('v_free', 'V');
const APPENDIX_SHORT_W_LATEX = () => notationColoredLatex('w_summed', 'W');
const APPENDIX_SHORT_S_W_LATEX = () => notationColoredLatex('s_w_summed', 'S(W)');
const APPENDIX_W_EQUALS_L_MINUS_V_LATEX = () => `${APPENDIX_SHORT_W_LATEX()} = ${notationLatex('l_labels')} \\setminus ${APPENDIX_SHORT_V_LATEX()}`;
const APPENDIX_M_SIGMA_LATEX = `${notationLatex('m_incidence')}_{${notationLatex('sigma_row_move')}}`;
const APPENDIX_PI_SIGMA_LATEX = `${notationLatex('pi_relabeling')}_{${notationLatex('sigma_row_move')}}`;
const APPENDIX_PI_RESTRICT_V_LATEX = () => `${notationLatex('pi_relabeling')}|_{${APPENDIX_SHORT_V_LATEX()}}`;
const APPENDIX_G_OUT_DEFINITION_LATEX = () =>
  `${notationLatex('g_output')} := \\{ ${APPENDIX_PI_RESTRICT_V_LATEX()} : ${notationLatex('pi_relabeling')} \\in ${notationLatex('g_pointwise')} \\text{ and } ${notationLatex('pi_relabeling')}(${APPENDIX_SHORT_V_LATEX()}) = ${APPENDIX_SHORT_V_LATEX()} \\}.`;

function getExampleExpressionParts(example) {
  const subscripts = example?.expression?.subscripts ?? '';
  const output = example?.expression?.output ?? '';
  const operandSubscripts = subscripts
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);
  if (!operandSubscripts.length) return null;

  const operandNames = (example?.expression?.operandNames ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

  const outputLabels = String(output).trim().split('').filter(Boolean);
  const freeSet = new Set(outputLabels);
  const summedLabels = [];
  for (const subs of operandSubscripts) {
    for (const label of subs) {
      if (!freeSet.has(label) && !summedLabels.includes(label)) summedLabels.push(label);
    }
  }

  return {
    operandSubscripts,
    operandNames,
    outputLabels,
    summedLabels,
  };
}

function buildExpandedEinsumEquation(example) {
  const parts = getExampleExpressionParts(example);
  if (!parts) return null;
  const {
    operandSubscripts,
    operandNames,
    outputLabels,
    summedLabels,
  } = parts;

  const lhsLatex = outputLabels.length ? `R[${outputLabels.join(',')}]` : 'R';
  const sumLatex = summedLabels.length ? `\\sum_{${summedLabels.join(',')}} ` : '';
  const factors = operandSubscripts.map((subs, index) => {
    const name = operandNames[index] || `T_{${index + 1}}`;
    return `${name}[${subs.split('').join(',')}]`;
  });
  return `${lhsLatex} = ${sumLatex}${factors.join(' \\cdot ')}`;
}

function buildFixedOutputEquation(example, outputValues) {
  const parts = getExampleExpressionParts(example);
  if (!parts) return null;
  const {
    operandSubscripts,
    operandNames,
    outputLabels,
    summedLabels,
  } = parts;

  const valueByOutputLabel = new Map(
    outputLabels.map((label, idx) => [label, outputValues[idx] ?? '?']),
  );
  const lhsLatex = outputLabels.length ? `R[${outputValues.join(',')}]` : 'R';
  const sumLatex = summedLabels.length ? `\\sum_{${summedLabels.join(',')}} ` : '';
  const factors = operandSubscripts.map((subs, index) => {
    const name = operandNames[index] || `T_{${index + 1}}`;
    const coords = subs
      .split('')
      .map((label) => (valueByOutputLabel.has(label) ? valueByOutputLabel.get(label) : label));
    return `${name}[${coords.join(',')}]`;
  });
  return `${lhsLatex} = ${sumLatex}${factors.join(' \\cdot ')}`;
}

function buildLabelValueMap(labelOrder, tuple) {
  const valueByLabel = new Map();
  (labelOrder ?? []).forEach((label, idx) => {
    valueByLabel.set(label, tuple?.[idx]);
  });
  return valueByLabel;
}

function buildValueMap(labels, values) {
  return new Map((labels ?? []).map((label, idx) => [label, values?.[idx]]));
}

function buildAssignmentLatex(labels, values) {
  if (!Array.isArray(labels) || !labels.length) return null;
  return labels
    .map((label, idx) => `${label} = ${values[idx] ?? '?'}`)
    .join(String.raw`,\; `);
}

function buildFormalOrbitExampleData({ example, labelOrder: labelOrderInput = [], witness }) {
  if (!example || !witness) return null;
  const parts = getExampleExpressionParts(example);
  if (!parts) return null;

  const { outputLabels, summedLabels } = parts;
  const labelOrder = Array.isArray(labelOrderInput) ? labelOrderInput : [];
  const valuesA = buildLabelValueMap(labelOrder, witness.tupleA);
  const valuesB = buildLabelValueMap(labelOrder, witness.tupleB);
  const outputValues = outputLabels.map((label) => valuesA.get(label) ?? '?');
  const summedValuesA = summedLabels.map((label) => valuesA.get(label) ?? '?');
  const summedValuesB = summedLabels.map((label) => valuesB.get(label) ?? '?');
  const outputEntryLatex = outputLabels.length ? `R[${outputValues.join(',')}]` : 'R';

  return {
    expandedEquation: buildExpandedEinsumEquation(example),
    outputAssignmentLatex: buildAssignmentLatex(outputLabels, outputValues),
    fixedOutputEquation: buildFixedOutputEquation(example, outputValues),
    outputValues,
    summedValuesA,
    summedValuesB,
    summedAssignmentA: buildAssignmentLatex(summedLabels, summedValuesA),
    summedAssignmentB: buildAssignmentLatex(summedLabels, summedValuesB),
    outputEntryLatex,
    outputTargetNoun: outputLabels.length ? 'output entry' : 'scalar output',
  };
}

function buildWorkedExampleFactors(example, outputValueMap = new Map(), summedValueMap = new Map()) {
  const parts = getExampleExpressionParts(example);
  if (!parts) return [];
  const outputSet = new Set(parts.outputLabels);
  const summedSet = new Set(parts.summedLabels);

  return parts.operandSubscripts.map((subs, index) => {
    const coords = subs.split('').map((label) => {
      if (outputValueMap.has(label)) return outputValueMap.get(label);
      if (summedValueMap.has(label)) return summedValueMap.get(label);
      return label;
    });
    const roles = subs.split('').map((label) => {
      if (outputSet.has(label)) return 'v';
      if (summedSet.has(label)) return 'w';
      return 'plain';
    });
    return {
      name: parts.operandNames[index] || `T_${index + 1}`,
      coords,
      roles,
    };
  });
}

function AppendixSupportSplit({
  article,
  support,
  strict = false,
  className = '',
  articleClassName = APPENDIX_ARTICLE_LANE_CLASS,
  supportClassName = APPENDIX_SUPPORT_SHELF_CLASS,
}) {
  return (
    <div
      className={[
        strict ? 'grid gap-y-6 gap-x-10 lg:grid-cols-2 lg:items-start' : 'grid gap-y-6 gap-x-10 xl:grid-cols-[0.95fr_1.25fr] xl:items-start',
        className,
      ].join(' ')}
    >
      <div className={['min-w-0', articleClassName].join(' ').trim()}>{article}</div>
      <div className={['min-w-0', supportClassName].join(' ').trim()}>{support}</div>
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

function WorkedExampleIndex({ value, role = 'plain' }) {
  const style =
    role === 'v' ? vStyle() :
    role === 'w' ? wStyle() :
    undefined;
  return <span style={style}>{value}</span>;
}

function WorkedExampleCoords({ coords = [], roles = [] }) {
  return coords.map((coord, idx) => (
    <span key={`${coord}-${idx}`}>
      {idx > 0 ? ',' : null}
      <WorkedExampleIndex value={coord} role={roles[idx] ?? 'plain'} />
    </span>
  ));
}

function WorkedExampleTensorRef({ name, coords = [], roles = [] }) {
  if (!coords.length) return <>{name}</>;
  return (
    <>
      {name}[<WorkedExampleCoords coords={coords} roles={roles} />]
    </>
  );
}

function WorkedExampleTensorProduct({ factors = [], scalarValues = null, total = null }) {
  return (
    <>
      {factors.map((factor, idx) => (
        <span key={`${factor.name}-${idx}`}>
          {idx > 0 ? ' · ' : null}
          <WorkedExampleTensorRef
            name={factor.name}
            coords={factor.coords}
            roles={factor.roles}
          />
        </span>
      ))}
      {Array.isArray(scalarValues) && scalarValues.length ? (
        <>
          {' = '}
          {scalarValues.map((value, idx) => (
            <span key={`${value}-${idx}`}>
              {idx > 0 ? ' · ' : null}
              {value}
            </span>
          ))}
          {total !== null ? (
            <>
              {' = '}
              <strong>{total}</strong>
            </>
          ) : null}
        </>
      ) : null}
    </>
  );
}

function WorkedExampleDisplayEquation({
  outputCoords = [],
  outputRoles = [],
  sumCoords = [],
  sumRoles = [],
  factors = [],
}) {
  return (
    <div className={`pl-0 sm:pl-4 ${APPENDIX_MONO_LEDGER_CLASS}`}>
      <div>
        <WorkedExampleTensorRef name="R" coords={outputCoords} roles={outputRoles} />
        {' = '}
        {sumCoords.length ? (
          <>
            ∑
            <sub className="text-[0.72em]">
              <WorkedExampleCoords coords={sumCoords} roles={sumRoles} />
            </sub>
            {' '}
          </>
        ) : null}
        <WorkedExampleTensorProduct factors={factors} />
      </div>
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

function AppendixTakeaway({ children }) {
  return (
    <EditorialCallout
      bodyClassName="mt-2 font-serif text-[16px] leading-[1.7] text-gray-900"
    >
      {children}
    </EditorialCallout>
  );
}

function AppendixDefinitionPanel({ children, className = '' }) {
  return (
    <EditorialCallout
      className={className}
      bodyClassName="mt-2 font-serif text-[17px] leading-[1.75] text-gray-900"
    >
      {children}
    </EditorialCallout>
  );
}

function AppendixRoadmap() {
  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-black bg-white px-4 py-3">
          <div className="font-mono text-[15px] font-semibold text-gray-900">
            <Latex math={notationLatex('g_pointwise')} />
          </div>
          <div className="mt-2 text-[12.5px] leading-6 text-gray-700">
            Summand-level symmetry. This is the group used by the main page&apos;s accumulation cost.
          </div>
        </div>
        <div className="rounded-lg border border-black bg-white px-4 py-3">
          <div className="font-mono text-[15px] font-semibold" style={{ color: notationColor('g_output') }}>
            <Latex math={notationLatex('g_output')} />
          </div>
          <div className="mt-2 text-[12.5px] leading-6 text-gray-700">
            The restriction <Latex math={String.raw`G_{\text{pt}}\|_V`} /> to output labels. This is the group used for output storage.
          </div>
        </div>
        <div className="rounded-lg border border-black bg-white px-4 py-3">
          <div className="font-mono text-[15px] font-semibold" style={{ color: notationColor('s_w_summed') }}>
            <Latex math={APPENDIX_SHORT_S_W_LATEX()} />
          </div>
          <div className="mt-2 text-[12.5px] leading-6 text-gray-700">
            These are alpha-renamings of bound summation variables: they preserve the completed expression by changing only bound-label names, not the summands themselves.
          </div>
        </div>
        <div className="rounded-lg border border-black bg-white px-4 py-3">
          <div className="font-mono text-[15px] font-semibold text-gray-900">
            <Latex math={`${notationLatex('g_formal')} = ${notationLatex('g_output')} \\times ${APPENDIX_SHORT_S_W_LATEX()}`} />
          </div>
          <div className="mt-2 text-[12.5px] leading-6 text-gray-700">
            The label-renaming symmetry of the completed expression. This explains expression symmetry, but it is not the accumulation-cost group.
          </div>
        </div>
      </div>
      <p className={APPENDIX_APP_TEXT_CLASS}>
        <InlineMathText>
          {String.raw`Throughout the appendix, $${notationLatex('l_labels')}$ denotes all labels in the contraction, $${APPENDIX_SHORT_V_LATEX()}$ denotes the free/output labels, and $${APPENDIX_W_EQUALS_L_MINUS_V_LATEX()}$ denotes the summed labels.`}
        </InlineMathText>
      </p>
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
  const runningExamplePreset = EXAMPLES_BY_ID.get('bilinear-trace');
  const selectedOperandItems = useMemo(
    () =>
      describeOperands(example).map((operand) => ({
        ...operand,
        chipName: operand.count > 1 ? `${operand.name}×${operand.count}` : operand.name,
      })),
    [example],
  );
  const alphaComparison = useMemo(
    () => computeExpressionAlphaComparison({ analysis, example }),
    [analysis, example],
  );
  const exampleExpressionParts = useMemo(
    () => getExampleExpressionParts(example),
    [example],
  );
  const formalOrbitExample = useMemo(
    () => buildFormalOrbitExampleData({ example, labelOrder: analysis?.symmetry?.allLabels ?? [], witness: alphaComparison.witness }),
    [analysis?.symmetry?.allLabels, alphaComparison.witness, example],
  );
  const runningExampleExpandedEquation = useMemo(
    () => buildExpandedEinsumEquation(runningExamplePreset),
    [runningExamplePreset],
  );
  const showBilinearFormalOrbitExample =
    example?.id === 'bilinear-trace' &&
    alphaComparison.state === 'mismatch' &&
    Boolean(alphaComparison.witness);
  const showDirectS2C3FormalOrbitExample =
    example?.id === 'direct-s2-c3' &&
    alphaComparison.state === 'mismatch' &&
    Boolean(alphaComparison.witness);
  const showMixedChainFormalOrbitExample =
    example?.id === 'mixed-chain' &&
    alphaComparison.state === 'mismatch' &&
    Boolean(alphaComparison.witness);

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
              Why expression symmetry is not the cost symmetry
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
              <SectionReferenceLink href="#cost-savings" beforeNavigate={onClose}>Section 5</SectionReferenceLink>
              <span>{' on the main page computed a symmetry-aware accumulation count. That count is deliberately based on pointwise equality of summands: two assignments may be identified only when they produce the same indexed product before summation. The completed einsum can have additional label-renaming symmetry after the summed labels have disappeared. This appendix separates those ideas: pointwise symmetry for accumulation, formal symmetry for the completed expression, and output symmetry for storage.'}</span>
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
          <AppendixRoadmap />

          <AppendixSection
            n={1}
            label="Pointwise group"
            anchorId="appendix-section-1"
            title="The main page computes a pointwise group"
            deck="The cost panels group together only those symmetries that preserve the indexed summand before summation."
            className="pt-10"
          >
            <AppendixSupportSplit
              strict={true}
              className="lg:grid-cols-[0.95fr_1.25fr]"
              articleClassName="space-y-4"
              supportClassName="space-y-4 xl:pt-1"
              article={(
                <>
                  <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                    <InlineMathText>
                      {`Let $${notationLatex('l_labels')}$ be the set of all labels in the einsum, $${APPENDIX_SHORT_V_LATEX()}$ the output labels, and $${APPENDIX_W_EQUALS_L_MINUS_V_LATEX()}$ the summed labels. The main page searches for symmetries by acting on the rows of the incidence matrix $${notationLatex('m_incidence')}$. A candidate row move $${notationLatex('sigma_row_move')}$ comes from repeated operands and declared operand symmetries. It is admissible only when the row-permuted matrix $${APPENDIX_M_SIGMA_LATEX}$ has column fingerprints that can be matched bijectively back to the original columns of $${notationLatex('m_incidence')}$.`}
                    </InlineMathText>
                  </p>
                  <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                    <InlineMathText>
                      {`When such a bijection exists, it induces a label relabeling $${APPENDIX_PI_SIGMA_LATEX}$. The group generated by the admissible relabelings is $${notationLatex('g_pointwise')}$, the pointwise symmetry group. This is the group used by the main page’s cost calculation, because its orbits contain assignments whose summand products are equal before the summation is performed.`}
                    </InlineMathText>
                  </p>
                  <div
                    data-takeaway={String.raw`G_{\text{pt}} is a cost group: it is valid for accumulation because it identifies genuinely equal indexed products.`}
                  >
                    <AppendixTakeaway>
                      <>
                        <Latex math={String.raw`G_{\text{pt}}`} />
                        <span>{' is a cost group: it is valid for accumulation because it identifies genuinely equal indexed products.'}</span>
                      </>
                    </AppendixTakeaway>
                  </div>
                </>
              )}
              support={(
                <div className="space-y-4 xl:pt-1">
                  <h4 className="font-heading text-[18px] font-semibold text-gray-900">
                    σ-loop audit
                  </h4>
                  <p className={APPENDIX_APP_TEXT_CLASS}>
                    <InlineMathText>
                      {String.raw`The table below is an implementation audit of the $\sigma$-loop for representative presets.`}
                    </InlineMathText>
                  </p>
                  <div className="overflow-x-auto">
                    <table className={`w-full border-collapse ${APPENDIX_SMALL_TEXT_CLASS} text-gray-700`}>
                      <thead className="border-y border-gray-200">
                        <tr className="text-left text-gray-500">
                          <th className="px-2 py-2 font-semibold text-gray-700">Preset</th>
                          <th className="px-2 py-2 font-semibold text-gray-700"><Latex math="L, V, W" /></th>
                          <th className="px-2 py-2 font-semibold text-gray-700"><Latex math="H_A, m_A" /></th>
                          <th className="px-2 py-2 font-semibold text-gray-700"><Latex math="|G_{\mathrm{wreath}}|" /></th>
                          <th className="px-2 py-2 font-semibold text-gray-700">Recorded</th>
                          <th className="px-2 py-2 font-semibold text-gray-700">Identity-only</th>
                          <th className="px-2 py-2 font-semibold text-gray-700">Rejected</th>
                          <th className="px-2 py-2 font-semibold text-gray-700"><Latex math="|G_{\text{pt}}|" /></th>
                        </tr>
                      </thead>
                      <tbody className="[&_tr]:border-b [&_tr]:border-gray-100">
                        <tr>
                          <td className="px-2 py-2">
                            <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('frobenius')} groupLabel="trivial" />
                          </td>
                          <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                          <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">2</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-[var(--status-success)]">0</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">2</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">1</td>
                        </tr>
                        <tr>
                          <td className="px-2 py-2">
                            <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('trace-product')} groupLabel="W: S2{i,j}" />
                          </td>
                          <td className="px-2 py-2"><Latex math="\{i,j\}, \varnothing, \{i,j\}" /></td>
                          <td className="px-2 py-2"><Latex math="\{e\}, 2" /></td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">2</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-[var(--status-success)]">1</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">2</td>
                        </tr>
                        <tr>
                          <td className="px-2 py-2">
                            <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('triangle')} groupLabel="C3{i,j,k}" />
                          </td>
                          <td className="px-2 py-2"><Latex math="\{i,j,k\}, \{i,j,k\}, \varnothing" /></td>
                          <td className="px-2 py-2"><Latex math="\{e\}, 3" /></td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">6</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-[var(--status-success)]">2</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-[var(--status-warning)]">3</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">3</td>
                        </tr>
                        <tr>
                          <td className="px-2 py-2">
                            <AppendixPresetHoverLabel preset={EXAMPLES_BY_ID.get('young-s3')} groupLabel="S3{i,j,k}" />
                          </td>
                          <td className="px-2 py-2"><Latex math="\{a,b,c\}, \{a,b\}, \{c\}" /></td>
                          <td className="px-2 py-2"><Latex math="S_3, 1" /></td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">6</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-[var(--status-success)]">5</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">1</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-500">0</td>
                          <td className="px-2 py-2 font-mono tabular-nums text-gray-900">6</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-4 space-y-3 border-t border-gray-200 pt-4">
                    <p className={APPENDIX_APP_TEXT_CLASS}>
                      <span className="font-semibold text-gray-900">Columns.</span>{' '}
                      <Latex math="L, V, W" />{' '}
                      <span>record all labels, output labels, and summed labels. </span>
                      <Latex math="H_A, m_A" />{' '}
                      <span>record the declared operand symmetry and operand multiplicity.</span>
                    </p>
                    <p className={APPENDIX_APP_TEXT_CLASS}>
                      <Latex math={String.raw`|G_{\mathrm{wreath}}|`} />{' '}
                      <span>counts candidate row moves before filtering. Recorded counts admissible non-identity relabelings, Identity-only counts admissible moves whose induced relabeling is the identity, Rejected counts failed fingerprint checks, and </span>
                      <Latex math={String.raw`|G_{\text{pt}}|`} />{' '}
                      <span>is the final detected pointwise group size.</span>
                    </p>
                    <p className={APPENDIX_APP_TEXT_CLASS}>
                      <span className="font-semibold text-gray-900">Frobenius.</span>{' '}
                      <InlineMathText>
                        {`The operand swap is admissible, but the induced relabeling stays the identity, so no non-identity pointwise symmetry is recorded.`}
                      </InlineMathText>
                    </p>
                    <p className={APPENDIX_APP_TEXT_CLASS}>
                      <span className="font-semibold text-gray-900">Triangle.</span>{' '}
                      <InlineMathText>
                        {`The same preset displays both accepted and rejected row moves, so the row-level boundary is visible in one table.`}
                      </InlineMathText>
                    </p>
                  </div>
                </div>
              )}
            />
          </AppendixSection>

          <AppendixSection
            n={2}
            label="Post-summation symmetry"
            anchorId="appendix-section-2"
            title="Summation creates a second kind of symmetry"
            deck={
              <InlineMathText>
                {`Once the $${APPENDIX_SHORT_W_LATEX()}$-labels have been summed over, they become dummy variables. Renaming them preserves the completed expression, even when it does not preserve individual summands.`}
              </InlineMathText>
            }
          >
            <AppendixSupportSplit
              strict={true}
              className="lg:grid-cols-[0.95fr_1.25fr]"
              articleClassName="space-y-4"
              supportClassName="space-y-4 xl:pt-1"
              article={(
                <>
                  <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                    <InlineMathText>
                      {`The pointwise group is not the only symmetry visible after the expression has been fully assembled. The labels in $${APPENDIX_SHORT_W_LATEX()}$ are bound summation variables. After the sum has been taken, a permutation of $${APPENDIX_SHORT_W_LATEX()}$ is an α-renaming of dummy variables: it changes the names used inside the summation, but not the value of the completed expression.`}
                    </InlineMathText>
                  </p>
                  <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                    <InlineMathText>
                      {`This gives a second group, $${notationLatex('s_w_summed')}$, the symmetric group on the summed labels. Its elements are formal symmetries of the completed expression. They are not, in general, pointwise symmetries of the summands.`}
                    </InlineMathText>
                  </p>
                  <AppendixTakeaway>
                    <InlineMathText>
                      {`$${notationLatex('s_w_summed')}$ is a post-summation symmetry. It preserves the completed sum, not necessarily the individual indexed products used to compute that sum.`}
                    </InlineMathText>
                  </AppendixTakeaway>
                </>
              )}
              support={(
                <div className="space-y-4">
                  <p className="text-[15px] font-semibold leading-7 text-gray-900">
                    Running example —{' '}
                    <AppendixPresetHoverLabel
                      preset={runningExamplePreset}
                      groupLabel={runningExamplePreset?.expectedGroup}
                      className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                    >
                      bilinear trace
                    </AppendixPresetHoverLabel>
                  </p>
                  <div className="inline-flex max-w-full rounded-xl border border-stone-200 bg-white px-3 py-2.5 shadow-sm">
                    <div className="font-mono text-[12px] leading-6 text-stone-900">
                      <FormulaHighlighted example={runningExamplePreset} hoveredLabels={null} />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                      <InlineMathText>
                        {`For the bilinear trace preset, the expression has output labels $${APPENDIX_SHORT_V_LATEX()} = \\{i,j\\}$ and summed labels $${APPENDIX_SHORT_W_LATEX()} = \\{k,l\\}$.`}
                      </InlineMathText>
                    </p>
                    {runningExampleExpandedEquation ? (
                      <div className="pl-0 sm:pl-4">
                        <div className={APPENDIX_PROSE_CLASS}>
                          <Latex math={runningExampleExpandedEquation} />
                        </div>
                      </div>
                    ) : null}
                    <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                      <InlineMathText>
                        {`The swap $(k\\;l)$ preserves the double sum as a formal expression. Under that swap, the summand $A[i,k]A[j,l]$ is carried to $A[i,l]A[j,k]$, so the transformed expression can be written as`}
                      </InlineMathText>
                    </p>
                    <div className="pl-0 sm:pl-4">
                      <div className={APPENDIX_PROSE_CLASS}>
                        <Latex math={String.raw`R[i,j] = \sum_{k,l} A[i,l] \cdot A[j,k]`} />
                      </div>
                    </div>
                    <p className={APPENDIX_PROSE_JUSTIFIED_CLASS}>
                      <InlineMathText>
                        {`Rename dummy variables back by swapping $k$ and $l$ inside the summation indices, and the completed double sum is unchanged. That is why $${notationLatex('s_w_summed')}$ is a symmetry of the finished expression even though it does not guarantee pointwise equality term by term.`}
                      </InlineMathText>
                    </p>
                  </div>
                </div>
              )}
            />
          </AppendixSection>

          <AppendixSection
            n={3}
            label="Output action"
            anchorId="appendix-section-3"
            title="The pointwise group also induces output symmetry"
            deck="Some pointwise relabelings act visibly on the output tensor. This visible action is the part that will later matter for storage."
          >
            {/* Source-contract marker: G_out is defined via the restriction pi|_V of G_pt to output labels. */}
            <AppendixSupportSplit
              articleClassName={APPENDIX_ARTICLE_LANE_CLASS}
              supportClassName="space-y-4 xl:pt-5"
              article={(
                <>
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`Among the elements of $${notationLatex('g_pointwise')}$, some preserve the output-label set $${APPENDIX_SHORT_V_LATEX()}$. Restricting those elements to $${APPENDIX_SHORT_V_LATEX()}$ gives the output group`}
                    </InlineMathText>
                  </p>

                  <div data-reader-facing-formula={APPENDIX_G_OUT_DEFINITION_LATEX()}>
                    <span className="sr-only">
                      {APPENDIX_G_OUT_DEFINITION_LATEX()}
                    </span>
                    <AppendixDefinitionPanel>
                      <Latex math={APPENDIX_G_OUT_DEFINITION_LATEX()} />
                    </AppendixDefinitionPanel>
                  </div>

                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`This group acts on output cells. If two output assignments lie in the same $${notationLatex('g_output')}$-orbit, the corresponding entries of the computed output tensor are equal. Unlike $${notationLatex('s_w_summed')}$, this symmetry is inherited from pointwise equality, so it is legitimate on the output tensor itself.`}
                    </InlineMathText>
                  </p>

                  <div className="mt-2">
                    <AppendixTakeaway>
                      <InlineMathText>
                        {`$${notationLatex('g_output')}$ is not an extra formal artifact. It is the visible output action inherited from $${notationLatex('g_pointwise')}$.`}
                      </InlineMathText>
                    </AppendixTakeaway>
                  </div>
                </>
              )}
              support={(
                /* Worked example — bilinear trace */
                <AppendixWorkedExample
                  preset={EXAMPLES_BY_ID.get('bilinear-trace')}
                  title="bilinear trace"
                  groupLabel={EXAMPLES_BY_ID.get('bilinear-trace')?.expectedGroup}
                  intro={
                    <>
                      <p>
                        <span>Let </span>
                        <Latex math={String.raw`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`} />
                        <span>.</span>
                      </p>
                      <p>
                        <InlineMathText>
                          {`For bilinear trace at $n = 2$, the detected pointwise symmetry swaps $i$ and $j$ together with the corresponding summed structure. Its restriction to $${APPENDIX_SHORT_V_LATEX()}$ is the transposition $(i\\;j)$, so the output tensor satisfies $R[i,j] = R[j,i]$.`}
                        </InlineMathText>
                      </p>
                      <p>
                        <InlineMathText>
                          {`A direct check shows $R[0,1] = R[1,0]$.`}
                        </InlineMathText>
                      </p>
                    </>
                  }
                >
                  <WorkedExampleEquationLedger>
                    <WorkedExampleEquation
                      assignment={
                        <>
                          R[<span style={vStyle()}>0</span>,<span style={vStyle()}>1</span>] =
                          A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>] + A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>] + A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>] + A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]
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
                          R[<span style={vStyle()}>1</span>,<span style={vStyle()}>0</span>] =
                          A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>] + A[<span style={vStyle()}>1</span>,<span style={wStyle()}>0</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>] + A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>0</span>] + A[<span style={vStyle()}>1</span>,<span style={wStyle()}>1</span>]·A[<span style={vStyle()}>0</span>,<span style={wStyle()}>1</span>]
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
                      {`The equality $R[0,1] = R[1,0]$ is therefore genuine on the computed output tensor itself.`}
                    </InlineMathText>
                  </WorkedExampleNote>
                </AppendixWorkedExample>
              )}
            />
          </AppendixSection>

          <AppendixSection
            n={4}
            label="Formal group"
            anchorId="appendix-section-4"
            title="The formal group is a product, not a cost group"
            deck="The completed expression combines the visible output action with dummy-label renamings."
          >
            <AppendixSupportSplit
              articleClassName={APPENDIX_ARTICLE_LANE_CLASS}
              supportClassName="space-y-4 xl:pt-1"
              article={(
                <>
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`We can now name the label-renaming formal group considered in this appendix. It has two independent factors: the visible output action $${notationLatex('g_output')}$, and the dummy-label renaming group $${notationLatex('s_w_summed')}$. Their direct product is`}
                    </InlineMathText>
                  </p>
                  <div className="py-1 text-center">
                    <Latex math={`${notationLatex('g_formal')} = ${notationLatex('g_output')} \\times ${APPENDIX_SHORT_S_W_LATEX()}`} />
                  </div>
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`An element of $${notationLatex('g_formal')}$ acts by applying an output relabeling on $${APPENDIX_SHORT_V_LATEX()}$ and an independent dummy renaming on $${APPENDIX_SHORT_W_LATEX()}$. This is a symmetry of the completed formal expression. It is not automatically a symmetry of the accumulation process.`}
                    </InlineMathText>
                  </p>
                  <AppendixTakeaway>
                    <InlineMathText>
                      {`$${notationLatex('g_formal')}$ explains label-renaming symmetry after summation. It should not be substituted for $${notationLatex('g_pointwise')}$ in the accumulation cost.`}
                    </InlineMathText>
                  </AppendixTakeaway>
                </>
              )}
              support={(
                <div>
                  <h4 className="font-heading text-[18px] font-semibold text-gray-900">
                    Formal-group construction
                  </h4>
                  <VSubSwConstruction
                    expressionGroup={expressionGroup}
                    vLabels={vLabels}
                    wLabels={wLabels}
                    showHeading={false}
                  />
                  <p className={`mt-4 ${APPENDIX_SMALL_TEXT_CLASS}`}>
                    The construction above enumerates the two factors and their product for the selected preset.
                  </p>
                </div>
              )}
            />
          </AppendixSection>

          <AppendixSection
            n={5}
            label="Accumulation boundary"
            anchorId="appendix-section-5"
            title="Why formal symmetry cannot replace pointwise symmetry in the cost"
            deck={
              <InlineMathText>
                {`The tempting shortcut is to apply Burnside to $${notationLatex('g_formal')}$. That shortcut is invalid for accumulation because formal orbits may contain unequal summands.`}
              </InlineMathText>
            }
          >
            {/* Source-contract marker: Let A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}. */}
            <AppendixSupportSplit
              articleClassName={APPENDIX_ARTICLE_LANE_CLASS}
              supportClassName="space-y-5 xl:pt-1"
              article={(
                <>
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {String.raw`The main page’s $\alpha$ counts accumulation representatives. Such a representative is valid only when every assignment in its orbit contributes the same indexed product. That condition holds for $${notationLatex('g_pointwise')}$ by construction. It need not hold for $${notationLatex('g_formal')}$, because the $${notationLatex('s_w_summed')}$ factor can rename dummy variables only after the terms have already been summed.`}
                    </InlineMathText>
                  </p>
                  <p className={APPENDIX_PROSE_CLASS}>
                    <InlineMathText>
                      {`Therefore, Burnside over $${notationLatex('g_formal')}$ answers a different question: it counts orbits of formal label renamings in the completed expression. It does not, in general, count the number of distinct summand computations required by the accumulation.`}
                    </InlineMathText>
                  </p>

                  <EditorialCallout bodyClassName="mt-2 font-serif text-[17px] leading-[1.75] text-gray-900">
                    <div>
                      <InlineMathText>
                        {`Use $${notationLatex('g_pointwise')}$ for accumulation. Use $${notationLatex('g_formal')}$ to describe formal symmetry of the completed expression. Do not use the dummy-label factor $${notationLatex('s_w_summed')}$ to remove summand computations.`}
                      </InlineMathText>
                    </div>
                  </EditorialCallout>
                </>
              )}
              support={(
                <div className="space-y-5">
                  {example ? (
                    <div className="space-y-2">
                      <p className={APPENDIX_APP_TEXT_STRONG_CLASS}>
                        <span className="font-semibold text-gray-900">Selected einsum:</span>{' '}
                        <span className="font-mono text-[13px] leading-6 text-stone-900">
                          <FormulaHighlighted example={example} hoveredLabels={null} />
                        </span>
                      </p>
                      {selectedOperandItems.length ? (
                        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[12px] leading-6 text-stone-700">
                          {selectedOperandItems.map((operand) => (
                            <SymmetryChip key={`${operand.name}-${operand.sym}`} name={operand.chipName} symmetry={operand.sym} />
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {onSelectPreset && BURNSIDE_GAP_PRESETS.length ? (
                    <div className="space-y-3">
                      <p className={APPENDIX_SMALL_TEXT_CLASS}>
                        Presets with a visible mismatch:
                      </p>
                      <div className="flex flex-wrap gap-2">
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

                  <div className="space-y-5">
                    {alphaComparison.state === 'mismatch' ? (
                      <>
                        <p className={APPENDIX_PROSE_CLASS}>
                          <InlineMathText>
                            {String.raw`For the selected preset, the difference is visible numerically. A naive formal count using $${notationLatex('g_formal')}$ gives $\alpha_{\text{formal}} = ${alphaComparison.exprAlpha}$. The pointwise accumulation count used by the engine is $\alpha_{\text{engine}} = ${alphaComparison.correctAlpha}$. The mismatch is not a bug; it is exactly the gap between formal equivalence and pointwise equality.`}
                          </InlineMathText>
                        </p>

                        {alphaComparison.witness ? (
                          showBilinearFormalOrbitExample ? (
                            <div className="space-y-4 border-y border-gray-200 py-5">
                              <p className="text-[15px] font-semibold leading-7 text-gray-900">
                                Worked example — bilinear trace
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <span>Let </span>
                                <Latex math={String.raw`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`} />
                                <span>.</span>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Then the expanded form of the einsum is:`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={['i', 'j']}
                                outputRoles={['v', 'v']}
                                sumCoords={['k', 'l']}
                                sumRoles={['w', 'w']}
                                factors={[
                                  { name: 'A', coords: ['i', 'k'], roles: ['v', 'w'] },
                                  { name: 'A', coords: ['j', 'l'], roles: ['v', 'w'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <span>Now set </span>
                                <Latex math={String.raw`i = 0,\; j = 1`} />
                                <span>.</span>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={[0, 1]}
                                outputRoles={['v', 'v']}
                                sumCoords={['k', 'l']}
                                sumRoles={['w', 'w']}
                                factors={[
                                  { name: 'A', coords: [0, 'k'], roles: ['v', 'w'] },
                                  { name: 'A', coords: [1, 'l'], roles: ['v', 'w'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Swapping the summed labels $k$ and $l$ sends $(0,1)$ to $(1,0)$, so $${notationLatex('g_formal')}$ places these two assignments in the same formal orbit.`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleEquationLedger>
                                <WorkedExampleEquation
                                  assignment={
                                    <InlineMathText>{`If $k = 0$ and $l = 1$, then`}</InlineMathText>
                                  }
                                  numeric={
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'A', coords: [0, 0], roles: ['v', 'w'] },
                                        { name: 'A', coords: [1, 1], roles: ['v', 'w'] },
                                      ]}
                                      scalarValues={[1, 4]}
                                      total={4}
                                    />
                                  }
                                />
                                <WorkedExampleEquation
                                  assignment={
                                    <InlineMathText>{`If $k = 1$ and $l = 0$, then`}</InlineMathText>
                                  }
                                  numeric={
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'A', coords: [0, 1], roles: ['v', 'w'] },
                                        { name: 'A', coords: [1, 0], roles: ['v', 'w'] },
                                      ]}
                                      scalarValues={[2, 3]}
                                      total={6}
                                    />
                                  }
                                />
                              </WorkedExampleEquationLedger>
                              <WorkedExampleNote>
                                <InlineMathText>
                                  {`Both terms contribute to the same output entry, and both must be accumulated. Formal symmetry has related them, but it has not made them equal.`}
                                </InlineMathText>
                              </WorkedExampleNote>
                            </div>
                          ) : showDirectS2C3FormalOrbitExample ? (
                            <div className="space-y-4 border-y border-gray-200 py-5">
                              <p className="text-[15px] font-semibold leading-7 text-gray-900">
                                Worked example —{' '}
                                <AppendixPresetHoverLabel
                                  preset={example}
                                  groupLabel={example?.expectedGroup}
                                  className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                                >
                                  {example?.name ?? 'selected preset'}
                                </AppendixPresetHoverLabel>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Let $T[0,0,0,1,2] = 1$ and $T[0,0,0,2,1] = 2$, and let every other entry of $T$ be $0$.`}
                                </InlineMathText>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Then the expanded form of the einsum is:`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={['a', 'b']}
                                outputRoles={['v', 'v']}
                                sumCoords={['c', 'd', 'e']}
                                sumRoles={['w', 'w', 'w']}
                                factors={[
                                  { name: 'T', coords: ['a', 'b', 'c', 'd', 'e'], roles: ['v', 'v', 'w', 'w', 'w'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <span>Now set </span>
                                <Latex math={String.raw`a = 0,\; b = 0`} />
                                <span>.</span>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={[0, 0]}
                                outputRoles={['v', 'v']}
                                sumCoords={['c', 'd', 'e']}
                                sumRoles={['w', 'w', 'w']}
                                factors={[
                                  { name: 'T', coords: [0, 0, 'c', 'd', 'e'], roles: ['v', 'v', 'w', 'w', 'w'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`The transposition $(d\,e)$ is allowed as a dummy relabeling in $${notationLatex('g_formal')}$, so it sends $(c,d,e) = (0,1,2)$ to $(0,2,1)$. That move is formal, but it is not a pointwise symmetry of the summands.`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleEquationLedger>
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>{`If $c = 0$, $d = 1$, and $e = 2$, then`}</InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'T', coords: [0, 0, 0, 1, 2], roles: ['v', 'v', 'w', 'w', 'w'] },
                                      ]}
                                      total={1}
                                    />
                                  )}
                                />
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>{`If $c = 0$, $d = 2$, and $e = 1$, then`}</InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'T', coords: [0, 0, 0, 2, 1], roles: ['v', 'v', 'w', 'w', 'w'] },
                                      ]}
                                      total={2}
                                    />
                                  )}
                                />
                              </WorkedExampleEquationLedger>
                              <WorkedExampleNote>
                                <InlineMathText>
                                  {String.raw`Both terms contribute to the same output entry $R[0,0]$, but $1 \neq 2$. Formal symmetry has related them, but it has not made them equal.`}
                                </InlineMathText>
                              </WorkedExampleNote>
                            </div>
                          ) : showMixedChainFormalOrbitExample ? (
                            <div className="space-y-4 border-y border-gray-200 py-5">
                              <p className="text-[15px] font-semibold leading-7 text-gray-900">
                                Worked example —{' '}
                                <AppendixPresetHoverLabel
                                  preset={example}
                                  groupLabel={example?.expectedGroup}
                                  className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                                >
                                  {example?.name ?? 'selected preset'}
                                </AppendixPresetHoverLabel>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <span>Let </span>
                                <Latex math={String.raw`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`} />
                                <span> and </span>
                                <Latex math={String.raw`B = \begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix}`} />
                                <span>.</span>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Then the expanded form of the einsum is:`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={['i', 'l']}
                                outputRoles={['v', 'v']}
                                sumCoords={['j', 'k']}
                                sumRoles={['w', 'w']}
                                factors={[
                                  { name: 'A', coords: ['i', 'j'], roles: ['v', 'w'] },
                                  { name: 'B', coords: ['j', 'k'], roles: ['w', 'w'] },
                                  { name: 'A', coords: ['k', 'l'], roles: ['w', 'v'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <span>Now set </span>
                                <Latex math={String.raw`i = 0,\; l = 0`} />
                                <span>.</span>
                              </p>
                              <WorkedExampleDisplayEquation
                                outputCoords={[0, 0]}
                                outputRoles={['v', 'v']}
                                sumCoords={['j', 'k']}
                                sumRoles={['w', 'w']}
                                factors={[
                                  { name: 'A', coords: [0, 'j'], roles: ['v', 'w'] },
                                  { name: 'B', coords: ['j', 'k'], roles: ['w', 'w'] },
                                  { name: 'A', coords: ['k', 0], roles: ['w', 'v'] },
                                ]}
                              />
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Swapping the summed labels $j$ and $k$ sends $(0,1)$ to $(1,0)$, so $${notationLatex('g_formal')}$ places these two assignments in the same formal orbit.`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleEquationLedger>
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>{`If $j = 0$ and $k = 1$, then`}</InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'A', coords: [0, 0], roles: ['v', 'w'] },
                                        { name: 'B', coords: [0, 1], roles: ['w', 'w'] },
                                        { name: 'A', coords: [1, 0], roles: ['w', 'v'] },
                                      ]}
                                      scalarValues={[1, 2, 3]}
                                      total={6}
                                    />
                                  )}
                                />
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>{`If $j = 1$ and $k = 0$, then`}</InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={[
                                        { name: 'A', coords: [0, 1], roles: ['v', 'w'] },
                                        { name: 'B', coords: [1, 0], roles: ['w', 'w'] },
                                        { name: 'A', coords: [0, 0], roles: ['w', 'v'] },
                                      ]}
                                      scalarValues={[2, 4, 1]}
                                      total={8}
                                    />
                                  )}
                                />
                              </WorkedExampleEquationLedger>
                              <WorkedExampleNote>
                                <InlineMathText>
                                  {String.raw`Both terms contribute to the same output entry $R[0,0]$, but $6 \neq 8$. Formal symmetry has related them, but it has not made them equal.`}
                                </InlineMathText>
                              </WorkedExampleNote>
                            </div>
                          ) : (
                            <div className="space-y-4 border-y border-gray-200 py-5">
                              <p className="text-[15px] font-semibold leading-7 text-gray-900">
                                Worked example —{' '}
                                <AppendixPresetHoverLabel
                                  preset={example}
                                  groupLabel={example?.expectedGroup}
                                  className="font-semibold cursor-help border-b border-dotted border-gray-300 text-left"
                                >
                                  {example?.name ?? 'selected preset'}
                                </AppendixPresetHoverLabel>
                              </p>
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`Then the expanded form of the einsum is:`}
                                </InlineMathText>
                              </p>
                              {exampleExpressionParts ? (
                                <WorkedExampleDisplayEquation
                                  outputCoords={exampleExpressionParts.outputLabels}
                                  outputRoles={exampleExpressionParts.outputLabels.map(() => 'v')}
                                  sumCoords={exampleExpressionParts.summedLabels}
                                  sumRoles={exampleExpressionParts.summedLabels.map(() => 'w')}
                                  factors={buildWorkedExampleFactors(example)}
                                />
                              ) : null}
                              {formalOrbitExample?.outputAssignmentLatex ? (
                                <>
                                  <p className={APPENDIX_PROSE_CLASS}>
                                    <span>Now set </span>
                                    <Latex math={formalOrbitExample.outputAssignmentLatex} />
                                    <span>.</span>
                                  </p>
                                  {exampleExpressionParts && formalOrbitExample.outputValues ? (
                                    <WorkedExampleDisplayEquation
                                      outputCoords={formalOrbitExample.outputValues}
                                      outputRoles={exampleExpressionParts.outputLabels.map(() => 'v')}
                                      sumCoords={exampleExpressionParts.summedLabels}
                                      sumRoles={exampleExpressionParts.summedLabels.map(() => 'w')}
                                      factors={buildWorkedExampleFactors(
                                        example,
                                        buildValueMap(exampleExpressionParts.outputLabels, formalOrbitExample.outputValues),
                                      )}
                                    />
                                  ) : null}
                                </>
                              ) : null}
                              <p className={APPENDIX_PROSE_CLASS}>
                                <InlineMathText>
                                  {`A formal relabeling on the summed labels places these two assignments in the same formal orbit.`}
                                </InlineMathText>
                              </p>
                              <WorkedExampleEquationLedger>
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>
                                      {`If $${formalOrbitExample?.summedAssignmentA ?? ''}$, then`}
                                    </InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={buildWorkedExampleFactors(
                                        example,
                                        buildValueMap(exampleExpressionParts?.outputLabels, formalOrbitExample?.outputValues),
                                        buildValueMap(exampleExpressionParts?.summedLabels, formalOrbitExample?.summedValuesA),
                                      )}
                                    />
                                  )}
                                />
                                <WorkedExampleEquation
                                  assignment={(
                                    <InlineMathText>
                                      {`If $${formalOrbitExample?.summedAssignmentB ?? ''}$, then`}
                                    </InlineMathText>
                                  )}
                                  numeric={(
                                    <WorkedExampleTensorProduct
                                      factors={buildWorkedExampleFactors(
                                        example,
                                        buildValueMap(exampleExpressionParts?.outputLabels, formalOrbitExample?.outputValues),
                                        buildValueMap(exampleExpressionParts?.summedLabels, formalOrbitExample?.summedValuesB),
                                      )}
                                    />
                                  )}
                                />
                              </WorkedExampleEquationLedger>
                              <WorkedExampleNote>
                                <InlineMathText>
                                  {`Both assignments contribute to the same ${formalOrbitExample?.outputTargetNoun ?? 'output entry'} $${formalOrbitExample?.outputEntryLatex ?? 'R'}$, but they remain different pointwise products and must be counted separately.`}
                                </InlineMathText>
                              </WorkedExampleNote>
                            </div>
                          )
                        ) : null}
                      </>
                    ) : null}

                    {alphaComparison.state === 'coincident' ? (
                      <p className={APPENDIX_PROSE_CLASS}>
                        <InlineMathText>
                          {`For the selected preset, the formal count and the pointwise count happen to agree at the current size. This numerical coincidence does not change the rule. The valid accumulation group is still $${notationLatex('g_pointwise')}$, because only $${notationLatex('g_pointwise')}$ is guaranteed to identify equal summands.`}
                        </InlineMathText>
                      </p>
                    ) : null}

                    {alphaComparison.state === 'none' ? (
                      <p className={APPENDIX_PROSE_CLASS}>
                        <InlineMathText>
                          {`For the selected preset, the formal group does not produce a distinct accumulation count to compare. The conceptual distinction remains the same: $${notationLatex('g_formal')}$ describes post-summation label-renaming symmetry, while $${notationLatex('g_pointwise')}$ is the group used for accumulation.`}
                        </InlineMathText>
                      </p>
                    ) : null}
                  </div>
                </div>
              )}
            />
          </AppendixSection>

          <AppendixSection
            n={6}
            label="Storage"
            anchorId="appendix-section-6"
            title="The valid leftover optimization is output storage"
            deckClassName="max-w-none"
            deck={
              <InlineMathText>
                {`After accumulation has been decided, $${notationLatex('g_output')}$ can still reduce mirrored output cells. The dummy-label factor $${APPENDIX_SHORT_S_W_LATEX()}$ cannot.`}
              </InlineMathText>
            }
          >
            {/* Source-contract marker: Accumulation is governed by G_pt. */}
            <div className="space-y-4">
              <p className={APPENDIX_PROSE_CLASS}>
                <InlineMathText>
                  {`The failure of $${notationLatex('g_formal')}$ as an accumulation group does not mean the additional structure is useless. It means the optimization must be placed on the correct axis.`}
                </InlineMathText>
              </p>
              <p className={APPENDIX_PROSE_CLASS}>
                <InlineMathText>
                  {String.raw`Accumulation is governed by $${notationLatex('g_pointwise')}$. Output storage is governed by $${notationLatex('g_output')}$. Once an output value has been computed, all output cells in the same $${notationLatex('g_output')}$-orbit contain the same value. A storage layout may therefore keep one representative per output orbit and recover the mirrored entries by symmetry.`}
                </InlineMathText>
              </p>
              <p className={APPENDIX_PROSE_CLASS}>
                <InlineMathText>
                  {`The dummy-label group $${notationLatex('s_w_summed')}$ contributes nothing to output storage, because its labels do not index output cells. They have already been summed away.`}
                </InlineMathText>
              </p>
            </div>

            <div className="mt-6 overflow-x-auto">
              <table className="w-full border-collapse text-[12px]">
                <thead className="border-b border-gray-200">
                  <tr className="text-left text-[12px] text-muted-foreground">
                    <th className="px-2 py-2 font-semibold">Preset</th>
                    <th className="px-2 py-2 font-semibold">Einsum</th>
                    <th className="px-2 py-2 font-semibold">Operand sym.</th>
                    <th className="px-2 py-2 font-semibold"><Latex math={APPENDIX_SHORT_V_LATEX()} /></th>
                    <th className="px-2 py-2 font-semibold"><Latex math={notationLatex('g_output')} /></th>
                    <th className="px-2 py-2 font-semibold text-right">
                      <div className="text-[13px] font-semibold text-gray-900">
                        <Latex math="\alpha_{\text{engine}}" />
                      </div>
                      <div className="mt-1 text-[11px] font-normal leading-5 text-gray-500">Accumulation representatives</div>
                    </th>
                    <th className="px-2 py-2 font-semibold text-right">
                      <div className="text-[13px] font-semibold text-gray-900">
                        <Latex math="\alpha_{\text{storage}}" />
                      </div>
                      <div className="mt-1 text-[11px] font-normal leading-5 text-gray-500">Output-storage representatives</div>
                    </th>
                    <th className="px-2 py-2 font-semibold text-right">
                      <div>Storage-only saving</div>
                    </th>
                  </tr>
                </thead>
                <tbody className="[&_tr]:border-b [&_tr]:border-gray-100">
                  {SAVINGS_TABLE_ROWS.map((r) => {
                    const hasSaving = r.saving > 0;
                    const preset = EXAMPLES_BY_ID.get(r.id);
                    const subs = preset?.expression?.subscripts ?? '';
                    const output = preset?.expression?.output ?? '';
                    const operands = describeOperands(preset);
                    const operandChips = operands.map((operand) => ({
                      ...operand,
                      chipName: operand.count > 1 ? `${operand.name}×${operand.count}` : operand.name,
                    }));
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
                        <td className="px-2 py-2">
                          <div className="flex flex-wrap gap-1.5">
                            {operandChips.map((operand) => (
                              <SymmetryChip key={`${operand.name}-${operand.sym}`} name={operand.chipName} symmetry={operand.sym} />
                            ))}
                          </div>
                        </td>
                        <td className="px-2 py-2 whitespace-nowrap">
                          <Latex math={r.v === '\\varnothing' ? '\\varnothing' : `\\{${r.v}\\}`} />
                        </td>
                        <td className="px-2 py-2 whitespace-nowrap"><Latex math={r.vSub} /></td>
                        <td className="px-2 py-2 text-right font-mono">{r.ae}</td>
                        <td className="px-2 py-2 text-right font-mono">{r.as}</td>
                        <td className={`px-2 py-2 text-right font-mono whitespace-nowrap ${hasSaving ? 'text-[var(--status-success)]' : ''}`}>
                          {hasSaving ? `${r.saving} (${r.pct}%)` : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className={`mt-2 ${APPENDIX_FOOTNOTE_CLASS}`}>
                All rows are computed at <Latex math="n = 3" />. Savings here are storage-only savings after accumulation has already been accounted for.
              </p>
            </div>

            <p className="mt-4 text-[12.5px] leading-6 text-stone-700">
              <InlineMathText>
                {`The table below records the storage-only savings available for the presets at $n = 3$. The column $\\alpha_{\\text{engine}}$ is the accumulation representative count used by the main page. The column $\\alpha_{\\text{storage}}$ is the number of output-storage representatives after grouping output cells into $${notationLatex('g_output')}$-orbits. These are different quantities; $\\alpha_{\\text{storage}}$ is not a replacement for the accumulation cost.`}
              </InlineMathText>
            </p>

            <div className="-mx-6 -mb-10 mt-8 border-t border-stone-200/70 bg-gray-50 px-6 py-4 md:-mx-8 md:px-8 lg:-mx-10 lg:px-10">
              <div className="font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
                SCOPE
              </div>
              <p className="mt-1.5 text-[12.5px] leading-6 text-stone-700">
                <InlineMathText>
                  {String.raw`The $\alpha$ shown on the main page counts distinct accumulation operations in the enumerate-and-accumulate evaluation model with $${notationLatex('g_pointwise')}$ as the summand-value equivalence relation.`}
                </InlineMathText>
              </p>
              <p className="mt-2 text-[12.5px] leading-6 text-stone-700">
                <InlineMathText>
                  {String.raw`Output-tensor storage collapse, algebraic restructuring such as factoring $R = v v^\top$, and contraction re-ordering all sit outside that scope and require different machinery than the pointwise orbit compression measured on the main page.`}
                </InlineMathText>
              </p>
            </div>
          </AppendixSection>
        </div>
      </div>
    </div>
  );
}
