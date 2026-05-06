import { useState } from 'react';
import CaseBadge from './CaseBadge.jsx';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import OrbitInspector from './OrbitInspector.jsx';
import RoleBadge from './RoleBadge.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';
import NotationSymbol from './NotationSymbol.jsx';
import { AnchorLink } from './ExplorerSectionCard.jsx';
import DecisionLadder from './DecisionLadder.jsx';
import PanZoomCanvas from './PanZoomCanvas.jsx';
import ExplorerModal from './ExplorerModal.jsx';
import MultiplicationCostCard from './MultiplicationCostCard.jsx';
import AccumulationHardCard from './AccumulationHardCard.jsx';
import ExplorerSubsectionHeader from './ExplorerSubsectionHeader.jsx';
import BranchingDemo from './BranchingDemo.jsx';
import UnavailableDetailsPanel from './UnavailableDetailsPanel.jsx';
import { getRegimePresentation } from './regimePresentation.js';
import {
  explorerThemeColor,
  explorerThemeTint,
} from '../lib/explorerTheme.js';
import {
  getActiveExplorerThemeId,
  notationColor,
  notationColoredLatex,
  notationLatex,
} from '../lib/notationSystem.js';

function isTrivial(comp) {
  return comp.shape === 'trivial';
}

// Per-component orbit count M_a, sourced from the engine field that
// `decomposeClassifyAndCount` now populates. The value is what the Act 5
// hero formula multiplies into ∏_a M_a.
function multiplicationCount(comp) {
  return comp.multiplication?.count ?? null;
}

function accumulationCount(comp) {
  return comp.accumulation?.count ?? null;
}

function accumulationFormula(comp) {
  return comp.accumulation?.latex ?? null;
}

function supportsOrbitEnumeration(comp) {
  return !isTrivial(comp) && comp.accumulation?.regimeId === 'bruteForceOrbit';
}

function denseTupleCount(comp, dimensionN) {
  const sizes = Array.isArray(comp.sizes) && comp.sizes.length > 0
    ? comp.sizes
    : Array(comp.labels?.length ?? 0).fill(dimensionN);
  return sizes.reduce((product, size) => product * size, 1);
}

/**
 * Method description: split "Technique — reason" on the em-dash and render
 * the technique bold + the reason in muted body. Inline math goes through the
 * shared notation-aware renderer so the regime copy stays color-consistent.
 */
function MethodDescription({ text }) {
  if (typeof text !== 'string' || !text.includes('—')) {
    return (
      <p className="text-[12.5px] leading-snug text-foreground">
        <InlineMathText>{text}</InlineMathText>
      </p>
    );
  }
  const [head, ...rest] = text.split('—');
  const technique = head.trim();
  const reason = rest.join('—').trim();
  return (
    <p className="text-[12.5px] leading-snug text-foreground">
      <span className="font-semibold"><InlineMathText>{technique}</InlineMathText></span>
      <span className="text-muted-foreground"> — </span>
      <span className="text-stone-700"><InlineMathText>{reason}</InlineMathText></span>
    </p>
  );
}

function LabelsCell({ comp }) {
  const orderedLabels = comp.labels ?? [];

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {orderedLabels.map((label) => {
        const role = (comp.va ?? []).includes(label) ? 'v' : 'w';
        return (
          <RoleBadge key={`${comp.shape ?? 'comp'}-${label}`} role={role}>
            {label}
          </RoleBadge>
        );
      })}
    </div>
  );
}

// V_a / W_a split cell — warm tones for V_free, cool tones for W_summed.
// Uses the same RoleBadge as LabelsCell so the coloring is consistent.
function VWSplitCell({ comp }) {
  const va = comp.va ?? [];
  const wa = comp.wa ?? [];
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-1">
        {va.length > 0 ? (
          va.map((label) => (
            <RoleBadge key={`va-${label}`} role="v">{label}</RoleBadge>
          ))
        ) : (
          <span className="text-[11px] text-muted-foreground">∅</span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1">
        {wa.length > 0 ? (
          wa.map((label) => (
            <RoleBadge key={`wa-${label}`} role="w">{label}</RoleBadge>
          ))
        ) : (
          <span className="text-[11px] text-muted-foreground">∅</span>
        )}
      </div>
    </div>
  );
}

// Heuristic: classify a refused-trace entry into one of the V3.1 §49 buckets.
// We look at the most-specific (latest) refusal entry that isn't the
// 'fallthrough' sentinel — its `reason` string carries the actionable signal.
function classifyFailedCondition(trace) {
  const list = Array.isArray(trace) ? trace : [];
  const declined = [...list]
    .reverse()
    .find((entry) => entry?.decision === 'refused' && entry.regimeId !== 'fallthrough');
  const reason = (declined?.reason ?? '').toLowerCase();
  if (reason.includes('partition')) return 'typed-partition';
  if (reason.includes('brute-force') || reason.includes('brute force') || reason.includes('tuple')) {
    return 'brute-force';
  }
  return 'no-shortcut';
}

// ─── Direction 1 redesign — Component as paragraph ─────────────────────────
// The earlier 7-column grid (labels | V_a/W_a | G_a | method | M_a | α_a |
// savings) crammed long method prose and dense-baseline numbers into ~60-char
// columns; on wide presets like Trilinear Trace each row was 13 lines tall
// and read like a spreadsheet trying to be a story. The redesign replaces
// each row with a self-contained ComponentCard:
//
//   ┌─ HEADER ──────────────────────────────────────────────────┐
//   │  CASE n · {regime label}                  [CaseBadge sm]  │
//   ├─ BODY (2-col on md+, stack on small) ────────────────────┤
//   │  IDENTITY                  │  METHOD                      │
//   │  Component  i j k l m n    │  Every detected symmetry…    │
//   │  Free V_a   i j k          │                              │
//   │  Sum  W_a   l m n          │       α_a = M_a = |X/G_a|    │
//   │  Group G_a  PermGroup⟨⟩    │                              │
//   │                            │  Enumerate orbits →          │
//   ├─ METRICS TAPE ────────────────────────────────────────────┤
//   │  M_a  2,925 / 15,625   α_a  2,925 / 15,625   ◖80% saved◗  │
//   └───────────────────────────────────────────────────────────┘
//
// Hover-bus integration: the whole card is the activeComponentId target;
// the method block inside is the activeAlphaMethod target. C39 + C29 buses
// keep firing on the same prop callbacks as before.

// Small kicker label used inside cards ("IDENTITY" / "METHOD" headings).
function CardKicker({ children, className = '' }) {
  return (
    <div
      className={`text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500 ${className}`}
    >
      {children}
    </div>
  );
}

// One field row inside the Identity column: small left label + value.
// Editorial register: kicker on left (~78px), value flows right.
function IdentityField({ label, children }) {
  return (
    <div className="grid grid-cols-[78px_minmax(0,1fr)] items-baseline gap-x-3">
      <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-500">
        {label}
      </span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

// One metric cell in the bottom tape: label kicker + primary value + dense
// baseline reference. Designed to read top-to-bottom in a 3-column tape.
// `valueTitle` and `denseTitle` carry the V3.1 Part-6 tooltip wording so
// the explanatory copy survives the redesign — same titles the original
// 7-column row carried, just on a different surface.
function MetricCell({
  label,
  value,
  dense,
  valueTitle = null,
  denseTitle = null,
  unavailableTitle = null,
}) {
  return (
    <div className="flex flex-col gap-1 items-start">
      <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-500">
        {label}
      </span>
      {value !== null ? (
        <div className="flex items-baseline gap-1.5">
          <code
            className="font-mono text-[15px] font-semibold text-foreground"
            title={valueTitle ?? undefined}
          >
            {value.toLocaleString()}
          </code>
          {dense != null ? (
            <span
              className="font-mono text-[11px] text-muted-foreground/70"
              title={denseTitle ?? undefined}
            >
              of {dense.toLocaleString()} dense
            </span>
          ) : null}
        </div>
      ) : (
        <span
          className="cursor-help rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800"
          title={unavailableTitle ?? undefined}
        >
          Unavailable
        </span>
      )}
    </div>
  );
}

// One ComponentCard renders all the data the old 7-column row carried for a
// single component, but laid out as identity ⟂ method with a metrics tape
// underneath. Wider presets like Trilinear Trace get the room they need;
// single-component presets read cleanly as a single card.
//
// Props:
//   comp                          — the component data object
//   index                         — 0-based card index (for the "Case n" header)
//   dimensionN                    — current dimension n (for dense baseline)
//   orbitRows                     — optional orbit list (for "Enumerate orbits")
//   explorerThemeId               — active theme (for savings pill colors)
//   onOpenOrbitModal              — callback when "Enumerate orbits" clicked
//   onActiveComponentHoverChange  — C20 hover bus
//   onActiveAlphaMethodHoverChange — C29 hover bus
//   onDimensionNChange            — C49 "Try smaller n" CTA support
function ComponentCard({
  comp,
  index,
  dimensionN,
  orbitRows,
  explorerThemeId,
  onOpenOrbitModal,
  onActiveComponentHoverChange,
  onActiveAlphaMethodHoverChange,
  onDimensionNChange,
}) {
  const M_a = multiplicationCount(comp);
  const canOpenOrbits = supportsOrbitEnumeration(comp) && (orbitRows?.length ?? 0) > 0;
  const denseCell = denseTupleCount(comp, dimensionN);
  const actualAcc = accumulationCount(comp);
  const pct = (actual, dense) =>
    dense > 0 ? Math.max(0, Math.round((1 - actual / dense) * 100)) : null;
  const multSavingsPct = M_a !== null ? pct(M_a, denseCell) : null;
  const accSavingsPct = actualAcc !== null ? pct(actualAcc, denseCell) : null;
  const totalSavingsPct = (M_a !== null && actualAcc !== null)
    ? pct(M_a + actualAcc, 2 * denseCell)
    : null;

  const leafId = comp.accumulation?.regimeId ?? comp.shape;
  const presentation = getRegimePresentation(leafId);
  const methodDescription = presentation?.tooltip?.body;
  const methodLatex = presentation?.tooltip?.latex;
  const caseLabel = presentation?.label ?? leafId ?? `Case ${index + 1}`;

  // The componentId is the comma-joined sorted label list, matching the
  // format used by C20 LabelInteractionGraph.
  const componentId = (comp.labels ?? []).join(',');

  const va = comp.va ?? [];
  const wa = comp.wa ?? [];

  return (
    <article
      className="overflow-hidden rounded-lg border border-gray-200 bg-white transition-shadow focus-within:shadow-sm hover:shadow-sm"
      tabIndex={0}
      role="row"
      aria-label={`Component ${componentId || index + 1}: ${caseLabel}`}
      onMouseEnter={() => onActiveComponentHoverChange?.(componentId)}
      onMouseLeave={() => onActiveComponentHoverChange?.(null)}
      onFocus={() => onActiveComponentHoverChange?.(componentId)}
      onBlur={() => onActiveComponentHoverChange?.(null)}
    >
      {/* ─── HEADER ─── */}
      <header className="flex flex-wrap items-center gap-3 border-b border-gray-100 bg-gray-50/40 px-5 py-3">
        {/* Tiny coloured dot anchored to the regime color — same hue the
            DecisionLadder uses for this leaf, so cards and tree stay
            visually linked without duplicating the badge label. */}
        <span
          aria-hidden="true"
          className="inline-block size-2 shrink-0 rounded-full"
          style={{ backgroundColor: presentation?.color ?? 'var(--gray-400)' }}
        />
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-500">
          Case {index + 1}
        </span>
        <span aria-hidden="true" className="text-gray-300">·</span>
        <span className="text-[14px] font-semibold leading-snug text-gray-900">
          {caseLabel}
        </span>
      </header>

      {/* ─── BODY: 2-column on md+, stacked on narrow ─── */}
      <div className="grid grid-cols-1 gap-x-8 gap-y-5 px-5 py-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        {/* IDENTITY column */}
        <div className="space-y-3">
          <CardKicker>Identity</CardKicker>
          <div className="space-y-2.5">
            <IdentityField label="Component">
              <LabelsCell comp={comp} />
            </IdentityField>
            <IdentityField label={<>Free <span title="Free labels Vₐ">V<sub>a</sub></span></>}>
              {va.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1">
                  {va.map((label) => (
                    <RoleBadge key={`va-${label}`} role="v">{label}</RoleBadge>
                  ))}
                </div>
              ) : (
                <span className="text-[12px] text-muted-foreground">∅</span>
              )}
            </IdentityField>
            <IdentityField label={<>Sum <span title="Summed labels Wₐ">W<sub>a</sub></span></>}>
              {wa.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1">
                  {wa.map((label) => (
                    <RoleBadge key={`wa-${label}`} role="w">{label}</RoleBadge>
                  ))}
                </div>
              ) : (
                <span className="text-[12px] text-muted-foreground">∅</span>
              )}
            </IdentityField>
            <IdentityField label={<>Group <span title="Local symmetry group Gₐ">G<sub>a</sub></span></>}>
              <SymmetryBadge value={comp.groupName || 'trivial'} />
            </IdentityField>
          </div>
        </div>

        {/* METHOD column — α-method hover-bus target */}
        <div
          className="space-y-3"
          tabIndex={0}
          role="button"
          aria-label={`Alpha method: ${caseLabel}`}
          onMouseEnter={() => onActiveAlphaMethodHoverChange?.(leafId)}
          onMouseLeave={() => onActiveAlphaMethodHoverChange?.(null)}
          onFocus={() => onActiveAlphaMethodHoverChange?.(leafId)}
          onBlur={() => onActiveAlphaMethodHoverChange?.(null)}
        >
          <CardKicker>Method</CardKicker>
          {methodDescription ? (
            <p className="font-serif text-[14.5px] leading-[1.65] text-gray-800">
              <InlineMathText>{methodDescription}</InlineMathText>
            </p>
          ) : null}
          {methodLatex ? (
            <div className="math-display-row mt-1 flex justify-center overflow-x-auto py-1 text-[14px] text-foreground">
              <Latex math={methodLatex} display />
            </div>
          ) : null}
          {canOpenOrbits ? (
            <button
              type="button"
              className="inline-flex cursor-pointer items-center gap-1 text-[12px] font-medium text-primary underline decoration-primary/40 decoration-dotted underline-offset-[3px] transition-colors hover:decoration-primary"
              onClick={() => onOpenOrbitModal?.(comp)}
            >
              Enumerate orbits →
            </button>
          ) : null}
        </div>
      </div>

      {/* ─── METRICS TAPE ─── */}
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 border-t border-gray-100 bg-gray-50/40 px-5 py-3.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.9fr)]">
        <MetricCell
          label={<>Product Orbits (<NotationSymbol id="m_component" mode="math" />)</>}
          value={M_a}
          dense={denseCell}
          valueTitle={`Dense tuple count = ${denseCell.toLocaleString()}, the full assignment-space product of n_ell before any symmetry collapse.`}
          denseTitle={`Dense tuple count = ${denseCell.toLocaleString()}, the full assignment-space product of n_ell before any symmetry collapse.`}
        />
        <MetricCell
          label={<>Accumulation Updates (<NotationSymbol id="alpha_component" mode="math" />)</>}
          value={actualAcc}
          dense={denseCell}
          denseTitle="Dense baseline: one update per full assignment before quotienting by the pointwise group."
          unavailableTitle={(() => {
            // Mirror the prior row-level α-unavailable tooltip: surface the
            // most-specific refusal entry from the regime trace.
            const trace = comp.accumulation?.trace ?? [];
            const declined = [...trace]
              .reverse()
              .find((t) => t.decision === 'refused' && t.regimeId !== 'fallthrough');
            const reason = declined?.reason ?? 'no regime fired';
            return `αₐ withheld: ${reason}.`;
          })()}
        />
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-500">
            Savings vs dense
          </span>
          {totalSavingsPct !== null ? (
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-1.5">
                <span
                  className="rounded-full px-2 py-0.5 font-mono text-[12px] font-semibold"
                  style={{
                    background: totalSavingsPct > 0
                      ? explorerThemeTint(explorerThemeId, 'quantity', 0.12)
                      : explorerThemeTint(explorerThemeId, 'freeSide', 0.12),
                    color: totalSavingsPct > 0
                      ? explorerThemeColor(explorerThemeId, 'quantity')
                      : explorerThemeColor(explorerThemeId, 'freeSide'),
                  }}
                  title={`Per-component direct savings: multiplication uses ${M_a?.toLocaleString?.() ?? '—'} product orbits and accumulation uses ${actualAcc?.toLocaleString?.() ?? '—'} stored-output-representative updates, compared with ${denseCell.toLocaleString()} dense assignments.`}
                >
                  {totalSavingsPct}% total
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-x-2 font-mono text-[11px] leading-tight">
                {multSavingsPct !== null ? (
                  <span
                    className="font-semibold"
                    style={{ color: notationColor('m_component') }}
                    title={`Mult savings: dense M_a would be ${denseCell.toLocaleString()}; symmetry gives ${M_a?.toLocaleString?.()}.`}
                  >
                    Mult {multSavingsPct}%
                  </span>
                ) : null}
                {multSavingsPct !== null && accSavingsPct !== null ? (
                  <span className="text-stone-300" aria-hidden="true">·</span>
                ) : null}
                {accSavingsPct !== null ? (
                  <span
                    className="font-semibold"
                    style={{ color: notationColor('alpha_component') }}
                    title={`Acc savings: dense alpha_a would be ${denseCell.toLocaleString()}; symmetry gives ${actualAcc?.toLocaleString?.()}.`}
                  >
                    Acc {accSavingsPct}%
                  </span>
                ) : null}
              </div>
            </div>
          ) : (
            <span className="text-[12px] text-muted-foreground">—</span>
          )}
        </div>
      </div>

      {/* V3.1 §49 — when α is unavailable, expose the verbose details panel
          beneath the metrics tape. Wrapped in a <details>/<summary> for free
          a11y; the panel itself carries the live numbers + CTAs. */}
      {actualAcc === null ? (
        <details className="group border-t border-gray-100 bg-amber-50/30 px-5 py-2.5">
          <summary
            className="cursor-pointer select-none text-[11px] font-medium text-muted-foreground hover:text-foreground"
            aria-label={`Show unavailable count details for component ${componentId || index + 1}`}
          >
            <span className="mr-1 inline-block transition-transform group-open:rotate-90" aria-hidden="true">▸</span>
            why is this unavailable?
          </summary>
          <div className="mt-2">
            <UnavailableDetailsPanel
              componentId={componentId}
              sizes={Array.isArray(comp.sizes) && comp.sizes.length > 0
                ? comp.sizes
                : Array(comp.labels?.length ?? 0).fill(dimensionN)}
              groupSize={comp.elements?.length ?? 1}
              failedCondition={classifyFailedCondition(comp.accumulation?.trace)}
              onLowerN={typeof onDimensionNChange === 'function' ? onDimensionNChange : null}
              currentN={dimensionN}
            />
          </div>
        </details>
      ) : null}
    </article>
  );
}

function ComponentSummaryTable({
  components,
  dimensionN,
  orbitRows,
  onOpenOrbitModal,
  onActiveComponentHoverChange,
  onActiveAlphaMethodHoverChange,
  onDimensionNChange,
}) {
  const explorerThemeId = getActiveExplorerThemeId();

  return (
    <div className="space-y-4 bg-white">
      <p className="px-1 text-[12px] leading-5 text-muted-foreground">
        The dense baseline is the same direct-event convention without symmetry: one product chain and one output update for every full label assignment. Each component below is one independent factor; multiply across them for the global counts.
      </p>

      {components.map((comp, idx) => (
        <ComponentCard
          key={`comp-${idx}`}
          comp={comp}
          index={idx}
          dimensionN={dimensionN}
          orbitRows={orbitRows}
          explorerThemeId={explorerThemeId}
          onOpenOrbitModal={onOpenOrbitModal}
          onActiveComponentHoverChange={onActiveComponentHoverChange}
          onActiveAlphaMethodHoverChange={onActiveAlphaMethodHoverChange}
          onDimensionNChange={onDimensionNChange}
        />
      ))}
    </div>
  );
}

// Compact stat read next to the card's title. Three numbers that answer
// "how big / connected is this graph" at a glance, in the same visual
// register as the rest of Act 4 (uppercase eyebrow + mono numeral).
function InteractionGraphMetricStrip({ labelCount, edgeCount, componentCount }) {
  const cells = [
    { label: 'labels', value: labelCount },
    { label: 'edges', value: edgeCount },
    { label: 'components', value: componentCount },
  ];
  return (
    <div className="mt-3 flex flex-wrap items-stretch gap-1.5">
      {cells.map((cell) => (
        <div
          key={cell.label}
          className="flex flex-col items-center rounded-md border border-border/70 bg-surface-raised px-2 py-1"
        >
          <span className="font-mono text-sm font-semibold leading-tight text-foreground">
            {cell.value}
          </span>
          <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {cell.label}
          </span>
        </div>
      ))}
    </div>
  );
}

// Inline legend for the graph's visual vocabulary. The V/W dot colors here
// must match LabelInteractionGraph's COLOR_V / COLOR_W so the legend stays
// truthful.
function InteractionGraphLegend() {
  const explorerThemeId = getActiveExplorerThemeId();
  const freeLabelColor = explorerThemeColor(explorerThemeId, 'hero');
  const summedLabelColor = explorerThemeColor(explorerThemeId, 'summedSide');
  const hullColor = explorerThemeColor(explorerThemeId, 'heroMuted');
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-full" style={{ backgroundColor: freeLabelColor }} />
        free label <span style={{ color: freeLabelColor, textTransform: 'none' }}>(<Latex math={notationLatex('v_free')} inheritColor />)</span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2.5 rounded-full" style={{ backgroundColor: summedLabelColor }} />
        summed label <span style={{ color: summedLabelColor, textTransform: 'none' }}>(<Latex math={notationLatex('w_summed')} inheritColor />)</span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-px w-5" style={{ backgroundColor: '#6B7280' }} />
        edge: co-permuted by some <Latex math={`${notationColoredLatex('sigma_row_move')} \\in ${notationColoredLatex('g_detected')}`} />
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block size-2.5 rounded-sm border"
          style={{ borderStyle: 'dashed', borderColor: hullColor }}
        />
        hull: one independent component
      </span>
    </div>
  );
}

export default function ComponentCostView({
  componentData,
  costModel,
  dimensionN,
  numTerms = 2,
  allLabels,
  vLabels,
  fullGenerators,
  selectedOrbitIdx,
  onSelectOrbit,
  onGraphHover,
  spotlightLeafIds,
  expressionInfo = null,
  hoveredLabels = null,
  showBranchingDemo = true,
  showCostCards = true,
  showDecisionLadder = true,
  // C39: hover buses — write activeComponentId / activeAlphaMethod on hover.
  // Setters are passed from App-level state (C20 and C01 respectively).
  onActiveComponentHoverChange = null,
  onActiveAlphaMethodHoverChange = null,
  // C49: dimension-N setter; threaded through to UnavailableDetailsPanel so
  // the "Try n = …" CTA can lower n when an unavailable state fires.
  // Optional — when null, the panel hides the lower-n button.
  onDimensionNChange = null,
}) {
  if (!componentData || !costModel) return null;

  const [showOrbitModal, setShowOrbitModal] = useState(false);
  const [orbitModalComponent, setOrbitModalComponent] = useState(null);
  const components = componentData.components ?? [];
  const orbitRows = costModel.orbitRows ?? [];

  return (
    <div className="min-w-0 space-y-6">
      {showBranchingDemo ? (
        /* ROW 0 — BranchingDemo lifted to right after the §4 intro so the matrix
            visualizes "product orbit" and "stored output rep" the moment the prose
            introduces those terms. The cost cards / classification tree / summary
            follow below. */
        <div id="demos-1col" className="border-b border-gray-100 pb-6">
          <BranchingDemo
            dimensionN={dimensionN}
            componentData={componentData}
            costModel={costModel}
            selectedOrbitIdx={selectedOrbitIdx}
            onSelectOrbit={onSelectOrbit}
            onHover={onGraphHover}
            expressionInfo={expressionInfo}
            hoveredLabels={hoveredLabels}
          />
        </div>
      ) : null}

      {showCostCards ? (
        /* ROW 2 — μ vs α-hard (the right contrast) */
        <div id="two-cost-cards" className="editorial-two-col-divider-lg editorial-two-col-divider-lg-inset border-y border-gray-100 py-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <MultiplicationCostCard
            components={components.map((comp) => ({
              ...comp,
              multiplicationCount: multiplicationCount(comp),
            }))}
            numTerms={numTerms}
          />
          <AccumulationHardCard />
        </div>
      ) : null}

      {showDecisionLadder ? (
        /* ROW 3 — Classification tree (roadmap) */
        <div id="classification-tree" className="bg-white p-4 scroll-mt-sticky">
          <ExplorerSubsectionHeader anchorId="classification-tree" labelText="Classification Tree">
            Classification Tree
          </ExplorerSubsectionHeader>
          <p className="explorer-support-prose mt-2">
            Each component is routed through a yes/no spine that dispatches to the
            cheapest applicable closed form, or to brute-force orbit projection
            when nothing else fits. The highlighted leaf on the left is where the
            current example lands.
          </p>
          <div className="mt-4">
            <DecisionLadder
              activeLeafIds={components
                .flatMap((c) => [c.accumulation?.regimeId, c.shape])
                .filter(Boolean)}
              spotlightLeafIds={spotlightLeafIds}
              liveReasonsByLeaf={(() => {
                const map = new Map();
                for (const comp of components) {
                  const trace = comp.accumulation?.trace ?? [];
                  for (const step of trace) {
                    if (!step?.regimeId || !step?.reason) continue;
                    const list = map.get(step.regimeId) ?? [];
                    if (!list.includes(step.reason)) list.push(step.reason);
                    map.set(step.regimeId, list);
                  }
                }
                return map;
              })()}
            />
          </div>
        </div>
      ) : null}

      {/* ROW 6 — Per-component summary table */}
      <ComponentSummaryTable
        components={components}
        dimensionN={dimensionN}
        orbitRows={orbitRows}
        onOpenOrbitModal={(comp) => {
          setOrbitModalComponent(comp);
          setShowOrbitModal(true);
        }}
        onActiveComponentHoverChange={onActiveComponentHoverChange}
        onActiveAlphaMethodHoverChange={onActiveAlphaMethodHoverChange}
        onDimensionNChange={onDimensionNChange}
      />

      <ExplorerModal
        title="Orbit Enumeration"
        titleId="orbit-inspector-modal-title"
        open={showOrbitModal}
        onClose={() => {
          setShowOrbitModal(false);
          setOrbitModalComponent(null);
        }}
      >
        <OrbitInspector
          orbitRows={orbitRows}
          selectedOrbitIdx={selectedOrbitIdx}
          onSelectOrbit={onSelectOrbit}
          showHeader={false}
          formulaMath={null}
          dimensionN={dimensionN}
          componentContext={orbitModalComponent}
        />
      </ExplorerModal>
    </div>
  );
}
