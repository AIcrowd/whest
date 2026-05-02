/**
 * TwoQuotientSchematic — §4 Rows and Columns
 *
 * Teaches the two quotients that define the O → Q matrix:
 *
 *   Product assignments X -- quotient by G_pt --> product-orbit rows O
 *   Output assignments Y  -- quotient by H ----> stored-output columns Q
 *   Projection π_V connects rows to columns as a reach relation.
 *
 * The primary tab reflects the current selected preset; the reference tabs are
 * fixed microscopes for the three edge cases the narrative uses repeatedly.
 */

import { useEffect, useMemo, useState } from 'react';

const TOKEN = {
  coral: 'var(--coral)',
  coralLight: 'var(--coral-light)',
  gray900: 'var(--gray-900)',
  gray700: 'var(--gray-700)',
  gray600: 'var(--gray-600)',
  gray500: 'var(--gray-500)',
  gray400: 'var(--gray-400)',
  gray300: 'var(--gray-300)',
  gray200: 'var(--gray-200)',
  gray100: 'var(--gray-100)',
  gray50: 'var(--gray-50)',
  white: 'var(--white)',
  einV: 'var(--ein-v)',
  einW: 'var(--ein-w)',
};

const REFERENCE_PRESETS = {
  crossS2: {
    id: 'crossS2',
    tabLabel: 'Cross S₂',
    title: 'Cross S₂ reference',
    caption: 'H is trivial. Columns are ordinary output assignments, and some product rows branch to multiple columns.',
    dimensionN: 2,
    hStatus: 'H trivial',
    projectionStatus: 'projection branches',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '4 columns' },
      { label: 'α', value: '5 filled cells' },
    ],
    productAssignments: ['(0,0,0)', '(0,1,0)', '(1,0,0)'],
    productRows: ['O₀', 'O₁', 'O₂'],
    outputAssignments: ['(0,0)', '(0,1)', '(1,0)'],
    outputColumns: ['Q₀', 'Q₁', 'Q₂'],
    reachTargets: 2,
  },
  bilinearTrace: {
    id: 'bilinearTrace',
    tabLabel: 'Bilinear trace',
    title: 'Bilinear trace reference',
    caption: 'H is nontrivial. Some output assignments share one stored representative, and projection is functional.',
    dimensionN: 2,
    hStatus: 'H nontrivial',
    projectionStatus: 'projection functional',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '3 columns' },
      { label: 'α', value: '3 filled cells' },
    ],
    productAssignments: ['(0,0,0,0)', '(0,1,0,1)', '(1,0,1,0)'],
    productRows: ['O₀', 'O₁', 'O₂'],
    outputAssignments: ['(0,0)', '(0,1)', '(1,0)'],
    outputColumns: ['Q₀', 'Q₁=(0,1)/(1,0)', 'Q₂'],
    reachTargets: 1,
  },
  tripleOuter: {
    id: 'tripleOuter',
    tabLabel: 'Triple outer',
    title: 'Triple outer reference',
    caption: 'All labels are visible. Projection drops nothing, so product-orbit rows and stored-output columns line up.',
    dimensionN: 2,
    hStatus: 'H nontrivial',
    projectionStatus: 'rows and columns line up',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '3 columns' },
      { label: 'α', value: '3 filled cells' },
    ],
    productAssignments: ['(0,0)', '(0,1)', '(1,0)'],
    productRows: ['O₀', 'O₁', 'O₂'],
    outputAssignments: ['(0,0)', '(0,1)', '(1,0)'],
    outputColumns: ['Q₀', 'Q₁', 'Q₂'],
    reachTargets: 1,
  },
};

const PRESET_ORDER = ['crossS2', 'bilinearTrace', 'tripleOuter'];
function normalizePresetId(name = '') {
  const normalized = String(name)
    .toLowerCase()
    .replaceAll('₂', '2')
    .replaceAll('₃', '3')
    .replaceAll('_', ' ')
    .replaceAll('-', ' ');
  if (normalized.includes('cross') && normalized.includes('s2')) return 'crossS2';
  if (normalized.includes('bilinear') && normalized.includes('trace')) return 'bilinearTrace';
  if (normalized.includes('triple') && normalized.includes('outer')) return 'tripleOuter';
  return null;
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e) => setReduced(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return reduced;
}

function formatNumber(value, fallback = '—') {
  return Number.isFinite(value) ? value.toLocaleString() : fallback;
}

function formatTuple(tuple) {
  if (Array.isArray(tuple)) return `(${tuple.join(',')})`;
  if (tuple && typeof tuple === 'object') {
    return `(${Object.entries(tuple).map(([key, value]) => `${key}=${value}`).join(', ')})`;
  }
  return String(tuple ?? '—');
}

function uniqueOutputSamples(orbitRows = []) {
  const seen = new Map();
  for (const row of orbitRows) {
    for (const output of row?.outputs ?? []) {
      const key = output?.outKey ?? formatTuple(output?.outTuple);
      if (!seen.has(key)) seen.set(key, formatTuple(output?.outTuple ?? key));
    }
  }
  return [...seen.values()];
}

function takeSamples(values, fallback) {
  const list = values.filter(Boolean).slice(0, 3);
  return list.length ? list : fallback;
}

function buildCurrentView(current) {
  const orbitRows = current?.orbitRows ?? [];
  const productAssignments = takeSamples(
    orbitRows[0]?.orbitTuples?.map(formatTuple) ?? [],
    ['full assignments in X'],
  );
  const productRows = takeSamples(
    orbitRows.map((row, index) => `O${index}: ${formatTuple(row?.repTuple)}`),
    ['product-orbit rows O'],
  );
  const outputSamples = uniqueOutputSamples(orbitRows);
  const outputAssignments = takeSamples(outputSamples, ['output assignments in Y']);
  const outputColumns = takeSamples(
    outputSamples.map((sample, index) => `Q${index}: ${sample}`),
    ['stored-output columns Q'],
  );
  const rowCount = current?.rowCount ?? orbitRows.length;
  const alpha = current?.alpha;
  const columnCount = current?.columnCount ?? outputSamples.length;
  const branchRows = current?.branchRows
    ?? orbitRows.filter((row) => (row?.outputCount ?? row?.outputs?.length ?? 0) > 1).length;
  const hSize = current?.hSize ?? 1;
  const hStatus = hSize > 1 ? 'H nontrivial' : 'H trivial';
  const projectionStatus = branchRows > 0 ? 'projection branches' : 'projection functional';
  const captionLead = hSize > 1
    ? branchRows > 0
      ? 'H is nontrivial; projection branches'
      : 'H is nontrivial; projection is functional'
    : branchRows > 0
      ? 'H is trivial; projection branches'
      : 'H is trivial; projection is functional';
  const presetName = current?.presetName ?? 'selected preset';
  const dimensionN = current?.dimensionN;

  return {
    id: 'current',
    referenceId: normalizePresetId(presetName),
    tabLabel: `Current: ${presetName}`,
    title: `Current preset — ${presetName}`,
    caption: `${captionLead}. These counts come from the selected contraction${Number.isFinite(dimensionN) ? ` at n=${dimensionN}` : ''}.`,
    dimensionN,
    hStatus,
    projectionStatus,
    metrics: [
      { label: 'n', value: formatNumber(dimensionN) },
      { label: 'M', value: `${formatNumber(rowCount)} rows` },
      { label: '|Y/H|', value: `${formatNumber(columnCount)} columns` },
      { label: 'α', value: `${formatNumber(alpha)} filled cells` },
    ],
    productAssignments,
    productRows,
    outputAssignments,
    outputColumns,
    reachTargets: branchRows > 0 ? 2 : 1,
  };
}

function Kicker({ children }) {
  return (
    <div
      className="font-sans text-[10px] font-semibold uppercase text-gray-400"
      style={{ letterSpacing: '0.18em' }}
    >
      {children}
    </div>
  );
}

function metricListForView(view) {
  const hasN = view.metrics.some((metric) => metric.label === 'n');
  return Number.isFinite(view.dimensionN) && !hasN
    ? [{ label: 'n', value: formatNumber(view.dimensionN) }, ...view.metrics]
    : view.metrics;
}

const METRIC_HELPER = {
  n: 'dimension',
  M: 'product rows',
  '|Y/H|': 'stored columns',
  α: 'filled cells',
};

function splitMetricValue(value) {
  const match = String(value).match(/^([0-9,]+|—)(?:\s+(.+))?$/);
  return match ? { number: match[1], unit: match[2] ?? '' } : { number: String(value), unit: '' };
}

function MetricStrip({ view }) {
  const metrics = metricListForView(view);
  return (
    <dl
      className="grid grid-cols-1 rounded-[var(--radius-md)] border-y border-gray-200 bg-white/70 py-1 sm:grid-cols-4 sm:py-3"
      aria-label="Current two-quotient metrics"
    >
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className="min-w-0 border-t border-gray-100 px-3 py-2 first:border-t-0 sm:border-l sm:border-t-0 sm:border-gray-200 sm:py-0 sm:first:border-l-0"
        >
          <dt className="font-sans text-[10px] font-semibold uppercase text-gray-400" style={{ letterSpacing: '0.14em' }}>
            <span className="font-mono text-[11px] normal-case tracking-normal text-gray-600">{metric.label}</span>
            <span className="ml-1">{METRIC_HELPER[metric.label]}</span>
          </dt>
          <dd className="mt-1 flex min-w-0 items-baseline gap-1.5 font-mono font-semibold text-gray-900">
            {(() => {
              const { number, unit } = splitMetricValue(metric.value);
              return (
                <>
                  <span className="text-[18px] leading-6">{number}</span>
                  {unit ? <span className="text-[11px] leading-4 text-gray-500">{unit}</span> : null}
                </>
              );
            })()}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function SampleList({ items }) {
  return (
    <div className="mt-3 grid gap-1.5">
      {items.slice(0, 3).map((item, index) => (
        <div
          key={`${item}-${index}`}
          className="min-w-0 rounded-[var(--radius-sm)] border border-gray-200 bg-gray-50 px-2 py-1 font-mono text-[11px] leading-4 text-gray-600"
          title={item}
        >
          <span className="block truncate">{item}</span>
        </div>
      ))}
    </div>
  );
}

function QuotientBox({ title, formula, items, active = false, result = false, reducedMotion }) {
  return (
    <div
      className="min-w-0 rounded-[var(--radius-lg)] border bg-white p-3"
      style={{
        borderColor: active ? TOKEN.coral : TOKEN.gray200,
        background: active ? TOKEN.coralLight : TOKEN.white,
        transition: reducedMotion ? 'none' : 'border-color 160ms ease, background 160ms ease',
      }}
    >
      <div className="font-sans text-[13px] font-semibold leading-5 text-gray-900">{title}</div>
      <div
        className="mt-1 font-mono text-[13px] font-semibold leading-5"
        style={{ color: active ? TOKEN.coral : result ? TOKEN.gray900 : TOKEN.gray700 }}
      >
        {formula}
      </div>
      <SampleList items={items} />
    </div>
  );
}

function ProcessArrow({ label, detail, active, onHoverStart, onHoverEnd, reducedMotion }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      aria-label={`Highlight ${label}`}
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      onFocus={onHoverStart}
      onBlur={onHoverEnd}
      className="group flex min-w-[7rem] flex-col items-center justify-center gap-1 self-stretch rounded-[var(--radius-md)] px-2 py-2 text-center focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--coral)]/20"
      style={{ color: active ? TOKEN.coral : TOKEN.gray600 }}
    >
      <div className="flex w-full items-center gap-2 max-md:flex-col">
        <div className="h-px flex-1 bg-gray-300 max-md:h-5 max-md:w-px max-md:flex-none" />
        <div
          className="rounded-full border bg-white px-2 py-1 font-sans text-[11px] font-semibold leading-4"
          style={{
            borderColor: active ? TOKEN.coral : TOKEN.gray200,
            transition: reducedMotion ? 'none' : 'border-color 160ms ease',
          }}
        >
          {label}
        </div>
        <div className="flex items-center gap-1 max-md:flex-col">
          <div className="h-px w-5 bg-gray-300 max-md:h-5 max-md:w-px" />
          <div
            className="h-0 w-0 border-y-[5px] border-l-[8px] border-y-transparent max-md:rotate-90"
            style={{ borderLeftColor: active ? TOKEN.coral : TOKEN.gray400 }}
            aria-hidden
          />
        </div>
      </div>
      <div className="max-w-[11rem] font-sans text-[10.5px] leading-4 text-gray-500">{detail}</div>
    </button>
  );
}

function QuotientLane({
  kicker,
  fromTitle,
  fromFormula,
  fromItems,
  toTitle,
  toFormula,
  toItems,
  processLabel,
  processDetail,
  active,
  onHoverStart,
  onHoverEnd,
  reducedMotion,
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-gray-200 bg-white p-3 md:p-4">
      <Kicker>{kicker}</Kicker>
      <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(8rem,10rem)_minmax(0,1fr)] md:items-center">
        <QuotientBox
          title={fromTitle}
          formula={fromFormula}
          items={fromItems}
          reducedMotion={reducedMotion}
        />
        <ProcessArrow
          label={processLabel}
          detail={processDetail}
          active={active}
          onHoverStart={onHoverStart}
          onHoverEnd={onHoverEnd}
          reducedMotion={reducedMotion}
        />
        <QuotientBox
          title={toTitle}
          formula={toFormula}
          items={toItems}
          active={active}
          result
          reducedMotion={reducedMotion}
        />
      </div>
    </div>
  );
}

function ReachRelationPanel({ view, active, onHoverStart, onHoverEnd, reducedMotion }) {
  const outputTargets = view.outputColumns.slice(0, Math.max(1, Math.min(3, view.reachTargets)));
  const reachesMany = view.reachTargets > 1;

  return (
    <div
      className="rounded-[var(--radius-lg)] border border-dashed bg-white p-3 md:p-4"
      style={{
        borderColor: active ? TOKEN.coral : TOKEN.gray300,
        background: active ? TOKEN.coralLight : TOKEN.white,
        transition: reducedMotion ? 'none' : 'border-color 160ms ease, background 160ms ease',
      }}
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <Kicker>Projection reach</Kicker>
          <p className="mt-2 font-serif text-[15px] leading-7 text-gray-700">
            Projection is not a third quotient. It asks which stored columns each product-orbit row reaches after dropping summed labels.
          </p>
        </div>
        <button
          type="button"
          aria-pressed={active}
          aria-label="Highlight projection reach relation"
          className="self-start rounded-full border bg-white px-3 py-1.5 font-sans text-[12px] font-semibold focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--coral)]/20"
          style={{ borderColor: active ? TOKEN.coral : TOKEN.gray200, color: active ? TOKEN.coral : TOKEN.gray600 }}
          onMouseEnter={onHoverStart}
          onMouseLeave={onHoverEnd}
          onFocus={onHoverStart}
          onBlur={onHoverEnd}
        >
          π_V reach
        </button>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] md:items-center">
        <div className="rounded-[var(--radius-md)] border border-gray-200 bg-gray-50 px-3 py-2">
          <div className="font-sans text-[11px] font-semibold uppercase text-gray-400" style={{ letterSpacing: '0.14em' }}>One row</div>
          <div className="mt-1 truncate font-mono text-[12px] font-semibold text-gray-900">{view.productRows[0] ?? 'O'}</div>
        </div>
        <div className="flex items-center justify-center gap-2 text-center max-md:flex-col">
          <div className="h-px w-8 bg-gray-300 max-md:h-5 max-md:w-px" />
          <div className="font-sans text-[11px] font-semibold text-gray-500">
            {reachesMany ? 'may fill multiple cells' : 'fills one cell here'}
          </div>
          <div className="h-px w-8 bg-gray-300 max-md:h-5 max-md:w-px" />
        </div>
        <div className="grid gap-1.5">
          {outputTargets.map((target, index) => (
            <div
              key={`${target}-${index}`}
              className="rounded-[var(--radius-sm)] border bg-white px-2 py-1 font-mono text-[11px] font-semibold text-gray-700"
              style={{ borderColor: reachesMany ? TOKEN.coral : TOKEN.gray200 }}
            >
              {target}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ExampleTabs({ entries, activeId, onSelect }) {
  return (
    <div
      role="group"
      aria-label="Select example preset"
      className="flex max-w-full flex-wrap gap-2 lg:justify-end"
    >
      {entries.map((entry) => {
        const active = entry.id === activeId;
        return (
          <button
            key={entry.id}
            type="button"
            aria-pressed={active}
            onClick={() => onSelect(entry.id)}
            className="rounded-full border px-3 py-1.5 font-sans text-[12px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--coral)]/20"
            style={{
              borderColor: active ? TOKEN.coral : TOKEN.gray200,
              background: active ? TOKEN.coralLight : TOKEN.white,
              color: active ? TOKEN.coral : TOKEN.gray600,
            }}
          >
            {entry.tabLabel}
          </button>
        );
      })}
    </div>
  );
}

function CurrentPresetSummary({ view }) {
  return (
    <section className="rounded-[var(--radius-lg)] border border-gray-200 bg-gray-50 p-4">
      <div className="grid gap-4">
        <div className="min-w-0">
          <Kicker>{view.id === 'current' ? 'Current preset' : 'Reference example'}</Kicker>
          <h3 className="mt-2 font-sans text-[15px] font-semibold text-gray-900">{view.title}</h3>
          <p className="mt-2 max-w-[46rem] font-serif text-[15px] leading-7 text-gray-700">{view.caption}</p>
        </div>
        <MetricStrip view={view} />
        <div className="rounded-[var(--radius-md)] border-l-[3px] border-[var(--coral)] bg-[var(--coral-light)] px-3 py-2 font-sans text-[12px] leading-5 text-gray-800">
          Count rows for <span className="font-mono font-semibold">M</span>. Count filled <span className="font-mono font-semibold">O→Q</span> cells for <span className="font-mono font-semibold">α</span>.
        </div>
      </div>
    </section>
  );
}

export default function TwoQuotientSchematic({ current = null }) {
  const [activeId, setActiveId] = useState('current');
  const [hoverTarget, setHoverTarget] = useState(null);
  const reducedMotion = usePrefersReducedMotion();
  const currentView = useMemo(() => buildCurrentView(current), [current]);
  const tabEntries = useMemo(() => [
    currentView,
    ...PRESET_ORDER
      .filter((id) => id !== currentView.referenceId)
      .map((id) => REFERENCE_PRESETS[id]),
  ], [currentView]);

  useEffect(() => {
    if (!tabEntries.some((entry) => entry.id === activeId)) setActiveId('current');
  }, [activeId, tabEntries]);

  const view = activeId === 'current' ? currentView : REFERENCE_PRESETS[activeId] ?? currentView;
  const isGptActive = hoverTarget === 'gpt';
  const isHActive = hoverTarget === 'h';
  const isPiActive = hoverTarget === 'pi';
  const startHover = (target) => setHoverTarget(target);
  const endHover = () => setHoverTarget(null);

  return (
    <figure
      className="rounded-[var(--radius-lg)] border border-gray-200 bg-white p-4 md:p-5"
      aria-label="Two-quotient schematic: current preset plus reference examples"
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <Kicker>Two quotient spaces</Kicker>
            <figcaption className="mt-2 max-w-[42rem] font-serif text-[16px] leading-7 text-gray-700">
              First compress full products into rows. Separately compress visible outputs into stored columns. Then projection marks the filled row-column cells counted by α.
            </figcaption>
          </div>
          <ExampleTabs entries={tabEntries} activeId={activeId} onSelect={setActiveId} />
        </div>

        <CurrentPresetSummary view={view} />

        <div className="grid min-w-0 gap-3 rounded-[var(--radius-lg)] border border-gray-200 bg-gray-50 p-3 md:p-4">
          <QuotientLane
            kicker="Row quotient"
            fromTitle="Full product assignments"
            fromFormula="X"
            fromItems={view.productAssignments}
            toTitle="Product-orbit rows"
            toFormula="O ∈ X/G_pt"
            toItems={view.productRows}
            processLabel="quotient by G_pt"
            processDetail="same product value"
            active={isGptActive}
            onHoverStart={() => startHover('gpt')}
            onHoverEnd={endHover}
            reducedMotion={reducedMotion}
          />
          <QuotientLane
            kicker="Column quotient"
            fromTitle="Visible output assignments"
            fromFormula="Y"
            fromItems={view.outputAssignments}
            toTitle="Stored-output columns"
            toFormula="Q ∈ Y/H"
            toItems={view.outputColumns}
            processLabel="quotient by H"
            processDetail="same stored output"
            active={isHActive}
            onHoverStart={() => startHover('h')}
            onHoverEnd={endHover}
            reducedMotion={reducedMotion}
          />
          <ReachRelationPanel
            view={view}
            active={isPiActive}
            onHoverStart={() => startHover('pi')}
            onHoverEnd={endHover}
            reducedMotion={reducedMotion}
          />
        </div>
      </div>
    </figure>
  );
}
