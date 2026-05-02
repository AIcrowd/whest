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
    labelOrder: ['i', 'j', 'k'],
    visibleLabels: ['i', 'k'],
    denseAssignmentCount: 8,
    hStatus: 'H trivial',
    projectionStatus: 'projection branches',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '4 columns' },
      { label: 'α', value: '5 filled cells' },
    ],
    productAssignments: [{ i: 0, j: 1, k: 0 }, { i: 1, j: 0, k: 0 }],
    productRows: ['O: {(i=0, j=1, k=0), (i=1, j=0, k=0)}'],
    outputAssignments: [{ i: 0, k: 0 }, { i: 1, k: 0 }],
    outputColumns: ['Q: (i=0, k=0)', 'Q: (i=1, k=0)'],
    reachTargets: 2,
  },
  bilinearTrace: {
    id: 'bilinearTrace',
    tabLabel: 'Bilinear trace',
    title: 'Bilinear trace reference',
    caption: 'H is nontrivial. Some output assignments share one stored representative, and projection is functional.',
    dimensionN: 2,
    labelOrder: ['i', 'k', 'j', 'l'],
    visibleLabels: ['i', 'j'],
    denseAssignmentCount: 16,
    hStatus: 'H nontrivial',
    projectionStatus: 'projection functional',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '3 columns' },
      { label: 'α', value: '3 filled cells' },
    ],
    productAssignments: ['(i=0, k=0, j=0, l=0)', '(i=0, k=1, j=0, l=1)'],
    productRows: ['O0: (i=0, k=0, j=0, l=0)'],
    outputAssignments: ['(i=0, j=0)', '(i=0, j=1)', '(i=1, j=0)'],
    outputColumns: ['Q0: (i=0, j=0)'],
    reachTargets: 1,
  },
  tripleOuter: {
    id: 'tripleOuter',
    tabLabel: 'Triple outer',
    title: 'Triple outer reference',
    caption: 'All labels are visible. Projection drops nothing, so product-orbit rows and stored-output columns line up.',
    dimensionN: 2,
    labelOrder: ['a', 'b', 'c'],
    visibleLabels: ['a', 'b', 'c'],
    denseAssignmentCount: 8,
    hStatus: 'H nontrivial',
    projectionStatus: 'rows and columns line up',
    metrics: [
      { label: 'M', value: '3 rows' },
      { label: '|Y/H|', value: '3 columns' },
      { label: 'α', value: '3 filled cells' },
    ],
    productAssignments: ['(a=0, b=0, c=0)', '(a=0, b=1, c=0)'],
    productRows: ['O0: (a=0, b=0, c=0)'],
    outputAssignments: ['(a=0, b=0, c=0)', '(a=0, b=1, c=0)', '(a=1, b=0, c=0)'],
    outputColumns: ['Q0: (a=0, b=0, c=0)'],
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

function uniqueLabels(...groups) {
  const labels = [];
  const seen = new Set();
  for (const group of groups) {
    for (const label of group ?? []) {
      if (!seen.has(label)) {
        labels.push(label);
        seen.add(label);
      }
    }
  }
  return labels;
}

function formatTuple(tuple, labels = null) {
  if (Array.isArray(tuple)) {
    if (Array.isArray(labels) && labels.length === tuple.length) {
      return `(${labels.map((label, idx) => `${label}=${tuple[idx]}`).join(', ')})`;
    }
    return `(${tuple.join(',')})`;
  }
  if (tuple && typeof tuple === 'object') {
    const orderedLabels = uniqueLabels(labels, Object.keys(tuple));
    return `(${orderedLabels
      .filter((key) => Object.prototype.hasOwnProperty.call(tuple, key))
      .map((key) => `${key}=${tuple[key]}`)
      .join(', ')})`;
  }
  return String(tuple ?? '—');
}

function uniqueOutputSamples(orbitRows = [], visibleLabels = null) {
  const seen = new Map();
  for (const row of orbitRows) {
    for (const output of row?.outputs ?? []) {
      const key = output?.outKey ?? formatTuple(output?.outTuple, visibleLabels);
      if (!seen.has(key)) seen.set(key, formatTuple(output?.outTuple ?? key, visibleLabels));
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
  const labelOrder = current?.labelOrder ?? [];
  const visibleLabels = current?.visibleLabels ?? [];
  let microscopeRowIndex = orbitRows.findIndex((row) => (row?.outputCount ?? row?.outputs?.length ?? 0) > 1);
  if (microscopeRowIndex < 0) microscopeRowIndex = orbitRows.findIndex((row) => (row?.orbitTuples?.length ?? 0) > 1);
  if (microscopeRowIndex < 0 && orbitRows.length > 0) microscopeRowIndex = 0;
  const microscopeRow = microscopeRowIndex >= 0 ? orbitRows[microscopeRowIndex] : null;
  const productAssignments = takeSamples(
    microscopeRow?.orbitTuples?.map((tuple) => formatTuple(tuple, labelOrder)) ?? [],
    ['full assignments in X'],
  );
  const productRows = takeSamples(
    microscopeRow
      ? [`O${microscopeRowIndex}: ${formatTuple(microscopeRow?.repTuple, labelOrder)}`]
      : orbitRows.map((row, index) => `O${index}: ${formatTuple(row?.repTuple, labelOrder)}`),
    ['product-orbit rows O'],
  );
  const outputSamples = uniqueOutputSamples(orbitRows, visibleLabels);
  const microscopeOutputs = microscopeRow?.outputs?.map((output) => formatTuple(output?.outTuple, visibleLabels)) ?? [];
  const outputAssignments = takeSamples(microscopeOutputs.length ? microscopeOutputs : outputSamples, ['output assignments in Y']);
  const outputColumns = takeSamples(
    (microscopeOutputs.length ? microscopeOutputs : outputSamples).map((sample, index) => `Q${index}: ${sample}`),
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
  const denseAssignmentCount = current?.denseAssignmentCount
    ?? (Number.isFinite(dimensionN) && labelOrder.length > 0 ? Math.pow(dimensionN, labelOrder.length) : null);

  return {
    id: 'current',
    referenceId: normalizePresetId(presetName),
    tabLabel: `Current: ${presetName}`,
    title: `Current preset — ${presetName}`,
    caption: `${captionLead}. These counts come from the selected contraction${Number.isFinite(dimensionN) ? ` at n=${dimensionN}` : ''}.`,
    dimensionN,
    denseAssignmentCount,
    labelOrder,
    visibleLabels,
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

function countLabel(count, singular, plural = `${singular}s`) {
  if (!Number.isFinite(count)) return null;
  return `${formatNumber(count)} ${count === 1 ? singular : plural}`;
}

function visibleAssignmentCount(view) {
  return Number.isFinite(view.dimensionN) && (view.visibleLabels?.length ?? 0) > 0
    ? Math.pow(view.dimensionN, view.visibleLabels.length)
    : null;
}

function AssignmentList({ items, active = false }) {
  return (
    <div className="mt-3 grid gap-1.5">
      {items.slice(0, 3).map((item, index) => (
        <div
          key={`${item}-${index}`}
          className="rounded-[var(--radius-sm)] border bg-white px-2.5 py-1.5 font-mono text-[11px] font-semibold leading-4 text-gray-700"
          style={{ borderColor: active ? TOKEN.coral : TOKEN.gray200 }}
          title={item}
        >
          <span className="block truncate">{item}</span>
        </div>
      ))}
    </div>
  );
}

function MicroscopeCard({ eyebrow, title, formula, note, items, active = false }) {
  return (
    <div
      className="min-w-0 rounded-[var(--radius-lg)] border bg-white p-4"
      style={{
        borderColor: active ? TOKEN.coral : TOKEN.gray200,
        background: active ? TOKEN.coralLight : TOKEN.white,
      }}
    >
      <div className="font-sans text-[10px] font-semibold uppercase text-gray-400" style={{ letterSpacing: '0.14em' }}>
        {eyebrow}
      </div>
      <h4 className="mt-2 font-sans text-[14px] font-semibold leading-5 text-gray-900">{title}</h4>
      {formula ? <div className="mt-2 font-mono text-[13px] font-semibold text-gray-800">{formula}</div> : null}
      {note ? <p className="mt-2 font-serif text-[14px] leading-6 text-gray-700">{note}</p> : null}
      {items?.length ? <AssignmentList items={items} active={active} /> : null}
    </div>
  );
}

function FlowArrow({ label }) {
  return (
    <div className="flex items-center justify-center gap-2 text-center text-gray-500 max-lg:py-1 lg:flex-col">
      <div className="h-px w-8 bg-gray-300 lg:h-8 lg:w-px" />
      <div className="rounded-full border border-gray-200 bg-white px-2.5 py-1 font-sans text-[11px] font-semibold leading-4 text-gray-600">
        {label}
      </div>
      <div className="flex items-center gap-1 lg:flex-col">
        <div className="h-px w-8 bg-gray-300 lg:h-8 lg:w-px" />
        <div
          className="h-0 w-0 border-y-[5px] border-l-[8px] border-y-transparent lg:rotate-90"
          style={{ borderLeftColor: TOKEN.gray400 }}
          aria-hidden
        />
      </div>
    </div>
  );
}

function isCrossS2View(view) {
  return normalizePresetId(view?.title ?? view?.tabLabel ?? '') === 'crossS2'
    || normalizePresetId(view?.referenceId ?? '') === 'crossS2';
}

function productRowNote(view) {
  if (isCrossS2View(view)) {
    return 'For the Cross S2 microscope, symmetry of A makes these two assignments the same representative product: A[0,1] * B[0] = A[1,0] * B[0].';
  }
  return view.reachTargets > 1
    ? 'These assignments share one representative product row before projection branches to stored outputs.'
    : 'These assignments share one representative product row; the evaluator multiplies that row once.';
}

function OneRowMicroscope({ view }) {
  const denseText = countLabel(view.denseAssignmentCount, 'assignment');
  const visibleText = countLabel(visibleAssignmentCount(view), 'visible output');
  const reachedCount = Math.max(1, Math.min(view.outputColumns.length, view.reachTargets));
  const reachText = reachedCount === 1
    ? 'This product row fills one stored column, so this row contributes 1 to α.'
    : `This product row fills ${reachedCount} stored columns, so this row contributes ${reachedCount} to α.`;

  return (
    <section className="rounded-[var(--radius-lg)] border border-gray-200 bg-gray-50 p-4 md:p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <Kicker>One row microscope</Kicker>
          <h3 className="mt-2 font-sans text-[16px] font-semibold text-gray-900">A tuple is one dense assignment</h3>
          <p className="mt-2 max-w-[48rem] font-serif text-[15px] leading-7 text-gray-700">
            Each tuple below is one choice of values for all labels, not the whole space. {denseText ? `The full space X has ${denseText} at n=${view.dimensionN}; this panel shows members of one product row.` : 'This panel shows members of one product row.'}
          </p>
        </div>
        <div className="rounded-full border border-gray-200 bg-white px-3 py-1.5 font-mono text-[12px] font-semibold text-gray-700 md:shrink-0">
          {view.labelOrder?.length ? `labels in X: ${view.labelOrder.join(', ')}` : 'labels in X'}
        </div>
      </div>

      <div className="mt-5 rounded-[var(--radius-lg)] border border-gray-200 bg-white p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <Kicker>1. Example assignments from X</Kicker>
            <p className="mt-2 font-serif text-[15px] leading-7 text-gray-700">
              A dense evaluator would visit these as separate assignments. Here they are members of the same product row.
            </p>
          </div>
          <div className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1.5 font-sans text-[11px] font-semibold text-gray-600">
            showing examples, not all of X
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {view.productAssignments.slice(0, 2).map((item, index) => (
            <div
              key={`${item}-${index}`}
              className="rounded-[var(--radius-md)] border border-gray-200 bg-gray-50 px-3 py-2 font-mono text-[12px] font-semibold text-gray-800"
            >
              {item}
            </div>
          ))}
          <div className="rounded-full border border-[var(--coral)] bg-[var(--coral-light)] px-3 py-1.5 font-sans text-[11px] font-semibold text-[var(--coral)]">
            same product under G_pt
          </div>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <MicroscopeCard
          eyebrow="2. Product row"
          title="Group equal products into O"
          formula="O ∈ X/G_pt"
          note={productRowNote(view)}
          items={view.productRows.slice(0, 1)}
          active
        />
        <MicroscopeCard
          eyebrow="3. Projection reach"
          title="Record reached Q columns"
          formula="π_V(O) meets Q"
          note={`${visibleText ? `Y has ${visibleText} at n=${view.dimensionN}. ` : ''}${reachText}`}
          items={view.outputColumns.slice(0, reachedCount)}
          active={reachedCount > 1}
        />
      </div>
    </section>
  );
}

function FormalQuotientSummary() {
  return (
    <section className="grid gap-3 md:grid-cols-3">
      {[
        ['Rows', 'X → X/G_pt', 'Group full assignments that produce the same pre-summation product.'],
        ['Columns', 'Y → Y/H', 'Group visible output assignments that share stored output representatives.'],
        ['Reach', 'X/G_pt ⇢ Y/H', 'Projection marks which row-column cells are filled; it may be a relation, not a function.'],
      ].map(([eyebrow, formula, text]) => (
        <div key={eyebrow} className="rounded-[var(--radius-lg)] border border-gray-200 bg-white p-4">
          <Kicker>{eyebrow}</Kicker>
          <div className="mt-2 font-mono text-[14px] font-semibold text-gray-900">{formula}</div>
          <p className="mt-2 font-serif text-[14px] leading-6 text-gray-700">{text}</p>
        </div>
      ))}
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
              Read one row first: a full assignment is one tuple of label values. Symmetry groups equal-product tuples into a product row, and projection shows which stored output columns that row fills.
            </figcaption>
          </div>
          <ExampleTabs entries={tabEntries} activeId={activeId} onSelect={setActiveId} />
        </div>

        <CurrentPresetSummary view={view} />

        <OneRowMicroscope view={view} />
        <FormalQuotientSummary />
      </div>
    </figure>
  );
}
