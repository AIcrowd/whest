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
import Latex from './Latex.jsx';

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
      className="grid grid-cols-1 border-y border-gray-200 py-1 sm:grid-cols-4 sm:py-3"
      aria-label="Selected two-quotient metrics"
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

function CurrentPresetCounts({ view }) {
  return (
    <section className="border-t border-gray-200 pt-4">
      <div className="grid gap-3">
        <div className="min-w-0">
          <Kicker>Selected counts</Kicker>
          <div className="mt-2 flex flex-col gap-1 md:flex-row md:items-baseline md:justify-between">
            <h3 className="font-sans text-[14px] font-semibold text-gray-900">{view.title}</h3>
            <p className="font-serif text-[14px] leading-6 text-gray-600">{view.caption}</p>
          </div>
        </div>
        <MetricStrip view={view} />
        <p className="font-sans text-[12px] leading-5 text-gray-600">
          Count rows for <span className="font-mono font-semibold">M</span>. Count filled <span className="font-mono font-semibold">O→Q</span> cells for <span className="font-mono font-semibold">α</span>.
        </p>
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

function AssignmentPill({ item }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-gray-200 bg-white px-3 py-2">
      <div className="font-sans text-[9px] font-semibold uppercase text-gray-400" style={{ letterSpacing: '0.12em' }}>
        one assignment
      </div>
      <div className="mt-1 font-mono text-[12px] font-semibold leading-4 text-gray-900">{item}</div>
    </div>
  );
}

function FlowCell({ eyebrow, title, formula, children, className = '' }) {
  return (
    <div className={`min-w-0 bg-white py-4 md:py-5 ${className}`}>
      <Kicker>{eyebrow}</Kicker>
      <h4 className="mt-2 font-sans text-[15px] font-semibold leading-5 text-gray-900">{title}</h4>
      {formula ? (
        <div className="mt-2 text-[13px] font-semibold text-gray-800">
          <Latex math={formula} colorize={false} />
        </div>
      ) : null}
      <div className="mt-3">{children}</div>
    </div>
  );
}

function ResultAssignmentList({ items }) {
  return (
    <div className="mt-3 grid gap-1.5">
      {items.map((item, index) => (
        <div
          key={`${item}-${index}`}
          className="rounded-[var(--radius-sm)] border bg-white px-2.5 py-1.5 font-mono text-[11px] font-semibold leading-4 text-gray-800"
          style={{ borderColor: TOKEN.coral }}
        >
          {item}
        </div>
      ))}
    </div>
  );
}

function ResultPunchline({ reachedCount }) {
  return (
    <div className="mt-3 rounded-[var(--radius-md)] border-l-[3px] border-[var(--coral)] bg-white px-3 py-2 font-sans text-[12px] leading-5 text-gray-800">
      This product row contributes <span className="font-mono font-semibold text-[var(--coral)]">{reachedCount}</span> to <span className="font-mono font-semibold">α</span>.
    </div>
  );
}

function MicroscopeConnector({ children }) {
  return (
    <div className="hidden items-center justify-center px-3 lg:flex">
      <div className="font-sans text-[10.5px] font-semibold uppercase leading-4 text-gray-400" style={{ letterSpacing: '0.08em' }}>
        {children}
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
    return 'For the Cross S₂ microscope, symmetry of A makes these two assignments the same representative product: A[0,1] * B[0] = A[1,0] * B[0].';
  }
  return view.reachTargets > 1
    ? 'These assignments share one representative product row before projection branches to stored outputs.'
    : 'These assignments share one representative product row; the evaluator multiplies that row once.';
}

function OneRowMicroscope({ view }) {
  const denseText = countLabel(view.denseAssignmentCount, 'assignment');
  const visibleText = countLabel(visibleAssignmentCount(view), 'visible output');
  const reachedCount = Math.max(1, Math.min(view.outputColumns.length, view.reachTargets));
  const reachesMany = reachedCount > 1;
  const headline = reachesMany
    ? `One product row, ${reachedCount} output columns`
    : 'One product row, one output column';
  const fillText = reachesMany
    ? `The row fills ${reachedCount} stored columns after projection.`
    : 'The row fills one stored column after projection.';

  return (
    <section className="border-t border-gray-200 pt-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <Kicker>One row microscope</Kicker>
          <h3 className="mt-2 font-serif text-[24px] font-semibold leading-8 text-gray-900">{headline}</h3>
          <p className="mt-2 max-w-[50rem] font-serif text-[16px] leading-7 text-gray-700">
            Each pill is one dense assignment: one choice of values for all labels. We show two members from one product row; projection then tells which stored output cells that row fills.
          </p>
        </div>
        <div className="rounded-full border border-gray-200 bg-white px-3 py-1.5 font-mono text-[12px] font-semibold text-gray-700 md:shrink-0">
          {view.labelOrder?.length ? `labels in X: ${view.labelOrder.join(', ')}` : 'labels in X'}
        </div>
      </div>

      <div className="mt-5 border-y border-gray-200">
        <div className="grid lg:grid-cols-[minmax(0,1.05fr)_auto_minmax(0,0.9fr)_auto_minmax(0,1fr)]">
          <FlowCell
            eyebrow="1. Dense assignments"
            title="Two example tuples from X"
            formula="x \\in X"
            className="lg:pr-5"
          >
            <p className="font-serif text-[14px] leading-6 text-gray-700">
              {denseText ? `The full space X has ${denseText} at n=${view.dimensionN}. ` : ''}These two examples are not the whole space.
            </p>
            <div className="mt-3 grid gap-2">
              {view.productAssignments.slice(0, 2).map((item, index) => (
                <AssignmentPill key={`${item}-${index}`} item={item} />
              ))}
            </div>
          </FlowCell>

          <MicroscopeConnector>same product</MicroscopeConnector>

          <FlowCell
            eyebrow="2. Product row"
            title="One row O"
            formula="O \\in X/G_{\\mathrm{pt}}"
            className="border-t border-gray-200 lg:border-l lg:border-t-0 lg:px-5"
          >
            <p className="font-serif text-[14px] leading-6 text-gray-700">{productRowNote(view)}</p>
            <AssignmentList items={view.productRows.slice(0, 1)} />
          </FlowCell>

          <MicroscopeConnector>project to V</MicroscopeConnector>

          <FlowCell
            eyebrow="3. Filled output cells"
            title="Which output cells get filled?"
            formula="\\pi_V(O) \\text{ meets } Q"
            className="border-t border-gray-200 lg:border-l lg:border-t-0 lg:pl-5"
          >
            <p className="font-serif text-[14px] leading-6 text-gray-800">
              {visibleText ? `Y has ${visibleText} at n=${view.dimensionN}. ` : ''}{fillText}
            </p>
            <ResultAssignmentList items={view.outputColumns.slice(0, reachedCount)} />
            <ResultPunchline reachedCount={reachedCount} />
          </FlowCell>
        </div>
      </div>
    </section>
  );
}

function FormalQuotientSummary() {
  return (
    <section className="grid border-t border-gray-200 md:grid-cols-3 md:divide-x md:divide-gray-200">
      {[
        ['Rows', 'X \\to X/G_{\\mathrm{pt}}', 'Group full assignments that produce the same pre-summation product.'],
        ['Columns', 'Y \\to Y/H', 'Group visible output assignments that share stored output representatives.'],
        ['Reach', 'X/G_{\\mathrm{pt}} \\dashrightarrow Y/H', 'Projection marks which row-column cells are filled; it may be a relation, not a function.'],
      ].map(([eyebrow, formula, text], index) => (
        <div
          key={eyebrow}
          className={`border-t border-gray-100 py-4 first:border-t-0 md:border-t-0 ${index === 0 ? 'md:pr-4' : 'md:px-4'}`}
        >
          <Kicker>{eyebrow}</Kicker>
          <div className="mt-2 text-[14px] font-semibold text-gray-900">
            <Latex math={formula} colorize={false} />
          </div>
          <p className="mt-2 font-serif text-[14px] leading-6 text-gray-700">{text}</p>
        </div>
      ))}
    </section>
  );
}

export default function TwoQuotientSchematic({ current = null }) {
  const [activeId, setActiveId] = useState('current');
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

        <OneRowMicroscope view={view} />
        <CurrentPresetCounts view={view} />
        <FormalQuotientSummary />
      </div>
    </figure>
  );
}
