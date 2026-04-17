import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import CaseBadge from './CaseBadge.jsx';

/**
 * Six-row reference table of the classification system's supported cases.
 * Sits to the right of the DecisionLadder in Act 4 (30% tree / 70% table).
 *
 * Detection and computation complexity + empirical median/max timings come
 * from the project's benchmark run. The numbers are static narrative, not
 * live telemetry — they move only when we rerun the bench and update this
 * array. (See TODOS.md for the benchmark rerun cadence.)
 */
const CASES = [
  {
    num: 1,
    id: 'trivial',
    detection:   { complexity: 'O(1)',          detail: '|G| = 1',            median: '<1 μs',  max: '<1 μs' },
    computation: { complexity: 'O(|L|)',        detail: 'product of sizes',    median: '<1 μs',  max: '<2 μs' },
  },
  {
    num: 2,
    id: 'allVisible',
    detection:   { complexity: 'O(1)',          detail: 'W = ∅',               median: '<1 μs',  max: '<1 μs' },
    computation: { complexity: 'O(|V|)',        detail: 'product',             median: '<1 μs',  max: '<2 μs' },
  },
  {
    num: 3,
    id: 'allSummed',
    detection:   { complexity: 'O(1)',          detail: 'V = ∅',               median: '<1 μs',  max: '<1 μs' },
    computation: { complexity: 'O(|G|·|L|)',    detail: 'size-aware Burnside', median: '5 μs',   max: '50 μs at |G|=48' },
  },
  {
    num: 4,
    id: 'singleton',
    detection:   { complexity: 'O(1)',          detail: '|V| = 1',             median: '0.38 μs', max: '3.9 μs' },
    computation: { complexity: 'O(|G|·|L|)',    detail: 'weighted incl-excl',  median: '48 μs',  max: '258 μs' },
  },
  {
    num: 5,
    id: 'directProduct',
    detection:   { complexity: 'O(|gens|·|L|)', detail: 'generator scan',      median: '6 μs',   max: '50 μs' },
    computation: { complexity: 'O(|G|·|W|)',    detail: 'Burnside on W',       median: '95 μs',  max: '266 μs' },
  },
  {
    num: 6,
    id: 'bruteForceOrbit',
    detection:   { complexity: 'O(1)',          detail: 'budget check',        median: '0.6 μs', max: '1.8 μs' },
    computation: { complexity: 'O(|X|·|G|)',    detail: 'orbit enumeration',   median: '107 μs', max: '5095 μs (budget-capped)' },
  },
];

export { CASES as COMPACT_CASE_ROWS };

function StackedCell({ primary, secondary, mono = false }) {
  return (
    <div className="leading-tight">
      <div className={mono ? 'font-mono text-[12.5px] text-stone-900' : 'text-[13px] text-stone-900'}>
        {primary}
      </div>
      {secondary ? (
        <div className="text-[11px] text-muted-foreground">{secondary}</div>
      ) : null}
    </div>
  );
}

function TimingCell({ median, max }) {
  return (
    <div className="font-mono text-[12px] leading-tight">
      <div className="text-stone-900">
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">p50 </span>
        {median}
      </div>
      <div className="text-stone-700">
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">max </span>
        {max}
      </div>
    </div>
  );
}

export default function CompactCaseTable({ activeRegimeIds = null, className = '' }) {
  const activeSet = (() => {
    if (!activeRegimeIds) return null;
    if (activeRegimeIds instanceof Set) return activeRegimeIds;
    if (Array.isArray(activeRegimeIds)) return new Set(activeRegimeIds.filter(Boolean));
    return null;
  })();

  return (
    <div
      className={`overflow-x-auto rounded-xl border border-stone-200 bg-white ${className}`}
      aria-label="Compact case table — six supported classification cases with empirical complexity"
    >
      <Table className="text-sm">
        <TableHeader className="bg-stone-50">
          <TableRow>
            <TableHead className="w-10 text-right text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              #
            </TableHead>
            <TableHead className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Case
            </TableHead>
            <TableHead className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Detection
            </TableHead>
            <TableHead className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Detection timing
            </TableHead>
            <TableHead className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Computation
            </TableHead>
            <TableHead className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
              Computation timing
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {CASES.map((row) => {
            const isActive = activeSet?.has(row.id) ?? false;
            return (
              <TableRow
                key={row.id}
                data-case-id={row.id}
                data-active={isActive ? 'true' : 'false'}
                className={isActive ? 'bg-accent/50 ring-1 ring-primary/30 ring-inset' : ''}
              >
                <TableCell className="text-right font-mono text-xs text-muted-foreground">
                  {row.num}
                </TableCell>
                <TableCell>
                  <CaseBadge regimeId={row.id} size="sm" variant="pill" active={isActive} />
                </TableCell>
                <TableCell>
                  <StackedCell mono primary={row.detection.complexity} secondary={row.detection.detail} />
                </TableCell>
                <TableCell>
                  <TimingCell median={row.detection.median} max={row.detection.max} />
                </TableCell>
                <TableCell>
                  <StackedCell mono primary={row.computation.complexity} secondary={row.computation.detail} />
                </TableCell>
                <TableCell>
                  <TimingCell median={row.computation.median} max={row.computation.max} />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
