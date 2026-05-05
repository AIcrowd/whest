import { cn } from '../lib/utils.js';

const BASE_SHIMMER = 'motion-safe:animate-pulse motion-reduce:animate-none';

function SkeletonBlock({ className = '', style = undefined }) {
  return (
    <div
      className={cn('rounded-sm bg-stone-200/80', BASE_SHIMMER, className)}
      style={style}
      aria-hidden="true"
    />
  );
}

function MatrixSkeleton() {
  return (
    <div className="mt-4 grid min-h-0 flex-1 grid-cols-[44px_1fr] grid-rows-[24px_minmax(0,1fr)] gap-2">
      <div />
      <div className="grid grid-cols-6 gap-1">
        {Array.from({ length: 6 }).map((_, idx) => (
          <SkeletonBlock key={idx} className="h-3" />
        ))}
      </div>
      <div className="grid min-h-0 grid-rows-8 gap-1">
        {Array.from({ length: 8 }).map((_, idx) => (
          <SkeletonBlock key={idx} className="h-3" />
        ))}
      </div>
      <div className="grid min-h-0 grid-cols-12 grid-rows-8 gap-px border border-stone-200 bg-white p-2">
        {Array.from({ length: 96 }).map((_, idx) => {
          const filled = idx % 11 === 0 || idx % 17 === 0 || (idx > 42 && idx < 60 && idx % 3 === 0);
          return (
            <div
              key={idx}
              className={cn(
                'min-h-3 rounded-[1px]',
                filled ? 'bg-coral/45' : 'bg-stone-100',
                filled && BASE_SHIMMER,
              )}
              aria-hidden="true"
            />
          );
        })}
      </div>
    </div>
  );
}

function GraphSkeleton() {
  return (
    <div className="mt-4 min-h-[260px] rounded border border-stone-200 bg-white p-5">
      <div className="grid h-full grid-cols-[1fr_1.25fr_1fr] items-center gap-5">
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, idx) => (
            <SkeletonBlock key={idx} className="h-8 rounded-full" />
          ))}
        </div>
        <div className="relative h-56">
          {Array.from({ length: 9 }).map((_, idx) => (
            <SkeletonBlock
              key={idx}
              className="absolute h-px rounded-none"
              style={{
                left: `${(idx % 3) * 26 + 8}%`,
                top: `${Math.floor(idx / 3) * 26 + 20}%`,
                width: '48%',
                transform: `rotate(${idx % 2 === 0 ? 17 : -17}deg)`,
              }}
            />
          ))}
        </div>
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, idx) => (
            <SkeletonBlock key={idx} className="h-8 rounded-full" />
          ))}
        </div>
      </div>
    </div>
  );
}

function CardsSkeleton() {
  return (
    <div className="mt-4 grid min-h-[220px] gap-3 md:grid-cols-2">
      {Array.from({ length: 4 }).map((_, idx) => (
        <div key={idx} className="rounded border border-stone-200 bg-white p-4">
          <SkeletonBlock className="h-3 w-1/3" />
          <SkeletonBlock className="mt-4 h-5 w-2/3" />
          <SkeletonBlock className="mt-3 h-3 w-full" />
          <SkeletonBlock className="mt-2 h-3 w-5/6" />
        </div>
      ))}
    </div>
  );
}

function CompactSkeleton() {
  return (
    <div className="mt-4 grid min-h-[150px] gap-3 sm:grid-cols-3">
      {Array.from({ length: 6 }).map((_, idx) => (
        <div key={idx} className="rounded border border-stone-200 bg-white p-3">
          <SkeletonBlock className="h-3 w-1/2" />
          <SkeletonBlock className="mt-3 h-7 w-2/3" />
        </div>
      ))}
    </div>
  );
}

function renderSkeleton(variant) {
  if (variant === 'matrix') return <MatrixSkeleton />;
  if (variant === 'graph') return <GraphSkeleton />;
  if (variant === 'compact') return <CompactSkeleton />;
  return <CardsSkeleton />;
}

export default function AnalysisLoadingBoundary({
  isLoading,
  label,
  variant = 'cards',
  minHeight = 220,
  children,
  className = '',
}) {
  return (
    <div
      className={cn('relative', className)}
      aria-busy={isLoading ? 'true' : undefined}
      data-analysis-loading-boundary={isLoading ? 'true' : undefined}
      data-analysis-loading-variant={isLoading ? variant : undefined}
      style={isLoading ? { minHeight } : undefined}
    >
      <div
        className={cn(isLoading && 'pointer-events-none select-none opacity-0')}
        aria-hidden={isLoading ? 'true' : undefined}
      >
        {children}
      </div>
      {isLoading ? (
        <div className="absolute inset-0 z-10 flex flex-col overflow-hidden rounded-lg border border-stone-200 bg-white/95 p-4 shadow-sm backdrop-blur-[1px]">
          <div
            role="status"
            aria-live="polite"
            className="flex items-center justify-between gap-3 font-sans text-[10px] font-semibold uppercase tracking-[0.18em] text-coral"
          >
            <span>Preparing {label}</span>
            <span className={cn('h-2 w-2 rounded-full bg-coral', BASE_SHIMMER)} aria-hidden="true" />
          </div>
          {renderSkeleton(variant)}
        </div>
      ) : null}
    </div>
  );
}
