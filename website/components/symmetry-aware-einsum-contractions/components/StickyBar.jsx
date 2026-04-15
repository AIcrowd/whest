import { EXPLORER_ACTS } from './explorerNarrative.js';

function navClasses(isActive) {
  return [
    'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
    isActive
      ? 'bg-coral-light text-coral'
      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
  ].join(' ');
}

function dotClasses(isActive) {
  return [
    'flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-mono font-bold',
    isActive ? 'bg-coral text-white' : 'bg-gray-200 text-gray-600',
  ].join(' ');
}

export default function StickyBar({ example, group, dimensionN, onDimensionChange, activeActId }) {
  return (
    <div className="sticky top-0 z-40 border-b border-gray-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-6 px-6 py-2">
        {example && (
          <div className="flex min-w-0 shrink items-center gap-2">
            <span className="shrink-0 rounded-full bg-coral px-2 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-wider text-white">
              einsum
            </span>
            <code className="truncate text-sm font-mono text-gray-800">
              {example.formula}
            </code>
            {group && (
              <span className="inline-flex shrink-0 items-center rounded-full bg-coral-light px-2.5 py-0.5 text-xs font-semibold text-coral">
                {group.fullGroupName || 'trivial'}
              </span>
            )}
          </div>
        )}

        <nav className="flex shrink-0 items-center gap-1">
          {EXPLORER_ACTS.map((act, idx) => {
            const isActive = activeActId === act.id;
            return (
              <a
                key={act.id}
                href={`#${act.id}`}
                className={navClasses(isActive)}
              >
                <span className={dotClasses(isActive)}>{idx + 1}</span>
                {act.navTitle}
              </a>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-2">
          <label className="flex cursor-pointer items-center gap-2">
            <span className="text-xs font-mono font-semibold text-gray-500">n =</span>
            <input
              type="range"
              min={2}
              max={25}
              value={dimensionN}
              onChange={(event) => onDimensionChange(Number(event.target.value))}
              className="h-1.5 w-40 cursor-pointer accent-coral"
            />
            <span className="w-6 text-center text-sm font-mono font-bold text-gray-800">
              {dimensionN}
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
