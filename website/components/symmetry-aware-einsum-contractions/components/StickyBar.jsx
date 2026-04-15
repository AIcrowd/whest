import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { EXPLORER_ACTS } from './explorerNarrative.js';

export default function StickyBar({ example, group, dimensionN, onDimensionChange, activeActId }) {
  return (
    <div className="sticky top-0 z-40 border-b border-border/70 bg-background/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1400px] flex-col gap-3 px-6 py-3 md:flex-row md:items-center md:justify-between">
        {example && (
          <div className="flex min-w-0 shrink items-center gap-2">
            <Badge className="shrink-0 bg-coral text-white hover:bg-coral">einsum</Badge>
            <code className="truncate text-sm font-mono text-foreground">
              {example.formula}
            </code>
            {group && (
              <Badge variant="outline" className="shrink-0 border-coral/30 bg-coral-light text-coral">
                {group.fullGroupName || 'trivial'}
              </Badge>
            )}
          </div>
        )}

        <nav className="flex shrink-0 items-center gap-1 overflow-x-auto pb-1 md:pb-0">
          {EXPLORER_ACTS.map((act, idx) => {
            const isActive = activeActId === act.id;
            return (
              <Button
                key={act.id}
                type="button"
                size="sm"
                variant={isActive ? 'secondary' : 'ghost'}
                className={isActive ? 'bg-coral-light text-coral hover:bg-coral-light/80' : 'text-muted-foreground'}
                onClick={() => {
                  window.location.hash = act.id;
                }}
              >
                <Badge
                  variant={isActive ? 'default' : 'outline'}
                  className={isActive ? 'bg-coral text-white' : 'text-muted-foreground'}
                >
                  {idx + 1}
                </Badge>
                {act.navTitle}
              </Button>
            );
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-3 self-end md:self-auto">
          <label className="flex cursor-pointer items-center gap-2">
            <span className="text-xs font-mono font-semibold text-muted-foreground">n =</span>
            <input
              type="range"
              min={2}
              max={25}
              value={dimensionN}
              onChange={(event) => onDimensionChange(Number(event.target.value))}
              className="h-1.5 w-40 cursor-pointer accent-coral"
            />
            <span className="w-6 text-center text-sm font-mono font-bold text-foreground">
              {dimensionN}
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
