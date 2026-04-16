import { Badge } from '@/components/ui/badge';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { EXPLORER_ACTS } from './explorerNarrative.js';

function symmetryLabel(variable) {
  if (!variable || variable.symmetry === 'none') return 'dense';
  const k = (variable.symAxes && variable.symAxes.length) || variable.rank;
  switch (variable.symmetry) {
    case 'symmetric':
      return `S${k}`;
    case 'cyclic':
      return `C${k}`;
    case 'dihedral':
      return `D${k}`;
    case 'custom':
      return 'custom';
    default:
      return null;
  }
}

function ParameterSymmetryRow({ variables = [] }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {variables.map((variable, idx) => (
        <span
          key={`sticky-sym-${variable.name}-${idx}`}
          className="inline-flex items-center gap-1 rounded-full border border-primary/18 bg-white px-2 py-0.5 text-xs font-mono text-foreground shadow-sm"
        >
          <span className="font-semibold text-primary">{variable.name}</span>
          <span className="text-muted-foreground">:</span>
          <span className="text-coral">{symmetryLabel(variable)}</span>
        </span>
      ))}
    </div>
  );
}

export default function StickyBar({ example, group, activeActId }) {
  const variables = example?.variables ?? [];

  return (
    <div className="sticky top-0 z-40 border-b border-border/70 bg-background/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1460px] flex-col gap-4 px-6 py-4 md:flex-row md:items-center md:justify-between md:px-8">
        <nav className="flex shrink-0 items-center gap-1 overflow-x-auto pb-1 md:pb-0">
          {EXPLORER_ACTS.map((act, idx) => {
            const isActive = activeActId === act.id;
            return (
              <a
                key={act.id}
                href={`#${act.id}`}
                className={cn(
                  buttonVariants({ size: 'sm', variant: 'ghost' }),
                  'inline-flex min-h-9 items-center gap-2 rounded-full border px-3 transition-colors',
                  isActive
                    ? 'border-primary/35 bg-primary/10 text-primary hover:border-primary/45 hover:bg-primary/15'
                    : 'border-transparent text-muted-foreground hover:border-primary/25 hover:bg-primary/8 hover:text-primary',
                )}
              >
                <Badge
                  variant={isActive ? 'default' : 'outline'}
                  className={cn(
                    'flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 py-0 text-[11px] font-semibold',
                    isActive
                      ? 'bg-primary/20 text-primary'
                      : 'border-primary/20 bg-background text-muted-foreground',
                  )}
                >
                  {idx + 1}
                </Badge>
                {act.navTitle}
              </a>
            );
          })}
        </nav>

        <div className="flex min-w-0 shrink flex-col items-start gap-2 self-end md:max-w-[42rem] md:items-end md:self-auto">
          {example && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="shrink-0">einsum</Badge>
                <code className="block rounded-md border border-primary/20 bg-primary/10 px-2.5 py-1 text-sm font-mono font-medium text-primary shadow-sm">
                  {example.formula}
                </code>
                {group && (
                  <Badge variant="outline" className="shrink-0 border-primary/25 bg-primary/10 text-primary">
                    {group.fullGroupName || 'trivial'}
                  </Badge>
                )}
              </div>
              <ParameterSymmetryRow variables={variables} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
