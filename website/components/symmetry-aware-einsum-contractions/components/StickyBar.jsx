import { Badge } from '@/components/ui/badge';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { EXPLORER_ACTS } from './explorerNarrative.js';

function symmetrySuperscript(variable) {
  if (!variable || variable.symmetry === 'none') return null;
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

function AnnotatedOperand({ name, subscript, variable }) {
  const superscript = symmetrySuperscript(variable);

  return (
    <span className="inline-flex items-start">
      <span>{name}</span>
      {superscript ? (
        <sup className="ml-0.5 text-[0.65rem] font-semibold leading-none text-coral">
          {superscript}
        </sup>
      ) : null}
      {subscript ? (
        <sub className="ml-0.5 text-[0.7rem] leading-none text-muted-foreground">
          {subscript}
        </sub>
      ) : null}
    </span>
  );
}

function EinsumExpression({ example }) {
  const expression = example?.expression;
  const variables = example?.variables ?? [];

  if (!expression) {
    return <span>{example?.formula ?? ''}</span>;
  }

  const operandNames = expression.operandNames.split(',').map((part) => part.trim()).filter(Boolean);
  const operandSubs = expression.subscripts.split(',').map((part) => part.trim());

  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-2 gap-y-1">
      {operandNames.map((name, idx) => {
        const variable = variables.find((entry) => entry.name === name) ?? null;
        return (
          <span key={`sticky-operand-${idx}`} className="inline-flex items-baseline">
            <AnnotatedOperand name={name} subscript={operandSubs[idx] || ''} variable={variable} />
            {idx < operandNames.length - 1 ? <span className="mx-1 text-muted-foreground">,</span> : null}
          </span>
        );
      })}
      <span className="mx-1 text-muted-foreground">→</span>
      <span className="inline-flex items-baseline">
        <span>Y</span>
        {expression.output ? (
          <sub className="ml-0.5 text-[0.7rem] leading-none text-muted-foreground">
            {expression.output}
          </sub>
        ) : null}
      </span>
    </span>
  );
}

export default function StickyBar({ example, group, activeActId }) {
  return (
    <div className="sticky top-0 z-40 border-b border-border/70 bg-background/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1460px] flex-col gap-4 px-6 py-4 md:flex-row md:items-center md:justify-between md:px-8">
        <div className="flex min-w-0 shrink items-center gap-2" />

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

        <div className="flex min-w-0 shrink items-center gap-2 self-end md:max-w-[32rem] md:self-auto">
          {example && (
            <>
              <Badge className="shrink-0">einsum</Badge>
              <code className="block rounded-md border border-primary/20 bg-primary/10 px-2.5 py-1 text-sm font-mono font-medium text-primary shadow-sm">
                <EinsumExpression example={example} />
              </code>
              {group && (
                <Badge variant="outline" className="shrink-0 border-primary/25 bg-primary/10 text-primary">
                  {group.fullGroupName || 'trivial'}
                </Badge>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
