import { Badge } from '@/components/ui/badge';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { EXPLORER_ACTS } from './explorerNarrative.js';

// Character-by-character renderer for a subscript/output *label region*
// (e.g. "ia,ib,ic" or "abc"). Letters matching `hoveredLabels` swap to
// pitch black with a coral underline; everything else passes through.
// Layout is stable: no bold (glyph widths stay fixed), no padding/ring
// (no per-char circle outlines).
function SubscriptTokens({ text, hoveredLabels }) {
  const chars = Array.from(String(text ?? ''));
  const hasHover = hoveredLabels instanceof Set && hoveredLabels.size > 0;
  return (
    <>
      {chars.map((ch, idx) => {
        const isLetter = /[A-Za-z]/.test(ch);
        const isHit = hasHover && isLetter && hoveredLabels.has(ch);
        if (!isHit) return <span key={idx}>{ch}</span>;
        // `decoration-coral` / `text-coral` are NOT defined in this Tailwind
        // theme — only `--primary` / `--color-primary` resolve to the coral
        // hex. `text-black` gives the strong contrast against the coral-tint
        // badge background that plain coral text can't.
        return (
          <span
            key={idx}
            className="text-black underline decoration-primary decoration-2 underline-offset-2"
          >
            {ch}
          </span>
        );
      })}
    </>
  );
}

// Renders the einsum formula with per-region tokenization. Only the
// subscript + output label regions are highlight-able — so hovering a
// label like `i` no longer lights up the `i` in "einsum". We rebuild from
// `example.expression` (structured) rather than walking the display string,
// so the visible formula always matches the authoritative subscripts.
function FormulaHighlighted({ example, hoveredLabels }) {
  const expr = example?.expression;
  if (!expr || typeof expr.subscripts !== 'string') {
    // Pre-normalized or malformed example — fall back to the raw string
    // with no highlighting. Better than crashing or silently highlighting
    // the wrong characters.
    return <>{example?.formula ?? ''}</>;
  }
  return (
    <>
      <span>{"einsum('"}</span>
      <SubscriptTokens text={expr.subscripts} hoveredLabels={hoveredLabels} />
      <span>{'→'}</span>
      <SubscriptTokens text={expr.output ?? ''} hoveredLabels={hoveredLabels} />
      <span>{`', ${expr.operandNames ?? ''})`}</span>
    </>
  );
}

function symmetryLabel(variable) {
  if (!variable || variable.symmetry === 'none') return 'dense';
  const k = (variable.symAxes && variable.symAxes.length) || variable.rank;
  const axes = Array.isArray(variable.symAxes) ? variable.symAxes : null;
  const hasExplicitPartialAxes = axes && axes.length > 0 && axes.length < variable.rank;

  switch (variable.symmetry) {
    case 'symmetric':
      return hasExplicitPartialAxes ? `S${k}{${axes.join(',')}}` : `S${k}`;
    case 'cyclic':
      return hasExplicitPartialAxes ? `C${k}{${axes.join(',')}}` : `C${k}`;
    case 'dihedral':
      return hasExplicitPartialAxes ? `D${k}{${axes.join(',')}}` : `D${k}`;
    case 'custom':
      return hasExplicitPartialAxes ? `custom{${axes.join(',')}}` : 'custom';
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

export default function StickyBar({ example, group, expressionGroup = null, activeActId, hoveredLabels = null }) {
  const variables = example?.variables ?? [];

  // Show the expression-level badge only when G_EXPR is strictly larger than
  // G_PT. When they coincide (|W| ≤ 1), the "counting group" and
  // "compression group" are the same, so displaying both is noise.
  const gptOrder = group?.fullElements?.length ?? group?.fullOrder ?? 1;
  const gExprOrder = expressionGroup?.order ?? gptOrder;
  const showExprBadge = gExprOrder > gptOrder;

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
                  <FormulaHighlighted example={example} hoveredLabels={hoveredLabels} />
                </code>
                {group && (
                  <Badge
                    variant="outline"
                    className="shrink-0 border-primary/25 bg-primary/10 text-primary"
                    title="Per-tuple symmetry group G_PT — drives compression (μ, α). Computed via σ-loop Sources A+B."
                  >
                    <span className="mr-1 text-[10px] uppercase tracking-wide opacity-70">per-tuple</span>
                    {group.fullGroupName || 'trivial'}
                  </Badge>
                )}
                {showExprBadge && (
                  <Badge
                    variant="outline"
                    className="shrink-0 border-amber-500/40 bg-amber-50 text-amber-800"
                    title="Expression-level group G_EXPR = V-sub × S(W) — counting symmetry. NOT used for compression (would over-compress). Larger than G_PT when |W| ≥ 2 because it includes dummy-rename permutations of summed labels."
                  >
                    <span className="mr-1 text-[10px] uppercase tracking-wide opacity-70">expr-level</span>
                    |G| = {gExprOrder}
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
