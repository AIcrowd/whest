import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

/**
 * Distinctive section label for the five top-level sections of the explorer.
 *
 * Visual grammar (Distill / academic-paper):
 *   - 'SECTION'  — small uppercase, letterspaced, muted gray (the caption role)
 *   - 'N'        — large serif italic, coral/primary, the eye-catching anchor
 *
 * Rendered side-by-side along a shared baseline. The number is the thing the
 * reader's eye tracks when scanning the page; 'SECTION' is the label.
 */
function SectionEyebrow({ n }) {
  return (
    <span className="inline-flex items-baseline gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
        Section
      </span>
      <span
        className="font-serif italic leading-none text-primary"
        style={{ fontSize: '22px', fontWeight: 700 }}
      >
        {n}
      </span>
    </span>
  );
}

function ExplorerSectionCard({
  title,
  description,
  eyebrow,
  action,
  children,
  className,
  contentClassName,
  ...props
}) {
  const hasContent = Array.isArray(children)
    ? children.some((child) => child != null && child !== false && child !== '')
    : children != null && children !== false && children !== '';

  return (
    <Card className={cn('border-border/70 shadow-sm', className)} {...props}>
      {(title || description || eyebrow || action) && (
        <CardHeader className="border-b border-border/70">
          <div className="flex items-start justify-between gap-2.5">
            <div className="space-y-2">
              {eyebrow ? (
                typeof eyebrow === 'string'
                  // Back-compat: any string eyebrow still gets the old
                  // uppercase-caption styling. Distinctive section-N labels
                  // come through as <SectionEyebrow n={...} /> JSX.
                  ? <CardDescription className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">{eyebrow}</CardDescription>
                  : <div>{eyebrow}</div>
              ) : null}
              {title ? <CardTitle className="text-lg leading-tight">{title}</CardTitle> : null}
              {description ? <CardDescription className="text-sm leading-6">{description}</CardDescription> : null}
            </div>
            {action ? <CardAction>{action}</CardAction> : null}
          </div>
        </CardHeader>
      )}
      {hasContent ? (
        <CardContent className={cn('pt-5', contentClassName)}>
          {children}
        </CardContent>
      ) : null}
    </Card>
  );
}

export { ExplorerSectionCard, SectionEyebrow };
export default ExplorerSectionCard;
