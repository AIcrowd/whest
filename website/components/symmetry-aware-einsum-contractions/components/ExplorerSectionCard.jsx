import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

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
              {eyebrow ? <CardDescription className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">{eyebrow}</CardDescription> : null}
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

export { ExplorerSectionCard };
export default ExplorerSectionCard;
