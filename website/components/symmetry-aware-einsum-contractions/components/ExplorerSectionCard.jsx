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
  return (
    <Card className={cn('gap-0', className)} {...props}>
      {(title || description || eyebrow || action) && (
        <CardHeader className="border-b border-border">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              {eyebrow ? <CardDescription className="text-[11px] font-semibold uppercase tracking-[0.18em]">{eyebrow}</CardDescription> : null}
              {title ? <CardTitle>{title}</CardTitle> : null}
              {description ? <CardDescription>{description}</CardDescription> : null}
            </div>
            {action ? <CardAction>{action}</CardAction> : null}
          </div>
        </CardHeader>
      )}
      <CardContent className={cn('pt-4', contentClassName)}>
        {children}
      </CardContent>
    </Card>
  );
}

export { ExplorerSectionCard };
export default ExplorerSectionCard;
