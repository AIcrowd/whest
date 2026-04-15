import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

function ExplorerSidebarItem({
  title,
  description,
  badge,
  active = false,
  className,
  badgeClassName,
  children,
  ...props
}) {
  return (
    <Card
      size="sm"
      className={cn(
        'gap-0 transition-colors',
        active ? 'ring-2 ring-primary/20' : 'ring-1 ring-foreground/10',
        className,
      )}
      {...props}
    >
      <CardContent className="flex items-start justify-between gap-3 py-3">
        <div className="min-w-0 space-y-1">
          {badge ? <Badge variant={active ? 'default' : 'outline'} className={badgeClassName}>{badge}</Badge> : null}
          {title ? <div className="font-heading text-sm font-medium text-foreground">{title}</div> : null}
          {description ? <div className="text-xs text-muted-foreground">{description}</div> : null}
          {children}
        </div>
      </CardContent>
    </Card>
  );
}

export { ExplorerSidebarItem };
export default ExplorerSidebarItem;
