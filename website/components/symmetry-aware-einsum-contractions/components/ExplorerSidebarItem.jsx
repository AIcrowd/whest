import { Badge } from '@/components/ui/badge';
import { cn } from '../lib/utils.js';

function ExplorerSidebarItem({
  title,
  description,
  badge,
  active = false,
  as: Component = 'div',
  type,
  className,
  badgeClassName,
  children,
  ...props
}) {
  return (
    <Component
      data-slot="card"
      data-size="sm"
      type={Component === 'button' ? (type ?? 'button') : undefined}
      className={cn(
        'group/card flex w-full flex-col gap-0 overflow-hidden rounded-xl py-3 text-left text-sm text-card-foreground transition-colors',
        active ? 'ring-2 ring-primary/20' : 'ring-1 ring-foreground/10',
        className,
      )}
      {...props}
    >
      <div className="px-3">
        <div className="min-w-0 space-y-1">
          {badge ? <Badge variant={active ? 'default' : 'outline'} className={badgeClassName}>{badge}</Badge> : null}
          {title ? <div className="font-heading text-sm font-medium text-foreground">{title}</div> : null}
          {description ? <div className="text-xs text-muted-foreground">{description}</div> : null}
          {children}
        </div>
      </div>
    </Component>
  );
}

export { ExplorerSidebarItem };
export default ExplorerSidebarItem;
