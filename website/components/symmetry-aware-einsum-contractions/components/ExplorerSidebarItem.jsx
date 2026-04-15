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
      <span className="px-3">
        <span className="min-w-0 space-y-1">
          {badge ? <Badge variant={active ? 'default' : 'outline'} className={badgeClassName}>{badge}</Badge> : null}
          {title ? <span className="block font-heading text-sm font-medium text-foreground">{title}</span> : null}
          {description ? <span className="block text-xs text-muted-foreground">{description}</span> : null}
          {children}
        </span>
      </span>
    </Component>
  );
}

export { ExplorerSidebarItem };
export default ExplorerSidebarItem;
