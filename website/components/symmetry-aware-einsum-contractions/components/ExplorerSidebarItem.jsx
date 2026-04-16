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
        'group/card flex w-full flex-col gap-1.5 overflow-visible rounded-xl border-2 border-border/40 py-4 text-left text-sm text-card-foreground transition-colors',
        active ? 'ring-2 ring-primary/20 border-primary/55' : 'border-border/40',
        className,
      )}
      {...props}
    >
      <span className="px-3">
        <span className="min-w-0 space-y-1.5">
          {badge ? <Badge variant={active ? 'default' : 'outline'} className={badgeClassName}>{badge}</Badge> : null}
          {title ? <span className="block font-heading text-sm font-medium text-foreground">{title}</span> : null}
          {description ? <span className="block text-xs leading-5 text-muted-foreground">{description}</span> : null}
          {children}
        </span>
      </span>
    </Component>
  );
}

export { ExplorerSidebarItem };
export default ExplorerSidebarItem;
