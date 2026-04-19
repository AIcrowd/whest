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
        // Flat preset row: no card border, no ring. The wrapping list
        // container (PresetSidebar) carries the gray-200 outline and the
        // 1px gray-100 dividers between siblings; the active state is
        // carried by coral-light bg + the 4px coral left-rail span.
        'group/card flex w-full flex-col gap-1.5 overflow-visible py-4 text-left text-sm text-card-foreground transition-colors',
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
