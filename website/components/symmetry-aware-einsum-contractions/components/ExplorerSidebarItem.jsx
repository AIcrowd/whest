import { cn } from '../lib/utils.js';

function ExplorerSidebarItem({
  title,
  description,
  formula,
  glyph,
  active = false,
  as: Component = 'div',
  type,
  className,
  children,
  ...props
}) {
  return (
    <Component
      type={Component === 'button' ? (type ?? 'button') : undefined}
      className={cn(
        // Flat editorial preset row mirroring the design-system UI kit:
        // title with coral glyph, compact description, subdued metadata,
        // quiet mono formula, and no inner card chrome.
        'group relative w-full overflow-visible px-4 py-3 pl-5 text-left transition-colors',
        className,
      )}
      {...props}
    >
      <span className="block min-w-0">
        {title ? (
          <span className="flex items-center gap-1.5 text-[13px] font-semibold text-gray-900">
            {glyph ? <span className="text-[13px] text-coral">{glyph}</span> : null}
            <span className="truncate">{title}</span>
          </span>
        ) : null}
        {description ? (
          <span className="mt-[3px] block text-[11px] leading-[1.45] text-gray-600">
            {description}
          </span>
        ) : null}
        {children ? (
          <span className="mt-2 flex flex-wrap items-center gap-1.5 text-gray-500">
            {children}
          </span>
        ) : null}
        {formula ? (
          <code className="mt-2 block truncate font-mono text-[11px] text-gray-400">
            {formula}
          </code>
        ) : null}
      </span>
    </Component>
  );
}

export { ExplorerSidebarItem };
export default ExplorerSidebarItem;
