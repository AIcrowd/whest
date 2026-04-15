'use client';

import { usePathname } from 'fumadocs-core/framework';
import { SidebarItem } from 'fumadocs-ui/components/sidebar/base';
import { useFolderDepth } from 'fumadocs-ui/components/sidebar/base';
import { STANDALONE_SYMMETRY_AWARE_EINSUM_URL } from '@/lib/docsTree';

function normalizePath(path) {
  return path === '/' ? path : path.replace(/\/+$/, '');
}

export default function DocsSidebarItem({ item }) {
  const pathname = usePathname();
  const depth = useFolderDepth();
  const isStandaloneLaunch = item.url === STANDALONE_SYMMETRY_AWARE_EINSUM_URL;
  const active = !isStandaloneLaunch && normalizePath(pathname) === normalizePath(item.url);
  const itemOffset = `calc(${2 + 3 * depth} * var(--spacing))`;
  const itemClasses = [
    'relative flex flex-row items-center gap-2 rounded-lg p-2 text-start text-fd-muted-foreground wrap-anywhere [&_svg]:size-4 [&_svg]:shrink-0 transition-colors hover:bg-fd-accent/50 hover:text-fd-accent-foreground/80 hover:transition-none data-[active=true]:bg-fd-primary/10 data-[active=true]:text-fd-primary data-[active=true]:hover:transition-colors',
    depth >= 1 ? "data-[active=true]:before:content-[''] data-[active=true]:before:bg-fd-primary data-[active=true]:before:absolute data-[active=true]:before:w-px data-[active=true]:before:inset-y-2.5 data-[active=true]:before:inset-s-2.5" : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <SidebarItem
      href={item.url}
      icon={item.icon}
      className={itemClasses}
      style={{ paddingInlineStart: itemOffset }}
      external={item.external}
      active={active}
      target={isStandaloneLaunch ? '_blank' : undefined}
      rel={isStandaloneLaunch ? 'noreferrer noopener' : undefined}
    >
      {item.name}
    </SidebarItem>
  );
}
