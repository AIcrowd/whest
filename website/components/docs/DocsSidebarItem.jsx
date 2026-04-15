'use client';

import { usePathname } from 'fumadocs-core/framework';
import { SidebarItem } from 'fumadocs-ui/components/sidebar/base';

const STANDALONE_SYMMETRY_AWARE_EINSUM_URL = '/symmetry-aware-einsum-contractions';

export default function DocsSidebarItem({ item }) {
  const pathname = usePathname();
  const isStandaloneLaunch = item.url === STANDALONE_SYMMETRY_AWARE_EINSUM_URL;
  const shouldOpenNewTab = Boolean(item.external || isStandaloneLaunch);

  return (
    <SidebarItem
      href={item.url}
      icon={item.icon}
      external={item.external}
      active={!isStandaloneLaunch && pathname === item.url}
      target={shouldOpenNewTab ? '_blank' : undefined}
      rel={shouldOpenNewTab ? 'noreferrer noopener' : undefined}
    >
      {item.name}
    </SidebarItem>
  );
}
