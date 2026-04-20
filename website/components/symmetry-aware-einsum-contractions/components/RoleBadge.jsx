import { Badge } from '@/components/ui/badge';
import { cn } from '../lib/utils.js';
import { notationColor, notationIdForRole, notationTint } from '../lib/notationSystem.js';

export const ROLE_BADGE_CLASSNAMES = {
  v: {
    base: 'mx-1 inline-flex rounded-full border px-2 py-0.5 align-middle font-mono text-[11px] font-semibold',
    interactive: '',
  },
  w: {
    base: 'mx-1 inline-flex rounded-full border px-2 py-0.5 align-middle font-mono text-[11px] font-semibold',
    interactive: '',
  },
};

function roleBadgeStyle(role) {
  const notationId = notationIdForRole(role);
  return {
    borderColor: notationTint(notationId, 0.28),
    backgroundColor: notationTint(notationId, 0.1),
    color: notationColor(notationId),
  };
}

export default function RoleBadge({ role, as = 'span', interactive = false, className, children }) {
  const roleClasses = ROLE_BADGE_CLASSNAMES[role] ?? ROLE_BADGE_CLASSNAMES.v;
  const badgeClassName = cn(roleClasses.base, interactive && roleClasses.interactive, className);
  const badgeStyle = roleBadgeStyle(role);

  if (as === 'badge') {
    return (
      <Badge variant="outline" className={badgeClassName} style={badgeStyle}>
        {children}
      </Badge>
    );
  }

  return <span className={badgeClassName} style={badgeStyle}>{children}</span>;
}
