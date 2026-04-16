import Link from 'next/link';
import {ChevronLeft, ChevronRight} from 'lucide-react';
import type {OperationNavLink} from './op-doc-types';

function NavCard({link, index}: {link: OperationNavLink; index: 0 | 1}) {
  const Icon = index === 0 ? ChevronLeft : ChevronRight;

  return (
    <Link
      href={link.href}
      className={[
        'flex flex-col gap-2 rounded-lg border p-4 text-sm transition-colors hover:bg-fd-accent/80 hover:text-fd-accent-foreground @max-lg:col-span-full',
        index === 1 ? 'text-end' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div
        className={[
          'inline-flex items-center gap-1.5 font-medium',
          index === 1 ? 'flex-row-reverse' : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        <Icon className="-mx-1 size-4 shrink-0 rtl:rotate-180" />
        <p>{link.label}</p>
      </div>
      <p className="text-fd-muted-foreground truncate">
        {index === 0 ? 'Previous Page' : 'Next Page'}
      </p>
    </Link>
  );
}

export default function OperationDocNav({
  previous,
  next,
}: {
  previous?: OperationNavLink | null;
  next?: OperationNavLink | null;
}) {
  if (!previous && !next) {
    return null;
  }

  return (
    <nav
      className={[
        '@container grid gap-4',
        previous && next ? 'grid-cols-2' : 'grid-cols-1',
      ].join(' ')}
      aria-label="Operation navigation"
    >
      {previous ? <NavCard link={previous} index={0} /> : null}
      {next ? <NavCard link={next} index={1} /> : null}
    </nav>
  );
}
