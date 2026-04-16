import type {ReactNode} from 'react';
import Link from 'next/link';
import styles from './styles.module.css';

type OperationDocLinkProps = {
  label: ReactNode;
  href?: string;
  externalUrl?: string;
  muted?: boolean;
  className?: string;
};

function joinClassNames(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ');
}

// Canonical renderer for operation links so supported whest ops always link consistently.
export default function OperationDocLink({
  label,
  href,
  externalUrl,
  muted = false,
  className,
}: OperationDocLinkProps) {
  const classes = joinClassNames(styles.opLink, muted && styles.opLinkMuted, className);

  if (href) {
    return (
      <Link href={href} className={classes}>
        {label}
      </Link>
    );
  }

  if (externalUrl) {
    return (
      <a href={externalUrl} target="_blank" rel="noreferrer" className={classes}>
        {label}
      </a>
    );
  }

  return <span className={classes}>{label}</span>;
}
