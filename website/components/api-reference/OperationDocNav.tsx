import Link from 'next/link';
import styles from './styles.module.css';
import type {OperationNavLink} from './op-doc-types';

function NavCard({
  title,
  link,
  align,
}: {
  title: 'Previous' | 'Next';
  link?: OperationNavLink | null;
  align: 'left' | 'right';
}) {
  if (!link) {
    return <div className={`${styles.docNavCard} ${styles.docNavCardMuted}`} />;
  }

  return (
    <Link
      href={link.href}
      className={`${styles.docNavCard} ${align === 'right' ? styles.docNavCardRight : ''}`}
    >
      <span className={styles.docNavLabel}>{title}</span>
      <span className={styles.docNavValue}>{link.label}</span>
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
    <nav className={styles.docNav} aria-label="Operation navigation">
      <NavCard title="Previous" link={previous} align="left" />
      <NavCard title="Next" link={next} align="right" />
    </nav>
  );
}
