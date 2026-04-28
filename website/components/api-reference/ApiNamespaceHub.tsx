import styles from './styles.module.css';
import { getFeaturedEntries, getNamespaceCards } from './public-api-data';

export default function ApiNamespaceHub() {
  const cards = getNamespaceCards();
  const featured = getFeaturedEntries();

  return (
    <div className={styles.referenceSection}>
      <section className={styles.sectionBlock}>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Browse the public API</h2>
            <p className={styles.sectionDescription}>
              Start with a namespace chapter, then drop into canonical per-symbol pages that mirror the public import path.
            </p>
          </div>
        </div>
        <div className={styles.namespaceGrid}>
          {cards.map((card) => (
            <article key={card.namespace} className={styles.namespaceCard}>
              <div className={styles.namespaceCardHeader}>
                <div>
                  <p className={styles.namespaceEyebrow}>{card.importPath}</p>
                  <h3 className={styles.namespaceTitle}>
                    <a className={styles.namespaceLink} href={card.href}>
                      {card.title}
                    </a>
                  </h3>
                </div>
                <span className={styles.namespaceCountChip}>{card.count} entries</span>
              </div>
              <p className={styles.namespaceSummary}>{card.summary}</p>
              <ul className={styles.namespaceHighlights}>
                {card.highlights.map((entry) => (
                  <li key={entry.import_path}>
                    <a className={styles.namespaceHighlightLink} href={entry.href}>
                      <span className={styles.namespaceHighlightText}>{entry.display_name}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.sectionBlock}>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Common entry points</h2>
            <p className={styles.sectionDescription}>
              High-signal entry points that cover the counted NumPy surface, top-level helpers, distributions, and analytical estimators.
            </p>
          </div>
        </div>
        <div className={styles.entryHighlightsList}>
          {featured.map((entry) => (
            <article key={entry.import_path} className={styles.entryHighlightCard}>
              <div>
                <p className={styles.entryHighlightEyebrow}>{entry.namespace}</p>
                <h3 className={styles.entryHighlightTitle}>
                  <a className={styles.entryHighlightLink} href={entry.href}>
                    {entry.display_name}
                  </a>
                </h3>
              </div>
              <p className={styles.entryHighlightDescription}>{entry.summary}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
