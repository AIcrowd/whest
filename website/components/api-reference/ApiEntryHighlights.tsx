import Link from 'next/link';
import {
  getHubEntryHighlights,
  getRelatedOperationAreas,
} from './public-api-data';
import styles from './styles.module.css';

const KIND_LABELS = {
  namespace: 'Namespace',
  symbol: 'API entry',
  operation: 'Operation',
} as const;

export default function ApiEntryHighlights(): React.ReactElement {
  const highlightedEntries = getHubEntryHighlights();
  const relatedAreas = getRelatedOperationAreas();

  return (
    <>
      <section className={styles.referenceSection}>
        <div className={styles.hubSectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Key entry points</h2>
            <p className={styles.sectionDescription}>
              A compact set of concrete destinations for the most common public
              API lookups.
            </p>
          </div>
        </div>
        <div className={styles.hubHighlightsGrid}>
          {highlightedEntries.map((entry) => (
            <article key={entry.href} className={styles.hubHighlightItem}>
              <p className={styles.hubHighlightEyebrow}>
                {KIND_LABELS[entry.kind]}
              </p>
              <h3 className={styles.hubHighlightTitle}>
                <Link className={styles.hubHighlightLink} href={entry.href}>
                  {entry.title}
                </Link>
              </h3>
              <p className={styles.hubHighlightDescription}>
                {entry.description}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.referenceSection}>
        <div className={styles.hubSectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Related operation areas</h2>
            <p className={styles.sectionDescription}>
              Operation-heavy submodules that still route directly into leaf
              pages while the hub remains focused on the authored namespace
              chapters.
            </p>
          </div>
        </div>
        <div className={styles.relatedAreaList}>
          {relatedAreas.map((area) => (
            <article key={area.import_path} className={styles.relatedAreaRow}>
              <div className={styles.relatedAreaCopy}>
                <h3 className={styles.relatedAreaTitle}>
                  <span>{area.import_path.replace(/^we\./, '')}</span>
                  <code className={styles.relatedAreaPath}>
                    {area.import_path}
                  </code>
                </h3>
                <p className={styles.relatedAreaDescription}>{area.summary}</p>
              </div>
              <ul className={styles.relatedAreaLinks}>
                {area.preview_links.map((entry) => (
                  <li key={`${area.import_path}-${entry.href}`}>
                    <Link className={styles.relatedAreaLink} href={entry.href}>
                      {entry.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}
