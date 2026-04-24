import Link from 'next/link';
import {
  getHubNamespacePreviewLinks,
  getHubNamespaces,
} from './public-api-data';
import styles from './styles.module.css';

export default function ApiNamespaceHub(): React.ReactElement {
  const topLevelNamespaces = getHubNamespaces();
  const summaryOverrides: Record<string, string> = {
    'we.random':
      'Counted samplers, generator helpers, and random-specific public utilities under we.random.',
    'we.stats':
      'Distribution objects with drill-down method pages for pdf, cdf, and ppf workflows.',
    'we.flops':
      'Pure cost-estimation helpers for planning, debugging, and comparing expected work.',
    'we.testing':
      'Assertion helpers re-exported on the public API for tests, diagnostics, and examples.',
  };

  return (
    <section className={styles.referenceSection}>
      <div className={styles.hubSectionHeader}>
        <div>
          <h2 className={styles.sectionTitle}>Public API namespaces</h2>
          <p className={styles.sectionDescription}>
            Start with the authored namespace chapters before drilling into
            generated leaf pages.
          </p>
        </div>
      </div>
      <div className={styles.hubNamespaceGrid}>
        {topLevelNamespaces.map((namespace) => (
          <article
            key={namespace.import_path}
            className={styles.hubNamespaceCard}
          >
            {/** Curated preview links keep the hub intentional instead of mirroring raw registry order. */}
            {(() => {
              const previewLinks = getHubNamespacePreviewLinks(
                namespace.import_path as
                  | 'we.random'
                  | 'we.stats'
                  | 'we.flops'
                  | 'we.testing',
              );
              return (
                <>
            <div className={styles.hubNamespaceCardHeader}>
              <p className={styles.hubNamespaceEyebrow}>Namespace</p>
              <h3 className={styles.hubNamespaceTitle}>
                <Link
                  className={styles.hubNamespaceTitleLink}
                  href={namespace.href}
                >
                  {namespace.import_path.replace(/^we\./, '')}
                </Link>
              </h3>
              <p className={styles.hubNamespacePath}>
                <code>{namespace.import_path}</code>
              </p>
            </div>
            <p className={styles.hubNamespaceSummary}>
              {summaryOverrides[namespace.import_path] ?? namespace.summary}
            </p>
            {previewLinks.length > 0 && (
              <ul className={styles.hubNamespaceHighlights}>
                {previewLinks.map((entry) => (
                  <li key={`${namespace.import_path}-${entry.href}`}>
                    <Link
                      className={styles.hubNamespaceHighlightLink}
                      href={entry.href}
                    >
                      {entry.title}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
                </>
              );
            })()}
          </article>
        ))}
      </div>
    </section>
  );
}
