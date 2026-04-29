import Link from 'next/link';
import styles from './styles.module.css';
import {
  getNamespaceInventory,
  type PublicApiNamespace,
} from './public-api-data';

export default function NamespaceInventory({ namespace }: { namespace: PublicApiNamespace }) {
  const sections = getNamespaceInventory(namespace);

  return (
    <div className={styles.referenceSection}>
      {sections.map((section) => (
        <section key={section.title} className={styles.namespaceSection}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>{section.title}</h2>
              <p className={styles.sectionDescription}>{section.description}</p>
            </div>
          </div>
          <div className={styles.namespaceList}>
            {section.entries.map((entry) => (
              <article key={entry.import_path} className={styles.namespaceRow}>
                <div className={styles.namespaceRowBody}>
                  <h3 className={styles.namespaceRowTitle}>
                    <Link className={styles.namespaceLink} href={entry.href}>
                      {entry.display_name}
                    </Link>
                  </h3>
                  <p className={styles.namespaceSummary}>{entry.summary}</p>
                </div>
                {entry.members && entry.members.length > 0 ? (
                  <div className={styles.namespaceMembers}>
                    {entry.members.map((member) => (
                      <Link key={member.href} className={styles.namespaceMemberLink} href={member.href}>
                        {member.label}
                      </Link>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
