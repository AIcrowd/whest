'use client';

import React from 'react';
import Link from 'next/link';
import {getNamespaceSectionData} from './public-api-data';
import styles from './styles.module.css';

type NamespaceSymbolListProps = {
  namespace: string;
};

type NamespaceMember = {
  href: string;
  import_path?: string;
  label?: string;
};

export default function NamespaceSymbolList({namespace}: NamespaceSymbolListProps): React.ReactElement | null {
  const {sections} = getNamespaceSectionData(namespace);

  if (sections.length === 0) {
    return null;
  }

  return (
    <>
      {sections.map((section) => (
        <section key={section.title} className={styles.referenceSection}>
          <div className={styles.subsectionHeader}>
            <h2 className={styles.subsectionTitle}>{section.title}</h2>
          </div>
          {section.description ? (
            <p className={styles.sectionDescription}>{section.description}</p>
          ) : null}
          <div className={styles.namespaceList}>
            {section.entries.map((entry) => (
              <article key={entry.href} className={styles.namespaceRow}>
                <div className={styles.namespaceRowBody}>
                  <div className={styles.namespaceRowTitle}>
                    <Link href={entry.href} className={styles.opLink}>
                      {entry.import_path}
                    </Link>
                  </div>
                  {entry.summary ? (
                    <p className={styles.namespaceSummary}>{entry.summary}</p>
                  ) : null}
                  {entry.members.length > 0 ? (
                    <div className={styles.namespaceMembers}>
                      {(entry.members as NamespaceMember[]).map((member, index) => {
                        const label = member.import_path ?? member.label ?? member.href;

                        return (
                          <span key={member.href}>
                            {index > 0 ? ', ' : null}
                            <Link href={member.href} className={styles.namespaceMemberLink}>
                              {label}
                            </Link>
                          </span>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}
    </>
  );
}
