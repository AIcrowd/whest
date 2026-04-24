import OperationDocBody from './OperationDocBody';
import OperationDocHeader from './OperationDocHeader';
import OperationDocSignature from './OperationDocSignature';
import OperationDocSection from './OperationDocSection';
import styles from './styles.module.css';
import type {PublicApiSymbolRecord} from './op-doc-types';

function formatKind(kind: string) {
  if (kind === 'cost_helper') return 'cost helper';
  return kind.replace(/_/g, ' ');
}

export default function PublicApiSymbolPage({
  symbol,
}: {
  symbol: PublicApiSymbolRecord;
}) {
  const aliasList = (symbol.aliases ?? []).filter((alias) => alias !== symbol.canonical_name);
  const members = symbol.members ?? [];
  const guides = symbol.related_guides ?? [];
  const memberLinks = members.map((member) => ({
    target: member.label,
    label: member.label,
    href: member.href,
  }));
  const guideLinks = guides.map((guide) => ({
    target: guide.title,
    label: guide.title,
    href: guide.href,
  }));

  return (
    <>
      <OperationDocHeader summary={symbol.summary} />

      <section className={styles.docOverlay} aria-label="Public API summary">
        <div className={styles.docMetaStrip}>
          <div className={styles.docMetaPair}>
            <span className={styles.docMetaLabel}>Kind</span>
            <span className={styles.docMetaValue}>{formatKind(symbol.kind)}</span>
          </div>
          <div className={styles.docMetaPair}>
            <span className={styles.docMetaLabel}>Import</span>
            <span className={styles.docMetaValue}>
              <code>{symbol.import_path}</code>
            </span>
          </div>
          <div className={styles.docMetaPair}>
            <span className={styles.docMetaLabel}>Module</span>
            <span className={styles.docMetaValue}>
              <code>{symbol.module}</code>
            </span>
          </div>
        </div>

        {aliasList.length > 0 ? (
          <div className={styles.docMetaRow}>
            <span className={styles.docMetaLabel}>Aliases</span>
            <span className={styles.docMetaValue}>
              {aliasList.map((alias, index) => (
                <span key={alias}>
                  <code>{alias}</code>
                  {index < aliasList.length - 1 ? ', ' : null}
                </span>
              ))}
            </span>
          </div>
        ) : null}

        {symbol.status_note ? (
          <div className={styles.docMetaRow}>
            <span className={styles.docMetaLabel}>Status</span>
            <p className={styles.docBodyText}>{symbol.status_note}</p>
          </div>
        ) : null}
      </section>

      {symbol.body_sections && symbol.body_sections.length > 0 ? (
        <OperationDocBody
          sections={symbol.body_sections}
          headerSummary=""
          signature={symbol.signature}
          whestSourceUrl={symbol.source_url}
          upstreamSourceUrl={symbol.upstream_source_url}
        />
      ) : (
        <OperationDocSignature
          signature={symbol.signature}
          whestSourceUrl={symbol.source_url}
          upstreamSourceUrl={symbol.upstream_source_url}
        />
      )}

      <OperationDocSection title="Members" links={memberLinks} />
      <OperationDocSection title="Related guides" links={guideLinks} />
    </>
  );
}
