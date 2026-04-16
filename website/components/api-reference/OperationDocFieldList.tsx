import type {ReactNode} from 'react';
import styles from './styles.module.css';

type OperationDocFieldListItem = {
  key: string;
  name: ReactNode;
  type?: ReactNode;
  body: ReactNode;
};

export default function OperationDocFieldList({
  items,
}: {
  items: OperationDocFieldListItem[];
}) {
  return (
    <dl className={styles.docDefinitionList}>
      {items.map((item) => (
        <div key={item.key} className={styles.docDefinitionItem}>
          <dt className={styles.docDefinitionTerm}>
            <span className={styles.docDefinitionLead}>
              <span className={styles.docDefinitionName}>{item.name}</span>
              {item.type ? (
                <>
                  <span className={styles.docDefinitionSeparator}>:</span>
                  <span className={styles.docDefinitionType}>{item.type}</span>
                </>
              ) : null}
            </span>
          </dt>
          <dd className={styles.docDefinitionBody}>{item.body}</dd>
        </div>
      ))}
    </dl>
  );
}
