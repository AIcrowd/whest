import styles from './styles.module.css';

const BAR_TITLES = new Set(['Parameters', 'Returns', 'Notes']);

export default function OperationDocSectionHeading({title}: {title: string}) {
  return (
    <h2 className={BAR_TITLES.has(title) ? styles.docSectionBar : undefined}>
      {title}
    </h2>
  );
}
