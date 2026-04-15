import {CodeBlock, Pre} from 'fumadocs-ui/components/codeblock';
import RichOutputPre from '../rich-output-pre';
import styles from './styles.module.css';
import type {DocExample} from './op-doc-types';

export default function OperationDocExample({example}: {example?: DocExample | null}) {
  if (!example) {
    return null;
  }

  return (
    <section className={styles.docSection}>
      <h2>Examples</h2>
      <CodeBlock
        title="python"
        viewportProps={{className: styles.docCodeViewport}}
        className={styles.docCodeBlock}
      >
        <Pre>{example.code}</Pre>
      </CodeBlock>
      {example.output ? <RichOutputPre title="output">{example.output}</RichOutputPre> : null}
    </section>
  );
}
