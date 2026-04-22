import {ServerCodeBlock} from 'fumadocs-ui/components/codeblock.rsc';
import {Terminal} from 'lucide-react';
import {apiCodeThemes} from './runtime-shiki-themes';
import styles from './styles.module.css';
import type {DocExample} from './op-doc-types';

export default async function OperationDocExample({example}: {example?: DocExample | null}) {
  if (!example) {
    return null;
  }

  return (
    <section className={styles.docSection}>
      <h2>Examples</h2>
      <ServerCodeBlock
        code={example.code}
        lang="python"
        themes={apiCodeThemes}
        codeblock={{
          allowCopy: false,
          keepBackground: true,
          className: styles.apiCodeFigure,
          viewportProps: {className: styles.apiCodeViewport},
        }}
      />
      {example.output ? (
        <ServerCodeBlock
          code={example.output}
          lang="python"
          themes={apiCodeThemes}
          codeblock={{
            allowCopy: false,
            keepBackground: true,
            title: 'output',
            icon: <Terminal className="size-3.5" />,
            className: `${styles.apiCodeFigure} ${styles.docOutputSpacing}`,
            viewportProps: {className: styles.apiCodeViewport},
          }}
        />
      ) : null}
    </section>
  );
}
