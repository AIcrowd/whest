import {highlight} from 'fumadocs-core/highlight';
import {CodeBlock, Pre} from 'fumadocs-ui/components/codeblock';
import {Terminal} from 'lucide-react';
import {
  Children,
  isValidElement,
  type ComponentProps,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from 'react';
import styles from './styles.module.css';
import type {DocExample} from './op-doc-types';

const editorTheme = {
  light: 'github-light',
  dark: 'github-dark',
} as const;

type HighlightedPre = ReactElement<{
  children?: ReactNode;
  className?: string;
  style?: CSSProperties;
  tabIndex?: number;
}>;

function renderHighlightedBlock(highlighted: ReactNode, props?: ComponentProps<typeof CodeBlock>) {
  const preNode = Children.toArray(highlighted).find(
    (child): child is HighlightedPre => isValidElement(child) && child.type === 'pre',
  );

  if (!preNode) {
    return <CodeBlock {...props}>{highlighted}</CodeBlock>;
  }

  return (
    <CodeBlock {...props} className={preNode.props.className} style={preNode.props.style}>
      <Pre tabIndex={preNode.props.tabIndex}>{preNode.props.children}</Pre>
    </CodeBlock>
  );
}

export default async function OperationDocExample({example}: {example?: DocExample | null}) {
  if (!example) {
    return null;
  }

  const [highlightedCode, highlightedOutput] = await Promise.all([
    highlight(example.code, {lang: 'python', themes: editorTheme}),
    example.output
      ? highlight(example.output, {lang: 'python', themes: editorTheme})
      : Promise.resolve(null),
  ]);

  return (
    <section className={styles.docSection}>
      <h2>Examples</h2>
      {renderHighlightedBlock(highlightedCode, {allowCopy: false})}
      {highlightedOutput ? (
        renderHighlightedBlock(highlightedOutput, {
          title: 'output',
          icon: <Terminal className="size-3.5" />,
          className: styles.docOutputSpacing,
        })
      ) : null}
    </section>
  );
}
