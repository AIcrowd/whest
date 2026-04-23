import { Fragment } from 'react';
import InlineMathText from '../components/InlineMathText.jsx';

export function renderProseBlocks(
  blocks = [],
  { renderCallout, strongClassName = null, keyPrefix = 'prose' } = {},
) {
  return blocks.map((block, index) => {
    const blockKey = `${keyPrefix}-${block.kind}-${index}`;
    if (block.kind === 'callout') {
      return renderCallout ? (
        <Fragment key={blockKey}>
          {renderCallout(block, index)}
        </Fragment>
      ) : (
        <InlineMathText key={blockKey} strongClassName={strongClassName}>{block.text}</InlineMathText>
      );
    }

    return (
      <InlineMathText key={blockKey} strongClassName={strongClassName}>
        {block.text}
      </InlineMathText>
    );
  });
}

export default renderProseBlocks;
