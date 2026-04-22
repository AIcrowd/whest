import { Fragment } from 'react';
import InlineMathText from '../components/InlineMathText.jsx';

export function renderProseBlocks(blocks = [], { renderCallout } = {}) {
  return blocks.map((block, index) => {
    if (block.kind === 'callout') {
      return renderCallout ? (
        <Fragment key={`${block.kind}-${index}`}>
          {renderCallout(block, index)}
        </Fragment>
      ) : (
        <InlineMathText key={`${block.kind}-${index}`}>{block.text}</InlineMathText>
      );
    }

    return (
      <InlineMathText key={`${block.kind}-${index}`}>
        {block.text}
      </InlineMathText>
    );
  });
}

export default renderProseBlocks;
