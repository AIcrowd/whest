import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';

export default function SectionIntroProse({
  paragraphs,
  blocks = null,
  className = '',
  columns = 'two',
  balancedColumns = false,
}) {
  const normalizedBlocks = blocks ?? (paragraphs ?? []).map((text) => ({ kind: 'paragraph', text }));
  const gridClassName = columns === 'one'
    ? 'grid gap-y-4'
    : 'editorial-two-col-divider-md grid gap-x-8 gap-y-4 md:grid-cols-2';
  const columnInsetClassName = columns === 'one'
    ? ''
    : 'md:px-4';
  const paragraphBaseClassName = 'font-serif text-[17px] leading-[1.75] text-gray-700';
  const renderParagraph = (paragraph, index) => {
    const paragraphStyle = paragraph?.align === 'left' ? { textAlign: 'left' } : { textAlign: 'justify' };

    return (
      <p
        key={index}
        className={[
          paragraphBaseClassName,
          paragraph?.lead ? 'text-[18px] leading-[1.72] text-gray-800' : '',
        ].filter(Boolean).join(' ')}
        style={paragraphStyle}
      >
        <InlineMathText>{paragraph?.text ?? paragraph}</InlineMathText>
      </p>
    );
  };
  const renderHeading = (text, index) => (
    <div
      key={index}
      className="font-sans text-[11px] font-semibold uppercase text-gray-700"
      style={{ letterSpacing: '0.18em' }}
    >
      {text}
    </div>
  );
  const renderEquation = (block, index) => (
    <div
      key={index}
      className="my-3 w-full max-w-full overflow-x-auto border-l-2 border-gray-200 py-1 pl-4 pr-2 text-left text-[18px] text-gray-900"
      data-section-intro-equation="true"
      style={block?.compact ? { fontSize: 'clamp(11px, 2.9vw, 18px)' } : null}
    >
      {block?.label ? (
        <div
          className="mb-2 font-sans text-[10px] font-semibold uppercase text-gray-400"
          style={{ letterSpacing: '0.16em' }}
        >
          {block.label}
        </div>
      ) : null}
      <div className="min-w-max text-center">
        <Latex math={block?.math ?? ''} display />
      </div>
    </div>
  );
  const renderBlock = (block, index) => {
    if (block?.kind === 'equation') return renderEquation(block, index);
    if (block?.kind === 'heading') return renderHeading(block.text, index);
    return renderParagraph(block ?? '', index);
  };

  const hasAssignedColumns = columns !== 'one'
    && normalizedBlocks.some((block) => block?.column === 1 || block?.column === 2);
  if (hasAssignedColumns) {
    const firstColumnIndex = normalizedBlocks.findIndex((block) => block?.column === 1 || block?.column === 2);
    const leftBlocks = normalizedBlocks.filter((block) => block?.column === 1);
    const rightBlocks = normalizedBlocks.filter((block) => block?.column === 2);
    const fullBeforeBlocks = normalizedBlocks.filter((block, index) => block?.column === 'full' && index < firstColumnIndex);
    const fullAfterBlocks = normalizedBlocks.filter((block, index) => block?.column === 'full' && index > firstColumnIndex);
    return (
      <div className={className}>
        {fullBeforeBlocks.length ? (
          <div className="mb-5 min-w-0 space-y-4 md:px-4">
            {fullBeforeBlocks.map((block, index) => renderBlock(block, index))}
          </div>
        ) : null}
        <div className={gridClassName}>
          <div className={[columnInsetClassName, 'min-w-0 space-y-4'].filter(Boolean).join(' ')}>
            {leftBlocks.map((block, index) => renderBlock(block, index))}
          </div>
          <div className={[columnInsetClassName, 'min-w-0 space-y-4'].filter(Boolean).join(' ')}>
            {rightBlocks.map((block, index) => renderBlock(block, index + leftBlocks.length))}
          </div>
        </div>
        {fullAfterBlocks.length ? (
          <div className="mt-5 min-w-0 space-y-4 border-t border-gray-100 pt-5 md:px-4">
            {fullAfterBlocks.map((block, index) => renderBlock(block, index + leftBlocks.length + rightBlocks.length + fullBeforeBlocks.length))}
          </div>
        ) : null}
      </div>
    );
  }

  if (balancedColumns && columns !== 'one' && !blocks && (paragraphs ?? []).length > 2) {
    const [leadParagraph, ...remainingParagraphs] = paragraphs ?? [];

    return (
      <div className={[gridClassName, className].filter(Boolean).join(' ')}>
        <div className={[columnInsetClassName, 'min-w-0'].filter(Boolean).join(' ')}>{renderParagraph(leadParagraph, 0)}</div>
        <div className={[columnInsetClassName, 'min-w-0 space-y-4'].filter(Boolean).join(' ')}>
          {remainingParagraphs.map((paragraph, index) => renderParagraph(paragraph, index + 1))}
        </div>
      </div>
    );
  }

  return (
    <div className={[gridClassName, className].filter(Boolean).join(' ')}>
      {normalizedBlocks.map((block, index) => (
        <div key={index} className={[columnInsetClassName, 'min-w-0'].filter(Boolean).join(' ')}>
          {renderBlock(block, index)}
        </div>
      ))}
    </div>
  );
}
