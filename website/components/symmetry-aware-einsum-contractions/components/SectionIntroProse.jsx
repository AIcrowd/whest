import InlineMathText from './InlineMathText.jsx';

export default function SectionIntroProse({
  paragraphs,
  className = '',
  columns = 'two',
  balancedColumns = false,
}) {
  const gridClassName = columns === 'one'
    ? 'grid gap-y-4'
    : 'editorial-two-col-divider-md grid gap-x-8 gap-y-4 md:grid-cols-2';
  const columnInsetClassName = columns === 'one'
    ? ''
    : 'md:px-4';
  const paragraphClassName = 'font-serif text-[17px] leading-[1.75] text-gray-700';
  const renderParagraph = (paragraph, index) => (
    <p
      key={index}
      className={paragraphClassName}
      style={{ textAlign: 'justify' }}
    >
      <InlineMathText>{paragraph}</InlineMathText>
    </p>
  );

  if (balancedColumns && columns !== 'one' && paragraphs.length > 2) {
    const [leadParagraph, ...remainingParagraphs] = paragraphs;

    return (
      <div className={[gridClassName, className].filter(Boolean).join(' ')}>
        <div className={columnInsetClassName}>{renderParagraph(leadParagraph, 0)}</div>
        <div className={[columnInsetClassName, 'space-y-4'].filter(Boolean).join(' ')}>
          {remainingParagraphs.map((paragraph, index) => renderParagraph(paragraph, index + 1))}
        </div>
      </div>
    );
  }

  return (
    <div className={[gridClassName, className].filter(Boolean).join(' ')}>
      {paragraphs.map((paragraph, index) => (
        <div key={index} className={columnInsetClassName}>
          {renderParagraph(paragraph, index)}
        </div>
      ))}
    </div>
  );
}
