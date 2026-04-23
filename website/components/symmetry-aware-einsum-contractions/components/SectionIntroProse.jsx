import InlineMathText from './InlineMathText.jsx';

export default function SectionIntroProse({
  paragraphs,
  className = '',
  columns = 'two',
}) {
  const gridClassName = columns === 'one'
    ? 'grid gap-y-4'
    : 'editorial-two-col-divider-md grid gap-x-8 gap-y-4 md:grid-cols-2';
  const columnInsetClassName = columns === 'one'
    ? ''
    : 'md:px-4';

  return (
    <div className={[gridClassName, className].filter(Boolean).join(' ')}>
      {paragraphs.map((paragraph, index) => (
        <div key={index} className={columnInsetClassName}>
          <p
            className="font-serif text-[17px] leading-[1.75] text-gray-700"
            style={{ textAlign: 'justify' }}
          >
            <InlineMathText>{paragraph}</InlineMathText>
          </p>
        </div>
      ))}
    </div>
  );
}
