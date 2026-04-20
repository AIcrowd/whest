import InlineMathText from './InlineMathText.jsx';

export default function SectionIntroProse({
  paragraphs,
  className = '',
  columns = 'two',
}) {
  const gridClassName = columns === 'one'
    ? 'grid gap-y-4'
    : 'grid gap-x-8 gap-y-4 md:grid-cols-2';

  return (
    <div className={[gridClassName, className].filter(Boolean).join(' ')}>
      {paragraphs.map((paragraph, index) => (
        <p
          key={index}
          className="font-serif text-[17px] leading-[1.75] text-gray-700"
          style={{ textAlign: 'justify' }}
        >
          <InlineMathText>{paragraph}</InlineMathText>
        </p>
      ))}
    </div>
  );
}
