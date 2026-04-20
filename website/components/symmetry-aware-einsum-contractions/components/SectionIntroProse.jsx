import InlineMathText from './InlineMathText.jsx';

export default function SectionIntroProse({ paragraphs, className = '' }) {
  return (
    <div className={['grid gap-x-8 gap-y-4 md:grid-cols-2', className].filter(Boolean).join(' ')}>
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
