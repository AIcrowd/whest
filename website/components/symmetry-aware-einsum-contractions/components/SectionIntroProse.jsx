import InlineMathText from './InlineMathText.jsx';

export default function SectionIntroProse({ paragraphs, className = '' }) {
  return (
    <div className={['space-y-4', className].filter(Boolean).join(' ')}>
      {paragraphs.map((paragraph, index) => (
        <p key={index} className="font-serif text-[17px] leading-[1.75] text-gray-700">
          <InlineMathText>{paragraph}</InlineMathText>
        </p>
      ))}
    </div>
  );
}
