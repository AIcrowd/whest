import InlineMathText from './InlineMathText.jsx';

export default function AppendixProofBlock({ children }) {
  return (
    <p className="font-serif text-[16px] leading-[1.85] text-gray-700">
      <span className="mr-1 italic font-semibold text-gray-800">Proof.</span>
      <InlineMathText>{children}</InlineMathText>
    </p>
  );
}
