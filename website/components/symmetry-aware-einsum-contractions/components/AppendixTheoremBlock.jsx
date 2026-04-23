import InlineMathText from './InlineMathText.jsx';

export default function AppendixTheoremBlock({ kind, children, lead = null }) {
  return (
    <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
      <span className="font-semibold text-gray-900">
        {kind}.
      </span>
      {lead ? (
        <span className="ml-1 font-semibold text-gray-900">
          <InlineMathText>{lead}</InlineMathText>
        </span>
      ) : null}
      {' '}
      <span className="italic">
        <InlineMathText>{children}</InlineMathText>
      </span>
    </p>
  );
}
