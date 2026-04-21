import InlineMathText from './InlineMathText.jsx';

export default function AppendixTheoremBlock({ kind, children, lead = null }) {
  return (
    <div className="my-6 border-l border-gray-200 pl-5">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-500">
        {kind}
      </div>
      {lead ? (
        <p className="mb-3 font-serif text-[16px] leading-[1.7] text-gray-700">
          <InlineMathText>{lead}</InlineMathText>
        </p>
      ) : null}
      <div className="font-serif text-[17px] leading-[1.8] text-gray-800">
        <InlineMathText>{children}</InlineMathText>
      </div>
    </div>
  );
}
