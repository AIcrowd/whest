export default function NarrativeCallout({ label, tone = 'muted', children }) {
  const toneClass = tone === 'accent'
    ? 'border-coral/25 bg-coral-light/60'
    : 'border-gray-200 bg-gray-50';

  return (
    <div className={`rounded-xl border px-5 py-4 ${toneClass}`}>
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-500">{label}</div>
      <p className="mt-2 text-[15px] leading-7 text-gray-700">{children}</p>
    </div>
  );
}
