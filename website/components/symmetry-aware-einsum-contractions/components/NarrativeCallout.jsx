import ExplorerSectionCard from './ExplorerSectionCard.jsx';

export default function NarrativeCallout({ label, tone = 'muted', children }) {
  const toneClass = {
    muted: 'border-border/70 bg-muted/30',
    algorithm: 'border-border/70 bg-white',
    accent: 'border-coral/20 bg-coral-light/60',
  }[tone] ?? 'border-border/70 bg-muted/30';

  return (
    <ExplorerSectionCard
      eyebrow={label}
      className={toneClass}
      contentClassName="pt-3"
    >
      <p className="text-[15px] leading-7 text-gray-700">{children}</p>
    </ExplorerSectionCard>
  );
}
