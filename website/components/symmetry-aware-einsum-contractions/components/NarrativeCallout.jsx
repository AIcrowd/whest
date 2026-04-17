import ExplorerSectionCard from './ExplorerSectionCard.jsx';

export default function NarrativeCallout({ label, tone = 'muted', children }) {
  const toneClass = {
    muted: 'border-border/70 bg-muted/30',
    algorithm: 'border-border/70 bg-white',
    accent: 'border-coral/20 bg-coral-light/60',
  }[tone] ?? 'border-border/70 bg-muted/30';

  // h-full stretches the card to match its sibling in a grid row; the content
  // area then grows (flex-1) so the shorter paragraph can be vertically
  // centred inside the card. The eyebrow stays pinned at the top via the
  // CardHeader — only the body text recentres.
  return (
    <ExplorerSectionCard
      eyebrow={label}
      className={`${toneClass} h-full`}
      contentClassName="pt-3 flex flex-1 flex-col justify-center"
    >
      <p className="text-[15px] leading-7 text-gray-700">{children}</p>
    </ExplorerSectionCard>
  );
}
