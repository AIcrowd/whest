import { AnchorLink } from './ExplorerSectionCard.jsx';

export default function ExplorerSubsectionHeader({
  anchorId,
  labelText,
  children,
  className = '',
}) {
  return (
    <div className={['min-w-0', className].filter(Boolean).join(' ')}>
      <div className="font-sans text-[12px] font-semibold uppercase tracking-[0.14em] text-coral">
        <AnchorLink anchorId={anchorId} labelText={labelText}>
          {children}
        </AnchorLink>
      </div>
    </div>
  );
}
