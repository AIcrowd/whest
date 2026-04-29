import { useEffect } from 'react';
import OrbitRepMatrix from './OrbitRepMatrix.jsx';
import OrbitDetailCard from './OrbitDetailCard.jsx';

const COLOR = {
  backdrop: 'rgba(41, 44, 45, 0.6)',  // gray-900 at 0.6 opacity
};

export default function OrbitRepMatrixModal({
  open,
  onClose,
  orbitRows,
  reps,
  cells,
  pin,
  onPin,
  expressionInfo,
  componentInfo,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,           // ignored — modal uses tooltip for hover, but keep prop for compat
}) {
  useEffect(() => {
    if (!open) return;
    function onKey(e) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="The O → Q matrix — expanded"
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: COLOR.backdrop }}
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-lg p-6 max-w-[1200px] w-[90vw] max-h-[90vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute top-3 right-4 text-gray-400 hover:text-gray-900 transition-colors text-lg"
          aria-label="Close"
        >
          ×
        </button>
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: '#F0524D' }}>
          The O → Q matrix · expanded
        </div>
        <div className="mt-3 grid gap-6 grid-cols-1 lg:grid-cols-[2fr_1fr]">
          <OrbitRepMatrix
            orbitRows={orbitRows}
            selectedOrbitIdx={selectedOrbitIdx}
            onSelectOrbit={onSelectOrbit}
            onHover={onHover}
            expressionInfo={expressionInfo}
            componentInfo={componentInfo}
            pin={pin}
            onPin={onPin}
          />
          <OrbitDetailCard
            pin={pin}
            orbitRows={orbitRows}
            reps={reps}
            cells={cells}
            expressionInfo={expressionInfo}
            componentInfo={componentInfo}
            onDismiss={() => onPin && onPin(null)}
            mode="inline"
          />
        </div>
      </div>
    </div>
  );
}
