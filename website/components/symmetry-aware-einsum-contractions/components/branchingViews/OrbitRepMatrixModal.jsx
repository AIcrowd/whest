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
  hover,                    // was `pin`
  onHoverChange,            // was `onPin`
  expressionInfo,
  componentInfo,
  selectedOrbitIdx = -1,
  onSelectOrbit = () => {},
  onHover = null,           // forwarded to inner matrix for compat with the legacy hover-graph callback
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
        className="relative flex h-[min(820px,90vh)] w-[min(1280px,94vw)] flex-col overflow-hidden rounded-lg bg-white p-6"
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
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: 'var(--coral)' }}>
          The O → Q matrix · expanded
        </div>
        <div className="mt-3 grid min-h-0 flex-1 grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="min-h-0 min-w-0 overflow-hidden">
            <OrbitRepMatrix
              orbitRows={orbitRows}
              selectedOrbitIdx={selectedOrbitIdx}
              onSelectOrbit={onSelectOrbit}
              onHover={onHover}
              expressionInfo={expressionInfo}
              componentInfo={componentInfo}
              hover={hover}
              onHoverChange={onHoverChange}
              canvasHeight={620}
            />
          </div>
          <div className="min-h-0 min-w-0 overflow-y-auto overflow-x-hidden rounded-md border border-gray-200 bg-white">
            {hover ? (
              <OrbitDetailCard
                hover={hover}
                orbitRows={orbitRows}
                reps={reps}
                cells={cells}
                expressionInfo={expressionInfo}
                componentInfo={componentInfo}
                onDismiss={() => onHoverChange && onHoverChange(null)}
                mode="inline"
              />
            ) : (
              <div data-testid="orbit-rep-matrix-modal-empty-detail" className="flex h-full flex-col justify-center p-6">
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-gray-900">
                  Inspect a cell
                </div>
                <p className="mt-3 font-serif text-[14px] leading-7 text-gray-600">
                  Hover over a filled or empty cell to see the product orbit, its projected output destination, and why that row contributes to <span className="font-semibold text-gray-800">α</span>.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
