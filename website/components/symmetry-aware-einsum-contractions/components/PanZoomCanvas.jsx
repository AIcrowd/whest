import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Lightweight pan/zoom wrapper for arbitrary children (e.g. SVG diagrams
 * that don't speak ReactFlow).
 *
 * - Wheel / trackpad pinch zooms around the pointer.
 * - Click + drag pans.
 * - Double-click resets to the initial viewport.
 * - Plus / minus buttons step the zoom in fixed increments.
 */
export default function PanZoomCanvas({
  children,
  className = '',
  minZoom = 0.4,
  maxZoom = 4,
  ariaLabel,
}) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef(null);
  const containerRef = useRef(null);

  const clampScale = useCallback(
    (next) => Math.max(minZoom, Math.min(maxZoom, next)),
    [minZoom, maxZoom],
  );

  const reset = useCallback(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  const zoomBy = useCallback((factor) => {
    setScale((prev) => clampScale(prev * factor));
  }, [clampScale]);

  const handleWheel = useCallback((event) => {
    if (!containerRef.current) return;
    // Only zoom when the pointer is over the canvas; let the page scroll otherwise.
    event.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const px = event.clientX - rect.left - rect.width / 2;
    const py = event.clientY - rect.top - rect.height / 2;
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
    setScale((prev) => {
      const next = clampScale(prev * factor);
      if (next === prev) return prev;
      // Keep the point under the pointer fixed while zooming.
      const k = next / prev;
      setOffset((prevOffset) => ({
        x: px - k * (px - prevOffset.x),
        y: py - k * (py - prevOffset.y),
      }));
      return next;
    });
  }, [clampScale]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    // Attach wheel as non-passive so preventDefault works.
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  const handlePointerDown = (event) => {
    if (event.button !== 0) return;
    const target = event.currentTarget;
    target.setPointerCapture(event.pointerId);
    dragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      baseX: offset.x,
      baseY: offset.y,
    };
  };

  const handlePointerMove = (event) => {
    if (!dragRef.current) return;
    const { startX, startY, baseX, baseY } = dragRef.current;
    setOffset({
      x: baseX + (event.clientX - startX),
      y: baseY + (event.clientY - startY),
    });
  };

  const handlePointerUp = (event) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  };

  return (
    <div className={`relative ${className}`} aria-label={ariaLabel}>
      <div
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDoubleClick={reset}
        className="relative h-full w-full touch-none select-none overflow-hidden"
        style={{ cursor: dragRef.current ? 'grabbing' : 'grab' }}
      >
        <div
          className="absolute inset-0 flex origin-center items-center justify-center"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transformOrigin: '50% 50%',
          }}
        >
          {children}
        </div>
      </div>

      <div className="pointer-events-auto absolute bottom-2 right-2 flex flex-col gap-1 rounded-md border border-border bg-white/90 p-1 text-xs shadow-sm backdrop-blur">
        <button
          type="button"
          aria-label="Zoom in"
          onClick={() => zoomBy(1.2)}
          className="flex h-6 w-6 items-center justify-center rounded text-foreground hover:bg-surface-raised"
        >
          +
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          onClick={() => zoomBy(1 / 1.2)}
          className="flex h-6 w-6 items-center justify-center rounded text-foreground hover:bg-surface-raised"
        >
          −
        </button>
        <button
          type="button"
          aria-label="Reset view"
          onClick={reset}
          className="flex h-6 w-6 items-center justify-center rounded text-foreground hover:bg-surface-raised"
          title="Reset"
        >
          ⟳
        </button>
      </div>
    </div>
  );
}
