import { useCallback, useEffect, useRef, useState } from 'react';
import PanZoomControls, { PanZoomHint } from './PanZoomControls.jsx';

/**
 * Lightweight pan/zoom wrapper for arbitrary children (e.g. SVG diagrams
 * that don't speak ReactFlow).
 *
 * Gesture contract:
 *   - Plain scroll passes through to the page. This is deliberate: users
 *     skimming the page by scrolling past a canvas must never get stuck
 *     inside it. The hint in the corner advertises the real zoom gesture.
 *   - ⌘-scroll (macOS) / Ctrl-scroll zooms around the pointer.
 *   - Click + drag pans.
 *   - Double-click resets to the initial viewport.
 *   - Buttons in the corner step zoom in fixed increments / reset.
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
    // Only zoom when the user explicitly opts in with ⌘ / Ctrl. Plain
    // wheel scrolls the page as normal — lets a reader skim past.
    if (!(event.ctrlKey || event.metaKey)) return;

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
    // Attach wheel as non-passive so preventDefault works when the user
    // IS zooming. When they aren't, we early-return above without
    // calling preventDefault — the event bubbles normally.
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

      <PanZoomHint className="absolute bottom-2 left-2" />
      <PanZoomControls
        onZoomIn={() => zoomBy(1.2)}
        onZoomOut={() => zoomBy(1 / 1.2)}
        onReset={reset}
        className="absolute bottom-2 right-2"
      />
    </div>
  );
}
