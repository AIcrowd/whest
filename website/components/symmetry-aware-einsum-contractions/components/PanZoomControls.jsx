/**
 * Shared zoom / reset button panel used by both the custom PanZoomCanvas
 * (SVG diagrams) and the ReactFlow-based DecisionLadder. Keeping one
 * visual style here avoids the two widgets drifting apart.
 */

import { useEffect, useState } from 'react';

const BTN_CLASS =
  'flex h-7 w-7 items-center justify-center rounded text-foreground transition-colors hover:bg-surface-raised';

export default function PanZoomControls({ onZoomIn, onZoomOut, onReset, className = '' }) {
  return (
    <div
      className={`pointer-events-auto flex flex-col gap-0.5 rounded-md border border-border bg-white/90 p-1 text-sm shadow-sm backdrop-blur ${className}`}
      role="group"
      aria-label="Pan and zoom controls"
    >
      <button type="button" aria-label="Zoom in" title="Zoom in" onClick={onZoomIn} className={BTN_CLASS}>
        +
      </button>
      <button type="button" aria-label="Zoom out" title="Zoom out" onClick={onZoomOut} className={BTN_CLASS}>
        −
      </button>
      <button type="button" aria-label="Reset view" title="Reset view" onClick={onReset} className={BTN_CLASS}>
        ⟳
      </button>
    </div>
  );
}

/**
 * Small floating hint in the corner of pan/zoom canvases that tells the
 * user the gesture has changed: scrolling alone now just scrolls the
 * page, and zoom requires holding ⌘ (mac) / Ctrl (other). Without this
 * hint, users who land on the page and try to scroll past an
 * interactive graph end up accidentally zooming and getting stuck.
 */
export function PanZoomHint({ className = '' }) {
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    if (typeof navigator === 'undefined') return;
    const platform = navigator.userAgentData?.platform ?? navigator.platform ?? '';
    setIsMac(/mac/i.test(platform));
  }, []);

  const key = isMac ? '⌘' : 'Ctrl';
  return (
    <div
      className={`pointer-events-none select-none rounded-md border border-border/70 bg-white/85 px-2 py-1 text-[11px] text-muted-foreground shadow-sm backdrop-blur ${className}`}
      aria-hidden="true"
    >
      <kbd className="rounded border border-border bg-surface-raised px-1 font-mono text-[10px] text-foreground">
        {key}
      </kbd>
      {' + scroll to zoom · drag to pan'}
    </div>
  );
}
