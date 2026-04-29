import { forwardRef, useImperativeHandle, useRef, useState } from 'react';

// Small floating tooltip that follows the cursor over the O → Q matrix.
//
// Performance contract: hover updates do NOT trigger React renders. The
// parent (OrbitRepMatrix) owns hover state in a ref and calls our
// imperative `update(content, position)` / `hide()` API on every rAF tick.
// Body content is written into a DOM node via textContent. The only
// React state we keep is `visible` (toggle display on mount/unmount), and
// that's set imperatively too — outside the render path.
//
// API (via ref):
//   tooltipRef.current.update({ contentLines, x, y }) — show + position
//   tooltipRef.current.hide()                         — hide

const STYLE = {
  base: {
    position: 'fixed',
    pointerEvents: 'none',
    zIndex: 50,
    background: '#FFFFFF',
    border: '1px solid #D9DCDC',
    borderRadius: 4,
    padding: '8px 10px',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: '#1F2526',
    lineHeight: 1.5,
    minWidth: 180,
    maxWidth: 320,
    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    transition: 'opacity 120ms cubic-bezier(0.4, 0, 0.2, 1)',
  },
};

function MatrixHoverTooltip(_props, ref) {
  const wrapperRef = useRef(null);
  const lineRefs = useRef([]); // refs to <div> children (one per line)
  const [visible, setVisible] = useState(false);

  useImperativeHandle(ref, () => ({
    update({ contentLines, x, y }) {
      const el = wrapperRef.current;
      if (!el) return;
      // Write content imperatively (no React render).
      contentLines.forEach((line, i) => {
        const lineEl = lineRefs.current[i];
        if (lineEl) lineEl.textContent = line;
      });
      // Hide any leftover lines from a previous longer tooltip.
      for (let i = contentLines.length; i < lineRefs.current.length; i += 1) {
        const lineEl = lineRefs.current[i];
        if (lineEl) lineEl.textContent = '';
      }
      // Position via CSS transform — cheaper than left/top during fast moves.
      el.style.transform = `translate(${x}px, ${y}px)`;
      if (!visible) {
        el.style.opacity = '1';
        setVisible(true);
      }
    },
    hide() {
      const el = wrapperRef.current;
      if (!el) return;
      el.style.opacity = '0';
      // Defer the visibility flip via React so it batches with other state.
      setVisible(false);
    },
  }), [visible]);

  return (
    <div
      ref={wrapperRef}
      data-testid="orbit-rep-matrix-tooltip"
      role="tooltip"
      aria-hidden={!visible}
      style={{
        ...STYLE.base,
        opacity: visible ? 1 : 0,
        // Initial transform off-screen until first update().
        transform: 'translate(-9999px, -9999px)',
      }}
    >
      {/* Three pre-allocated lines; the imperative update writes textContent. */}
      <div ref={(el) => { lineRefs.current[0] = el; }} style={{ fontWeight: 600 }} />
      <div ref={(el) => { lineRefs.current[1] = el; }} style={{ color: '#9AA0A0' }} />
      <div ref={(el) => { lineRefs.current[2] = el; }} style={{ color: '#B23E3A', fontWeight: 600 }} />
    </div>
  );
}

export default forwardRef(MatrixHoverTooltip);
