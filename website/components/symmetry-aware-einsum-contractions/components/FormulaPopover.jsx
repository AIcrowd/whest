import React, { useState, useRef, useEffect } from 'react';
import Latex from './Latex.jsx';
import { REGIME_SPEC } from '../engine/regimeSpec.js';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';

function lookup(id) {
  return REGIME_SPEC[id] || SHAPE_SPEC[id] || null;
}

export default function FormulaPopover({ regimeId, children }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const spec = lookup(regimeId);

  useEffect(() => {
    if (!open) return undefined;
    function onClick(e) {
      if (!ref.current) return;
      if (ref.current.contains(e.target)) return;
      setOpen(false);
    }
    window.addEventListener('mousedown', onClick);
    return () => window.removeEventListener('mousedown', onClick);
  }, [open]);

  if (!spec) return children;

  return (
    <span className="relative inline-block" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="cursor-help underline decoration-dotted underline-offset-2"
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        {children}
      </button>
      {open ? (
        <span
          role="dialog"
          aria-label={`Explain ${spec.label}`}
          className="absolute left-0 top-full z-40 mt-2 w-80 rounded-lg border border-gray-200 bg-white p-3 text-xs shadow-lg"
        >
          <div className="mb-1 text-sm font-semibold">{spec.label}</div>
          <div className="mb-2 text-[11px] uppercase tracking-wider text-muted-foreground">
            When: {spec.when}
          </div>
          <div className="mb-2 leading-relaxed text-gray-700">{spec.description}</div>
          <div className="text-xs">
            <Latex math={spec.latex} />
          </div>
        </span>
      ) : null}
    </span>
  );
}
