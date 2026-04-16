import { useEffect } from 'react';

function isTextInput(el) {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
  if (el.isContentEditable) return true;
  return false;
}

/**
 * Bind a map of single-key → handler for keyboard shortcuts. Skips dispatch
 * when the active element is a text input so users can type freely.
 *
 * Usage:
 *   useKeyboardShortcuts({
 *     ArrowLeft: () => prev(),
 *     ArrowRight: () => next(),
 *     r: () => randomize(),
 *     '/': () => focusSearch(),
 *   });
 */
export function useKeyboardShortcuts(bindings) {
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    function onKey(e) {
      if (isTextInput(document.activeElement)) return;
      const handler = bindings[e.key];
      if (handler) {
        e.preventDefault();
        handler(e);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bindings]);
}
