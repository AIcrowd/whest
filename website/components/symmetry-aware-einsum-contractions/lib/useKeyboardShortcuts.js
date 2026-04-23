import { useEffect } from 'react';

function isTextInput(el) {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
  if (el.isContentEditable) return true;
  return false;
}

const modifierKeys = ['metaKey', 'ctrlKey', 'shiftKey', 'altKey'];

function normalizeBindings(bindings) {
  if (Array.isArray(bindings)) return bindings;
  return Object.entries(bindings ?? {}).map(([key, handler]) => ({ key, handler }));
}

function modifiersMatch(entry, event) {
  return modifierKeys.every((modifier) => event[modifier] === (entry.modifiers?.[modifier] ?? false));
}

/**
 * Bind keyboard shortcuts. Supports either:
 * - an object map of single-key → handler
 * - an array of { key, handler, modifiers } entries
 *
 * Skips dispatch when the active element is a text input so users can type freely.
 */
export function useKeyboardShortcuts(bindings) {
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const normalizedBindings = normalizeBindings(bindings);
    function onKey(e) {
      if (isTextInput(document.activeElement)) return;
      const entry = normalizedBindings.find((candidate) => candidate.key === e.key && modifiersMatch(candidate, e));
      if (entry?.handler) {
        e.preventDefault();
        entry.handler(e);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bindings]);
}
