import { useState } from 'react';

// useShowMore — small hook for "show first N, click to expand" lists.
//
// Returns:
//   visible : the slice of items currently rendered (full array when expanded).
//   showAll : bool — whether the list is fully expanded.
//   toggle  : () => void — flips the expanded state.
//   hidden  : number — how many items are currently hidden (0 when expanded).
//   total   : number — items.length (convenience for callers building the
//             "+N more" affordance).
//
// Default limit of 8 matches the rhythm of the editorial subsection cards on
// the einsum-contractions page (μ + α-hard + per-component table all sit
// around 6–8 lines before scrolling). Pass a different limit explicitly when
// the caller has a more constrained slot.
export function useShowMore(items = [], limit = 8) {
  const [showAll, setShowAll] = useState(false);
  const total = items.length;
  const visible = showAll ? items : items.slice(0, limit);
  const hidden = Math.max(0, total - limit);
  return {
    visible,
    showAll,
    toggle: () => setShowAll((v) => !v),
    setShowAll,
    hidden,
    total,
  };
}
