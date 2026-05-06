// savingsCategoryBus.js — module-level pub/sub for the C40 Cost Savings Spread
// hover state.
//
// Why a bus? The C40 Cost Savings Spread surfaces the dense-vs-symmetry-aware
// columns plus a small supporting-rows table (Product chains / Updates /
// Active alpha method / Speedup / Savings). Hovering a row should cross-link
// to the matching visual elsewhere in §5: the product-chain row highlights
// the product-orbit visuals, the updates row highlights the O→Q matrix, and
// the active-method row highlights the classification tree + partition
// counter (which is already covered by alphaMethodBus.js).
//
// Rather than thread a new prop chain through ComponentCostView and friends,
// a tiny module-level bus mirrors the alphaMethodBus pattern (see
// alphaMethodBus.js for the original rationale). One-directional flow:
//
//   CostSavingsSpread → publishes a category id when a row is hovered/focused
//   downstream visuals → subscribe via useSyncExternalStore
//
// Categories are a small fixed enum so subscribers can pattern-match without
// stringly-typed surprises:
//
//   'product-chains'  → product-orbit visuals (M_a, μ)
//   'updates'         → O→Q accumulation matrix (α)
//   'active-method'   → classification tree + partition counter
//   'speedup'         → no-op highlight by default; reserved for future use
//   'savings'         → no-op highlight by default; reserved for future use
//   null              → no row hovered

let currentCategory = null;
const subscribers = new Set();

/**
 * Publish a new active savings-row category. Pass `null` to clear.
 */
export function setActiveSavingsCategory(categoryId) {
  if (categoryId === currentCategory) return;
  currentCategory = categoryId;
  for (const notify of subscribers) notify(currentCategory);
}

/**
 * Subscribe to savings-category changes. Returns an unsubscribe fn.
 * Designed for useSyncExternalStore: subscribe + getSnapshot.
 */
export function subscribeActiveSavingsCategory(callback) {
  subscribers.add(callback);
  return () => { subscribers.delete(callback); };
}

export function getActiveSavingsCategory() {
  return currentCategory;
}
