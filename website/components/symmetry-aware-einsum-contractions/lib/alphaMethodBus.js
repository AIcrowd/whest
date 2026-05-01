// alphaMethodBus.js — module-level pub/sub for the active α-method id.
//
// Why a bus? The α-method hover state lives in App-level React state
// (activeAlphaMethodHover) and flows down to StickyBar and ComponentCostView
// via props. DecisionLadder is rendered INSIDE ComponentCostView, which is
// not allowed to be modified in this task. Rather than add a new prop chain
// through ComponentCostView, we use a tiny module-level bus (the same pattern
// as CaseBadge's tooltipSubscribers) so:
//
//   App            → calls setActiveAlphaMethod(id)  (in a useEffect)
//   DecisionLadder → calls useActiveAlphaMethod()     (subscribes here)
//
// This keeps the API surface minimal and the coupling one-directional.

let currentMethod = null;
const subscribers = new Set();

/**
 * Publish a new active α-method id. Pass `null` to clear.
 * Called by the App when activeAlphaMethodHover changes.
 */
export function setActiveAlphaMethodBus(methodId) {
  if (methodId === currentMethod) return;
  currentMethod = methodId;
  for (const notify of subscribers) notify(currentMethod);
}

/**
 * Subscribe to α-method changes.
 * Returns the current value and an unsubscribe function.
 * Designed for useSyncExternalStore: subscribe + getSnapshot.
 */
export function subscribeActiveAlphaMethod(callback) {
  subscribers.add(callback);
  return () => { subscribers.delete(callback); };
}

export function getActiveAlphaMethod() {
  return currentMethod;
}
