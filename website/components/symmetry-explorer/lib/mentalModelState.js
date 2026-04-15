const CONTEXT_CHANGE_ACTIONS = new Set([
  'selectPreset',
  'customMode',
  'customExample',
]);

export function reduceMentalModelVisibility(isVisible, action) {
  if (!isVisible) return false;
  if (CONTEXT_CHANGE_ACTIONS.has(action)) return false;
  return isVisible;
}
