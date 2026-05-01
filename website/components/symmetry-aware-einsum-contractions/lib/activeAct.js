export function mergeObservedActEntries(previousEntries, nextEntries) {
  const merged = new Map(previousEntries);

  (nextEntries ?? []).forEach((entry) => {
    merged.set(entry.target.id, entry);
  });

  return merged;
}

export function pickTopVisibleAct(entries, fallbackId) {
  const visible = (entries ?? [])
    .filter((entry) => entry.isIntersecting)
    .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

  const entered = visible.filter((entry) => entry.boundingClientRect.top >= 0);

  return (entered[0] ?? visible[0])?.target?.id ?? fallbackId;
}
