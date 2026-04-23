export function denseTupleCountFromComponents(components = []) {
  if (!Array.isArray(components)) return 0;
  if (components.length === 0) return 1;
  let total = 1;
  for (const component of components) {
    const sizes = Array.isArray(component.sizes) ? component.sizes : [];
    for (const size of sizes) total *= size;
  }
  return total;
}

export function denseDirectEventCostFromComponents(components = [], numTerms = 1) {
  const denseTuples = denseTupleCountFromComponents(components);
  const multiplicationFactor = Math.max(numTerms - 1, 0);
  return multiplicationFactor * denseTuples + denseTuples;
}

export function denseGridScalingLatex({ labelCount = 0, hasHeterogeneousSizes = false } = {}) {
  if (hasHeterogeneousSizes) return String.raw`\prod_{\ell \in L} n_\ell`;
  if (labelCount <= 0) return '1';
  if (labelCount === 1) return 'n';
  return `n^{${labelCount}}`;
}

export function hasHeterogeneousLabelSizesFromOverrides(labelSizes = {}, defaultSize) {
  const values = Object.values(labelSizes ?? {});
  if (values.length === 0) return false;
  return new Set(values).size > 1;
}
