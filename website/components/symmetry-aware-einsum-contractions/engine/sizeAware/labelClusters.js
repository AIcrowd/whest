// website/components/symmetry-aware-einsum-contractions/engine/sizeAware/labelClusters.js

/**
 * A "cluster" is a G-orbit on labels. Labels in the same cluster must share
 * a size, because the group maps one to another.
 *
 * Returns an array of { id, labels: string[], size: number|null }.
 * The default size is null until the caller fills it from UI or preset.
 */

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
  }
  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }
  union(a, b) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent[ra] = rb;
  }
}

export function computeLabelClusters(labels, generators, defaultSize = null) {
  const n = labels.length;
  const uf = new UnionFind(n);
  for (const g of generators) {
    for (let i = 0; i < n; i += 1) {
      uf.union(i, g.arr[i]);
    }
  }
  const groups = new Map();
  for (let i = 0; i < n; i += 1) {
    const root = uf.find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(i);
  }
  return [...groups.values()]
    .map((indices) => indices.map((i) => labels[i]).sort())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map((memberLabels) => ({
      id: memberLabels.join(','),
      labels: memberLabels,
      size: defaultSize,
    }));
}

export function validateClusterSizes(clusters) {
  for (const c of clusters) {
    if (!Number.isInteger(c.size) || c.size < 1) {
      throw new Error(
        `cluster ${c.id} has invalid size ${c.size} — expected positive integer`,
      );
    }
  }
}

/**
 * Build a label→size map from clusters. Every label in L must appear in
 * exactly one cluster.
 */
export function labelSizesMap(clusters) {
  const map = new Map();
  for (const c of clusters) {
    for (const label of c.labels) {
      if (map.has(label)) {
        throw new Error(`label ${label} appears in multiple clusters`);
      }
      map.set(label, c.size);
    }
  }
  return map;
}
