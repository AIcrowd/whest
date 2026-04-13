/**
 * Minimal permutation group library — JS port of mechestim/_perm_group.py
 */

export class Permutation {
  constructor(arrayForm) {
    this.arr = Array.isArray(arrayForm) ? arrayForm : [...arrayForm];
  }

  get size() { return this.arr.length; }

  get isIdentity() {
    return this.arr.every((v, i) => v === i);
  }

  compose(other) {
    // (self ∘ other)[i] = self[other[i]]
    return new Permutation(this.arr.map((_, i) => this.arr[other.arr[i]]));
  }

  inverse() {
    const inv = new Array(this.size);
    for (let i = 0; i < this.size; i++) inv[this.arr[i]] = i;
    return new Permutation(inv);
  }

  apply(i) { return this.arr[i]; }

  /** Disjoint cycles including fixed points as 1-cycles. */
  fullCyclicForm() {
    const visited = new Set();
    const cycles = [];
    for (let i = 0; i < this.size; i++) {
      if (visited.has(i)) continue;
      const cycle = [];
      let j = i;
      while (!visited.has(j)) {
        cycle.push(j);
        visited.add(j);
        j = this.arr[j];
      }
      cycles.push(cycle);
    }
    return cycles;
  }

  /** Disjoint cycles excluding fixed points. */
  cyclicForm() {
    return this.fullCyclicForm().filter(c => c.length > 1);
  }

  /** Human-readable cycle notation with labels. */
  cycleNotation(labels) {
    const cycles = this.cyclicForm();
    if (cycles.length === 0) return 'e';
    return cycles.map(c =>
      '(' + c.map(i => labels ? labels[i] : i).join(' ') + ')'
    ).join('');
  }

  key() { return this.arr.join(','); }

  equals(other) {
    if (this.size !== other.size) return false;
    return this.arr.every((v, i) => v === other.arr[i]);
  }

  static identity(n) {
    return new Permutation(Array.from({ length: n }, (_, i) => i));
  }
}

/**
 * Dimino's algorithm — enumerate all group elements from generators.
 * Port of mechestim/_perm_group.py:341-374
 */
export function dimino(generators) {
  const n = generators[0].size;
  const identity = Permutation.identity(n);
  const elements = [identity];
  const seen = new Set([identity.key()]);

  for (const gen of generators) {
    if (seen.has(gen.key())) continue;
    const coset = [gen];
    seen.add(gen.key());
    let newElements = [gen];
    while (newElements.length > 0) {
      const nextNew = [];
      for (const elem of newElements) {
        for (const g of generators) {
          const product = elem.compose(g);
          if (!seen.has(product.key())) {
            seen.add(product.key());
            nextNew.push(product);
          }
          const productR = g.compose(elem);
          if (!seen.has(productR.key())) {
            seen.add(productR.key());
            nextNew.push(productR);
          }
        }
      }
      newElements = nextNew;
      coset.push(...nextNew);
    }
    elements.push(...coset);
  }
  return elements;
}

/**
 * Burnside's lemma: count unique tensor elements.
 * Port of mechestim/_perm_group.py:238-272
 *
 * @param {Permutation[]} elements - all group elements
 * @param {number[]} sizes - dimension size for each position
 * @returns {{ perElement: Array, totalFixed: number, uniqueCount: number }}
 */
export function burnsideCount(elements, sizes) {
  const perElement = [];
  let totalFixed = 0;

  for (const g of elements) {
    const cycles = g.fullCyclicForm();
    let fixed = 1;
    const cycleDetails = cycles.map(cycle => {
      const sz = sizes[cycle[0]];
      fixed *= sz;
      return { cycle, size: sz };
    });
    perElement.push({
      element: g,
      cycles: cycleDetails,
      fixedCount: fixed,
    });
    totalFixed += fixed;
  }

  const uniqueCount = totalFixed / elements.length;
  return { perElement, totalFixed, uniqueCount: Math.round(uniqueCount) };
}
