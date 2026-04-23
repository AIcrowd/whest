// website/components/symmetry-aware-einsum-contractions/engine/sizeAware/burnside.js

/**
 * Size-aware Burnside for orbit counting under a finite group action on
 * heterogeneous-size label assignments.
 *
 * M = (1 / |G|) * Σ_{g ∈ G}  Π_{c ∈ cycles(g)} n_c
 *
 * where n_c is the common size of the labels in cycle c. Within any cycle of
 * a valid symmetry g, all labels must have the same size (invariant asserted
 * during group construction).
 */

export function cyclesOfG(perm) {
  return perm.fullCyclicForm();
}

function commonSizeOrThrow(cycle, sizes) {
  const n0 = sizes[cycle[0]];
  for (const idx of cycle) {
    if (sizes[idx] !== n0) {
      throw new Error(
        `cycle size mismatch: labels ${cycle.join(',')} have sizes `
        + cycle.map(i => sizes[i]).join(',')
        + ` — a permutation can only mix labels of equal size.`,
      );
    }
  }
  return n0;
}

export function sizeAwareBurnside(elements, sizes) {
  if (!elements || elements.length === 0) {
    throw new Error('sizeAwareBurnside requires at least one group element');
  }
  let total = 0;
  for (const g of elements) {
    let contrib = 1;
    for (const cycle of cyclesOfG(g)) {
      contrib *= commonSizeOrThrow(cycle, sizes);
    }
    total += contrib;
  }
  if (total % elements.length !== 0) {
    throw new Error(
      `Burnside sum ${total} not divisible by |G|=${elements.length} — `
      + `group elements probably incomplete or inconsistent.`,
    );
  }
  return total / elements.length;
}
