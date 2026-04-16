// website/components/symmetry-aware-einsum-contractions/engine/budget.js
export const BRUTE_FORCE_BUDGET = 1_500_000;

export function bruteForceEstimate(sizes, groupOrder) {
  let total = groupOrder;
  for (const s of sizes) total *= s;
  return total;
}

export function withinBruteForceBudget(sizes, groupOrder, budget = BRUTE_FORCE_BUDGET) {
  return bruteForceEstimate(sizes, groupOrder) <= budget;
}
