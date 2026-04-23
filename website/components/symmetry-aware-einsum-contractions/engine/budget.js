// website/components/symmetry-aware-einsum-contractions/engine/budget.js
// Production default: 1.5M bounds UI latency (~100ms on a modern laptop).
// Offline validation can override via BRUTE_FORCE_BUDGET_OVERRIDE env var
// (Node only; browser context has no process.env, so production is unaffected).
const DEFAULT_BUDGET = 1_500_000;
export const BRUTE_FORCE_BUDGET = (typeof process !== 'undefined'
  && process.env
  && process.env.BRUTE_FORCE_BUDGET_OVERRIDE)
  ? Number(process.env.BRUTE_FORCE_BUDGET_OVERRIDE)
  : DEFAULT_BUDGET;

export function bruteForceEstimate(sizes, groupOrder) {
  let total = groupOrder;
  for (const s of sizes) total *= s;
  return total;
}

export function withinBruteForceBudget(sizes, groupOrder, budget = BRUTE_FORCE_BUDGET) {
  return bruteForceEstimate(sizes, groupOrder) <= budget;
}
