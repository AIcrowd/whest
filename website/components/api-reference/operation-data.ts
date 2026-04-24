import opsData from '../../public/ops.json';
import opRefsJson from '../../.generated/op-refs.json';

type OperationRefEntry = {
  href: string;
  canonical_name: string;
};

export interface Operation {
  name: string;
  module: string;
  whest_ref: string;
  numpy_ref: string;
  category: string;
  cost_formula: string;
  cost_formula_latex: string;
  free: boolean;
  blocked: boolean;
  status: string;
  notes: string;
  weight: number;
  area: 'core' | 'linalg' | 'fft' | 'random' | 'stats';
  display_type: 'counted' | 'custom' | 'free' | 'blocked';
  href?: string;
}

export type OperationArea = Operation['area'];
export type OperationDisplayType = Operation['display_type'];

let cachedOperations: Operation[] | undefined;

export function getPublicApiOperations(): Operation[] {
  if (cachedOperations) return cachedOperations;

  const opRefs = opRefsJson as Record<string, OperationRefEntry>;
  const operationByName = new Map(opsData.operations.map((op) => [op.name, op]));

  cachedOperations = opsData.operations.map((op) => {
    const ref = opRefs[op.name];
    const canonical = operationByName.get(ref?.canonical_name ?? op.name);
    const area =
      op.area ??
      (canonical ? normalizeArea(canonical.module) : normalizeArea(op.module));
    const displayType = canonical
      ? op.display_type ?? displayTypeForCategory(canonical.category)
      : op.display_type ?? displayTypeForCategory(op.category);

    return {
      ...op,
      area: area as Operation['area'],
      display_type: displayType as Operation['display_type'],
      href: op.blocked ? undefined : op.detail_href ?? ref?.href,
    };
  });

  return cachedOperations;
}

function normalizeArea(module: string): Operation['area'] {
  if (module === 'numpy.linalg') return 'linalg';
  if (module === 'numpy.fft') return 'fft';
  if (module === 'numpy.random') return 'random';
  if (module === 'whest.stats') return 'stats';
  return 'core';
}

function displayTypeForCategory(category: string): Operation['display_type'] {
  if (category === 'blacklisted') return 'blocked';
  if (category === 'free') return 'free';
  if (category === 'counted_custom') return 'custom';
  if (category.startsWith('counted_')) return 'counted';
  return 'custom';
}
