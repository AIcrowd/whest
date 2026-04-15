'use client';

import opsData from '../../public/ops.json';
import opDocsJson from '../../.generated/op-docs.json';
import opRefsJson from '../../.generated/op-refs.json';
import ApiReferenceInner from './ApiReference';
import type {Operation} from './OperationRow';

type OperationDocPreview = Pick<Operation, 'area' | 'display_type'> & {href: string};
type OperationRefEntry = {href: string; canonical_name: string};

export default function ApiReference() {
  const opDocs = opDocsJson as Record<string, OperationDocPreview>;
  const opRefs = opRefsJson as Record<string, OperationRefEntry>;

  const operations: Operation[] = opsData.operations.map((op) => {
    const ref = opRefs[op.name];
    const docRecord = ref ? opDocs[ref.canonical_name] : undefined;

    return {
      ...op,
      area: docRecord?.area ?? normalizeArea(op.module),
      display_type: docRecord?.display_type ?? displayTypeForCategory(op.category),
      href: op.blocked ? undefined : ref?.href,
    };
  });

  return <ApiReferenceInner operations={operations} />;
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
