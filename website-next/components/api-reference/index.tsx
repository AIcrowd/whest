'use client';

import opsData from '../../public/ops.json';
import ApiReferenceInner from './ApiReference';

export default function ApiReference() {
  return <ApiReferenceInner operations={opsData.operations} />;
}
