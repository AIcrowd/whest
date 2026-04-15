import Link from 'next/link';

import refsJson from '../.generated/op-refs.json';

type OpRefEntry = {
  canonical_name: string;
  href: string;
  label: string;
};

const refs = refsJson as Record<string, OpRefEntry>;

function stripCodeTicks(value: string): string {
  return value.replace(/`/g, '');
}

export default function OpRef({name}: {name: string}) {
  const ref = refs[name];

  if (!ref) {
    throw new Error(`Unknown op ref: ${name}`);
  }

  return (
    <Link href={ref.href}>
      <code>{stripCodeTicks(ref.label)}</code>
    </Link>
  );
}
