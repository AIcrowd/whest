'use client';

import dynamic from 'next/dynamic';

const SymmetryAwareEinsumContractionsApp = dynamic(
  () =>
    import('./SymmetryAwareEinsumContractionsApp.jsx').then((mod) => ({
      default: mod.default,
    })),
  { ssr: false, loading: () => <div>Loading Symmetry Aware Einsum Contractions...</div> },
);

export default function SymmetryAwareEinsumContractionsAppWrapper() {
  return <SymmetryAwareEinsumContractionsApp />;
}
