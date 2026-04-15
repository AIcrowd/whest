'use client';

import dynamic from 'next/dynamic';

const SymmetryExplorer = dynamic(
  () => import('./SymmetryExplorer.jsx').then(mod => ({default: mod.default})),
  { ssr: false, loading: () => <div>Loading Symmetry Explorer...</div> }
);

export default function SymmetryExplorerWrapper() {
  return <SymmetryExplorer />;
}
