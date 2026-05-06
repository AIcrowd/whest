import type { Metadata } from 'next';
import SymmetryAwareEinsumContractionsApp from '@/components/symmetry-aware-einsum-contractions';

export const metadata: Metadata = {
  title: 'Symmetry Aware Einsum Contractions',
  description: 'Interactive walkthrough of flopscope symmetry detection for einsum contractions.',
};

export default function Page() {
  return (
    <main className="symmetry-aware-einsum-contractions-page min-h-screen bg-background">
      <div className="symmetry-aware-einsum-contractions-page-shell w-full">
        <SymmetryAwareEinsumContractionsApp />
      </div>
    </main>
  );
}
