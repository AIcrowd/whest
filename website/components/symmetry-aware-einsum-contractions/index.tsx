'use client';

import { Loader2 } from 'lucide-react';
import dynamic from 'next/dynamic';

function SymmetryAwareEinsumContractionsLoading() {
  return (
    <div className="flex min-h-[100svh] items-center justify-center bg-white px-6">
      <div className="relative flex h-88 w-88 items-center justify-center">
        <Loader2
          aria-hidden
          className="pointer-events-none absolute h-full w-full animate-spin"
          strokeWidth={1.05}
          style={{ color: 'var(--coral)', opacity: 0.92 }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-[18%] rounded-full"
          style={{ border: '1px solid color-mix(in oklab, var(--coral) 18%, transparent)' }}
        />
        <span className="flopscope-wordmark relative z-10 text-[38px] leading-none sm:text-[48px]" aria-label="Flopscope.">
          Flopscope<span className="flopscope-wordmark__dot">.</span>
        </span>
      </div>
    </div>
  );
}

const SymmetryAwareEinsumContractionsApp = dynamic(
  () =>
    import('./SymmetryAwareEinsumContractionsApp.jsx').then((mod) => ({
      default: mod.default,
    })),
  { ssr: false, loading: () => <SymmetryAwareEinsumContractionsLoading /> },
);

export default function SymmetryAwareEinsumContractionsAppWrapper() {
  return <SymmetryAwareEinsumContractionsApp />;
}
