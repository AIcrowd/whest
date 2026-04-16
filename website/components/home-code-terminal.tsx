'use client';

import { Check, Clipboard } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { Terminal as MagicTerminal } from '@/components/ui/terminal';

export default function HomeCodeTerminal({
  children,
  copyText,
  className,
  center = false,
  sequence = false,
  startOnView = false,
}: {
  children: ReactNode;
  copyText: string;
  className?: string;
  center?: boolean;
  sequence?: boolean;
  startOnView?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return undefined;
    const timeout = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timeout);
  }, [copied]);

  async function handleCopy() {
    await navigator.clipboard.writeText(copyText);
    setCopied(true);
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleCopy}
        aria-label={copied ? 'Copied' : 'Copy code'}
        className="absolute right-3 top-3 z-10 rounded-md p-1.5 text-slate-300 transition hover:bg-white/8 hover:text-white"
      >
        {copied ? <Check className="size-4" /> : <Clipboard className="size-4" />}
      </button>
      <MagicTerminal
        sequence={sequence}
        startOnView={startOnView}
        className={cn(
          'h-auto max-h-none max-w-none overflow-hidden rounded-2xl border-[#14303c] bg-[#071018] text-[#f8fafc] shadow-[0_18px_56px_rgba(7,16,24,0.18)] [&>div:first-child]:border-[#14303c] [&>div:first-child]:bg-[#071018] [&>div:first-child]:p-4 [&>pre]:bg-[#071018] [&>pre]:px-5 [&>pre]:py-4 [&>pre]:shadow-none [&>pre>code]:w-full',
          center && '[&>pre]:overflow-x-auto',
          className,
        )}
      >
        {children}
      </MagicTerminal>
    </div>
  );
}
