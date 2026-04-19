'use client';

import { Check, Clipboard } from 'lucide-react';
import {
  Children,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { motion, useInView } from 'motion/react';
import { cn } from '@/lib/utils';

// =========================================================================
// HomeCodeTerminal — Whest Design System "paper" code block.
// Pure-white canvas, 1px gray-200 border, 8px radius. Replaces the old
// #071018 dark terminal with traffic-light chrome.
//
// Keeps the staggered line-reveal motion (the one moment of motion the
// home page earns) via a lightweight sequence context. Lines wrap their
// children in <AnimatedSpan> and reveal in order — same behaviour as the
// previous <Terminal> primitive, stripped of the shell-window framing.
// =========================================================================

interface SequenceContextValue {
  completeItem: (index: number) => void;
  activeIndex: number;
  sequenceStarted: boolean;
}

const SequenceContext = createContext<SequenceContextValue | null>(null);
const ItemIndexContext = createContext<number | null>(null);

function useSequence() {
  return useContext(SequenceContext);
}
function useItemIndex() {
  return useContext(ItemIndexContext);
}

export const AnimatedSpan = ({
  children,
  delay = 0,
  className,
  startOnView = false,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  startOnView?: boolean;
}) => {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(elementRef as React.RefObject<Element>, {
    amount: 0.3,
    once: true,
  });

  const sequence = useSequence();
  const itemIndex = useItemIndex();
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    if (!sequence || itemIndex === null || !sequence.sequenceStarted || hasStarted) return;
    if (sequence.activeIndex === itemIndex) {
      setHasStarted(true);
    }
  }, [sequence, itemIndex, hasStarted]);

  const shouldAnimate = sequence ? hasStarted : startOnView ? isInView : true;

  return (
    <motion.div
      ref={elementRef}
      initial={{ opacity: 0, y: -4 }}
      animate={shouldAnimate ? { opacity: 1, y: 0 } : { opacity: 0, y: -4 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1], delay: sequence ? 0 : delay / 1000 }}
      className={cn('grid', className)}
      onAnimationComplete={() => {
        if (!sequence || itemIndex === null) return;
        sequence.completeItem(itemIndex);
      }}
    >
      {children}
    </motion.div>
  );
};

export default function HomeCodeTerminal({
  children,
  copyText,
  className,
  label,
  labelAccent = false,
  center = false,
  sequence = false,
  startOnView = false,
}: {
  children: ReactNode;
  copyText: string;
  className?: string;
  label?: string;
  labelAccent?: boolean;
  center?: boolean;
  sequence?: boolean;
  startOnView?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(containerRef as React.RefObject<Element>, {
    amount: 0.3,
    once: true,
  });
  const [activeIndex, setActiveIndex] = useState(0);
  const sequenceHasStarted = sequence ? !startOnView || isInView : false;

  useEffect(() => {
    if (!copied) return undefined;
    const timeout = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timeout);
  }, [copied]);

  async function handleCopy() {
    await navigator.clipboard.writeText(copyText);
    setCopied(true);
  }

  const contextValue = useMemo<SequenceContextValue | null>(() => {
    if (!sequence) return null;
    return {
      completeItem: (index: number) => {
        setActiveIndex((current) => (index === current ? current + 1 : current));
      },
      activeIndex,
      sequenceStarted: sequenceHasStarted,
    };
  }, [activeIndex, sequence, sequenceHasStarted]);

  const wrappedChildren = useMemo(() => {
    if (!sequence) return children;
    return Children.toArray(children).map((child, index) => (
      <ItemIndexContext.Provider key={index} value={index}>
        {child as ReactNode}
      </ItemIndexContext.Provider>
    ));
  }, [children, sequence]);

  const content = (
    <div ref={containerRef} className={className}>
      {label ? (
        <div
          className={cn(
            'mb-3 flex items-center gap-2 font-sans text-[10px] font-semibold uppercase',
            labelAccent
              ? 'text-[#F0524D]'
              : 'text-gray-400 dark:text-gray-500',
          )}
          style={{ letterSpacing: '0.2em' }}
        >
          <span aria-hidden className={cn('h-px w-5', labelAccent ? 'bg-[#F0524D]' : 'bg-gray-300 dark:bg-gray-600')} />
          {label}
        </div>
      ) : null}

      <div
        className={cn(
          'relative overflow-hidden rounded-lg border border-[var(--gray-200)] bg-white',
          'dark:border-[#2B2F30] dark:bg-[var(--code-bg-dark)]',
        )}
      >
        <button
          type="button"
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          className={cn(
            'absolute end-3 top-3 z-10 rounded-md p-1.5 text-gray-400 transition',
            'hover:bg-gray-100 hover:text-gray-900',
            'dark:hover:bg-[#22262A] dark:hover:text-gray-100',
          )}
        >
          {copied ? <Check className="size-3.5" /> : <Clipboard className="size-3.5" />}
        </button>
        <pre
          className={cn(
            'px-5 py-4 text-[13.5px] leading-[1.55] text-gray-900 dark:text-gray-100',
            center && 'overflow-x-auto',
          )}
          style={{ fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)' }}
        >
          <code className="grid gap-y-0.5 overflow-auto">{wrappedChildren}</code>
        </pre>
      </div>
    </div>
  );

  if (!sequence) return content;
  return <SequenceContext.Provider value={contextValue}>{content}</SequenceContext.Provider>;
}
