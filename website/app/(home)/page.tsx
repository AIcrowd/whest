import HomeCodeTerminal, { AnimatedSpan } from '@/components/home-code-terminal';
import { withBasePath } from '@/lib/base-path';
import type { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'whest',
  description:
    'A NumPy-compatible math library that counts every FLOP analytically, so compute budgets stop being guesswork.',
};

// =========================================================================
// Whest Design System — Home page, paper-register edition.
// Pure white canvas, left-aligned editorial layout, Newsreader display
// serif for headings + wordmark, Source Serif 4 for body prose, Inter for
// UI chrome + kickers, JetBrains Mono for code. Code palette derives from
// the same tokens as the Shiki `whest-paper` theme in lib/shiki-themes.ts.
// =========================================================================

const features = [
  {
    number: '508',
    label: 'Operations',
    desc: 'NumPy-compatible operations with analytical FLOP costs.',
  },
  {
    number: '\u03C0',
    label: 'Symmetry-aware',
    desc: 'Symmetry metadata propagates through the chain, so costs scale with unique elements.',
  },
  {
    number: '\u221E',
    label: 'Composable',
    desc: 'Budget contexts, namespaces, and per-operation breakdowns.',
  },
];

const navCards = [
  {
    title: 'Get started',
    desc: 'Install whest and create your first FLOP budget.',
    link: '/docs/getting-started/installation',
  },
  {
    title: 'Migrate from NumPy',
    desc: 'Convert existing code with FLOP annotations.',
    link: '/docs/guides/migrate-from-numpy',
  },
  {
    title: 'API reference',
    desc: 'Search all 508 operations with interactive filters.',
    link: '/docs/api',
  },
  {
    title: 'FLOP counting model',
    desc: 'How costs are computed analytically.',
    link: '/docs/understanding/flop-counting-model',
  },
  {
    title: 'Symmetry explorer',
    desc: 'Interactive einsum symmetry visualization.',
    link: '/symmetry-aware-einsum-contractions',
  },
  {
    title: 'For AI agents',
    desc: 'Machine-readable resources and rules.',
    link: '/docs/api/for-agents',
  },
];

const installCode = 'uv add git+https://github.com/AIcrowd/whest.git';

const numpyCode = `import numpy as np

depth, width = 5, 256

# Weight init
scale = np.sqrt(2 / width)
weights = [
    np.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = np.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = np.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = np.maximum(h, 0)
# Total FLOPs? No idea.`;

const whestCode = `import whest as we

depth, width = 5, 256

# Weight init
scale = we.sqrt(2 / width)
weights = [
    we.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = we.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = we.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = we.maximum(h, 0)
we.budget_summary()  # 984,321 FLOPs`;

// Whest-paper-tuned syntax palette. These are the same hex values the
// Shiki `whestLight` theme emits (see lib/shiki-themes.ts) — using them
// directly here lets us keep the per-line typing animation while still
// rendering with the design system's code colors.
const TOKENS = {
  base: 'text-[#292C2D] dark:text-[#E8EAEB]',                 // ink
  muted: 'text-[#AAACAD] italic dark:text-[#6B7072]',          // comment
  keyword: 'text-[#5E36C4] dark:text-[#B99BFF]',               // violet
  number: 'text-[#0B6D7A] dark:text-[#6BD4CE]',                // teal
  string: 'text-[#D23934] dark:text-[#FF8A7A]',                // coral-hover
  op: 'text-[#F0524D]',                                        // brand coral
  func: 'text-[#2959C4] dark:text-[#8FB4FF]',                  // V-blue
} as const;

type Token = {
  text: string;
  className?: string;
};

const installLine: Token[] = [
  { text: 'uv', className: TOKENS.func },
  { text: ' add git+https://github.com/AIcrowd/whest.git', className: TOKENS.string },
];

const numpyLines: Token[][] = [
  [
    { text: 'import', className: TOKENS.keyword },
    { text: ' numpy ', className: TOKENS.base },
    { text: 'as', className: TOKENS.keyword },
    { text: ' np', className: TOKENS.base },
  ],
  [],
  [
    { text: 'depth, width ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' 5, 256', className: TOKENS.number },
  ],
  [],
  [{ text: '# Weight init', className: TOKENS.muted }],
  [
    { text: 'scale ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' np.sqrt', className: TOKENS.func },
    { text: '(', className: TOKENS.base },
    { text: '2', className: TOKENS.number },
    { text: ' / ', className: TOKENS.op },
    { text: 'width', className: TOKENS.base },
    { text: ')', className: TOKENS.base },
  ],
  [
    { text: 'weights ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' [', className: TOKENS.base },
  ],
  [
    { text: '    np.random.randn', className: TOKENS.func },
    { text: '(width, width)', className: TOKENS.base },
    { text: ' * ', className: TOKENS.op },
    { text: 'scale', className: TOKENS.base },
  ],
  [
    { text: '    for', className: TOKENS.keyword },
    { text: ' _ ', className: TOKENS.base },
    { text: 'in', className: TOKENS.keyword },
    { text: ' range', className: TOKENS.func },
    { text: '(depth)', className: TOKENS.base },
  ],
  [{ text: ']', className: TOKENS.base }],
  [],
  [{ text: '# Forward pass', className: TOKENS.muted }],
  [
    { text: 'x ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' np.random.randn', className: TOKENS.func },
    { text: '(width)', className: TOKENS.base },
  ],
  [
    { text: 'h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' x', className: TOKENS.base },
  ],
  [
    { text: 'for', className: TOKENS.keyword },
    { text: ' i, W ', className: TOKENS.base },
    { text: 'in', className: TOKENS.keyword },
    { text: ' enumerate', className: TOKENS.func },
    { text: '(weights):', className: TOKENS.base },
  ],
  [
    { text: '    h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' np.einsum', className: TOKENS.func },
    { text: "('ij,j->i', W, h)", className: TOKENS.string },
  ],
  [
    { text: '    if', className: TOKENS.keyword },
    { text: ' i ', className: TOKENS.base },
    { text: '<', className: TOKENS.op },
    { text: ' depth ', className: TOKENS.base },
    { text: '-', className: TOKENS.op },
    { text: ' 1:', className: TOKENS.number },
  ],
  [
    { text: '        h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' np.maximum', className: TOKENS.func },
    { text: '(h, ', className: TOKENS.base },
    { text: '0', className: TOKENS.number },
    { text: ')', className: TOKENS.base },
  ],
  [{ text: '# Total FLOPs? No idea.', className: TOKENS.muted }],
];

const whestLines: Token[][] = [
  [
    { text: 'import', className: TOKENS.keyword },
    { text: ' whest ', className: TOKENS.base },
    { text: 'as', className: TOKENS.keyword },
    { text: ' we', className: TOKENS.base },
  ],
  [],
  [
    { text: 'depth, width ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' 5, 256', className: TOKENS.number },
  ],
  [],
  [{ text: '# Weight init', className: TOKENS.muted }],
  [
    { text: 'scale ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' we.sqrt', className: TOKENS.func },
    { text: '(', className: TOKENS.base },
    { text: '2', className: TOKENS.number },
    { text: ' / ', className: TOKENS.op },
    { text: 'width', className: TOKENS.base },
    { text: ')', className: TOKENS.base },
  ],
  [
    { text: 'weights ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' [', className: TOKENS.base },
  ],
  [
    { text: '    we.random.randn', className: TOKENS.func },
    { text: '(width, width)', className: TOKENS.base },
    { text: ' * ', className: TOKENS.op },
    { text: 'scale', className: TOKENS.base },
  ],
  [
    { text: '    for', className: TOKENS.keyword },
    { text: ' _ ', className: TOKENS.base },
    { text: 'in', className: TOKENS.keyword },
    { text: ' range', className: TOKENS.func },
    { text: '(depth)', className: TOKENS.base },
  ],
  [{ text: ']', className: TOKENS.base }],
  [],
  [{ text: '# Forward pass', className: TOKENS.muted }],
  [
    { text: 'x ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' we.random.randn', className: TOKENS.func },
    { text: '(width)', className: TOKENS.base },
  ],
  [
    { text: 'h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' x', className: TOKENS.base },
  ],
  [
    { text: 'for', className: TOKENS.keyword },
    { text: ' i, W ', className: TOKENS.base },
    { text: 'in', className: TOKENS.keyword },
    { text: ' enumerate', className: TOKENS.func },
    { text: '(weights):', className: TOKENS.base },
  ],
  [
    { text: '    h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' we.einsum', className: TOKENS.func },
    { text: "('ij,j->i', W, h)", className: TOKENS.string },
  ],
  [
    { text: '    if', className: TOKENS.keyword },
    { text: ' i ', className: TOKENS.base },
    { text: '<', className: TOKENS.op },
    { text: ' depth ', className: TOKENS.base },
    { text: '-', className: TOKENS.op },
    { text: ' 1:', className: TOKENS.number },
  ],
  [
    { text: '        h ', className: TOKENS.base },
    { text: '=', className: TOKENS.op },
    { text: ' we.maximum', className: TOKENS.func },
    { text: '(h, ', className: TOKENS.base },
    { text: '0', className: TOKENS.number },
    { text: ')', className: TOKENS.base },
  ],
  [
    { text: 'we.budget_summary', className: TOKENS.func },
    { text: '()', className: TOKENS.base },
    { text: '  # 984,321 FLOPs', className: TOKENS.muted },
  ],
];

function TerminalLine({
  tokens,
  animated = false,
  noWrap = false,
  delayMs = 0,
}: {
  tokens: Token[];
  animated?: boolean;
  noWrap?: boolean;
  delayMs?: number;
}) {
  const content =
    tokens.length === 0 ? (
      <span className={TOKENS.base}>&nbsp;</span>
    ) : (
      tokens.map((token, index) => (
        <span key={`${token.text}-${index}`} className={token.className ?? TOKENS.base}>
          {token.text}
        </span>
      ))
    );

  const className = [
    'leading-[1.55]',
    noWrap ? 'whitespace-nowrap' : 'whitespace-pre',
  ]
    .filter(Boolean)
    .join(' ');

  if (!animated) {
    return <div className={className}>{content}</div>;
  }

  return (
    <AnimatedSpan className="leading-[1.55]" delay={delayMs} startOnView={false}>
      <span className={className}>{content}</span>
    </AnimatedSpan>
  );
}

// Base unstyled wordmark used in the big hero. Colors + fonts are supplied
// by .whest-wordmark / .whest-wordmark__dot utilities in global.css.
function HeroHeadline({ children, dot }: { children: string; dot?: string }) {
  return (
    <h1
      className="m-0 font-semibold"
      style={{
        fontFamily: 'var(--font-display-serif), Georgia, serif',
        fontVariationSettings: "'opsz' 72",
        fontSize: 'clamp(44px, 7vw, 64px)',
        letterSpacing: '-0.02em',
        lineHeight: 1.02,
        color: 'var(--gray-900, #292C2D)',
      }}
    >
      {children}
      <span style={{ color: '#F0524D' }}>{dot ?? '.'}</span>
    </h1>
  );
}

export default function HomePage() {
  return (
    <main className="bg-white dark:bg-[#0E0F10]">
      {/* 1. Masthead — left-aligned editorial hero. Text block hugs the left,
          capped at --prose-max (720px); brush mark hugs the right edge of the
          content-max container. At wide viewports this reads as a classic
          editorial pairing rather than a floating illustration in a fat column. */}
      <section className="mx-auto w-full max-w-[var(--content-max)] px-6 pt-16 pb-14 md:px-8 md:pt-24 md:pb-20">
        <div className="grid grid-cols-1 items-center gap-10 md:grid-cols-[minmax(0,var(--prose-max))_minmax(0,1fr)] md:gap-0">
          <div className="max-w-[var(--prose-max)]">
          <div
            className="mb-6 font-sans text-[10px] font-semibold uppercase text-gray-400 dark:text-gray-500"
            style={{ letterSpacing: '0.2em' }}
          >
            <span aria-hidden className="mr-2 inline-block h-px w-8 align-middle bg-gray-300 dark:bg-gray-600" />
            WHEST · A FLOP-COUNTING NUMPY
          </div>

          <HeroHeadline>Count every FLOP</HeroHeadline>

          <p
            className="mt-6 mb-9 max-w-[600px] text-[17px] italic text-gray-600 dark:text-gray-300"
            style={{
              fontFamily: 'var(--font-paper-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 18",
              lineHeight: 1.6,
            }}
          >
            A NumPy-compatible math library that counts every FLOP
            analytically, so compute budgets stop being guesswork.
          </p>

          <div className="mb-10 flex flex-wrap items-center gap-3">
            <Link
              href="/docs/getting-started/installation"
              className="inline-flex items-center rounded-lg bg-[#F0524D] px-5 py-2.5 text-sm font-medium text-white no-underline transition-colors hover:bg-[#D23934] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#F0524D]/20"
            >
              Install
            </Link>
            <Link
              href="/docs/understanding/how-whest-works"
              className="inline-flex items-center rounded-lg border border-gray-200 px-5 py-2.5 text-sm font-medium text-gray-900 no-underline transition-colors hover:border-[#F0524D] hover:text-[#F0524D] dark:border-gray-700 dark:text-gray-100"
            >
              How it works &rarr;
            </Link>
          </div>

          <HomeCodeTerminal copyText={installCode} className="max-w-[600px]">
            <TerminalLine tokens={installLine} noWrap />
          </HomeCodeTerminal>
          </div>

          {/* Brush mark — horizontally centered in its grid cell (both
              mobile and desktop). The left column is capped at --prose-max
              so the right cell is tight enough that centering reads as
              balanced rather than floating. */}
          <div className="row-start-1 flex items-center justify-center self-center md:row-start-auto">
            <Image
              src={withBasePath('/logo.png')}
              alt="whest"
              width={280}
              height={150}
              priority
              className="h-auto w-[180px] select-none md:w-[220px] lg:w-[260px]"
            />
          </div>
        </div>
      </section>

      {/* 2. Metric row — 508 · 0 · ∞ trio (breadth · methodology · composition) */}
      <section className="mx-auto w-full max-w-[var(--content-max)] grid grid-cols-1 gap-10 px-6 py-16 text-center sm:grid-cols-3 sm:gap-8 md:gap-12 md:py-20">
        {features.map((feature) => (
          <div key={feature.label} className="flex min-h-[140px] flex-col items-center justify-center">
            <span
              className="mb-3 block text-[#F0524D]"
              style={{
                fontFamily: 'var(--font-display-serif), Georgia, serif',
                fontVariationSettings: "'opsz' 72",
                fontWeight: 700,
                fontSize: '48px',
                lineHeight: 1,
                letterSpacing: '-0.015em',
              }}
            >
              {feature.number}
            </span>
            <span
              className="mb-3 block font-sans text-[10px] font-semibold uppercase text-gray-400 dark:text-gray-500"
              style={{ letterSpacing: '0.2em' }}
            >
              {feature.label}
            </span>
            <p
              className="m-0 max-w-[280px] text-[14px] text-gray-600 dark:text-gray-400"
              style={{
                fontFamily: 'var(--font-paper-serif), Georgia, serif',
                fontVariationSettings: "'opsz' 14",
                lineHeight: 1.6,
              }}
            >
              {feature.desc}
            </p>
          </div>
        ))}
      </section>

      {/* 3. Diptych — NumPy vs whest */}
      <section className="mx-auto w-full max-w-[var(--content-max)] px-6 pb-20 md:px-8">
        <div className="max-w-[var(--prose-max)]">
          <div
            className="mb-3 font-sans text-[10px] font-semibold uppercase text-gray-400 dark:text-gray-500"
            style={{ letterSpacing: '0.2em' }}
          >
            <span aria-hidden className="mr-2 inline-block h-px w-8 align-middle bg-gray-300 dark:bg-gray-600" />
            IN ACTION
          </div>
          <h2
            className="mb-3 m-0 text-gray-900 dark:text-gray-100"
            style={{
              fontFamily: 'var(--font-display-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 32",
              fontWeight: 600,
              fontSize: '28px',
              letterSpacing: '-0.015em',
              lineHeight: 1.2,
            }}
          >
            What does this code cost<span style={{ color: '#F0524D' }}>?</span>
          </h2>
          <p
            className="mb-10 max-w-[600px] text-[15px] italic text-gray-600 dark:text-gray-400"
            style={{
              fontFamily: 'var(--font-paper-serif), Georgia, serif',
              fontVariationSettings: "'opsz' 15",
              lineHeight: 1.6,
            }}
          >
            Same five-layer MLP, written twice. The one on the right counts
            every FLOP analytically as it runs.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-6">
          <div className="text-left">
            <HomeCodeTerminal copyText={numpyCode} label="NumPy">
              {numpyLines.map((tokens, index) => (
                <TerminalLine key={index} tokens={tokens} />
              ))}
            </HomeCodeTerminal>
          </div>
          <div className="text-left">
            <HomeCodeTerminal copyText={whestCode} label="Whest" labelAccent>
              {whestLines.map((tokens, index) => (
                <TerminalLine key={index} tokens={tokens} animated delayMs={index * 140} />
              ))}
            </HomeCodeTerminal>
          </div>
        </div>

        <p
          className="mt-6 max-w-[var(--prose-max)] text-[14px] italic text-gray-500 dark:text-gray-400"
          style={{
            fontFamily: 'var(--font-paper-serif), Georgia, serif',
            fontVariationSettings: "'opsz' 14",
            lineHeight: 1.6,
          }}
        >
          The answer, on the right:{' '}
          <span className="not-italic font-medium text-[#F0524D]">984,321 FLOPs</span>
          {' '}— counted analytically as the NumPy call runs.
        </p>
      </section>

      {/* 4. Explore the docs — editorial index-card grid */}
      <section className="mx-auto w-full max-w-[var(--content-max)] px-6 pb-24 md:px-8">
        <div
          className="mb-3 font-sans text-[10px] font-semibold uppercase text-gray-400 dark:text-gray-500"
          style={{ letterSpacing: '0.2em' }}
        >
          <span aria-hidden className="mr-2 inline-block h-px w-8 align-middle bg-gray-300 dark:bg-gray-600" />
          READ ON
        </div>
        <h2
          className="mb-10 m-0 text-gray-900 dark:text-gray-100"
          style={{
            fontFamily: 'var(--font-display-serif), Georgia, serif',
            fontVariationSettings: "'opsz' 32",
            fontWeight: 600,
            fontSize: '28px',
            letterSpacing: '-0.015em',
            lineHeight: 1.2,
          }}
        >
          Explore the docs<span style={{ color: '#F0524D' }}>.</span>
        </h2>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {navCards.map((card) => (
            <Link
              key={card.title}
              href={card.link}
              className="group relative block rounded-lg border border-gray-200 p-6 text-inherit no-underline transition-colors hover:border-[#F0524D] dark:border-gray-700"
            >
              <span
                className="absolute end-5 top-5 text-lg text-gray-400 transition-colors group-hover:text-[#F0524D]"
                aria-hidden="true"
              >
                &rarr;
              </span>
              <div
                className="mb-2 text-[19px] italic text-gray-900 underline-offset-4 group-hover:underline dark:text-gray-100"
                style={{
                  fontFamily: 'var(--font-display-serif), Georgia, serif',
                  fontVariationSettings: "'opsz' 24",
                  fontWeight: 600,
                  letterSpacing: '-0.005em',
                  lineHeight: 1.25,
                }}
              >
                {card.title}
              </div>
              <p
                className="m-0 max-w-[22rem] text-[14px] text-gray-600 dark:text-gray-400"
                style={{
                  fontFamily: 'var(--font-paper-serif), Georgia, serif',
                  fontVariationSettings: "'opsz' 14",
                  lineHeight: 1.6,
                }}
              >
                {card.desc}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* 5. Colophon */}
      <footer
        className="mx-auto w-full max-w-[var(--prose-max)] px-6 pb-16 text-center text-[13px] italic text-gray-400 dark:text-gray-500"
        style={{
          fontFamily: 'var(--font-paper-serif), Georgia, serif',
          fontVariationSettings: "'opsz' 13",
          lineHeight: 1.6,
        }}
      >
        whest is maintained by AIcrowd. The design system extends the{' '}
        <Link
          href="/symmetry-aware-einsum-contractions"
          className="italic text-gray-500 underline-offset-4 hover:text-[#F0524D] hover:underline dark:text-gray-400"
        >
          Symmetry-Aware Einsum Contractions
        </Link>{' '}
        explorer.
      </footer>
    </main>
  );
}
