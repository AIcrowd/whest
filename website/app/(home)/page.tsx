import HomeCodeTerminal from '@/components/home-code-terminal';
import { AnimatedSpan } from '@/components/ui/terminal';
import Image from 'next/image';
import Link from 'next/link';

const features = [
  {
    kind: 'metric' as const,
    number: '508',
    label: 'Operations',
    desc: 'NumPy-compatible operations with analytical FLOP costs',
  },
  {
    kind: 'logo' as const,
  },
  {
    kind: 'metric' as const,
    number: '\u221E',
    label: 'Composable',
    desc: 'Budget contexts, namespaces, and per-operation breakdowns',
  },
];

const navCards = [
  {
    title: 'Get Started',
    desc: 'Install whest and create your first FLOP budget',
    link: '/docs/getting-started/installation',
  },
  {
    title: 'Migrate from NumPy',
    desc: 'Convert existing code with FLOP annotations',
    link: '/docs/how-to/migrate-from-numpy',
  },
  {
    title: 'API Reference',
    desc: 'Search all 508 operations with interactive filters',
    link: '/docs/api',
  },
  {
    title: 'FLOP Counting Model',
    desc: 'How costs are computed analytically',
    link: '/docs/concepts/flop-counting-model',
  },
  {
    title: 'Symmetry Explorer',
    desc: 'Interactive einsum symmetry visualization',
    link: '/docs/explanation/symmetry-explorer',
  },
  {
    title: 'For AI Agents',
    desc: 'Machine-readable resources and rules',
    link: '/docs/reference/for-agents',
  },
];

const installCode = `uv add git+https://github.com/AIcrowd/whest.git`;

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

const TOKENS = {
  base: 'text-[#c9d1d9]',
  muted: 'text-[#8b949e]',
  keyword: 'text-[#ff7b72]',
  number: 'text-[#79c0ff]',
  string: 'text-[#a5d6ff]',
  op: 'text-[#ff7b72]',
  func: 'text-[#d2a8ff]',
} as const;

type Token = {
  text: string;
  className?: string;
};

const installLine: Token[] = [
  {text: 'uv', className: TOKENS.func},
  {text: ' add git+https://github.com/AIcrowd/whest.git', className: TOKENS.string},
];

const numpyLines: Token[][] = [
  [
    {text: 'import', className: TOKENS.keyword},
    {text: ' numpy ', className: TOKENS.base},
    {text: 'as', className: TOKENS.keyword},
    {text: ' np', className: TOKENS.base},
  ],
  [],
  [
    {text: 'depth, width ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' 5, 256', className: TOKENS.number},
  ],
  [],
  [{text: '# Weight init', className: TOKENS.muted}],
  [
    {text: 'scale ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' np.sqrt', className: TOKENS.func},
    {text: '(', className: TOKENS.base},
    {text: '2', className: TOKENS.number},
    {text: ' / ', className: TOKENS.op},
    {text: 'width', className: TOKENS.base},
    {text: ')', className: TOKENS.base},
  ],
  [
    {text: 'weights ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' [', className: TOKENS.base},
  ],
  [
    {text: '    np.random.randn', className: TOKENS.func},
    {text: '(width, width)', className: TOKENS.base},
    {text: ' * ', className: TOKENS.op},
    {text: 'scale', className: TOKENS.base},
  ],
  [
    {text: '    for', className: TOKENS.keyword},
    {text: ' _ ', className: TOKENS.base},
    {text: 'in', className: TOKENS.keyword},
    {text: ' range', className: TOKENS.func},
    {text: '(depth)', className: TOKENS.base},
  ],
  [{text: ']', className: TOKENS.base}],
  [],
  [{text: '# Forward pass', className: TOKENS.muted}],
  [
    {text: 'x ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' np.random.randn', className: TOKENS.func},
    {text: '(width)', className: TOKENS.base},
  ],
  [
    {text: 'h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' x', className: TOKENS.base},
  ],
  [
    {text: 'for', className: TOKENS.keyword},
    {text: ' i, W ', className: TOKENS.base},
    {text: 'in', className: TOKENS.keyword},
    {text: ' enumerate', className: TOKENS.func},
    {text: '(weights):', className: TOKENS.base},
  ],
  [
    {text: '    h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' np.einsum', className: TOKENS.func},
    {text: "('ij,j->i', W, h)", className: TOKENS.string},
  ],
  [
    {text: '    if', className: TOKENS.keyword},
    {text: ' i ', className: TOKENS.base},
    {text: '<', className: TOKENS.op},
    {text: ' depth ', className: TOKENS.base},
    {text: '-', className: TOKENS.op},
    {text: ' 1:', className: TOKENS.number},
  ],
  [
    {text: '        h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' np.maximum', className: TOKENS.func},
    {text: '(h, ', className: TOKENS.base},
    {text: '0', className: TOKENS.number},
    {text: ')', className: TOKENS.base},
  ],
  [{text: '# Total FLOPs? No idea.', className: TOKENS.muted}],
];

const whestLines: Token[][] = [
  [
    {text: 'import', className: TOKENS.keyword},
    {text: ' whest ', className: TOKENS.base},
    {text: 'as', className: TOKENS.keyword},
    {text: ' we', className: TOKENS.base},
  ],
  [],
  [
    {text: 'depth, width ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' 5, 256', className: TOKENS.number},
  ],
  [],
  [{text: '# Weight init', className: TOKENS.muted}],
  [
    {text: 'scale ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' we.sqrt', className: TOKENS.func},
    {text: '(', className: TOKENS.base},
    {text: '2', className: TOKENS.number},
    {text: ' / ', className: TOKENS.op},
    {text: 'width', className: TOKENS.base},
    {text: ')', className: TOKENS.base},
  ],
  [
    {text: 'weights ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' [', className: TOKENS.base},
  ],
  [
    {text: '    we.random.randn', className: TOKENS.func},
    {text: '(width, width)', className: TOKENS.base},
    {text: ' * ', className: TOKENS.op},
    {text: 'scale', className: TOKENS.base},
  ],
  [
    {text: '    for', className: TOKENS.keyword},
    {text: ' _ ', className: TOKENS.base},
    {text: 'in', className: TOKENS.keyword},
    {text: ' range', className: TOKENS.func},
    {text: '(depth)', className: TOKENS.base},
  ],
  [{text: ']', className: TOKENS.base}],
  [],
  [{text: '# Forward pass', className: TOKENS.muted}],
  [
    {text: 'x ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' we.random.randn', className: TOKENS.func},
    {text: '(width)', className: TOKENS.base},
  ],
  [
    {text: 'h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' x', className: TOKENS.base},
  ],
  [
    {text: 'for', className: TOKENS.keyword},
    {text: ' i, W ', className: TOKENS.base},
    {text: 'in', className: TOKENS.keyword},
    {text: ' enumerate', className: TOKENS.func},
    {text: '(weights):', className: TOKENS.base},
  ],
  [
    {text: '    h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' we.einsum', className: TOKENS.func},
    {text: "('ij,j->i', W, h)", className: TOKENS.string},
  ],
  [
    {text: '    if', className: TOKENS.keyword},
    {text: ' i ', className: TOKENS.base},
    {text: '<', className: TOKENS.op},
    {text: ' depth ', className: TOKENS.base},
    {text: '-', className: TOKENS.op},
    {text: ' 1:', className: TOKENS.number},
  ],
  [
    {text: '        h ', className: TOKENS.base},
    {text: '=', className: TOKENS.op},
    {text: ' we.maximum', className: TOKENS.func},
    {text: '(h, ', className: TOKENS.base},
    {text: '0', className: TOKENS.number},
    {text: ')', className: TOKENS.base},
  ],
  [
    {text: 'we.budget_summary', className: TOKENS.func},
    {text: '()', className: TOKENS.base},
    {text: '  # 984,321 FLOPs', className: TOKENS.muted},
  ],
];

function TerminalLine({
  tokens,
  center = false,
  animated = false,
  noWrap = false,
  delayMs = 0,
}: {
  tokens: Token[];
  center?: boolean;
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
    'font-mono text-sm leading-5',
    noWrap ? 'whitespace-nowrap' : 'whitespace-pre',
    center ? 'mx-auto w-fit text-center' : '',
  ]
    .filter(Boolean)
    .join(' ');

  if (!animated) {
    return <div className={className}>{content}</div>;
  }

  return (
    <AnimatedSpan className="font-mono text-sm leading-5" delay={delayMs} startOnView={false}>
      <span className={className}>{content}</span>
    </AnimatedSpan>
  );
}

export default function HomePage() {
  return (
    <main>
      <section className="-mt-px bg-gradient-to-b from-red-50 to-white dark:from-[#1a1020] dark:to-transparent text-center pt-16 pb-20 px-6">
        <h1 className="text-5xl font-bold tracking-tight text-gray-900 dark:text-gray-100 mb-4 leading-tight">
          Count every FLOP.
        </h1>
        <p className="text-lg text-gray-500 dark:text-gray-400 max-w-xl mx-auto mb-8 leading-relaxed">
          NumPy-compatible math primitives with analytical FLOP counting. Set
          budgets, track costs, and understand your compute.
        </p>
        <div className="flex gap-3 justify-center flex-wrap mb-8">
          <Link
            href="/docs/getting-started/installation"
            className="bg-[#F0524D] text-white px-6 py-2.5 rounded-lg font-semibold hover:bg-[#D23934] transition-all hover:-translate-y-0.5 no-underline"
          >
            Get Started
          </Link>
          <Link
            href="/docs/api"
            className="border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 px-6 py-2.5 rounded-lg font-semibold hover:border-[#F0524D] hover:text-[#F0524D] transition-all no-underline"
          >
            API Reference
          </Link>
        </div>
        <div className="mx-auto max-w-xl">
          <HomeCodeTerminal copyText={installCode} center>
            <TerminalLine tokens={installLine} center noWrap />
          </HomeCodeTerminal>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto py-16 px-6 text-center">
        {features.map((f) => (
          <div
            key={f.kind === 'logo' ? 'logo' : f.label}
            className="flex min-h-[112px] flex-col items-center justify-center"
          >
            {f.kind === 'logo' ? (
              <Image
                src="/logo.png"
                alt="whest"
                width={180}
                height={96}
                className="h-auto w-36 md:w-[180px]"
              />
            ) : (
              <>
                <span className="text-4xl font-bold text-[#F0524D] leading-none mb-2">
                  {f.number}
                </span>
                <span className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-3">
                  {f.label}
                </span>
                <p className="text-sm text-gray-500 dark:text-gray-400 max-w-[280px] m-0 leading-relaxed">
                  {f.desc}
                </p>
              </>
            )}
          </div>
        ))}
      </section>

      <section className="max-w-3xl mx-auto px-6 pb-16">
        <h2 className="text-center text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100 mb-6">
          See it in action
        </h2>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-5">
          <div className="mx-auto w-full max-w-[520px] text-left">
            <div className="mb-3 text-center text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 dark:text-gray-500">
              NumPy
            </div>
            <HomeCodeTerminal copyText={numpyCode} className="w-full">
              {numpyLines.map((tokens, index) => (
                <TerminalLine key={index} tokens={tokens} />
              ))}
            </HomeCodeTerminal>
          </div>
          <div className="mx-auto w-full max-w-[520px] text-left">
            <div className="mb-3 text-center text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 dark:text-gray-500">
              whest
            </div>
            <HomeCodeTerminal copyText={whestCode} className="w-full">
              {whestLines.map((tokens, index) => (
                <TerminalLine
                  key={index}
                  tokens={tokens}
                  animated
                  delayMs={index * 140}
                />
              ))}
            </HomeCodeTerminal>
          </div>
        </div>
      </section>

      <section className="max-w-4xl mx-auto px-6 pb-20">
        <h2 className="text-center text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100 mb-8">
          Explore the docs
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {navCards.map((card) => (
            <Link
              key={card.title}
              href={card.link}
              className="group relative border border-gray-200 dark:border-gray-700 rounded-xl p-6 no-underline text-inherit hover:border-[#F0524D] hover:-translate-y-0.5 transition-all hover:shadow-md"
            >
              <span
                className="absolute top-5 right-5 text-lg text-gray-400 transition-all group-hover:translate-x-1 group-hover:text-[#F0524D]"
                aria-hidden="true"
              >
                &rarr;
              </span>
              <div className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-1.5">
                {card.title}
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 m-0 leading-relaxed">
                {card.desc}
              </p>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
