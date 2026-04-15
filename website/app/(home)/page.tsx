import Link from 'next/link';

const features = [
  {
    number: '508',
    label: 'Operations',
    desc: 'NumPy-compatible operations with analytical FLOP costs',
  },
  {
    number: '0',
    label: 'Runtime overhead',
    desc: 'Costs computed from tensor shapes, not measured at runtime',
  },
  {
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

const exampleCode = `import whest as we

with we.BudgetContext(flop_budget=10**8, namespace="demo") as budget:
    scale = we.sqrt(we.array(2 / 256))
    W = we.multiply(we.random.randn(256, 256), scale)
    x = we.einsum('ij,j->i', W, we.random.randn(256))

we.budget_summary()`;

export default function HomePage() {
  return (
    <main>
      {/* Hero */}
      <section className="bg-gradient-to-b from-red-50 to-white dark:from-[#1a1020] dark:to-transparent text-center py-20 px-6">
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
        <code className="inline-block bg-[#0F1B2D] text-[#E8F0FE] px-5 py-2.5 rounded-lg font-mono text-sm max-w-full overflow-x-auto">
          uv add git+https://github.com/AIcrowd/whest.git
        </code>
      </section>

      {/* Feature numbers */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto py-16 px-6 text-center">
        {features.map((f) => (
          <div key={f.label} className="flex flex-col items-center">
            <span className="text-4xl font-bold text-[#F0524D] leading-none mb-2">
              {f.number}
            </span>
            <span className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-3">
              {f.label}
            </span>
            <p className="text-sm text-gray-500 dark:text-gray-400 max-w-[280px] m-0 leading-relaxed">
              {f.desc}
            </p>
          </div>
        ))}
      </section>

      {/* Code example */}
      <section className="max-w-2xl mx-auto px-6 pb-16">
        <h2 className="text-center text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100 mb-6">
          See it in action
        </h2>
        <pre className="bg-[#0F1B2D] text-[#E8F0FE] rounded-xl p-5 overflow-x-auto text-sm leading-relaxed">
          <code>{exampleCode}</code>
        </pre>
      </section>

      {/* Navigation cards */}
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
