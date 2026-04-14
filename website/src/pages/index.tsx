import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import styles from './index.module.css';

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
    link: '/getting-started/installation',
  },
  {
    title: 'Migrate from NumPy',
    desc: 'Convert existing code with FLOP annotations',
    link: '/how-to/migrate-from-numpy',
  },
  {
    title: 'API Reference',
    desc: 'Search all 508 operations with interactive filters',
    link: '/api',
  },
  {
    title: 'FLOP Counting Model',
    desc: 'How costs are computed analytically',
    link: '/concepts/flop-counting-model',
  },
  {
    title: 'Symmetry Explorer',
    desc: 'Interactive einsum symmetry visualization',
    link: '/explanation/symmetry-explorer',
  },
  {
    title: 'For AI Agents',
    desc: 'Machine-readable resources and rules',
    link: '/reference/for-agents',
  },
];

const exampleCode = `import whest as we

with we.BudgetContext(flop_budget=10**8, namespace="demo") as budget:
    scale = we.sqrt(we.array(2 / 256))
    W = we.multiply(we.random.randn(256, 256), scale)
    x = we.einsum('ij,j->i', W, we.random.randn(256))

we.budget_summary()`;

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Home"
      description="NumPy-compatible math primitives with analytical FLOP counting"
    >
      {/* Hero */}
      <section className={styles.hero}>
        <h1 className={styles.heroTitle}>Count every FLOP.</h1>
        <p className={styles.heroSubtitle}>
          NumPy-compatible math primitives with analytical FLOP counting.
          Set budgets, track costs, and understand your compute.
        </p>
        <div className={styles.heroButtons}>
          <Link className={styles.primaryButton} to="/getting-started/installation">
            Get Started
          </Link>
          <Link className={styles.secondaryButton} to="/api">
            API Reference
          </Link>
        </div>
        <code className={styles.heroInstall}>
          uv add git+https://github.com/AIcrowd/whest.git
        </code>
      </section>

      {/* Feature numbers */}
      <section className={styles.features}>
        {features.map((f) => (
          <div key={f.label} className={styles.featureItem}>
            <span className={styles.featureNumber}>{f.number}</span>
            <span className={styles.featureLabel}>{f.label}</span>
            <p className={styles.featureDesc}>{f.desc}</p>
          </div>
        ))}
      </section>

      {/* Code example */}
      <section className={styles.codeSection}>
        <h2 className={styles.codeSectionTitle}>See it in action</h2>
        <CodeBlock language="python">{exampleCode}</CodeBlock>
      </section>

      {/* Navigation cards */}
      <section className={styles.navSection}>
        <h2 className={styles.navSectionTitle}>Explore the docs</h2>
        <div className={styles.navGrid}>
          {navCards.map((card) => (
            <Link key={card.title} className={styles.navCard} to={card.link}>
              <span className={styles.navCardArrow} aria-hidden="true">&rarr;</span>
              <div className={styles.navCardTitle}>{card.title}</div>
              <p className={styles.navCardDesc}>{card.desc}</p>
            </Link>
          ))}
        </div>
      </section>
    </Layout>
  );
}
