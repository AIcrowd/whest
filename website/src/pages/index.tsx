import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import styles from './index.module.css';

const cards = [
  {
    emoji: '\u{1F680}',
    title: 'Get Started',
    description: 'Install whest and create your first FLOP budget',
    link: '/getting-started/installation',
  },
  {
    emoji: '\u{1F6E0}',
    title: 'Troubleshoot',
    description: 'Common errors and fixes',
    link: '/troubleshooting/common-errors',
  },
  {
    emoji: '\u{1F4C8}',
    title: 'Write Efficient Code',
    description: 'Migrate from NumPy, use einsum, exploit symmetry',
    link: '/how-to/migrate-from-numpy',
  },
  {
    emoji: '\u{1F9E0}',
    title: 'Understand the Model',
    description: 'How FLOP costs are computed analytically',
    link: '/concepts/flop-counting-model',
  },
  {
    emoji: '\u{1F3D7}',
    title: 'Architecture',
    description: 'Client-server model and Docker setup',
    link: '/architecture/client-server',
  },
  {
    emoji: '\u{1F4D6}',
    title: 'API Reference',
    description: 'All 482 operations with interactive search',
    link: '/api',
  },
];

const exampleCode = `import whest.np as wnp
from whest import BudgetContext

with BudgetContext(max_flops=1e9) as ctx:
    a = wnp.random.randn(512, 512)
    b = wnp.random.randn(512, 512)
    c = wnp.matmul(a, b)

print(f"FLOPs used: {ctx.flops_used:,.0f}")`;

export default function Home(): React.JSX.Element {
  return (
    <Layout title="Home" description="NumPy-compatible math primitives with analytical FLOP counting">
      {/* Hero */}
      <div className={styles.hero}>
        <img src="/whest/img/logo.png" alt="whest logo" className={styles.logo} />
        <h1 className={styles.title}>whest</h1>
        <p className={styles.tagline}>
          NumPy-compatible math primitives with analytical FLOP counting
        </p>
        <div className={styles.buttons}>
          <Link className={styles.primaryButton} to="/getting-started/installation">
            Get Started
          </Link>
          <Link className={styles.secondaryButton} to="/api">
            API Reference
          </Link>
        </div>
      </div>

      {/* Navigation Cards */}
      <div className={styles.cardsSection}>
        <div className={styles.cardsGrid}>
          {cards.map((card) => (
            <Link key={card.title} className={styles.card} to={card.link}>
              <div className={styles.cardTitle}>
                {card.emoji} {card.title}
              </div>
              <p className={styles.cardDescription}>{card.description}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* Quick Code Example */}
      <div className={styles.codeSection}>
        <h2 className={styles.codeSectionTitle}>Quick Example</h2>
        <CodeBlock language="python">{exampleCode}</CodeBlock>
      </div>
    </Layout>
  );
}
