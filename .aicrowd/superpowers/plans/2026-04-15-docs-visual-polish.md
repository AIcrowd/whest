# Docs Visual Polish — "Apple Marketing meets Stripe" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the whest Docusaurus docs from a generic template into a polished, distinctive site that fuses Apple Marketing's bold typography and dramatic whitespace with Stripe Docs' crisp code presentation and refined sidebar.

**Architecture:** Pure CSS + layout changes. No new React components. Complete rewrite of `custom.css`, landing page (`index.tsx` + `index.module.css`), and Docusaurus config theme adjustments. The coral accent from whestbench-explorer is replaced with a richer palette inspired by Stripe's navy + accent system, adapted for whest's identity.

**Tech Stack:** CSS custom properties, Docusaurus theme swizzling (CSS only), responsive breakpoints (735px / 1068px — Apple's system)

**Spec:** `.aicrowd/superpowers/specs/2026-04-15-docs-cathedral-direction.md`

---

## Design System Reference

Fused from Stripe Docs + Apple Developer Docs + whestbench-explorer:

```
TYPOGRAPHY
  Font stack:     Inter (body), Montserrat (hero/h1), IBM Plex Mono (code)
  Body:           16px / 1.6 line-height / #292C2D
  H1 (hero):      48px / 1.08 / Montserrat 700 / -0.03em tracking
  H1 (page):      32px / 1.25 / Montserrat 700 / -0.02em tracking
  H2:             24px / 1.3 / Inter 600
  H3:             18px / 1.4 / Inter 600
  Code inline:    14px / IBM Plex Mono / on #F1F3F5 bg
  Code block:     13.5px / IBM Plex Mono / on #0a2540 (dark) bg
  Sidebar nav:    14px / Inter 400
  Sidebar heads:  11px / Inter 600 / uppercase / 0.08em tracking / #AAACAD
  Small/meta:     13px / Inter 400 / #5D5F60

COLORS
  Background:     #FAFBFC (page) — between Apple #f5f5f7 and Stripe #f6f9fc
  Surface:        #FFFFFF (cards, sidebar, content panels)
  Text primary:   #292C2D (whest gray-900)
  Text secondary: #5D5F60 (whest gray-600)
  Text tertiary:  #AAACAD (whest gray-400)
  Accent:         #F0524D (whest coral — kept from explorer)
  Accent hover:   #D23934
  Accent light:   #FEF2F1
  Border:         #E3E8EE (Stripe-style light blue-gray, softer than #D9DCDC)
  Code bg (dark): #0F1B2D (near Stripe's #0a2540, slightly warmer)
  Code text:      #E8F0FE (light blue-white on dark bg)
  Success:        #23B761 (whest green)
  Warning:        #FA9E33 (whest amber)

SPACING (8px base grid)
  4px / 8px / 12px / 16px / 20px / 24px / 32px / 48px / 64px / 80px

BORDER RADIUS
  Small:  4px (inline code, tooltips)
  Medium: 6px (inputs, buttons)
  Large:  8px (cards, code blocks)
  XL:     12px (hero cards)
  Pill:   9999px (badges)

BREAKPOINTS
  Mobile:  <= 735px
  Tablet:  736px — 1068px
  Desktop: >= 1069px

LAYOUT
  Sidebar:        260px
  Content max:    720px
  Page max:       1200px
  Nav height:     56px

TRANSITIONS
  Fast:   0.15s ease (hover states, micro-interactions)
  Medium: 0.3s ease (layout shifts, reveals)
```

---

## File Structure

```
website/src/
  css/
    custom.css              ← REWRITE (273 lines → ~350 lines)
  pages/
    index.tsx               ← REWRITE (96 lines → ~120 lines)
    index.module.css        ← REWRITE (141 lines → ~200 lines)
  components/
    api-reference/
      styles.module.css     ← MODIFY (update to match new design tokens)
    shared/
      SortableTable.module.css ← MODIFY (update to match new design tokens)
  theme/                    ← CREATE (swizzled overrides)
    CodeBlock/
      Content/
        styles.module.css   ← CREATE (dark code blocks)

website/
  docusaurus.config.ts      ← MODIFY (prism dark theme, navbar tweaks)
```

---

## Task 1: Rewrite Global CSS Design System

**Files:**
- Rewrite: `website/src/css/custom.css`

This is the foundation — every other task depends on it looking right.

- [ ] **Step 1: Replace `website/src/css/custom.css` entirely**

Replace the full file content with the new design system. The new CSS must:

1. Define all design tokens as CSS custom properties in `:root`
2. Set `html { font-size: 16px }` and `body { font-size: 1rem; line-height: 1.6 }`
3. Style the navbar: white bg, 56px height, 1px bottom border `#E3E8EE`, Montserrat title
4. Style the sidebar: white bg, right border `#E3E8EE`, 260px width, category heads 11px uppercase `#AAACAD`, nav items 14px with 6px/12px padding, active state coral bg `#FEF2F1` + coral text + left 3px border, hover `#FAFBFC` bg, transition 0.15s
5. Style headings: h1 Montserrat 700 32px, h2 Inter 600 24px with `#E3E8EE` border-bottom, h3 Inter 600 18px
6. Style body text: 16px Inter, `#292C2D`, 1.6 line-height
7. Style inline code: IBM Plex Mono 14px, `#F1F3F5` bg, `#5D5F60` text, 4px radius, 2px 6px padding
8. Style code blocks: `#0F1B2D` dark bg, `#E8F0FE` text, 8px radius, 16px 20px padding, 13.5px font, 1px border `rgba(255,255,255,0.06)`
9. Style tables: `#FFFFFF` bg, `#E3E8EE` 1px borders, th 13px uppercase `#AAACAD` with `#FAFBFC` bg, td 14px, row hover `#FAFBFC`, 8px radius on table wrapper
10. Style admonitions: 8px radius, 4px left border, refined padding
11. Style TOC (right sidebar): 13px, `#AAACAD` color, coral on active, transition 0.15s
12. Style footer: white bg, top border, 13px centered text, `#AAACAD`
13. Style pagination: 8px radius cards, coral hover border
14. Set content max-width: 720px
15. Page background: `#FAFBFC`
16. All interactive elements: `transition: all 0.15s ease`

Here is the complete CSS:

```css
/* ══════════════════════════════════════════════════════════
   whest docs — Apple Marketing meets Stripe Docs
   Coral accent from whestbench-explorer
   ══════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600;700&family=Montserrat:wght@600;700&display=swap');

/* ── Design Tokens ────────────────────────────────────── */
:root {
  /* Accent (coral) */
  --whest-accent: #F0524D;
  --whest-accent-hover: #D23934;
  --whest-accent-light: #FEF2F1;
  --whest-accent-lighter: #FFF5F5;

  /* Neutrals */
  --whest-white: #FFFFFF;
  --whest-bg: #FAFBFC;
  --whest-gray-50: #F6F8FA;
  --whest-gray-100: #F1F3F5;
  --whest-gray-200: #E3E8EE;
  --whest-gray-400: #AAACAD;
  --whest-gray-600: #5D5F60;
  --whest-gray-900: #292C2D;
  --whest-code-bg: #0F1B2D;
  --whest-code-text: #E8F0FE;

  /* Success / Warning */
  --whest-success: #23B761;
  --whest-warning: #FA9E33;

  /* Infima overrides */
  --ifm-color-primary: var(--whest-accent);
  --ifm-color-primary-dark: #EE403A;
  --ifm-color-primary-darker: var(--whest-accent-hover);
  --ifm-color-primary-darkest: #B52E2A;
  --ifm-color-primary-light: #F36B67;
  --ifm-color-primary-lighter: #F7A09D;
  --ifm-color-primary-lightest: var(--whest-accent-light);

  --ifm-font-family-base: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --ifm-font-family-monospace: 'IBM Plex Mono', 'SF Mono', SFMono-Regular, ui-monospace, monospace;
  --ifm-heading-font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

  --ifm-heading-color: var(--whest-gray-900);
  --ifm-font-color-base: var(--whest-gray-900);
  --ifm-font-color-secondary: var(--whest-gray-600);
  --ifm-background-color: var(--whest-bg);
  --ifm-background-surface-color: var(--whest-white);

  --ifm-global-radius: 8px;
  --ifm-code-border-radius: 4px;
  --ifm-code-font-size: 0.875em;

  --ifm-global-shadow-lw: 0 1px 3px rgba(0, 0, 0, 0.04);
  --ifm-global-shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);

  --docusaurus-highlighted-code-line-bg: rgba(240, 82, 77, 0.08);
}

/* Match dark mode to light (light-only site) */
[data-theme='dark'] {
  --ifm-color-primary: var(--whest-accent);
  --ifm-background-color: var(--whest-bg);
  --ifm-background-surface-color: var(--whest-white);
  --ifm-heading-color: var(--whest-gray-900);
  --ifm-font-color-base: var(--whest-gray-900);
}

/* ── Base ─────────────────────────────────────────────── */
html {
  font-size: 16px !important;
}

body {
  font-size: 1rem !important;
  line-height: 1.6;
  background: var(--whest-bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ── Navbar ───────────────────────────────────────────── */
.navbar {
  background: var(--whest-white);
  border-bottom: 1px solid var(--whest-gray-200);
  box-shadow: none;
  height: 56px;
  padding: 0 24px;
}

.navbar__inner {
  max-width: 1200px;
  margin: 0 auto;
}

.navbar__title {
  font-family: 'Montserrat', var(--ifm-font-family-base);
  font-weight: 700;
  font-size: 17px;
  letter-spacing: -0.02em;
  color: var(--whest-gray-900);
}

.navbar__link {
  font-size: 14px;
  font-weight: 500;
  color: var(--whest-gray-600);
  transition: color 0.15s ease;
}

.navbar__link:hover {
  color: var(--whest-gray-900);
}

/* ── Sidebar ──────────────────────────────────────────── */
.theme-doc-sidebar-container {
  border-right: 1px solid var(--whest-gray-200) !important;
}

nav.menu {
  background: var(--whest-white);
  padding: 20px 12px;
}

.menu__link {
  font-size: 14px;
  font-weight: 400;
  color: var(--whest-gray-600);
  border-radius: 6px;
  padding: 6px 12px;
  transition: all 0.15s ease;
  border-left: 3px solid transparent;
}

.menu__link:hover {
  background: var(--whest-gray-50);
  color: var(--whest-gray-900);
  text-decoration: none;
}

.menu__link--active:not(.menu__link--sublist) {
  color: var(--whest-accent) !important;
  font-weight: 600;
  background: var(--whest-accent-light);
  border-left-color: var(--whest-accent);
}

/* Category labels in sidebar */
.menu__list-item-collapsible > .menu__link {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--whest-gray-400);
  border-left: none;
}

.menu__list-item-collapsible > .menu__link:hover {
  background: transparent;
  color: var(--whest-gray-600);
}

/* ── Headings ─────────────────────────────────────────── */
.markdown h1,
article h1 {
  font-family: 'Montserrat', var(--ifm-font-family-base);
  font-weight: 700;
  font-size: 32px;
  line-height: 1.25;
  letter-spacing: -0.02em;
  color: var(--whest-gray-900);
  margin-bottom: 16px;
}

.markdown h2 {
  font-weight: 600;
  font-size: 24px;
  line-height: 1.3;
  color: var(--whest-gray-900);
  border-bottom: 1px solid var(--whest-gray-200);
  padding-bottom: 8px;
  margin-top: 48px;
  margin-bottom: 16px;
}

.markdown h3 {
  font-weight: 600;
  font-size: 18px;
  line-height: 1.4;
  color: var(--whest-gray-900);
  margin-top: 32px;
  margin-bottom: 12px;
}

/* ── Body text ────────────────────────────────────────── */
.markdown p {
  margin-bottom: 16px;
}

.markdown li {
  margin-bottom: 6px;
}

.markdown a {
  color: var(--whest-accent);
  text-decoration: none;
  transition: color 0.15s ease;
}

.markdown a:hover {
  color: var(--whest-accent-hover);
  text-decoration: underline;
}

/* ── Inline code ──────────────────────────────────────── */
:not(pre) > code {
  font-family: var(--ifm-font-family-monospace);
  background: var(--whest-gray-100);
  color: var(--whest-gray-600);
  border: none;
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 0.875em;
}

/* ── Code blocks (dark, Stripe-inspired) ──────────────── */
pre {
  background: var(--whest-code-bg) !important;
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 8px;
  padding: 16px 20px;
}

pre code {
  color: var(--whest-code-text);
  font-size: 13.5px;
  line-height: 1.6;
  background: none !important;
}

/* Prism token overrides for dark bg */
.token.comment,
.token.prolog {
  color: #637777;
}

.token.keyword {
  color: #c792ea;
}

.token.string {
  color: #a5d6ff;
}

.token.function {
  color: #7ee787;
}

.token.number {
  color: #f9ae58;
}

.token.operator {
  color: #79c0ff;
}

.token.builtin,
.token.class-name {
  color: #ffa657;
}

/* Copy button */
.copyButtonCopied_node_modules-\@docusaurus-theme-classic-lib-theme-CodeBlock-CopyButton-styles-module,
[class*="copyButton"] {
  color: var(--whest-gray-400);
  transition: color 0.15s ease;
}

[class*="copyButton"]:hover {
  color: var(--whest-code-text);
}

/* ── Tables ───────────────────────────────────────────── */
table {
  border: 1px solid var(--whest-gray-200);
  border-radius: 8px;
  overflow: hidden;
  font-size: 14px;
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}

table th {
  background: var(--whest-gray-50);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--whest-gray-400);
  padding: 10px 16px;
  border-bottom: 1px solid var(--whest-gray-200);
  text-align: left;
}

table td {
  font-size: 14px;
  color: var(--whest-gray-900);
  padding: 10px 16px;
  border-bottom: 1px solid var(--whest-gray-200);
}

table tr:last-child td {
  border-bottom: none;
}

table tr:hover td {
  background: var(--whest-bg);
}

/* ── Admonitions ──────────────────────────────────────── */
.theme-admonition {
  border-radius: 8px;
  border-left-width: 4px;
  background: var(--whest-white);
  box-shadow: var(--ifm-global-shadow-lw);
}

/* ── TOC (right sidebar) ─────────────────────────────── */
.table-of-contents {
  border-left: 1px solid var(--whest-gray-200);
  padding-left: 12px;
}

.table-of-contents__link {
  font-size: 13px;
  color: var(--whest-gray-400);
  transition: color 0.15s ease;
}

.table-of-contents__link:hover {
  color: var(--whest-gray-900);
}

.table-of-contents__link--active {
  color: var(--whest-accent);
  font-weight: 500;
}

/* ── Breadcrumbs ──────────────────────────────────────── */
.breadcrumbs__link {
  font-size: 13px;
  color: var(--whest-gray-400);
}

.breadcrumbs__link:hover {
  color: var(--whest-accent);
}

/* ── Footer ───────────────────────────────────────────── */
.footer {
  background: var(--whest-white);
  border-top: 1px solid var(--whest-gray-200);
  padding: 20px;
  text-align: center;
}

.footer__copyright {
  font-size: 13px;
  color: var(--whest-gray-400);
}

/* ── Pagination ───────────────────────────────────────── */
.pagination-nav__link {
  border: 1px solid var(--whest-gray-200);
  border-radius: 8px;
  transition: all 0.15s ease;
}

.pagination-nav__link:hover {
  border-color: var(--whest-accent);
  box-shadow: var(--ifm-global-shadow-lw);
}

.pagination-nav__sublabel {
  font-size: 12px;
  color: var(--whest-gray-400);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}

/* ── Content width ────────────────────────────────────── */
.markdown {
  max-width: 720px;
}

/* ── Scrollbar (webkit) ───────────────────────────────── */
.menu::-webkit-scrollbar {
  width: 4px;
}

.menu::-webkit-scrollbar-thumb {
  background: var(--whest-gray-200);
  border-radius: 4px;
}

/* ── Responsive ───────────────────────────────────────── */
@media (max-width: 735px) {
  .markdown h1, article h1 {
    font-size: 24px;
  }

  .markdown h2 {
    font-size: 20px;
    margin-top: 32px;
  }

  .markdown h3 {
    font-size: 16px;
  }
}

@media (min-width: 736px) and (max-width: 1068px) {
  .markdown h1, article h1 {
    font-size: 28px;
  }

  .markdown h2 {
    font-size: 22px;
  }
}
```

- [ ] **Step 2: Verify build**

```bash
cd website && rm -rf .docusaurus && npm run build
```

Expected: Build succeeds.

- [ ] **Step 3: Start dev server and visually verify**

```bash
cd website && npm start
```

Check in browser:
- Landing page: coral accents, new spacing
- Docs page (getting-started/installation): dark code blocks, proper heading sizes, sidebar styling
- API Reference: table styling consistent with new tokens
- Verify no console errors

- [ ] **Step 4: Commit**

```bash
git add website/src/css/custom.css
git commit -m "style: rewrite design system — Apple Marketing meets Stripe

Dark code blocks, refined sidebar with active left-border accent,
Montserrat hero headings, 8px grid spacing, Stripe-inspired borders
and hover transitions. 16px base text with 1.6 line-height."
```

---

## Task 2: Rewrite Landing Page

**Files:**
- Rewrite: `website/src/pages/index.tsx`
- Rewrite: `website/src/pages/index.module.css`

- [ ] **Step 1: Replace `website/src/pages/index.tsx`**

The new landing page should have:
1. **Hero section** — large Montserrat heading (48px), tagline, two CTA buttons, and an `install` code snippet
2. **Feature highlights** — 3 cards in a row explaining core value props (not navigation cards — *value* cards: "Analytical FLOP Counting", "508 Operations", "Budget Control")
3. **Navigation cards** — 6 cards in 3x2 grid linking to doc sections (same as current but restyled)
4. **Code example** — with dark code block styling

Key design decisions:
- Hero has a subtle gradient or colored band at the top (coral-tinted, very subtle)
- The install snippet uses dark code block styling inline in the hero
- Feature cards have icons/numbers, not emojis
- Navigation cards use the `→` arrow pattern from Stripe docs

Replace with:

```tsx
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
    number: '∞',
    label: 'Composable',
    desc: 'Budget contexts, namespaces, and per-operation breakdowns',
  },
];

const navCards = [
  { title: 'Get Started', desc: 'Install whest and create your first FLOP budget', link: '/getting-started/installation' },
  { title: 'Migrate from NumPy', desc: 'Convert existing code with FLOP annotations', link: '/how-to/migrate-from-numpy' },
  { title: 'API Reference', desc: 'Search all 508 operations with interactive filters', link: '/api' },
  { title: 'FLOP Counting Model', desc: 'How costs are computed analytically', link: '/concepts/flop-counting-model' },
  { title: 'Symmetry Explorer', desc: 'Interactive einsum symmetry visualization', link: '/explanation/symmetry-explorer' },
  { title: 'For AI Agents', desc: 'Machine-readable resources and rules', link: '/reference/for-agents' },
];

const exampleCode = `import whest as we

with we.BudgetContext(flop_budget=10**8) as budget:
    W = we.random.randn(256, 256)
    x = we.random.randn(256)
    y = we.einsum('ij,j->i', W, x)

we.budget_summary()`;

export default function Home(): React.JSX.Element {
  return (
    <Layout title="whest" description="NumPy-compatible math primitives with analytical FLOP counting">
      {/* Hero */}
      <header className={styles.hero}>
        <div className={styles.heroInner}>
          <h1 className={styles.heroTitle}>
            Count every FLOP.
          </h1>
          <p className={styles.heroSubtitle}>
            NumPy-compatible math primitives with analytical FLOP counting.
            Every operation has a cost. Every cost is a choice.
          </p>
          <div className={styles.heroCta}>
            <Link className={styles.btnPrimary} to="/getting-started/installation">
              Get Started
            </Link>
            <Link className={styles.btnSecondary} to="/api">
              API Reference
            </Link>
          </div>
          <div className={styles.heroInstall}>
            <code>uv add git+https://github.com/AIcrowd/whest.git</code>
          </div>
        </div>
      </header>

      {/* Feature numbers */}
      <section className={styles.features}>
        {features.map((f) => (
          <div key={f.label} className={styles.feature}>
            <div className={styles.featureNumber}>{f.number}</div>
            <div className={styles.featureLabel}>{f.label}</div>
            <p className={styles.featureDesc}>{f.desc}</p>
          </div>
        ))}
      </section>

      {/* Code example */}
      <section className={styles.codeSection}>
        <h2 className={styles.sectionTitle}>See it in action</h2>
        <CodeBlock language="python">{exampleCode}</CodeBlock>
      </section>

      {/* Navigation cards */}
      <section className={styles.navSection}>
        <h2 className={styles.sectionTitle}>Explore the docs</h2>
        <div className={styles.navGrid}>
          {navCards.map((card) => (
            <Link key={card.title} to={card.link} className={styles.navCard}>
              <h3 className={styles.navCardTitle}>{card.title}</h3>
              <p className={styles.navCardDesc}>{card.desc}</p>
              <span className={styles.navCardArrow}>→</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Minimal footer spacer */}
      <div className={styles.footerSpacer} />
    </Layout>
  );
}
```

- [ ] **Step 2: Replace `website/src/pages/index.module.css`**

```css
/* ── Hero ─────────────────────────────────────────────── */
.hero {
  background: linear-gradient(180deg, var(--whest-accent-lighter) 0%, var(--whest-bg) 100%);
  padding: 80px 24px 64px;
}

.heroInner {
  max-width: 680px;
  margin: 0 auto;
  text-align: center;
}

.heroTitle {
  font-family: 'Montserrat', var(--ifm-font-family-base);
  font-size: 48px;
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.03em;
  color: var(--whest-gray-900);
  margin: 0 0 20px;
}

.heroSubtitle {
  font-size: 18px;
  line-height: 1.6;
  color: var(--whest-gray-600);
  margin: 0 0 32px;
  max-width: 520px;
  margin-left: auto;
  margin-right: auto;
}

.heroCta {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-bottom: 24px;
}

.btnPrimary {
  background: var(--whest-accent);
  color: white;
  padding: 10px 24px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 15px;
  text-decoration: none;
  transition: all 0.15s ease;
}

.btnPrimary:hover {
  background: var(--whest-accent-hover);
  color: white;
  text-decoration: none;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(240, 82, 77, 0.2);
}

.btnSecondary {
  border: 1px solid var(--whest-gray-200);
  color: var(--whest-gray-900);
  padding: 10px 24px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 15px;
  text-decoration: none;
  transition: all 0.15s ease;
}

.btnSecondary:hover {
  border-color: var(--whest-gray-900);
  color: var(--whest-gray-900);
  text-decoration: none;
  transform: translateY(-1px);
}

.heroInstall {
  display: inline-block;
  background: var(--whest-code-bg);
  border-radius: 8px;
  padding: 10px 20px;
}

.heroInstall code {
  color: var(--whest-code-text);
  font-family: var(--ifm-font-family-monospace);
  font-size: 14px;
  background: none;
  padding: 0;
}

/* ── Features ─────────────────────────────────────────── */
.features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  max-width: 960px;
  margin: 0 auto;
  padding: 64px 24px;
}

.feature {
  text-align: center;
  padding: 32px 24px;
}

.featureNumber {
  font-family: 'Montserrat', var(--ifm-font-family-base);
  font-size: 40px;
  font-weight: 700;
  color: var(--whest-accent);
  letter-spacing: -0.02em;
  line-height: 1;
  margin-bottom: 8px;
}

.featureLabel {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--whest-gray-400);
  margin-bottom: 12px;
}

.featureDesc {
  font-size: 15px;
  color: var(--whest-gray-600);
  line-height: 1.5;
  margin: 0;
}

/* ── Code Section ─────────────────────────────────────── */
.codeSection {
  max-width: 640px;
  margin: 0 auto;
  padding: 0 24px 64px;
}

.sectionTitle {
  font-family: 'Montserrat', var(--ifm-font-family-base);
  font-weight: 700;
  font-size: 28px;
  letter-spacing: -0.02em;
  color: var(--whest-gray-900);
  text-align: center;
  margin-bottom: 32px;
}

/* ── Nav Cards ────────────────────────────────────────── */
.navSection {
  max-width: 960px;
  margin: 0 auto;
  padding: 0 24px 80px;
}

.navGrid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.navCard {
  background: var(--whest-white);
  border: 1px solid var(--whest-gray-200);
  border-radius: 12px;
  padding: 24px;
  text-decoration: none;
  color: inherit;
  transition: all 0.15s ease;
  position: relative;
}

.navCard:hover {
  border-color: var(--whest-accent);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  text-decoration: none;
  color: inherit;
  transform: translateY(-2px);
}

.navCardTitle {
  font-size: 16px;
  font-weight: 600;
  color: var(--whest-gray-900);
  margin: 0 0 6px;
}

.navCardDesc {
  font-size: 14px;
  color: var(--whest-gray-600);
  line-height: 1.5;
  margin: 0;
}

.navCardArrow {
  position: absolute;
  top: 24px;
  right: 24px;
  font-size: 18px;
  color: var(--whest-gray-200);
  transition: all 0.15s ease;
}

.navCard:hover .navCardArrow {
  color: var(--whest-accent);
  transform: translateX(3px);
}

/* ── Footer spacer ────────────────────────────────────── */
.footerSpacer {
  height: 0;
}

/* ── Responsive ───────────────────────────────────────── */
@media (max-width: 735px) {
  .hero {
    padding: 48px 20px 40px;
  }

  .heroTitle {
    font-size: 32px;
  }

  .heroSubtitle {
    font-size: 16px;
  }

  .features {
    grid-template-columns: 1fr;
    gap: 0;
    padding: 32px 20px;
  }

  .feature {
    padding: 20px;
  }

  .featureNumber {
    font-size: 32px;
  }

  .navGrid {
    grid-template-columns: 1fr;
  }

  .sectionTitle {
    font-size: 24px;
  }
}

@media (min-width: 736px) and (max-width: 1068px) {
  .heroTitle {
    font-size: 40px;
  }

  .navGrid {
    grid-template-columns: repeat(2, 1fr);
  }
}
```

- [ ] **Step 3: Build and verify**

```bash
cd website && rm -rf .docusaurus && npm run build
```

- [ ] **Step 4: Start dev server and take screenshots**

Verify:
- Hero section: gradient bg, large title, subtitle, two buttons, install snippet
- Feature numbers: 3 columns, coral numbers, uppercase labels
- Code block: dark bg, syntax highlighted
- Nav cards: 3x2 grid, white cards, coral hover, → arrow
- Mobile: 1 column, smaller title

- [ ] **Step 5: Commit**

```bash
git add website/src/pages/
git commit -m "style: redesign landing page — bold hero, feature numbers, dark code

Apple-style hero with 48px Montserrat heading, subtle coral gradient.
Feature numbers section (508 ops, 0 overhead, ∞ composable).
Stripe-style nav cards with → arrow and hover lift.
Dark code block with install snippet."
```

---

## Task 3: Update Docusaurus Config for Dark Code Theme

**Files:**
- Modify: `website/docusaurus.config.ts`

The dark code blocks need the Prism dark theme enabled by default (not just for dark mode).

- [ ] **Step 1: Change the Prism theme configuration**

In `website/docusaurus.config.ts`, change the prism config from:

```typescript
prism: {
  theme: prismThemes.github,
  darkTheme: prismThemes.dracula,
  additionalLanguages: ['python', 'bash', 'json'],
},
```

To:

```typescript
prism: {
  theme: prismThemes.dracula,
  darkTheme: prismThemes.dracula,
  additionalLanguages: ['python', 'bash', 'json'],
},
```

This makes code blocks use the dark theme in both light and dark modes, which pairs with our dark code block CSS.

- [ ] **Step 2: Build and verify**

```bash
cd website && rm -rf .docusaurus && npm run build
```

Expected: Code blocks render with dark theme syntax colors.

- [ ] **Step 3: Commit**

```bash
git add website/docusaurus.config.ts
git commit -m "style: use Dracula prism theme for dark code blocks"
```

---

## Task 4: Update API Reference Styles

**Files:**
- Modify: `website/src/components/api-reference/styles.module.css`

Align the API reference component with the new design tokens. Key changes:
- Border color: `#D9DCDC` → `#E3E8EE` (softer Stripe-style)
- Table header: match new table th styling (11px uppercase, `#AAACAD`)
- Input/select styling: match new border color and focus ring
- Toggle buttons: keep coral active state, update border/bg colors
- Badge colors: keep current (already good)

- [ ] **Step 1: Update the CSS module**

In `website/src/components/api-reference/styles.module.css`, make these specific changes:

Replace all instances of `var(--ifm-color-emphasis-300)` with `var(--whest-gray-200, #E3E8EE)`.

Replace all instances of `var(--ifm-color-emphasis-200)` with `var(--whest-gray-200, #E3E8EE)`.

Replace all instances of `var(--ifm-color-emphasis-100)` with `var(--whest-gray-50, #F6F8FA)`.

Replace all instances of `var(--ifm-color-emphasis-50, rgba(0,0,0,0.02))` with `var(--whest-bg, #FAFBFC)`.

Replace all instances of `var(--ifm-background-surface-color, #f8f9fa)` with `var(--whest-gray-50, #F6F8FA)`.

In `.searchInput:focus`, change the box-shadow from `rgba(64, 81, 181, 0.15)` to `rgba(240, 82, 77, 0.15)` (coral focus ring).

Update `.opsTable` font-size from `0.84rem` to `14px`.

Update `.opsTable td` padding from `0.5rem 0.75rem` to `10px 16px`.

Update `.opsTable thead th` padding from `0.6rem 0.75rem` to `10px 16px`.

Update `.detailLabel` to use `var(--whest-gray-400, #AAACAD)`.

Remove all `[data-theme='dark']` overrides (we're light-only).

- [ ] **Step 2: Update SortableTable styles similarly**

In `website/src/components/shared/SortableTable.module.css`:

Replace `#f8f9fa` with `var(--whest-gray-50, #F6F8FA)`.
Replace `#1a1a2e` with `var(--whest-gray-900, #292C2D)`.
Replace `#666` with `var(--whest-gray-600, #5D5F60)`.
Remove all `[data-theme='dark']` overrides.
Update border colors to `var(--whest-gray-200, #E3E8EE)`.
Update focus ring to `rgba(240, 82, 77, 0.15)`.

- [ ] **Step 3: Build and verify**

```bash
cd website && rm -rf .docusaurus && npm run build
```

- [ ] **Step 4: Verify API reference page visually**

Check: filter bar border colors, toggle button styling, table header font, table cell padding, badge colors, expanded row styling.

- [ ] **Step 5: Commit**

```bash
git add website/src/components/
git commit -m "style: align API reference and table components with new design tokens

Update border colors, focus rings, font sizes, padding to match
the Apple-meets-Stripe design system. Remove dark mode overrides."
```

---

## Parallelization Guide

```
Task 1 (Global CSS)
  ├─→ Task 2 (Landing page) — depends on CSS tokens
  ├─→ Task 3 (Prism theme) — independent of CSS, can run parallel
  └─→ Task 4 (Component styles) — depends on CSS tokens

Recommended order: Task 1 first, then Tasks 2+3+4 in parallel.
```
