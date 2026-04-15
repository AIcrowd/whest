# Docusaurus Migration Design Spec

Migrate whest documentation from MkDocs Material to Docusaurus, enabling full React ecosystem access throughout the docs.

## Motivation

The current MkDocs Material setup is mature (42 markdown files, auto-generated API docs, MathJax, Mermaid, interactive tables, Symmetry Explorer, CI/CD to GitHub Pages). However, React components are confined to isolated islands (the Symmetry Explorer is a separate Vite app with its own build step). The migration to Docusaurus makes React a first-class citizen on every page via MDX, enabling animated visualizations, interactive explorers, and rich widgets throughout the documentation.

## Design Principles

- **Agent-first** — Docs must generate `llms.txt` and `llms-full.txt` for AI agent consumption.
- **React everywhere** — Any doc page can import and use React components via MDX.
- **API docs from data** — The 482-operation inventory (ops.json) is consumed directly by React components, not pre-rendered to markdown.
- **Big bang migration** — Full replacement of MkDocs, no transition period.

## 1. Project Structure

Docusaurus lives in a `website/` subdirectory, cleanly separated from the Python project.

```
whest/
├── website/                     ← Docusaurus root
│   ├── docs/                    ← .md/.mdx content (migrated from docs/)
│   │   ├── getting-started/
│   │   ├── how-to/
│   │   ├── concepts/
│   │   ├── architecture/
│   │   ├── explanation/
│   │   ├── development/
│   │   ├── api/                 ← MDX pages importing <ApiReference />
│   │   ├── reference/
│   │   ├── troubleshooting/
│   │   └── changelog.md
│   ├── src/
│   │   ├── components/          ← React components
│   │   │   ├── api-reference/
│   │   │   │   ├── ApiReference.tsx
│   │   │   │   ├── OperationRow.tsx
│   │   │   │   ├── FilterBar.tsx
│   │   │   │   └── CostBadge.tsx
│   │   │   ├── symmetry-explorer/
│   │   │   │   ├── SymmetryExplorer.tsx
│   │   │   │   ├── GraphRenderer.tsx
│   │   │   │   ├── EquationPanel.tsx
│   │   │   │   └── engine/      ← symmetry detection logic
│   │   │   └── shared/
│   │   │       ├── SortableTable.tsx
│   │   │       └── CodeBlock.tsx
│   │   ├── pages/
│   │   │   └── index.tsx        ← custom landing page
│   │   ├── plugins/
│   │   │   ├── plugin-llmstxt/
│   │   │   │   └── index.js
│   │   │   └── plugin-api-docs/
│   │   │       └── index.js
│   │   ├── css/
│   │   │   └── custom.css       ← ported from extra.css
│   │   └── theme/               ← swizzled overrides only
│   ├── static/
│   │   ├── img/                 ← logo, assets
│   │   └── ops.json             ← operation metadata (198KB)
│   ├── docusaurus.config.js
│   ├── sidebars.js
│   └── package.json
├── scripts/generate_api_docs.py ← stays at repo root
└── pyproject.toml               ← mkdocs deps removed
```

### Key Decisions

- **`website/` subdirectory**: Keeps Node tooling isolated from the Python project. Clean separation of concerns.
- **Symmetry Explorer absorbed**: No longer a separate Vite app in `docs/visualization/symmetry-explorer/`. Components move to `website/src/components/symmetry-explorer/`. Shared build, shared dependencies, no separate `npm ci` step in CI.
- **Content stays `.md` mostly**: Most of the 42 existing markdown files migrate as-is. Only pages that need embedded React components get renamed to `.mdx`. Docusaurus handles both formats.
- **`ops.json` in `static/`**: The API operation metadata is served as a static asset and loaded at build time via the plugin-api-docs plugin.

## 2. Content Migration

### Feature Mapping

| MkDocs Feature | Docusaurus Equivalent | Effort |
|---|---|---|
| `!!! warning "title"` admonitions | `:::warning[title]` native admonitions | Script |
| MathJax (`pymdownx.arithmatex`) | KaTeX via `@docusaurus/plugin-math` | Plugin |
| Mermaid diagrams (`pymdownx.superfences`) | `@docusaurus/theme-mermaid` | Plugin |
| Code highlighting (`pymdownx.highlight`) | Prism.js built-in (Python + line numbers) | Free |
| Tablesort interactive tables | React `<SortableTable />` component | Build |
| `mkdocstrings` API autodoc | React `<ApiReference />` from ops.json | Build |
| `llmstxt-md` plugin | Custom Docusaurus plugin (postBuild hook) | Build |
| Search (MkDocs built-in) | `@docusaurus/plugin-search-local` or Algolia | Plugin |
| Custom CSS (extra.css) | `website/src/css/custom.css` — port theme vars + overrides | Port |

### File Migration Categories

**Copy as-is (~30 files)**: Getting-started, how-to, concepts, architecture, development, troubleshooting, changelog. Changes limited to:
- Frontmatter additions (`sidebar_position`, `sidebar_label`)
- Admonition syntax conversion (`!!!` → `:::`)
- Both changes are scriptable via a migration script

**Convert to MDX (~8 files)**: Pages that embed React components:
- `explanation/symmetry-explorer.md` → `.mdx` (imports `<SymmetryExplorer />`)
- `reference/operation-audit.md` → `.mdx` (imports `<ApiReference />` or `<SortableTable />`)
- `reference/empirical-weights.md` → `.mdx` (imports `<SortableTable />`)
- `reference/cheat-sheet.md` → `.mdx` (imports `<SortableTable />`)
- API reference: 13 current generated files collapse into a single MDX page importing `<ApiReference />`

**Deleted**: `mkdocs.yml`, `docs/javascripts/mathjax.js`, `docs/javascripts/tablesort.js`, `docs/stylesheets/extra.css`, `docs/visualization/symmetry-explorer/` (absorbed into website/src/components/)

### Sidebar Navigation

Docusaurus auto-generates the sidebar from directory structure. Each folder gets a `_category_.json` file for label and position. Individual docs use frontmatter (`sidebar_position`, `sidebar_label`) for ordering. This replaces the manual `nav:` block in mkdocs.yml entirely.

```js
// sidebars.js
module.exports = {
  docs: [{ type: 'autogenerated', dirName: '.' }],
};
```

## 3. React Components (Day 1)

### `<ApiReference />`

- **Source**: ops.json (198KB, 482 operations)
- **Features**: Search/filter by name, category, cost. Group by module (linalg, fft, random, etc.). Expandable rows with signature, docstring, FLOP formula. Sort by cost, name, call count. Toggle between counted vs free vs all.
- **Replaces**: 13 separate generated API markdown files
- **Data loading**: Via plugin-api-docs `usePluginData()` hook — no runtime fetch, data statically embedded at build time

### `<SymmetryExplorer />`

- **Source**: Existing Vite app (`docs/visualization/symmetry-explorer/`) — React 19, KaTeX
- **Migration**: Move `src/` components into `website/src/components/symmetry-explorer/`. Remove Vite config, use Docusaurus bundler. KaTeX already a shared dependency. Tests migrate to website's test setup.
- **Replaces**: Separate Vite app with its own build step

### `<SortableTable />`

- **Features**: Click column headers to sort. Numeric-aware sorting (FLOP costs). Optional search/filter row. Sticky header for long tables.
- **Replaces**: tablesort.js script injection
- **Usage**: Import in any .mdx file

### Future Components (unlocked by React, not in scope for migration)

- `<FlopCalculator />` — interactive cost estimator with sliders for array dimensions
- `<BudgetVisualizer />` — animated budget consumption timeline
- `<EinsumPlayground />` — type an einsum string, see the contraction animated step by step
- `<MigrationDiff />` — side-by-side NumPy ↔ whest code comparison with animated highlighting

## 4. Custom Docusaurus Plugins

### plugin-llmstxt

- **Hook**: `postBuild`
- **Behavior**: Walks the generated site content, strips HTML, extracts text per doc page. Generates `llms.txt` (index with section labels + one-line descriptions) and `llms-full.txt` (all content concatenated). Uses section config from plugin options mirroring the current mkdocs.yml `llmstxt-md` configuration.
- **Output**: `build/llms.txt` and `build/llms-full.txt`

### plugin-api-docs

- **Hook**: `loadContent`
- **Behavior**: Reads `static/ops.json` at build time. Makes data available to React components via `usePluginData('plugin-api-docs')` hook. Data is statically embedded in the JS bundle and code-split per page. No runtime fetch of the 198KB JSON file.

## 5. CI/CD & Deployment

### GitHub Actions Pipeline

The docs job in `.github/workflows/ci.yml` changes to:

```yaml
docs:
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  needs: [lint, test]
  runs-on: ubuntu-latest
  permissions:
    contents: write
    pages: write
    id-token: write
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Install uv
      uses: astral-sh/setup-uv@v4

    - name: Set up Python
      run: uv python install 3.12

    - name: Install Python dependencies
      run: uv sync --all-extras

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'
        cache-dependency-path: website/package-lock.json

    - name: Install website dependencies
      run: cd website && npm ci

    - name: Generate API data
      run: uv run python scripts/generate_api_docs.py

    - name: Build website
      run: cd website && npm run build

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v4
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: website/build
```

### Changes from Current Pipeline

- **One fewer build step**: No separate `cd symmetry-explorer && npm ci && npm run build`.
- **Single `npm ci`**: One package.json for everything (website/).
- **Same trigger**: Push to main, gated behind lint + test jobs.
- **Same URL**: `https://aicrowd.github.io/whest/` stays unchanged.

### docusaurus.config.js Key Settings

```js
module.exports = {
  title: 'whest',
  tagline: 'NumPy-compatible math primitives with FLOP counting',
  url: 'https://aicrowd.github.io',
  baseUrl: '/whest/',
  organizationName: 'AIcrowd',
  projectName: 'whest',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  // ...
};
```

### Local Development

```bash
# Start dev server with hot reload
cd website && npm start
# → http://localhost:3000/whest/

# Build for production
cd website && npm run build

# Test production build locally
cd website && npm run serve
```

Makefile targets (`make docs-build`, `make docs-serve`) update to point to the Docusaurus commands. Developer workflow stays the same.

## 6. generate_api_docs.py Changes

The existing script currently generates markdown files for mkdocstrings. Post-migration, it simplifies to only generating `ops.json` (which it already produces as an intermediate artifact). The markdown generation codepath is removed. The `--verify` flag is updated to verify `ops.json` is in sync with the source code rather than verifying generated markdown.

## 7. Migration Script

A one-time migration script (`scripts/migrate_to_docusaurus.py`) handles the mechanical conversion:

1. Copy markdown files from `docs/` to `website/docs/`, preserving directory structure
2. Convert admonition syntax: `!!! type "title"` → `:::type[title]`
3. Add frontmatter (`sidebar_position`, `sidebar_label`) based on current mkdocs.yml nav ordering
4. Generate `_category_.json` files for each directory
5. Flag files containing `{: .tablesort}` or similar MkDocs-specific syntax for manual review
6. Copy static assets (logo, images) to `website/static/`

## 8. What Gets Deleted

After migration is verified:
- `mkdocs.yml`
- `docs/` directory (content moved to `website/docs/`)
- `docs/visualization/symmetry-explorer/` (absorbed into `website/src/components/`)
- `docs/javascripts/` (MathJax and tablesort replaced by Docusaurus plugins and React)
- `docs/stylesheets/extra.css` (ported to `website/src/css/custom.css`)
- MkDocs dependencies from `pyproject.toml` (`mkdocs-material`, `mkdocstrings[python]`, `mkdocs-llmstxt-md`)
