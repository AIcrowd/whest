# Symmetry Explorer

Interactive React/Vite explorer for understanding `einsum` symmetry detection, full-label `π` mappings, Burnside evaluation counting, and orbit-projected reduction cost.

## Development

```bash
cd /Users/mohanty/.codex/worktrees/0775/whest/docs/visualization/symmetry-explorer
npm install
npm run dev -- --host 127.0.0.1
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/).

## Test and verify

```bash
cd /Users/mohanty/.codex/worktrees/0775/whest/docs/visualization/symmetry-explorer
npm run test
npm run verify
```

`npm run verify` runs:

- ESLint across the explorer source
- the Node-based engine and teaching-model tests
- a production Vite build

## Visual smoke check

```bash
cd /Users/mohanty/.codex/worktrees/0775/whest/docs/visualization/symmetry-explorer
npm run dev -- --host 127.0.0.1
npx playwright screenshot http://127.0.0.1:5173 /tmp/symmetry-explorer.png
```

When checking the UI, verify:

- the pseudocode rail stays visible while scrolling
- the `σ`/`π` step surfaces `cross`, `V-only`, `W-only`, and `correlated` behavior
- Group Construction treats the full group on active labels as the primary object
- Burnside is framed as evaluation cost
- the cost section explains reduction cost through orbit-projected output updates

## Explorer model

The explorer now teaches the cost model directly through:

```python
evaluation_cost = 0
reduction_cost = 0

for rep in RepSet:
    base_val = product_of_operand_entries_at(rep)
    evaluation_cost += max(num_terms - 1, 0)

    for out in Outs(rep):
        R[out] += coeff(rep, out) * base_val
        reduction_cost += 1
```

with:

- `RepSet`: one representative from each full symmetry orbit
- `Outs(rep)`: distinct output bins reached by that orbit
- `coeff(rep, out)`: multiplicity of dense tuples in that orbit landing in that output bin
