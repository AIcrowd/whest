# Symmetry Explorer

Interactive visualization of the subgraph symmetry detection algorithm used by
`mechestim.einsum` to find and exploit permutation symmetry in einsum
expressions.

Choose a preset example or define your own einsum expression with custom
operand symmetries to walk through the algorithm step by step:

1. **Bipartite Graph** — how operands and index labels form the graph
2. **Incidence Matrix M** — the matrix representation and column fingerprints
3. **σ-Loop & π Detection** — animated row/column shuffles showing how π recovers M
4. **Group Construction** — from valid π's to generators via Dimino's algorithm
5. **Burnside Counting** — counting unique elements under the symmetry group
6. **Cost Reduction** — FLOP savings from symmetry

<iframe src="../../visualization/symmetry-explorer/dist/index.html" style="width: 100%; height: 2400px; border: 1px solid #e0e0e0; border-radius: 8px;" loading="lazy" title="Subgraph Symmetry Explorer"></iframe>

!!! tip "Standalone mode"
    You can also run the explorer locally for a better experience:

    ```bash
    cd docs/visualization/symmetry-explorer
    npm install
    npm run dev
    ```

    Or open `docs/visualization/symmetry-explorer/dist/index.html` directly
    in your browser.
