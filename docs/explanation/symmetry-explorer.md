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

<div style="position: relative; width: 100%; height: 0; padding-bottom: 160%; overflow: hidden;">
  <iframe
    src="../visualization/symmetry-explorer/dist/index.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="Subgraph Symmetry Explorer">
  </iframe>
</div>

!!! tip "Standalone mode"
    You can also run the explorer locally for a better experience:

    ```bash
    cd docs/visualization/symmetry-explorer
    npm install
    npm run dev
    ```

    Or open `docs/visualization/symmetry-explorer/dist/index.html` directly
    in your browser.
