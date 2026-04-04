# Free Operations

Zero-cost operations (0 FLOPs) for tensor creation, reshaping, indexing,
sorting, and data movement. These never consume budget.

**Includes:** `array`, `zeros`, `ones`, `eye`, `arange`, `linspace`,
`reshape`, `transpose`, `concatenate`, `stack`, `split`, `sort`,
`argsort`, `where`, `copy`, `astype`, and 200+ more.

!!! note
    `sort` and `argsort` are free in mechestim. This is intentional —
    comparison-based sorting has no floating-point arithmetic cost.

## API Reference

::: mechestim._free_ops
