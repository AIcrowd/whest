# Free Operations

Zero-cost operations (0 FLOPs) for tensor creation, reshaping, indexing,
and data movement. These never consume budget.

**Includes:** `zeros`, `ones`, `eye`, `reshape`, `transpose`, `stack`,
`split`, `where`, `copy`, `astype`, and 140+ more.

> **Note:** `array`, `linspace`, `arange`, and `concatenate` are **not**
> free. Each charges `numel(output)` FLOPs (one per output element) and
> is classified as a custom-cost operation. See their entries in the API
> reference below for details.

## API Reference

::: whest._free_ops
