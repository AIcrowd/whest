# mechestim

NumPy-compatible math primitives with analytical FLOP counting for the Mechanistic Estimation Challenge.

## Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```

## Usage

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as budget:
    W = me.array(weight_matrix)
    x = me.zeros((256,))
    h = me.einsum('ij,j->i', W, x)
    h = me.maximum(h, 0)
    print(budget.summary())
```

## Documentation

Full documentation at [mechestim docs](https://aicrowd.github.io/mechestim/) (coming soon).

## Development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
uv run pytest
```

## License

MIT
