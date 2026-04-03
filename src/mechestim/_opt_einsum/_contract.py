"""Contains contract_path and supporting types (stripped from opt_einsum contract.py).

Excluded: contract, _core_contract, ContractExpression, _einsum, _tensordot,
_transpose, backends/sharing imports, _filter_einsum_defaults,
format_const_einsum_str, shape_only.
"""

from collections.abc import Collection, Sequence
from decimal import Decimal
from typing import Any, Literal, overload

from . import _blas as blas
from . import _helpers as helpers
from . import _parser as parser
from . import _paths as paths
from ._typing import (
    ArrayIndexType,
    ArrayType,
    ContractionListType,
    OptimizeKind,
    PathType,
)

__all__ = [
    "contract_path",
    "PathInfo",
]

## Common types

_MemoryLimit = None | int | Decimal | Literal["max_input"]


class PathInfo:
    """A printable object to contain information about a contraction path."""

    def __init__(
        self,
        contraction_list: ContractionListType,
        input_subscripts: str,
        output_subscript: str,
        indices: ArrayIndexType,
        path: PathType,
        scale_list: Sequence[int],
        naive_cost: int,
        opt_cost: int,
        size_list: Sequence[int],
        size_dict: dict[str, int],
    ):
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.path = path
        self.indices = indices
        self.scale_list = scale_list
        self.naive_cost = Decimal(naive_cost)
        self.opt_cost = Decimal(opt_cost)
        self.speedup = self.naive_cost / max(self.opt_cost, Decimal(1))
        self.size_list = size_list
        self.size_dict = size_dict

        self.shapes = [tuple(size_dict[k] for k in ks) for ks in input_subscripts.split(",")]
        self.eq = f"{input_subscripts}->{output_subscript}"
        self.largest_intermediate = Decimal(max(size_list, default=1))

    def __repr__(self) -> str:
        # Return the path along with a nice string representation
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = [
            f"  Complete contraction:  {self.eq}\n",
            f"         Naive scaling:  {len(self.indices)}\n",
            f"     Optimized scaling:  {max(self.scale_list, default=0)}\n",
            f"      Naive FLOP count:  {self.naive_cost:.3e}\n",
            f"  Optimized FLOP count:  {self.opt_cost:.3e}\n",
            f"   Theoretical speedup:  {self.speedup:.3e}\n",
            f"  Largest intermediate:  {self.largest_intermediate:.3e} elements\n",
            "-" * 80 + "\n",
            "{:>6} {:>11} {:>22} {:>37}\n".format(*header),
            "-" * 80,
        ]

        for n, contraction in enumerate(self.contraction_list):
            _, _, einsum_str, remaining, do_blas = contraction

            if remaining is not None:
                remaining_str = ",".join(remaining) + "->" + self.output_subscript
            else:
                remaining_str = "..."
            size_remaining = max(0, 56 - max(22, len(einsum_str)))

            path_run = (
                self.scale_list[n],
                do_blas,
                einsum_str,
                remaining_str,
                size_remaining,
            )
            path_print.append("\n{:>4} {:>14} {:>22}    {:>{}}".format(*path_run))

        return "".join(path_print)


def _choose_memory_arg(memory_limit: _MemoryLimit, size_list: list[int]) -> int | None:
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError("memory_limit must be None, int, or the string Literal['max_input'].")

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


# Overload for contract_path(einsum_string, *operands)
@overload
def contract_path(
    subscripts: str,
    *operands: ArrayType,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
) -> tuple[PathType, PathInfo]: ...


# Overload for contract_path(operand, indices, operand, indices, ....)
@overload
def contract_path(
    subscripts: ArrayType,
    *operands: ArrayType | Collection[int],
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
) -> tuple[PathType, PathInfo]: ...


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
) -> tuple[PathType, PathInfo]:
    """Find a contraction order `path`, without performing the contraction.

    Parameters:
          subscripts: Specifies the subscripts for summation.
          *operands: These are the arrays for the operation.
          use_blas: Do you use BLAS for valid operations, may use extra memory for more intermediates.
          optimize: Choose the type of path the contraction will be optimized with.
                - if a list is given uses this as the path.
                - `'optimal'` An algorithm that explores all possible ways of
                contracting the listed tensors. Scales factorially with the number of
                terms in the contraction.
                - `'dp'` A faster (but essentially optimal) algorithm that uses
                dynamic programming to exhaustively search all contraction paths
                without outer-products.
                - `'greedy'` An cheap algorithm that heuristically chooses the best
                pairwise contraction at each step. Scales linearly in the number of
                terms in the contraction.
                - `'random-greedy'` Run a randomized version of the greedy algorithm
                32 times and pick the best path.
                - `'random-greedy-128'` Run a randomized version of the greedy
                algorithm 128 times and pick the best path.
                - `'branch-all'` An algorithm like optimal but that restricts itself
                to searching 'likely' paths. Still scales factorially.
                - `'branch-2'` An even more restricted version of 'branch-all' that
                only searches the best two options at each step. Scales exponentially
                with the number of terms in the contraction.
                - `'auto'` Choose the best of the above algorithms whilst aiming to
                keep the path finding time below 1ms.
                - `'auto-hq'` Aim for a high quality contraction, choosing the best
                of the above algorithms whilst aiming to keep the path finding time
                below 1sec.

          memory_limit: Give the upper bound of the largest intermediate tensor contract will build.
                - None or -1 means there is no limit
                - `max_input` means the limit is set as largest input tensor
                - a positive integer is taken as an explicit limit on the number of elements

                The default is None. Note that imposing a limit can make contractions
                exponentially slower to perform.

          shapes: Whether ``contract_path`` should assume arrays (the default) or array shapes have been supplied.

    Returns:
          path: The optimized einsum contraction path
          PathInfo: A printable object containing various information about the path found.

    Notes:
          The resulting path indicates which terms of the input contraction should be
          contracted first, the result of this contraction is then appended to the end of
          the contraction list.

    Examples:
          We can begin with a chain dot example. In this case, it is optimal to
          contract the b and c tensors represented by the first element of the path (1,
          2). The resulting tensor is added to the end of the contraction and the
          remaining contraction, `(0, 1)`, is then executed.

      ```python
      path_info = contract_path('ij,jk,kl->il', (2,3), (3,4), (4,5), shapes=True)
      print(path_info[0])
      #> [(1, 2), (0, 1)]
      ```
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    # Python side parsing
    operands_ = [subscripts] + list(operands)
    input_subscripts, output_subscript, operands_prepped = parser.parse_einsum_input(operands_, shapes=shapes)

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [frozenset(x) for x in input_list]
    if shapes:
        input_shapes = list(operands_prepped)
    else:
        input_shapes = [parser.get_shape(x) for x in operands_prepped]
    output_set = frozenset(output_subscript)
    indices = frozenset(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict: dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]

        if len(sh) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [helpers.compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    num_ops = len(input_list)

    # Compute naive cost
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)

    # Compute the path
    if optimize is False:
        path_tuple: PathType = [tuple(range(num_ops))]
    elif not isinstance(optimize, (str, paths.PathOptimizer)):
        # Custom path supplied
        path_tuple = optimize  # type: ignore
    elif num_ops <= 2:
        # Nothing to be optimized
        path_tuple = [tuple(range(num_ops))]
    elif isinstance(optimize, paths.PathOptimizer):
        # Custom path optimizer supplied
        path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = paths.get_path_fn(optimize)
        path_tuple = path_optimizer(input_sets, output_set, size_dict, memory_arg)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path_tuple):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = helpers.flop_count(idx_contract, bool(idx_removed), len(contract_inds), size_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, "".join(out_inds), idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path_tuple)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shapes.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining: tuple[str, ...] | None = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    path_print = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path_tuple,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path_tuple, path_print
