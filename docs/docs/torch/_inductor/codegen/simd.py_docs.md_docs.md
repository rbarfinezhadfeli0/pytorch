# Documentation: `docs/torch/_inductor/codegen/simd.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/simd.py_docs.md`
- **Size**: 54,051 bytes (52.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/simd.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/simd.py`
- **Size**: 121,499 bytes (118.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import textwrap
from collections import Counter
from typing import Any, Generic, Optional, TYPE_CHECKING, Union
from typing_extensions import TypeVar

import sympy

import torch
import torch._logging
from torch._inductor import metrics
from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.tiling_utils import analyze_memory_coalescing
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.immutable_collections import immutable_dict
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.symbol import (
    free_symbol_is_type,
    prefix_str,
    symbol_is_type,
    SymT,
)

from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..analyze_preserves_zero_mask import prologue_preserves_zero_mask
from ..codecache import code_hash, PyCodeCache
from ..dependencies import MemoryDep, StarDep, WeakDep


if TYPE_CHECKING:
    from collections.abc import Callable

    from ..ir import IRNode

from ..optimize_indexing import indexing_dtype_strength_reduction
from ..runtime.coordinate_descent_tuner import CoordescTuner
from ..runtime.hints import DeviceProperties
from ..runtime.runtime_utils import green_text, next_power_of_2, yellow_text
from ..scheduler import BaseSchedulerNode, BaseScheduling, WhyNoFuse
from ..utils import (
    cache_property_on_self,
    expr_fits_within_32bit,
    get_dtype_size,
    IndentedBuffer,
    Placeholder,
    prefix_is_reduction,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    unique,
)
from ..virtualized import ops, OpsWrapper, V
from .block_analysis import BlockPatternMatcher
from .common import CSEVariable, index_prevent_reordering, Kernel, PythonPrinter
from .multi_kernel import MultiKernel, SizeHintMultiKernel
from .simd_kernel_features import (
    DisableReduction,
    EnableReduction,
    NodeScheduleEntry,
    NodeScheduleMarker,
    SIMDKernelFeatures,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from torch._inductor.tiling_utils import CoalesceVarAnalysis


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


pexpr = PythonPrinter().doprint

all_prefixes = OrderedSet(["z", "y", "x", "r0_", "r1_"])


def get_max_tiles(default: int = 2) -> int:
    max_tiles = torch._inductor.config.triton.max_tiles
    return max_tiles if max_tiles is not None else default


@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: list[sympy.Symbol],
        var_ranges: dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        *,
        kernel: SIMDKernel,
        divisor=sympy.S.One,
        length=sympy.S.One,
        root: IterationRangesRoot,
    ) -> None:
        super().__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length
        self.kernel = kernel
        self.root = root

    @property
    @cache_property_on_self
    def is_reduction(self) -> bool:
        return prefix_is_reduction(self.prefix)

    def symbol(self) -> sympy.Symbol:
        return sympy_index_symbol(self.name)

    @property
    @cache_property_on_self
    def symt(self) -> SymT:
        prefix_to_symt = {prefix: symt for symt, prefix in prefix_str.items()}
        return prefix_to_symt[self.prefix]


class IterationRangesRoot(IterationRanges):
    """
    Root of a iteration range tree that represents a single
    tiled dimension in the output kernel. It contains multiple
    sets of iteration represented with IterationRangesEntry.
    """

    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        prefix: str,
        index: int,
        kernel: SIMDKernel,
        pid_cache: Optional[dict[str, str]] = None,
        *,
        is_loop: bool,
        tensor_dim: Optional[int],
        grid_dim: Optional[int],
        has_zdim: bool,
    ) -> None:
        if pid_cache is None:
            pid_cache = {}
        super().__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
            kernel=kernel,
            root=self,
        )
        self.index = index
        # Store all the nodes in one flat list
        self.nodes: dict[sympy.Expr, IterationRangesEntry] = {}
        # This is for re-ordering program ID in triton mm template
        # pid_cache["tl.program_id(0)"] = pid_m
        self.pid_cache: dict[str, str] = pid_cache

        # True if the dimension is implemented as a single program looping over
        # the full dimension (currently only used for non-persistent reduction)
        # pyrefly: ignore [missing-argument]
        assert not is_loop or (self.is_reduction and grid_dim is None)
        self.is_loop = is_loop
        # Index of corresponding dimension on triton tensors
        self.tensor_dim = tensor_dim
        # Index of corresponding dimension in the triton grid
        self.grid_dim = grid_dim
        self.has_zdim = has_zdim

    def __repr__(self) -> str:
        return f"IterationRangesRoot({self.name!r}, {self.numel}, ...)"

    def cache_clear(self) -> None:
        for node in self.nodes.values():
            node.cache_clear()

    def index_sym(self) -> sympy.Symbol:
        return sympy_index_symbol(f"{self.prefix}index")

    def lookup(self, divisor: sympy.Expr, length: sympy.Expr) -> IterationRangesEntry:
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(self.index_sym(), divisor)
        else:
            expr = ModularIndexing(self.index_sym(), divisor, length)

        if expr not in self.nodes:
            node = IterationRangesEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct_entries(
        self, lengths: list[sympy.Expr]
    ) -> list[IterationRangesEntry]:
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return [*reversed(itervars)]

    def construct(self, lengths: list[sympy.Expr]) -> list[sympy.Symbol]:
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(
        self, index: sympy.Expr
    ) -> tuple[list[sympy.Symbol], list[sympy.Expr]]:
        """Figure out vars from this tree used in index"""

        def get_sort_key(x: IterationRangesEntry) -> tuple[int, bool]:
            """
            Gets the key for sorting nodes. When two nodes have the
            same divisor, the node with length as 1 should be handled
            first so the current divisor is not changed after multiplied
            node.length. Returns `not length_is_one_hint` for ascending
            sort.
            """
            divisor_hint = V.graph.sizevars.size_hint(
                x.divisor, fallback=config.unbacked_symint_fallback
            )
            length_is_one_hint = (
                V.graph.sizevars.size_hint(
                    x.length, fallback=config.unbacked_symint_fallback
                )
                == 1
            )
            return (divisor_hint, not length_is_one_hint)

        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: get_sort_key(x))
        divisor = sympy.S.One
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length

        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                # fill in unused index var
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            # fill in unused index var
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))

        return [*reversed(index_vars)], [*reversed(sizes)]


class IterationRangesEntry(IterationRanges):
    def __init__(
        self,
        name: str,
        divisor: sympy.Expr,
        length: sympy.Expr,
        expr: sympy.Expr,
        parent: IterationRanges,
    ) -> None:
        super().__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
            kernel=parent.kernel,
            root=parent.root,
        )
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def __repr__(self) -> str:
        return f"IterationRangesEntry({self.name}, {self.divisor}, {self.length}, {self.expr}, {self.var_ranges})"

    def set_name(self, name: str) -> None:
        self.codegen = lambda: name  # type: ignore[assignment]
        self.codegen.cache_clear = lambda: None  # type: ignore[method-assign]
        self.name = name

    def cache_clear(self) -> None:
        self.codegen.cache_clear()

    def _codegen(self) -> str:
        V.kernel.codegen_iteration_ranges_entry(self)
        return self.name

    def precomputed_args(self) -> list[sympy.Expr]:
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args: list[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(
                    symbol_is_type(s, SymT.SIZE) for s in symbols
                ):
                    precomputed_args.append(arg)
        return precomputed_args

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, IterationRangesEntry)
        return self.name == other.name


def constant_repr(value: Union[int, float]) -> str:
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


CSEVariableType = TypeVar("CSEVariableType", bound=CSEVariable, default=CSEVariable)


@dataclasses.dataclass
class PartialAccumulate:
    buffer_name: str
    reduction_type: str
    value: Any


class SIMDKernel(Kernel[CSEVariableType], Generic[CSEVariableType]):
    """
    Common base class for Triton/Halide codegen which both use flattened indexing rather than loop nests.
    """

    sexpr: Callable[[sympy.Expr], str] = pexpr
    kexpr: Callable[[sympy.Expr], str]
    allow_block_ptr: bool = False
    # pyrefly: ignore [bad-override]
    kernel_name: str

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        features: SIMDKernelFeatures,
        pid_cache: Optional[dict[str, str]] = None,
        override_persistent_reduction: Optional[bool] = None,
        override_cooperative_reduction: Optional[bool] = None,
        tiling_scores: Optional[dict[str, sympy.Expr]] = None,
        mix_order_reduction: bool = False,
    ) -> None:
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.features = features
        self.mutations = features.get_mutations()
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.numels = {
            prefix: V.graph.sizevars.simplify(val) for prefix, val in tiling.items()
        }
        self.range_trees: list[IterationRangesRoot] = []
        self.range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry] = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = features.is_reduction()
        self.cooperative_reduction: bool = (
            override_cooperative_reduction
            if override_cooperative_reduction is not None
            else self.should_use_cooperative_reduction()
        )
        self.tiling_scores: Optional[dict[str, sympy.Expr]] = tiling_scores
        self.tiling: dict[str, sympy.Expr] = tiling
        self.persistent_reduction: bool = (
            override_persistent_reduction
            if override_persistent_reduction is not None
            else self.should_use_persistent_reduction()
        )
        self.mix_order_reduction: bool = mix_order_reduction
        self.no_x_dim = self.want_no_x_dim()
        self.code_hash: Optional[str] = None
        # Info to enable multiple store_output calls for epilogue subtiling
        self.store_output_ctr = itertools.count()
        self.is_native_matmul = False
        if config.triton.native_matmul:
            for node in self.features.node_schedule:
                if (
                    isinstance(node, scheduler.SchedulerNode)
                    and isinstance(node.node, ir.ComputedBuffer)
                    and node.node.get_reduction_type() == "dot"
                ):
                    self.is_native_matmul = True
                    break

        # define this in a closure to make cache local to object
        @functools.cache
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)

            return self.combine_modular_indexing_pairs(index)

        self.simplify_indexing = simplify_indexing
        self.initialize_range_tree(pid_cache)

        self.rsplit_size = 0
        self.saved_partial_accumulate: list[PartialAccumulate] = []

    def _get_store_output_subgraph_name(self, i: int) -> str:
        return f"<STORE_OUTPUT_{i}>"

    def get_store_output_count(self):
        total = next(self.store_output_ctr)
        self.store_output_ctr = itertools.count(start=total - 1, step=1)
        return total

    @property
    @cache_property_on_self
    def num_reduction_dims(self) -> int:
        return sum(prefix_is_reduction(prefix) for prefix in self.numels)

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        raise NotImplementedError

    def get_index_dtype_as_torch_dtype(self) -> torch.dtype:
        return self.features.select_index_dtype()

    @property
    def index_dtype(self) -> str:
        return self.dtype_to_str(self.get_index_dtype_as_torch_dtype())

    def want_no_x_dim(self) -> bool:
        return False

    def construct_range_trees(
        self,
        pid_cache: Optional[dict[str, str]],
        inside_reduction: bool,
        is_reduction: bool,
        numels: dict[str, sympy.Expr],
        no_x_dim: bool,
    ) -> list[IterationRangesRoot]:
        active_prefixes = OrderedSet(
            prefix for prefix in all_prefixes if prefix in numels
        )
        no_r_dim = not inside_reduction or not is_reduction

        def filtered_index_map(seq, mask) -> dict[Any, int]:
            return {
                val: idx for idx, val in enumerate(val for val in seq if val in mask)
            }

        grid_dims = ["x", "y", "z"]
        pointwise_tensor_dims = list(reversed(grid_dims))
        reduction_dims = ["r0_", "r1_"]
        if no_x_dim:
            tensor_dims = reduction_dims
        elif no_r_dim:
            tensor_dims = pointwise_tensor_dims
        else:
            tensor_dims = pointwise_tensor_dims + reduction_dims

        # Filter out unused tensor dims.
        # Convert to dicts for O(1) index lookup.
        tensor_dim_map = filtered_index_map(tensor_dims, active_prefixes)
        grid_dim_map = filtered_index_map(grid_dims, all_prefixes)

        range_trees = []
        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix_is_reduction(prefix)
            tensor_dim = tensor_dim_map.get(prefix)
            grid_dim = grid_dim_map.get(prefix)
            index = i if grid_dim is None else grid_dim
            range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    numels[prefix],
                    prefix,
                    index,
                    self,  # type: ignore[arg-type]
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim="z" in numels,
                )
            )
        return range_trees

    def initialize_range_tree(self, pid_cache: dict[str, str]) -> None:
        range_trees = self.construct_range_trees(
            pid_cache,
            self.inside_reduction,
            self.features.is_reduction(),
            self.numels,
            self.no_x_dim,
        )
        self.range_trees.extend(range_trees)

    def finalize_indexing(self, indices: Sequence[sympy.Expr]) -> None:
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.
        """

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

    def should_use_cooperative_reduction(self) -> bool:
        return False  # defined in subclass

    def should_use_persistent_reduction(self) -> bool:
        return False  # defined in subclass

    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

    def triton_tensor_ndim(self) -> int:
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def indexing_size_str(self, i: int) -> str:
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

    def dense_size_list(self) -> list[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            # pyrefly: ignore [missing-argument]
            if not tree.is_reduction or self.inside_reduction:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        return sizes

    def create_constant_mask(self, entry) -> str:
        x = entry.prefix
        if entry.tensor_dim is None:
            sizestr = self.dense_size_str()
            return f"{x}mask = tl.full({sizestr}, True, tl.int1)"
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[entry.tensor_dim] = ":"
        suffix = ", ".join(sizes)
        out = f"{x}mask = tl.full([{x.upper()}BLOCK], True, tl.int1)[{suffix}]"
        return out

    def dense_size_str(self) -> str:
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        if not isinstance(index, ModularIndexing):
            return index
        x = index.args[0]
        if (tree_node := self.range_tree_nodes.get(x)) is None:
            return index
        new_index = sympy_subs(index, {x: tree_node.expr})
        new_index = V.graph.sizevars.combine_modular_indexing_pairs(new_index)
        # the index now contains xindex/etc, which is nonstandard, fix it up
        return sympy_subs(
            new_index,
            {
                tree_node.root.index_sym(): tree_node.root.lookup(
                    sympy.S.One, tree_node.root.numel
                ).symbol()
            },
        )

    def combine_contiguous_dims(
        self, index: sympy.Expr, tree: IterationRangesRoot
    ) -> sympy.Expr:
        if expand_res := V.graph.sizevars.expand_floor_div(index):
            new_index, denominator = expand_res  # type: ignore[misc]
            return FloorDiv(self._combine_contiguous_dims(new_index, tree), denominator)
        else:
            return self._combine_contiguous_dims(index, tree)

    def _combine_contiguous_dims(
        self, index: sympy.Expr, tree: IterationRangesRoot
    ) -> sympy.Expr:
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        should_flush = self.range_trees[-1].is_loop or self.cooperative_reduction

        @contextlib.contextmanager
        def ctx():
            if not self.features.is_reduction():
                assert not self.inside_reduction
                yield
                return
            if should_flush:
                # calling codegen_body() will flush all the pending buffers
                # and write out a reduction loop
                self.codegen_body()
            self.inside_reduction = False
            try:
                yield
                if should_flush:
                    # flush out any code before opening the next loop
                    self.codegen_body()
            finally:
                self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths: sympy.Expr) -> list[sympy.Symbol]:
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ) -> tuple[
        list[list[sympy.Expr]], list[list[Callable[[list[sympy.Expr]], sympy.Expr]]]
    ]:
        # Special case: if a node's sizes are ([], []), there's nothing to split.
        if all(len(length) == 0 for length in lengths):
            return [[] for group in groups], []

        sv = V.graph.sizevars
        new_ranges: list[list[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i: int, expr: sympy.Expr) -> int:
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(
            sizes: list[sympy.Expr], idxs: list[int]
        ) -> Callable[[list[sympy.Expr]], sympy.Expr]:
            """
            Builds the nested expression:
              ((...((s1*v[i1] + v[i2]) * s2 + v[i3]) ... ) * sk + v[i(k+1)])
            """
            assert len(idxs) == len(sizes) + 1

            def getter(flat_vars: list[sympy.Expr]) -> sympy.Expr:
                expr = flat_vars[idxs[0]]
                for s, idx in zip(sizes, idxs[1:]):
                    expr = s * expr + flat_vars[idx]
                return expr

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.S.Zero)
                    continue

                while current_group < len(remaining) and sv.statically_known_equals(
                    remaining[current_group],
                    1,  # type: ignore[arg-type]
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                # During native matmul on bmm, we enforce tiling order (z, y, x, r).
                # When fusing a bmm node with loop (z, y, x, r) with a pw node
                # of shape (z*y*x, 1), we need to split the pw iteration range
                # into three dimensions.
                # The group becomes [z, y, x, 1], with lengths ([z*y*x], []).
                # In this case, we decompose the combined size z*y*x into three
                # consecutive groups. Previously, _split_iteration_ranges supported
                # splitting into at most two dimensions, but we now extend it to do
                # three splits when the total size is divisible by all three.

                # is group having (z,y,x,r=1) form?
                is_bmm_then_pw = len(remaining) == 4 and remaining[-1] == 1
                if (
                    current_group + 2 < len(remaining)
                    and sv.statically_known_gt(
                        size, remaining[current_group] * remaining[current_group + 1]
                    )
                    and is_bmm_then_pw
                ):
                    # need to break size in three
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group] * remaining[current_group + 1]
                    ):
                        raise CantSplit

                    size1 = remaining[current_group]
                    size2 = remaining[current_group + 1]
                    size3 = FloorDiv(size, size1 * size2)
                    return_getters.append(
                        make_combined(
                            [size2, size3],
                            [
                                add_range(current_group, size1),
                                add_range(current_group + 1, size2),
                                add_range(current_group + 2, size3),
                            ],
                        )
                    )

                # Two-dimensional tiling
                elif current_group + 1 < len(remaining) and sv.statically_known_gt(
                    size, remaining[current_group]
                ):
                    # need to break size in two
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group]
                    ):
                        raise CantSplit

                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            [size2],
                            [
                                add_range(current_group, size1),
                                add_range(current_group + 1, size2),
                            ],
                        )
                    )
                else:
                    if current_group < len(remaining):
                        return_getters.append(
                            operator.itemgetter(add_range(current_group, size))
                        )
            return_getters_groups.append(return_getters)

        assert all(V.graph.sizevars.size_hint(s) == 1 for s in remaining), (
            f"failed to set ranges {remaining} {lengths}"
        )
        return new_ranges, return_getters_groups

    @classmethod
    def prepare_split_iteration_lengths(
        cls,
        groups: Iterable[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        reduction_numel: sympy.Expr = sympy.S.One,
    ) -> Sequence[Sequence[sympy.Expr]]:
        "Fill in the reduction numel of lengths if missing"
        sizevars = V.graph.sizevars
        if len(lengths[1]) == 0 and (
            not sizevars.statically_known_equals(reduction_numel, sympy.S.One)
            and sizevars.statically_known_equals(
                sympy_product(groups),
                sympy_product(lengths[0]) * reduction_numel,
            )
        ):
            return (lengths[0], [reduction_numel])

        return lengths

    @classmethod
    def is_compatible(
        cls,
        groups: Iterable[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        reduction_numel: sympy.Expr = sympy.S.One,
    ) -> bool:
        lengths = cls.prepare_split_iteration_lengths(groups, lengths, reduction_numel)

        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(
        self, lengths: Sequence[Sequence[sympy.Expr]]
    ) -> list[list[sympy.Expr]]:
        """
        Split and set iteration ranges for the kernel based on the provided lengths.

        This method maps the kernel's tiling structure to the node's iteration space,
        handling both pointwise and reduction dimensions appropriately.

        Args:
            lengths: A sequence of sequences of symbolic expressions representing
                    the sizes of different dimensions for each node.

        Returns:
            A list of lists of symbolic expressions representing the mapped
            iteration variables for each dimension.
        """
        # Create a dictionary mapping each range tree prefix to its total number of elements
        tiling = {rt.prefix: rt.numel for rt in self.range_trees}

        # If we're not inside a reduction loop, set all reduction dimensions to 1
        # This effectively disables reduction dimensions when not needed
        if not self.inside_reduction:
            for prefix in tiling:
                if prefix_is_reduction(prefix):
                    tiling[prefix] = sympy.S.One

        # Extract the values from the tiling dictionary to create groups
        groups = [*tiling.values()]

        # Map the kernel's group structure to the node's sizes and set the ranges
        # using the set_ranges method, returning the resulting iteration variables
        return self.map_kernel_groups_to_node_sizes(groups, lengths, self.set_ranges)

    @classmethod
    def map_kernel_groups_to_node_sizes(
        cls,
        groups: Sequence[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        set_ranges,
    ) -> list[list[sympy.Expr]]:
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        if len(lengths) == len(groups) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return set_ranges(*lengths)

        new_ranges, return_getters_groups = cls._split_iteration_ranges(groups, lengths)
        itervars = [*itertools.chain.from_iterable(set_ranges(*new_ranges))]
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr) -> bool:
        # tmpX  means indirect indexing
        return free_symbol_is_type(index, SymT.TMP)

    def is_broadcasted(self, index: sympy.Expr) -> bool:
        # Note. This may not be correct when there is indirect indexing
        if self.is_indirect_indexing(index):
            return False

        index_numels = [1] * len(self.numels)
        for symbol in index.free_symbols:
            if symbol not in self.range_tree_nodes:
                # Non-iterated variables, e.g. strides
                continue
            entry = self.range_tree_nodes[symbol]  # type: ignore[index]
            assert isinstance(entry.parent, IterationRangesRoot)
            index_numels[entry.parent.index] *= entry.length

        # If the index variables only iterate over a subset of the kernel
        # numels, then it must be broadcasted.
        simplify = V.graph.sizevars.simplify
        return any(
            simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
            for idx_range, iter_range in zip(index_numels, self.numels.values())
        )

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in output code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the generated kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        return self.kexpr(self.rename_indexing(index))  # type: ignore[call-arg]

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ) -> sympy.Expr:
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        # last resort, if no range vars are in the expr, hoist it
        # TODO instead of trying to blindly find complicated exprs, we should hoist the
        # inputs/outputs sizes and strides, but at the time indexing is generated
        # kernel inputs and outputs are not set yet, we'd need a deeper refactor
        # to do it this way

        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                    symbol_is_type(s, (SymT.SIZE, SymT.PRECOMPUTED_SIZE))
                    for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        simp_index = self.simplify_indexing(index)

        # Now that we are done simplifying we can unwrap Identity so that downstream handling
        # for its contained expression will work. previously, tl.full wrapping of sympy.Integer
        # would not occur
        simp_index = (
            simp_index if not isinstance(simp_index, Identity) else simp_index.args[0]
        )

        return self.codegen_indexing(simp_index)

    def active_range_trees(self) -> list[IterationRangesRoot]:
        return [
            t
            for t in self.range_trees
            # pyrefly: ignore [missing-argument]
            if not t.is_reduction or self.inside_reduction
        ]

    def codegen_indexing(self, expr: sympy.Expr) -> sympy.Expr:
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # if indexing expression is complicated, we precompute it on the host side
                # and send the result as a kernel argument
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr,
                        replacements,  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr

    def codegen_nan_check(self) -> None:
        raise NotImplementedError("NYI: codegen_nan_check")

    def deallocate_workspaces(self):
        wrapper = V.graph.wrapper_code
        for ws in reversed(self.args.workspace_args):
            wrapper.generate_workspace_deallocation(ws)

    def call_kernel(
        self, name: str, node: Optional[IRNode] = None, deallocate_ws: bool = True
    ) -> None:
        raise NotImplementedError("NYI: call_kernel")

    @contextlib.contextmanager
    def mask_loads(
        self, mask: Union[str, OpsWrapper], value: Union[int, float]
    ) -> Iterator[str]:
        """Context manager to add an additional mask to tl.load/store"""
        prior = self._load_mask
        prior_val = self._load_other
        if prior:
            mask = ops.logical_and(mask, prior)

        mask = OpsWrapper._unwrap(mask)
        self._load_mask = mask
        self._load_other = value
        try:
            # TODO(jansel): do we need a reshape here?
            yield mask
        finally:
            self._load_mask = prior
            self._load_other = prior_val

    def get_strides_of_load(self, index: sympy.Expr) -> dict[sympy.Symbol, sympy.Expr]:
        """
        This gets the stride of the index for each of the tiling variables
        (technically, it does it at index 0)

        For example, if
        xindex = x0 + 512*x1 + 1024*r0
        x0 = (xindex//512)
        x1 = (xindex % 512)
        r0 = rindex // 1024

        this function would return
        {xindex: 512, rindex: 1024}
        """
        index_to_tile_indexes = {k: v.expr for k, v in self.range_tree_nodes.items()}
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)  # type: ignore[arg-type]
        strides = {}
        for range_tree in self.range_trees:
            s = sympy_index_symbol(range_tree.name)
            strides[s] = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(
                index_in_tile_vars, {s: 0}
            )
        return strides

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)

    def estimate_flops(self) -> Optional[int]:
        flops = [
            node.estimate_flops()
            for node in NodeScheduleMarker.only_nodes(self.features.node_schedule)
        ]
        return sum(filter(None, flops))

    def estimate_kernel_num_bytes(self):
        """
        Try the best to estimate the total size (in bytes) of the
        kernel's inputs and outputs, which is used for estimating the memory
        throughput of this kernel. This information is used for checking how
        far we are from the peak memory bandwidth. It's important that
        we want to avoid overestimating the sizes of the inputs and outputs,
        because it can wrongfully give us a very large memory traffic value,
        which may be even larger than the theoretical bandwidth and thus
        become very misleading. This is particularly problematic for cases
        where we slice some inputs. In those cases, we should only count
        the size of the "slices" instead of the original inputs, because
        only the slices contribute to the real memory traffic.
        """
        nbytes = []
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        _, call_args, _, _ = self.args.python_argdefs()
        buf_accesses = self.features.buf_accesses()

        # For pointwise and reduction kernels, this is the upper-bound numels
        # for the output buffer.
        # FIXME: This is not exactly right for cases like below:
        #    def foo(tensor0, tensor1):
        #        x0 = narrow(tensor0)
        #        return cat(x0, tensor1)
        # For this example, we will end up overestimate the size for the
        # slice s0. Potentially, we could have precise inputs information
        # if we maintained the original inputs of the Pointwise kernel created
        # for the "cat". However, I think it might be a bit overwhelming that
        # we add such complexity only for handling some particular cases for
        # benchmarking.
        out_numel = V.graph.sizevars.size_hint(
            sympy_product(self.numels.values()),
            fallback=config.unbacked_symint_fallback,
        )
        for i, arg in enumerate(call_args):
            # "buf" may be narrowed. In this case, the number of memory accesses
            # should be estimated based on the reinterpreted layout.
            # On the other hand, buf may be broadcasted. In this case,
            # counting the size of the underline storage would give us
            # a better estimation in terms of memory accesses.
            if arg not in buf_accesses:
                nbytes.append(0)
                continue
            arg_numel = V.graph.get_numel(arg)
            buf_size = V.graph.sizevars.size_hint(
                arg_numel, fallback=config.unbacked_symint_fallback
            )
            if buf_size > out_numel:
                # This arg points to a buf that has been sliced.
                # We need to count each individual slice to have
                # a better estimation.
                indices = OrderedSet[Any]()
                no_index_dep_count = 0
                for dep in buf_accesses[arg]:
                    if isinstance(dep, (StarDep, WeakDep)):
                        indices.add(f"no_index_dep_{no_index_dep_count}")
                        no_index_dep_count += 1
                    else:
                        indices.add(dep.index)
                numel = len(indices) * out_numel
            else:
                numel = buf_size
            dtype = V.graph.get_dtype(arg)
            dtype_size = get_dtype_size(dtype)
            # pyrefly: ignore [bad-argument-type]
            nbytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(nbytes)

    def warn_mix_layout(self, kernel_name):
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
        if (
            len(self.args.input_buffers) == 1
            and len(self.args.output_buffers) == 1
            and len(self.args.inplace_buffers) == 0
        ):
            # even if input buffer and output buffer have different layout,
            # this can be a layout conversion kernel. No need to warn for
            # the mix layouts.
            return

        argdefs, call_args, _signature, _ = self.args.python_argdefs()
        uniform_stride_order = None
        # pyrefly: ignore [bad-assignment]
        for arg_name in call_args:
            buf = V.graph.try_get_buffer(arg_name)
            if not buf:
                continue
            layout = buf.get_layout()
            if len(layout.size) == 4:
                # ignore the tensor if only 1 dimension is non-zero
                if len([x for x in layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(
                        f"Expected stride order {uniform_stride_order}, but found stride order"
                        + f" {stride_order} for kernel {kernel_name}"
                    )
                    log.warning(msg)

                    stride_order_list = [
                        ir.get_stride_order(
                            V.graph.get_buffer(name).get_layout().stride
                        )
                        if V.graph.try_get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    size_list = [
                        V.graph.get_buffer(name).get_layout().size
                        if V.graph.try_get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    source_list = [
                        "GraphInput"
                        if name in V.graph.graph_inputs
                        else "IntermediateBuffer"
                        if name in V.graph.name_to_buffer
                        else None
                        for name in call_args
                    ]

                    argdef_names = [x.name for x in argdefs]
                    msg = yellow_text(
                        f"  param names {argdef_names}\n  buf names {call_args}\n  strides {stride_order_list}"
                        + f"\n  sizes {size_list}\n  sources {source_list}\n"
                    )
                    log.warning(msg)
                    return
        msg = green_text(
            f"All the inputs for the triton kernel {kernel_name} have uniform layout"
        )
        log.warning(msg)

    def welford_reduce_fallback(self, dtype, value):
        sum_ = ops.reduction(dtype, dtype, "sum", value)
        self.inside_reduction = False
        rnumel = ops.index_expr(self.features.reduction_numel, dtype)
        mean = ops.truediv(sum_, rnumel)

        self.inside_reduction = True
        dx = ops.sub(value, mean)
        dx2 = ops.mul(dx, dx)
        m2 = ops.reduction(dtype, dtype, "sum", dx2)
        return OpsWrapper._unwrap((mean, m2, rnumel))

    def prepare_softmax_twopass_fallback(self, dtype, value):
        vmax = ops.reduction(dtype, dtype, "max", value)
        sub = ops.sub(value, vmax)
        exp = ops.exp(sub)
        vsum = ops.reduction(dtype, dtype, "sum", exp)
        return OpsWrapper._unwrap((vmax, vsum))

    def codegen_kernel(self):
        raise NotImplementedError

    def codegen_body(self):
        pass

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        pass


class SIMDScheduling(BaseScheduling):
    """
    Single Instruction Multiple Data parent class used for fusion across
    multiple different backends.
    """

    kernel_type: type[Any] = SIMDKernel  # override in subclass

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
            node2, scheduler.ForeachKernelSchedulerNode
        ):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group
        why = WhyNoFuse(node1, node2)

        if node1.is_split_scan() and not node2.is_split_scan():
            if node2.is_reduction():
                why("Split scan cannot fuse with reductions")
        elif node2.is_split_scan() and not node1.is_split_scan():
            if node1.is_reduction():
                why("Split scan cannot fuse with reductions")

        if node1.is_reduction() and node2.is_reduction():
            reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
            if not reduction_can_fuse:
                from torch._inductor.scheduler import MixOrderReduction

                reduction_can_fuse = MixOrderReduction.can_fuse(node1, node2)

            if
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `simd.py_docs.md_docs.md`
- **Keyword Index**: `simd.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
