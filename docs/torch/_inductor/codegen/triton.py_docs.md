# Documentation: `torch/_inductor/codegen/triton.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/triton.py`
- **Size**: 239,932 bytes (234.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

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
import os
import textwrap
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import Any, cast, Optional, TYPE_CHECKING, Union

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
import torch._logging
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity, preserve_rng_state
from torch._prims_common import is_integer_dtype
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._triton import has_triton_package, has_triton_stable_tma_api

from ...utils._sympy.symbol import free_symbol_is_type, prefix_str, symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir, metrics
from ..async_compile import AsyncCompile
from ..codecache import code_hash, get_path, PyCodeCache, write_atomic
from ..debug import set_kernel_post_grad_provenance_tracing
from ..ops_handler import DefaultHandler
from ..runtime import triton_heuristics
from ..runtime.benchmarking import benchmarker
from ..runtime.hints import (
    AutotuneHint,
    DeviceProperties,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from ..runtime.runtime_utils import get_max_y_grid, next_power_of_2
from ..scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from ..shape_propagation import get_broadcasted_shape
from ..utils import (
    cache_on_self,
    DelayMaybeLine,
    DelayReplaceLine,
    get_bounds_index_expr,
    get_fused_kernel_name,
    get_kernel_metadata,
    is_welford_reduction,
    Placeholder,
    prefix_is_reduction,
    sympy_dot,
    sympy_product,
    sympy_subs,
    triton_type,
    triton_version_uses_attrs_dict,
    upcast_compute_type,
)
from ..virtualized import _ops as ops, ReductionType, StoreMode, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .block_analysis import BlockPatternMatcher
from .common import (
    ArgName,
    BackendFeature,
    ConstexprArg,
    CSE,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    InplacedBuffer,
    is_buffer_removed,
    OpOverrides,
    PythonPrinter,
    RemovedArg,
    SizeArg,
    TensorArg,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .simd import (
    constant_repr,
    IterationRanges,
    IterationRangesEntry,
    IterationRangesRoot,
    PartialAccumulate,
    SIMDKernel,
    SIMDScheduling,
)
from .triton_utils import (
    config_of,
    equal_1_arg_indices,
    non_constexpr_signature,
    should_unwrap_unspec_arg,
    signature_to_meta,
)
from .wrapper import SymbolicCallArg


if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeVar

    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler

    from ..ir import IRNode
    from .common import BlockShapeType
    from .simd_kernel_features import SIMDKernelFeatures

    _T = TypeVar("_T")

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
async_compile = AsyncCompile()


def get_triton_reduction_function(reduction_type):
    use_helper = reduction_type in ("any", "max", "min", "prod")
    module = "triton_helpers" if use_helper else "tl"
    if reduction_type in ("max", "min"):
        return f"{module}.{reduction_type}2"
    else:
        return f"{module}.{reduction_type}"


def is_sympy_integer_like(expr: object):
    """ "
    Is this expression a Sympy Integer or is it an integer sympy Expr
    containing no free symbols. The latter case can happen with Identity expr.
    """
    if not isinstance(expr, sympy.Expr):
        return False
    return isinstance(expr, sympy.Integer) or (
        expr.is_integer and len(expr.free_symbols) == 0
    )


class OpDtypeSupport:
    """
    Some Triton ops such as libdevice and tl.math only support float32 and float64.
    This class records which dtypes are supported by specific IR ops.
    """

    supported_dtypes: dict[str, OrderedSet[torch.dtype]] = {}
    convert_outputs: dict[str, bool] = {}

    @classmethod
    def register_upcast(cls, func: Callable[..., str], convert_output: bool) -> None:
        op_name = func.__name__
        cls.supported_dtypes[op_name] = OrderedSet([torch.float32, torch.float64])
        cls.convert_outputs[op_name] = convert_output


@lru_cache(None)
def gen_attr_descriptor_import() -> str:
    """
    import AttrsDescriptor if the triton version is new enough to have this
    class defined.
    """
    if not has_triton_package():
        return ""

    import triton.compiler.compiler

    # Note: this works because triton.compiler.compiler imports AttrsDescriptor from triton.backends.compiler
    # When support for the legacy AttrsDescriptor is removed then this import path should be changed.
    if hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        return "from triton.compiler.compiler import AttrsDescriptor"
    else:
        return ""


@lru_cache(None)
def gen_common_triton_imports() -> str:
    imports = IndentedBuffer()
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)

    imports.splice(
        """
        from torch._inductor.runtime import triton_helpers, triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
        """
    )
    return imports.getvalue()


class TritonSymbols:
    """
    Stores sympy.Symbol instances and constants associated with triton codegen.
    """

    reduction_types = OrderedSet([SymT.R0_INDEX, SymT.R1_INDEX])
    block_types = OrderedSet([SymT.XBLOCK, SymT.YBLOCK, SymT.ZBLOCK, *reduction_types])

    block_offsets = {
        symt: sympy.Symbol(f"{prefix_str[symt]}offset", integer=True, nonnegative=True)
        for symt in block_types
    }

    block_sizes = {
        symt: sympy.Symbol(
            f"{prefix_str[symt].upper()}BLOCK", integer=True, positive=True
        )
        for symt in block_types
    }

    @classmethod
    def get_block_shape(cls, expr: sympy.Expr) -> BlockShapeType:
        # return block shape of sympy Expression
        # e.g.,
        # tmp13 = y1
        # tmp14 = x0 - tmp13
        #
        # get_block_shape(y1) = (YBLOCK,1,1)
        # get_block_shape(x0-tmp13) = (YBLOCK,XBLOCK,1)

        expr_shape: BlockShapeType = ()
        expr_vars = expr.free_symbols
        for var in expr_vars:
            if symbol_is_type(var, SymT.TMP):
                cse_var = V.kernel.cse.varname_map[var.name]
                var_shape = cse_var.shape
            elif symbol_is_type(
                var,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.INDEX,
                    SymT.FLOAT,
                    SymT.UNBACKED_FLOAT,
                ),
            ):
                var_shape = ()
            else:
                symbol_matches = [
                    symt for symt in cls.block_types if symbol_is_type(var, symt)
                ]
                assert len(symbol_matches) == 1, f"Ambiguous type: {var.name}"

                sym = symbol_matches[0]
                ndim = V.kernel.triton_tensor_ndim()
                shape = ["1"] * ndim

                tree_match = [
                    tree
                    for tree in V.kernel.active_range_trees()
                    if prefix_str[sym] == tree.prefix
                ]
                assert len(tree_match) == 1, "# of Match expected to 1"

                shape[tree_match[0].tensor_dim] = str(cls.get_block_size(tree_match[0]))
                var_shape = tuple(shape)

            # Union current variable shape
            expr_shape = get_broadcasted_shape(expr_shape, var_shape)

        assert expr_shape is not None

        # Below logic handles when index symbols does not match with convention range tree order.
        # Mainly, it is for TMA template where TMA indices are expected to be in (x,y), not (y,x).
        # so in such case, the get_block_shape(yindex) should be (1,YBLOCK), not (YBLOCK,1).
        if isinstance(V.kernel, torch._inductor.select_algorithm.TritonTemplateKernel):
            out_shape = V.kernel.template_out_shape
            if out_shape == ("XBLOCK", "YBLOCK") and V.kernel.tma_store:
                expr_shape = (expr_shape[1], expr_shape[0], *expr_shape[2:])

        return expr_shape

    @classmethod
    def get_block_size(cls, tree: IterationRanges) -> sympy.Symbol:
        return cls.block_sizes[tree.symt]

    @classmethod
    def get_block_offset(cls, tree: IterationRanges) -> sympy.Symbol:
        return cls.block_offsets[tree.symt]


@dataclasses.dataclass
class IndexingOptions:
    index_str: str
    mask_vars: OrderedSet[str]
    expand_str: Optional[str]
    _has_rindex: bool
    index: sympy.Expr
    expand_shape: Optional[Sequence[Union[int, str]]]

    def has_mask(self) -> bool:
        return bool(self.mask_vars)

    def has_indirect(self) -> bool:
        return free_symbol_is_type(self.index, SymT.TMP)

    def has_rindex(self) -> bool:
        return self._has_rindex

    def has_tmpmask(self) -> bool:
        return any(str(mask).startswith("tmp") for mask in self.mask_vars)

    def has_rmask(self) -> bool:
        return any(str(mask).startswith("r") for mask in self.mask_vars)

    @property
    def mask_str(self) -> str:
        # The sorted call is added to make sure the order is still
        # deterministic if self.mask_vars contains mix of string
        # and TritonCSEVariable
        return (
            " & ".join(sorted(map(str, self.mask_vars))) if self.mask_vars else "None"
        )


@dataclasses.dataclass
class BlockDescriptorOptions:
    """
    This is a base class that describes a block descriptor used in Triton kernels.
    It can be used to create either a tensor descriptor (with TensorDescriptorOptions)
    or a block pointer (with BlockPtrOptions).
    """

    params: BlockParameters
    constant_offset: sympy.Expr
    order: list[int]
    mask_vars: OrderedSet[str]
    broadcast_shape: Sequence[sympy.Expr]
    broadcasting_dims: list[bool]
    final_shape: Sequence[sympy.Expr]
    _boundary_check: Optional[list[int]] = None
    # Can we safely lift the constructor
    # to the top of the kernel?
    can_lift: bool = False

    @property
    def shape(self) -> list[sympy.Expr]:
        return self.params.shape

    @property
    def block_shape(self) -> list[sympy.Expr]:
        return self.params.block_shape

    @property
    def strides(self) -> list[sympy.Expr]:
        return self.params.strides

    @property
    def offsets(self) -> list[sympy.Expr]:
        return self.params.offsets

    @classmethod
    def create(
        cls,
        *,
        params: BlockParameters,
        constant_offset: sympy.Expr,
        range_trees: list[IterationRangesRoot],
        mask_vars: OrderedSet[str],
        get_max_block: Callable[[str], int],
        can_lift=False,
        transpose_contiguous=False,
    ) -> BlockDescriptorOptions:
        """Helper to create a BlockDescriptorOptions instance"""

        sizevars = V.graph.sizevars

        def lookup_size(exprs: Iterable[sympy.Expr]) -> list[sympy.Expr]:
            return [sizevars.lookup_precomputed_size(expr) for expr in exprs]

        # Look up precomputed sizes
        params.shape = lookup_size(params.shape)
        params.strides = lookup_size(params.strides)

        # Strip out dimensions of stride 0.
        # These will be restored with tl.broadcast_to.
        broadcasting_dims = [
            sizevars.statically_known_equals(stride, 0) for stride in params.strides
        ]

        # Strip out dimensions of size 1.
        # These will be restored by tl.reshape.
        singleton_dims = [
            sizevars.statically_known_equals(dim, 1) for dim in params.block_shape
        ]
        if all(singleton_dims):
            # Handle a pure singletons, e.g. [1, 1]
            singleton_dims[-1] = False

        # Record the post-broadcast shape before broadcasting dims are removed.
        # The pre-broadcast shape is identical to this, except broadcasting dims are
        # replaced with 1.
        broadcast_shape = [
            dim
            for dim, is_singleton in zip(params.block_shape, singleton_dims)
            if not is_singleton
        ]

        # Combine all removable dims.
        removable_dims = [any(dims) for dims in zip(singleton_dims, broadcasting_dims)]

        # Remove singleton_dims from broadcasting_dims so that
        # broadcast_shape and broadcasting_dims have the same length
        broadcasting_dims = [
            dim
            for dim, is_singleton in zip(broadcasting_dims, singleton_dims)
            if not is_singleton
        ]

        def remove_dims(it):
            """Removes any broadcasting or singleton dims from a given sequence"""
            return [
                item
                for item, is_removable in zip(it, removable_dims)
                if not is_removable
            ]

        # Drop removable dimensions from the input.
        params = BlockParameters(
            **{
                key: remove_dims(val) for key, val in dataclasses.asdict(params).items()
            },
        )
        # TODO: Generalize to ND tensors.
        transpose = transpose_contiguous and params.strides[-1] != 1
        if transpose:
            params = params.transpose()

        # Compute the final shape, adjusting for special kernel types.
        final_shape = [TritonSymbols.get_block_size(tree) for tree in range_trees]
        if V.kernel.no_x_dim:
            assert range_trees[0].prefix == "x"
            final_shape.pop(0)

        # Check for when BlockParams have been transposed.
        order = list(reversed(range(len(params.shape))))
        if transpose:
            final_shape.reverse()
            order.reverse()

        reduction_ndim = V.kernel.num_reduction_dims
        if (
            not V.kernel.inside_reduction
            and len(params.strides) == len(V.kernel.numels) - reduction_ndim
            and V.kernel.features.is_reduction()
        ):
            # Need to expand rank to match the rank used inside the reduction loop
            final_shape += [sympy.S.One] * reduction_ndim

        result = cls(
            params=params,
            constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
            order=order,
            mask_vars=mask_vars,
            final_shape=final_shape,
            broadcast_shape=broadcast_shape,
            broadcasting_dims=broadcasting_dims,
            can_lift=can_lift,
        )
        result.compute_boundary_check(get_max_block, range_trees)
        return result

    def replace_offset(
        self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT
    ) -> sympy.Expr:
        """
        Replaces instances of {symt}_offset with the new expression.
        """
        roffset = TritonSymbols.block_offsets[symt]
        return sympy_subs(expr, {roffset: replacement})

    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr:
        for symt in TritonSymbols.reduction_types:
            expr = self.replace_offset(expr, sympy.Integer(0), symt)
        return expr

    def compute_boundary_check(
        self,
        get_max_block: Callable[[str], int],
        range_trees: list[IterationRangesRoot],
    ) -> None:
        """List of indices to pass to tl.load(boundary_check=...)"""
        sizevars = V.graph.sizevars

        # Substitute maximum block sizes in shape expressions.
        # This works in multiple_of checks because block sizes are powers of 2.
        block_to_max: dict[sympy.Expr, Any] = {
            TritonSymbols.block_sizes[t.symt]: get_max_block(prefix_str[t.symt])
            for t in range_trees
        }

        # Also see Note: Constant mask optimisation
        # if ynumel / YBLOCK > max_ygrid, then the z dimension is used to handle
        # the remaining programs that cannot fit into the y dimension. This means
        # it's possible that more than the required number of programs are launched,
        # possibly leading to out-of-bounds accesses. So even if ynumel divides YBLOCK,
        # boundary checking is required in the dimensions that are based on YBLOCK
        # e.g. for [YBLOCK // 16, YBLOCK, XBLOCK] dimensions 0 and 1 need boundary
        # checks when max_ygrid is exceeded.
        needs_overflow_grid = any(map(V.kernel.needs_yz_grid_overflow, range_trees))
        self._boundary_check = [
            idx
            for idx in range(len(self.shape))
            if (
                not sizevars.statically_known_equals(self.strides[idx], sympy.S.Zero)
                and (
                    (
                        needs_overflow_grid
                        and TritonSymbols.block_sizes[SymT.YBLOCK]
                        in self.block_shape[idx].free_symbols
                    )
                    or (
                        not sizevars.statically_known_multiple_of(
                            self.shape[idx], self.block_shape[idx]
                        )
                        and not sizevars.statically_known_multiple_of(
                            self.shape[idx],
                            sympy_subs(self.block_shape[idx], block_to_max),
                        )
                    )
                )
                and not (
                    V.kernel.no_x_dim
                    and self.block_shape[idx] == TritonSymbols.block_sizes[SymT.XBLOCK]
                )
            )
        ]

    def boundary_check(self) -> list[int]:
        assert self._boundary_check is not None
        return self._boundary_check

    def has_indirect(self) -> bool:
        return False  # block_ptr can't do indirect indexing

    def has_rindex(self) -> bool:
        return any(
            free_symbol_is_type(expr, TritonSymbols.reduction_types)
            for expr in self.block_shape
        )

    def has_rmask(self) -> bool:
        return self.has_rindex()

    def has_tmpmask(self) -> bool:
        return False  # block_ptr can't do indirect indexing

    def has_mask(self) -> bool:
        return bool(self.boundary_check())

    def codegen_broadcast_and_reshape(
        self,
        value: str,
        initial_shape: Sequence[sympy.Expr],
        final_shape: Sequence[sympy.Expr],
        allow_implicit: bool,
    ) -> str:
        """
        Generate a broadcast and a reshape for the block descriptor.
        This restores stride-0 dimensions which were removed from the block descriptor.
        """

        # Reshape to add singletons.
        pre_broadcast_shape = [
            sympy.S.One if is_broadcasting else dim
            for dim, is_broadcasting in zip(
                self.broadcast_shape, self.broadcasting_dims
            )
        ]
        value = triton_reshape(value, initial_shape, pre_broadcast_shape)

        # Broadcast singletons.
        # For loads, we can often implicitly broadcast singleton dimensions.
        # We need an explicit broadcast for stores, or if the final reshape does more
        # than add singletons.
        sizevars = V.graph.sizevars
        supports_implicit_broadcast = allow_implicit and (
            len(pre_broadcast_shape) == len(final_shape)
            and all(
                sizevars.statically_known_equals(pre_dim, 1)
                or sizevars.statically_known_equals(pre_dim, post_dim)
                for pre_dim, post_dim in zip(pre_broadcast_shape, final_shape)
            )
        )

        if any(self.broadcasting_dims) and not supports_implicit_broadcast:
            value = f"tl.broadcast_to({value}, {V.kernel.index_to_str(self.broadcast_shape)})"

        # Reshape to the final shape.
        value = triton_reshape(value, self.broadcast_shape, final_shape)

        return value


@dataclasses.dataclass
class TensorDescriptorOptions(BlockDescriptorOptions):
    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_tensor_descriptor()

        Args:
            name: variable name for pointer
            roffset: unused, but kept for compatibility with BlockPtrOptions.format()

        Returns:
            "tl.make_tensor_descriptor(...)"
        """

        f = V.kernel.index_to_str
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
        ]

        return f"tl.make_tensor_descriptor({', '.join(args)})"


@dataclasses.dataclass
class BlockPtrOptions(BlockDescriptorOptions):
    def replace_offset(
        self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT
    ) -> sympy.Expr:
        """
        Replaces instances of {symt}_offset with the new expression.
        """
        roffset = TritonSymbols.block_offsets[symt]
        return sympy_subs(expr, {roffset: replacement})

    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr:
        for symt in TritonSymbols.reduction_types:
            expr = self.replace_offset(expr, sympy.Integer(0), symt)
        return expr

    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_block_ptr()

        Args:
            name: variable name for pointer
            roffset: should rn_offset be included in offsets=..., for use with tl.advance()

        Returns:
            "tl.make_block_ptr(...)"
        """
        f = V.kernel.index_to_str
        offsets = [*self.offsets]
        if not roffset:
            offsets = [self.remove_roffsets(offset) for offset in offsets]
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
            f"order={f(self.order)}",
            f"offsets={f(offsets)}",
        ]
        return f"tl.make_block_ptr({', '.join(args)})"

    def advance_roffset(self, symt: SymT) -> sympy.Expr:
        """
        Codegen string to pass to tl.advance(name, ...).

        Advance is the difference between offsets in each loop iteration.
        To compute it, we replace rN_offset with multiples of RN_BLOCK.
        Since we expect rN_offset to vary in range(0, rN_numel, RN_BLOCK), the first
        iteration has rN_offset=0, while the second has rN_offset=RN_BLOCK.
        """
        rblock = TritonSymbols.block_sizes[symt]
        advance = [
            (
                self.replace_offset(offset, rblock, symt)
                - self.replace_offset(offset, sympy.S.Zero, symt)
            )
            for offset in self.offsets
        ]
        return advance


def triton_reshape(
    value: str, old_shape: Sequence[sympy.Expr], new_shape: Sequence[sympy.Expr]
) -> str:
    """Workaround https://github.com/triton-lang/triton/issues/2836"""
    assert isinstance(old_shape, list) and isinstance(new_shape, list)

    old_shape_str = [V.kernel.index_to_str(shape) for shape in old_shape]
    new_shape_str = [V.kernel.index_to_str(shape) for shape in new_shape]

    if old_shape_str == new_shape_str:
        return value
    if [s for s in new_shape_str if s != "1"] != old_shape_str:
        return f"tl.reshape({value}, [{', '.join(new_shape_str)}])"
    # rewrite to [:, None] syntax, which is less buggy
    idx = 0
    expand = []
    for size in new_shape_str:
        if idx < len(old_shape_str) and size == old_shape_str[idx]:
            expand.append(":")
            idx += 1
        else:
            assert size == "1"
            expand.append("None")
    assert idx == len(old_shape_str)
    return f"{value}[{', '.join(expand)}]"


def enable_pdl_codegen():
    if not torch._inductor.config.triton.enable_pdl:
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major >= 9


# NB: Inheriting from PythonPrinter is somewhat dangerous, because there are a
# number of operators which Triton "implements", but in a way that is
# inconsistent with Python semantics (and consistent with C semantics).  We
# must override all of these, or it is potential silent correctness problem
class TritonPrinter(PythonPrinter):
    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.trunc({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr.is_integer:
            # sympy considers 0.0 to be integer, but triton doesn't.
            # this workaround prints the float as an integer
            # xref: https://github.com/sympy/sympy/issues/26620
            ret = str(int(expr))
        elif config.is_fbcode() and torch.version.hip:
            ret = f"{expr}"
        else:
            ret = f"tl.full([], {expr}, tl.float64)"
        return ret

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
        return f"{s}.to(tl.float64)"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.remainder_integer({quot_s}, {div_s})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        assert expr.is_integer
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " // ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.div_floor_integer({quot_s},  {div_s})"

    # TODO: This is wrong, when lhs, rhs > 2**53, Python does a higher
    # precision algorithm, which we would need to replicate here
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    # NB: sympy.floor/ceiling produce integers, so we have to do the
    # conversion to index dtype
    def _print_floor(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        # work around for https://github.com/pytorch/pytorch/issues/165738
        if torch.xpu.is_available():
            return f"libdevice.sqrt(({self._print(expr)}).to(tl.float32))"
        return f"tl.sqrt_rn(({self._print(expr)}).to(tl.float32))"

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        return (
            f"libdevice.pow({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        )

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        if expr.args[0].is_Integer:
            return f"libdevice.pow({float(expr.args[0])}, {self._print(expr.args[1])})"
        return (
            f"libdevice.pow({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        )

    def _print_Where(self, expr: sympy.Expr) -> str:
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"tl.where({c}, {p}, {q})"

    def _print_min_max_helper(self, expr: sympy.Expr, cmp: str) -> str:
        """
        Helper for max/min code generation.
        cmp: > or <
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        cls = type(expr)
        a = self._print(cls(*expr.args[:mid]))
        b = self._print(cls(*expr.args[mid:]))

        # Use a macro so we can propagate constexprs.
        # https://github.com/triton-lang/triton/issues/3815
        a, b = tuple(f"({x})" for x in (a, b))
        assert cmp in (">", "<"), f"Unexpected comparator: '{cmp}'"
        return f"({a} * ({a} {cmp}= {b}) + {b} * ({b} {cmp} {a}))"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, "<")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, ">")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"tl_math.abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.log2(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.llrint({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )

        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"libdevice.nearbyint(1e{ndigits} * {number_str}) * 1e{-ndigits}"


texpr = TritonPrinter().doprint


def triton_compute_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type and upcast [b]float16 to float32"""
    return triton_type(upcast_compute_type(dtype))


def triton_store_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with fix for storing tl.bool"""
    if dtype == torch.bool:
        dtype = torch.int8
    return triton_type(dtype)


def upcast_acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Implicit upcasts used for Triton reduction types"""
    if is_integer_dtype(dtype) and dtype.is_signed and dtype.itemsize <= 4:
        return torch.int32
    return upcast_compute_type(dtype)


def triton_acc_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with reduction upcasts"""
    return triton_compute_type(upcast_acc_dtype(dtype))


def low_precision_fp(dtype: torch.dtype) -> bool:
    return dtype.itemsize <= 2 and dtype.is_floating_point


def low_precision_fp_var(var: Union[CSEVariable, Any]) -> bool:
    if not isinstance(var, CSEVariable):
        return False

    dtype = var.dtype
    return low_precision_fp(dtype) if isinstance(dtype, torch.dtype) else False


class TritonCSEVariable(CSEVariable):
    def __init__(
        self,
        name: str,
        bounds: ValueRanges[Any],
        dtype: torch.dtype,
        shape: BlockShapeType = None,
    ) -> None:
        super().__init__(name, bounds, dtype, shape=shape)
        # We'll use this to track which masks the variable needs when used for indirect indexing
        self.mask_vars: OrderedSet[str] = OrderedSet()
        assert dtype is not None, "TritonCSEVariable must have dtype"
        # TODO: uncomment this and fix the few failures left
        # assert shape is not None, "TritonCSEVariable must have shape"

    def update_on_args(self, name, args, kwargs):
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol):
                # most of the time index vars don't need masks associated with them
                # however, when index vars are used to compute indices for indirect reads
                # those reads should subsequently be masked,
                for symt in TritonSymbols.block_types:
                    if symbol_is_type(arg, symt):
                        self.mask_vars.update([f"{prefix_str[symt]}mask"])
                        break


def get_dtype_handler() -> DtypePropagationOpsHandler:
    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler

    return DtypePropagationOpsHandler()


def maybe_upcast_float32(convert_output: bool = True) -> Callable[[_T], _T]:
    """
    Codegen helper to upcast arguments to float32, depending on the config and dtype.
    This decorates tl.math/libdevice codegen functions.
    """

    def needs_upcast(var) -> bool:
        return (
            not config.triton.codegen_upcast_to_fp32
            and isinstance(var, CSEVariable)
            and var.dtype in (torch.float16, torch.bfloat16)
        )

    def maybe_upcast_arg(var) -> str:
        upcast_string = ".to(tl.float32)" if needs_upcast(var) else ""
        return f"{var}{upcast_string}"

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Record that this function only supports float32 and float64.
        OpDtypeSupport.register_upcast(func, convert_output)

        def wrapped(*args, **kwargs) -> str:
            # Optionally upcast args to float32.
            upcast_args = [maybe_upcast_arg(arg) for arg in args]
            upcast_kwargs = {key: maybe_upcast_arg(val) for key, val in kwargs.items()}

            # Call the decorated function, optionally downcasting the result.
            result = func(*upcast_args, **upcast_kwargs)
            any_needs_upcast = convert_output and any(
                needs_upcast(var) for var in itertools.chain(args, kwargs.values())
            )
            result_dtype = (
                None
                if not any_needs_upcast
                else getattr(get_dtype_handler(), func.__name__)(*args, **kwargs)
            )
            needs_downcast = result_dtype not in (torch.float32, None)
            downcast_string = (
                f".to({triton_type(result_dtype)})"
                if needs_downcast and result_dtype is not None
                else ""
            )
            return f"{result}{downcast_string}"

        return wrapped

    return decorator  # type: ignore[return-value]


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton e.g., ops.to_dtype(x,...) -> x.to(...)"""

    _LOG_2_E = math.log2(math.e)

    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types=True,
    ):
        def _get_min_elements_per_thread(
            src_dtype: torch.dtype, dst_dtype: torch.dtype
        ) -> int:
            if src_dtype == dst_dtype:
                # No data type conversion is needed. No requirements on min_elem_per_thread.
                return 0

            # fp8 data type conversions has min_elem_per_thread requirements.
            # Refer to Triton implementations here:
            # https://github.com/triton-lang/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10.
            fp8_dtypes = (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            # Triton doesn't support type conversions between fp8_e4m3 and fp8_e5m2.
            assert not (
                src_dtype in fp8_dtypes
                and dst_dtype in fp8_dtypes
                and src_dtype != dst_dtype
            ), "Conversions between float8_e5m2 and float8_e4m3fn is not supported!"
            if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
                return 4
            if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
                return 2
            # No requirements on min_elem_per_thread.
            return 0

        if src_dtype is not None:
            # Both dtype and src_dtype are set. This is used by torch to(dtype=dtype).
            # It takes the maximum min_elem_per_thread if there are multiple fp8 conversions
            # in the same kernel.
            V.kernel.min_elem_per_thread = max(
                _get_min_elements_per_thread(src_dtype, dtype),
                V.kernel.min_elem_per_thread,
            )

        if dtype == torch.bool:
            return f"({x} != 0)"
        elif dtype == torch.uint8 and (
            src_dtype is not None and src_dtype.is_floating_point or src_dtype is None
        ):
            # to work around llvm uint conversion semantics that produces 0's for negative
            # values when converting from floating types.
            # optimization - if source type is known and it's not a floating type, then
            # do not apply conversion to the intermediate type.
            return f"{x}.to(tl.int16).to(tl.uint8)"

        if use_compute_types:
            out_dtype = triton_compute_type(dtype)
        else:
            out_dtype = triton_store_type(dtype)

        return f"{x}.to({out_dtype})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        assert src_dtype.itemsize == dtype.itemsize
        # We may promote float16 or bfloat16 to float32 and cause the
        # bitwidth of dtype to be different from the input tensor (i.e. float32).
        # In such as case, we will have to convert the input tensor to
        # its src_type, perform bitcast, and then convert the bit-casted
        # tensor back to float to ensure we use values with the right precision.
        if x.dtype != src_dtype:
            x = f"{x}.to({triton_type(src_dtype)})"

        out = f"{x}.to({triton_type(dtype)}, bitcast=True)"
        if upcast_compute_type(dtype) != dtype:
            out = f"{out}.to({triton_type(upcast_compute_type(dtype))})"

        return out

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = constant_repr(type_(value))
        triton_type = triton_compute_type(dtype)

        if triton_type == "tl.float32":
            # Float constants are always f32 in triton
            return triton_val

        # NOTE: We use a tensor here in order to get the expected type.
        # Otherwise, e.g. float64 constants would be truncated to float32.
        if value < 0 and not dtype.is_signed:
            triton_signed_type = f"tl.{triton_type[4:]}"
            return f"tl.full({shape}, {triton_val}, {triton_signed_type}).to({triton_type})"
        else:
            return f"tl.full({shape}, {triton_val}, {triton_type})"

    @classmethod
    def constant(cls, value, dtype):
        return cls._shaped_constant(value, dtype, shape=[])

    @staticmethod
    @maybe_upcast_float32()
    def abs(x):
        return f"tl_math.abs({x})"

    # TODO - register these ops as having divergent dtype
    # output if doing graph pass to remove consecutive casts

    @staticmethod
    def truediv(x, y):
        x_dtype = getattr(x, "dtype", None)
        y_dtype = getattr(y, "dtype", None)

        if (
            x_dtype == torch.float32
            and y_dtype == torch.float32
            and config.emulate_divison_rounding
        ):
            # x / y in Triton is lowered to div.full which is approx
            # we want div_rn to adhere with eager
            out = f"triton.language.div_rn({x}, {y})"
        else:
            out = f"({x} / {y})"

        # Workaround here since the functionality of div_rn has not ready on XPU.
        # TODO: remove this workaround after https://github.com/intel/intel-xpu-backend-for-triton/issues/5306
        # resolved.
        if torch.xpu.is_available():
            out = f"({x} / {y})"

        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().truediv(x, y)
            if out_dtype in (torch.float16, torch.float32):
                out = f"{out}.to({triton_type(out_dtype)})"

        return out

    @staticmethod
    def mod(x, y):
        out = f"({x} % {y})"
        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().mod(x, y)
            if out_dtype in (torch.float16, torch.float32):
                out = f"{out}.to({triton_type(out_dtype)})"
        return out

    @staticmethod
    @maybe_upcast_float32()
    def exp(x):
        """
        When use_fast_math, use the ftz (flushing to zero) variant
        of exponent computation.

        Check https://github.com/triton-lang/triton/issues/5735 for
        more details.
        """
        if config.use_fast_math:
            return f"tl_math.exp({x})"
        else:
            return f"libdevice.exp({x})"

    @staticmethod
    @maybe_upcast_float32()
    def exp2(x):
        return f"libdevice.exp2({x})"

    @staticmethod
    @maybe_upcast_float32()
    def expm1(x):
        return f"libdevice.expm1({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sqrt(x):
        # work around for https://github.com/pytorch/pytorch/issues/165738
        if torch.xpu.is_available():
            return f"libdevice.sqrt({x})"
        return f"tl.sqrt_rn({x})"

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            # NB: this only triggers runtime error as long as input
            # is not all zero
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == "accuracy":
            return f"{x} + 1"
        elif bug is None:
            return ops.maximum(ops.constant(0, torch.int32), x)
        else:
            raise AssertionError(
                f"unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        if torch.version.hip:
            return f"tl.minimum({a}, {b}, tl.PropagateNan.ALL)"
        else:
            return f"triton_helpers.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        if torch.version.hip:
            return f"tl.maximum({a}, {b}, tl.PropagateNan.ALL)"
        else:
            return f"triton_helpers.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def dot(a, b):
        """
        Triton code generation for lowering ops.dot to tl.dot.

        The logic is as follows:

        1. Downcasting for performance
           If the data was previously upcasted to fp32, we downcast back to the
           original dtype (e.g., fp16 or bf16) for better performance. While
           surrounding operations may run in fp32, matmul itself is executed at the
           original precision to optimize throughput.

        2. Handling non-constant reduction masks
           If the reduction mask is not constant and there was any operation between
           tl.load and tl.dot, we zero out regions outside the mask using
           tl.where(r0_mask, val, 0).
           This ensures that values outside the mask do not contribute to the dot
           product, preventing incorrect results.

        3. Shape alignment for tl.dot
           We massage shapes to match the tl.dot requirement of (Y, R) x (R, X).
           Current codegen eagerly broadcasts tl.arange to create unique axes. We
           reshape, transpose, or broadcast to align with the (Y, R) x (R, X) shape.
           We avoid using 3D dot ((Z, Y, R) x (Z, R, X)) because 3D tl.dot has
           poor performance. During batched matmul (bmm), we keep ZBLOCK=1 and call
           the 2D dot kernel instead.
        """
        assert V.kernel.is_native_matmul
        orig_a, orig_b = a, b

        def is_where_needed(var):
            # Skip if the variable doesn't have a reduction mask
            if not any(map(prefix_is_reduction, var.mask_vars)):
                return False

            reduction_range = V.kernel.range_trees[-1]
            assert reduction_range.is_reduction

            # Skip if reduction mask was already constant
            if V.kernel._has_constant_mask(reduction_range):
                return False

            # Skip if the variable is already zeroed outside the mask
            # (e.g., from tl.load(..., other=0.0))
            # TODO : track the value of outside of mask region with cse
            for k, v in V.kernel.cse._cache.items():
                if v == var and "tl.load" in k and "other=0.0" in k:
                    return False

            return True

        def where_cond(var):
            default = ir.Reduction.default_value("dot", var.dtype)
            reduction_mask = [
                f"{tree.prefix}mask"
                for tree in V.kernel.range_trees
                if tree.is_reduction
            ]

            assert len(reduction_mask) == 1, "don't tile reduction when native matmul"

            where_var = TritonKernelOverrides.where(reduction_mask[0], var, default)
            return V.kernel.cse.generate(
                V.kernel.compute, where_var, dtype=var.dtype, shape=var.shape
            )

        # When computing expressions like ((A+1) @ (B+2)),
        # native codegen will do
        #
        # a = tl.load(..., r0_mask, other=0.0)
        # b = tl.load(..., r0_mask, other=0.0)
        # tmp0 = a+1
        # tmp1 = b+2
        # tmp2 = tl.dot(tmp0, tmp1)
        #
        # This produces incorrect results because outside of r0_mask is not zero.
        # So before calling tl.dot, apply tl.where to zero out values properly.
        # TODO: Optimize - We don't need both operands to be zeroed except NaN * 0
        if is_where_needed(orig_a):
            a = where_cond(a)
        if is_where_needed(orig_b):
            b = where_cond(b)

        def reshape_transpose_broadcast_for_dot(
            value,
            initial_shape: Sequence[sympy.Expr],
            final_shape: Sequence[sympy.Expr],
        ) -> str:
            """
            Generate a reshape, transpose, and broadcast for the tl.dot.
            tl.dot requires specific shape requirement : (Y,R) x (R,X)
            but the current triton codegen eagerly broadcast the tl.arange so
            it needs to be reshaped to meet the requirement.

            This is done by three steps.
            1. remove the empty dimension (dim with size 1) and make it 2d with tl.reshape
            2. permute the dimension if needed (e.g., (X,R) -> (R,X)) with tl.trans
            3. broadcast if needed with broadcast_to.
                - This shows up when matmul operand is broadcasted with torch.expand/repeat.
                - e.g., torch.rand((16,)).expand(16,16) @ B

            e.g., (Y,1,R), (Y,R) -> tl.reshape(var, (Y,R))
            e.g., (1,X,R), (R,X) -> tl.trans(tl.reshape(var, (X,R)))
            e.g., (1,X,1), (R,X) -> tl.broadcast_to(tl.trans(tl.reshape(var, (X,1))), (R,X))

            TODO : eventually we want to remove this function when lazy broadcasting arrives
            """

     
```



## High-Level Overview


This Python file contains 23 class(es) and 300 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OpDtypeSupport`, `TritonSymbols`, `IndexingOptions`, `BlockDescriptorOptions`, `TensorDescriptorOptions`, `BlockPtrOptions`, `TritonPrinter`, `TritonCSEVariable`, `TritonOverrides`, `TritonKernelOverrides`, `HelperFunctions`, `BlockParameters`, `CooperativeReductionWorkspaceCache`, `FixedTritonConfig`, `TritonCSE`, `TMACompatibilityChecker`, `TritonKernel`, `CSEProxy`, `TritonScheduling`

**Functions defined**: `get_triton_reduction_function`, `is_sympy_integer_like`, `register_upcast`, `gen_attr_descriptor_import`, `gen_common_triton_imports`, `get_block_shape`, `get_block_size`, `get_block_offset`, `has_mask`, `has_indirect`, `has_rindex`, `has_tmpmask`, `has_rmask`, `mask_str`, `shape`, `block_shape`, `strides`, `offsets`, `create`, `lookup_size`

**Key imports**: annotations, collections, contextlib, dataclasses, functools, itertools, logging, math, operator, os


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `collections`
- `contextlib`
- `dataclasses`
- `functools`
- `itertools`
- `logging`
- `math`
- `operator`
- `os`
- `textwrap`
- `collections.abc`: Callable, Iterable, Sequence
- `typing`: Any, cast, Optional, TYPE_CHECKING, Union
- `sympy`
- `sympy.printing.precedence`: PRECEDENCE
- `torch`
- `torch._logging`
- `torch.utils._pytree as pytree`
- `torch._dynamo.device_interface`: get_interface_for_device
- `torch._dynamo.utils`: identity, preserve_rng_state
- `torch._prims_common`: is_integer_dtype
- `torch.utils._ordered_set`: OrderedSet
- `torch.utils._sympy.functions`: CeilDiv, FloorDiv, ModularIndexing
- `torch.utils._triton`: has_triton_package, has_triton_stable_tma_api
- `...utils._sympy.symbol`: free_symbol_is_type, prefix_str, symbol_is_type, SymT
- `...utils._sympy.value_ranges`: ValueRanges
- `..`: config, ir, metrics
- `..async_compile`: AsyncCompile
- `..codecache`: code_hash, get_path, PyCodeCache, write_atomic


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`torch/_inductor/codegen`):

- [`cpp_wrapper_mps.py_docs.md`](./cpp_wrapper_mps.py_docs.md)
- [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- [`cpp_flex_attention_template.py_docs.md`](./cpp_flex_attention_template.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- [`block_analysis.py_docs.md`](./block_analysis.py_docs.md)
- [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- [`cpp_bmm_template.py_docs.md`](./cpp_bmm_template.py_docs.md)
- [`python_wrapper_mtia.py_docs.md`](./python_wrapper_mtia.py_docs.md)
- [`cpp_template.py_docs.md`](./cpp_template.py_docs.md)


## Cross-References

- **File Documentation**: `triton.py_docs.md`
- **Keyword Index**: `triton.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
