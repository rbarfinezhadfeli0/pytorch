# Documentation: triton.py

## File Metadata
- **Path**: `torch/_inductor/codegen/triton.py`
- **Size**: 239932 bytes
- **Lines**: 6047
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

            # Triton 3d dot is slower than 2d dot, so we want to keep block shape in 2d
            # by fixing ZBLOCK=1 in the autotune config
            if ZBLOCK in initial_shape:
                initial_shape = ["1" if dim == ZBLOCK else dim for dim in initial_shape]

            if final_shape == [YBLOCK, RBLOCK]:
                assert XBLOCK not in initial_shape, (
                    "left tl.dot operand cannot depend on x"
                )

                shape_2d = ["1", "1"]
                if YBLOCK in initial_shape:
                    shape_2d[0] = YBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d
                value = triton_reshape(value, initial_shape, shape_2d)

                # broadcast if needed
                broadcast_needed = shape_2d != [YBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({YBLOCK}, {RBLOCK}))"

            elif final_shape == [RBLOCK, XBLOCK]:
                assert YBLOCK not in initial_shape, (
                    "right tl.dot operand cannot depend on y"
                )

                shape_2d = ["1", "1"]
                if XBLOCK in initial_shape:
                    shape_2d[0] = XBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d (X,R)
                value = triton_reshape(value, initial_shape, shape_2d)

                # transpose to (R,X)
                value = f"tl.trans({value})"

                # broadcast if needed
                broadcast_needed = shape_2d != [XBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({RBLOCK}, {XBLOCK}))"
            else:
                raise NotImplementedError

            return value

        assert len(V.kernel.dense_size_list()) >= 3, "tl.dot can only do mm and bmm"

        XBLOCK = str(TritonSymbols.block_sizes[SymT.XBLOCK])
        YBLOCK = str(TritonSymbols.block_sizes[SymT.YBLOCK])
        ZBLOCK = str(TritonSymbols.block_sizes[SymT.ZBLOCK])
        RBLOCK = str(TritonSymbols.block_sizes[SymT.R0_INDEX])

        a = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(a, list(a.shape), [YBLOCK, RBLOCK]),
            dtype=a.dtype,
            shape=(YBLOCK, RBLOCK),
        )

        b = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(b, list(b.shape), [RBLOCK, XBLOCK]),
            dtype=b.dtype,
            shape=(RBLOCK, XBLOCK),
        )

        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            input_precision = "tf32"
        else:
            input_precision = "ieee"

        return f'tl.dot({a}, {b}, input_precision="{input_precision}")'

    @staticmethod
    def inline_asm_elementwise(
        *inputs, asm, constraints=None, dtype=torch.float32, is_pure=True, pack=1
    ):
        triton_type = triton_compute_type(dtype)
        input_refs = ", ".join([str(i) for i in inputs])
        if constraints is None:
            constraints = ", ".join(["=r"] + ["r" for _ in inputs])
        return f"tl.inline_asm_elementwise('{asm}', '{constraints}', [{input_refs}], dtype={triton_type}, is_pure={is_pure}, pack={pack})"  # noqa: B950

    @staticmethod
    @maybe_upcast_float32()
    def cos(x):
        return f"tl_math.cos({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sin(x):
        return f"tl_math.sin({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        raise NotImplementedError("ops.index_expr not implemented outside a kernel")

    @staticmethod
    def masked(mask, body, other):
        raise NotImplementedError("ops.masked not implemented outside a kernel")

    @staticmethod
    @maybe_upcast_float32()
    def lgamma(x):
        return f"libdevice.lgamma({x})"

    @staticmethod
    @maybe_upcast_float32()
    def erf(x):
        return f"libdevice.erf({x})"

    @staticmethod
    @maybe_upcast_float32()
    def cosh(x):
        return f"libdevice.cosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sinh(x):
        return f"libdevice.sinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def acos(x):
        return f"libdevice.acos({x})"

    @staticmethod
    @maybe_upcast_float32()
    def acosh(x):
        return f"libdevice.acosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def asin(x):
        return f"libdevice.asin({x})"

    @staticmethod
    @maybe_upcast_float32()
    def asinh(x):
        return f"libdevice.asinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def atan2(x, y):
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def atan(x):
        return f"libdevice.atan({x})"

    @staticmethod
    @maybe_upcast_float32()
    def atanh(x):
        return f"libdevice.atanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def copysign(x, y):
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def erfc(x):
        return f"libdevice.erfc({x})"

    @staticmethod
    @maybe_upcast_float32()
    def erfinv(x):
        return f"libdevice.erfinv({x})"

    @staticmethod
    @maybe_upcast_float32()
    def hypot(x, y):
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def log10(x):
        return f"libdevice.log10({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log2(x):
        return f"libdevice.log2({x})"

    @staticmethod
    @maybe_upcast_float32()
    def nextafter(x, y):
        return f"libdevice.nextafter({x}, {y})"

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.randn({seed}, {offset})"

    @staticmethod
    def randint64(seed, offset, low, high):
        offset = f"({offset}).to(tl.uint32)"
        return f"triton_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("ops.load_seed not implemented outside a kernel")

    @staticmethod
    @maybe_upcast_float32()
    def rsqrt(x):
        if torch.version.hip:
            return f"tl.rsqrt({x})"
        else:
            return f"libdevice.rsqrt({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log1p(x):
        return f"libdevice.log1p({x})"

    @staticmethod
    @maybe_upcast_float32()
    def tan(x):
        return f"libdevice.tan({x})"

    @staticmethod
    @maybe_upcast_float32()
    def tanh(x):
        return f"libdevice.tanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return (
            f"(libdevice.signbit({x}) != 0) if ({x}).dtype is tl.float32 else {x} < 0"
        )

    @staticmethod
    @maybe_upcast_float32()
    def fmod(a, b):
        return f"libdevice.fmod({a}, {b})"

    @staticmethod
    @maybe_upcast_float32()
    def pow(a, b):
        return f"libdevice.pow({a}, {b})"

    @staticmethod
    @maybe_upcast_float32()
    def log(x):
        return f"tl_math.log({x})"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isinf(x):
        return f"libdevice.isinf({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isnan(x):
        return f"libdevice.isnan({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32()
    def round(x):
        return f"libdevice.nearbyint({x})"

    @staticmethod
    @maybe_upcast_float32()
    def floor(x):
        return f"libdevice.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def sign(x):
        z = ops.constant(0, torch.int32)
        left = ops.to_dtype((ops.lt(z, x)), torch.int8)
        right = ops.to_dtype((ops.lt(x, z)), torch.int8)
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    @maybe_upcast_float32()
    def trunc(x):
        return f"libdevice.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    @maybe_upcast_float32()
    def ceil(x):
        return f"libdevice.ceil({x})"


TritonOverrides._initialize_pointwise_overrides("triton")


class TritonKernelOverrides(TritonOverrides):
    """Map element-wise ops to Triton within a TritonKernel

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # happens in __init__ unlike _initialize_pointwise_overrides
        # because the libdevice registrations are populated during lowerings
        self._setup_libdevice_routing()

    @classmethod
    @functools.cache
    def _setup_libdevice_routing(cls):
        """Set up routing to libdevice implementations for fp64 inputs."""

        from torch._inductor.codegen.common import OpDecompositions

        for fn_name in torch._inductor.utils.op_requires_libdevice_fp64:
            assert hasattr(cls, fn_name)
            original_impl = getattr(cls, fn_name)

            def decomposition_router(x, _original_impl, _fn_name):
                if x.dtype != torch.float64:
                    return _original_impl(x)
                else:
                    return getattr(OpDecompositions, _fn_name)(x).value

            if fn_name == "sigmoid":
                assert hasattr(OpDecompositions, "sigmoid")
                fn = functools.partial(
                    decomposition_router, _original_impl=original_impl, _fn_name=fn_name
                )
                fn.__name__ = fn_name  # type: ignore[attr-defined]
                setattr(cls, fn_name, staticmethod(fn))
                continue

            def dtype_router(x, _original_impl, _fn_name):
                if x.dtype == torch.float64:
                    return f"libdevice.{_fn_name}({x})"
                else:
                    return _original_impl(x)

            fn = functools.partial(
                dtype_router, _original_impl=original_impl, _fn_name=fn_name
            )
            fn.__name__ = fn_name  # type: ignore[attr-defined]
            setattr(cls, fn_name, staticmethod(fn))

    @classmethod
    def constant(cls, value, dtype):
        # NOTE: Cannot use shape=[] as it's not supported by triton-rocm
        # We could use shape=[1] instead but starting with the correct
        # ndim avoids extra `tt.expand_dim` ops appearing in the triton IR.
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return cls._shaped_constant(value, dtype, shape=shape)

    @classmethod
    def index_expr(cls, expr, dtype):
        indexing = V.kernel.indexing(
            expr, block_ptr=False, tma_compatibility_checker=None
        )
        assert isinstance(indexing, IndexingOptions)

        shape: BlockShapeType
        if indexing.expand_shape:
            shape = indexing.expand_shape
        else:
            shape = TritonSymbols.get_block_shape(indexing.index)

        # Our sympy expr printing casts to the current kernel index dtype.
        # we only respect non int32-int64 dtypes and otherwise use current kernel indexing dtype
        index_dtype = V.kernel.get_index_dtype_as_torch_dtype()
        dtype = dtype if dtype not in (torch.int32, torch.int64) else index_dtype

        # after we emit this var we cast it to the correct dtype
        orig = config.test_configs.runtime_triton_dtype_assert
        try:
            config.test_configs.runtime_triton_dtype_assert = False
            var = V.kernel.cse.generate(
                V.kernel.compute,
                indexing.index_str,
                bounds=get_bounds_index_expr(expr),
                dtype=dtype,
                shape=shape,
            )
        finally:
            config.test_configs.runtime_triton_dtype_assert = orig

        if dtype not in (torch.int32, torch.int64):
            var = V.kernel.cse.generate(
                V.kernel.compute,
                cls.to_dtype(var, dtype),
                dtype=upcast_compute_type(dtype),
                shape=var.shape,
            )
        else:
            # TODO: we are not always consistent in enforcing that the output of the index expr printing
            # results in the indexing dtype. So if we detect that we have an input which might type promote
            # to a dtype other than indexing dtype, add a cast.
            # Trying to avoid
            dtype = index_dtype
            for index_var in expr.free_symbols:
                if symbol_is_type(index_var, SymT.TMP):
                    dtype = torch.promote_types(
                        dtype, V.kernel.cse.varname_map[index_var.name].dtype
                    )

            if dtype != index_dtype:
                var = V.kernel.cse.generate(
                    V.kernel.compute,
                    cls.to_dtype(var, index_dtype),
                    dtype=index_dtype,
                    shape=var.shape,
                )

        var.mask_vars = indexing.mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        if mask is not None and torch.version.hip is not None:
            mask = V.kernel.cse.generate(
                V.kernel.compute,
                f"{mask}.to(tl.int1)",
                dtype=torch.bool,
                shape=mask.shape,
            )

        nodes = body.graph.find_nodes(op="output")
        assert nodes, "graph for body does not contain an output"

        need_where = False
        # If we have a tl.load with a masking operator and no other value
        # we can add the mask here and the other value to the tl.load
        # operator to save the branching cost.
        for node in nodes:
            for arg in node.args:
                if arg.target != "load" or should_unwrap_unspec_arg(arg.args[1]):
                    need_where = True
                    break

        value = None if need_where else other

        with V.kernel.mask_loads(mask, value=value) as new_mask:
            result = body()

        if need_where:
            # Remove once CSEVariables track the dtype
            if result.bounds.is_bool:
                other = bool(other)
            # Take dtype from result to prevent accidental promotion
            other = V.kernel.cse.generate(
                V.kernel.compute,
                f"tl.full({result}.shape, {constant_repr(other)}, {result}.dtype)",
                bounds=ValueRanges.wrap(other),
                dtype=result.dtype,
                shape=result.shape,
            )
            ret = ops.where(new_mask, result, other)
        else:
            ret = result

        ret.mask_vars.discard(new_mask)
        return ret

    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return (
            f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"
        )

    @staticmethod
    def frexp(x):
        cache_key = f"frexp({x})"
        if cse_val := V.kernel.cse.try_get(cache_key):
            return cse_val

        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        exponent = V.kernel.cse.newvar(dtype=torch.int32, shape=x.shape)
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        V.kernel.cse.put(cache_key, (mantissa, exponent))
        return (mantissa, exponent)

    @staticmethod
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class HelperFunctions:
    """An ordered set of helper functions."""

    _templates_seen: dict[str, str]  # Template code to function name
    finalized_helpers: list[str]

    def __init__(self) -> None:
        self._templates_seen = {}
        self.finalized_helpers = []

    def add(self, template_code: str, *, base_name="_triton_helper_fn") -> str:
        """This accepts a function definition with the function name
        left as a format specifier e.g.

            @triton.jit
            def {name}(arg0, arg1):
                return arg0 + arg1

        We add the templated code to the function set and return the name
        assigned to that function.

        """
        existing_name = self._templates_seen.get(template_code)
        if existing_name is not None:
            # Don't duplicate existing helpers
            return existing_name

        name = f"{base_name}{len(self.finalized_helpers)}"
        self._templates_seen[template_code] = name
        self.finalized_helpers.append(template_code.format(name=name))
        return name

    def __iter__(self):
        return iter(self.finalized_helpers)

    def __getitem__(self, idx):
        return self.finalized_helpers[idx]


@dataclasses.dataclass
class BlockParameters:
    """
    Class representing ND block dimensions, for block pointer analysis.
    """

    shape: list[sympy.Expr] = dataclasses.field(default_factory=list)
    block_shape: list[sympy.Expr] = dataclasses.field(default_factory=list)
    strides: list[sympy.Expr] = dataclasses.field(default_factory=list)
    offsets: list[sympy.Expr] = dataclasses.field(default_factory=list)

    def __add__(self, other: BlockParameters) -> BlockParameters:
        """
        Concatenates block parameters.
        """
        cls = type(self)
        a, b = tuple(dataclasses.asdict(x) for x in (self, other))
        return cls(**{key: a[key] + b[key] for key in a})

    def transpose(self) -> BlockParameters:
        return BlockParameters(
            self.shape[::-1],
            self.block_shape[::-1],
            self.strides[::-1],
            self.offsets[::-1],
        )


class CooperativeReductionWorkspaceCache:
    """
    The scratch space used for cooperative reductions can be reused
    after two reduction loops.  This keeps track of what can be reused.
    """

    def __init__(self, args):
        self.args = args
        self.current_loop = []
        self.prior_loop = []
        self.ready_for_reuse = collections.defaultdict(collections.deque)
        self.loop_count = 0
        self.store_count = 0

    def allocate(self, nbytes: sympy.Expr):
        cached = self.ready_for_reuse.get(nbytes)
        if cached:
            return cached.popleft()
        ws_name, _, ws_offset = self.args.workspace(nbytes, False)
        self.current_loop.append((nbytes, ws_name, ws_offset))
        return (ws_name, ws_offset)

    def on_loop_end(self):
        # Buffers can be reused after 2 loop ends
        for nbytes, ws_name, ws_offset in self.prior_loop:
            self.ready_for_reuse[nbytes].append((ws_name, ws_offset))
        self.prior_loop = self.current_loop
        self.current_loop = []
        self.loop_count += 1

    def increment_store_count(self):
        prior = self.store_count
        self.store_count += 1
        return prior


@dataclasses.dataclass
class FixedTritonConfig:
    config: dict[str, int]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config


class TritonCSE(CSE[TritonCSEVariable, Union[str, tuple[str, str]]]):
    """
    Subclasses CSE to apply the current load mask to the cache key to avoid CSEing
    variables across separate masked blocks.
    """

    def augment_key(self, cache_key: str) -> Union[str, tuple[str, str]]:
        if mask := V.kernel._load_mask:
            return (cache_key, mask.name)
        else:
            return cache_key


@dataclasses.dataclass
class TMACompatibilityChecker:
    """
    Checks if the TMA API can be used for load / store triton operations.
    """

    kernel: TritonKernel
    dtype: torch.dtype
    for_store: bool
    force: bool

    def __post_init__(self):
        self.failed_debug_prefix = "Cannot use TMA descriptor for load / store since: "

    # Also see Note: TMA API Restrictions for the below
    def can_use_tma(
        self,
    ) -> bool:
        if self.force:
            return True
        if not (
            V.graph.get_current_device_or_throw().type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 9
            and config.triton.use_tensor_descriptor
            and config.assume_aligned_inputs
            and has_triton_stable_tma_api()
            # For CUDA The base ptr needs to be aligned
        ):
            log.debug(
                (
                    "%s Requires triton>=3.4.0, a CUDA device with cc>=9.0 and"
                    " `use_tensor_descriptor` and `assume_aligned_inputs` options enabled"
                ),
                self.failed_debug_prefix,
            )
            return False

        # `no_x_dim` => XBLOCK=1, and for reductions this means only one element
        # is to be stored . However the TMA API requires that
        # the store will be 16 byte aligned, which is not attainable with a single
        # element
        if self.for_store and self.kernel.no_x_dim:
            log.debug(
                "%s stores with `no_x_dim` cannot load 16 bytes.",
                self.failed_debug_prefix,
            )
            return False

        return True

    def are_block_parameters_compatible(
        self,
        block_params: BlockParameters,
    ) -> bool:
        """
        Check if the block parameters are valid for TMA.
        If force, we allow relying on symbolic hints equivalent
        to what we check for Triton templates.
        """
        if self.force:
            strides = [
                V.graph.sizevars.symbolic_hint(st) for st in block_params.strides
            ]
        else:
            strides = block_params.strides

        # The TMA API requires that the innermost stride is 1
        # and that the outer strides are 16 byte aligned
        if not V.graph.sizevars.statically_known_equals(strides[-1], sympy.Integer(1)):
            log.debug(
                "%s TMA API requires innermost stride to be 1.",
                self.failed_debug_prefix,
            )
            return False

        element_size = self.dtype.itemsize
        for stride in strides[:-1]:
            if not V.graph.sizevars.statically_known_equals(
                ModularIndexing(stride * element_size, 1, sympy.Integer(16)),
                sympy.Integer(0),
            ):
                log.debug(
                    "%s TMA API requires outer strides to be 16 byte aligned.",
                    self.failed_debug_prefix,
                )
                return False

        # Now compute the minimum value of the block type that is used
        # in the innermost block size that can guarantee that 16 bytes of data
        # can be loaded / stored.
        # Start with finding the innermost block type
        innermost_block_shape = block_params.block_shape[-1]
        innermost_block_type = None
        innermost_block_symt = None
        for block_type_str in innermost_block_shape.free_symbols:
            for block_symt in TritonSymbols.block_types:
                if symbol_is_type(block_type_str, block_symt):
                    innermost_block_type = block_type_str
                    innermost_block_symt = block_symt
                    break
        assert innermost_block_type and innermost_block_symt, (
            f"{innermost_block_shape} expr must contain a single block type from {TritonSymbols.block_types}"
        )

        # For persistent reductions, the reduction block sizes are fixed at compile time
        if self.kernel.persistent_reduction and not self.for_store:
            # For a discontiguous tensor, a 1D block will be split across several
            # dimensions, e.g. R0_BLOCK:
            # block_shape=[XBLOCK, ((R0_BLOCK + 31)//32), Min(1, ((R0_BLOCK + 31)//32)), Min(32, R0_BLOCK)]
            # The persistent R0_BLOCK will be a power of 2 that is at least r0_numel So it
            # should be guaranteed that Min(32, R0_BLOCK) * element_size >= 16
            innermost_tree_prefix = prefix_str[innermost_block_symt]
            tree_numel = None
            for t in self.kernel.range_trees:
                if t.is_reduction:
                    if t.prefix == innermost_tree_prefix:
                        tree_numel = t.numel
                        break
            assert tree_numel is not None
            persistent_rblock = self.kernel._get_persistent_RBLOCK(tree_numel)
            innermost_block_bytes = (
                innermost_block_shape.subs({innermost_block_type: persistent_rblock})
                * element_size
            )
            if not V.graph.sizevars.statically_known_geq(
                innermost_block_bytes, sympy.Integer(16)
            ):
                log.debug(
                    "%s persistent reduction innermost block shape cannot load 16 bytes.",
                    self.failed_debug_prefix,
                )
                return False

        else:
            # E.g. if the innermost block shape is Min(2, XBLOCK)
            # then the TMA API can only be used if the dtype has an 8 byte element
            # size so that 16 bytes of data can be loaded in the innermost dimension
            try:
                min_block_size = next_power_of_2(
                    int(
                        sympy.nsolve(
                            innermost_block_shape * element_size - 16,
                            innermost_block_type,
                            1,
                        )
                    )
                )

                block_type_str = V.kernel.index_to_str(innermost_block_type)
                # Check block sizes if the user has provided a fixed triton config
                if self.kernel.fixed_config:
                    if min_block_size > self.kernel.fixed_config[block_type_str]:
                        log.debug(
                            "%s For block %s, fixed config block size %d is smaller "
                            "than the minimum required: %d",
                            self.failed_debug_prefix,
                            block_type_str,
                            self.kernel.fixed_config[block_type_str],
                            min_block_size,
                        )
                        return False
                else:
                    # Update the minimum block sizes that are passed to triton
                    # heuristics
                    self.kernel.tma_min_block_sizes[block_type_str] = max(
                        min_block_size,
                        self.kernel.tma_min_block_sizes.get(block_type_str, 1),
                    )

            except ValueError:
                log.debug(
                    "%s innermost block shape cannot load 16 bytes.",
                    self.failed_debug_prefix,
                )
                return False

        return True

    def can_lift(self) -> bool:
        """
        Can you lift the make_tensor_descriptor
        call to the top of the kernel? This requires
        being certain that all of the shape, stride,
        and block_shape information is handled in arguments
        or top level definitions.

        Right now we assume this is always possible if you force TMA.
        """
        return self.force


class TritonKernel(SIMDKernel[TritonCSEVariable]):
    """A class to represent a triton kernel and helpers to generate
    triton kernel programmatically
    """

    overrides = TritonKernelOverrides  # type: ignore[assignment]
    helper_functions: HelperFunctions
    kexpr: Callable[[sympy.Expr], str] = texpr
    allow_block_ptr = True
    tma_compatibility_checker_cls = TMACompatibilityChecker
    block_ptr_options_cls: type[BlockPtrOptions] = BlockPtrOptions
    tensor_descriptor_options_cls: type[TensorDescriptorOptions] = (
        TensorDescriptorOptions
    )

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.optimize_mask: bool = optimize_mask
        self.fixed_config = fixed_config
        super().__init__(tiling, **kwargs)
        self.cse = TritonCSE(self.newvar_prefix, self.suffix)
        # Cache of values that can be reused for the prologue.
        self.prologue_cache: dict[str, str] = {}
        self.prologue: IndentedBuffer = IndentedBuffer()
        self.post_loop_combine: IndentedBuffer = IndentedBuffer()
        self.post_loop_store: IndentedBuffer = IndentedBuffer()
        self.outside_loop_vars = OrderedSet[Any]()
        self.min_elem_per_thread = min_elem_per_thread
        self.block_ptr_id = itertools.count()
        self.block_ptr_to_buffer = dict[str, str]()
        self.helper_functions = HelperFunctions()
        self.pointer_advancements: dict[SymT, dict[str, list[sympy.Expr]]] = (
            collections.defaultdict(dict)
        )
        self.tma_min_block_sizes = dict[str, int]()
        self.hint_override = hint_override
        self._load_counts: collections.Counter[str] = collections.Counter()
        self._load_index = 0

        # A set of autotuning hints to pass as part of triton_meta
        self.autotune_hints = OrderedSet[AutotuneHint]()
        self.triton_meta: Optional[dict[str, Any]] = None

        if self.inside_reduction:
            self.codegen_reduction_numels(self.body)

        if self.cooperative_reduction:
            self.init_cooperative_reduction()

        self.codegen_range_tree()

        if self.cooperative_reduction:
            self.init_cooperative_reduction_mask()

        self.has_load_with_contiguous_rdim = False
        # We track the store name since a store can be canceled later
        self.stores_with_contiguous_rdim: list[str] = []

    @staticmethod
    def _has_stride1_on_rdim(index) -> bool:
        # These analysis is only needed in deterministic mode so far
        # to filter triton configs. Return false immediately to avoid
        # increasing compilation time when the mode is off.
        if not (
            config.deterministic or config.test_configs.force_filter_reduction_configs
        ):
            return False
        support_vars = index.free_symbols
        reduce_vars = [
            var
            for var in support_vars
            if symbol_is_type(var, TritonSymbols.reduction_types)
        ]

        if len(reduce_vars) == 0:
            return False

        # for expression "x0 + 150528*((x1//(s27*s38))) + 3*(ModularIndexing(x1, 1, s38)) + 672*(ModularIndexing(x1, s38, s27))"
        # stride_vars will results in DivisionByZero error
        try:
            stride_vars = V.graph.sizevars.stride_vars(index, reduce_vars, support_vars)
        except ZeroDivisionError:
            return False

        return any(stride == 1 for stride in stride_vars)

    @property
    def has_store_with_contiguous_rdim(self) -> bool:
        return not all(
            is_buffer_removed(name) for name in self.stores_with_contiguous_rdim
        )

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return triton_type(dtype)

    def should_use_cooperative_reduction(self) -> bool:
        return self.inside_reduction and V.choices.should_use_cooperative_reduction(
            self.features
        )

    def init_cooperative_reduction(self):
        """One time setup code for cooperative reductions."""
        assert self.cooperative_reduction

        # shift all the grids over since tl.program_id(0) is for rsplit
        for tree in self.range_trees:
            if tree.grid_dim is not None:
                tree.grid_dim += 1

        sem_count = self.numels["x"]
        if self.fixed_config:
            sem_count = CeilDiv(sem_count, self.fixed_config["XBLOCK"])
        self.semaphores_name = self.args.semaphores(sem_count)
        self.cooperative_reduction_workspace_cache = CooperativeReductionWorkspaceCache(
            self.args
        )
        self.body.splice(
            """\
            RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
            RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
            HAS_RSPLIT: tl.constexpr = RSPLIT > 1
            rsplit_id = tl.program_id(0)
            num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
            rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
            rsplit_start = rsplit_chunk * rsplit_id
            rsplit_end = rsplit_chunk * (rsplit_id + 1)
            """,
        )
        if any(
            not self._has_constant_mask(tree)
            for tree in self.range_trees
            if tree.is_reduction
        ):
            self.body.writeline(
                "rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)"
            )

    def init_cooperative_reduction_mask(self):
        rsplit_arange = "tl.arange(0, RSPLIT_NEXT_POWER_OF_2)"
        if not self.no_x_dim:
            rsplit_arange = f"{rsplit_arange}[None, :]"
        self.body.writeline(f"rsplit_arange = {rsplit_arange}")

        if self._has_constant_xmask():
            self.body.splice(
                """\
                if RSPLIT_IS_POWER_OF_2:
                    rsplit_mask: tl.constexpr = None
                else:
                    rsplit_mask = rsplit_arange < RSPLIT
                """
            )
        else:
            assert not self.no_x_dim
            self.body.writeline(
                "rsplit_mask = xmask if RSPLIT_IS_POWER_OF_2 else ((rsplit_arange < RSPLIT) & xmask)"
            )

    def codegen_range_tree(self):
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if not tree.is_loop:
                self.iteration_ranges_codegen_header(tree, self.body)
            elif self.inside_reduction:
                # workaround for this issue:
                # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
                self.body.writeline(
                    f"{tree.prefix}base = {self.iteration_ranges_ranges_code(tree)}"
                )

        if self.inside_reduction:
            if any(tree.is_loop for tree in self.range_trees):
                # If the kernel contains loops, compute rbase.
                rn_bases = self._get_reduction_symbols(
                    "base", integer=True, nonnegative=True
                )
                rbase = self._flatten_reduction_indices(rn_bases)
                self.body.splice(f"rbase = {self.index_to_str(rbase)}")
            else:
                # For looped reductions, indexing is deferred to the innermost loop.
                self.codegen_reduction_indices(self.body)

    def need_numel_args(self):
        """
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
        return True

    def should_use_persistent_reduction(self) -> bool:
        return self.inside_reduction and V.choices.should_use_persistent_reduction(
            self.features, self.cooperative_reduction
        )

    def want_no_x_dim(self):
        return (
            self.persistent_reduction
            and len(self.numels) == self.num_reduction_dims + 1
            and self.fixed_config
            and self.fixed_config["XBLOCK"] == 1
        )

    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape: Optional[Union[str, tuple[str]]] = None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
        tma_compatibility_checker: Optional[TMACompatibilityChecker] = None,
    ):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.prepare_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: OrderedSet[str] = OrderedSet()
        for var in sorted(index_vars, key=operator.attrgetter("name")):
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or symbol_is_type(
                var, TritonSymbols.reduction_types
            )
            if override_mask:
                pass
            elif symbol_is_type(var, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
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
                pass
            else:
                # var is one of xN, yN, r0_N or r1_N
                prefix_matches = [
                    prefix_str[symt]
                    for symt in TritonSymbols.block_types
                    if symbol_is_type(var, symt)
                ]
                if len(prefix_matches) == 0:
                    pass
                assert len(prefix_matches) == 1, f"Ambiguous type: {var.name}"
                mask_vars.add(f"{prefix_matches[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars: OrderedSet[str] = OrderedSet()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            (
                (block_ptr and self.allow_block_ptr and config.triton.use_block_ptr)
                or (
                    tma_compatibility_checker
                    and tma_compatibility_checker.can_use_tma()
                )
            )
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/triton-lang/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):

            def match_affine_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Matches expressions of the form:
                    idx = s * xindex

                This implies stride (s,), and shape (XBLOCK,).
                """
                stride = BlockPatternMatcher.match_affine_block_expr(
                    index, range_tree.symbol()
                )
                if stride is None:
                    return None

                return BlockParameters(
                    shape=[range_tree.numel],
                    block_shape=[TritonSymbols.get_block_size(range_tree)],
                    strides=[stride],
                    offsets=[TritonSymbols.get_block_offset(range_tree)],
                )

            def match_mod_div_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Matches higher-dimensional blocks coming from FloorDiv and ModularIndexing.

                Example expression to match:
                   sN * ((rindex//(d1 * ... * d(N-1))))
                       + s1 * ModularIndexing(rindex, 1, d1)
                       + ...
                       + s(N-1) * ModularIndexing(rindex, d1 * ... * d(N-2), d(N-1))

                This iterates over a block of shape (dN, ..., d1) and stride
                (sN, ..., s1). (d1,...,d(N-1)) and (s1,...,sN) are
                wildcards that we match.

                Note that dN does not appear in the expression, but we solve for it
                using range tree numels and the other dims.
                """

                index_var = range_tree.symbol()

                # Bound the possible number of dims. We use the following heuristics:
                # - At least one dim for each range tree node.
                # - At least one dim for every FloorDiv or ModularIndexing op.
                # - At least 2 dims to pattern match.
                denom, modulo = sympy.symbols(
                    "denom modulo",
                    cls=functools.partial(sympy.Wild, exclude=[index_var]),
                )
                num_dims = max(
                    2,
                    # range_tree.nodes only includes the entries for the range tree
                    # len(range_tree.nodes) <= self.range_tree_nodes
                    len(range_tree.nodes),
                    (
                        index.count(FloorDiv(index_var, denom))
                        + index.count(ModularIndexing(index_var, denom, modulo))
                    ),
                )

                match_result = BlockPatternMatcher.match_mod_div_block_expr(
                    index, index_var, range_tree.numel, num_dims
                )
                if match_result is None:
                    return None

                (
                    dims,
                    strides,
                    block_index_exprs,
                ) = match_result
                slice_numels = BlockPatternMatcher.get_slice_numels(dims)

                # Check for applicable iteration range sizes.
                # When mapping a 1D block into an ND one, we need to know that
                # the number of elements is not changed. This means the slice numels of
                # the ND iteration range must evenly divide the length of the 1D block.
                # There are two cases where we can guarantee this:
                #  1. Numels are powers of 2. If numel == 2 ** n, and we know XBLOCK == 2 ** m,
                #     with n and m integers, then either numel is a multiple of XBLOCK, or numel
                #     is less than XBLOCK. (If numel is less than XBLOCK, we round up to 1 below.)
                #  2. Numels are multiples of the maximum possible block size.
                sizevars = V.graph.sizevars
                max_block = self.max_block(range_tree.prefix)
                if any(
                    not sizevars.statically_known_multiple_of(numel, max_block)
                    and not sizevars.statically_known_power_of_2(numel)
                    for numel in slice_numels
                ):
                    return None

                # Compute the ND block shape from the linear block size.
                # Use CielDiv to round leading dimensions up to 1.
                # Non-leading dimensions are clamped to the size of the iteration range,
                # while the leading dimension can exceed this to accommodate a larger
                # block size.
                linear_block_size = TritonSymbols.get_block_size(range_tree)
                block_shape: list[sympy.Expr] = [
                    CeilDiv(linear_block_size, slice_numels[0])
                ] + [
                    sympy.Min(CeilDiv(linear_block_size, numel), dim)
                    for numel, dim in zip(slice_numels[1:], dims[1:])
                ]

                # Compute block offsets from {xyzr}offset and the matched expressions.
                block_offsets: list[sympy.Expr] = [
                    sympy_subs(
                        expr, {index_var: TritonSymbols.get_block_offset(range_tree)}
                    )
                    for expr in block_index_exprs
                ]

                return BlockParameters(
                    shape=dims,
                    block_shape=block_shape,
                    strides=strides,
                    offsets=block_offsets,
                )

            def match_block_subexpr(
                expr: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Match a block indexing subexpression involving a single range tree.
                """
                for match_func in (
                    match_affine_block,
                    match_mod_div_block,
                ):
                    match = match_func(expr, range_tree)
                    if match is not None:
                        return match

                return None

            def match_block_expr() -> Optional[BlockDescriptorOptions]:
                index_relative_to_xyr_index = sympy_subs(
                    index, {v: t.expr for v, t in self.range_tree_nodes.items()}
                )
                range_trees = self.active_range_trees()

                # Partition the index into subexpressions pertaining to each range tree.
                # For example xindex * 5 + r0_index * 3 is partitioned to
                # (xindex * 5, r0_index * 3).
                index_subexprs = [
                    BlockPatternMatcher.get_subexpr_involving_symbol(
                        index_relative_to_xyr_index, tree.symbol()
                    )
                    for tree in range_trees
                ]

                # Match each range tree's subexpression separately.
                range_symbols = OrderedSet(tree.symbol() for tree in range_trees)
                block_params = BlockParameters()
                for tree, subexpr in zip(range_trees, index_subexprs):
                    # Reject mixed terms, e.g. xindex * r0_index.
                    # NB: the zero expression is allowed, for broadcasting.
                    if len(range_symbols.intersection(subexpr.free_symbols)) > 1:
                        return None

                    # Match the subexpression for this range tree.
                    params = match_block_subexpr(subexpr, tree)
                    if params is None:
                        return None
                    block_params += params

                # Collect leftover terms as a constant offset.
                offset = index_relative_to_xyr_index - sum(index_subexprs)

                # Form the block pointer or TMA descriptor.
                self.filter_masks(mask_vars)

                options_class = (
                    self.block_ptr_options_cls
                    if config.triton.use_block_ptr
                    else self.tensor_descriptor_options_cls
                )
                nonlocal tma_compatibility_checker
                if config.triton.use_block_ptr:
                    can_lift = False
                    transpose_contiguous = False
                else:
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    can_lift = tma_compatibility_checker.can_lift()
                    # Only try transpose if we know the output shape
                    # in case we need to transpose the data.
                    transpose_contiguous = copy_shape is not None

                options = options_class.create(
                    params=block_params,
                    constant_offset=offset,
                    range_trees=range_trees,
                    mask_vars=mask_vars,
                    get_max_block=self.max_block,
                    can_lift=can_lift,
                    transpose_contiguous=transpose_contiguous,
                )
                if isinstance(options_class, TensorDescriptorOptions):
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    if not tma_compatibility_checker.are_block_parameters_compatible(
                        options.params
                    ):
                   

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 19 class(es): OpDtypeSupport, TritonSymbols, class, class, class, class, TritonPrinter, TritonCSEVariable, TritonOverrides, TritonKernelOverrides, HelperFunctions, class, CooperativeReductionWorkspaceCache, class, TritonCSE, class, TritonKernel, CSEProxy, TritonScheduling

### Functions
This file defines 300 function(s): get_triton_reduction_function, is_sympy_integer_like, register_upcast, gen_attr_descriptor_import, gen_common_triton_imports, get_block_shape, get_block_size, get_block_offset, has_mask, has_indirect, has_rindex, has_tmpmask, has_rmask, mask_str, shape, block_shape, strides, offsets, create, lookup_size, remove_dims, replace_offset, remove_roffsets, compute_boundary_check, boundary_check, has_indirect, has_rindex, has_rmask, has_tmpmask, has_mask


## Key Components

The file contains 18968 words across 6047 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 239932 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
