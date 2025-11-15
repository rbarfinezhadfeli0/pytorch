# Documentation: `docs/torch/_inductor/codegen/wrapper.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/wrapper.py_docs.md`
- **Size**: 54,432 bytes (53.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/wrapper.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/wrapper.py`
- **Size**: 152,827 bytes (149.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import dis
import functools
import inspect
import logging
import operator
import random
import re
import tempfile
from collections.abc import Callable
from itertools import chain, count
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy
from sympy import Expr

import torch
import torch._ops
import torch.utils._pytree as pytree
from torch import dtype as torch_dtype
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codegen.debug_utils import DebugPrinterManager
from torch._inductor.codegen.multi_kernel import MultiKernelState
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._logging import trace_structured
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
    resolve_unbacked_bindings,
    SymTypes,
)
from torch.fx.node import _get_qualified_name
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import async_compile, config, ir
from ..codecache import output_code_log
from ..ir import IRNode, ReinterpretView
from ..runtime import triton_heuristics
from ..runtime.hints import DeviceProperties
from ..utils import (
    cache_on_self,
    DelayReplaceLine,
    get_benchmark_name,
    get_dtype_size,
    IndentedBuffer,
    is_codegen_graph_partition_subgraph,
    is_using_cudagraph_partition,
    LineContext,
    sympy_product,
    sympy_str,
    sympy_subs,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    CodeGen,
    DeferredLine,
    PythonPrinter,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .cpp_utils import cexpr
from .triton_utils import config_of, should_unwrap_unspec_arg, signature_to_meta


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import triton

    from ..graph import GraphLowering
    from ..ir import ExternKernel
    from ..scheduler import BaseSchedulerNode
    from .wrapper_fxir import FxConverter


log = logging.getLogger(__name__)

pexpr = PythonPrinter().doprint


ReuseKey = tuple[torch.device, torch.dtype, str, bool]
BufferLike = Union[ir.Buffer, WorkspaceArg]
FxConversionFunc = Callable[["WrapperLine"], None]


def buffer_reuse_key(node: BufferLike) -> ReuseKey:
    storage_size = V.graph.get_allocation_storage_size(node)
    alignment = node.get_name() not in V.graph.unaligned_buffers
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        alignment,
    )


def can_match_buffer_size(input_buf: BufferLike, output_buf: BufferLike):
    # Return True if input_buf can be re-inplaced for output_buf.
    # This differs from `buffer_reuse_key` for general buffer reuse.
    if input_buf.get_device_or_error() != output_buf.get_device_or_error():
        return False

    if input_buf.get_dtype() != output_buf.get_dtype():
        return False

    input_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(input_buf)
    )
    output_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(output_buf)
    )

    if (
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(input_size) == sympy_str(output_size)
    ) or (
        # statically known that 0.95 * input_size <= output_size <= input_size
        V.graph.sizevars.statically_known_geq(output_size, 0.95 * input_size)
        and V.graph.sizevars.statically_known_leq(output_size, input_size)
    ):
        return True

    return False


# TODO: Move to a well known place
TritonMetaParams = dict[str, int]
TritonGrid = Union[
    tuple[Union[int, sympy.Expr], ...], Callable[[TritonMetaParams], tuple[int, ...]]
]


def user_defined_kernel_grid_fn_code(
    name: str,
    configs: list[triton.Config],  # type: ignore[name-defined]
    grids: list[TritonGrid],
    wrapper: Optional[PythonWrapperCodegen] = None,
    original_fxnode_name: Optional[str] = None,
) -> tuple[str, str]:
    output = IndentedBuffer()

    def _convert_to_sympy_expr(item: Union[int, sympy.Expr]) -> sympy.Expr:
        return item if isinstance(item, sympy.Expr) else sympy.Integer(item)

    def determine_grid(
        grid: TritonGrid,
        example_grid: Optional[TritonGrid] = None,
    ):
        """
        This function return a tuple of two values: the first one is for the real grid
        which is used in the generated code; the second one is an example grid with
        concreate values which is used in the autotune block to run the generated
        kernels at compile time.
        """
        if wrapper is None or callable(grid):
            # return as-is when used in eager mode or when grid is callable
            return grid, grid
        # Grid contains ints/Expr, so utilize wrapper's expr printer for codegen
        sympy_grid = tuple(_convert_to_sympy_expr(g) for g in grid)
        if not example_grid:
            example_grid = sympy_grid
        return (
            wrapper.codegen_python_shape_tuple(sympy_grid),
            (
                wrapper.codegen_python_shape_tuple(
                    tuple(
                        wrapper.generate_example_arg_value(g, type(g))
                        for g in example_grid  # type: ignore[union-attr]
                    )
                )
                if config.triton.autotune_at_compile_time
                else None
            ),
        )

    def writeline(line: str, example_grid: Optional[str] = None):
        output.writeline(line)
        if (
            wrapper
            and config.triton.autotune_at_compile_time
            and name not in wrapper.kernel_autotune_names
        ):
            wrapper.kernel_autotune_calls.writeline(example_grid or line)

    fn_name = f"grid_wrapper_for_{name}"
    writeline(f"def {fn_name}(meta):")
    kernel_autotune_calls_indent = (
        wrapper.kernel_autotune_calls.indent()
        if wrapper and config.triton.autotune_at_compile_time
        else contextlib.nullcontext()
    )
    with output.indent(), kernel_autotune_calls_indent:
        if (
            config.triton.autotune_at_compile_time
            and original_fxnode_name
            and V.graph.autotuning_grids
            and original_fxnode_name in V.graph.autotuning_grids
        ):
            example_grids = V.graph.autotuning_grids[original_fxnode_name]
        else:
            example_grids = [None] * len(grids)
        if len(grids) == 1:
            grid, example_grid = determine_grid(grids[0], example_grids[0])
            writeline(f"return {grid}", f"return {example_grid}")
        else:
            assert len(grids) > 1
            assert len(grids) == len(configs)
            seen: OrderedSet[str] = OrderedSet()
            # sort the configs from the largest # of kwargs to the smallest to
            # emit the grids in the order of (approximately) decreasing specificity
            # TODO(aakhundov): the sorting below is generally not sufficient, so
            # maybe we'll need to restrict the supported cases to identical kwarg
            # names in all autotuning configs.
            for grid, c, example_grid in sorted(
                zip(grids, configs, example_grids),
                key=lambda x: len(x[1].kwargs),
                reverse=True,
            ):
                guardslist = []
                if c.kwargs:
                    # Remove AMD specific kwargs.
                    for kwarg in c.kwargs:
                        if kwarg not in [
                            "matrix_instr_nonkdim",
                            "waves_per_eu",
                            "kpack",
                        ]:
                            guardslist.append(f"meta['{kwarg}'] == {c.kwargs[kwarg]}")
                if guardslist:
                    guards = " and ".join(guardslist)
                else:
                    guards = "True"  # for configs with empty kwargs
                grid, example_grid = determine_grid(grid, example_grid)
                statement = f"if {guards}: return {grid}"
                if statement in seen:
                    continue
                seen.add(statement)
                writeline(statement, f"if {guards}: return {example_grid}")

    return fn_name, output.getvalue()


def user_defined_triton_kernel_transitive_closure_source_code(kernel) -> str:
    """
    Given a triton kernel function pointer collect the transitive closure of
    its dependencies
    """
    compile_wrapper = IndentedBuffer()
    compile_wrapper.splice(kernel.src, strip=True)

    # Also include any possible kernel being called indirectly
    import triton
    from triton import JITFunction  # type: ignore[name-defined, attr-defined]
    from triton.language import constexpr  # type: ignore[name-defined]

    # global constexpr vars handled above
    symbols_included = OrderedSet([kernel.__name__])

    def traverse(cur_kernel):
        # here we extract the unqualified names (i.e., not attributes and
        # without prepended module name) loaded in the kernel code, which
        # are matched with the co_names and __globals__ below to codegen
        # the respective imports necessary for the kernel compilation
        unqualified_loads = OrderedSet(
            inst.argval
            for inst in dis.Bytecode(cur_kernel.fn)
            if inst.opname == "LOAD_GLOBAL"
        )
        global_annotations = cur_kernel.fn.__globals__.get("__annotations__", {})
        for symbol_name in cur_kernel.fn.__code__.co_names:
            if symbol_name in symbols_included:
                continue
            if symbol_name in cur_kernel.fn.__globals__:
                symbol = cur_kernel.fn.__globals__[symbol_name]
                if isinstance(symbol, JITFunction):
                    compile_wrapper.newline()
                    compile_wrapper.writeline("@triton.jit")
                    # pyrefly: ignore  # missing-attribute
                    compile_wrapper.splice(symbol.src, strip=True)
                    symbols_included.add(symbol_name)
                    traverse(symbol)
                elif hasattr(triton, "constexpr_function") and isinstance(
                    # pyrefly: ignore  # missing-attribute
                    symbol,
                    # pyrefly: ignore  # missing-attribute
                    triton.runtime.jit.ConstexprFunction,
                ):
                    compile_wrapper.newline()
                    compile_wrapper.writeline("@triton.constexpr_function")
                    compile_wrapper.splice(symbol.src, strip=True)
                    symbols_included.add(symbol_name)
                    traverse(symbol)
                elif isinstance(symbol, (int, str, bool, constexpr)):
                    compile_wrapper.newline()
                    if isinstance(symbol, constexpr):
                        symbol_str = f"tl.constexpr({symbol.value!r})"
                    else:
                        symbol_str = f"{symbol!r}"
                    if annotation := global_annotations.get(symbol_name):
                        if isinstance(annotation, type):
                            annotation_code = (
                                f": {annotation.__module__}.{annotation.__name__}"
                            )
                        else:
                            annotation_code = f": {annotation!r}"
                        compile_wrapper.writeline(
                            f"{symbol_name}{annotation_code} = {symbol_str}"
                        )
                    else:
                        compile_wrapper.writeline(f"{symbol_name} = {symbol_str}")
                    symbols_included.add(symbol_name)
                elif (
                    symbol_name in unqualified_loads
                    and symbol_name != "tl"  # already imported
                    and hasattr(symbol, "__module__")
                    # only codegen imports from triton; JITFunctions
                    # imported from other modules will be codegened
                    # in the separate branch above
                    and symbol.__module__.startswith("triton")
                ):
                    # a global symbol imported from triton is referenced
                    # without module qualification (i.e., `store` instead
                    # of `tl.store`): need to codegen an import
                    compile_wrapper.writeline(
                        f"from {symbol.__module__} import {symbol.__name__} as {symbol_name}"
                    )
                    symbols_included.add(symbol_name)

    traverse(kernel)
    return compile_wrapper.getvalue()


@dataclasses.dataclass
class SymbolicCallArg:
    inner: sympy.Symbol
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: dict[ReuseKey, list[FreeIfNotReusedLine]] = (
            collections.defaultdict(list)
        )
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key: ReuseKey) -> bool:
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        assert not item.is_reused
        self.reuse_pool[key].append(item)


class WrapperLine:
    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        raise NotImplementedError(f"FX codegen not yet supported for type {type(self)}")


@dataclasses.dataclass
class EnterSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    graph: GraphLowering

    def __post_init__(self) -> None:
        self.wrapper.push_computed_sizes(self.wrapper.computed_sizes)

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.push_codegened_graph(self.graph)
        code.do_indent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_enter_subgraph


@dataclasses.dataclass
class ConditionalLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.Conditional

    def codegen(self, code: IndentedBuffer) -> None:
        raise NotImplementedError("Only supports FX codegen")

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        return converter._generate_conditional


@dataclasses.dataclass
class CommentLine(WrapperLine):
    line: LineContext

    def codegen(self, code: IndentedBuffer) -> None:
        code.writeline(self.line)

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comment


@dataclasses.dataclass
class DynamicScalarLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.DynamicScalar

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._codegen_dynamic_scalar(self.node)

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        return converter._generate_dynamic_scalar


@dataclasses.dataclass
class ExitSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def __post_init__(self) -> None:
        self.wrapper.computed_sizes = self.wrapper.pop_computed_sizes()

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.pop_codegened_graph()
        code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_exit_subgraph


@dataclasses.dataclass
class EnterDeviceContextManagerLine(WrapperLine):
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer) -> None:
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if V.graph.aot_mode:
                # In AOT mode, we have a stream provided as a param. A stream is
                # associated with a device, so we never expect the device to change.
                # CUDAStreamGuard sets the stream and the device.
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"{V.graph.device_ops.cpp_aoti_stream_guard()} stream_guard(stream, this->device_idx_);"
                    )
                else:
                    assert self.last_seen_device_guard_index == self.device_idx, (
                        "AOTInductor only supports running on one CUDA device"
                    )
            else:
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"{V.graph.device_ops.cpp_aoti_device_guard()} device_guard({self.device_idx});"
                    )
                else:
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_enter_device_context_manager


class ExitDeviceContextManagerLine(WrapperLine):
    def codegen(self, code: IndentedBuffer) -> None:
        if not V.graph.cpp_wrapper:
            code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_exit_device_context_manager


@dataclasses.dataclass
class ExternKernelAllocLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelAlloc

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs()]
        self.wrapper._generate_extern_kernel_alloc_helper(self.node, args)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_extern_kernel_alloc


@dataclasses.dataclass
class ExternKernelOutLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelOut

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs(skip_out=True)]
        kernel_name = node.get_kernel_name()
        if (
            V.graph.cpp_wrapper
            and node.cpp_kernel_name == "torch::inductor::_mm_plus_mm"
        ):
            # For https://github.com/pytorch/pytorch/issues/128474
            kernel_name = "aoti_torch__mm_plus_mm_out"
        else:
            kernel_name = node.get_kernel_name()
        device = d.type if (d := node.get_device()) else V.graph.device_type
        self.wrapper._generate_extern_kernel_out_helper(
            kernel_name,
            node.codegen_reference(),
            node.output_view.codegen_reference() if node.output_view else None,
            args,
            device,
            self.node.get_stack_traces(),
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_extern_kernel_out


@dataclasses.dataclass
class FreeLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: Union[BufferLike, ir.TorchBindObject]

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        code.writeline(self.wrapper.make_buffer_free(self.node))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_free


@dataclasses.dataclass
class KernelCallLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    kernel_name: str
    call_args: tuple[Any, ...]
    raw_keys: tuple[Any, ...]
    raw_args: tuple[Any, ...]
    arg_types: list[str]
    triton: bool
    triton_meta: dict[str, Any]
    device: torch.device
    graph_name: str
    original_fxnode_name: str

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._generate_kernel_call_helper(
            self.kernel_name,
            self.call_args,
            triton=self.triton,
            arg_types=self.arg_types,
            raw_keys=self.raw_keys,
            raw_args=self.raw_args,
            triton_meta=self.triton_meta,
            device=self.device,
            graph_name=self.graph_name,
            original_fxnode_name=self.original_fxnode_name,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_kernel_call


@dataclasses.dataclass
class KernelDefinitionLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    kernel_name: str
    kernel_body: str
    metadata: Optional[str] = None
    gpu: bool = True
    cpp_definition: Optional[str] = None

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._define_kernel_helper(
            self.kernel_name,
            self.kernel_body,
            metadata=self.metadata,
            gpu=self.gpu,
            cpp_definition=self.cpp_definition,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_kernel_definition


@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        """Second pass to output code"""

    def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        args: list[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"


class EfficientPeakEstimate:
    def __init__(self):
        from ..memory import estimate_peak_memory, get_freeable_input_buf

        scheduler_nodes = V.graph.scheduler.nodes
        graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
        graph_outputs = OrderedSet(V.graph.get_output_names())
        names_to_freeable_bufs = get_freeable_input_buf(scheduler_nodes, graph_inputs)
        self.overall_peak_memory, peak_by_scheduler_node = estimate_peak_memory(
            scheduler_nodes,
            names_to_freeable_bufs,
            graph_outputs,
        )

        from .segmented_tree import SegmentedTree

        self.segmented_tree = SegmentedTree(
            peak_by_scheduler_node, operator.add, max, 0
        )

    def _get_size(self, node: BufferLike) -> int:
        return V.graph.sizevars.size_hint(
            V.graph.get_allocation_storage_size(node), fallback=0
        ) * get_dtype_size(node.get_dtype())

    def peak_between(self, line_a: FreeIfNotReusedLine, line_b: AllocateLine):
        return self.segmented_tree.summarize_range(
            line_a.scheduler_node_index + 1, line_b.scheduler_node_index - 1
        )

    def update_peak_between(self, line_a: FreeIfNotReusedLine, line_b: AllocateLine):
        if line_a.scheduler_node_index + 1 == line_b.scheduler_node_index:
            return
        self.segmented_tree.update_range(
            line_a.scheduler_node_index + 1,
            line_b.scheduler_node_index - 1,
            self._get_size(line_b.node),
        )


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: BufferLike

    def __post_init__(self):
        assert V.graph.scheduler.current_node is not None
        self.scheduler_node_index = V.graph.scheduler.nodes.index(
            V.graph.scheduler.current_node
        )

    def should_reuse_buffer(self, free_line: FreeIfNotReusedLine, size: int) -> bool:
        if free_line.scheduler_node_index + 1 == self.scheduler_node_index:
            return True
        overall_peak_memory = self.wrapper.estimate_peak.overall_peak_memory
        peak_memory_in_range = self.wrapper.estimate_peak.peak_between(free_line, self)
        new_peak_memory = size + peak_memory_in_range
        return new_peak_memory <= overall_peak_memory

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            size = V.graph.sizevars.size_hint(
                V.graph.get_allocation_storage_size(self.node), fallback=0
            ) * get_dtype_size(self.node.get_dtype())
            if self.should_reuse_buffer(free_line, size):
                free_line.is_reused = True
                self.wrapper.estimate_peak.update_peak_between(free_line, self)
                return ReuseLine(self.wrapper, free_line.node, self.node)
            else:
                state.push(key, free_line)
                return self

        if self.node.get_device_or_error().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_allocate


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: BufferLike
    is_reused: bool = False

    def __post_init__(self):
        assert V.graph.scheduler.current_node is not None
        self.scheduler_node_index = V.graph.scheduler.nodes.index(
            V.graph.scheduler.current_node
        )

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_free_if_not_reused


@dataclasses.dataclass
class ReinterpretLine(MemoryPlanningLine):
    node: BufferLike
    reused_as: BufferLike
    layout: ir.Layout

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert isinstance(self.layout, ir.NonOwningLayout)
        assert isinstance(self.layout.view, ir.ReinterpretView)
        self.wrapper.codegen_deferred_allocation(
            self.reused_as.get_name(), self.layout.view
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_reinterpret


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: BufferLike
    reused_as: BufferLike
    delete_old: bool = True

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_reuse


class NullLine(MemoryPlanningLine):
    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_null


@dataclasses.dataclass
class CommBufferLine(WrapperLine):
    wrapper: PythonWrapperCodegen  # type: ignore[name-defined] # noqa: F821
    node: ir.Buffer

    @property
    def size(self) -> int:
        from torch._inductor.utils import is_symbolic

        numel = self.node.get_numel()
        dtype = self.node.get_dtype()
        if is_symbolic(numel):
            raise AssertionError(
                f"The size of a comm buffer can't be symbolic: {self.node}"
            )
        return int(numel) * dtype.itemsize

    @property
    def comm_buffer_type(self) -> ir.CommBufferType:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.comm_buffer_type

    @property
    def group_name(self) -> str:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.group_name


@dataclasses.dataclass
class CommBufferAllocateLine(CommBufferLine):
    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        name = self.node.get_name()
        device = self.node.get_device()
        dtype = self.node.get_dtype()
        shape = tuple(self.node.get_size())
        stride = tuple(self.node.get_stride())
        code.writeline(
            self.make_allocation_line(
                self.comm_buffer_type,
                self.group_name,
                self.wrapper,
                name,
                device,
                dtype,
                shape,
                stride,
            )
        )

    @staticmethod
    def make_allocation_line(
        comm_buffer_type, group_name, wrapper, name, device, dtype, shape, stride
    ):
        if comm_buffer_type == ir.CommBufferType.SYMM_MEM:
            return (
                f"{name} = empty_strided_p2p("
                f"{wrapper.codegen_shape_tuple(shape)}, "
                f"{wrapper.codegen_shape_tuple(stride)}, "
                f"{dtype}, "
                f'torch.device("cuda:{device.index}"), '
                f'group_name="{group_name}", '
                f"alloc_id={random.randint(0, 2**64 - 1)})"
            )
        else:
            raise NotImplementedError(
                f"Unsupported comm buffer type: {comm_buffer_type}"
            )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comm_buffer_allocate


@dataclasses.dataclass
class CommBufferFreeLine(CommBufferLine):
    def codegen(self, code: IndentedBuffer) -> None:
        line = self.wrapper.make_buffer_free(self.node)
        code.writeline(f"{line} # {self.comm_buffer_type.value} buffer free")

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comm_buffer_free


@dataclasses.dataclass
class MultiOutputLine(WrapperLine):
    """
    Given a MultiOutputLayout buffer, indexes actual buffer(s) from the result.
    """

    wrapper: PythonWrapperCodegen
    result_name: str
    arg_name: str
    indices: Sequence[Any]

    def codegen(self, code: IndentedBuffer) -> None:
        def codegen_list_tuple_access(basename, indices):  # type: ignore[no-untyped-def]
            if len(indices) > 0:
                itype, i = indices[0]
                if issubclass(itype, list):
                    return codegen_list_tuple_access(f"{basename}[{i}]", indices[1:])
                elif issubclass(itype, tuple):
                    # cpp wrapper code needs to use std::get<> to access a tuple
                    tuple_access = self.wrapper.codegen_tuple_access(
                        basename, self.result_name, str(i)
                    )
                    return codegen_list_tuple_access(tuple_access, indices[1:])
                elif issubclass(itype, dict):
                    return codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
                else:
                    raise AssertionError("non supported index type: ", itype)
            else:
                return basename

        value = codegen_list_tuple_access(self.arg_name, self.indices)
        code.writeline(
            f"{self.wrapper.declare}{self.result_name} = {value}{self.wrapper.ending}"
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_multi_output


@dataclasses.dataclass
class IndexPutFallbackLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.IndexPutFallback
    indices: list[Optional[ir.IRNode]]

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        assert ir.is_node_sequence(node.inputs)
        (x, values) = (t.codegen_reference() for t in node.inputs[:2])
        indices = [
            idx.codegen_reference() if idx else self.wrapper.none_str
            for idx in self.indices
        ]

        self.wrapper._generate_index_put_fallback(
            node.get_kernel_name(), x, indices, values, *node.codegen_const_args()
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_index_put_fallback


@dataclasses.dataclass
class ScatterFallbackLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.ScatterFallback

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        assert ir.is_node_sequence(node.inputs)
        if node.src_is_tensor:
            (x, index, src) = (t.codegen_reference() for t in node.inputs)
        else:
            (x, index) = (t.codegen_reference() for t in node.inputs)
            src = node.constant_args[1]
        device = d.type if (d := node.get_device()) else V.graph.device_type
        self.wrapper._generate_scatter_fallback(
            x,
            [x, node.constant_args[0], index, src],
            node.cpp_kernel_name,
            node.python_kernel_name,
            node.src_is_tensor,
            node.kwargs["reduce"],
            node.codegen_kwargs(),
            device,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_scatter_fallback


@dataclasses.dataclass
class SymbolicCallArgLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    arg: SymbolicCallArg
    graph: GraphLowering

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._generate_symbolic_call_arg_helper(self.arg, self.graph)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_symbolic_call_arg


@dataclasses.dataclass
class UnbackedSymbolDefsLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    output_name: str
    outputs: Any
    unbacked_bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]]

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._codegen_unbacked_symbol_defs_for_outputs(
            self.output_name, self.outputs, self.unbacked_bindings
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_unbacked_symbol_defs


BufferName = str
Line = Union[MemoryPlanningLine, LineContext]


class PythonWrapperCodegen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    supports_caching = True  # Whether the output code is cacheable.

    def __init__(self):
        super().__init__()
        self._names_iter: Iterator[int] = count()
        self.args_to_buffers: dict[
            str, Union[None, ir.TensorBox, ir.Buffer, ir.TorchBindObject]
        ] = {}
        self.imports = IndentedBuffer()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.kernel_declarations = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.kernel_autotune_defs = IndentedBuffer()
        self.kernel_autotune_calls = IndentedBuffer()
        self.subgraph_definitions = IndentedBuffer()
        self.kernel_autotune_names: OrderedSet[str] = OrderedSet()
        # Map key is the kernel argument name; value is a tuple of the resulting example
        # tensor name with the kernel where that tensor was most recently used.
        self.kernel_autotune_example_args: dict[str, tuple[str, str]] = {}
        self.kernel_autotune_tmp_arg_idx: int = 0
        # If the generated source code is exactly the same, reuse the
        # pre-existing kernel for it
        self.src_to_kernel: dict[str, str] = {}
        self.kernel_numel_expr: OrderedSet[tuple[str, GraphLowering]] = OrderedSet()
        self.lines: list[Line] = []
        self.declare = ""
        self.declare_maybe_reference = ""
        self.ending = ""
        self.comment = "#"
        self.none_str = "None"
        self.move_begin = "std::move(" if V.graph.cpp_wrapper else ""
        self.move_end = ")" if V.graph.cpp_wrapper else ""
        self.last_seen_device_guard_index: Optional[int] = None
        self.supports_intermediate_hooks = True
        self.user_defined_kernel_cache: dict[tuple[Any, ...], tuple[str, Any]] = {}
        self.unbacked_symbol_decls: OrderedSet[str] = (
            OrderedSet()
        )  # str of sympy.Symbol
        self.computed_sizes: OrderedSet[sympy.Symbol] = OrderedSet()
        self.launcher_fn_name = None
        # This function can be overridden to change the launcher name
        self.set_launcher_fn_name()

        # this is used for tracking which GraphLowering instance---parent graph
        # or (nested) subgraph---is currently codegened; the primary use case is
        # including the graph instance into a cache key to avoid cross-graph
        # caching during lowering of nested subgraphs
        self.codegened_graph_stack = []
        self.computed_sizes_stack = []

        self.write_header()

        if not is_codegen_graph_partition_subgraph(self):
            # See [Note: Removed Graph Partition Arguments]
            self.write_prefix()

        self.write_kernel_autotune_defs_header()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated = OrderedSet[BufferName]()
        self.freed = OrderedSet[BufferName]()

        # maps from reusing buffer to reused buffer
        self.reuses: dict[BufferName, BufferName] = {}

        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

        @functools.cache
        def add_import_once(line: str) -> None:
            self.imports.writeline(line)
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(line)

        self.add_import_once = add_import_once
        self._metas: dict[str, str] = {}
        self._meta_vars: OrderedSet[str] = OrderedSet()
        self.multi_kernel_state = MultiKernelState()
        self.already_codegened_subgraphs: OrderedSet[str] = OrderedSet()
        self.allocated_workspaces: dict[str, Any] = {}

        # intermediate tensor value printing utility
        self.debug_printer = DebugPrinterManager(
            debug_printer_level=config.aot_inductor.debug_intermediate_value_printer,
            use_array_ref=config.aot_inductor.allow_stack_allocation,
        )

        # Additional files that are dependent to the wrapper (ex. cubin files)
        self.additional_files = []

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperCodegen()

    def set_launcher_fn_name(self) -> None:
        # pyrefly: ignore [bad-assignment]
        self.launcher_fn_name = "call"

    def write_constant(self, name: str, hashed: str) -> None:
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self) -> None:
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        inductor_debug_utils = ""
        if int(config.aot_inductor.debug_intermediate_value_printer) > 0:
            inductor_debug_utils = "from torch._inductor.codegen.debug_utils import _print_debugging_tensor_value_info"
        elif torch._inductor.config.test_configs.track_memory_lifecycle:
            inductor_debug_utils = "from torch._inductor.runtime.debug_utils import tracked_empty_strided\n"

        self.imports.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long, c_int
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from cmath import nanj
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align
                from torch import device, empty_strided
                from {async_compile.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                {inductor_debug_utils}
            """,
            strip=True,
        )
        self.header.splice(
            """
                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
                empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()
            """,
            strip=True,
        )
        try:
            # Only add empty_strided_p2p() if distributed and SymmetricMemory
            # is available
            from torch._C._distributed_c10d import _SymmetricMemory  # noqa: F401

            self.header.splice(
                """
                empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
                """,
                strip=True,
            )
        except (AttributeError, ImportError):
            pass
        if config.annotate_training:
            self.header.writeline("from torch.cuda import nvtx")

    def include_extra_header(self, header: str):
        pass

    def write_kernel_autotune_defs_header(self) -> None:
        self.kernel_autotune_defs.splice(
            f"""
                import torch
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
            """
        )

        try:
            from torch._C import _cuda_getCurrentRawStream  # noqa: F401

            self.kernel_autotune_defs.splice(
                """
                get_raw_stream = torch._C._cuda_getCurrentRawStream
                """,
                strip=True,
            )
        except (ImportError, AttributeError):
            pass

    @cache_on_self
    def write_triton_header_once(self) -> None:
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import start_graph, end_graph
            """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    def write_get_raw_stream_header(self) -> None:
        import_get_raw_stream_str = V.graph.device_ops.import_get_raw_stream_as(
            "get_raw_stream"
        )
        if config.triton.autotune_at_compile_time:
            if not self.kernel_autotune_calls.contains(import_get_raw_stream_str):
                self.kernel_autotune_calls.writeline(import_get_raw_stream_str)
        if not V.graph.cpp_wrapper:
            if not self.imports.contains(import_get_raw_stream_str):
                self.imports.writeline(import_get_raw_stream_str)

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        self.write_get_raw_stream_header()

    def add_meta_once(self, meta: TritonMetaParams) -> str:
        # pyrefly: ignore [bad-assignment]
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            # pyrefly: ignore [unsupported-operation]
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(f"{var} = {meta}")
                self._meta_vars.add(var)
        # pyrefly: ignore [index-error]
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self) -> list[str]:
        return [
            x.codegen_reference(self.wrapper_call) for x in self.get_graph_outputs()
        ]

    def mark_output_type(self) -> None:
        return

    def get_graph_inputs(
        self,
    ) -> dict[str, Union[ir.TensorBox, ir.TorchBindObject, sympy.Expr]]:
        return V.graph.graph_inputs

    def get_graph_outputs(self) -> list[IRNode]:
        return V.graph.graph_outputs

    def codegen_input_size_asserts(self) -> None:
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue

            # a graph partition may take an IRNode output from a previous partition
            if name not in V.graph.graph_input_names or isinstance(
                buf, ir.GeneratorState
            ):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_python_shape_tuple(buf.get_size())
            stride = self.codegen_python_shape_tuple(buf.get_stride())
            self.prefix.writeline(f"assert_size_stride({name}, {size}, {stride})")

    def codegen_input_nan_asserts(self) -> None:
        self.prefix.writeline("# make sure graph inputs are not nan/inf")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue

            line = f"assert not {name}.isnan().any().item()"
            self.prefix.writeline(line)
            line = f"assert not {name}.isinf().any
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
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `wrapper.py_docs.md_docs.md`
- **Keyword Index**: `wrapper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
