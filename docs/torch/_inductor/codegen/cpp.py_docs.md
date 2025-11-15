# Documentation: `torch/_inductor/codegen/cpp.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/cpp.py`
- **Size**: 234,718 bytes (229.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import contextlib
import dataclasses
import functools
import itertools
import math
import operator
import re
import sys
import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, cast, Optional, Union

import sympy

import torch
import torch.fx
from torch._inductor import dependencies
from torch._prims_common import is_float_dtype, is_integer_dtype
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from ..._dynamo.utils import counters
from .. import config, cpp_builder, cpu_vec_isa, ir, metrics
from ..debug import set_kernel_post_grad_provenance_tracing
from ..loop_body import LoopBody
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    ExternKernelSchedulerNode,
    ForeachKernelSchedulerNode,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from ..utils import (
    cache_on_self,
    get_bounds_index_expr,
    get_fused_kernel_name,
    has_free_symbols,
    is_multi_outputs_template,
    is_welford_reduction,
    parallel_num_threads,
    Placeholder,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
)
from ..virtualized import NullKernelHandler, ops, OpsValue, V
from .common import (
    BackendFeature,
    BracesBuffer,
    CSE,
    CSEVariable,
    DataTypePropagation,
    DeferredLine,
    DTYPE_TO_COMPUTATION_DTYPE,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
    OptimizationContext,
)
from .cpp_utils import (
    _get_dtype_from_loopbodies,
    _get_loop_body,
    cexpr,
    cexpr_index,
    codegen_rand,
    CppCSEVariable,
    DTYPE_TO_CPP,
    get_promote_dtype,
    INDEX_TYPE,
    LocalBufferContext,
    may_unify_binary_op_mask_type,
    promote_args,
    template_fusion_with_epilogues_supported,
    unify_mask_base_type,
    value_to_cpp,
)


_IS_WINDOWS = sys.platform == "win32"


@functools.cache
def get_export_declaration():
    return "__declspec(dllexport)" if _IS_WINDOWS else ""


schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

NATIVE_OMP_RTYPES = OrderedSet(["+", "*", "^", "||", "min", "max"])
RTYPE_TO_CPP = {
    "sum": "+",
    "prod": "*",
    "xor_sum": "^",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
    "welford_reduce": "welford",
    "welford_combine": "welford",
}
VECTORIZABLE_RTYPES = OrderedSet(
    [
        "max",
        "min",
        "sum",
        "prod",
        "xor_sum",
        "welford_reduce",
        "welford_combine",
        "argmin",
        "argmax",
        "any",
    ]
)

PYTHON_TO_CPP = {
    "Tensor": "at::Tensor",
    "int": "long",
    "float": "double",
    "bool": "bool",
    "str": "std::string",
    "ScalarType": "c10::ScalarType",
    "MemoryFormat": "at::MemoryFormat",
    "Layout": "at::Layout",
    "Device": "at::Device",
    "number": "at::Scalar",
}

CONTAINER_PYTHON_TO_CPP = {
    "List": "std::vector",
    "Optional": "std::optional",
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]

VECTORIZABLE_DTYPES: list[torch.dtype] = [
    torch.float64,
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

MASKED_VECTORIZABLE_DTYPES: list[torch.dtype] = [
    torch.float64,
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


def reduction_init(reduction_type, dtype):
    if dtype in DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type in ("max", "argmax", "min", "argmin"):
        cdtype = DTYPE_TO_CPP[dtype]
        if dtype == torch.bool and reduction_type in ("argmin", "argmax"):
            cdtype = DTYPE_TO_CPP[torch.float]
        min_var = (
            f"-std::numeric_limits<{cdtype}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{cdtype}>::min()"
        )
        max_var = (
            f"std::numeric_limits<{cdtype}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{cdtype}>::max()"
        )
        init_var = min_var if reduction_type in ("max", "argmax") else max_var
        return (
            init_var
            if reduction_type in ("max", "min")
            else f"IndexValue<{cdtype}>{{0, {init_var}}}"
        )
    if is_welford_reduction(reduction_type):
        return f"Welford<{DTYPE_TO_CPP[dtype]}>()"
    raise AssertionError(reduction_type)


def reduction_acc_type(reduction_type, dtype):
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    if is_welford_reduction(reduction_type):
        return f"Welford<{scalar_type}>"
    if reduction_type in ("argmin", "argmax"):
        if dtype == torch.bool:
            scalar_type = DTYPE_TO_CPP[torch.float]
        return f"IndexValue<{scalar_type}>"
    return scalar_type


def reduction_combine(
    reduction_type,
    var,
    next_value,
    helper_val=None,
    index: Optional[sympy.Symbol] = None,
    src_dtype=None,
):
    is_bool = src_dtype == torch.bool
    if reduction_type == "sum":
        if helper_val:
            return f"cascade_sum_combine({next_value}, &{helper_val})"
        else:
            conjunction = "|" if is_bool else "+"
            return f"{var} {conjunction} {next_value}"
    if reduction_type == "prod":
        return f"{var} * {next_value}"
    if reduction_type == "xor_sum":
        return f"{var} ^ {next_value}"
    if reduction_type == "any":
        return f"{var} || {next_value}"
    if reduction_type in ("min", "max"):
        return f"{reduction_type}_propagate_nan({var}, {next_value})"
    if reduction_type == "welford_reduce":
        return f"welford_combine({var}, {next_value})"
    if reduction_type == "welford_combine":
        if isinstance(next_value, tuple):
            mean, m2, weight = next_value
        else:
            mean, m2, weight = reduction_project(reduction_type, next_value)
        return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
    if reduction_type in ("argmin", "argmax"):
        if (
            hasattr(next_value, "dtype")
            and next_value.dtype == torch.bool
            and not next_value.is_vec
        ):
            if index is not None:
                return f"{reduction_type}_combine({var}, static_cast<float>({next_value}), {index})"
            else:
                return (
                    f"{reduction_type}_combine({var}, static_cast<float>({next_value}))"
                )
        if index is not None:
            return f"{reduction_type}_combine({var}, {next_value}, {index})"
        else:
            return f"{reduction_type}_combine({var}, {next_value})"
    raise AssertionError(reduction_type)


def reduction_project(reduction_type, acc):
    if is_welford_reduction(reduction_type):
        return f"{acc}.mean", f"{acc}.m2", f"{acc}.weight"
    elif reduction_type in ("argmin", "argmax"):
        return f"{acc}.index"
    return acc


def move_code_under_inner_loop(
    code: IndentedBuffer,
    iter_var: sympy.Expr,
    new_iter_var: str,
    loop_start: sympy.Expr,
    loop_end: sympy.Expr,
) -> BracesBuffer:
    r"""
    f(iter_var) is transformed to f(new_iter_var) under the inner loop
      \/
    for (new_iter_var = loop_start; new_iter_var < loop_end; new_iter_var++) {
        f(new_iter_var)
    }
    Please be careful while using this function,
    as the variable defined in f(iter_var) will be invalid outside the for loop.
    For example:
    auto tmp0 = in_ptr[x0]; ->
    for (new_x0 = start; new_x0 < end; new_x0++){
        auto tmp0 = in_ptr[new_x0];
    }
    The tmp0 is invalid outside the loop.
    """
    transformed_code = BracesBuffer()
    with contextlib.ExitStack() as stack:
        transformed_code.writeline(
            f"for ({INDEX_TYPE} {new_iter_var} = {cexpr_index(loop_start)};"
            + f"{new_iter_var} < {cexpr_index(loop_end)}; {new_iter_var}++)"
        )
        stack.enter_context(transformed_code.indent())
        for _, line in enumerate(code._lines):
            assert isinstance(
                line,
                (
                    str,
                    DeferredLine,
                ),
            )
            deferred_name = None
            if isinstance(line, DeferredLine):
                deferred_name = line.name
                line = line.line
            new_line = re.sub(r"\b" + f"{iter_var}" + r"\b", f"{new_iter_var}", line)
            if deferred_name:
                new_line = DeferredLine(deferred_name, new_line)  # type: ignore[assignment]
            transformed_code.writeline(new_line)
    return transformed_code


def reduction_prefix_array(
    acc_var: Union[str, CSEVariable],
    acc_type: str,
    reduction_type: str,
    dtype: torch.dtype,
    len: Union[str, int],
    init_fn,
):
    """
    MSVC don't support dynamic array(VLA). So we use std::unique_ptr here.
    Ref: https://stackoverflow.com/questions/56555406/creating-dynamic-sized-array-using-msvc-c-compiler
    MSVC is the only one compiler without VLA. support. Since MSVC can't get good performance here.
    We just use unique_ptr make it works on MSVC.
    For other compilers, we continue to use VLA to get best performance.
    """
    code_buffer = IndentedBuffer()
    acc_decl = (
        f"auto {acc_var}_arr = std::make_unique<{acc_type}[]>({len});"
        if cpp_builder.is_msvc_cl()
        else f"{acc_type} {acc_var}_arr[{len}];"
    )
    code_buffer.writeline(f"{acc_decl}")
    code_buffer.writelines(
        [
            f"for (int i = 0; i < {len}; i++)",
            "{",
            f"    {acc_var}_arr[i] = {init_fn(reduction_type, dtype)};",
            "}",
        ],
    )
    return code_buffer


def replace_acc_name(buffer: IndentedBuffer, name: str, new_name: str):
    for i, line in enumerate(buffer._lines):
        assert isinstance(
            line,
            (
                str,
                DeferredLine,
            ),
        )
        if isinstance(line, DeferredLine):
            line.line = re.sub(r"\b" + f"{name}" + r"\b", f"{new_name}", line.line)
        else:
            buffer._lines[i] = re.sub(r"\b" + f"{name}" + r"\b", f"{new_name}", line)


def replace_cascade_sum_with_add(buffer: IndentedBuffer):
    """
    Replaces `acc = cascade_sum_combine(value, ...)` with `acc = acc + value;`
    """

    pattern = r"(.*?)\s*=\s*cascade_sum_combine\(([^,]+),.*?\);"
    for i, line in enumerate(buffer._lines):
        assert isinstance(
            line,
            (
                str,
                DeferredLine,
            ),
        )
        content = line.line if isinstance(line, DeferredLine) else line
        match = re.search(pattern, content)
        if match:
            acc, value = match.groups()
            new_content = re.sub(pattern, f"{acc} = {acc} + {value};", content)
            if isinstance(line, DeferredLine):
                line.line = new_content
            else:
                buffer._lines[i] = new_content


@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
    if not index.has(var):
        # see test_torchinductor_dynamic_shapes.py::test_full_boolean_dynamic_shapes_cpu
        # which has tmp0 = ops.index_expr(s0 >= 1024, torch.bool) and fails below calculation.
        # in this case, there is no dependencies between index and var.
        return sympy.S.Zero
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)  # type: ignore[arg-type]
    return sympy.simplify(new_index - index)


@functools.lru_cache
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(
    index: sympy.Expr, var: sympy.Symbol, vec_length: Optional[int] = None
):
    if vec_length:
        index = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index, var)


@dataclasses.dataclass
class ParallelDepth:
    """
    A class representing parallel depth.
    Includes the starting depth of parallelism and the depth of parallelism.
    """

    parallel_depth: int
    start_depth: int


class OuterLoopFusedSchedulerNode(FusedSchedulerNode):
    @classmethod
    def fuse(  # type: ignore[override]
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode, outer_loop_fusion_depth
    ):
        assert node1.scheduler is node2.scheduler
        assert all(
            type(node)
            in (
                OuterLoopFusedSchedulerNode,
                SchedulerNode,
                FusedSchedulerNode,
            )
            for node in (node1, node2)
        )
        if any(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return cls(
                node1.scheduler,
                # pyrefly: ignore [bad-argument-type]
                (
                    list(node1.get_outer_nodes())
                    if type(node1) is OuterLoopFusedSchedulerNode
                    else [
                        node1,
                    ]
                )
                + (
                    list(node2.get_outer_nodes())
                    if type(node2) is OuterLoopFusedSchedulerNode
                    else [
                        node2,
                    ]
                ),
                outer_loop_fusion_depth,
            )
        else:
            return cls(node1.scheduler, [node1, node2], outer_loop_fusion_depth)  # type: ignore[list-item]

    def __init__(
        self,
        scheduler: "Scheduler",
        outer_fused_nodes: list[Union[FusedSchedulerNode, SchedulerNode]],
        outer_loop_fusion_depth,
    ):
        self.outer_fused_nodes: list[Union[FusedSchedulerNode, SchedulerNode]] = (
            outer_fused_nodes
        )
        self.outer_loop_fusion_depth = outer_loop_fusion_depth
        flatten_snodes = []
        for _node in self.outer_fused_nodes:
            assert isinstance(_node, (SchedulerNode, FusedSchedulerNode))
            flatten_snodes.extend(list(_node.get_nodes()))
        super().__init__(scheduler, flatten_snodes)  # type: ignore[arg-type]

    def get_outer_nodes(self):
        return self.outer_fused_nodes

    def check_outer_fusion_loop_level_attr(
        self, cpp_kernel_proxy_list, outer_loop_fusion_depth
    ):
        # This function ensures that the same tiling split is applied at each loop level within the outer loop fusion depth.
        # In the fusion stage, we only examine nodes with same vars and reduce.
        # However, for nodes with same vars and reduce, the loops may still have different tile splits.
        # For example (test_expr_vec_non_contiguous in test_cpu_repro.py):
        #   * buf0 tiling along the 2nd loop level, buf1 tiling along the 3rd loop level.
        # If the check failed, we should fall back to standard loop codegen.
        def _inner(
            left_loop_nest: LoopNest,
            right_loop_nest: LoopNest,
            loop_fusion_depth: int,
            current_checking_depth: int,
        ) -> bool:
            assert left_loop_nest.loops
            assert right_loop_nest.loops
            left_loop_level = left_loop_nest.loops[current_checking_depth]
            right_loop_level = right_loop_nest.loops[current_checking_depth]
            # Check if same loop level attr
            outer_loops_attr_compare_list = [
                "var",
                "size",
                "offset",
                "steps",
            ]
            if not (
                all(
                    getattr(left_loop_level, attr_compare)
                    == getattr(right_loop_level, attr_compare)
                    for attr_compare in outer_loops_attr_compare_list
                )
            ):
                return False

            assert loop_fusion_depth >= 1
            if (loop_fusion_depth := loop_fusion_depth - 1) > 0:
                # Check next loop level attr
                current_checking_depth = current_checking_depth + 1
                assert current_checking_depth < len(left_loop_nest.loops)
                assert current_checking_depth < len(right_loop_nest.loops)
                if not _inner(
                    left_loop_nest,
                    right_loop_nest,
                    loop_fusion_depth,
                    current_checking_depth,
                ):
                    return False

            return True

        for idx in range(len(cpp_kernel_proxy_list) - 1):
            left_loop_nest = cpp_kernel_proxy_list[idx].loop_nest
            right_loop_nest = cpp_kernel_proxy_list[idx + 1].loop_nest
            if not _inner(
                left_loop_nest,
                right_loop_nest,
                outer_loop_fusion_depth,
                0,
            ):
                return False

        for cpp_kernel_proxy in cpp_kernel_proxy_list:
            outer_ranges = functools.reduce(
                operator.mul,
                cpp_kernel_proxy.ranges[:outer_loop_fusion_depth],
            )
            # When the range of the first inner loop is much larger than the range of
            # all outer loops, do not fuse outer loop and fallback to standard loop codegen,
            # so that the inner loops with larger range have a chance to be parallelized.
            # We set a conservative threshold here:
            # First inner loop range / all outer loops range > 300.
            if (
                len(cpp_kernel_proxy.ranges) > outer_loop_fusion_depth
                and isinstance(outer_ranges, sympy.Integer)
                and isinstance(
                    cpp_kernel_proxy.ranges[outer_loop_fusion_depth],
                    sympy.Integer,
                )
                and outer_ranges * 300
                < cpp_kernel_proxy.ranges[outer_loop_fusion_depth]
            ):
                return False

        return True

    def merge_outer_fusion_kernels(
        self,
        cpp_kernel_proxy_list,
    ):
        kernel_group = cpp_kernel_proxy_list[0].kernel_group
        outer_loop_fused_kernel = OuterLoopFusedKernel(kernel_group)
        outer_loop_fused_kernel.inner = [
            proxy.loop_nest.from_loop_level(self.outer_loop_fusion_depth)
            for proxy in cpp_kernel_proxy_list
        ]
        outer_fused_proxy = cpp_kernel_proxy_list[0]
        outer_fused_proxy.loop_nest.kernel = outer_loop_fused_kernel
        outer_fused_proxy.loop_nest.loops = outer_fused_proxy.loop_nest.loops[
            : self.outer_loop_fusion_depth
        ]
        return outer_fused_proxy


class RecordOptimizationContext:
    def __init__(self, func_name: str = ""):
        self.func_name = func_name
        self.current_node: Optional[torch.fx.Node] = None
        self.opt_ctx: Optional[OptimizationContext] = None

    def __enter__(self):
        assert V.interpreter
        assert V.interpreter.current_node

        self.current_node = V.interpreter.current_node
        assert self.current_node is not None
        if OptimizationContext.key in self.current_node.meta:
            self.opt_ctx = self.current_node.meta[OptimizationContext.key]
        else:
            self.opt_ctx = OptimizationContext()
        assert self.opt_ctx is not None
        self.opt_ctx.ops_name = self.func_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.current_node
        assert self.opt_ctx
        self.current_node.meta[OptimizationContext.key] = self.opt_ctx

    def get_opt_ctx(self):
        return self.opt_ctx

    def get_fx_node(self):
        assert self.current_node
        return self.current_node


def decltype_promoted(*args):
    assert not any(isinstance(arg, CppCSEVariable) and arg.is_vec for arg in args), (
        "Promotion of vector types is not supported"
    )

    if (dt := get_promote_dtype(args)) is not None:
        return DTYPE_TO_CPP[dt]
    else:
        return f"decltype({args[0]})"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def add(a, b):
        return f"{decltype_promoted(a, b)}({a} + {b})"

    @staticmethod
    def sub(a, b):
        return f"{decltype_promoted(a, b)}({a} - {b})"

    @staticmethod
    def mul(a, b):
        return f"{decltype_promoted(a, b)}({a} * {b})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        assert isinstance(x, CppCSEVariable)
        if src_dtype is None:
            src_dtype = x.dtype
        expr = V.kernel.get_to_dtype_expr(x, dtype, src_dtype)
        csevar = V.kernel.cse.generate(V.kernel.compute, expr)
        csevar.update_on_args("to_dtype", (x, dtype), {"src_dtype": src_dtype})
        if dtype in DTYPE_LOWP_FP and src_dtype == torch.float:
            """
            https://github.com/pytorch/pytorch/issues/115260
            For FusedSchedulerNode[node1, node2], the node2 loads what node1 stores and the buffer is
            in low-precision floating point data type. When the output of node1 also serves as the output of the
            kernel, the result of nodes would be different from the case when output of node1 is not the output
            of the kernel (where we don't need to insert `to_dtype` for legalization). To address the problem, on
            storing the lowp node1 output, we also add the inverse dtype conversion to high precision data type
            to the cse cache.

            Example (pseudo code):
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                store(buf, node1_output_lowp)
                node2_input_lowp = load(buf)
                node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

            Without cse cache trick:
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                store(buf, node1_output_lowp)
                node2_input_lowp = node_output_lowp # hit store cache
                node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

            With cse cache trick:
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                # also add `to_dtype(node1_input_lowp, dtype=torch.float)` -> `node1_output` to cse cache
                store(buf, node1_output_lowp)
                node2_input_lowp = node_output_lowp # hit store cache
                node2_input = node1_output # hit cse cache
            """
            V.kernel.cache_dtype_convert(x, src_dtype, csevar, dtype)
        return csevar

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def neg(x):
        return f"decltype({x})(-{x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def exp2(x):
        return f"std::exp2({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def erfc(x):
        return f"std::erfc({x})"

    @staticmethod
    def erfinv(x):
        return f"calc_erfinv({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::log1p({x})"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def tan(x):
        return f"std::tan({x})"

    @staticmethod
    def tanh(x):
        return f"std::tanh({x})"

    @staticmethod
    def signbit(x):
        """
        On windows std::signbit only support float type.
        Ref: https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/signbit?view=msvc-170
        """
        return (
            f"std::signbit(static_cast<float>({x}))"
            if _IS_WINDOWS
            else f"std::signbit({x})"
        )

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def acos(x):
        return f"std::acos({x})"

    @staticmethod
    def acosh(x):
        return f"std::acosh({x})"

    @staticmethod
    def cosh(x):
        return f"std::cosh({x})"

    @staticmethod
    def sinh(x):
        return f"std::sinh({x})"

    @staticmethod
    def asin(x):
        return f"std::asin({x})"

    @staticmethod
    def asinh(x):
        return f"std::asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"std::atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"std::atan({x})"

    @staticmethod
    def atanh(x):
        return f"std::atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"std::copysign({x}, {y})"

    @staticmethod
    def frexp(x):
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(V.kernel.cse.try_get(cache_key) is not None for cache_key in cache_keys):
            return tuple(V.kernel.cse.try_get(cache_key) for cache_key in cache_keys)

        code = BracesBuffer()
        exponent = V.kernel.cse.newvar(dtype=torch.int32, shape=x.shape)
        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        code.writeline(f"int32_t {exponent};")
        code.writeline(f"auto {mantissa} = std::frexp({x}, &{exponent});")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.put(cache_key, cse_var)
        return mantissa, exponent

    @staticmethod
    def hypot(x, y):
        return f"std::hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"std::log10({x})"

    @staticmethod
    def log2(x):
        return f"std::log2({x})"

    @staticmethod
    def nextafter(x, y):
        return f"std::nextafter({x}, {y})"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::max({x}, decltype({x})(0))"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        return f"min_propagate_nan({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"max_propagate_nan({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        return value_to_cpp(val, DTYPE_TO_CPP[dtype])

    @staticmethod
    def index_expr(expr, dtype):
        idx_str = cexpr(V.kernel.rename_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()

        # Write masked operation into a lambda
        body_var = V.kernel.cse.newvar()
        code.writeline(f"auto {body_var} = [&]")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        # Use the lambda's return type as the type of other
        other_code = value_to_cpp(other, f"decltype({body_var}())")
        return f"{mask} ? {body_var}() : {other_code}"

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_not(a):
        return f"!{a}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"{a} != {b}"

    @staticmethod
    def bitwise_and(a, b):
        return f"decltype({a})({a} & {b})"

    @staticmethod
    def bitwise_not(a):
        return f"decltype({a})(~{a})"

    @staticmethod
    def bitwise_or(a, b):
        return f"decltype({a})({a} | {b})"

    @staticmethod
    def bitwise_xor(a, b):
        return f"decltype({a})({a} ^ {b})"

    @staticmethod
    def bitwise_left_shift(a, b):
        code = BracesBuffer()
        code.writeline("[&]()")
        with code.indent():
            scalar_t = DTYPE_TO_CPP[a.dtype]
            code.writeline(
                f"constexpr decltype({b}) max_shift = sizeof({scalar_t}) * CHAR_BIT;"
            )
            code.writeline(
                f"if ((static_cast<std::make_signed_t<{scalar_t}>>({b}) < 0) || ({b} >= max_shift))"
            )
            with code.indent():
                code.writeline(f"return decltype({a})(0);")
            code.writeline(
                f"return decltype({a})(static_cast<std::make_unsigned_t<{scalar_t}>>({a}) << {b});"
            )
        code.writeline("()")
        return code

    @staticmethod
    def bitwise_right_shift(a, b):
        code = BracesBuffer()
        code.writeline("[&]()")
        with code.indent():
            scalar_t = DTYPE_TO_CPP[a.dtype]
            code.writeline(
                f"constexpr decltype({b}) max_shift = sizeof({scalar_t}) * CHAR_BIT - std::is_signed_v<{scalar_t}>;"
            )
            code.writeline(
                f"if ((static_cast<std::make_signed_t<{scalar_t}>>({b}) < 0) || ({b} >= max_shift))"
            )
            with code.indent():
                code.writeline(f"return decltype({a})({a} >> max_shift);")
            code.writeline(f"return decltype({a})({a} >> {b});")
        code.writeline("()")
        return code

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr):
        return f"normalized_rand_cpu({seed}, {offset})"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr):
        return f"randn_cpu({seed}, {offset})"

    @staticmethod
    def randint64(seed: sympy.Expr, offset: sympy.Expr, low, high):
        return f"randint64_cpu({seed}, {offset}, {low}, {high})"

    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        scalar_zero = f"decltype({x})(0)"
        scalar_one = f"decltype({x})(1)"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {x} > 0 ? {scalar_one} : {scalar_zero};")
            code.writeline(f"auto right = {x} < 0 ? {scalar_one} : {scalar_zero};")
            code.writeline("return left - right;")
        code.writeline("()")
        return code

    def partial_accumulate(
        self,
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


CppOverrides._initialize_pointwise_overrides("cpp")


class CppVecOverrides(CppOverrides):
    """Map element-wise ops to aten vectorization C++"""

    def __new__(cls, *args, **kargs):
        self = super().__new__(cls)

        def wrap(func):
            # `CppVecKernel` generates both scalar ops and vector ops according to
            # whether the inputs are scalars or vectors while all ops in `CppVecOverrides`
            # (except for some ops explained below) assume the inputs are vectors. We wrap the ops in
            # `CppVecOverrides` to broadcast scalar inputs to vectors if needed or fallback to
            # `CppOverrides` when all inputs are scalars.
            #
            # Notes on ops handled separately in their own functions:
            # `ops.masked`:
            #     needs recursive handling of masked body.
            # `ops.index_expr`:
            #     needs to further analyze the dependency of the index expression on
            #     the tiling itervar.
            def wrapper(*args, **kwargs):
                scalars = [
                    arg
                    for arg in args
                    if isinstance(arg, (int, sympy.Expr))
                    or (isinstance(arg, CppCSEVariable) and not arg.is_vec)
                ]
                vectors = [
                    arg
                    for arg in args
                    if isinstance(arg, CppCSEVariable) and arg.is_vec
                ]
                new_args = list(args)
                if scalars and vectors:
                    new_args = []
                    for arg in args:
                        if isinstance(arg, (int, sympy.Expr)):
                            if isinstance(arg, sympy.Expr) and not arg.is_number:
                                arg = ops.index_expr(arg, torch.int64)
                            else:
                                arg = ops.constant(arg, torch.int64)
                            arg = arg.value if isinstance(arg, OpsValue) else arg
                        new_args.append(arg)

                # DType Promotion
                if vectors:
                    # We have saw several data type mismatch issues related with index_expr in
                    # the lowering phase of torch.int8. torch.int32, torch.int64.
                    # 1. int32 and int64 in test_torchinductor.py::test_max_pool2d_with_indices_backward3_cpu
                    # 2. int8 and int32 in test_torchinductor.py::test_max_pool2d5_cpu
                    # 3. int32 and fp32 in test_torchinductor_dynamic_shapes.py::test_avg_pool2d8_dynamic_shapes_cpu
                    if len(new_args) == 2:
                        new_args = promote_args(new_args)
                    elif func is CppVecOverrides.where:
                        new_args[1:] = promote_args(new_args[1:])

                # Broadcast scalar args to vector
                if scalars and vectors:
                    assert isinstance(V.kernel, CppVecKernel)
                    new_args = [
                        (
                            V.kernel.broadcast(new_arg)
                            if (
                                isinstance(new_arg, CppCSEVariable)
                                and not new_arg.is_vec
                                and func
                                not in [
                                    CppVecOverrides.rand,
                                    CppVecOverrides.randn,
                                    CppVecOverrides.randint64,
                                ]
                            )
                            else new_arg
                        )
                        for new_arg in new_args
                    ]

                if vectors:
                    return func(*new_args, **kwargs)
                else:
                    # fallback to scalar ops
                    scalar_ops = super(CppVecOverrides, self)
                    scalar_func = getattr(scalar_ops, func.__name__)
                    assert scalar_func is not None
                    return scalar_func(*args, **kwargs)

            return wrapper

        for name, method in vars(CppVecOverrides).items():
            if getattr(method, "__class__", None) is staticmethod and name not in [
                "masked",
                "index_expr",
            ]:
                setattr(self, name, wrap(method.__func__))

        return self

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def truediv(a, b):
        return f"{a} / {b}"

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def exp2(x):
        return f"{x}.exp2()"

    @staticmethod
    def expm1(x):
        # decompose for a better performance
        vec_one = f"decltype({x})(1)"
        return f"{x}.exp() - {vec_one}"

    @staticmethod
    def erf(x):
        return f"{x}.erf()"

    @staticmethod
    def erfc(x):
        return f"{x}.erfc()"

    @staticmethod
    def erfinv(x):
        return f"{x}.erfinv()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def eq(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} == {y})"

    @staticmethod
    def ne(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        if x.dtype == torch.bool:
            assert y.dtype == torch.bool
            x_cast, y_cast = unify_mask_base_type(V.kernel.compute, (x, y))
            return f"{x_cast} != {y_cast}"
        else:
            assert x.dtype is not None
            return f"{V.kernel._get_mask_type(x.dtype)}({x} != {y})"

    @staticmethod
    def lt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} < {y})"

    @staticmethod
    def gt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} > {y})"

    @staticmethod
    def le(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} <= {y})"

    @staticmethod
    def ge(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} >= {y})"

    @staticmethod
    def and_(x, y):
        return f"{x} & {y}"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"~{a}"

    @staticmethod
    def logical_or(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_and(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def load_seed(name, offset):
        assert isinstance(V.kernel, CppVecKernel)
        return f"{V.kernel.load(name, offset)}"

    @staticmethod
    def rand(seed, offset):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = (
            f"result[offset_idx] = normalized_rand_cpu({seed}, offset[offset_idx]);"
        )
        return codegen_rand(offset, code, rand_function)

    @staticmethod
    def randn(seed, offset):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = f"result[offset_idx] = randn_cpu({seed}, offset[offset_idx]);"
        return codegen_rand(offset, code, rand_function)

    @staticmethod
    def randint64(seed, offset, low, high):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = f"result[offset_idx] = randint64_cpu({seed}, offset[offset_idx], {low}, {high});"
        return codegen_rand(offset, code, rand_function, torch.int64)

    @staticmethod
    def remainder(a, b):
        assert a.dtype == b.dtype, (
            "remainder vec implementation expect the same inputs' dtype."
        )
        return f"{a} - ({CppVecOverrides.floordiv(a, b)}) * {b}"

    @staticmethod
    def tan(a):
        return f"{a}.tan()"

    @staticmethod
    def tanh(a):
        if config.cpp.use_decompose_tanh:
            vec_one = f"decltype({a})(1)"
            vec_two = f"decltype({a})(2)"
            vec_minus_two = f"decltype({a})(-2)"
            return (
                f"{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}"
            )
        else:
            return f"{a}.tanh()"

    @staticmethod
    def reciprocal(a):
        return f"{a}.reciprocal()"

    @staticmethod
    def atan(x):
        return f"{x}.atan()"

    @staticmethod
    def acos(x):
        return f"{x}.acos()"

    @staticmethod
    def asin(x):
        return f"{x}.asin()"

    @staticmethod
    def cosh(x):
        return f"{x}.cosh()"

    @staticmethod
    def sinh(x):
        return f"{x}.sinh()"

    @staticmethod
    def log10(x):
        return f"{x}.log10()"

    @staticmethod
    def log2(x):
        return f"{x}.log2()"

    @staticmethod
    def nextafter(x, y):
        return f"{x}.nextafter({y})"

    @staticmethod
    def copysign(a, b):
        return f"{a}.copysign({b})"

    @staticmethod
    def atan2(a, b):
        return f"{a}.atan2({b})"

    @staticmethod
    def hypot(a, b):
        return f"{a}.hypot({b})"

    @staticmethod
    def atanh(x):
        # For real x, atanh(x) = 1/2 * log((1+x)/(1-x))
        vec_one = f"decltype({x})(1)"
        vec_one_half = f"decltype({x})(0.5)"
        return f"{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()"

    @staticmethod
    def asinh(x):
        return f"{x}.asinh()"

    @staticmethod
    def acosh(x):
        return f"{x}.acosh()"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"at::vec::clamp_min({x}, decltype({x})(0))"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    # TODO: this seems to be dead
    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())"

    @staticmethod
    def neg(x):
        return f"{x}.neg()"

    @staticmethod
    def floordiv(a, b):
        if is_float_dtype(a.dtype):
            assert a.dtype == b.dtype, (
                "div_floor_floating_vec implementation expect the same inputs' dtype."
            )
            return f"div_floor_floating_vec({a}, {b})"
        else:
            assert all(is_integer_dtype(item.dtype) for item in [a, b])
            # a and b are integer type
            _t = f"decltype({a})"
            if V.kernel._get_raw_num_vectors(b.dtype) < 1:
                # Doing blend to set the remaining bits of b to non-zero
                b = f"{_t}::blend<{(1 << V.kernel.tiling_factor) - 1}>({_t}(1), {b})"
            quot = f"{a} / {b}"
            has_rem = f"({a} % {b} != {_t}(0))"
            is_neg = f"(({a} < {_t}(0)) != ({b} < {_t}(0)))"
            return f"{_t}::blendv({quot}, {quot} - {_t}(1), {has_rem} & {is_neg})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        if V.kernel._get_raw_num_vectors(b.dtype) < 1:
            # Doing blend to set the remaining bits of b to non-zero
            _t = f"decltype({b})"
            b = f"{_t}::blend<{(1 << V.kernel.tiling_factor) - 1}>({_t}(1), {b})"
        return f"{a} / {b}"

    @staticmethod
    def minimum(a, b):
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} & {b_cast}"
        else:
            return f"at::vec::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} | {b_cast}"
        else:
            return f"at::vec::maximum({a}, {b})"

    @staticmethod
    def square(a):
        return f"{a} * {a}"

    @s
```



## High-Level Overview


This Python file contains 24 class(es) and 350 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ParallelDepth`, `OuterLoopFusedSchedulerNode`, `RecordOptimizationContext`, `CppOverrides`, `CppVecOverrides`, `CppTile2DOverrides`, `CppKernel`, `CppVecKernel`, `CppTile2DKernel`, `TilingSelect`, `CppKernelProxy`, `OuterLoopFusedKernel`, `ReasonFusedNodes`, `CppScheduling`, `KernelGroup`, `WorkSharing`, `LoopLevel`, `LoopNest`

**Functions defined**: `get_export_declaration`, `reduction_init`, `reduction_acc_type`, `reduction_combine`, `reduction_project`, `move_code_under_inner_loop`, `reduction_prefix_array`, `replace_acc_name`, `replace_cascade_sum_with_add`, `stride_at`, `simplify_index_in_vec_range`, `visit_indexing_div`, `visit_modular_indexing`, `stride_at_vec_range`, `fuse`, `__init__`, `get_outer_nodes`, `check_outer_fusion_loop_level_attr`, `_inner`, `merge_outer_fusion_kernels`

**Key imports**: contextlib, dataclasses, functools, itertools, math, operator, re, sys, warnings, Callable, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `dataclasses`
- `functools`
- `itertools`
- `math`
- `operator`
- `re`
- `sys`
- `warnings`
- `collections.abc`: Callable, Sequence
- `enum`: Enum
- `typing`: Any, cast, Optional, Union
- `sympy`
- `torch`
- `torch.fx`
- `torch._inductor`: dependencies
- `torch._prims_common`: is_float_dtype, is_integer_dtype
- `torch.utils._ordered_set`: OrderedSet
- `torch.utils._sympy.functions`: CeilDiv, FloorDiv, ModularIndexing
- `torch.utils._sympy.symbol`: free_symbol_is_type, symbol_is_type, SymT
- `..._dynamo.utils`: counters
- `..`: config, cpp_builder, cpu_vec_isa, ir, metrics
- `..debug`: set_kernel_post_grad_provenance_tracing
- `..loop_body`: LoopBody
- `..virtualized`: NullKernelHandler, ops, OpsValue, V


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `cpp.py_docs.md`
- **Keyword Index**: `cpp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
