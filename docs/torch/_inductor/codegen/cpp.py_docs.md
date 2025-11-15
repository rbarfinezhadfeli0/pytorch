# Documentation: cpp.py

## File Metadata
- **Path**: `torch/_inductor/codegen/cpp.py`
- **Size**: 234718 bytes
- **Lines**: 5839
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

    @staticmethod
    def where(a, b, c):
        assert isinstance(V.kernel, CppVecKernel)
        if b.dtype == torch.bool:
            assert c.dtype == torch.bool
            blendv_a, blendv_b, blendv_c = unify_mask_base_type(
                V.kernel.compute, (a, b, c)
            )
            return f"decltype({blendv_b})::blendv({blendv_c}, {blendv_b}, {blendv_a})"
        else:
            return f"decltype({b})::blendv({c}, {b}, {V.kernel._get_mask_cast(a, b.dtype)})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        vec_zero = f"decltype({x})(0)"
        vec_one = f"decltype({x})(1)"
        blendv_l = f"decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})"
        blendv_r = f"decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {blendv_l};")
            code.writeline(f"auto right = {blendv_r};")
            code.writeline("return left - right;")
        code.writeline("()")
        return code

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_dtypes=True):
        assert dtype in [
            torch.bool,
            torch.float64,
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ], f"{__name__} does not support {dtype}"
        assert isinstance(x, CppCSEVariable)
        src_dtype = x.dtype
        expr = V.kernel.get_to_dtype_expr(x, dtype, src_dtype)
        csevar = V.kernel.cse.generate(V.kernel.compute, expr)
        csevar.update_on_args("to_dtype", (x, dtype), {"src_dtype": src_dtype})
        if dtype in DTYPE_LOWP_FP and src_dtype == torch.float:
            V.kernel.cache_dtype_convert(x, src_dtype, csevar, dtype)
        return csevar

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"{x}.log1p()"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def masked(mask, body, other):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        with V.kernel.masked(mask) as new_mask:
            code.writeline(f"auto {var} = [&]")
            with V.kernel.swap_buffers(code), code.indent():
                result = body()
                code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        dtype = result.dtype
        body_code = f"{var}()"

        def maskify_or_vecify(code):
            return (
                f"{V.kernel._get_mask_type()}::from({code})"
                if dtype == torch.bool
                else f"{V.kernel._get_vec_type(dtype)}({code})"
            )

        if result.is_vec:
            body_code_vec = body_code
        else:
            body_code_vec = maskify_or_vecify(body_code)
        other_code = value_to_cpp(other, DTYPE_TO_CPP[dtype])
        # loading bool as VecMask<float, N>
        other_code_vec = maskify_or_vecify(other_code)
        assert isinstance(new_mask, CppCSEVariable), new_mask
        if new_mask.is_vec:
            code = BracesBuffer()
            code.writeline("[&]")
            with V.kernel.swap_buffers(code), code.indent():
                code.writeline(f"if ({new_mask}.all_zero())")
                with code.indent():
                    code.writeline(f"return {other_code_vec};")
                code.writeline("else")
                with code.indent():
                    # Create cse variable to reuse kernel.overrides.where
                    body_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        body_code_vec,
                    )
                    other_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        other_code_vec,
                    )
                    assert isinstance(body_vec_var, CppCSEVariable), body_vec_var
                    assert isinstance(other_vec_var, CppCSEVariable), other_vec_var
                    body_vec_var.dtype = dtype
                    other_vec_var.dtype = dtype
                    overrides: type[Union[CppOverrides, CppVecOverrides]] = (
                        # pyrefly: ignore [bad-assignment]
                        V.kernel.overrides
                    )  # type: ignore[has-type]
                    code.writeline(
                        f"return {overrides.where(new_mask, body_vec_var, other_vec_var)};"
                    )
            code.writeline("()")
            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                code,
            )
        elif result.is_vec:
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code_vec} : {other_code_vec}"
            )
        else:
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code} : {other_code}"
            )
        # `result` is explicitly added to the args for correct propagation
        # of relevant itervars and vectorization status.
        csevar.update_on_args("masked", (mask, body, other, result), {})
        return csevar

    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppVecKernel)
        index = V.kernel.rename_indexing(expr)
        tiling_var = V.kernel.itervars[V.kernel.tiling_idx]
        stride = V.kernel._try_get_const_stride(index, tiling_var)
        if stride == 0:
            return CppOverrides.index_expr(expr, dtype)
        elif stride is not None:
            idx = V.kernel.cse.generate(
                V.kernel.compute, cexpr(index), bounds=get_bounds_index_expr(expr)
            )
            value = ops.to_dtype(idx, dtype)
            if isinstance(value, OpsValue):
                value = value.value
            csevar = V.kernel.arange(value, stride)
        else:
            csevar = V.kernel._load_or_store_non_contiguous(  # type: ignore[assignment]
                None, index, dtype, V.kernel.compute
            )
        # pyrefly: ignore [missing-attribute]
        csevar.update_on_args("index_expr", (expr, dtype), {})
        return csevar

    @staticmethod
    def frexp(x):
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(V.kernel.cse.try_get(cache_key) is not None for cache_key in cache_keys):
            return tuple(V.kernel.cse.try_get(cache_key) for cache_key in cache_keys)

        cdtype = DTYPE_TO_CPP[x.dtype]
        size = V.kernel.tail_size if V.kernel.tail_size else V.kernel.tiling_factor
        code = BracesBuffer()
        exponent = V.kernel.cse.newvar(dtype=torch.int32)
        mantissa = V.kernel.cse.newvar(dtype=x.dtype)
        exponent.update_on_args("frexp", (x,), kwargs={})
        mantissa.update_on_args("frexp", (x,), kwargs={})
        n_vec = V.kernel._get_num_vectors(x.dtype)
        mantissa_t = (
            f"at::vec::Vectorized<{cdtype}>"
            if n_vec == 1
            else f"at::vec::VectorizedN<{cdtype}, {n_vec}>"
        )
        code.writeline(
            f"at::vec::Vectorized<int32_t> {exponent};"
            if n_vec == 1
            else f"at::vec::VectorizedN<int32_t, {n_vec}> {exponent};"
        )
        code.writeline(f"{mantissa_t} {mantissa};")
        code.writeline("[&]()")
        with code.indent():
            code.writeline(
                f"__at_align__ std::array<{cdtype}, {V.kernel.tiling_factor}> tmpbuf;"
            )
            code.writeline(f"{x}.store(tmpbuf.data(), {cexpr_index(size)});")
            code.writeline(
                f"__at_align__ std::array<int32_t, {V.kernel.tiling_factor}> tmpbuf_exponent;"
            )
            code.writeline(
                f"__at_align__ std::array<{cdtype}, {V.kernel.tiling_factor}> tmpbuf_mantissa;"
            )
            code.writeline(f"for (int i = 0; i < {cexpr_index(size)}; i++)")
            with code.indent():
                code.writeline(
                    "tmpbuf_mantissa[i] = std::frexp(tmpbuf[i], &tmpbuf_exponent[i]);"
                )
            code.writeline(
                f"{exponent} = at::vec::Vectorized<int32_t>::loadu(tmpbuf_exponent.data(), {cexpr_index(size)});"
                if n_vec == 1
                else f"{exponent} = at::vec::VectorizedN<int32_t, {n_vec}>::loadu(tmpbuf_exponent.data(), {cexpr_index(size)});"
            )
            code.writeline(
                f"{mantissa} = {mantissa_t}::loadu(tmpbuf_mantissa.data(), {cexpr_index(size)});"
            )
        code.writeline("();")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.put(cache_key, cse_var)
        return mantissa, exponent

    @classmethod
    def _scalarize(cls, scalar_func):
        def inner(*args, **kwargs):
            assert not kwargs
            kernel = V.kernel
            assert isinstance(kernel, CppVecKernel)
            code = BracesBuffer()
            code.writeline("[&]()")
            vec_dtype = args[0].dtype
            n_vec = kernel._get_num_vectors(vec_dtype)
            size = kernel.tail_size if kernel.tail_size else kernel.tiling_factor
            scalar_args = []
            cdtype = DTYPE_TO_CPP[vec_dtype]
            output_mask = scalar_func.__name__ in (
                "isinf",
                "isnan",
                "signbit",
            )
            octype = "bool" if output_mask else cdtype
            octype = (
                DTYPE_TO_CPP[args[-2]]
                if (scalar_func.__name__ == "to_dtype_bitcast")
                else octype
            )
            with code.indent():
                for argidx, arg in enumerate(args):
                    if isinstance(arg, CppCSEVariable):
                        assert arg.is_vec
                        assert arg.dtype == vec_dtype
                        code.writeline(
                            f"__at_align__ std::array<{cdtype}, {kernel.tiling_factor}> tmpbuf{argidx};"
                        )
                        code.writeline(
                            f"{arg}.store(tmpbuf{argidx}.data(), {cexpr_index(size)});"
                        )
                        scalar_args.append(f"tmpbuf{argidx}[i]")
                    else:
                        scalar_args.append(arg)
                code.writeline(
                    f"__at_align__ std::array<{octype}, {kernel.tiling_factor}> tmpbuf_out;"
                )
                res = scalar_func(*scalar_args)
                code.writeline(f"for (int i = 0; i < {cexpr_index(size)}; i++)")
                with code.indent():
                    code.writeline(f"tmpbuf_out[i] = {res};")
                if output_mask:
                    assert not kernel.tail_size
                    load_args = "tmpbuf_out.data()"
                    load_fn = f"at::vec::VecMask<{cdtype},{n_vec}>::from"
                else:
                    load_args = f"tmpbuf_out.data(), {cexpr_index(size)}"
                    if n_vec == 1:
                        load_fn = f"at::vec::Vectorized<{octype}>::loadu"
                    else:
                        load_fn = f" at::vec::VectorizedN<{octype}, {n_vec}>::loadu"
                code.writeline(f"return {load_fn}({load_args});")
            code.writeline("()")
            return code

        return inner

    @classmethod
    def _initialize_scalarize(cls):
        vec_vars = vars(CppVecOverrides)
        for name, method in vars(CppOverrides).items():
            if isinstance(method, staticmethod) and name not in vec_vars:
                func = cls._scalarize(method.__func__)
                func.__name__ = name
                setattr(cls, name, staticmethod(func))


CppVecOverrides._initialize_pointwise_overrides("cppvec")
CppVecOverrides._initialize_scalarize()


class CppTile2DOverrides(CppVecOverrides):
    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppTile2DKernel)
        expr = V.kernel.transform_indexing(expr)
        return CppVecOverrides.index_expr(expr, dtype)


class CppKernel(Kernel):
    """
    Base class for C++ kernel code generation in PyTorch Inductor.
    This class is responsible for generating C++ code from the intermediate representation.

    Args:
        args: Kernel arguments used for code generation
        num_threads: Number of threads for parallel execution
    """

    overrides = CppOverrides  # type: ignore[assignment]
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super().__init__(args)
        # Indicate when this kernel is active, for example
        # {x0, {24, 26}} -> this kernel is active when x0 >= 24 and x0 < 26
        self.active_ranges: dict[sympy.Expr, tuple[sympy.Expr, ...]] = {}
        # Indicate this kernel will be moved under the inner for-loop
        # See move_code_under_inner_loop
        self.inner_itervars: list[sympy.Symbol] = []
        self.call_ranges: Optional[tuple[sympy.Expr, ...]] = None
        self.ranges: list[sympy.Expr] = []
        self.itervars: list[sympy.Symbol] = []
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        # We need this because when we run "reduction" nodes here, we lack
        # "loop" information to decide whether we need a scalar init or an array init
        # in the reduction prefix. Meanwhile, we have other information like
        # reduction types and dtype to generate the reduction prefix. We record the information
        # with a callable lambda function, and when we have enough information to finalize
        # the reduction prefix, we can invoke the functions here with additional information.
        self.reduction_prefix_generators: list[Callable] = []  # type: ignore[type-arg]
        self.reduction_suffix = IndentedBuffer()
        self.parallel_reduction_prefix = IndentedBuffer()
        self.parallel_reduction_suffix = IndentedBuffer()
        self.local_reduction_init = IndentedBuffer()
        self.local_reduction_stores = IndentedBuffer()
        self.is_reduction = False
        self.non_parallel_reduction_prefix = IndentedBuffer()
        self.non_parallel_reduction_suffix = IndentedBuffer()
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.welford_helper_cse = CSE(
            self.newvar_prefix, self.suffix, name_prefix="welford_helper"
        )
        self.cascade_helper_cse = CSE(
            self.newvar_prefix, self.suffix, name_prefix="cascade_helper"
        )
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads  # num_threads the kernel specialized for
        self.reduction_omp_dec: dict[tuple[str, str], str] = {}
        self.reduction_var_names: list[str] = []

    def _gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=reduction_combine,
        reduction_init_fn=reduction_init,
    ):
        if config.cpp.dynamic_threads and not self.parallel_reduction_prefix:
            self.parallel_reduction_prefix.writeline(
                "int max_threads = omp_get_max_threads();"
            )
        acc_local = f"{acc}_local"
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        acc_local_in_array = f"{acc}_arr[tid]"
        self.local_reduction_init.writeline(
            f"{acc_type} {acc_local} = {reduction_init_fn(reduction_type, dtype)};"
        )
        self.parallel_reduction_prefix.splice(
            reduction_prefix_array(
                acc,
                acc_type,
                reduction_type,
                dtype,
                num_threads,
                reduction_init_fn,
            )
        )
        self.local_reduction_stores.writeline(f"{acc_local_in_array} = {acc_local};")
        self.parallel_reduction_suffix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc} = {reduction_combine_fn(reduction_type, acc, acc_local_in_array, src_dtype=dtype)};",
                "}",
            ],
        )

    def update_stores_with_parallel_reduction(self):
        for var_name in self.reduction_var_names:
            replace_acc_name(self.stores, var_name, f"{var_name}_local")

    def gen_body(self, code: Optional[BracesBuffer] = None):
        assert code is None
        code = BracesBuffer()
        with contextlib.ExitStack() as stack:
            if hasattr(self, "codegen_inner_loops"):
                code.splice(self.preloads)
                self.codegen_inner_loops(code)
                stack.enter_context(code.indent())
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)
        if hasattr(self, "codegen_inner_loops"):
            code.splice(self.poststores)

        if self.inner_itervars:
            for idx in self.inner_itervars:
                start, end = self.active_ranges[idx]
                code = move_code_under_inner_loop(code, idx, f"{idx}_tail", start, end)
        return code

    @contextlib.contextmanager
    def masked(self, mask):
        """Context manager to add an additional mask to loads and stores."""
        prior = self._load_mask
        if prior:
            mask = ops.and_(mask, prior)
            if isinstance(mask, OpsValue):
                mask = mask.value
                assert isinstance(mask, CppCSEVariable)
                # see NOTE [dtype of CppCSEVariable]
                # mask's dtype should be bool
                mask.dtype = torch.bool

        # pyrefly: ignore [bad-assignment]
        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def scale_index_with_offset(
        self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0
    ):
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(index, replacement)
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in cpp code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the cpp kernel.
        """
        return cexpr(self.rename_indexing(index))

    def index_indirect_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        """
        Check if an index has free symbol CppCSEVariable that depends on `itervar`.
        """
        return any(
            self.cse.varname_map[s.name].depends_on(itervar)  # type: ignore[attr-defined]
            for s in index.free_symbols
            if s.name in self.cse.varname_map  # type: ignore[attr-defined]
            and isinstance(self.cse.varname_map[s.name], CppCSEVariable)  # type: ignore[attr-defined]
        )

    def index_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        return itervar in index.free_symbols or self.index_indirect_depends_on(
            index, itervar
        )

    def var_ranges(self):
        return dict(zip(self.itervars, self.ranges))

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ):
        if not (lower or upper):
            return

        indirect = free_symbol_is_type(expr, SymT.TMP)
        if indirect:
            # indexing in compute
            csevar = ops.index_expr(expr, torch.int64).value
            buffer = V.kernel.compute
        else:
            # indexing in loads
            prior_compute = V.kernel.compute
            try:
                V.kernel.compute = self.loads
                csevar = ops.index_expr(expr, torch.int64).value
            finally:
                V.kernel.compute = prior_compute
            buffer = self.loads

        size_str = V.kernel.sexpr(self.rename_indexing(size)) if upper else None

        line = self.indirect_assert(
            csevar, "0" if lower else None, size_str, self._load_mask
        )
        self.cse.generate(buffer, line, assignment=False)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr_index(index)}]"
        csevar = self.cse.generate(self.loads, line, dtype=V.graph.get_dtype(name))
        csevar.update_on_args("load", (self, name, index), {})
        return csevar

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr_index(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr_index(index)}] += {value};"
            else:
                dtype = V.graph.get_dtype(name)
                # mirroring static_cast<float>(...) in load:
                value = f"static_cast<{DTYPE_TO_CPP[dtype]}>({value})"
                line = f"atomic_add(&{var}[{cexpr_index(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))

    def device_assert_async(self, cond, msg):
        self.compute.writeline(
            f'({cond} ? 0 : (throw std::runtime_error("{msg}"), 0));'
        )

    def _gen_reduction_prefix(
        self,
        acc: Union[CSEVariable, str],
        acc_type: str,
        rtype: str,
        dtype: torch.dtype,
        init_fn,
    ):
        # Generate reduction prefix
        # If size is None, we will define and initialize a single reduction variable
        # => float tmp_acc0 = 0;
        # Otherwise, we will define and initialize a reduction array
        # => float tmp_acc0_arr[size];
        # => for (int i = 0; i < size; i++) tmp_acc0_arr[i] = 0;
        def inner(size: Optional[int] = None):
            if size is None:
                return f"{acc_type} {acc} = {init_fn(rtype, dtype)};"
            else:
                return reduction_prefix_array(
                    acc,
                    acc_type,
                    rtype,
                    dtype,
                    size,
                    init_fn,
                )

        return inner

    def finalize_reduction_prefix(self, size: Optional[int] = None):
        for gen_fn in self.reduction_prefix_generators:
            self.reduction_prefix.splice(gen_fn(size))

    def need_use_acc_helper(self, reduction_type, dtype, use_scalar):
        # Check if we need accumulate helper for the reduction operation.
        # using accumulate helper generates the necessary code to improve precision for
        # sum and welford
        # Note: using helper has non-negligible impact on performance

        # keep the original behavior for welford_reduce
        # acc helper is not used for scalar welford_reduce
        if reduction_type == "welford_reduce":
            return not use_scalar

        # TODO add supports for more data types when needed
        if reduction_type == "sum" and dtype == torch.float:
            assert self.call_ranges is not None
            reduction_size = functools.reduce(
                operator.mul, self.call_ranges[self.reduction_depth :]
            )

            # chunk size to balance accuracy and performance
            chunk_size = 4096

            # use acc helper If cannot get size_hint
            try:
                reduction_size_hint = V.graph.sizevars.size_hint(reduction_size)
            except Exception:
                return True

            if reduction_size_hint > chunk_size:
                # use helper if the reduction size is too large
                V.graph.sizevars.check_lt(chunk_size, reduction_size)
                return True
            else:
                V.graph.sizevars.check_leq(reduction_size, chunk_size)
        return False

    def _acc_helper_init(
        self,
        reduction_type,
        helper_val,
        helper_range,
        dtype,
        num_threads=None,
        use_scalar=False,
    ):
        num_range_thread = (
            CeilDiv(helper_range, num_threads) if num_threads else helper_range
        )
        num_range_thread_expr = cexpr_index(num_range_thread)
        assert reduction_type in ["welford_reduce", "sum"]
        chunk_size = 4096
        num_chunks = CeilDiv(num_range_thread, chunk_size)
        helper_type = (
            "WelfordHelper"
            if reduction_type == "welford_reduce"
            else "CascadeSumHelper"
        )
        if use_scalar:
            h_type = DTYPE_TO_CPP[dtype]
        else:
            h_type = (
                self._get_vec_type(dtype)
                if hasattr(self, "_get_vec_type")
                else DTYPE_TO_CPP[dtype]
            )
        helper_init_line = (
            f"{helper_type}<{h_type}, {chunk_size}> {helper_val}"
            f"("
            f"{num_range_thread_expr}"
            f");"
        )
        if reduction_type == "sum":
            return helper_init_line
        if isinstance(num_chunks, sympy.Integer) and num_chunks <= 1:
            # When the number of chunks <= 1, there is no need to use cascade summation to improve
            # reduction accuracy. We can initialize a static WelfordHelper to improve performance.
            return f"static {helper_init_line}"
        else:
            return helper_init_line

    def _use_acc_helper(
        self, reduction_type, acc, helper_val, helper_range, dtype, use_scalar=False
    ):
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        self.non_parallel_reduction_prefix.writeline(
            self._acc_helper_init(
                reduction_type, helper_val, helper_range, dtype, None, use_scalar
            )
        )
        self.local_reduction_init.writeline(
            self._acc_helper_init(
                reduction_type, helper_val, helper_range, dtype, num_threads, use_scalar
            )
        )
        result = acc if use_scalar else f"{acc}_vec"
        if reduction_type == "welford_reduce":
            self.non_parallel_reduction_suffix.writeline(
                f"{result} = welford_combine({result}, &{helper_val});"
            )
            self.local_reduction_stores.writeline(
                f"{result}_local = welford_combine({result}_local, &{helper_val});"
            )
        else:
            self.non_parallel_reduction_suffix.writeline(
                f"{result} = cascade_sum_final(&{helper_val});"
            )
            self.local_reduction_stores.writeline(
                f"{result}_local = cascade_sum_final(&{helper_val});"
            )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in ("argmax", "argmin")
        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]

        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        self.reduction_var_names.append(f"{acc}")
        self.is_reduction = True
        init_dtype = src_dtype if argmax_or_argmin else dtype
        acc_type = reduction_acc_type(reduction_type, init_dtype)
        self.reduction_prefix_generators.append(
            self._gen_reduction_prefix(
                acc, acc_type, reduction_type, init_dtype, reduction_init
            )
        )

        if self.need_use_acc_helper(reduction_type, dtype, True):
            # use cascade_helper for vec kernel
            reduction_size = functools.reduce(
                operator.mul, self.ranges[self.reduction_depth :]
            )
            helper_val = self.cascade_helper_cse.generate(
                self.compute, f"reduction {reduction_key}", write=False
            )
            # rename the helper variable to distinguish it from vectorized version
            scalar_helper_val = f"scalar_{helper_val}"
            self._use_acc_helper(
                reduction_type,
                acc,
                scalar_helper_val,
                reduction_size,
                dtype,
                use_scalar=True,
            )
            self.stores.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, value, scalar_helper_val)};"
            )
        else:
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            self.stores.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, value, index=index)};"
            )

        self._gen_parallel_reduction_buffers(acc, acc_type, reduction_type, init_dtype)
        result = reduction_project(reduction_type, acc)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        self.reduction_suffix.writeline(
            DeferredLine(name, f"{var}[{cexpr_index(index)}] = {value};")
        )

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(reduction_lengths), (
                f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            )
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [
                sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
                for n in range(len(self.ranges))
            ]
            # pyrefly: ignore [bad-assignment]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        assert self.call_ranges is not None
        return V.graph.sizevars.size_hint(
            sympy_product(self.call_ranges), fallback=8192
        )

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        assert isinstance(self, CppKernelProxy)
        threads = parallel_num_threads()
        assert self.call_ranges is not None
        if isinstance(loop_nest.kernel, OuterLoopFusedKernel):
            par_depth = loop_nest.kernel.decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )
        else:
            par_depth = self.decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )

        is_reduction_loop = (
            loop_nest.loops is not None
            and loop_nest.loops[par_depth.start_depth].is_reduction
        )
        with contextlib.ExitStack() as stack:
            if par_depth.parallel_depth:
                if is_reduction_loop:
                    # need to close the worksharing scope to define reduction vars outside it
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_kernel(_loop_nest: LoopNest):
                def is_parallel_reduction():
                    assert _loop_nest.loops
                    root = _loop_nest.loops[par_depth.start_depth]
                    return root.is_reduction and root.parallel

                kernel = _loop_nest.get_kernel()
                if isinstance(kernel, OuterLoopFusedKernel):
                    for _loop_nest in kernel.inner:
                        gen_loop_nest(_loop_nest)
                else:
                    assert isinstance(kernel, CppKernelProxy)
                    if _loop_nest.loops is not None and is_parallel_reduction():
                        kernel.update_stores_with_parallel_reduction()
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(code.indent())
                        kernel.gen_body(code)

            def get_reduction_prefix_suffix(kernel, parallel=False, is_suffix=False):
                if is_suffix:
                    suffix = kernel.reduction_suffix
                    if parallel:
                        suffix = kernel.parallel_reduction_suffix + suffix
                    else:
                        suffix = kernel.non_parallel_reduction_suffix + suffix
                    return suffix
                else:
                    prefix = kernel.reduction_prefix
                    if parallel:
                        prefix = prefix + kernel.parallel_reduction_prefix
                    else:
                        prefix = prefix + kernel.non_parallel_reduction_prefix
                    return prefix

            def gen_loop_with_reduction(
                _loop_nest: LoopNest, depth: int = 0, in_reduction=False
            ):
                kernel = _loop_nest.get_kernel()
                assert _loop_nest.loops
                loop = _loop_nest.loops[depth]
                with contextlib.ExitStack() as stack_outer:
                    if loop.is_reduction and not in_reduction:
                        reduction_prefix = get_reduction_prefix_suffix(
                            kernel, loop.parallel, is_suffix=False
                        )
                        if reduction_prefix:
                            stack_outer.enter_context(code.indent())
                        code.splice(reduction_prefix)
                    if is_reduction_loop and loop.parallel:
                        worksharing.parallel(threads)
                        if kernel.local_reduction_init:
                            assert kernel.local_reduction_stores
                            code.splice(kernel.local_reduction_init)

                    gen_loop_at(_loop_nest, depth)

                    if is_reduction_loop and loop.parallel:
                        if kernel.local_reduction_stores:
                            code.splice(kernel.local_reduction_stores)
                        worksharing.close()
                    if loop.is_reduction and not in_reduction:
                        code.splice(
                            get_reduction_prefix_suffix(
                                kernel, loop.parallel, is_suffix=True
                            )
                        )

            def gen_loop_at(_loop_nest: LoopNest, depth: int = 0):
                with contextlib.ExitStack() as stack:
                    assert _loop_nest.loops
                    loop = _loop_nest.loops[depth]
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    gen_loop_nest(_loop_nest, depth + 1, loop.is_reduction)

            def gen_loop_nest(
                _loop_nest: LoopNest,
                depth: int = 0,
                in_reduction: bool = False,
            ):
                if _loop_nest.loops is None or depth == len(_loop_nest.loops):  # type: ignore[arg-type]
                    gen_kernel(_loop_nest)
                else:
                    gen_loop_with_reduction(_loop_nest, depth, in_reduction)

            stack.enter_context(code.indent())

            if (
                isinstance(loop_nest.kernel, OuterLoopFusedKernel)
                and isinstance(V.local_buffer_context, LocalBufferContext)
                and V.local_buffer_context.local_buffers
            ):
                # Allocate local buffer
                local_buffers = V.local_buffer_context.local_buffers
                for local_buffer in local_buffers.values():
                    # For dynamic size, rename s to ks
                    local_buf_size = sympy_product(
                        [
                            self.rename_indexing(size_val)
                            for size_val in local_buffer.get_layout().size
                        ]
                    )
                    local_buf_dtype = DTYPE_TO_CPP[local_buffer.get_layout().dtype]
                    allocate = f"std::make_unique<{local_buf_dtype} []>({cexpr(local_buf_size)})"
                    local_buffer_name = local_buffer.get_name()
                    code.splice(
                        f"std::unique_ptr<{local_buf_dtype} []> buf_{local_buffer_name} = {allocate};"
                    )
                    code.splice(
                        f"{local_buf_dtype}* {local_buffer_name} = buf_{local_buffer_name}.get();"
                    )
            gen_loop_nest(loop_nest)

    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNest.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    @property
    def assert_function(self) -> str:
        if V.graph.aot_mode:
            return "AOTI_TORCH_CHECK"
        else:
            return "TORCH_CHECK"

    def decide_parallel_depth(self, max_parallel_depth, threads):
        assert self.call_ranges is not None
        ranges = self.call_ranges[
            max_parallel_depth.start_depth : (
                max_parallel_depth.start_depth + max_parallel_depth.parallel_depth
            )
        ]
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr, fallback=8192)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        return ParallelDepth(
            parallel_depth=depth, start_depth=max_parallel_depth.start_depth
        )

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior

    def create_cse_var(self, *args, **kwargs):
        return CppCSEVariable(*args, **kwargs)

    def get_to_dtype_expr(self, src, dtype, src_dtype):
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({src})"

    def cache_dtype_convert(self, dst, dst_dtype, src, src_dtype):
        expr = self.get_to_dtype_expr(src, dst_dtype, src_dtype)
        self.cse.put(expr, dst)

    def codegen_conditions(
        self,
        code: BracesBuffer,
        prefix: Optional[str] = None,
        var: Optional[sympy.Symbol] = None,
    ):
        if prefix is None:
            prefix = ""
        if not self.active_ranges:
            return True
        conditions = []

        def gen(start, end, var):
            if start == end:
                return False
            var_id = None
            for i, _var in enumerate(self.itervars):
                if var == _var:
                    var_id = i
                    break
            if (
                type(self) is CppKernel
                and var_id
                and start == 0
                and end == self.ranges[var_id]
            ):
                end = 1
            # pyrefly: ignore [bad-argument-type]
            conditions.append(f"{var} >= {cexpr_index(start)}")
            # pyrefly: ignore [bad-argument-type]
            conditions.append(f"{var} < {cexpr_index(end)}")
            return True

        if var is not None:
            assert var in self.active_ranges
            start, end = self.active_ranges[var]
            if not gen(start, end, var):
                return False
        else:
            for _var, _range in self.active_ranges.items():
                start, end = _range
                if not gen(start, end, _var):
                    return False
        joined_conditions = " && ".join(conditions)
        if joined_conditions:
            code.writeline(f"if({prefix}({joined_conditions}))")
            return True
        else:
            return False


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides  # type: ignore[assignment]

    def __init__(
        self,
        args,
        num_threads,
        tiling_factor,
        tiling_idx,
        tail_size=None,
    ):
        super().__init__(args, num_threads)
        self.vec_isa = cpu_vec_isa.pick_vec_isa()
        assert self.vec_isa
        assert tiling_factor > 0, "Expect pass in Non-Zero tiling_factor explicitly"
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx
        self.tail_size = tail_size
        self.num_elems = tail_size if tail_size else tiling_factor

    def _try_get_const_stride(self, index: sympy.Expr, itervar: sympy.Symbol):
        if self.index_indirect_depends_on(index, itervar):
            return None
        for indirect_var in (
            self.cse.varname_map[s.name]  # type: ignore[attr-defined]
            for s in index.free_symbols
            if symbol_is_type(s, SymT.TMP)
        ):
            assert isinstance(indirect_var, CppCSEVariable)
            if indirect_var.is_vec:
                return None
        stride = stride_at_vec_range(index, itervar, self.tiling_factor)
        return stride if stride.is_number else None

    def _get_num_vectors(self, dtype: torch.dtype) -> int:
        num_vectors = math.ceil(
            self.tiling_factor * dtype.itemsize * 8 / self.vec_isa.bit_width()
        )
        assert num_vectors >= 1
        return num_vectors

    def _get_raw_num_vectors(self, dtype: torch.dtype) -> float:
        # This utility function is used to check if the vector lanes has been
        # fully utilized. For example, uint8 will only use 1/4 of the vector lanes.
        return self.tiling_factor * dtype.itemsize * 8 / self.vec_isa.bit_width()

    def _get_vec_type(self, dtype: torch.dtype) -> str:
        num_vectors = self._get_num_vectors(dtype)
        if num_vectors == 1:
            return f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>"
        else:
            return f"at::vec::VectorizedN<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

    def _get_mask_type(self, dtype: torch.dtype = torch.float) -> str:
        if dtype == torch.bool:
            return ""
        num_vectors = self._get_num_vectors(dtype)
        return f"at::vec::VecMask<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

    def _get_mask_cast(self, mask: CppCSEVariable, dtype: torch.dtype) -> str:
        assert mask.dtype == torch.bool, repr(mask)
        num_vectors = self._get_num_vectors(dtype)
        return f"{mask}.template cast<{DTYPE_TO_CPP[dtype]},{num_vectors}>()"

    def _get_vec_load_line(
        self,
        var: str,
        index: sympy.Expr,
        dtype: torch.dtype,
        load_mask: Optional[CppCSEVariable] = None,
    ):
        """
        Get a load line str that loads a vector from `var` at `index` of type `dtype`.
        If `load_mask` is not None, we do a masked load accordingly.
        Notes on the `dtype`:
        1. We always load `self.tiling_factor` number of elements regardless of the `dtype`.
           It means we load half of the vector lanes for 16-bit data types and quarter of the
           vector lanes for 8-bit data types.
        2. `torch.bool` and `torch.uint8` could mean masks and we load them as float mask vectors.
        """
        cpp_type = DTYPE_TO_CPP[dtype]
        num_vectors = self._get_num_vectors(dtype)
        load_mask_str = None
        if load_mask:
            if not load_mask.is_vec:
                # TODO: avoid hard-code torch.float
                load_mask_str = f"{self._get_mask_type(torch.float)}::from({load_mask})"
            else:
                load_mask_str = f"{self._get_mask_cast(load_mask, torch.float)}"
        loadbuf = f"{var} + {cexpr_index(index)}" if index != 0 else var
        if dtype == torch.bool:
            # TODO: should we consider load mask here?
            line = f"{self._get_mask_type()}::from({loadbuf})"
        else:
            line = (
                f"{load_mask_str}.template loadu<{cpp_type},{num_vectors}>({loadbuf})"
                if load_mask_str
                else f"{self._get_vec_type(dtype)}::loadu({loadbuf}, {cexpr_index(self.num_elems)})"
            )
        return line

    def _load_or_store_non_contiguous(
        self,
        var: Optional[str],
        index: sympy.Expr,
        dtype: torch.dtype,
        buffer: Optional[IndentedBuffer] = None,
        store_value: Optional[Union[str, CppCSEVariable]] = None,
        accu_store: bool = False,
    ) -> Optional[CppCSEVariable]:
        """
        Load or store a vector in a non-contiguous way. The vector is initialized from an array that is
        filled in an inner loop over the tiling factor.
        :param var: buffer to load from or store to, i.e. `var[transformed(index)]`. If None, we load the index
                    as index expression, i.e. `transformed(index)`.
        :param index: index into the `var` or the index expression by its own if `var` is None.
                      The `index` could contain indirect indexing or the tiling itervar. When used in
                      the inner loop, the index is transformed as follows:
                      1. the index is linearized along the tiling dim.
                      2. the indirect indexing vector variables are transformed into arrays over the tiling dim.
        :param dtype: data type of `var` or `index` if `var` is None.
        :param buffer: the code buffer to write the generated code to. If None, we write to `self.loads`.
        :param store_value: the value to store. If None, we load the vector.
        :param accu_store: whether accumulate the store_value to store_ptr. If True, a store_value should be provided
        :return: a CppCSEVariable that represents the loaded vector or None if it is a store.
        """
        assert not store_value or var is not None, "store var must be provided"
        if accu_store:
            assert store_value
        if buffer is None:
            buffer = self.loads

        def get_result_size(dtype: torch.dtype) -> int:
            if dtype.itemsize < 4:
                return self.num_elems * (4 // dtype.itemsize)
            else:
                return self.num_elems

        def get_tiling_size(dtype: torch.dtype) -> int:
            if dtype.itemsize < 4:
                return self.tiling_factor * (4 // dtype.itemsize)
            else:
                return self.tiling_factor

        def vec_to_array(vec_var: CppCSEVariable) -> CppCSEVariable:
            assert vec_var.is_vec
            code = BracesBuffer()
            code.writeline("[&]")
            with code.indent():
                vec_dtype = vec_var.dtype
                assert vec_dtype is not None
                if vec_dtype == torch.bool:
                    vec_dtype = torch.float
                result_size = get_result_size(vec_dtype)
                tiling_size = get_tiling_size(vec_dtype)
                code.writeline(
                    f"__at_align__ std::array<{DTYPE_TO_CPP[vec_dtype]}, {tiling_size}> tmpbuf;"
                )
                line = f"{vec_var}.store(tmpbuf.data(), {cexpr_index(result_size)});"
                code.writeline(line)
                code.writeline("return tmpbuf;")
            code.writeline("()")
            csevar = self.cse.generate(buffer, code)
            assert isinstance(csevar, CppCSEVariable)
            return csevar

        code = BracesBuffer()
        code.writeline("[&]")
        with code.indent():
            result_size = get_result_size(dtype)
            tiling_size = get_tiling_size(dtype)
            result_declare = (
                f"__at_align__ std::array<{DTYPE_TO_CPP[dtype]}, {tiling_size}> tmpbuf;"
            )
            code.writeline(result_declare)
            if store_value:
                code.writeline(
                    f"{store_value}.store(tmpbuf.data(), {cexpr_index(result_size)});"
                )
            itervar_inner = sympy_index_symbol(
                f"{self.itervars[self.tiling_idx]}_inner"
            )
            replacements = {}
            for indirect_var in (
                self.cse.varname_map[s.name]  # type: ignore[attr-defined]
                for s in index.free_symb

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 18 class(es): class, OuterLoopFusedSchedulerNode, RecordOptimizationContext, CppOverrides, CppVecOverrides, CppTile2DOverrides, CppKernel, CppVecKernel, CppTile2DKernel, TilingSelect, CppKernelProxy, OuterLoopFusedKernel, ReasonFusedNodes, CppScheduling, KernelGroup, WorkSharing, class, class

### Functions
This file defines 350 function(s): get_export_declaration, reduction_init, reduction_acc_type, reduction_combine, reduction_project, move_code_under_inner_loop, reduction_prefix_array, replace_acc_name, replace_cascade_sum_with_add, stride_at, simplify_index_in_vec_range, visit_indexing_div, visit_modular_indexing, stride_at_vec_range, fuse, __init__, get_outer_nodes, check_outer_fusion_loop_level_attr, _inner, merge_outer_fusion_kernels, __init__, __enter__, __exit__, get_opt_ctx, get_fx_node, decltype_promoted, add, sub, mul, to_dtype


## Key Components

The file contains 17755 words across 5839 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 234718 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
