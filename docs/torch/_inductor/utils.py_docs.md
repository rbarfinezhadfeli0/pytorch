# Documentation: `torch/_inductor/utils.py`

## File Metadata

- **Path**: `torch/_inductor/utils.py`
- **Size**: 134,846 bytes (131.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

## Original Source

```python
from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import importlib
import inspect
import io
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import statistics
import sys
import sysconfig
import tempfile
import textwrap
import time
import unittest
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
)
from datetime import datetime
from io import StringIO
from typing import (
    Any,
    cast,
    Concatenate,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import dataclass_transform, ParamSpec, Self
from unittest import mock

import sympy

import torch
import torch.utils._pytree as pytree
from torch._inductor.analysis.device_info import datasheet_tops
from torch._inductor.runtime.hints import DeviceProperties
from torch.fx.passes.regional_inductor import _needs_inductor_compile
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten, tree_map_only


if TYPE_CHECKING:
    from pathlib import Path

OPTIMUS_EXCLUDE_POST_GRAD = [
    "activation_quantization_aten_pass",
    "inductor_autotune_lookup_table",
]

from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    IterateExprs,
    ShapeEnv,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence, ValuesView

    from torch import SymBool, SymFloat, SymInt
    from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
    from torch.fx import GraphModule
    from torch.fx.node import Node

    from .codegen.common import WorkspaceArg
    from .codegen.wrapper import PythonWrapperCodegen
    from .dependencies import Dep
    from .graph import GraphLowering
    from .ir import Buffer, ExternKernel, IRNode, Layout, Operation, ReinterpretView
    from .output_code import CompiledFxGraph
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]
T = TypeVar("T")


# defines here before import torch._dynamo is for avoiding circular import
# when get_gpu_type is imported from dynamo
@functools.cache
def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    assert len(avail_gpus) <= 1
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()
    return gpu_type


from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import detect_fake_mode
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._sympy.functions import (
    CeilDiv,
    CleanDiv,
    FloorDiv,
    Identity,
    ModularIndexing,
)
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from . import config
from .runtime.runtime_utils import ceildiv as runtime_ceildiv


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")


_T = TypeVar("_T")
VarRanges = dict[sympy.Expr, sympy.Expr]
InputType = Optional[Union[torch.Tensor, int, torch.SymInt]]

GPU_KERNEL_BIN_EXTS = {"cuda": ".cubin", "xpu": ".spv"}

GPU_ALIGN_BYTES = 16
ALIGNMENT = 16

TMA_ALIGNMENT = 16
TMA_DESCRIPTOR_SIZE = 128

ALIGN_BYTES = 64
assert (ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0 and ALIGN_BYTES >= 8, "must be power of 2"


def _align(nbytes: int) -> int:
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


def _is_aligned(v: sympy.Expr) -> bool:
    """v can be statically proven to be a multiple of ALIGN_BYTES"""
    if isinstance(v, (sympy.Add, sympy.Max)):
        return all(map(_is_aligned, v.args))
    return isinstance(v, align) or sympy.gcd(v, ALIGN_BYTES) == ALIGN_BYTES


class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""

    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value: sympy.Expr) -> Optional[sympy.Expr]:
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value


@dataclasses.dataclass(frozen=True)
class GraphPartitionMap:
    """
    Mapping from the partition info (e.g., input/output) to the graph info
    """

    # a unique id of graph partition
    id: int

    # map partition input/output indices to graph input/output indices. None indicates
    # a partition input/output is not a graph input/output.
    input_index_mapping: list[Optional[int]]
    output_index_mapping: list[Optional[int]]

    # name of constants read/written by the graph partition
    constant_names: list[str]


def fp8_bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.float16, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        torch.cuda.synchronize()
        for i in range(n_repeat):
            cache.zero_()
            start_event[i].record()
            with torch.cuda.nvtx.range("RunCudaModule"):
                fn()
            end_event[i].record()
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        )

    res = torch.mean(times).item()
    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))
    filtered_events = EventList(
        [
            event
            for event in p.events()
            if (
                event.device_type == DeviceType.CUDA
                and re.match(r"fused_abs_max_\d", event.name) is not None
            )
        ]
    )
    if filtered_events:
        res -= (
            statistics.mean(event.device_time_total for event in filtered_events)
            / 1000.0
        )

    log.debug("profiling results: %s ms", res)
    return res


def do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    # We did't use decorator may_distort_benchmarking_result directly since that
    # requires us to import torch._inductor.runtime.benchmarking into global scope.
    # Importing torch._inductor.runtime.benchmarking will cause cuda initialization
    # (because of calling torch.cuda.available in global scope)
    # which cause failure in vllm when it create child processes. Check log:
    #   https://gist.github.com/shunting314/c194e147bf981e58df095c14874dd65a
    #
    # Another way to solve the issue is to just move do_bench_using_profiling
    # to torch._inductor.runtime.benchmarking and change all the call site.
    # But that's not trivial due to so many call sites in and out of pytorch.

    from torch._inductor.runtime.benchmarking import may_distort_benchmarking_result

    return may_distort_benchmarking_result(_do_bench_using_profiling)(
        fn, warmup, rep, is_vetted_benchmarking
    )


def _do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    if not is_vetted_benchmarking:
        from torch._inductor.runtime.benchmarking import may_ban_benchmarking

        may_ban_benchmarking()

    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        # Benchmark
        for _ in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            fn()
        # Record clocks
        torch.cuda.synchronize()

    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))

    filtered_events = EventList(
        [
            event
            for event in p.events()
            if event.device_type == DeviceType.CUDA and event.name != "Context Sync"
        ]
    )
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError(
            "Failed to divide all profiling events into #repeat groups. "
            "#CUDA events: %d, #repeats: %s",
            len(filtered_events),
            n_repeat,
        )
    num_event_per_group = len(filtered_events) / n_repeat
    actual_events = EventList(
        [
            event
            for i, event in enumerate(filtered_events)
            if i % num_event_per_group != 0
        ]
    )
    actual_events._build_tree()
    actual_events = actual_events.key_averages()

    log.debug("profiling time breakdown")
    log.debug(actual_events.table(row_limit=-1))

    res = sum(event.device_time_total for event in actual_events) / 1000.0 / n_repeat
    log.debug("profiling results: %s ms", res)
    return res


@functools.cache
def has_torchvision_roi_align() -> bool:
    try:
        from torchvision.ops import roi_align  # noqa: F401

        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False
    except RuntimeError as e:
        assert "torchvision::nms does not exist" in str(e)
        return False


def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type not in ("cpu", "meta") and device.index is None:
        device_interface = get_interface_for_device(device.type)
        return torch.device(device.type, index=device_interface.Worker.current_device())
    return device


def sympy_product(it: Iterable[sympy.Expr]) -> sympy.Expr:
    return functools.reduce(operator.mul, it, sympy.S.One)


def sympy_dot(seq1: Sequence[sympy.Expr], seq2: Sequence[sympy.Expr]) -> sympy.Expr:
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


def ceildiv(
    number: Union[int, sympy.Expr], denom: Union[int, sympy.Expr]
) -> Union[int, sympy.Expr]:
    if isinstance(number, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(sympy.sympify(number), sympy.sympify(denom))
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    assert isinstance(number, int) and isinstance(denom, int), (
        f"{number}: {type(number)}, {denom}: {type(denom)}"
    )
    return runtime_ceildiv(number, denom)


def _type_of(key: Optional[torch.dtype]) -> str:
    # Use the function here to get rid of dependencies on the Triton during the codegen.
    # Refer to Triton implementation here:
    # https://github.com/triton-lang/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
    # `None` is nullptr.  Implicitly convert to *i8.
    if key is None:
        return "*i8"
    dtype_str = str(key).split(".")[-1]
    tys = {
        "bool": "i1",
        "float8e4nv": "fp8e4nv",
        "float8e5": "fp8e5",
        "float8e4b15": "fp8e4b15",
        "float8e4b15x4": "fp8e4b15x4",
        "float8_e4m3fn": "fp8e4nv",
        "float8_e5m2": "fp8e5",
        # TODO: remove when support is added in triton
        # https://github.com/triton-lang/triton/issues/6054
        "float8_e8m0fnu": "u8",
        "float4_e2m1fn_x2": "u8",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    # reinterpret can create triton type
    tys.update({v: v for v in list(tys.values())})
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"


def convert_shape_to_inductor(
    lst: Iterable[Union[int, torch.SymInt]],
) -> list[sympy.Expr]:
    """
    Gets the shape and stride of a tensor. For non-symbolic tensors, this is
    trivial. But for symbolic tensors, we need to map from SymIntNode into
    sympy.Expr.
    """
    return [sympy.sympify(i) for i in lst]


def convert_to_symint(i: Union[int, sympy.Expr]) -> Union[int, torch.SymInt]:
    """
    Like convert_shape_to_symint, but operates on a single expression.
    """
    from .virtualized import V

    return (
        i
        if isinstance(i, int)
        else (
            int(i)
            if isinstance(i, sympy.Integer)
            else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        )
    )


def convert_shape_to_symint(
    lst: Iterable[Union[int, sympy.Expr]],
) -> list[Union[int, torch.SymInt]]:
    """
    Takes a list of shapes from Inductor and converts them into symints (or just
    ints if all shapes are static).
    """
    return [convert_to_symint(i) for i in lst]


def is_view(op: torch._ops.OpOverload) -> bool:
    """
    Does this op overload have aliasing
    """
    return any(a.alias_info is not None for a in op._schema.arguments)


def is_pointwise_use(
    use: Node,
    is_pointwise_fn: Callable[[torch._ops.OpOverload], bool] = lambda _: False,
) -> bool:
    """
    Do all uses of this op have torch.Tag.pointwise or return True for optional `is_pointwise_fn`

    Uses in views ops will follow the views uses
    """

    if use.op != "call_function":
        return False
    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    target = cast(torch._ops.OpOverload, use.target)
    if target is operator.getitem or is_view(target):
        return all(is_pointwise_use(u, is_pointwise_fn) for u in use.users)

    return torch.Tag.pointwise in target.tags or is_pointwise_fn(target)


def gen_gm_and_inputs(
    target: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[GraphModule, list[torch.Tensor]]:
    g = torch.fx.Graph()
    graph_args: list[torch.Tensor] = []

    def add_tensor_arg(arg: torch.Tensor) -> Node:
        graph_args.append(arg)
        return g.placeholder(f"arg{len(graph_args)}")

    node = g.call_function(
        target, *tree_map_only(torch.Tensor, add_tensor_arg, (args, kwargs))
    )
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)  # type: ignore[assignment]
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, graph_args


def synchronize(device: str = "cuda") -> None:
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


def timed(
    model: Callable[..., Any],
    example_inputs: Sequence[Any],
    times: int = 1,
    device: str = "cuda",
) -> float:
    synchronize(device)
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize(device)
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None  # type: ignore[possibly-undefined]
    return t1 - t0


def print_performance(
    model: Callable[..., Any],
    example_inputs: Sequence[Any] = (),
    times: int = 10,
    repeat: int = 10,
    baseline: float = 1.0,
    device: str = "cuda",
) -> float:
    timings = torch.tensor(
        [timed(model, example_inputs, times, device) for _ in range(repeat)]
    )
    took = torch.median(timings) / times
    print(f"{took / baseline:.6f}")
    return took.item()


def precompute_method(obj: Any, method: str) -> None:
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: list[str]) -> None:
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a: int, b: int) -> int:
    return int(a > b) - int(a < b)


def pad_listlike(x: Union[int, Sequence[int]], size: int) -> Sequence[int]:
    if isinstance(x, int):
        return [x] * size
    if len(x) == 1:
        return type(x)([x[0]]) * size  # type: ignore[call-arg, operator, return-value]
    return x


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x: tuple[_T, ...]) -> list[_T]:
    if len(x) == 0:
        return []

    def sort_func(elem: _T) -> str:
        if isinstance(elem, str):
            return elem

        from .scheduler import BaseSchedulerNode

        assert isinstance(elem, BaseSchedulerNode)
        return elem.get_name()

    return sorted(x, key=sort_func)


P = ParamSpec("P")
RV = TypeVar("RV", covariant=True)
FN_TYPE = Callable[Concatenate[Any, P], RV]


class CachedMethod(Protocol, Generic[P, RV]):
    @staticmethod
    def clear_cache(cache: Any) -> None: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RV: ...


# See https://github.com/python/mypy/issues/13222#issuecomment-1193073470 to understand the type signature
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    name = fn.__name__
    key = f"__{name}_cache"

    # wrapper is likely on the hot path, compile a specialized version of it
    ctx = {"fn": fn}
    exec(
        f"""\
        def {name}_cache_on_self(self):
            try:
                return self.{key}
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "{key}", rv)
            return rv
        """.lstrip(),
        ctx,
    )
    wrapper = functools.wraps(fn)(ctx[f"{name}_cache_on_self"])

    def clear_cache(self: Any) -> None:
        if hasattr(self, key):
            delattr(self, key)

    wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def cache_property_on_self(fn: Callable[P, RV]) -> CachedMethod[P, RV]:
    """
    Variant of cache_on_self for properties. The only difference is the type signature.
    """
    # pyrefly: ignore [bad-argument-type]
    return cache_on_self(fn)


def cache_on_self_and_args(
    class_name: str,
) -> Callable[[FN_TYPE[P, RV]], FN_TYPE[P, RV]]:
    # include both class_name and fn_name in the key to support `super().fn(self, **args, **kwargs)` calls.

    def wrapper(
        fn: FN_TYPE[P, RV],
    ) -> FN_TYPE[P, RV]:
        key = f"__{class_name}_{fn.__name__}_cache"

        # wrapper is likely on the hot path, compile a specialized version of it
        ctx = {"fn": fn}
        exec(
            f"""\
            def inner(self: Any, *args: P.args, **kwargs: P.kwargs) -> RV:
                args_kwargs = (args, tuple(sorted(kwargs.items())))

                if not hasattr(self, "{key}"):
                    object.__setattr__(self, "{key}", {{}})

                cache = self.{key}

                try:
                    return cache[args_kwargs]
                except KeyError:
                    pass

                rv = fn(self, *args, **kwargs)

                cache[args_kwargs] = rv
                return rv
            """.lstrip(),
            ctx,
        )
        inner = functools.wraps(fn)(ctx["inner"])

        def clear_cache(self: Any) -> None:
            if hasattr(self, key):
                delattr(self, key)

        inner.clear_cache = clear_cache  # type: ignore[attr-defined]
        return inner

    return wrapper


def aggregate_origins(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
) -> OrderedSet[Node]:
    from . import ir

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                # pyrefly: ignore [missing-attribute]
                node.node.origins
                for node in node_schedule
                if hasattr(node, "node") and node.node
            ],
            OrderedSet(),
        )
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return OrderedSet()


def get_fused_kernel_name(
    node_schedule: Sequence[BaseSchedulerNode],
    descriptive_names: Literal[True, "torch", "original_aten", "inductor_node"],
) -> str:
    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == "original_aten":

        def get_origin_meta_str(origin):
            original_aten = origin.meta["original_aten"]
            key = ""
            if isinstance(original_aten, torch._ops.OpOverload):
                key = original_aten._overloadpacket.__name__
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            return key

        # Bases the kernel name off of the top-level aten operator (i.e. pre-decompositions)
        sources = [
            get_origin_meta_str(origin)
            for origin in all_origins
            if origin.op == "call_function"
            and "original_aten" in origin.meta
            and origin.meta["original_aten"] is not None
        ]
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "torch":
        # Bases the kernel name off of the top-level "torch" operator (i.e. post-dynamo graph)
        sources = []
        for origin in all_origins:
            if origin.op == "call_function":
                source_fn = None
                suffix = ""
                if "source_fn_stack" in origin.meta:
                    source_fn = origin.meta["source_fn_stack"][-1]
                elif "fwd_source_fn_stack" in origin.meta:
                    # backward nodes have "fwd_source_fn_stack" instead
                    source_fn = origin.meta["fwd_source_fn_stack"][-1]
                    suffix = "backward"
                if not source_fn:
                    continue
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1] + suffix)
                else:
                    sources.append(source_fn[1].__name__ + suffix)

        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError
    return "_".join(["fused"] + sources)


def get_kernel_metadata(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
    wrapper: PythonWrapperCodegen,
) -> tuple[str, str]:
    """
    Retrieves metadata information for a kernel.
    Args:
        node_schedule (Union[Sequence[BaseSchedulerNode], ExternKernel]):
            Either a sequence of BaseSchedulerNode objects or an ExternKernel instance.
        wrapper (PythonWrapperCodegen):
            An instance of PythonWrapperCodegen, used to define the code comment format.
    Returns:
        tuple[str, str]:
            A tuple containing two strings:
                - The first string represents the kernel's metadata.
                - The second string represent the kernel's detailed metadata.
    """

    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)

    # Attempt to sort `inductor_nodes` topologically. Note that the case
    # where `inductor_nodes` contains nodes from multiple graph instances
    # is not supported. An example of this is conditional statements.
    single_graph = None
    if inductor_nodes:
        unique_graphs = OrderedSet(n.graph for n in inductor_nodes)
        if len(unique_graphs) == 1:
            single_graph = inductor_nodes[0].graph
            # create a map of idx -> node and cache it
            if not hasattr(single_graph, "_inductor_kernel_metadata_node_to_idx_map"):
                node_to_idx_map = {n: idx for idx, n in enumerate(single_graph.nodes)}
                single_graph._inductor_kernel_metadata_node_to_idx_map = node_to_idx_map  # type: ignore[attr-defined]
            inductor_nodes.sort(
                key=lambda n: single_graph._inductor_kernel_metadata_node_to_idx_map[n]  # type: ignore[attr-defined]
            )

    for node in inductor_nodes:
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            original_aten = node.meta["original_aten"]
            key = None
            if isinstance(original_aten, torch._ops.OpOverload):
                key = str(original_aten._overloadpacket)
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            if key:
                original_aten_dict[key].append(node.name)
        if "from_node" in node.meta:
            key = node.meta["from_node"][0].name
            from_node_dict[key].append(node.name)
        elif node.meta.get("partitioner_tag") == "is_backward":
            # backward nodes currently don't have a "from node"
            from_node_dict[node.name].append(node.name)
    sort_str = "Topologically Sorted" if single_graph is not None else "Unsorted"
    metadata = (
        f"{wrapper.comment} {sort_str} Source Nodes: [{', '.join(from_node_dict.keys())}], "
        f"Original ATen: [{', '.join(original_aten_dict.keys())}]"
    )

    # trace back to original node here
    detailed_metadata = [f"{wrapper.comment} Source node to ATen node mapping:"]
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment}   {original_node} => {', '.join(sorted(nodes))}"
        )

    # print the aot_autograd graph fragment
    if single_graph is not None:
        from . import ir

        detailed_metadata.append(f"{wrapper.comment} Graph fragment:")
        all_reads: OrderedSet[str] = OrderedSet()
        all_writes: list[str] = []
        if not isinstance(node_schedule, ir.ExternKernel):
            from .virtualized import V

            def get_buffer_info(
                buffer: Union[ir.TensorBox, ir.Buffer, ir.TorchBindObject], rw_name: str
            ) -> tuple[str, ir.Layout | None]:
                if isinstance(buffer, ir.TensorBox) and isinstance(
                    buffer.data, ir.StorageBox
                ):
                    origin_node = buffer.data.data.origin_node
                else:
                    origin_node = buffer.origin_node
                if origin_node is None:
                    # use the read/write name if no origin node is found
                    name = rw_name
                else:
                    name = origin_node.name
                try:
                    layout = buffer.get_layout()
                except NotImplementedError:
                    layout = None
                return name, layout

            def stringify_shape(shape: Iterable[int]) -> str:
                return f"[{', '.join([str(x) for x in shape])}]"

            def stringfy_layout(layout: ir.Layout | None) -> str:
                if layout is None:
                    return ""
                shape_annotation = f"{stringify_shape(layout.size)}"
                stride_annotation = f"{stringify_shape(layout.stride)}"
                device_annotation = f"{layout.device}"

                return (
                    f'"{dtype_abbrs[layout.dtype]}{shape_annotation}'
                    f'{stride_annotation}{device_annotation}"'
                )

            for n in node_schedule:
                if not hasattr(n, "read_writes") or n.read_writes is None:
                    continue
                if hasattr(n.read_writes, "reads") and n.read_writes.reads is not None:
                    for r in n.read_writes.reads:
                        # Remove the dupricated inputs
                        if r.name in all_reads:
                            continue
                        all_reads.add(r.name)
                        buffer = V.graph.try_get_buffer(r.name)
                        if buffer is None:
                            continue
                        input_name, layout = get_buffer_info(buffer, r.name)
                        detailed_metadata.append(
                            f"{wrapper.comment}   %{input_name} : Tensor "
                            f"{stringfy_layout(layout)} = PlaceHolder[target={input_name}]"
                        )

                if (
                    hasattr(n.read_writes, "writes")
                    and n.read_writes.writes is not None
                ):
                    for w in n.read_writes.writes:
                        buffer = V.graph.try_get_buffer(w.name)
                        if buffer is None:
                            continue
                        output_name, _ = get_buffer_info(buffer, w.name)

                        all_writes.append("%" + output_name)

        for node in inductor_nodes:
            detailed_metadata.append(
                f"{wrapper.comment}   {node.format_node(include_tensor_metadata=True)}"
            )

        detailed_metadata.append(f"{wrapper.comment}   return {','.join(all_writes)}")

    return metadata, "\n".join(detailed_metadata)


def dominated_nodes(
    initial_queue: Iterable[torch.fx.Node],
    skip_filter: Optional[Callable[[Any], bool]] = None,
) -> OrderedSet[torch.fx.Node]:
    """Returns the set of nodes whose values depend on those within initial_queue"""
    initial_queue = list(initial_queue)
    dominated_set = OrderedSet(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def gather_origins(
    args: Sequence[IRNode], kwargs: dict[str, IRNode]
) -> OrderedSet[torch.fx.Node]:
    from . import ir

    def is_unrealized_node(n: IRNode) -> bool:
        if isinstance(n, ir.TensorBox):
            return is_unrealized_node(n.data)
        if isinstance(n, ir.StorageBox):
            return is_unrealized_node(n.data)
        return isinstance(n, ir.IRNode) and not isinstance(
            n,
            (
                ir.ComputedBuffer,
                ir.InputsKernel,
                ir.InputBuffer,
                ir.TemplateBuffer,
            ),
        )

    # kwargs and args may include a container of node, for example torch.cat([t1, t2])
    # flatten them before search the unrealized nodes
    kwargs_flatten, _ = tree_flatten(kwargs)
    kwargs_origins = [val.origins for val in kwargs_flatten if is_unrealized_node(val)]
    args_flatten, _ = tree_flatten(args)
    args_origins = [val.origins for val in args_flatten if is_unrealized_node(val)]
    return OrderedSet(itertools.chain(*args_origins, *kwargs_origins))


def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """

    def is_neg_lead(expr: sympy.Expr) -> bool:
        return (
            isinstance(expr, sympy.Mul) and len(expr.args) == 2 and expr.args[0] == -1
        )

    def sympy_str_add(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Add):
            # Special case 'a - b'. Note that 'a - b - c' will still appear as
            # 'a + -1 * b + -1 * c'.
            if len(expr.args) == 2 and is_neg_lead(expr.args[1]):
                return f"{sympy_str_mul(expr.args[0])} - {sympy_str_mul(expr.args[1].args[1])}"
            else:
                return " + ".join(map(sympy_str_mul, expr.args))
        else:
            return sympy_str_mul(expr)

    def sympy_str_mul(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Mul):
            if is_neg_lead(expr):
                # Special case '-a'. Note that 'a * -b' will still appear as
                # '-1 * a * b'.
                return f"-{sympy_str_atom(expr.args[1])}"
            else:
                return " * ".join(map(sympy_str_atom, expr.args))
        else:
            return sympy_str_atom(expr)

    def sympy_str_atom(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Symbol):
            return expr.name
        elif isinstance(expr, (sympy.Add, sympy.Mul)):
            return f"({sympy_str_add(expr)})"
        elif isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv, Identity)):
            return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
        else:
            return str(expr)

    return sympy_str_add(expr)


def get_bounds_index_expr(index: sympy.Expr) -> ValueRanges[Any]:
    from .virtualized import V

    # If this expression does not come from an FX node, we compute its bounds
    if (
        config.compute_all_bounds
        and (fx_node := getattr(V.interpreter, "current_node", None))
        and fx_node.target != "index_expr"
    ):
        return bound_sympy(index)
    else:
        return ValueRanges.unknown()


def prefix_is_reduction(prefix: str) -> bool:
    return prefix[0] == "r"


def sympy_index_symbol_with_prefix(prefix: SymT, idx: int) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert prefix != SymT.SIZE
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return make_symbol(prefix, idx, integer=True, nonnegative=True)


def generate_assert(check: bool) -> bool:
    return (check or config.debug_index_asserts) and config.assert_indirect_indexing


def sympy_index_symbol(name: str) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_subs(expr: sympy.Expr, replacements: dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(
        replaced: sympy.Expr, replacement: Union[sympy.Expr, str]
    ) -> sympy.Symbol:
        assert isinstance(replaced, sympy.Expr)
        if isinstance(replacement, str):
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,  # type: ignore[attr-defined]
                nonnegative=replaced.is_nonnegative,  # type: ignore[attr-defined]
            )
        else:
            return replacement

    # xreplace is faster than subs, but is way more picky
    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}
    )


def is_symbolic(a: Any) -> TypeGuard[Union[torch.SymInt, torch.Tensor]]:
    return isinstance(a, torch.SymInt) or (
        isinstance(a, torch.Tensor)
        and any(is_symbolic(x) for x in itertools.chain(a.size(), a.stride()))
    )


def any_is_symbolic(*args: Any) -> bool:
    return any(is_symbolic(a) for a in args)


def get_first_incompatible_cudagraph_node(
    gm: torch.fx.GraphModule,
) -> Optional[torch.fx.Node]:
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    forbidden_set = OrderedSet(
        [
            "aten._fused_moving_avg_obs_fq_helper.default",
            "aten._fused_moving_avg_obs_fq_helper_functional.default",
            "fbgemm.dense_to_jagged.default",
            "fbgemm.jagged_to_padded_dense.default",
            "run_and_save_rng_state",
            "run_with_rng_state",
            "aten._local_scalar_dense",
            # Technically, it's not necessary to ban this, because an
            # assert_scalar with constant arguments can be validly run
            # with CUDA graphs, but the operator is also pointless with
            # constant arguments, so might as well ban
            "aten._assert_scalar",
        ]
    )
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update(
            (
                "aten._unsafe_index_put.default",
                "aten._unsafe_masked_index_put_accumulate.default",
                "aten.index_put.default",
                "aten.index_put_.default",
                "aten.scatter.src",
                "aten.scatter.reduce",
                "aten.scatter.value_reduce",
                "aten.scatter_add_",
                "aten.scatter_add.default",
                "aten.scatter_reduce.two",
                "aten.scatter_reduce_.two",
                "aten.scatter_reduce.two_out",
            )
        )

    for node in gm.graph.nodes:
        if str(node.target) in forbidden_set:
            return node

        if (
            not torch._inductor.config.graph_partition
            and isinstance(node.target, torch._ops.OpOverload)
            and torch._C.Tag.cudagraph_unsafe in node.target.tags  # type: ignore[attr-defined]
        ):
            # skip cudagraph if a cudagraph_unsafe op is detected.
            # graph_partition helps by splitting on this cudagraph_unsafe
            # op and cudagraphifying the subgraphs.
            return node

        if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
            return node

    return None


def output_node(gm: torch.fx.GraphModule) -> Node:
    """Get the output node from an FX graph"""
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return last_node


def get_all_devices(gm: torch.fx.GraphModule) -> OrderedSet[torch.device]:
    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    input_devices: OrderedSet[torch.device] = OrderedSet(
        node.meta["val"].device
        for node in placeholder_nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    )

    out_arg = output_node(gm).args[0]  # type: ignore[union-attr]
    out_args = out_arg if isinstance(out_arg, tuple) else (out_arg,)
    out_devices: OrderedSet[torch.device] = OrderedSet(
        arg.meta["val"].device
        for arg in out_args
        if isinstance(arg, torch.fx.Node)
        and isinstance(arg.meta.get("val"), torch.Tensor)
    )
    return input_devices | out_devices


import gc


def unload_xpu_triton_pyds() -> None:
    # unload __triton_launcher.pyd
    for module_name in list(sys.modules.keys()):
        if not module_name.startswith("torch._inductor.runtime.compile_tasks."):
            continue
        m = sys.modules[module_name]
        for attr_name in m.__dict__:
            if attr_name.startswith("triton_"):
                kernel = getattr(m, attr_name)
                if isinstance(
                    kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner
                ):
                    for result in kernel.compile_results:
                        if isinstance(
                            result,
                            torch._inductor.runtime.triton_heuristics.TritonCompileResult,
                        ):
                            # pyrefly: ignore [missing-attribute]
                            result.kernel.run.mod.__del__()
        del sys.modules[module_name]

    # unload spirv_utils.pyd
    if "triton.runtime.driver" in sys.modules:
        mod = sys.modules["triton.runtime.driver"]
        del type(mod.driver.active.utils).instance
        del mod.driver.active.utils

    gc.collect()


_registered_caches: list[Any] = []


def clear_on_fresh_cache(obj: Any) -> Any:
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_cache().
    """
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def clear_caches() -> None:
    """
    Clear all registered caches.
    """
    for obj in _registered_caches:
        obj.cache_clear()


@contextlib.contextmanager
def fresh_cache(
    cache_entries: Optional[dict[str, Any]] = None,
    dir: Optional[str] = None,
    delete: bool = True,
) -> Iterator[None]:
    """
    Contextmanager that provides a clean tmp cachedir for pt2 caches.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    clear_caches()

    from torch._inductor.cpp_builder import normalize_path_separator

    inductor_cache_dir = normalize_path_separator(tempfile.mkdtemp(dir=dir))
    try:
        with mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": inductor_cache_dir}
        ):
            log.debug("Using inductor cache dir %s", inductor_cache_dir)
            triton_cache_dir = normalize_path_separator(
                os.path.join(inductor_cache_dir, "triton")
            )
            with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": triton_cache_dir}):
                yield
                if isinstance(cache_entries, dict):
                    assert len(cache_entries) == 0, "expected empty cache_entries dict"
                    if os.path.exists(triton_cache_dir):
                        files = os.listdir(triton_cache_dir)
                        cache_entries.update(
                            {
                                f: os.path.getsize(os.path.join(triton_cache_dir, f))
                                for f in files
                                if ".lock" not in f
                            }
                        )
        if delete:
            if is_windows() and torch.xpu.is_available():
                unload_xpu_triton_pyds()

            shutil.rmtree(
                inductor_cache_dir,
                # Let's not fail if we can't clean up the temp dir. Also note that for
                # Windows, we can't delete the loaded modules because the module binaries
                # are open.
                ignore_errors=is_windows(),
                onerror=lambda func, path, exc_info: log.warning(
                    "Failed to remove temporary cache dir at %s",
                    inductor_cache_dir,
                    exc_info=exc_info,
                ),
            )
    except Exception:
        log.warning("on error, temporary cache dir kept at %s", inductor_cache_dir)
        raise
    finally:
        clear_caches()


# Deprecated functions -- only keeping them for BC reasons
clear_on_fresh_inductor_cache = clear_on_fresh_cache
clear_inductor_caches = clear_caches
fresh_inductor_cache = fresh_cache


def argsort(seq: Sequence[Any]) -> list[int]:
    # preserve original order for equal strides
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


def argsort_sym(
    shape_env: ShapeEnv, seq: Sequence[Union[int, torch.SymInt, sympy.Expr]]
) -> list[int]:
    def cmp(a: tuple[int, sympy.Expr], b: tuple[int, sympy.Expr]) -> int:
        a_idx, a_val = a
        b_idx, b_val = b

        def evaluate(expr: Union[bool, torch.SymInt, sympy.Expr]) -> bool:
            if isinstance(expr, bool):
                return expr
            return shape_env.evaluate_expr(expr, size_oblivious=True)

        if evaluate(a_val < b_val):
            return -1
        if evaluate(a_val > b_val):
            return 1
        # If strides are the same, prefer the original order.
        # (this matches argsort's algorithm).
        # For strides = [2048, 2048, 16, 1], this is
        # [3, 2, 1, 0].
        if a_idx < b_idx:
            return 1
        if a_idx > b_idx:
            return -1
        return 0

    # Strategy: convert all symints to sympy.Expr, then use a custom comparator
    exprs = [
        (idx, s.node.expr if isinstance(s, torch.SymInt) else s)
        for idx, s in enumerate(seq)
    ]
    exprs = sorted(exprs, key=functools.cmp_to_key(cmp))
    result = [idx for idx, _ in exprs]
    return result


@functools.lru_cache(8)
def get_dtype_size(dtype: torch.dtype) -> int:
    # TODO: Investigate why uint64 tensor creation causes overflow error:
    # Workaround for RuntimeError in memory size calculation, but underlying cause unclear
    if dtype == torch.uint64:
        return 8
    return torch.empty((), dtype=dtype).element_size()


class LineContext(NamedTuple):
    context: Any


@dataclasses.dataclass
class ValueWithLineMap:
    value: str
    line_map: list[tuple[int, LineContext]]


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        self._lines: list[Union[DeferredLineBase, LineContext, str]] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> ValueWithLineMap:
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            assert isinstance(line, str)
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    def prefix(self) -> str:
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        self.writeline("\n")

    def writeline(se
```



## High-Level Overview


This Python file contains 21 class(es) and 275 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `align`, `GraphPartitionMap`, `CachedMethod`, `LineContext`, `ValueWithLineMap`, `IndentedBuffer`, `FakeIndentedBuffer`, `DeferredLineBase`, `DelayReplaceLine`, `DelayMaybeLine`, `CKGemmOperation`, `DebugDirManager`, `DummyModule`, `Placeholder`, `BoxedBool`, `OpDtypeRule`, `ScopedDict`, `TritonAttrsDescriptorVersion`, `CUDAGraphWrapperMetadata`, `CUDAGraphWrapper`

**Functions defined**: `get_gpu_type`, `_align`, `_is_aligned`, `eval`, `fp8_bench`, `do_bench_using_profiling`, `_do_bench_using_profiling`, `has_torchvision_roi_align`, `decode_device`, `sympy_product`, `sympy_dot`, `unique`, `ceildiv`, `_type_of`, `convert_shape_to_inductor`, `convert_to_symint`, `convert_shape_to_symint`, `is_view`, `is_pointwise_use`, `gen_gm_and_inputs`

**Key imports**: annotations, collections, contextlib, dataclasses, enum, functools, importlib, inspect, io, itertools


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `collections`
- `contextlib`
- `dataclasses`
- `enum`
- `functools`
- `importlib`
- `inspect`
- `io`
- `itertools`
- `logging`
- `math`
- `operator`
- `os`
- `platform`
- `re`
- `shutil`
- `statistics`
- `sys`
- `sysconfig`
- `tempfile`
- `textwrap`
- `time`
- `unittest`
- `datetime`: datetime
- `typing_extensions`: dataclass_transform, ParamSpec, Self
- `sympy`
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


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
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
