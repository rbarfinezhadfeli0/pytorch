# Documentation: `docs/torch/_inductor/codegen/common.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/common.py_docs.md`
- **Size**: 54,242 bytes (52.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/common.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/common.py`
- **Size**: 104,446 bytes (102.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import math
import operator
import os
import re
import tempfile
from abc import ABC, abstractmethod
from enum import auto, Enum
from itertools import chain
from typing import (
    Any,
    cast,
    ClassVar,
    Generic,
    NamedTuple,
    Optional,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self, TypeVar

import sympy

import torch
import torch.fx
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils import _pytree as pytree
from torch.utils._config_module import ConfigModule
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter as _PythonPrinter
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config, metrics
from ..dtype_propagation import DtypePropagationOpsHandler
from ..ops_handler import BasicMathOpsMixin, DefaultHandler
from ..shape_propagation import ShapePropagationOpsHandler
from ..utils import (
    boolean_ops,
    DeferredLineBase,
    generate_assert,
    get_current_backend,
    IndentedBuffer,
    ir_dataclass,
    ScopedDict,
    sympy_dot,
    sympy_index_symbol,
    sympy_subs,
    triton_type,
    unique,
)
from ..virtualized import (
    NullHandler,
    ops,
    OpsHandler,
    OpsValue,
    ReductionType,
    StoreMode,
    V,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, MutableMapping, Sequence

    from torch.fx import GraphModule

    from ..custom_graph_pass import CustomGraphModulePass
    from ..ir import Buffer, ChoiceCaller, FixedLayout, IRNode
    from ..loop_body import LoopBody
    from ..scheduler import BaseScheduling, Scheduler, SchedulerNode
    from ..shape_propagation import BlockShapeType
    from .wrapper import PythonWrapperCodegen

    _T = TypeVar("_T")
    SchedulingConstructor = Callable[[Optional[Scheduler]], BaseScheduling]
    WrapperConstructor = type[PythonWrapperCodegen]
    SymbolLike = Union[str, sympy.Symbol]

    # OpVarT should really be Union[CSEVariable, str], however this
    # causes typing errors in subclasses (defined in other files).
    OpVarT = str

schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
log = logging.getLogger(__name__)


def data_type_logger(msg: str) -> None:
    if schedule_log.isEnabledFor(logging.DEBUG):
        schedule_log.debug("Data type propagation: %s", msg)


@dataclasses.dataclass
class FileBackedGraphModule:
    """
    Output of FX wrapper codegen. Exposes the same methods as ModuleType, but these
    map back to a GraphModule instead of Python source.
    """

    gm: GraphModule
    compiled_fn: Callable[..., Any]

    def __post_init__(self) -> None:
        # Write the code to a file for compatibility with debugging utilities.
        # The file is deleted upon program termination.
        self.tempfile = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        )
        atexit.register(os.remove, self.tempfile.name)
        with self.tempfile as f:
            f.write(self.value)

    @property
    def __file__(self) -> str:
        return self.tempfile.name

    def call(self, args: list[Any]) -> Any:
        return self.compiled_fn(*args)

    @property
    def value(self) -> str:
        return self.gm.code


class WorkspaceZeroMode(enum.Enum):
    UNINITIALIZED = 0
    ZERO_ON_CALL = 1  # kernel may leave workspace dirty
    ZERO_PER_GRAPH = 2  # must be re-zeroed by kernel

    @staticmethod
    def combine(a: WorkspaceZeroMode, b: WorkspaceZeroMode) -> WorkspaceZeroMode:
        if a == b or b == WorkspaceZeroMode.UNINITIALIZED:
            return a
        if a == WorkspaceZeroMode.UNINITIALIZED:
            return b
        raise NotImplementedError(f"WorkspaceZeroMode.combine({a!r}, {b!r})")

    @staticmethod
    def from_bool(zero_fill: bool) -> WorkspaceZeroMode:
        if zero_fill:
            return WorkspaceZeroMode.ZERO_ON_CALL
        return WorkspaceZeroMode.UNINITIALIZED


class CodegenSymbol(ABC):
    """
    An IR object possibly corresponding to a variable in the wrapper code.
    """

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        pass


@ir_dataclass(frozen=True)
class WorkspaceArg(CodegenSymbol):
    """A temporary buffer used for a single kernel, then discarded.

    Not registered as a traditional buffer since there are no users,
    so it would be dead code eliminated.

    Args:
        nbytes: The size of the buffer in bytes.
        zero_fill: Whether the buffer should be initialized to zero.

    """

    count: sympy.Expr
    zero_mode: WorkspaceZeroMode
    device: torch.device
    outer_name: str
    inner_name: str = "ws_ptr"
    dtype: torch.dtype = torch.uint8

    @staticmethod
    def unique_name(prefix: str = "workspace_") -> str:
        return f"{prefix}{next(V.graph.workspace_id)}"

    @staticmethod
    def can_join(a: WorkspaceArg, b: WorkspaceArg) -> bool:
        return (
            a.inner_name == b.inner_name and a.dtype == b.dtype and a.device == b.device
        )

    @staticmethod
    def join(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
        return WorkspaceArg(
            count=a.count + b.count,
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    @staticmethod
    def maximum(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
        assert (
            a.dtype == b.dtype and a.device == b.device and a.inner_name == b.inner_name
        )
        return WorkspaceArg(
            count=sympy.Max(a.count, b.count),
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    # These methods let WorkspaceArg pretend it is a buffer to reuse allocation code
    def get_device(self) -> torch.device:
        return self.device

    get_device_or_error = get_device

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        return self.get_layout().get_example()

    def get_layout(self) -> FixedLayout:
        from ..ir import FixedLayout

        return FixedLayout(
            device=self.device,
            dtype=self.dtype,
            size=[self.count],
            stride=[1],
        )

    @property
    def layout(self) -> FixedLayout:
        return self.get_layout()

    get_output_spec = get_layout
    maybe_get_output_spec = get_layout
    maybe_get_layout = get_layout

    def get_offset(self) -> sympy.Expr:
        return sympy.S.Zero

    def get_size(self) -> list[sympy.Expr]:
        return [self.count]

    def get_stride(self) -> list[sympy.Expr]:
        return [sympy.S.One]

    def get_name(self) -> str:
        return self.outer_name

    def get_is_pinned(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> list[str]:
        return []


class TritonScratchWorkspace:
    def __init__(self, size: int, generate_dtype_str: Callable[..., str]):
        self.size = size
        self._generate_dtype_str = generate_dtype_str

    def generate_dtype_str(self) -> str:
        return self._generate_dtype_str()


@dataclasses.dataclass
class TensorArg:
    name: str
    buffer: str
    dtype: torch.dtype
    offset: sympy.Expr = sympy.S.Zero  # c++ only
    alias_of: Optional[str] = None  # halide only


@dataclasses.dataclass
class SizeArg:
    name: str
    expr: sympy.Expr

    @property
    def alias_of(self) -> Optional[str]:
        return None


@dataclasses.dataclass
class ConstexprArg:
    name: str


@dataclasses.dataclass
class TMADescriptorArg:
    name: str
    api_type: str  # "experimental" or "stable"
    block_shape: Optional[list[sympy.Expr]]  # only needed for "stable"
    dtype: Optional[torch.dtype]  # only needed for "stable"


@dataclasses.dataclass
class DeviceCodegen:
    scheduling: SchedulingConstructor
    wrapper_codegen: WrapperConstructor
    cpp_wrapper_codegen: Optional[WrapperConstructor] = None
    fx_wrapper_codegen: Optional[WrapperConstructor] = None


KernelArgType = Union[WorkspaceArg, TensorArg, SizeArg, TMADescriptorArg, ConstexprArg]

device_codegens: dict[str, DeviceCodegen] = {}


class DeviceOpOverrides:
    def import_get_raw_stream_as(self, name: str) -> str:
        raise NotImplementedError

    def set_device(self, device_idx: int) -> str:
        raise NotImplementedError

    def synchronize(self) -> str:
        raise NotImplementedError

    def device_guard(self, device_idx: int) -> str:
        raise NotImplementedError

    def cpp_device_guard(self) -> str:
        raise NotImplementedError

    def cpp_aoti_device_guard(self) -> str:
        raise NotImplementedError

    def cpp_stream_guard(self) -> str:
        raise NotImplementedError

    def cpp_aoti_stream_guard(self) -> str:
        raise NotImplementedError

    def cpp_getStreamFromExternal(self) -> str:
        raise NotImplementedError

    def kernel_header(self) -> str:
        raise NotImplementedError

    def kernel_driver(self) -> str:
        raise NotImplementedError

    def cpp_stream_type(self) -> str:
        raise NotImplementedError

    def aoti_get_stream(self) -> str:
        raise NotImplementedError

    def cpp_kernel_type(self) -> str:
        raise NotImplementedError

    def cpp_device_ptr(self) -> str:
        raise NotImplementedError

    def tma_descriptor_helpers(self) -> str:
        raise NotImplementedError

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: Optional[str] = None
    ) -> Optional[tuple[list[str], str]]:
        # optionally return (scratch definition, arg name)
        raise NotImplementedError


device_op_overrides_dict: dict[str, DeviceOpOverrides] = {}
custom_backend_passes: dict[str, Optional[CustomGraphModulePass]] = {}
custom_backend_codegen_configs: dict[str, Optional[ConfigModule]] = {}


# The code generated by Inductor consists of two main parts: kernel code and wrapper code.
# For any new backend looking to integrate with Inductor, customization of these two main
# parts are necessary to generate its specific code.
#
# Kernel code generation is determined by different Scheduling. Consequently, a new
# backend needs to provide a custom Scheduling for its unique kernel code generation. Currently,
# CppScheduling and TritonScheduling serve the C++/OpenMP and Triton backends, respectively.
#
# For the Wrapper, Inductor provides a PythonWrapperCodegen class to generate the Python wrapper code
# that bridges kernels. This allows out-of-tree backends to inherit from PythonWrapperCodegen,
# and override specific member functions to create backend-specific Python wrapper code.
#
# Other classes, such as CppKernel and TritonKernel, used for code generation, typically form part
# of the logic for either Scheduling or PythonWrapperCodegen. So the Scheduling and PythonWrapperCodegen interfaces
# provide flexibility to the backend. A backend can choose to implement these classes from scratch,
# or reuse them by extending and overriding as necessary. And Inductor provides the registration API,
# register_backend_for_device, to equip a new backend at runtime.
#
# Intel has developed a new backend on top of Triton to support Intel GPUs, leveraging these interfaces.
# This backend can be used as a reference:
# https://github.com/intel/intel-extension-for-pytorch/blob/5dcc9d57e5422cf295e1a1ee97896d6b6a554a85/intel_extension_for_pytorch/_inductor/__init__.py#L9
def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: Optional[WrapperConstructor] = None,
    device_fx_wrapper_codegen: Optional[WrapperConstructor] = None,
    device_custom_pass: Optional[CustomGraphModulePass] = None,
    device_custom_config: Optional[ConfigModule] = None,
) -> None:
    device_codegens[device] = DeviceCodegen(
        device_scheduling,
        device_wrapper_codegen,
        device_cpp_wrapper_codegen,
        device_fx_wrapper_codegen,
    )
    custom_backend_passes[device] = device_custom_pass
    if device_custom_config:
        assert (
            isinstance(device_custom_config, ConfigModule)
            and device_custom_config is not config
        ), (
            f"{device_custom_config=} cannot be the same as the default inductor config {config=}"
        )
    custom_backend_codegen_configs[device] = device_custom_config


class BackendFeature(Enum):
    FOREACH = auto()
    BUCKETIZE = auto()
    INPLACE_BUFFERS = auto()
    MASKED_SCATTER_WITH_INDEX = auto()
    SCAN = auto()
    SORT = auto()
    TUPLE_REDUCTION = auto()
    PREFER_STORE_LOOP_ORDER = auto()
    TRITON_TEMPLATES = auto()
    REDUCE_TO_SINGLE_ELEMENT = auto()


def get_backend_features(
    device: Union[torch.device, str, None],
) -> OrderedSet[BackendFeature]:
    if device is None:
        return OrderedSet()
    init_backend_registration()
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        assert isinstance(device, str), type(device)
        device_type = device
        device = torch.device(device_type)
    scheduling_ctor = get_scheduling_for_device(device_type)
    assert scheduling_ctor
    scheduling = scheduling_ctor(None)
    return scheduling.get_backend_features(device)


def has_backend_feature(
    device: Union[torch.device, str, None], feature: BackendFeature
) -> bool:
    """See also V.graph.has_feature"""
    assert isinstance(feature, BackendFeature)
    return feature in get_backend_features(device)


def get_scheduling_for_device(device: str) -> Optional[SchedulingConstructor]:
    return device_codegens[device].scheduling if device in device_codegens else None


def get_wrapper_codegen_for_device(
    device: str, cpp_wrapper: bool = False, fx_wrapper: bool = False
) -> Optional[WrapperConstructor]:
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        if fx_wrapper:
            return wrapper_codegen_obj.fx_wrapper_codegen
        elif cpp_wrapper:
            return wrapper_codegen_obj.cpp_wrapper_codegen
        else:
            return wrapper_codegen_obj.wrapper_codegen
    return None


def get_custom_backend_pass_for_device(device: str) -> Optional[CustomGraphModulePass]:
    return custom_backend_passes.get(device)


def get_custom_backend_config_for_device(device: str) -> Optional[ConfigModule]:
    return custom_backend_codegen_configs.get(device)


@functools.cache
def init_backend_registration() -> None:
    """
    Register the backend for different devices, including the scheduling
    for kernel code generation and the host side wrapper code generation.
    """
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef
    from .cpp_wrapper_gpu import CppWrapperGpu
    from .cpp_wrapper_mps import CppWrapperMps
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .mps import MetalScheduling
    from .pallas import PallasScheduling
    from .python_wrapper_mtia import PythonWrapperMtia
    from .triton import TritonScheduling
    from .wrapper import PythonWrapperCodegen
    from .wrapper_fxir import WrapperFxCodegen

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {
            "cpp": CppScheduling,
            "halide": HalideScheduling,
            "triton": TritonScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cpu",
            lambda scheduling: cpu_backends[config.cpu_backend](scheduling),
            PythonWrapperCodegen,
            CppWrapperCpuArrayRef
            if config.aot_inductor.allow_stack_allocation
            else CppWrapperCpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {
            "triton": CUDACombinedScheduling,
            "halide": HalideScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cuda",
            lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device(
            "xpu",
            TritonScheduling,
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("mps") is None:
        register_backend_for_device(
            "mps",
            MetalScheduling,
            PythonWrapperCodegen,
            CppWrapperMps,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("mtia") is None:
        register_backend_for_device(
            "mtia",
            TritonScheduling,
            PythonWrapperMtia,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    private_backend = torch._C._get_privateuse1_backend_name()
    if (
        private_backend != "privateuseone"
        and get_scheduling_for_device(private_backend) is None
    ):
        from torch.utils.backend_registration import _get_custom_mod_func

        try:
            device_scheduling = _get_custom_mod_func("Scheduling")
            wrapper_codegen = _get_custom_mod_func("PythonWrapperCodegen")
            cpp_wrapper_codegen = _get_custom_mod_func("CppWrapperCodegen")
            fx_wrapper_codegen = _get_custom_mod_func("WrapperFxCodegen")
            if device_scheduling and wrapper_codegen and cpp_wrapper_codegen:
                register_backend_for_device(
                    private_backend,
                    device_scheduling,
                    wrapper_codegen,
                    cpp_wrapper_codegen,
                    fx_wrapper_codegen,
                )
        except RuntimeError:
            pass


def index_prevent_reordering(
    index: Sequence[sympy.Expr],
    index_vars: Sequence[sympy.Expr],
    sizes: Sequence[sympy.Expr],
) -> list[sympy.Expr]:
    from ..ir import FlexibleLayout

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def register_device_op_overrides(
    device: str, device_op_overrides: DeviceOpOverrides
) -> None:
    device_op_overrides_dict[device] = device_op_overrides


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    assert isinstance(device, str), type(device)

    if not device_op_overrides_dict:
        from . import cpu_device_op_overrides, mps_device_op_overrides  # noqa: F401
        from .cuda import device_op_overrides  # noqa: F401
        from .mtia import device_op_overrides as mtia_op_overrides  # noqa: F401
        from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

    return device_op_overrides_dict[device]


DTYPE_TO_COMPUTATION_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.bfloat16: torch.float,
    torch.float16: torch.float,
    **{
        dtype: dtype
        for dtype in [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ]
    },
}


def deduce_output_dtype_by_name(
    op_name: str,
    *args: Any,
    **kwargs: Any,
) -> Optional[torch.dtype]:
    """
    Given op name and a list of input dtypes, deduce the output dtype
    """
    if op_name in boolean_ops():
        return torch.bool
    elif op_name in (
        "to_dtype",
        "index_expr",
    ):
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "rand",
        "randn",
    ):
        return torch.float
    elif op_name in (
        "get_index",
        "randint64",
        "load_seed",
    ):
        return torch.int64
    elif op_name == "reduction":
        return kwargs["dtype"] if "dtype" in kwargs else args[1]
    elif op_name == "constant":
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "load",
        "store",
        "store_reduction",
    ):
        buf_name = args[1]
        return V.graph.get_dtype(buf_name)  # type: ignore[arg-type]
    elif op_name == "to_dtype_bitcast":
        return kwargs["dtype"] if "dtype" in kwargs else args[-2]
    return None


def check_dtype(
    buffer: IndentedBuffer, var: CSEVariableType, dtype: torch.dtype
) -> None:
    backend = get_current_backend()
    if config.test_configs.runtime_triton_dtype_assert and backend == "triton":
        buffer.writeline(f"tl.static_assert({var}.dtype == {triton_type(dtype)})")
    elif config.test_configs.static_cpp_dtype_assert and backend == "cpp":
        from .cpp_utils import CppCSEVariable, DTYPE_TO_CPP

        assert isinstance(var, CppCSEVariable), type(var)
        if dtype == torch.bool:
            if var.is_vec:
                is_same_dt = f"IsVecMaskType<decltype({var})>::value"
            else:
                # operator&(bool, bool) returns int and it can be used as boolean in C++
                is_same_dt = f"std::is_same_v<decltype({var}), bool> || std::is_same_v<decltype({var}), int>"
        else:
            c_var_type = f"decltype({var})"
            if var.is_vec:
                c_var_type = f"typename {c_var_type}::value_type"
            is_same_dt = f"std::is_same_v<{c_var_type}, {DTYPE_TO_CPP[dtype]}>"

        buffer.writeline(f"static_assert({is_same_dt});")


def check_shape(
    buffer: IndentedBuffer, var: CSEVariableType, shape: BlockShapeType
) -> None:
    backend = get_current_backend()
    assert shape is not None
    if config.test_configs.runtime_triton_shape_assert and backend == "triton":
        shape_str = (
            ", ".join(str(d) for d in shape) if len(shape) != 1 else f"{shape[0]},"
        )
        buffer.writeline(f"tl.static_assert({var}.shape == ({shape_str}))")


def check_nan(buffer: IndentedBuffer, var: CSEVariableType) -> None:
    backend = get_current_backend()
    if backend == "triton":
        msg = "NaN or Inf found"
        buffer.writeline(
            f"tl.device_assert(({var} == {var}) & ({var} != float('inf')) & ({var} != float('-inf')), '{msg}')"
        )


class DataTypePropagation:
    def __init__(self, body: LoopBody) -> None:
        self.body = body
        self.graphs: dict[Union[Callable[..., Any], str], Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node) -> Optional[torch.dtype]:
        inputs = node.all_input_nodes
        input_nodes = [
            n for n in inputs if isinstance(n, torch.fx.Node) and n.op != "placeholder"
        ]
        if len(input_nodes) == 0:
            return None

        all_input_nodes_propagated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propagated:
            return None

        return functools.reduce(
            torch.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )

    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node) -> torch.dtype:
        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        assert dtype
        return dtype

    def deduce_node_dtype(self, node: torch.fx.Node) -> Optional[torch.dtype]:
        if node.op == "placeholder":
            return None

        if node.target == "output" and len(node.args) != 1:
            # we can infer output node if it only have 1 arg
            return None

        if node.target is operator.getitem:
            node_arg = node.args[0]
            assert isinstance(node_arg, torch.fx.Node), type(node_arg)
            return self.deduce_node_dtype(node_arg)

        assert isinstance(node.target, str), type(node.target)

        if node.target.startswith("masked_subblock"):
            return self.deduce_node_dtype_by_subgraph(node)

        if (
            output_dtype := deduce_output_dtype_by_name(
                node.target,
                *node.args,
                **node.kwargs,
            )
        ) is not None:
            return output_dtype

        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: torch.fx.Graph) -> Optional[torch.dtype]:
        assert graph.nodes
        graph_dtype: Optional[torch.dtype] = None
        # For masked_subblock, we use output's dtype to represent
        # the dtype of this subgraph. For other cases, graph_dtype
        # might be None
        for node in graph.nodes:
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()

            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx
            if node.target == "output":
                graph_dtype = opt_ctx.dtype
        return graph_dtype

    def propagate(self) -> Optional[torch.dtype]:
        return self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body: LoopBody) -> Optional[torch.dtype]:
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node: SchedulerNode) -> Optional[torch.dtype]:
        from ..loop_body import LoopBody
        from ..scheduler import SchedulerNode

        assert isinstance(node, SchedulerNode), type(node)
        assert isinstance(node._body, LoopBody), type(node._body)
        return DataTypePropagation.propagate_loopbody(node._body)


class PythonPrinter(_PythonPrinter):
    def doprint(
        self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True
    ) -> str:
        # TODO: why are people passing strings to the printer here :think:
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)

    def parenthesize(self, item: sympy.Expr, level: int, strict: bool = False) -> str:
        if isinstance(item, sympy.Mod):
            # use parenthesis to enforce precedence.
            # in sympy 1.13.3, -2*Mod(x,y) becomes -2*x%y, which is wrong.
            return f"({self._print(item)})"
        else:
            return super().parenthesize(item, level, strict)


class OpDecompositions:
    """
    Decomposes inductor ops
    """

    @staticmethod
    def identity(value: OpVarT) -> OpVarT:
        # used to trigger cse
        return value

    @staticmethod
    def reciprocal(x: OpVarT) -> OpVarT:
        return ops.truediv(ops.constant(1, torch.int32), x)

    @staticmethod
    def square(x: OpVarT) -> OpVarT:
        return ops.mul(x, x)

    @staticmethod
    def erfc(x: OpVarT) -> OpVarT:
        return ops.sub(ops.constant(1, torch.float32), ops.erf(x))

    @staticmethod
    def erfcx(x: OpVarT) -> OpVarT:
        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x: OpVarT) -> OpVarT:
        return ops.sub(ops.exp(x), ops.constant(1, torch.float32))

    @staticmethod
    def log10(x: OpVarT) -> OpVarT:
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), torch.float32))

    @staticmethod
    def log2(x: OpVarT) -> OpVarT:
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), torch.float32))

    @staticmethod
    def exp2(x: OpVarT) -> OpVarT:
        return ops.exp(ops.mul(x, ops.constant(math.log(2), torch.float32)))

    @staticmethod
    def log1p(x: OpVarT) -> OpVarT:
        return ops.log(ops.add(x, ops.constant(1, torch.int32)))

    @staticmethod
    def sigmoid(x: OpVarT) -> OpVarT:
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def relu(x: OpVarT) -> OpVarT:
        return ops.maximum(x, ops.constant(0, torch.int32))

    @staticmethod
    def fma(x: OpVarT, y: OpVarT, z: OpVarT) -> OpVarT:
        # for backends that don't override this (halide)
        return ops.add(ops.mul(x, y), z)

    @staticmethod
    def floor_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.floor(a), dtype)

    @staticmethod
    def ceil_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.ceil(a), dtype)

    @staticmethod
    def trunc_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.trunc(a), dtype)

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> OpVarT:
        r = ops.mod(a, b)
        cond = ops.and_(
            ops.ne(r, ops.constant(0, torch.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        return ops.where(cond, ops.add(r, b), r)

    @staticmethod
    def round_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        return ops.to_dtype(ops.round(a), dtype)


_RE_PAREN_NOT_NEEDED = re.compile(r"[a-z0-9_.]+|\([^)]*\)|", flags=re.IGNORECASE)


def _all_in_parens(string: str) -> bool:
    if string[0] != "(" or len(string) < 2:
        return False
    count = 1
    for i, char in enumerate(string[1:]):
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count == 0 and i != len(string) - 2:
            return False
    assert count == 0
    return True


class OpOverrides(BasicMathOpsMixin, OpDecompositions, OpsHandler[Any]):
    @staticmethod
    def paren(string: OpVarT) -> OpVarT:
        if (
            isinstance(string, CSEVariable)
            or _RE_PAREN_NOT_NEEDED.fullmatch(string)
            or _all_in_parens(string)
        ):
            # don't put extra parens for strings that are already wrapped in parens
            # pyrefly: ignore [bad-return]
            return string
        return f"({string})"

    @staticmethod
    def constant(value: Union[bool, float, int], dtype: torch.dtype) -> OpVarT:
        return repr(value)

    @staticmethod
    def bitwise_not(x: OpVarT) -> OpVarT:
        return f"~{OpOverrides.paren(x)}"

    @staticmethod
    def logical_not(a: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} & {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_or(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} | {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_xor(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} ^ {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_left_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} << {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_right_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        return f"{OpOverrides.paren(x)} >> {OpOverrides.paren(y)}"

    @staticmethod
    def int_truediv(a: OpVarT, b: OpVarT) -> OpVarT:
        # TODO: this is wrong
        # TODO: an easy bandaid is to generate runtime asserts that it's
        # <= 2**53, which is when this equation is correct
        return ops.truediv(a, b)

    @staticmethod
    def load_seed(name: str, offset: OpVarT) -> OpVarT:
        return ops.load(name, sympy.Integer(offset))

    def indirect_indexing(
        self,
        var: OpVarT,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        return sympy_index_symbol(str(var))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: check_bounds should be handled by CSEProxy"
        )

    def load(self, name: str, index: sympy.Expr) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: load should be handled by CSEProxy"
        )

    def store(
        self, name: str, index: sympy.Expr, value: OpVarT, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store should be handled by CSEProxy"
        )

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def store_reduction(self, name: str, index: sympy.Expr, value: OpVarT) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store_reduction should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[OpVarT, tuple[OpVarT, ...]],
    ) -> Union[OpVarT, tuple[OpVarT, ...]]:
        raise NotImplementedError(
            f"{type(self).__name__}: reduction should be handled by CSEProxy"
        )

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[OpVarT, ...], tuple[OpVarT, ...]],
            tuple[OpVarT, ...],
        ],
        values: tuple[OpVarT, ...],
    ) -> tuple[OpVarT, ...]:
        raise NotImplementedError(
            f"{type(self).__name__}: scan should be handled by CSEProxy"
        )

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[OpVarT, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[OpVarT, ...]:
        raise NotImplementedError(
            f"{type(self).__name__}: sort should be handled by CSEProxy"
        )

    def bucketize(
        self,
        values: OpVarT,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: OpVarT,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[OpVarT] = None,
    ) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: bucketize should be handled by CSEProxy"
        )

    def halide_clamp(self, value: OpVarT, size: sympy.Expr, check: bool) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: halide_clamp only implemented for Halide backend"
        )

    def dot(self, x: OpVarT, y: OpVarT) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: dot only implemented for Triton backend"
        )

    def inline_asm_elementwise(
        self,
        *inputs: OpVarT,
        asm: str,
        constraints: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
    ) -> OpVarT:
        raise NotImplementedError(
            f"{type(self).__name__}: inline_asm_elementwise only implemented for Triton backend"
        )

    def output(self, *args: OpVarT) -> None:
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear at codegen time"
        )

    def placeholder(self, index: int) -> OpVarT:
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear at codegen time"
        )

    @staticmethod
    def _unimplemented(name: str) -> Callable[..., OpVarT]:
        def unimplemented(self: OpOverrides, *args: Any, **kwargs: Any) -> OpVarT:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement ops.{name}"
            )

        unimplemented.__name__ = name
        unimplemented.is_unimplemented = True  # type: ignore[attr-defined]
        return unimplemented

    @classmethod
    def _is_unimplemented(cls, name: str) -> bool:
        fn = getattr(cls, name, None)
        default_fn = getattr(OpsHandler, name, None)
        return not fn or fn == default_fn or getattr(fn, "is_unimplemented", False)

    @classmethod
    def _initialize_pointwise_overrides(cls, target: str) -> None:
        assert target in ("triton", "cpp", "cppvec", "halide", "mps"), target

        for funcname, data in pointwise_overrides_data.items():
            impl = getattr(data, target)
            if impl is None:
                if cls._is_unimplemented(funcname):
                    setattr(cls, funcname, cls._unimplemented(funcname))
            else:
                assert funcname not in cls.__dict__, (
                    f"multiple definitions of {funcname} on {cls.__name__}"
                )
                impl.__name__ = funcname
                setattr(cls, funcname, staticmethod(impl))


@dataclasses.dataclass
class OverridesData:
    name: str
    cpp: Callable[..., str]
    # None when not impl in libdevice/triton
    triton: Optional[Callable[..., str]] = None
    # None when not impl in aten/.../vec
    cppvec: Optional[Callable[..., str]] = None
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND = (
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    halide: Optional[Callable[..., str]] = None
    mps: Optional[Callable[..., str]] = None


# NB: if you add a new special function, don't forget to update
# torch._inductor.ops_handler too
pointwise_overrides_data: dict[str, OverridesData] = dict(
    airy_ai=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"airy_ai_forward({x})",
        name="special_airy_ai",
    ),
    bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j0_forward({x})",
        triton=lambda x: f"libdevice.j0({x})",
        name="special_bessel_j0",
    ),
    bessel_j1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j1_forward({x})",
        triton=lambda x: f"libdevice.j1({x})",
        name="special_bessel_j1",
    ),
    bessel_y0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y0_forward({x})",
        triton=lambda x: f"libdevice.y0({x})",
        name="special_bessel_y0",
    ),
    bessel_y1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y1_forward({x})",
        triton=lambda x: f"libdevice.y1({x})",
        name="special_bessel_y1",
    ),
    digamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_digamma({x})",
        cppvec=lambda x: f"{x}.digamma()",
        name="digamma",
    ),
    # no cpp nor triton implementation for entr, it is defined as decomposition
    # erf, erfc
    erfcx=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_erfcx({x})",
        triton=lambda x: f"libdevice.erfcx({x})",
        name="special_erfcx",
    ),
    fma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y, z: f"std::fma({x}, {y}, {z})",
        cppvec=lambda x, y, z: f"fmadd({x}, {y}, {z})",
        triton=lambda x, y, z: f"libdevice.fma({x}, {y}, {z})",
        name="fma",
    ),
    # erfinv, exp2, expit, gammaln
    igamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="igamma",
    ),
    igammac=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="igammac",
    ),
    gammainc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="special_gammainc",
    ),
    gammaincc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="special_gammaincc",
    ),
    i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        cppvec=lambda x: f"{x}.i0()",
        name="i0",
    ),
    i0e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0e({x})",
        cppvec=lambda x: f"{x}.i0e()",
        name="special_i0e",
    ),
    i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_i1",
    ),
    i1e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1e({x})",
        name="special_i1e",
    ),
    log_ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_log_ndtr({x})",
        name="special_log_ndtr",
    ),
    # logit
    modified_bessel_i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i0_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i0({x})",
        name="special_modified_bessel_i0",
    ),
    modified_bessel_i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i1_forward({x})",
        triton=lambda x: f"libdevice.cyl_bessel_i1({x})",
        name="special_modified_bessel_i1",
    ),
    modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k0_forward({x})",
        name="special_modified_bessel_k0",
    ),
    modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k1_forward({x})",
        name="special_modified_bessel_k1",
    ),
    # multigamma
    ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtr({x})",
        name="special_ndtr",
    ),
    ndtri=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtri({x})",
        name="special_ndtri",
    ),
    polygamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x,
        y: f"{x} == 0 ? calc_digamma({y}) : ({x} == 1 ? trigamma({y}) : calc_polygamma({y}, {x}))",
        name="polygamma",
    ),
    # psi - alias to digamma
    # round
    scaled_modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k0_forward({x})",
        name="special_scaled_modified_bessel_k0",
    ),
    scaled_modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k1_forward({x})",
        name="special_scaled_modified_bessel_k1",
    ),
    # sinc
    spherical_bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"spherical_bessel_j0_forward({x})",
        name="special_spherical_bessel_j0",
    ),
    zeta=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"zeta({x}, {y})",
        name="special_zeta",
    ),
    chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_t_forward({x}, {y})",
        name="special_chebyshev_polynomial_t",
    ),
    chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_u_forward({x}, {y})",
        name="special_chebyshev_polynomial_u",
    ),
    chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_v_forward({x}, {y})",
        name="special_chebyshev_polynomial_v",
    ),
    chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_w_forward({x}, {y})",
        name="special_chebyshev_polynomial_w",
    ),
    legendre_polynomial_p=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"legendre_polynomial_p_forward({x}, {y})",
        name="special_legendre_polynomial_p",
    ),
    shifted_chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_t_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_t",
    ),
    shifted_chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_u_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_u",
    ),
    shifted_chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_v_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_v",
    ),
    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
)


def is_buffer_removed(name: str) -> bool:
    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )


class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name: str, line: str):
        super().__init__(line)
        self.name = name
        assert not isinstance(line, DeferredLineBase)

    def __call__(self) -> Optional[str]:
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> DeferredLine:
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(NamedTuple):
    inner_name: str
    other_names: list[str]


@dataclasses.dataclass
class ArgName:
    name: str
    # is_constexpr=True is used to attach a " : tl.constexpr" into the argument list
    is_constexpr: bool = False

    def full_name(self) -> str:
        return f"{self.name}{' : tl.constexpr' if self.is_constexpr else ''}"


class RemovedArg:
    def __str__(self) -> str:
        return "REMOVED"


REMOVED = RemovedArg()


class KernelArgs:
    @st
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
- **Abstract Base Classes**: Defines abstract interfaces
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

- **File Documentation**: `common.py_docs.md_docs.md`
- **Keyword Index**: `common.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
