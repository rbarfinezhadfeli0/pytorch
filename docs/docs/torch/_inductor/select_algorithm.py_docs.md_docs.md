# Documentation: `docs/torch/_inductor/select_algorithm.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/select_algorithm.py_docs.md`
- **Size**: 54,202 bytes (52.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/select_algorithm.py`

## File Metadata

- **Path**: `torch/_inductor/select_algorithm.py`
- **Size**: 158,021 bytes (154.32 KB)
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
import hashlib
import inspect
import itertools
import json
import logging
import math
import operator
import os
import re
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, Optional, TYPE_CHECKING, Union
from typing_extensions import Self
from unittest.mock import patch

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_chromium_event_logger,
    identity,
    preserve_rng_state,
)
from torch._inductor.await_utils import await_sync
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._filelock import FileLock
from torch.utils._ordered_set import OrderedSet

from ..utils._sympy.functions import CeilDiv
from . import config, ir
from .autotune_process import (
    TensorMeta,
    TritonBenchmarkRequest,
    TritonCPUBenchmarkRequest,
    TritonGPUBenchmarkRequest,
)
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import (
    CSEVariable,
    IndentedBuffer,
    KernelTemplate,
    OpOverrides,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .codegen.simd_kernel_features import SIMDKernelFeatures
from .codegen.subgraph import SubgraphChoiceCaller
from .codegen.triton import (
    gen_common_triton_imports,
    texpr,
    TMACompatibilityChecker,
    TritonKernel,
    TritonScheduling,
)
from .codegen.triton_utils import config_of, equal_1_arg_indices, signature_to_meta
from .codegen.wrapper import pexpr
from .exc import CUDACompileError
from .fx_utils import count_flops_fx
from .ir import ChoiceCaller, PrimitiveInfoType
from .ops_handler import StoreMode
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties
from .runtime.triton_compat import HAS_WARP_SPEC
from .runtime.triton_heuristics import FixedGrid
from .utils import (
    ceildiv,
    do_bench_using_profiling,
    FakeIndentedBuffer,
    get_dtype_size,
    is_gpu,
    Placeholder,
    restore_stdout_stderr,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    triton_type,
    triton_type_to_torch,
    unique,
)
from .virtualized import V


log = logging.getLogger(__name__)

# correctness checks struggle with fp16/tf32
VERIFY: dict[str, Any] = {}
PRINT_AUTOTUNE = True
DEBUG = False


if TYPE_CHECKING:
    import concurrent

    from torch._inductor.codegen.simd import IterationRangesEntry, IterationRangesRoot

    from .codegen.common import CSE


class KernelNamespace:
    pass


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()


@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: list[torch.Tensor]
    output_tensor: Optional[torch.Tensor]

    def unpack(self):
        return self.input_tensors, self.output_tensor


@dataclasses.dataclass
class AutotuneArgs:
    """During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """

    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: Optional[torch.Tensor] = None

    def get_benchmark_tensors(self, extern=False) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
        bench_tensors = self.extern if extern else self.triton
        return bench_tensors

    @classmethod
    def from_choice_args(
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> Self:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
        return cls(
            triton=BenchmarkTensors(example_inputs, out),
            extern=BenchmarkTensors(example_inputs_extern, out_extern),
            expected=expected,
        )

    def verify(self, **kwargs):
        """Verify the correctness of the benchmarking results"""

        torch.testing.assert_close(self.extern.output_tensor, self.expected, **kwargs)


class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    HookFn = Callable[[], str]

    def __init__(
        self, code: str, replacement_hooks: dict[str, Optional[HookFn]]
    ) -> None:
        super().__init__()
        self._code: str = code
        self.replacement_hooks: dict[str, Optional[PartialRender.HookFn]] = (
            replacement_hooks
        )

    @property
    def code(self) -> str:
        """
        The fully rendered code. Will **error** if any hooks have yet to be
        finalized.
        """
        remaining_active_hooks = [
            key for key, fn in self.replacement_hooks.items() if fn is not None
        ]
        assert len(remaining_active_hooks) == 0, (
            f"The following hooks have not yet been finalized:\n {remaining_active_hooks=}"
        )
        return self._code

    def finalize_hook(self, hook_key: str, strict: bool = True) -> None:
        """
        Finalize a hook by name.

        :param strict: If ``True``, raise an error if the hook wasn't found.

        NOTE: Will **error** if the hook has already been finalized.
        """
        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} not registered in self.replacement_hooks"
                )
            else:
                return

        hook = self.replacement_hooks[hook_key]
        assert hook is not None, f"Hook key {hook_key} can only be called once"
        self._code = self._code.replace(hook_key, hook())

        self.replacement_hooks[hook_key] = None

    def finalize_remaining(self) -> str:
        """
        Finalize the remaining active hooks. This function can be used in cases
        where the caller uses `finalize_hook` rather than `finalize_all`.
        Note: `finalize_all` errors if a hook that has already been finalized
        is attempted to be called again. This function only attempts to
        finalize active hooks.
        """
        for key, fn in self.replacement_hooks.items():
            if fn is not None:
                self.finalize_hook(key)
        return self.code

    def finalize_all(self) -> str:
        """
        Finalize all active hooks.

        NOTE: unlike ``finalize_remaining``, this method will **error** if any
        hook has already been finalized.
        """
        for key in self.replacement_hooks:
            self.finalize_hook(key)
        return self.code


# This is used to store info needed for lowering each subgraph in triton
# templates


@dataclasses.dataclass()
class SubgraphInfo:
    body: IndentedBuffer
    template_mask: Optional[str] = None
    template_out_shape: Optional[Union[str, tuple[str]]] = None
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]
    cse: Optional["CSE[Any]"] = None

    # only copied over if not None
    range_trees: Optional[list["IterationRangesRoot"]] = None
    range_tree_nodes: Optional[dict[sympy.Symbol, "IterationRangesEntry"]] = None
    numels: Optional[dict[str, sympy.Expr]] = None

    def __post_init__(self):
        self.only_copy_if_non_none_fields = (
            "range_trees",
            "range_tree_nodes",
            "numels",
            "cse",
        )

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class ModificationWrapper(V.WrapperHandler):  # type: ignore[name-defined]
    """Handles placeholder substitutions during subgraph processing."""

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: Optional[str],
    ):
        super().__init__(V.ops)
        self.name = f"PlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = fixed_inputs
        self.mask = mask

    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed input."""
        if name not in self.fixed_inputs:
            index_str = self._process_indexing(index)
            var = self._add_kernel_input(name)
            buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype
            line = f"tl.load({var} + {index_str})"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(
                self.kernel.compute, line, dtype=var_dtype, shape=()
            )
            return out

        return self.kernel.cse.generate(
            self.kernel.compute,
            f"({self.fixed_inputs[name]})",
            dtype=torch.float32,
            shape=(),
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        return sympy_index_symbol(str(index_var))

    # pyrefly: ignore [bad-override]
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        """Currently only supports stores for atomic adds coming from scatter nodes
        This is used by flex_attention's backwards grad for captured buffers, see
        zeros_and_scatter lowering
        """
        assert self.mask is not None, (
            "Mask is required for inner stores in modifications"
        )
        assert mode == "atomic_add", "Only atomic_add is supported for inner stores"

        buf_name = self._add_kernel_input(name)
        index_str = self._process_indexing(index)
        index_str = f"tl.broadcast_to({index_str}, {value}.shape)"
        store = f"tl.atomic_add({buf_name} + {index_str}, {value}, {self.mask}, sem='relaxed')"
        return store

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        return self.kernel.args.input(name)

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        return self.kernel.kexpr(self.kernel.rename_indexing(index))


# Function name, followed by args and kwargs.
RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]


class TritonTemplateKernel(TritonKernel):
    """
    A specialized kernel class for Triton templates that handles code generation
    for templated Triton kernels.

    This class extends TritonKernel to provide additional functionality for
    template-based kernel generation, including support for subgraphs, workspace
    arguments, and prologue/epilogue fusion.
    """

    def __init__(
        self,
        kernel_name,
        input_nodes: tuple[ir.IRNode],
        output_node,
        defines,
        num_stages,
        num_warps,
        grid_fn,
        meta,
        call_sizes,
        num_consumer_groups=0,
        num_buffers_warp_spec=0,
        use_jit=False,
        tma_store=False,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        prologue_loads_all_inputs=False,
        hint_override: Optional[int] = None,
    ) -> None:
        if tma_store:
            pass
        numel = sympy_product(output_node.get_size())
        if tma_store:
            assert len(output_node.get_size()) == 2, (
                "TMA store only supported for 2D with templates"
            )
            tiling = {
                "x": output_node.get_size()[0],
                "y": output_node.get_size()[1],
                "r0_": sympy.S.One,
            }
        else:
            tiling = {
                "x": numel,
                "r0_": sympy.S.One,
            }
        super().__init__(
            tiling,
            features=SIMDKernelFeatures([], numel),
            hint_override=hint_override,
        )
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}  # type: ignore[var-annotated]
        self.defines = defines
        self.kernel_name = kernel_name
        self.use_jit = use_jit
        self.tma_store = tma_store
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.grid_fn = grid_fn
        self.meta = meta
        self.call_sizes = call_sizes
        # for templates with fixed epilogues
        self.prefix_args = prefix_args
        self.suffix_args = suffix_args
        # pyrefly: ignore [invalid-type-var]
        self.epilogue_fn = epilogue_fn
        self.render_hooks = {}  # type: ignore[var-annotated]
        self.triton_meta: Optional[dict[str, object]] = None
        # For Templated Attention this can be a list of ir.Subgraph
        self.subgraphs: Optional[list[ir.ComputedBuffer]] = subgraphs

        # Some templates use extra global memory as a workspace
        self.workspace_arg = workspace_arg
        if workspace_arg is not None:
            self.args.workspace_args.append(workspace_arg)

        # The following attributes (body, template_mask, output_val) are all
        # used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        self.subgraph_bodies: dict[str, SubgraphInfo] = {}

        # input buffers which we are allowed to prologue fuse into
        self.prologue_supported_inputs: OrderedSet[str] = OrderedSet()

        # input buffers which we are fusing into
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        # input buffers which we are fusing into, which preserve a zero mask
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()

        # The following attributes are all used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        # NB: the names here must match the fields in SubgraphInfo
        self.body: IndentedBuffer = FakeIndentedBuffer()
        self.compute: IndentedBuffer = FakeIndentedBuffer()
        self.indexing_code: IndentedBuffer = FakeIndentedBuffer()
        self.loads: IndentedBuffer = FakeIndentedBuffer()
        self.stores: IndentedBuffer = FakeIndentedBuffer()
        self.template_mask: Optional[str] = None
        self.template_out_shape: Optional[Union[str, tuple[str]]] = None
        self.ops_handler: Optional[V.WrapperHandler] = None  # type: ignore[name-defined]

        # When caching is enabled, the generated code is not dependent on the input nodes names, or
        # symbolic sizes names.
        # However, some of the variables returned by generate_and_load that are computed during the
        # triton template expansions (code generation) are dependent on those.
        # In order to cache the code generation and avoid redoing it for similar inputs that varies only by
        # input names or symbol names, we do a record and replay method.
        # During template expansions we record all function calls that change input_dependent_preserved_state
        # and replay them on a cache hit to regenerate them.
        self.cached_replay_events: Optional[RecordedEventsType] = None

        # Update each time an input is marked frozen, used to replay the freezing of inputs on a cache hit.
        self.frozen_layouts_cnt = 0

        # When prologue_loads_all_inputs is true, prologue_supported_inputs is populated during def_kernel
        # by adding all inputs.
        self.prologue_loads_all_inputs = prologue_loads_all_inputs

        # Extra functions to be exposed during partial template rendering.
        self.extra_template_env_fns: list[Callable[..., Any]] = []

        # Tracking for intermediate variables
        self.tmp_var_ctr = itertools.count()

    def _gen_tmp_var(self) -> str:
        return f"_tmp_var{next(self.tmp_var_ctr)}"

    def input_dependent_preserved_state(self) -> str:
        # Not adding self.args.output_buffers on purpose. But we do not need to reproduce it on a cache hit.
        # (never accessed).
        return repr(
            [
                self.args.input_buffers,
                self.args.sizevars,
                self.args.workspace_args,
                self.prologue_supported_inputs,
                self.frozen_layouts_cnt,
            ]
        )

    def record_input_dependent_tracked_event(self) -> Callable[..., Any]:
        def decorator(fn) -> Callable[..., Any]:
            def wrapper(*args, **kwargs) -> Any:
                pre_state = self.input_dependent_preserved_state()
                result = fn(*args, **kwargs)
                post_state = self.input_dependent_preserved_state()
                if pre_state != post_state:
                    assert self.cached_replay_events is not None
                    self.cached_replay_events.append((fn.__name__, [*args], {**kwargs}))
                return result

            return wrapper

        return decorator

    def replay_cached_events(self, events: RecordedEventsType) -> None:
        for f, args, kwargs in events:
            getattr(self, f)(*args, **kwargs)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        assert all(
            hasattr(self, field.name) for field in dataclasses.fields(SubgraphInfo)
        )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(SubgraphInfo)
        }

        assert body_name in self.subgraph_bodies, body_name

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in subgraph.only_copy_if_non_none_fields:
                continue
            setattr(self, key, value)

        context = (
            contextlib.nullcontext
            if not self.ops_handler
            # pyrefly: ignore [not-callable]
            else lambda: V.set_ops_handler(self.ops_handler(V.get_ops_handler()))
        )
        with context():  # type: ignore[operator]
            yield
        self.subgraph_bodies[body_name] = SubgraphInfo(
            **{
                key.name: getattr(self, key.name)
                for key in dataclasses.fields(SubgraphInfo)
            }
        )
        for key, value in old_state.items():
            setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, clear_cse: bool = False):
        assert body_name not in self.subgraph_bodies
        self.subgraph_bodies[body_name] = SubgraphInfo(
            IndentedBuffer(), None, None, cse=self.cse.clone() if clear_cse else None
        )
        with self.set_subgraph_body(body_name):
            yield

    def need_numel_args(self):
        return False

    def estimate_kernel_num_bytes(self):
        """
        Estimate the total number of bytes this kernel takes.
        For in/out nodes, sizes are counted twice: once for reading and
        once for writing.
        """
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        num_bytes = []
        for i, inp in enumerate(itertools.chain(self.input_nodes, (self.output_node,))):
            size = V.graph.sizevars.size_hints(inp.get_size(), fallback=0)
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def estimate_flops(self) -> int:
        for node in self.input_nodes:
            for fx_node in node._current_origins:
                f = count_flops_fx(fx_node)
                if f is not None:
                    return V.graph.sizevars.size_hint(f, fallback=0)
        return 0

    def jit_lines(self):
        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature, _ = self.args.python_argdefs()
        triton_meta: dict[str, Any] = {
            "signature": signature_to_meta(
                signature,
                size_dtype=self.index_dtype,
                argdefs=argdefs,
                is_template=True,
            ),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]
        matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", None)
        waves_per_eu = self.meta.get("waves_per_eu", None)
        kpack = self.meta.get("kpack", None)
        if matrix_instr_nonkdim:
            triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim
        if waves_per_eu:
            triton_meta["waves_per_eu"] = waves_per_eu
        if kpack:
            triton_meta["kpack"] = kpack

        self.triton_meta = triton_meta

        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **self.inductor_meta_common(),
            **FixedGrid.setup_grid_as_args(),
        }
        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb
        if config.benchmark_kernel:
            flops = self.estimate_flops()
            inductor_meta["kernel_flop"] = flops

        inductor_meta["config_args"] = self.meta

        template_args = f"""
            num_stages={self.num_stages},
            num_warps={self.num_warps},
            triton_meta={triton_meta!r},
            inductor_meta={inductor_meta!r},
        """

        if HAS_WARP_SPEC:
            template_args += f"""
            num_consumer_groups={self.num_consumer_groups},
            num_buffers_warp_spec={self.num_buffers_warp_spec},
        """

        return f"""
            @triton_heuristics.template(
                {template_args}
            )
            @triton.jit
        """

    def gen_argdefs(self):
        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            return f"{', '.join(x.full_name() for x in arg_defs)}"

        return self._register_hook("<ARGDEFS>", hook, allow_overwriting=True)

    def gen_defines(self):
        return self.defines

    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
        assert all(isinstance(x, str) for x in argnames)
        renames = IndentedBuffer(initial_indent=1)

        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        assert len(argnames) == len(named_args), (
            len(argnames),
            len(named_args),
            self.prefix_args,
            len(self.input_nodes),
        )

        for input_node in self.input_nodes[: self.prefix_args]:
            # get args in correct order
            self.args.input(input_node.get_name())

        for name, input_node in zip(argnames, named_args):
            arg_name = f"arg_{name}"
            self.named_input_nodes[name] = input_node
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input_buffers[input_node.get_name()] = arg_name

        # The args may be duplicated, so renaming must be after args are de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            if self.prologue_loads_all_inputs:
                self.prologue_supported_inputs.add(input_node.get_name())
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            # get args in correct order
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input(input_node.get_name())

        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(gen_common_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(
                f"def {self.kernel_name}({', '.join(x.full_name() for x in arg_defs)}):"
            )
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
                self.codegen_prologue(code)
            return code.getvalue()

        return self._register_hook("<DEF_KERNEL>", hook)

    def size(self, name: Optional[str], index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        assert isinstance(index, int)
        if name is None:
            val = self.output_node.get_size()[index]
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_size()[index]
        return texpr(self.rename_indexing(val))

    def stride(self, name, index=None):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        if name is None:
            val = self.output_node.get_stride()
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_stride()

        if isinstance(index, int):
            return texpr(self.rename_indexing(val[index]))
        return ", ".join([texpr(self.rename_indexing(i)) for i in val])

    def _get_subgraph(self, subgraph_number: int):
        assert isinstance(subgraph_number, int)
        assert isinstance(self.subgraphs, list)
        assert subgraph_number < len(self.subgraphs), (
            f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
        )
        assert self.body.getvalue() == "", (
            "Body should be clear before adding a modification"
        )
        return self.subgraphs[subgraph_number]

    def _handle_scatter_graph(self, scatter_graph):
        """Handle processing for a single scatter graph.

        Args:
            scatter_graph: The scatter graph to process
        """
        assert isinstance(scatter_graph, ir.ComputedBuffer), (
            f"scatter_graph must be an instance of ComputeBuffer but got {type(scatter_graph)}"
        )

        def contiguous_strides(x):
            # We always create a fresh contiguous grad for scattering into
            return sum(
                x_i * stride for x_i, stride in zip(x, scatter_graph.get_stride())
            )

        return scatter_graph.data.store_output(  # type: ignore[attr-defined]
            scatter_graph.name, contiguous_strides, []
        )

    def modification(
        self,
        subgraph_number: int,
        output_name: Optional[str],
        mask: Optional[str] = None,
        **fixed_inputs,
    ) -> str:
        """This creates a modification function for a subgraph.
        To use this inside a template, the first argument should specify which subgraph to codegen for

        Args:
            subgraph_number (int): The index of the subgraph in self.subgraphs
            output_name (Optional[str]): The name of the output variable to store the result in
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
        """
        num = 0
        out = None
        scatters = []
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapper(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_ops_handler(modification_handler):
                assert isinstance(subgraph, (ir.ComputedBuffer, list)), (
                    f"Expected the subgraph to be a ComputedBuffer or a List[ComputedBuffer], got {type(subgraph)}"
                )
                # Handle scatter stores
                if isinstance(subgraph, list):
                    for scatter_graph in subgraph:
                        scatters.append(self._handle_scatter_graph(scatter_graph))
                elif isinstance(subgraph.data, ir.InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
                    out = subgraph.data.inner_fn(())

            self.codegen_body()
            if output_name is not None:
                assert isinstance(output_name, str)
                assert out is not None
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                assert out is None
                for scatter in scatters:
                    self.body.writeline(str(scatter))

            body_val = self.body.getvalue()
            self.cse.invalidate(OrderedSet())
            return body_val

    def load_input(
        self,
        input_name: str,
        output_name: str,
        indices: Union[list[Any], tuple[Any]],
        mask: Optional[str] = None,
        other: Optional[Union[float, int]] = 0.0,
        indent_width: int = 4,
        index_shape: Optional[tuple[str]] = None,
    ):
        """Loads an input and applies any necessary preprocessing or masking.

        Args:
            input_name (str): The name of the input to load.
            indices (Union[List, Tuple]): The index for each dimension of the input.
            val (str): The name of the variable to store the loaded value.
            mask (Optional[str]): An optional mask to use for the load operation.
            other (Optional[Union[float, int]]): The value to use for masked elements. Default is 0.0.
            indent_width (int): The number of spaces to use for indentation.
        """

        input_node = self.named_input_nodes[input_name]
        if not self.prologue_loads_all_inputs:
            self.prologue_supported_inputs.add(input_node.get_name())

        tilings = (sympy_product(input_node.get_size()), sympy.Integer(1))
        groups = {
            "x": tilings[0],
            "r0_": tilings[1],
        }

        range_trees = self.construct_range_trees(
            pid_cache=None,
            inside_reduction=False,
            is_reduction=False,
            numels=groups,
            no_x_dim=False,
        )
        load_code = None

        with self.create_subgraph_body(f"<LOAD_INPUT_{input_name}>"):
            assert isinstance(indices, (list, tuple))
            assert isinstance(output_name, str)
            assert isinstance(mask, (str, type(None)))
            self.range_trees = range_trees
            self.numels = {k: V.graph.sizevars.simplify(v) for k, v in groups.items()}
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]

            lengths = [V.graph.sizevars.simplify(s) for s in input_node.get_size()]
            assert len(indices) == len(lengths)

            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            assert len(indices) == len(lengths)

            # glue to make generated code use same indexing from template

            # TODO (from reviewers as well)
            # in codegen_template,
            # prologue_node.codegen(kernel.split_and_set_ranges(prologue_node.get_ranges()))
            # the ranges need to reflect the group of the prologue input or it will error
            # not sure if there is any difference between original range_tree_entry in
            # and new one from correct lengths/groups... both actually seem to work
            for name, range_tree_entry in zip(
                indices, self.range_trees[0].construct_entries(lengths)
            ):
                range_tree_entry.set_name(name)
            contiguous_index = sympy_dot(
                ir.FlexibleLayout.contiguous_strides(lengths), index_symbols
            )
            contiguous_index = self.rename_indexing(contiguous_index)
            self.body.writeline("xindex = " + texpr(contiguous_index))

            xindex_range_root = self.range_trees[0].lookup(
                sympy.Integer(1), sympy_product(lengths)
            )
            xindex_range_root.set_name("xindex")

            # Note - ["None" override_mask]
            # MM Templates work by taking out of bounds index values and wrapping them around to 0
            # so that no mask is required on the load: offs_a_m = `rm % M`
            # We should to override the mask to be "None" instead of inheriting the mask that would
            # have been loaded otherwise.
            # We are using "None" for clarity in output code, but
            # we could alternatively emit `xmask = tl.full([xindex.shape], True, tl.int1)`
            self.template_mask = mask if mask is not None else "None"
            self.template_out_shape = index_shape if index_shape else "xindex"
            self.template_indices = indices
            self.named_input_nodes[input_name].data.freeze_layout()
            self.cse.invalidate(OrderedSet())

            template_mask = self.template_mask

            class StoreOutputSubstitution(V.WrapperHandler):  # type: ignore[name-defined]
                name = "StoreOutputSubstitution"

                def store(
                    self,
                    name: str,
                    index: sympy.Expr,
                    value: "CSEVariable",
                    mode: "StoreMode" = None,
                ):
                    V.kernel.store_buffer_names.add(name)
                    V.kernel.cse.store_cache[name] = value
                    if name in V.kernel.prologue_fused_inputs:
                        # We load masked out values with 0, then apply a prologue.
                        # The masked out values may not necessariliy be 0 any more
                        # so we need to reapply the mask.
                        value_dtype = value.dtype
                        value_str = str(value)
                        if template_mask != "None" and (
                            name not in V.kernel.prologue_fused_inputs_preserve_zero
                            or other != 0
                        ):
                            value_str = (
                                f"tl.where({template_mask}, {value_str}, {other})"
                            )

                        if value_dtype != V.graph.get_buffer(name).dtype:
                            value_str = f"{value_str}.to({triton_type(V.graph.get_buffer(name).dtype)})"

                        # TODO: we should have intermediary var shapes
                        V.kernel.compute.writeline(
                            f"{output_name} = {value_str}.broadcast_to(xindex.shape)"
                        )

            # pyrefly: ignore [bad-assignment]
            self.ops_handler = StoreOutputSubstitution

            input_node = self.named_input_nodes[input_name]
            output_index = input_node.make_indexer()(index_symbols)

            # in def_kernel above we define the inputs with the storage offset adjusted
            # creating the load in input_node.make_indexer() will also adjust by storage offset
            # so subtract here to not double increment
            if not V.graph.sizevars.statically_known_equals(
                input_node.layout.offset, 0
            ):
                output_index = output_index - self.rename_indexing(
                    input_node.get_layout().offset
                )

            output_index = self.rename_indexing(output_index)

            if output_index == contiguous_index:
                output_index_str = "xindex"
            else:
                out_indexing = self.indexing(
                    output_index,
                    copy_shape=self.template_out_shape,
                    override_mask=self.template_mask,
                )
                from .codegen.triton import IndexingOptions

                assert isinstance(out_indexing, IndexingOptions)
                output_index_str = (
                    f"({out_indexing.index_str}).broadcast_to(xindex.shape)"
                )

            # Generate load code
            load_code = f"{output_name} = tl.load({input_name} + ({output_index_str})"

            if mask:
                load_code += f", mask={mask}, other={other})"
            else:
                load_code += ")"

        hook_key = f"<LOAD_INPUT_{input_name}>"

        def hook():
            with self.set_subgraph_body(hook_key):
                self.cse.invalidate(OrderedSet())
                self.codegen_body()
                self.cse.invalidate(OrderedSet())
                if input_node.get_name() not in self.prologue_fused_inputs:
                    assert load_code is not None
                    self.body.writeline(load_code)

                return textwrap.indent(self.body.getvalue(), " " * indent_width).strip()

        return self._register_hook(hook_key, hook)

    def _generate_index_from_tma_index(
        self,
        output_name: str,
        offset_name: str,
        tma_index: sympy.Symbol,
        block_size: str,
        dim: int,
        num_dims: int,
        block_name: Optional[str] = None,
    ) -> list[str]:
        """
        Generate the logic to compute the regular tl.load index from the provided
        tma index. This is used to ensure variables can support fusions.

        Args:
            output_name (str): The output variable name.
            offset_name (str): The name used for the intermediate offset.
            tma_index (sympy.Symbol): The symbol used for the original TMA index.
            block_size (str): The block size of the index.
            dim (int): Which dimension to project the index in.
            num_dims (int): The total number of dimensions in the output.
            block_name (Optional[str]): The name of the block variable. If not passed
                in then we aren't reusing standard symbol names.

        Returns:
            list[str]: The lines used to generate the index.

        """
        if block_name:
            # Generate the expected names for the structure:
            # XBLOCK/YBLOCK and xoffset/yoffset. We append XBLOCK/YBLOCK
            # to the top of the kernel so we can safely extract the tensor
            # descriptor construction to the top of the kernel.
            if block_name in self.prologue_cache:
                assert self.prologue_cache[block_name] == block_size, (
                    f"Constant {block_name} must be used for all stores"
                )
            else:
                self.prologue_cache[block_name] = block_size
                self.prologue.writeline(f"{block_name}: tl.constexpr = {block_size}")
        else:
            block_name = block_size
        line0 = f"{offset_name} = {texpr(tma_index)}"
        expr = f"({offset_name} + tl.arange(0, {block_name}))"
        prefix_none = "".join(["None, "] * dim)
        suffix_none = ", ".join(["None"] * (num_dims - (dim + 1)))
        line1 = f"{output_name} = {expr}[{prefix_none}:, {suffix_none}]"
        return [line0, line1]

    def _generated_mask_for_tma(
        self,
        index_name: str,
        shape_val: str,
        output_name: str,
    ) -> str:
        """
        Generate the mask logic to feed to fusions for mask. The expectation
        is that if we have X/Y there will be a variable named xmask and ymask.

        Args:
            index_name (str): The index used in the mask. Should be one of
                xindex or yindex.
            shape_val (str): The expression for the upper bound shape.
            output_name (str): The expression used for the output.

        Returns:
            str: The mask generation line.
        """
        return f"{output_name} = {index_name} < {shape_val}"

    def store_output(
        self,
        indices: Union[list[Any], tuple[Any]],
        val: str,
        mask: Optional[str] = None,
        indent_width: int = 4,
        val_shape: Optional[tuple[str]] = None,
        block_indexing: bool = False,
    ):
        """Stores the final output and appends any epilogue fusions if the buffer hasn't been optimized away.

        Args:
            indices (Union[List, Tuple]): The index for each dimension of the output. The dot product of
                these indices and output strides must match `val`.
            val (str): The value to store.
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
            indent_width (int): The number of spaces to use for indentation. This is used when the call to
                store_output is indented in the kernel definition.
            block_indexing (bool): Are the input indices presented as offsets for creating the block (e.g.
                inputs to TMA) or are they tensors that should be passed in directly.
        """
        subgraph_name = self._get_store_output_subgraph_name(
            next(self.store_output_ctr)
        )
        with self.create_subgraph_body(subgraph_name, clear_cse=True):
            assert isinstance(indices, (list, tuple))
            assert isinstance(val, str)
            assert isinstance(mask, (str, type(None)))
            assert isinstance(val_shape, (tuple, type(None)))
            assert isinstance(block_indexing, bool)
            assert self.template_mask is None
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            lengths = [
                V.graph.sizevars.simplify(s) for s in self.output_node.get_size()
            ]
            assert len(indices) == len(lengths)

            output_layout = self.output_node.get_layout()
            self.template_out = val
            if block_indexing:
                assert val_shape, "Blocking indexing requires passing in val_shape"
                assert len(val_shape) == 2, (
                    "Blocking indexing only supports 2D data at this time"
                )
                assert not mask, "Mask is not supported with blocking indexing"
                intermediate_lines: list[str] = []
                epilogue_index_symbols: list[sympy.Symbol] = []
                if self.tma_store:
                    # Generate the expected indexing symbols.
                    # Note: TMA indices are expected to be in the
                    # format (x, y), but the range_tree is always
                    # (yindex, xindex).
                    index_order = [1, 0]
                    val_shape_copy = list(val_shape)
                    for i, range_tree in zip(index_order, self.range_trees[:-1]):
                        name = range_tree.name
                        symbol = range_tree.symbol()
                        epilogue_index_symbols.append(symbol)
                        lookup_output = range_tree.lookup(sympy.S.One, lengths[i])
                        old_name = lookup_output.symbol()
                        lookup_output.set_name(name)
                        # Update var_list and var_range
                        range_tree.var_list[range_tree.var_list.index(old_name)] = (
                            symbol
                        )
                        range_val = range_tree.var_ranges[old_name]
                        del range_tree.var_ranges[old_name]
                        range_tree.var_ranges[symbol] = range_val
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                name,
                                "xoffset" if name == "xindex" else "yoffset",
                                index_symbols[i],
                                val_shape[i],
                                i,
                                len(index_order),
                                # pyrefly: ignore [missing-argument]
                                block_name=range_tree.symt.name,
                            )
                        )
                        # Generate the xmask and ymask
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                name,
                                self.size(None, i),
                                "xmask" if name == "xindex" else "ymask",
                            )
                        )
                        # Update the val_shape information to use consistent naming
                        # after the remapping.
                        # pyrefly: ignore [missing-argument]
                        val_shape_copy[i] = range_tree.symt.name
                    # Reverse the index symbols because TMA is indexed
                    # as (x, y) whereas the variables will naturally be indexed
                    # as (y, x)
                    epilogue_index_symbols.reverse()
                    val_shape = tuple(val_shape_copy)
                else:
                    mask_vars: list[str] = []
                    for i, (index, shape) in enumerate(zip(index_symbols, val_shape)):
                        index_name = self._gen_tmp_var()
                        offset_name = self._gen_tmp_var()
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                index_name,
                                offset_name,
                                index,
                                shape,
                                i,
                                len(index_symbols),
                            )
                        )
                        epilogue_index_symbols.append(
                            sympy.Symbol(index_name, integer=True)
                        )
                        mask_name = self._gen_tmp_var()
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                index_name,
                                self.size(None, i),
                                mask_name,
                            )
                        )
                        mask_vars.append(mask_name)
                    final_mask_var = self._gen_tmp_var()
                    final_mask_rhs = " & ".join(
                        f"{mask_name}" for mask_name in mask_vars
                    )
                    intermediate_lines.append(f"{final_mask_var} = {final_mask_rhs}")
                    self.template_mask = final_mask_var
                index_symbols = epilogue_index_symbols
                contiguous_index = sympy_dot(output_layout.stride, index_symbols)
                if not self.tma_store:
                    # Convert to just use xindex.
                    contiguous_index = self.rename_indexing(contigu
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `select_algorithm.py_docs.md_docs.md`
- **Keyword Index**: `select_algorithm.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
