# Documentation: `torch/_dynamo/output_graph.py`

## File Metadata

- **Path**: `torch/_dynamo/output_graph.py`
- **Size**: 166,763 bytes (162.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Core graph building functionality for PyTorch's Dynamo system. This module contains
the essential components for constructing and managing FX graphs during compilation:

- OutputGraph: Manages the overall graph construction and compilation process. It owns
  a SubgraphTracer and handles graph compilation, execution, and state management.
  OutputGraph also manages features like graph deduplication, symbolic shape handling,
  and tracking of side effects.

- SubgraphTracer: Handles the actual FX graph construction by tracing Python code.
  It supports advanced features like higher-order operators through nested tracers,
  lifting of free variables, and handling of symbolic shapes.

The module supports key Dynamo features including:
- Higher-order operators through nested SubgraphTracers
- Graph deduplication for optimization
- Symbolic shape handling and propagation
- Side effect tracking and management
- Guard insertion and management
"""

import collections
import contextlib
import copy
import functools
import inspect
import itertools
import logging
import operator
import re
import sys
import traceback
import warnings
import weakref
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field as dc_field
from types import CodeType
from typing import Any, cast, Optional, TYPE_CHECKING, Union
from typing_extensions import ParamSpec, TypeVar

import sympy

import torch._guards
import torch._logging
import torch.distributed as dist
import torch.nn
import torch.utils._pytree as pytree
from torch import fx, Tensor
from torch._C._dynamo import guards
from torch._dynamo.exc import ShortenTraceback, TensorifyScalarRestartAnalysis
from torch._guards import (
    CompileContext,
    CompileId,
    GlobalContextCheckpointState,
    Source,
    tracing,
    TracingContext,
)
from torch._library.opaque_object import is_opaque_type
from torch._subclasses.fake_tensor import FakeTensor
from torch._utils_internal import signpost_event
from torch.export.dynamic_shapes import _ConstraintTarget
from torch.fx._lazy_graph_module import _make_graph_module  # type: ignore[attr-defined]
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    guard_scalar,
    is_symbolic,
    ShapeEnv,
    Specialization,
    uninteresting_files,
)
from torch.fx.node import Target
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from . import config, exc, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
    create_binary_slice,
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
    create_dup_top,
    create_instruction,
    create_load_const,
    create_rot_n,
    create_swap,
    Instruction,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .device_interface import get_interface_for_device
from .exc import (
    BackendCompilerFailed,
    exceptions_allowed_to_be_fallback,
    SkipFrame,
    unimplemented,
    unimplemented_with_warning,
)
from .graph_bytecode_inputs import has_user_objects, index_to_bytecode_constructor
from .graph_deduplication import apply_graph_deduplication
from .graph_region_tracker import GraphRegionTracker
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import AttributeMutationExisting, SideEffects, ValueMutationExisting
from .source import (
    _get_source_debug_name,
    AttrSource,
    BackwardStateSource,
    ConstantSource,
    GetItemSource,
    GlobalStateSource,
    is_constant_source,
    is_from_local_source,
    LocalSource,
    NumpyTensorSource,
    ParamBufferSource,
    ShapeEnvSource,
    SyntheticLocalSource,
    TensorProperty,
    TensorPropertySource,
)
from .utils import (
    _extract_tensor_dict,
    checkpoint_params,
    CleanupHook,
    clone_inputs,
    count_calls,
    counters,
    dynamo_timed,
    get_instruction_source_311,
    get_locals_to_steal,
    get_static_address_type,
    get_unique_name_wrt,
    graph_break_reasons,
    increment_op_count,
    istype,
    lazy_format_graph_code,
    LazyString,
    nn_module_proxy,
    same,
    set_example_value,
)
from .variables.base import VariableTracker
from .variables.builder import (
    BackwardStateGraphArg,
    GraphArg,
    TrackedFake,
    wrap_fx_proxy,
)
from .variables.ctx_manager import ContextWrappingVariable
from .variables.lists import BaseListVariable
from .variables.misc import NullVariable
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .variables.torch_function import TensorWithTFOverrideVariable
from .variables.user_defined import UserDefinedDictVariable


if TYPE_CHECKING:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)
graph_tabular_log = torch._logging.getArtifactLogger(__name__, "graph")
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")
graph_sizes_log = torch._logging.getArtifactLogger(__name__, "graph_sizes")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")

RootGuardManager = guards.RootGuardManager


# Capture fn pointer at import time
# This is to guard against trying to mark the iterated tensors
# as static in case user overrides fn ptr
og_module_named_buffers_fn_ptr = torch.nn.Module.named_buffers
og_module_named_parameters_fn_ptr = torch.nn.Module.named_parameters


@dataclass(frozen=True)
class VariableTrackerCacheKey:
    vt_id: int
    # Two different source can point to the same object. However, Dynamo handles
    # globals and local source differently when it comes to guards and possibly
    # some other parts as well. So, cache also relies on the source.
    source: Source


@dataclass(frozen=True)
class AliasingInfo:
    has_aliasing: bool
    msg: str


@dataclass(frozen=True)
class MutationInfo:
    has_mutation: bool
    msg: str


class VariableTrackerCache:
    def __init__(self) -> None:
        self.cache: dict[VariableTrackerCacheKey, VariableTracker] = {}

    def lookup(self, value: Any, source: Source) -> Optional[VariableTracker]:
        key = VariableTrackerCacheKey(id(value), source)
        if key not in self.cache:
            return None
        return self.cache[key]

    def add(self, value: Any, source: Source, vt: VariableTracker) -> None:
        key = VariableTrackerCacheKey(id(value), source)
        self.cache[key] = vt

    def clone(self) -> "VariableTrackerCache":
        # Needed for copy and restore graph state
        new_cache = VariableTrackerCache()
        new_cache.cache.update(self.cache)
        return new_cache

    def clear(self) -> None:
        self.cache.clear()


@functools.cache
def _step_logger() -> Any:
    return torchdynamo_logging.get_step_logger(log)


@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""

    reason: str
    user_stack: list[traceback.FrameSummary]

    # Indicates if this was a graph break reason due to graph break.
    graph_break: bool = True

    def __post_init__(self) -> None:
        if self.graph_break:
            graph_break_reasons.append(self)


def _get_gen_rand_values_fn(random_calls: Any) -> Callable[[], list[Any]]:
    def _gen_rand_values() -> list[Any]:
        return [fn(*args, **kwargs) for fn, args, kwargs in random_calls]

    return _gen_rand_values


class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""

    def __init__(self, nn_modules: dict[str, torch.nn.Module]):
        super().__init__()
        for k, v in nn_modules.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return "FakeRootModule(...)"

    def add_nn_modules(self, nn_modules: dict[str, torch.nn.Module]) -> None:
        for k, v in nn_modules.items():
            setattr(self, k, v)


class WrapperBackend:
    def __init__(self, backend: CompilerFn) -> None:
        self.backend: CompilerFn = backend

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        self.restore = checkpoint_params(gm)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, example_inputs)

        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.candidate

        # if verify_correctness=True
        try:
            correct = self.gm.forward(*clone_inputs(example_inputs))
            result = self.candidate(*clone_inputs(example_inputs))

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.candidate

            raise RuntimeError(f"incorrect results of backend {self}")

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


Scope = dict[str, object]


@dataclass
class OutputGraphGuardsState:
    """
    A base class containing fields that are considered "persistent" when we
    want to save all the important state for reconstrucing guards in a different
    process. Normally we don't need to add states here, but we may have to when
    the information is needed to serialize the guards, so the fields here are
    supposed to be serializable as a requirement.
    """

    local_scope: Scope
    global_scope: Scope
    # This records the initial torch function mode stack for guarding
    torch_function_mode_stack: list[torch.overrides.TorchFunctionMode]
    guard_on_key_order: set[Source]
    # Map from graph input's `Source` to sizes / strides metadata
    input_source_to_sizes_strides: dict[Source, dict[str, Any]]
    dual_level: int
    functorch_layers: list[torch._functorch.pyfunctorch.FuncTorchInterpreter]
    current_device: Optional[torch.device]
    global_state_guard: torch._C._dynamo.guards.GlobalStateGuard
    _guards: torch._guards.GuardsSet
    _aotautograd_guards: list[torch._guards.GuardEnvExpr]

    # Whether or not the guards should be checked for correctness

    export: bool = False
    skip_guards_check: bool = False
    export_constraints: bool = False
    name_of_builtins_dict_key_in_fglobals: Optional[str] = None

    @property
    def shape_env(self) -> ShapeEnv:
        raise AssertionError(f"shape_env shouldn't be accessed from {type(self)}")

    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self._guards

    @property
    def aotautograd_guards(self) -> list[torch._guards.GuardEnvExpr]:
        return self._aotautograd_guards

    def dump_guards_state(self) -> "OutputGraphGuardsState":
        # Dump a serializable version of self without extras
        return OutputGraphGuardsState(
            local_scope=self.local_scope,
            global_scope=self.global_scope,
            torch_function_mode_stack=self.torch_function_mode_stack,
            guard_on_key_order=self.guard_on_key_order,
            input_source_to_sizes_strides=self.input_source_to_sizes_strides,
            dual_level=self.dual_level,
            functorch_layers=self.functorch_layers,
            current_device=self.current_device,
            global_state_guard=self.global_state_guard,
            name_of_builtins_dict_key_in_fglobals=self.name_of_builtins_dict_key_in_fglobals,
            export=self.export,
            export_constraints=self.export_constraints,
            _guards=self.guards,
            _aotautograd_guards=self.aotautograd_guards,
            skip_guards_check=self.skip_guards_check,
        )


@dataclass
class StackLocalsMetadata:
    """
    Stores metadata for a frame's stack and locals for the purposes of building resume functions
    """

    num_stack: int = 0  # number of stack elements, minus removed NULLs
    locals_names: dict[str, int] = dc_field(
        default_factory=dict
    )  # order of locals codegen'd to the stack
    stack_null_idxes: list[int] = dc_field(default_factory=list)
    locals_null_keys: list[str] = dc_field(default_factory=list)
    stack_ctx_args: list[tuple[int, tuple[Any, ...]]] = dc_field(default_factory=list)
    stack_ctx_idxes_orig: list[int] = dc_field(default_factory=list)
    locals_ctx_args: list[tuple[str, tuple[Any, ...]]] = dc_field(default_factory=list)


# TODO we should expand this to make it work for atribtrary in/out
@dataclass
class ExportMetaData:
    # maps graph input index to its' source which is later
    # used in export to map to correct user input. In its' flat form,
    # just looks like GetItem(base=LocalSource("foo", idx=0))
    graph_input_idx_to_local_source: dict[int, Source] = dc_field(default_factory=dict)
    # maps user output idx to what type of output it is. There are 3 options:
    # 1) graph out
    # 2) user input
    # 3) constants
    output_return_type: dict[int, tuple[str, Any]] = dc_field(default_factory=dict)
    # output spec of the traced function
    out_spec: Union[torch.utils._pytree.TreeSpec, torch.utils._pytree.LeafSpec] = (
        torch.utils._pytree._LEAF_SPEC
    )
    module_call_spec: dict[
        str,
        dict[str, Union[torch.utils._pytree.TreeSpec, torch.utils._pytree.LeafSpec]],
    ] = dc_field(default_factory=dict)


def get_builtins_dict(global_scope: Scope) -> dict[str, Any]:
    # f_globals["__builtins__"] can be a dict or a module. This is an
    # implementation detail -
    # https://docs.python.org/3/library/builtins.html.

    # This makes guarding on any builtin messy because the guard check_fn
    # has to check if the __builtins__ is a module or dict, and then access
    # by either using getattr or getitem respectively.

    # To solve this problem, we insert a new entry in f_globals which points
    # to the builtins __dict__ and then we guard any builtin on this dict.
    # To avoid any collision with the pre-existing keys, we use the
    # install_global to give us a unique dict key.

    f_builtins = global_scope["__builtins__"]
    if not isinstance(f_builtins, dict):
        f_builtins = f_builtins.__dict__
    return f_builtins


class OutputGraphCommon(OutputGraphGuardsState):
    """
    A minimal interface for full graph capture. It is intended to be
    the target of any tracer that feeds into backends.

    Currently dynamo's OutputGraph is the only known implementation
    of this interface, used by (aot) precompile and (strict) export.
    Importantly, that implementation also contains many other fields
    that are using during tracing but not included in this interface
    because they are not used once tracing is complete.

    It should be safe to assume that (caching) precompile also uses
    this interface.

    In the future, we want make_fx, used by (non-strict) export, to
    also implement this interface.

    The serializable part of this interface is OutputGraphGuardsState.
    We do not need to serialize other parts; however it will pay to
    be disciplined about what those other parts are, especially since
    we want other tracers to be able to meaningfully implement them,
    and we should generally try to cut them down when possible.
    """

    def __init__(
        self,
        output_graph_guards_state: OutputGraphGuardsState,
        import_sources: Optional[dict[str, str]] = None,
        shape_env: Optional[ShapeEnv] = None,
        export_metadata: Optional[ExportMetaData] = None,
        tracked_fakes_id_to_source: Optional[dict[int, list[Source]]] = None,
    ):
        super().__init__(
            output_graph_guards_state.local_scope,
            output_graph_guards_state.global_scope,
            output_graph_guards_state.torch_function_mode_stack,
            output_graph_guards_state.guard_on_key_order,
            output_graph_guards_state.input_source_to_sizes_strides,
            output_graph_guards_state.dual_level,
            output_graph_guards_state.functorch_layers,
            output_graph_guards_state.current_device,
            output_graph_guards_state.global_state_guard,
            output_graph_guards_state._guards,
            output_graph_guards_state._aotautograd_guards,
            output_graph_guards_state.export,
            output_graph_guards_state.skip_guards_check,
            output_graph_guards_state.export_constraints,
            output_graph_guards_state.name_of_builtins_dict_key_in_fglobals,
        )

        self.import_sources = import_sources or {}
        # The following fields are currently known to be used by clients.
        # In particular, we need:
        # - shape_env, for building guards
        # - export_metadata, for un/flattening inputs and outputs
        # - tracked_fakes_id_to_source, for processing tensor dim constraints
        self._shape_env = shape_env or ShapeEnv()  # private for inheritance
        self.export_metadata = export_metadata or ExportMetaData()
        self.tracked_fakes_id_to_source: dict[int, list[Source]] = (
            tracked_fakes_id_to_source or {}
        )

    @property
    def shape_env(self) -> ShapeEnv:
        return self._shape_env

    def bypass_package(self, reason: str = "", **kwargs: Any) -> None:
        # NOTE: currently there are no tests for this but it is reachable
        # when building guards, so technically necessary to include here.
        # It is unclear whether we should include packaging altogether.
        raise NotImplementedError


class OutputGraph(OutputGraphCommon):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    side_effects: SideEffects

    def __init__(
        self,
        code_options: dict[str, Any],
        compiler_fn: Optional[CompilerFn],
        root_tx: "InstructionTranslatorBase",
        export: bool,
        export_constraints: Sequence[_ConstraintTarget],
        frame_state: Any,
        local_scope: Scope,
        global_scope: Scope,
        f_code: CodeType,
        torch_function_mode_stack: list[torch.overrides.TorchFunctionMode],
        package: Optional["CompilePackage"],
        one_graph: bool = False,
    ) -> None:
        OutputGraphGuardsState.__init__(
            self,
            local_scope,
            global_scope,
            torch_function_mode_stack,
            guard_on_key_order=set(),
            input_source_to_sizes_strides={},
            dual_level=torch.autograd.forward_ad._current_level,
            functorch_layers=torch._functorch.pyfunctorch.retrieve_all_functorch_interpreters(),
            current_device=torch.utils._device.CURRENT_DEVICE,
            # initial_global_state is only None during NopTest.
            global_state_guard=torch._dynamo.convert_frame.initial_global_state
            or torch._C._dynamo.guards.GlobalStateGuard(),
            # These are set by @property instead, just initialize them as blank
            _guards=torch._guards.GuardsSet(),
            _aotautograd_guards=[],
        )
        self.tracers = [SubgraphTracer(self, is_export=export)]
        # Map from graph input's `Source` to its `VariableTracker` to
        # de-duplicate graph inputs by source and reuse the tracker
        self.input_source_to_var: dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints  # type: ignore[assignment]
        self.frame_state = frame_state
        self.cleanup_hooks: list[Callable[[], Any]] = []
        # compile_id is an id number for the current torch.compile
        self.compile_id: int = next(_compile_id_counter)
        # Set of globals installed via install_global* APIs
        self.installed_globals: set[str] = set()

        # TODO: maybe should just pass the entire f_code in here?  Not
        # sure...
        self.co_fields = {
            "co_name": f_code.co_name,
            "co_filename": f_code.co_filename,
            "co_firstlineno": f_code.co_firstlineno,
        }

        self.region_tracker = GraphRegionTracker()

        # tracked_fakes says where any tensor that was wrapped to fake came
        # from.  It is similar to GraphArg, in that all GraphArgs will get
        # will get added to TrackedFakes, but TrackedFakes also contains
        # GraphArgs that got pruned, and things like Tensor attributes which
        # aren't explicit graph inputs.  Used by shape guard
        self.tracked_fakes: list[TrackedFake] = []

        shape_env = ShapeEnv(
            # Reference Cycle!
            # Share a reference to the list of TrackedFake.
            #
            # ShapeEnv needs this in order to be able to reproduce the call
            # to produce_guards at an arbitrary time point. That is because
            # TrackedFake instances may have its metadata changed throughout
            # the program execution.
            tracked_fakes=self.tracked_fakes,
            # We want to allow capture scalar outputs and allow_dynamic_output_shape_ops when fullgraph=True
            allow_scalar_outputs=one_graph or config.capture_scalar_outputs,
            allow_dynamic_output_shape_ops=one_graph
            or config.capture_dynamic_output_shape_ops,
            prefer_deferred_runtime_asserts_over_guards=config.prefer_deferred_runtime_asserts_over_guards,
            co_fields=self.co_fields,
        )

        # In export mode, we force the shape_env to strictly disallow any constraining
        # of the user marked dynamic dims
        import torch._functorch.config as _config

        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            fake_mode = torch._subclasses.FakeTensorMode(
                shape_env=shape_env,
                # TODO (tmanlaibaatar) Remove this once we always lift params and buffers
                allow_non_fake_inputs=bool(self.export),
                export=self.export,
            )
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        self.tracing_context.traced_code.append(f_code)
        self.traced_code = self.tracing_context.traced_code
        self.dynamo_compile_id: Optional[CompileId] = (
            CompileContext.current_compile_id()
        )
        self.init_ambient_guards()

        # Map each tensor id to a list of sources. This is necessary because
        # tensor ids cannot be recovered from tracked fakes (in general).
        # We use this map to interpret (i.e., check for violations of) constraints,
        # specifically equality constraints, which have shared tensor ids in them.
        # This map should also be generally useful, e.g., for (de)serialization.
        self.tracked_fakes_id_to_source: dict[int, list[Source]] = (
            collections.defaultdict(list)
        )
        # Stores the full fqn of a param or buffer to the relevant source.
        self.param_name_to_source: Optional[dict[str, Source]] = {}
        self.side_effects = SideEffects(self)
        # Cached variable trackers. This makes symbolic analysis of LOAD_GLOBAL
        # and LOAD_ATTR for same python objects free.
        self.variable_tracker_cache = VariableTrackerCache()
        self.unique_var_id = itertools.count()
        self.code_options: dict[str, Any] = dict(code_options)
        self.output_instructions: list[Instruction] = []
        # used to track nodes that are added between calls of copy_graphstate
        # and restore_graphstate
        self.timestamp = 0

        # A list of register_finalizer_fns to apply to the output graph module
        self.register_finalizer_fns: list[Callable[[fx.GraphModule], None]] = []

        # Not checkpointed
        self.compiler_fn: Optional[CompilerFn] = compiler_fn
        self.root_tx = root_tx

        self.package = package
        # Given a source, what are the user stacks of all locations that
        # accessed it?
        #
        # For efficiency, we only populate this:
        #   - During export, and
        #   - If the source could potentially lead to a spurious export input
        #
        # Feel free to populate this more frequently if other use-cases arise,
        # but be aware that we have to generate full stacks for each
        # recording!
        self.source_to_user_stacks: dict[Source, list[traceback.StackSummary]] = {}

        self._current_tx: list[InstructionTranslatorBase] = []
        self.cleanups: list[CleanupHook] = []
        self.should_exit = False
        self.unspec_variable_map: dict[str, UnspecializedPythonVariable] = {}

        # This returns false if TF Overall (both mode and subclass) is disabled OR that TF Mode stack is empty
        self.torch_function_mode_enabled = torch._C._is_torch_function_mode_enabled()

        # Tracks if the output graph has a user defined allowed function in the
        # graph. This is used later to determine if we should fallback to eager
        # for certain exceptions. THe idea is that if the user has applied
        # allow_in_graph, they would like to see the error instead of falling
        # back for backend errors.
        self.has_user_defined_allowed_in_graph = False

        # Tracks a list of called ops that were not tagged with "pt2_compliant_tag".
        # This information is useful for logging.
        self.non_compliant_ops: set[torch._ops.OpOverload] = set({})

        # Tracks a list of called custom ops that were tagged with "pt2_compliant_tag".
        # This information is useful for logging.
        self.compliant_custom_ops: set[torch._ops.OpOverload] = set({})

        # We save the global torch state here to be restored in case of graph
        # breaks. The relevant issue is seen here
        # https://github.com/pytorch/pytorch/pull/100570#issuecomment-1543427086
        # where inlining of a function changes the global state (because of the
        # presence of torch.no_grad) and there is a graph break.
        self.save_global_state()

        # Tracks the original FQNs of the constant tensors from the original graph,
        # i.e. buffers and parameters.
        self.dynamo_flat_name_to_original_fqn: dict[str, str] = {}

        # All calls to random() are replaced with a single call to __gen_rand_values
        # functions that returns a tuple of random values for each original call.
        # random_calls tracks calls to random() and random_values_var stores the name of
        # the variable that stores __gen_rand_values results.
        self.random_calls: list[
            tuple[Callable[..., object], tuple[object, ...], dict[str, object]]
        ] = []
        self.random_values_var: Any = None

        # Bytecode to insert right before we call the graph
        self.pregraph_bytecode: list[Instruction] = []

        # Use to pass values to backward hooks when using compiled autograd
        self.backward_state: dict[str, VariableTracker] = {}
        self.backward_state_proxy: Optional[torch.fx.Proxy] = None
        self.backward_state_var: Optional[str] = None

        # pyrefly: ignore [bad-override]
        self.name_of_builtins_dict_key_in_fglobals: str = (
            self.install_builtins_dict_in_fglobals()
        )

        self.compiler_trace_stack = contextlib.ExitStack()

        # These are the ambient, currently-global saved_tensor_hooks stashed in autograd,
        # that are set for the entire duration of the compiled region.
        # This is an invariant today because we graph break on the saved_tensor_hook
        # context manager inside a compiled region
        self.saved_tensors_hooks_subgraph_names: Optional[list[str]] = (
            self.maybe_install_saved_tensors_hooks_subgraphs()
        )

        # mangled alias -> module fqn name
        self.import_sources: dict[str, str] = {}

        self.export_metadata = ExportMetaData()

        # Set of inlined unspecialized modules names to generate the
        # dynamo_flat_name_to_original_fqn mapping.
        self.used_inlined_inbuilt_modules_names: OrderedSet[str] = OrderedSet()

    def mark_bytecode_tracing_start(self) -> None:
        self.compiler_trace_stack.enter_context(
            dynamo_timed(
                "bytecode_tracing",
                log_pt2_compile_event=True,
            )
        )

    def mark_bytecode_tracing_stop(self) -> None:
        self.compiler_trace_stack.close()

    def install_builtins_dict_in_fglobals(self) -> str:
        f_builtins = get_builtins_dict(self.global_scope)
        return self.install_global("__builtins_dict__", f_builtins)

    def add_backward_state_hook(
        self, hook: VariableTracker, prefix: str = "hook"
    ) -> tuple[str, torch.fx.Proxy]:
        name = f"{prefix}{len(self.backward_state)}"
        assert name not in self.backward_state
        self.backward_state[name] = hook
        return name, self.get_backward_state_proxy()

    def get_backward_state_proxy(self) -> torch.fx.Proxy:
        if self.backward_state_proxy is None:
            if self.export:
                unimplemented(
                    gb_type="backward_state does not support export",
                    context="",
                    explanation="Compiled autograd doesn't work with `torch.export`.",
                    hints=[],
                )
            example_value = BackwardState()
            self.backward_state_proxy = self.root_tracer.create_graph_input(
                "dynamo_backward_state",
                type(example_value),
                example_value,
                source=BackwardStateSource(),
            )
            self.backward_state_proxy.node.meta["grapharg"] = BackwardStateGraphArg()
            self.backward_state_var = self.new_var()
        return self.backward_state_proxy

    # This gets its own helper function so guards DEBUG logs are more informative
    def init_ambient_guards(self) -> None:
        # Register a SHAPE_ENV guard to make sure we setup shape guards
        # that show up in ShapeEnv
        self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))

        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.DETERMINISTIC_ALGORITHMS)
        )

        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GRAD_MODE))

        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DEFAULT_DEVICE))

        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GLOBAL_STATE))
        self.guards.add(
            GlobalStateSource().make_guard(GuardBuilder.TORCH_FUNCTION_STATE)
        )

        ci = torch._C._functorch.peek_interpreter_stack()
        if ci is not None:
            self.guards.add(
                GlobalStateSource().make_guard(GuardBuilder.FUNCTORCH_STACK_MATCH)
            )
        if not torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            self.guards.add(
                GlobalStateSource().make_guard(
                    GuardBuilder.AUTOGRAD_SAVED_TENSORS_HOOKS
                )
            )

    def maybe_install_saved_tensors_hooks_subgraphs(self) -> Optional[list[str]]:
        if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            return None

        get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
        are_inline_hooks = (
            torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
        )
        hooks = get_hooks()
        if not are_inline_hooks(hooks):
            return None

        # If GraphModule provided by user contains fx.wrap,
        # We can only rely on user provided cache hash in this case.
        # If user did not provide cache hash - then we always bypass cache.

        pack_gm, unpack_gm = hooks
        pack_subgraph_name = self.install_subgraph(
            "saved_tensors_hooks_pack",
            torch.fx.GraphModule(self.nn_modules, pack_gm.graph),
        )
        unpack_subgraph_name = self.install_subgraph(
            "saved_tensors_hooks_unpack",
            torch.fx.GraphModule(self.nn_modules, unpack_gm.graph),
        )
        assert pack_subgraph_name == "saved_tensors_hooks_pack_0"
        assert unpack_subgraph_name == "saved_tensors_hooks_unpack_0"
        return [pack_subgraph_name, unpack_subgraph_name]

    def synthetic_graph_input(
        self, fn: Callable[..., Any], args: tuple[Any, ...]
    ) -> VariableTracker:
        """
        call fn(*args) before the graph runs and turn the result into a fake input.
        """
        example_value = fn(*args)
        varname = self.new_var()
        cg = PyCodegen(self.root_tx)
        cg.add_push_null(
            lambda: cg.load_import_from(
                fn.__module__,
                fn.__name__,
            )
        )
        cg.foreach(map(variables.ConstantVariable.create, args))
        cg.call_function(len(args), False)
        cg.store(varname)
        self.pregraph_bytecode.extend(cg.get_instructions())
        source = SyntheticLocalSource(varname)
        result = VariableTracker.build(self.root_tx, example_value, source)
        # Realize the VT because we will delete the guards on it in the next line.
        result = result.realize()
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def add_cleanup_hook(self, fn: Callable[[], Any]) -> None:
        self.cleanup_hooks.append(fn)

    def call_cleanup_hooks(self) -> None:
        for hook in reversed(self.cleanup_hooks):
            hook()
        self.cleanup_hooks.clear()

    @property
    def root_tracer(self) -> "SubgraphTracer":
        return self.tracers[0]

    @property
    def current_tracer(self) -> "SubgraphTracer":
        return self.tracers[-1]

    def is_root_tracer(self) -> bool:
        # Helper to tell if we are inside the higher order operator tracing.
        return len(self.tracers) == 1

    @property
    def graph(self) -> torch.fx.Graph:
        return self.current_tracer.graph

    # TODO(rzou): can delete after we refactor speculate_subgraph to use nested GraphTracer.
    @graph.setter
    def graph(self, value: torch.fx.Graph) -> None:
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self) -> dict[str, fx.Proxy]:
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self) -> dict[fx.Node, torch.Tensor]:
        return self.current_tracer.real_value_cache

    @property
    def bound_symbols(self) -> dict[sympy.Symbol, Union[torch.fx.Proxy, "LazyProxy"]]:
        return self.current_tracer.bound_symbols

    # If you are here, and you're looking for create_graph_input,
    # to avoid ambiguity, please call one of the following:
    # - self.current_tracer.create_graph_input
    # - self.root_tracer.create_graph_input
    # See NOTE [HigherOrderOperator tracing design] for more context.

    def create_proxy(self, *args: Any, **kwargs: Any) -> torch.fx.Proxy:
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args: Any, **kwargs: Any) -> torch.fx.Node:
        return self.current_tracer.create_node(*args, **kwargs)

    def remove_node(self, *args: Any, **kwargs: Any) -> None:
        return self.current_tracer.remove_node(*args, **kwargs)

    @contextlib.contextmanager
    def subtracer(
        self, source_target: Optional[Target], prior_tracer: "SubgraphTracer"
    ) -> Generator[fx.Tracer, None, None]:
        new_scope_ctx = enter_new_scope()
        try:
            if prior_tracer:
                # Lineage MUST stay preserved
                assert prior_tracer.parent is self.current_tracer
            new_scope_ctx.__enter__()
            tracer = (
                prior_tracer
                if prior_tracer
                else SubgraphTracer(
                    self,
                    parent=self.current_tracer,
                    source_target=source_target,
                    is_export=self.current_tracer.is_export,
                )
            )
            self.tracers.append(tracer)
            yield tracer
        finally:
            new_scope_ctx.__exit__(None, None, None)
            self.tracers.pop()

    @property
    def output(self) -> "OutputGraph":
        return self

    @property
    def fake_mode(self) -> torch._subclasses.FakeTensorMode:
        assert self.tracing_context.fake_mode is not None
        return self.tracing_context.fake_mode

    @property
    def shape_env(self) -> ShapeEnv:
        assert self.tracing_context.fake_mode is not None
        assert self.tracing_context.fake_mode.shape_env is not None
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self.tracing_context.guards_context.dynamo_guards

    @property
    def nn_modules(self) -> dict[str, Any]:
        return self.tracing_context.module_context.nn_modules

    @property
    def aotautograd_guards(self) -> list[torch._guards.GuardEnvExpr]:
        return self.tracing_context.guards_context.aotautograd_guards

    def save_global_state(
        self, out: Optional[dict[str, tuple[Callable[..., Any], bool]]] = None
    ) -> None:
        """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
        global_state = cast(
            dict[str, tuple[Callable[..., Any], bool]],
            (
                out
                if out is not None
                else self.tracing_context.global_context.global_state
            ),
        )

        global_state["grad_enabled"] = (torch.set_grad_enabled, torch.is_grad_enabled())

        global_state["autocast_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cuda"),
            torch.is_autocast_enabled("cuda"),
        )
        global_state["autocast_cpu_enabled"] = (
            functools.partial(torch.set_autocast_enabled, "cpu"),
            torch.is_autocast_enabled("cpu"),
        )
        global_state["autocast_gpu_dtype"] = (  # type:ignore[assignment]
            functools.partial(torch.set_autocast_dtype, "cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        global_state["autocast_cpu_dtype"] = (  # type:ignore[assignment]
            functools.partial(torch.set_autocast_dtype, "cpu"),
            torch.get_autocast_dtype("cpu"),
        )
        global_state["autocast_cache_enabled"] = (
            torch.set_autocast_cache_enabled,
            torch.is_autocast_cache_enabled(),
        )

    def push_tx(self, tx: "InstructionTranslatorBase") -> None:
        self._current_tx.append(tx)

    def pop_tx(self) -> "InstructionTranslatorBase":
        return self._current_tx.pop()

    @property
    def current_tx(self) -> "InstructionTranslatorBase":
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def count_calls(self) -> int:
        return count_calls(self.graph)

    def is_empty_graph(self) -> bool:
        return len(list(self.graph.nodes)) == 0

    def has_outputs(self) -> bool:
        return len([x for x in self.graph.nodes if x.op == "output"]) > 0

    def get_submodule(self, keys: str) -> Union[torch.nn.Module, Any]:
        assert keys
        obj: Union[torch.nn.Module, dict[str, torch.nn.Module]] = self.nn_modules
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def new_var(self, name: str = "tmp") -> str:
        existing = set(self.code_options["co_varnames"])
        # In common case, this will be O(1)
        while True:
            var = f"{name}_{next(self.unique_var_id)}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)
                return var

    def update_co_names(self, name: str) -> None:
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)

    @staticmethod
    def module_key_name(*names: Any) -> str:
        # create a new unique name
        name = "_".join(map(str, names))
        # Strip _buffers[..]/_parameters[..]/_modules[..] names
        name = re.sub(
            r"\._(?:modules|parameters|buffers)\[(['\"])([^'\"\]]+)\1\]", r".\2", name
        )
        # Replace getattr(a, b) with a.b
        name = re.sub(
            r"getattr\(\s*([^,]+?)\s*,\s*(['\"])([^'\"]+)\2\s*\)", r"\1.\3", name
        )
        # Strip the guard lookup L/G access
        name = re.sub(r"^[GL]\['?(.*?)'?\]$", r"\1", name)
        # e.g. replace abc.xyz[123].qkv with abc.xyz_123.qkv
        name = re.sub(r"\[(\d+)\]", r"_\g<1>", name)
        # e.g. replace abc.xyz_123.qkv with abc_xyz_123_qkv
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        if not name or not name[0].isalpha():
            name = "sub" + name

        return name

    def register_static_attr_and_return_proxy(
        self, attr_prefix: str, attr_value: Any
    ) -> fx.Proxy:
        # Check if the module already exists, if it does, return the already
        # added proxy. This is important for executorch tests.
        if isinstance(attr_value, torch.nn.Module):
            for name, mod in self.nn_modules.items():
                if mod is attr_value:
                    proxy = self.create_proxy("get_attr", name, (), {})
                    return proxy

        attr_name = get_unique_name_wrt(attr_prefix, self.nn_modules)
        # TODO `nn_modules` has been historically overloaded to store a lot more
        # than just nn module objects, fix that.
        self.nn_modules[attr_name] = attr_value
        proxy = self.create_proxy("get_attr", attr_name, (), {})
        set_example_value(proxy.node, attr_value)
        return proxy

    def register_attr_or_module(
        self,
        target: Union[torch.nn.Module, torch.Tensor, Any],
        *names: Any,
        **options: Any,
    ) -> VariableTracker:
        if is_dynamic_nn_module(target, self.export):
            # Instead of returning UnspecializedNNModuleVariable, call
            # VariableTracker.build so that it is tracked for mutation.
            return VariableTracker.build(self.current_tx, target, **options)

        options = dict(options)
        assert "source" in options
        source = options["source"]
        assert not isinstance(source, ParamBufferSource)

        if isinstance(target, torch.Tensor):
            tracer = self.current_tracer
            if not self.is_root_tracer():
                # For higher order ops, we don't want to insert the get_attr in
                # innermost graph. Instead, we want to raise the params/buffers
                # as inputs to the higher-order graph, and register them as
                # get_attrs in the root tracer.

                # Note that Dynamo will still call lift_tracked_freevar_to_input
                # when these inputs are encountered for the inner graph. The
                # only difference is what happens at the root tracer for
                # nn.Parameters vs free inputs. The free inputs are registered
                # as placeholders in the root graph, whereas the nn.Parameters
                # are registered as get_attr nodes in the root graph.
                tracer = self.root_tracer

            def wrap_name(module_key: str) -> VariableTracker:
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source

                # Check if the attr has already been registered. This can happen
                # when two different sources point to the same tensor.
                assert self.root_tx is not None
                if target in self.root_tx.output.side_effects:
                    return self.root_tx.output.side_effects[target]

                if get_static_address_type(target) == "guarded" and not isinstance(
                    source, NumpyTensorSource
                ):
                    install_guard(source.make_guard(GuardBuilder.ID_MATCH))
                elif not is_constant_source(source):
                    install_guard(source.make_guard(GuardBuilder.TENSOR_MATCH))

                vt = wrap_fx_proxy(
                    self.root_tx,
                    tracer.create_proxy("get_attr", module_key, (), {}),
                    example_value=target,
                    **options,
                )

                # Track the object so to avoid duplicate registration in case of
                # different sources pointing to the same tensor object.
                vt = self.root_tx.output.side_effects.track_object_existing(target, vt)

                assert "tensor_dict" not in vt.as_proxy().node.meta
                # pyrefly: ignore [bad-argument-type]
                vt.as_proxy().node.meta["tensor_dict"] = _extract_tensor_dict(target)

                return vt

        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)

            if source:
                install_guard(source.make_guard(GuardBuilder.NN_MODULE))

                def wrap_name(module_key: str) -> VariableTracker:
                    # pyrefly: ignore [bad-argument-type]
                    return NNModuleVariable(type(target), module_key, target, **options)

            else:
                # This is Dynamo created graph module, e.g., graph module coming
                # from higher order ops. NNModuleVariable tracker can't be
                # sourceless, so let's return a unspecializedNNModule variable
                # tracker.
                def wrap_name(module_key: str) -> VariableTracker:
                    return variables.UnspecializedNNModuleVariable(target, **options)

        elif isinstance(target, (torch.SymInt, torch.SymFloat)):
            # HACKY CODE REGION BEGIN
            # WE ARE PIGGYBACKING ON EXISTING INFRA TO REGISTER ATTRS
            # This ultimately gets written to self.nn_modules, which is unfortunate
            # Attrs that are tenors and symints and such need to be migrated to have their
            # own storage
            # alas, this is like this for now

            def wrap_name(module_key: str) -> VariableTracker:
                return SymNodeVariable.create(
                    self,
                    self.create_proxy("get_attr", module_key, (), {}),
                    sym_num=target,
                    **options,
                )

            # HACKY CODE REGION END
        else:

            def wrap_name(module_key: str) -> VariableTracker:
                self.output.update_co_names(module_key)
                self.global_scope[module_key] = target
                return VariableTracker.build(
                    self,  # type: ignore[arg-type]
                    target,
                    ConstantSource(source_name=module_key),
                )

        for k, v in self.nn_modules.items():
            if v is target:
                # it already exists
                return wrap_name(k)

        name = OutputGraph.module_key_name(*names)
        name = get_unique_name_wrt(name, self.nn_modules, self.global_scope)
        self.nn_modules[name] = target
        if isinstance(target, torch.nn.Module):

            def register_leaf_name(leaf_name: str) -> None:
                assert self.param_name_to_source is not None
                new_source = ParamBufferSource(source, leaf_name)
                new_name = f"{name}.{leaf_name}"
                self.param_name_to_source[new_name] = new_source
                if isinstance(source, LocalSource):
                    self.dynamo_flat_name_to_original_fqn[
                        OutputGraph.module_key_name(new_source.name())
                    ] = leaf_name

            # annoying, but there are cases when we do not have parameters
            # see test_nn_moduledict_contains
            if hasattr(target, "_parameters"):
                for leaf_name, _ in target.named_parameters():
                    register_leaf_name(leaf_name)
            if hasattr(target, "_buffers"):
                for leaf_name, _ in target.named_buffers():
                    register_leaf_name(leaf_name)

        return wrap_name(name)

    def handle_aliases_for_stolen_lists(
        self, tx: "InstructionTranslatorBase"
    ) -> tuple[list[Instruction], dict[Source, Source]]:
        # If list inputs are stolen, but still needed after the function call, create aliases to keep them alive
        maybe_gm = self.local_scope.get("self")
        stolen_list_names = get_locals_to_steal(maybe_gm)
        if not stolen_list_names:
            return [], {}

        alias_insts = []
        needs_alias: dict[str, list[VariableTracker]] = {}

        queue = [
            *tx.stack,
            *tx.symbolic_locals.values(),
            *self.side_effects.store_attr_mutations.keys(),
        ]

        while queue:
            x = queue.pop()
            if isinstance(x, BaseListVariable):
                assert isinstance(x.items, list)
                queue += x.items
 
```



## High-Level Overview

"""Core graph building functionality for PyTorch's Dynamo system. This module containsthe essential components for constructing and managing FX graphs during compilation:- OutputGraph: Manages the overall graph construction and compilation process. It owns  a SubgraphTracer and handles graph compilation, execution, and state management.  OutputGraph also manages features like graph deduplication, symbolic shape handling,  and tracking of side effects.- SubgraphTracer: Handles the actual FX graph construction by tracing Python code.  It supports advanced features like higher-order operators through nested tracers,  lifting of free variables, and handling of symbolic shapes.The module supports key Dynamo features including:- Higher-order operators through nested SubgraphTracers- Graph deduplication for optimization- Symbolic shape handling and propagation- Side effect tracking and management- Guard insertion and management

This Python file contains 20 class(es) and 132 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `VariableTrackerCacheKey`, `AliasingInfo`, `MutationInfo`, `VariableTrackerCache`, `GraphCompileReason`, `FakeRootModule`, `WrapperBackend`, `OutputGraphGuardsState`, `StackLocalsMetadata`, `ExportMetaData`, `OutputGraphCommon`, `OutputGraph`, `DynamoTracerOutput`, `LazyProxy`, `SubgraphTracer`

**Functions defined**: `__init__`, `lookup`, `add`, `clone`, `clear`, `_step_logger`, `__post_init__`, `_get_gen_rand_values_fn`, `_gen_rand_values`, `__init__`, `__repr__`, `add_nn_modules`, `__init__`, `__call__`, `shape_env`, `guards`, `aotautograd_guards`, `dump_guards_state`, `get_builtins_dict`, `__init__`

**Key imports**: collections, contextlib, copy, functools, inspect, itertools, logging, operator, re, sys


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `contextlib`
- `copy`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `operator`
- `re`
- `sys`
- `traceback`
- `warnings`
- `weakref`
- `collections.abc`: Callable, Generator, Sequence
- `dataclasses`: dataclass, field as dc_field
- `types`: CodeType
- `typing`: Any, cast, Optional, TYPE_CHECKING, Union
- `typing_extensions`: ParamSpec, TypeVar
- `sympy`
- `torch._guards`
- `torch._logging`
- `torch.distributed as dist`
- `torch.nn`
- `torch.utils._pytree as pytree`
- `torch`: fx, Tensor
- `torch._C._dynamo`: guards
- `torch._dynamo.exc`: ShortenTraceback, TensorifyScalarRestartAnalysis
- `torch._library.opaque_object`: is_opaque_type
- `torch._subclasses.fake_tensor`: FakeTensor
- `torch._utils_internal`: signpost_event


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `output_graph.py_docs.md`
- **Keyword Index**: `output_graph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
