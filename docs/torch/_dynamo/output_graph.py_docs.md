# Documentation: output_graph.py

## File Metadata
- **Path**: `torch/_dynamo/output_graph.py`
- **Size**: 166763 bytes
- **Lines**: 3873
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
                continue

            if not (
                (
                    x not in self.side_effects.store_attr_mutations
                    or isinstance(x.mutation_type, AttributeMutationExisting)
                )
                and isinstance(x.source, GetItemSource)
                and isinstance(x.source.base, LocalSource)
                and x.source.base.local_name in stolen_list_names
            ):
                continue

            stolen_name = x.source.base.local_name
            if stolen_name not in needs_alias:
                needs_alias[stolen_name] = []
            needs_alias[stolen_name].append(x)

        visited = {}
        overridden_sources: dict[Source, Source] = {}
        for arg in self.graphargs:
            if not (
                isinstance(arg._example, list)
                and isinstance(arg.source, LocalSource)
                and arg.source.local_name in needs_alias
            ):
                continue

            # arg is a list that will be cleared by the compiled function
            list_name = arg.source.local_name
            assert list_name in self.code_options["co_varnames"]
            for x in needs_alias[list_name]:
                # Skip if already handled.
                if x.source in overridden_sources:
                    continue

                # A small codegen optimization because we might have different
                # VariableTrackers that share the same source.
                assert x.source is not None
                list_idx = x.source.index  # type: ignore[attr-defined]
                if list_idx not in visited:
                    alias_name = self.new_var(
                        f"{list_name}_ref"
                    )  # self.new_var already adds unique id suffix

                    visited[list_idx] = alias_name
                    # bytecode of `alias_name = list_name[list_idx]`
                    alias_insts.extend(
                        [
                            create_instruction("LOAD_FAST", argval=list_name),
                            create_load_const(list_idx),
                            create_binary_subscr(),
                            create_instruction("STORE_FAST", argval=alias_name),
                        ]
                    )

                # operate on alias, handled by suffix codegen
                assert x.source is not None
                old_source = x.source
                overridden_sources[old_source] = LocalSource(visited[list_idx])

        # NOTE: we need `overridden_sources` because (1) we want to codegen for
        # these list items to use the new local source, but (2) we want to avoid
        # updating `source` in place because that might break invariants in
        # other parts of Dynamo like guards.
        return alias_insts, overridden_sources

    def _get_stack_values_to_restore(
        self, tx: "InstructionTranslatorBase", stack_pops: int
    ) -> tuple[list[VariableTracker], StackLocalsMetadata]:
        """
        Gets the stack + locals values belonging to tx that need to be restored.

        Also prunes dead tx locals and realizes all VTs in the tx's stack.

        NullVariables in stack/locals will NOT be restored, unless they are the top `stack_pops`
        elements of the stack - it is expected that the next instruction to run will pop the top
        `stack_pops` elements of the stack, so we should codegen NULLs.

        Returns:
            - stack_values: stack and locals values that need to be restored
            - meta: locations of NULLs and ContextWrappingVariables in the stack/locals
                (ignores the top `stack_pops` values on the stack)
        """
        tx.prune_dead_locals()

        stack_values = []
        meta = StackLocalsMetadata()

        # realize any unrealized tensor VTs in case they
        # need to be added to self.nn_modules as attributes
        for i, value in enumerate(tx.stack):
            variables.LazyVariableTracker.realize_all(value)
            # ignore top `stack_pops` values on the stack
            if len(tx.stack) - i <= stack_pops:
                stack_values.append(value)
                continue
            if isinstance(value, NullVariable):
                meta.stack_null_idxes.append(i)
            else:
                stack_values.append(value)
            if isinstance(value, ContextWrappingVariable):
                target_values = (
                    () if value.target_values is None else tuple(value.target_values)
                )
                # NOTE: track index in stack after NULLs have been removed
                meta.stack_ctx_args.append((len(stack_values) - 1, target_values))
                meta.stack_ctx_idxes_orig.append(i)

        meta.num_stack = len(stack_values)

        cell_and_freevars = set(tx.cellvars() + tx.freevars())

        # NB: Typically (i.e., for graph compile from RETURN_VALUE),
        # symbolic_locals will be empty at this point, as prune_dead_locals
        # will clear out all of symbolic_locals because RETURN_VALUE is the
        # last instruction and no more locals are used.  The fanciness here
        # is only needed for partial graphs.
        # NOTE: All cell and free variables are represented as CellVariable,
        # so checks for NULLs and context managers in the case of codegen'ing resume
        # functions will not be performed on them. This is expected behavior.
        for k, v in tx.symbolic_locals.items():
            # Note! this explicitly uses .local_name for matching
            # Failure to do so will cause spurious registrations in val_to_names.
            # This will in turn result in spurious variables showing up in the graph.
            # This was very tricky to debug. For an example, dump the graph at call_user_compiler
            # while running test_subgraphs.py
            # Do not include top-frame unmodified locals here - otherwise, the compiled graph may
            # erroneously include them as part of the return. We manually codegen them afterward.
            if (
                isinstance(v.source, LocalSource)
                and v.source.local_name == k
                and tx is self.root_tx
            ):
                continue
            # Do not load cell/free vars
            if k in cell_and_freevars:
                continue
            # Do not load variable if it is NULL.
            if sys.version_info >= (3, 12):
                # NOTE: do not use isinstance, since it realizes lazy VT's
                # Continuation function will load the NULL for v.
                if type.__instancecheck__(NullVariable, v):
                    meta.locals_null_keys.append(k)
                    continue
            else:
                # A variable should never be NULL in < 3.12
                assert not type.__instancecheck__(NullVariable, v)
            meta.locals_names[k] = len(meta.locals_names)
            if isinstance(v, ContextWrappingVariable):
                target_values = (
                    () if v.target_values is None else tuple(v.target_values)
                )
                meta.locals_ctx_args.append((k, target_values))
            stack_values.append(v)

        return stack_values, meta

    def compile_subgraph(
        self,
        tx: "InstructionTranslatorBase",
        reason: GraphCompileReason,
        partial_convert: bool = False,
        stack_pops: int = 0,
    ) -> list[StackLocalsMetadata]:
        """
        Compiles the current subgraph, with inputs w.r.t. self.root_tx, and codegens:
            - Call the compiled subgraph
            - Apply side effects
            - Codegen stack and locals
            - Store the locals

        Python does not allow NULL to be an arg to a function, so we do not codegen NULLs on the stack,
        unless the value is one of the top `stack_pops` values on the stack (these values are expected to be
        popped immediately after this generated code. The prologue of the resume function is expected to restore
        any dropped NULLs.

        Returns stack indices and locals keys where we dropped NULLs, and where we found inactive context manager objects.
        """

        assert self.root_tx is not None

        if not config.nested_graph_breaks:
            # expect to only compile 1 frame
            assert self.root_tx is tx

        # bytecode tracing has finished. Pop the context manager for dynamo_timed
        self.mark_bytecode_tracing_stop()

        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason
        self.should_exit = True

        log.debug("COMPILING GRAPH due to %s", reason)

        # prefix instructions (Python 3.11+)
        prefix_insts: list[Instruction] = []
        if sys.version_info >= (3, 11):
            for inst in self.root_tx.prefix_insts:
                if inst.opname == "COPY_FREE_VARS":
                    prefix_insts.append(
                        create_instruction(
                            "COPY_FREE_VARS",
                            arg=len(self.root_tx.code_options["co_freevars"]),
                        )
                    )
                else:
                    prefix_insts.append(copy.copy(inst))

        # stack values and restore vars for each frame are pushed in reverse order
        # i.e. last element corresponds to root frame (1),
        # first element corresponds to current frame (N)
        all_stack_values = []
        all_stack_locals_metas = []
        cur_tx: Optional[InstructionTranslatorBase] = tx
        while cur_tx is not None:
            # this should have been checked by the caller
            assert all(block.can_restore() for block in cur_tx.block_stack)

            stack_values, meta = self._get_stack_values_to_restore(
                cur_tx, stack_pops if cur_tx is tx else 0
            )
            all_stack_values.append(stack_values)
            all_stack_locals_metas.append(meta)

            # Exit from all context manager variables to make sure global state is restored
            for block in reversed(cur_tx.block_stack):
                block.exit(cur_tx, is_graph_break=reason.graph_break)

            cur_tx = cur_tx.parent

        # "Garbage collect the heap".
        self.side_effects.prune_dead_object_new(tx)

        self.add_output_instructions(prefix_insts)

        assert not (self.pregraph_bytecode and self.export), (
            "export does not support pregraph_bytecode"
        )
        self.add_output_instructions(self.pregraph_bytecode)

        alias_insts, overridden_sources = self.handle_aliases_for_stolen_lists(
            self.root_tx
        )
        self.add_output_instructions(alias_insts)

        self.cleanup_graph()

        # Use nn.Module "proxies" in the constructed GraphModule so that
        # the resulting GM does not hold additional strong references to the original modules.
        # This prevents a strong ref cycle where Dynamo created code holds on to references
        # to modules that also have Dynamo code cache invalidation checks.
        # When cache invalidation runs, the generated GM will be invalidated, which also deletes
        # the proxies.
        nn_modules_proxies = {
            name: nn_module_proxy(mod) for name, mod in self.nn_modules.items()
        }
        root = FakeRootModule(nn_modules_proxies)

        from .decorators import disable

        if has_user_objects():
            # NB: This is where we store possible user objects before running the graph
            # index_to_user_object_weakref is the function used in the graph to translate
            # the dynamo-generated index into the actual object passed to the compiled function.
            # We generate bytecode to store all user objects at the proper index in the below
            # call.
            codegen = PyCodegen(
                self.root_tx, root, overridden_sources=overridden_sources
            )
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,
                    "store_user_object_weakrefs",
                )
            )
            tmp_vars = []
            for constructor in index_to_bytecode_constructor.values():
                constructor(codegen)
                var_name = (
                    self.new_var()
                )  # keep alive any temp objects for the rest of the frame
                codegen.store(var_name)
                tmp_vars.append(var_name)

            for var_name in tmp_vars:
                codegen.append_output(codegen.create_load(var_name))

            codegen.call_function(len(index_to_bytecode_constructor), False)
            codegen.pop_top()
            self.add_output_instructions(codegen.get_instructions())

        # to handle random calls
        if len(self.random_calls) > 0:
            random_calls_instructions = []
            self.random_values_var = self.new_var("random_values")
            rand_fn = disable(
                _get_gen_rand_values_fn(self.random_calls),
                reason="do not trace into Dynamo rng recovery function",
            )
            rand_fn_name = self.install_global("__gen_rand_values", rand_fn)
            codegen = PyCodegen(
                self.root_tx, root, overridden_sources=overridden_sources
            )
            random_calls_instructions.extend(
                codegen.load_function_name(rand_fn_name, True)
            )
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(
                codegen.create_store(self.random_values_var),
            )
            self.add_output_instructions(random_calls_instructions)

        # Codegen stack convention before the unsupported instruction
        # NOTE: in these comment blocks, "locals" EXCLUDE free and cell vars.
        # NOTE: stack/locals/cells must be codegen'd BEFORE the unsupported instruction, since the latter
        # can arbitrarily mutate the former.
        # [frame N cells, .., frame 1 cells],
        # [
        #   frame N locals,
        #   frame N-1 stack + locals,
        #   ...,
        #   frame 1 stack + locals,
        # ], frame N stack

        # see symbolic_convert.py for
        # codegen stack convention after the unsupported instruction
        # NOTE: cells will be loaded into continuation functions directly by symbolic_convert

        # this determines the order that values are codegen'd to the stack
        stack_values_flat = [val for vals in all_stack_values for val in vals]
        stored_graph_output_var = False
        graph_output_var = None

        # call compiled fx graph and codegen all values - stack and locals
        if (
            self.root_tx is tx  # single frame
            and stack_values_flat
            and all(
                not isinstance(
                    v,
                    (
                        UnspecializedPythonVariable,
                        NumpyNdarrayVariable,
                        TensorWithTFOverrideVariable,
                    ),
                )
                and not (isinstance(v, SymNodeVariable) and v.python_type() is float)
                for v in stack_values_flat
            )
            and all(isinstance(x, TensorVariable) for x in stack_values_flat)
            and len(set(stack_values_flat)) == len(stack_values_flat)
            and self.side_effects.is_empty()
            and not tx.debug_locals
            and not self.backward_state
            and not all_stack_locals_metas[-1].stack_null_idxes
            and not all_stack_locals_metas[-1].locals_null_keys
        ):
            # optimization to generate better code in a common case

            # codegen cells
            # no side effects, so no new cells created - no need to call side_effects.codegen_save_tempvars
            cell_cg = PyCodegen(self.root_tx)
            self.codegen_cells(tx, cell_cg)
            self.add_output_instructions(
                [
                    # load in reverse since UNPACK_SEQUENCE will reverse
                    *self.compile_and_call_fx_graph(
                        tx, list(reversed(stack_values_flat)), root
                    ),
                    *cell_cg.get_instructions(),
                    *create_swap(2),
                    create_instruction("UNPACK_SEQUENCE", arg=len(stack_values_flat)),
                ]
            )
            # function output will be moved to the correct places below
        else:
            graph_output_var = self.new_var("graph_out")
            # load stack values in a flat manner - we will codegen bytecode to place them correctly
            # according to our convention above
            pass1 = PyCodegen(
                self.root_tx,
                root,
                graph_output_var,
                overridden_sources=overridden_sources,
            )
            self.codegen_suffix(tx, stack_values_flat, pass1)

            # Use `pass1.uses` to selectively cache multi-user variables into a
            # temporary local source. This (a). speeds up loading VTs with long
            # chained source, and (b). avoids redundantly saving single-user VT
            # into a temporary local.
            tempvars = {}  # type: ignore[var-annotated]
            for val, count in pass1.uses.items():
                # If it's already a local source, no need to cache it
                if count > 1 and not istype(val, (SyntheticLocalSource, LocalSource)):
                    tempvars[val] = None
            pass2 = PyCodegen(
                self.root_tx,
                root,
                graph_output_var,
                tempvars=tempvars,
                overridden_sources=overridden_sources,
            )
            self.codegen_suffix(tx, stack_values_flat, pass2)

            if (
                torch._dynamo.config.log_graph_in_out_metadata
                and stack_values_flat
                and len(stack_values_flat) == 1
            ):
                vt = stack_values_flat[0]
                if (
                    isinstance(vt, torch._dynamo.variables.NamedTupleVariable)
                    and vt.tuple_cls
                    is torch._dynamo.functional_export.ExportTracerOutput
                ):
                    flat_returns = vt.items[0]
                    out_spec = vt.items[1]
                    assert isinstance(
                        flat_returns, torch._dynamo.variables.ListVariable
                    )

                    vt_to_graph_out_idx: dict[VariableTracker, int] = {}
                    for value in pass2.graph_outputs.values():
                        assert isinstance(value, torch._dynamo.codegen.GraphOutputEntry)
                        variable: VariableTracker = value.variable
                        vt_to_graph_out_idx[variable] = value.index

                    for idx, vt in enumerate(flat_returns.items):
                        if vt in vt_to_graph_out_idx:
                            self.export_metadata.output_return_type[idx] = (
                                "graph_out",
                                vt_to_graph_out_idx[vt],
                            )
                        elif (
                            vt.source is not None
                            and (source := getattr(vt.source, "base", None))  # type: ignore[assignment]
                            and source.is_input
                        ):
                            self.export_metadata.output_return_type[idx] = (
                                "input",
                                vt.source,
                            )
                        elif isinstance(vt, torch._dynamo.variables.ConstantVariable):
                            self.export_metadata.output_return_type[idx] = (
                                "constant",
                                vt.as_python_constant(),
                            )
                        else:
                            assert f"Encountered unrecognized type {vt} at output {idx}"  # noqa: PLW0129

                    self.export_metadata.out_spec = out_spec.as_python_constant()

            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(
                    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
                )

                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                    stored_graph_output_var = True
                else:
                    output.append(create_instruction("POP_TOP"))
            else:
                # NB: Important to run compiler collective even when there is
                # a graph break
                self.run_compiler_collective()
            self.add_output_instructions(output + pass2.get_instructions())

        # store all stack and locals for each frame
        # current state of the stack:
        # all cells,
        # *(frame N stack), *(frame N locals),
        # ...,
        # *(frame 1 stack), *(frame 1 locals)

        self.add_output_instructions(
            [
                create_instruction(
                    "BUILD_LIST",
                    arg=len(stack_values_flat) - all_stack_locals_metas[0].num_stack,
                ),
            ]
        )

        # current state of the stack:
        # all cells,
        # *(frame N stack), [
        #     *(frame N locals),
        #     *(frame N-1 stack), *(frame N-1 locals),
        #     ...
        #     *(frame 1 stack), *(frame 1 locals),
        # ]
        # iterate current frame (N) to root frame (1)
        # sliding window over frame stack/locals
        start_idx = 0
        end_idx = 0
        for i, meta in enumerate(all_stack_locals_metas):
            # do not pack frame N's stack into the value list
            n_vals = len(meta.locals_names)
            if i != 0:
                n_vals += meta.num_stack
            if n_vals == 0:
                self.add_output_instructions(
                    [
                        create_instruction("BUILD_LIST", arg=0),
                        *create_swap(2),
                    ]
                )
                # [], stack_values_flat
            else:
                end_idx += n_vals
                self.add_output_instructions(
                    [
                        create_dup_top(),
                        *create_binary_slice(start_idx, end_idx),
                        *create_swap(2),
                    ]
                )
                start_idx += n_vals
                # stack_values_flat[x:y], stack_values_flat

            # add root frame's unmodified locals here
            if i == len(all_stack_locals_metas) - 1:
                root_cg = PyCodegen(self.root_tx)
                unmodified_locals_names: dict[str, int] = {}
                for k, v in self.root_tx.symbolic_locals.items():
                    if isinstance(v.source, LocalSource) and v.source.local_name == k:
                        root_cg.append_output(root_cg.create_load(k))
                        unmodified_locals_names[k] = len(meta.locals_names) + len(
                            unmodified_locals_names
                        )
                self.add_output_instructions(
                    root_cg.get_instructions()
                    + [
                        create_instruction(
                            "BUILD_LIST", arg=len(unmodified_locals_names)
                        ),
                        # arg=2 because we already swapped the locals list back
                        create_instruction("LIST_EXTEND", arg=2),
                    ]
                )
                meta.locals_names.update(unmodified_locals_names)

            # *(frame N stack), metas[0] stack + locals, ..., metas[i] stack + locals, stack_values_flat

        # current state of the stack:
        # all cells,
        # *(frame N stack),
        # frame N locals,
        # frame N-1 stack, frame N-1 locals,
        # ...
        # frame 1 stack, frame 1 locals,
        # stack_values_flat
        #

        self.add_output_instructions(
            [
                create_instruction("POP_TOP"),
                create_instruction("BUILD_LIST", arg=len(all_stack_locals_metas)),
                *create_rot_n(all_stack_locals_metas[0].num_stack + 1),
            ]
        )

        # final state of the stack before running the unsupported bytecode:
        # all cells,
        # [
        #   [frame N locals],
        #   [frame N-1 stack + locals],
        #   ...,
        #   [frame 1 stack + locals],
        # ], *(frame N stack)

        if graph_output_var and stored_graph_output_var:
            self.add_output_instructions(
                [create_instruction("DELETE_FAST", argval=graph_output_var)]
            )

        if torch._dynamo.config.side_effect_replay_policy in ["warn", "error"]:
            from torch.export._trace import _ExportModuleSpecTrackerDict

            potential_side_effects = []
            for var in self.side_effects._get_modified_vars():
                if hasattr(var, "mutation_type"):
                    mut_type = var.mutation_type
                    # Make sure to skip codegen specific mutations
                    if isinstance(
                        mut_type, (AttributeMutationExisting, ValueMutationExisting)
                    ):
                        if isinstance(var, UserDefinedDictVariable) and isinstance(
                            var.value, _ExportModuleSpecTrackerDict
                        ):
                            for k, v in var.items.items():
                                specs = {}
                                for k_spec, val in v.items.items():
                                    specs[k_spec.vt.as_python_constant()] = (
                                        val.as_python_constant()
                                    )
                                assert ["in_spec", "out_spec"] == list(specs.keys())
                                self.export_metadata.module_call_spec[
                                    k.vt.as_python_constant()
                                ] = specs
                        # export uses tracepoint pass to dump submodule inp/out spec
                        # into global state, so we filter it here
                        if not (
                            isinstance(var, UserDefinedDictVariable)
                            and isinstance(var.value, _ExportModuleSpecTrackerDict)
                        ):
                            potential_side_effects.append(var)
            side_effect_refs = [
                _get_source_debug_name(var.source) for var in potential_side_effects
            ]

            if side_effect_refs:
                if torch._dynamo.config.side_effect_replay_policy == "warn":
                    warnings.warn(
                        f"While compiling, we found certain side effects happened in the model.forward. "
                        f"Here are the list of potential sources you can double check: {side_effect_refs}"
                    )
                else:
                    raise RuntimeError(
                        f"While compiling, we found certain side effects happened in the model.forward. "
                        f"Here are the list of potential sources you can double check: {side_effect_refs}"
                    )

        return all_stack_locals_metas

    def codegen_cells(self, tx: "InstructionTranslatorBase", cg: PyCodegen) -> None:
        # no need to codegen if reason.graph_break is False (since we won't resume)
        if self.compile_subgraph_reason.graph_break:
            tx_cnt = 0
            cur_tx: Optional[InstructionTranslatorBase] = tx
            while cur_tx is not None:
                # NOTE: we generate cells in the same order as resume_execution.py: sorted freevars + cellvars
                # Emitting `LOAD_FAST/LOAD_CLOSURE` with names in `co_freevars`
                # requires that in the generated bytecode, these cells would keep
                # their original local names, which we ensure via
                # `CellVariable.local_name`.
                freevars = tuple(sorted(cur_tx.cell_and_freevars()))
                for cell in freevars:
                    if cur_tx is self.root_tx:  # root frame
                        cg.append_output(cg.create_load_closure(cell))
                    else:  # nested frame
                        assert cur_tx.post_prune_cell_and_freevars
                        cg(cur_tx.post_prune_cell_and_freevars[cell])
                cg.append_output(create_build_tuple(len(freevars)))
                cur_tx = cur_tx.parent
                tx_cnt += 1
            cg.append_output(create_instruction("BUILD_LIST", arg=tx_cnt))
        else:
            cg.append_output(create_instruction("BUILD_LIST", arg=0))

    def codegen_suffix(
        self,
        tx: "InstructionTranslatorBase",
        stack_values: list[VariableTracker],
        cg: PyCodegen,
    ) -> None:
        # NOTE: `codegen_save_tempvars` must run first to update `source` fields
        # for variables with `AttributeMutationNew`, as they don't implement
        # `reconstruct` themselves.
        self.side_effects.codegen_save_tempvars(cg)
        if self.backward_state:
            assert not self.export
            for name, val in self.backward_state.items():
                cg(val)
                assert self.backward_state_var is not None
                cg.append_output(cg.create_load(self.backward_state_var))
                cg.store_attr(name)
        if config.replay_side_effects:
            self.side_effects.codegen_hooks(cg)

        # TODO get debug_locals working for nested graph breaks
        # Return variables used for logging at the end
        for debug_var, args in tx.debug_locals:
            cg.add_push_null(lambda: cg(debug_var))
            for arg in args:
                cg(arg)
            cg.extend_output(create_call_function(len(args), False))
            cg.extend_output([create_instruction("POP_TOP")])

        # codegen cells before we apply side effects
        self.codegen_cells(tx, cg)

        cg.restore_stack(stack_values, value_from_source=not tx.export)
        if config.replay_side_effects:
            self.side_effects.codegen_update_mutated(cg)

    def cleanup_graph(self) -> None:
        """
        Remove "creation_timestamp" from node meta

        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        """
        assert self.should_exit
        nodes = list(self.graph.nodes)
        for node in nodes:
            node.meta.pop("creation_timestamp", None)

        grad_enabled = torch.is_grad_enabled()
        for node1, node2 in itertools.pairwise(nodes):
            if (
                node1.target is torch._C._set_grad_enabled
                and tuple(node1.args) == (not grad_enabled,)
                and not node1._erased
            ):
                grad_enabled = node1.args[0]
                if (
                    node2.target is torch._C._set_grad_enabled
                    and tuple(node2.args) == (not grad_enabled,)
                    and not node2._erased
                ):
                    grad_enabled = node2.args[0]
                    self.graph.erase_node(node1)
                    self.graph.erase_node(node2)

    def bypass_package(self, reason: str = "", **kwargs: Any) -> None:
        """
        Do not save this output graph to the CompilePackage
        """
        if not self.package:
            return
        if torch._dynamo.config.strict_precompile:
            raise torch._dynamo.exc.PackageError(
                "Detected a package bypass: %s", reason
            )
        log.warning("Detected a package bypass: %s", reason)
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "precompile_cache_bypass",
                "encoding": "json",
            },
            payload_fn=lambda: {
                # precede with underscore so it always appear first in JSON in tlparse
                "_reason": reason,
                **kwargs,
            },
        )
        self.package.bypass_current_entry()
        self.package = None

    def get_graph_sizes_structured(self) -> dict[str, list[Union[int, str]]]:
        ret: dict[str, list[Union[int, str]]] = {}
        for node in self.graph.nodes:
            example_value = node.meta.get("example_value", None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                ret[node.name] = [s if isinstance(s, int) else repr(s) for s in size]
        return ret

    def get_graph_sizes(self, name: str) -> str:
        graph_sizes_str = "TRACED GRAPH TENSOR SIZES\n"
        graph_sizes_str += f"===== {name} =====\n"
        for node in self.graph.nodes:
            example_value = node.meta.get("example_value", None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                graph_sizes_str += f"{node.name}: {tuple(size)}\n"
                concrete_size = []
                has_symint = False
                for sz in size:
                    if isinstance(sz, int):
                        concrete_size.append(sz)
                    elif isinstance(sz, torch.SymInt):
                        has_symint = True
                        concrete_size.append(sz.node.hint)
                    else:
                        break
                else:
                    if has_symint:
                        graph_sizes_str += (
                            f"{node.name} (concrete): {tuple(concrete_size)}\n"
                        )
        return graph_sizes_str

    @contextlib.contextmanager
    def restore_global_state(self) -> Any:
        """
        Momentarily restores the global state to what it was prior to tracing the current output
        """
        prior_global_state = self.tracing_context.global_context.copy_graphstate()
        current_global_state: dict[str, tuple[Any, bool]] = {}
        self.save_global_state(out=current_global_state)
        try:
            # Set to state prior to tracing the graph
            self.tracing_context.global_context.restore_graphstate(prior_global_state)
            yield
        finally:
            # Reset to state at the current time (e.g. before calling the user compiler)
            self.tracing_context.global_context.restore_graphstate(
                GlobalContextCheckpointState(current_global_state)
            )

    def run_compiler_collective(self) -> None:
        tx = self.root_tx
        assert tx is not None
        if (ds := tx.distributed_state) is not None and ds.all_states is None:
            compile_pg = ds.compile_pg

            log.info("compiler_collective %s", ds.local_state)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compiler_collective",
                    "encoding": "string",
                },
                payload_fn=lambda: ds.local_state.render(),
            )
            device_types = compile_pg._device_types
            assert len(device_types) == 1, (
                "Expect only one device type but got {}".format("+".join(device_types))
            )
            with (
                get_interface_for_device(device_types.pop()).device(  # type: ignore[attr-defined]
                    compile_pg.rank() % torch.accelerator.device_count()
                ),
                dynamo_timed("compiler_collective", log_pt2_compile_event=True),
            ):
                all_states: list[Any] = [None] * compile_pg.size()

                dist.all_gather_object(all_states, ds.local_state, group=compile_pg)

                ds.all_states = all_states
            # Clear speculation log, because are tracing may diverge due to
            # this information from the compiler collective
            tx.speculation_log.clear()
            raise exc.CompileCollectiveRestartAnalysis

    def compile_and_call_fx_graph(
        self,
        tx: "InstructionTranslatorBase",
        rv: list[VariableTracker],
        root: FakeRootModule,
    ) -> list[Instruction]:
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.

        Code is generated w.r.t. self.root_tx.
        tx is only used for preserving GraphModule metadata
        """
        with torch._guards.TracingContext.clear_frame():
            from .decorators import disable

            assert self.should_exit

            self.run_compiler_collective()
            if count_calls(self.graph) == 0 and len(rv) == 0:
                return []

            name = unique_id("__compiled_fn", with_uuid=True)

            assert isinstance(rv, list)
            assert isinstance(root, FakeRootModule)

            output_node = self.create_node(
                "output",
                "output",
                (self.current_tracer.create_arg(tuple(x.as_proxy() for x in rv)),),
                {},
            )
            sub_gms = self.dedup_pass()
            root.add_nn_modules(sub_gms)  # type: ignore[arg-type]

            self.current_tracer._maybe_preserve_original_meta(tx, output_node)
            if not config.do_not_emit_runtime_asserts:
                # There is a rare scenario where codegen_suffix adds a new entry
                # to self.nn_modules while `root` knows only about the
                # nn_modules at the time of its creation. This causes failures
                # while creating the graph module because self.graph and root
                # are out of sync. This only happens for `get_attr` nodes, so
                # here we clean up the get_attr nodes that are unused.
                for attr in dir(root):
                    subgraph = getattr(root, attr)
                    if isinstance(subgraph, fx.GraphModule):
                        insert_deferred_runtime_asserts(
                            subgraph,
                            self.shape_env,
                            name,
                            export=self.export,
                        )
                self.remove_unused_get_attr_nodes()
                insert_deferred_runtime_asserts(
                    fx.GraphModule(root, self.graph),
                    self.shape_env,
                    name,
                    export=self.export,
                )
            # NB: deferred runtime asserts can keep graphargs live, so make sure
            # those are inserted before pruning
            self.remove_unused_graphargs()
            ncalls = count_calls(self.graph)
            counters["stats"]["calls_captured"] += ncalls

            self.remove_tensorify_specialized_graphargs()

            # free a bit of memory
            self.real_value_cache.clear()

            gm = _make_graph_module(root, self.graph)

            # Saved tensors hooks are not used by the graph.
            # GraphModule by default only copies used in the graph submodules.
            # Copying them into the result graph manually.
            if self.saved_tensors_hooks_subgraph_names:
                for subgraph_name in self.saved_tensors_hooks_subgraph_names:
                    setattr(gm, subgraph_name, getattr(root, subgraph_name))

            for register_finalizer in self.register_finalizer_fns:
                register_finalizer(gm)

            if next(gm.parameters(), None) is not None:
                # If dynamo produces a graph with parameters, skip package stuff
                # Bypass output graph
                self.bypass_package(
                    "Graph contains named parameters: either inline_inbuilt_nn_modules=False or there are static addresses.",
                    inline_builtin_nn_modules=torch._dynamo.config.inline_inbuilt_nn_modules,
                    gm=gm.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    ),
                )

            if self.package is not None:
                gm._backend_id = name

            gm.compile_subgraph_reason = self.compile_subgraph_reason
            gm.meta["dynamo_flat_name_to_original_fqn"] = (
                self.dynamo_flat_name_to_original_fqn.copy()
            )
            gm.meta["dynamo_compile_id"] = self.dynamo_compile_id
            gm.meta["backend_id"] = name

            graph_code_log.debug(
                "%s",
                lazy_format_graph_code(
                    name, gm, include_stride=True, include_device=True, colored=True
                ),
            )
            torch._logging.trace_structured(
                "dynamo_output_graph",
                lambda: {"sizes": self.get_graph_sizes_structured()},
                payload_fn=lambda: gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            self.call_cleanup_hooks()
            old_fake_mode = self.tracing_context.fake_mode
            assert old_fake_mode is not None
            if not self.export:
                import torch._functorch.config as _config

                with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                    # TODO(voz): The way export uses gm, and fake tensors, is not supported with us resetting

                    # Why create a new FakeTensorMode?
                    #
                    # The reason this needs to be done is because when we do Dynamo tracing, fake
                    # tensors can have their metadata mutated. Thus, the fake tensor we allocated
                    # for any given tensor may no longer be valid for the beginning trace of the
                    # graph. Nor is it convenient to "clone" the input tensors before mutating them,
                    # since you have to preserve aliasing. So we just reconstruct the FakeTensorMode
                    # from scratch when we go to AOTAutograd. But the ShapeEnv must be preserved as
                    # Dynamo made decisions about what is dynamic or not / guards from the user code
                    # that is not in graph.
                    backend_fake_mode = torch._subclasses.FakeTensorMode(
                        shape_env=old_fake_mode.shape_env,
                    )
                # TODO(voz): Ostensibly, this should be scoped and
                # restore back to old_fake_mode, but doing so currently violates
                # a lot of fake_tensor ownership assumptions and runs afoul of detect_fake_mode
                self.tracing_context.fake_mode = backend_fake_mode

            with self.restore_global_state():
                compiled_fn = self.call_user_compiler(gm, self.example_inputs())

            from torch.fx._lazy_graph_module import _LazyGraphModule

            if isinstance(compiled_fn, _LazyGraphModule) or (
                isinstance(getattr(compiled_fn, "__self__", None), _LazyGraphModule)
                and compiled_fn.__name__ == "_lazy_forward"  # type: ignore[attr-defined]
            ):
                # Since dynamo will run the forward method for the GraphModule shortly
                # anyways, it does not hurt to do the real recompilation here if
                # this is a _LazyGraphModule. This makes it easier for dynamo to
                # optimize a _LazyGraphModule.

                lazy_gm = (
                    compiled_fn
                    if isinstance(compiled_fn, _LazyGraphModule)
                    else compiled_fn.__self__  # type: ignore[attr-defined]
                )

                _LazyGraphModule.force_recompile(lazy_gm)

                if not isinstance(compiled_fn, _LazyGraphModule):
                    # replace compiled_fn with the real forward method
                    compiled_fn = lazy_gm.forward

            if self.package is not None:
                self.package.add_backend_id(name, compiled_fn)

            compiled_fn = disable(
                compiled_fn, reason="do not trace Dynamo-compiled graph"
            )

            counters["stats"]["unique_graphs"] += 1
            assert old_fake_mode.shape_env is not None
            if specializations := old_fake_mode.shape_env.specializations:
                specialization_guards = []
                specialization_cache: dict[Specialization, Callable[[Any], Any]] = {}
                sources = [a.source for a in self.graphargs]
                for specialization in specializations:
                    source_index = sources.index(specialization.source)
                    check_fn_source = inspect.getsource(specialization.check_fn).strip()
                    # Required because the LABDA_GUARD API requires a root guard manager
                    unused_root_guard_manager = RootGuardManager()
                    check_fn = guards.LAMBDA_GUARD(  # type: ignore[attr-defined]
                        unused_root_guard_manager,
                        specialization.check_fn,
                        [check_fn_source],
                    )

                    log.debug(
                        "Compiling backend specialized graph with specialization=%s",
                        check_fn_source,
                    )

                    specialization_guards.append(
                        (
                            functools.partial(
                                lambda idx, args, check_fn=check_fn: check_fn(
                                    args[idx]
                                ),
                                source_index,
                            ),
                            specialization,
                        )
                    )

                @torch._dynamo.disable(reason="do not trace Dynamo-compiled graph")  # type: ignore[misc]
                def specialized_dispatch(*args: Any, **kwargs: Any) -> Any:
                    for check_fn, specialization in specialization_guards:
                        if check_fn(args):
                            if specialization in specialization_cache:
                                return specialization_cache[specialization](
                                    *args, **kwargs
                                )

                            with self.shape_env.patch_source_specialization(
                                specialization.source, specialization.check_fn
                            ):
                                # Modify gm so AOTAutogradCache key changes per specialization
                                gm.meta["specialization"] = specialization
                                example_inputs: list[Tensor] = list(args)
                                with tracing(self.tracing_context):
                                    specialization_cache[specialization] = (
                                        self.call_user_compiler(gm, example_inputs)
                                    )

                            return specialization_cache[specialization](*args, **kwargs)
                    return compiled_fn(*args, **kwargs)

                # This is safe because we pre-process name to be unique
                self.install_global_unsafe(name, specialized_dispatch)
            else:
                # This is safe because we pre-process name to be unique
                self.install_global_unsafe(name, compiled_fn)

            assert self.root_tx is not None
            cg = PyCodegen(self.root_tx)

            for idx, arg in enumerate(self.graphargs):
                self.export_metadata.graph_input_idx_to_local_source[idx] = arg.source

            cg.make_call_generated_code(name)
            return cg.get_instructions()

    @property
    def placeholders(self) -> list[fx.Node]:
        return self.graph.find_nodes(op="placeholder")

    @property
    def graphargs(self) -> list[GraphArg]:
        return [node.meta["grapharg"] for node in self.placeholders]

    def call_user_compiler(
        self, gm: fx.GraphModule, example_inputs: list[Tensor]
    ) -> CompiledFn:
        with dynamo_timed(
            "OutputGraph.call_user_compiler",
            phase_name="backend_compile",
            log_pt2_compile_event=True,
            log_waitcounter=True,
            waitcounter_name_override="compile_aot_autograd",
            dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
        ):
            return self._call_user_compiler(gm, example_inputs)

    def _call_user_compiler(
        self, gm: fx.GraphModule, example_inputs: list[Tensor]
    ) -> CompiledFn:
        assert self.compiler_fn is not None
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                tot += 1
            if node.op == "placeholder":
                placeholders.append(node)
        increment_op_count(tot)
        for pl in placeholders:
            if not hasattr(pl, "_dynamo_source"):
                arg = pl.meta["grapharg"]
                # TODO: Why isn't this stored in meta :think:
                # NOTE: can't move these into meta: https://github.com/pytorch/pytorch/issues/141640
                pl._dynamo_source = arg.source

        # NOTE: can't move these into meta: https://github.com/pytorch/pytorch/issues/141640
        gm._param_name_to_source = self.param_name_to_source  # type: ignore[assignment]
        gm._source_to_user_stacks = self.source_to_user_stacks  # 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 15 class(es): VariableTrackerCacheKey, AliasingInfo, MutationInfo, VariableTrackerCache, class, FakeRootModule, WrapperBackend, class, class, class, OutputGraphCommon, OutputGraph, DynamoTracerOutput, LazyProxy, SubgraphTracer

### Functions
This file defines 132 function(s): __init__, lookup, add, clone, clear, _step_logger, __post_init__, _get_gen_rand_values_fn, _gen_rand_values, __init__, __repr__, add_nn_modules, __init__, __call__, shape_env, guards, aotautograd_guards, dump_guards_state, get_builtins_dict, __init__, shape_env, bypass_package, __init__, mark_bytecode_tracing_start, mark_bytecode_tracing_stop, install_builtins_dict_in_fglobals, add_backward_state_hook, get_backward_state_proxy, init_ambient_guards, maybe_install_saved_tensors_hooks_subgraphs


## Key Components

The file contains 14374 words across 3873 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 166763 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
