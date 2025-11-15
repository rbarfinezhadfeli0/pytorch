# Documentation: builder.py

## File Metadata
- **Path**: `torch/_dynamo/variables/builder.py`
- **Size**: 169959 bytes
- **Lines**: 3941
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: ignore-errors

"""
This module contains classes and utilities for building variable trackers in Dynamo.
Variable trackers are used to convert Python values into symbolic representations
that can be traced and transformed during graph capture.

The key classes are:

- VariableBuilder: Handles source-tracked objects that need guards and proper
  reconstruction in the output graph. Used for inputs, module attributes, etc.

- SourcelessBuilder: Handles ephemeral objects created during tracing that don't
  need source tracking or guards. Used for temporary lists, intermediate values, etc.

Variable trackers enable Dynamo to track the flow of values through the program,
maintain guards for dynamic properties, and reconstruct values in the output graph.
The builders in this module handle converting Python values into appropriate
VariableTracker instances based on their type and usage context.
"""

import abc
import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import math
import operator
import random
import re
import sys
import traceback
import types
import weakref
from collections.abc import Callable, MutableMapping
from typing import Any, NamedTuple, Optional, TYPE_CHECKING, Union

import sympy

import torch
from torch import SymInt
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.graph_bytecode_inputs import (
    get_external_object_by_index,
    register_user_object,
)
from torch._dynamo.utils import (
    get_metrics_context,
    is_int_specialization_case,
    is_torch_sym,
    set_feature_use,
)
from torch._guards import TracingContext
from torch._higher_order_ops.flat_apply import flat_apply
from torch._higher_order_ops.torchbind import call_torchbind
from torch._library.opaque_object import is_opaque_type
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch._subclasses.meta_utils import is_sparse_any, safe_grad
from torch._utils_internal import justknobs_check
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental._dynamism import normalize_source_name
from torch.fx.experimental.sym_node import _DynamicScalar, DynamicInt
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    _nested_int_aware_sort,
    DimDynamic,
    RelaxedUnspecConstraint,
    StatefulSymbolicContext,
    SubclassSymbolicContext,
    SymbolicContext,
    SymIntSymbolicContext,
    TrackedFake,
)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.nn.utils._expanded_weights import ExpandedWeight
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    is_traceable_wrapper_subclass_type,
)
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils.weak import TensorWeakRef

from .. import config, graph_break_hints, mutation_guard, replay_record, trace_rules
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..pgo import (
    auto_dynamic,
    auto_unset,
    FrameStateSizeEntry,
    InferStride,
    process_automatic_dynamic,
)
from ..side_effects import SideEffects
from ..source import (
    AttrProxySource,
    AttrSource,
    CallMethodItemSource,
    ChainedSource,
    ConstDictKeySource,
    ConvertIntSource,
    DictGetItemSource,
    DictSubclassGetItemSource,
    DynamicScalarSource,
    FloatTensorSource,
    GetItemSource,
    GradSource,
    is_constant_source,
    is_from_closure_source,
    is_from_global_source,
    is_from_nonlocal_source,
    is_from_optimizer_source,
    is_from_unspecialized_nn_module_source,
    ListGetItemSource,
    LocalSource,
    NonSerializableSetGetItemSource,
    NumpyTensorSource,
    OptimizerSource,
    RandomValueSource,
    Source,
    SubclassAttrListSource,
    TupleIteratorGetItemSource,
    UnspecializedBuiltinNNModuleSource,
    UnspecializedNNModuleSource,
)
from ..utils import (
    _extract_tensor_dict,
    build_checkpoint_variable,
    build_invoke_subgraph_variable,
    clone_input,
    common_constant_types,
    dict_keys,
    get_fake_value,
    get_items_from_dict,
    get_locals_to_steal,
    get_static_address_type,
    is_frozen_dataclass,
    is_function,
    is_function_or_wrapper,
    is_invoke_subgraph,
    is_lru_cache_wrapped_function,
    is_namedtuple,
    is_parameter_freezing,
    is_typing,
    is_utils_checkpoint,
    is_wrapper_or_member_descriptor,
    istype,
    namedtuple_fields,
    odict_values,
    proxy_args_kwargs,
    range_iterator,
    set_example_value,
    tensor_always_has_static_shape,
    tuple_iterator,
    tuple_iterator_getitem,
    tuple_iterator_len,
    unwrap_with_attr_name_if_wrapper,
    wrap_fake_exception,
)
from .base import (
    AttributeMutationNew,
    typestr,
    ValueMutationExisting,
    ValueMutationNew,
    VariableTracker,
    VariableTrackerMeta,
)
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    AutocastModeVariable,
    DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable,
    NullContextVariable,
    PreserveVersionContextVariable,
)
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictKeySetVariable,
    FrozensetVariable,
    MappingProxyVariable,
    SetVariable,
)
from .distributed import (
    DeviceMeshVariable,
    PlacementClassVariable,
    PlacementVariable,
    ProcessGroupVariable,
    WorldMetaClassVariable,
)
from .functions import (
    BuiltinMethodVariable,
    CollectionsNamedTupleFunction,
    CollectiveFunctionRewriteVariable,
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    FunctoolsPartialVariable,
    FunctoolsWrapsVariable,
    SysFunctionVariable,
    TracebackVariable,
    TritonKernelVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperUserFunctionVariable,
)
from .higher_order_ops import (
    LocalMapWrappedHigherOrderVariable,
    TorchHigherOrderOperatorVariable,
)
from .iter import ItertoolsVariable
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    SizeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradEngineVariable,
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    ComptimeVariable,
    DebuggingVariable,
    DelayGraphBreakVariable,
    GetAttrVariable,
    GetSetDescriptorVariable,
    LambdaVariable,
    LoggingLoggerVariable,
    MethodWrapperVariable,
    NumpyDTypeVariable,
    NumpyTypeInfoVariable,
    NumpyVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    RegexPatternVariable,
    SavedTensorBox,
    TorchVersionVariable,
    TypingVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .script_object import TorchScriptObjectVariable
from .sdpa import SDPAParamsVariable
from .streams import EventVariable, StreamContextVariable, StreamVariable
from .tensor import (
    NumpyNdarrayVariable,
    supported_const_comparison_op_values,
    SymNodeVariable,
    TensorSubclassVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .torch import (
    DispatchKeySetVariable,
    FuncTorchInterpreterVariable,
    TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)
from .torch_function import (
    TensorWithTFOverrideVariable,
    torch_function_mode_stack_state_mgr,
    TorchFunctionModeVariable,
)
from .user_defined import (
    FrozenDataClassVariable,
    IntWrapperVariable,
    KeyedJaggedTensorVariable,
    MutableMappingVariable,
    SourcelessGraphModuleVariable,
    UserDefinedClassVariable,
    UserDefinedDictVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedListVariable,
    UserDefinedObjectVariable,
    UserDefinedSetVariable,
    UserDefinedTupleVariable,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


DimList = list


def safe_has_grad(t):
    with torch._logging.hide_warnings(torch._logging._internal.safe_grad_filter):
        return hasattr(t, "grad")


class _missing:
    pass


@dataclasses.dataclass
class GraphArg:
    source: Source
    # TODO: storing a SymInt here but not a FakeTensor is a pretty strange
    # thing to do.  Probably should have example (which stores an int) and
    # fake_example
    _example: Union[TensorWeakRef, torch.SymInt]
    # When True, this indicates that this GraphArg is a Python quantity (e.g.,
    # a float or int) which we pass to the FX graph as a Tensor.  This
    # controls how we codegen calls into the Dynamo graph: we will call
    # torch.as_tensor on the quantity before passing it in.
    #
    # Note that we typically do not pass dynamic integers as tensors, because
    # they will most frequently just be used for size computation.  But this
    # is a policy decision that we can change our mind on; in particular, when
    # an int comes from a random number generator (e.g., random.randint), we
    # DO pass it as a tensor.
    #
    # It's also worth noting that our current tracing rules for
    # pass_arg_as_tensor as subtly broken: we just pun the variable as a
    # 0d scalar Tensor and pray that the semantics are the same.  Which they
    # often are, but not necessarily.  ezyang(May 2024) plans to fix this
    # soon.
    pass_arg_as_tensor: bool
    fake_tensor: Optional[torch._subclasses.fake_tensor.FakeTensor]
    # UnspecializedPythonVariable often masquerades as a tensor.
    # We MUST NOT generate shape guard code
    # that actually tries to access tensor properties on these values.
    # is_tensor lets us tell if this graph arg actually is a tensor
    # or not.
    is_tensor: bool = True
    # Sometimes, the Tensor we pass to example is freshly allocated (smh).
    # Then we cannot only keep a weak reference to it.  This lets you
    # stash a strong reference too.
    example_strong_ref: Optional[torch.Tensor] = None

    def __setattr__(self, name, value):
        # Use object.__setattr__ to bypass Dynamo's STORE_ATTR interception.
        # This is needed because when PYTORCH_TEST_WITH_DYNAMO=1, even internal
        # GraphArg creation can be traced, and with replay_side_effects=False,
        # normal STORE_ATTR bytecode only records mutations without applying them.
        object.__setattr__(self, name, value)

    @property
    def example(self):
        if isinstance(self._example, TensorWeakRef):
            r = self._example()
            assert r is not None
            return r
        else:
            return self._example

    def __post_init__(self):
        if isinstance(self._example, torch.Tensor):
            self._example = TensorWeakRef(self._example)
            assert is_fake(self.fake_tensor)

    def reconstruct(self, codegen: "PyCodegen"):
        codegen(self.source)

    def erase(self):
        self._example = None
        self.example_strong_ref = None

    def __eq__(self, other):
        return self.source.name() == other.source.name()


class BackwardStateGraphArg(GraphArg):
    def __init__(self) -> None:
        super().__init__(
            source=None,
            _example=BackwardState(),
            pass_arg_as_tensor=False,
            fake_tensor=None,
            is_tensor=False,
        )

    def reconstruct(self, codegen: "PyCodegen"):
        assert codegen.tx.output.backward_state_var
        codegen.add_push_null(
            lambda: codegen.load_import_from(BackwardState.__module__, "BackwardState")
        )
        codegen.call_function(0, False)
        codegen.dup_top()
        codegen.store(codegen.tx.output.backward_state_var)


# All class-based iterators in itertools
# NOTE: use id() because some objects are not hashable, it will raise error during lookup
ITERTOOLS_TYPE_IDS: frozenset[int] = frozenset(
    id(member)
    for name, member in vars(itertools).items()
    if not name.startswith("_") and inspect.isclass(member)
)
# Will be updated later in substitute_in_graph in torch/_dynamo/polyfills/itertools.py
ITERTOOLS_POLYFILLED_TYPE_IDS: set[int] = set()

# Capture fn pointer at import time
# This is to guard against trying to mark the iterated tensors
# as static in case user overrides fn ptr
og_module_named_buffers_fn_ptr = torch.nn.Module.named_buffers
og_module_named_parameters_fn_ptr = torch.nn.Module.named_parameters


class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(
        self,
        tx,
        source: Source,
    ) -> None:
        assert source is not None, (
            "Consider SourcelessBuilder for ephemeral objects, usually objects created locally."
        )
        assert TracingContext.try_get() is not None, "Expected active TracingContext"
        super().__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        if value in self.tx.output.side_effects:
            side_effect_result = self.tx.output.side_effects[value]
            dup_guard = make_dupe_guard(self.source, side_effect_result.source)
            if dup_guard:
                self.install_guards(dup_guard)

            if isinstance(value, torch.nn.Module) and isinstance(
                side_effect_result, UnspecializedNNModuleVariable
            ):
                # This means that two nn module instances with different sources
                # have the same id. NN modules are somewhat special objects,
                # because we have to track their nn_module_stack for ease of
                # use. But if we don't do anything, we will just return the
                # older variable tracker with the older nn_module_stack. So,
                # lets return the old variable tracker but update its
                # nn_module_stack
                side_effect_result.set_nn_module_stack_source(self.source)
            return side_effect_result

        cached_vt = self.tx.output.variable_tracker_cache.lookup(value, self.source)
        if cached_vt:
            return cached_vt

        vt = self._wrap(value)

        if vt.source is None:
            vt.source = self.source

        def _is_deduplicable_sym_variable(value, vt):
            # Constants like 0, 1, 2, etc. can be unspecialized as SymNodeVariables sometimes, but we
            # should NOT track them. If we use a single SymNodeVariable instance to track them
            # across multiple uses, then guards created for one usage will incorrectly apply to
            # all other usages of that constant, leading to unnecessary recompilations.
            return (
                is_torch_sym(value) or isinstance(value, _DynamicScalar)
            ) and isinstance(vt, SymNodeVariable)

        if (
            (
                self._can_lift_attrs_to_inputs(vt)
                or _is_deduplicable_sym_variable(value, vt)
            )
            and value not in self.tx.output.side_effects
            and not is_wrapper_or_member_descriptor(value)
        ):
            vt = self.tx.output.side_effects.track_object_existing(value, vt)

        self.tx.output.variable_tracker_cache.add(value, self.source, vt)
        return vt

    def _can_lift_attrs_to_inputs(self, vt):
        return type(vt) in {
            TensorVariable,
            TensorWithTFOverrideVariable,
            UserDefinedObjectVariable,
            NumpyNdarrayVariable,
        }

    def get_source(self):
        return self.source

    def install_guards(self, *guards):
        source = self.get_source()
        try:
            tmp = [source.make_guard(guard) for guard in guards]
        except NotImplementedError:
            return None
        install_guard(*tmp, skip=1)
        return {}

    @classmethod
    def _type_dispatch(cls):
        return cls._type_dispatch_impl(config.trace_numpy)

    @classmethod
    @functools.cache
    def _type_dispatch_impl(cls, trace_numpy):
        # NB: Careful not to close over self to avoid ref cycle from lru_cache
        entries = [
            (
                (
                    torch.Tensor,
                    torch.nn.Parameter,
                    torch._subclasses.FakeTensor,
                    torch._subclasses.functional_tensor.FunctionalTensor,
                ),
                cls.wrap_tensor,
            ),
            (
                (tuple, list, odict_values, collections.deque, torch.Size),
                cls.wrap_listlike,
            ),
            (tuple_iterator, cls.wrap_tuple_iterator),
            (range_iterator, cls.wrap_range_iterator),
            ((slice, range), cls.wrap_slice_range),
            (tuple(common_constant_types), cls.wrap_literal),
            (re.Pattern, cls.wrap_regex_pattern),
            (weakref.ReferenceType, cls.wrap_weakref),
            (torch.utils.hooks.RemovableHandle, cls.wrap_removable_handle),
            (torch.jit.ScriptFunction, cls.wrap_jit_function),
            (types.MappingProxyType, cls.wrap_mapping_proxy),
        ]

        if trace_numpy and np:
            entries.append((np.ndarray, cls.wrap_numpy_ndarray))

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, tuple) else (ts,):
                assert t not in result
                result[t] = fn

        return result

    def wrap_regex_pattern(self, value: re.Pattern):
        # TODO(jansel): something like a REPR_MATCH might be more robust here
        self.install_guards(GuardBuilder.ID_MATCH)
        return RegexPatternVariable(value)

    def wrap_weakref(self, value: weakref.ReferenceType):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return WeakRefVariable.build(self.tx, value, source=self.source)

    def wrap_removable_handle(self, value):
        # This means that the removable handle was created in some other frame.
        # Our current infra requires the hook to be registered and removed in
        # the same frame. So graph break.
        # Related test - PYTORCH_TEST_WITH_DYNAMO=1 python test/test_autograd.py -k TestAutograd.test_hooks
        unimplemented(
            gb_type="Attempted to represent unregistered RemovableHandle",
            context="",
            explanation="Dynamo attempted to build a representation of a torch.utils.hooks.RemovableHandle, "
            "which is not supported. This happens because the RemovableHandle was created in another frame.",
            hints=[],
        )

    def wrap_jit_function(self, value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        return WrapperUserFunctionVariable(
            value, "_torchdynamo_inline", source=self.source
        )

    def wrap_mapping_proxy(self, value):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        # This might be suboptimal compared to dict guards. But mappingproxy is
        # not very common, so its ok to guard on all keys.
        self.install_guards(GuardBuilder.MAPPING_KEYS_CHECK)
        all_const = all(ConstantVariable.is_literal(k) for k in value)

        if not all_const:
            unimplemented(
                gb_type="non-const keys in mappingproxy",
                context=f"non-const keys: {[k for k in value.keys() if not ConstantVariable.is_literal(k)]}",  # noqa: SIM118
                explanation="Dynamo expects mappingproxy keys to be constants.",
                hints=[
                    "Ensure your mappingproxy keys are constants (e.g. int, float, strings)",
                ],
            )

        def build_key_value(k, v):
            key = ConstantVariable.create(k)
            source_key = k

            source_value = GetItemSource(self.get_source(), source_key)
            res_value = LazyVariableTracker.create(v, source_value)

            return key, res_value

        items = dict(build_key_value(k, v) for k, v in value.items())

        # Create a dict_vt to be used in the mapping proxy variable
        dict_vt = ConstDictVariable(items, source=None)
        result = MappingProxyVariable(dict_vt, source=self.source)
        return self.tx.output.side_effects.track_mutable(value, result)

    @classmethod
    @functools.cache
    def _id_dispatch(
        cls,
    ) -> dict[int, Callable[["VariableBuilder", Any], VariableTracker]]:
        from ..comptime import comptime

        entries = [
            (comptime, lambda self, value: ComptimeVariable()),
            (
                dataclasses.fields,
                lambda self, value: LambdaVariable(
                    _dataclasses_fields_lambda,
                    source=self.source,
                    **self.install_guards(GuardBuilder.CLOSURE_MATCH),
                ),
            ),
            (torch.__version__, lambda self, value: TorchVersionVariable()),
        ]

        result = {}
        for ts, fn in entries:
            for t in ts if isinstance(ts, (tuple, list)) else (ts,):
                assert t not in result
                result[id(t)] = fn

        return result

    def _wrap(self, value):
        # import here to avoid circular dependencies
        from torch.utils._triton import (
            has_triton,
            has_triton_experimental_host_tma,
            has_triton_tensor_descriptor_host_tma,
        )

        from ..decorators import (
            DynamoConfigPatchProxy,
            ErrorOnGraphBreakDecoratorContextManager,
        )

        if has_triton():
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        else:

            class JITFunction:
                pass

            class Autotuner:
                pass

        # default implementations, in case we don't have triton (or the wrong triton version)
        def create_1d_tma_descriptor():
            pass

        def create_2d_tma_descriptor():
            pass

        class TensorDescriptor:
            @staticmethod
            def from_tensor():
                pass

        if has_triton_experimental_host_tma():
            from triton.tools.experimental_descriptor import (  # noqa: F811
                create_1d_tma_descriptor,
                create_2d_tma_descriptor,
            )
        if has_triton_tensor_descriptor_host_tma():
            from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F811

        # Handle exact type() match
        type_dispatch = self._type_dispatch().get(type(value))
        if type_dispatch is not None:
            return type_dispatch(self, value)

        # Handle exact id() match
        id_dispatch = self._id_dispatch().get(id(value))
        if id_dispatch is not None:
            return id_dispatch(self, value)

        # Everything else (NB: order matters!)
        if (
            isinstance(value, torch.Tensor)
            and type(value)
            not in (
                # These torch-native subclasses have overly restrictive
                # `__torch_function__` which prevents Dynamo from reading their
                # tensor attributes like `is_nested` or calling methods like
                # `_is_view`.
                torch.nn.parameter.UninitializedBuffer,
                torch.nn.parameter.UninitializedParameter,
                ExpandedWeight,
            )
            and type(value) not in config.nontraceable_tensor_subclasses
        ):
            if (
                type(value).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
                or is_traceable_wrapper_subclass(value)
            ):
                return self.wrap_tensor(value)

        if is_namedtuple(value):
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)
            output = [
                LazyVariableTracker.create(
                    getattr(value, name),
                    source=AttrSource(self.source, name),
                )
                for name in namedtuple_fields(type(value))
            ]
            result = NamedTupleVariable(
                output, tuple_cls=type(value), source=self.source
            )
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif istype(value, (dict, collections.defaultdict, collections.OrderedDict)):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            all_const = all(ConstantVariable.is_literal(k) for k in value)

            # For all_const, we don't have to guard on anything yet. We guard on
            # keys lazily by adding a dict_getitem entry for each accessed key.
            # For cases where we need to guard on all keys, we lazily put guards
            # during the dict call_method (check dicts.py)
            if not all_const:
                # Guard on the key order
                # This is not ideal, i.e., there is no need to guard on the key
                # order. But we guard on the key order because of the complexity
                #
                # 1) For non-constant objects, we can't save the key in the
                # guard context because it can be memory heavy. We can add
                # weakrefs but this complicates the accesses.
                #
                # 2) For non-constant objects, we also have to guard on the keys
                # (like TENSOR_MATCH on tensor). We might also have guards on
                # the attributes of the keys (like tensor.grad). To make this
                # work in tree structure is complicated.
                #
                # So, instead we guard on the key order. While guarding on key
                # order, we just save the indices and use it to access keys and
                # values. Indices are cheap to save.
                self.tx.output.guard_on_key_order.add(self.source)

            # We need all the keys to be hashable. We do this within the
            # _HashableTracker class in dicts.py
            def build_key_value(i, k, v):
                base = self.get_source()
                if all_const:
                    key = ConstantVariable.create(k)
                    source_key = k
                else:
                    source_key = ConstDictKeySource(base, i)
                    key = LazyVariableTracker.create(k, source_key)
                source_value = DictGetItemSource(base, source_key)
                res_value = LazyVariableTracker.create(v, source_value)

                return key, res_value

            # Ensure that we call dict.keys and not value.keys (which can call
            # overridden keys method). In the C++ guards, we relied on
            # PyDict_Next to traverse the dictionary, which uses the internal
            # data structure and does not call the overridden keys method.
            result = dict(
                build_key_value(i, k, v)
                for i, (k, v) in enumerate(get_items_from_dict(value))
            )

            if istype(value, collections.defaultdict):
                factory_source = AttrSource(self.source, "default_factory")
                result = DefaultDictVariable(
                    result,
                    type(value),
                    default_factory=VariableBuilder(self.tx, factory_source)(
                        value.default_factory
                    ),
                    source=self.source,
                )
            else:
                result = ConstDictVariable(
                    result, user_cls=type(value), source=self.source
                )

            return self.tx.output.side_effects.track_mutable(value, result)
        elif isinstance(value, torch.nn.Module):
            return self.wrap_module(value)
        elif ConstantVariable.is_literal(value):  # non-atomic literals
            return self.wrap_literal(value)
        elif isinstance(value, torch.overrides.TorchFunctionMode):
            var = TorchFunctionModeVariable(value, source=self.source)
            self.tx.output.side_effects.track_object_existing(value, var)
            return var
        elif istype(value, set):
            if any(isinstance(x, torch.Tensor) for x in value):
                unimplemented(
                    gb_type="Attempted to wrap a set with tensors",
                    context="Python set containing torch.Tensor elements",
                    explanation=(
                        "Dynamo cannot trace sets of tensors. To get a stable ordering, "
                        "Dynamo needs to convert the set into a list and the order might not be "
                        "stable if the set contains tensors."
                    ),
                    hints=[
                        "Use a dictionary where the keys are tensors.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            # The list gives a ordering for the set items. The ordering is based
            # on the Python hash and it is not related to object ordering inside
            # the set object. The order being incorrect at runtime will lead to
            # a recompilation.
            L = list(value)
            items = [
                LazyVariableTracker.create(
                    v, source=NonSerializableSetGetItemSource(self.source, i)
                )
                for i, v in enumerate(L)
            ]
            result = SetVariable(items, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif istype(value, frozenset) and all(
            (
                # For DBR quantization, we could get a frozenset of torch funcs.
                (type(x) is types.BuiltinMethodType and x.__module__ == "torch")
                or
                # Another commonly used frozenset of types.
                x in torch.utils._pytree.BUILTIN_TYPES
            )
            for x in value
        ):
            # For the limited cases of frozenset here, we know the items won't
            # change across runs, so we can safely create sourceless VTs for
            # them and only guard on the frozenset id.
            # TODO support source for sets and remove the special logics here.
            items = [SourcelessBuilder.create(self.tx, v) for v in value]
            self.install_guards(GuardBuilder.ID_MATCH)
            return FrozensetVariable(items, source=self.source)
        elif isinstance(
            value, (enum.Enum, torch.DispatchKey, torch._C._functorch.TransformType)
        ):
            self.install_guards(GuardBuilder.ID_MATCH)
            return EnumVariable(value=value, source=self.source)
        elif DebuggingVariable.is_reorderable_logging_function(value):
            # Put this above builtin_callable so that print() can be handled
            # along with other builtin debugging functions
            self.install_guards(GuardBuilder.BUILTIN_MATCH)
            return DebuggingVariable(value, source=self.source)
        elif isinstance(value, logging.Logger):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return LoggingLoggerVariable(value, source=self.source)
        elif is_utils_checkpoint(value):
            return build_checkpoint_variable(source=self.source)
        elif is_invoke_subgraph(value):
            return build_invoke_subgraph_variable(source=self.source)
        elif LocalMapWrappedHigherOrderVariable.should_wrap_in_hop(value):
            return LocalMapWrappedHigherOrderVariable.build(source=self.source)
        elif isinstance(value, functools.partial):
            func_src = AttrSource(self.get_source(), "func")
            func_obj = VariableBuilder(self.tx, func_src)(value.func)

            args = []
            args_source = AttrSource(self.get_source(), "args")
            for i, arg in enumerate(value.args):
                args.append(
                    VariableBuilder(self.tx, GetItemSource(args_source, i))(arg)
                )

            keywords = {}
            keywords_source = AttrSource(self.get_source(), "keywords")
            for k, v in value.keywords.items():
                if not ConstantVariable.is_literal(k):
                    unimplemented(
                        gb_type="functools.partial() with non-literal keyword",
                        context=f"non-literal keyword: {k}",
                        explanation="functools.partial() expects literal/string keywords",
                        hints=[*graph_break_hints.USER_ERROR],
                    )
                keywords[k] = VariableBuilder(
                    self.tx, DictGetItemSource(keywords_source, k)
                )(v)

            install_guard(
                self.get_source().make_guard(GuardBuilder.TYPE_MATCH),
                keywords_source.make_guard(GuardBuilder.DICT_KEYS_MATCH),
                args_source.make_guard(GuardBuilder.SEQUENCE_LENGTH),
            )
            return FunctoolsPartialVariable(func_obj, args, keywords)
        elif is_typing(value):
            # typing.List, typing.Mapping, etc.
            self.install_guards(GuardBuilder.ID_MATCH)
            return TypingVariable(
                value,
                source=self.source,
            )
        elif np is not None and isinstance(value, np.generic):
            # numpy array scalars: convert to 0D arrays
            return self.wrap_numpy_ndarray(np.asarray(value))
        elif trace_rules.is_numpy(value):
            assert np
            if istype(value, types.MethodType):
                # Dont guard on cython functions as they dont change ids
                if inspect.isfunction(value.__func__):
                    install_guard(
                        AttrSource(self.source, "__func__").make_guard(
                            GuardBuilder.CLOSURE_MATCH
                        )
                    )
            elif inspect.isclass(value):
                self.install_guards(GuardBuilder.CLASS_MATCH)
            elif inspect.isfunction(value):
                self.install_guards(GuardBuilder.CLOSURE_MATCH)
            elif callable(value):
                self.install_guards(GuardBuilder.ID_MATCH)
            else:
                self.install_guards(GuardBuilder.TYPE_MATCH)
            return NumpyVariable(value, source=self.source)
        elif trace_rules.is_numpy_dtype(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return NumpyDTypeVariable(value, source=self.source)
        elif trace_rules.is_numpy_type_info(value):
            if isinstance(value, np.iinfo):
                self.install_guards(GuardBuilder.TYPE_MATCH)
                dt_source = AttrSource(self.source, "dtype")
                install_guard(dt_source.make_guard(GuardBuilder.ID_MATCH))
            else:
                self.install_guards(GuardBuilder.ID_MATCH)
            return NumpyTypeInfoVariable(value, source=self.source)
        # NB: These can't be put in type_dispatch, they have to run later
        elif CollectiveFunctionRewriteVariable.can_rewrite(value):
            self.install_guards(GuardBuilder.CLOSURE_MATCH)
            return CollectiveFunctionRewriteVariable.create(
                self.tx,
                value,
                source=self.source,
            )
        elif istype(value, torch.autograd.function.FunctionMeta):
            self.install_guards(GuardBuilder.CLASS_MATCH)
            return AutogradFunctionVariable(
                value,
                source=self.source,
            )
        elif isinstance(value, torch.autograd.function.FunctionCtx):
            actual_saved_tensors = None
            try:
                actual_saved_tensors = value.saved_tensors
            except RuntimeError:
                pass

            saved_tensors = []
            guards = [self.source.make_guard(GuardBuilder.TYPE_MATCH)]
            if isinstance(actual_saved_tensors, tuple):
                saved_tensors_source = AttrSource(self.source, "saved_tensors")
                guards.append(
                    saved_tensors_source.make_guard(GuardBuilder.SEQUENCE_LENGTH)
                )
                for i, v in enumerate(actual_saved_tensors):
                    saved_tensors.append(
                        VariableBuilder(
                            self.tx, GetItemSource(saved_tensors_source, i)
                        )(v)
                    )
            install_guard(*guards)

            return self.tx.output.side_effects.track_object_existing(
                value,
                AutogradFunctionContextVariable(
                    value,
                    source=self.source,
                    saved_tensors=SavedTensorBox(saved_tensors),
                ),
            )
        elif (
            isinstance(value, types.MethodType)
            and istype(
                getattr(value, "__self__", None), torch.autograd.function.FunctionMeta
            )
            and getattr(value, "__name__", "") == "apply"
            and value == getattr(value.__self__, "apply", None)
        ):
            # handle aliased autograd function `apply` calls
            install_guard(
                AttrSource(self.get_source(), "__func__").make_guard(
                    GuardBuilder.CLOSURE_MATCH
                )
            )
            return GetAttrVariable(
                AutogradFunctionVariable(
                    value.__self__, source=AttrSource(self.source, member="__self__")
                ),
                "apply",
            )
        elif isinstance(value, torch._C._ImperativeEngine):
            self.install_guards(GuardBuilder.ID_MATCH)
            return AutogradEngineVariable(value, source=self.source)
        elif (
            value
            is torch._dynamo.external_utils.FakeCompiledAutogradEngine._exec_final_callbacks_stub
        ):
            self.install_guards(GuardBuilder.CLOSURE_MATCH)
            return LambdaVariable(
                lambda: UserFunctionVariable(
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.exec_final_callbacks,
                ).call_function(
                    self.tx,
                    (self.tx.output.side_effects.get_ca_final_callbacks_var(),),
                    {},
                )
            )
        elif isinstance(value, DynamoConfigPatchProxy):
            return DynamoConfigPatchVariable(value.changes)
        elif isinstance(value, ErrorOnGraphBreakDecoratorContextManager):
            return ErrorOnGraphBreakVariable(value.error_on_graph_break)
        elif callable(value) and trace_rules.lookup_callable(value) is not None:
            if trace_rules.is_callable_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return trace_rules.lookup_callable(value).create_with_source(
                value, source=self.source
            )
        elif np and isinstance(value, np.number):
            return self.wrap_unspecialized_primitive(value)
        elif isinstance(value, HigherOrderOperator):
            if value is torch._higher_order_ops.invoke_subgraph:
                unimplemented(
                    gb_type="Attempted to wrap torch._higher_order_ops.invoke_subgraph",
                    context="",
                    explanation="Directly using invoke_subgraph is not supported. Use nested_compile_region",
                    hints=[],
                )
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return TorchHigherOrderOperatorVariable.make(value, source=self.source)
        elif isinstance(value, torch.cuda.StreamContext):
            self.install_guards(GuardBuilder.ID_MATCH)
            stream_source = AttrSource(self.source, "stream")
            stream_var = VariableBuilder(self.tx, stream_source)(value.stream)
            return StreamContextVariable.create(self.tx, stream_var)
        elif isinstance(value, torch.Stream):
            # This refers to the device-agnostic torch.Stream
            self.install_guards(GuardBuilder.TYPE_MATCH)
            index = register_user_object(value, self.source)
            stream_proxy = self.tx.output.create_proxy(
                "call_function", get_external_object_by_index, (index,), {}
            )
            set_example_value(stream_proxy.node, value)
            var = StreamVariable(
                stream_proxy, value, source=self.source, user_object_index=index
            )
            return self.tx.output.side_effects.track_object_existing(value, var)
        elif isinstance(value, (torch._C._SDPAParams)):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return SDPAParamsVariable.create(self.tx, value, self.source)
        elif isinstance(value, torch._functorch.pyfunctorch.FuncTorchInterpreter):
            self.install_guards(GuardBuilder.ID_MATCH)
            return FuncTorchInterpreterVariable(value)
        elif isinstance(value, torch.Event):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            index = register_user_object(value, self.source)
            event_proxy = self.tx.output.create_proxy(
                "call_function",
                get_external_object_by_index,
                (index,),
                {},
            )
            set_example_value(event_proxy.node, value)
            return EventVariable(
                event_proxy,
                value,
                index,
                source=self.source,
            )
        elif (
            istype(value, contextlib.nullcontext)
            and inspect.getattr_static(value, "enter_result", None) is None
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return NullContextVariable(source=self.source)
        elif KeyedJaggedTensorVariable.is_matching_object(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = KeyedJaggedTensorVariable(value, source=self.source)
            # TODO: this doing it manually is bad
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, torch.optim.Optimizer):
            self.install_guards(GuardBuilder.ID_MATCH)
            self.source = OptimizerSource(self.source)
            return OptimizerVariable(value, source=self.source)
        elif isinstance(value, torch.DispatchKeySet):
            self.install_guards(GuardBuilder.DISPATCH_KEY_SET_MATCH)
            return DispatchKeySetVariable(value)
        elif WorldMetaClassVariable.is_group_member_type(value):
            return WorldMetaClassVariable(value, source=self.source)
        elif ProcessGroupVariable.is_process_group(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return ProcessGroupVariable(value, source=self.source)
        elif DeviceMeshVariable.is_device_mesh(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.EQUALS_MATCH)
            return DeviceMeshVariable(value, source=self.source)
        elif PlacementClassVariable.is_placement_type(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.ID_MATCH)
            return PlacementClassVariable(value, source=self.source)
        elif PlacementVariable.is_placement(value):
            # TODO: see if we need to add custom guard instead of a simple ID_MATCH
            self.install_guards(GuardBuilder.EQUALS_MATCH)
            return PlacementVariable(
                value,
                source=self.source,
            )
        elif (
            id(value) in ITERTOOLS_TYPE_IDS
            and id(value) not in ITERTOOLS_POLYFILLED_TYPE_IDS
        ):
            self.install_guards(GuardBuilder.CLASS_MATCH)
            return ItertoolsVariable(value, source=self.source)
        elif isinstance(value, _DynamicScalar):
            is_int = isinstance(value, DynamicInt)
            source = DynamicScalarSource(self.source, is_int)
            if id(value) in self.tx.output.root_tracer.dynamic_scalar_nodes:
                # If we've already seen this dynamic scalar, reuse the existing
                # SymInt/SymFloat node.
                node = self.tx.output.root_tracer.dynamic_scalar_nodes[id(value)]
            else:
                sym = self.tx.output.shape_env.create_unspecified_symbol(
                    value.real,
                    source=source,
                    dynamic_dim=DimDynamic.DYNAMIC,
                )
                node = self.tx.output.shape_env.create_symintnode(
                    sym,
                    hint=value.real,
                    source=source,
                )

            # Bind to graph input
            sym_node_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(node),
                node,
                source=source,
            )
            sym_node_proxy.node.meta["grapharg"] = GraphArg(
                source,
                node,
                False,
                None,
                is_tensor=False,
                example_strong_ref=node,
            )
            sym_expr = node.node.expr
            assert isinstance(sym_expr, sympy.Symbol), (
                f"{sym_expr} is not a basic Symbol."
            )
            self.tx.output.tracked_fakes.append(TrackedFake(node, source, None))
            return SymNodeVariable.create(self.tx, sym_node_proxy, node)
        elif is_torch_sym(value):
            # Note: this doesn't handle nested symints.
            # For SymBool input, we reuse the infra for SymInt by simulating SymBool with a SymInt in dynamo.

            # Concretely,
            # 1. We create a SymInt in dynamo's shape_env, whose source is constructed as ConvertIntSource(self.source).
            # so that guards on the SymInts can be effectively applied on the original SymBool in user program.
            # 2. We create a SymBool based on the SymInt in dynamo's ShapeEnv. Because the original user program
            # depends on the value being a SymBool. This allows dynamo to interpret the user's program correctly.
            source = (
                self.source
                if isinstance(value, torch.SymInt)
                else ConvertIntSource(self.source)
            )
            if value.node.has_hint():
                new_symint = (
                    self.tx.output.shape_env.create_unspecified_symint_and_symbol(
                        int(value.node.hint),
                        source,
                        dynamic_dim=DimDynamic.DYNAMIC,
                    )
                )
            else:
                if isinstance(value, torch.SymBool):
                    # We need to create an unbacked symint to replace the unbacked symbool.
                    new_symint = self.tx.output.shape_env.create_unbacked_symint()
                else:
                    # TODO (yidi): we need to figure out a way to propagate the guards
                    # we accumulated when tracing the subggraph to outer shape_env. For normal symints,
                    # this is automatically done by evaluating the guards once but this
                    # will cause data-dependent error when we evaluate the outer unbacked symints.
                    # The test case that triggers this graph break is test_cond_unbacked_symint_closure
                    unimplemented(
                        gb_type="Attempted to wrap unbacked SymInt",
                        context="",
                        explanation="Unbacked SymInt input is not supported yet.",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

            sym_node_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(new_symint),
                new_symint,
                source=source,
            )

            sym_node_proxy.node.meta["grapharg"] = GraphArg(
                source,
                new_symint,
                False,
                None,
                is_tensor=False,
                example_strong_ref=new_symint,
            )
            # We bind the new_symint to graph input.
            sym_expr = new_symint.node.expr
            assert isinstance(sym_expr, sympy.Symbol), (
                f"{sym_expr} is not a basic Symbol."
            )
            self.tx.output.tracked_fakes.append(TrackedFake(new_symint, source, None))

            tracing_symint = (
                new_symint if isinstance(value, torch.SymInt) else new_symint == 1
            )  # cast it back to symbool for tracing
            return SymNodeVariable(sym_node_proxy, tracing_symint)

        elif isinstance(value, (JITFunction, Autotuner)):
            self.install_guards(GuardBuilder.ID_MATCH)
            return TritonKernelVariable(
                value,
                None,  # No kernel idx provided
                None,  # No grid provided
                source=self.source,
            )
        elif value is create_1d_tma_descriptor:
            return CreateTMADescriptorExperimentalVariable(rank=1)
        elif value is create_2d_tma_descriptor:
            return CreateTMADescriptorExperimentalVariable(rank=2)
        elif value is TensorDescriptor.from_tensor:
            return CreateTMADescriptorStableVariable()
        elif isinstance(value, torch.amp.autocast_mode.autocast):
            self.install_guards(GuardBuilder.ID_MATCH)
            return AutocastModeVariable(
                target_values=[
                    value.device,
                    value.fast_dtype,
                    value._enabled,
                    value._cache_enabled,
                ],
                source=self.source,
            )
        elif TorchCtxManagerClassVariable.is_matching_cls(value):
            if inspect.isclass(value):
                self.install_guards(GuardBuilder.CLASS_MATCH)
            elif inspect.isfunction(value):
                self.install_guards(GuardBuilder.CLOSURE_MATCH)
            return TorchCtxManagerClassVariable(value, source=self.source)
        elif inspect.getattr_static(value, "__script_if_tracing_wrapper", False):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(
                value, "__original_fn", source=self.source
            )
        elif is_lru_cache_wrapped_function(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(value, "__wrapped__", source=self.source)
        elif value is traceback.clear_frames:
            return TracebackVariable(source=self.source)
        elif value is sys.exc_info or (
            sys.version_info >= (3, 11) and value is sys.exception
        ):
            return SysFunctionVariable(value, source=self.source)
        elif is_function_or_wrapper(value) and inspect.getattr_static(
            value, "_torchdynamo_inline", False
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return WrapperUserFunctionVariable(
                value, "_torchdynamo_inline", source=self.source
            )
        elif value is functools.wraps:
            self.install_guards(GuardBuilder.ID_MATCH)
            return FunctoolsWrapsVariable(value, source=self.source)
        elif value is collections.namedtuple:
            self.install_guards(GuardBuilder.ID_MATCH)
            return CollectionsNamedTupleFunction(value, source=self.source)
        elif isinstance(
            value, types.BuiltinMethodType
        ) and BuiltinMethodVariable.is_supported_builtin_method(value):
            self.install_guards(GuardBuilder.ID_MATCH)
            return BuiltinMethodVariable(value, source=self.source)
        elif is_function(value) and value in (float.fromhex, float.hex):
            self.install_guards(GuardBuilder.ID_MATCH)
            return GetAttrVariable(
                BuiltinVariable(float, source=self.source),
                value.__name__,
            )
        elif is_function_or_wrapper(value):
            value, attr_name = unwrap_with_attr_name_if_wrapper(value)
            # For these wrappers, Dynamo points to the wrapped function,
            # so source needs to be updated as well.
            if attr_name is not None:
                self.source = AttrSource(self.source, attr_name)
            return trace_rules.lookup(value).create_with_source(
                value, source=self.source
            )
        elif value is random.Random:
            self.install_guards(GuardBuilder.ID_MATCH)
            return RandomClassVariable(source=self.source)
        elif istype(value, random.Random) and RandomVariable.is_supported_random_obj(
            value
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = RandomVariable(value, source=self.source)
            self.tx.output.side_effects.track_mutable(value, result)
            return result
        # Don't use istype, since some python modules are not subclasses of types.ModuleType directly.
        # E.g, type(torch.ops) -> <class 'torch._ops._Ops'>,
        # type(torch.backends.cudnn) -> <class 'torch.backends.cudnn.CudnnModule'>
        elif isinstance(value, (types.ModuleType, replay_record.DummyModule)):
            self.install_guards(GuardBuilder.MODULE_MATCH)
            result = PythonModuleVariable(
                value,
                source=self.source,
            )
            self.tx.output.side_effects.track_object_existing(value, result)
            return result
        elif isinstance(value, types.MethodType) and isinstance(
            value.__self__, (torch.nn.Module, torch.utils._pytree.TreeSpec)
        ):
            # don't let MethodTypes fall through to UserDefinedObject,
            # which doesn't support 'CALL_FUNCTION'

            # TODO(whc): Why do we limit this to methods on NNModules?
            # I don't have a good reason for this, but it preserves the existing behavior
            # for MBartForConditionalGeneration, which generates many graph breaks and OOMs otherwise.
            # I suspect we probably want to relax this check and dig deeper there.

            # In order to construct a MethodVariable in Dynamo, we start with an actual method obj from python,
            # but need to separately wrap its underlying `__func__` and its `self` argument.  We wrap `self` here
            # and then `__func__` gets wrapped inside UserMethodVariable.
            self_obj = VariableBuilder(
                self.tx, source=AttrSource(self.source, "__self__")
            )(value.__self__)
            assert self_obj and isinstance(self_obj, VariableTracker), (
                "Failed to produce a valid self obj"
            )
            return UserMethodVariable(
                value.__func__,
                self_obj,
                source=self.source,
            )
        elif isinstance(value, types.GetSetDescriptorType):
            # GetSet descriptors are C functions attached to an attribute lookup
            # using PyGetSetDef. Python, on attribute lookup, can decide to
            # create a new object on the fly, and therefore the `id` of the
            # descriptors is not guaranteed to be same for different attribute
            # accesses. Since these are unlikely to change during the program
            # execution, we can skip guarding on them.
            return GetSetDescriptorVariable(value)
        elif isinstance(value, types.MethodWrapperType):
            # Method-wrappers are written in C, and they are not guaranteed to
            # return the same object on attribute lookup. Therefore, we cannot
            # insert a ID_MATCH guard here. method-wrappers are very
            # unlikely to change, so its ok to skip the guard here.
            return MethodWrapperVariable(value)
        elif issubclass(type(value), type) and issubclass(value, BaseException):
            # match user defined exceptions
            self.install_guards(GuardBuilder.ID_MATCH)
            return UserDefinedExceptionClassVariable(value)
        elif issubclass(type(value), type):
            if value in (
                torch.utils.hooks.BackwardHook,
                torch.nn.Parameter,
                torch.nn.Buffer,
            ):
                # TODO(jansel): combine this case with the one above
                return trace_rules.lookup(value).create_with_source(
                    value, source=self.source
                )
            if value is torch.autograd._unsafe_preserve_version_counter:
                self.install_guards(GuardBuilder.CLASS_MATCH)
                return PreserveVersionContextVariable.constructor(self.tx)
            if (
                # `value` must be a strict subclass of `torch.Tensor`
                issubclass(value, torch.Tensor)
                and value is not torch.Tensor
                # `TensorSubclassVariable` is not for subclass that overrides
                # `torch_dispatch`.
                and value.__torch_dispatch__ is torch.Tensor.__torch_dispatch__
                # `TensorSubclassVariable` would lead to construction of
                # `TensorWithTFOverrideVariable`, but we don't want that for
                # traceable wrapper subclasses (we wrap those subclass instances
                # into `TensorVariable`).
                and not is_traceable_wrapper_subclass_type(value)
            ):
                return TensorSubclassVariable(value, source=self.source)

            if not is_from_closure_source(self.source):
                # For closure source, the variable comes from LOAD_SUPER_ATTR,
                # which calls self.__class__. This is internal Cpython
                # implementation, and it is rare for the user to modify
                # self.__class__ manually.
                # For other cases, this is a userdefined class, so install an
                # ID_MATCH even if its a global variable.
                self.install_guards(GuardBuilder.CLASS_MATCH)

            return UserDefinedClassVariable(
                value,
                source=self.source,
            )
        elif TorchScriptObjectVariable.is_matching_cls(type(value)):
            from ..source import (
                FlattenScriptObjectSource,
                ScriptObjectQualifiedNameSource,
            )

            if torch._library.fake_class_registry.tracing_with_real(value):
                proxy = self.tx.output.root_tracer.create_graph_input(
                    re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                    type(value),
                    value,
                    source=self.source,
                )

                # setting is_unspecialized=False to not insert a as_tensor call in reconstruct by default
                # setting example to be real value because these example values will be used
                # as example_inputs for user compiler.
                proxy.node.meta["grapharg"] = GraphArg(
                    self.source, value, False, None, False, value
                )
                return TorchScriptObjectVariable.create(
                    proxy,
                    value,
                    source=self.source,
                )

            if is_opaque_type(type(value)):
                self.install_guards(GuardBuilder.TYPE_MATCH)

            elif not hasattr(value, "__obj_flatten__"):
                # This exists to allow a smoother transition.
                # The implications are:
                # The script objects won't be tracked as proxies.
                # Methods on these objects won't show up in the graph.
                # The original script object might be mutated.
                return self.wrap_user_defined(value)
            else:
                # Install the guards on the fully qualified name of the script object
                LazyVariableTracker.realize_all(
                    VariableBuilder(
                        self.tx, ScriptObjectQualifiedNameSource(self.source)
                    )(
                        value._type().qualified_name()  # type: ignore[attr-defined]
                    )
                )
                # Install the guards on the content of the script object by setting the source
                # to be FlattenScriptObjectSource, which calls __obj_flatten__() to get the contents.
                LazyVariableTracker.realize_all(
                    VariableBuilder(self.tx, FlattenScriptObjectSource(self.source))(
                        value.__obj_flatten__()
                    )
                )

            fake_script_obj = torch._library.fake_class_registry.maybe_to_fake_obj(
                self.tx.output.fake_mode, value
            )

            proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(value),
                fake_script_obj,
                source=self.source,
            )

            # setting is_unspecialized=False to not insert a as_tensor call in reconstruct by default
            # setting example to be real value because these example values will be used
            # as example_inputs for user compiler.
            proxy.node.meta["grapharg"] = GraphArg(
                self.source, value, False, None, False, fake_script_obj
            )
            return TorchScriptObjectVariable.create(
                proxy,
                fake_script_obj,
                source=self.source,
            )
        elif (
            isinstance(value, (dict, collections.OrderedDict))
            and type(value).__new__ is dict.__new__
        ):
            # Construct a dict_vt that will reside inside the UserDefinedDictVariable
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            # Guard on the key order
            self.tx.output.guard_on_key_order.add(self.source)

            # We need all the keys to be hashable. We do this within the
            # _HashableTracker class in dicts.py
            def build_key_value(i, k, v):
                base = self.get_source()
                source_key = ConstDictKeySource(base, i)
                key = LazyVariableTracker.create(k, source_key)

                source_value = DictSubclassGetItemSource(base, source_key)
                res_value = LazyVariableTracker.create(v, source_value)

                return key, res_value

            # Ensure that we call dict.keys and not value.keys (which can call
            # overridden keys method). In the C++ guards, we relied on
            # PyDict_Next to traverse the dictionary, which uses the internal
            # data structure and does not call the overridden keys method.
            result = dict(
                build_key_value(i, k, v)
                for i, (k, v) in enumerate(get_items_from_dict(value))
            )

            dict_vt = ConstDictVariable(
                result,
                user_cls=(
                    collections.OrderedDict
                    if isinstance(value, collections.OrderedDict)
                    else dict
                ),
                mutation_type=ValueMutationExisting(),
                source=self.source,
            )
            # Force this to reconstruct on mutation to keep the reconstruction
            # bytecode simple
            dict_vt.should_reconstruct_all = True

            result = UserDefinedDictVariable(value, dict_vt=dict_vt, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, tuple):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            # NB - Be careful in not triggering user code. Guards also work on
            # the underlying tuple data structure.
            output = [
                LazyVariableTracker.create(
                    tuple.__getitem__(value, i),
                    source=GetItemSource(self.get_source(), i),
                )
                for i in range(tuple.__len__(value))
            ]

            tuple_vt = TupleVariable(
                output, source=self.source, mutation_type=ValueMutationExisting()
            )
            result = UserDefinedTupleVariable(
                value, tuple_vt=tuple_vt, source=self.source
            )
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, list):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            # NB - Be careful in not triggering user code. Guards also work on
            # the underlying list data structure.
            output = [
                LazyVariableTracker.create(
                    list.__getitem__(value, i),
                    source=ListGetItemSource(self.get_source(), i),
                )
                for i in range(list.__len__(value))
            ]
            list_vt = ListVariable(
                output, source=self.source, mutation_type=ValueMutationExisting()
            )
            result = UserDefinedListVariable(value, list_vt=list_vt, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, (set, frozenset)):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

            L = list(dict.fromkeys(value))
            output = [
                LazyVariableTracker.create(
                    list.__getitem__(L, i),
                    source=NonSerializableSetGetItemSource(self.get_source(), i),
                )
                for i in range(list.__len__(L))
            ]
            set_vt_cls = SetVariable if isinstance(value, set) else FrozensetVariable
            set_vt = set_vt_cls(
                output, source=self.source, mutation_type=ValueMutationExisting()
            )
            result = UserDefinedSetVariable(value, set_vt=set_vt, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif issubclass(type(value), MutableMapping):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = MutableMappingVariable(value, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif is_frozen_dataclass(value):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = FrozenDataClassVariable.create(self.tx, value, source=self.source)
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif isinstance(value, dict_keys):
            if all(ConstantVariable.is_literal(k) for k in value):
                # If the dict_keys object is passed from outside the compile region, it must either be passed along with
                # the corresponding dict object or treated as a set (when only the keys are passed into the compiled region).
                # - If it is passed along with the dict, the dict object itself is already guarded.
                # - If only the dict_keys object is passed, we add EQUALS_MATCH and SEQUENCE_LENGTH guards
                #   to ensure it remains unchanged across multiple runs.
                items = [SourcelessBuilder.create(self.tx, v) for v in value]
                install_guard(
                    self.get_source().make_guard(GuardBuilder.SEQUENCE_LENGTH),
                    self.get_source().make_guard(GuardBuilder.EQUALS_MATCH),
                )
                return DictKeySetVariable(items, source=self.source)
            else:
                unimplemented(
                    gb_type="non-const keys in dict_keys",
                    context=f"non-const keys: {[k for k in value if not ConstantVariable.is_literal(k)]}",
                    explanation="Dynamo expects dict_keys keys to be constants.",
                    hints=[
                        "Ensure your dict_keys keys are constants (e.g. int, float, strings)",
                    ],
                )
        elif IntWrapperVariable.is_matching_object(value):
            from torch.export.dynamic_shapes import _DimHintType

            if value.dynamism is None or value.dynamism.type == _DimHintType.STATIC:
                return self.wrap_symint(value.val)
            elif value.dynamism.type == _DimHintType.DYNAMIC:
                log.debug(
                    "%s marked %s via IntWrapper",
                    self.source.name(),
                    DimDynamic.DYNAMIC,
                )
                return self.wrap_symint(
                    value.val,
                    dynamism=DimDynamic.DYNAMIC,
                    context=SymIntSymbolicContext(
                        constraint=RelaxedUnspecConstraint(warn_only=False)
                    ),
                )
            elif value.dynamism.type == _DimHintType.AUTO:
                log.debug(
                    "%s marked %s via IntWrapper",
                    self.source.name(),
                    DimDynamic.DYNAMIC,
                )
                return self.wrap_symint(value.val, dynamism=DimDynamic.DYNAMIC)
            else:
                raise RuntimeError(f"Undefined dynamism {value.dynamism}")
        else:
            return self.wrap_user_defined(value)

    def wrap_user_defined(self, value: Any):
        self.install_guards(GuardBuilder.TYPE_MATCH)
        result = UserDefinedObjectVariable(value, source=self.source)
        if not SideEffects.cls_supports_mutation_side_effects(type(value)):
            # don't allow STORE_ATTR mutation with custom __setattr__
            return result
        return self.tx.output.side_effects.track_object_existing(value, result)

    def wrap_listlike(self, value: Union[tuple, list, odict_values, NamedTuple]):
        for item in value:
            if item is value:
                unimplemented(
                    gb_type="list elements are pointing to the list itself",
                    context="",
                    explanation="Dynamo does not support lists whose items reference to itself",
                    hints=["Avoid using self referential list"],
                )

        if config.specialize_int and type(value) is torch.Size:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return ConstantVariable.create(value=value)

        # One can index a tensor with a list/tuple. Therefore, we need to
        # have a stricter match.
        self.install_guards(GuardBuilder.SEQUENCE_LENGTH)

        # Tuples are immutable objects, so we should mark its items static. This
        # avoids wrapping of tuple items as symints. This helps for nn module
        # attributes like conv2d strides, dilations.
        if (
            istype(value, tuple)
            and all(ConstantVariable.is_literal(item) for item in value)
            and self.source.guard_source().is_unspecialized_nn_module()
        ):
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            return TupleVariable([ConstantVariable.create(item) for item in value])

        output = [
            LazyVariableTracker.create(
                item,
                source=GetItemSource(self.get_source(), i),
            )
            for i, item in enumerate(value)
        ]

        maybe_gm = self.tx.output.local_scope.get("self")
        if isinstance(
            self.source, LocalSource
        ) and self.source.local_name in get_locals_to_steal(maybe_gm):
            # The input tensor list to dynamo from compiled autograd may contain activations
            # which are freed as they are used in inductor. Dynamo's default behavior is to
            # lift all tensors to the graph inputs, but this will cause dynamo to hold an
            # extra reference to the activation tensors and increase peak memory usage.
            # To allow freeing ASAP, we keep the list as graph argument to the dynamo output
            # graph, and unpack it locally.
            # e.g. instead of `def forward(self, L_inputs_0_, L_inputs_1_, ...):`, we have
            # `def forward(self, L_inputs_):`
            source = self.source
            assert isinstance(value, list)
            tensor_list_proxy = self.tx.output.root_tracer.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                type(value),
                value,
                source=source,
            )
            tensor_list_proxy.node.meta["steal_arg"] = True

            list_variable = wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=self.tx,
                proxy=tensor_list_proxy,
                example_value=value,
                subclass_type=None,
                source=source,
            )

            # Apply relevant logic from `VariableTracker.build(value[i])`
            # (except for the `create_graph_input` stuff).
            guards = []
            for i, tensor_variable in enumerate(list_variable.items):
                source_i = GetItemSource(base=source, index=i, index_is_slice=False)
                # access unpacked tensor from this list instead of from a lifted arg
                self.tx.output.input_source_to_var[source_i] = tensor_variable
                tensor_variable.proxy.node.meta["tensor_dict"] = _extract_tensor_dict(
                    value[i]
                )
                guard = functools.partial(
                    GuardBuilder.TENSOR_MATCH, value=TensorWeakRef(value[i])
                )
                guards.append(source_i.make_guard(guard))

            install_guard(*guards, skip=1)

            grapharg = GraphArg(
                source,
                value,
                pass_arg_as_tensor=False,
                fake_tensor=None,
                is_tensor=False,
            )
            tensor_list_proxy.node.meta["grapharg"] = grapharg

            # The following is very important for maintaining the "python object
            # <==> variable tracker" 1-to-1 mapping, which is mainly handled via
            # `side_effects`. Note that constructing `tensor_variable` above
            # already adds it to graph arg, but we never registered it with
            # `side_effects`. The preemptive `realize` calls here basically
            # does that registration (at the end of `self.__call__`).
            #
            # A slightly cleaner alternative is to register the
            # `tensor_variable`s above with `side_effects` directly, and just
            # return the `list_variable`, but that breaks some tensor-subclass
            # related tests like `test_inputs_aliasing_bytecode_stack_restore`,
            # because `tensor_variable` is constructed via
            # `handle_traced_output`, which doesn't really expect/handle tensor
            # subclass.
            #
            # Eventually, we expect to fix remove all of these by having Dynamo
            # auto-boxing inputs to the compiled graph, see
            # https://github.com/pytorch/pytorch/issues/153701.
            for vt in output:
                vt.realize()

        result = BaseListVariable.cls_for_instance(value)(output, source=self.source)
        if istype(value, (list, collections.deque)):
            return self.tx.output.side_effects.track_mutable(value, result)
        return result

    def wrap_tuple_iterator(self, value: tuple_iterator):
        self.install_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
        output = [
            VariableBuilder(self.tx, TupleIteratorGetItemSource(self.get_source(), i))(
                tuple_iterator_getitem(value, i)
            )
            for i in range(tuple_iterator_len(value))
        ]
        result = TupleIteratorVariable(output, source=self.source)
        return self.tx.output.side_effects.track_mutable(value, result)

    def wrap_range_iterator(self, value: range_iterator):
        self.install_guards(GuardBuilder.RANGE_ITERATOR_MATCH)
        # Get all the values from the range iterator; no need to install guards
        # on items since `RANGE_ITERATOR_MATCH` guarantees the same items.
        items = [ConstantVariable.create(v) for v in copy.deepcopy(value)]
        result = ListIteratorVariable(items, source=self.source)
        return self.tx.output.side_effects.track_mutable(value, result)

    def wrap_slice_range(self, value: Union[slice, range]):
        items = [
            VariableBuilder(self.tx, AttrSource(self.get_source(), k))(
                getattr(value, k)
            )
            for k in ("start", "stop", "step")
        ]
        self.install_guards(GuardBuilder.TYPE_MATCH)
        if isinstance(value, slice):
            return SliceVariable(items, self.tx, source=self.source)
        else:
            return RangeVariable(items, source=self.source)

    def mark_static_input(self, value: torch.Tensor, guard: bool):
        from ..decorators import mark_static_address

        static_inputs_log.debug(
            "Marking static input %s, id: %s)", self.source.name(), id(value)
        )
        mark_static_address(value, guard=guard)

        # Check if we've seen this tensor before and update graph metadata if needed
        # As long as this runs before AOT this is sound
        if value in self.tx.output.side_effects:
            var = self.tx.output.side_effects[value]
            var.proxy.node.meta["tensor_dict"]["_dynamo_static_input_type"] = (
                value._dynamo_static_input_type
            )

    def wrap_module(self, value: torch.nn.Module):
        from ..eval_frame import OptimizedModule

        if len(value.__dict__) == 0:
            unimplemented(
                gb_type="Uninitialized nn.Module",
                context=typestr(value),
                explanation=f"Attempted to trace an uninitialized nn.Module of type {typestr(value)}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                    "Ensure your nn.Module instance has called `super().__init__()`.",
                ],
            )
        if istype(value, OptimizedModule):
            # Check if the optimized module was disabled
            if inspect.getattr_static(value.forward, "_torchdynamo_disable", False):
                # This bytecode is mostly of kind LOAD_ATTR or LOAD_METHOD. If
                # we graph break here, Dynamo does not know how to create
                # continuation functions for such bytecodes. So, we delay the
                # graph break to CALL_FUNCTION.
                msg = inspect.getattr_static(
                    value.forward, "_torchdynamo_disable_msg", None
                )
                return DelayGraphBreakVariable(
                    source=self.source,
                    msg=f"Optimized `nn.Module` is wrapped with `torch.compiler.disable` (reason: {msg})",
                )

            self.install_guards(GuardBuilder.TYPE_MATCH)
            self.source = AttrSource(self.source, "_orig_mod")
            return self.wrap_module(value._orig_mod)

        if (
            isinstance(value, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM))
            and not config.allow_rnn
        ):
            unimplemented(
                gb_type="Attempted to wrap RNN, GRU, or LSTM",
                context=str(value),
                explanation="Dynamo does not support RNN, GRU, or LSTM.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        if getattr(value, "_is_fsdp_managed_module", False):
            # See note [Dynamo treats FSDP wrapped modules as UnspecializedNNModule]
            # in fully_sharded_data_parallel.py for more information

            # we can't do this assert inside FSDP constructor,
            # since we don't know yet whether dynamo will be used
            if not getattr(value, "_fsdp_use_orig_params", False):
                unimplemented(
                    gb_type="FSDP with use_orig_params=False",
                    context="",
                    explanation="Dynamo only supports FSDP with use_orig_params=True",
                    hints=[],
                )

            # Note on FSDP guarding
            # Eager FSDP already assumes (requires, but without enforcement)
            # that users don't mutate their model parameters/structure after
            # FSDP wrapping, because FSDP wouldn't notice or update its
            # FlatParams.
            #
            # Therefore, torch.compile can skip guarding on params or submodule
            # structure of fsdp_managed modules, by using FSDPNNModuleSource as
            # the guard source.  This behavior is gated on
            # config.skip_fsdp_guards.
            self.install_guards(GuardBuilder.TYPE_MATCH)
            result = FSDPManagedNNModuleVariable(value, source=self.get_source())
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif mutation_guard.is_dynamic_nn_module(value, self.tx.export):
            # created dynamically, don't specialize on it

            # Note [Tracing a torch.compiled function]
            # when make_fx tracing a compiled function, we need
            if isinstance(value, torch.fx.experimental.proxy_tensor._AttrProxy):
                value = value.get_base()
                self.source = AttrProxySource(self.source)

            if torch._dynamo.config.inline_inbuilt_nn_modules:
                freezing = is_parameter_freezing()

                # Guard against the case where user may overwrite named parameters
                # / named buffers
                # NOTE: This is not likely to happen but worth guarding to avoid
                # exception
                if (
                    callable(value.named_parameters)
                    and value.named_parameters.__func__
                    is og_module_named_parameters_fn_ptr
                ):
                    try:  # catch TypeErrors in named_parameters() from unserializable nn modules
                        for _, p in value.named_parameters():
                            self.mark_static_input(p, guard=freezing)
                    except TypeError as e:
                        raise_observed_exception(type(e), self.tx, args=list(e.args))

                if (
                    callable(value.named_buffers)
                    and value.named_buffers.__func__ is og_module_named_buffers_fn_ptr
                ):
                    try:  # catch TypeErrors in named_parameters() from unserializable nn modules
                        for _, b in value.named_buffers():
                            self.mark_static_input(b, guard=freezing)
                    except TypeError as e:
                        raise_observed_exception(type(e), self.tx, args=list(e.args))

                if freezing:
                    # we need to add the module to tracing context
                    # in order to allow its params to get invalidated
                    # this will get cleaned up once compile ends
                    self.tx.output.nn_modules[self.name] = value

            if (
                value.__module__.startswith(("torch.nn.modules", "torch.ao."))
                and not value.__module__.startswith("torch.nn.modules.container")
            ) or getattr(value.__class__, "_dynamo_marked_static", False):
                new_source = self.source
                if config.inline_inbuilt_nn_modules and (
                    not self.tx.output.export or config.install_free_tensors
                ):
                    # Export corner case - look at test_repros.py test_inlining_cornercase
                    new_source = UnspecializedBuiltinNNModuleSource(self.source)
                result = UnspecializedBuiltinNNModuleVariable(value, source=new_source)
                install_guard(new_source.make_guard(GuardBuilder.TYPE_MATCH))
            else:
                new_source = self.source
                if config.inline_inbuilt_nn_modules and (
                    not self.tx.output.export or config.install_free_tensors
                ):
                    # Export corner case - look at test_repros.py test_inlining_cornercase
                    new_source = UnspecializedNNModuleSource(self.source)
                result = UnspecializedNNModuleVariable(value, source=new_source)
                install_guard(new_source.make_guard(GuardBuilder.TYPE_MATCH))

            self.tx.output.add_fqn_info_for_inlined_modules(value, self.source)

            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(value, result)
        elif issubclass(
            value.__class__, torch.nn.parallel.distributed.DistributedDataParallel
        ):
            self.install_guards(GuardBuilder.TYPE_MATCH)
            return UnspecializedNNModuleVariable(value, source=self.get_source())
        else:
            return self.tx.output.register_attr_or_module(
                value,
                self.name,
                source=self.get_source(),
                # Guards are added inside register_attr_or_module
            )

    def wrap_literal(self, value):
        if type(value) is int:
            # allowlist has higher precedence over specialization control.
            if is_dynamic_source(self.source.name()):
                log.debug("%s marked dynamic via source whitelist", self.source.name())
                return self.wrap_symint(value, dynamism=DimDynamic.DYNAMIC)

            if is_unbacked_source(self.source.name()):
                log.debug("%s marked unbacked via source whitelist", self.source.name())
                return self.wrap_symint(value, dynamism=DimDynamic.SIZE_LIKE_UNBACKED)

            if not config.specialize_int:
                # unspecializing int by default, but still
                # specialize for the following conditions
                if is_int_specialization_case(value, self.source):
                    recompile_hint = None
                    if (
                        self.source.guard_source().is_unspecialized_builtin_nn_module()
                        or self.source.guard_source().is_unspecialized_nn_module()
                    ):
                        # This means that it is an integer from a NN module.
                        # Dynamo considers nn module int attributes to be static
                        # (a good heuristic). But a user might want to mark the
                        # int attribute to be a symint, so track this integer
                        # for recompilation later.
                        recompile_hint = (
                            "torch.compile considers integer attributes of the nn.Module to be static. "
                            "If you are observing recompilation, you might want to make this integer dynamic "
                            "using torch._dynamo.config.allow_unspec_int_on_nn_module = True, or convert this "
                            "integer into a tensor."
                        )

                    process_automatic_dynamic(
                        self.tx,
                        self.source.name(),
                        FrameStateSizeEntry.make_scalar(value),
                        is_unspecialized_nn_module=self.source.guard_source().is_unspecialized_nn_module(),
                    )
                    self.install_guards(
                        functools.partial(
                            GuardBuilder.EQUALS_MATCH, recompile_hint=recompile_hint
                        )
                    )
                    return ConstantVariable.create(value=value, source=self.source)

            return self.wrap_symint(value)
        elif not config.specialize_float and type(value) is float:
            return self.wrap_symfloat(value)
        else:
            self.install_guards(GuardBuilder.CONSTANT_MATCH)
            result = ConstantVariable.create(value=value, source=self.source)
            if isinstance(value, (list, set)):
                return self.tx.output.side_effects.track_mutable(value, result)
            return result

    def assert_not_wrapped_by_this_graph(self, value: torch.Tensor):
        if is_fake(value) and maybe_get_fake_mode(value) is self.tx.fake_mode:
            raise InternalTorchDynamoError(
                "Cannot wrap a Tensor that has already been",
                "wrapped by this instance of Dynamo",
            )

    def wrap_tensor(self, value: torch.Tensor):
        source = self.get_source()

        # We cannot already be tracking the tensor, which implies
        # it would have already been wrapped
        assert value not in self.tx.output.side_effects

        is_static_input = get_static_address_type(value) is not None

        if (
            config.inline_inbuilt_nn_modules
            and not is_static_input
            and (
                isinstance(value, torch.nn.Parameter)
                # mark tensor attributes of nn modules static. This is done to keep inline_inbuilt_nn_modules behavior
                # compatible with previous behavior.
                or (source and source.guard_source().is_unspecialized_nn_module())
            )
        ):
            self.mark_static_input(value, guard=is_parameter_freezing())
            is_static_input = True

        # Install any tensors which are "free" variables; that is:
        # 1. Globals
        # 2. NonLocals
        # 3. tensors that are attributes of nn module
        should_install_free_tensor = config.install_free_tensors and (
            is_from_global_source(source)
            or is_from_nonlocal_source(source)
            or is_from_unspecialized_nn_module_source(source)
        )

        make_graph_attribute = is_static_input and (
            not config.inline_inbuilt_nn_modules
            or is_parameter_freezing()
            or torch._dynamo.config.prepare_freezing
        )

        if should_install_free_tensor or (
            (source.guard_source().is_specialized_nn_module() or make_graph_attribute)
            and not source.guard_source().is_fsdp_module()
        ):
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(
                value, self.name, source=source
            )

        if get_static_address_type(value) == "guarded":
            # If it's a guarded tensor, we can install the parameter directly
            # into  the Fx graph instead of lifting it as an input. Lifting
            # offers no benefit,  such as regional compilation, since we still
            # guard on the tensor's ID.  Moreover, installing it in the Fx graph
            # eliminates the pre-graph bytecode  required to extract the tensor
            # from locals/globals, reducing overhead.  This can lead to
            # significant cost savings, especially for optimizers  handling many
            # tensors.
            self.install_guards(GuardBuilder.ID_MATCH)
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(
                value, self.name, source=source
            )

        if is_constant_source(source):
            self.assert_not_wrapped_by_this_graph(value)
            return self.tx.output.register_attr_or_module(
                value,
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
                source=source,
                # Guards are added inside register_attr_or_module
            )

        # NB: this just says we accessed a tensor from the same source again
        # (e.g., a tensor lives in a global foo, and we LOAD_GLOBAL it twice).
        # This is distinct from two distinct sources mapping to the same
        # Tensor (per id())!  No guard is necessary here.  See below for the
        # other case.
        is_duplicate_tensor = source in self.tx.output.input_source_to_var
        if is_duplicate_tensor:
            return self.tx.output.input_source_to_var[source]

        options = {}
        subclass_type = infer_subclass_type(value)
        if subclass_type is not None:
            self.install_guards(GuardBuilder.TYPE_MATCH)

        if get_static_address_type(value) == "guarded":
            self.install_guards(GuardBuilder.ID_MATCH)

        # By this point, we should have deduplicated all tensors
        self.assert_not_wrapped_by_this_graph(value)

        if (
            isinstance(value, torch.Tensor)
            and value.is_nested
            and not isinstance(value, torch.nested._internal.nested_tensor.NestedTensor)
        ):
            unimplemented(
                gb_type="Attempted to wrap strided NestedTensor",
                context="",
                explanation="torch.compile does not support strided NestedTensor",
                hints=[],
            )

        # TODO(pearu,sparse-team) - Add the corresponding SPARSE_TENSOR_MATCH guards
        if (
            isinstance(value, torch.Tensor)
            and is_sparse_any(value)
            and (not self.tx.export or not config.capture_sparse_compute)
        ):
            # A hot fix for sparse tensors + torch.compile. Support for
            # export + sparsity is being added but we need to create
            # SPARSE_TENSOR_GUARDS for guards to work properly.
            unimplemented(
                gb_type="Attempted to wrap sparse Tensor",
                context="",
                explanation="torch.compile does not support sparse Tensors",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        if (
            safe_has_grad(value)
            and safe_grad(value) is not None
            and value.dtype != safe_grad(value).dtype
        ):
            unimplemented(
                gb_type="dtype mismatch between tensor and its gradient",
                context=f"tensor dtype: {value.dtype}; grad dtype: {safe_grad(value).dtype}",
                explanation="Inconsistent dtype between tensor and its gradient. "
                "This can happen in FSDP and crashes meta tensor creation.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        # tx.output has multiple tracers if we're introspecting HigherOrderOperator.
        # When we've discovered an untracked tensor, then we actually need
        # to get Dynamo to track the tensor (which is what this function does)
        # and put it as a graph input on the root tracer. Later on,
        # if the input is actually used in the body of the HigherOrderOperator,
        # then the relevant SubgraphTracer will lift it to being an input of
        # the subgraph.
        # See NOTE [HigherOrderOperator tracing design] for more details.

        example_value = wrap_to_fake_tensor_and_record(
            value, tx=self.tx, is_tensor=True, source=source
        )

        tensor_proxy = self.tx.output.root_tracer.create_graph_input(
            re.sub(r"[^a-zA-Z0-9]+", "_", self.name),
            type(value),
            example_value,
            source=source,
        )
        cache_real_value_when_export(self.tx, tensor_proxy, value)

        tensor_variable = wrap_fx_proxy(
            tx=self.tx,
            proxy=tensor_proxy,
            example_value=example_value,
            subclass_type=subclass_type,
            source=source,
            **options,
        )

        if value._is_view():
            # If value is a view, add its base tensor to the tracked fakes list.
            # This is so we are able to access the correct source for its symbolic
            # shape values, in case we need them.
            wrap_to_fake_tensor_and_record(
                value._base,
                tx=self.tx,
                source=AttrSource(source, "_base"),
                is_tensor=True,
            )

        guard_type = GuardBuilder.TENSOR_MATCH

        if isinstance(source, GradSource) and is_from_optimizer_source(source):
            guard_type = GuardBuilder.NOT_NONE_MATCH

        is_dtensor = torch.distributed.is_available() and isinstance(
            value, torch.distributed.tensor.DTensor
        )
        if not is_dtensor:
            # We guard on the _local_tensor and the _spec, and therefore we dont
            # have to guard on the outer DTensor.
            self.install_guards(
                functools.partial(
                    guard_type,
                    value=(
                        value
                        if isinstance(source, NumpyTensorSource)
                        else TensorWeakRef(value)
                    ),
                )
            )

        # We install TYPE_MATCH guards for traceable wrapper subclass object,
        # and recursively install corresponding guard for each inner attribute.
        if is_traceable_wrapper_subclass(value):
            # Tensor subclass guards are very expensive because they are
            # implemented in Python. Since DTensor is PyTorch-maintained class,
            # we can skip a lot of these guards.
            if is_dtensor:
                self.install_guards(GuardBuilder.TYPE_MATCH)

                # The inner tensor name is always _local_tensor. If its not, we
                # raise assertion to update the check accordingly.
                inner_tensor_name = value.__tensor_flatten__()[0][0]
                if inner_tensor_name != "_local_tensor":
                    raise RuntimeError(
                        "Expecting Dtensor inner tensor name to be _local_tensor"
                    )

                # Now selectively guard on the flattening context
                flattening_ctx = value.__tensor_flatten__()[1]
                # This is supposed to be (self._spec, self.requires_grad)
                if not (
                    len(flattening_ctx) == 2
                    and flattening_ctx[0] == value._spec
                    and flattening_ctx[1] == value.requires_grad
                ):
                    # If not, raise an assertion to update to the new guards
                    raise RuntimeError(
                        "Expecting Dtensor flattening ctx to be _spec, requires_grad"
                    )
                # Guard on the dtensor spec
                install_guard(
                    AttrSource(self.source, "_spec").make_guard(
                        GuardBuilder.DTENSOR_SPEC_MATCH
                    )
                )
                # Move this to C++
                install_guard(
                    AttrSource(self.source, "requires_grad").make_guard(
                        GuardBuilder.EQUALS_MATCH
        

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 9 class(es): _missing, class, BackwardStateGraphArg, VariableBuilder, JITFunction, Autotuner, TensorDescriptor, SourcelessBuilder, SourcelessUserDefinedObjectBuilder

### Functions
This file defines 73 function(s): safe_has_grad, __setattr__, example, __post_init__, reconstruct, erase, __eq__, __init__, reconstruct, __init__, __call__, _is_deduplicable_sym_variable, _can_lift_attrs_to_inputs, get_source, install_guards, _type_dispatch, _type_dispatch_impl, wrap_regex_pattern, wrap_weakref, wrap_removable_handle, wrap_jit_function, wrap_mapping_proxy, build_key_value, _id_dispatch, _wrap, create_1d_tma_descriptor, create_2d_tma_descriptor, from_tensor, build_key_value, build_key_value


## Key Components

The file contains 13228 words across 3941 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 169959 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
