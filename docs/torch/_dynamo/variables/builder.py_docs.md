# Documentation: `torch/_dynamo/variables/builder.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/builder.py`
- **Size**: 169,959 bytes (165.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
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
        elif v
```



## High-Level Overview

"""This module contains classes and utilities for building variable trackers in Dynamo.Variable trackers are used to convert Python values into symbolic representationsthat can be traced and transformed during graph capture.The key classes are:- VariableBuilder: Handles source-tracked objects that need guards and proper  reconstruction in the output graph. Used for inputs, module attributes, etc.- SourcelessBuilder: Handles ephemeral objects created during tracing that don't  need source tracking or guards. Used for temporary lists, intermediate values, etc.Variable trackers enable Dynamo to track the flow of values through the program,maintain guards for dynamic properties, and reconstruct values in the output graph.The builders in this module handle converting Python values into appropriateVariableTracker instances based on their type and usage context.

This Python file contains 26 class(es) and 73 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_missing`, `GraphArg`, `BackwardStateGraphArg`, `VariableBuilder`, `JITFunction`, `Autotuner`, `TensorDescriptor`, `SourcelessBuilder`, `SourcelessUserDefinedObjectBuilder`

**Functions defined**: `safe_has_grad`, `__setattr__`, `example`, `__post_init__`, `reconstruct`, `erase`, `__eq__`, `__init__`, `reconstruct`, `__init__`, `__call__`, `_is_deduplicable_sym_variable`, `_can_lift_attrs_to_inputs`, `get_source`, `install_guards`, `_type_dispatch`, `_type_dispatch_impl`, `wrap_regex_pattern`, `wrap_weakref`, `wrap_removable_handle`

**Key imports**: abc, collections, contextlib, copy, dataclasses, enum, functools, inspect, itertools, logging


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `collections`
- `contextlib`
- `copy`
- `dataclasses`
- `enum`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `math`
- `operator`
- `random`
- `re`
- `sys`
- `traceback`
- `types`
- `weakref`
- `collections.abc`: Callable, MutableMapping
- `typing`: Any, NamedTuple, Optional, TYPE_CHECKING, Union
- `sympy`
- `torch`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._guards`: TracingContext
- `torch._higher_order_ops.flat_apply`: flat_apply
- `torch._higher_order_ops.torchbind`: call_torchbind
- `torch._library.opaque_object`: is_opaque_type
- `torch._ops`: HigherOrderOperator
- `torch._subclasses.fake_tensor`: FakeTensor, is_fake, maybe_get_fake_mode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`torch/_dynamo/variables`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`nn_module.py_docs.md`](./nn_module.py_docs.md)
- [`higher_order_ops.py_docs.md`](./higher_order_ops.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`torch.py_docs.md`](./torch.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`dicts.py_docs.md`](./dicts.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`lists.py_docs.md`](./lists.py_docs.md)


## Cross-References

- **File Documentation**: `builder.py_docs.md`
- **Keyword Index**: `builder.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
