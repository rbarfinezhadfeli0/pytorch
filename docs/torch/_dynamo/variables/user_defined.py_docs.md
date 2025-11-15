# Documentation: `torch/_dynamo/variables/user_defined.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/user_defined.py`
- **Size**: 92,210 bytes (90.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

"""
This module contains variable classes for handling user-defined objects in Dynamo's tracing system.

The key classes are:
- UserDefinedVariable: Base class for representing custom Python objects
- UserDefinedClassVariable: Handles Python class objects/types
- UserDefinedObjectVariable: Fallback class for instance objects, with support for method calls,
  attribute access, and other Python object behaviors.
- Specialized subclasses for common patterns:
  - UserDefinedDictVariable: For dict subclasses
  - UserDefinedSetVariable: For set subclasses
  - UserDefinedTupleVariable: For tuple subclasses
  - UserDefinedExceptionObjectVariable: For exception subclasses
  - FrozenDataClassVariable: Special handling of frozen dataclasses
  - MutableMappingVariable: For collections.abc.MutableMapping subclasses

Dynamo specializes to VariableTracker subclasses like FrozenDataClassVariable if available; if no
subclass qualifies, it falls back to UserDefinedObjectVariable.

These classes help Dynamo track and handle arbitrary Python objects during tracing,
maintaining proper semantics while enabling optimizations where possible.
"""

import _collections
import builtins
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import random
import sys
import threading
import types
import warnings
import weakref
from typing import TYPE_CHECKING
from typing_extensions import is_typeddict

import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    ObservedKeyError,
    ObservedTypeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    unimplemented,
)
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
    DataclassFieldsSource,
    DictGetItemSource,
    GetItemSource,
    RandomValueSource,
    TypeDictSource,
    TypeMROSource,
    TypeSource,
    UnspecializedParamBufferSource,
)
from ..utils import (
    check_constant_args,
    cmp_name_to_op_mapping,
    dict_methods,
    frozenset_methods,
    get_custom_getattr,
    has_torch_function,
    is_frozen_dataclass,
    is_lru_cache_wrapped_function,
    is_namedtuple_cls,
    is_wrapper_or_member_descriptor,
    istype,
    list_methods,
    namedtuple_fields,
    object_has_getattribute,
    proxy_args_kwargs,
    raise_args_mismatch,
    set_methods,
    tensortype_to_dtype,
    tuple_methods,
    unpatched_nn_module_getattr,
)
from .base import raise_type_error_exc, ValueMutationNew, VariableTracker
from .dicts import ConstDictVariable, DefaultDictVariable
from .lists import SizeVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from torch.utils._cxx_pytree import PyTreeSpec
except ImportError:
    PyTreeSpec = type(None)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


def is_standard_setattr(val):
    return val in (object.__setattr__, BaseException.__setattr__)


def is_standard_delattr(val):
    return val in (object.__delattr__, BaseException.__delattr__)


def is_forbidden_context_manager(ctx):
    f_ctxs = []

    try:
        from _pytest.python_api import RaisesContext
        from _pytest.recwarn import WarningsChecker

        f_ctxs.append(RaisesContext)
        f_ctxs.append(WarningsChecker)
    except ImportError:
        pass

    if m := sys.modules.get("torch.testing._internal.jit_utils"):
        f_ctxs.append(m._AssertRaisesRegexWithHighlightContext)

    return ctx in f_ctxs


def is_cython_function(obj):
    return (
        callable(obj)
        and hasattr(type(obj), "__name__")
        and type(obj).__name__ == "cython_function_or_method"
    )


class UserDefinedVariable(VariableTracker):
    value: object


class UserDefinedClassVariable(UserDefinedVariable):
    value: type[object]

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value
        # Used when we materialize class.__dict__ to a MappingProxyObject. In
        # this case, we don't want to allow mutation in the class because there
        # is no way to reflect it in the created MappingProxyVariable.
        self.ban_mutation = False

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    @staticmethod
    @functools.cache
    def _constant_fold_classes():
        return {
            torch.device,
            torch.finfo,
            torch.iinfo,
            torch.Size,
        }

    @staticmethod
    @functools.cache
    def _in_graph_classes():
        _in_graph_class_list = {
            torch.Tensor,
            torch.cuda.FloatTensor,
            torch.cuda.DoubleTensor,
            torch.cuda.HalfTensor,
            torch.cuda.BFloat16Tensor,
            torch.cuda.ByteTensor,
            torch.cuda.CharTensor,
            torch.cuda.IntTensor,
            torch.cuda.ShortTensor,
            torch.cuda.LongTensor,
            torch.Stream,
            torch.Event,
            torch.cuda.Stream,
            torch.cuda.Event,
            torch.xpu.Stream,
            torch.xpu.Event,
        }
        if hasattr(torch, "hpu"):
            _in_graph_class_list.update(
                {
                    torch.hpu.Stream,
                    torch.hpu.Event,
                }
            )

        return set(tensortype_to_dtype.keys()) | _in_graph_class_list

    @staticmethod
    @functools.cache
    def supported_c_new_functions():
        exceptions = [
            getattr(builtins, name).__new__
            for name in dir(builtins)
            if isinstance(getattr(builtins, name), type)
            and issubclass(getattr(builtins, name), BaseException)
        ]
        return {
            object.__new__,
            dict.__new__,
            set.__new__,
            frozenset.__new__,
            tuple.__new__,
            list.__new__,
        }.union(exceptions)

    @staticmethod
    def is_supported_new_method(value):
        # TODO(anijain2305) - Extend this to support objects with default tp_new
        # functions.
        return value in UserDefinedClassVariable.supported_c_new_functions()

    def can_constant_fold_through(self):
        return self.value in self._constant_fold_classes()

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key):
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        from . import ConstantVariable, EnumVariable

        source = AttrSource(self.source, name) if self.source is not None else None

        if name == "__name__":
            return ConstantVariable.create(self.value.__name__)
        elif name == "__qualname__":
            return ConstantVariable.create(self.value.__qualname__)
        elif name == "__dict__":
            options = {"source": source}
            return variables.GetAttrVariable(self, name, **options)
        elif name == "__mro__":
            attr_source = self.source and TypeMROSource(self.source)
            return VariableTracker.build(tx, self.value.__mro__, attr_source)

        # Special handling of collections.OrderedDict.fromkeys()
        # Wrap it as GetAttrVariable(collections.OrderedDict, "fromkeys") to make it consistent with
        # collections.defaultdict, and both will be handled at UserDefinedClassVariable.call_method().
        # Otherwise, it would be wrapped as UserDefinedObjectVariable(collections.OrderedDict.fromkeys),
        # and we need duplicate code to handle both cases.
        if (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return super().var_getattr(tx, name)

        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            if type(self.value) is type:
                raise_observed_exception(
                    AttributeError,
                    tx,
                    msg=f"type object '{self.value.__name__}' has no attribute '{name}'",
                )
            else:
                # Cannot reason about classes with a custom metaclass
                # See: test_functions::test_getattr_metaclass
                obj = None

        if name == "__new__" and UserDefinedClassVariable.is_supported_new_method(obj):
            return super().var_getattr(tx, name)

        if name in cmp_name_to_op_mapping and not isinstance(obj, types.FunctionType):
            return variables.GetAttrVariable(self, name, source=source)

        if isinstance(obj, staticmethod):
            return VariableTracker.build(tx, obj.__get__(self.value), source)
        elif isinstance(obj, classmethod):
            if isinstance(obj.__func__, property):
                fget_vt = VariableTracker.build(tx, obj.__func__.fget)
                return fget_vt.call_function(tx, [self], {})
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif isinstance(obj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static(dict, "fromkeys")
            #       inspect.getattr_static(itertools.chain, "from_iterable")
            func = obj.__get__(None, self.value)
            return VariableTracker.build(tx, func, source)
        elif source:
            if inspect.ismemberdescriptor(obj):
                return VariableTracker.build(tx, obj.__get__(self.value), source)

        if ConstantVariable.is_literal(obj):
            return ConstantVariable.create(obj)
        elif isinstance(obj, enum.Enum):
            return EnumVariable(obj)
        elif self.value is collections.OrderedDict:
            return variables.GetAttrVariable(self, name)
        elif name in getattr(self.value, "__dict__", {}) or (
            self.value.__module__.startswith("torch.")
            or self.value.__module__ == "torch"
        ):
            if source:
                return VariableTracker.build(tx, obj, source)

        if (
            source
            and not inspect.ismethoddescriptor(obj)
            and not is_wrapper_or_member_descriptor(obj)
        ):
            return VariableTracker.build(tx, obj, source)

        return super().var_getattr(tx, name)

    def _call_cross_entropy_loss(self, tx: "InstructionTranslator", args, kwargs):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight=ConstantVariable.create(None),
            size_average=ConstantVariable.create(None),
            ignore_index=ConstantVariable.create(-100),
            reduce=ConstantVariable.create(None),
            reduction=ConstantVariable.create("mean"),
            label_smoothing=ConstantVariable.create(0.0),
        ):
            return (
                weight,
                size_average,
                ignore_index,
                reduce,
                reduction,
                label_smoothing,
            )

        (
            weight,
            size_average,
            ignore_index,
            reduce_arg,
            reduction,
            label_smoothing,
        ) = normalize_args(*args, **kwargs)

        def fake_cross_entropy_loss(input, target):
            from .builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.nn.functional.cross_entropy,
                    *proxy_args_kwargs(
                        [
                            input,
                            target,
                            weight,
                            size_average,
                            ignore_index,
                            reduce_arg,
                            reduction,
                            label_smoothing,
                        ],
                        {},
                    ),
                ),
            )

        return variables.LambdaVariable(fake_cross_entropy_loss)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__subclasses__"
            and len(args) == 0
            and not kwargs
            and "__subclasses__" not in self.value.__dict__
        ):
            source = self.source
            if self.source:
                source = AttrSource(self.source, "__subclasses__")
                source = CallFunctionNoArgsSource(source)
            return VariableTracker.build(tx, self.value.__subclasses__(), source)
        elif (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return variables.BuiltinVariable.call_custom_dict_fromkeys(
                tx, self.value, *args, **kwargs
            )
        elif self.value is collections.OrderedDict and name == "move_to_end":
            return args[0].call_method(tx, name, [*args[1:]], kwargs)
        elif name == "__eq__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value == args[0].value)
        elif name == "__ne__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value != args[0].value)
        elif issubclass(self.value, dict) and name != "__new__":
            # __new__ is handled below
            return variables.BuiltinVariable(dict).call_method(tx, name, args, kwargs)
        elif issubclass(self.value, (set, frozenset)) and name != "__new__":
            # __new__ is handled below
            return variables.BuiltinVariable(set).call_method(tx, name, args, kwargs)
        elif (
            name == "__new__"
            and self.value is collections.OrderedDict
            and isinstance(args[0], UserDefinedClassVariable)
            and args[0].value is collections.OrderedDict
        ):
            if kwargs and len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return variables.ConstDictVariable(
                {}, collections.OrderedDict, mutation_type=ValueMutationNew()
            )
        elif name == "__new__" and UserDefinedClassVariable.is_supported_new_method(
            self.value.__new__
        ):
            return tx.output.side_effects.track_new_user_defined_object(
                self,
                args[0],
                args[1:],
            )
        elif name == "__setattr__" and self.ban_mutation:
            unimplemented(
                gb_type="Class attribute mutation when the __dict__ was already materialized",
                context=str(self.value),
                explanation="Dyanmo does not support tracing mutations on a class when its __dict__ is materialized",
                hints=graph_break_hints.SUPPORTABLE,
            )
        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..side_effects import SideEffects
        from .builder import wrap_fx_proxy

        constant_args = check_constant_args(args, kwargs)

        if self.can_constant_fold_through() and constant_args:
            # constant fold
            return variables.ConstantVariable.create(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif self.value is torch.nn.CrossEntropyLoss:
            return self._call_cross_entropy_loss(tx, args, kwargs)
        elif self.value is contextlib.nullcontext:
            # import here to avoid circular dependency
            from .ctx_manager import NullContextVariable

            return NullContextVariable(*args, **kwargs)
        elif self.value is collections.OrderedDict:
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.construct_dict),
                [self, *args],
                kwargs,
            )
        elif self.value is collections.defaultdict:
            if len(args) == 0:
                default_factory = variables.ConstantVariable.create(None)
            else:
                default_factory, *args = args
            dict_vt = variables.BuiltinVariable.call_custom_dict(
                tx, dict, *args, **kwargs
            )
            return DefaultDictVariable(
                dict_vt.items,
                collections.defaultdict,
                default_factory,
                mutation_type=ValueMutationNew(),
            )
        elif is_typeddict(self.value):
            if self.value.__optional_keys__:
                unimplemented(
                    gb_type="TypedDict with optional keys",
                    context=str(self.value),
                    explanation="Dyanmo does not support tracing TypedDict with optional keys",
                    hints=[
                        "Avoid using TypedDict with optional keys",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            return variables.BuiltinVariable(dict).call_dict(tx, *args, **kwargs)
        elif self.value is collections.deque:
            maxlen = variables.ConstantVariable.create(None)

            def deque_signature(iterable=None, maxlen=None):
                pass

            try:
                bound_args = inspect.signature(deque_signature).bind(*args, **kwargs)
            except TypeError as e:
                unimplemented(
                    gb_type="collections.deque() with bad arguments",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Detected call to collections.deque() with bad arguments.",
                    hints=[
                        "Fix the call to collections.deque().",
                        *graph_break_hints.USER_ERROR,
                    ],
                    from_exc=e,
                )

            if "iterable" in bound_args.arguments:
                if not bound_args.arguments["iterable"].has_force_unpack_var_sequence(
                    tx
                ):
                    unimplemented(
                        gb_type="collections.deque() with bad iterable argument",
                        context=f"args={args}, kwargs={kwargs}",
                        explanation="Call to collections.deque() has an iterable argument that Dynamo cannot "
                        "convert to a list.",
                        hints=[
                            "Use a simpler sequence type that Dynamo can convert to a list "
                            "(e.g. list, tuple, list iterator, etc.)",
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
                items = bound_args.arguments["iterable"].force_unpack_var_sequence(tx)
            else:
                items = []

            if "maxlen" in bound_args.arguments:
                maxlen = bound_args.arguments["maxlen"]

            return variables.lists.DequeVariable(
                items, maxlen=maxlen, mutation_type=ValueMutationNew()
            )
        elif self.value is weakref.ref:
            if len(args) > 1:
                callback = args[1]
            else:
                callback = variables.ConstantVariable.create(None)
            return variables.WeakRefVariable(args[0], callback)
        elif self.value is functools.partial:
            if not args:
                unimplemented(
                    gb_type="missing args to functools.partial",
                    context="",
                    explanation="functools.partial requires at least one argument",
                    hints=[
                        "Fix the functools.partial call.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            # The first arg, a callable (the ctor below will assert on types)
            fn = args[0]
            rest_args = args[1:]
            # guards for the produced FunctoolsPartialVariable are installed in FunctoolsPartialVariable ctor from the
            # args and keywords
            return variables.functions.FunctoolsPartialVariable(
                fn, args=rest_args, keywords=kwargs
            )
        elif self.value is warnings.catch_warnings and not args:
            return variables.CatchWarningsCtxManagerVariable.create(tx, kwargs)
        elif self.value is torch.cuda.device and not kwargs and len(args) == 1:
            if not args[0].is_python_constant():
                raise_type_error_exc(
                    tx, "torch.cuda.device() requires a constant argument"
                )
            return variables.CUDADeviceVariable.create(tx, args[0].as_python_constant())
        elif (
            issubclass(type(self.value), type)
            and hasattr(
                self.value, "__enter__"
            )  # TODO(voz): These can invoke user code!
            and hasattr(
                self.value, "__exit__"
            )  # TODO(voz): These can invoke user code!
            and self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
            and not is_forbidden_context_manager(self.value)
        ):
            from . import TorchCtxManagerClassVariable
            from .functions import (
                BaseUserFunctionVariable,
                FunctionDecoratedByContextlibContextManagerVariable,
            )

            # graph break on any contextlib.* that it is not contextlib.contextmanager
            # Some of the APIs below are not supported because they rely on features
            # that Dynamo doesn't play well today (i.e. contextlib.suppress)
            if self.value in (
                contextlib._AsyncGeneratorContextManager,
                contextlib.closing,
                contextlib.redirect_stdout,
                contextlib.redirect_stderr,
                contextlib.suppress,
                contextlib.ExitStack,
                contextlib.AsyncExitStack,
            ):
                # We are not changing the behavior of Dynamo as these function were
                # already ignored on trace_rules.py before #136033 landed
                unimplemented(
                    gb_type="unsupported contextlib.* API",
                    context=f"{self.value}",
                    explanation=f"{self.value} not supported. This may be due to its use of "
                    "context-specific operations that are not supported in "
                    "Dynamo yet (i.e. Exception handling)",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            if self.value is contextlib._GeneratorContextManager and isinstance(
                args[0], (BaseUserFunctionVariable, TorchCtxManagerClassVariable)
            ):
                if not torch._dynamo.config.enable_trace_contextlib:
                    unimplemented(
                        gb_type="attempted to trace contextlib.contextmanager",
                        context=f"args={args}",
                        explanation="Tracing contextlib.contextmanager is disabled.",
                        hints=[
                            "Set torch._dynamo.config.enable_trace_contextlib = True",
                        ],
                    )

                # Special treatments for certain context managers created via
                # contextlib, because
                # 1. we (pytorch) own their impls
                # 2. it's tedious to trace through them, so we effectively
                #    "allow in graph" them without sacrificing soundness.
                #
                # We would typically reach here via either
                # 1. the instance construction in `with ctx_manager(...):`:
                #    https://github.com/python/cpython/blob/3.12/Lib/contextlib.py#L301
                # 2. calling a function decorated with a context manager:
                #    https://github.com/python/cpython/blob/3.12/Lib/contextlib.py#L122
                #
                # So we basically trace through the surface part of the
                # contextlib code, and then special case the shared remaining
                # logic (the actual context manager instance construction and
                # usage later on).
                if isinstance(args[0], TorchCtxManagerClassVariable):
                    fn_var = args[0]
                    args_list = args[1].items
                    kwargs_dict = args[2].keys_as_python_constant()
                    return fn_var.call_function(tx, args_list, kwargs_dict)

                # Wrap UserFunctionVariable in FunctionDecoratedByContextlibContextManagerVariable
                # if the function is annotated with @contextlib.contextmanager
                # This shouldn't be necessary once generator functions are fully
                # supported in dynamo
                args = [
                    FunctionDecoratedByContextlibContextManagerVariable(
                        args[0], source=args[0].source
                    )
                ] + args[1:]

            cm_obj = tx.output.side_effects.track_new_user_defined_object(
                variables.BuiltinVariable(object),
                self,
                args,
            )
            cm_obj.call_method(tx, "__init__", args, kwargs)
            return cm_obj
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            # check if this a quasi-namedtuple or a real one
            if self.value.__module__ == "torch.return_types":
                if kwargs or len(args) != 1:
                    raise_args_mismatch(
                        tx,
                        "torch.return_types",
                        "1 args and 0 kwargs",
                        f"{len(args)} args and {len(kwargs)} kwargs",
                    )
                items = args[0].force_unpack_var_sequence(tx)
            else:
                field_defaults = self.value._field_defaults

                items = list(args)
                items.extend([None] * (len(fields) - len(items)))

                var_tracker_kwargs = {}
                for field_name, var_tracker in zip(fields, items):
                    if var_tracker is None:
                        if field_name in kwargs:
                            field_var = kwargs[field_name]
                        else:
                            assert field_name in field_defaults
                            field_var = VariableTracker.build(
                                tx, field_defaults[field_name]
                            )
                        var_tracker_kwargs[field_name] = field_var

                for name, value in var_tracker_kwargs.items():
                    assert name in fields
                    items[fields.index(name)] = value

                assert all(x is not None for x in items)

            # Modify mutability of namedtuple for sourcelesss instantiations.
            from .base import AttributeMutationNew

            return variables.NamedTupleVariable(
                items, self.value, mutation_type=AttributeMutationNew()
            )
        elif self.value is torch.Size:
            # This simulates `THPSize_pynew`, the C impl for `Size.__new__`.
            tup = variables.BuiltinVariable(tuple).call_function(tx, args, kwargs)
            return SizeVariable(tup.items)
        elif is_frozen_dataclass(self.value) and self.is_standard_new():
            fields = dataclasses.fields(self.value)
            fields_source = DataclassFieldsSource(self.source)
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))

            default_kwargs = {}
            for ind, field, var_tracker in zip(itertools.count(), fields, items):
                if var_tracker is None:
                    if field.name in kwargs:
                        var_tracker = kwargs[field.name]
                    else:
                        if not field.init:
                            continue

                        if field.default is not dataclasses.MISSING:
                            var_tracker = VariableTracker.build(
                                tx,
                                field.default,
                                source=AttrSource(
                                    GetItemSource(fields_source, ind), "default"
                                ),
                            )
                        elif field.default_factory is not dataclasses.MISSING:
                            factory_fn = VariableTracker.build(
                                tx, field.default_factory
                            )
                            var_tracker = factory_fn.call_function(tx, [], {})
                        else:
                            # if we are subclass, the constructor could possibly
                            # be missing args
                            continue

                    default_kwargs[field.name] = var_tracker
            kwargs.update(default_kwargs)

            var = tx.output.side_effects.track_new_user_defined_object(
                variables.BuiltinVariable(object), self, args
            )
            var.call_method(tx, "__init__", args, kwargs)
            return var
        elif (
            self.value in self._in_graph_classes()
            or is_traceable_wrapper_subclass_type(self.value)
        ):
            # torch.LongTensor cannot accept a list of FakeTensors.
            # So we stack the list of FakeTensors instead.
            if (
                np
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], variables.ListVariable)
                and len(args[0].items) > 1
                and all(isinstance(x, variables.TensorVariable) for x in args[0].items)
            ):
                # Stack FakeTensor
                stacked = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        torch.stack,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
                args = [stacked]

            if issubclass(self.value, torch.Stream):
                from .constant import ConstantVariable
                from .lists import TupleVariable

                # Register newly created stream for reconstruction
                var_kwargs = ConstDictVariable(
                    {ConstantVariable(k): v for k, v in kwargs.items()}
                )
                var_args = TupleVariable(list(args))
                stream = self.value(
                    *(var_args.as_python_constant()),
                    **(var_kwargs.as_python_constant()),
                )
                from ..graph_bytecode_inputs import register_graph_created_object
                from .streams import StreamVariable

                ind = register_graph_created_object(
                    stream,
                    StreamVariable.make_construct_in_graph_stream_fn(
                        var_args, var_kwargs
                    ),
                )
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function", get_external_object_by_index, (ind,), {}
                    ),
                )
            elif issubclass(self.value, torch.Event):
                from .constant import ConstantVariable
                from .lists import TupleVariable

                # Register newly created event for reconstruction
                var_kwargs = ConstDictVariable(
                    {ConstantVariable(k): v for k, v in kwargs.items()}
                )
                var_args = TupleVariable(list(args))
                event = self.value(
                    *(var_args.as_python_constant()),
                    **(var_kwargs.as_python_constant()),
                )
                from ..graph_bytecode_inputs import register_graph_created_object
                from .streams import EventVariable

                ind = register_graph_created_object(
                    event,
                    EventVariable.make_construct_in_graph_event_fn(
                        var_args, var_kwargs
                    ),
                )
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function", get_external_object_by_index, (ind,), {}
                    ),
                )
            else:
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.value,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )

            return tensor_variable
        elif self.value is random.Random:
            if len(args) == 1 and isinstance(args[0], variables.ConstantVariable):
                seed = args[0].value
            else:
                seed = None
            random_object = random.Random(seed)
            return RandomVariable(random_object)
        elif (
            self.value is types.MappingProxyType
            and len(args) == 1
            and isinstance(args[0], variables.ConstDictVariable)
        ):
            # types.MappingProxyType is a read-only proxy of the dict. If the
            # original dict changes, the changes are reflected in proxy as well.
            return variables.MappingProxyVariable(args[0])
        elif SideEffects.cls_supports_mutation_side_effects(self.value) and self.source:
            with do_not_convert_to_tracable_parameter():
                return tx.inline_user_function_return(
                    VariableTracker.build(
                        tx, polyfills.instantiate_user_defined_class_object
                    ),
                    [self, *args],
                    kwargs,
                )
        return super().call_function(tx, args, kwargs)

    def is_standard_new(self):
        """Check for __new__ being overridden"""
        new_fn = inspect.getattr_static(self.value, "__new__", None)
        if isinstance(new_fn, staticmethod):
            new_fn = new_fn.__func__
        return new_fn is object.__new__

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "VariableTracker":
        if self.source:
            source = AttrSource(self.source, name)
            install_guard(source.make_guard(GuardBuilder.HASATTR))
            return variables.ConstantVariable(hasattr(self.value, name))
        return super().call_obj_hasattr(tx, name)

    def const_getattr(self, tx: "InstructionTranslator", name):
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)


class UserDefinedExceptionClassVariable(UserDefinedClassVariable):
    @property
    def fn(self):
        return self.value


class NO_SUCH_SUBOBJ:
    pass


def call_random_fn(tx, fn, args, kwargs):
    from .builder import VariableBuilder

    args = [x.as_python_constant() for x in args]
    kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
    random_call_index = len(tx.output.random_calls)
    example_value = fn(*args, **kwargs)
    source = RandomValueSource(random_call_index)
    tx.output.random_calls.append((fn, args, kwargs))
    # TODO: arguably, this should route to wrap_symint/wrap_symfloat
    # (currently hypothetical), but I'm not going to poke my hand in
    # this nest for now
    return VariableBuilder(tx, source).wrap_unspecialized_primitive(example_value)


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    _nonvar_fields = {
        "value",
        "value_type",
        "attrs_directly_modifed_on_dict",
        *UserDefinedVariable._nonvar_fields,
    }

    def __init__(
        self,
        value,
        *,
        value_type=None,
        cls_source=None,
        base_cls_vt=None,
        init_args=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type
        # This is used with __new__, when the new object is sourceless but the user class can be sourceful.
        self.cls_source = cls_source
        if cls_source is None and self.source is not None:
            self.cls_source = TypeSource(self.source)

        # These attributes are used to reconstruct the user defined object. The
        # pseudo code looks like this. Builtin C __new__ do not support kwargs,
        # so init_args is sufficient.
        #   obj = base_cls.__new__(user_cls, *args)
        self.base_cls_vt = base_cls_vt
        self.init_args = init_args

        # This records names of the attributes that were modified via instance
        # `__dict__` directly, rather than the normal setattr path.
        #
        # TODO consider emulating `obj.__dict__` as a `ConstDictVariable` to get
        # rid of these workarounds here and in `GetAttrVariable`.
        self.attrs_directly_modifed_on_dict = set()

        import torch.utils._pytree as pytree

        self.is_pytree_constant_class = pytree.is_constant_class(self.value_type)
        if pytree.is_constant_class(self.value_type) and self.source:
            install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))

    def __str__(self) -> str:
        inner = self.value_type.__name__
        if inner in [
            "builtin_function_or_method",
            "getset_descriptor",
            "method_descriptor",
            "method",
        ]:
            inner = str(getattr(self.value, "__name__", None))
        return f"{self.__class__.__name__}({inner})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value_type.__name__})"

    def is_underlying_vt_modified(self, side_effects):
        return False

    def python_type(self):
        return self.value_type

    def as_python_constant(self):
        if self.is_pytree_constant_class and self.source:
            # NOTE pytree constants created in the torch.compile region will
            # NOT be guarded (even though they have a source set)
            return self.value
            # TODO else try reconstructing the object by, e.g., leveraging side
            # effects and `as_python_constant`.
        return super().as_python_constant()

    def guard_as_python_constant(self):
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            return self.value
        return super().guard_as_python_constant()

    def torch_function_check(self):
        assert has_torch_function(self), (
            f"calling torch function on object without __torch_function__ {self}"
        )

    def get_torch_fn(self, tx):
        self.torch_function_check()
        from .torch_function import get_torch_function_fn

        return get_torch_function_fn(tx, self)

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        self.torch_function_check()

        from .torch_function import call_torch_function

        return call_torch_function(
            tx,
            self.get_torch_fn(tx),
            fn,
            types,
            args,
            kwargs,
        )

    @staticmethod
    @functools.cache
    def _supported_random_functions():
        fns = {
            random.random,
            random.randint,
            random.randrange,
            random.uniform,
        }
        return fns

    def _maybe_get_baseclass_method(self, name):
        if name not in getattr(self.value, "__dict__", {}):
            try:
                return inspect.getattr_static(type(self.value), name)
            except AttributeError:
                pass
        return None

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, UserMethodVariable

        method = self._maybe_get_baseclass_method(name)
        if method is not None:
            if method is object.__init__:
                return ConstantVariable.create(None)

            if is_standard_setattr(method) or isinstance(self.value, threading.local):
                return self.method_setattr_standard(tx, *args, **kwargs)

            if is_standard_delattr(method):
                return self.method_setattr_standard(
                    tx, args[0], variables.DeletedVariable()
                )

            if method is object.__eq__ and len(args) == 1 and not kwargs:
                other = args[0]
                if not isinstance(other, UserDefinedObjectVariable):
                    return variables.ConstantVariable.create(NotImplemented)

                # TODO(anijain2305) - Identity checking should already be a part
                # of the cmp_eq  polyfill function.
                return ConstantVariable.create(self.value is other.value)

            if torch._dynamo.config.enable_faithful_generator_behavior and isinstance(
                self.value, types.GeneratorType
            ):
                unimplemented(
                    gb_type="call_method on generator",
                    context=f"object={self.value}, method={name}, args={args}, kwargs={kwargs}",
                    explanation="Detected a method call to a user-defined generator object. "
                    "This is not fully supported.",
                    hints=[
                        "Set `torch._dynamo.config.enable_faithful_generator_behavior = False`. Note that this "
                        "may cause silent incorrectness, since we will eagerly unpack generators instead of lazily "
                        "evaluating them.",
                    ],
                )

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = self.source
                source_fn = None
                if source:
                    source_fn = self.get_source_by_walking_mro(name)
                # TODO(jansel): add a guard to check for monkey patching?
                from ..mutation_guard import unpatched_nn_module_init

                if method is torch.nn.Module.__init__:
                    method = unpatched_nn_module_init
                return UserMethodVariable(
                    method, self, source_fn=source_fn, source=source
                ).call_function(tx, args, kwargs)

            if method is list.__len__ and self.source and not (args or kwargs):
                install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
                return ConstantVariable(len(self.value))

        return super().call_method(tx, name, args, kwargs)

    def method_setattr_standard(
        self, tx: "InstructionTranslator", name, value, directly_update_dict=False
    ):
        try:
            name = name.as_python_constant()
        except NotImplementedError:
            unimplemented(
                gb_type="non-const setattr name on user-defined object",
                context=f"object={self}, name={name}, value={value}",
                explanation="Detected a call to `setattr` of a user-defined object with a non-constant name.",
                hints=["Ensure that the name is a string."],
            )
        assert tx.output.side_effects.is_attribute_mutation(self), (
            "Attempted setattr on a user-defined object that does not have "
            "an AttributeMutation mutation_type"
        )

        if directly_update_dict:
            self.attrs_directly_modifed_on_dict.add(name)
        else:
            tmp = self.try_get_descritor_and_setter_py_func(name)
            if tmp:
                descriptor, setter = tmp
                # Emulate
                # https://github.com/python/cpython/blob/3.11/Objects/object.c#L1371-L1452
                desc_source = None
                func_source = None
                if self.cls_source:
                    desc_source = self.get_source_by_walking_mro(name)
                    # use `type(...)` to ignore instance attrs.
                    func_source = AttrSource(TypeSource(desc_source), "__set__")
                desc_var = VariableTracker.build(tx, descriptor, desc_source)
                func_var = VariableTracker.build(tx, setter, func_source)
                args = [desc_var, self, value]
                return func_var.call_function(tx, args, {})
            # NOTE: else we assume the descriptor (if any) has a
            # side-effect-free `__set__` as far as Dynamo tracing is concerned.

        # Emulate the standard setattr on instance dict.
        tx.output.side_effects.store_attr(self, name, value)
        return variables.ConstantVariable(None)

    def needs_slow_setattr(self):
        return not is_standard_setattr(
            inspect.getattr_static(self.value, "__setattr__", None)
        ) and not isinstance(self.value, threading.local)

    def unpack_var_sequence(self, tx):
        if (
            self.source
            and self._maybe_get_baseclass_method("__iter__") is list.__iter__
            and self._maybe_get_baseclass_method("__len__") is list.__len__
            and self._maybe_get_baseclass_method("__getitem__") is list.__getitem__
        ):
            install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return [
                variables.LazyVariableTracker.create(
                    self.value[k],
                    source=GetItemSource(self.source, k),
                )
                for k in range(len(self.value))
            ]
        return super().unpack_var_sequence(tx)

    def has_force_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        try:
            variables.BuiltinVariable(iter).call_function(tx, [self], {})
            return True
        except ObservedTypeError:
            handle_observed_exception(tx)
            return False

    def force_unpack_var_sequence(self, tx):
        result = []
        iter_ = variables.BuiltinVariable(iter).call_function(tx, [self], {})

        while True:
            try:
                r = iter_.next_variable(tx)
                result.append(r)
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break
        return result

    def next_variable(self, tx):
        return self.call_method(tx, "__next__", [], {})

    def is_supported_random(self):
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            # TypeError: unhashable type
            return False

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            self.is_supported_random()
            and all(k.is_python_constant() for k in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            return call_random_fn(tx, self.value, args, kwargs)
        elif istype(self.value, types.MethodType):
            func = self.value.__func__
            obj = self.value.__self__
            if (
                func is torch.utils._contextlib._DecoratorContextManager.clone
                and variables.TorchCtxManagerClassVariable.is_matching_cls(
                    obj.__class__
                )
                and not (args or kwargs)
            ):
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, args, kwargs)

            if (
                func is torch.autograd.grad_mode.inference_mode.clone
         
```



## High-Level Overview

"""This module contains variable classes for handling user-defined objects in Dynamo's tracing system.The key classes are:- UserDefinedVariable: Base class for representing custom Python objects- UserDefinedClassVariable: Handles Python class objects/types- UserDefinedObjectVariable: Fallback class for instance objects, with support for method calls,  attribute access, and other Python object behaviors.- Specialized subclasses for common patterns:  - UserDefinedDictVariable: For dict subclasses  - UserDefinedSetVariable: For set subclasses  - UserDefinedTupleVariable: For tuple subclasses  - UserDefinedExceptionObjectVariable: For exception subclasses  - FrozenDataClassVariable: Special handling of frozen dataclasses  - MutableMappingVariable: For collections.abc.MutableMapping subclassesDynamo specializes to VariableTracker subclasses like FrozenDataClassVariable if available; if nosubclass qualifies, it falls back to UserDefinedObjectVariable.These classes help Dynamo track and handle arbitrary Python objects during tracing,maintaining proper semantics while enabling optimizations where possible.

This Python file contains 44 class(es) and 109 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UserDefinedVariable`, `UserDefinedClassVariable`, `UserDefinedExceptionClassVariable`, `NO_SUCH_SUBOBJ`, `UserDefinedObjectVariable`, `FrozenDataClassVariable`, `HashWrapper`, `SourcelessGraphModuleVariable`, `UserDefinedExceptionObjectVariable`, `KeyedJaggedTensorVariable`, `IntWrapperVariable`, `RemovableHandleClass`, `RemovableHandleVariable`, `UserDefinedDictVariable`, `UserDefinedSetVariable`, `UserDefinedListVariable`, `UserDefinedTupleVariable`, `MutableMappingVariable`, `RandomVariable`

**Functions defined**: `is_standard_setattr`, `is_standard_delattr`, `is_forbidden_context_manager`, `is_cython_function`, `__init__`, `as_python_constant`, `as_proxy`, `__repr__`, `_constant_fold_classes`, `_in_graph_classes`, `supported_c_new_functions`, `is_supported_new_method`, `can_constant_fold_through`, `has_key_in_generic_dict`, `var_getattr`, `_call_cross_entropy_loss`, `normalize_args`, `fake_cross_entropy_loss`, `call_method`, `call_function`

**Key imports**: _collections, builtins, collections, contextlib, dataclasses, enum, functools, inspect, itertools, random


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `_collections`
- `builtins`
- `collections`
- `contextlib`
- `dataclasses`
- `enum`
- `functools`
- `inspect`
- `itertools`
- `random`
- `sys`
- `threading`
- `types`
- `warnings`
- `weakref`
- `typing`: TYPE_CHECKING
- `typing_extensions`: is_typeddict
- `torch._dynamo.config`
- `torch.nn`
- `torch._guards`: TracingContext
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass_type
- `..`: graph_break_hints, polyfills, variables
- `..bytecode_transformation`: create_call_function
- `..create_parameter_op`: do_not_convert_to_tracable_parameter
- `..graph_bytecode_inputs`: get_external_object_by_index
- `..guards`: GuardBuilder, install_guard
- `.base`: raise_type_error_exc, ValueMutationNew, VariableTracker
- `.dicts`: ConstDictVariable, DefaultDictVariable
- `.lists`: SizeVariable
- `numpy as np`


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

- **File Documentation**: `user_defined.py_docs.md`
- **Keyword Index**: `user_defined.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
