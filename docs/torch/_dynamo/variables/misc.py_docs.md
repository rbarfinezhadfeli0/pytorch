# Documentation: `torch/_dynamo/variables/misc.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/misc.py`
- **Size**: 82,789 bytes (80.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

"""
This module contains miscellaneous variable tracker implementations for various Python types
and features used in Dynamo's symbolic execution. These classes help track and propagate
information about different kinds of variables during graph capture.

Key classes include:
- SuperVariable: Handles super() calls and method resolution
- ExceptionVariable: Tracks exception objects
- RandomVariable: Manages random number generators
- GetAttrVariable: Tracks attribute access
- MethodWrapperVariable: Handles method wrappers
- PythonModuleVariable: Tracks Python modules
- NumpyVariable: Handles numpy functions and types
- StringFormatVariable: Manages string formatting
- DebuggingVariable: Handles print and logging
"""

import dataclasses
import enum
import functools
import inspect
import itertools
import random
import re
import sys
import types
import warnings
from typing import Optional, TYPE_CHECKING

import torch._C
import torch._numpy as tnp
import torch.utils._pytree as pytree

from .. import config, graph_break_hints, trace_rules, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import unpatched_nn_module_init
from ..source import (
    AttrSource,
    GenericAttrSource,
    GetItemSource,
    TypeMROSource,
    TypeSource,
    WeakRefCallSource,
)
from ..utils import (
    check_unspec_or_constant_args,
    cmp_name_to_op_mapping,
    identity,
    is_tensor_base_attr_getter,
    istype,
    list_methods,
    proxy_args_kwargs,
    raise_args_mismatch,
    tuple_methods,
)
from .base import (
    AsPythonConstantNotImplementedError,
    raise_type_error_exc,
    VariableTracker,
)
from .constant import ConstantVariable
from .functions import NestedUserFunctionVariable, UserFunctionVariable
from .user_defined import call_random_fn, is_standard_setattr, UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


class NO_SUCH_SUBOBJ:
    pass


class SuperVariable(VariableTracker):
    _nonvar_fields = {
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, typevar, objvar=None, **kwargs) -> None:
        super().__init__(**kwargs)
        # typevar is the first argument to super(). In the case where no argument
        # is provided to super(), it is the __class__ object where
        # the super() function is being called
        self.typevar = typevar
        # objvar here must be an instance or subtype of typevar.
        # In the case where super() is called without arguments, it is the first argument
        # to the current function where super() is called from (self for regular method,
        # cls for a classmethod)
        self.objvar = objvar

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(lambda: codegen(variables.BuiltinVariable(super)))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            codegen.extend_output(create_call_function(2, False))
        else:
            codegen.extend_output(create_call_function(1, False))

    def _resolved_getattr_and_source(self, tx: "InstructionTranslator", name):
        if not self.objvar:
            unimplemented(
                gb_type="1-arg super not implemented",
                context="",
                explanation=f"Dynamo failed to trace attribute `{name}` accessed "
                f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
                "because one-argument of super() is not supported.",
                hints=[
                    "Use two-argument super(type, object_or_type).",
                ],
            )
        search_type = self.typevar.as_python_constant()

        # The rest of this function does two things:
        #   - Walk the mro to find where the attribute comes from to be
        #     able to provide accurate source
        #   - Call the getattr to get the object

        # Find the class object, where the function lives.
        # When objvar is "self", use type(self), when objvar is "cls", use it as-is
        type_to_use = self.objvar.python_type()
        type_to_use_source = (
            TypeSource(self.objvar.source) if self.objvar.source else None
        )
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value
            type_to_use_source = self.objvar.source

        source = None
        search_mro = type_to_use.__mro__

        try:
            start_index = search_mro.index(search_type) + 1
        except ValueError:
            # Corner case where the typevar is not in the mro of the objvar
            # https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8843-L8844
            return getattr(super(search_type, type_to_use), name), None
        # Implemented based on https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8812
        # super has its getattro implementation. The key point is that instead of calling getattr, it checks the
        # attribute in the class __dict__
        for index in range(start_index, len(search_mro)):
            # Dont call getattr, just check the __dict__ of the class
            if resolved_getattr := search_mro[index].__dict__.get(name, NO_SUCH_SUBOBJ):
                if resolved_getattr is not NO_SUCH_SUBOBJ:
                    # Equivalent of something like type(L['self']).__mro__[1].attr_name
                    if type_to_use_source:
                        source = AttrSource(
                            GetItemSource(TypeMROSource(type_to_use_source), index),
                            name,
                        )
                    return resolved_getattr, source

        unimplemented(
            gb_type="Unable to resolve super getattr",
            context="",
            explanation=f"Dynamo failed to trace attribute `{name}` accessed "
            f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
            "because the resolved attribute type is not supported.",
            hints=[
                "Ensure the attribute exists in the parent class.",
                "Check the arguments passed to `super()`.",
            ],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        # Check if getattr is a constant. If not, delay the actual work by
        # wrapping the result in GetAttrVariable. Mostly super is called with a
        # method, so most of the work is delayed to call_function.
        #
        # We could have just implemented a const_getattr. However, super is
        # special when it comes to finding sources. Compared to other VTs, super
        # requires the attr name to walk the mro and find the actual source (and
        # not just AttrSource).
        value, source = self._resolved_getattr_and_source(self, name)
        if not variables.ConstantVariable.is_literal(value):
            return GetAttrVariable(self, name)
        if source:
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        inner_fn, source = self._resolved_getattr_and_source(self, name)
        # This essentially simulates CPython's `super_getattro`:
        # https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/typeobject.c#L11138-L11168
        # where `inner_fn` is the VT for `res = _super_lookup_descr(...)`.
        #
        # However, `res`'s type needs to be checked for `tp_descr_get`, and
        # applied if it has one. We currently don't have polyfills for all the
        # relevant `tp_descr_get`, so we explicitly handle the cases we care
        # about here (e.g., note the staticmethod, classmethod cases).
        if inner_fn is object.__init__:
            return LambdaVariable(identity)
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew

            if (
                isinstance(objvar, variables.UserDefinedObjectVariable)
                and isinstance(objvar.mutation_type, AttributeMutationNew)
                and not (args or kwargs)
            ):
                with do_not_convert_to_tracable_parameter():
                    fn_vt = VariableTracker.build(
                        tx, unpatched_nn_module_init, source=source
                    )
                    return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported super().__init__() call",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation="Dynamo encountered a super().__init__() call "
                    f"on {objvar} that resolved to a `torch.nn.Module.__init__()` "
                    "call that we cannot trace.",
                    hints=[*graph_break_hints.DIFFICULT],
                )
        elif (
            self.objvar.source
            and hasattr(inner_fn, "__name__")
            and inner_fn.__name__ == "__new__"
            and variables.UserDefinedClassVariable.is_supported_new_method(inner_fn)
        ):
            user_cls = inner_fn.__self__
            if hasattr(user_cls, "__module__") and user_cls.__module__ == "builtins":
                user_cls_vt = variables.BuiltinVariable(user_cls)
            else:
                user_cls_source = source.member
                user_cls_vt = variables.UserDefinedClassVariable(
                    user_cls, source=user_cls_source
                )
            return user_cls_vt.call_method(tx, "__new__", args, kwargs)
        elif isinstance(inner_fn, staticmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            fn_vt = VariableTracker.build(tx, inner_fn.__func__, source=source)
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(inner_fn, classmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            if isinstance(self.objvar, variables.UserDefinedClassVariable):
                # super().classmethod is called from a classmethod itself. So,
                # super was converted to super(__class__, cls) in bytecode and
                # therefore we have to propagate the cls.
                cls_variable = self.objvar
            else:
                # current function is an instance method, therefore super was
                # converted to super(__class__, self). We have to find
                # type(self) to bind the cls to the parent classmethod.
                # Note that it can't be the self.typevar because __class__ is
                # the class where the method is defined, which could be
                # different from type(self) with polymorphism.
                cls_source = None
                if self.objvar.source:
                    cls_source = TypeSource(self.objvar.source)
                cls_variable = VariableTracker.build(
                    tx, self.objvar.value_type, cls_source
                )

            fn_vt = VariableTracker.build(
                tx, inner_fn.__func__, source=AttrSource(source, "__func__")
            )
            return fn_vt.call_function(tx, [cls_variable, *args], kwargs)
        elif isinstance(inner_fn, types.FunctionType):
            fn_vt = VariableTracker.build(tx, inner_fn, source=source)
            return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source
            ).call_function(tx, args, kwargs)
        elif is_standard_setattr(inner_fn) and isinstance(
            self.objvar, UserDefinedObjectVariable
        ):
            return self.objvar.method_setattr_standard(tx, *args, **kwargs)
        elif inner_fn is object.__delattr__:
            attr = args[0]
            try:
                attr = attr.as_python_constant()
            except NotImplementedError as exc:
                unimplemented(
                    gb_type="Non-constant attribute given to `super().__delattr__()`",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo requires the attribute name passed to "
                    "`super().__delattr__(...)` to be a constant (string).",
                    hints=[
                        "Ensure the attribute name is a string literal or a constant variable."
                    ],
                    from_exc=exc,
                )
            if not tx.output.side_effects.is_attribute_mutation(self.objvar):
                unimplemented(
                    gb_type="Attempted super().__delattr__() on an object without mutation tracking",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo needs to track mutations on an object "
                    "before `super().__delattr__` can be used on it. But the "
                    f"object ({self.objvar}) doesn't have attribute mutation "
                    "tracking enabled.",
                    hints=[
                        "Ensure the object is tracked by Dynamo's side effect system.",
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )

            tx.output.side_effects.store_attr(
                self.objvar, attr, variables.DeletedVariable()
            )
            return variables.ConstantVariable(None)
        elif (
            isinstance(self.objvar, variables.UserDefinedDictVariable)
            and inner_fn in self.objvar._dict_methods
        ):
            return self.objvar._dict_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedSetVariable)
            and inner_fn in self.objvar._set_methods
        ):
            return self.objvar._set_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedTupleVariable)
            and inner_fn in tuple_methods
        ):
            return self.objvar._tuple_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedListVariable)
            and inner_fn in list_methods
        ):
            return self.objvar._list_vt.call_method(tx, name, args, kwargs)
        elif inner_fn is object.__getattribute__:
            # object.__getattribute__ has no side-effects. We can directly call
            # __getattribute__ to access the attribute.
            attr_name = args[0].value
            if tx.output.side_effects.has_pending_mutation_of_attr(
                self.objvar, attr_name
            ):
                result = tx.output.side_effects.load_attr(
                    self.objvar, attr_name, deleted_ok=True
                )
                if isinstance(result, variables.DeletedVariable):
                    raise_observed_exception(AttributeError, tx)
                return result

            try:
                # NB - use object.__getattribute__ to prevent running any user code
                attr_value = object.__getattribute__(self.objvar.value, attr_name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)

            attr_source = None
            if self.objvar.source is not None:
                # setup a object.__getattribute__(self.objvar, name) source
                attr_source = GenericAttrSource(self.objvar.source, attr_name)
            return VariableTracker.build(tx, attr_value, attr_source)
        elif inner_fn is torch._C._disabled_torch_function_impl:
            # See `THPModule_disable_torch_function` for the C impl.
            # The signature of _disabled_torch_function_impl is similar to
            # `__torch_function__`, just without the first `cls` argument:
            #  * (func, types, args, kwargs)
            func = args[0]
            tf_kwargs = {}
            tf_args = args[2].items
            for hash_key_vt, value_vt in args[3].items.items():
                key_str = hash_key_vt.vt.as_python_constant()
                tf_kwargs[key_str] = value_vt

            tx_old = tx.symbolic_torch_function_state.torch_function_subclass_enabled
            tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
            try:
                return func.call_function(tx, tf_args, tf_kwargs)
            finally:
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    tx_old
                )
        elif (
            isinstance(inner_fn, types.MethodDescriptorType)
            and inner_fn in trace_rules.get_tensor_method()
        ):
            # FunctionType but implementation is in C, we support some of these,
            # e.g., tensor ops like `torch.Tensor.to`.
            fn_var = VariableTracker.build(tx, inner_fn, source)
            return fn_var.call_function(tx, [self.objvar] + args, kwargs)

        unimplemented(
            gb_type="Attempted to call a super() attribute that is "
            "not a function or method",
            context=f"call_method {self} {name}",
            explanation="Dynamo does not know how to trace the call "
            f"`super().{name}()` because `super().{name}` is not a "
            "function or method attribute.",
            hints=[
                "Ensure the attribute accessed via `super()` is a standard method or function.",
            ],
        )


class ExceptionVariable(VariableTracker):
    # The ExceptionVariable corresponds to the BaseException class in Python
    def __init__(
        self, exc_type, args, init_kwargs=None, source=None, mutation_type=None
    ) -> None:
        super().__init__(source=source, mutation_type=mutation_type)
        self.exc_type = exc_type
        self.args = args
        if init_kwargs:
            unimplemented(
                gb_type="Keyword args passed to exception constructor",
                context=f"{self} with kwargs {init_kwargs}",
                explanation="Dynamo does not know how to handle keyword args passed to an exception constructor",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # When raising a new exception while another exception is already being
        # handled, the new exception's __context__ attribute is automatically
        # set to the handled exception.
        self.__context__ = ConstantVariable(None)
        # Set when user raised an exception from another:
        # raise ... from ...
        self.__cause__ = ConstantVariable(None)
        # Boolean flag that controls whether the __context__ attribute is set
        self.__suppress_context__ = ConstantVariable(False)
        # Contains the call stack where the exception was raised. Dynamo does
        # not track traceback. So, this variable is always set to None
        self.__traceback__ = ConstantVariable(None)

    def set_context(self, context: "ExceptionVariable"):
        self.__context__ = context

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", self.exc_type.__name__)
        )
        codegen.foreach(self.args)
        codegen.call_function(len(self.args), False)

        def codegen_attr(name: str) -> None:
            attr = getattr(self, name)
            if istype(attr, ConstantVariable):
                assert attr.value in (True, False, None), attr
            else:
                codegen.dup_top()
                codegen(attr)
                codegen.extend_output(codegen.rot_n(2))
                codegen.store_attr(name)

        codegen_attr("__context__")
        codegen_attr("__cause__")
        codegen_attr("__suppress_context__")

    def python_type(self):
        return self.exc_type

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        name_var: VariableTracker,
        val: VariableTracker,
    ):
        def raise_error(msg):
            raise_observed_exception(TypeError, tx, args=[ConstantVariable(msg)])

        name = name_var.as_python_constant()
        if name == "__context__":
            self.set_context(val)
        elif name == "__cause__":
            if (isinstance(val, ConstantVariable) and val.value is None) or isinstance(
                val,
                (
                    variables.BuiltinVariable,
                    variables.ExceptionVariable,
                    variables.UserDefinedExceptionClassVariable,
                    variables.UserDefinedExceptionObjectVariable,
                ),
            ):
                self.__cause__ = val
                self.__suppress_context__ = variables.ConstantVariable(True)
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__suppress_context__":
            if isinstance(val, ConstantVariable) and val.value in (True, False):
                self.__suppress_context__ = val
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__traceback__":
            if isinstance(val, ConstantVariable) and val.value is None:
                self.__traceback__ = val
            else:
                unimplemented(
                    gb_type="Set Exception object `__traceback__` attribute to not-`None`",
                    context=f"call_setattr {self} {name}",
                    explanation="Dynamo does not support setting the attribute "
                    "'__traceback__' on tracked exception objects to anything "
                    "other than None.",
                    hints=[
                        "Avoid setting '__traceback__' on exception objects "
                        "within traced code, or set it to None."
                    ],
                )
        else:
            unimplemented(
                gb_type="Unsupported attribute assignment on Exception object",
                context=f"call_setattr {self} {name}",
                explanation="Dynamo does not support setting the attribute "
                f"'{name}' on tracked exception objects. Only `__context__`, "
                "`__cause__`, `__suppress_context__`, and `__traceback__` are supported.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return variables.ConstantVariable(None)

    def call_method(self, tx, name, args, kwargs):
        if name == "__setattr__":
            return self.call_setattr(tx, *args)
        elif name == "with_traceback":
            [tb] = args
            self.call_setattr(tx, ConstantVariable("__traceback__"), tb)
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        if name == "__context__":
            return self.__context__
        elif name == "__cause__":
            return self.__cause__
        elif name == "__suppress_context__":
            return self.__suppress_context__
        elif name == "__traceback__":
            return variables.ConstantVariable(None)
        elif name == "args":
            return variables.ListVariable(self.args, source=self.source)
        return super().var_getattr(tx, name)

    def __str__(self):
        return f"{self.__class__.__name__}({self.exc_type})"

    __repr__ = __str__


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """


class DelayGraphBreakVariable(UnknownVariable):
    """
    Used to insert a dummy variable in the stack to do the graph break at CALL_FUNCTION.
    """

    def __init__(self, msg=None, **kwargs):
        super().__init__(**kwargs)
        self.msg = msg

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented(
            gb_type="Unsupported function call (delayed)",
            context=f"source: {self.source}",
            explanation="Dynamo determined that a graph break should occur "
            f"when calling `{self.source.name()}`. Reason: {self.msg}",
            hints=[],
        )


class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen: "PyCodegen"):
        raise NotImplementedError("comptime is special form")

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        from ..comptime import comptime

        # To support the comptime.print_graph convenience accessors
        return VariableTracker.build(
            tx, getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well
        # Second argument is runtime lambda, ignored
        if kwargs or len(args) > 2:
            raise_args_mismatch(
                tx,
                "comptime()",
                "at most 2 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            # We have to manually bind the freevars ourselves
            code = fn.get_code()
            if fn.closure:
                raise_type_error_exc(
                    tx,
                    f"comptime function must not have free variables, but these variables were free: {code.co_freevars}",
                )
            func = types.FunctionType(
                code,
                fn.f_globals,
                fn.fn_name.as_python_constant(),
                tuple(fn.defaults.items) if fn.defaults else None,
                # We could automatically promote free variables into
                # ComptimeVar but this is confusing if you access
                # a free variable that we actually DO have the runtime
                # value for
                # tuple(make_cell(ComptimeVar(i)) for i in fn.closure.items)
                (),
            )
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f"unsupported argument to comptime: {type(fn)}")

        return variables.ConstantVariable.create(None)


class CellVariable(VariableTracker):
    # If the cell existed before Dynamo tracing started, this will be the
    # VariableTracker that represents the cell content.
    #
    # Note that all mutation to the cell (i.e., its content) will be buffered in
    # SideEffects, rather than being reflected here. One can think of
    # `CellVariable` as a special case for `UserDefinedObjectVariable`.
    pre_existing_contents: Optional[VariableTracker]

    # This is set when this cell can be referenced via `LOAD/STORE_DEREF` in the
    # root frame via this name (e.g., the name is in `co_cellvars/co_freevars`).
    local_name: Optional[str] = None

    def __init__(
        self, pre_existing_contents: Optional[VariableTracker] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pre_existing_contents = pre_existing_contents


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def produce_trampoline_autograd_apply(fn_cls):
    def trampoline_autograd_apply(*args, **kwargs):
        return fn_cls.apply(*args, **kwargs)

    trampoline_autograd_apply._origin = produce_trampoline_autograd_apply
    return trampoline_autograd_apply


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    _nonvar_fields = {
        "fn_cls",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, fn_cls, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx: "InstructionTranslator", args, kwargs):
        requires_grad = False

        def visit(vt):
            nonlocal requires_grad
            if isinstance(vt, variables.TensorVariable):
                if vt.requires_grad is not False:
                    requires_grad = True
            if isinstance(vt, variables.NNModuleVariable):
                if vt.is_training(tx):
                    requires_grad = True

        VariableTracker.visit(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            if config.capture_autograd_function is False:
                warnings.warn(
                    "The config.capture_autograd_function flag is deprecated, it's now always true."
                )

            from torch._functorch.autograd_function import (
                autograd_function_forward_rewritten,
            )
            from torch.autograd.function import _is_setup_context_defined

            forward_fn = self.fn_cls.forward

            is_setup_ctx_defined = _is_setup_context_defined(self.fn_cls.setup_context)
            if is_setup_ctx_defined:
                # If setup_context is defined, we generate a new forward function which includes
                # the original forward and setup_context function, and trace the new forward function.
                forward_fn = autograd_function_forward_rewritten(
                    self.fn_cls.forward, self.fn_cls.setup_context
                )

            vjp_fn = self.fn_cls.vjp  # type: ignore[attr-defined]
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented(
                    gb_type="Unsupported custom vjp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `vjp` method.",
                    hints=[
                        "Remove the custom `vjp` method if possible.",
                        "Use standard `backward` instead if applicable.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            jvp_fn = self.fn_cls.jvp  # type: ignore[attr-defined]
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented(
                    gb_type="Unsupported custom jvp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `jvp` method.",
                    hints=[
                        "Remove the custom `jvp` method if possible.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            from .higher_order_ops import AutogradFunctionApplyVariable

            source = self.source
            if source is None:
                source = AttrSource(
                    tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__
                )

            val = AutogradFunctionApplyVariable(
                forward_fn,
                self.fn_cls.backward,
                source,
                source=AttrSource(source, member="apply"),
            ).call_function(tx, args, kwargs)
            # Inside of AutogradFunctionApplyVariable.call_function, we use sourceless variable wrapping
            # the forward function, as we don't want to generate guards for new_forward.__closure__
            # if forward is rewritten by autograd_function_forward_rewritten.
            # But we still need to generate correct guards for the original forward and setup_context
            # functions, so we have to add guards manually.
            if self.source:
                fwd_src = AttrSource(self.source, "forward")
                install_guard(fwd_src.make_guard(GuardBuilder.CLOSURE_MATCH))
                if is_setup_ctx_defined:
                    setup_ctx_src = AttrSource(self.source, "setup_context")
                    install_guard(setup_ctx_src.make_guard(GuardBuilder.CLOSURE_MATCH))

            return val

        if self.source:
            source = AttrSource(self.source, "forward")
        else:
            source = None

        fn = self.fn_cls.forward
        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        args = [ctx, *args]
        if isinstance(fn, types.FunctionType):
            sig = inspect.signature(fn)
            if len(args) - 1 == len(sig._parameters):
                args = args[1:]  # Don't use context
            fn_vt = VariableTracker.build(tx, fn, source=source)
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__,
                variables.UserDefinedClassVariable(self.fn_cls),
                source=source,
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                gb_type="Non-function or method in subclass of torch.autograd.Function",
                context=f"call_apply {self} {args} {kwargs}",
                explanation="Dynamo requires the `forward` attribute of a "
                "`torch.autograd.Function` subclass to be a standard Python "
                f"function or method. Found type `{type(fn).__name__}` instead.",
                hints=[
                    "Ensure the `forward` method is defined as a regular "
                    "function or instance method."
                ],
            )

    def call_backward(self, tx: "InstructionTranslator", args, kwargs):
        fn = self.fn_cls.backward
        assert type(args[0].value) is torch._dynamo.external_utils.FakeBackwardCFunction
        assert isinstance(fn, types.FunctionType)

        fn_source = AttrSource(self.source, "backward")
        fn_vt = VariableTracker.build(tx, fn, source=fn_source)
        return fn_vt.call_function(tx, args, kwargs)

    def call_function(self, tx: "InstructionTranslator", args, kwargs):
        return AutogradFunctionVariable(self.fn_cls)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ):
        from .builder import wrap_fx_proxy

        if name == "apply":
            if trace_rules.is_callable_allowed(self.fn_cls):
                trampoline_autograd_apply = produce_trampoline_autograd_apply(
                    self.fn_cls
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        trampoline_autograd_apply,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
            else:
                return self.call_apply(tx, args, kwargs)

        elif name == "backward":
            return self.call_backward(tx, args, kwargs)
        else:
            source = AttrSource(self.source, name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                func = obj.__get__(self.fn_cls)
                if source is not None:
                    return (
                        trace_rules.lookup(func)
                        .create_with_source(func, source=source)
                        .call_function(tx, args, kwargs)
                    )
                else:
                    return trace_rules.lookup(func)(func).call_function(
                        tx, args, kwargs
                    )
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported autograd.Function method",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo does not support calling the method "
                    f"`{name}` directly on the `torch.autograd.Function` "
                    "instance. Supported methods include `apply`, `backward`, "
                    "static methods, and class methods.",
                    hints=[
                        "Ensure the method is decorated with `@staticmethod` "
                        "or `@classmethod` if it's meant to be called on the class.",
                    ],
                )


@dataclasses.dataclass
class SavedTensorBox:
    tensors: list[VariableTracker] = dataclasses.field(default_factory=list)


class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """

    _nonvar_fields = {
        "proxy",
        "inference",
        "saved_tensors",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value,
        value_type=None,
        inference=False,
        saved_tensors=None,
        needs_input_grad=None,
        non_differentiable=None,
        **kwargs,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference
        self.saved_tensors = saved_tensors
        self.needs_input_grad = needs_input_grad
        self.non_differentiable = non_differentiable

    @staticmethod
    def create(tx: "InstructionTranslator", args=None, kwargs=None):
        needs_input_grad = None
        if args and not kwargs:
            needs_input_grad = tuple(
                isinstance(x, variables.TensorVariable) and x.requires_grad
                for x in args
            )
        out = tx.output.side_effects.track_object_new(
            None,
            torch.autograd.function.FunctionCtx,
            functools.partial(
                AutogradFunctionContextVariable,
                inference=True,
                saved_tensors=SavedTensorBox(),
                needs_input_grad=needs_input_grad,
            ),
            {},
        )
        return out

    def as_proxy(self):
        if self.proxy is None:
            unimplemented(
                gb_type="proxy not set",
                context=f"as_proxy {self}",
                explanation="Dynamo requires the autograd.Function context "
                "to be initialized with a proxy.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        return self.proxy

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__setattr__":
            return super().call_method(tx, name, args, kwargs)
        elif name == "mark_non_differentiable":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            self.non_differentiable = proxy_args_kwargs(args, {})[0]
            return variables.ConstantVariable.create(None)

        if name != "save_for_backward":
            unimplemented(
                gb_type="Unsupported autograd.Function context method",
                context=f"call_method {self} {name}",
                explanation="Dynamo does not support calling the method "
                f"`{name}` on `autograd.Function` context objects. Supported "
                "methods are `__setattr__`, `save_for_backward` and "
                "`mark_non_differentiable`.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if self.saved_tensors is None:
            unimplemented(
                gb_type="Unsupported autograd.Function context `save_for_backward`",
                context=f"call_method {self} {name}",
                explanation="Dynamo requires the `saved_tensors` attribute "
                "to be initialized on the `autograd.Function` context object.",
                hints=[
                    "Ensure that the `saved_tensors` attribute is properly "
                    "initialized before calling `save_for_backward`. "
                    "`save_for_backward` only supported on a newly constructed `torch.autograd.function.FunctionCtx`.",
                ],
            )

        if not self.inference:
            if kwargs or not self.source:
                raise_type_error_exc(
                    tx, "save_for_backward() requires a source and no keyword arguments"
                )
            tx.output.side_effects.track_save_for_backward(self, args)

        # In eager mode, multiple calls to .save_for_backward() will overwrite previous calls.
        if len(self.saved_tensors.tensors) > 0:
            self.saved_tensors.tensors = []
        for arg in args:
            self.saved_tensors.tensors.append(arg)
        return variables.ConstantVariable.create(None)

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name in ["save_for_backward", "mark_non_differentiable"]:
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            )
        if name == "saved_tensors" and self.saved_tensors is not None:
            return variables.TupleVariable(list(self.saved_tensors.tensors))
        if name == "needs_input_grad":
            if self.needs_input_grad is not None:
                return variables.ConstantVariable.create(self.needs_input_grad)
            if self.source:
                source = AttrSource(self.source, "needs_input_grad")
                return VariableTracker.build(tx, self.value.needs_input_grad, source)

        return super().var_getattr(tx, name)


class AutogradEngineVariable(UserDefinedObjectVariable):
    """
    Represents a torch._C._ImperativeEngine instance.
    """

    def __init__(
        self,
        value,
        value_type=None,
        **kwargs,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "queue_callback":
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                assert tx.one_graph or tx.error_on_graph_break, (
                    "queue_callback() is only supported when Compiled Autograd is enabled with fullgraph=True"
                )
                # queue_callback is a method-wrapper, no need to insert a guard.
                fn_vt = VariableTracker.build(
                    tx,
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.queue_callback,
                )
                return fn_vt.call_function(
                    tx,
                    (tx.output.side_effects.get_ca_final_callbacks_var(), *args),
                    kwargs,
                )
            else:
                unimplemented(
                    gb_type="Unsupported torch._C._ImperativeEngine.queue_callback()",
                    context=f"call_method {self} {name}",
                    explanation="queue_callback() is only supported when "
                    "Compiled Autograd is enabled with fullgraph=True.",
                    hints=[],
                )
        else:
            unimplemented(
                gb_type="Unsupported torch._C._ImperativeEngine method",
                context=f"call_method {self} {name}",
                explanation="Dynamo only supports the `queue_callback` method "
                f"on a torch._C._ImperativeEngine instance, but found: `{name}`.",
                hints=[],
            )


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn = fn

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return self.fn(*args, **kwargs)


class GetAttrVariable(VariableTracker):
    _nonvar_fields = {
        "name",
        "py_type",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, obj, name, py_type=None, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name
        self.py_type = py_type  # In some cases we know the type (ex. tensor methods)

    def python_type(self):
        if self.py_type is not None:
            return self.py_type
        else:
            return super().python_type()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr):
        return getattr(base_proxy, attr)

    def as_proxy(self):
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def as_python_constant(self):
        constant = self.obj.as_python_constant()
        try:
            return getattr(constant, self.name)
        except AttributeError:
            raise NotImplementedError(f"{self} is not a constant") from None

    def const_getattr(self, tx: "InstructionTranslator", name):
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen: "PyCodegen"):
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return self.obj.call_method(tx, self.name, args, kwargs)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name in ("__getitem__", "get")
            and self.name == "__dict__"
            and not kwargs
            and args[0].is_python_constant()
            and isinstance(
                self.obj,
                (
                    variables.UserDefinedObjectVariable,
                    variables.NNModuleVariable,
                    variables.UserDefinedClassVariable,
                ),
            )
        ):
            obj = self.obj
            key = args[0].as_python_constant()
            if obj.has_key_in_generic_dict(tx, key):
                # redirect to var_getattr on the original obj
                return obj.var_getattr(tx, key)

            # Return the default value for get
            if name == "get":
                if len(args) == 2:
                    return args[1]
                else:
                    return variables.ConstantVariable(None)

        elif (
            name == "__contains__"
            and self.name == "__dict__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
            and isinstance(
                self.obj,
                (
                    variables.UserDefinedObjectVariable,
                    variables.NNModuleVariable,
                    variables.UserDefinedClassVariable,
                ),
            )
        ):
            obj = self.obj
            key = args[0].as_python_constant()
            if obj.has_key_in_generic_dict(tx, key):
                return variables.ConstantVariable(True)
            else:
                return variables.ConstantVariable(False)

        elif name == "__setitem__" and self.name == "__dict__" and not kwargs:
            if isinstance(self.obj, variables.UserDefinedObjectVariable):
                # Bypass any custom setattr as we are updating the `__dict__` itself
                return self.obj.method_setattr_standard(
                    tx, args[0], args[1], directly_update_dict=True
                )
            if isinstance(self.obj, variables.NNModuleVariable):
                # This matches how `setattr` is handled for NNModuleVariable
                self.obj.convert_to_unspecialized(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_forwarded_dict(self, tx):
        assert (
            self.na
```



## High-Level Overview

"""This module contains miscellaneous variable tracker implementations for various Python typesand features used in Dynamo's symbolic execution. These classes help track and propagateinformation about different kinds of variables during graph capture.Key classes include:- SuperVariable: Handles super() calls and method resolution- ExceptionVariable: Tracks exception objects- RandomVariable: Manages random number generators- GetAttrVariable: Tracks attribute access- MethodWrapperVariable: Handles method wrappers- PythonModuleVariable: Tracks Python modules- NumpyVariable: Handles numpy functions and types- StringFormatVariable: Manages string formatting- DebuggingVariable: Handles print and logging

This Python file contains 41 class(es) and 114 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NO_SUCH_SUBOBJ`, `SuperVariable`, `ExceptionVariable`, `UnknownVariable`, `DelayGraphBreakVariable`, `ComptimeVariable`, `CellVariable`, `NewGlobalVariable`, `AutogradFunctionVariable`, `SavedTensorBox`, `AutogradFunctionContextVariable`, `AutogradEngineVariable`, `LambdaVariable`, `GetAttrVariable`, `MethodWrapperVariable`, `GetSetDescriptorVariable`, `PythonModuleVariable`, `TypingVariable`, `NumpyVariable`, `NullVariable`

**Functions defined**: `__init__`, `reconstruct`, `_resolved_getattr_and_source`, `var_getattr`, `call_method`, `__init__`, `set_context`, `reconstruct`, `codegen_attr`, `python_type`, `call_setattr`, `raise_error`, `call_method`, `var_getattr`, `__str__`, `__init__`, `call_function`, `reconstruct`, `var_getattr`, `call_function`

**Key imports**: dataclasses, enum, functools, inspect, itertools, random, re, sys, types, warnings


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `enum`
- `functools`
- `inspect`
- `itertools`
- `random`
- `re`
- `sys`
- `types`
- `warnings`
- `typing`: Optional, TYPE_CHECKING
- `torch._C`
- `torch._numpy as tnp`
- `torch.utils._pytree as pytree`
- `..`: config, graph_break_hints, trace_rules, variables
- `..create_parameter_op`: do_not_convert_to_tracable_parameter
- `..exc`: raise_observed_exception, unimplemented
- `..guards`: GuardBuilder, install_guard
- `..mutation_guard`: unpatched_nn_module_init
- `.constant`: ConstantVariable
- `.functions`: NestedUserFunctionVariable, UserFunctionVariable
- `.user_defined`: call_random_fn, is_standard_setattr, UserDefinedObjectVariable
- `torch._dynamo.codegen`: PyCodegen
- `torch._dynamo.symbolic_convert`: InstructionTranslator
- `..side_effects`: AttributeMutationNew
- `..comptime`: comptime
- `torch.autograd.function`: _is_setup_context_defined
- `.higher_order_ops`: AutogradFunctionApplyVariable
- `.builder`: wrap_fx_proxy


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `misc.py_docs.md`
- **Keyword Index**: `misc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
