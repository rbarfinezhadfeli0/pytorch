# Documentation: `torch/_dynamo/variables/functions.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/functions.py`
- **Size**: 103,473 bytes (101.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Function-related variable tracking classes for Dynamo's symbolic execution.

This module contains classes that track different types of functions during graph
compilation, including:
- User-defined functions and methods
- Built-in functions and methods
- Wrapped functions (e.g. from decorators)
- Special function types (e.g. functools.partial)
- Triton kernels and related function types

These classes are responsible for:
- Tracking function calls and their arguments
- Managing function closures and cell variables
- Handling function attributes and special methods
- Maintaining guards for function identity and closure contents
- Supporting function inlining and specialization
- Enabling proper symbolic execution of different function types

The variable trackers here work together with the rest of Dynamo to enable
accurate graph capture while handling Python's various function-related behaviors.
"""

import builtins
import functools
import inspect
import itertools
import logging
import sys
import traceback
import types
from collections.abc import Callable, Sequence
from types import CellType, FunctionType
from typing import Any, Optional, TYPE_CHECKING, TypeVar
from typing_extensions import Never
from weakref import WeakKeyDictionary

import torch
from torch._dynamo.exc import get_stack_above_dynamo
from torch._guards import Source

from .. import config, graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_rot_n, is_generator
from ..exc import (
    format_skip_frame_message,
    get_dynamo_observed_exception,
    handle_observed_exception,
    InfiniteGeneratorError,
    ObservedException,
    ObservedGeneratorExit,
    ObservedUserStopIteration,
    raise_observed_exception,
    SkipFrame,
    StepUnsupported,
    unimplemented,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ClosureSource,
    ConstantSource,
    DefaultsSource,
    GetItemSource,
    SkipGuardSource,
)
from ..utils import (
    check_constant_args,
    check_unspec_or_constant_args,
    cmp_name_to_op_mapping,
    identity,
    is_function,
    is_wrapper_or_member_descriptor,
    istype,
    make_cell,
)
from .base import (
    AsPythonConstantNotImplementedError,
    AttributeMutationNew,
    raise_type_error_exc,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable


try:
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import (
        InstructionTranslator,
        InstructionTranslatorBase,
    )
    from torch._dynamo.variables.ctx_manager import ContextWrappingVariable
    from torch._higher_order_ops.triton_kernel_wrap import (
        TritonGridType,
        TritonKernelType,
    )

    from .lists import BaseListVariable, ListVariable
    from .tensor import TensorVariable


_F = TypeVar("_F", bound=Callable[..., Any])
CO_VARARGS = 0x04
CO_VARKEYWORDS = 0x08


# Module-level cache keyed by the function object
_spec_cache: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


class FunctionSpec:
    def __init__(self, func: FunctionType):
        code = func.__code__
        vn = code.co_varnames

        self.posonly_count = code.co_posonlyargcount
        self.arg_count = code.co_argcount
        self.kwonly_count = code.co_kwonlyargcount

        self.posonly_names = vn[: self.posonly_count]
        self.pos_or_kw_names = vn[self.posonly_count : self.arg_count]
        self.all_pos_names = self.posonly_names + self.pos_or_kw_names
        self.kwonly_names = vn[self.arg_count : self.arg_count + self.kwonly_count]

        off = self.arg_count + self.kwonly_count
        self.varargs_name = vn[off] if code.co_flags & CO_VARARGS else None
        off += 1 if self.varargs_name else 0
        self.varkw_name = vn[off] if code.co_flags & CO_VARKEYWORDS else None

    def update_defaults(self, func: FunctionType) -> None:
        # Defaults can change from function call to function call. So re-update
        # them on every call.
        self.defaults = func.__defaults__ or ()
        self.kwdefaults = func.__kwdefaults__ or {}

        # Map positional-default names → their index in self.defaults
        self.pos_default_map = dict(
            zip(self.all_pos_names[-len(self.defaults) :], range(len(self.defaults)))
        )


def _get_spec(func: FunctionType) -> FunctionSpec:
    spec = _spec_cache.get(func)
    if spec is None:
        spec = FunctionSpec(func)
        _spec_cache[func] = spec
    return spec


def bind_args_cached(
    func: FunctionType,
    tx: "InstructionTranslator",
    fn_source: Optional[Source],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> dict[str, VariableTracker]:
    spec = _get_spec(func)
    spec.update_defaults(func)
    ba = {}
    rem_kw = dict(kwargs)

    # 1) Bind all positional (pos-only + pos-or-kw)
    # 1.1) Apply pos-defaults first (maybe overridden later)
    for name, idx in spec.pos_default_map.items():
        default_source = None
        if fn_source and not (
            ConstantVariable.is_literal(spec.defaults[idx])
            and config.skip_guards_on_constant_func_defaults
        ):
            default_source = DefaultsSource(fn_source, idx)
        ba[name] = wrap_bound_arg(tx, spec.defaults[idx], default_source)
    # 1.2) Fill in provided positional args
    for i, name in enumerate(spec.all_pos_names):
        if i < len(args):
            # Maybe override pos-defaults applied above
            ba[name] = wrap_bound_arg(tx, args[i])
        elif name in rem_kw and (
            # `kwargs` can have the same key as a pos-only arg `name`.
            # If this case happens, we should not consume the `name` here and
            # keep it in `kwargs`:
            #   >>> def fn(a, /, **kwargs): return (a, kwargs)
            #   >>> fn(1, a=2)
            #   (1, {'a': 2})
            name not in spec.posonly_names
        ):
            # Maybe override pos-defaults applied above
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name not in ba:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    ConstantVariable.create(
                        f"Missing required positional argument: {name}"
                    )
                ],
            )

    # 2) *args
    extra = args[len(spec.all_pos_names) :]
    if spec.varargs_name:
        ba[spec.varargs_name] = wrap_bound_arg(tx, tuple(extra))
    elif extra:
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                ConstantVariable.create(
                    f"Too many positional arguments: got {len(args)}, expected {len(spec.all_pos_names)}"
                )
            ],
        )

    # 3) Keyword-only
    for name in spec.kwonly_names:
        if name in rem_kw:
            ba[name] = wrap_bound_arg(tx, rem_kw.pop(name))
        elif name in spec.kwdefaults:
            kwdefault_source = None
            if fn_source:
                kwdefault_source = DefaultsSource(fn_source, name, is_kw=True)
            ba[name] = wrap_bound_arg(tx, spec.kwdefaults[name], kwdefault_source)
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    ConstantVariable.create(
                        f"Missing required keyword-only argument: {name}"
                    )
                ],
            )

    # 4) **kwargs
    if spec.varkw_name:
        ba[spec.varkw_name] = wrap_bound_arg(tx, rem_kw)
    elif rem_kw:
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                ConstantVariable.create(f"Unexpected keyword arguments: {list(rem_kw)}")
            ],
        )

    return ba


def wrap_bound_arg(
    tx: "InstructionTranslator", val: Any, source: Optional[Source] = None
) -> VariableTracker:
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        return VariableTracker.build(tx, val)
    else:
        # Create a lazy variable to avoid guarding on __defaults__ unless really
        # needed.
        return variables.LazyVariableTracker.create(val, source)


def wrap_args_kwargs(tx: "InstructionTranslator", result: dict[str, Any]) -> None:
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(
    parent: "InstructionTranslator",
    result: dict[str, VariableTracker],
    code: types.CodeType,
) -> None:
    """
    Update `result` to add mapping from local name to new cells created
    directly by `code`, or update SideEffects in `parent` if the a local cell is
    already in `result` (cell argument).
    """
    side_effects = parent.output.side_effects

    for name in code.co_cellvars:
        new_cell = side_effects.track_cell_new()
        if name in result:
            # This handles when a function argument is a cell (e.g., captured by
            # a nested func). See `MAKE_CELL` bytecode for more info.
            side_effects.store_cell(new_cell, result.pop(name))
        result[name] = new_cell


def _create_nested_fn(
    code: types.CodeType,
    f_globals: dict[str, Any],
    name: str,
    defaults: Optional[tuple[object, ...]],
    closure: Optional[tuple[CellType]],
    kwdefaults: Optional[dict[str, Any]],
    annotations: Optional[dict[str, Any]],
) -> types.FunctionType:
    from types import FunctionType

    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults

    if isinstance(annotations, tuple):
        from itertools import pairwise

        annotations = dict(pairwise(annotations))

    # TypeError: __annotations__ must be set to a dict object
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations  # type: ignore[assignment]

    return func


fn_known_dunder_attrs = {
    "__annotations__",
    "__defaults__",
    "__kwdefaults__",
    "__code__",
    "__globals__",
    "__closure__",
    "__doc__",
}


def fn_var_getattr(
    tx: "InstructionTranslator", fn: object, source: Optional[Source], name: str
) -> VariableTracker:
    source = source and AttrSource(source, name)

    if source and name == "__annotations__":
        # We get a large number of silly guards from annotations from inspect
        # module. Changing annotations is rare, and it impacting the extracted
        # graph is even rarer. So skip guards.
        source = SkipGuardSource(source)

    subobj = None
    try:
        subobj = inspect.getattr_static(fn, name)
    except AttributeError:
        # function does not have a __getattr__ or __getattribute__ method,
        # so we can safely assume that this attribute is absent
        raise_observed_exception(AttributeError, tx)

    # Special handling for known dunder attributes
    if name in fn_known_dunder_attrs:
        subobj = getattr(fn, name)
    if source:
        return variables.LazyVariableTracker.create(subobj, source)
    return VariableTracker.build(tx, subobj)


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self) -> str:
        return self.get_code().co_filename  # type: ignore[attr-defined]

    def get_name(self) -> str:
        return self.get_code().co_name  # type: ignore[attr-defined]

    def get_globals(self):
        raise NotImplementedError

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)  # type: ignore[attr-defined]

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        result = False

        try:
            result = hasattr(self.get_function(), name)  # type: ignore[attr-defined]
        except NotImplementedError:
            if name == "__name__" and isinstance(self, NestedUserFunctionVariable):
                result = True
        return variables.ConstantVariable.create(result)

    def closure_vars(self, tx: "InstructionTranslator") -> dict[str, VariableTracker]:
        return {}

    # Override to set whether or not nested graph breaks should be allowed
    # if we create an inlining tx for this BaseUserFunctionVariable.
    # See symbolic_convert.py for where this function is called.
    def should_allow_nested_graph_breaks(self):
        return True


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    _nonvar_fields = {
        "fn",
        "is_constant",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Any) -> "UserFunctionVariable":
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(value, source=source)

    def __init__(
        self,
        fn: types.FunctionType | torch.jit.ScriptFunction,  # type: ignore[type-arg]
        is_constant: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        # TODO putting this here to avoid duplication, because we could hit this
        # from several paths (e.g., SuperVariable or `var_getattr`s).
        if not isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)):
            unimplemented(
                gb_type="can't handle functions not implemented in python ",
                context=f"{fn}",
                explanation="Dynamo can only handle functions defined in python",
                hints=[
                    "Move usage of this function out of `torch.compile` region",
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )
        # TODO(anijain2305) - Replace directly calling UserFunctionVariable with
        # VariableBuilder, which handles the wrapping of _torchdynamo_inline.
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        self.fn = fn

    def as_python_constant(self) -> Any:
        if istype(self, UserFunctionVariable):
            return self.fn
        # subclasses (such as methods) usually aren't a constant
        return super().as_python_constant()

    def self_args(self) -> list[VariableTracker]:
        return []

    def get_function(self) -> types.FunctionType:
        return self.fn

    def get_code(self) -> types.CodeType:
        return self.fn.__code__

    def python_type(self) -> type:
        return types.FunctionType

    def has_self(self) -> bool:
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self) -> dict[str, Any]:
        return self.fn.__globals__

    def get_source(self) -> Source:
        source = self.source

        if source and isinstance(self, variables.UserMethodVariable):
            source = self.source_fn  # type: ignore[assignment]
        return source  # type: ignore[return-value]

    def bind_args(
        self,
        parent: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> dict[str, VariableTracker]:
        """
        Assume `args` and `kwargs` are VariableTracker arguments for a call to
        this function, create new bindings for initial locals.
        """
        assert not self.is_constant

        fn: types.FunctionType = self.fn

        if not isinstance(fn, FunctionType):
            raise TypeError("Only supports regular Python functions.")
        root_tx = parent.output.root_tx

        source = self.get_source()
        result = bind_args_cached(fn, root_tx, source, args, kwargs)  # type: ignore[arg-type]

        init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            # TODO refactor these 3 branches.
            side_effects = parent.output.side_effects
            if cell in side_effects:
                cell_var = side_effects[cell]

            elif source:
                closure_cell = GetItemSource(ClosureSource(source), idx)
                closure_cell_contents = AttrSource(closure_cell, "cell_contents")
                try:
                    contents_var = VariableTracker.build(
                        parent, cell.cell_contents, closure_cell_contents
                    )
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(
                    closure_cell, cell, contents_var
                )

            else:
                # TODO figure out why source isn't available here, and whether
                # we can fix that and remove this branch.
                try:
                    contents_var = VariableTracker.build(parent, cell.cell_contents)
                except ValueError:
                    # Cell has not yet been assigned
                    contents_var = variables.DeletedVariable()
                cell_var = side_effects.track_cell_existing(None, cell, contents_var)

            result[name] = cell_var

        return result

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        source = self.get_source()
        return fn_var_getattr(tx, self.fn, source, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        result = hasattr(self.fn, name)
        return variables.ConstantVariable.create(result)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Handle patch_dynamo_config call
        if self.fn is torch._dynamo.patch_dynamo_config:
            try:
                args_const = [arg.as_python_constant() for arg in args]
                kwargs_const = {
                    key: val.as_python_constant() for key, val in kwargs.items()
                }
                changes = torch._dynamo.patch_dynamo_config(
                    *args_const, **kwargs_const
                ).changes
                return variables.DynamoConfigPatchVariable(changes)
            except AsPythonConstantNotImplementedError as e:
                raise RuntimeError(
                    "Cannot convert patch_dynamo_config args/kwargs to constants. "
                    "Please fix your call to patch_dynamo_config by using simpler inputs. "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        elif self.fn is torch._dynamo.error_on_graph_break:
            try:
                bound = inspect.signature(self.fn).bind(*args, **kwargs)
                error_on_graph_break = bound.arguments[
                    "error_on_graph_break"
                ].as_python_constant()
                assert isinstance(error_on_graph_break, bool)
                return variables.ErrorOnGraphBreakVariable(error_on_graph_break)
            except Exception as e:
                raise RuntimeError(
                    "Improper error_on_graph_break() call. Please fix your call to error_on_graph_break(). "
                    f"args: {args}, kwargs: {kwargs}"
                ) from e
        # Handle a `nonstrict_trace(fn)` call
        elif self.fn is torch._dynamo.nonstrict_trace:
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            fn_var = bound.args[0]
            if not isinstance(fn_var, BaseUserFunctionVariable):
                typ = fn_var.python_type()
                msg = f"`nonstrict_trace` expects a callable, but got value of type <{typ.__name__}>"
                unimplemented(
                    gb_type="TypeError from user code",
                    context=f"call_function({self.value}, {args}, {kwargs})",  # type: ignore[attr-defined]
                    explanation=msg,
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

            if not isinstance(fn_var, UserFunctionVariable):
                fn_name = fn_var.get_name()
                msg = f"Applying `nonstrict_trace` to function <{fn_name}>; however, `nonstrict_trace` currently requires the function to be defined outside `torch.compile` region."  # noqa: B950
                unimplemented(
                    gb_type="Limitation of `nonstrict_trace",
                    context=f"{self}",
                    explanation=msg,
                    hints=[
                        f"make sure definition of {fn_name} is outside ",
                        "`torch.compile` region",
                    ],
                )
            # pyrefly: ignore[missing-attribute]
            fn = fn_var.fn
            return variables.TorchInGraphFunctionVariable(fn, nonstrict_traceable=True)

        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        if (
            not tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
            and self.fn
            is torch._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer
        ):
            with torch._dynamo.side_effects.allow_externally_visible_side_effects_in_subtracer(
                tx
            ):
                return super().call_function(tx, args, kwargs)

        if (
            tx.output.current_tracer.under_activation_checkpoint
            and not tx.output.current_tracer.allow_side_effects_under_checkpoint
        ):
            try:
                from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
            except Exception:
                FSDPState = None  # type: ignore[assignment, misc]
            if FSDPState is not None and self.fn in [
                FSDPState._pre_forward,
                FSDPState._post_forward,
            ]:
                with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
                    return super().call_function(tx, args, kwargs)
        return super().call_function(tx, args, kwargs)


class BuiltinMethodVariable(BaseUserFunctionVariable):
    def __init__(
        self, fn: types.BuiltinMethodType, is_constant: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(fn, types.BuiltinMethodType)
        self.fn = fn

    @staticmethod
    def is_supported_builtin_method(obj: Any) -> bool:
        method_self = obj.__self__
        method_name = obj.__name__

        # TODO(anijain2305) - Add support for more builtin methods
        # Supports tuple.__new__ and frozenset({....}).__contains__
        return (method_self is tuple and method_name == "__new__") or (
            type(method_self) is frozenset and method_name == "__contains__"
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        method_self = self.fn.__self__
        name = self.fn.__name__
        obj_source = self.source and AttrSource(self.source, "__self__")
        obj_vt = VariableTracker.build(tx, method_self, obj_source)
        return obj_vt.call_method(tx, name, args, kwargs)


class LocalGeneratorObjectVariable(VariableTracker):
    def __init__(
        self,
        code: types.CodeType,
        f_globals: dict[str, Any],
        inline_tracer: Optional["InstructionTranslator"],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.code = code
        self.f_globals = f_globals
        self.inline_tracer = inline_tracer

    def get_code(self) -> types.CodeType:
        return self.code

    def get_filename(self) -> str:
        return self.get_code().co_filename

    def get_name(self) -> str:
        return self.get_code().co_name

    def get_function(self) -> Never:
        raise NotImplementedError

    def has_self(self) -> bool:
        return False

    def __name__(self) -> str:
        return self.get_name()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.get_name()})"

    __repr__ = __str__

    def reconstruct(self, codegen: "PyCodegen") -> None:
        from torch._dynamo.side_effects import disallow_side_effects_in_generator
        from torch._dynamo.symbolic_convert import (
            InstructionTranslator,
            save_and_restart_speculation_log,
            temporarely_allow_writes_to_output_graph,
        )

        tx = InstructionTranslator.current_tx()
        save = save_and_restart_speculation_log(tx)
        disallow = disallow_side_effects_in_generator(tx)
        temp = temporarely_allow_writes_to_output_graph(tx)

        with save, disallow, temp:
            tracer = self._get_inline_tracer(tx)
            if not tracer.generator_exhausted:
                self.remaining_items = self.force_unpack_var_sequence(tx)
            variables.ListIteratorVariable(self.remaining_items).reconstruct(codegen)

    def bind_args(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> dict[str, VariableTracker]:
        return self.vt.bind_args(tx, args, kwargs)  # type: ignore[attr-defined]

    def get_globals(self) -> dict[str, Any]:
        return self.f_globals

    def python_type(self) -> type:
        return types.GeneratorType

    def _get_inline_tracer(self, tx: "InstructionTranslator") -> Any:
        from torch._dynamo.symbolic_convert import InliningInstructionTranslator

        if self.inline_tracer is None:
            self.inline_tracer = InliningInstructionTranslator.build_inline_tracer(  # type: ignore[assignment]
                tx, self, [], {}
            )
        return self.inline_tracer

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        tracer = self._get_inline_tracer(tx)

        if self._is_generator_exhausted():
            raise_observed_exception(StopIteration, tx)

        try:
            # Hierarchically, tx can be seen as the parent of the inline tracer
            # created on call_function. Any exception needs to be propagated to tx
            # for Dynamo to behave correctly
            return tracer.inline_call_()
        except ObservedException as e:
            tracer.generator_exhausted = True
            raise e
        except InfiniteGeneratorError:
            # test/dynamo/test_misc.py::test_iterator_limit
            raise
        except Unsupported as e:
            torch._dynamo.eval_frame.skip_code(self.get_code())
            raise SkipFrame from e

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        if name in self.python_type().__dict__:
            return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return False

    def has_force_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return True

    def force_unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list[VariableTracker]:
        result: list[VariableTracker] = []
        self.force_apply_to_var_sequence(tx, result.append)
        return result

    def force_apply_to_var_sequence(
        self, tx: "InstructionTranslator", fn: Callable[[VariableTracker], Any]
    ) -> None:
        while True:
            try:
                fn(self.next_variable(tx))
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break

    # no nested graph breaks in generators
    def should_allow_nested_graph_breaks(self):
        return False

    def _setup_exception(
        self, tx: "InstructionTranslator", exc: VariableTracker
    ) -> None:
        tracer = self._get_inline_tracer(tx)
        try:
            tracer._raise_exception_variable(exc)
        except ObservedException as e:
            # if no handler is available (i.e. user code doesn't catch it), the
            # exception is raised again.
            tracer.exception_handler(e)

    def _is_generator_just_started(self) -> bool:
        return self.inline_tracer is None or self.inline_tracer.instruction_pointer == 0

    def _is_generator_exhausted(self) -> bool:
        return getattr(self.inline_tracer, "generator_exhausted", False)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__next__":
            return self.next_variable(tx)
        elif name == "__iter__":
            # iter(gen) returns itself
            return self
        elif name == "send":
            # Sends a value into the generator function. Returns the next value
            # yielded by the generator, or raises StopIteration if the generator
            # exits without yielding another value
            if self._is_generator_just_started() and len(args):
                # can't send non-None value to a just-started generator
                # Test: GeneratorCPythonTests.test_send_non_none_to_new_gen
                if not all(
                    isinstance(arg, ConstantVariable) and arg.value is None
                    for arg in args
                ):
                    raise_observed_exception(TypeError, tx)
            tracer = self._get_inline_tracer(tx)
            tracer.push_many(args)
            return self.next_variable(tx)
        elif name == "close":
            # * Raises a GeneratorExit at the point where the generator function was paused.
            # * If the generator function catches the exception and returns a
            # value, this value is returned from close() - Python 3.13+
            # * If the generator function is already closed, or raises GeneratorExit
            # (by not catching the exception), close() returns None.
            # * If the generator yields a value, a RuntimeError is raised.
            # * If the generator raises any other exception, it is propagated to the caller.
            # * If the generator has already exited due to an exception or normal
            # exit, close() returns None and has no other effect.

            # Return None if close is called on a just-started generator
            # See test GeneratorCloseCpythonTests::test_close_not_started

            tracer = self._get_inline_tracer(tx)
            if self._is_generator_just_started() or self._is_generator_exhausted():
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)

            # Raise GeneratorExit to see if user code catches it. Any other exception
            # is propagated to the parent frame.
            try:
                self._setup_exception(
                    tx, variables.ExceptionVariable(GeneratorExit, ())
                )
                # There's an extra block on Python 3.12+ to handle StopIteration
                # see: https://github.com/python/cpython/blob/8f93dd8a8f237b277abad20d566df90c5cbd7f1e/Objects/genobject.c#L394-L397
                #
                #   1           0 RETURN_GENERATOR
                #               2 POP_TOP
                #               4 RESUME                   0

                #   2           6 LOAD_CONST               1 (1)
                #               8 YIELD_VALUE              1
                #              10 RESUME                   1
                #              12 POP_TOP
                #              14 RETURN_CONST             0 (None)
                #         >>   16 CALL_INTRINSIC_1         3 (INTRINSIC_STOPITERATION_ERROR)
                #              18 RERAISE                  1
                # ExceptionTable:
                #   4 to 14 -> 16 [0] lasti
                if (
                    sys.version_info >= (3, 12)
                    and tracer.next_instruction.opname == "CALL_INTRINSIC_1"
                ):
                    tracer.generator_exhausted = True
                    return variables.ConstantVariable(None)
            except ObservedGeneratorExit:
                # If it doesn't catch, we just return None, as per the text above
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)

            try:
                # Raise RuntimeError if the generator yields any other value
                if self.next_variable(tx):
                    raise_observed_exception(RuntimeError, tx)
            except ObservedGeneratorExit:
                tracer.generator_exhausted = True
                return variables.ConstantVariable(None)
            except ObservedUserStopIteration:
                # In Python 3.13+, one can capture GeneratorExit and return a value
                # See test_generator.py::test_close_capture_GeneratorExit_return
                # https://discuss.python.org/t/let-generator-close-return-stopiteration-value/24786/26
                # https://github.com/python/cpython/pull/104771
                assert tracer.symbolic_result is not None
                return tracer.symbolic_result
        elif name == "throw":
            # * Raises an exception at the point where the generator was paused, and
            # returns the next value yielded by the generator.
            # * If the generator exits without yielding, raise StopIteration
            # * If the generator function does not catch the passed-in exception,
            # or raises a different exception, then that exception propagates to the caller.

            # Setup the exception table and jump target in case of try...finally
            tracer = self._get_inline_tracer(tx)
            try:
                # In Python 3.9, the exception is represented as a triple (typ, val, tb)
                # In such cases, we re-raise the exception object given to avoid
                # creating a new object, so that IS_OP works.
                # See: https://github.com/pytorch/pytorch/pull/146496
                self._setup_exception(tx, args[1] if len(args) == 3 else args[0])
            except ObservedException:  # noqa: TRY203
                # propagate the exception back to the parent caller
                raise

            retval = self.next_variable(tx)

            # The exception raised before is still active. We need to check the exception
            # table one more time to find the next target. But why? Let's walk
            # through an example and its generated bytecode: https://godbolt.org/z/ebdTbMv8M
            #
            #     z = 0
            #     def whoo():
            #         global z
            #         z = 0
            #         try:
            #             yield 1
            #         except ValueError:
            #             yield 2
            #         finally:
            #             z += 1
            #         z += 10
            #
            #     gen = whoo()
            #     next(gen)
            #     gen.throw(ValueError)
            #     print('z', z)  -> z = 1
            #
            #              ...
            #         >>   58 PUSH_EXC_INFO
            #
            #   8          60 LOAD_GLOBAL              2 (ValueError)
            #              70 CHECK_EXC_MATCH
            #              72 POP_JUMP_IF_FALSE        7 (to 88)
            #              74 POP_TOP
            #
            #   9          76 LOAD_CONST               3 (2)
            #              78 YIELD_VALUE              3      <------ ValueError is still active here
            #              80 RESUME                   1
            #              82 POP_TOP
            #              84 POP_EXCEPT
            #              86 jump_backward           34 (to 20)
            #              ...
            #
            #     ExceptionTable:
            #     4 to 8 -> 124 [0] lasti
            #     12 to 18 -> 58 [0]
            #     20 to 56 -> 124 [0] lasti
            #     58 to 82 -> 90 [1] lasti     <------ move to 90
            #     84 to 86 -> 96 [0]
            #     88 to 88 -> 90 [1] lasti
            #     90 to 94 -> 96 [0]
            #     96 to 116 -> 118 [1] lasti
            #     118 to 122 -> 124 [0] lasti
            #
            # In this scenario, a generator can yield after `throw()` is called. Even
            # after the exception is raised a few lines above, it remains active
            # within the `78 YIELD_VALUE` instruction. When the generator resumes
            # after the second yield on instruction `80 RESUME`, we cannot simply
            # return the control flow to the next instruction. Instead, one must
            # check the exception table (or equivalent) to find the next target
            # In this case, it says the instruction pointer must be moved to 90.
            #
            # Without this step, if we let the trace proceed to the next
            # instruction, it would follow the control flow where the exception
            # raised by `throw()` was handled and swallowed, potentially leading
            # to incorrect behavior.
            exc_type = type("__InternalThrowException", (Exception,), {})

            try:
                self._setup_exception(tx, variables.ExceptionVariable(exc_type, ()))
                self.next_variable(tx)
            except get_dynamo_observed_exception(exc_type):
                # We should get back the exception raised before.
                pass
            else:
                raise_observed_exception(RuntimeError, tracer)
            return retval

        return super().call_method(tx, name, args, kwargs)


class ContextlibContextManagerLocalGeneratorObjectVariable(
    LocalGeneratorObjectVariable
):
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager

        It is a special case of a generator function as we do not allow return a context manager
        from a torch.compile function.
    """


class LocalGeneratorFunctionVariable(BaseUserFunctionVariable):
    """functions that behaves like iterators

    .. note::

        This is a wrapper around (Nested)UserFunctionVariable
    """

    def __init__(
        self,
        vt: VariableTracker,
        *,
        generator_cls: type = LocalGeneratorObjectVariable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vt = vt
        self.generator_cls = generator_cls

    def __getattr__(self, name):
        if name in self.__class__.__dict__:
            return getattr(self, name)
        return getattr(self.vt, name)

    def get_globals(self) -> dict[str, Any]:
        return self.vt.get_globals()  # type: ignore[attr-defined]

    def _build_inline_tracer(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "InstructionTranslatorBase":
        from torch._dynamo.symbolic_convert import InliningInstructionTranslator

        return InliningInstructionTranslator.build_inline_tracer(
            tx,
            self,
            args,
            kwargs,
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not is_generator(self.vt.get_code()):  # type: ignore[attr-defined]
            unimplemented(
                gb_type="non-generator contextlib.contextmanager",
                context=str(self.vt.get_code()),  # type: ignore[attr-defined]
                explanation="Cannot compile function decorated with `@contextlib.contextmanager` that is not a generator"
                ", i.e. does not use `yield`",
                hints=[
                    "Use `yield` in the function body instead of `return`.",
                    "Remove the `@contextlib.contextmanager` decorator.",
                ],
            )

        inline_tracer = self._build_inline_tracer(tx, list(args), kwargs)
        code = self.vt.get_code()  # type: ignore[attr-defined]
        f_globals = self.vt.get_globals()  # type: ignore[attr-defined]

        # calling a generator returns a generator object
        return self.generator_cls(
            code,
            f_globals,
            inline_tracer,  # type: ignore[arg-type]
            source=self.source,
        )


class FunctionDecoratedByContextlibContextManagerVariable(
    LocalGeneratorFunctionVariable
):
    """
    .. note::

        This is only used when the function is annotated with @contextlib.contextmanager
    """

    def __init__(self, vt: VariableTracker, **kwargs: Any):
        super().__init__(
            vt,
            generator_cls=ContextlibContextManagerLocalGeneratorObjectVariable,
            **kwargs,
        )

    def _build_inline_tracer(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "InstructionTranslatorBase":
        # NOTE: This only exists to not break support for context manager when
        # config.enable_faithful_generator_behavior = False and
        # config.enable_trace_contextlib = True. In case the former is false,
        # Dynamo should still be able to trace through @contextmanager functions
        tracer = super()._build_inline_tracer(tx, args, kwargs)
        assert isinstance(
            tracer,
            torch._dynamo.symbolic_convert.InliningGeneratorInstructionTranslator,
        )
        tracer.is_generator_from_ctx_manager = True
        return tracer


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(
        self,
        fn: Callable[..., Any],
        obj: VariableTracker,
        source_fn: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fn=fn, **kwargs)  # type: ignore[arg-type]
        self.obj = obj
        self.source_fn = source_fn
        # Note on source and source_fn
        # Be careful with `source` when delegating to UserFunctionVariable
        # (base-class) methods. In this __init__, `source` is a *bound method*
        # object, but the base class expects the underlying *function* object.
        # One way is to simplly use `__func__` to unwrap it.
        #
        # For recursive dict-tag optimizations, it can be faster to fetch the
        # function directly from `cls.__dict__`; that's why we pass on
        # `source_fn`. Whenever it is possible to access the function from
        # cls.__dict__, we pass that on to `source_fn`. Because bind_args
        # operates on the unbound function, most guards should target
        # `source_fn` rather than the original `source`.
        if source_fn is None and kwargs.get("source") is not None:
            self.source_fn = AttrSource(kwargs.get("source"), "__func__")  # type: ignore[assignment, arg-type]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self) -> list[VariableTracker]:
        return [self.obj]

    def python_type(self) -> type[types.MethodType]:
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # NOTE this is to handle methods annotated by `nonstrict_trace`.
        # a `nonstrict_trace`-ed function will be wrapped by
        # `VariableTracker.build` and route to `TorchInGraphFunctionVariable`,
        # but in the case of method, we manually wrap it with `UserMethodVariable`
        # inside `UserDefinedObjectVariable.var_getattr`.
        #
        # We might be able to simplify this away by canonicalizing the
        # function/method wrapping code paths.
        from ..trace_rules import is_nonstrict_trace_callable

        if is_nonstrict_trace_callable(self.fn):
            call_args = [*self.self_args(), *args]
            var = variables.TorchInGraphFunctionVariable(
                self.fn, nonstrict_traceable=True
            )
            return var.call_function(tx, call_args, kwargs)

        # For nn.Module methods, redirecting to NNModuleVariable.call_method for optimized solution
        # rather than simple inlining. E.g, putting `call_method` op in FX graph for `forward` method
        # since we ensure `forward` of allowed modules can be traced by AOT safely.
        # Note this is not only for allowed modules, as user customized modules can extend from
        # allowed modules but using parent's `forward` method, which is also covered by this branch.

        # If we are tracing the higher order op, we want Dynamo to step inside
        # the module call so that Dynamo can see the underlying parameters and
        # buffers and raise them as inputs to the graph. The is_root_tracer
        # check bypasses the if condition for non-root tracers and directly
        # calls the super().call_function at the end, which is basically
        # equivalent of inlining the method.
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            # inline torch.nn.utils.parametrize
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                and module_attr != "torch.nn.utils.parametrize"
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, list(args), kwargs, constant=self.is_constant
                )
        elif (
            _fsdp_param_group is not None
            and self.fn is _fsdp_param_group.FSDPParamGroup.use_training_state  # type: ignore[attr-defined]
        ):
            return variables.TorchCtxManagerClassVariable(self.fn).call_function(
                tx, (self.obj, *args), kwargs
            )
        if self.is_constant:
            fn = getattr(self.obj.value, self.fn.__name__)  # type: ignore[attr-defined]
            return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
        return super().call_function(tx, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__self__":
            return self.obj
        if name == "__func__":
            # We might have a better way to access the function object, this
            # information is stored in self.source_fn, use that to construct the
            # variable tracker.
            return VariableTracker.build(tx, self.fn, self.source_fn)  # type: ignore[arg-type]
        return super().var_getattr(tx, name)


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(
        self,
        wrapped: UserMethodVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen(self.context))  # type: ignore[arg-type]
        codegen(self.wrapped)
        codegen.extend_output(create_call_function(1, False))


class WrappedUserFunctionVariable(UserFunctionVariable):
    def __init__(
        self,
        wrapped: UserFunctionVariable,
        context: "ContextWrappingVariable",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("fn", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen(self.context)) 
```



## High-Level Overview

"""Function-related variable tracking classes for Dynamo's symbolic execution.This module contains classes that track different types of functions during graphcompilation, including:- User-defined functions and methods- Built-in functions and methods- Wrapped functions (e.g. from decorators)- Special function types (e.g. functools.partial)- Triton kernels and related function typesThese classes are responsible for:- Tracking function calls and their arguments- Managing function closures and cell variables- Handling function attributes and special methods- Maintaining guards for function identity and closure contents- Supporting function inlining and specialization- Enabling proper symbolic execution of different function typesThe variable trackers here work together with the rest of Dynamo to enableaccurate graph capture while handling Python's various function-related behaviors.

This Python file contains 32 class(es) and 172 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FunctionSpec`, `BaseUserFunctionVariable`, `UserFunctionVariable`, `BuiltinMethodVariable`, `LocalGeneratorObjectVariable`, `ContextlibContextManagerLocalGeneratorObjectVariable`, `LocalGeneratorFunctionVariable`, `FunctionDecoratedByContextlibContextManagerVariable`, `UserMethodVariable`, `WrappedUserMethodVariable`, `WrappedUserFunctionVariable`, `NestedUserFunctionVariable`, `WrappedNestedUserFunctionVariable`, `SkipFunctionVariable`, `WrappedSkipFunctionVariable`, `WrapperUserFunctionVariable`, `WrapperUserMethodVariable`, `CollectiveFunctionRewriteVariable`, `FunctoolsWrapsVariable`, `CollectionsNamedTupleFunction`

**Functions defined**: `__init__`, `update_defaults`, `_get_spec`, `bind_args_cached`, `fn`, `wrap_bound_arg`, `wrap_args_kwargs`, `init_cellvars`, `_create_nested_fn`, `fn_var_getattr`, `get_filename`, `get_name`, `get_globals`, `call_function`, `call_obj_hasattr`, `closure_vars`, `should_allow_nested_graph_breaks`, `create_with_source`, `__init__`, `as_python_constant`

**Key imports**: builtins, functools, inspect, itertools, logging, sys, traceback, types, Callable, Sequence, CellType, FunctionType


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `builtins`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `sys`
- `traceback`
- `types`
- `collections.abc`: Callable, Sequence
- `typing`: Any, Optional, TYPE_CHECKING, TypeVar
- `typing_extensions`: Never
- `weakref`: WeakKeyDictionary
- `torch`
- `torch._dynamo.exc`: get_stack_above_dynamo
- `torch._guards`: Source
- `..`: config, graph_break_hints, polyfills, variables
- `..bytecode_transformation`: create_call_function, create_rot_n, is_generator
- `..guards`: GuardBuilder, install_guard
- `.constant`: ConstantVariable
- `torch.distributed.fsdp._fully_shard`: _fsdp_param_group
- `torch._dynamo.codegen`: PyCodegen
- `torch._dynamo.variables.ctx_manager`: ContextWrappingVariable
- `.lists`: BaseListVariable, ListVariable
- `.tensor`: TensorVariable
- `torch.distributed.fsdp._fully_shard._fsdp_state`: FSDPState
- `torch._dynamo.side_effects`: disallow_side_effects_in_generator
- `torch._dynamo.symbolic_convert`: InliningInstructionTranslator


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `functions.py_docs.md`
- **Keyword Index**: `functions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
