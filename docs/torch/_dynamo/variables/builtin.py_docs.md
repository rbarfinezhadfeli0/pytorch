# Documentation: `torch/_dynamo/variables/builtin.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/builtin.py`
- **Size**: 127,979 bytes (124.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

## Original Source

```python
"""
Built-in function and type variable tracking for TorchDynamo's symbolic execution.

This module contains variable tracker classes for Python built-in functions, types,
and operations during graph compilation. It handles symbolic execution of:

- Built-in functions (len, getattr, isinstance, etc.)
- Type constructors (int, float, str, list, dict, etc.)
- Built-in operators and methods
- Special Python constructs (super, hasattr, etc.)

Key classes:
- BuiltinVariable: Tracks built-in functions and handles their execution
- TypeVariable: Manages type constructor calls and type checking
- SuperVariable: Handles super() calls in class hierarchies

These variable trackers ensure that built-in Python operations are correctly
handled during symbolic execution, either by executing them directly when safe
or by creating appropriate graph nodes when needed.
"""

import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
import typing
import unittest
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterable, KeysView, Sequence
from typing import Any, TYPE_CHECKING, Union

import torch
from torch import sym_float, sym_int
from torch._subclasses.meta_utils import is_sparse_any
from torch.overrides import BaseTorchFunctionMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config, graph_break_hints, polyfills, variables
from ..exc import (
    AttributeMutationError,
    ObservedAttributeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    unimplemented,
    Unsupported,
    UserError,
    UserErrorType,
)
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    is_constant_source,
    Source,
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_numpy_ndarray_args,
    check_unspec_or_constant_args,
    check_unspec_python_args,
    cmp_name_to_op_mapping,
    dict_methods,
    extract_fake_example_value,
    frozenset_methods,
    get_fake_value,
    guard_if_dyn,
    is_tensor_getset_descriptor,
    is_wrapper_or_member_descriptor,
    istype,
    numpy_operator_wrapper,
    proxy_args_kwargs,
    raise_args_mismatch,
    set_methods,
    str_methods,
    tensortype_to_dtype,
)
from .base import AsPythonConstantNotImplementedError, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictKeysVariable,
    DictViewVariable,
    FrozensetVariable,
    is_hashable,
    SetVariable,
)
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    SizeVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .streams import EventVariable, StreamVariable
from .tensor import (
    FakeItemVariable,
    supported_comparison_ops,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .user_defined import (
    MutableMappingVariable,
    UserDefinedDictVariable,
    UserDefinedObjectVariable,
    UserDefinedVariable,
)


if TYPE_CHECKING:
    # Cyclic dependency...
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator

log = logging.getLogger(__name__)


IN_PLACE_DESUGARING_MAP = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.itruediv: operator.truediv,
    operator.imod: operator.mod,
    operator.imatmul: operator.imatmul,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}


_HandlerCallback = Callable[
    ["InstructionTranslator", typing.Any, typing.Any], VariableTracker | None
]
_TrackersType = Union[type[VariableTracker], tuple[type[VariableTracker], ...]]
polyfill_fn_mapping = {
    operator.eq: polyfills.cmp_eq,
    operator.ne: polyfills.cmp_ne,
    operator.lt: polyfills.cmp_lt,
    operator.le: polyfills.cmp_le,
    operator.gt: polyfills.cmp_gt,
    operator.ge: polyfills.cmp_ge,
}

bin_ops = (
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
)

bin_int_ops = (
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
)

un_int_ops = (operator.invert,)

tensor_and_int_ops = (
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
)

un_ops = (
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
)

BUILTIN_TO_TENSOR_FN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# These functions represent the r* versions of the above ops
# Basically, if __add__(1, Tensor) is called, it is translated
# to __radd__(Tensor, 1).
# In the builtin var, we check if there is a tensor in the first args position,
# if not, we swap the args and use the r* version of the op.
BUILTIN_TO_TENSOR_RFN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}


def populate_builtin_to_tensor_fn_map() -> None:
    global BUILTIN_TO_TENSOR_FN_MAP
    if len(BUILTIN_TO_TENSOR_FN_MAP) > 0:
        # Only populate once; after there are elements present no need to
        # repopulate
        return
    most_recent_func: Callable[..., Any] | None = None

    class GetMethodMode(BaseTorchFunctionMode):
        """
        Mode to extract the correct methods from torch function invocations
        (Used to get the correct torch.Tensor methods from builtins)
        """

        def __torch_function__(
            self,
            func: Callable[..., Any],
            types: Any,
            args: Sequence[Any] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            nonlocal most_recent_func
            most_recent_func = func
            return func(*args, **kwargs)

    inp0 = torch.ones(1)
    inp1 = torch.ones(1)
    inp0_int = torch.ones(1, dtype=torch.int32)
    inp1_int = torch.ones(1, dtype=torch.int32)
    with GetMethodMode():
        setups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (lambda o: o(inp0), un_ops),
            (lambda o: o(inp0_int), un_int_ops),
            (lambda o: o(inp0, inp1), bin_ops),
            (lambda o: o(inp0_int, inp1_int), bin_int_ops),
            (lambda o: o(inp0_int, 0), tensor_and_int_ops),
        ]
        for setup_fn, op_list in setups_and_oplists:
            for op in op_list:
                setup_fn(op)
                assert most_recent_func is not None
                BUILTIN_TO_TENSOR_FN_MAP[op] = most_recent_func

        # gather the reverse functions
        rsetups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (
                lambda o: o(1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: o(1, inp1_int), bin_int_ops),
            (lambda o: o(0, inp0_int), tensor_and_int_ops),
        ]

        rskips = {operator.matmul, operator.imatmul, operator.getitem}
        for setup_fn, op_list in rsetups_and_oplists:
            for op in op_list:
                if op in rskips:
                    continue
                setup_fn(op)
                assert most_recent_func is not None
                if most_recent_func != BUILTIN_TO_TENSOR_FN_MAP[op]:
                    BUILTIN_TO_TENSOR_RFN_MAP[op] = most_recent_func


class BuiltinVariable(VariableTracker):
    """
    A VariableTracker that represents a built-in value (functions and operators).
    A lot of the code here assumes it will be a function object.

    The BuiltinVariable class wraps Python built-in functions (like len, isinstance, etc.)
    and operators (like +, -, *, etc.) to enable symbolic execution during tracing. This allows
    Dynamo to properly handle these operations when converting Python code to FX graphs while
    maintaining correct semantics and enabling optimizations.
    """

    _SENTINEL = object()
    _nonvar_fields = {
        "fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BuiltinVariable":
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.cache
    def _constant_fold_functions() -> set[Callable[..., Any]]:
        fns: set[Callable[..., Any]] = {
            abs,
            all,
            any,
            bool,
            callable,
            chr,
            complex,
            divmod,
            float,
            getattr,
            int,
            len,
            max,
            min,
            ord,
            pow,
            repr,
            round,
            str,
            str.format,
            sum,
            type,
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.truth,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
            operator.index,
        }
        from .tensor import supported_comparison_ops

        fns.update(supported_comparison_ops.values())
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self) -> bool:
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.cache
    def _fx_graph_functions() -> set[Callable[..., Any]]:
        fns = {
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.lt,
            operator.gt,
            operator.ge,
            operator.le,
            operator.ne,
            operator.eq,
            operator.sub,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.getitem,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        return fns  # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _binops() -> dict[
        Callable[..., object], tuple[list[str], Callable[..., object]]
    ]:
        # function -> ([forward name, reverse name, in-place name], in-place op)
        fns: dict[Callable[..., object], tuple[list[str], Callable[..., object]]] = {
            operator.add: (["__add__", "__radd__", "__iadd__"], operator.iadd),
            operator.sub: (["__sub__", "__rsub__", "__isub__"], operator.isub),
            operator.mul: (["__mul__", "__rmul__", "__imul__"], operator.imul),
            operator.truediv: (
                ["__truediv__", "__rtruediv__", "__itruediv__"],
                operator.itruediv,
            ),
            operator.floordiv: (
                ["__floordiv__", "__rfloordiv__", "__ifloordiv__"],
                operator.ifloordiv,
            ),
            operator.mod: (["__mod__", "__rmod__", "__imod__"], operator.imod),
            pow: (["__pow__", "__rpow__", "__ipow__"], operator.ipow),
            operator.pow: (["__pow__", "__rpow__", "__ipow__"], operator.ipow),
            operator.lshift: (
                ["__lshift__", "__rlshift__", "__ilshift__"],
                operator.ilshift,
            ),
            operator.rshift: (
                ["__rshift__", "__rrshift__", "__irshift__"],
                operator.irshift,
            ),
            operator.xor: (["__xor__", "__rxor__", "__ixor__"], operator.xor),
            # NB: The follow binary operators are not supported for now, since the
            # corresponding magic methods aren't defined on SymInt / SymFloat:
            # operator.matmul
            # divmod
            # operator.and_
            # operator.or_
        }
        return fns

    @staticmethod
    @functools.cache
    def _binop_handlers() -> dict[
        Callable[..., object],
        list[
            tuple[
                tuple[
                    type[VariableTracker],
                    _TrackersType,
                ],
                _HandlerCallback,
            ]
        ],
    ]:
        # Multiple dispatch mechanism defining custom binop behavior for certain type
        # combinations. Handlers are attempted in order, and will be used if the type checks
        # match. They are expected to have the signature:
        # fn(tx, arg0: VariableTracker, arg1: VariableTracker) -> VariableTracker
        from .functions import BaseUserFunctionVariable, UserFunctionVariable
        from .nn_module import NNModuleVariable
        from .tensor import supported_const_comparison_ops
        from .torch import BaseTorchVariable
        from .user_defined import (
            UserDefinedClassVariable,
            UserDefinedObjectVariable,
            UserDefinedVariable,
        )

        # Override table contains: op_fn -> [list of handlers]
        op_handlers: dict[Any, list[Any]] = {}
        for (
            op,
            (magic_method_names, in_place_op),
        ) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []

            forward_name, reverse_name, inplace_name = magic_method_names

            # User-defined args (highest precedence)
            def user_defined_handler(
                tx: "InstructionTranslator",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = forward_name,
                reverse_name: str = reverse_name,
            ) -> VariableTracker:
                # Manually handle reversing logic if needed (e.g. call __radd__)

                # TODO: If we expand this to handle tensor args, we need to manually
                # handle cases like this:
                #
                # class A(int):
                #     def __radd__(self, other):
                #         print("woof")
                # torch.randn(3) + A(3)
                #
                # In this example, A.__radd__() is not called -> nothing is printed, because
                # Tensor.__add__ only does a subtype test against int, ignoring the subclass.
                # To be fully correct, we should not call A.__radd__() here, and there may be
                # other cases to reason about and add exceptions for.
                if isinstance(a, UserDefinedVariable):
                    return a.call_method(tx, forward_name, [b], {})
                else:
                    return b.call_method(tx, reverse_name, [a], {})

            op_handlers[op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_handler)
            )

            def user_defined_inplace_handler(
                tx: "InstructionTranslator",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = inplace_name,
            ) -> VariableTracker:
                return a.call_method(tx, forward_name, [b], {})

            op_handlers[in_place_op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_inplace_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_inplace_handler)
            )

            # Dynamic shape args
            def dynamic_handler(
                tx: "InstructionTranslator",
                a: VariableTracker,
                b: VariableTracker,
                *,
                fn: Callable[..., Any] = op,
            ) -> VariableTracker:
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function", fn, *proxy_args_kwargs([a, b], {})
                    ),
                )

            op_handlers[op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

            # NB: Prefer out-of-place op when calling in-place op to generate valid graph
            op_handlers[in_place_op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

        # Special cases - lower precedence but still prefer these over constant folding

        # List-like addition (e.g. [1, 2] + [3, 4])
        def tuple_add_handler(
            tx: "InstructionTranslator", a: BaseListVariable, b: VariableTracker
        ) -> VariableTracker:
            return TupleVariable([*a.items, *b.unpack_var_sequence(tx)])

        def size_add_handler(
            tx: "InstructionTranslator", a: BaseListVariable, b: VariableTracker
        ) -> VariableTracker:
            return SizeVariable([*a.items, *b.unpack_var_sequence(tx)])

        list_like_addition_handlers: list[
            tuple[
                tuple[
                    type[VariableTracker],
                    _TrackersType,
                ],
                _HandlerCallback,
            ]
        ] = [
            # NB: Prefer the tuple-specific logic over base logic because of
            # some SizeVariable weirdness. Specifically, the tuple-specific logic
            # drops the subclass type (e.g. SizeVariable) and returns TupleVariables.
            (
                (SizeVariable, SizeVariable),
                size_add_handler,
            ),
            (
                (SizeVariable, TupleVariable),
                size_add_handler,
            ),
            (
                (TupleVariable, SizeVariable),
                size_add_handler,
            ),
            (
                (TupleVariable, TupleVariable),
                tuple_add_handler,
            ),
            (
                (TupleVariable, ConstantVariable),
                tuple_add_handler,
            ),
            (
                (ConstantVariable, TupleVariable),
                lambda tx, a, b: TupleVariable(
                    [
                        *a.unpack_var_sequence(tx),
                        *b.items,
                    ],
                ),
            ),
            (
                (
                    ListVariable,
                    (BaseListVariable, ConstantVariable, ListIteratorVariable),
                ),
                lambda tx, a, b: ListVariable(
                    [*a.items, *b.unpack_var_sequence(tx)],
                    mutation_type=ValueMutationNew(),
                ),
            ),
            (
                (BaseListVariable, BaseListVariable),
                lambda tx, a, b: type(a)(
                    [
                        *a.items,
                        *b.items,
                    ]
                ),
            ),
        ]
        op_handlers[operator.add].extend(list_like_addition_handlers)

        def list_iadd_handler(
            tx: "InstructionTranslator", a: BaseListVariable, b: VariableTracker
        ) -> Any:
            if a.is_immutable() or not b.has_unpack_var_sequence(tx):
                # Handler doesn't apply
                return None

            seq = b.unpack_var_sequence(tx)
            tx.output.side_effects.mutation(a)
            a.items.extend(seq)
            return a

        list_like_iadd_handlers: list[Any] = [
            (
                (ListVariable, VariableTracker),
                list_iadd_handler,
            ),
            (
                (TupleVariable, TupleVariable),
                tuple_add_handler,
            ),
            (
                (TupleVariable, ConstantVariable),
                tuple_add_handler,
            ),
        ]
        op_handlers[operator.iadd].extend(list_like_iadd_handlers)

        # List-like expansion (e.g. [1, 2, 3] * 3)
        def expand_list_like(
            tx: "InstructionTranslator", lst: VariableTracker, const: VariableTracker
        ) -> VariableTracker:
            if isinstance(lst, ConstantVariable):
                lst, const = const, lst
            try:
                assert isinstance(lst, BaseListVariable)
                return lst.__class__(
                    items=lst.items * const.as_python_constant(),
                    mutation_type=ValueMutationNew(),
                )
            except MemoryError as exc:
                raise_observed_exception(
                    type(exc),
                    tx,
                    args=list(map(ConstantVariable.create, exc.args)),
                )

        list_like_expansion_handlers: list[
            tuple[
                tuple[type[VariableTracker], type[VariableTracker]],
                _HandlerCallback,
            ]
        ] = [
            ((ListVariable, ConstantVariable), expand_list_like),
            ((TupleVariable, ConstantVariable), expand_list_like),
            ((ConstantVariable, ListVariable), expand_list_like),
            ((ConstantVariable, TupleVariable), expand_list_like),
        ]
        op_handlers[operator.mul].extend(list_like_expansion_handlers)

        def create_cmp_op_handlers(
            op: Callable[..., Any],
        ) -> list[tuple[tuple[_TrackersType, _TrackersType], _HandlerCallback]]:
            def compare_by_value(
                tx: "InstructionTranslator", a: VariableTracker, b: VariableTracker
            ) -> VariableTracker:
                try:
                    return ConstantVariable(op(a.value, b.value))  # type: ignore[attr-defined]
                except TypeError as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(map(ConstantVariable.create, exc.args)),
                    )

            result: list[
                tuple[
                    tuple[
                        _TrackersType,
                        _TrackersType,
                    ],
                    _HandlerCallback,
                ]
            ] = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in polyfill_fn_mapping:
                # For constants, speedup the comparison instead of using
                # polyfill. Removing this line causes major regression for pr
                # time benchmark - add_loop_eager.
                result = [((ConstantVariable, ConstantVariable), compare_by_value)]

                op_var = BuiltinVariable(op)
                # Special handling of SymNode variable
                result.extend(
                    [
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                def handler(
                    tx: "InstructionTranslator", a: VariableTracker, b: VariableTracker
                ) -> VariableTracker:
                    return tx.inline_user_function_return(
                        VariableTracker.build(tx, polyfill_fn_mapping[op]), [a, b], {}
                    )

                result.append(((VariableTracker, VariableTracker), handler))
                return result

            result = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in supported_const_comparison_ops.values() and op.__name__.startswith(
                "is_"
            ):
                # Tensor is None, List is not None, etc
                none_result = op(object(), None)

                def never(
                    tx: "InstructionTranslator", a: VariableTracker, b: VariableTracker
                ) -> VariableTracker:
                    return ConstantVariable(none_result)

                obj_op_none = never
                none_op_obj = never

                types_that_are_never_none = (
                    TensorVariable,
                    SymNodeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                    BaseTorchVariable,
                )
                result.extend(
                    [
                        (
                            (types_that_are_never_none, ConstantVariable),
                            obj_op_none,
                        ),
                        (
                            (ConstantVariable, types_that_are_never_none),
                            none_op_obj,
                        ),
                    ]
                )

                op_var = BuiltinVariable(op)
                result.extend(
                    [
                        (
                            (
                                (UserFunctionVariable, BuiltinVariable),
                                (UserFunctionVariable, BuiltinVariable),
                            ),
                            lambda tx, a, b: ConstantVariable(op(a.fn, b.fn)),
                        ),
                        (
                            (
                                NNModuleVariable,
                                NNModuleVariable,
                            ),
                            lambda tx, a, b: ConstantVariable(
                                op(
                                    tx.output.get_submodule(a.module_key),
                                    tx.output.get_submodule(b.module_key),
                                )
                            ),
                        ),
                        (
                            (UserDefinedObjectVariable, UserDefinedObjectVariable),
                            compare_by_value,
                        ),
                        (
                            (UserDefinedClassVariable, UserDefinedClassVariable),
                            compare_by_value,
                        ),
                        (
                            (
                                (StreamVariable, EventVariable, ConstantVariable),
                                (StreamVariable, EventVariable, ConstantVariable),
                            ),
                            compare_by_value,
                        ),
                        (
                            (TensorVariable, VariableTracker),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (VariableTracker, TensorVariable),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                def handle_is(
                    tx: "InstructionTranslator",
                    left: VariableTracker,
                    right: VariableTracker,
                ) -> VariableTracker | None:
                    # If the two objects are of different type, we can safely return False
                    # and True for `is` and `is not`, respectively
                    if type(left) is not type(right):
                        return ConstantVariable.create(op.__name__ != "is_")
                    if left is right:
                        return ConstantVariable.create(op(left, right))
                    if (
                        istype(left, variables.ExceptionVariable)
                        and istype(right, variables.ExceptionVariable)
                        and left.exc_type is not right.exc_type
                    ):
                        return ConstantVariable.create(op(left, right))
                    return None

                result.append(((VariableTracker, VariableTracker), handle_is))  # type: ignore[arg-type]

            return result

        for op in supported_comparison_ops.values():
            assert callable(op)
            assert op not in op_handlers
            op_handlers[op] = create_cmp_op_handlers(op)

        return op_handlers

    @staticmethod
    def _find_binop_handler(
        op: Callable[..., Any], a_type: type[VariableTracker], b_type: type
    ) -> list[_HandlerCallback] | None:
        handlers = BuiltinVariable._binop_handlers().get(op)
        if handlers is None:
            return None

        matches = []
        for (type1, type2), handler in handlers:
            if issubclass(a_type, type1) and issubclass(b_type, type2):
                matches.append(handler)
        return matches

    def can_insert_in_graph(self) -> bool:
        return self.fn in self._fx_graph_functions()

    def __init__(self, fn: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fn = fn

    def __repr__(self) -> str:
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def as_python_constant(self) -> Any:
        return self.fn

    def as_proxy(self) -> Any:
        DTYPE = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
        }
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen: "PyCodegen") -> None:
        name = self.fn.__name__
        assert self.fn.__module__ == "builtins"
        assert name not in codegen.tx.f_globals, "shadowed global"
        codegen.append_output(codegen.create_load_global(name, add=True))

    def constant_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args: VariableTracker) -> bool:
        any_tensor = False
        for arg in args:
            if isinstance(arg, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or isinstance(arg, variables.TensorVariable)
        return any_tensor

    def tensor_args_type(self, arg_types: list[type]) -> bool:
        any_tensor = False
        for arg_type in arg_types:
            if issubclass(arg_type, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or issubclass(arg_type, variables.TensorVariable)
        return any_tensor

    def python_and_tensor_constant_only(
        self, *args: VariableTracker, **kwargs: VariableTracker
    ) -> bool:
        tensor_args = []
        non_tensor_args = []
        for i in itertools.chain(args, kwargs.values()):
            if isinstance(i, variables.TensorVariable):
                tensor_args.append(i)
            else:
                non_tensor_args.append(i)
        return all(
            is_constant_source(t.source) if t.source is not None else False
            for t in tensor_args
        ) and self.constant_args(*non_tensor_args)

    @staticmethod
    def unwrap_unspec_args_kwargs(
        args: Sequence[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[list[Any], dict[str, Any]]:
        return [x.as_python_constant() for x in args], {
            k: v.as_python_constant() for k, v in kwargs.items()
        }

    def has_constant_handler(
        self, args: Sequence[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> bool:
        return self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        )

    @staticmethod
    def _make_handler(
        fn: Callable[..., Any], arg_types: list[type], has_kwargs: bool
    ) -> Callable[
        [
            "InstructionTranslator",
            Sequence[VariableTracker],
            dict[str, VariableTracker],
        ],
        VariableTracker | None,
    ]:
        from .lazy import LazyVariableTracker

        obj = BuiltinVariable(fn)
        handlers: list[_HandlerCallback] = []

        if any(issubclass(t, LazyVariableTracker) for t in arg_types):
            return lambda tx, args, kwargs: obj.call_function(
                tx, [v.realize() for v in args], kwargs
            )

        if inspect.isclass(fn) and (
            issubclass(fn, Exception)
            # GeneratorExit doesn't inherit from Exception
            # >>> issubclass(GeneratorExit, Exception)
            # False
            or fn is GeneratorExit
        ):

            def create_exception_class_object(
                tx: "InstructionTranslator",
                args: Sequence[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker:
                if fn is AssertionError and not all(
                    isinstance(x, variables.ConstantVariable)
                    and isinstance(x.value, str)
                    for x in args
                ):
                    unimplemented(
                        gb_type="assert with non-string message",
                        context=str(args),
                        explanation="Dynamo only supports asserts with string messages",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

                return variables.ExceptionVariable(fn, args, kwargs)

            return create_exception_class_object

        if obj.can_insert_in_graph() and not (
            fn is operator.getitem
            and not issubclass(arg_types[0], variables.TensorVariable)
        ):
            if obj.tensor_args_type(arg_types):
                return obj._handle_insert_op_in_graph
            elif has_kwargs:
                # need runtime check for kwargs
                handlers.append(obj._handle_insert_op_in_graph)

        # Handle binary ops (e.g. __add__ / __radd__, __iadd__, etc.)
        # NB: Tensor args are handled above and not here
        if len(arg_types) == 2 and not has_kwargs:
            # Try to find a handler for the arg types; otherwise, fall through to constant handler
            binop_handlers = BuiltinVariable._find_binop_handler(fn, *arg_types)
            if not binop_handlers:
                pass
            elif len(binop_handlers) == 1:
                (binop_handler,) = binop_handlers
                handlers.append(lambda tx, args, _: binop_handler(tx, *args))
            else:

                def call_binop_handlers(
                    tx: "InstructionTranslator", args: Any, _: Any
                ) -> Any:
                    # pyrefly: ignore [not-iterable]
                    for fn in binop_handlers:
                        rv = fn(tx, *args)
                        if rv:
                            return rv
                    return None

                handlers.append(call_binop_handlers)

        self_handler = getattr(obj, f"call_{fn.__name__}", None)
        if self_handler:

            def call_self_handler(
                tx: "InstructionTranslator",
                args: Sequence[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                try:
                    # pyrefly: ignore [not-callable]
                    return self_handler(tx, *args, **kwargs)
                except TypeError:
                    # Check if binding is bad. inspect signature bind is expensive.
                    # So check only when handler call fails.
                    try:
                        # pyrefly: ignore [bad-argument-type]
                        inspect.signature(self_handler).bind(tx, *args, **kwargs)
                    except TypeError as e:
                        has_constant_handler = obj.has_constant_handler(args, kwargs)
                        if not has_constant_handler:
                            log.warning(  # noqa: G200
                                "incorrect arg count %s %s and no constant handler",
                                self_handler,
                                e,
                            )
                            unimplemented(
                                gb_type="invalid call to builtin op handler",
                                context=f"invalid args to {self_handler}: {args} {kwargs}",
                                explanation=f"Encountered TypeError when trying to handle op {fn.__name__}",
                                hints=[*graph_break_hints.DIFFICULT],
                            )
                    else:
                        raise
                except Unsupported as exc:
                    has_constant_handler = obj.has_constant_handler(args, kwargs)
                    if not has_constant_handler:
                        raise
                    # Actually, we will handle this just fine
                    exc.remove_from_stats()
                return None

            handlers.append(call_self_handler)

        if obj.can_constant_fold_through():
            if (
                all(issubclass(x, ConstantVariable) for x in arg_types)
                and not has_kwargs
            ):

                def constant_fold_handler(
                    tx: "InstructionTranslator",
                    args: Sequence[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # fast path
                    try:
                        res = fn(
                            *[x.as_python_constant() for x in args],
                        )
                    except Exception as exc:
                        raise_observed_exception(
                            type(exc),
                            tx,
                            args=list(map(ConstantVariable.create, exc.args)),
                        )
                    except AsPythonConstantNotImplementedError as exc:
                        unimplemented(
                            gb_type="constant fold exception",
                            context=f"attempted to run function {fn} with arguments {args}",
                            explanation="Encountered exception when attempting to constant fold.",
                            hints=[*graph_break_hints.DYNAMO_BUG],
                            from_exc=exc,
                        )
                    # pyrefly: ignore [unbound-name]
                    return VariableTracker.build(tx, res)

            else:

                def constant_fold_handler(
                    tx: "InstructionTranslator",
                    args: Sequence[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # path with a runtime check
                    if check_unspec_or_constant_args(args, kwargs):
                        try:
                            res = fn(
                                *[x.as_python_constant() for x in args],
                                **{
                                    k: v.as_python_constant() for k, v in kwargs.items()
                                },
                            )
                        except AsPythonConstantNotImplementedError as exc:
                            unimplemented(
                                gb_type="constant fold exception",
                                context=f"attempted to run function {fn} with arguments {args}",
                                explanation="Encountered exception when attempting to constant fold.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                                from_exc=exc,
                            )
                        except Exception as exc:
                            raise_observed_exception(
                                type(exc),
                                tx,
                                args=list(map(ConstantVariable.create, exc.args)),
                            )
                        # pyrefly: ignore [unbound-name]
                        return VariableTracker.build(tx, res)
                    return None

            handlers.append(constant_fold_handler)

        def call_unimplemented(args: Sequence[VariableTracker]) -> None:
            real_arg_types = [arg.python_type_name() for arg in args]
            unimplemented(
                gb_type="Failed to trace builtin operator",
                context=f"builtin {fn.__name__} {arg_types} {has_kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator `{fn.__name__}` "
                f"with argument types {real_arg_types} (has_kwargs {has_kwargs})",
                hints=[
                    f"Avoid calling builtin `{fn.__name__}` with argument types {real_arg_types}. "
                    f"Consider using an equivalent alternative function/method to `{fn.__name__}`.",
                    "If you are attempting to call a logging function (e.g. `print`), "
                    "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                    "Please report an issue to PyTorch.",
                ],
            )

        if len(handlers) == 0:
            return lambda tx, args, kwargs: call_unimplemented(args)
        elif len(handlers) == 1:
            (handler,) = handlers

            def builtin_dispatch(
                tx: "InstructionTranslator",
                args: Sequence[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = handler(tx, args, kwargs)
                if rv:
                    return rv
                call_unimplemented(args)
                return rv

        else:

            def builtin_dispatch(
                tx: "InstructionTranslator",
                args: Sequence[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = None
                for fn in handlers:
                    rv = fn(tx, args, kwargs)
                    if rv:
                        return rv
                call_unimplemented(args)
                return rv

        return builtin_dispatch

    def call_vars(self, tx: "InstructionTranslator", *args: Any) -> VariableTracker:
        if len(args) == 0:
            unimplemented(
                gb_type="unimplemented builtin op vars() with no arguments",
                context=f"vars: {self} {args}",
                explanation=f"Dynamo does not know how to trace builtin operator {self.fn} with no arguments",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        assert len(args) == 1
        # vars(obj) is obj.__dict__ if __dict__ is present else TypeError
        try:
            return args[0].var_getattr(tx, "__dict__")
        except ObservedAttributeError:
            raise_observed_exception(TypeError, tx)

    def _handle_insert_op_in_graph(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        if kwargs and not self.tensor_args(*args, *kwargs.values()):
            return None

        # insert handling for torch function here
        from .builder import SourcelessBuilder
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        global BUILTIN_TO_TENSOR_RFN_MAP, BUILTIN_TO_TENSOR_FN_MAP
        if can_dispatch_torch_function(tx, args, kwargs):
            # Only remap the fn to tensor methods if we aren't exporting
            # export serde does not handle method descriptors today
            if not tx.export:
                # Ensure the builtin maps are populated before accessing them
                populate_builtin_to_tensor_fn_map()
                # Use sourceless builder, we built the map ourselves
                if not isinstance(args[0], TensorVariable):
                    if self.fn in BUILTIN_TO_TENSOR_RFN_MAP:
                        func = BUILTIN_TO_TENSOR_RFN_MAP[self.fn]
                    else:
                        func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]

                    tmp = args[0]
                    # swap args and call reverse version of func
                    args[0] = args[1]  # type: ignore[index]
                    args[1] = tmp  # type: ignore[index]
                else:
                    func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]
            else:
                func = self.fn

            fn_var = SourcelessBuilder.create(tx, func)

            return dispatch_torch_function(tx, fn_var, args, kwargs)

        fn = self.fn
        try:
            # Constant fold for constant tensor and python constants
            if self.python_and_tensor_constant_only(*args, **kwargs):
                from ..bytecode_transformation import unique_id
                from .functions import invoke_and_store_as_constant

                return invoke_and_store_as_constant(
                    tx, fn, unique_id(fn.__name__), args, kwargs
                )

            if fn in IN_PLACE_DESUGARING_MAP and isinstance(
                args[0], variables.ConstantVariable
            ):
                # In-place operators like += usually mustate tensor
                # values, but in the edge case of immutable values they
                # re-bind the variable.
                #
                # The easiest way to keep the graph consistent in this
                # scenario is to de-sugar eagerly.
                fn = IN_PLACE_DESUGARING_MAP[fn]
                args = [args[0], args[1]]  # type: ignore[assignment]

            if fn is operator.getitem and isinstance(args[1], SymNodeVariable):
                # Standard indexing will force specialization due to
                # __index__.  Rewrite as a regular torch op which will
                # trace fine
                fn = torch.select
                args = [
                    args[0],
                    variables.ConstantVariable.create(0),
                    args[1],
                ]  # type: ignore[assignment]

            # Interaction between ndarray and tensors:
            #   We prefer the tensor op whenever there are tensors involved
            if check_numpy_ndarray_args(args, kwargs) and not any(
                type(arg) is variables.TensorVariable for arg in args
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    numpy_operator_wrapper(fn),
                    *proxy_args_kwargs(args, kwargs),
                )

                return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)

            if (
                fn is operator.eq
                and len(args) == 2
                and isinstance(args[0], variables.TensorVariable)
            ):
                # Dynamo expects `__eq__` str while operator.eq gives just `eq`
                # TODO - supporting all comparison operators could also work but
                # it fails lots of tests because graph str changes.
                return args[0].call_method(tx, "__eq__", args[1:], kwargs)
            proxy = tx.output.create_proxy(
                "call_function",
                fn,
                *proxy_args_kwargs(args, kwargs),
            )
     
```



## High-Level Overview

"""Built-in function and type variable tracking for TorchDynamo's symbolic execution.This module contains variable tracker classes for Python built-in functions, types,and operations during graph compilation. It handles symbolic execution of:- Built-in functions (len, getattr, isinstance, etc.)- Type constructors (int, float, str, list, dict, etc.)- Built-in operators and methods- Special Python constructs (super, hasattr, etc.)Key classes:- BuiltinVariable: Tracks built-in functions and handles their execution- TypeVariable: Manages type constructor calls and type checking- SuperVariable: Handles super() calls in class hierarchiesThese variable trackers ensure that built-in Python operations are correctlyhandled during symbolic execution, either by executing them directly when safeor by creating appropriate graph nodes when needed.

This Python file contains 11 class(es) and 111 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GetMethodMode`, `BuiltinVariable`, `A`

**Functions defined**: `populate_builtin_to_tensor_fn_map`, `__torch_function__`, `create_with_source`, `_constant_fold_functions`, `can_constant_fold_through`, `_fx_graph_functions`, `_binops`, `_binop_handlers`, `user_defined_handler`, `__radd__`, `user_defined_inplace_handler`, `dynamic_handler`, `tuple_add_handler`, `size_add_handler`, `list_iadd_handler`, `expand_list_like`, `create_cmp_op_handlers`, `compare_by_value`, `handler`, `never`

**Key imports**: contextlib, functools, inspect, itertools, logging, math, operator, types, typing, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `math`
- `operator`
- `types`
- `typing`
- `unittest`
- `collections`: defaultdict, OrderedDict
- `collections.abc`: Callable, Iterable, KeysView, Sequence
- `torch`
- `torch._subclasses.meta_utils`: is_sparse_any
- `torch.overrides`: BaseTorchFunctionMode
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass
- `..`: config, graph_break_hints, polyfills, variables
- `..guards`: GuardBuilder, install_guard
- `..replay_record`: DummyModule
- `.base`: AsPythonConstantNotImplementedError, ValueMutationNew, VariableTracker
- `.constant`: ConstantVariable
- `.streams`: EventVariable, StreamVariable
- `torch._dynamo.codegen`: PyCodegen
- `torch._dynamo.symbolic_convert`: InstructionTranslator
- `.tensor`: supported_comparison_ops
- `.functions`: BaseUserFunctionVariable, UserFunctionVariable
- `.nn_module`: NNModuleVariable


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `builtin.py_docs.md`
- **Keyword Index**: `builtin.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
