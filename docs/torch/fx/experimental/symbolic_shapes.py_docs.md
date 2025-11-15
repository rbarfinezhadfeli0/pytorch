# Documentation: symbolic_shapes.py

## File Metadata
- **Path**: `torch/fx/experimental/symbolic_shapes.py`
- **Size**: 335453 bytes
- **Lines**: 8109
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
from __future__ import annotations

import sympy
from sympy import S

from torch._prims_common import BoolLike, FloatLike, IntLike


"""
``torch.fx.experimental.symbolic_shapes`` provides interfaces for interacting with
our symbolic shapes reasoning system that is used heavily in torch.compile.  Although
this is not generally considered public API, when writing framework code in PyTorch
as well as extensions to PyTorch (e.g., in custom operator implementations), you may
need to make use of these APIs to setup dynamic shapes support appropriately.
"""

import abc
import atexit
import collections
import dis
import functools
import hashlib
import inspect
import itertools
import logging
import math
import operator
import os
import re
import sys
import threading
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    Any,
    cast,
    Generic,
    NamedTuple,
    NoReturn,
    Optional,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import deprecated, ParamSpec

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import SymBool, SymFloat, SymInt
from torch._C._functorch import get_unwrapped, is_batchedtensor
from torch._guards import ShapeGuard, SLoc, Source, TracingContext
from torch._logging import dtrace_structured, LazyString, structured, trace_structured
from torch._subclasses.meta_utils import is_sparse_any
from torch._utils_internal import signpost_event
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
    FakeTensorMeta,
    record_shapeenv_event,
    replay_shape_env_events,
    shape_env_check_state_equal,
    ShapeEnvEvent,
)
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import (
    Application,
    CeilToInt,
    CleanDiv,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Mod,
    PythonMod,
    TruncToInt,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import CppPrinter, PythonPrinter
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import make_symbol, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRangeError,
    ValueRanges,
)
from torch.utils._traceback import CapturedTraceback, format_frame


if TYPE_CHECKING:
    import types

    from torch import Tensor
    from torch._dynamo.source import TensorPropertySource
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType


InputList = list
DimList = list

log = logging.getLogger(__name__)


class GuardOnDataDependentSymNode(RuntimeError):
    cond: sympy.Basic

    def __init__(self, cond: sympy.Basic, *args: Any) -> None:
        super().__init__(*args)
        self.cond = cond


class PendingUnbackedSymbolNotFound(RuntimeError):
    pass


aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = [
    "size_hint",
    "guard_or_false",
    "guard_or_true",
    "has_symbolic_sizes_strides",
    "create_contiguous",
    "ShapeEnv",
    "is_concrete_int",
    "is_concrete_float",
    "is_concrete_bool",
    "has_static_value",
    "guard_int",
    "guard_float",
    "guard_scalar",
    "canonicalize_bool_expr",
    "hint_int",
    "SYMPY_INTERP",
    "free_symbols",
    "is_symbol_binding_fx_node",
    "is_nested_int",
    "SHAPEENV_EVENT_KEY",
    "CURRENT_NODE_KEY",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "sym_and",
    "sym_eq",
    "sym_or",
    "SymbolicContext",
    "StatelessSymbolicContext",
    "StatefulSymbolicContext",
    "SubclassSymbolicContext",
    "SymIntSymbolicContext",
    "TrackedFake",
    "statically_known_true",
    "statically_known_false",
    "guard_size_oblivious",
    "check_consistent",
    "compute_unbacked_bindings",
    "ConvertIntKey",
    "rebind_unbacked",
    "resolve_unbacked_bindings",
    "is_accessor_node",
    "ValueRangesSLoc",
    "SymIntEqByExpr",
    "Specialization",
]

# FX node metadata keys for symbolic shape FX graph.
SHAPEENV_EVENT_KEY = "shapeenv_event"
CURRENT_NODE_KEY = "current_node"


def log_lru_cache_stats(wrapped_f: functools._lru_cache_wrapper[object]) -> None:
    log.debug(
        "lru_cache_stats %s: %s",
        wrapped_f.__name__,  # type: ignore[attr-defined]
        wrapped_f.cumulative_cache_info(),  # type: ignore[attr-defined]
    )


# Note about Sympy Expr/SympyBoolean/Basic typing: the Sympy hierarchy is
#
#   Basic
#       Expr
#       SympyBoolean
#           Relational
#
# Notably, Expr and SympyBoolean are not related.  So use Basic when the
# expression could denote int, float OR bool, and otherwise use the more
# specific Expr for int/float and SympyBoolean for bool.
#
# In obscure Meta only situations, sympy.logic.boolalg doesn't exist at runtime.
# So make sure only type checker evaluates this alias.
# Xref: https://www.internalfb.com/diff/D53324783
SympyBoolean: TypeAlias = "sympy.logic.boolalg.Boolean"


_T = TypeVar("_T")
_SympyT = TypeVar("_SympyT", sympy.Expr, SympyBoolean, sympy.Basic)


class SymIntEqByExpr:
    """
    This is a wrapper around SymInt which has alternative semantics for
    equality and pickling.  Specifically, instead of erroring or guarding, we
    instead will hash/compare equality based on the underlying sympy
    expression; e.g., s0 and s1 will always compare as False.

    NB: This does NOT do fancy analysis that maybe_evaluate_static does;
    we can only reason through equalities that occur because to expressions
    canonicalize to the same expression via regular simplification.
    """

    @staticmethod
    def _extract(val: Union[torch.SymInt, int]) -> sympy.Expr:
        if isinstance(val, torch.SymInt):
            return val.node.expr
        else:
            return sympy.Integer(val)

    def __init__(self, val: Union[torch.SymInt, int]) -> None:
        self.val: sympy.Expr = SymIntEqByExpr._extract(val)

    def __repr__(self) -> str:
        return repr(self.val)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SymIntEqByExpr)
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)


def _nested_int_aware_sort(
    tup: tuple[IntLikeType, int],
) -> tuple[int, IntLikeType, int]:
    return (
        # Order nested ints by their coefficients.
        # 1 here to order nested ints after non-nested-ints.
        (1, tup[0].node.nested_int_coeff(), tup[1])
        if is_nested_int(tup[0])
        else (0, *tup)
    )


def size_hint(x: int | torch.SymInt, *, allow_none: bool = False) -> int | None:
    """Gets a size hint for a given expression from the underlying shapes we had.
    Does not introduce a guard, so only use this when you can guarantee that
    your code is still valid for arbitrary shapes (such as optimization decisions)
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    return x.node.shape_env.size_hint(x.node.expr, allow_none=allow_none)


# Wrapper on lru_cache that reports statistics at process end
def lru_cache(
    maxsize: Optional[int],
) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]:
    def inner(f: Callable[..., _T]) -> functools._lru_cache_wrapper[_T]:
        wrapped_f = functools.lru_cache(maxsize)(f)
        old_cache_clear = wrapped_f.cache_clear
        prev_hits = 0
        prev_misses = 0

        # TODO: There's a ref-cycle here (wrapped_f -> cumulative_cache_info
        # -> wrapped_f) but cannot be solved with weakref as wrapped_f is not
        # weakref'able on some versions of Python

        def cumulative_cache_info() -> functools._CacheInfo:
            cur = wrapped_f.cache_info()
            return functools._CacheInfo(
                prev_hits + cur.hits,
                prev_misses + cur.misses,
                cur.maxsize,
                cur.currsize,
            )

        def new_cache_clear() -> None:
            nonlocal prev_hits, prev_misses
            cur = wrapped_f.cache_info()
            prev_hits += cur.hits
            prev_misses += cur.misses
            old_cache_clear()

        wrapped_f.cache_clear = new_cache_clear  # type: ignore[attr-defined, method-assign]
        wrapped_f.cumulative_cache_info = cumulative_cache_info  # type: ignore[attr-defined, method-assign]
        if log.isEnabledFor(logging.DEBUG):
            atexit.register(log_lru_cache_stats, wrapped_f)  # type: ignore[arg-type]
        return wrapped_f

    return inner


# These are modules that contain generic code for interacting with ShapeEnv
# which are unlikely to identify a particular interesting guard statement
@lru_cache(None)
def uninteresting_files() -> set[str]:
    import torch._compile
    import torch._dynamo.eval_frame
    import torch._inductor.sizevars
    import torch._library.custom_ops
    import torch._library.fake_impl
    import torch._logging
    import torch._subclasses.fake_tensor
    import torch._subclasses.meta_utils
    import torch.export._trace

    mods = [
        sys.modules[__name__],
        torch.export._trace,
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch.fx.interpreter,
        torch.fx._symbolic_trace,
        torch,
        torch._compile,
        torch._dynamo.eval_frame,
        torch._inductor.sizevars,
        torch._library.custom_ops,
        torch._library.fake_impl,
        torch._subclasses.meta_utils,
        torch._subclasses.fake_tensor,
        torch._logging._internal,
        torch._logging.structured,
    ]
    import torch._dynamo.guards

    return (
        {inspect.getfile(m) for m in mods}
        | torch._dynamo.guards.uninteresting_files()
        | {"<string>"}
    )


class ConstraintViolationError(RuntimeError):
    pass


def has_symbolic_sizes_strides(elem: torch.Tensor) -> bool:
    return elem._has_symbolic_sizes_strides


Int: TypeAlias = Union[torch.SymInt, int]


def create_contiguous(shape: Sequence[Int]) -> list[Int]:
    strides: list[Int] = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])  # type: ignore[operator]
    return list(reversed(strides))


def hint_int(a: Union[torch.SymInt, int], fallback: Optional[int] = None) -> int:
    """
    Retrieve the hint for an int (based on the underlying real values as observed
    at runtime).  If no hint is available (e.g., because data dependent shapes),
    if fallback is not None, use that instead (otherwise raise an error).
    """
    if isinstance(a, torch.SymInt):
        return a.node.require_hint(fallback)
    assert type(a) is int, a
    return a


Scalar: TypeAlias = Union[torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool]


def has_hint(a: Scalar) -> bool:
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True


def is_concrete_int(a: IntLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or int): Object to test if it int
    """
    assert isinstance(a, (SymInt, int))

    if isinstance(a, int):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Integer):
        return True

    return False


def is_concrete_float(a: FloatLikeType) -> bool:
    r"""Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or float): Object to test if it float
    """
    assert isinstance(a, (SymFloat, float))

    if isinstance(a, float):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Float):
        return True

    return False


def is_concrete_bool(a: BoolLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymBool is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymBool or bool): Object to test if it bool
    """
    assert isinstance(a, (SymBool, bool))

    if isinstance(a, bool):
        return True

    if isinstance(
        a.node.expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)
    ):
        return True

    return False


def has_static_value(a: Union[SymBool, SymFloat, SymInt, bool, float, int]) -> bool:
    """
    User-code friendly utility to check if a value is static or dynamic.
    Returns true if given a constant, or a symbolic expression with a fixed value.

    Args:
        a (Union[SymBool, SymFloat, SymInt, bool, float, int]): Object to test
    """
    assert isinstance(a, BoolLike + FloatLike + IntLike)
    if (
        isinstance(a, BoolLike)
        and is_concrete_bool(a)  # type: ignore[arg-type]
        or isinstance(a, FloatLike)
        and is_concrete_float(a)  # type: ignore[arg-type]
        or isinstance(a, IntLike)
        and is_concrete_int(a)  # type: ignore[arg-type]
    ):
        return True

    assert isinstance(a, py_sym_types)
    return a.node.shape_env.bound_sympy(a.node.expr).is_singleton()  # type: ignore[union-attr]


def guard_size_oblivious(expr: Union[torch.SymBool, bool]) -> bool:
    """
    Perform a guard on a symbolic boolean expression in a size oblivious way.
    This is typically used when a non-oblivious test would result in a guard
    on a data dependent value of which we don't know the value of at compile time.
    When a guard is tested this way, we may diverge in behavior from how regular
    PyTorch semantics would treat it.  For more information, see
    https://github.com/pytorch/pytorch/pull/118579
    """
    if isinstance(expr, torch.SymBool):
        return expr.node.guard_size_oblivious("", 0)
    else:
        assert isinstance(expr, bool), expr
        return expr


def check_consistent(new: _T, old: _T) -> None:
    """
    Test that two "meta" values (typically either Tensor or SymInt) have
    the same values, e.g., after retracing.  If we don't understand the
    quantities in question, we'll just skip the consistency check.
    """
    # TODO: do boolean equality test too, see
    # https://github.com/pytorch/pytorch/issues/124110
    scalar_types = (torch.SymInt, torch.SymFloat, int, float)

    if isinstance(new, torch.Tensor):
        assert isinstance(old, torch.Tensor)
        torch._check(
            old.dim() == new.dim(), lambda: f"{old.shape} != {new.shape} (old != new)"
        )
        # Do this manually so that each individual test is irrefutable
        # (TODO: should be a helper for this, maybe sym_eq?  That
        # gives us a compound expression and I'm not sure it
        # simplifies right now)
        for i, j in zip(old.shape, new.shape):
            torch._check(i == j, lambda: f"{old.shape} != {new.shape} (old != new)")
    # NB: bool is subclass of int
    elif isinstance(new, scalar_types) and not isinstance(new, bool):
        assert isinstance(old, scalar_types) and not isinstance(old, bool), (
            f"{old} != {new}"
        )
        torch._check(old == new, lambda: f"{old} != {new} (old != new)")


def resolve_unbacked_bindings(
    shape_env: Optional[ShapeEnv],
    bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]],
) -> Optional[dict[sympy.Symbol, pytree.KeyPath]]:
    """
    When we do fake tensor prop, we oftentimes will allocate new unbacked symints.
    We then run proxy tensor mode, which populates node.meta["unbacked_bindings"]
    with these new symints. To ensure consistency we use PropagateUnbackedSymInts
    to rename unbacked bindings to their old ones. But all of the node metas are
    still using the old bindings from before the renaming. This function helps to
    post facto apply any renamings discovered in the PropogateUnbackedSymInts pass.
    """
    if bindings is None:
        return None
    assert shape_env is not None
    return {shape_env.unbacked_renamings.get(k, k): v for k, v in bindings.items()}


Result: TypeAlias = Union[torch.Tensor, tuple[torch.Tensor, ...]]


def rebind_unbacked(
    shape_env: Optional[ShapeEnv], n: torch.fx.Node, result: Result
) -> None:
    """
    Suppose we are retracing a pre-existing FX graph that previously had
    fake tensor propagation (and therefore unbacked SymInts).  When we retrace,
    we re-propagate fake tensors, which results in new unbacked SymInts.
    When this happens, we need to tell the shape environment about the equivalence
    of the old and new unbacked SymInts.  Pass us the old torch.fx.Node (which
    has the old binding information) and the new result (which we can extract the
    new unbacked SymInts out from).
    """

    # Inputs never need rebinding
    if n.op == "placeholder":
        return

    if bindings := resolve_unbacked_bindings(
        shape_env, n.meta.get("unbacked_bindings")
    ):
        assert shape_env is not None
        for raw_u0, path in bindings.items():
            u1 = pytree.key_get(result, path)

            # Sometimes, things were previously unbacked bindings become constants.
            # There are two situations this can happen.
            #
            # First, you might have a runtime assert that causes the
            # constant-ification.  In this case, the /binding/ itself will
            # still be an unbacked symbol (because we will only force it
            # to be a constant later in fake tensor propagation).  In this
            # case, u1 is a SymInt and we still do all our work as normal.
            #
            # But second, it might be that fake tensor propagation DIRECTLY
            # converted the unbacked SymInt into a constant.  This happens
            # more rarely, but we have identified two situations it can
            # validly occur:
            #
            # - If you have a tensor_version operator, these are initially
            #   allocated as unbacked SymInts, but after AOTAutograd they
            #   get forced specialized to specific values.  In this case,
            #   there is no reason to do runtime asserts on them, this is
            #   just a hack to properly keep track of them to start.
            #
            # - If you have an item() call on a constant tensor, the result
            #   of the item() call is constant and we do not need runtime
            #   asserts on this symbol.  In
            #   https://github.com/pytorch/pytorch/issues/140625 we have a
            #   case where in the initial trace of the program we are unable
            #   to determine that torch.tensor is constant, but then
            #   subsequent passes cause torch.tensor to become a constant and
            #   then the unbacked symbol goes poof.
            #
            # In all of these cases, it is no longer necessary to generate
            # deferred runtime asserts, since other subsystems (e.g., the
            # constant-ification pass) ensure that the quantity is now truly
            # static and cannot change at runtime.  So it's OK to discard
            # in these situations.
            #
            # There is one more hazard (re
            # https://github.com/pytorch/pytorch/issues/141248), the problem
            # is that you can end up with "dangling" unbacked symbols that
            # exist in the ShapeEnv but are never bound anywhere.  You might
            # like an invariant that unbacked symbols never get lost.  But
            # we do not have this invariant, so do not try to enforce it.
            if isinstance(u1, (int, float)):
                log.info(
                    "rebind_unbacked: discard %s %s %s -> %s",
                    n.target,
                    raw_u0,
                    path,
                    u1,
                )
                continue

            # We only care about rebinding unbacked things
            if u1.node.hint is not None:
                continue

            # unbacked symbols bindings might be replaced to other backed or
            # unbacked replacements.
            #
            # Example:
            #   u = x.item()
            #   torch._check(u == 5)
            #
            # The safest approach is to retrieve raw_u1 from u1.node._expr
            # and perform the rebinding on the original unbacked symbol,
            # even if it’s no longer directly referenced.
            #
            # In other words, we should always rebind the original symbol
            # before any replacements are applied.
            #   u0 -> u0 == s1
            raw_u1 = u1.node._expr

            # TODO Do we still need this logic below?
            # Simplify SymBool binding
            if (
                isinstance(raw_u1, sympy.Piecewise)
                and len(raw_u1.args) == 2
                and (
                    raw_u1_args0 := cast(
                        tuple[sympy.Basic, sympy.Basic], raw_u1.args[0]
                    )
                )
                and raw_u1_args0[0] == 1
                and isinstance(eq := raw_u1_args0[1], sympy.Eq)
                and isinstance(new_raw_u1 := eq.lhs, sympy.Symbol)
                and shape_env.var_to_range[new_raw_u1].issubset(ValueRanges(0, 1))
                and eq.rhs == 1
                and cast(tuple[sympy.Basic, sympy.Basic], raw_u1.args[1]) == (0, True)
            ):
                # This is what the pattern match above is testing
                repacked = _sympy_cast_symbool_to_symint_guardless(
                    sympy.Eq(new_raw_u1, 1)
                )
                assert repacked == raw_u1, f"{repacked} != {raw_u1}"
                # Cancel the to_int(to_bool(x)). This is sound because x in
                # [0, 1]

                raw_u1 = new_raw_u1

            if not isinstance(raw_u1, sympy.Symbol):
                assert not raw_u1.free_symbols, (
                    f"should have been constant, but got {raw_u1}"
                )
                continue

            # The old and new could be the same if you improperly hit the memo
            # while retracing.  Make sure you updated FakeTensorMode.epoch
            assert raw_u0 != raw_u1, f"{raw_u0} possible memo disaster"
            # Reuse the OLD symbol name
            shape_env._rename_unbacked_to(raw_u1, raw_u0)


# NB: You could try to expand this to cover more cases by simply
# detecting whenever you have an int output, but this is a bit
# dangerous in case someone adds a function that returns an int but is
# mutating.  So manually whitelist for now.
def is_accessor_node(node: torch.fx.Node) -> bool:
    """
    Helper function to determine if a node is trying to access
    a symbolic integer such as size, stride, offset or item. Currently
    primarily only used in a DCE pass to figure out purity.
    """

    # Dynamo only exercised condition
    if (
        node.op == "call_method"
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[0].meta.get("example_value"), torch.Tensor)
        and node.target in ["size", "stride", "storage_offset", "item"]
    ):
        return True

    if node.op == "call_function" and node.target in [
        torch.ops.aten.sym_size,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_stride,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_storage_offset,
        torch.ops.aten.sym_storage_offset.default,
        torch.ops.aten.sym_numel.default,
    ]:
        return True

    return False


def canonicalize_bool_expr(expr: _T) -> _T:
    """
    Canonicalize a boolean expression by transforming it into a lt / le
    inequality and moving all the non-constant terms to the rhs.
    We canonicalize And / Ors / Not via cnf and then canonicalize their subexpr
    recursively
    nb. sympy.Rel.canonical is not good enough https://github.com/sympy/sympy/issues/25924

    Args:
        expr (sympy.Expr): Expression to canonicalize
    """
    # Canonicalise an inequality by transforming it into a lt / le
    # inequality and moving all the non-constant terms to the rhs
    # We canonicalise And / Ors / Not via cnf
    # nb. Relational.canonical in sympy is broken
    # https://github.com/sympy/sympy/issues/25924

    if not isinstance(
        expr, (sympy.Rel, sympy.And, sympy.Or, sympy.Not, sympy.Eq, sympy.Ne)
    ):
        return expr

    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        expr = sympy.logic.boolalg.to_cnf(expr)
    return _canonicalize_bool_expr_impl(expr)  # type: ignore[arg-type, return-value]


def _sympy_from_args(
    cls: type[Union[sympy.Add, sympy.Mul]],
    args: list[sympy.Expr],
    sort: bool = True,
    is_commutative: Optional[bool] = None,
) -> sympy.Expr:
    """
    Create a sympy expression from a list of arguments, optimizing for performance.

    This function creates a sympy Add or Mul expression from a list of arguments
    while avoiding expensive operations like flattening. It handles sorting the
    arguments appropriately based on the expression type.

    Args:
        cls: The sympy class to create (Add or Mul)
        args: List of sympy expressions to combine
        sort: Whether to sort the arguments (default: True)
        is_commutative: Whether the operation is commutative (default: None)

    Returns:
        A sympy expression of type cls combining all arguments

    Raises:
        ValueError: If cls is not sympy.Add or sympy.Mul
    """

    if not args:
        return cls.identity  # type: ignore[union-attr]

    # These args are already in canonical form, so we avoid calling
    # Add(*args) to avoid expensive Add.flatten operation
    if sort:
        if cls is sympy.Add:
            sort_fn = sympy.core.add._addsort
        elif cls is sympy.Mul:
            sort_fn = sympy.core.mul._mulsort
        else:
            raise ValueError(f"Unknown cls: {cls}")

        # we don't support non commutative with sort
        assert is_commutative is True
        if args[0].is_Number:
            rest = args[1:]
            sort_fn(rest)
            return cls._from_args([args[0]] + rest, is_commutative=is_commutative)  # type: ignore[attr-defined]
        else:
            args = args.copy()
            sort_fn(args)
            return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]
    else:
        # if the args are already sorted, we create directly
        return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]


def _canonicalize_bool_expr_impl(expr: SympyBoolean) -> SympyBoolean:
    """
    After canonicalization, we are guaranteed to have eliminated Ge/Gt relations
    (rewriting them to Le/Lt, respectively).
    """
    if isinstance(expr, (sympy.And, sympy.Or)):
        return type(expr)(*map(canonicalize_bool_expr, expr.args))

    opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
    t: Union[type[Any]]
    if isinstance(expr, tuple(opposite.keys())):
        rhs = expr.lhs - expr.rhs  # type: ignore[attr-defined]
        t = opposite[type(expr)]  # type: ignore[index]
    else:
        assert isinstance(expr, (sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne))
        rhs = expr.rhs - expr.lhs
        t = type(expr)

    def is_neg(t: sympy.Expr) -> bool:
        return (t.is_Number and t.is_negative) or (
            isinstance(t, sympy.Mul) and t.args[0].is_Number and t.args[0].is_negative
        )

    lhs = S.Zero
    rhs = _reduce_to_lowest_terms(rhs)
    if isinstance(rhs, sympy.Add):
        pos = []
        neg = []
        for term in rhs.args:
            if is_neg(term):
                neg.append(-term)
            else:
                pos.append(term)
        # these are already sorted
        rhs = _sympy_from_args(sympy.Add, pos, sort=False, is_commutative=True)
        # the terms were changed, so needs a sorting
        lhs = _sympy_from_args(sympy.Add, neg, sort=True, is_commutative=True)
    elif is_neg(rhs):
        # lhs == 0
        lhs, rhs = -rhs, S.Zero
    # We don't have to evaluate here because lhs, rhs came from a Boolean
    # and it was already simplified
    return t(lhs, rhs, evaluate=False)


def _reduce_to_lowest_terms(expr: sympy.Expr) -> sympy.Expr:
    """
    Eliminates any integer factor from a given expression.
    E.g., 6x + 4y reduces to 3x + 2y.

    Useful when an expression is == or != to 0.
    """

    def integer_coefficient(x: sympy.Expr) -> int:
        if x.is_Integer:
            return abs(int(x))
        elif x.is_Mul:
            # If one of the args of a Mul is an Integer, it is the
            # first arg. eg: args(2*x*3*y) == (6, x, y)
            return abs(int(x.args[0])) if x.args[0].is_Integer else 1  # type: ignore[call-overload]
        else:
            return 1

    def div_by_factor(x: sympy.Expr, factor: int) -> sympy.Expr:
        if x.is_Integer:
            return x / factor
        elif x.is_Mul:
            if x.args[0] != factor:
                args = [x.args[0] / sympy.Integer(factor), *x.args[1:]]
            else:
                # Mul._from_args require a canonical list of args
                # so we remove the first arg (x.args[0] / factor) if it was 1
                args = list(x.args[1:])
            return _sympy_from_args(sympy.Mul, args, is_commutative=x.is_commutative)
        else:
            raise AssertionError(f"illegal arg to div_by_factor: {x}")

    if expr.is_Add:
        atoms = cast(Sequence[sympy.Expr], expr.args)
        factor = functools.reduce(math.gcd, map(integer_coefficient, atoms))
        if factor == 1:
            return expr
        # pyrefly: ignore [bad-argument-type]
        atoms = [div_by_factor(x, factor) for x in atoms]
        return _sympy_from_args(
            sympy.Add, atoms, sort=True, is_commutative=expr.is_commutative
        )
    elif expr.is_Integer:
        return S.One
    elif expr.is_Mul:
        return div_by_factor(expr, integer_coefficient(expr))
    return expr


def is_nested_int(s: IntLikeType) -> TypeGuard[SymInt]:
    return isinstance(s, torch.SymInt) and s.node.is_nested_int()


IterateExprsAtom: TypeAlias = Union[
    SymInt, SymFloat, SymBool, int, float, bool, sympy.Basic, torch.Tensor
]
IterateExprs: TypeAlias = Union[IterateExprsAtom, Sequence[IterateExprsAtom]]


def _iterate_exprs(val: IterateExprs) -> Iterator[sympy.Basic]:
    """
    Recursively iterate through a value and yield all sympy expressions contained within it.

    This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
    any symbolic expressions they contain. It's used for operations like finding free symbols
    in complex nested structures.

    Args:
        val: The value to extract sympy expressions from. Can be a symbolic type (SymInt, SymFloat, SymBool),
             a sympy expression, a primitive type (int, float, bool), a container (tuple, list),
             a sparse tensor, a regular tensor, None, or a torch.Generator.

    Yields:
        sympy.Basic: Each sympy expression found in the value.

    Raises:
        AssertionError: If the value is of an unsupported type.
    """
    # This is almost close enough to implement in terms of _iterate_nodes()
    # except that it needs to handle `list[sympy.Basic]` which _iterate_nodes()
    # can't handle.
    if isinstance(val, SymTypes):
        # This allow applies to the jagged layout NestedTensor case as
        # nested ints are not symbolic
        if is_symbolic(val):
            yield val.node.expr
    elif isinstance(val, SymNode):
        yield val.expr
    elif isinstance(val, sympy.Basic):
        yield val
    elif isinstance(val, (int, float, bool)):
        pass
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iterate_exprs(s)
    elif is_sparse_any(val):
        yield from _iterate_exprs(val.size())
    elif isinstance(val, torch.Tensor):
        yield from _iterate_exprs(val.size())
        yield from _iterate_exprs(val.stride())
        yield from _iterate_exprs(val.storage_offset())
    elif val is None:
        pass
    # see Note: [Generator arguments in AOTDispatcher]
    elif isinstance(val, torch.Generator):
        pass
    else:
        raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")


def _iterate_nodes(val: Any) -> Iterator[SymNode]:
    """
    Recursively iterate through a value and yield all SymNodes contained
    within it.
    """
    if isinstance(val, SymNode):
        yield val
    elif isinstance(val, py_sym_types):
        # This allow applies to the jagged layout NestedTensor case as
        # nested ints are not symbolic
        if is_symbolic(val):
            yield val.node
    elif isinstance(val, (tuple, list, torch.Size)):
        for s in val:
            yield from _iterate_nodes(s)
    elif isinstance(val, torch.Tensor):
        yield from _iterate_nodes(val.size())
        if not is_sparse_any(val):
            yield from _iterate_nodes(val.stride())
            yield from _iterate_nodes(val.storage_offset())


def free_symbols(val: IterateExprs) -> OrderedSet[sympy.Symbol]:
    """
    Recursively collect all free symbols from a value.

    This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
    all sympy symbols contained within them. It's useful for finding all symbolic variables
    that a complex nested structure depends on.

    Args:
        val: The value to extract symbols from. Can be a symbolic type (SymInt, SymFloat, SymBool),
             a container (tuple, list), a tensor, or None.

    Returns:
        OrderedSet[sympy.Symbol]: An ordered set of all free symbols found in the value.
    """
    if val is None:
        return OrderedSet()

    itr = _iterate_exprs(val)

    # we need at least 1 to call union, so we hand code the identity
    try:
        first_expr = next(itr)
    except StopIteration:
        return OrderedSet()

    # TODO: Apparently, returning an OrderedSet here breaks
    # python test/distributed/tensor/test_dtensor_compile.py TestDTensorCompile.test_dtensor_dynamic
    return first_expr.free_symbols.union(*(e.free_symbols for e in itr))  # type: ignore[return-value]


def has_free_symbols(val: IterateExprs) -> bool:
    """Faster version of bool(free_symbols(val))"""
    return not all((e.is_number or e.is_Boolean) for e in _iterate_exprs(val))


def has_free_unbacked_symbols(x: IterateExprs) -> bool:
    """Faster version of bool(free_unbacked_symbols(val))"""
    from sympy.core.traversal import iterargs

    for s in _iterate_exprs(x):
        for arg in iterargs(s):
            if arg.is_Symbol and symbol_is_type(
                arg, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
            ):
                return True
    return False


def free_unbacked_symbols(x: IterateExprs) -> OrderedSet[sympy.Symbol]:
    """Like free_symbols, but filtered to only report unbacked symbols"""

    # NB: keep synced with is_unbacked_symint
    return OrderedSet(
        s
        for s in free_symbols(x)
        if symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
    )


def _free_non_source_unbacked_symbols(
    x: IterateExprs, unbacked_inputs: OrderedSet[sympy.Symbol]
) -> OrderedSet[sympy.Symbol]:
    """Unbacked symbols that are not inputs to the graph. These are symbols that originated from
    data-dependent operations as opposed to mark_unbacked calls."""
    unbacked_symbols = free_unbacked_symbols(x)
    non_source_symbols = unbacked_symbols - unbacked_inputs
    return non_source_symbols


# WARNING: Don't use this on Dynamo produced graphs, they don't have meta
# setup!
def is_symbol_binding_fx_node(node: torch.fx.Node) -> Optional[sympy.Symbol]:
    """
    Check if a given FX node is a symbol binding node.

    A symbol binding node is one that has a SymInt value in its meta that contains
    a sympy Symbol expression, and is either a placeholder node or contains unbacked symbols.

    Args:
        node (torch.fx.Node): The FX node to check

    Returns:
        Optional[sympy.Symbol]: The sympy Symbol if the node is a symbol binding node, None otherwise
    """
    if (
        "val" in node.meta
        and isinstance(node.meta["val"], torch.SymInt)
        and isinstance(node.meta["val"].node.expr, sympy.Symbol)
        and (
            node.op == "placeholder"
            or free_unbacked_symbols(node.meta["val"].node.expr)
        )
    ):
        return node.meta["val"].node.expr
    return None


def find_symbol_binding_fx_nodes(
    graph: torch.fx.Graph,
) -> dict[sympy.Symbol, torch.fx.Node]:
    """
    Find all nodes in an FX graph that bind sympy Symbols.

    This function scans through all nodes in the given FX graph and identifies
    nodes that bind sympy Symbols (typically placeholder nodes with SymInt values).
    When multiple nodes bind the same symbol, only the first occurrence is kept.

    Args:
        graph: The FX graph to search for symbol binding nodes

    Returns:
        A dictionary mapping from sympy Symbols to their binding FX nodes
    """
    r = {}
    # NB: Prefer first occurrence of symbol
    for node in graph.nodes:
        if (s := is_symbol_binding_fx_node(node)) is not None and s not in r:
            r[s] = node
    return r


@dataclass(frozen=True)
class Specialization:
    """
    This class is used in multi-graph compilation contexts where we generate
    multiple specialized graphs and dispatch to the appropriate one at runtime.
    This allows us to optimize the trade-off between performance and generality
    by creating specialized versions for common patterns (e.g., x.shape[0] % 16 == 0)
    while maintaining a general fallback.
    """

    source: TensorPropertySource
    check_fn: Callable


# Analogous to ConvertIntSource
@dataclass(frozen=True)
class ConvertIntKey:
    def __str__(self) -> str:
        return ".cast_symbool_to_symint_guardless()"

    def get(self, b: bool) -> IntLikeType:
        """Get the int value from bool"""
        return cast_symbool_to_symint_guardless(b)


@dataclass(frozen=True)
class CallMethodKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}()"

    def get(self, o: Any) -> Any:
        """Call the method on object"""
        return getattr(o, self.name)()


@dataclass(frozen=True)
class InnerTensorKey:
    inner_name: str

    def __str__(self) -> str:
        return f".{self.inner_name}"

    def get(self, o: Any) -> Any:
        """Get the inner tensor attribute"""
        return getattr(o, self.inner_name)


@dataclass(frozen=True)
class DivideByKey:
    divisor: IntLikeType

    def __str__(self) -> str:
        return f".__floordiv__({self.divisor})"

    def get(self, o: int) -> int:
        """Divide object by divisor"""
        return o // self.divisor


def _free_unbacked_symbols_with_path(
    a: object,
    path: pytree.KeyPath,
    real: Optional[object] = None,
    shape_env: Optional[ShapeEnv] = None,
    pending: Optional[set[sympy.Symbol]] = None,
    simplify: bool = False,
) -> dict[sympy.Symbol, pytree.KeyPath]:
    """
    Recursively traverses a structure to find unbacked symbols and their access paths.

    This function walks through tensors, lists, tuples, and symbolic values to locate
    unbacked symbols that are in the pending set, and returns a mapping from those
    symbols to their access paths in the structure.

    Args:
        a: The object to traverse (tensor, list, tuple, SymInt, etc.)
        path: The current path in the object tree
        real: Optional real tensor corresponding to the fake tensor being traversed
        shape_env: Optional ShapeEnv to register unbacked values with
        pending: Set of unbacked symbols to look for (will be modified in-place)
        simplify: Whether to use simplified expressions

    Returns:
        A dictionary mapping unbacked symbols to their access paths
    """
    go = functools.partial(
        _free_unbacked_symbols_with_path,
        shape_env=shape_env,
        pending=pending,
        simplify=simplify,
    )

    def expr(s: Union[SymInt, SymFloat, SymBool]) -> sympy.Expr:
        if simplify:
            return s.node.expr
        # (When called from compute_unbacked_bindings)
        # NB: Intentionally access _expr, not expr, do not want
        # simplification!
        return s.node._expr

    if pending is None:
        pending = set()
    r = {}
    if isinstance(a, (tuple, list)):
        # NB: real is apparently not always a tuple/list here
        # python test/inductor/test_torchinductor.py CpuTests.test_index_propagation_nested_indirect_indexing_cpu
        for i in range(len(a)):
            r.update(
                go(
                    a[i],
                    path + (pytree.SequenceKey(i),),
                    real=real[i] if real is not None else None,  # type: ignore[index]
                )
            )
    elif is_traceable_wrapper_subclass(a):
        # TODO: Determine if this is correct
        attrs, _ = a.__tensor_flatten__()
        for attr in attrs:
            sub = getattr(a, attr)
            r.update(go(sub, path + (InnerTensorKey(attr),)))
    elif isinstance(a, torch.Tensor) and is_batchedtensor(a):
        unwrapped_tensor = get_unwrapped(a)
        r.update(go(unwrapped_tensor, path))
    elif isinstance(a, torch.Tensor) and not is_batchedtensor(a):
        from torch._subclasses.fake_tensor import FakeTensor

        assert isinstance(a, FakeTensor)
        r.update(
            go(
                a.size(),
                path + (CallMethodKey("size"),),
                real=a.real_tensor.size() if a.real_tensor is not None else None,
            )
        )
        if a.layout not in [
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            r.update(
                go(
                    a.stride(),
                    path + (CallMethodKey("stride"),),
                    real=a.real_tensor.stride() if a.real_tensor is not None else None,
                )
            )
        r.update(
            go(
                a.storage_offset(),
                path + (CallMethodKey("storage_offset"),),
                real=(
                    a.real_tensor.storage_offset()
                    if a.real_tensor is not None
                    else None
                ),
            )
        )

    elif (
        isinstance(a, (torch.SymInt, torch.SymFloat))
        and isinstance(s := expr(a), sympy.Symbol)
        and s in pending
    ):
        r[s] = path
        if shape_env and real is not None:
            assert isinstance(real, (int, float))

            shape_env.set_unbacked_var_to_val(s, real)

        pending.remove(s)
    # When an unbacked SymInt is perfectly divisible by an integer
    # constant, we replace it with the integer constant to improve
    # reasoning capabilities.  However, in synthetic examples, it is
    # then possible that the factor never is explicitly allocated.
    # Fortunately, we can compute it by division.
    elif (
        isinstance(a, torch.SymInt)
        and isinstance(s := expr(a), sympy.Mul)
        and len(s.args) == 2
        and isinstance(lhs := s.args[0], (sympy.Integer, sympy.Symbol))
        and isinstance(rhs := s.args[1], sympy.Symbol)
        # support exactly one unbacked for now
        and ((rhs in pending) ^ (lhs in pending))
        # support constant coefficient or backed symbolic coefficient
        and (
            isinstance(coeff := lhs if lhs not in pending else rhs, sympy.Integer)
            or shape_env
            and coeff in shape_env.var_to_val
        )
    ):

        def _symint_wrap(s: sympy.Symbol) -> SymInt:
            return shape_env.create_symintnode(  # type: ignore[union-attr]
                s,
                hint=int(shape_env.var_to_val[s]),  # type: ignore[union-attr]
                source=shape_env.var_to_sources.get(s, [None])[0],  # type: ignore[union-attr]
            )

        unbacked = lhs if lhs in pending else rhs
        divisor: IntLikeType = (
            int(coeff)
            if shape_env and isinstance(coeff, sympy.Integer)
            else _symint_wrap(coeff)
        )
        # TODO: DivideByKey needs to test divisibility at runtime!

        r[unbacked] = path + (DivideByKey(divisor),)
        if real is not None:
            assert isinstance(real, int)
            val = (
                real // int(coeff)
                if isinstance(coeff, sympy.Integer)
                else CleanDiv(real, coeff)
            )
            if shape_env:
                shape_env.set_unbacked_var_to_val(unbacked, val)
        pending.remove(unbacked)
    # The annoyance here arises from the fact that SymBool is
    # allocated by allocating a SymInt and then testing if it's equal
    # to one.  So you have a complicated binding site logic for this.
    elif (
        isinstance(a, torch.SymBool)
        and isinstance(s := expr(a), sympy.Eq)
        # This must match create_unbacked_symbool EXACTLY
        and isinstance(s.lhs, sympy.Symbol)
        and s.rhs == 1
        and s.lhs in pending
    ):
        r[s.lhs] = path + (ConvertIntKey(),)
        if real is not None:
            assert type(real) is bool
            if shape_env:
                shape_env.set_unbacked_var_to_val(s, int(real))

        pending.remove(s.lhs)

    return r


def compute_unbacked_bindings(
    shape_env: Optional[ShapeEnv],
    example_value: object,
    old_example_value: Optional[object] = None,
    peek: bool = False,
) -> Optional[dict[sympy.Symbol, pytree.KeyPath]]:
    """
    After having run fake tensor propagation and producing example_value
    result, traverse example_value looking for freshly bound unbacked
    symbols and record their paths for later.  It is an error if
    we have allocated an unbacked SymInt but it cannot be found in
    example_value.  (NB: this means if you have a multi-output
    function, you must call this on the tuple of tensor output, you
    cannot wait!)

    The peek parameter lets you check out what the bindings are without
    changing the affected list.  This is primarily useful for ensuring
    unbacked_var_to_val is promptly populated when propagate_real_tensors is on.
    """
    if shape_env is None:
        return None

    fs = shape_env.pending_fresh_unbacked_symbols

    pending = set(fs)
    if not pending:
        return None

    if not peek:
        log.info("compute_unbacked_bindings %s", fs)
        fs.clear()

    symbol_to_path = _free_unbacked_symbols_with_path(
        example_value, (), shape_env=shape_env, pending=pending, simplify=False
    )
    if not peek and pending:
        extra = (
            repr((example_value.stride(), example_value.storage_offset()))
            if isinstance(example_value, torch.Tensor)
            else ""
        )
        raise PendingUnbackedSymbolNotFound(
            f"Pending unbacked symbols {pending} not in returned outputs {example_value} {extra}.\n"
            "Did you accidentally call new_dynamic_size() or item() more times "
            "than you needed to in your fake implementation?\n"
            "For more help, see https://docs.google.com/document/d/1RWrH-3wLEpzR9kCS6gGBNen_-Fs-8PVbWWFE5AcgeWE/edit"
        )

    # Why do we have to do some rebinding here?  If the original FX node
    # wasn't a binding site because you had a memo hit, but post
    # translation you aren't a memo hit anymore, there's now a new binding
    # site... but we know (because it's the same FX node) that the value
    # is actually the same, they're just not obviously equal anymore.
    #
    # The logic here is written carefully, because unlike the
    # bind_unbacked case, we are not guaranteed to have a symbol for
    # old_sym.  If we have a symbol, do regular rename unbacked to; but if
    # we don't, we need to specially eliminate the fresh unbacked symbol
    # (NB: we are /trusting/ that the memoization is correct, and that we
    # don't need to generate a new runtime assert.  This is load bearing,
    # as repropagation can happen after we've frozen runtime asserts.)
    if old_example_value is not None:
        for keypath in symbol_to_path.values():
            old_sym = pytree.key_get(old_example_value, keypath)
            new_sym = pytree.key_get(example_value, keypath)
            if isinstance(new_sym, SymTypes) and isinstance(
                new_s := new_sym.node.expr, sympy.Symbol
            ):
                if (
                    isinstance(old_sym, SymTypes)
                    and (old_s := old_sym.node.expr) != new_s
                ):
                    # If old_s is not an unbacked_symbol,
                    # we assume that the original unbacked symbol is replaced
                    # by a backed symbol (old_s). This can happen
                    # when this node reuses the original symbol (due to memoi)
                    # and the original symbol gets replaced by the backed symbol.
                    # When this happens we just replace new_s by the old_s
                    # because we know the value is the same.

                    if isinstance(old_s, sympy.Symbol) and free_unbacked_symbols(old_s):
                        shape_env._rename_unbacked_to(new_s, old_s)
                    else:
                        shape_env._eliminate_unbacked(new_s, old_s)
                elif not isinstance(old_sym, SymTypes):
                    shape_env._eliminate_unbacked(new_s, sympy.sympify(old_sym))

    return symbol_to_path


# Note [guard_or_]
# The following two functions are common utilities used while defining unbacked semantics
# of various framework code. Those would be used in situations you prefer to guard and know
# the result of the expression over not guarding, but in case you hit a data dependent error
# you are ok with just returning true or false.
#
# When to use this?
# (1) If you can use a higher level combinator prefer using those instead, they are definitely safe (modulo short-circuiting).
#
# (2) It can be used if the program would behave equivalently if _guard_or returned true or false.
# Many inductor optimizations fall in this bracket for example.
#
# (3) Finally, it's even be OK if the program wouldn't behave equivalently, so long as the
# change is semantics preserving.  It can be semantics preserving if the program errors in more
# cases than it did previously (but otherwise behaves identically), or if it changes some quantity
# in a way that doesn't matter (e.g., strides often fall in this bucket.)
#
# (4) Specialize for the general case and add a runtime assertion that would fail during
#     runtime if the conditions for the general case are not satisfied. Examples for this are;
#      assuming expand/reshape inputs are not -1. or assuming the non-broadcasting path.
#
def _guard_or(a: BoolLikeType, default: bool) -> bool:
    """
    Try to guard a, if data dependent error encountered just return default.
    """
    if not isinstance(a, SymBool):
        assert isinstance(a, bool)
        return a

    # if backed_size_oblivious is True we treat backed as unbacked here.
    if torch.fx.experimental._config.backed_size_oblivious:
        result = _static_eval_sym_bool(a)
        return result if result is not None else default

    shape_env = getattr(a.node, "shape_env", None)

    # xla symnode path.
    if shape_env is None:
        return guard_bool(a)

    sym_node = a.node
    r = sym_node.shape_env.evaluate_sym_node(
        sym_node, size_oblivious=False, fallback_value=default
    )
    return bool(r)


def guard_or_false(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return false.
    """
    return _guard_or(a, False)


def guard_or_true(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return true.
    """
    return _guard_or(a, True)


def _static_eval_sym_bool(x: SymBool) -> Optional[bool]:
    assert isinstance(x, SymBool)
    expr = x.node.expr

    try:
        # Shape env access is inside the try on purpose. xla symnode does not
        # have it on its attributes.
        shape_env = x.node.shape_env
        simplified = shape_env._maybe_evaluate_static(expr)
        if simplified is not None:
            return bool(simplified)
        else:
            return None
    except Exception:
        log.debug("Could not simplify %s", expr)
        return None


def statically_known_false(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is False.
    If x cannot be evaluated from static, we return False

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to False at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
    if not isinstance(x, SymBool):
        assert isinstance(x, bool)
        return not x

    result = _static_eval_sym_bool(x)
    if result is None:
        return False

    return not result


def statically_known_true(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is true.

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to true at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
    if not isinstance(x, SymBool):
        assert isinstance(x, bool)
        return x
    result = _static_eval_sym_bool(x)
    if result is None:
        return False

    return result


def sym_and(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    and, but for symbolic expressions, without bool casting.
    """
    if len(others) == 0:
        return x
    for y in others:
        x = operator.and_(x, y)
    return x


def sym_eq(x: _T, y: _T) -> BoolLikeType:
    """
    Like ==, but when run on list/tuple, it will recursively test equality
    and use sym_and to join the results together, without guarding.
    """
    if isinstance(x, (tuple, list)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return False
        return functools.reduce(operator.and_, map(sym_eq, x, y), True)
    elif isinstance(x, (int, torch.SymInt)) and isinstance(y, (int, torch.SymInt)):
        return x == y
    else:
        raise AssertionError(f"unexpected sym_eq between {type(x)} {type(y)}")


def sym_or(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    or, but for symbolic expressions, without bool casting.
    """
    if len(others) == 0:
        return x
    for y in others:
        x = operator.or_(x, y)
    return x


def guard_scalar(
    a: Union[SymBool, SymInt, SymFloat, int, bool, float],
) -> Union[bool, int, float]:
    """
    Guard a scalar value, which can be a symbolic or concrete boolean, integer, or float.

    This function dispatches to the appropriate guard function based on the type of the input.

    Args:
        a: A symbolic or concrete scalar value (bool, int, or float)

    Returns:
        The concrete value after guarding

    Raises:
        AssertionError: If the input is not a recognized scalar type
    """
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f"unrecognized scalar {a}")


def _advise_is_size(a: SymInt) -> None:
    """
    Don't use this directly; use torch._check_is_size instead.

    This is a softer version of _constrain_range_for_size (with min=0,
    max=Inf).  Instead of forcibly constraining a variable (and erroring if we
    failed to constrain it), it will simply advise us that a size is
    constrained in some way.  We will always defer a runtime assert for this
    constraint if we cannot prove it at compile-time, but we we only
    *sometimes* learn useful extra information at compile-time with this
    information.  This is in contrast to constrain_range_for_size, where if
    you don't call that on a fresh unbacked symint, chances are we will choke.

    TODO: Make Dynamo handle this appropriately if this is seen in Dynamo-ed
    code.  Right now this is only really used in code with AOTAutograd trace
    through, so it is not a big problem that this isn't supported, but in
    principle all of this code should be Dynamo'able too.

    TODO: I didn't support min/max because I didn't have a use case where this
    actually helped.  In principle we can support it, it just makes the
    implementation below more complicated.
    """

    # This must always succeed, because the sole allowed caller _check_is_size
    # was responsible for expect_true'ing this
    # This assert triggers expensive sym compute, do not do it until its cheap.
    # assert a >= 0

    # NB: it's important not to constrain range for size for *hinted* SymInts,
    # because it is not only unsound, it will immediately trip our asserts
    # that hints have to be consistent with static analysis!  If you somehow
    # have an unbounded SymInt that later constrains to 1, this will be
    # inconsistent with the range
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and isinstance(a.node.expr, sympy.Symbol)
        and a.node.shape_env.is_unbacked_symint(a.node.expr)
    ):
        _constrain_range_for_size(a)


def _advise_is_bounded(a: SymInt, upper_bound: IntLikeType) -> None:
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and isinstance(a.node.expr, sympy.Symbol)
        and a.node.shape_env.is_unbacked_symint(a.node.expr)
        and isinstance(upper_bound, int)  # TODO: relax
    ):
        a.node.shape_env._constrain_is_bounded(a.node.expr, upper_bound)


def _constrain_range_for_size(
    a: SymInt, min: Optional[int] = None, max: Optional[int] = None
) -> None:
    """
    This function is NOT INTENDED to be used by itself.
    """

    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat/SymBool is nyi")

    assert isinstance(a, SymInt), "can only constrain range for SymInt"
    assert isinstance(a.node.expr, sympy.Symbol), f"constraining non-Symbols NYI: {a}"

    a.node.shape_env._constrain_range_for_size(a.node.expr, min, max)


# inclusive both ways
def constrain_range(
    a: SymInt, *, min: Optional[int], max: Optional[int] = None
) -> None:
    """
    Applies a constraint that the passed in SymInt must lie between min-max
    inclusive-inclusive, WITHOUT introducing a guard on the SymInt (meaning
    that it can be used on unbacked SymInts).  If min/max are None, we assume
    that the dimension is unbounded in that direction.  Repeated application
    of constrain_range intersects the ranges.  This is a fairly low level API
    that doesn't have a lot of safety guarantees (TODO: provide higher level
    APIs).

    Currently, we use this API in the following circumstance: when we allocate
    an unbacked SymInt, denoting an integer quantity which is data dependent,
    we ordinarily do not know anything about what values it may take.  This
    means that any sort of guard on it will immediately fail.  However, in
    many cases, we know something about the unbacked SymInt: for example, we
    know that nonzero(x).size(0) must be >= 0.  We use constrain_range to
    narrow the possible range, declaring that negative symbols are impossible.
    This permits to definitely answer True to queries like 'nnz >= 0', even if
    we don't know what the actual (hinted) value of 'nnz' is.  In fact, we
    actually use constrain_range to unsoundly discharge common guards: for an
    unbacked SymInt produced by nonzero, we will also assume that it is not
    equal to 0/1 (even though these are perfectly possible values at runtime),
    because we generally expect graphs that are valid for N=2 to also be valid
    for N=1.
    """
    if min is None:
        min = -int_oo
    if max is None:
        max = int_oo

    if max < min:
        raise ValueError(
            "Maximum value to constrain_as_size can't be less than the specified min value, "
            f"received min={min} and max={max}"
        )

    if isinstance(a, int):
        if not (min <= a <= max):
            raise ValueError(f"Invalid value {a} for range [{min}:{max}]")
        return

    a.node.shape_env._constrain_range(a.node.expr, min, max)


def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None:
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
            return
        else:
            shape_env = b.node.shape_env
    else:
        shape_env = a.node.shape_env

    shape_env._constrain_unify(a, b)


# Assume that a boolean is true for the purposes of subsequent symbolic
# reasoning.  This will keep track of corresponding runtime checks to verify
# that the result is upheld: either as a regular guard, or as a special set
# of asserts which are triggered when an unbacked SymInt is allocated.
#
# DO NOT use this function for these cases:
#
#  - This is inappropriate for "branching" conditions (where both
#    true and false result in valid programs).  We will always assume
#    the condition evaluates true, and so it will never be possible
#    to trace the false condition when you use it.  For true branching
#    on unbacked SymInts, you must use torch.cond; if you incorrectly
#    use expect_true in this case, you will make the false branch
#    unreachable (as we will simply assume that only the true branch
#    is ever exercised).
#
#  - This is inappropriate for situations where you know some other system
#    invariant guarantees that this property holds, since you don't
#    really need to insert a runtime check in that case.  Use something
#    like constrain_range in that case.
#
# This API has a hitch.  To avoid having to reimplement error reporting
# capabilities, this function CAN return False.  The invariant is that
# the surrounding code must raise an error when this function returns
# False.  This is quite low level, so we recommend using other functions
# like check() which enforce this in a more intuitive way.
#
# By the way, this name is a nod to the __builtin_expect macro,
# which is used similarly (but unlike __builtin_expect, you MUST fail
# in the unlikely branch.)  (I think expect is a good name; in recent
# versions of C++, this is replaced with [[likely]], which is weaker
# and not accurate for this function!)
def expect_true(a: BoolLikeType, skip: int = 0) -> bool:
    if isinstance(a, SymBool):
        # TODO: check perf implications of this
        frame = inspect.currentframe()
        for _ in range(skip + 1):  # always run this loop at least once
            if frame is None:
                break
            frame = frame.f_back
        return a.node.expect_true(
            frame.f_code.co_filename if frame else "", frame.f_lineno if frame else 0
        )
    assert type(a) is bool, a
    return a


def guard_bool(a: BoolLikeType) -> bool:
    if isinstance(a, SymBool):
        return a.node.guard_bool("", 0)  # NB: uses Python backtrace
    assert type(a) is bool, a
    return a


def guard_int(a: IntLikeType) -> int:
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    assert type(a) is int, a
    return a


def guard_float(a: FloatLikeType) -> float:
    if isinstance(a, SymFloat):
        return a.node.guard_float("", 0)  # NB: uses Python backtrace
    assert isinstance(a, float), a
    return a


# Given a GraphModule, return all the FakeTensors for all the placeholders
def fx_placeholder_vals(gm: torch.fx.GraphModule) -> list[object]:
    return [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]


def fx_placeholder_targets(gm: torch.fx.GraphModule) -> list[str]:
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]


# Given a GraphModule and arguments to run it with, evaluate that the guards
# for its associated ShapeEnv are satisfied by the passed arguments.  This
# WILL check for duck sizing.
def eval_guards(
    gm: torch.fx.GraphModule, *args: Tensor, ignore_static: bool = True
) -> bool:
    assert gm.shape_env is not None
    return gm.shape_env.evaluate_guards_for_args(  # type: ignore[operator, union-attr]
        fx_placeholder_vals(gm), args, ignore_static=ignore_static
    )


def bind_symbols(gm: torch.fx.GraphModule, *args: Tensor) -> dict[sympy.Symbol, int]:
    assert gm.shape_env is not None
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)  # type: ignore[operator, union-attr]


class DimDynamic(Enum):
    """
    Controls how to perform symbol allocation for a dimension.  It is always
    sound to default this to DYNAMIC, but the policies DUCK and STATIC can
    result in better trace-time and compile-time performance, as they reduce
    the number of allocated symbols and generally make your graph more static.

    NB: If we notice you've applied a constraint to the dimension, we will
    force it to DYNAMIC for simplicity.

    DimDynamic is controlled by a variety of higher level UX features.
    Currently:

    - In eager mode, the default policy is DUCK.
        - The default is changed to STATIC with assume_static_by_default.
        - An individual dim is marked DYNAMIC if you mark_dynamic_dim.
    - In export mode, the default policy is STATIC.
        - An individual dim is marked DYNAMIC if you specify it in
          dynamic_shapes passed to export.
    """

    # Treat the dimension symbolically
    DYNAMIC = 0
    # Treat the dimension symbolically, but if its hint matches another
    # dynamic dimension, unify the two symbols ("duck sizing")
    DUCK = 1
    # Treat the dimension statically based on its hint
    STATIC = 2
    # Treat the dimension as a size-like unbacked
    SIZE_LIKE_UNBACKED = 3
    # Infer the strides from stride. If size is static, strides will be static as well.
    INFER_STRIDE = 4
    # Like SIZE_LIKE_UNBACKED, but there's a hint
    OBLIVIOUS_SIZE = 5


# NB: These constraints affect both clients and backends: given some
# constraint C, the client must pass inputs that satisfy the constraint,
# while a backend must not introduce guards BEYOND this constraint.
# For clarity, we document the implications on both sides for both the client
# and the backend.
#
# NB: These constraints are on a *single* dimension.  In principle, we could
# also have multi-dimension constraints, but our guess is that this is not
# actually useful and so we are not supporting it right now.
#
# NB: Strict constraints are typically only suitable for export, as in eager
# a backend like inductor may validly introduce extra, discretionary guards
# to improve performance of code.  A StrictMinMaxConstraint would be brittle
# under future optimizations performed by inductor; we don't guarantee
# eager code with StrictMinMaxConstraint will keep working in the future!


@dataclass(frozen=True)
class Constraint:
    warn_only: bool


@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    """
    For clients: the size at this dimension must be within 'vr' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    """

    vr: ValueRanges

    def render(self, source: Source) -> str:
        """Format the constrain equation"""
        # TODO: better printing for -oo and oo
        return f"{self.vr.lower} <= {source.name()} <= {self.vr.upper}"


@dataclass(frozen=True)
class RelaxedUnspecConstraint(Constraint):
    """
    For clients: no explicit constraint; constraint is whatever is implicitly
    inferred by guards from tracing.

    For backends: there must exist at least TWO possible values for the
    size at this dimension which satisfy the guards for this dimension.

    In other words, this constraint helps us distinguish between "we don't
    care if this dimension specializes or not" versus "this dimension must be
    unspecialized."  However, this constraint doesn't say very much about what
    specialization is permitted; for example, if we guard on a size being
    even, this would still be acceptable under an unspec constraint.  This
    makes RelaxedUnspecConstraint useful for eager mode, where your backend compiler
    may add constraints to otherwise dynamic dimensions; we can't assert that
    there are NO guards as this is brittle because compilers should be able to
    add extra constraints.  If you want to assert that there are no guards,
    use StrictMinMaxConstraint with an unbounded ValueRanges.
    """

    def render(self, source: Source) -> str:
        return f"RelaxedUnspecConstraint({source.name()})"


# NB: None here indicates the client constraint is whatever is implicitly
# inferred by guards from tracing, and that a backend can add whatever guards
# it wants (including fully specializing the value).
DimConstraint = Union[StrictMinMaxConstraint, RelaxedUnspecConstraint, None]


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """
    Represent and decide various kinds of equality constraints between input sources.

    A "source pair" is a pair of input sources for dynamic dimensions that
    are specified equal. We represent `source_pairs` in a union-find forest
    so that we can efficiently check whether two such sources are transitively equal.

    A "derived equality" relates an input source to an expression over a root.
    The root can be another input source, corresponding to some dynamic dimension,
    or a phantom symbol that does not directly represent any dynamic dimension. We
    represent `derived_equalities` involving input sources in a transitively-closed map
    so that we can efficiently check whether an input source is transitively equal to
    a given expression over another input source.
    (NOTE: In contrast, it is easy to decide whether an input source is transitively equal
    to a given expression over a phantom symbol; such expressions are already in canonical
    form and so the problem reduces to symbolic expression equality.)
    """

    source_pairs: list[tuple[Source, Source]]
    derived_equalities: list[
        tuple[Source, Union[Source, sympy.Symbol], Callable[[sympy.Expr], sympy.Expr]]
    ]
    phantom_symbols: list[sympy.Symbol]
    relaxed_sources: set[Source]

    _parents: dict[Source, Source] = field(init=False)
    _defs: dict[Source, sympy.Expr] = field(init=False)

    def __post_init__(self) -> None:
        """
        Pre-processing to answer queries `is_equal` and `is_derived` below.

        Example: Suppose we are given:
          source_pairs [a = b, b = c]
          derived_equalities [d = c + 1, e = d - 1]
        We first construct a union find with source_pairs:
          _parents = {a: a, b: a, c: a}
        Then we compute canonical symbolic expressions, recursively applying derived_equalities
        until we bottom out:
          _defs = {d: c + 1, e: (c + 1) - 1 aka c}
        """

        # self._parents is a map from input sources to input sources where, conceptually,
        # these are directed edges in a union-find forest
        _parents: dict[Source, Source] = {}
        object.__setattr__(self, "_parents", _parents)
        # self._defs is a map from input sources to "canonical" symbolic expressions,
        # i.e., unary expressions with symbols that corresponds to regular Dims (i.e.,
        # not derived Dims)
        _defs: dict[Source, sympy.Expr] = {}
        object.__setattr__(self, "_defs", _defs)

        for source1, source2 in self.source_pairs:
            # preprocess into a union-find forest
            self._union(self._find(source1), self._find(source2))
        for source, root, fn in self.derived_equalities:
            # preprocess into a transitively-closed map
            # NOTE(avik): we reuse the union-find forest for canonicalizing input sources
            if isinstance(root, (sympy.Symbol, sympy.Integer)):
                self._defs[self._find(source)] = fn(root)
            else:
                self._defs[self._find(source)] = fn(self._rewrite(root))

    def _find(self, source: Source) -> Source:
        # chase edges to find the root of this equivalence class
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1: Source, root2: Source) -> None:
        # merge two equivalence classes by adding an edge from one root to the other
        if root1 != root2:
            self._parents[root1] = root2

    def _rewrite(self, src: Source) -> sympy.Expr:
        # always represent the given source by the root of its equivalence class
        src = self._find(src)
        if src in self._defs:
            # simply look up the definition if it exists
            # NOTE(avik): This works because definitions are always transitively-closed;
            # otherwise we would have to do recursive rewriting.
            return self._defs[src]
        else:
            # otherwise, create a symbol representing the source
            return sympy.Symbol(src.name())

    def is_equal(self, source1: Source, source2: Source) -> bool:
        return (
            # check whether source1 and source2 have the same root
            # or are relaxed
            (src1 := self._find(source1)) in self.relaxed_sources
            or (src2 := self._find(source2)) in self.relaxed_sources
            or src1 == src2
            # check whether source1 is derived equal to source2
            or self.is_derived(source1, source2, lambda x: x)
        )

    def is_derived(
        self, src: Source, symbol_src: Source, fn: Callable[[sympy.Expr], sympy.Expr]
    ) -> bool:
        # check whether both src and symbol_src have the same definition
        return self._rewrite(src) == fn(self._rewrite(symbol_src))


def _assert_symbol_context(symbolic_context: object) -> TypeGuard[SymbolicContext]:
    assert isinstance(symbolic_context, SymbolicContext), (
        "Invalid symbolic_context object"
    )
    assert type(symbolic_context) is not SymbolicContext, (
        "Illegal usage of symbolic_context ABC"
    )
    return True


def _is_supported_equivalence(expr: sympy.Expr) -> bool:
    # Currently supported Dim ops are linear expressions with integer coefficients.
    # So check that expr only contains +, *, ints, and a single occurrence of a symbol.
    # (See also documentation of dynamic_shapes._DerivedDim.)
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        if len(expr.args) > 2:
            return False
        lhs, rhs = expr.args
        return (_is_supported_equivalence(lhs) and isinstance(rhs, sympy.Integer)) or (
            isinstance(lhs, sympy.Integer) and _is_supported_equivalence(rhs)
        )
    return isinstance(expr, sympy.Symbol)


def _has_uninterpretable_sympy_function(expr: sympy.Basic) -> bool:
    """
    Add functions that our sympy interpreter can't reify into FX nodes
    """
    return expr.has(
        torch.utils._sympy.functions.ToFloat,
        torch.utils._sympy.functions.TruncToInt,
        torch.utils._sympy.functions.CeilToInt,
    )


@dataclass(frozen=True)
class SymbolicContext:
    """
    Data structure specifying how we should create symbols in
    ``create_symbolic_sizes_strides_storage_offset``; e.g., should
    they be static or dynamic.

    This is an abstract base class because we are probably going to add
    another version of this that says "use exactly these SymInts, don't
    allocate fresh symbols."
    """


@dataclass(frozen=True)
class SymIntSymbolicContext(SymbolicContext):
    """
    Data structure specifying any constraints on a SymInt input
    """

    constraint: DimConstraint


_P1 = ParamSpec("_P1")
_T1 = TypeVar("_T1")


@dataclass(frozen=True)
class StatelessSymbolicContext(SymbolicContext, Generic[_P1, _T1]):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by ``DimDynamic`` and ``DimConstraint``.
    This will cause fresh symbols to be allocated
    """

    dynamic_sizes: DimList[DimDynamic]
    dynamic_strides: DimList[DimDynamic] = None  # type: ignore[assignment]
    constraint_sizes: DimList[DimConstraint] = None  # type: ignore[assignment]
    constraint_strides: DimList[DimConstraint] = None  # type: ignore[assignment]
    specialize_on: Optional[list[list[Callable[_P1, _T1]]]] = None
    # If the tensor is a view, this should be populated for the base. It contains
    # information on how to allocate symbols when recursively fakeifying the base
    # during view fake-ification.
    view_base_context: Optional[SymbolicContext] = None
    # TODO: add storage offset and stride symbolic_context

    def __post_init__(self) -> None:
        if self.specialize_on is None:
            object.__setattr__(
                self,
                "specialize_on",
                [[]] * len(self.dynamic_sizes),
            )
        if self.dynamic_strides is None:
            object.__setattr__(
                self,
                "dynamic_strides",
                [DimDynamic.INFER_STRIDE] * len(self.dynamic_sizes),
            )
        if self.constraint_sizes is None:
            object.__setattr__(
                self, "constraint_sizes", [None] * len(self.dynamic_sizes)
            )
        if self.constraint_strides is None:
            object.__setattr__(
                self, "constraint_strides", [None] * len(self.dynamic_sizes)
            )
        assert all(
            stride in (DimDynamic.INFER_STRIDE, DimDynamic.DYNAMIC, DimDynamic.DUCK)
            for stride in self.dynamic_strides
        )


# note [Tensor Fakification and Symbol Caching]
#
# As of the time of this note, dynamo creates a fresh fake tensor mode for backends.
# The reason we do this is because there are certain classes of operations, namely,
# metadata mutations, that change tensor size, stride, etc. This means that the fake tensor
# state at the end of a dynamo trace is different than the fake tensor state at the beginning
# of a trace. Backends like aot_autograd need a fresh fake tensor to correctly track metadata mutation,
# view relationships, etc.
#
# As we create a new fake mode, we also lose the memoization that comes with it. Rather than
# transfer the memoization cache, we instead transfer the shape env. However, with this
# comes nuance - as dynamo is selective in how it makes symbolic shapes. Due to strategies in
# automatic dynamic and constraints, the policy for which dims are dynamic is nuanced and varies across
# recompilations.
#
# In order to preserve the symbolic decisions made during dynamo tensor fakification, we pass
# a StatefulSymbolicContext at creation time. This object is tracked, per tensor, on the TracingContext.
# The lifecycle of this object should match the lifecycle of the original dynamo tracked tensor, and it is
# safe to reuse this object as many times as necessary to create a fake tensor. Fake tensors
# created with new fake modes should produce the same exact symbols as the original, providing the same shape_env
# is used.
# TODO(voz): Shape env validation
@dataclass(frozen=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by a cache of Source:Symbol. A cache hit
    will reuse a stored symbol, and a cache miss will write to this cache.

    This behaves like StatelessSymbolicContext, except the cache supersedes the
    other values - dynamic_sizes and constraint_sizes will not be read if we cache
    hit.

    It is the cache owner's responsibility to maintain the lifecycle of the cache
    with respect to different shape_envs, clearing, etc.
    """

    tensor_source: Source = None  # type: ignore[assignment]
    # Why is this keyed on int first?
    # That integer is actually the id of the shape_env. This cache short-circuits symbol
    # creation, and we must store it per shape env. Now, while tracing invariants are a single
    # shape env per tracing context, and every new frame gets a new shape_env. So where would we have
    # multiple shape envs? The answer lies in recording. When we are replaying, replay_shape_env_events
    # is invoked, and creates a new shape_env. Replaying events against this new shape_env will
    # cause it to fail with unknown symbols, as the symbols cached here will skip creation, and never
    # get recorded in var_to_val, etc.
    # TODO(voz): consider a weakref to the shape_env here
    shape_env_to_source_to_symbol_cache: dict[int, dict[str, sympy.Expr]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        # The None default is annoying, but required because of dataclass limitations
        assert self.tensor_source is not None
        if not self.shape_env_to_source_to_symbol_cache:
            object.__setattr__(self, "shape_env_to_source_to_symbol_cache", {})


@dataclass(frozen=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    """
    The correct symbolic context for a given inner tensor of a traceable tensor subclass
    may differ from that of the outer symbolic context. This structure allows for this
    flexibility, with inner symbolic contexts mapped via attr -> symbolic context.
    """

    inner_contexts: dict[str, SymbolicContext] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.inner_contexts is None:
            # pyrefly: ignore [bad-assignment]
            self.inner_contexts = {}


@dataclass
class TrackedFake:
    """
    Tracks the sources of all fake tensors we wrap in Dynamo.
    Used by shape guard computation.
    """

    fake: Union[FakeTensor, SymInt]
    source: Source
    symbolic_context: Optional[SymbolicContext]

    def __hash__(self) -> int:
        return hash((self.fake, self.source.name()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedFake):
            return self.fake is other.fake and self.source.name() == other.source.name()
        return False


def is_symbolic(
    val: Union[int, SymInt, float, SymFloat, bool, SymBool],
) -> TypeGuard[Union[SymInt, SymFloat, SymBool]]:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()


IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)


def _expandsums(args: list[sympy.Expr]) -> tuple[sympy.Expr, bool]:
    """
    Expand products of sums into sums of products.

    This function takes a list of sympy expressions and separates them into
    additive expressions (those with is_Add=True) and other expressions.
    It then computes the distributive product, expanding (a+b)*(c+d) into a*c + a*d + b*c + b*d.

    Args:
        args: A list of sympy expressions to expand

    Returns:
        A tuple containing:
        - The expanded expression as a sympy.Expr
        - A boolean indicating whether expansion occurred (True if multiple additive
          expressions were present or if there was at least one additive and one other expression)
    """
    adds, other = [], []
    for arg in args:
        if arg.is_Add:
            adds.append(arg)
        else:
            other.append(arg)

    result = [sympy.Mul(*other)]
    for add in adds:
        result = [a * b for a, b in itertools.product(result, add.args)]

    result = sympy.Add(*result)
    return result, len(adds) > 1 or (len(adds) > 0 and len(other) > 0)


def _fast_expand(expr: _SympyT) -> _SympyT:
    """
    A faster implementation of sympy's expand function for common cases.

    This function expands expressions like (a+b)^n or (a+b)*(c+d) into sums of products,
    but avoids the expensive checks and features of sympy's full expand implementation.
    It only recreates objects when necessary to avoid expensive operations.

    Args:
        expr: A sympy expression to expand

    Returns:
        The expanded expression
    """

    # The expand algorithm in sympy is slow due to all the features is supports
    # For eg: e^(-x)*(x-1)/(x+1) is expanded to (x-1)/(e^x + e^x*x) if x is
    # positive and (e^(-x)*x-e^(-x))/(x+1) if x is negative. We do not implement
    # such features here to avoid expensive checks. We also make sure that we
    # only re-create the objects if any of the args changed to avoid expensive
    # checks when re-creating objects.
    new_args = [_fast_expand(arg) for arg in expr.args]  # type: ignore[arg-type]
    # pyrefly: ignore [missing-attribute]
    if any(arg is not new_arg for arg, new_arg in zip(expr.args, new_args)):
        # pyrefly: ignore [missing-attribute]
        return _fast_expand(expr.func(*new_args))

    # pyrefly: ignore [missing-attribute]
    if expr.is_Pow:
        base: sympy.Expr
        exp: sympy.Expr
        base, exp = expr.args  # type: ignore[assignment]
        if exp.is_Integer and base.is_Add:
            if exp > 1:
                return sympy.expand_multinomial(expr, deep=False)
            elif exp < 0:
                return S.One / sympy.expand_multinomial(S.One / expr, deep=False)
    # pyrefly: ignore [missing-attribute]
    elif expr.is_Mul:
        num: list[sympy.Expr] = []
        den: list[sympy.Expr] = []
        # pyrefly: ignore [missing-attribute]
        for arg in expr.args:
            if arg.is_Pow and arg.args[1] == -1:
                den.append(S.One / arg)  # type: ignore[operator, arg-type]
            else:
                num.append(arg)  # type: ignore[arg-type]

        num, num_changed = _expandsums(num)
        den, den_changed = _expandsums(den)
        if num_changed or den_changed:
            return num / den

    return expr


@lru_cache(256)
def safe_expand(r: _SympyT) -> _SympyT:
    """
    Expand the given symbolic expression by recursively rewriting product of
    sums into sum of products (with the product being either a multiplication or
    exponentiation).

    NOTE: using this on an intermediate expression may prevent simplification
    down the line, e.g., if we eagerly expand `(a + b)^2` into `a^2 + 2ab + b^2`,
    we won't be able to simplify `(a^2 + 2ab + b^2) / (a + b)` as easily.
    """
    if hasattr(r, "expand"):
        try:
            return _fast_expand(r)
        except RecursionError:
            log.warning("RecursionError in _fast_expand(%s)", r)
            return r
    else:
        return r


class _SymbolInfo(NamedTuple):
    k: sympy.Symbol
    vr: Optional[ValueRanges]
    val: Optional[sympy.Integer]
    is_size_like: bool


@lru_cache(None)
def _maybe_evaluate_static_worker(
    expr: _SympyT,
    # NB: this is a tuple to ensure it can be LRU cached
    symbol_info: tuple[_SymbolInfo, ...],
    unbacked_only: bool,
    size_oblivious: bool,
) -> Optional[_SympyT]:
    """
    This variant of ShapeEnv._maybe_evaluate_static has no dependence on
    ShapeEnv and thus can be cached indefinitely.  It does the "heavy" lifting
    for static evaluation, including nontrivial reliance on Sympy simplification
    that occurs when we reallocate the symbols
    """

    # Simplify making use of value range lower bound
    new_shape_env = {}
    new_range_env = {}
    for idx, sinfo in enumerate(symbol_info):
        k, vr, val, is_size_like = sinfo
        if isinstance(val, SingletonInt):
            # Skip var_ranges logic for SingletonInt which is only used
            # for jagged layout NestedTensors today
            continue
        assert vr is not None
        if size_oblivious and is_size_like:
            lower = max(2, vr.lower)
            # Clamping size-oblivious to some quantity below sys.maxsize
            # helps us determine that f(u0) != sys.maxsize, which is a
            # test that is looking for sys.maxsize as a sentinel, but you
            # don't really want to worry about it for unbacked SymInts.
            # This is similar to the flavor where size oblivious omits
            # 0/1, it changes semantics but in a benign way.
            upper = min(2**48, vr.upper)
            # Excluding the very upper bound can be helpful
            if upper > lower:
                upper = upper - 1
            # This is a bit dodgy: what this means is that there was a
            # size-like unbacked symbol whose upper bound < 2.  This
            # causes... problems.
            if lower <= upper:
                vr = ValueRanges(lower, upper)
        else:
            lower = vr.lower
        # Don't do anything if we don't have a nontrivial lower bound
        # Also don't do anything if we asked only to simplify unbacked
        # SymInt
        if lower is -int_oo or (unbacked_only and val is not None) or not vr.is_int:
            new_range_env[k] = vr
            continue
        # The goal is to take our symbols which have various lower bounds
        # and reallocate them into new symbols which are exactly positive;
        # e.g., if we have s0 in [2, inf], we want to turn it into ess0 in
        # [1, inf], where s0 = ess0 + 1.  This gives the most information
        # to sympy for subsequent simplifications.
        #
        # Positive means >= 1
        # Positive - 1 means >= 0
        # Positive + lower - 1 means >= lower
        # The new symbol 's' is "too low", so when we substitute it in
        # we have to increase it by offset (and conversely, the new
        # variables have to have their value range bounds adjusted as
        # well)
        s = sympy.Symbol(f"evaluate_static_shape_{idx}", positive=True, integer=True)

        # Note:
        #   Offset might be a fraction(e.g. aten.split.Tensor), but shapes are always integers.
        #   Sympy might give unexpected results when comparing an integer with a non-integer
        #   Therefore, we cast offset to int here.
        #   For example:
        #       shape_0 = sympy.Symbol("shape_0", positive=True, integer=True)
        #       expr = sympy.Eq(shape_0 - 1/3, 4)
        #       expr.xreplace({}) # False
        offset = int(lower - 1)
        new_shape_env[k] = s + offset
        new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

    # TODO: remove this try catch (esp for unbacked_only)
    try:
        # pyrefly: ignore [missing-attribute]
        new_expr = expr.xreplace(new_shape_env)
    except RecursionError:
        log.warning("RecursionError in sympy.xreplace(%s, %s)", expr, new_shape_env)
        return None

    # We need to canonicalize, as after expand we may have something like `a + b = a` and
    # sympy will not simplify the a. The two appeareances of the a will then make value ranges
    # analysis give lose bounds
    new_expr = canonicalize_bool_expr(safe_expand(new_expr))
    if new_expr.is_number:
        return new_expr

    # Check if the range can solve it statically
    out = bound_sympy(new_expr, new_range_env)
    if out.is_singleton():
        return out.lower

    return new_expr if unbacked_only else None


def error() -> NoReturn:
    raise AssertionError("shouldn't be hit")


# TODO: Deduplicate this with torch/_prims_common/__init__.py
def eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> int:
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))


def _eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> bool:
    """
    Evaluates whether a tensor with the given sizes and strides is non-overlapping and dense.

    A tensor is non-overlapping if there's no memory location that belongs to more than one element.
    A tensor is dense if all elements are stored in memory without gaps.

    Args:
        sizes: Sequence of dimension sizes for the tensor
        strides: Sequence of strides for the tensor

    Returns:
        True if the tensor is non-overlapping and dense, False otherwise
    """
    dim = len(sizes)

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    # or it is a 0/1 element tensor
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    lengths_and_strides = sorted(zip(sizes, strides), key=operator.itemgetter(1))

    # Unlike the C++ code, we don't move the 0/1 size dimensions to the
    # end.  So we have to keep going for this code.
    expected_stride = 1
    for length, stride in lengths_and_strides:
        if length == 1:
            continue

        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


def _sympy_cast_symbool_to_symint_guardless(x: SympyBoolean) -> sympy.Expr:
    return sympy.Piecewise((1, x), (0, True))


def cast_symbool_to_symint_guardless(
    symbool: Union[bool, torch.SymBool],
) -> Union[int, torch.SymInt]:
    """
    Converts a SymBool or bool to a SymInt or int without introducing guards.

    This function maps True to 1 and False to 0, preserving the symbolic nature
    of the input when it's a SymBool. Unlike regular casting which might introduce
    guards, this function performs the conversion without adding any guards.

    Args:
        symbool: A boolean value, either a concrete bool or symbolic SymBool

    Returns:
        The corresponding integer value (1 for True, 0 for False) as either
        a concrete int or symbolic SymInt
    """
    if isinstance(symbool, bool):
        return 1 if symbool else 0
    int_sym = _sympy_cast_symbool_to_symint_guardless(symbool.node.expr)
    return symbool.node.shape_env.create_symintnode(
        int_sym, hint=int(symbool.node.require_hint()) if has_hint(symbool) else None
    )


SYMPY_INTERP = {
    "IsNonOverlappingAndDenseIndicator": eval_is_non_overlapping_and_dense,
    "cast_symbool_to_symint_guardless": cast_symbool_to_symint_guardless,
    "math": math,
    "torch": torch,
}


def _lru_cache(
    fn: Callable[..., _T], maxsize: Optional[int] = None
) -> functools._lru_cache_wrapper[_T]:
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.

    Also note that this depends on _update_version_counter being called on the
    shape environment whenever the constraints are updated, otherwise the cache
    will not be cleared.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_version = 0

    if config.validate_shape_env_version_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()

            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                assert prior_key == self._get_key(), (
                    "ShapeEnv cache key changed without version being updated!"
                )

            return fn_cache(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:  # type: ignore[misc]
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter

            return fn_cache(self, *args, **kwargs)

    wrapper.cache_clear = fn_cache.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@dataclass(frozen=True)
class RuntimeAssert:
    """
    This is pretty similar to ShapeGuard but it also comes with a message,
    and is exclusively used for things that MUST be true (unlike guards,
    which can evaluate False, in which case you just choose not to use
    a particular specialization)
    """

    expr: SympyBoolean
    msg: str = field(repr=False)
    stack: CapturedTraceback = field(repr=False)


# Used for printing SymExprs in compile_fx
class SymExprPrinter(PythonPrinter):
    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))


class _ShapeGuardPrinter(abc.ABC):
    """
    Abstract base class for printers that convert symbolic expressions to string representations.

    This class provides common functionality for printing symbolic expressions with
    special handling for symbols that represent tensor shapes, strides, etc.
    Subclasses implement specific formatting for different output languages.

    Args:
        symbol_to_source: Mapping from sympy symbols to their source objects
        source_ref: Function to convert a source to its string representation
        var_to_sources: Mapping from sympy symbols to their source objects (for error reporting)
    """

    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Source]],
        source_ref: Callable[[Source], str],
        var_to_sources: Mapping[sympy.Symbol, list[Source]],
    ) -> None:
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources
        super().__init__()

    def _print_Float(self, expr: sympy.Float) -> str:
        """Convert a sympy Float to a Python float string representation."""
        return str(float(expr))

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        """
        Convert a sympy Symbol to its source representation.

        This method looks up the symbol in symbol_to_source mapping and returns
        the string representation of its first source. If the symbol is not in
        symbol_to_source (which

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 39 class(es): GuardOnDataDependentSymNode, PendingUnbackedSymbolNotFound, SymIntEqByExpr, ConstraintViolationError, Specialization, ConvertIntKey, CallMethodKey, InnerTensorKey, DivideByKey, DimDynamic, Constraint, StrictMinMaxConstraint, RelaxedUnspecConstraint, EqualityConstraint, if, SymbolicContext, SymIntSymbolicContext, StatelessSymbolicContext, StatefulSymbolicContext, SubclassSymbolicContext

### Functions
This file defines 284 function(s): __init__, log_lru_cache_stats, _extract, __init__, __repr__, __eq__, __hash__, _nested_int_aware_sort, size_hint, lru_cache, inner, cumulative_cache_info, new_cache_clear, uninteresting_files, has_symbolic_sizes_strides, create_contiguous, hint_int, has_hint, is_concrete_int, is_concrete_float, is_concrete_bool, has_static_value, guard_size_oblivious, check_consistent, resolve_unbacked_bindings, rebind_unbacked, is_accessor_node, canonicalize_bool_expr, _sympy_from_args, _canonicalize_bool_expr_impl


## Key Components

The file contains 33965 words across 8109 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 335453 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
