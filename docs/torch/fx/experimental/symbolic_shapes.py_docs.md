# Documentation: `torch/fx/experimental/symbolic_shapes.py`

## File Metadata

- **Path**: `torch/fx/experimental/symbolic_shapes.py`
- **Size**: 335,455 bytes (327.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
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

```



## High-Level Overview

"""``torch.fx.experimental.symbolic_shapes`` provides interfaces for interacting withour symbolic shapes reasoning system that is used heavily in torch.compile.  Althoughthis is not generally considered public API, when writing framework code in PyTorchas well as extensions to PyTorch (e.g., in custom operator implementations), you mayneed to make use of these APIs to setup dynamic shapes support appropriately.

This Python file contains 53 class(es) and 284 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GuardOnDataDependentSymNode`, `PendingUnbackedSymbolNotFound`, `SymIntEqByExpr`, `ConstraintViolationError`, `Specialization`, `ConvertIntKey`, `CallMethodKey`, `InnerTensorKey`, `DivideByKey`, `DimDynamic`, `Constraint`, `StrictMinMaxConstraint`, `RelaxedUnspecConstraint`, `EqualityConstraint`, `SymbolicContext`, `SymIntSymbolicContext`, `StatelessSymbolicContext`, `StatefulSymbolicContext`, `SubclassSymbolicContext`, `TrackedFake`

**Functions defined**: `__init__`, `log_lru_cache_stats`, `_extract`, `__init__`, `__repr__`, `__eq__`, `__hash__`, `_nested_int_aware_sort`, `size_hint`, `lru_cache`, `inner`, `cumulative_cache_info`, `new_cache_clear`, `uninteresting_files`, `has_symbolic_sizes_strides`, `create_contiguous`, `hint_int`, `has_hint`, `is_concrete_int`, `is_concrete_float`

**Key imports**: annotations, sympy, S, BoolLike, FloatLike, IntLike, abc, atexit, collections, dis, functools, hashlib


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `sympy`
- `torch._prims_common`: BoolLike, FloatLike, IntLike
- `abc`
- `atexit`
- `collections`
- `dis`
- `functools`
- `hashlib`
- `inspect`
- `itertools`
- `logging`
- `math`
- `operator`
- `os`
- `re`
- `sys`
- `threading`
- `traceback`
- `collections.abc`: Callable, Generator, Iterator, Mapping, Sequence
- `contextlib`: _GeneratorContextManager, contextmanager
- `dataclasses`: asdict, dataclass, field
- `enum`: Enum
- `typing_extensions`: deprecated, ParamSpec
- `torch`
- `torch.fx`
- `torch.fx.traceback as fx_traceback`
- `torch.utils._pytree as pytree`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/fx/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- [`validator.py_docs.md`](./validator.py_docs.md)
- [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- [`unify_refinements.py_docs.md`](./unify_refinements.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`const_fold.py_docs.md`](./const_fold.py_docs.md)
- [`merge_matmul.py_docs.md`](./merge_matmul.py_docs.md)
- [`rewriter.py_docs.md`](./rewriter.py_docs.md)
- [`partitioner_utils.py_docs.md`](./partitioner_utils.py_docs.md)


## Cross-References

- **File Documentation**: `symbolic_shapes.py_docs.md`
- **Keyword Index**: `symbolic_shapes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
