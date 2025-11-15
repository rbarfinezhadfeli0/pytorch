# Documentation: `torch/fx/experimental/proxy_tensor.py`

## File Metadata

- **Path**: `torch/fx/experimental/proxy_tensor.py`
- **Size**: 106,080 bytes (103.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import logging
import operator
import threading
import typing
import typing_extensions
import weakref
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Concatenate,
    Optional,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, Self, TypeVarTuple, Unpack
from weakref import WeakKeyDictionary

import torch
import torch._ops
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import SymBool, SymInt, Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._logging import trace_structured
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_impls import fast_detach
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    is_fake,
    unset_fake_temporarily,
)
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.graph_module import _assign_attr
from torch.fx.node import (
    _side_effectful_need_to_be_preserved_pre_dispatch,
    Argument,
    Target,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
    autograd_would_have_decomposed,
    TorchDispatchMode,
)
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary

from ._backward_state import BackwardState
from .sym_node import SymNode


if TYPE_CHECKING:
    import types
    from collections.abc import MutableMapping

    import sympy

    from torch._ops import OpOverload
    from torch.fx._symbolic_trace import PHBase
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "selective_decompose",
    "py_sym_types",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "maybe_enable_thunkify",
    "maybe_disable_thunkify",
]

_ProxyTracer = Union["PythonKeyTracer", "_GraphAppendingTracerEx"]

_AnyScriptObject = (torch.ScriptObject, FakeScriptObject)
_AnyScriptObjectType = Union[torch.ScriptObject, FakeScriptObject]

aten = torch.ops.aten
prim = torch.ops.prim

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

CURRENT_DECOMPOSITION_TABLE: Mapping[OpOverload, Callable] = {}

CONSTANT_NUMEL_LIMIT = 1

T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")
_Ts = TypeVarTuple("_Ts")

null_ctx_type = type(nullcontext)
# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    # pyrefly: ignore [bad-argument-type]
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
    serialized_type_name="torch.Size",
)
# Ideally unflattening should not lose info, but we unflatten
# torch.Size to tuple (see above). This is necessary because the
# torch.Size constructor only accepts ints whereas our infra often
# transforms them to non-ints, e.g. symint proxies. Anyway, losing
# such info can cause pytree mapping or spec matching to fail, so
# work around this problem using the following dict as needed.
_pytree_subclasses_that_lose_info = {torch.Size: tuple}


def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(
    decomposition_table: Optional[Mapping[OpOverload, Callable]],
) -> Generator[Mapping[OpOverload, Callable], None, None]:
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table or {}
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table


# ensure we cannot collide with other properties
proxy_slot = object()


class _NoDefault:
    pass


no_default = _NoDefault()

from torch.types import py_sym_types, PySymType


class _HasMeta(Protocol):
    meta: dict[str, PySymType]


def is_sym_node(node: _HasMeta) -> bool:
    assert hasattr(node, "meta"), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta["val"], py_sym_types)


@overload  # type: ignore[no-overload-impl]
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...


@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, proxy: Proxy
) -> None: ...


@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None: ...


class _DisableUpdateTensorTracker(threading.local):
    value: bool = False


_disable_update_tensor_tracker_tls = _DisableUpdateTensorTracker()


_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = {}


def _is_proxy_tensor_update_tensor_tracker_disabled() -> bool:
    """
    Returns current state of disabling update tensor tracker.
    """
    return _disable_update_tensor_tracker_tls.value


@contextmanager
def _proxy_tensor_disable_update_tensor_tracker() -> Generator[None, None, None]:
    """
    NOTE "Do not clobber inplace ops"
    By default tensor_tracker is updated every time.
    This leads to chaining every operation by the FakeTensor.
    For example for mutable ops if we have several consecutive mutable operations:

    def f(x, y, z):
        x.copy_(y)
        x.copy_(z)
        return x

    Default graph result:
    def f_graph(x, y, z)
        x_1 = x.copy_(y)
        x_2 = x_1.copy_(z)
        return x_2

    This chaining simplifies the fx passes and helps to prevent the reordering.
    But in some cases, we want those nodes to be disconnected.
    E.g. in case of splitting joint graph into forward and backward.
    If first inplace op happened in forward, second in backward,
    we want them after split to be properly placed.

    Enabling this context manager for copy_ will result in:
    def f_graph_2(x, y, z):
        x_1 = x.copy_(y)
        x_2 = x.copy_(z)
        return x

    Results of copy_ x1 and x2 will have empty users in the graph.
    The reason why this behavior is not enabled for all inplace ops is that
    some fx passes (e.g. fx quantization) rely on chaining inplace ops like add_
    in their fusions passes.
    We could revisit enabling this logic for all inplace ops in future.
    """
    orig_value = _disable_update_tensor_tracker_tls.value
    _disable_update_tensor_tracker_tls.value = True
    try:
        yield
    finally:
        _disable_update_tensor_tracker_tls.value = orig_value


def set_proxy_slot(  # type: ignore[no-redef]
    obj: Union[PySymType, _AnyScriptObjectType, Tensor],
    tracer: _ProxyTracer,
    proxy: object,
) -> None:
    log.debug("set_proxy_slot %s (%s) %s", obj, id(obj), proxy)
    if isinstance(obj, Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        assert isinstance(proxy, _ProxyTensor)
        # see NOTE [Do not clobber inplace ops]
        if not _is_proxy_tensor_update_tensor_tracker_disabled():
            tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (_AnyScriptObject)):
        # We DO want to clobber proxies, with a similar rationale as for tensors.
        assert isinstance(proxy, Proxy)
        tracer.script_object_tracker[obj] = proxy
    else:
        # NB: Never clobber pre-existing proxy.  Although the proxies
        # are in principle equivalent, when we do graph partitioning
        # we need there not to be spurious dependencies on tangent inputs.
        # This works because primals get their SymInts set first, and
        # THEN later we allocate tangent inputs.  Make sure if a SymInt
        # is derivable from a primal that we use that.
        assert isinstance(obj, py_sym_types), type(obj)
        if obj not in tracer.symnode_tracker:
            proxy = typing.cast(_PySymProxyType, proxy)
            tracer.symnode_tracker[obj] = proxy

            # WAR: python test/dynamo/test_subclasses.py
            # TestNestedTensor.test_basic_autograd
            #
            # AOTAutograd doesn't pass the "outer sizes" as an actual argument
            # to make_fx, but it is made use of internally in AOTAutograd's
            # call to tensor unflatten.  Because the outer sizes isn't passed
            # as an argument, it is therefore untracked.  However, it turns
            # out you luck out, because *Dynamo* will manually add the outer
            # sizes as an argument so you can fix up the proxy'ness.
            #
            # This is probably fixed in
            # https://github.com/pytorch/pytorch/pull/125941/
            import sympy

            if isinstance(obj.node.expr, sympy.Symbol):
                tracer.sympy_expr_tracker[obj.node.expr] = _SympyExprTrackerValue(
                    proxy, obj
                )


def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool:
    assert isinstance(obj, (Tensor, SymNode)), type(obj)
    # pyrefly: ignore [no-matching-overload]
    return bool(get_proxy_slot(obj, tracer, False, lambda _: True))


_PySymProxyType = Thunk[Proxy]


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
) -> _ProxyTensor: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
) -> Union[_ProxyTensor, U]: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> Union[R, U]: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
) -> Proxy: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
) -> Union[Proxy, U]: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> Union[R, U]: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
) -> _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: T,
) -> Union[T, _PySymProxyType]: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> Union[R, U]: ...


# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(
    obj: Union[Tensor, _AnyScriptObjectType, PySymType],
    tracer: _ProxyTracer,
    default: object = no_default,
    transform: Callable = lambda x: x,
) -> object:
    tracker: Any
    if isinstance(obj, Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, _AnyScriptObject):
        tracker = tracer.script_object_tracker
    else:
        assert isinstance(obj, py_sym_types), type(obj)
        tracker = tracer.symnode_tracker

    # pyrefly: ignore [index-error]
    # pyrefly: ignore [no-matching-overload, bad-argument-type]
    value = tracker.get(obj)

    if value is None and isinstance(obj, py_sym_types):
        if obj.node.is_symbolic():
            # Last ditch - we found a SymInt (SymBool, etc) we don't know
            # about.
            if (tmp := tracer.sympy_expr_tracker.get(obj.node.expr)) is not None:
                value = tmp.proxy

            else:
                # Attempt to build it from first principles.
                _build_proxy_for_sym_expr(tracer, obj.node.expr, obj)
                # pyrefly: ignore [no-matching-overload]
                value = tracker.get(obj)

    if value is None:
        # We don't know this value - return the default.
        if isinstance(default, _NoDefault):
            raise RuntimeError(
                f"{obj} ({type(obj)}, {id(obj)})is not tracked with proxy for {tracer}"
            )
        return default

    res = transform(value)
    return res


@functools.cache
def _sympy_handlers() -> dict[type[sympy.Expr], Callable[..., Any]]:
    """
    Returns a dict converting sympy functions to python operators
    (i.e. `sympy.Mul` -> `operator.mul`)
    """
    import torch.utils._sympy.interp

    handlers = {}
    for k, v in torch.utils._sympy.interp.handlers().items():
        op = getattr(operator, v, None)
        if op is not None:
            handlers[k] = op
    return handlers


def _build_proxy_for_sym_expr(
    tracer: _ProxyTracer, expr: sympy.Expr, out: PySymType | None = None
) -> IntLikeType | FloatLikeType | BoolLikeType | None:
    """
    Decompose `expr` and look for the pieces as inputs. If `out` is provided
    then that will be the resulting SymNode (and `out.expr` must be the same as
    `expr`).

    This function is used when the ProxyTorchDispatchMode sees a SymNode
    that it hasn't seen before to try to associate it with traced inputs.

    How can this happen?

    First thing to remember is that although sympy.Exprs are interned (so
    `sympy.Expr("s3*s4")` will always have the same `id` and will always compare
    equal) SymNode does not (so doing `SymNode("s3")*SymNode("s4")` twice in a
    row will give two unique SymNodes).

    - On way for this to happen is if we turn off tracing to compute an
      intermediate value and then USE that value with tracing turned on - for
      example if we turn off tracing to do some FakeTensor propagation to
      compute a size (dtensor does this) but then turn tracing back on and use
      that computed size.

    - Another way is if we compute a size in one graph and stash it somewhere
      hidden (such as in some meta-data) and later use it in a different graph
      (dtensor does this too). Since the size was computed in the first graph
      and it's not an official input to the second graph it's not tracked
      properly. This is often going to show up as it usually works in fullgraph
      but a graph break causes a failure.

    To handle this we decompose the sympy.Expr and look for the pieces as
    inputs. But there are problems with this approach:

    - We lose operation provanance: We end up figuring out where to get the
      inputs - but those may not actually be correct. If we have "s1" coming in
      from both tensor1 and tensor2 and we pick the wrong one we could end up
      keeping a tensor alive longer than intended.

    - There's no guarantee that those values are inputs to the graph: If we have
      "s1*s2" computed in a graph #1 and used in graph #2 there's no guarantee
      that the input that holds "s1" is actually an input on graph #2.

    - The decomposition isn't guaranteed to be the same: Sympy can "simplify"
      expressions so it's possible that our inputs are "s1*s2" and "s3" but we
      decompose it into "s1" and "s2*s3" - which wouldn't be found.

    Other ways we could handle this:

    - Don't: Just require that all inputs are tracked properly. This is the
      "correct" solution but harder because you need to track down each
      potential problem one by one and fix them. And when it fails it's a lot of
      work to figure out both why it's failing and the right way to fix it. This
      is complicated by the fact that a stashed value could be incorrect but
      work fine until we happen to get an graph break in the wrong place - so it
      may be a while before the bug is found. (Maybe we need a "dynamo abuse
      mode" where we run tests with as many graph breaks inserted as possible?)

    - Track SymNode ops separately from proxy tracing: Right now SymNode
      operations are tracked as part of the proxy tracing - so when we disable
      proxy tracing we also disable SymNode tracing. But we don't have to do
      that - we could instead always have SymNodes track where they came from
      and just use that when needed. This solves the problem of tracing being
      temporarily turned off but doesn't help if an input isn't present after a
      graph break.

    - Better decomposition: Right now the decomposition is pretty simple. We do
      have a sat-solver available to us so we could theoretically do a better
      job figuring out a "correct" decomposition. But that still relies on
      having the inputs available at all - which isn't a guarantee.
    """

    if (value := tracer.sympy_expr_tracker.get(expr)) is not None:
        assert not out
        return value.value

    if isinstance(expr, (int, float, bool)):
        return expr
    if expr.is_Integer:
        return int(expr)
    if expr.is_Float:
        return float(expr)

    args = []
    for arg in expr.args:
        if (arg_value := _build_proxy_for_sym_expr(tracer, arg)) is None:
            return None
        args.append(arg_value)
    args = tuple(args)

    func: OpOverload | None = _sympy_handlers().get(expr.func)  # type: ignore[assignment]
    if not func:
        # Handler not found
        return None

    if out is None:
        out = func(*args)
    else:
        _sym_register(tracer, func, args, out)
    return out


def snapshot_fake(val: Tensor, include_real: bool = False) -> Optional[Tensor]:
    # val.detach() will also eventually call fast_detach(),
    # but this saves us a full trip into __torch_dispatch__
    # (snapshot_fake is called a lot)
    if isinstance(val, FakeTensor):
        return fast_detach(val.fake_mode, val, include_real)
    else:
        return val.detach()


_ExtractValType = Optional[
    Union[
        PySymType,
        _AnyScriptObjectType,
        BackwardState,
        list["_ExtractValType"],
        tuple["_ExtractValType", ...],
        dict[str, "_ExtractValType"],
        Tensor,
        int,
        float,
        bool,
    ]
]


def extract_val(val: _ExtractValType, include_real: bool = False) -> _ExtractValType:
    if is_fake(val):
        return snapshot_fake(val, include_real=include_real)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, _AnyScriptObject):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, dict):
        return {k: extract_val(v) for k, v in val.items()}
    elif isinstance(val, Tensor):
        if not val.is_sparse:
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            from torch._guards import detect_fake_mode

            fake_tensor_mode = detect_fake_mode(val)
            if not fake_tensor_mode:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(
                    val.shape, val.stride(), device=val.device, dtype=val.dtype
                )
        else:
            return None
    elif isinstance(val, (int, float, bool)):
        return val
    elif val is None:
        return None

    typing_extensions.assert_never(val)


@contextmanager
def _enable_thunkify(
    tracer: _ProxyTracer, *, enable: bool = True
) -> Generator[None, None, None]:
    """
    Enable thunkification inside the context manager.  Thunkification prevents
    SymNode computation from directly being traced into an FX graph; instead,
    the compute is only added to the graph if it is actually used.  This helps
    us track SymNode compute when it is computed (since we need /something/
    to put in the tracker) even if it is unlikely to be used.
    """
    old = tracer.enable_thunkify
    tracer.enable_thunkify = enable
    try:
        yield
    finally:
        tracer.enable_thunkify = old


@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    """Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer, enable=False):
            yield
    else:
        yield


@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    """Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer):
            yield
    else:
        yield


# Note [invariants for node meta 'val']
# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy:
    proxy.node.meta["val"] = extract_val(
        val, include_real=(proxy.node.op == "placeholder")
    )

    with _enable_thunkify(proxy.tracer):  # type: ignore[arg-type]
        # Best effort tensor_meta setting; prefer using val!
        if is_fake(val):
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        elif isinstance(val, Tensor) and not val.is_sparse:
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
    return proxy


def thunkify(
    tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> Thunk[R]:
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    if tracer.enable_thunkify:
        return Thunk(functools.partial(f, *args, **kwargs))
    else:
        r = f(*args, **kwargs)
        return Thunk(lambda: r)


def track_tensor(
    tensor: Tensor, proxy: Proxy, *, constant: Optional[Tensor], tracer: _ProxyTracer
) -> None:
    def try_set_proxy_slot(
        outer_s: IntLikeType,
        proxy_callable: Callable[Concatenate[PySymType, _P], Proxy],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            with _enable_thunkify(tracer):
                set_proxy_slot(
                    outer_s,
                    tracer,
                    thunkify(tracer, proxy_callable, outer_s, *args, **kwargs),
                )

    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(
            s,
            lambda x, i: set_meta(
                tracer.create_proxy(
                    "call_function", torch.ops.aten.sym_size.int, (proxy, i), {}
                ),
                x,
            ),
            i,
        )

    if not is_sparse_any(tensor):
        for i, s in enumerate(tensor.stride()):
            try_set_proxy_slot(
                s,
                lambda x, i: set_meta(
                    tracer.create_proxy(
                        "call_function", torch.ops.aten.sym_stride.int, (proxy, i), {}
                    ),
                    x,
                ),
                i,
            )

    try_set_proxy_slot(
        tensor.numel(),
        lambda x: set_meta(
            tracer.create_proxy(
                "call_function", torch.ops.aten.sym_numel.default, (proxy,), {}
            ),
            x,
        ),
    )
    if not is_sparse_any(tensor):
        try_set_proxy_slot(
            tensor.storage_offset(),
            lambda x: set_meta(
                tracer.create_proxy(
                    "call_function",
                    torch.ops.aten.sym_storage_offset.default,
                    (proxy,),
                    {},
                ),
                x,
            ),
        )
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))


_NestedProxys = Union[
    Proxy, Sequence["_NestedProxys"], Mapping[object, "_NestedProxys"]
]
_NestedTensors = Union[
    Tensor, Sequence["_NestedTensors"], Mapping[object, "_NestedTensors"]
]


def track_tensor_tree(
    inner_res: T,
    proxy_res: _NestedProxys,
    *,
    constant: Optional[_NestedTensors],
    tracer: _ProxyTracer,
) -> T:
    # NB: We call set_unbacked_bindings only on the *topmost* call to
    # track_tensor_tree, not recursive calls.  This is because there must
    # be only ONE unbacked_binding proxy call, and it should be the one
    # where all of the unbacked SymInts actually first come into existence.
    # If you call this again on the inner proxies for the tuple projections,
    # you will have multiple unbacked_bindings for the same symbol, but
    # they're not going to show up anywhere.
    #
    # I was briefly deceived into setting unbacked bindings recursively when
    # working on https://github.com/pytorch/pytorch/pull/133585 because I
    # observed that some extra unbacked bindings were needed to handle some
    # higher order operator code.  But actually it looks like this was
    # just an unrelated bug that needed to be fixed separately.
    _set_unbacked_bindings(inner_res, proxy_res)

    def wrap_with_proxy(
        e: object, proxy: _NestedProxys, constant: Optional[_NestedTensors]
    ) -> None:
        if isinstance(e, Tensor):
            assert isinstance(proxy, Proxy)
            assert constant is None or isinstance(constant, Tensor)
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            assert isinstance(proxy, Proxy)
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, thunkify(tracer, lambda: proxy))
        elif isinstance(e, _AnyScriptObject):
            assert isinstance(proxy, Proxy)
            set_proxy_slot(e, tracer, proxy)
            set_meta(proxy, e)
        elif isinstance(e, (tuple, list)):
            # example use case: allreduce_ returns ([tensor], work)
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            def get_constant(
                c: Optional[_NestedTensors], idx: int
            ) -> Optional[_NestedTensors]:
                if c is None:
                    return None
                else:
                    assert isinstance(c, (list, tuple))
                    return c[idx]

            for idx, ee in enumerate(e):
                # Use an indexer here - if proxy is a List then it will unwrap
                # it. If it's a Proxy then it will proxy the getelem.
                wrap_with_proxy(ee, proxy[idx], get_constant(constant, idx))  # type: ignore[index]

        elif isinstance(e, dict):
            # example use case: triton_kernel_wrapper takes arguments as kwargs

            # In theory we could support const-prop when proxy-tensor-tracing
            # operators that returns dicts of tensors, but we have no use case
            # for it today (since the only op we currently trace that can
            # return a dict is triton_kernel_wrapper_functional/mutation,
            # which does not participate in const-prop)
            assert constant is None

            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)  # type: ignore[index]

        elif isinstance(e, BackwardState):
            assert isinstance(proxy, Proxy)
            set_meta(proxy, e)
            e.proxy = proxy
        else:
            # intentionally pass on primitives
            pass

    wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[Tensor]


def fetch_sym_proxy(
    tracer: _ProxyTracer,
) -> Callable[[PySymType], Union[bool, int, float, Proxy]]:
    def inner(e: PySymType) -> Union[int, bool, float, Proxy]:
        n = e.node
        if n.constant is not None:
            return n.constant
        if e.node.expr.is_number:
            if isinstance(e, SymBool):
                return bool(e.node.expr)
            elif isinstance(e, SymInt):
                return int(e.node.expr)
            return float(e.node.expr)
        else:
            assert isinstance(e, py_sym_types)
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(e, tracer).force()

    return inner


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: Tensor
) -> Union[_ProxyTensor, Tensor]: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Union[Proxy, _AnyScriptObjectType]: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> Union[_PySymProxyType, PySymType]: ...


def fetch_object_proxy(
    tracer: _ProxyTracer, t: Union[Tensor, _AnyScriptObjectType, PySymType]
) -> object:
    return get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (Tensor, torch.nn.Parameter, FakeTensor)


def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    """
    Records operators whose tensor outputs or inputs are fp16/bf16 so downstream pointwise code can
    emulate eager's rounding behavior when emulate_precision_casts is enabled.
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if not isinstance(func, torch._ops.OpOverload):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    t = last_node.meta.get("val")
    low_pr_fp = (torch.bfloat16, torch.float16)

    output_low_precision = isinstance(t, torch.Tensor) and t.dtype in low_pr_fp

    if not output_low_precision:
        for input_node in last_node.all_input_nodes:
            val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if isinstance(val, torch.Tensor) and val.dtype in low_pr_fp:
                output_low_precision = True
                break

    if not output_low_precision:
        return

    last_node.meta["low_precision_pointwise_barrier"] = True


def _fetch_proxies_and_all_constant_flag(
    flat_args_kwargs: Union[list[object], tuple[object, ...]], tracer: _ProxyTracer
) -> tuple[list[object], tuple[object, ...], bool]:
    """
    Given flat arguments, fetch the proxies and whether they are all constants.
    This is later used in proxy_call or when someone is trying to stitch together
    graph node in tf or td modes.
    """
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer, x)
            if isinstance(x, (Tensor, _AnyScriptObject))
            else x
        )
        for x in flat_args_kwargs
    ]

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        not any(
            t.constant is None
            for t in f_flat_args_kwargs
            if isinstance(t, _ProxyTensor)
        )
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and not any(isinstance(x, py_sym_types) for x in flat_args_kwargs)
    )

    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e for e in f_flat_args_kwargs
    ]

    proxy_flat_args_kwargs = [
        (fetch_sym_proxy(tracer)(e) if isinstance(e, py_sym_types) else e)
        for e in proxy_flat_args_kwargs
    ]

    return f_flat_args_kwargs, tuple(proxy_flat_args_kwargs), all_constant


def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    unrecognized_types: list[type] = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x: Tensor) -> bool:
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) is torch._subclasses.FakeTensor
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not all(can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, Tensor)):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        _maybe_record_pointwise_barrier(func, proxy_mode)
        return r

    # For pre-autograd tracing, we do not want to run CompositeImplicit decomps.
    if (
        not pre_dispatch
        and func
        not in [
            torch.ops.aten.size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.storage_offset.default,
        ]
        and autograd_would_have_decomposed(func, flat_args_kwargs)
    ):
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    if func is torch.ops.aten.is_nonzero.default:
        with proxy_mode:
            torch._check(
                args[0].numel() == 1,  # type: ignore[attr-defined]
                lambda: "Boolean value of Tensor with more than one value is ambiguous",
            )
            return (args[0] != 0).item()  # type: ignore[attr-defined]

    tracer = proxy_mode.tracer
    f_flat_args_kwargs, proxy_flat_args_kwargs, all_constant = (
        _fetch_proxies_and_all_constant_flag(flat_args_kwargs, tracer)
    )

    if torch.Tag.data_dependent_output in func.tags:
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            with unset_fake_temporarily():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            Tensor, lambda t: not is_fake(t), (args, kwargs)
        ):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call."
            )

    proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat_args_kwargs, spec)

    # When we trace through a torch.tensor invocation, you never actually
    # see a torch.ops.aten.tensor call. Instead, the way this function is
    # implemented internally is that we allocate a plain tensor (this is
    # *guaranteed* to be a plain tensor, we disable all modes when doing
    # so), and then call at::lift_fresh on it (to give modes a chance to do
    # their stuff).  Furthermore, the tensor argument to lift_fresh is guaranteed
    # to be freshly allocated, so we want lift_fresh to be a no-op (directly
    # returning the input argument).
    #
    # Here is the basic problem: when we trace this sequence of executions
    # into an FX graph, what happens to this call sequence?  Traditionally,
    # tensor constants get interned as buffers on the FX GraphModule.  But
    # this is dangerous.  Consider:
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default

    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function",
        func,
        proxy_args,
        proxy_kwargs,
        name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__),
    )

    with _enable_thunkify(proxy_mode.tracer):
        out = func(*args, **kwargs)

    # In some circumstances, we will be tracing in a situation where a tensor
    # is *statically* known to be a constant (currently, this only happens if
    # you run torch.tensor; deterministic factory functions like torch.arange
    # don't get this treatment).  When the tensor in question is small, it's
    # helpful to due constant propagation in case we call item() (in which
    # case we can return the constant value that is known, rather than give
    # an error.)  The logic here tests if constant propagation is possible
    # (because all of the inputs are constant).  If so, we disable fake tensor
    # mode (if it is on) and do true compute on the constant.
    #
    # It's worth highlighting that we're making a policy decision here.
    # There is a potential that the tensor is actually quite large, and we
    # don't actually want to run the compute.  The tensor being quite large
    # is one of the reasons why factory functions don't get this treatment
    # (since they can be quite large; if a parameter is initialized to a
    # constant value it will be!)  Similarly, there is also a potential
    # to run an operator that blows up the size of a small tensor; we don't
    # protect against this case, but we could force, e.g., only single
    # element constant computation by testing the numel of the result before
    # propagating const-ness.  Similarly, we don't require the constant to
    # live on CPU, but we could.
    any_constant = any(
        t.constant is not None
        for t in f_flat_args_kwargs
        if isinstance(t, _ProxyTensor)
    )

    constant = None

    def tensor_numel_in_limit(t: Tensor) -> bool:
        return t.numel() <= CONSTANT_NUMEL_LIMIT

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if (
        func is torch.ops.aten.lift_fresh_copy.default
        and out.numel() <= CONSTANT_NUMEL_LIMIT
    ):
        with unset_fake_temporarily():
            assert isinstance(args[0], (Proxy, Tensor)), type(args[0])
            constant = args[0].clone()
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags
        and all_constant
        and any_constant
        and pytree.tree_all_only(Tensor, tensor_numel_in_limit, out)
    ):
        # NB: do NOT include factories as constants
        with unset_fake_temporarily():
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None

    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    _maybe_record_pointwise_barrier(func, proxy_mode)
    return out


class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """

    def __init__(self) -> None:
        self.sym_node_dict: dict[PySymType, _PySymProxyType] = {}

    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None:
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: PySymType) -> _PySymProxyType:
        return self.sym_node_dict[key.node]

    def __contains__(self, key: PySymType) -> bool:
        return key.node in self.sym_node_dict

    def get(
        self, key: PySymType, default: Optional[_PySymProxyType] = None
    ) -> _PySymProxyType:
        # dict.get()'s annotation doesn't accept `None` when the value type
        # isn't Optional.
        return self.sym_node_dict.get(key.node, default)  # type: ignore[arg-type, return-value]

    def __iter__(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sym_node_dict)


@dataclass
class _SympyExprTrackerValue:
    proxy: _PySymProxyType
    value: PySymType


class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self) -> None:
        super().__init__(autowrap_modules=())  # type: ignore[arg-type]
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = _SymNodeDict()
        self.script_object_tracker = WeakIdKeyDictionary(
            dict=None, ref_type=_WeakHashRef
        )
        self.sympy_expr_tracker = {}

        # Stores the torch function that was called during tracing
        self.torch_fn_metadata = None
        # Stores the counts for every torch function called. This is to help
        # distinguish between different calls to the same torch function.
        self.torch_fn_counts = {}
        self.enable_thunkify = False

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object:
        return attr_val

    def create_arg(self, a: object) -> fx.node.Node:
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})

            qualname = self.get_fresh_qualname("_param_constant")
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        elif isinstance(a, py_sym_types):
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)  # type: ignore[return-value]

    @overload
    def unwrap_proxy(self, e: Tensor) -> Union[Proxy, Tensor]: ...

    @overload
    def unwrap_proxy(self, e: PySymType) -> Union[Proxy, PySymType]: ...

    @overload
    def unwrap_proxy(
        self, e: _AnyScriptObjectType
    ) -> Union[Proxy, _AnyScriptObjectType]: ...

    def unwrap_proxy(self, e: T) -> object:
        if isinstance(e, Tensor):
            return get_proxy_slot(e, self, e, lambda x: x.proxy)  # type: ignore[attr-defined]
        elif isinstance(e, py_sym_types):
            return get_proxy_slot(e, self, e, lambda e: e.force())
        elif isinstance(e, _AnyScriptObject):
            return get_proxy_slot(e, self, e)
        else:
            return e

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)  # type: ignore[arg-type]

        if node.op in ["placeholder", "output"] and "stack_trace" in node.meta:
            del node.meta["stack_trace"]

        if kind == "get_attr":
            assert isinstance(target, str)
            attr = getattr(self.root, target)
            if isinstance(attr, torch.Tensor):
                with disable_proxy_modes_tracing():
                    node.meta["val"] = extract_val(attr)

        def map_fn(v: Any) -> Optional[_ExtractValType]:
            if not isinstance(v, torch.fx.Node) or "val" not in v.meta:
                return None
            val = v.meta["val"]
            # other subclasses like FunctionalTensor error on `extract_val`
            # "Attempting to use FunctionalTensor on its own." just store FakeTensors for now
            if isinstance(val, torch.Tensor) and not isinstance(val, FakeTensor):
                return None
            return extract_val(v.meta["val"])

        if _should_save_eager_input_vals(target, (args, kwargs)):
            # NOTE "eager_input_vals"
            # We save the original (args, kwargs) FakeTensor values for nodes
            # that have exact stride requirements. This is useful downstream.
            # We use this information inside Inductor to ensure that inputs to
            # stride-sensitive operators have the correct strides.
            arg_inp, kwarg_inp = torch.fx.node.map_aggregate((args, kwargs), map_fn)  # type: ignore[misc, arg-type]
            node.meta["eager_input_vals"] = (arg_inp, kwarg_inp)

        return node


def _should_save_eager_input_vals(
    target: Any,
    args_kwargs: Optional[tuple[tuple[Argument, ...], dict[str, Argument]]] = None,
) -> bool:
    from torch._higher_order_ops.invoke_subgraph import InvokeSubgraphHOP

    if not callable(target):
        return False
    if isinstance(
        target,
        (
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
            InvokeSubgraphHOP,
        ),
    ):
        return True
    if args_kwargs is not None and (
        target is torch.ops.higher_order.auto_functionalized
        or target is torch.ops.higher_order.auto_functionalized_v2
    ):
        args = args_kwargs[0]
        assert isinstance(
            args[0], (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        )
        return _should_save_eager_input_vals(args[0], None)
    if target is torch.ops.higher_order.with_effects:
        # TODO: inductor lowering for with_effects needs to be updated to propagate
        # the arg_kwarg_vals
        return False
    if isinstance(target, torch._ops.HigherOrderOperator):
        if pytree.tree_any(_should_save_eager_input_vals, args_kwargs):
            raise RuntimeError(
                f"NYI: The HOP {target} has an input that is an OpOverload that "
                f
```



## High-Level Overview


This Python file contains 22 class(es) and 144 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_NoDefault`, `_HasMeta`, `_DisableUpdateTensorTracker`, `_ProxyTensor`, `_SymNodeDict`, `_SympyExprTrackerValue`, `PythonKeyTracer`, `TorchFunctionMetadataMode`, `PreDispatchTorchFunctionMode`, `ProxyTorchDispatchMode`, `_GraphAppendingTracerEx`, `DecompositionInterpreter`, `_SelectiveDecomposeInterpreter`, `_ModuleNotInstalledAsSubmoduleError`, `_AttrProxy`, `_ModuleStackTracer`, `AttrProxy`, `_MakefxTracer`

**Functions defined**: `fake_signature`, `decompose`, `is_sym_node`, `set_proxy_slot`, `set_proxy_slot`, `set_proxy_slot`, `_is_proxy_tensor_update_tensor_tracker_disabled`, `_proxy_tensor_disable_update_tensor_tracker`, `f`, `f_graph`, `f_graph_2`, `set_proxy_slot`, `has_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`, `get_proxy_slot`

**Key imports**: annotations, functools, inspect, logging, operator, threading, typing, typing_extensions, weakref, defaultdict, OrderedDict


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `inspect`
- `logging`
- `operator`
- `threading`
- `typing`
- `typing_extensions`
- `weakref`
- `collections`: defaultdict, OrderedDict
- `collections.abc`: Callable, Generator, Mapping, Sequence
- `contextlib`: _GeneratorContextManager, contextmanager, ExitStack, nullcontext
- `dataclasses`: dataclass
- `torch`
- `torch._ops`
- `torch.fx as fx`
- `torch.fx.traceback as fx_traceback`
- `torch.utils._pytree as pytree`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._library.fake_class_registry`: FakeScriptObject
- `torch._library.opaque_object`: is_opaque_type
- `torch._logging`: trace_structured
- `torch._subclasses.fake_impls`: fast_detach
- `torch._subclasses.meta_utils`: is_sparse_any
- `torch.fx`: GraphModule, Proxy, Tracer
- `torch.fx.graph_module`: _assign_attr


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `proxy_tensor.py_docs.md`
- **Keyword Index**: `proxy_tensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
