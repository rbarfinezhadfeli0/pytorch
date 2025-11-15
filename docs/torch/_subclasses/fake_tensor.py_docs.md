# Documentation: `torch/_subclasses/fake_tensor.py`

## File Metadata

- **Path**: `torch/_subclasses/fake_tensor.py`
- **Size**: 136,166 bytes (132.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import logging
import math
import os
import threading
import traceback
import types
import typing
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Literal,
    Optional,
    TYPE_CHECKING,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import Self
from weakref import ReferenceType

import torch
import torch._library.utils as library_utils
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C._functorch import is_functorch_wrapped_tensor, is_legacy_batchedtensor
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.fake_profile import MissingOpProfile
from torch._logging import dtrace_structured
from torch._prims_common import suggest_memory_format
from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)
from torch._utils import render_call
from torch.fx.immutable_collections import immutable_dict
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.types import IntLikeType, py_sym_types
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import KeyPath, keystr, PyTree, tree_map, tree_map_, TreeSpec
from torch.utils._stats import count
from torch.utils._traceback import CapturedTraceback

from ._fake_tensor_utils import _CacheKeyState, _PySymInputStub, _SymIntOutputStub


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
    from types import TracebackType

    from torch._guards import Source
    from torch._ops import OpOverload
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# TODO: Hack to unblock https://github.com/pytorch/pytorch/pull/108186
# Proper fix tracked by https://github.com/pytorch/pytorch/issues/120105
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        not_implemented_log = logging.getLogger(__name__ + ".not_implemented")
    else:
        raise e


DimList = list

pytree = torch.utils._pytree
T = TypeVar("T")

aten = torch._ops.ops.aten

CONSTANT_NUMEL_LIMIT = 1

RECURSION_COUNT = 0


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:
    def __init__(self) -> None:
        global RECURSION_COUNT
        RECURSION_COUNT += 1

    def __del__(self) -> None:
        global RECURSION_COUNT
        RECURSION_COUNT -= 1


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str


@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload


@dataclass
class DataDependentOutputException(RuntimeError):
    func: OpOverload


@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload


@dataclass
class UnsupportedMutationAliasingException(RuntimeError):
    reason: str


@dataclass
class MetadataMismatchError(RuntimeError):
    reason: str


class FakeTensorTLS(threading.local):
    # Default to None, otherwise it'll be used to override _all_
    # `FakeTensorMode.allow_non_fake_inputs` in this thread.
    allow_non_fake_inputs_override: Optional[bool]
    non_strict_export_fake_tensor_tracker: weakref.WeakSet

    def __init__(self) -> None:
        self.allow_non_fake_inputs_override = None
        self.non_strict_export_fake_tensor_tracker = weakref.WeakSet()


fake_tensor_tls = FakeTensorTLS()


def ordered_set(*items: T) -> dict[T, Literal[True]]:
    return dict.fromkeys(items, True)


@contextlib.contextmanager
def unset_fake_temporarily() -> Generator[Optional[TorchDispatchMode], None, None]:
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


@contextlib.contextmanager
def disable_fake_tensor_cache(fake_mode: FakeTensorMode) -> Generator[None, None, None]:
    old_value: bool = fake_mode.cache_enabled
    try:
        fake_mode.cache_enabled = False
        yield
    finally:
        fake_mode.cache_enabled = old_value


def get_plain_tensors(
    subclass: Tensor, *, out: list[Union[Tensor, int, SymInt]]
) -> list[Union[Tensor, int, SymInt]]:
    # This function is used in Runtime, do not add redundant asserts
    todo = [subclass]
    while todo:
        curr = todo.pop()
        if not is_traceable_wrapper_subclass(curr):
            out.append(curr)
            continue

        inner_keys, _ = curr.__tensor_flatten__()
        todo.extend(getattr(curr, key) for key in reversed(inner_keys))

    return out


def is_fake(x: object) -> TypeGuard[Tensor]:
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(x, FakeTensor):
        return True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)
        flattened_tensors = [getattr(x, attr) for attr in attrs]
        all_fake = all(is_fake(x) for x in flattened_tensors)
        any_fake = any(is_fake(x) for x in flattened_tensors)
        assert all_fake == any_fake, "got mixed fake and real tensors!"
        return all_fake
    elif isinstance(x, FunctionalTensor):
        return is_fake(x.elem)
    elif isinstance(x, Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)
        return is_fake(unwrapped)
    elif isinstance(x, Tensor) and is_functorch_wrapped_tensor(x):
        unwrapped = torch._C._functorch.get_unwrapped(x)
        return is_fake(unwrapped)
    return False


def maybe_get_fake_mode(t: object) -> Optional[FakeTensorMode]:
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FakeTensor):
        return t.fake_mode
    if is_traceable_wrapper_subclass(t):
        inner_tensor_names, _ = t.__tensor_flatten__()
        modes = [
            maybe_get_fake_mode(getattr(t, t_name)) for t_name in inner_tensor_names
        ]
        m = modes[0]
        assert all(m is x for x in modes)
        return m
    elif isinstance(t, FunctionalTensor):
        return maybe_get_fake_mode(t.elem)
    elif isinstance(t, Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)
        return maybe_get_fake_mode(unwrapped)
    elif isinstance(t, Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)
        return maybe_get_fake_mode(unwrapped)
    return None


@functools.cache
def get_schema_info(func: OpOverload) -> torch._C._SchemaInfo:
    return torch._C._SchemaInfo(func._schema)


# many of the decompositions registered to torch/_prims do not at the moment model
# aliasing or strides, so as an incremental step, just enable the decompositions in
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them
@functools.cache
def torch_decomp_decompositions(func: OpOverload) -> bool:
    from torch._decomp import decomposition_table

    decompositions = torch._decomp.decompositions
    # Note that the function in the decomposition table might be
    # different from the one in the module because of the difference
    # in out handling in aten API and torch public API
    return decomposition_table[func].__module__.startswith(
        "torch._decomp"
    ) and decomposition_table[func].__name__ in dir(decompositions)


def tree_flatten_only(ty: type[T], tree: PyTree) -> list[T]:
    flat_vals = pytree.tree_leaves(tree)
    return [elem for elem in flat_vals if isinstance(elem, ty)]


def _is_plain_tensor(t: object) -> bool:
    return (
        type(t) is Tensor
        and t.layout == torch.strided
        and not (
            t.is_sparse
            or t.is_nested
            or is_functorch_wrapped_tensor(t)
            or is_legacy_batchedtensor(t)
            or torch._is_functional_tensor(t)
        )
    )


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter:
    @property
    def tensor_memo(
        self,
    ) -> weakref.WeakValueDictionary:
        # not valid until py3.10
        # weakref.WeakValueDictionary["torch._subclasses.meta_utils.MetaTensorId", Optional["FakeTensor"]]
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: dict[StorageWeakRef, list[ReferenceType]]
    export: bool

    def __init__(self, *, copy_data: bool = False, export: bool = False) -> None:
        self.meta_converter = MetaConverter(copy_data=copy_data)
        self.export = export

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor: FakeTensor) -> None:
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor: Tensor) -> None:
        assert not isinstance(tensor, FakeTensor)

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def _get_memo(self, t: Tensor) -> Optional[FakeTensor]:
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t: Tensor, v: FakeTensor) -> None:
        tid = self.meta_converter.describer.get_tensor_id(t)
        self.meta_converter.tensor_memo[tid] = v

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def from_real_tensor(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        make_constant: bool = False,
        shape_env: Optional[ShapeEnv] = None,
        *,
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
        trace: bool = True,
    ) -> FakeTensor:
        # see note [Tensor Fakification and Symbol Caching]
        if not symbolic_context and not source and shape_env:
            if tracing_context := torch._guards.TracingContext.try_get():
                if t in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[t]
                    from torch.fx.experimental.symbolic_shapes import (
                        StatefulSymbolicContext,
                    )

                    assert isinstance(symbolic_context, StatefulSymbolicContext)
                    source = symbolic_context.tensor_source

        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        if type(t) is torch.nn.Parameter:
            assert not make_constant

        constant = t if make_constant else None

        # This callback is used by both subclass and inner tensors. Require the
        # caller to explicitly specify the device in case outer and inner tensors
        # have different devices.
        def mk_fake_tensor(
            make_meta_t: Callable[[], object], device: Union[torch.device, str]
        ) -> FakeTensor:
            # NB: don't use in_kernel_invocation_manager. to
            # ensure FakeTensor can internally do constant computation
            # as necessary.  Invocation manager is "more correct" as
            # it works for more operators in make_meta_t, but
            # invariant is that make_meta_t only calls factories
            # for which it is not strictly necessary to use the
            # invocation manager (I think!)
            with no_dispatch():
                return FakeTensor(
                    fake_mode,
                    # pyrefly: ignore [bad-argument-type]
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                    # TODO: callback might be used in recursive contexts, in
                    # which case using t is wrong!  BUG!
                    constant=constant,
                )

        out = self.meta_converter(
            t,
            shape_env=shape_env,
            callback=mk_fake_tensor,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )
        if out is NotImplemented:
            raise UnsupportedFakeTensorException("meta converter nyi")

        from torch._dynamo.source import RandomValueSource

        value = None
        if (
            not self.export
            and _is_plain_tensor(t)  # mostly, we want to know if item() works
            and t.dim() == 0
            and t.device.type == "cpu"
            # All integer types are fair game, because signed overflow is UB
            # (and even int64 can overflow, since integers in Python are
            # arbitrary precision). But only float64 is OK for float, because
            # switching between float32 and float64 changes semantics in an
            # observable way without hitting UB.
            and t.dtype
            in [torch.int64, torch.int32, torch.int16, torch.int8, torch.float64]
            and source is not None
            # Impede setting up item() on things coming from random.  These
            # are not "real" item() calls, instead UnspecializedPythonVariable
            # is unsafely pretending an int is a tensor, which can sometimes
            # implicitly cause an item call.  The problem is this is pretty
            # unsound: there's no reason substituting an int with a Tensor is
            # going to give the same results.  Today, you mostly get around
            # this by typically not having capture_scalar_outputs on and graph
            # breaking when someone tries to use the unspec variable in an
            # int-y context.  But allowing it through here would break that.
            # So don't.
            #
            # Once random values are setup to be represented as
            # SymNodeVariable, this condition can be removed.  To check if
            # you've done it right, this is a good test:
            #
            #   PYTORCH_TEST_WITH_DYNAMO=1 python test/test_reductions.py -k
            #   TestReductionsCPU.test_dim_reduction_fns_fn_name_amax_cpu_bfloat16
            and not isinstance(source, RandomValueSource)
            # In Dynamo, shape_env is never none (even with static shapes).
            # However, FakeTensorMode can be used by hand and in some cases
            # ShapeEnv is not allocated.
            and shape_env is not None
        ):
            from torch._dynamo.source import CallMethodItemSource, FloatTensorSource
            from torch.fx.experimental.symbolic_shapes import DimDynamic

            with no_dispatch():
                value = t.item()
            if not math.isnan(value) and not math.isinf(value):
                # Peephole strip out unnecessary torch.as_tensor(x).item()
                if isinstance(source, FloatTensorSource):
                    item_source = source.base
                else:
                    item_source = CallMethodItemSource(source)
                symbol = shape_env.create_unspecified_symbol(
                    value,
                    source=item_source,
                    dynamic_dim=DimDynamic.DYNAMIC,
                    symbolic_context=symbolic_context,
                )
                # NB: reusing item_memo here ensures that we invalidate on
                # mutation
                if t.dtype == torch.int64:
                    out.item_memo = shape_env.create_symintnode(
                        symbol,
                        hint=value,
                        source=item_source,
                    )
                elif t.dtype == torch.float64:
                    out.item_memo = shape_env.create_symfloatnode(
                        symbol,
                        hint=value,
                        source=item_source,
                    )
        if make_constant:
            self.add_constant_storage_mapping(out)
        # NB: meta_converter set the memo
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        device: torch.device,
        pytype: Optional[type[torch.Tensor]] = None,
        dispatch_keys: Optional[torch.DispatchKeySet] = None,
    ) -> FakeTensor:
        assert t.device.type == "meta", (
            f"tensor's device must be `meta`, got {t.device.type} instead"
        )
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(
            fake_mode, t, device, pytype=pytype, dispatch_keys=dispatch_keys
        )
        self.set_tensor_memo(t, out)
        return out


@functools.cache
def init_gpu_context(device: torch.device) -> None:
    # Backward will error with cuda Fake Tensors if no cuda tensors have been initialized first
    if torch.cuda.is_available() or torch.xpu.is_available():
        (
            torch.empty(1, device=device)
            if torch.version.hip is None
            else torch.zeros(1, device=device)
        )


@contextlib.contextmanager
def in_kernel_invocation_manager(
    fake_mode: FakeTensorMode,
) -> Generator[None, None, None]:
    # See: note [Fake Tensor Dispatch Keys]
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f"{meta_in_tls}, {prev_in_kernel}"

    with torch._C._DisableTorchDispatch():
        fake_mode.in_kernel_invocation = True
        # Unfortunately _set_meta_in_tls_dispatch_include(False) can leave
        # `Dense` turned on (because it's implied by `Meta`)
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield
            finally:
                fake_mode.in_kernel_invocation = prev_in_kernel
                # torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)


# Return if the function allows Python numbers to bind to Tensors
def should_allow_numbers_as_tensors(func: OpOverload) -> bool:
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


class FakeTensorConfig:
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", "0") == "1"


# This memorizes unbacked SymInt or SymFloats representing quantities like the
# number of nonzero elements in this tensor or learning rate. There is one
# instance of the descriptor per particular quantity to memoize.
#
# Memoization is helpful if you do something like x[mask] and y[mask];
# mask.nonzero() gets repeatedly called and should give a consistent unbacked
# SymInt. It needs to be invalidated in the same way constant is.
#
# Making this a descriptor may seem overly fancy, but actually it's the most
# convenient way to ensure access to FakeTensor during access, which is
# required for testing version counter and epoch validity.
class SymNumberMemoDescriptor:
    _name: str

    # By default, SymInts in this memo are invalidated across versions/epochs.
    # nested_ints however are preserved across epochs and across versions.
    # Preserving across versions is okay for nested int since the association
    # of a nested int is agnostic to the underlying data and nested ints are not
    # shared across multiple distinct tensors.
    _is_nested_int: bool

    def __init__(self, *, is_nested_int: bool = False) -> None:
        self._is_nested_int = is_nested_int

    def __set_name__(self, owner: str, name: str) -> None:
        self._name = name

    def _memo(self, obj: FakeTensor) -> str:
        return f"_{self._name}"

    def _memo_vc(self, obj: FakeTensor) -> str:
        return f"_{self._name}_vc"

    # When we retrace, we need to invalidate all the memos so that we can
    # accurately identify the first time unbacked SymInts are allocated.
    # This is only relevant for inputs; for intermediates, they will get fresh
    # fake tensors so you won't have a memo anyway
    def _memo_epoch(self, obj: FakeTensor) -> str:
        return f"_{self._name}_epoch"

    def __get__(
        self, obj: FakeTensor, objtype: Optional[type[FakeTensor]] = None
    ) -> Optional[Union[torch.SymInt, torch.SymFloat]]:
        if (r := getattr(obj, self._memo(obj))) is None:
            return None

        # If backed, it's ok to preserve memo since we know it won't renumber.
        if isinstance(r, torch.SymFloat) and r.node.hint is not None:
            return r

        # Version counter based tracking isn't 100% sound but it's close
        # enough
        if (
            not self._is_nested_int and getattr(obj, self._memo_vc(obj)) != obj._version
        ) or (
            not self._is_nested_int
            and getattr(obj, self._memo_epoch(obj)) != obj.fake_mode.epoch
        ):
            setattr(obj, self._memo(obj), None)
            return None
        return r

    def __set__(
        self, obj: FakeTensor, value: Optional[Union[torch.SymInt, torch.SymFloat]]
    ) -> None:
        if value is None:
            setattr(obj, self._memo(obj), None)
            setattr(obj, self._memo_vc(obj), None)
            setattr(obj, self._memo_epoch(obj), None)
        elif not obj.is_inference() or self._is_nested_int:
            setattr(obj, self._memo(obj), value)
            if not self._is_nested_int:
                setattr(obj, self._memo_vc(obj), obj._version)
            setattr(obj, self._memo_epoch(obj), obj.fake_mode.epoch)


class FakeTensor(Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    fake_device: torch.device
    fake_mode: FakeTensorMode
    constant: Optional[Tensor]
    real_tensor: Optional[Tensor]

    # TODO: Generalize this as needed, e.g., into a trie of memos, if
    # you do something like x[0].item()  (x[0] is fresh each time, so
    # memo mechanism here won't work)
    nonzero_memo = SymNumberMemoDescriptor()
    item_memo = SymNumberMemoDescriptor()
    unique_memo = SymNumberMemoDescriptor()
    unique_consecutive_memo = SymNumberMemoDescriptor()

    # We expect nested_int_memo to be None when an offsets is a graph
    # intermediate, or an input that has never been associated with a
    # nested int.
    nested_int_memo = SymNumberMemoDescriptor(is_nested_int=True)

    # FakeTensor doesn't fully emulate the original tensor's Python type
    # and dispatch key set, therefore sometimes we want to track them
    # separately.
    pytype: Optional[type[Tensor]]
    dispatch_keys: Optional[torch.DispatchKeySet]

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    # pyrefly: ignore [bad-override]
    def device(self) -> torch.device:
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    @device.setter
    def device(self, _: torch.device) -> None:
        raise NotImplementedError

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch_dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below
    # NOTE: this probably will not do the right thing for backends
    # that have dispatch keys which are higher than the "meta" key:
    # https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h#L189

    # We don't support named tensors; graph break
    @property
    # pyrefly: ignore [bad-override]
    def names(self) -> list[str]:
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    @names.setter
    def names(self, _: list[str]) -> None:
        raise NotImplementedError

    @staticmethod
    def __new__(
        cls,
        fake_mode: FakeTensorMode,
        elem: Tensor,
        device: torch.device,
        constant: Optional[Tensor] = None,
        real_tensor: Optional[Tensor] = None,
        pytype: Optional[type[Tensor]] = None,
        dispatch_keys: Optional[torch.DispatchKeySet] = None,
    ) -> Self:
        self = Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        if not fake_mode._allow_unsafe_data_ptr_access:
            torch._C._set_throw_on_mutable_data_ptr(self)
        else:
            torch._C._set_warn_deprecated_on_mutable_data_ptr(self)

        assert elem.device.type == "meta", elem.device.type
        device = device if isinstance(device, torch.device) else torch.device(device)
        # NB: it is fine, if a little confusing, for device to be meta
        # (we are faking a meta tensor in that case).  However, it often
        # indicates some sort of confusion (e.g., you accidentally passed
        # in a meta tensor when you should have passed in the real tensor).
        # So by default we disallow meta, and if you are working in a situation
        # where it is helpful (e.g., crossref testing) you can turn it back
        # on
        if not fake_mode.allow_meta:
            assert device.type != "meta"
        # normalize device.
        if device.type in ["cuda", "xpu"]:
            init_gpu_context(device)

        if (
            device.type
            in [
                "cuda",
                "hpu",
                "xpu",
                "mps",
                "mtia",
                torch._C._get_privateuse1_backend_name(),
            ]
            and device.index is None
        ):
            if device.type != "mps" and getattr(torch, device.type).is_initialized():
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        # pyrefly: ignore [read-only]
        self.fake_device = device
        self.fake_mode = fake_mode
        self.constant = constant
        self.pytype = pytype
        self.dispatch_keys = dispatch_keys
        assert not isinstance(real_tensor, FakeTensor)
        self.real_tensor = real_tensor
        self.nonzero_memo = None
        self.item_memo = None
        self.unique_memo = None
        self.unique_consecutive_memo = None
        self.nested_int_memo = None

        if FakeTensorConfig.debug:
            self._debug_trace = CapturedTraceback.extract()  # type: ignore[attr-defined]
        return self

    # In some circumstances, a conventional Tensor constructor
    # will get rewritten to call into FakeTensor.  We must provide an
    # __init__ method that can accept the Python interpreters initialization
    # in such a situation; we must also be able to handle direct fake
    # tensor construction via FakeTensor().
    #
    # In particular, the __init__ call will look funny in the following case:
    #
    #   with FakeTensorMode():
    #       x = Tensor([1, 2, 3])
    #
    # this desugars into:
    #
    #   with FakeTensorMode():
    #       x = Tensor.__new__([1, 2, 3])
    #       # NB: x is a fake tensor, because of the mode!
    #       x.__init__([1, 2, 3])  # not the normal fake tensor args!
    #
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        if (
            torch.compiler.is_exporting()
            and torch._export.config.detect_non_strict_fake_tensor_leaks
        ):
            fake_tensor_tls.non_strict_export_fake_tensor_tracker.add(self)

    @staticmethod
    def from_tensor(t: Tensor, fake_mode: FakeTensorMode) -> FakeTensor:
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(  # type: ignore[override] # TODO
        cls,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func is torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # this handler must be done inside FakeTensor subclass, not mode, because
        # we can end up dispatching here when we have a fake tensor with
        # symbolic sizes running under in_kernel_invocation_manager.
        # The subclass is asked to handle this query because size (not
        # sym_size) was called, but we are unable to serve it directly because
        # there are symbolic sizes in the class.  The use of
        # in_kernel_invocation_manager means it's incorrect to activate a
        # mode to actually handle this (this caused
        # https://github.com/pytorch/pytorch/issues/122772).
        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break

        assert fake_mode is not None

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FAKE
        )
        if maybe_cur_fake_mode:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_mode,
                maybe_cur_fake_mode,
            )
            return NotImplemented

        assert not fake_mode.in_kernel_invocation

        with fake_mode:
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(
        func: OpOverload, flat_args: Sequence[object]
    ) -> tuple[torch.device, bool]:
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        # list of ops which can have args(tensor/tensorList) in mixed device
        mixed_device_fns = ordered_set(
            aten._foreach_copy.default,
        )

        # list of ops not using zero dim cpu tensor logic to align with the eager mode.
        bypass_zero_dim_cpu_tensor_check_ops = ordered_set(
            aten.nextafter.default,
        )

        def check_cpu_device(device: torch.device) -> bool:
            return device.type == "cpu"

        def cpu_zero_dim(t: Tensor) -> bool:
            return check_cpu_device(t.device) and t.dim() == 0

        def merge_devices(t: object) -> None:
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            is_bypass_zero_dim_cpu_tensor_check_op = (
                func in bypass_zero_dim_cpu_tensor_check_ops
            )

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # if still device mismatches we will check ops which can work
            # on different devices for ex. _foreach_copy, and one of the
            # device must be cpu in this case we will return from here without
            # throwing an error
            if func in mixed_device_fns:
                if any(map(check_cpu_device, (common_device, t.device))):
                    return

            # if prefer_device_type is set, prefer that device type over others
            prefer_device_type = torch._functorch.config.fake_tensor_prefer_device_type
            if prefer_device_type is not None:
                common_has_preferred = prefer_device_type in common_device.type
                t_has_preferred = prefer_device_type in t.device.type

                if not common_has_preferred and t_has_preferred:
                    # Switch to the preferred device type
                    common_device = t.device
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                    return
                elif common_has_preferred and not t_has_preferred:
                    # Keep the existing preferred device type
                    return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise RuntimeError(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        for arg in flat_args:
            merge_devices(arg)

        # some functions that allow Python numbers to bind to Tensors
        # if we have failed to find a device, and we're running one of these operators,
        # we must have scalar only inputs
        if should_allow_numbers_as_tensors(func) and common_device is None:
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        assert common_device is not None, f"Could not find common device for {func}"

        return common_device, has_scalar_only_inputs

    def get_nested_int(
        self,
        *,
        coeff: Union[int, torch.SymInt] = 1,
    ) -> torch.SymInt:
        if self.nested_int_memo is None:
            self.nested_int_memo = self.fake_mode.create_symbolic_nested_int(
                nt_tensor_id=None
            )
        assert isinstance(self.nested_int_memo, torch.SymInt)
        return self.nested_int_memo * coeff

    # Similar to FunctionalTensor.tolist
    def tolist(self) -> Any:
        if self.dim() == 0:
            return self.item()
        elif self.dim() == 1:
            return [elem.item() for elem in self]
        else:
            return [elem.tolist() for elem in self]


_MetadataIntLike = Union[IntLikeType, "_PySymInputStub", "_SymIntOutputStub"]


@dataclass(slots=True)
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    dtype: torch.dtype
    shape: tuple[_MetadataIntLike, ...]
    stride: tuple[_MetadataIntLike, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: _MetadataIntLike
    storage_bytes: Optional[_MetadataIntLike]
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool  # read: is sparse COO
    is_coalesced: Optional[bool]
    dense_dim: Optional[int]
    sparse_dim: Optional[int]

    def _flatten_into(
        self,
        result: list[object],
        mode: FakeTensorMode,
        state: _CacheKeyState,
    ) -> None:
        # Flatten the TensorMetadata out into `result`.  Make sure to call
        # state.convert_sym_int() on any SymInts.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (tuple, list, torch.Size)):
                # This will recursively flatten the iterable, calling
                # convert_sym_int() as necessary.
                id_hashed_objects: list[object] = []
                mode._prep_args_for_hash(result, value, state, id_hashed_objects)
                id_hashed_objects.clear()
            elif isinstance(value, SymInt):
                state.convert_sym_int(result, value)
            else:
                result.append(value)


def extract_tensor_metadata(t: Tensor) -> TensorMetadata:
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format = suggest_memory_format(t)
    # Don't call is_contiguous() on a Tensor which has symbolic sizes or things
    # will go badly (guards will be messed up?)
    if (
        t._has_symbolic_sizes_strides
        or is_sparse_any(t)
        or not t.is_contiguous(memory_format=memory_format)
    ):
        memory_format = None  # type: ignore[assignment]

    storage_offset = t.storage_offset()

    return TensorMetadata(
        t.dtype,
        t.shape,
        t.stride() if t.layout == torch.strided else (),
        t.device,
        t.layout,
        memory_format,
        storage_offset,
        # Only set storage_bytes for tensors that have storage (not sparse)
        t.untyped_storage().nbytes() if not is_sparse_any(t) else None,
        t.requires_grad,
        t.is_quantized,
        t.is_conj(),
        t.is_neg(),
        t.is_inference(),
        t.is_sparse,
        t.is_coalesced() if t.is_sparse else None,
        t.dense_dim() if is_sparse_any(t) else None,
        t.sparse_dim() if is_sparse_any(t) else None,
    )


@dataclass(slots=True)
class _DispatchCacheKey:
    """
    Key for the FakeTensor dispatch cache.
    """

    key: tuple[object, ...]
    hashvalue: int

    def __init__(self, tup: tuple[object, ...]) -> None:
        self.key = tup
        self.hashvalue = hash(tup)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DispatchCacheKey) and self.key == other.key

    def __hash__(self) -> int:
        return self.hashvalue

    def strip_shape_env(self) -> None:
        # We need to strip the ShapeEnv from any values before we store in the
        # cache so the cache doesn't keep our ShapeEnvs alive.
        for v in self.key:
            if isinstance(v, _PySymInputStub):
                v.strip_shape_env()


# Default value for constant_value in _DispatchCacheEntryOutputInfo. This is
# only for checking and differentiates from None.
class SingletonConstant:
    pass


@dataclass(frozen=True, slots=True)
class _DispatchCacheEntryOutputInfo:
    """
    Entry type for the FakeTensor dispatch cache for an output. Accounts for three
    possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a
       given index.
    2) We need to synthesize a new FakeTensor given tensor metadata. For view
       ops, we further capture the index of the arg to alias.
    3) if the tensor related fields are None, then it is a constant value (e.g.
    None or integer)
    """

    inplace_idx: Optional[int]
    metadata: Optional[TensorMetadata]
    view_idx: Optional[int]
    constant_value: Optional[Any] = SingletonConstant


@dataclass(frozen=True, slots=True)
class _DispatchCacheValidEntry:
    """
    Entry type for the FakeTensor dispatch cache. It supports two types of outputs
    1) tensor
    2) tuple of tensors

    is_output_tuple flag helps in differentiating the return type
    """

    output_infos: tuple[_DispatchCacheEntryOutputInfo]
    is_output_tuple: bool = False


@dataclass(frozen=True, slots=True)
class _DispatchCacheBypassEntry:
    """
    Entry type for a negative cache entry.
    """

    reason: str


if TYPE_CHECKING:
    _DispatchCacheEntry = Union[_DispatchCacheValidEntry, _DispatchCacheBypassEntry]


@dataclass(frozen=True, slots=True)
class _BypassDispatchCache(Exception):
    """
    Signals cases that should skip FakeTensor caching.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class DispatchCacheInfo:
    """
    Information about the state of the FakeTensor dispatch cache.
    """

    hits: int
    misses: int
    bypasses: dict[str, int]
    size: int


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.


class FakeTensorMode(TorchDispatchMode):
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: dict[str, int] = defaultdict(int)
    # Every time you retrace using the same fake tensor mode, you should
    # advance the epoch so we don't reuse unbacked memos
    epoch: int = 0
    in_kernel_invocation: bool = False
    static_shapes: bool
    shape_env: Optional[ShapeEnv]
    _stack: Optional[str]
    allow_meta: bool

    # NestedTensor uses a tensor_id_counter to uniquely identify offsets.
    # This counter is incremented when an offsets is used to create an NJT
    # for the first time. To avoid mutating eager state if we construct NJT
    # during tracing, we maintain a separate counter on the FakeTensorMode.
    # The initial count is set to the current eager tensor_id_counter value
    # upon initialization, and every time you retrace using the same fake tensor
    # mode, you should reset the counter to the initial count.
    nt_tensor_id_counter: int = -1
    nt_tensor_id_initial_count: int = -1

    def __init__(
        self,
        *,
        allow_fallback_kernels: bool = True,
        allow_non_fake_inputs: bool = False,
        shape_env: Optional[ShapeEnv] = None,
        static_shapes: Optional[bool] = None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        export: bool = False,
    ) -> None:
        log.debug("create_mode 0x%x", id(self))
        super().__init__()
        self.allow_fallback_kernels = allow_fallback_kernels

        import torch._dynamo.config
        import torch._functorch.config

        self.propagate_real_tensors = (
            torch._functorch.config.fake_tensor_propagate_real_tensors
        )
        self.fake_tensor_converter = FakeTensorConverter(
            copy_data=self.propagate_real_tensors,
            export=export,
        )

        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None

        # This is temporarily patched to True in Dynamo to grandfather in some
        # places where we unconditionally allow scalar outputs, TO BE REMOVED
        self.allow_scalar_outputs = False

        self._allow_unsafe_data_ptr_access = (
            torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access
        )
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        self.cache_enabled: bool = (
            torch._dynamo.config.fake_tensor_cache_enabled
            and not self.propagate_real_tensors
        )
        self.cache_crosscheck_enabled = (
            torch._dynamo.config.fake_tensor_cache_crosscheck_enabled
        )

        # A flag that controls, whether we want to invoke ops on mix of
        # real weights/global variables and fake inputs
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # when FakeTensor is invoked in user code, .device should return
        # the fake_device of the tensor so that code such as as `if x.is_cuda`
        # or torch.zeros([10, 10], device=x.device) continues to execute as if
        # the FakeTensor were real. However, within kernel execution, we return
        # the `Meta` device because all computation within the kernels should
        # behave as if the Tensors are on meta devices. Kernels should allocate
        # new tensors on meta devices, and checks like `is_meta` should return true.
        # within python refs, we always return the real device by defining
        # the device property
        self.in_kernel_invocation = False

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        # If another fake mode was already active when we enter, we also stash it here.
        # That way when we exit, we know to re-enable the previous fake mode.
        self.enter_stack: list[
            tuple[bool, Optional[TorchDispatchMode], Optional[bool]]
        ] = []

        self.shape_env = shape_env

        s
```



## High-Level Overview


This Python file contains 32 class(es) and 111 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IncrementRecursionCount`, `UnsupportedFakeTensorException`, `DynamicOutputShapeException`, `DataDependentOutputException`, `UnsupportedOperatorException`, `UnsupportedMutationAliasingException`, `MetadataMismatchError`, `FakeTensorTLS`, `FakeTensorConverter`, `FakeTensorConfig`, `SymNumberMemoDescriptor`, `FakeTensor`, `TensorMetadata`, `_DispatchCacheKey`, `SingletonConstant`, `_DispatchCacheEntryOutputInfo`, `_DispatchCacheValidEntry`, `_DispatchCacheBypassEntry`, `_BypassDispatchCache`, `DispatchCacheInfo`

**Functions defined**: `__init__`, `__del__`, `__init__`, `ordered_set`, `unset_fake_temporarily`, `disable_fake_tensor_cache`, `get_plain_tensors`, `is_fake`, `maybe_get_fake_mode`, `get_schema_info`, `torch_decomp_decompositions`, `tree_flatten_only`, `_is_plain_tensor`, `tensor_memo`, `__init__`, `add_constant_storage_mapping`, `invalidate_constant_aliases`, `_get_memo`, `set_tensor_memo`, `from_real_tensor`

**Key imports**: annotations, atexit, contextlib, dataclasses, functools, logging, math, os, threading, traceback


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_subclasses`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `atexit`
- `contextlib`
- `dataclasses`
- `functools`
- `logging`
- `math`
- `os`
- `threading`
- `traceback`
- `types`
- `typing`
- `weakref`
- `collections`: defaultdict
- `typing_extensions`: Self
- `torch`
- `torch._library.utils as library_utils`
- `torch._C._functorch`: is_functorch_wrapped_tensor, is_legacy_batchedtensor
- `torch._library.fake_class_registry`: FakeScriptObject
- `torch._library.fake_profile`: MissingOpProfile
- `torch._logging`: dtrace_structured
- `torch._prims_common`: suggest_memory_format
- `torch._utils`: render_call
- `torch.fx.immutable_collections`: immutable_dict
- `torch.fx.operator_schemas`: normalize_function
- `torch.multiprocessing.reductions`: StorageWeakRef
- `torch.overrides`: TorchFunctionMode


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_subclasses`):

- [`fake_utils.py_docs.md`](./fake_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_fake_tensor_utils.py_docs.md`](./_fake_tensor_utils.py_docs.md)
- [`fake_impls.py_docs.md`](./fake_impls.py_docs.md)
- [`functional_tensor.py_docs.md`](./functional_tensor.py_docs.md)
- [`meta_utils.py_docs.md`](./meta_utils.py_docs.md)
- [`schema_check_mode.py_docs.md`](./schema_check_mode.py_docs.md)


## Cross-References

- **File Documentation**: `fake_tensor.py_docs.md`
- **Keyword Index**: `fake_tensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
