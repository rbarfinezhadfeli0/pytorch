# Documentation: `docs/torch/_inductor/cudagraph_trees.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/cudagraph_trees.py_docs.md`
- **Size**: 56,178 bytes (54.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/cudagraph_trees.py`

## File Metadata

- **Path**: `torch/_inductor/cudagraph_trees.py`
- **Size**: 104,013 bytes (101.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
CUDA graph trees are a safety abstraction over CUDAGraphs, similar to make_graph_callables,
which share the same memory pool.  Sharing a memory pool is an extremely
important optimization when chaining multiple CUDA graphs together, as it
prevents you from needing to copy intermediate tensors from one graph to the
next, and reduces overall memory usage by allowing dead memory from the first
pool to be reused in the second.

The standard graph/make_graph_callables support sharing memory pool, but
with a lot of caveats.  CUDA graph trees remove these restrictions:

* Previously, if you recorded graphs A, B, you had to replay A, B in that
  order.  With CUDA graph trees, after replaying A, you can change your
  mind and record/replay a different graph B'; we will support efficient
  execution of both A, B and A, B', using only max(mem(A, B), mem(A, B')).  In
  other words: we support arbitrary trees of CUDA graph operations, not just
  sequences (this is why this feature is called CUDA graph trees.)

* Previously, if you executed graph A, some non-CUDA graph code, and then
  graph B, after executing graph B, it was not safe to retain any references
  to intermediates produced by A.  With CUDA graph trees, we track if any
outputs of graph A are still live by the time graph B is run, and make
  sure graph B doesn't clobber there memory when reusing the CUDA graphs
  pool.  You'll get a separate recording of B depending on what tensors
  stay live or dead.

CUDA graph trees are flexible enough to be used in Dynamo across graph breaks,
which is their primary use case.

The ability to switch from replay to record is fairly nontrivial: remember that
when you replay a CUDA graph, you only replay CUDA operations; no CPU side state
is updated.  In particular, the CPU-side book-keeping for the allocator is not
reconstructed.  However, to record a new child CUDA graph, we must restore this
book-keeping.  This is what checkpoint pool state is used for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import itertools
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Any, cast, Optional, TYPE_CHECKING, TypeVar, Union

import torch.fx
from torch import Tensor
from torch._dynamo.callback import CallbackTrigger
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import counters, dynamo_timed, preserve_rng_state
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    check_for_mutation,
    CheckInvariantStatus,
    FunctionID,
    log_cudagraph_skip_and_bump_counter,
    log_data_ptr_mismatch,
    maybe_warning_due_to_dynamic_shape,
    ModelType,
    OutputType,
    PlaceholderInfo,
    WrappedFunction,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.weak import TensorWeakRef


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence

    from torch._guards import CompileId
    from torch._inductor.utils import InputType
    from torch.cuda import _POOL_HANDLE
    from torch.types import _bool

StorageWeakRefPointer = int
StorageDataPtr = int
NBytes = int
S = TypeVar("S", bound="StorageWeakRefWrapper")


if torch.backends.cuda.is_built():
    from torch._C import (
        _cuda_CUDAAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")


from . import config


@dataclasses.dataclass(frozen=True)
class GraphID:
    "Unique counter of a cuda graph recording"

    id: int


def clear_cublass_cache() -> None:
    """
    Cublas keeps a persistent workspace allocation for running matmuls. This poses a problem for
    doing warmup within a CUDAGraph private pool because we do not want persistent allocations from
    one one run to the next. When we begin a new run of a cudagraphs path (generation), all tensors
    from the previous generation are freed. This frees them the memory pool, but not elsewhere.
    A tensor in the cublas workspace would continue to be in use the workspace but would also get allocated
    in the next run. The memory would be in use in two places.

    To solve this, we clear cublas caches before and after warming up or recording. If a workspace is required
    it will be allocated to the cudagraph private pool and accounted for in the allocator for the duration of the
    program. There is no overhead to this on replay since cudagraphs removes allocation overhead.
    """
    torch._C._cuda_clearCublasWorkspaces()


@contextlib.contextmanager
def clear_cublas_manager() -> Generator[None, None, None]:
    "Context manager around clearing cublas caches that will clear on enter and exit"
    clear_cublass_cache()
    try:
        yield
    finally:
        clear_cublass_cache()


@contextlib.contextmanager
def disable_conv_cache_emptying() -> Generator[None, None, None]:
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)


@contextlib.contextmanager
def enable_history_recording() -> Generator[None, None, None]:
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch._C._cuda_isHistoryEnabled()
    try:
        if not enabled:
            torch.cuda.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.cuda.memory._record_memory_history(None)


def get_history_recording() -> AbstractContextManager[None]:
    # TODO - remove, prevents cleanup
    if not config.triton.cudagraph_trees_history_recording:
        return contextlib.nullcontext()
    return enable_history_recording()


class TreeManagerContainer:
    """
    Manages the lifetime of the tree manager. Like `PrivatePool` in cuda caching allocator,
    the tree and its corresponding memory pool should be kept alive as long as any outstanding
    graph or tensor which is an output of a graph remains alive.

    There is a single tree manager container per device.

    The lifecycle of a tree_manager is:
    -  Is constructed, no graph, no fns, no tensors
    -  Tree manager is fetched, resulting in tree manager being allocated
    -  We generate a bunch of functions, calling add_strong_reference
    -  These functions die, calling finalize_reference
    -  When all the functions die, we finalize_tree_manager.

    TODO: in the future, we would like to do the following once storage weak refs land
    -  We look for all the live storages and add references to THOSE
    -  We count as storages die
    -  All the storages are dead, we deallocate the tree manager
    """

    def __init__(self, device_index: int) -> None:
        # This class keeps a strong reference to tree_manager,
        # but upon all other strong references to the tree_manager will reset it to None.
        # We need a strong reference so that we can still access its attributes upon cleanup.
        self.tree_manager: Optional[CUDAGraphTreeManager] = None

        # Number of outstanding references to the current tree manager
        self.live_cudagraphify_fns = 0

        self.device_index = device_index

        # Following two objects are only set in the case that Tensor outputs outlive
        # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
        # deallocation.
        self.live_storages_count = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        self.lock = threading.Lock()

    def _finalize_tensor(self) -> None:
        with self.lock:
            self.live_storages_count -= 1
            if self.live_storages_count == 0:
                self.graph = None

                # manager was used again after existing cleanup,
                # we shouldn't set it to None
                if self.live_cudagraphify_fns == 0:
                    self.tree_manager = None

    def finalize_cudagraphify_fn(self) -> None:
        with self.lock:
            self.live_cudagraphify_fns -= 1
            if self.live_cudagraphify_fns == 0:
                self._finalize_tree_manager()

    def _finalize_tree_manager(self) -> None:
        assert self.lock.locked()
        self.tree_manager = None

        # TODO - when issue #91395 is landed, we can set a weakref on
        # storages and trigger a deallocation when all outputs of the
        # cudagraph are dead.

        # live_storages = list(
        #     tree_manager.live_cudagraph_pool_storages_in_curr_execution()
        # )

        # # Maintain reference to graph to keep tensors alive
        # assert len(tree_manager.roots) > 0, "expected at least one use"
        # root = next(tree_manager.get_roots())
        # self.graph = root.graph
        # seen_storages = set()
        # for stor in live_storages:
        #     if stor in seen_storages:
        #         continue
        #     seen_storages.add(stor)
        #     self.live_storages_count += 1
        # .   weakref.finalize(stor, self._finalize_tensor)

    def add_strong_reference(self, fn: Callable[..., Any]) -> None:
        with self.lock:
            self.live_cudagraphify_fns += 1

        weakref.finalize(fn, self.finalize_cudagraphify_fn)

    def get_tree_manager(self) -> CUDAGraphTreeManager:
        with self.lock:
            if self.tree_manager is None:
                self.tree_manager = CUDAGraphTreeManager(self.device_index)
            return self.tree_manager


local = threading.local()

# one tree manager per device
local.tree_manager_containers = {}
local.tree_manager_locks = defaultdict(threading.Lock)


# only incremented by user call of mark_step_begin
class MarkStepBox:
    mark_step_counter = 0


# We need to register this as an object that will be copied over as TLS when new
# threads are created in autograd
torch._C._stash_obj_in_tls("tree_manager_containers", local.tree_manager_containers)
torch._C._stash_obj_in_tls("tree_manager_locks", local.tree_manager_locks)


def mark_step_begin() -> None:
    "Indicates that a new iteration of inference or training is about to begin."

    # iterate down to distinguish from GenerationTracking counter
    MarkStepBox.mark_step_counter -= 1


def reset_cudagraph_trees() -> None:
    "Clear all cudagraph trees"
    # see shutdown below for why this is necessary
    container_dict = get_obj(local, "tree_manager_containers")
    locks_dict = get_obj(local, "tree_manager_locks")
    for device, lock in locks_dict.items():
        with lock:
            container = container_dict.get(device)
            if not container or not container.tree_manager:
                continue

            container.tree_manager.shutdown()

    _set_cached_tensors_enabled(False)
    container_dict.clear()

    MarkStepBox.mark_step_counter = 0


def get_obj(local: Any, attr_name: str) -> Any:
    if hasattr(local, attr_name):
        return getattr(local, attr_name)
    else:
        assert torch._C._is_key_in_tls(attr_name)
        return torch._C._get_obj_in_tls(attr_name)


def get_container(device_index: int) -> TreeManagerContainer:
    container_dict = get_obj(local, "tree_manager_containers")
    lock = get_obj(local, "tree_manager_locks")[device_index]

    with lock:
        if device_index not in container_dict:
            container_dict[device_index] = TreeManagerContainer(device_index)

        return container_dict[device_index]


def get_manager(
    device_index: int, create_if_none_exists: bool = True
) -> Optional[CUDAGraphTreeManager]:
    if create_if_none_exists:
        return get_container(device_index).get_tree_manager()
    return get_container(device_index).tree_manager


def is_cudagraph_capture_sizes(int_key: Union[int, tuple[int, ...]]) -> bool:
    """
    Returns true if all dynamic shapes should be captured or the dynamic shape
    int_key should be captured.
    """
    return (
        config.triton.cudagraph_capture_sizes is None
        or int_key in config.triton.cudagraph_capture_sizes
    )


def cudagraphify_impl(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    *args: Any,
    **kwargs: Any,
) -> ModelType:
    fn_cache: dict[tuple[int, ...], Callable[..., Any]] = {}

    # Detect int inputs: we need to index on these
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    has_warn = False

    del inputs

    def deferred_cudagraphify(inputs: list[InputType]) -> OutputType:
        nonlocal has_warn

        int_key = get_ints(inputs)

        if not is_cudagraph_capture_sizes(int_key):
            return model(inputs)

        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)

        if int_key is None:
            log.info("recording cudagraph tree for graph without symints")
        else:
            log.info("recording cudagraph tree for symint key %s", int_key)

        if not has_warn:
            has_warn = maybe_warning_due_to_dynamic_shape(fn_cache, int_key)

        # first get indices we need to check to align, then update our static inputs,
        # and finally copy
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)

        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        # cudagraph will already clones input locally, no need to copy back
        mutated_input_idxs: OrderedSet[int] = OrderedSet()
        fn = align_inputs_from_check_idxs(
            fn, inputs_to_check=check_input_idxs, mutated_input_idxs=mutated_input_idxs
        )
        # pyrefly: ignore [unsupported-operation]
        fn_cache[int_key] = fn

        return out

    return deferred_cudagraphify


@contextlib.contextmanager
def dynamo_timed_cudagraph(
    name: str,
    compile_id: Optional[CompileId],
    mode: Optional[CompilationMode],
) -> Generator[Any, None, None]:
    """
    Makes usages of dynamo_timed in this file less verbose. NOTE: This CM sums
    all durations into a single column in the dynamo_compile table. Use only if
    you consider the timed region to be part of the runtime overhead associated
    with the compiler.
    """
    with dynamo_timed(
        name,
        log_pt2_compile_event=True,
        compile_id=compile_id,
        is_backward=mode == CompilationMode.BACKWARD,
        dynamo_compile_column_us="runtime_cudagraphify_time_us",
    ):
        yield


def cudagraphify(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: Optional[StackTraces] = None,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: tuple[PlaceholderInfo, ...] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    compile_id: Optional[CompileId] = None,
) -> tuple[ModelType, OutputType]:
    assert not (is_backward and is_inference)
    mode = (
        CompilationMode.BACKWARD
        if is_backward
        else (CompilationMode.INFERENCE if is_inference else CompilationMode.FORWARD)
    )

    with dynamo_timed_cudagraph("cudagraphify.get_container", compile_id, mode):
        manager = get_container(device_index).get_tree_manager()

    return manager.add_function(
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
        constants,
        placeholders,
        mutated_input_idxs,
        compile_id,
    )


class StorageWeakRefWrapper:
    """
    Wrapper around a storage weak ref. Will deallocate it upon expiration if invoked.
    """

    __slots__ = ["ref", "_data_ptr", "extra_ref_check"]

    storage_ref: Optional[StorageWeakRef]

    def __init__(
        self,
        inp: Union[Tensor, UntypedStorage],
        extra_ref_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        extra_ref_check is an additional check we need to run to check if the
        weak ref has expired. in checking storage use count we assume extra_ref_check
        will hold an additional reference to the storage.
        """
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            assert isinstance(inp, UntypedStorage)
            stor = inp
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = extra_ref_check

    @classmethod
    def from_weakref_and_data_ptr(
        cls: type[StorageWeakRefWrapper],
        cdata: Any,
        data_ptr: int,
        extra_ref_check: Optional[Callable[[], bool]] = None,
    ) -> StorageWeakRefWrapper:
        instance = cls.__new__(cls)
        instance._data_ptr = data_ptr
        instance.ref = StorageWeakRef.from_weakref(cdata)
        instance.extra_ref_check = extra_ref_check
        return instance

    def __call__(self) -> Optional[StorageWeakRefPointer]:
        if self.expired():
            return None

        return self.ref.cdata

    def swap_weakref(self, cdata: Any) -> None:
        self.ref.__del__()
        self.ref.cdata = cdata

    def data_ptr(self) -> int:
        "NB: returns the data ptr even if the storage has expired"
        return self._data_ptr

    def remove_extra_reference(self) -> None:
        self.extra_ref_check = None

    def expired(self) -> bool:
        if self.extra_ref_check is not None and not self.extra_ref_check():
            return False

        # if extra_ref_check is not None we expect an additional reference
        stor_count = torch._C._storage_Use_Count(self.ref.cdata)
        return (stor_count - (self.extra_ref_check is not None)) == 0

    def __repr__(self) -> str:
        if self.ref is None or self.ref.expired():
            return f"StorageWeakRefWrapper to {self.data_ptr()}; dead"
        else:
            return f"StorageWeakRefWrapper to {self.data_ptr()}; alive"


def is_live(weak_ref: Optional[StorageWeakRefWrapper]) -> bool:
    return maybe_deref(weak_ref) is not None


def maybe_deref(
    weak_ref: Optional[StorageWeakRefWrapper],
) -> Optional[tuple[StorageWeakRefPointer, int]]:
    if weak_ref is None:
        return None
    r = weak_ref()
    if r is None:
        return None
    # NB: r.data_ptr() does not necessarily equal weak_ref.data_ptr()
    return r, weak_ref.data_ptr()


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(
    device: int, mem_pool: tuple[int, int], stream: torch.cuda.Stream
) -> Generator[None, None, None]:
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.device(device):
        # Begin allocate to mem pool for all memory allocation on the current thread.
        # This is thread safe since a thread can only warmup or record 1 cudagraph
        # at the same time.
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mem_pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateToPool(device, mem_pool)
            torch._C._cuda_releasePool(device, mem_pool)

    torch.cuda.current_stream().wait_stream(stream)


def map_to_ref(t: Optional[Tensor]) -> Optional[StorageWeakRefWrapper]:
    if not isinstance(t, torch.Tensor):
        assert t is None
        return None
    return StorageWeakRefWrapper(t)


# A path index of (depth, offset) indices into a graph that is `depth`` number of nodes from the root
# at graph output offset
PathOutputIndex = tuple[int, int]

# For each node in the path, for each output, is the output alive
PathLiveness = list[list[bool]]

StackTraces = list[Optional[str]]


class CUDAWarmupNode:
    """
    Simplified Wrapper around A CUDA Model that wraps outputs in storage refs and exposes
    apis to get the live storages in the current chain of warmup.

    A CUDAWarmupNode may have either CUDAGraphNode or CUDAWarmupNode as a parent, but may only have
    CUDAWarmupNode as children, because we cannot record or execute with tensors which do not have stable
    memory addresses.

    CUDAWarmupNode and CUDAGraphNode have a number of differences that make it easier to use separate classes.
    - Much of the CUDAGraphNode logic & initialization is based on the tensor properties of first recording. In the
    first instance of warmup, these are not finalized yet.
    - All Inputs to the RecordedFunction must be copied over to the cuda graph memory pool, this is unnecessary in warmup.
    - CUDAWarmup is only used once and so does not need to optimize as much bookkeeping. It is much simpler.

    NB: this class and CUDAGraphNode need to expose `path_live_weakrefs`, `all_outputs_are_dead`, and
    `self.outputs_weakrefs`, `stack_traces`, and `tensor_weakrefs` for compatibility.
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        parent: Optional[Union[CUDAGraphNode, CUDAWarmupNode]],
        cuda_graphs_pool: tuple[int, int],
        existing_cuda_graph: Optional[torch.cuda.CUDAGraph],
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.cuda.Stream,
        already_warm: bool,
        id: GraphID,
    ) -> None:
        self.wrapped_function = wrapped_function
        self.parent: Optional[Union[CUDAGraphNode, CUDAWarmupNode]] = parent
        self.cuda_graphs_pool = cuda_graphs_pool
        self.outputs_weakrefs: list[Optional[StorageWeakRefWrapper]] = []
        self.tensor_weakrefs: list[Optional[TensorWeakRef]] = []
        self.existing_cuda_graph = existing_cuda_graph
        self.has_run = False
        self.device_index = device_index
        self.stack_traces = stack_traces
        self.stream = stream
        self.already_warm = already_warm
        self.id = id

    def run(self, new_inputs: Any) -> OutputType:
        assert not self.has_run, "Wrapped function should never be run twice"

        # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
        # storages in path_live_weakrefs.
        existing_path_data_ptrs = OrderedSet(
            [t.data_ptr() for t in self.path_live_weakrefs() if t()]
        )

        def get_non_cudagraph_inps() -> list[weakref.ReferenceType[UntypedStorage]]:
            non_cudagraph_inps = [
                weakref.ref(t.untyped_storage())
                for t in itertools.chain(new_inputs, self.wrapped_function.constants)
                if isinstance(t, torch.Tensor)
                and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
            ]
            return non_cudagraph_inps

        non_cudagraph_inps_storages = get_non_cudagraph_inps()

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

        with (
            torch.cuda.device(self.device_index),
            disable_conv_cache_emptying(),
            clear_cublas_manager(),
            _use_cuda_memory_pool_manager(
                self.device_index, self.cuda_graphs_pool, self.stream
            ),
            get_history_recording(),
        ):
            out = self.wrapped_function.model(new_inputs)

        # We need to know which outputs are allocated within the cudagraph pool
        # so that we can deallocate them at the beginning of the next cudagraph step,
        # and set their access to error.
        # We use a weakref to the inputs storage, in case a block which was previously
        # allocated to the general caching allocator pool gets reallocated to a private pool.

        non_cudagraph_inps_storage_ptrs = OrderedSet[Any]()
        for storage in non_cudagraph_inps_storages:
            s = storage()
            if s is not None:
                non_cudagraph_inps_storage_ptrs.add(s._cdata)

        assert len(new_inputs) == 0

        # sdpa returns cpu tensors when not recording cuda graph
        def add_ref(o: Any) -> bool:
            return (
                isinstance(o, torch.Tensor)
                and o.is_cuda
                and o.untyped_storage()._cdata not in non_cudagraph_inps_storage_ptrs
                and o.untyped_storage().data_ptr() != 0
            )

        self.outputs_weakrefs.extend(
            [map_to_ref(o) if add_ref(o) else None for o in out]
        )
        self.tensor_weakrefs.extend(
            [TensorWeakRef(o) if add_ref(o) else None for o in out]
        )

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            out_refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, out_refs)

        return out

    @property
    def _path_from_root(
        self,
    ) -> Generator[Union[CUDAGraphNode, CUDAWarmupNode], None, None]:
        nodes = []
        node: Union[CUDAGraphNode, CUDAWarmupNode] = self
        while node:
            nodes.append(node)
            node = node.parent  # type: ignore[assignment]

        yield from reversed(nodes)

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        "Returns all live storages weakrefs that created by nodes in this path"
        for node in self._path_from_root:
            for output in node.outputs_weakrefs:
                if is_live(output):
                    yield output  # type: ignore[misc]

    def all_outputs_are_dead(self) -> bool:
        return not list(self.path_live_weakrefs())

    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor) -> bool:
        for storage_weak_ref in self.path_live_weakrefs():
            if t.untyped_storage().data_ptr() == storage_weak_ref.data_ptr():
                return True
        return False


# Aliases for List that say what the indices denote
InputList = list  # input indexes
OutputList = list  # output indexes
LevelList = list  # levels (distance from root of tree)


class OutputAliasInfo:
    pass


class _UnaliasedStorage(OutputAliasInfo):
    "Singleton to mark that the graph output constructs a new alias or is None"


UnaliasedStorage = _UnaliasedStorage()


class AliasesPriorGraphOutput(OutputAliasInfo):
    "Marks that the graph output aliases an output of a prior graph"

    __slots__ = ["index"]

    index: PathOutputIndex

    def __init__(self, index: PathOutputIndex) -> None:
        assert isinstance(index, tuple)
        self.index = index


class AliasesNewOutput(OutputAliasInfo):
    "Marks that the graph output aliases an index in the new, returned outputs"

    __slots__ = ["index"]

    index: int

    def __init__(self, index: int) -> None:
        assert isinstance(index, int)
        self.index = index


class CUDAGraphNode:
    """
    A single recording of a function into a CUDA Graph. Recordings of CUDA Graphs share a single memory pool
    and are structured into a tree, where there is a single recording that can precede it (parent) and multiple
    subsequent recordings that may follow (children). A node will have no parent if it is the first recording
    in a tree; i.e., when it is first recorded, there are no live tensors from a previous recording which
    would force a dependency.

    On first recording, all of the live tensors in the current CUDA Graph Node path will be
    reflected in the corresponding private pool. On subsequent executions, the caching allocator
    is unaffected when the graph is replayed.

    In order to support recording a subsequent cuda graph recording after execution of this graph,
    we checkpoint the state of the memory pool so that it may later be resumed.

    WrappedFunction should have already been warmed up prior to invocation.

    See [setCheckpointPoolState] for further explanation, as well as
    https://user-images.githubusercontent.com/13564/222815509-374f3400-f83d-4f7d-8fa6-4a092b3250bb.png
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        id: GraphID,
        parent: Optional[CUDAGraphNode],
        inputs: list[InputType],
        cuda_graphs_pool: _POOL_HANDLE,
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.cuda.Stream,
        mode: Optional[CompilationMode],
        compile_id: Optional[CompileId],
    ) -> None:
        assert isinstance(inputs, (list, tuple))

        self.wrapped_function = wrapped_function
        self.id = id
        self.device = device_index
        self.stack_traces = stack_traces
        self.stream = stream

        # Enable re-record a cudagraph when static tensor address changed.
        # if not we should error when it changed.
        self.rerecord_if_static_inputs_change = (
            torch._dynamo.config.inline_inbuilt_nn_modules
            or torch._inductor.config.triton.cudagraph_support_input_mutation
        )

        # if this is a root parent will be None. use weakref to prevent reference cycle
        self._parent = weakref.ref(parent) if parent is not None else None
        # reference to the shared memory pool for the entire cuda graphs tree
        self.cuda_graphs_pool = cuda_graphs_pool

        # A single wrapped function may be recorded multiple times if memory patterns or
        # invariants change from one execution to the next
        self.children: dict[FunctionID, list[CUDAGraphNode]] = defaultdict(list)

        # StorageWeakRef maintains whether the Storage C++ object remains allocated,
        # not whether the corresponding memory has been deallocated. In order
        # to use them to track memory deallocations we must maintain a single StorageWeakRef
        # for all Storages that reference that memory (even if we are constructing Storages
        # that do not have a deallocator function). We maintain one single storage_cache
        # as we execute any tree path. When we retrieve a storage from the cache we
        # check that it is still alive, and we hash based on observed recording data ptr
        # and storage cdata.

        # we preserve a single reference to executed outputs that is then referenced
        # in children to avoid children having to chase parent pointers in the hot path
        # DO NOT reassign output_weakrefs, only call `clear()`
        # Path is a series of nodes from root to the current node
        self.outputs_weakrefs: OutputList[Optional[StorageWeakRefWrapper]] = []
        self.path_weakrefs: LevelList[OutputList[Optional[StorageWeakRefWrapper]]] = [
            node.outputs_weakrefs for node in self._path_from_root
        ]
        self.path_stacktraces: LevelList[Optional[StackTraces]] = [
            node.stack_traces for node in self._path_from_root
        ]
        self.tensor_weakrefs: OutputList[Optional[TensorWeakRef]] = []

        # tensors which are outputs of previous graphs in the tree
        self.cudagraph_managed_idxs: list[int] = [
            idx
            for idx, t in enumerate(inputs)
            if isinstance(t, torch.Tensor) and self._is_cuda_graph_recorded_tensor(t)
        ]

        # (depth, offset) of live tensors which are alias of previous graph outputs
        self.live_cudagraph_managed_path_refs: InputList[Optional[PathOutputIndex]] = [
            (
                self._is_alias_of_live_recorded_tensor(t)
                if isinstance(t, torch.Tensor)
                else None
            )
            for t in inputs
        ]

        # when replay, preserve the liveness of an input if it AliasesPriorGraphOutput
        # and also aliases an output of the current CUDAGraphNode
        self.preserved_aliased_inputs: InputList[bool] = [False] * len(inputs)

        self.static_input_idxs: list[int] = list(
            OrderedSet(wrapped_function.static_input_idxs)
            | OrderedSet(self.cudagraph_managed_idxs)
        )

        self.non_static_input_idx: LevelList[int] = [
            i for i in range(len(inputs)) if i not in self.static_input_idxs
        ]

        counters["inductor"]["cudagraph_recorded_non_static_inputs"] += len(
            self.non_static_input_idx
        )

        self.non_managed_static_input_idxs: LevelList[int] = [
            i
            for i in wrapped_function.static_input_idxs
            if i not in self.cudagraph_managed_idxs
        ]

        def maybe_get_static_data_ptr(
            idx: int,
            inputs: list[InputType],
            static_input_idxs: list[int],
        ) -> Optional[int]:
            inp = inputs[idx]
            if isinstance(inp, torch.Tensor) and idx in static_input_idxs:
                return inp.data_ptr()
            return None

        self.static_input_data_ptrs: InputList[Optional[int]] = [
            # pyrefly: ignore [bad-argument-type]
            maybe_get_static_data_ptr(i, inputs, self.static_input_idxs)
            for i in range(len(inputs))
        ]

        # When we checkpoint, and free generations, we will be manually freeing the outputs
        # of CUDAGraphNodes. We should not be freeing parameters, not do we need to account for
        # their liveness (they are static), so we need to compute which outputs are aliases of
        # parameters. Some static inputs are saved tensors from the forward that die in the backward.
        # Their locations are static but lifetimes are not. We only include the persistent static
        # data ptrs below because the non persistent data ptrs may be outputs of this record and
        # fresh allocations.

        # precompute expanded dims to avoid computing in the hot path
        self.expanded_dims: list[list[int]] = [
            get_expanded_dims(x)
            if isinstance(x, torch.Tensor) and idx not in self.static_input_idxs
            else []
            for idx, x in enumerate(inputs)
        ]

        # For each node in path, which outputs were observed to be live
        # before invoking graph recording, and after graph recording
        self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
        self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []

        # List of tuples of (depth, output_index) that index into node at depth
        # number of nodes from root and output_index of outputs. Will index into
        # path_weakrefs.
        self.expected_dead_indices_before_graph: list[PathOutputIndex] = []
        self.expected_dead_indices_after_graph: list[PathOutputIndex] = []

        # all live indices after graph recording
        self.live_indices_after_graph: list[PathOutputIndex] = []

        if self.parent is not None:
            previous_liveness = self.parent.recorded_liveness_after_graph
            curr_liveness = self._get_liveness(self.path_weakrefs)

            different_indices = self._get_different_indices(
                previous_liveness, curr_liveness
            )

            self.recorded_liveness_before_graph = curr_liveness
            self.expected_dead_indices_before_graph = different_indices

        rng_states = [inp for inp in inputs if isinstance(inp, torch.Generator)]
        # pyrefly: ignore [bad-argument-type]
        recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
        # recording inputs will copy over memory, so we can free non recording inputs
        # pyrefly: ignore [missing-attribute]
        inputs.clear()
        del inputs

        # graph used for recording model invocation
        self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()

        # TODO: register_generator_state should potentially take explicit device
        with torch.cuda.device(self.device):
            for rng_state in rng_states:
                self.graph.register_generator_state(rng_state)

        # we allocate non-static inputs within the same memory pool as the CUDAGraph
        # which we will record the model with. For memory efficiency, it is important
        # to reclaim the input memory when the inputs are no longer live. To accomplish this,
        # we reconstruct tensors at the correct data pointers of our inputs which are
        # non owning and do not prevent deallocation. On subsequent executions, input values
        # will be copied over to these tensors.
        self.reconstructed_inputs: list[InputType] = [
            self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
            if isinstance(x, torch.Tensor)
            else x
            for x in recording_inputs
        ]

        # DO THE RECORDING!!!
        # We record the CUDA graph in the constructor of CUDAGraphNode, which
        # gives you what the CPU side compute of the function would do.  We
        # don't throw the recording outputs away: their memory is
        # correctly accounted for in the CUDAGraphs caching allocator.  This
        # means on the very FIRST run of the CUDA graph node, we can directly
        # do more recording, because we have a valid caching allocator state.
        # NB: This relies on run() being called immediately after the
        # constructor, otherwise this optimization would not be valid.

        # initialized below in _record

        self.checkpointed_caching_state: Optional[AllocatorState] = None

        # Output Storage Alias information, can be:
        # - A new, unaliased storage, or the output is None
        # - An alias of an output of a prior graph
        # - An alias of an output already created in the reconstructed outputs
        # This is None if the output in question is an int
        self.output_storage_alias: OutputList[Optional[OutputAliasInfo]] = []

        # is the output Storage unaliased in subsequent outputs, of all subsequent paths
        # if it is, we cached the output tensor and adjust storage liveness tracking to also
        # check if the output tensor does not have an additional python reference.
        # If a descendent node discovers it has an alias of a prior output, then the output
        # will no longer be cached in the ancestor.
        # The large majority of tensors are unaliased, and preserving aliased output tensors would add
        # significant additional complexity with marginal gains
        # The cached tensor outputs are added on the first execution, and cleared whenever we need
        # to do subsequent recording
        self.unaliased_in_all_paths: OutputList[bool] = []
        self.cached_tensor_outputs: OutputList[Optional[Tensor]] = []

        # if an output aliases a static, persistent input then the corresponding Tensor will
        # be set here. These are different than cached tensors, because they are tensors that
        # are aliases of parameters that are always live.
        self.static_output_tensors: OutputList[Optional[Tensor]] = []

        # Cleared after recording
        with dynamo_timed_cudagraph("CUDAGraphNode.record", compile_id, mode):
            self.recording_outputs: Optional[OutputType] = self._record(
                wrapped_function.model, recording_inputs
            )
        self.outputs_metadata: OutputList[Union[dict[str, Any], int, None]] = []

        # As with inputs, we do not want to keep the outputs permanently alive because that would prevent
        # their memory being reclaimed in subsequent cuda graph recordings. We record the tensor metadata
        # needed to reconstruct instead.
        assert self.recording_outputs is not None
        for out in self.recording_outputs:
            if isinstance(out, torch.Tensor):
                self.outputs_metadata.append(
                    self._tensor_metadata(out, ignore_storage_offset=False)
                )
            else:
                assert isinstance(out, (int, type(None))), type(out)
                self.outputs_metadata.append(out)

        self.graph.replay()

    def _copy_inputs_and_remove_from_src(
        self, dsts: list[InputType], srcs: list[InputType]
    ) -> None:
        dst_tensors = []
        src_tensors = []
        for idx in self.non_static_input_idx:
            if not isinstance(srcs[idx], torch.Tensor):
                continue
            expanded_dims = self.expanded_dims[idx]
            dst_tensors.append(index_expanded_dims(dsts[idx], expanded_dims))  # type: ignore[arg-type]
            src_tensors.append(index_expanded_dims(srcs[idx], expanded_dims))  # type: ignore[arg-type]
            srcs[idx] = None  # type: ignore[call-overload]
        # Fails on empty lists
        if dst_tensors:
            torch._foreach_copy_(dst_tensors, src_tensors)

    def check_static_inputs_are_stable(self, new_inputs: list[InputType]) -> None:
        # avoid checking managed tensor static points since we already checked those in check_invariants
        if (
            not self.rerecord_if_static_inputs_change
            and not torch._C._tensors_data_ptrs_at_indices_equal(
                new_inputs,  # type: ignore[arg-type]
                self.static_input_data_ptrs,
                self.non_managed_static_input_idxs,
            )
        ):
            # this should error
            error_msg = log_data_ptr_mismatch(
                self.wrapped_function.placeholders,
                new_inputs,
                self.static_input_data_ptrs,
                self.non_managed_static_input_idxs,
                CheckInvariantStatus.StaticInputIdxMismatch,
            )
            torch._check(False, lambda: error_msg)

    def run_first_inputs(self, new_inputs: list[InputType]) -> OutputType:
        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_before_invocation()

        # graph is already invoked in the __init__
        # inputs are copied over in _allocate_recording_inputs and subsequently cleared
        assert len(new_inputs) == 0
        outputs = self.recording_outputs
        self.recording_outputs = None
        assert outputs is not None
        return outputs

    def run(self, new_inputs: list[InputType]) -> OutputType:
        self.check_static_inputs_are_stable(new_inputs)

        self._copy_inputs_and_remove_from_src(self.reconstructed_inputs, new_inputs)

        self.run_graph()

        outputs = self.reconstruct_outputs()
        new_inputs.clear()

        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_after_invocation()

        if config.triton.force_cudagraph_sync:
            torch.cuda.synchronize()

        # Reset this to run the check in the future
        self.static_inputs_stable = False

        return outputs

    def reconstruct_outputs(self) -> OutputType:
        "Reconstruct output tensors according to their saved metadata and alias information"

        # Cached tensors will not yet be set on the first execution
        # They are also cleared in checkpointing, so if we checkpoint this node
        # and then execute it again we will need to repopulate cached tensors
        if not self.cached_tensor_outputs:
            self._initialize_cached_tensors()

        outputs: OutputType = []

        for i, (storage_info, metadata) in enumerate(
            zip(self.output_storage_alias, self.outputs_metadata)
        ):
            if not isinstance(metadata, dict):  # tensor metadata
                assert isinstance(metadata, (int, type(None)))
                outputs.append(metadata)
                continue

            cached_t = self.cached_tensor_outputs[i]
            if cached_t is not None:
                # this output represents a fresh allocated tensor.
                # We return the same TensorImpl from run to run to avoid overhead.
                # autograd.Function will reset the Autograd meta of output tensors
                # as part of aot_autograd, but _backward_hooks are stored on tensors separately,
                # so we need to manually reset hooks.
                if cached_t._backward_hooks is not None:
                    cached_t._backward_hooks = None

                # No need to update weakrefs, already correctly initialized
                outputs.append(cached_t)
                continue

            static_t = self.static_output_tensors[i]
            if static_t is not None:
                assert self.outputs_weakrefs[i] is None
                outputs.append(static_t)
                continue

            storage = self.prepare_alias_info_for_tensor_construction(
                storage_info, metadata
            )

            if isinstance(storage, UntypedStorage) or storage is None:
                out = self._reconstruct_from_tensor_metadata(metadata, storage)
            else:
                assert isinstance(storage, int)
                out = self._reconstruct_from_tensor_metadata(
                    metadata, cast(torch.Tensor, outputs[storage]).untyped_storage()
                )

            outputs.append(out)
            w = self.outputs_weakrefs[i]
            assert w is not None
            w.swap_weakref(out.untyped_storage()._weak_ref())

        return outputs

    def prepare_alias_info_for_tensor_construction(
        self,
        out_alias_info: Optional[OutputAliasInfo],
        metadata: Union[dict[str, Any], int, None],
    ) -> Union[UntypedStorage, None, int]:
        if (
            isinstance(metadata, (int, type(None)))
            or out_alias_info is UnaliasedStorage
        ):
            return None

        if isinstance(out_alias_info, AliasesPriorGraphOutput):
            depth, existing_output_index = out_alias_info.index
            ref = self.path_weakrefs[depth][existing_output_index]
            assert ref is not None
            return torch.UntypedStorage._new_with_weak_ptr(ref())

        assert isinstance(out_alias_info, AliasesNewOutput)
        return out_alias_info.index

    def prepare_storages_for_construction(
        self,
    ) -> list[Union[UntypedStorage, None, int]]:
        output_storages = []
        for output_storage_alias, metadata in zip(
            self.output_storage_alias, self.outputs_metadata
        ):
            output_storages.append(
                self.prepare_alias_info_for_tensor_construction(
                    output_storage_alias, metadata
                )
            )

        return output_storages

    def run_graph(self) -> None:
        assert self.graph is not None
        self.graph.replay()

    def all_outputs_are_dead(self) -> bool:
        "All outputs of the path from this node to its root are dead"
        for depth, output_index in self.live_indices_after_graph:
            if is_live(self.path_weakrefs[depth][output_index]):
                return False
        return True

    def _record(self, model: ModelType, inputs: list[InputType]) -> OutputType:
        "Record the model"
        assert self.graph is not None

        def static_input_iter() -> Generator[torch.Tensor, None, None]:
            for i in self.wrapped_function.static_input_idxs:
                _inp = inputs[i]
                if isinstance(
                    _inp, torch.Tensor
                ) and not self._is_cuda_graph_recorded_tensor(_inp):
                    yield _inp

        # see: output_is_alias_of_persistent_static_inputs above
        static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper] = {
            inp.untyped_storage().data_ptr(): StorageWeakRefWrapper(inp)
            for inp in itertools.chain(
                static_input_iter(), self.wrapped_function.constants
            )
        }

        if config.triton.slow_path_cudagraph_asserts:
            # need to use parent live weakrefs because live_indices isn't set yet
            memory = (
                [] if self.parent is None else list(self.parent.path_live_weakrefs())
            )
            memory += [
                StorageWeakRefWrapper(elem)
                for i, elem in enumerate(inputs)
                if isinstance(elem, torch.Tensor)
                and i not in self.wrapped_function.static_input_idxs
                and elem.untyped_storage().data_ptr() != 0
            ]
            check_memory_pool(self.device, self.cuda_graphs_pool, memory)

        with (
            preserve_rng_state(),
            torch.cuda.device(self.device),
            clear_cublas_manager(),
            torch.cuda.graph(
                self.graph,
                stream=self.stream,
                pool=self.cuda_graphs_pool,
                capture_error_mode="thread_local",
  
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `cudagraph_trees.py_docs.md_docs.md`
- **Keyword Index**: `cudagraph_trees.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
