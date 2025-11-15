# Documentation: `docs/torch/distributed/optim/zero_redundancy_optimizer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/optim/zero_redundancy_optimizer.py_docs.md`
- **Size**: 54,433 bytes (53.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/optim/zero_redundancy_optimizer.py`

## File Metadata

- **Path**: `torch/distributed/optim/zero_redundancy_optimizer.py`
- **Size**: 72,897 bytes (71.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

r"""Zero Redundancy Optimizer."""

import collections
import copy
import enum
import inspect
import io
import logging
from collections.abc import Callable
from itertools import chain
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer


__all__ = ["ZeroRedundancyOptimizer"]


logger = logging.getLogger(__name__)


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if possible.

    Non-tensor values are passed as-is in the result.

    .. note::
        These are all copies, so if there are two objects that reference
        the same object, then after this call, there will be two different objects
        referenced on the device.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = [
            _recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            for val in value
        ]
        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )
            for key, val in value.items()
        }

    return value


def _is_trainable(param: torch.Tensor) -> bool:
    r"""Return if a parameter is trainable, where trainability is equivalent to requiring a gradient."""
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    device: torch.device = torch.device("cpu"),
) -> Any:
    r"""
    Broadcasts an object to the given group.

    It will be sending the object if called from the source rank and receiving
    the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.group.WORLD``).
        device (``torch.device``, optional): device to send from or receive
            to (default: ``torch.device("cpu")``).

    Returns:
        The broadcasted object.
    """
    if dist.get_rank() == src_rank:
        # Send the object
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(device)
        data_send_tensor = torch.ByteTensor(data).to(device)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Receive the object
        length_tensor = torch.LongTensor([0]).to(device)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty(
            [int(length_tensor.item())], dtype=torch.uint8, device=device
        )
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device, weights_only=False)
    return obj


class _ZeROJoinHook(JoinHook):
    def __init__(self, zero):
        assert isinstance(zero, ZeroRedundancyOptimizer), (
            "ZeRO join hook requires passing in a ZeroRedundancyOptimizer "
            "instance as the state"
        )
        self.zero = zero
        super().__init__()

    def main_hook(self):
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()


class _DDPBucketAssignment:
    r"""
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[torch.Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device (torch.device): device on which the parameters are stored.
        tensor (torch.Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """

    def __init__(
        self,
        bucket_index: int,
        parameters: list[torch.Tensor],
        offset: int,
    ):
        self.bucket_index = bucket_index
        self.parameters = parameters
        self.offset = offset
        if len(self.parameters) == 0:
            raise ValueError("Empty bucket assignment")
        # DDP guarantees all parameters in the bucket have the same device
        # pyrefly: ignore [read-only]
        self.device: torch.device = self.parameters[0].device
        self.tensor: Optional[torch.Tensor] = None


class _OverlapStatus(enum.IntEnum):
    r"""
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """

    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2


class _OverlapInfo:
    r"""
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity following
            a threshold given by the total parameter size divided by the world
            size; if ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank);
            this should be set to the value passed into the hook constructor.
        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        num_bucket_assignments (int): total number of bucket assignments across
            all ranks; this is equal to the number of
            :class:`DistributedDataParallel` gradient buckets if
            ``shard_buckets=False`` and possibly greater otherwise.
        total_size (int, optional): total size of all buckets (i.e. sum of
            ``param.numel()`` for all ``param`` across all buckets) if
            ``shard_buckets=True``; otherwise, ``None``.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """

    def __init__(self, world_size) -> None:
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        self.shard_buckets: bool = False

        # Modified per bucket reconstruction
        self.params_per_bucket: list[list[torch.Tensor]] = []
        self.params_per_rank: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
        self.offsets: dict[int, int] = {}
        # Group Ranks
        self.assigned_ranks_per_bucket: list[set[int]] = []
        self.num_bucket_assignments: int = 0
        self.total_size: Optional[int] = None

        # Modified per iteration
        self.broadcast_handles: list[Any] = []
        self.bucket_indices_seen: list[int] = []
        # Used by `hook_with_zero_step()`
        self.bucket_index_to_future: dict[int, torch.futures.Future] = {}
        self.bucket_index_to_bucket: dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        r"""
        Wait for all parameter broadcasts.

        This function should be called once all broadcasts have been scheduled,
        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``
        in preparation for the next iteration.
        """
        assert len(self.broadcast_handles) == self.num_bucket_assignments, (
            f"Missing at least one broadcast handle on rank {dist.get_rank()}"
        )
        _ = [x.wait() for x in self.broadcast_handles]
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        r"""
        Clear the data structures that are modified per-iteration.

        This function should be called at the end of an iteration.
        """
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()


class ZeroRedundancyOptimizer(Optimizer, Joinable):
    r"""
    Wrap an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>` and shards its states across ranks in the group.

    The sharing is done as described by `ZeRO <https://arxiv.org/abs/1910.02054>`_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    Example::

        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``. To adjust for this, one option
        is to prepend dummy inputs.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.
    """

    def __init__(
        self,
        params,
        optimizer_class: type[Optimizer],
        process_group: Optional[Any] = None,
        parameters_as_bucket_view: bool = False,
        overlap_with_ddp: bool = False,
        **defaults: Any,
    ):
        r"""Init."""
        # Perform type and assumption checks on the input parameters
        params = self._verify_and_init_params(params)
        self._verify_same_dense_param_type()

        # NOTE: The parent constructor uses `add_param_group()` which is
        # partially overloaded in ZeroRedundancyOptimizer, so we use the
        # `initialized` flag to dissociate the behaviour of `add_param_group()`
        # between the parent and child.
        self.initialized = False

        Optimizer.__init__(self, params, defaults)
        Joinable.__init__(self)
        # Now, all parameters are held in both `self._all_params` and
        # `self.param_groups`

        # Internal data structures (`_cache` indicates lazily evaluated)
        self._param_to_rank_cache: dict[torch.Tensor, int] = {}
        self._param_to_index_cache: dict[torch.Tensor, int] = {}
        self._partition_parameters_cache: list[list[dict]] = []
        self._index_to_param_cache: list[torch.Tensor] = []
        self._device_to_params_per_rank_cache: dict[
            torch.device, list[list[torch.Tensor]]
        ] = {}
        self._bucket_assignments_per_rank_cache: list[
            dict[int, _DDPBucketAssignment]
        ] = []
        self._is_trainable_mask = self._get_is_trainable_mask()

        # Default device for collective communication and buckets
        self._default_device = self._all_params[0].device

        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        self.world_size: int = dist.get_world_size(self.process_group)
        self.rank: int = dist.get_rank(self.process_group)
        self.global_rank: int = dist.distributed_c10d.get_global_rank(
            # pyrefly: ignore [bad-argument-type]
            self.process_group,
            self.rank,
        )

        self._overlap_with_ddp: bool = overlap_with_ddp
        self._optim_defaults = defaults
        self._optim_constructor = self._get_optimizer_constructor(optimizer_class)

        # If `overlap_with_ddp=True`, local optimizer initialization is delayed
        # to run time after the necessary information has been collected
        if not overlap_with_ddp:
            self._init_local_optimizer()
        else:
            self._overlap_info: _OverlapInfo = _OverlapInfo(self.world_size)
            if parameters_as_bucket_view:
                logger.warning(
                    "`parameters_as_bucket_view=True` will be ignored since "
                    "`overlap_with_ddp=True`; instead, a different bucketing "
                    "strategy will be used"
                )

        # `self._buckets` is used if `parameters_as_bucket_view=True`, in
        # which case parameter data is flattened into contiguous bucket tensors
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._buckets: list[list[torch.Tensor]] = []
        self._build_param_buckets()

        # Optional consolidated optimizer state, only populated if this rank
        # is the target in `consolidate_state_dict()`
        self._all_state_dicts: list[dict[str, Any]] = []

        self.initialized = True

    def _clear_cache(self) -> None:
        r"""Clear the cached data structures giving partition information."""
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()
        self._device_to_params_per_rank_cache.clear()
        self._bucket_assignments_per_rank_cache.clear()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        r"""
        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
        if self.initialized and self._overlap_with_ddp:
            raise RuntimeError(
                "ZeroRedundancyOptimizer with `overlap_with_ddp=True` only "
                "supports a single parameter group"
            )

        super().add_param_group(param_group)
        # NOTE: The rest of the method assumes that the call to the parent's
        # `add_param_group()` appends the new parameter group and preserves
        # the previous parameter-group ordering

        if self.initialized:
            # Force a re-partitioning of the parameters
            self._clear_cache()
            param_groups = self._partition_parameters()[self.rank]
            # NOTE: All parameters in the old parameter groups should be
            # assigned to the same ranks so that the local optimizers do not
            # need to be reinitialized

            # Add the parameters assigned to this rank from the new parameter
            # group to the local optimizer, if any
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.

        .. warning:: This needs to be called on all ranks.
        """
        self._check_overlap_initialized()

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Pull the sharded state from all ranks and store them in rank order
        empty_messenger = torch.tensor(
            [0], dtype=torch.uint8, device=self._default_device
        )

        # NOTE: We wastefully use `broadcast()` (e.g. instead of `gather()`)
        # due to compatibility issues with NCCL backend; a possible follow-up
        # is to move all sharded state management to RPC RRef
        self._all_state_dicts = []
        for rank in range(self.world_size):
            global_rank = dist.distributed_c10d.get_global_rank(
                # pyrefly: ignore [bad-argument-type]
                self.process_group,
                rank,
            )
            if self.rank == to:
                # Consolidate all local `state_dict`s on this rank, storing on
                # CPU to save GPU memory
                if rank == self.rank:
                    # Directly append own optimizer state
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            self.optim.state_dict(),
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
                else:
                    # Receive the optimizer state from the source rank
                    local_state_dict = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            local_state_dict,
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
            else:
                if rank == self.rank:
                    # Send the optimizer state to the target rank
                    _ = _broadcast_object(
                        self.optim.state_dict(),
                        src_rank=self.global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                elif rank != to:
                    # Discard the received object; `broadcast()` is used for
                    # compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )

    def _verify_params_per_rank(
        self,
        params_per_rank: list[list[torch.Tensor]],
    ) -> None:
        r"""
        Verify ``params_per_rank`` for :meth:`_partition_parameters`.

        The verification is done by checking that ``params_per_rank`` has length equal
        to the world size and that it does not contain any parameters not passed into the
        :class:`ZeroRedundancyOptimizer` constructor.

        The parameters in ``params_per_rank`` being a strict subset of those
        passed into the constructor is valid since some parameters may be
        frozen.

        Raises:
            ValueError: if ``params_per_rank`` does not have length equal to
                the world size or if it contains a parameter that was not
                passed into the :class:`ZeroRedundancyOptimizer` constructor.
        """
        if len(params_per_rank) != self.world_size:
            raise ValueError(
                "`params_per_rank` must have length equal to the world size"
            )
        all_params_set = set(self._all_params)
        for params in params_per_rank:
            for param in params:
                if param not in all_params_set:
                    raise ValueError(
                        "Passing a new parameter in `params_per_rank` that "
                        "was not passed into the ZeroRedundancyOptimizer "
                        "constructor"
                    )

    def _partition_param_group(
        self, param_group: dict[str, Any], params_per_rank: list[list[torch.Tensor]]
    ) -> None:
        r"""
        Partition the parameter group ``param_group`` according to ``params_per_rank``.

        The partition will modify the ``self._partition_parameters_cache``. This method should
        only be used as a subroutine for :meth:`_partition_parameters`.

        Arguments:
            param_group (dict[str, Any]): a parameter group as normally defined
                in an optimizer state.
            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
                length world size containing :class:`list` s of parameters to
                assign to each rank.
        """
        for rank, params in enumerate(params_per_rank):
            rank_param_group = copy.copy(param_group)
            rank_param_group["params"] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(
        self,
        params_per_rank: Optional[list[list[torch.Tensor]]] = None,
    ) -> list[list[dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank (list[list[torch.Tensor]], optional): a
                :class:`list` of length world size containing :class:`list` s
                of parameters to assign to each rank; this provides a way to
                specify a partition manually.
                If ``None``, the parameters are partitioned according to an
                internal algorithm.
                (default: ``None``)

        Returns:
            A :class:`list` where each element of the list contains the
            ``param_groups`` for a rank (which itself is a :class:`list` of
            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank
            stores the ``param_groups`` for all ranks for the collective
            communication in :meth:`step`.

        Raises:
            ValueError: see :meth:`_validate_params_per_rank`.
            RuntimeError: if ``params_per_rank`` is not ``None`` and this
                :class:`ZeroRedundancyOptimizer` instance is using more than
                one parameter group.
        """
        if params_per_rank is None:
            # Partition the parameters optimizing for uniformity
            if len(self._partition_parameters_cache) == 0:
                self._partition_parameters_cache = [[] for _ in range(self.world_size)]
                sizes = [0] * self.world_size
                for param_group in self.param_groups:
                    param_group_params_per_rank: list[list] = [
                        [] for _ in range(self.world_size)
                    ]
                    # Sort the parameters by size (largest first)
                    params_sorted = sorted(
                        param_group["params"], key=lambda t: t.numel(), reverse=True
                    )
                    for param in params_sorted:
                        # Greedily add the parameter to rank with smallest size so far
                        rank = self._get_min_index(sizes)
                        param_group_params_per_rank[rank].append(param)
                        sizes[rank] += param.numel()
                    # Apply the constructed partition of the parameter group
                    self._partition_param_group(
                        param_group, param_group_params_per_rank
                    )

            return self._partition_parameters_cache

        # Partition the parameters according to `params_per_rank`
        assert len(self._partition_parameters_cache) == 0, (
            "Specifying `params_per_rank` should only be done when the "
            "parameters have not been partitioned yet"
        )
        if len(self.param_groups) != 1:
            raise RuntimeError(
                "Specifying `params_per_rank` only supports a single parameter group"
            )
        self._verify_params_per_rank(params_per_rank)
        self._partition_parameters_cache = [[] for _ in range(self.world_size)]

        # Apply the passed-in partition of the parameter group
        param_group = self.param_groups[0]
        self._partition_param_group(param_group, params_per_rank)

        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> dict[torch.Tensor, int]:
        r""":class:`dict` mapping parameters to their assigned data parallel rank in the partition."""
        if len(self._param_to_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> dict[torch.Tensor, int]:
        r"""
        :class:`dict` mapping parameters to their indices in the global optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                p: i
                for i, p in enumerate(
                    chain.from_iterable(g["params"] for g in self.param_groups)
                )
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> list[torch.Tensor]:
        r"""List mapping parameter indices in the global optimizer scheme to the actual params."""
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(
                chain.from_iterable(g["params"] for g in self.param_groups)
            )
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        r"""
        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
        assert not self._overlap_with_ddp, (
            "`_broadcast_params_from_rank()` should not be used if "
            "`overlap_with_ddp=True`; instead, the broadcasting should "
            "happen in the DDP communication hook"
        )
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = dist.distributed_c10d.get_global_rank(
                    # pyrefly: ignore [bad-argument-type]
                    self.process_group,
                    rank,
                )
                handles.append(
                    dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                )
        else:
            param_groups = self._partition_parameters()[rank]
            global_rank = dist.distributed_c10d.get_global_rank(
                # pyrefly: ignore [bad-argument-type]
                self.process_group,
                rank,
            )
            for param_group in param_groups:
                handles.extend(
                    dist.broadcast(
                        tensor=param.data,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    for param in param_group["params"]
                )
        return handles

    def _sync_params(self):
        r"""
        Sync all parameter shards across the ranks.

        This rank sends its shard of the parameters to all other ranks and
        receives a shard from each other rank. This is done using
        ``broadcast()``. Parameters are sent bucket-by-bucket if
        ``parameters_as_bucket_view=True``and sent parameter-by-parameter
        otherwise.
        """
        handles = []
        for rank in range(self.world_size):
            handles.extend(self._broadcast_params_from_rank(rank))
        _ = [x.wait() for x in handles]

    @property
    def _device_to_params_per_rank(
        self,
    ) -> dict[torch.device, list[list[torch.Tensor]]]:
        r"""
        Return device parameters assigned per rank.

        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
        assert self.parameters_as_bucket_view, (
            "`_device_to_params_per_rank` should only be used if "
            "`parameters_as_bucket_view=True`"
        )
        if len(self._device_to_params_per_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        device = param.device
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [
                                [] for _ in range(self.world_size)
                            ]
                        self._device_to_params_per_rank_cache[device][rank].append(
                            param
                        )
        return self._device_to_params_per_rank_cache

    def _get_min_index(
        self,
        values: list[int],
        disallowed_indices: Optional[set[int]] = None,
    ) -> int:
        r"""
        Return ``values.index(min(values))``, except only uses one pass.

        It also excludes any indices in ``disallowed_indices`` if provided.

        Arguments:
            values: (List[int]): :class:`list` of values.
            disallowed_indices (Optional[set[int]]): indices that are
                disallowed from being the returned min index.
        """
        min_index = -1
        min_value = float("inf")
        for i, value in enumerate(values):
            if disallowed_indices and i in disallowed_indices:
                continue
            if value < min_value:
                min_value = value
                min_index = i
        assert min_index >= 0, "All indices are disallowed"
        return min_index

    def _assign_bucket_subset_to_rank(
        self,
        bucket_index: int,
        bucket_params: list[torch.Tensor],
        bucket_offset: int,
        assigned_rank: int,
        assigned_ranks_per_bucket: list[set[int]],
    ) -> None:
        r"""
        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.

        The model parameters given by ``bucket_params`` represents a (possibly non-strict)
        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                gradient bucket.
            bucket_params (List[torch.Tensor]): subset of the parameters
                corresponding to the bucket to assign.
            bucket_offset (int): offset giving the index of the first element
                in ``bucket_params`` in the bucket's full parameter list.
            assigned_rank (int): group rank to assign to.
            assigned_ranks_per_bucket (list[set[int]]): :class:`set` of group ranks
                assigned to each bucket.
        """
        overlap_info = self._overlap_info
        if len(bucket_params) == 0:
            raise ValueError("Empty bucket assignment")
        params_per_rank = overlap_info.params_per_rank
        offsets = overlap_info.offsets

        self._bucket_assignments_per_rank_cache[assigned_rank][bucket_index] = (
            _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)
        )
        if self.global_rank == assigned_rank:
            offsets[bucket_index] = len(params_per_rank[assigned_rank])
        params_per_rank[assigned_rank].extend(bucket_params)
        assigned_ranks_per_bucket[bucket_index].add(assigned_rank)
        self._overlap_info.num_bucket_assignments += 1

    @property
    def _bucket_assignments_per_rank(self) -> list[dict[int, _DDPBucketAssignment]]:
        r"""
        Return DDP bucket parameters assigned per rank.

        :class:`list` of length world size consisting of :class:`dict` s
        mapping bucket indices to :class:`_DDPBucketAssignment` s for each
        rank.
        """
        assert self._overlap_with_ddp, (
            "`_bucket_assignments_per_rank` only be used if `overlap_with_ddp=True`"
        )
        if len(self._bucket_assignments_per_rank_cache) > 0:
            return self._bucket_assignments_per_rank_cache

        overlap_info = self._overlap_info
        assert overlap_info.status == _OverlapStatus.INITIALIZED

        self._bucket_assignments_per_rank_cache = [{} for _ in range(self.world_size)]
        params_per_bucket = overlap_info.params_per_bucket

        if overlap_info.shard_buckets:
            # Define the assignment threshold to approximate uniformity
            assert overlap_info.total_size is not None, "`total_size` was not computed"
            threshold = overlap_info.total_size / self.world_size  # type: ignore[operator]
            size_per_rank = [0 for _ in range(self.world_size)]

        num_buckets = len(params_per_bucket)
        overlap_info.assigned_ranks_per_bucket = [set() for _ in range(num_buckets)]
        assigned_ranks_per_bucket = overlap_info.assigned_ranks_per_bucket
        if not overlap_info.shard_buckets:
            # Assign each DDP bucket entirely to a single rank
            for bucket_index, bucket_params in enumerate(params_per_bucket):
                assert len(bucket_params) > 0, "Empty bucket"
                assigned_rank = self._get_assigned_rank(bucket_index)
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params,
                    0,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
        else:
            # Assign each DDP bucket to possibly multiple ranks
            # Specifically, sort the DDP buckets by increasing size, and for
            # each bucket, iteratively assign the maximal unassigned subset
            # with size less than `threshold` to the rank with the least total
            # size so far -- each such assignment is represented by a
            # `_DDPBucketAssignment` instance and only contains parameters from
            # a single DDP bucket
            params_per_bucket_enum = sorted(
                enumerate(params_per_bucket), key=lambda x: sum(p.numel() for p in x[1])
            )
            for bucket_index, bucket_params in params_per_bucket_enum:
                assert len(bucket_params) > 0, "Empty bucket"
                bucket_offset = 0
                assignment_size = 0
                for param_index, param in enumerate(bucket_params):
                    param_numel = param.numel()
                    if (
                        # pyrefly: ignore [unbound-name]
                        assignment_size + param_numel >= threshold
                        and param_index > bucket_offset
                    ):
                        assigned_rank = self._get_min_index(
                            # pyrefly: ignore [unbound-name]
                            size_per_rank,
                            assigned_ranks_per_bucket[bucket_index],
                        )
                        # Include up to but not including the parameter that
                        # exceeded the threshold
                        self._assign_bucket_subset_to_rank(
                            bucket_index,
                            bucket_params[bucket_offset:param_index],
                            bucket_offset,
                            assigned_rank,
                            assigned_ranks_per_bucket,
                        )
                        # pyrefly: ignore [unbound-name]
                        size_per_rank[assigned_rank] += assignment_size
                        bucket_offset = param_index
                        assignment_size = 0
                    assignment_size += param_numel
                # Assign the remainder of the bucket so that no assignment
                # spans across two buckets
                assigned_rank = self._get_min_index(
                    # pyrefly: ignore [unbound-name]
                    size_per_rank,
                    assigned_ranks_per_bucket[bucket_index],
                )
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params[bucket_offset:],
                    bucket_offset,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
                # pyrefly: ignore [unbound-name]
                size_per_rank[assigned_rank] += assignment_size

        return self._bucket_assignments_per_rank_cache

    def _local_step(
        self,
        gradients: Optional[list[Optional[torch.Tensor]]] = None,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        r"""
        Perform a single optimizer step without syncing parameters across ranks.

        Arguments:
            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`
                of length equal to the number of parameters assigned to this
                rank containing gradient tensors or ``None`` as its elements;
                a ``None`` in the :class:`list` indicates that the
                corresponding parameter should not be updated.
                If the argument itself is ``None``, then all parameters are
                updated, and the gradients are assumed to be already populated.
                (default: ``None``)
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers and should be
                ``None`` if ``gradients`` is not ``None``; (default: ``None``)
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. warning::
            The argument ``gradients`` should only be specified (i.e. not
            ``None``) if ``overlap_with_ddp=True``, in which case
            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.
        """
        Join.notify_join_context(self)
        # Check if the model trainability has changed
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            if self._overlap_with_ddp:
                raise RuntimeError(
                    "ZeroRedundancyOptimizer with `overlap_with_ddp=True` "
                    "does not support changing parameter trainability at run "
                    "time"
                )
            logger.warning(
                "ZeroRedundancyOptimizer detected that the trainable "
                "parameters changed; rebuilding the parameter buckets if "
                "enabled"
            )
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only
        if gradients is None:
            loss = (
                self.optim.step(**kwargs)
                if closure is None
                else self.optim.step(closure=closure, **kwargs)
            )
        else:
            assert self._overlap_with_ddp, (
                "Specifying `gradients` should not "
                "be used when `overlap_with_ddp=False`"
            )
            assert closure is None, (
                "`closure` is not supported when using a local functional optimizer"
            )
            loss = self.optim.step(gradients=gradients)

        # Sync any updated attributes in the local optimizer to the exposed
        # `param_groups`
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    # pyrefly: ignore [bad-override]
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        r"""
        Perform a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note:: Any extra parameters are passed to the base optimizer as-is.
        """
        if self._overlap_with_ddp:
            logger.warning(
                "`step()` should not be included in the training loop when "
                "`overlap_with_ddp=True`"
            )
            return None

        # Perform the local optimizer step
        loss = self._local_step(closure=closure, **kwargs)

        # Sync all of the updated parameter shards across the ranks
        self._sync_params()

        return loss

    def join_hook(self, **kwargs):
        r"""
        Return the ZeRO join hook.

        It enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs`
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/distributed/optim`):

- [`apply_optimizer_in_backward.py_docs.md_docs.md`](./apply_optimizer_in_backward.py_docs.md_docs.md)
- [`functional_rprop.py_kw.md_docs.md`](./functional_rprop.py_kw.md_docs.md)
- [`functional_adagrad.py_docs.md_docs.md`](./functional_adagrad.py_docs.md_docs.md)
- [`_deprecation_warning.py_kw.md_docs.md`](./_deprecation_warning.py_kw.md_docs.md)
- [`zero_redundancy_optimizer.py_kw.md_docs.md`](./zero_redundancy_optimizer.py_kw.md_docs.md)
- [`functional_rmsprop.py_docs.md_docs.md`](./functional_rmsprop.py_docs.md_docs.md)
- [`functional_rprop.py_docs.md_docs.md`](./functional_rprop.py_docs.md_docs.md)
- [`named_optimizer.py_docs.md_docs.md`](./named_optimizer.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `zero_redundancy_optimizer.py_docs.md_docs.md`
- **Keyword Index**: `zero_redundancy_optimizer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
