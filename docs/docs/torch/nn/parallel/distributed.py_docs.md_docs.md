# Documentation: `docs/torch/nn/parallel/distributed.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/parallel/distributed.py_docs.md`
- **Size**: 54,078 bytes (52.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/parallel/distributed.py`

## File Metadata

- **Path**: `torch/nn/parallel/distributed.py`
- **Size**: 109,859 bytes (107.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._utils import _get_device_index
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from torch.utils._pytree import tree_flatten, tree_unflatten


RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _rank_not_in_group,
        ReduceOp,
    )
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if dist.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["DistributedDataParallel"]

logger = logging.getLogger(__name__)


@dataclass
class _MixedPrecision:
    """
    This configures DDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs (when ``cast_forward_inputs`` is set to
            ``True``), and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``_MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and DDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise. For example, ``_MixedPrecision(param_dtype=torch.float16)``
        would result in communication occurring in fp16.
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    buffer_dtype: torch.dtype | None = None
    # TODO (rohan-varma): keep_low_precision_grads: bool = False
    # TODO (rohan-varma): APIs to allow users to run batchnorm and layernorm
    # in full precision. For DDP, this can be implemented by not performing the
    # parameter cast for BN and LN units.


def _cast_buffers(mixed_precision_config, root_module):
    """Casts buffers to the given ``buffer_dtype``."""
    for buf in root_module.buffers():
        if hasattr(buf, "_ddp_ignored") and buf._ddp_ignored:
            continue

        buf.data = buf.to(dtype=mixed_precision_config.buffer_dtype)


def _setup_mixed_precision_params(mixed_precision_config, root_module):
    """Create and free storage for the mixed precision parameters."""
    for param in root_module.parameters():
        # Do not setup mixed precision for DDP ignored parameters.
        if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
            continue

        if not hasattr(param, "_mp_param"):
            param._mp_param = torch.zeros_like(
                param,
                device=param.device,
                dtype=mixed_precision_config.param_dtype,
                requires_grad=param.requires_grad,
            )
            _free_storage(param._mp_param)
            # _fp_param will point to the full precision param so it can be switched
            # back to at the end of forward / backward.
            param._fp_param = param.data


def _tree_flatten_with_rref(output):
    output_is_rref = RPC_AVAILABLE and isinstance(output, RRef)
    if output_is_rref:
        output_tensor_list, treespec = tree_flatten(output.local_value())
    else:
        output_tensor_list, treespec = tree_flatten(output)
    # Need to return flattened tensors, spec to re-pack them, as well
    # as if the return type was actually an RRef to reconstruct.
    return output_tensor_list, treespec, output_is_rref


def _tree_unflatten_with_rref(output, treespec, output_is_rref):
    output = tree_unflatten(output, treespec)
    if output_is_rref:
        output = RRef(output)
    return output


def _find_tensors(obj):
    r"""Recursively find all tensors contained in the specified object."""
    if RPC_AVAILABLE and isinstance(obj, RRef):
        # If the current node is the owner of the RRef, unwrap it and try to
        # find Tensors.
        # TODO: Expand to remote RRefs.
        if obj.is_owner():
            return _find_tensors(obj.local_value())
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain.from_iterable(map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain.from_iterable(map(_find_tensors, obj.values()))
    if is_dataclass(obj):
        return itertools.chain.from_iterable(
            map(_find_tensors, (getattr(obj, f.name) for f in fields(obj)))
        )

    return []


def _dump_DDP_relevant_env_vars():
    relevant_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        # More NCCL env vars:
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ.get(var, "N/A")
        formatted_output += f"env:{var}={value}\n"
    print(formatted_output)


class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation


# Add a DDPSink to run various functions when backwards starts, such as
# queueing call back of out-most backward/graph task,
# this helps call back is fired after all gradients' calculation
# is completed.
class _DDPSink(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, ddp_weakref, *inputs):
        # set_materialize_grads(False) will ensure that None gradients stay as
        # None and are not filled with zeros.
        ctx.set_materialize_grads(False)
        ctx.ddp_weakref = ddp_weakref
        ret = inputs
        if ddp_weakref()._ddp_sink_clone:
            ret = tuple(
                inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
            )
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Enqueue delay allreduce for static graph training on the first
        # iteration.
        ddp_weakref = ctx.ddp_weakref()
        reducer = ddp_weakref.reducer
        static_graph = ddp_weakref.static_graph
        delay_ar_enqueued = (
            static_graph and ddp_weakref._static_graph_delay_allreduce_enqueued
        )
        if static_graph and not delay_ar_enqueued:
            Variable._execution_engine.queue_callback(  # type: ignore[call-arg,misc]
                reducer._delay_all_reduce
            )
            ddp_weakref._static_graph_delay_allreduce_enqueued = True

        return (None, *grad_outputs)


class _DDPJoinHook(JoinHook):
    def __init__(self, ddp, divide_by_initial_world_size):
        """Set config variables for internal usage."""
        assert isinstance(ddp, DistributedDataParallel), (
            "DDP join hook requires passing in a DistributedDataParallel "
            "instance as the state"
        )
        assert ddp.logger is not None
        ddp.logger._set_uneven_input_join()
        self.ddp = ddp
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self):
        """Shadow the DDP collective communication operations in the forward and backward passes."""
        ddp = self.ddp
        # Buckets are rebuilt only once during a training period
        ddp.reducer._rebuild_buckets()

        # Schedule a broadcast if we are syncing module buffers in the
        # forward pass
        # TODO: make DDP uneven inputs context manager support buffer
        # comm hook (https://github.com/pytorch/pytorch/issues/65436)
        ddp._check_and_sync_module_buffers()

        # Check if need to sync in the backward pass
        should_sync_backwards = ddp._check_global_requires_backward_grad_sync(
            is_joined_rank=True
        )
        # Forward parameter sync is disabled in the next iteration if we
        # are skipping gradient sync this iteration, so set
        # `require_forward_param_sync` accordingly
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return

        # Schedule one allreduce per gradient bucket to match the backward
        # pass allreduce
        ddp._match_all_reduce_for_bwd_pass()

        # Check if we need to allreduce locally unused parameters
        if ddp.find_unused_parameters:
            ddp._match_unused_params_allreduce()

        # Rebuilt parameters are pushed only once during a training period
        ddp.reducer._push_all_rebuilt_params()

    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""
        self.ddp._sync_final_model(is_last_joiner)


class DistributedDataParallel(Module, Joinable):
    r"""Implement distributed data parallelism based on ``torch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-ddp-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    To use ``DistributedDataParallel`` on a host with N GPUs, you should spawn
    up ``N`` processes, ensuring that each process exclusively works on a single
    GPU from 0 to N-1. This can be done by either setting
    ``CUDA_VISIBLE_DEVICES`` for every process or by calling the following API for GPUs,

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.cuda.set_device(i)

    or calling the unified API for :ref:`accelerator<accelerators>`,

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.accelerator.set_device_index(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> if torch.accelerator.is_available():
        >>>     device_type = torch.accelerator.current_accelerator().type
        >>>     vendor_backend = torch.distributed.get_default_backend_for_device(device_type)
        >>>
        >>> torch.distributed.init_process_group(
        >>>     backend=vendor_backend, world_size=N, init_method='...'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    Or you can use the latest API for initialization:

        >>> torch.distributed.init_process_group(device_id=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note::
        ``DistributedDataParallel`` can be used in conjunction with
        :class:`torch.distributed.optim.ZeroRedundancyOptimizer` to reduce
        per-rank optimizer states memory footprint. Please refer to
        `ZeroRedundancyOptimizer recipe <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`__
        for more details.

    .. note:: ``nccl`` backend is currently the fastest and highly recommended
        backend when using GPUs. This applies to both single-node and
        multi-node distributed training.

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of ``fp16`` and ``fp32``, the gradient reduction on these
        mixed types of parameters will just work fine.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
        gradient will be ``M`` times smaller when compared to the same model
        trained on a single node with ``batch=M*N`` if the loss is summed (NOT
        averaged as usual) across instances in a batch (because the gradients
        between different nodes are averaged). You should take this into
        consideration when you want to obtain a mathematically equivalent
        training process compared to the local training counterpart. But in most
        cases, you can just treat a DistributedDataParallel wrapped model, a
        DataParallel wrapped model and an ordinary model on a single GPU as the
        same (E.g. using the same learning rate for equivalent batch size).

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    .. note::
        If you are using DistributedDataParallel in conjunction with the
        :ref:`distributed-rpc-framework`, you should always use
        :meth:`torch.distributed.autograd.backward` to compute gradients and
        :class:`torch.distributed.optim.DistributedOptimizer` for optimizing
        parameters.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch.distributed.autograd as dist_autograd
            >>> from torch.nn.parallel import DistributedDataParallel as DDP
            >>> import torch
            >>> from torch import optim
            >>> from torch.distributed.optim import DistributedOptimizer
            >>> import torch.distributed.rpc as rpc
            >>> from torch.distributed.rpc import RRef
            >>>
            >>> t1 = torch.rand((3, 3), requires_grad=True)
            >>> t2 = torch.rand((3, 3), requires_grad=True)
            >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))
            >>> ddp_model = DDP(my_model)
            >>>
            >>> # Setup optimizer
            >>> optimizer_params = [rref]
            >>> for param in ddp_model.parameters():
            >>>     optimizer_params.append(RRef(param))
            >>>
            >>> dist_optim = DistributedOptimizer(
            >>>     optim.SGD,
            >>>     optimizer_params,
            >>>     lr=0.05,
            >>> )
            >>>
            >>> with dist_autograd.context() as context_id:
            >>>     pred = ddp_model(rref.to_here())
            >>>     loss = loss_func(pred, target)
            >>>     dist_autograd.backward(context_id, [loss])
            >>>     dist_optim.step(context_id)

    .. note::
        DistributedDataParallel currently offers limited support for gradient
        checkpointing with :meth:`torch.utils.checkpoint`.
        If the checkpoint is done with use_reentrant=False (recommended), DDP
        will work as expected without any limitations.
        If, however, the checkpoint is done with use_reentrant=True (the default),
        DDP will work as expected when there are no unused parameters in the model
        and each layer is checkpointed at most once (make sure you are not passing
        `find_unused_parameters=True` to DDP). We currently do not support the
        case where a layer is checkpointed multiple times, or when there unused
        parameters in the checkpointed model.

    .. note::
        To let a non-DDP model load a state dict from a DDP model,
        :meth:`~torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`
        needs to be applied to strip the prefix "module." in the DDP state dict before loading.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) are distributed synchronization
        points. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient ``allreduce`` following the reverse order of the
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module allows parameters with non-rowmajor-contiguous strides.
        For example, your model may contain some parameters whose
        :class:`torch.memory_format` is ``torch.contiguous_format``
        and others whose format is ``torch.channels_last``.  However,
        corresponding parameters in different processes must have the
        same strides.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with ``DistributedDataParallel``. Because, when
        wrapping up your model with ``DistributedDataParallel``, the constructor
        of ``DistributedDataParallel`` will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters afterwards,
        gradient reduction functions no longer match the correct set of
        parameters.

    .. warning::
        Using ``DistributedDataParallel`` in conjunction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices.
                   1) For single-device modules, ``device_ids`` can
                   contain exactly one device id, which represents the only
                   CUDA device where the input module corresponding to this process resides.
                   Alternatively, ``device_ids`` can also be ``None``.
                   2) For multi-device modules and CPU modules,
                   ``device_ids`` must be ``None``.

                   When ``device_ids`` is ``None`` for both cases,
                   both the input data for the forward pass and the actual module
                   must be placed on the correct device.
                   (default: ``None``)
        output_device (int or torch.device): Device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be ``None``, and the module itself
                      dictates the output location. (default: ``device_ids[0]``
                      for single-device modules)
        broadcast_buffers (bool): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``True``)
        init_sync (bool): Whether to sync during initialization to verify param
                          shapes and broadcast parameters and buffers.
                          WARNING: if this is set to False the user is required
                          to ensure themselves that the weights are the same on
                          all ranks.
                          (default: ``True``)
        process_group: The process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
        bucket_cap_mb: ``DistributedDataParallel`` will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in
                       MebiBytes (MiB). If ``None``, a default size of 25 MiB
                       will be used. (default: ``None``)
        find_unused_parameters (bool): Traverse the autograd graph from all
                               tensors contained in the return value of the
                               wrapped module's ``forward`` function. Parameters
                               that don't receive gradients as part of this
                               graph are preemptively marked as being ready to
                               be reduced. In addition, parameters that may have
                               been used in the wrapped module's ``forward``
                               function but were not part of loss computation and
                               thus would also not receive gradients are
                               preemptively marked as ready to be reduced.
                               (default: ``False``)
        check_reduction: This argument is deprecated.
        gradient_as_bucket_view (bool): When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.
                      Note that gradients will be views after first iteration, so
                      the peak memory saving should be checked after first iteration.
        static_graph (bool): When set to ``True``, DDP knows the trained graph is
                     static. Static graph means 1) The set of used and unused
                     parameters will not change during the whole training loop; in
                     this case, it does not matter whether users set
                     ``find_unused_parameters = True`` or not. 2) How the graph is trained
                     will not change during the whole training loop (meaning there is
                     no control flow depending on iterations).
                     When static_graph is set to be ``True``, DDP will support cases that
                     can not be supported in the past:
                     1) Reentrant backwards.
                     2) Activation checkpointing multiple times.
                     3) Activation checkpointing when model has unused parameters.
                     4) There are model parameters that are outside of forward function.
                     5) Potentially improve performance when there are unused parameters,
                     as DDP will not search graph in each iteration to detect unused
                     parameters when static_graph is set to be ``True``.
                     To check whether you can set static_graph to be ``True``, one way is to
                     check ddp logging data at the end of your previous model training,
                     if ``ddp_logging_data.get("can_set_static_graph") == True``, mostly you
                     can set ``static_graph = True`` as well.

                     Example::
                         >>> # xdoctest: +SKIP("undefined variables")
                         >>> model_DDP = torch.nn.parallel.DistributedDataParallel(model)
                         >>> # Training loop
                         >>> ...
                         >>> ddp_logging_data = model_DDP._get_ddp_logging_data()
                         >>> static_graph = ddp_logging_data.get("can_set_static_graph")
        delay_all_reduce_named_params (list of tuple of str and torch.nn.Parameter): a list
                    of named parameters whose all reduce will be delayed when the gradient of
                    the parameter specified in ``param_to_hook_all_reduce`` is ready. Other
                    arguments of DDP do not apply to named params specified in this argument
                    as these named params will be ignored by DDP reducer.
        param_to_hook_all_reduce (torch.nn.Parameter): a parameter to hook delayed all reduce
                    of parameters specified in ``delay_all_reduce_named_params``.
        skip_all_reduce_unused_params: When set to True, DDP will skip reducing unused parameters.
                    This requires that unused parameters remain the same across all ranks throughout
                    the entire training process. If this condition is not met, it may cause
                    desynchronization and result in training hang.


    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.parallel.DistributedDataParallel(model)
    """

    # used to track whether the given thread is inside ddp forward for torchdynamo purposes
    _active_ddp_module: Optional["DistributedDataParallel"] = None

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        init_sync=True,
        process_group=None,
        bucket_cap_mb=None,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision: _MixedPrecision | None = None,
        device_mesh=None,
        skip_all_reduce_unused_params=False,
    ):
        super().__init__()
        Joinable.__init__(self)
        self._use_python_reducer = (
            torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer"
        )
        self.logger: dist.Logger | None = None
        if bool(delay_all_reduce_named_params is not None) != bool(
            param_to_hook_all_reduce is not None
        ):
            self._log_and_throw(
                ValueError,
                "delay_all_reduce_named_params and param_to_hook_all_reduce "
                "need to be set at the same time.",
            )

        if process_group and device_mesh is not None:
            raise RuntimeError(
                "Cannot specify both process_group and device_mesh arguments."
            )
        elif process_group is None and device_mesh is None:
            self.process_group = _get_default_group()
        elif device_mesh is None:
            # pyrefly: ignore [bad-assignment]
            self.process_group = process_group
        else:
            if device_mesh.ndim != 1:
                raise RuntimeError(
                    f"Only 1D device mesh is supported, but got {device_mesh}."
                )
            self.device_mesh = device_mesh
            self.process_group = device_mesh.get_group(mesh_dim=0)

            root_mesh = device_mesh._get_root_mesh()
            # if a root mesh is not the same as device_mesh,
            # meaning the device_mesh is sliced out from the root mesh.
            if root_mesh != device_mesh:
                # TODO: This is a temporary work around to enable DDP + TP.
                # We should do the logic in DDP so that the 2D implementation is
                # sound and the state_dict works out of the box.
                # This has to be done before check UninitializedParameter.
                from torch.distributed.tensor.parallel.ddp import (
                    _pre_dp_module_transform,
                )

                _pre_dp_module_transform(module)

        self._delay_all_reduce_params = []
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()
        if delay_all_reduce_named_params is not None:
            for name, param in delay_all_reduce_named_params:
                self.parameters_to_ignore.add(name)
                self._delay_all_reduce_params.append(param)

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            if len(self._delay_all_reduce_params):
                logger.info("Delay the AllReduce of all parameters.")
            else:
                self._log_and_throw(
                    RuntimeError,
                    "DistributedDataParallel is not needed when a module "
                    "doesn't have any parameter that requires a gradient.",
                )

        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = (
            len({p.device for p in self._module_parameters}) > 1
        )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                f"the same type of devices, but input module parameters locate in {distinct_device_types}.",
            )

        self.device_type = next(iter(distinct_device_types))

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    f"but got device_ids {device_ids}, output_device {output_device}, "
                    f"and module parameters { ({p.device for p in self._module_parameters}) }.",
                )

            self.device_ids = None
            self.output_device = None
        else:
            # pyrefly: ignore [bad-assignment]
            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            # pyrefly: ignore [bad-assignment]
            self.output_device = _get_device_index(output_device, True)

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = next(iter(self._module_parameters)).device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.mixed_precision = mixed_precision
        if self.mixed_precision is not None:
            logger.warning("Received mixed precision config %s", self.mixed_precision)

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it.",
                FutureWarning,
                stacklevel=2,
            )

        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * 1024 * 1024

        # reduction bucket size
        if bucket_cap_mb is None:
            # default case (bucket cap is 25 MiB)
            bucket_cap_mb = 25
            self.bucket_bytes_cap_default = True
        else:
            self.bucket_bytes_cap_default = False
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Initialize gradient buffers and register all reduce hook
        self._delay_grad_buffer: torch.Tensor | None = None
        self._delay_grad_views: list[torch.Tensor] = []
        self._delay_all_reduce_all_params = False
        if len(self._delay_all_reduce_params) != 0:
            self._register_delay_all_reduce_hook(
                bucket_cap_mb=bucket_cap_mb,
                param_to_hook_all_reduce=param_to_hook_all_reduce,
                device_ids=device_ids,
            )
            if self._delay_all_reduce_all_params:
                return

        self.skip_all_reduce_unused_params = skip_all_reduce_unused_params

        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()

        # All collectives during initialization are gated by this flag.
        if init_sync:
            # Verify model equivalence.
            _verify_param_shape_across_processes(self.process_group, parameters)
            # Sync params and buffers. Ensures all DDP models start off at the same value.
            _sync_module_states(
                module=self.module,
                process_group=self.process_group,
                broadcast_bucket_size=self.broadcast_bucket_size,
                src=0,
                params_and_buffers_to_ignore=self.parameters_to_ignore,
                broadcast_buffers=self.broadcast_buffers,
            )

        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)

        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph,
        )
        self._comm_hooks: list[tuple[Callable, object]] = []

        if self.mixed_precision is not None:
            _setup_mixed_precision_params(self.mixed_precision, self.module)
            _cast_buffers(self.mixed_precision, self.module)
            # Stream used for async low precision copies.
            self._mp_stream = torch.Stream()
            self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
            # Add forward pre-hook to root module to kick off copies to lower
            # precision.
            self.module.register_forward_pre_hook(
                self._root_copy_hook, prepend=False, with_kwargs=True
            )
            # Add forward pre hook to all submodules to wait for copy events
            # before running computation.
            for module in self.module.modules():
                module.register_forward_pre_hook(
                    self._module_wait_for_copy_hook,
                    prepend=False,
                    with_kwargs=True,
                )
            # Set up callbacks in backward to upcast and use full precision
            # params. TODO (rohan-varma): Make this compose with general
            # comm hooks and apply_optimizer_in_backward. Importing inline to
            # avoid circular import issue.
            from torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks import (
                _AllreduceUpcastHookState,
                _reducer_allreduce_and_upcast_hook,
            )

            upcast_hook_state = _AllreduceUpcastHookState(
                ddp_weakref=weakref.ref(self),
                upcast_stream=torch.Stream(),
            )
            self.register_comm_hook(
                upcast_hook_state,
                _reducer_allreduce_and_upcast_hook,
            )
            # Inform reducer of reduced precision param dtype for correctness
            # of type checks between gradient and bucket.
            self.reducer._set_mixed_precision_param_dtype(  # type: ignore[attr-defined]
                self.mixed_precision.param_dtype
            )

        self._has_rebuilt_buckets = False

        if static_graph:
            self._set_static_graph()

        self._lazy_init_ran = False

        # Register the AccumulateGrad post hooks if optimize_ddp is
        # True. The hooks will be deregistered if compiled_autograd is not
        # enabled.
        self._accum_grad_hooks: list[RemovableHandle] = []
        if self._use_python_reducer:
            # pyrefly: ignore [bad-assignment]
            torch._inductor.config._fuse_ddp_communication = True
            torch._inductor.config._fuse_ddp_bucket_size = bucket_cap_mb
            # Directly adding this to the trace rule will disturb the users
            # who are using DDPOptimizer.
            torch._dynamo.trace_rules.LEGACY_MOD_INLINELIST.add(
                "torch.nn.parallel.distributed"
            )
            torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
            # NOTE: we should init these lazily
            self._register_accum_grad_hook()

        # Whether or not DDPSink performs a clone.
        self._ddp_sink_clone = True

    def _register_accum_grad_hook(self):
        import torch.distributed._functional_collectives as fcol

        def compiled_accum_grad_hook(
            param,
            *,
            param_index: int,
        ):
            if not self.require_backward_grad_sync:
                return

            if param.grad is None:
                return

            if self._comm_hooks:
                for hook, state in self._comm_hooks:
                    hook(state, (param.grad, param))
            else:
                gradient = param.grad / self.process_group.size()
                gradient = fcol.all_reduce(gradient, "sum", self.process_group)
                param.grad.copy_(gradient)

        for index, param in enumerate(self._module_parameters):
            if not param.requires_grad:
                continue
            self._accum_grad_hooks.append(
                param.register_post_accumulate_grad_hook(
                    functools.partial(
                        compiled_accum_grad_hook,
                        param_index=index,
                    )
                )
            )

    def _delayed_all_reduce_hook(self, grad):
        world_size = dist.get_world_size(self.process_group)

        self._delay_grad_buffer.div_(world_size)  # type: ignore[union-attr]
        _ = dist.all_reduce(
            self._delay_grad_buffer, group=self.process_group, async_op=True
        )
        return grad

    def _register_delay_all_reduce_hook(
        self,
        bucket_cap_mb,
        param_to_hook_all_reduce,
        device_ids,
    ):
        # 1. Create gradient buffer
        device = torch.device("cpu") if device_ids is None else device_ids[0]
        self._delay_grad_buffer = torch.zeros(
            sum(p.numel() for p in self._delay_all_reduce_params),
            device=device,
        )

        # 2. Broadcast the parameters
        detached_params = [p.detach() for p in self._delay_all_reduce_params]
        dist._broadcast_coalesced(self.process_group, detached_params, bucket_cap_mb, 0)

        # 3. Hook all reduce to the specified parameter
        param_to_hook_all_reduce.register_hook(self._delayed_all_reduce_hook)

        # 4. Build tensor views for gradients
        offset = 0
        for param in self._delay_all_reduce_params:
            grad_view = self._delay_grad_buffer[offset : (offset + param.numel())].view(
                param.shape
            )
            self._delay_grad_views.append(grad_view)
            offset = offset + param.numel()

        # 5. Check whether the all reduce of all params requiring grad is delayed.
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{module_name}.{param_name}"
                    if full_name not in self.parameters_to_ignore:
                        # There is at least a param whose all reduce will not be delayed.
                        # In this case, we should not set self._delay_all_reduce_all_params
                        # to True.
                        return
        self._delay_all_reduce_all_params = True

    def _setup_in_backward_optimizers(self):
        # Check if user has used apply_optim_in_backward to overlap optimizer
        # step + DDP backward. Current constraints:
        # 1. Only allreduce is supported at the moment, no custom communication.
        # 2. For DDP-managed parameters that have their optimizer run in
        # backward, their gradients are set to ``None``. If your use case
        # requires DDP parameters grad not to be set to ``None`` after their
        # in-backward optimizer runs, please ping
        # https://github.com/pytorch/pytorch/issues/90052.
        # NOTE: we use self._module_parameters instead of .parameters() since
        # the former excludes ignored (non-DDP managed) parameters.
        if any(hasattr(p, "_in_backward_optimizers") for p in self._module_parameters):
            torch._C._log_api_usage_once("ddp.optimizer_in_backward")
            # Remove hooks that apply_optim_in_backward had registered because
            # DDP customizes how optimizer is overlapped with backward due to
            # the allreduce.
            param_to_handle_map = (
                dist.optim.apply_optimizer_in_backward.param_to_optim_hook_handle_map
            )
            for p in self._module_parameters:
                for handle in param_to_handle_map.get(p, []):
                    handle.remove()

            # Need a weakref to DDP instance to run all_reduce (from reducer)
            # and get managed DDP parameters.
            ddp_weakref = weakref.ref(self)
            # Note: importing in function, otherwise this will cause a circular
            # import.
            from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
                _apply_optim_in_backward_hook,
            )

            self.register_comm_hook(
                ddp_weakref,
                _apply_optim_in_backward_hook(
                    gradient_is_bucket_view=self.gradient_as_bucket_view
                ),
            )

            self.reducer._set_optimizer_in_backward()  # type: ignore[attr-defined]

    def _fire_reducer_autograd_hook(self, idx, *unused):
        """
        Fire the reducer's autograd hook to allreduce params in a Reducer bucket.

        Note that this is only used during mixed precision training as the
        Reducer's hooks installed during construction time would not be called
        as we're working in the low precision parameter setting.
        """
        self.reducer._autograd_hook(idx)  # type: ignore[attr-defined]

    def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        For DDP mixed precision, put low precision copies on separate stream and create events to wait for them.

        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
        # Clear out previous iteration submodule to event. This is because we
        # may have populated some events for modules that didn't end up being
        # used.
        self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
        with self._mp_stream:
            for submodule in self.module.modules():
                for param in submodule.parameters(recurse=False):
                    # Do not cast DDP ignored parameters.
                    if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                        continue
                    _alloc_storage(param._mp_param, param.size())
                    # copy() implicitly casts to low precision
                    with torch.no_grad():
                        param._mp_param.copy_(param.data)
                        # TODO: when zero_grad(set_to_none=False) or in grad
                        # accumulation case, accumulated grads can be in fp32
                        # wh
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/parallel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nn/parallel`):

- [`replicate.py_docs.md_docs.md`](./replicate.py_docs.md_docs.md)
- [`replicate.py_kw.md_docs.md`](./replicate.py_kw.md_docs.md)
- [`scatter_gather.py_kw.md_docs.md`](./scatter_gather.py_kw.md_docs.md)
- [`parallel_apply.py_docs.md_docs.md`](./parallel_apply.py_docs.md_docs.md)
- [`parallel_apply.py_kw.md_docs.md`](./parallel_apply.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`comm.py_kw.md_docs.md`](./comm.py_kw.md_docs.md)
- [`comm.py_docs.md_docs.md`](./comm.py_docs.md_docs.md)
- [`_functions.py_docs.md_docs.md`](./_functions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `distributed.py_docs.md_docs.md`
- **Keyword Index**: `distributed.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
