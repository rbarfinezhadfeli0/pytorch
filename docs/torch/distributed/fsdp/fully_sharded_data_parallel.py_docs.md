# Documentation: `torch/distributed/fsdp/fully_sharded_data_parallel.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/fully_sharded_data_parallel.py`
- **Size**: 101,286 bytes (98.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

import contextlib
import copy
import functools
import math
import traceback
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from enum import auto, Enum
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_param_to_fqns,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    HandleTrainingState,
    TrainingState,
)
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_extension,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)
from torch.distributed.fsdp._runtime_utils import (
    _get_fsdp_root_states,
    _is_fsdp_root,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
    _unshard,
    _wait_for_computation_stream,
)
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.tensor import DeviceMesh
from torch.distributed.utils import _p_assert

from ._flat_param import FlatParameter, FlatParamHandle
from ._optim_utils import (
    _flatten_optim_state_dict,
    _get_param_id_to_param_from_optim_input,
    _get_param_key_to_param,
    _get_param_to_param_id_from_optim_input,
    _get_param_to_param_key,
    _optim_state_dict,
    _rekey_sharded_optim_state_dict,
    _set_optim_use_dtensor,
)
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
    _deregister_orig_params,
    _register_flat_param,
    _register_orig_params,
    _unshard_params,
    _unshard_params_for_summon,
)
from .wrap import CustomPolicy, ModuleWrapPolicy


__all__ = [
    "FullyShardedDataParallel",
    "OptimStateKeyType",
]


FLAT_PARAM = "_flat_param"


class OptimStateKeyType(Enum):
    """Represents the type of key in an optimizer state-dict."""

    PARAM_NAME = auto()
    PARAM_ID = auto()


class FullyShardedDataParallel(nn.Module, _FSDPState):
    """A wrapper for sharding module parameters across data parallel workers.

    This is inspired by `Xu et al. <https://arxiv.org/abs/2004.13336>`_ as
    well as the ZeRO Stage 3 from `DeepSpeed <https://www.deepspeed.ai/>`_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    Using FSDP involves wrapping your module and then initializing your
    optimizer after. This is required since FSDP changes the parameter
    variables.

    When setting up FSDP, you need to consider the destination CUDA
    device. If the device has an ID (``dev_id``), you have three options:

    * Place the module on that device
    * Set the device using ``torch.cuda.set_device(dev_id)``
    * Pass ``dev_id`` into the ``device_id`` constructor argument.

    This ensures that the FSDP instance's compute device is the
    destination device. For option 1 and 3, the FSDP initialization
    always occurs on GPU. For option 2, the FSDP initialization
    happens on module's current device, which may be a CPU.

    If you're using the ``sync_module_states=True`` flag, you need to
    ensure that the module is on a GPU or use the ``device_id``
    argument to specify a CUDA device that FSDP will move the module
    to in the FSDP constructor. This is necessary because
    ``sync_module_states=True`` requires GPU communication.

    FSDP also takes care of moving input tensors to the forward method
    to the GPU compute device, so you don't need to manually move them
    from CPU.

    For ``use_orig_params=True``,
    ``ShardingStrategy.SHARD_GRAD_OP`` exposes the unsharded
    parameters, not the sharded parameters after forward, unlike
    ``ShardingStrategy.FULL_SHARD``. If you want
    to inspect the gradients, you can use the ``summon_full_params``
    method with ``with_grads=True``.

    With ``limit_all_gathers=True``, you may see a gap in the FSDP
    pre-forward where the CPU thread is not issuing any kernels. This is
    intentional and shows the rate limiter in effect. Synchronizing the CPU
    thread in that way prevents over-allocating memory for subsequent
    all-gathers, and it should not actually delay GPU kernel execution.

    FSDP replaces managed modules' parameters with ``torch.Tensor``
    views during forward and backward computation for autograd-related
    reasons. If your module's forward relies on saved references to
    the parameters instead of reacquiring the references each
    iteration, then it will not see FSDP's newly created views,
    and autograd will not work correctly.

    Finally, when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
    with the sharding process group being intra-node and the
    replication process group being inter-node, setting
    ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
    the replication process group for some cluster setups.

    **Limitations**

    There are several limitations to be aware of when using FSDP:

    * FSDP currently does not support gradient accumulation outside
      ``no_sync()`` when using CPU offloading. This is because FSDP
      uses the newly-reduced gradient instead of accumulating with any
      existing gradient, which can lead to incorrect results.

    * FSDP does not support running the forward pass of a submodule
      that is contained in an FSDP instance. This is because the
      submodule's parameters will be sharded, but the submodule itself
      is not an FSDP instance, so its forward pass will not all-gather
      the full parameters appropriately.

    * FSDP does not work with double backwards due to the way it
      registers backward hooks.

    * FSDP has some constraints when freezing parameters.
      For ``use_orig_params=False``, each FSDP instance must manage
      parameters that are all frozen or all non-frozen. For
      ``use_orig_params=True``, FSDP supports mixing frozen and
      non-frozen parameters, but it's recommended to avoid doing so to
      prevent higher than expected gradient memory usage.

    * As of PyTorch 1.12, FSDP offers limited support for shared
      parameters. If enhanced shared parameter support is needed for
      your use case, please post in
      `this issue <https://github.com/pytorch/pytorch/issues/77724>`__.

    * You should avoid modifying the parameters between forward and
      backward without using the ``summon_full_params`` context, as
      the modifications may not persist.

    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]):
            This is the process group over which the model is sharded and thus
            the one used for FSDP's all-gather and reduce-scatter collective
            communications. If ``None``, then FSDP uses the default process
            group. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a tuple of
            process groups, representing the groups over which to shard and
            replicate, respectively. If ``None``, then FSDP constructs process
            groups for the user to shard intra-node and replicate inter-node.
            (Default: ``None``)
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy, which may trade off memory
            saving and communication overhead. See :class:`ShardingStrategy`
            for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], ModuleWrapPolicy, CustomPolicy]]):
            This specifies a policy to apply FSDP to submodules of ``module``,
            which is needed for communication and computation overlap and thus
            affects performance. If ``None``, then FSDP only applies to
            ``module``, and users should manually apply FSDP to parent modules
            themselves (proceeding bottom-up). For convenience, this accepts
            ``ModuleWrapPolicy`` directly, which allows users to specify the
            module classes to wrap (e.g. the transformer block). Otherwise,
            this should be a callable that takes in three arguments
            ``module: nn.Module``, ``recurse: bool``, and
            ``nonwrapped_numel: int`` and should return a ``bool`` specifying
            whether the passed-in ``module`` should have FSDP applied if
            ``recurse=False`` or if the traversal should continue into the
            module's subtree if ``recurse=True``. Users may add additional
            arguments to the callable. The ``size_based_auto_wrap_policy`` in
            ``torch.distributed.fsdp.wrap.py`` gives an example callable that
            applies FSDP to a module if the parameters in its subtree exceed
            100M numel. We recommend printing the model after applying FSDP
            and adjusting as needed.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     nonwrapped_numel: int,
                >>>     # Additional custom arguments
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return nonwrapped_numel >= min_num_params
                >>> # Configure a custom `min_num_params`
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. If
            ``None``, then FSDP does not backward prefetch, and there is no
            communication and computation overlap in the backward pass. See
            :class:`BackwardPrefetch` for details. (Default: ``BACKWARD_PRE``)
        mixed_precision (Optional[MixedPrecision]):
            This configures native mixed precision for FSDP. If this is set to
            ``None``, then no mixed precision is used. Otherwise, parameter,
            buffer, and gradient reduction dtypes can be set. See
            :class:`MixedPrecision` for details. (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should
            be initialized onto an actual device. As of v1.12, FSDP detects
            modules with parameters or buffers on meta device via ``is_meta``
            and either applies ``param_init_fn`` if specified or calls
            ``nn.Module.reset_parameters()`` otherwise. For both cases, the
            implementation should *only* initialize the parameters/buffers of
            the module, not those of its submodules. This is to avoid
            re-initialization. In addition, FSDP also supports deferred
            initialization via torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init()`` API, where the deferred modules are initialized
            by calling ``param_init_fn`` if specified or torchdistX's default
            ``materialize_module()`` otherwise. If ``param_init_fn`` is
            specified, then it is applied to all meta-device modules, meaning
            that it should probably case on the module type. FSDP calls the
            initialization function before parameter flattening and sharding.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module: nn.Module):
                >>>     # E.g. initialize depending on the module type
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or
            ``torch.device`` giving the CUDA device on which FSDP
            initialization takes place, including the module initialization
            if needed and the parameter sharding. This should be specified to
            improve initialization speed if ``module`` is on CPU. If the
            default CUDA device was set (e.g. via ``torch.cuda.set_device``),
            then the user may pass ``torch.cuda.current_device`` to this.
            (Default: ``None``)
        sync_module_states (bool): If ``True``, then each FSDP module will
            broadcast module parameters and buffers from rank 0 to ensure that
            they are replicated across ranks (adding communication overhead to
            this constructor). This can help load ``state_dict`` checkpoints
            via ``load_state_dict`` in a memory efficient way. See
            :class:`FullStateDictConfig` for an example of this. (Default:
            ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next forward-pass all-gather before the current forward
            computation. This is only useful for CPU-bound workloads, in which
            case issuing the next all-gather earlier may improve overlap. This
            should only be used for static-graph models since the prefetching
            follows the first iteration's execution order. (Default: ``False``)
        limit_all_gathers (bool): If ``True``, then FSDP explicitly
            synchronizes the CPU thread to ensure GPU memory usage from only
            *two* consecutive FSDP instances (the current instance running
            computation and the next instance whose all-gather is prefetched).
            If ``False``, then FSDP allows the CPU thread to issue all-gathers
            without any extra synchronization. (Default: ``True``) We often
            refer to this feature as the "rate limiter". This flag should only
            be set to ``False`` for specific CPU-bound workloads with low
            memory pressure in which case the CPU thread can aggressively issue
            all kernels without concern for the GPU memory usage.
        use_orig_params (bool): Setting this to ``True`` has FSDP use
            ``module`` 's original parameters. FSDP exposes those original
            parameters to the user via :meth:`nn.Module.named_parameters`
            instead of FSDP's internal :class:`FlatParameter` s. This means
            that the optimizer step runs on the original parameters, enabling
            per-original-parameter hyperparameters. FSDP preserves the original
            parameter variables and manipulates their data between unsharded
            and sharded forms, where they are always views into the underlying
            unsharded or sharded :class:`FlatParameter`, respectively. With the
            current algorithm, the sharded form is always 1D, losing the
            original tensor structure. An original parameter may have all,
            some, or none of its data present for a given rank. In the none
            case, its data will be like a size-0 empty tensor. Users should not
            author programs relying on what data is present for a given
            original parameter in its sharded form. ``True`` is required to
            use ``torch.compile()``. Setting this to ``False`` exposes FSDP's
            internal :class:`FlatParameter` s to the user via
            :meth:`nn.Module.named_parameters`. (Default: ``False``)
        ignored_states (Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]):
            Ignored parameters or modules that will not be managed by this FSDP
            instance, meaning that the parameters are not sharded and their
            gradients are not reduced across ranks. This argument unifies with
            the existing ``ignored_modules`` argument, and we may deprecate
            ``ignored_modules`` soon. For backward compatibility, we keep both
            ``ignored_states`` and `ignored_modules``, but FSDP only allows one
            of them to be specified as not ``None``.
        device_mesh (Optional[DeviceMesh]): DeviceMesh can be used as an alternative to
            process_group. When device_mesh is passed, FSDP will use the underlying process
            groups for all-gather and reduce-scatter collective communications. Therefore,
            these two args need to be mutually exclusive. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a 2D DeviceMesh instead
            of a tuple of process groups. For 2D FSDP + TP, users are required to pass in
            device_mesh instead of process_group. For more DeviceMesh info, please visit:
            https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[
            Union[Callable, ModuleWrapPolicy, CustomPolicy]
        ] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            warnings.warn(
                "FSDP will not all-gather parameters for containers that do "
                f"not implement forward: {module}",
                stacklevel=2,
            )
        _init_ignored_module_states(self, module, ignored_modules, ignored_states)
        _init_device_handle(self, module, self._ignored_params, device_id)

        # Add module annotations for Dynamo support (see function for details)
        _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

        # Initializes self.process_group, along with rank and world size. This will
        # also set another attribute, _inter_node_pg, to control the process group
        # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
        # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
        # the same process group state as the root FSDP module.
        self._device_mesh = device_mesh
        _init_process_group_state(
            self,
            process_group,
            sharding_strategy,
            auto_wrap_policy,
            device_mesh,
        )
        if auto_wrap_policy is not None:
            root_kwargs = {
                "process_group": process_group,
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "mixed_precision": mixed_precision,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "sync_module_states": sync_module_states,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
                "use_orig_params": use_orig_params,
                "ignored_states": self._ignored_params,
                "device_mesh": device_mesh,
            }
            if sharding_strategy in HYBRID_SHARDING_STRATEGIES and device_mesh is None:
                # Share root process groups with children to maintain
                # the invariant that all FSDP modules will have the same
                # process groups.
                root_kwargs["process_group"] = (self.process_group, self._inter_node_pg)

            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                root_kwargs,
                FullyShardedDataParallel,
            )

        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            sharding_strategy,
            mixed_precision,
            cpu_offload,
            limit_all_gathers,
            use_orig_params,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        _init_runtime_state(self)
        _init_prefetching_state(self, backward_prefetch, forward_prefetch)
        _init_buffer_state(self, module)
        # extension needs to be set before `_init_param_handle_from_module()`
        _init_extension(self, device_mesh)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
            sync_module_states,
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            _register_flat_param(self, self)

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)
        _register_all_state_dict_hooks(self)
        self._zero_scalar = None

    @property
    def module(self) -> nn.Module:
        """Return the wrapped module."""
        # FSDP's `.module` must refer to the innermost wrapped module when
        # composing with other module wrappers in order for state dict to work
        if isinstance(self._fsdp_wrapped_module, ActivationWrapper):
            return getattr(self._fsdp_wrapped_module, _CHECKPOINT_WRAPPED_MODULE)
        return self._fsdp_wrapped_module

    @property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
        return hasattr(self, "_handle") and self._handle is not None

    @property
    def _flat_param(self) -> Optional[FlatParameter]:
        return self._handle.flat_param if self._handle else None

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        if hasattr(self, FSDP_WRAPPED_MODULE):
            return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]
        return super().__getitem__(key)

    def check_is_root(self) -> bool:
        """Check if this instance is a root FSDP module."""
        return _is_fsdp_root(self, self)

    @staticmethod
    def fsdp_modules(
        module: nn.Module,
        root_only: bool = False,
    ) -> list["FullyShardedDataParallel"]:
        """Return all nested FSDP instances.

        This possibly includes ``module`` itself and only includes FSDP root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
        if root_only:
            return _get_fsdp_root_states(module)
        return traversal_utils._get_fsdp_states(module)

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState.IDLE)
        # Use `_unshard_params_for_summon()` with `recurse=False` instead of
        # `_unshard_fsdp_state_params()` directly to perform lazy
        # initialization, which is needed to initialize `FlatParameter`
        # parameter attributes as required by the unshard logic
        with _unshard_params_for_summon(
            self,
            self,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False,
        ):
            ret = super().apply(fn)

        # Reset lazy init called in `_unshard_params_for_summon()` since
        # `apply()` may have been called on FSDP instance that is not truly a
        # root, in which case it will be incorrectly marked as one.
        if uninitialized and self._is_root:
            for module in traversal_utils._get_fsdp_states(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """Return whether the user explicitly enabled buffer mixed precision.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _low_precision_hook_enabled(self) -> bool:
        """Whether a low precision hook is registered or not."""
        return self._comm_hook is not None and self._comm_hook in LOW_PRECISION_HOOKS

    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
        self._is_root: Optional[bool] = None

    @staticmethod
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
    ) -> StateDictSettings:
        """Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        Also takes (optional) configuration for the model's and optimizer's state dict.
        The target module does not have to be a FSDP module. If the target
        module is a FSDP module, its ``state_dict_type`` will also be changed.

        .. note:: This API should be called for only the top-level (root)
            module.

        .. note:: This API enables users to transparently use the conventional
            ``state_dict`` API to take model checkpoints in cases where the
            root FSDP module is wrapped by another ``nn.Module``. For example,
            the following will ensure ``state_dict`` is called on all non-FSDP
            instances, while dispatching into `sharded_state_dict` implementation
            for FSDP:

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>>     state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
            >>>     optim_state_dict_config = OptimStateDictConfig(offload_to_cpu=True),
            >>> )
            >>> param_state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the configuration
                for the optimizer state dict.

        Returns:
            A StateDictSettings that include the previous state_dict type and
            configuration for the module.
        """
        warnings.warn(
            "FSDP.state_dict_type() and FSDP.set_state_dict_type() are being "
            "deprecated. Please use APIs, get_state_dict() and set_state_dict(), "
            "which can support different parallelisms, FSDP1, FSDP2, DDP. "
            "API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html"
            "#torch.distributed.checkpoint.state_dict.get_state_dict ."
            "Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .",
            FutureWarning,
            stacklevel=2,
        )
        _state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }
        _optim_state_dict_type_to_config = {
            StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig,
        }

        # Use the default config if a state_dict config is not set.
        state_dict_config_type = _state_dict_type_to_config[state_dict_type]
        optim_state_dict_config_type = _optim_state_dict_type_to_config[state_dict_type]
        if state_dict_config is None:
            state_dict_config = state_dict_config_type()
        if optim_state_dict_config is None:
            optim_state_dict_config = optim_state_dict_config_type()
        if state_dict_config_type is not type(state_dict_config):
            raise RuntimeError(
                f"Expected state_dict_config of type {state_dict_config_type} "
                f"but got {type(state_dict_config)}"
            )
        if optim_state_dict_config_type is not type(optim_state_dict_config):
            raise RuntimeError(
                f"Expected optim_state_dict_config of type {optim_state_dict_config_type} "
                f"but got {type(optim_state_dict_config)}"
            )

        # Set the state_dict type and configurations.
        prev_state_dict_type = None
        prev_state_dict_config = None
        prev_optim_state_dict_config = None
        for submodule in traversal_utils._get_fsdp_states(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = submodule._state_dict_type
            else:
                if prev_state_dict_type != submodule._state_dict_type:
                    raise AssertionError(
                        "All FSDP modules should have the same state_dict_type."
                    )
            if prev_state_dict_config is None:
                prev_state_dict_config = submodule._state_dict_config
            else:
                if not isinstance(
                    submodule._state_dict_config, type(prev_state_dict_config)
                ):
                    raise AssertionError(
                        "All FSDP modules must have the same type of state_dict_config."
                    )
            if prev_optim_state_dict_config is None:
                prev_optim_state_dict_config = submodule._optim_state_dict_config
            else:
                if not isinstance(
                    submodule._optim_state_dict_config,
                    type(prev_optim_state_dict_config),
                ):
                    raise AssertionError(
                        "All FSDP modules must have the same type of optim_state_dict_config."
                    )

            submodule._state_dict_type = state_dict_type
            submodule._state_dict_config = state_dict_config
            submodule._optim_state_dict_config = optim_state_dict_config

        return StateDictSettings(
            prev_state_dict_type, prev_state_dict_config, prev_optim_state_dict_config
        )

    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings:
        """Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at ``module``.

        The target module does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
        state_dict_settings: Optional[StateDictSettings] = None
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if state_dict_settings is None:
                state_dict_settings = StateDictSettings(
                    state_dict_type=submodule._state_dict_type,
                    state_dict_config=submodule._state_dict_config,
                    optim_state_dict_config=submodule._optim_state_dict_config,
                )
                _set_optim_use_dtensor(submodule, state_dict_settings)
            else:
                submodule_settings = StateDictSettings(
                    submodule._state_dict_type,
                    submodule._state_dict_config,
                    submodule._optim_state_dict_config,
                )
                if state_dict_settings != submodule_settings:
                    raise AssertionError(
                        "All FSDP modules must have the same state dict settings."
                        f"Got {submodule_settings} and {state_dict_settings}."
                    )
                _set_optim_use_dtensor(submodule, submodule_settings)
        return state_dict_settings

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
    ) -> Generator:
        """Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        This context manager has the same functions as :meth:`set_state_dict_type`. Read the document of
        :meth:`set_state_dict_type` for the detail.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>> ):
            >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the model ``state_dict``
                configuration for the target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the optimizer
               ``state_dict`` configuration for the target ``state_dict_type``.
        """
        prev_state_dict_settings = FullyShardedDataParallel.set_state_dict_type(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        )
        yield
        FullyShardedDataParallel.set_state_dict_type(
            module,
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
            prev_state_dict_settings.optim_state_dict_config,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic."""
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module: nn.Module,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
        with_grads: bool = False,
    ) -> Generator:
        r"""Expose full params for FSDP instances with this context manager.

        Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking. It can take a non-FSDP
        module and will summon full params for all contained FSDP modules as
        well as their children, depending on the ``recurse`` argument.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when ``world_size == 1``, or ``NO_SHARD``
            config, the modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        .. warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
            result in full parameters being redundantly copied to CPU memory for
            GPUs that reside on the same machine, which may incur the risk of
            CPU OOM. It is recommended to use ``offload_to_cpu`` with
            ``rank0_only=True``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True).
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exits;
                disabling this can be slightly more efficient (default: True)
            rank0_only (bool, Optional): if ``True``, full parameters are
                materialized on only global rank 0. This means that within the
                context, only rank 0 will have full parameters and the other
                ranks will have sharded parameters. Note that setting
                ``rank0_only=True`` with ``writeback=True`` is not supported,
                as model parameter shapes will be different across ranks
                within the context, and writing to them can lead to
                inconsistency across ranks when the context is exited.
            offload_to_cpu (bool, Optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1 or ``NO_SHARD`` config). It is recommended
                to use ``offload_to_cpu`` with ``rank0_only=True`` to avoid
                redundant copies of model parameters being offloaded to the same CPU memory.
            with_grads (bool, Optional): If ``True``, gradients are also
                unsharded with the parameters. Currently, this is only
                supported when passing ``use_orig_params=True`` to the FSDP
                constructor and ``offload_to_cpu=False`` to this method.
                (Default: ``False``)
        """
        with _unshard_params(
            module, recurse, writeback, rank0_only, offload_to_cpu, with_grads
        ):
            yield

    @contextlib.contextmanager
    def _deregister_orig_params_ctx(self):
        """Deregister the original parameters and expose the :class:`FlatParameter`.

        If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method should
        only be called when using the original parameters.
        """
        _p_assert(
            self._use_orig_params,
            "`_deregister_orig_params_ctx()` should only be called when "
            "`_use_orig_params=True`",
        )
        for fsdp_module in traversal_utils._get_fsdp_states(self):
            _deregister_orig_params(fsdp_module, fsdp_module)
        try:
            yield
        finally:
            for fsdp_module in traversal_utils._get_fsdp_states(self):
                _register_orig_params(fsdp_module, fsdp_module)

    def _apply(self, *args, **kwargs):
        """Deregister the original parameters and expose the :class:`FlatParameter` s before calling ``_apply()``."""
        # When using the original parameters: Since (1) the `FlatParameter`s
        # own the storage and (2) `_apply()` is the subroutine underlying the
        # most common storage-changing ops like `to()` and `cuda()`, we
        # override `_apply()` to have the storage change directly performed on
        # the `FlatParameter`s instead of applying to the original parameters
        # and then writing back to the `FlatParameter`s.
        context = (
            self._deregister_orig_params_ctx()
            if self._use_orig_params
            else contextlib.nullcontext()
        )
        with context:
            return super()._apply(*args, **kwargs)

    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.

        Intercepts buffer names and removes all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                buffer_name = buffer_name.replace(FSDP_PREFIX, "")
            yield (buffer_name, buffer)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself.

        Intercepts parameter names and removes all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

    def _assert_state(self, state: Union[TrainingState, list[TrainingState]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        """Disable gradient synchronizations across FSDP instances.

        Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`no_sync()` on inner FSDP instances is not supported. Please call `no_sync()` on root FSDP module."
            )
        self._asse
```



## High-Level Overview


This Python file contains 3 class(es) and 48 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OptimStateKeyType`, `FullyShardedDataParallel`, `UnshardHandle`

**Functions defined**: `custom_auto_wrap_policy`, `my_init_fn`, `__init__`, `module`, `_has_params`, `_flat_param`, `__getattr__`, `__getitem__`, `check_is_root`, `fsdp_modules`, `apply`, `_mixed_precision_enabled_for_buffers`, `_low_precision_hook_enabled`, `_reset_lazy_init`, `set_state_dict_type`, `get_state_dict_type`, `state_dict_type`, `forward`, `summon_full_params`, `_deregister_orig_params_ctx`

**Key imports**: contextlib, copy, functools, math, traceback, warnings, Callable, Generator, Iterable, Iterator, contextmanager, auto, Enum, Any, Optional, Union


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `functools`
- `math`
- `traceback`
- `warnings`
- `collections.abc`: Callable, Generator, Iterable, Iterator
- `enum`: auto, Enum
- `typing`: Any, Optional, Union
- `torch`
- `torch.distributed as dist`
- `torch.distributed.fsdp._traversal_utils as traversal_utils`
- `torch.nn as nn`
- `torch.distributed.algorithms._comm_hooks`: LOW_PRECISION_HOOKS
- `torch.distributed.fsdp._dynamo_utils`: _annotate_modules_for_dynamo
- `torch.distributed.fsdp._wrap_utils`: _auto_wrap
- `torch.distributed.tensor`: DeviceMesh
- `torch.distributed.utils`: _p_assert
- `._flat_param`: FlatParameter, FlatParamHandle
- `._state_dict_utils`: _register_all_state_dict_hooks
- `.wrap`: CustomPolicy, ModuleWrapPolicy
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/distributed/fsdp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_limiter_utils.py_docs.md`](./_limiter_utils.py_docs.md)
- [`_traversal_utils.py_docs.md`](./_traversal_utils.py_docs.md)
- [`_runtime_utils.py_docs.md`](./_runtime_utils.py_docs.md)
- [`_common_utils.py_docs.md`](./_common_utils.py_docs.md)
- [`_wrap_utils.py_docs.md`](./_wrap_utils.py_docs.md)
- [`_exec_order_utils.py_docs.md`](./_exec_order_utils.py_docs.md)
- [`sharded_grad_scaler.py_docs.md`](./sharded_grad_scaler.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `fully_sharded_data_parallel.py_docs.md`
- **Keyword Index**: `fully_sharded_data_parallel.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
