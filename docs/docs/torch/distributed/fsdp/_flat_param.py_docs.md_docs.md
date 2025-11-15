# Documentation: `docs/torch/distributed/fsdp/_flat_param.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/fsdp/_flat_param.py_docs.md`
- **Size**: 53,926 bytes (52.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/fsdp/_flat_param.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_flat_param.py`
- **Size**: 126,271 bytes (123.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import contextlib
import functools
import logging
import os
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from enum import auto, Enum
from itertools import accumulate, chain
from typing import Any, cast, NamedTuple, no_type_check, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _named_parameters_with_duplicates,
    _no_dispatch_record_stream,
    _set_fsdp_flattened,
    HandleTrainingState,
)
from torch.distributed.utils import (
    _alloc_storage,
    _data_ptr_allocated,
    _free_storage,
    _p_assert,
)
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

from ._fsdp_extensions import (
    _ext_post_unflatten_transform,
    _ext_pre_flatten_transform,
    FSDPExtensions,
)


__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleShardingStrategy",
]

logger = logging.getLogger(__name__)


"""
[Note: Fully Sharded Module]
We define the "fully sharded module" to be the original ``nn.Module`` that owns
a ``FlatParamHandle``. It is the *single* module logically responsible for the
*single* unshard/reshard pair for the handle's ``FlatParameter`` for a given
forward or backward pass. The fully sharded module should be passed to the
``FlatParamHandle`` constructor.

For the wrapper code path:
- The ``FullyShardedDataParallel`` module wrapping the fully sharded module
runs the unshard/reshard on behalf of the fully sharded module by overriding
``nn.Module.forward``.
- The fully sharded module is exactly the module passed to the
``FullyShardedDataParallel`` constructor's ``module`` argument.

For the non-wrapper code path:
- Hooks registered on the fully sharded module run the unshard/reshard.
- The fully sharded module may either be the direct argument to ``fully_shard``
or a submodule chosen by the provided wrapping policy.
"""

# Environment variable toggling whether to use unsafe `setattr()` for view
# setting in `_use_sharded_views()` and `_use_unsharded_views()`
# We should use 'safe' by default since it respects method overrides, but for
# special cases such as for high CPU overhead or for intentionally bypassing
# checks in the overrides, we may use 'unsafe'.
_FSDP_USE_UNSAFE_SETATTR = "FSDP_USE_UNSAFE_SETATTR"

# Environment variable toggling whether to check for parameter/gradient
# writeback in case their storages change after FSDP initialization
# We should check by default since it prevents silent correctness errors, but
# since such changes are atypical, we may want to skip the check to save CPU
# overhead, especially since the check happens in the pre-forward and
# pre-backward each iteration.
_FSDP_SKIP_WRITEBACK_CHECK = "FSDP_SKIP_WRITEBACK_CHECK"

# Env var toggling whether when model is in .eval() mode, should we run in fp32
# or the reduced precision.
_FSDP_USE_FULL_PREC_IN_EVAL = "FSDP_USE_FULL_PREC_IN_EVAL"

# Some value to set padding in tensors to for debuggability
_FLAT_PARAM_PADDING_VALUE = 42

# Environment variables for disabling the all-gather and reduce-scatter
# communication ops for ablation studies. Note that without these communication
# ops the training won't converge, and you probably need to disable correctness
# checks in your model.
_FSDP_USE_FAKE_ALL_GATHER = "FSDP_USE_FAKE_ALL_GATHER"
_FSDP_USE_FAKE_REDUCE = "FSDP_USE_FAKE_REDUCE"


# TODO: Define this for now to avoid circular imports. See if we can remove.
class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.HYBRID_SHARD,
)
NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.SHARD_GRAD_OP,
    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str
    prim_param_name: str  # unprefixed
    prim_module: nn.Module
    prim_module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]
    numel_in_shard: Optional[int]
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g. either
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]` or
    # `param.as_strided((param.numel(),), (1,))[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]
    intra_param_end_idx: Optional[int]  # inclusive


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_strides (Tuple[torch.Size, ...]): Parameter strides of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_contiguities (Tuple[bool, ...]): Parameter `.contiguous` call results
            of this rank's shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """

    param_names: tuple[str, ...]
    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_contiguities: tuple[bool, ...]
    param_numels: tuple[int, ...]
    param_offsets: tuple[tuple[int, int], ...]


class _FlatParameterMeta(_ParameterMeta):
    # Make `isinstance(t, FlatParameter)` return True for custom tensor
    # instances that have the _is_flat_param flag for BC
    def __instancecheck__(self, instance):
        # NB: do NOT test the super implementation
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    """
    This is the flat parameter used by :class:`FullyShardedDataParallel`.

    It is comprised of one or more original parameters, which are flattened and
    concatenated to construct the flat parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.

    NOTE: Since ``use_orig_params=True`` supports intra-``FlatParameter``
    padding, we have two versions of the per-parameter numels, one that
    includes the padding (``_numels_with_padding``) and one that does not
    (``_numels``). The former may have length longer than the other data
    structures, while the latter has the same length as the number of actual
    original parameters like the other per-parameter data structures.

    NOTE: This is not a real class; instead, you will always get a Parameter
    back out if you try to create one of these.  This is similar to the trick
    we implemented for Parameter to get it to work with subclasses; this
    is primarily so that FlatParameter supports combination with FakeTensor.

    Attributes:
        _unpadded_unsharded_size (torch.Size): Unsharded flat parameter's size
            without right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding.
        _padded_unsharded_size (torch.Size): Unsharded flat parameter's size
            with right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding. This
            is only set for sharded strategies since they require padding for
            the all-gather.
        _sharded_size (torch.Size): Sharded flat parameter's size with padding.
            This is also set for ``NO_SHARD``, in which case it is the same as
            the unsharded sizes. (We omit "padded" because there is no
            analogous unpadded one.)

        _num_params (int): Number of original parameters flattened into this
            flat parameter. This is the length of the per-parameter data
            structures.
        _param_infos (Tuple[ParamInfo, ...]): Each parameter's parameter info
            entry; see :class:`ParamInfo` for details.
        _shapes (Tuple[torch.Size, ...]): Each parameter's original shape.
        _strides (Tuple[torch.Size, ...]): Each parameter's original stride.
        _contiguities (Tuple[bool, ...]): Each parameter's ``contiguous()``
            call result.
        _fqns (Tuple[str, ...]): Each parameter's fully-qualified name (FQN)
            prefixed from the ``_fully_sharded_module``. The names are
            guaranteed to be unique in the subtree rooted at that module.
        _param_extensions (Tuple[Optional[Any], ...]): Each parameter's
            extension (i.e. some per-parameter state) used to customize
            pre-flatten and post-unflatten behavior or ``None``. This is
            experimental, and users should not depend on its existence in the
            future.
        _numels_with_padding (Tuple[int, ...]): Each parameter's numel
            including entries for the padding. This is used to construct views
            into the flat parameter via ``torch.split()``. This may have length
            longer than ``_num_params``.
        _numels (Tuple[int, ...]): Each parameter's numel excluding entries for
            padding. This has length equal to ``_num_params``.
        _shard_param_infos (Tuple[_ShardParamInfo, ...]): Each parameter's
            shard parameter info; see :class:`_ShardParamInfo` for details.
        _shared_param_infos (Tuple[SharedParamInfo, ...]): Shared parameter
            info entries; see :class:`SharedParamInfo` for details.
        _modules (set[nn.Module]): Modules that contain some original parameter
            that is flattened into the flat parameter.

        _shard_numel_padded (int): Numel padded for this rank's sharded flat
            parameter.
        _local_shard (Tensor): Sharded flat parameter with padding if using a
            sharded strategy. If using ``NO_SHARD``, then this is the unpadded
            unsharded flat parameter, and there is no notion of a sharded flat
            parameter or padded unsharded flat parameter.
        _full_param_padded (Tensor): Unsharded flat parameter with padding.
            This is not defined for ``NO_SHARD``. When using mixed precision
            for parameters, this has the low precision.
        _full_prec_full_param_padded (Tensor): Full precision unsharded flat
            parameter with padding. This is used for unsharding outside of
            computation when using mixed precision for parameters. This is
            never defined for ``NO_SHARD``.
        _post_backward_hook_handle (RemovableHandle):
            Flat parameter's post-backward hook handle. (Compile only)
        _post_backward_hook_state (Tuple[AccumulateGrad, RemovableHandle]):
            Flat parameter's :class:`AccumulateGrad` object and post-backward
            hook handle. (Eager only)
        _mp_shard (Tensor): Low precision sharded flat parameter with padding.
            This is only defined when parameter mixed precision is enabled. For
            ``NO_SHARD``, this is used for computation.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
            This is only defined when offloading parameters is enabled.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.

        _params (Optional[List[nn.Parameter]]): If ``use_orig_params=True``,
            then each original parameter variable; otherwise, ``None``. This
            does not include any padding tensors.
        _shared_params (Optional[List[nn.Parameter]]): The original shared
            parameter variables if ``use_orig_params=True`` and ``None``
            otherwise.
        _tensors (Optional[List[Optional[Tensor]]]): This saves the ``Tensor``
            views created in the forward and tracked by autograd when
            ``use_orig_params=True`` and is ``None`` otherwise. This is to
            preserve those ``Tensor`` variables for the backward to ensure that
            the ``FlatParameter`` 's ``AccumulateGrad`` object does not change
            in which case the post-backward hook does not run. This is relevant
            for cases like reentrant activation checkpointing.
        _is_grad_none_mask (Optional[List[bool]]): If ``use_orig_params=True``,
            a mask over the original parameters' gradients indicating if it is
            logically ``None`` or not; otherwise, ``None``. This does not
            include entries for padding. This mask is needed because only some
            of the parameters may have ``None`` gradient, in which case the
            flat gradient must be non-``None`` and must use zeros to
            approximate those original ``None`` gradients. This mask informs
            FSDP to set the original parameter gradients to ``None`` (instead
            of zeros) as needed.
    """

    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: tuple[ParamInfo, ...]
    _shapes: tuple[torch.Size, ...]
    _strides: tuple[tuple[int, ...], ...]
    _contiguities: tuple[bool, ...]
    _fqns: tuple[str, ...]
    _param_extensions: tuple[Optional[Any], ...]
    _numels_with_padding: tuple[int, ...]
    _numels: tuple[int, ...]
    _shard_param_infos: tuple[_ShardParamInfo, ...]
    _shared_param_infos: tuple[SharedParamInfo, ...]
    _modules: set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    # Eager only
    _post_backward_hook_state: tuple[Any, Any]
    # Compile only
    _post_backward_hook_handle: Any
    _mp_shard: Tensor
    _cpu_grad: Tensor
    _saved_grad_shard: Tensor
    _params: Optional[list[nn.Parameter]]
    _shared_params: Optional[list[nn.Parameter]]
    _tensors: Optional[list[Optional[Tensor]]]
    _is_grad_none_mask: Optional[list[bool]]

    _is_padding_mask: list[bool]

    def __new__(cls, data=None, requires_grad=True):
        if cls is not FlatParameter:
            raise AssertionError("subclasses FlatParameter not supported")
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        r._is_flat_param = True  # type: ignore[attr-defined]
        return r

    # NB: This is not a regular method, because FlatParameters are not actually
    # instances of this class (see __new__ above).  So you must indirectly
    # call this directly through the classmethod.
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: list[ParamInfo],
        numels: list[int],
        shapes: list[torch.Size],
        strides: list[tuple[int, ...]],
        contiguities: list[bool],
        fqns: list[str],
        shared_param_infos: list[SharedParamInfo],
        param_extensions: list[Optional[Any]],
        params: Optional[list[nn.Parameter]],
        shared_params: Optional[list[nn.Parameter]],
        is_padding_mask: list[bool],
    ) -> None:
        """
        Initialize attributes holding metadata about the original parameters comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
        if len(param_infos) != len(shapes):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match shapes length {len(shapes)}"
            )
        if len(param_infos) != len(strides):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match strides length {len(strides)}"
            )
        if len(param_infos) != len(contiguities):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match contiguities length {len(contiguities)}"
            )
        if len(param_infos) != len(fqns):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match fqns length {len(fqns)}"
            )
        if len(param_infos) != len(param_extensions):
            raise AssertionError(
                f"Expected param_infos length {len(param_infos)} to match param_extensions length {len(param_extensions)}"
            )
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._strides = strides
        self._contiguities = contiguities
        self._fqns = fqns
        self._param_extensions = param_extensions
        self._is_padding_mask = is_padding_mask

        numels_without_padding: list[int] = []
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        self._numels = tuple(numels_without_padding)
        self._numels_with_padding = tuple(numels)
        if len(self._numels) != self._num_params:
            raise AssertionError(
                f"Expected _numels length {len(self._numels)} to equal _num_params {self._num_params}"
            )

        self._shared_param_infos = tuple(shared_param_infos)
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        if (params is None) != (shared_params is None):
            raise AssertionError(
                "Expected params and shared_params to both be None or both be not None"
            )
        if params is not None:
            if shared_params is None or len(shared_params) != len(shared_param_infos):
                raise AssertionError(
                    f"Expected shared_params to be not None and have length {len(shared_param_infos)}, got {shared_params}"
                )
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)
            if shared_params is not None:
                self._shared_params = shared_params
            else:
                self._shared_params = []
            # Mark the original parameters to avoid flattening them into
            # another `FlatParameter` during recursive construction
            for param in chain(self._params, self._shared_params):
                _set_fsdp_flattened(param)
            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            self._tensors = [None for _ in range(self._num_params)]
        else:
            self._params = None
            self._shared_params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        _set_fsdp_flattened(self)
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
        self._post_backward_called = False


class FlatParamHandle:
    """
    A handle that manages a flat parameter (:class:`FlatParameter`).

    This includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to flatten into the
            flat parameter.
        fully_sharded_module (nn.Module): See [Note: Fully Sharded Module].
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        sharding_strategy (ShardingStrategy): Sharding strategy to apply to
            this handle's ``FlatParameter``.
        offload_params (bool): Whether to offload the handle's
            ``FlatParameter`` to CPU.
        mp_param_dtype (Optional[torch.dtype]): Parameter mixed precision
            setting passed to the FSDP constructor.
        mp_reduce_dtype (Optional[torch.dtype]): Gradient reduction mixed
            precision setting passed to the FSDP constructor.
        keep_low_precision_grads (bool): Whether to keep gradients in low
            precision.
        use_orig_params (bool): If ``True``, then FSDP preserves the original
            parameter variables and returns them from ``named_parameters()``
            (e.g. to support different optimizer hyperparameters within one
            :class:`FlatParameter`). If ``False``, then FSDP reconstructs the
            parameters every iteration and returns the :class:`FlatParameter` s
            from ``named_parameters()``.
    """

    ##################
    # INITIALIZATION #
    ##################
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, Tensor]],
        fully_sharded_module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        offload_params: bool,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        keep_low_precision_grads: bool,
        process_group: dist.ProcessGroup,
        use_orig_params: bool,
        *,
        fsdp_extension: Optional[FSDPExtensions] = None,
    ):
        super().__init__()
        params = list(params)
        if len(params) == 0:
            raise ValueError(
                f"Cannot construct a {self.__class__.__name__} with an empty parameter list"
            )
        self._init_setattr_fns()
        self._skip_writeback_check = (
            os.environ.get(_FSDP_SKIP_WRITEBACK_CHECK, "") == "1"
        )
        self._use_full_prec_in_eval = (
            os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
        )
        self._use_fake_all_gather = os.environ.get(_FSDP_USE_FAKE_ALL_GATHER, "") == "1"
        self._use_fake_reduce = os.environ.get(_FSDP_USE_FAKE_REDUCE, "") == "1"
        if self._skip_writeback_check:
            _warn_skip_writeback_check(
                logger,
                f"Since {_FSDP_SKIP_WRITEBACK_CHECK}=1, FSDP will not check "
                "for parameter or gradient writeback. Changing parameter or "
                "gradient storages may lead to silent correctness errors.",
            )
        if self._use_fake_all_gather:
            _warn_use_fake_all_gather(
                logger,
                f"Since {_FSDP_USE_FAKE_ALL_GATHER}=1, FSDP will not execute "
                "all-gather ops. Your training will be incorrect, but "
                "can reveal how much time spent on all-gather ops.",
            )
        if self._use_fake_reduce:
            _warn_use_fake_reduce(
                logger,
                f"Since {_FSDP_USE_FAKE_REDUCE}=1, FSDP will not execute "
                "reduce-scatter ops. Your training will be incorrect, but "
                "can reveal how much time spent on reduce-scatter ops.",
            )
        # Only align addresses for `use_orig_params=True` (for now)
        align_addresses = use_orig_params
        self._init_get_unflat_views_fn(align_addresses)
        # pyrefly: ignore [read-only]
        self.device = device
        self._device_handle = _FSDPDeviceHandle.from_device(self.device)
        self.process_group = process_group
        if self._use_fake_all_gather or self._use_fake_reduce:
            self._fake_process_group = FakeProcessGroup._create_internal(
                rank=process_group.rank(), world_size=process_group.size()
            )
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self._sharding_strategy = sharding_strategy
        self._offload_params = offload_params
        self._use_orig_params = use_orig_params
        self._keep_low_precision_grads = keep_low_precision_grads
        self._training_state = HandleTrainingState.IDLE
        self._debug_level = dist.get_debug_level()
        self._fully_sharded_module = fully_sharded_module
        # For strategies that do not free after forward, we skip using sharded
        # views after forward since the unsharded data exists. We still switch
        # `self.flat_param` to point to the sharded flat parameter since what
        # it points to parameterizes behavior. We use the following attribute
        # to track which tensor data the parameters are unsharded views into.
        self._unsharded_flat_param_for_skipped_views: Optional[Tensor] = None
        # The index in the state's `all_handles`, which must be the
        # same across ranks for the execution order validation to work
        self._handle_index: Optional[int] = None
        # Index in handles_to_pre_forward_order
        self._pre_forward_order_index: Optional[int] = None
        # Index in `handles_post_forward_order`
        self._post_forward_index: Optional[int] = None
        # Used for guarding against mistargeted forward prefetches
        self._needs_pre_forward_unshard = False
        # Used for guarding against mistargeted backward prefetches
        self._needs_pre_backward_unshard = False
        # Was the handle prefetched? Set on successful _prefetch_handle and unshard
        self._prefetched = False
        # Optimistically assume a valid input `params` and set dtype attributes
        # before `_init_flat_param()`, which performs the actual validation
        self._orig_param_dtype = params[0].dtype
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        if self._fwd_bwd_param_dtype is None:
            raise AssertionError("Expected _fwd_bwd_param_dtype to be not None")  # mypy
        self._aligned_numel = (
            _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype)
            if align_addresses
            else 0
        )
        self._fsdp_extension = fsdp_extension
        self._init_flat_param_and_metadata(
            params,
            fully_sharded_module,
            self._aligned_numel,
            use_orig_params,  # type: ignore[arg-type]
        )
        self._use_unsharded_views(as_params=False)

    def __repr__(self):
        return f"FlatParamHandle(flat_param.fqns={self.flat_param._fqns})"

    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, "") == "1"
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool):
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    def _init_flat_param_and_metadata(
        self,
        params: list[Union[Tensor, nn.Parameter]],
        module: nn.Module,
        aligned_numel: int,
        use_orig_params: bool,
    ) -> None:
        """
        Initialize the ``FlatParameter`` and its metadata.

        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: list[ParamInfo] = []
        numels: list[int] = []
        shapes: list[torch.Size] = []
        strides: list[tuple[int, ...]] = []
        contiguities: list[bool] = []
        fqns: list[str] = []
        shared_param_infos: list[SharedParamInfo] = []
        shared_param_memo: dict[
            Union[Tensor, nn.Parameter], tuple[nn.Module, str, str]
        ] = {}
        params_to_flatten: list[Union[Tensor, nn.Parameter]] = []
        shared_params: list[Union[Tensor, nn.Parameter]] = []
        param_extensions: list[Any] = []
        is_padding_mask: list[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    if aligned_numel > 0:
                        numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(
                                numel_to_pad, dtype, False, device
                            )
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    transform_t, extension = _ext_pre_flatten_transform(
                        param,
                        self._fsdp_extension,
                    )
                    param = cast(nn.Parameter, transform_t)
                    param_extensions.append(extension)
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    is_padding_mask.append(False)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    strides.append(param.stride())
                    contiguities.append(_is_truly_contiguous(param))
                    fqn = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    fqns.append(fqn)
                    total_numel += param.numel()
                    total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        if (
            self.rank == 0
            and aligned_numel > 0
            and total_numel != total_numel_without_padding
        ):
            logger.debug(
                "FSDP FlatParameter address alignment created "
                "%s numel of padding (%s vs. %s)",
                total_numel - total_numel_without_padding,
                total_numel,
                total_numel_without_padding,
            )
        if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                if self.rank == 0:
                    logger.info(
                        "FSDP FlatParameter world size divisibility created "
                        "%s numel of padding",
                        numel_to_pad,
                    )
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
        )
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            strides,
            contiguities,
            fqns,
            shared_param_infos,
            param_extensions,
            _convert_to_params(params_to_flatten) if use_orig_params else None,
            _convert_to_params(shared_params) if use_orig_params else None,
            is_padding_mask,
        )

    def _validate_tensors_to_flatten(
        self, tensors: list[Union[Tensor, nn.Parameter]]
    ) -> tuple:
        """Validate the tensors to flatten and returns any necessary metadata."""
        dtype: Optional[torch.dtype] = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        # For `use_orig_params=True`, permit non-uniform `requires_grad`
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is None and not tensor.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if (
                not self._use_orig_params
                and flat_param_requires_grad is not None
                and tensor.requires_grad != flat_param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        if flat_param_requires_grad is None:
            raise AssertionError("Requires non-empty `tensors` list")
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: list[Tensor],
        aligned_numel: int,
    ) -> Tensor:
        """
        Flatten ``tensors`` into a single flat tensor.

        The flattening optionally includes
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: list[Tensor] = []
        if aligned_numel > 0:
            total_numel = 0
            for tensor in tensors:
                numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(
                        numel_to_pad, dtype, False, device
                    )
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                flat_tensors.append(
                    torch.flatten(_detach_if_needed(tensor))
                    if _is_truly_contiguous(tensor)
                    else _detach_if_needed(tensor).as_strided((tensor.numel(),), (1,))
                )
                total_numel += tensor.numel()
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            flat_tensors = [
                torch.flatten(_detach_if_needed(tensor))
                if _is_truly_contiguous(tensor)
                else _detach_if_needed(tensor).as_strided((tensor.numel(),), (1,))
                for tensor in tensors
            ]
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: list[Tensor],
        aligned_numel: int,
        requires_grad: bool,
    ) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(
        self,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
    ) -> None:
        """
        Initialize param and reduce dtypes.

        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        # Save whether these dtypes were specified so that we permit the
        # parameter dtype to change up until the lazy initialization
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if (
            self._low_prec_param_dtype_specified
            and not self._low_prec_reduce_dtype_specified
        ):
            # Special case: infer gradient reduction mixed precision
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = self._fwd_bwd_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        if self._fwd_bwd_param_dtype is None:
            raise AssertionError("Expected _fwd_bwd_param_dtype to be not None")
        if self._reduce_dtype is None:
            raise AssertionError("Expected _reduce_dtype to be not None")

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self):
        """
        Shard the handle's ``FlatParameter``.

        This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                allocated = flat_param._typed_storage()._size() > 0
                if allocated:
                    flat_param._typed_storage()._resize_(0)
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initialize shard-related metadata for this rank's shard of the flat parameter.

        This includes ``_sharded_size``, ``_shard_param_infos``, and ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()  # type: ignore[attr-defined]
        sharded_flat_param_numel = flat_param.numel()  # includes `numel_padded`
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        if len(shard_param_infos) != flat_param._num_params:
            raise AssertionError(
                f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
            )
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> tuple[_ShardParamInfo, ...]:
        """
        Compute the shard metadata based on ``unsharded_start_idx`` and ``unsharded_end_idx`` (inclusive).

        ``unsharded_start_idx`` and ``unsharded_end_idx`` give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        if len(flat_param_offsets) != len(self.flat_param._numels_with_padding):
            raise AssertionError(
                f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
            )
        shard_param_infos: list[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # `unsharded_param_start_idx` and `unsharded_param_end_idx` are indices
        # into the unsharded flat parameter (inclusive) of the given parameter
        for (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in zip(flat_param_offsets, self.flat_param._is_padding_mask):
            if is_padding:
                continue
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                if not (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ):
                    raise AssertionError(
                        f"Invalid `offset_in_shard` of {offset_in_shard} for "
                        f"sharded flat parameter with {sharded_flat_param_numel} numel"
                    )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    @staticmethod
    def _get_unpadded_shard(
        tensor: Tensor,
        rank: int,
        
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/fsdp`, which is part of the **core PyTorch library**.



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
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/distributed/fsdp`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_limiter_utils.py_kw.md_docs.md`](./_limiter_utils.py_kw.md_docs.md)
- [`_optim_utils.py_kw.md_docs.md`](./_optim_utils.py_kw.md_docs.md)
- [`fully_sharded_data_parallel.py_kw.md_docs.md`](./fully_sharded_data_parallel.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`_exec_order_utils.py_docs.md_docs.md`](./_exec_order_utils.py_docs.md_docs.md)
- [`_wrap_utils.py_kw.md_docs.md`](./_wrap_utils.py_kw.md_docs.md)
- [`sharded_grad_scaler.py_docs.md_docs.md`](./sharded_grad_scaler.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_flat_param.py_docs.md_docs.md`
- **Keyword Index**: `_flat_param.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
