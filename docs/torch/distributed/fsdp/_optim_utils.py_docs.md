# Documentation: `torch/distributed/fsdp/_optim_utils.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_optim_utils.py`
- **Size**: 90,432 bytes (88.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import functools
import logging
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, cast, NamedTuple, no_type_check, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _get_param_to_fqns,
    _module_handle,
    _named_parameters_with_duplicates,
    clean_tensor_name,
)
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (
    ShardingStrategy,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import tree_map_only


if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor import ShardedTensor


logger = logging.getLogger(__name__)


@dataclass
class FSDPParamInfo:
    state: _FSDPState
    handle: FlatParamHandle
    param_indices: dict[str, int]
    param_requires_grad: list[bool]


def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    keys = sorted(dictionary.keys())
    for k in keys:
        yield k, dictionary[k]


@dataclass
class _ConsolidatedOptimState:
    """
    This holds the consolidated optimizer state on the target rank. Positive-
    dimension tensor state is communicated across ranks, while zero-dimension
    tensor state and non-tensor state is taken directly from the target rank.

    PyTorch version 1.12 moved to using zero-dimension tensors for scalar
    values, but user implemented optimizers may still use float (i.e. a
    non-tensor). Thus, we support both and handle them identically.

    Attributes:
        tensor_state (Dict[str, torch.Tensor]): Mapping from positive-dimension
            tensor state name to the unsharded flat tensor representing the
            state.
        zero_dim_tensor_state (Dict[str, torch.Tensor]): Mapping from zero-
            dimension tensor state name to its value.
        non_tensor_state (Dict[str, Any]): Mapping from non-tensor state
            name to its value.
    """

    tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    zero_dim_tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_state: dict[str, Any] = field(default_factory=dict)


class _PosDimTensorInfo(NamedTuple):
    """
    Metadata for positive-dimension tensors used internally for
    :meth:`scatter_full_optim_state_dict`.

    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype


class _OptimStateKey(NamedTuple):
    """
    This represents an optimizer state key that may be used commonly across
    ranks. It is based on the unflattened parameter names rather than parameter
    IDs to make it independent of each rank's own optimizer construction.
    """

    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool


def _unflatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool,
) -> list[dict[str, Any]]:
    """
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): Entry for the flat parameter in the
            "state" part of the optimizer state dict.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter if on the target
        rank or an empty :class:`list` otherwise. The final optimizer state
        dict will need to map these entries using the proper unflattened
        parameter IDs.
    """
    if shard_state and not to_save:
        raise AssertionError("If ``shard_state`` is True, ``to_save`` has to be True.")
    consolidated_state = _communicate_optim_state(
        fsdp_param_info,
        flat_param_state,
    )
    if to_save:
        unflat_param_state = _unflatten_communicated_optim_state(
            fsdp_param_info,
            consolidated_state,
            shard_state,
        )
        for optim_state in unflat_param_state:
            # We can't use .items() below cuz we'd run into a concurrent modification error
            if cpu_offload:
                for key in list(optim_state.keys()):
                    state = optim_state[key]
                    if not isinstance(state, torch.Tensor):
                        continue
                    optim_state[key] = state.cpu()
        return unflat_param_state
    else:
        return []


def _is_zero_dim_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 0


def _communicate_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
) -> _ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flat parameter across ranks. All
    ranks will hold the entire non-sharded optimizer state on GPU.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flat parameter.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for the target
        flat parameter.
    """
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    state = _ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )

    for state_name, value in sorted_items(flat_param_state):
        # Positive-dimension tensor state: communicate across ranks
        if torch.is_tensor(value) and value.dim() > 0:
            # If the parameter is not sharded, then neither is the
            # positive-dimension tensor state, so no need to communicate it --
            # we take the target rank's value
            if (
                fsdp_state.world_size == 1
                or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
            ):
                tensor_state[state_name] = value
                continue
            if fsdp_state.compute_device is None:
                raise AssertionError("compute_device has not been initialized")
            if value.device.type != fsdp_state.compute_device.type:
                value = value.to(fsdp_state.compute_device)
            # Assume that positive-dimension tensor optimizer state
            # has the same shape as the sharded flat parameter
            buffer_size = flat_param._full_param_padded.size()  # type: ignore[attr-defined]
            tensor_buffer = value.new_zeros(*buffer_size)
            dist.all_gather_into_tensor(
                tensor_buffer, value, group=fsdp_state.process_group
            )
            fsdp_state._device_handle.synchronize()
            unpadded_numel = cast(
                nn.Parameter, flat_param._unpadded_unsharded_size
            ).numel()
            tensor_state[state_name] = tensor_buffer[:unpadded_numel]
        # Zero-dimension tensor state and non-tensor state: take this rank's
        # value directly
        else:
            if _is_zero_dim_tensor(value):
                zero_dim_tensor_state[state_name] = value.detach().clone()
            else:
                non_tensor_state[state_name] = value
    return state


def _unflatten_communicated_optim_state(
    fsdp_param_info: FSDPParamInfo,
    state: _ConsolidatedOptimState,
    shard_state: bool,
) -> list[dict[str, Any]]:
    """
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flat
    parameter. This should only be called on the target rank.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        state (_ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter. The final
        optimizer state dict will need to map these entries using the proper
        unflattened parameter IDs.
    """
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param
    unflat_param_state: list[dict[str, Any]] = []
    flat_param_views: dict[str, Iterator] = {}
    num_unflat_params = flat_param._num_params
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )

    for _ in range(num_unflat_params):
        unflat_state_param = {}
        # Add positive-dimension tensor state: unflatten with views
        for state_name, flat_tensor in sorted_items(tensor_state):
            views_generated = state_name in flat_param_views
            if not views_generated:
                views = handle._get_unflat_views(flat_tensor)
                flat_param_views[state_name] = views
            else:
                views = flat_param_views[state_name]
            optim_state: Union[torch.Tensor, ShardedTensor, DTensor] = next(views)
            if shard_state:
                osd_config = fsdp_state._optim_state_dict_config
                if getattr(osd_config, "_use_dtensor", False):
                    if fsdp_state._device_mesh is None:
                        raise AssertionError(
                            f"Expected _device_mesh to be not None, got {fsdp_state._device_mesh}"
                        )
                    optim_state = _ext_chunk_dtensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state._device_mesh,
                        fsdp_state._fsdp_extension,
                    )
                else:
                    if fsdp_state.process_group is None:
                        raise AssertionError(
                            f"Expected process_group to be not None, got {fsdp_state.process_group}"
                        )
                    optim_state = _ext_chunk_tensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state.world_size,
                        fsdp_state._device_handle.device_count(),
                        fsdp_state.process_group,
                        fsdp_state._fsdp_extension,
                    )
            unflat_state_param[state_name] = optim_state

        # Add zero-dimension tensor state: take the target rank's value
        unflat_state_param.update(sorted_items(zero_dim_tensor_state))
        # Add non-tensor state: take the target rank's value
        unflat_state_param.update(sorted_items(non_tensor_state))
        unflat_param_state.append(unflat_state_param)
    return unflat_param_state


def _broadcast_processed_state(
    fsdp_state: _FSDPState,
    optim_state: dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> dict[str, Any]:
    objects: list[Any] = [None]
    if dist.get_rank(group) == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    dist.broadcast_object_list(objects, src=0, group=group)
    if dist.get_rank(group) == 0:
        return optim_state
    else:
        return objects[0]


def _broadcast_state(
    fsdp_state: _FSDPState, state: Any, group: Optional[dist.ProcessGroup]
) -> Any:
    if dist.get_rank(group) == 0:
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state
        tensor = state.to(fsdp_state.compute_device)
    else:
        if isinstance(state, torch.Tensor):
            if state.dim() != 0:
                raise AssertionError(
                    "For non-zero ranks, a tensor state should have zero dimension, "
                    f"but got the state with shape {state.shape}."
                )
            return state
        elif not isinstance(state, _PosDimTensorInfo):
            return state
        tensor = torch.zeros(
            state.shape, dtype=state.dtype, device=fsdp_state.compute_device
        )
    dist.broadcast(tensor, src=0, group=group)
    return tensor


def _shard_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    fqn: str,
    optim_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Shard the optimizer state for the original parameter with the name ``fqn``.
    This API should only be used when ``use_orig_params`` is True.
    """
    if not optim_state:
        return {}
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    param_idx = fsdp_param_info.param_indices[fqn]
    shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
    optim_state = _gather_state_dict(
        optim_state, pg=fsdp_state.process_group, device=fsdp_state.compute_device
    )
    if not shard_param_info.in_shard:
        return {}
    # Flatten and shard the state.
    new_optim_state: dict[str, Any] = {}
    intra_param_start_idx = shard_param_info.intra_param_start_idx
    intra_param_end_idx = shard_param_info.intra_param_end_idx
    for state_name, value in optim_state.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            value = value.flatten()[
                intra_param_start_idx : intra_param_end_idx  # type: ignore[operator]
                + 1
            ].clone()
        new_optim_state[state_name] = value
    return new_optim_state


def _flatten_optim_state_dict(
    optim_state_dict: dict[str, Any],
    model: nn.Module,
    use_orig_params: bool = False,
    optim: Optional[torch.optim.Optimizer] = None,
    rank0_only: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()

    unflat_osd = optim_state_dict
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError(
            '`optim_state_dict` must have the keys "state"'
            "to be a valid optimizer state dict"
        )
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True.
    if rank0_only:
        unflat_osd = _broadcast_processed_state(fsdp_state, unflat_osd, group=group)

    # Construct the "state" part
    flat_osd_state: dict[Union[_OptimStateKey, str], Any] = {}
    unflat_osd_state = unflat_osd["state"]
    all_state_keys = set(unflat_osd_state.keys())

    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn]:
                    unflat_osd_state[fqn][state_name] = _broadcast_state(
                        fsdp_state, unflat_osd_state[fqn][state_name], group=group
                    )
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
                    flat_state = _shard_orig_param_state(
                        fsdp_param_info,
                        fqn,
                        unflat_osd_state[fqn],
                    )
            else:
                flat_state = _flatten_optim_state(
                    fsdp_param_info,
                    unflat_osd_state,
                    fqns,
                )
            key = _OptimStateKey(tuple(fqns), True)
            # Only include non-empty states since as expected by
            # `torch.optim.Optimizer` s unless the optimizer is KeyedOptimizer
            # or NamedOptimizer.
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                if len(fqns) != 1:
                    raise AssertionError(
                        f"use_orig_params is True but there are multiple FQNs, {fqns}."
                    )
                if optim is not None:  # NamedOptimizer or KeyedOptimizer case.
                    state = optim.state.get(param, None)  # type: ignore[call-overload]
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(
                            f"optim_state[{key}] is not on rank{fsdp_state.rank}.",
                            stacklevel=2,
                        )

            else:
                raise RuntimeError(
                    f"The state of {key} is empty. This should happen when "
                    "use_orig_params=True."
                )
        else:  # do not flatten non-FSDP parameters' states
            if len(fqns) != 1:
                raise AssertionError(f"Expected len(fqns) == 1, got {len(fqns)}")
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        # Deference the tensor so that PyTorch can collect the memory.
                        del unflat_osd_state[fqn][state_name]
                    else:
                        # Move the tensor in the original osd back to CPU to make the
                        # original osd unaffected.
                        unflat_osd_state[fqn][state_name] = param_state.cpu()

    # Handle user-defined state, states that are not associated with parameters.
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)

    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict() profiling: ")
    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}


def _flatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    unflat_osd_state: dict[str, dict[str, Any]],
    unflat_param_names: list[str],
) -> dict[str, Any]:
    """
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flat parameter in ``fsdp_param_info`` corresponding to the unflattened
    parameter names in ``unflat_param_names``.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flat parameter ``flat_param``.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flat parameter. The sharded optimizer state dict's "state"
        part will map a key to this returned value.
    """
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param
    num_unflat_params = len(unflat_param_names)
    if num_unflat_params <= 0:
        raise AssertionError(
            "Expects at least one unflattened parameter corresponding to the flat parameter"
        )
    unflat_param_shapes = flat_param._shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    if num_unflat_params != num_unflat_param_shapes:
        raise AssertionError(
            f"Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}"
        )

    # Check if these unflattened parameters have any optimizer state
    has_state = [
        bool(unflat_param_name in unflat_osd_state)
        for unflat_param_name in unflat_param_names
    ]
    # If none of the unflattened parameters comprising this flat parameter have
    # any state, then we do not want an entry in the optimizer state dict
    if not any(has_state):
        return {}  # no need to flatten any state
    # There may still be some unflattened parameters with state and some
    # without
    unflat_param_states = [
        _gather_state_dict(
            unflat_osd_state[unflat_param_name],
            pg=fsdp_state.process_group,
            device=fsdp_state.compute_device,
        )
        if unflat_param_name in unflat_osd_state
        else None
        for unflat_param_name in unflat_param_names
    ]
    # Check that the unflattened parameters have the same state names
    state_names = None
    # pyrefly: ignore [bad-assignment]
    for unflat_param_state in unflat_param_states:
        if unflat_param_state is None:
            continue
        if state_names is None:
            state_names = set(unflat_param_state.keys())
        else:
            if state_names != set(unflat_param_state.keys()):
                raise ValueError(
                    "Differing optimizer state names for the unflattened "
                    f"parameters: {unflat_param_names}"
                )
    if state_names is None:
        raise AssertionError(f"Expected state_names to be not None, got {state_names}")

    # Flatten the state
    flat_state: dict[str, Optional[torch.Tensor]] = {}
    for state_name in state_names:
        state_values = [
            unflat_param_state[state_name] if unflat_param_state is not None else None
            for unflat_param_state in unflat_param_states
        ]
        non_none_state_values = [v for v in state_values if v is not None]
        # If all ranks have None, this is a None value
        if not non_none_state_values:
            flat_state[state_name] = None
            continue
        are_pos_dim_tensors = are_zero_dim_tensors = are_non_tensors = True
        for v in non_none_state_values:
            are_pos_dim_tensors &= torch.is_tensor(v) and v.dim() > 0
            are_zero_dim_tensors &= _is_zero_dim_tensor(v)
            are_non_tensors &= not torch.is_tensor(v)
        types = {type(v) for v in non_none_state_values}
        if len(types) != 1 or not (
            are_pos_dim_tensors or are_zero_dim_tensors or are_non_tensors
        ):
            raise ValueError(
                f"Differing optimizer state types for state {state_name}, "
                f"values {non_none_state_values}, and unflattened parameter "
                f"names {unflat_param_names}"
            )
        if are_pos_dim_tensors:
            flat_tensor = _flatten_tensor_optim_state(
                state_name,
                state_values,  # type: ignore[arg-type]
                unflat_param_names,
                unflat_param_shapes,
                handle,
            )
            # Shard the flattened tensor immediately to minimize max memory
            # usage
            if (
                fsdp_state.world_size != 1
                and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
            ):
                sharded_flat_tensor, _ = FlatParamHandle._get_shard(
                    flat_tensor,
                    fsdp_state.rank,
                    fsdp_state.world_size,
                )
            else:
                sharded_flat_tensor = flat_tensor
            flat_state[state_name] = sharded_flat_tensor
        elif are_zero_dim_tensors:
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(
                state_name,
                state_values,  # type: ignore[arg-type]
                unflat_param_names,
            )
        else:
            if not are_non_tensors:
                raise AssertionError(
                    f"Expected are_non_tensors to be True, got {are_non_tensors}"
                )
            flat_state[state_name] = _flatten_non_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )

    return flat_state


def _flatten_tensor_optim_state(
    state_name: str,
    pos_dim_tensors: list[torch.Tensor],
    unflat_param_names: list[str],
    unflat_param_shapes: Sequence[torch.Size],
    handle: FlatParamHandle,
) -> torch.Tensor:
    """
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flat parameter
    from ``handle`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter's tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam's "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flat parameter.
        handle (FlatParamHandle): The flat parameter's handle.

    Returns:
        torch.Tensor: A flat tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    """
    flat_param = handle.flat_param
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    # Check that all are tensors with the same dtype
    dtypes = {t.dtype for t in non_none_tensors}
    if len(dtypes) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have positive-dimension tensor state with the "
            f"same dtype but got dtypes {dtypes} for state {state_name} and "
            f"unflattened parameter names {unflat_param_names}"
        )
    dtype = next(iter(dtypes))
    # Check that each tensor state matches its parameter's shape
    for tensor, shape in zip(pos_dim_tensors, unflat_param_shapes):
        if tensor is None and len(shape) == 0:
            raise ValueError("Flattening a zero-dimension parameter is not supported")
        elif tensor is not None and tensor.shape != shape:
            raise ValueError(
                "Tensor optimizer state does not have same shape as its "
                f"parameter: {tensor.shape} {shape}"
            )
    # Flatten the tensor states: we do not need to add any right-hand-side
    # padding since the flat optimizer state tensor is sharded via
    # `_get_shard()`, which pads the shard as needed (just like for the flat
    # parameter)
    cpu_device = torch.device("cpu")
    tensors_to_flatten = [
        torch.flatten(state_value.to(cpu_device))
        if state_value is not None
        else torch.flatten(
            torch.zeros(
                size=shape,
                dtype=dtype,
                device=cpu_device,
            )
        )
        for state_value, shape in zip(pos_dim_tensors, unflat_param_shapes)
    ]
    flat_tensor = handle.flatten_tensors(tensors_to_flatten, handle._aligned_numel)
    flat_param_shape = flat_param._unpadded_unsharded_size  # type: ignore[attr-defined]
    if flat_tensor.shape != flat_param_shape:
        raise AssertionError(
            f"tensor optim state: {flat_tensor.shape} flat parameter: {flat_param_shape}"
        )
    return flat_tensor


def _flatten_zero_dim_tensor_optim_state(
    state_name: str,
    zero_dim_tensors: list[torch.Tensor],
    unflat_param_names: list[str],
) -> torch.Tensor:
    """
    Flattens the zero-dimension tensor optimizer state given by the values
    ``zero_dim_tensors`` for the state ``state_name`` for a single flat
    parameter corresponding to the unflattened parameter names
    ``unflat_param_names`` by enforcing that all tensors are the same and using
    that common value.

    NOTE: The requirement that the tensors are the same across all unflattened
    parameters comprising the flat parameter is needed to maintain the
    invariant that FSDP performs the same computation as its non-sharded
    equivalent. This means that none of the unflattened parameters can be
    missing this state since imposing a value may differ from having no value.
    For example, for Adam's "step", no value means maximum bias correction,
    while having some positive value means less bias correction.

    Args:
        state_name (str): Optimizer state name.
        zero_dim_tensors (List[torch.Tensor]): Zero-dimension optimizer state
            for the unflattened parameters corresponding to the single
            flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        torch.Tensor: A zero-dimensional tensor giving the value of the state
        ``state_name`` for all unflattened parameters corresponding to the
        names ``unflat_param_names``.
    """
    non_none_tensors = [t for t in zero_dim_tensors if t is not None]
    # Enforce that all have the same value and dtype
    values_set = {t.item() if t is not None else None for t in zero_dim_tensors}
    dtypes = {t.dtype if t is not None else None for t in zero_dim_tensors}
    if (
        len(non_none_tensors) != len(zero_dim_tensors)
        or len(values_set) != 1
        or len(dtypes) != 1
    ):
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {values_set} and dtypes {dtypes} for state "
            f"{state_name} and unflattened parameter names "
            f"{unflat_param_names}"
        )
    value = next(iter(values_set))
    dtype = next(iter(dtypes))
    return torch.tensor(value, dtype=dtype, device=torch.device("cpu"))


def _flatten_non_tensor_optim_state(
    state_name: str,
    non_tensors: list[Any],
    unflat_param_names: list[str],
) -> Any:
    """
    Flattens the non-tensor optimizer state given by the values ``non_tensors``
    for the state ``state_name`` for a single flat parameter corresponding
    to the unflattened parameter names ``unflat_param_names`` by enforcing that
    all values are the same and using that common value.

    See the note in :func:`_flatten_zero_dim_tensor_optim_state`.

    Args:
        state_name (str): Optimizer state name.
        non_tensors (List[Any]): Non-tensor optimizer state for the unflattened
            parameters corresponding to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        Any: A non-tensor giving the value of the state ``state_name`` for all
        unflattened parameters corresponding to the names
        ``unflat_param_names``.
    """
    non_none_non_tensors = [nt for nt in non_tensors if nt is not None]
    # Enforce that all have the same value (same type already checked)
    non_tensor_set = set(non_tensors)
    if len(non_none_non_tensors) != len(non_tensors) or len(non_tensor_set) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {non_tensor_set} for state {state_name} and  "
            f"unflattened parameter names {unflat_param_names}"
        )
    non_tensor = next(iter(non_tensor_set))
    return non_tensor


def _rekey_sharded_optim_state_dict(
    sharded_osd: dict[str, Any],
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_input: Optional[
        Union[
            list[dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ],
    using_optim_input: bool,
    is_named_optimizer: bool = False,
) -> dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to flat
    parameter IDs according to the calling rank's ``optim``, which may be
    different across ranks. In particular, the unflattened parameter names are
    represented as :class:`_OptimStateKey` s.
    """
    param_to_fqns = _get_param_to_fqns(model)
    flat_param_to_fqn = _get_flat_param_to_fqn(model)
    param_to_param_key: dict[nn.Parameter, Union[int, str]] = cast(
        dict[nn.Parameter, Union[int, str]],
        (
            _get_param_to_param_id_from_optim_input(model, optim_input)
            if using_optim_input
            else _get_param_to_param_key(
                optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
            )
        ),
    )
    # All parameter keys in `param_to_param_key` should be in
    # `param_to_fqns` -- strict inequality follows when not all parameters are
    # passed to the optimizer
    if len(param_to_param_key) > len(param_to_fqns):
        raise AssertionError(
            f"Expected len(param_to_param_key) <= len(param_to_fqns), got {len(param_to_param_key)} > {len(param_to_fqns)}"
        )

    unflat_param_names_to_flat_param_key: dict[
        tuple[str, ...], Union[int, str]
    ] = {}  # for "state"
    unflat_param_name_to_flat_param_key: dict[
        str, Union[int, str]
    ] = {}  # for "param_groups"
    for param, unflat_param_names in param_to_fqns.items():
        if param not in param_to_param_key:
            # This parameter was not passed to the optimizer
            continue
        flat_param_key = param_to_param_key[param]
        unflat_param_names_to_flat_param_key[tuple(unflat_param_names)] = flat_param_key
        for unflat_param_name in unflat_param_names:
            unflat_param_name_to_flat_param_key[unflat_param_name] = flat_param_key

    sharded_osd_state = sharded_osd["state"]
    rekeyed_osd_state: dict[Union[str, int], Any] = {}
    for key, param_state in sharded_osd_state.items():
        if isinstance(key, str):
            rekeyed_osd_state[key] = param_state
            continue
        flat_param_key = unflat_param_names_to_flat_param_key.get(
            key.unflat_param_names, key.unflat_param_names
        )
        # pyrefly: ignore [unsupported-operation]
        rekeyed_osd_state[flat_param_key] = param_state

    # Only process param_groups if it exists in sharded_osd
    if "param_groups" in sharded_osd:
        rekeyed_osd_param_groups: list[dict[str, Any]] = []
        for unflat_param_group in sharded_osd["param_groups"]:
            flat_param_group = copy.deepcopy(unflat_param_group)
            flat_param_keys = sorted(
                {
                    unflat_param_name_to_flat_param_key[unflat_param_name]
                    for unflat_param_name in unflat_param_group["params"]
                }
            )
            flat_param_group["params"] = flat_param_keys
            rekeyed_osd_param_groups.append(flat_param_group)
        return {"state": rekeyed_osd_state, "param_groups": rekeyed_osd_param_groups}
    else:
        return {"state": rekeyed_osd_state}


def _get_param_id_to_param_from_optim_input(
    model: nn.Module,
    optim_input: Optional[
        Union[
            list[dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ] = None,
) -> dict[int, nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

    NOTE: This method is only preserved for backward compatibility. The method
    :meth:`_get_param_key_to_param` is the preferred code path that does not
    rely on ``optim_input``.

    NOTE: We critically assume that, whether the optimizer input is a list of
    parameters or a list of parameter groups, :class:`torch.optim.Optimizer`
    enumerates the parameter IDs in order. In other words, for a parameter list
    input, the parameter IDs should be in that list order, and for a parameter
    groups input, the parameter IDs should be in order within each parameter
    group and in order across parameter groups.

    Args:
        model (nn.Module): Model whose parameters are passed into the
            optimizer.
        optim_input (Optional[Union[List[Dict[str, Any]],
        Iterable[nn.Parameter]]]): Input passed into the optimizer
            representing either a :class:`list` of parameter groups or an
            iterable of parameters; if ``None``, then this method assumes the
            input was ``model.parameters()``. (Default: ``None``)

    Returns:
        List[nn.Parameter]: Mapping from parameter IDs to parameters,
        where the parameter ID is implicitly the index in the :class:`list`.
    """
    # Assume the standard case of passing `model.parameters()` to the optimizer
    # if `optim_input` is not specified
    if optim_input is None:
        return dict(enumerate(model.parameters()))
    try:
        # pyrefly: ignore [no-matching-overload]
        # pyrefly: ignore [redundant-cast]
        params = cast(list[nn.Parameter], list(optim_input))
    except TypeError as e:
        raise TypeError(
            "Optimizer input should be an iterable of Tensors or dicts, "
            f"but got {optim_input}"
        ) from e
    if len(params) == 0:
        raise ValueError("Optimizer input should not be empty")

    # Check if the optimizer input represents tensors or parameter groups
    all_tensors = True
    all_dicts = True
    for param in params:
        all_tensors &= isinstance(param, torch.Tensor)
        all_dicts &= isinstance(param, dict)
    if not all_tensors and not all_dicts:
        raise TypeError("Optimizer input should be an iterable of Tensors or dicts")
    if all_tensors:
        return dict(enumerate(params))
    if not all_dicts:
        raise AssertionError(f"Expected all_dicts to be True, got {all_dicts}")
    param_id_to_param: list[nn.Parameter] = []
    for param_group in params:
        has_params_key = "params" in param_group  # type: ignore[operator]
        if not has_params_key:
            raise AssertionError(
                'A parameter group should map "params" to a list of the parameters in the group'
            )
        # Implicitly map `flat_param_id` (current length of the list) to
        # `param`
        param_id_to_param.extend(param_group["params"])  # type: ignore[index]
    return dict(enumerate(param_id_to_param))


def _get_flat_param_to_fqn(model: torch.nn.Module) -> dict[FlatParameter, str]:
    """
    Constructs a mapping from ``FlatParameter`` to a cleaned (devoid of prefixes
    from wrappers) fully qualified name (FQN). Note that this FQN is "non-canonical"
    because ``FlatParameter``  s do not come from the original module but are
    registered only after FSDP has been applied. This function returns the FSDP-given
    name for the ``FlatParameter`` (usually module._flat_param) as opposed to the
    canonical FQNs returned for ``FlatParameter`` s in ``_common_utils._get_param_to_fqns(...)``).

    Consequently, this function will only return a non-empty mapping if FSDP was
    applied with ``use_orig_params=False`` as, otherwise, the original parameters
    are used within the module and there would be no ``FlatParameter`` s in the module.

    """

    def module_fn(module, prefix, tree_level, flat_param_to_fqn):
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            if not isinstance(param, FlatParameter):
                continue
            fqn = clean_tensor_name(prefix + param_name)
            flat_param_to_fqn[param] = fqn

    def return_fn(flat_param_to_fqn):
        return flat_param_to_fqn

    flat_param_to_fqn_ret: dict[FlatParameter, str] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [fqn for fqn, _ in _named_parameters_with_duplicates(model)],
        flat_param_to_fqn_ret,
    )


def _get_param_key_to_param(
    optim: torch.optim.Optimizer,
    model: Optional[nn.Module] = None,
    is_named_optimizer: bool = False,
    param_to_fqns: Optional[dict[nn.Parameter, list[str]]] = None,
    flat_param_to_fqn: Optional[dict[FlatParameter, str]] = None,
) -> dict[Union[int, str], nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """
    clean_fqn_to_curr_fqn: dict[str, str] = {}
    if is_named_optimizer:
        if param_to_fqns is None or flat_param_to_fqn is None:
            raise AssertionError(
                "The optimizer is a NamedOptimizer, `param_to_fqns` must not be None."
            )
        if model is None:
            raise AssertionError(f"Expected model to be not None, got {model}")
        for key, _ in _named_parameters_with_duplicates(model):
            clean_fqn_to_curr_fqn[clean_tensor_name(key)] = key

    param_key_to_param: dict[Union[str, int], nn.Parameter] = {}
    pid = 0
    for param_group in optim.param_groups:
        if is_named_optimizer:
            for param in param_group["params"]:
                if flat_param_to_fqn is None:
                    raise AssertionError(
                        f"Expected flat_param_to_fqn to be not None, got {flat_param_to_fqn}"
                    )
                if param in flat_param_to_fqn:
                    # FlatParameter case
                    key = flat_param_to_fqn[param]
                else:
                    if param_to_fqns is None:
                        raise AssertionError(
                            f"Expected param_to_fqns to be not None, got {param_to_fqns}"
                        )
                    # use_orig_params case
                    if len(param_to_fqns[param]) != 1:
                        raise AssertionError(
                            f"Expected len(param_to_fqns[param]) == 1, got {len(param_to_fqns[param])}"
                        )
                    key = param_to_fqns[param][0]
                try:
                    key = clean_fqn_to_curr_fqn[key]
                except KeyError as e:
                    raise KeyError(
                        f"Can't find {key} from {list(clean_fqn_to_curr_fqn.keys())}."
                    ) from e
                param_key_to_param[key] = param
        else:
            for param in param_group["params"]:
                param_key_to_param[pid] = param
                pid += 1

    return param_key_to_param


def _get_param_to_param_key(
    optim: torch.optim.Optimizer,
    model: Optional[nn.Module] = None,
    is_named_optimizer: bool = False,
    param_to_fqns: Optional[dict[nn.Parameter, list[str]]] = None,
    flat_param_to_fqn: Optional[dict[FlatParameter, str]] = None,
) -> dict[nn.Parameter, Union[int, str]]:
    """
    Constructs the inverse mapping of :func:`_get_param_key_to_param`. This API
    only supports the case where `optim` is a regular optimizer, not NamedOptimizer.
    So the parameter keys will be parameter ids.
    """
    param_id_to_param = _get_param_key_to_param(
        optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
    )
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _get_param_to_param_id_from_optim_input(
    model: nn.Module,
    optim_input: Optional[
        Union[
            list[dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ] = None,
) -> dict[nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param_from_optim_input`."""
    param_id_to_param = _get_param_id_to_param_from_optim_input(model, optim_input)
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _check_missing_keys_on_rank(
    r0_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, Union[str, int]],
    param_key_to_param: dict[Union[str, int], nn.Parameter],
    group: Optional[dist.ProcessGroup],
) -> None:
    # Ensure that all ranks have at least the optimizer states needed by
    # rank 0's optimizer
    missing_keys: list[_OptimStateKey] = []
    for r0_optim_state_key in r0_optim_state_keys:
        if r0_optim_state_key not in optim_state_key_to_param_key:
            # A parameter from rank 0's optimizer does not exist for this
            # rank's optimizer
            missing_keys.append(r0_optim_state_key)
            continue
        param_key = optim_state_key_to_param_key[r0_optim_state_key]
        if isinstance(param_key, int):
            if not (param_key >= 0 and param_key < len(param_key_to_param)):
                raise AssertionError("Check the `param_key_to_param` construction")
    # We cannot use FSDPState.compute_device as this API is a global view.
    device = _get_pg_default_device(group)
    num_missing = torch.tensor([len(missing_keys)], dtype=torch.int32, device=device)
    dist.all_reduce(num_missing, group=group)
    if num_missing.item() > 0:
        obj_list = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(obj_list, missing_keys, group=group)
        error_msg = (
            "FSDP currently requires each rank to have at least the "
```



## High-Level Overview


This Python file contains 5 class(es) and 37 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FSDPParamInfo`, `_ConsolidatedOptimState`, `_PosDimTensorInfo`, `_OptimStateKey`, `StateInfo`

**Functions defined**: `sorted_items`, `_unflatten_optim_state`, `_is_zero_dim_tensor`, `_communicate_optim_state`, `_unflatten_communicated_optim_state`, `_broadcast_processed_state`, `_broadcast_state`, `_shard_orig_param_state`, `_flatten_optim_state_dict`, `_flatten_optim_state`, `_flatten_tensor_optim_state`, `_flatten_zero_dim_tensor_optim_state`, `_flatten_non_tensor_optim_state`, `_rekey_sharded_optim_state_dict`, `_get_param_id_to_param_from_optim_input`, `_get_flat_param_to_fqn`, `module_fn`, `return_fn`, `_get_param_key_to_param`, `_get_param_to_param_key`

**Key imports**: copy, functools, logging, warnings, Iterable, Iterator, Sequence, ExitStack, dataclass, field, chain, Any, cast, NamedTuple, no_type_check, Optional, TYPE_CHECKING, Union, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `functools`
- `logging`
- `warnings`
- `collections.abc`: Iterable, Iterator, Sequence
- `contextlib`: ExitStack
- `dataclasses`: dataclass, field
- `itertools`: chain
- `typing`: Any, cast, NamedTuple, no_type_check, Optional, TYPE_CHECKING, Union
- `torch`
- `torch.distributed as dist`
- `torch.distributed.fsdp._traversal_utils as traversal_utils`
- `torch.nn as nn`
- `torch.distributed._state_dict_utils`: _gather_state_dict
- `torch.distributed.distributed_c10d`: _get_pg_default_device
- `torch.distributed.fsdp._debug_utils`: SimpleProfiler
- `torch.distributed.fsdp._flat_param`: FlatParameter, FlatParamHandle
- `torch.distributed.tensor`: DTensor, Replicate
- `torch.utils._pytree`: tree_map_only
- `torch.distributed._shard.sharded_tensor`: ShardedTensor


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `_optim_utils.py_docs.md`
- **Keyword Index**: `_optim_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
