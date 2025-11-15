# Documentation: `torch/distributed/distributed_c10d.py`

## File Metadata

- **Path**: `torch/distributed/distributed_c10d.py`
- **Size**: 247,592 bytes (241.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Distributed Collective Communication (c10d)."""

import collections.abc
import contextlib
import copy
import ctypes
import hashlib
import io
import itertools
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from collections.abc import Callable
from datetime import timedelta
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import deprecated

import torch
from torch._C import _DistStoreError as DistStoreError
from torch._C._distributed_c10d import (
    _DistributedBackendOptions,
    _register_process_group,
    _resolve_process_group,
    _unregister_all_process_groups,
    _unregister_process_group,
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    DebugLevel,
    GatherOptions,
    get_debug_level,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    Work,
)
from torch._utils_internal import set_pytorch_distributed_envs_from_justknobs
from torch.monitor import _WaitCounter
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._typing_utils import not_none

from .c10d_logger import _exception_logger, _time_logger
from .constants import default_pg_nccl_timeout, default_pg_timeout
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401


__all__ = [
    "Backend",
    "BackendConfig",
    "GroupMember",
    "P2POp",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_object",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "send_object_list",
    "recv_object_list",
    "broadcast_object_list",
    "destroy_process_group",
    "gather",
    "gather_object",
    "get_backend_config",
    "get_backend",
    "get_default_backend_for_device",
    "get_rank",
    "get_world_size",
    "get_pg_count",
    "group",
    "init_process_group",
    "irecv",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_backend_available",
    "is_nccl_available",
    "is_torchelastic_launched",
    "is_ucc_available",
    "is_xccl_available",
    "isend",
    "monitored_barrier",
    "new_group",
    "new_subgroups",
    "new_subgroups_by_enumeration",
    "recv",
    "reduce",
    "reduce_scatter",
    "scatter",
    "scatter_object_list",
    "send",
    "supports_complex",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "AllToAllOptions",
    "BarrierOptions",
    "BroadcastOptions",
    "GatherOptions",
    "PrefixStore",
    "ProcessGroup",
    "ReduceOp",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    "Store",
    "DebugLevel",
    "get_debug_level",
    "Work",
    "default_pg_timeout",
    "get_group_rank",
    "get_global_rank",
    "get_process_group_ranks",
    "reduce_op",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "get_node_local_rank",
    "split_group",
    "shrink_group",
]

_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
_UCC_AVAILABLE = True
_XCCL_AVAILABLE = True

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


# Change __module__ of all imported types from torch._C._distributed_c10d that are public
def _export_c_types() -> None:
    _public_types_to_change_module = [
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        BarrierOptions,
        BroadcastOptions,
        GatherOptions,
        PrefixStore,
        ProcessGroup,
        ReduceOp,
        ReduceOptions,
        ReduceScatterOptions,
        ScatterOptions,
        Store,
        DebugLevel,
        get_debug_level,
        Work,
    ]
    for type in _public_types_to_change_module:
        type.__module__ = "torch.distributed.distributed_c10d"


_export_c_types()

try:
    from torch._C._distributed_c10d import ProcessGroupMPI

    ProcessGroupMPI.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupMPI"]
except ImportError:
    _MPI_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupNCCL

    ProcessGroupNCCL.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupNCCL"]
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from torch._C._distributed_c10d import _ProcessGroupWrapper, ProcessGroupGloo

    ProcessGroupGloo.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupGloo"]
except ImportError:
    _GLOO_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupUCC

    ProcessGroupUCC.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupUCC"]
except ImportError:
    _UCC_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupXCCL

    ProcessGroupXCCL.__module__ = "torch.distributed.distributed_c10d"
    __all__ += ["ProcessGroupXCCL"]
except ImportError:
    _XCCL_AVAILABLE = False

logger = logging.getLogger(__name__)

PG_WRAPPER_STORE_PREFIX = "pg_wrapper"


# Some reduce ops are not supported by complex numbers and will result in an error.
# We currently provide complex support to the distributed API by viewing
# complex tensors as real (torch.view_as_real), meaning that calling
# these unsupported ops will return garbage values rather than error out.
# (e.g. max(2+3i, 3+2i) = 3+3i)
# We'd like calls to unsupported ops to error out accordingly,
# rather than returning garbage values.
def supports_complex(reduceOp: ReduceOp) -> bool:
    """Return true if reduce ops is supported. False otherwise."""
    denyList = [
        ReduceOp.MAX,
        ReduceOp.MIN,
        ReduceOp.PRODUCT,
        ReduceOp.BAND,
        ReduceOp.BOR,
        ReduceOp.BXOR,
    ]
    return reduceOp not in denyList


# TODO refactor into enum/strenum
class Backend(str):  # noqa: SLOT000
    """
    An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """

    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"

    _BackendPlugin = namedtuple("_BackendPlugin", ["creator_fn", "extended_api"])

    _plugins: dict[str, _BackendPlugin] = {}

    backend_list = [UNDEFINED, GLOO, NCCL, XCCL, UCC, MPI]

    # 3rd-party devices can register the default backend support here
    default_device_backend_map: dict[str, str] = {
        "cpu": GLOO,
        "cuda": NCCL,
        "xpu": XCCL,
        "mps": GLOO,
    }

    backend_capability: dict[str, list[str]] = {
        GLOO: ["cpu", "cuda"],
        NCCL: ["cuda"],
        XCCL: ["xpu"],
        UCC: ["cpu", "cuda"],
        MPI: ["cpu", "cuda"],
    }

    backend_type_map: dict[str, ProcessGroup.BackendType] = {
        UNDEFINED: ProcessGroup.BackendType.UNDEFINED,
        GLOO: ProcessGroup.BackendType.GLOO,
        NCCL: ProcessGroup.BackendType.NCCL,
        XCCL: ProcessGroup.BackendType.XCCL,
        UCC: ProcessGroup.BackendType.UCC,
        MPI: ProcessGroup.BackendType.MPI,
    }

    def __new__(cls, name: str):
        """Create and return a new instance of the class."""
        if not isinstance(name, str):
            raise ValueError("Backend constructor parameter must be string-ish")
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value == Backend.UNDEFINED:
            value = name.lower()
        return value

    @classmethod
    def register_backend(
        cls,
        name,
        func,
        extended_api=False,
        devices: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """
        Register a new backend with the given name and instantiating function.

        This class method is used by 3rd party ``ProcessGroup`` extension to
        register new backends.

        Args:
            name (str): Backend name of the ``ProcessGroup`` extension. It
                        should match the one in ``init_process_group()``.
            func (function): Function handler that instantiates the backend.
                             The function should be implemented in the backend
                             extension and takes four arguments, including
                             ``store``, ``rank``, ``world_size``, and ``timeout``.
            extended_api (bool, optional): Whether the backend supports extended argument structure.
                                           Default: ``False``. If set to ``True``, the backend
                                           will get an instance of ``c10d::DistributedBackendOptions``, and
                                           a process group options object as defined by the backend implementation.
            device (str or list of str, optional): device type this backend
                            supports, e.g. "cpu", "cuda", etc. If `None`,
                            assuming both "cpu" and "cuda"

        .. note:: This support of 3rd party backend is experimental and subject to change.

        """
        # This takes care of CUSTOM Out-of-tree backend types, update in backend_list indicates availability
        if not hasattr(Backend, name.upper()):
            setattr(Backend, name.upper(), name.lower())
        if name.lower() not in Backend.backend_list:
            Backend.backend_list.append(name.lower())

        if devices is not None:
            for device in devices:
                if device not in Backend.default_device_backend_map:
                    Backend.default_device_backend_map[device] = name.lower()
        Backend.backend_type_map[name.lower()] = ProcessGroup.BackendType.CUSTOM

        # Update device capability matrix in Backend class
        if devices is None:
            # This is more of a backward support for groups like `threaded`:
            # assume default devices "cpu" and "cuda", but warn
            warnings.warn(
                f"Device capability of {name} unspecified, assuming `cpu` and "
                "`cuda` or `xpu`. Please specify it via the `devices` argument of "
                "`register_backend`.",
                stacklevel=2,
            )
            Backend.backend_capability[name.lower()] = (
                ["cpu", "cuda", "xpu"] if torch.xpu.is_available() else ["cpu", "cuda"]
            )
        elif isinstance(devices, str):
            # Single device string specified. Simply convert to list.
            Backend.backend_capability[name.lower()] = [devices]
        else:
            Backend.backend_capability[name.lower()] = devices

        Backend._plugins[name.upper()] = Backend._BackendPlugin(func, extended_api)


class BackendConfig:
    """Backend configuration class."""

    def __init__(self, backend: Backend):
        """Init."""
        self.device_backend_map: dict[str, Backend] = {}
        # pyrefly: ignore [bad-assignment]
        backend = str(backend)

        if backend == Backend.UNDEFINED:
            # Detect the accelerator on the machine. If no accelerator is
            # available, it returns CPU.
            device_type = torch._C._get_accelerator().type
            try:
                backend_str = Backend.default_device_backend_map[device_type]
                self.device_backend_map[device_type] = Backend(backend_str)
            except KeyError:
                raise ValueError(
                    f"We detected accelerator {device_type} on your machine. "
                    f"But we don't know which communication backend to use for this accelerator. "
                    f"Please specify the `backend` argument in the `init_process_group` call."
                ) from None
        elif backend.lower() in Backend.backend_list:
            # Cases for when backend is a single string (without device types)
            # e.g. "nccl", "gloo", "ucc", "mpi"
            supported_devices = Backend.backend_capability[backend.lower()]
            backend_val = Backend(backend)

            self.device_backend_map = dict.fromkeys(supported_devices, backend_val)
        elif ":" in backend.lower():
            # Backend specified in "device:backend" format
            # make sure the backend string is in the correct format
            # "{device_type1}:{backend1},{device_type2}:{backend2}"
            # e.g. "cpu:gloo,cuda:nccl"
            backend_str_error_message = f"""The custom backend string argument is invalid: {backend}.
                Custom backend string is an experimental feature where the backend string must be in the format:
                "<device_type1>:<backend1>,<device_type2>:<backend2>...". e.g. 'cpu:gloo,cuda:nccl'"""

            # parse the backend string and populate the device_backend_map
            for device_backend_pair_str in backend.lower().split(","):
                device_backend_pair = device_backend_pair_str.split(":")
                if len(device_backend_pair) != 2:
                    raise ValueError(
                        f"Invalid device:backend pairing: \
                                     {device_backend_pair_str}. {backend_str_error_message}"
                    )
                # pyrefly: ignore [bad-assignment]
                device, backend = device_backend_pair
                if device in self.device_backend_map:
                    raise ValueError(
                        f"Duplicate device type {device} \
                                     in backend string: {backend}. {backend_str_error_message}"
                    )
                self.device_backend_map[device] = Backend(backend)
        else:
            # User specified a single backend name whose device capability is
            # unknown, assuming it can support the default devices of PyTorch
            # (cpu and cuda)
            warnings.warn(
                f"Device capability of {backend} unknown, assuming `cpu` and "
                "`cuda`. You can specify it in `device:backend` format in "
                "`init_process_group` call.",
                stacklevel=2,
            )
            backend_val = Backend(backend)
            self.device_backend_map = {
                "cpu": backend_val,
                "cuda": backend_val,
                "xpu": backend_val,
            }

        logger.info("Using backend config: %s", self.device_backend_map)

    def __repr__(self):
        """Return all the device:backend pairs separated by commas."""
        return ",".join(
            f"{device}:{backend}" for device, backend in self.device_backend_map.items()
        )

    def get_device_backend_map(self) -> dict[str, Backend]:
        """Return backend map of the device."""
        return self.device_backend_map


class _reduce_op:
    r"""
    Deprecated enum-like class.

    For reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """

    def __init__(self) -> None:
        # __members__ is a dict storing key-value pairs for enum classes
        for k, v in ReduceOp.RedOpType.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.RedOpType.__members__

    @deprecated(
        "`torch.distributed.reduce_op` is deprecated, "
        "please use `torch.distributed.ReduceOp` instead",
        category=FutureWarning,
    )
    def __getattribute__(self, key):
        return object.__getattribute__(self, key)


reduce_op = _reduce_op()


class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int, optional): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
        group_peer (int, optional): Destination or source rank.
    """

    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_peer: Optional[int] = None,
    ):
        """Init."""
        self.op = op
        self.tensor = tensor
        self.group = _group_or_default_group(group)
        self.peer = _canonicalize_group_rank(
            self.group, peer, group_peer, return_global=True
        )
        self.tag = tag
        self.group_peer = _canonicalize_group_rank(self.group, peer, group_peer)

    def __new__(
        cls,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        tag: int = 0,
        group_peer: Optional[int] = None,
    ):
        """Create and return a new instance of the class."""
        _check_op(op)
        _check_single_tensor(tensor, "tensor")

        return object.__new__(cls)

    def __repr__(self):
        my_group_rank = get_rank(self.group)
        op_name = self.op.__name__
        group_name = self.group.group_name if self.group else "default_pg"
        if "send" in op_name:
            s = my_group_rank
            d = self.group_peer
        elif "recv" in op_name:
            s = self.group_peer
            d = my_group_rank
        else:
            return super().__repr__()

        return f"P2POp({op_name} pg={group_name}, group_src={s}, group_dst={d},  {self.tensor.shape}, {self.tensor.dtype})"


class _CollOp:
    """
    A class to capture collective operations.

    Args:
        op (Callable): A collective function, e.g. ``torch.distributed.all_reduce``.
        tensor (Tensor): Tensor to operate on.
        dst_tensor (Tensor, optional): Provided when source and destination tensors are not the same.
        redop (ReduceOp, optional): reduce operation.
        root (int, optional): root of broadcast or reduce.
    """

    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        dst_tensor: Optional[torch.Tensor] = None,
        redop: Optional[ReduceOp] = None,
        root: Optional[int] = None,
    ):
        self.op = op
        self.tensor = tensor
        self.dst_tensor = dst_tensor
        self.redop = redop
        self.root = root


# DO NOT USE THESE FIELDS DIRECTLY.
# Use them through the _world object to make sure the _world override mechanism
_pg_map: dict[ProcessGroup, tuple[str, Store]] = {}
_pg_names: dict[ProcessGroup, str] = {}
_pg_group_ranks: dict[ProcessGroup, dict[int, int]] = {}
# For a pg, it is a map from ProcessGroup to BackendConfig
_pg_backend_config: dict[ProcessGroup, str] = {}
_group_count = 0
_tags_to_pg: dict[str, list[ProcessGroup]] = {}
_pg_to_tag: dict[ProcessGroup, str] = {}
_backend: Optional[str] = None


class _World:
    """
    Container class for c10d process group state.

    This is used during registration and lookup of PG state.

    .. warning:: This is an experimental API intended to expose the inner workings
       of c10d and is subject to change..
    """

    def __init__(self) -> None:
        self._default_pg = None
        self._pg_coalesce_state: dict[ProcessGroup, list[_CollOp]] = {}

    @property
    def default_pg(self) -> Optional[ProcessGroup]:
        """
        Process group that includes all ranks of the cluster.

        This default ProcessGroup is used by c10d APIs when a ProcessGroup is needed
        but None is provided.
        """
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value) -> None:
        self._default_pg = value

    @property
    def pg_map(self) -> dict[ProcessGroup, tuple[str, Store]]:
        """
        Provide Mapping from ProcessGroup to backend name and store.

        For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
        For MPI pg, it is a map from ProcessGroup to (Backend, None)

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_map
        return _pg_map

    @property
    def pg_names(self) -> dict[ProcessGroup, str]:
        """
        Process group's names, map from ProcessGroup to str.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_names
        return _pg_names

    @property
    def pg_group_ranks(self) -> dict[ProcessGroup, dict[int, int]]:
        """
        Process group's global rank to local rank mapping.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_backend_config(self) -> dict[ProcessGroup, str]:
        """
        Process group's backend config.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_backend_config
        return _pg_backend_config

    @property
    def group_count(self) -> int:
        """
        Process group count for default naming.

        TODO don't expose group_count, use something else instead
        """
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value: int) -> None:
        """Use to compute the name of ProcessGroups when using global synchronization."""
        global _group_count
        _group_count = value

    @property
    def tags_to_pg(self) -> dict[str, list[ProcessGroup]]:
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> dict[ProcessGroup, str]:
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_coalesce_state(self) -> dict[ProcessGroup, list[_CollOp]]:
        return self._pg_coalesce_state

    @property
    def pg_config_info(self) -> list[dict[str, Any]]:
        """
        Return a list of dict with process groups and backends.

        Along with their unique IDs and configurations (types and ranks).
        """
        config_info: list[dict[str, Any]] = []
        default_pg_size = _get_group_size(None)
        for pg in self.pg_map:
            ranks = self.pg_group_ranks[pg]
            config_info.append(
                {
                    "pg_name": self.pg_names[pg],
                    "pg_desc": pg.group_desc,
                    "backend_config": self.pg_backend_config[pg],
                    "ranks": (
                        list(ranks.keys()) if len(ranks) != default_pg_size else []
                    ),  # 'ranks' is an empty list when all ranks are involved in a pg
                    "group_size": len(ranks),
                    "group_count": self.group_count,
                }
            )
        return config_info


_world = _World()
"""Holds the singleton instance of ``_World`` used by c10. Experimental extension point to override it"""


class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """

    # Points to the default PG once initialized.
    @property
    def WORLD(cls) -> Optional[ProcessGroup]:
        return _world.default_pg

    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):
        _world.default_pg = pg


class group(metaclass=_WorldMeta):
    """Group class. Placeholder."""


class GroupMember(metaclass=_WorldMeta):
    """Group member class."""

    NON_GROUP_MEMBER = -100


def _get_default_timeout(backend: Backend) -> timedelta:
    # see note on nccl vs other backend timeout (constants.py)
    if backend == Backend.NCCL:
        if not isinstance(default_pg_nccl_timeout, timedelta):
            # TODO moco benchmark on CPU initializes pgnccl backend today, triggered this assert in CI before it was
            # changed to be a warning.  We should fix the moco model.
            warnings.warn(
                "Attempted to get default timeout for nccl backend, but NCCL support is not compiled",
                stacklevel=2,
            )
            return default_pg_timeout
        return default_pg_nccl_timeout
    else:
        return default_pg_timeout


def _check_valid_timeout(timeout: Any) -> None:
    if not isinstance(timeout, timedelta):
        raise TypeError(
            f"Expected timeout argument to be of type datetime.timedelta, got {timeout}"
        )


# Default process group state
_default_pg_init_method: Optional[str] = None

STORE_BASED_BARRIER_PREFIX = "store_based_barrier_key"


def _get_object_coll_device(group: Optional[ProcessGroup] = None) -> str:
    """
    .. note:: This is an internal helper and does not have backward
        compatibility, please use with caution.

    Return the device type to use with ``group`` for object collectives or
    barrier.

    There are selection rules:
        1. If user specifies exactly one backend in ``init_process_group`` call:
            use that backend
        2. Else if user specifies multiple "device:backend" pairs in init_process_group:
            If "cpu" is among those pairs, use "cpu" (because the object is in cpu memory);
            Otherwise, use the first backend (sort of a random pick).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        str: The device type to use for object collective with ``group``.

    """
    group = group or _get_default_group()

    if not isinstance(group, ProcessGroup):
        warnings.warn(
            f"You are using a Backend {type(group)} as a ProcessGroup. "
            "This usage is deprecated since PyTorch 2.0. Please use a public API "
            "of PyTorch Distributed instead.",
            stacklevel=2,
        )
        # Provide backward compatibility to cases where `group` passed in is
        # actually a Backend (like `ProcessGroupGloo`) rather than a
        # `ProcessGroup` in PT 2.0 sense
        if isinstance(group, ProcessGroupGloo):
            # RPC uses Gloo for object collectives
            return "cpu"
        else:
            raise ValueError(f"Expecting a ProcessGroup, but got a {type(group)}.")

    """
    ``group._device_types`` is a property pybind that returns the devices
    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the
    ``group`` supports multiple devices.
    """
    devices = group._device_types

    if len(devices) == 1:
        # User fixed exactly one backend in `init_process_group`
        return devices[0].type
    elif len(devices) == 0:
        # No backend has been registered with this PG (maybe because no
        # collective has been run?) We pick cpu as the default and hopefully
        # this would lazily init Gloo or other available cpu backend.
        return "cpu"
    elif torch.device("cpu") in devices:
        # There are multiple backends in this PG and cpu is among them.
        # cpu is preferred as the object is in cpu memory. No need for device
        # copy.
        return "cpu"
    else:
        # No cpu in the backend list. Randomly pick the first backend
        return devices[0].type


def _get_pg_default_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """
    .. note:: This method will be deprecated, it only stays for
        backward-compatiblity reason. Alternatives:

        - If you need to find a device for object collectives, please use
        `_get_object_coll_device(group)`.

        - If you need to query the device types supported by group, please use
        `_device_capability(group)`.

    Return the device type registered with ``group``.

    For example, if `init_process_group("nccl", ...)` was called, the returned
    value would be `torch.device("cuda")`.

    Errors out if no device has been registered.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.device: The device type registered with ``group``.
    """

    warnings.warn(
        "`_get_pg_default_device` will be deprecated, it only stays for "
        "backward-compatiblity reason. If you need to find a device for object "
        "collectives, please use `_get_object_coll_device`. If you need to query "
        "the device types supported by group, please use "
        "`_device_capability(group)`. ",
        stacklevel=2,
    )
    group = group or _get_default_group()

    if not isinstance(group, ProcessGroup):
        # Provide backward compatibility to cases where `group` passed in is
        # actually a Backend (like `ProcessGroupGloo`) rather than a
        # `ProcessGroup` in PT 2.0 sense
        warnings.warn(
            f"You are using a Backend {type(group)} as a ProcessGroup. "
            "This usage is deprecated since PyTorch 2.0. Please use a public API "
            "of PyTorch Distributed instead.",
            FutureWarning,
            stacklevel=3,
        )
        # Most users create Gloo with private API for object collectives
        return torch.device("cpu")

    """
    ``group._device_types`` is a property pybind that returns the devices
    ("cpu", "cuda", etc) supported by ``group``. Can be multiple if the
    ``group`` supports multiple devices.
    """
    devices = group._device_types

    if len(devices) == 1:
        # User fixed exactly one backend in `init_process_group`
        return devices[0]
    elif len(devices) == 0:
        raise RuntimeError(
            "Default device not found, because no backend has been registered "
            "with this ProcessGroup."
        )
    else:
        # There are multiple backends in this PG.
        if torch.device("cpu") in devices:
            rv = torch.device("cpu")
        else:
            rv = devices[0]
        warnings.warn(
            "Multiple backends are registered with this ProcessGroup. We cannot "
            f"determine which one is the default. Returning {rv}. "
            "Please consider using other APIs.",
            stacklevel=2,
        )
        return rv


def _device_capability(group: Optional[ProcessGroup] = None) -> list[str]:
    """
    Return the device type(s) supported by ``group``.

    Args:
        group (ProcessGroup, optional): The process group to query. If None,
            the default process group will be used.

    Returns:
        List[str]: A list of device types supported by ``group``.
    """
    group = group or _get_default_group()
    return [device.type for device in group._device_types]


@_time_logger
def _store_based_barrier(
    rank,
    store,
    group_name,
    rendezvous_count,
    timeout,
    logging_interval=timedelta(seconds=10),
) -> None:
    """
    Store based barrier for synchronizing processes.

    Barrier based on store which is used for synchronizing processes after
    ``init_process_group`` or ``new_group``. Intended to be used only with
    those two methods and is not a generic alternative to ``barrier()``.
    """
    store_key = f"{STORE_BASED_BARRIER_PREFIX}:{group_name}"
    store.add(store_key, 1)
    logger.debug("Added key: %s to store for rank: %s", store_key, rank)

    # Now wait for all workers to check in with the store.
    world_size = rendezvous_count
    worker_count = store.add(store_key, 0)

    last_worker_key = f"{store_key}:last_worker"
    if worker_count == world_size:
        store.set(last_worker_key, "1")

    # adjust the timeout to be at least 10secs + 1sec per thousand ranks to reduce the odds of timeout
    # this value was empirically found while scale testing.
    logging_interval = max(logging_interval, timedelta(seconds=10 + world_size / 1000))

    start = time.time()
    while True:
        try:
            # This will throw an exception after the logging_interval in which we print out
            # the status of the group or time out officially, throwing runtime error
            store.wait([last_worker_key], logging_interval)
            break
        except RuntimeError as e:
            worker_count = store.add(store_key, 0)
            # Print status periodically to keep track.
            logger.debug(  # noqa: G200
                "Waiting in store based barrier to initialize process group for %s seconds"
                "rank: %s, key: %s (world_size=%s, num_workers_joined=%s, timeout=%s error=%s)",
                time.time() - start,
                rank,
                store_key,
                world_size,
                worker_count,
                timeout,
                e,
            )

            if timedelta(seconds=(time.time() - start)) > timeout:
                raise DistStoreError(  # noqa: B904
                    "Timed out initializing process group in store based barrier on "
                    f"rank {rank}, for key: {store_key} (world_size={world_size}, "
                    f"num_workers_joined={worker_count}, timeout={timeout} error={e})"
                )

    logger.info(
        "Rank %s: Completed store-based barrier for key:%s with %s nodes.",
        rank,
        store_key,
        world_size,
    )


def _rank_not_in_group(group: Optional[ProcessGroup]) -> bool:
    """Check if the current process's rank is not in a given group."""
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


def _warn_not_in_group(op_name) -> None:
    global_rank = -1 if GroupMember.WORLD is None else GroupMember.WORLD.rank()
    warnings.warn(
        f"Running {op_name} on global rank {global_rank} which does not "
        "belong to the given group.",
        stacklevel=2,
    )


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is GroupMember.WORLD:
        return global_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    group_ranks = _world.pg_group_ranks[group]
    if global_rank not in group_ranks:
        raise ValueError(f"Global rank {global_rank} is not part of group {group}")

    return group_ranks[global_rank]


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """
    Translate a group rank into a global rank.

    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the global rank from.
        group_rank (int): Group rank to query.

    Returns:
        Global rank of ``group_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is GroupMember.WORLD:
        return group_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(
            f"Group {group} is not registered, please create group with torch.distributed.new_group API"
        )
    for rank, grp_rank in _world.pg_group_ranks[group].items():
        if grp_rank == group_rank:
            return rank
    raise ValueError(f"Group rank {group_rank} is not part of group {group}")


# TODO: remove this once the ecosystem moves away from it.
@deprecated(
    "`torch.distributed.distributed_c10d._get_global_rank` is deprecated, "
    "please use `torch.distributed.distributed_c10d.get_global_rank` instead",
    category=FutureWarning,
)
def _get_global_rank(group, rank) -> int:
    """Use get_global_rank as this method is deprecated."""
    return get_global_rank(group, rank)


def get_process_group_ranks(group: Optional[ProcessGroup]) -> list[int]:
    """
    Get all ranks associated with ``group``.

    Args:
        group (Optional[ProcessGroup]): ProcessGroup to get all ranks from.
            If None, the default process group will be used.

    Returns:
        List of global ranks ordered by group rank.
    """
    return list(_world.pg_group_ranks[group or _get_default_group()].keys())


def _get_group_size(group) -> int:
    """Get a given group's world size."""
    if group is GroupMember.WORLD or group is None:
        default_pg = _get_default_group()
        return default_pg.size()
    return group.size()


def _get_group_size_by_name(group_name: str) -> int:
    group = _resolve_process_group(group_name)
    return group.size()


def _resolve_group_name_by_ranks_and_tag(ranks: list[int], tag: str) -> str:
    # TODO(yifu): remove this function once ranks + tag is not a supported
    # identifier for process group for functional collectives.
    group = _find_pg_by_ranks_and_tag(tag, ranks)
    if group is None:
        raise ValueError("")
    return group.group_name


def _check_single_tensor(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a single tensor."""
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type torch.Tensor
             but got {type(param)} instead."""
        )


def _check_tensor_list(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a list of tensors."""
    if not isinstance(param, list):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} instead."""
        )
    elif not all(isinstance(p, torch.Tensor) for p in param):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} with elements of type {[type(p) for p in param]}."""
        )


def _group_or_default_group(group: Optional[ProcessGroup] = None) -> ProcessGroup:
    if group is None or group is GroupMember.WORLD:
        group = _get_default_group()
    return group


def _canonicalize_group_rank(
    group: ProcessGroup,
    global_rank: Optional[int] = None,
    group_rank: Optional[int] = None,
    return_global: bool = False,
) -> int:
    """
    Helper method to take _either_ a global rank or a group rank and produce a group rank.

    If 'return_global' is true, produce a global rank instead of a group rank.
    """

    if group_rank is not None:
        if global_rank is not None:
            raise ValueError("Can't specify both group_rank and global_rank")
        if return_global:
            return get_global_rank(group, group_rank)
    else:
        if global_rank is None:
            raise ValueError("Must specify global_rank or group_rank")
        if return_global:
            return global_rank
        group_rank = get_group_rank(group, global_rank)
    return group_rank


def _check_not_self_rank(group: ProcessGroup, rank: int, rank_type: str):
    if group.rank() == rank:
        raise ValueError(
            f"Invalid {rank_type} rank: {rank_type} rank should not be the same as "
            "the rank of the current process."
        )


def _as_iterable(obj) -> collections.abc.Iterable:
    return obj if isinstance(obj, list) else (obj,)


def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    # pyrefly: ignore [bad-assignment]
    for tensor in itertools.chain.from_iterable(map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        # Mixing complex and its element type is allowed
        if tensor_dtype.is_complex:
            tensor_dtype = (
                torch.float32 if tensor_dtype == torch.complex64 else torch.complex128
            )

        if last_dtype is None:
            last_dtype = tensor_dtype
        else:
            if last_dtype != tensor_dtype:
                raise ValueError(
                    "Invalid usage of tensors with different dtypes"
                    f"Found {last_dtype} and  {tensor.dtype}"
                )


def _check_op(op) -> None:
    """Check that the ``op`` is either isend or irecv."""
    if op not in [isend, irecv]:
        raise ValueError(
            "Invalid ``op``. Expected ``op`` "
            "to be of type ``torch.distributed.isend`` or "
            "``torch.distributed.irecv``."
        )


def _check_p2p_op_list(p2p_op_list) -> None:
    """
    Check that the ``p2p_op_list`` is a list of P2POp instances.

    Also, check that all ops use the same group.
    """
    if not isinstance(p2p_op_list, list) or not all(
        isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list
    ):
        raise ValueError(
            "Invalid ``p2p_op_list``. Each op is expected to "
            "to be of type ``torch.distributed.P2POp``."
        )

    group = p2p_op_list[0].group
    if not all(group == p2p_op.group for p2p_op in p2p_op_list):
        raise ValueError("All ops need to use the same group.")


def is_mpi_available() -> bool:
    """Check if the MPI backend is available."""
    return _MPI_AVAILABLE


def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
    return _NCCL_AVAILABLE


def is_gloo_available() -> bool:
    """Check if the Gloo backend is available."""
    return _GLOO_AVAILABLE


def is_ucc_available() -> bool:
    """Check if the UCC backend is available."""
    return _UCC_AVAILABLE


def is_xccl_available() -> bool:
    """Check if the XCCL backend is available."""
    return _XCCL_AVAILABLE


def _check_single_backend_availability(backend_name: str) -> bool:
    """
    Helper function to check if a single backend is available.
    """
    available_func = getattr(
        torch.distributed, f"is_{str(backend_name).lower()}_available", None
    )
    if available_func:
        return available_func()
    return str(backend_name).lower() in Backend.backend_list


def is_backend_available(backend: str) -> bool:
    """
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """
    # If the backend has an ``is_backend_available`` function, return the result of that function directly
    if ":" in backend.lower():  # composite backend like "cpu:gloo"
        backend_config = BackendConfig(Backend(backend))
        device_backend_map = backend_config.get_device_backend_map()
        return all(
            _check_single_backend_availability(str(backend_name))
            for backend_name in device_backend_map.values()
        )
    else:
        # Handle simple backend strings like "nccl", "gloo"
        return _check_single_backend_availability(backend)


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return GroupMember.WORLD is not None


def is_torchelastic_launched() -> bool:
    """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    The existence of ``TORCHELASTIC_RUN_ID`` environment
    variable is used as a proxy to determine whether the current process
    was launched with torchelastic. This is a reasonable proxy since
    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a
    non-null value indicating the job id for peer discovery purposes..
    """
    return os.getenv("TORCHELASTIC_RUN_ID") is not None


def _is_barrier_after_init() -> int:
    # Environment variable to control whether process group should perform a
    # barrier after its init. Default value is 0, i.e. no barrier. If you
    # experience issue with this setting, you may set
    # `TORCH_DIST_INIT_BARRIER=1` to add the barrier.
    return int(os.getenv("TORCH_DIST_INIT_BARRIER", "0"))


def _get_default_group() -> ProcessGroup:
    """Get the default process group created by init_process_group."""
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    if TYPE_CHECKING:
        return not_none(GroupMember.WORLD)
    else:
        return GroupMember.WORLD


def _get_default_store() -> Store:
    """Get the default store created by init_process_group."""
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, "
            "please make sure to call init_process_group."
        )
    default_pg = _get_default_group()
    _, default_store = _world.pg_map[default_pg]
    return default_store


def _update_default_pg(pg) -> None:
    _world.default_pg = pg
    rank = pg.rank() if pg is not None and pg != GroupMember.NON_GROUP_MEMBER else -1
    torch._C._distributed_c10d._set_global_rank(rank)


def get_backend_config(group: Optional[ProcessGroup] = None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
    pg = group or _get_default_group()
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")
    backend_config = _world.pg_backend_config.get(pg)
    return str(not_none(backend_config))


def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
    """
    Return the backend of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    pg = group or _get_default_group()
    if _rank_not_in_group(pg):
        raise ValueError("Invalid process group specified")

    pg_store = _world.pg_map.get(pg, None)
    if pg_store is None:
        raise ValueError(
            f"Process group {pg} is not initialized in the world group map. Please initialize the group first."
        )

    return Backend(not_none(pg_store)[0])


def get_default_backend_for_device(device: Union[str, torch.device]) -> str:
    """
    Return the default backend for the given device.

    Args:
        device (Union[str, torch.device]): The device to get the default backend for.

    Returns:
        The default backend for the given device as a lower case string.

    """
    if isinstance(device, torch.device):
        device_str = device.type
    else:
        device_str = torch.device(device).type

    backend = Backend.default_device_backend_map.get(device_str)
    if backend is None:
        raise ValueError(f"Default backend not registered for device : {device}")

    return backend


def _get_process_group_uid(pg: ProcessGroup) -> int:
    backend = None
    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        pass
    if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
        return backend.uid
    return -1


def _get_pg_config(group: Optional[ProcessGroup] = None) -> dict[str, Any]:
    """
    Return the pg configuration of the given process group.

    """
    pg = group or _get_default_group()
    return {
        "pg_name": _get_process_group_name(pg),
        "pg_desc": pg.group_desc,
        "backend_config": get_backend_config(pg),
        "pg_size": _get_group_size(pg),
        "ranks": get_process_group_ranks(pg),
    }


def _get_all_pg_configs() -> list[dict[str, Any]]:
    """
    Return the pg configuration of all the process groups.

    """
    config_info: list[dict[str, Any]] = [_get_pg_config(pg) for pg in _world.pg_map]
    return config_info


def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
    return _world.group_count


def get_node_local_rank(fallback_rank: Optional[int] = None) -> int:
    """
    Return the local rank of the current process relative to the node.

    Semantically, this is a useful concept for mapping processes to devices.
    For example, on a node with 8 accelerator you could use the node local rank to decide
    which accelerator device to bind the process to.

    In practice, the actual assignment of node local ranks is handled by the process launcher outside of pytorch,
    and communicated via the `LOCAL_RANK` environment variable.

    Torchrun will auto
```



## High-Level Overview

"""Distributed Collective Communication (c10d)."""import collections.abcimport contextlibimport copyimport ctypesimport hashlibimport ioimport itertoolsimport loggingimport osimport pickleimport sysimport timeimport warningsfrom collections import namedtuplefrom collections.abc import Callablefrom datetime import timedeltafrom typing import Any, Optional, TYPE_CHECKING, Unionfrom typing_extensions import deprecatedimport torchfrom torch._C import _DistStoreError as DistStoreErrorfrom torch._C._distributed_c10d import (    _DistributedBackendOptions,    _register_process_group,    _resolve_process_group,    _unregister_all_process_groups,    _unregister_process_group,    AllgatherOptions,    AllreduceCoalescedOptions,    AllreduceOptions,    AllToAllOptions,    BarrierOptions,    BroadcastOptions,    DebugLevel,    GatherOptions,    get_debug_level,    PrefixStore,    ProcessGroup,    ReduceOp,    ReduceOptions,    ReduceScatterOptions,    ScatterOptions,    Store,    Work,)from torch._utils_internal import set_pytorch_distributed_envs_from_justknobsfrom torch.monitor import _WaitCounter

This Python file contains 25 class(es) and 150 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Backend`, `BackendConfig`, `_reduce_op`, `P2POp`, `_CollOp`, `_World`, `_WorldMeta`, `group`, `GroupMember`, `_IllegalWork`, `_CoalescingManager`, `_TimeEstimator`

**Functions defined**: `_export_c_types`, `supports_complex`, `__new__`, `register_backend`, `__init__`, `__repr__`, `get_device_backend_map`, `__init__`, `__getattribute__`, `__init__`, `__new__`, `__repr__`, `__init__`, `__init__`, `default_pg`, `default_pg`, `pg_map`, `pg_names`, `pg_group_ranks`, `pg_backend_config`

**Key imports**: collections.abc, contextlib, copy, ctypes, hashlib, io, itertools, logging, os, pickle


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`
- `contextlib`
- `copy`
- `ctypes`
- `hashlib`
- `io`
- `itertools`
- `logging`
- `os`
- `pickle`
- `sys`
- `time`
- `warnings`
- `collections`: namedtuple
- `datetime`: timedelta
- `typing`: Any, Optional, TYPE_CHECKING, Union
- `typing_extensions`: deprecated
- `torch`
- `torch._C`: _DistStoreError as DistStoreError
- `torch._utils_internal`: set_pytorch_distributed_envs_from_justknobs
- `torch.monitor`: _WaitCounter
- `torch.overrides`: handle_torch_function, has_torch_function
- `torch.utils._typing_utils`: not_none
- `.c10d_logger`: _exception_logger, _time_logger
- `.constants`: default_pg_nccl_timeout, default_pg_timeout
- `.rendezvous`: register_rendezvous_handler, rendezvous  
- `torch._C._distributed_c10d`: ProcessGroupMPI


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `distributed_c10d.py_docs.md`
- **Keyword Index**: `distributed_c10d.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
