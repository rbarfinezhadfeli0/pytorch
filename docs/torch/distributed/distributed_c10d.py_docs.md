# Documentation: distributed_c10d.py

## File Metadata
- **Path**: `torch/distributed/distributed_c10d.py`
- **Size**: 247590 bytes
- **Lines**: 6275
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

    Torchrun will automatically populate `LOCAL_RANK`, but other launchers may not.  If `LOCAL_RANK` is unspecified,
    this API will fall back to the provided kwarg 'fallback_rank' if specified, otherwise it will raise an error. The
    intent is to allow writing an application that runs either in single or multi device contexts without error.

    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif fallback_rank is not None:
        return int(fallback_rank)
    raise RuntimeError(
        "LOCAL_RANK is not in the environment. Consider passing fallback_rank to allow `get_node_local_rank` to work, "
        "assuming you are not running in a multi-device context and want the code to run locally instead."
    )


def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
    """
    This API adds an ephemeral timeout extension for all PGs locally
    on one rank. The timeout gets reset when the first collective issued
    after API called finished.
    NOTE: We only support to set timeout for cuda backends for now.
    NOTE: While this feature
    provides flexibility in specific scenarios, it introduces statefulness
    to timeout setting. Therefore, it is advisable to use this API sparingly
    and consider alternative approaches, such as directly setting the timeout
    or utilizing a barrier collective (one can set any timeout to the barrier),
    whenever feasible.

    Args:
        timeout (timedelta): The delta of timeout to extend.

    Returns:
        None.
    """
    for pg in _world.pg_map:
        devices = pg._device_types
        if torch.device("cuda") in devices:
            backend = pg._get_backend(torch.device("cuda"))
            if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
                backend._add_ephemeral_timeout(timeout)


def _set_pg_timeout(timeout: timedelta, group: Optional[ProcessGroup] = None) -> None:
    """
    Set the timeout for the given process group when users want to use a different timeout instead of
    default values.

    Args:
        timeout (timedelta): Timeout for operations executed against the process group which
            users want to set. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.

        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        None
    """
    if group is None:
        group = _get_default_group()
    if _rank_not_in_group(group):
        raise ValueError("Invalid process group specified")
    if not isinstance(group, ProcessGroup):
        raise AssertionError(f"Expected ProcessGroup, got {type(group)}")
    devices = group._device_types
    backends = set()
    if torch.device("cpu") in devices and is_gloo_available():
        backend = group._get_backend(torch.device("cpu"))
        if isinstance(backend, ProcessGroupGloo):
            backends.add(backend)
    if torch.device("cuda") in devices:
        backend = group._get_backend(torch.device("cuda"))
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backends.add(backend)  # type: ignore[arg-type]
        elif is_gloo_available() and isinstance(backend, ProcessGroupGloo):
            backends.add(backend)  # type: ignore[arg-type]
    if len(backends) == 0:
        warnings.warn(
            "Set timeout is now only supported for either nccl or gloo.", stacklevel=2
        )
    for backend in backends:
        backend._set_default_timeout(timeout)


@_exception_logger
@_time_logger
def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[Union[torch.device, int]] = None,
    _ranks: Optional[list[int]] = None,
) -> None:
    """
    Initialize the default distributed process group.

    This will also initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".


    Args:
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            ``nccl``, ``ucc``, ``xccl`` or one that is registered by a third-party
            plugin.
            Since 2.6, if ``backend`` is not provided, c10d will use a backend
            registered for the device type indicated by the `device_id` kwarg
            (if provided). The known default registrations today are: ``nccl``
            for ``cuda``, ``gloo`` for ``cpu``, ``xccl`` for ``xpu``.
            If neither ``backend`` nor ``device_id`` is provided, c10d will
            detect the accelerator on the run-time machine and use a backend
            registered for that detected accelerator (or ``cpu``).
            This field can be given as a lowercase string (e.g., ``"gloo"``),
            which can also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).
            If using multiple processes per machine with ``nccl`` backend, each
            process must have exclusive access to every GPU it uses, as sharing
            GPUs between processes can result in deadlock or NCCL invalid usage.
            ``ucc`` backend is experimental.
            Default backend for the device can be queried with
            :func:`get_default_backend_for_device`.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.

        group_name (str, optional, deprecated): Group name. This argument is ignored
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. As of now, the only
            options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            the nccl backend can pick up high priority cuda streams when
            there're compute kernels waiting. For other available options to config nccl,
            See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
        device_id (torch.device | int, optional): a single, specific device
            this process will work on, allowing for backend-specific
            optimizations.  Currently this has two effects, only under
            NCCL: the communicator is immediately formed (calling
            ``ncclCommInit*`` immediately rather than the normal lazy
            call) and sub-groups will use ``ncclCommSplit`` when
            possible to avoid unnecessary overhead of group creation. If you
            want to know NCCL initialization error early, you can also use this
            field. If an `int` is provided, the API assumes that the accelerator
            type at compile time will be used.
        _ranks: The ranks in the process group. If provided, the process
               group name will be the hash of all the ranks in the group.

    .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.

    .. note:: Support for multiple backends is experimental. Currently when no backend is
        specified, both ``gloo`` and ``nccl`` backends will be created. The ``gloo`` backend
        will be used for collectives with CPU tensors and the ``nccl`` backend will be used
        for collectives with CUDA tensors. A custom backend can be specified by passing in
        a string with format "<device_type>:<backend_name>,<device_type>:<backend_name>", e.g.
        "cpu:gloo,cuda:custom_backend".

    """

    global _world

    global _backend
    global _default_pg_init_method

    if GroupMember.WORLD is not None:
        raise ValueError("trying to initialize the default process group twice!")

    set_pytorch_distributed_envs_from_justknobs()

    # Depending on the import order, some trace_rules functions may be evaluated
    # during the import phase. In such a case, these functions may not correctly
    # add the distributed related rules due to import circular dependency.
    # We need to clear the lru_cache during the runtime to ensure the correctness
    # of these trace_rules.
    #
    # Since this API must be called before all distributed code being compiled,
    # clearing the cache here should be safe.
    if "torch._dynamo" in sys.modules:
        torch._dynamo.trace_rules.clear_lru_cache()

    if not ((store is None) or (init_method is None)):
        raise AssertionError("Cannot specify both init_method and store.")

    if store is not None:
        if not world_size > 0:
            raise AssertionError("world_size must be positive if using store")
        if not rank >= 0:
            raise AssertionError("rank must be non-negative if using store")
    elif init_method is None:
        init_method = "env://"

    # Get the compile-time accelerator type.
    # None indicates no accelerator support.
    acc = torch.accelerator.current_accelerator()

    # Auto complete device id
    if isinstance(device_id, int):
        if acc is None:
            raise ValueError(
                "device_id is an int, but no accelerator support is found from the current compilation. "
                "Please use a different compiled version that supports your accelerator."
            )
        device_id = torch.device(acc.type, device_id)

    # Sanity check device_id
    if device_id is not None and device_id.type != "cpu":
        # Type
        if acc is None or device_id.type != acc.type:
            raise ValueError(
                f"device_id {device_id} does not match the current compilation's accelerator support: {acc}. "
                "Please use a different compiled version that supports your accelerator."
            )
        # Index
        if device_id.index is None:
            raise ValueError("Please use a device_id with index.")
        # Range
        if device_id.index >= torch.accelerator.device_count():
            raise ValueError(
                f"device_id {device_id} is out of range. Please use a device index less than "
                f"the number of accelerators available: {torch.accelerator.device_count()}."
            )

    logger.info("Using device: %s", device_id)

    # If user did not provide a backend string but provided a device id, e.g.
    # >>> init_process_group(device_id=device)
    # we try to figure out the backend name based on the device type.
    if backend is None and device_id is not None:
        # Note: 3rd-party devices can register default backend through the
        # default map below.
        backend = Backend.default_device_backend_map.get(device_id.type)

    # If we still cannot figure it out, e.g.
    # >>> init_process_group()
    # we set it to `undefined` and rely on lazy init.
    if backend is None:
        backend = "undefined"

    # Convert string into `Backend` type
    backend = Backend(backend)

    if timeout is None:
        timeout = _get_default_timeout(backend)

    _check_valid_timeout(timeout)

    """
    Group name is not visible to users unless they access
    internals of c10d. This means we can ignore the value
    they provide as it not exposed in a public way.
    """
    if _ranks is None or len(_ranks) == 0:
        group_name = _process_group_name([], use_hashed_name=False)
    else:
        group_name = _process_group_name(_ranks, use_hashed_name=True)
    if backend == Backend.MPI:
        if world_size != -1 or rank != -1:
            warnings.warn(
                f"For MPI backend, world_size ({world_size}) and rank ({rank}) "
                "are ignored since they are assigned by the "
                "MPI runtime.",
                stacklevel=2,
            )

        default_pg, _ = _new_process_group_helper(
            -1,
            -1,
            [],
            backend,
            Store(),  # Placeholder value since store cannot be None
            group_name,
            timeout=timeout,
            group_desc="default_pg",
        )
    else:
        # backward compatible API
        if store is None:
            if backend == "fake":
                from torch.testing._internal.distributed.fake_pg import FakeStore

                store = FakeStore()
            else:
                rendezvous_iterator = rendezvous(
                    not_none(init_method), rank, world_size, timeout=timeout
                )
                store, rank, world_size = next(rendezvous_iterator)
                store.set_timeout(timeout)

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)

        default_pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name,
            backend_options=pg_options,
            timeout=timeout,
            device_id=device_id,
            group_desc="default_pg",
        )

    _update_default_pg(default_pg)

    _world.pg_group_ranks[GroupMember.WORLD] = {  # type: ignore[index]
        i: i
        for i in range(GroupMember.WORLD.size())  # type: ignore[attr-defined]
    }
    _backend = _world.pg_map[not_none(GroupMember.WORLD)][0]
    _default_pg_init_method = init_method

    old_hook = sys.excepthook
    excepthook_prefix = f"[rank{get_rank()}]"

    def _distributed_excepthook(*args):
        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            old_hook(*args)
        finally:
            sys.stderr = old_stderr
        msg = buf.getvalue()
        msg = "\n".join(
            f"{excepthook_prefix}: {s}" if s != "" else "" for s in msg.split("\n")
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    sys.excepthook = _distributed_excepthook

    if _is_barrier_after_init() == 1:
        # barrier at the end to ensure that once we return from this method, all
        # process groups including global variables (if any) are updated
        # correctly on all ranks.
        # Update 04/2023: for large-scale runs, this barrier (esp. store-based
        # barrier) may be costly and/or unscalable. Also, in a lot of cases,
        # these barriers may be unnecessary, as proven by a green CI after
        # removal. An environment variable `TORCH_DIST_INIT_BARRIER` has been
        # added which enables this barrier only when set to 1.
        logger.debug(
            "Performing barrier after ProcessGroup initialization since "
            "TORCH_DIST_INIT_BARRIER = 1"
        )
        if backend == Backend.MPI:
            # MPI backend doesn't use store.
            barrier()
        else:
            # Use store based barrier here since barrier() used a bunch of
            # default devices and messes up NCCL internal state.
            _store_based_barrier(rank, store, group_name, world_size, timeout)


def _get_split_source(pg):
    split_from = None
    if pg.bound_device_id:
        split_from = pg._get_backend(pg.bound_device_id)
    elif pg is _world.default_pg:
        try:
            # pyrefly: ignore [missing-attribute]
            split_from = pg._get_backend(torch.device("cuda"))
        except RuntimeError:
            # no cuda device associated with this backend
            pass

    if not split_from or not split_from.supports_splitting:
        return None

    # If necessary, find a backend to split from by peeling process
    # group wrappers from our potentially wrapped process group.
    while _GLOO_AVAILABLE and isinstance(split_from, _ProcessGroupWrapper):
        split_from = split_from.wrapped_pg

    return split_from


def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world

    if group_name in _world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if device_id is not None and (device_id.index is None or device_id.type == "cpu"):
        raise ValueError(
            "init_process_group device_id parameter must be an accelerator with an index"
        )

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = len(global_ranks_in_group) == 0

    # nccl and potentially other backends allow creation of
    # communicators based on pre-existing ones, which can save
    # initialization time.  Due to lazy initialization of
    # communicators in some backends, we have to be careful and only
    # split when we *know* the default PG has already started communicator initialization.
    # We know this if we have bound a device id to the default pg (eager initialized).
    if is_initialized() and _get_default_group().bound_device_id:
        split_from = _get_split_source(_get_default_group())
    else:
        split_from = None

    # If this is a subgroup (which means group_ranks is specified),
    # we check if the current process is a member of the new group.
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # If we are using `ncclCommSplit` (or similar split from
            # other APIs) to create the communicator, we will need to
            # call `ncclCommSplit` on *all* ranks in this new group's
            # parent group, even those not in the new group.  This is
            # a requirement of the NCCL API as otherwise we would get
            # out of sync.
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    backend_config = BackendConfig(backend)
    # Set the default backend when single backend is passed in.
    if "," not in str(backend) and ":" not in str(backend):
        if backend not in Backend.backend_type_map:
            raise AssertionError(f"Unknown backend type {backend}")
        if backend == Backend.UNDEFINED:
            # Currently when backend is UNDEFINED, only one backend will be initialized
            # we use nccl (if cuda is available) or gloo as default backend
            # so we can correctly call getDefaultBackend which in ProcessGroup.
            if Backend.NCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.NCCL)
            else:
                pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        else:
            pg._set_default_backend(Backend.backend_type_map[backend])
    # In order to correctly call pg._has_hooks(), we should set the default backend
    # when multi backend is passed in
    else:
        if Backend.NCCL in backend_config.device_backend_map.values():
            pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        elif Backend._plugins.keys():
            custom_backend = next(iter(Backend._plugins.keys()))
            if custom_backend in backend_config.device_backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        else:
            pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    if device_id:
        pg.bound_device_id = device_id
    backend_class: torch._C._distributed_c10d.Backend
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError(
                    "Distributed package doesn't have MPI built in."
                    " MPI is only included if you build PyTorch from"
                    " source on a host that has MPI installed."
                )
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER, None
            # create new process group with accurate rank and size
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(
                    backend_prefix_store,
                    backend_class.rank(),
                    backend_class.size(),
                )
                pg._set_default_backend(backend_type)
        elif backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            if not is_gloo_available():
                raise RuntimeError("Distributed package doesn't have Gloo built in")
            backend_class = ProcessGroupGloo(
                backend_prefix_store,
                group_rank,
                group_size,
                # pyrefly: ignore [bad-argument-type]
                timeout=timeout,
            )
            backend_class.options.global_ranks_in_group = global_ranks_in_group
            backend_class.options.group_name = group_name
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if backend_options is not None:
                if not isinstance(backend_options, ProcessGroupNCCL.Options):
                    raise AssertionError(
                        "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                    )
                if backend_options._timeout != timeout:
                    warnings.warn(
                        "backend_options._timeout was specified, "
                        "but timeout kwarg has a default value that will always override it. ",
                        stacklevel=2,
                    )
            else:
                # default backend_options for NCCL
                backend_options = ProcessGroupNCCL.Options()
                backend_options.is_high_priority_stream = False
            # pyrefly: ignore [bad-argument-type]
            backend_options._timeout = timeout

            if split_from:
                backend_options.split_from = split_from
                backend_options.split_color = _process_group_color(
                    global_ranks_in_group
                )
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            backend_class = ProcessGroupNCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            # TODO: once UCC plugin is fully deprecated, remove
            # is_ucc_available() from above elif-condition and raise
            # RuntimeError if is_ucc_available() returns false.

            backend_class = ProcessGroupUCC(
                backend_prefix_store,
                group_rank,
                group_size,
                # pyrefly: ignore [bad-argument-type]
                timeout=timeout,
            )
            backend_type = ProcessGroup.BackendType.UCC
        elif backend_str == Backend.XCCL:
            if not is_xccl_available():
                raise RuntimeError("Distributed package doesn't have XCCL built in")
            backend_options = ProcessGroupXCCL.Options()
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            # pyrefly: ignore [bad-argument-type]
            backend_options._timeout = timeout
            backend_class = ProcessGroupXCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.XCCL
        else:
            if backend_str.upper() not in Backend._plugins:
                raise AssertionError(f"Unknown c10d backend type {backend_str.upper()}")

            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM

            if not extended_api:
                backend_class = creator_fn(
                    backend_prefix_store, group_rank, group_size, timeout
                )
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                # pyrefly: ignore [bad-argument-type]
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                backend_class = creator_fn(dist_backend_opts, backend_options)

        # Set sequence numbers for gloo and nccl backends.
        if backend_str == Backend.GLOO:
            if not isinstance(backend_class, ProcessGroupGloo):
                raise AssertionError(
                    f"Expected ProcessGroupGloo, got {type(backend_class)}"
                )
            backend_class._set_sequence_number_for_group()
        elif backend_str == Backend.NCCL:
            if not isinstance(backend_class, ProcessGroupNCCL):
                raise AssertionError(
                    f"Expected ProcessGroupNCCL, got {type(backend_class)}"
                )
            backend_class._set_sequence_number_for_group()

        # If the type is a subclass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class  # type: ignore[assignment]
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if (
            backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
            or backend_str.upper() in Backend._plugins
        ):
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        # pyrefly: ignore [bad-argument-type]
                        timeout=timeout,
                    )

        # register only a single backend when all get_device_backend_map values are the same
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break

        pg._register_backend(torch.device(device), backend_type, backend_class)

    # set group_name and group_dsec to backend
    if group_name is None:
        raise AssertionError("group_name must not be None")
    if group_desc is None:
        raise AssertionError("group_desc must not be None")
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)

    _world.pg_backend_config[pg] = str(backend_config)
    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return pg, prefix_store


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    if pg is None:
        raise AssertionError("Process group cannot be None")
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # When users register Python onCompletion hooks, those hooks will run on a
    # different thread than the main thread. Today, the ProcessGroup dtor does
    # wait for that thread. However, the dtor might finish after the Python
    # Interpreter exits. After that grabbing the GIL for the Python hook will crash.
    # We can either revive the interpreter when running hooks or keep the main one
    # alive until all works and hooks are done. The current implementation does the
    # latter. Therefore, we explicitly call _wait_for_pending_works() here to wait
    # for the pending hooks to finish.
    if type(pg) is ProcessGroup and pg._has_hooks():
        pg._wait_for_pending_works()

    if group is None or group == GroupMember.WORLD:
        # shutdown all backends in the order of pg names. shutting down in order because
        # ncclCommAbort() was a 'collective' call in some versions of NCCL.
        for pg_to_shutdown in sorted(
            _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
        ):
            pg_to_shutdown.shutdown()

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        pg.shutdown()
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_coalesce_state:
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned.",
                stacklevel=2,
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def _abort_process_group(group: Optional[ProcessGroup] = None):
    """
    Abort a given process group. If group.WORLD (i.e. `None`) is given, all
    process groups including the default one will be aborted.

    Args:
        group (ProcessGroup, optional): The process group to be aborted.

    .. note:: this API is experimental and currently only works with the NCCL
        backend.

    .. note:: this API should be used with `TORCH_NCCL_ASYNC_ERROR_HANDLING`
        turned off (i.e. set to 0). Otherwise, ProcessGroupNCCL's watchdog may
        automatically handle errors or timeouts for you including aborting the
        ProcessGroup.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    pg = group or GroupMember.WORLD

    if pg is None:
        raise AssertionError("Process group cannot be None")
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified or has been destroyed.")

    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        backend = None

    if group is None or group == GroupMember.WORLD:
        # Abort all backends within a ncclGroupStart|End semantic.
        # This ensures that different NCCL communicators' abort calls won't
        # deadlock each other.
        # For details, please see: https://github.com/pytorch/pytorch/issues/119797
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backend._group_start()
        for pg_to_abort in sorted(
            _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
        ):
            pg_to_abort.abort()
        if is_nccl_available() and isinstance(backend, ProcessGroupNCCL):
            backend._group_end()

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        pg.abort()
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_coalesce_state:
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is aborted. They will be cleaned.",
                stacklevel=2,
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    default_pg = _get_default_group()
    if group is None or group is GroupMember.WORLD:
        return default_pg.rank()

    return get_group_rank(group, default_pg.rank())


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    return _get_group_size(group)


def isend(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_dst: Optional[int] = None,
) -> Optional[Work]:
    """
    Send a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike send, which is blocking, isend allows src == dst rank, i.e. send to self.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst)
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("isend")
        return None

    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    return group.send([tensor], group_dst, tag)


def irecv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_src: Optional[int] = None,
) -> Optional[Work]:
    """
    Receives a tensor asynchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("irecv")
        return None

    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)

    group = _group_or_default_group(group)
    if src is None and group_src is None:
        return group.recv_anysource([tensor], tag)
    else:
        group_src = _canonicalize_group_rank(group, src, group_src)
        return group.recv([tensor], group_src, tag)


@_exception_logger
def send(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_dst: Optional[int] = None,
) -> None:
    """
    Send a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument).
            Destination rank should not be the same as the rank of the current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``.

    """
    group = _group_or_default_group(group)
    group_dst = _canonicalize_group_rank(group, dst, group_dst)
    _check_not_self_rank(group, group_dst, "destination")
    work = isend(tensor, group=group, tag=tag, group_dst=group_dst)
    if work is not None:
        work.wait()


@_exception_logger
def recv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
    group_src: Optional[int] = None,
) -> int:
    """
    Receives a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.
    Returns:
        Sender rank
        -1, if not part of the group

    """
    work = irecv(tensor, src=src, group=group, tag=tag, group_src=group_src)
    if work is None:
        return -1
    work.wait()
    if src is None:
        if group_src is None:
            group_src = work._source_rank()
        group = _group_or_default_group(group)
        _check_not_self_rank(group, group_src, "source")
        src = get_global_rank(group, group_src)
    return src


class _IllegalWork(Work):
    def __getattribute__(self, name):
        if name in [
            "is_success",
            "exception",
            "wait",
            "source_rank",
            "_source_rank",
            "result",
            "synchronize",
        ]:
            raise ValueError(f"Illegal to call {name} on IllegalWork object")


class _CoalescingManager:
    def __init__(self) -> None:
        self.works: list[Work] = []

    def append(self, work: Optional[Work] = None):
        if work:
            self.works.append(work)

    def wait(self):
        for work in self.works:
            work.wait()


@contextlib.contextmanager
def _coalescing_manager(
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
    async_ops: bool = False,
):
    """
    Context manager used to coalesce collectives or P2P operations when possible.

    Args:
        group (`ProcessGroup`, optional): The process group to work on. If None,
            the default process group will be used.
        device (`torch.device`, optional): Default is None, set to a device if
            there isn't a `**_coalesced` implementation by the backend.
        async_ops (`bool`, optional): whether the coalesced ops are async ops.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # Synchronous ops
        >>> with _coalescing_manager():
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> # Asynchronous ops
        >>> with _coalescing_manager(async_ops=True) as cm:
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> cm.wait()

    .. warning::
       :func:`_coalescing_manager` currently do not support coalescing
       all-reduces with different reduce operators, e.g.  `ReduceOp.SUM` mixed
       with `ReduceOp.PRODUCT`.
    """
    group = group or _get_default_group()
    op_list = _world.pg_coalesce_state.setdefault(group, [])
    if op_list:
        raise ValueError(
            "ProcessGroup has non-empty op list at the start of coalescing"
        )
    if device:
        group._start_coalescing(device)
    cm = _CoalescingManager()
    yield cm
    work = None
    op_list = _world.pg_coalesce_state.pop(group)
    if op_list:
        # Collectives supporting "Fast Path" coalescing are captured.
        # See implementation in corresponding collective APIs.
        # Currently supported:
        # - coalesced `all_reduce`
        # - coalesced `all_gather_into_tensor`
        # - coalesced `reduce_scatter_tensor`
        op0 = op_list[0].op
        if op0 is all_reduce:
            tensors = [op.tensor for op in op_list]
            all_reduce_opts = AllreduceCoalescedOptions()
            all_reduce_opts.reduceOp = not_none(op_list[0].redop)
            all_reduce_opts.asyncOp = async_ops
            work = group.allreduce_coalesced(tensors, all_reduce_opts)
        elif op0 is all_gather_into_tensor:
            inputs = []
            outputs = []
            for op in op_list:
        

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 13 class(es): Backend, if, BackendConfig, _reduce_op, P2POp, _CollOp, _World, _WorldMeta, group, GroupMember, _IllegalWork, _CoalescingManager, _TimeEstimator

### Functions
This file defines 150 function(s): _export_c_types, supports_complex, __new__, register_backend, __init__, __repr__, get_device_backend_map, __init__, __getattribute__, __init__, __new__, __repr__, __init__, __init__, default_pg, default_pg, pg_map, pg_names, pg_group_ranks, pg_backend_config, group_count, group_count, tags_to_pg, pg_to_tag, pg_coalesce_state, pg_config_info, WORLD, WORLD, _get_default_timeout, _check_valid_timeout


## Key Components

The file contains 26592 words across 6275 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 247590 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
