# Documentation: `docs/torch/serialization.py_docs.md`

## File Metadata

- **Path**: `docs/torch/serialization.py_docs.md`
- **Size**: 54,112 bytes (52.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/serialization.py`

## File Metadata

- **Path**: `torch/serialization.py`
- **Size**: 85,555 bytes (83.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copyreg
import difflib
import functools
import io
import os
import pickle
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import threading
import warnings
from collections.abc import Callable
from contextlib import closing, contextmanager
from enum import Enum
from typing import Any, cast, Generic, IO, Optional, TypeAlias, TypeVar, Union
from typing_extensions import TypeIs

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch._sources import get_source_lines_and_file
from torch._utils import _import_dotted_name
from torch.storage import _get_dtype_from_pickle_storage_type
from torch.types import FileLike, Storage


__all__ = [
    "SourceChangeWarning",
    "mkdtemp",
    "register_package",
    "check_module_version_greater_or_equal",
    "validate_cuda_device",
    "validate_hpu_device",
    "location_tag",
    "default_restore_location",
    "normalize_storage_type",
    "storage_to_tensor_type",
    "save",
    "load",
    "StorageType",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
    "clear_safe_globals",
    "get_safe_globals",
    "add_safe_globals",
    "safe_globals",
    "get_unsafe_globals_in_checkpoint",
    "skip_data",
]

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct("=l").size
INT_SIZE = struct.Struct("=i").size
SHORT_SIZE = struct.Struct("=h").size

MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ","

MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, dict[str, str]]
]
STORAGE: TypeAlias = Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage]

IS_WINDOWS = sys.platform == "win32"

UNSAFE_MESSAGE = (
    "In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` "
    "from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, "
    "but it can result in arbitrary code execution. Do it only if you got the file from a "
    "trusted source."
)

if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None  # type: ignore[assignment]


def _default_to_weights_only(pickle_module):
    is_fbcode = not hasattr(torch.version, "git_version")
    return pickle_module is None and not is_fbcode


# _serialization_tls is used to store thread local state specific to serialization
# that needs to be propagated to other files, in particular we use this for
# (1) map_location (needed for wrapper subclasses/third party devices to torch._utils)
# (2) skip_data (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
# (3) materialize_fake_tensors (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
class _SerializationLocal(threading.local):
    def __init__(self):
        super().__init__()
        self.map_location: Optional[MAP_LOCATION] = None
        self.skip_data: bool = False
        self.materialize_fake_tensors: bool = False


_serialization_tls = _SerializationLocal()


class SourceChangeWarning(Warning):
    pass


@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)


_package_registry: list[
    tuple[
        int,
        Callable[[STORAGE], Optional[str]],
        Callable[[STORAGE, str], Optional[STORAGE]],
    ]
] = []


class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


def get_default_load_endianness() -> Optional[LoadEndianness]:
    """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """
    from torch.utils.serialization import config

    return config.load.endianness


def set_default_load_endianness(endianness):
    """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    from torch.utils.serialization import config

    config.load.endianness = endianness


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    from torch.utils.serialization import config

    return config.save.compute_crc32


def set_crc32_options(compute_crc32: bool):
    """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 computation flag
    """
    from torch.utils.serialization import config

    config.save.compute_crc32 = compute_crc32


def get_default_mmap_options() -> Optional[int]:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """
    from torch.utils.serialization import config

    return config.load.mmap_flags


def _get_storage_alignment() -> int:
    """
    Gets alignment for storages in torch.save files/

    Defaults to 64.

    Returns:
        storage_alginment: int
    """
    from torch.utils.serialization import config

    return config.save.storage_alignment


class set_default_mmap_options:
    """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """

    def __init__(self, flags: int) -> None:
        if IS_WINDOWS:
            raise RuntimeError(
                "Changing the default mmap options is currently not supported for Windows"
            )
        if flags != MAP_PRIVATE and flags != MAP_SHARED:
            raise ValueError(
                "Invalid argument in function set_default_mmap_options, "
                f"expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got {flags}"
            )
        # global config
        from torch.utils.serialization import config

        self.prev = config.load.mmap_flags
        config.load.mmap_flags = flags

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        from torch.utils.serialization import config

        config.load.mmap_flags = self.prev


def clear_safe_globals() -> None:
    """
    Clears the list of globals that are safe for ``weights_only`` load.
    """
    _weights_only_unpickler._clear_safe_globals()


def get_safe_globals() -> list[Union[Callable, tuple[Callable, str]]]:
    """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """
    return _weights_only_unpickler._get_safe_globals()


def add_safe_globals(safe_globals: list[Union[Callable, tuple[Callable, str]]]) -> None:
    """
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Each item in the list can either be a function/class or a tuple of the form
    (function/class, string) where string is the full path of the function/class.

    Within the serialized format, each function is identified with its full
    path as ``{__module__}.{__qualname__}``. When calling this API, you can provide this
    full path that should match the one in the checkpoint otherwise the default
    ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Args:
        safe_globals (List[Union[Callable, Tuple[Callable, str]]]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """
    _weights_only_unpickler._add_safe_globals(safe_globals)


class safe_globals(_weights_only_unpickler._safe_globals):
    r"""Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    """


def get_unsafe_globals_in_checkpoint(f: FileLike) -> list[str]:
    """Returns a list of strings of functions/classes in a ``torch.save`` object that are not safe for ``weights_only``.

    For a given function or class ``f``, the corresponding string will be of the form
    ``{f.__module__}.{f.__name__}``.

    This function will return any GLOBALs in the checkpoint that are not in the set marked safe
    for ``weights_only`` (either via :func:`add_safe_globals` or :class:`safe_globals` context or
    allowlisted by ``torch`` by default).

    .. note::
        This function will statically disassemble the pickle file in the checkpoint.
        The implication is any classes dynamically pushed onto the stack during unpickling
        will not be included in the output.

    Args:
        f: File-like object or string containing the checkpoint object saved via ``torch.save``

    Returns:
        A list of strings of pickle GLOBALs in the checkpoint that are not allowlisted for ``weights_only``.
    """
    default_safe_globals_strings = set(
        _weights_only_unpickler._get_allowed_globals().keys()
    )
    user_safe_global_strings = set(
        _weights_only_unpickler._get_user_allowed_globals().keys()
    )
    safe_global_strings = default_safe_globals_strings.union(user_safe_global_strings)

    with _open_file_like(f, "rb") as opened_file:
        if not _is_zipfile(opened_file):
            raise ValueError("Expected input to be a checkpoint returned by torch.save")
        with _open_zipfile_reader(opened_file) as zip_file:
            if _is_torchscript_zip(zip_file):
                raise ValueError(
                    "Expected input to be a checkpoint returned by torch.save but got a torchscript checkpoint"
                )
            data_file = io.BytesIO(zip_file.get_record("data.pkl"))
            all_globals = _weights_only_unpickler.get_globals_in_pkl(data_file)
            return list(all_globals.difference(safe_global_strings))


class skip_data:
    """
    Context-manager that skips writing/reading storage bytes for ``torch.save`` / ``torch.load`` calls.

    For the save path, storages will still be saved, but the space that their bytes would usually be written to
    will be empty space. The storage bytes can then be populated in a separate pass.

    For the load path, tensors will be loaded per the checkpoint but their storages will not be populated with data.

    .. warning::
        The ``skip_data`` context manager is an early prototype and is subject to change.

    Args:
        materialize_fake_tensors: Whether to materialize FakeTensors during save. This is a no-op for the load path.

    Example:
        >>> # xdoctest: +SKIP("NamedTemporaryFile on Windows")
        >>> import tempfile
        >>> t = torch.randn(2, 3)
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     with torch.serialization.skip_data():
        ...         torch.save(t, f.name)
        ...     torch.load(f.name, weights_only=True)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    """

    def __init__(self, materialize_fake_tensors: bool = False):
        self.materialize_fake_tensors = materialize_fake_tensors

    def __enter__(self):
        global _serialization_tls
        self._old_skip_data = _serialization_tls.skip_data
        self._old_materialize_fake_tensors = _serialization_tls.materialize_fake_tensors
        _serialization_tls.skip_data = True
        _serialization_tls.materialize_fake_tensors = self.materialize_fake_tensors

    def __exit__(self, type, value, tb):
        global _serialization_tls
        _serialization_tls.skip_data = self._old_skip_data
        _serialization_tls.materialize_fake_tensors = self._old_materialize_fake_tensors


def _is_zipfile(f) -> bool:
    # This is a stricter implementation than zipfile.is_zipfile().
    # zipfile.is_zipfile() is True if the magic number appears anywhere in the
    # binary. Since we expect the files here to be generated by torch.save or
    # torch.jit.save, it's safe to only check the start bytes and avoid
    # collisions and assume the zip has only 1 file.
    # See bugs.python.org/issue28494.

    start = f.tell()
    # Read the first few bytes and match against the ZIP file signature
    local_header_magic_number = b"PK\x03\x04"
    read_bytes = f.read(len(local_header_magic_number))
    f.seek(start)
    return read_bytes == local_header_magic_number


def register_package(
    priority: int,
    tagger: Callable[[STORAGE], Optional[str]],
    deserializer: Callable[[STORAGE, str], Optional[STORAGE]],
):
    """
    Registers callables for tagging and deserializing storage objects with an associated priority.
    Tagging associates a device with a storage object at save time while deserializing moves a
    storage object to an appropriate device at load time. :attr:`tagger` and :attr:`deserializer`
    are run in the order given by their :attr:`priority` until a tagger/deserializer returns a
    value that is not `None`.

    To override the deserialization behavior for a device in the global registry, one can register a
    tagger with a higher priority than the existing tagger.

    This function can also be used to register a tagger and deserializer for new devices.

    Args:
        priority: Indicates the priority associated with the tagger and deserializer, where a lower
            value indicates higher priority.
        tagger: Callable that takes in a storage object and returns its tagged device as a string
            or None.
        deserializer: Callable that takes in storage object and a device string and returns a storage
            object on the appropriate device or None.

    Returns:
        `None`

    Example:
        >>> def ipu_tag(obj):
        >>>     if obj.device.type == 'ipu':
        >>>         return 'ipu'
        >>> def ipu_deserialize(obj, location):
        >>>     if location.startswith('ipu'):
        >>>         ipu = getattr(torch, "ipu", None)
        >>>         assert ipu is not None, "IPU device module is not loaded"
        >>>         assert torch.ipu.is_available(), "ipu is not available"
        >>>         return obj.ipu(location)
        >>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
    """
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()


def check_module_version_greater_or_equal(
    module,
    req_version_tuple,
    error_if_malformed=True,
):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        version_strs = module.__version__.split(".")
        # Cast module version fields to match the types of the required version
        module_version = tuple(
            type(req_field)(version_strs[idx])
            for idx, req_field in enumerate(req_version_tuple)
        )
        requirement_is_met = module_version >= req_version_tuple

    except Exception as e:
        message = (
            f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared"
            f" with tuple {str(req_version_tuple)}"
        )
        if error_if_malformed:
            raise RuntimeError(message) from e
        else:
            warnings.warn(
                message + ", but continuing assuming that requirement is met",
                stacklevel=2,
            )
            requirement_is_met = True

    return requirement_is_met


def _cpu_tag(obj):
    if obj.device.type == "cpu":
        return "cpu"


def _mps_tag(obj):
    if obj.device.type == "mps":
        return "mps"


def _meta_tag(obj):
    if obj.device.type == "meta":
        return "meta"


def _backend_tag(backend_name, obj):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if obj.device.type == backend_name:
        if obj.device.index is None:
            return backend_name
        else:
            return backend_name + ":" + str(obj.device.index)


def _cpu_deserialize(obj, location):
    if location == "cpu":
        return obj


def _mps_deserialize(obj, location):
    if location.startswith("mps"):
        return obj.mps()


def _meta_deserialize(obj, location):
    if location == "meta":
        return torch.UntypedStorage(obj.nbytes(), device="meta")


def _validate_device(location, backend_name):
    """
    Check whether the device index of specified backend is valid

    In case of privateuse1 backend, your must first register a device_module for
    privateuse1 using torch._register_device_module. Implement the following
    methods in device_module like cuda: device_module._utils._get_device_index(location, True),
    device_module.device_count().

    Args:
        location: string of device
        backend_name: the backend name or the name of privateuse1, which can be renamed

    Returns:
        device_index: int
    """
    if not hasattr(torch, backend_name):
        raise RuntimeError(
            f"The {backend_name.upper()} device module is not registered. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_module = getattr(torch, backend_name)
    if hasattr(device_module, "_utils") and hasattr(
        device_module._utils, "_get_device_index"
    ):
        device_index = device_module._utils._get_device_index(location, True)
        device = torch.device(backend_name, device_index)
    else:
        device = torch.device(location)
        device_index = device.index if device.index else 0
    if hasattr(device_module, "is_available") and not device_module.is_available():
        raise RuntimeError(
            f"Attempting to deserialize object on a {backend_name.upper()} "
            f"device but torch.{backend_name}.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    if hasattr(device_module, "device_count"):
        device_count = device_module.device_count()
        if device_index >= device_count:
            raise RuntimeError(
                f"Attempting to deserialize object on {backend_name.upper()} device "
                f"{device_index} but torch.{backend_name}.device_count() is {device_count}. "
                "Please use torch.load with map_location to map your storages "
                "to an existing device."
            )
    return device


def validate_cuda_device(location):
    return _validate_device(location, "cuda").index


def validate_hpu_device(location):
    return _validate_device(location, "hpu").index


def _deserialize(backend_name, obj, location):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if location.startswith(backend_name):
        device = _validate_device(location, backend_name)
        return obj.to(device=device)


register_package(10, _cpu_tag, _cpu_deserialize)
register_package(
    20,
    functools.partial(_backend_tag, "cuda"),
    functools.partial(_deserialize, "cuda"),
)
register_package(21, _mps_tag, _mps_deserialize)
register_package(22, _meta_tag, _meta_deserialize)
register_package(
    23,
    functools.partial(_backend_tag, "privateuse1"),
    functools.partial(_deserialize, "privateuse1"),
)
register_package(
    24,
    functools.partial(_backend_tag, "hpu"),
    functools.partial(_deserialize, "hpu"),
)
register_package(
    25,
    functools.partial(_backend_tag, "xpu"),
    functools.partial(_deserialize, "xpu"),
)
register_package(
    26,
    functools.partial(_backend_tag, "mtia"),
    functools.partial(_deserialize, "mtia"),
)


def location_tag(
    storage: Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage],
):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)
        if location:
            return location
    raise RuntimeError(
        "don't know how to determine data location of " + torch.typename(storage)
    )


def default_restore_location(storage, location):
    """
    Restores `storage` using a deserializer function registered for the `location`.

    This function looks in the registry for deserializer functions that match the `location`.
    If found, it attempts to use them, in priority order, to restore `storage` until one
    returns a not `None` result. If no deserializer can be found in the registry, or all found fail
    to bear a result, it raises a `RuntimeError`.

    Args:
        storage (STORAGE): the storage object to restore
        location (str): the location tag associated with the storage object

    Returns:
        storage: Optional[STORAGE]

    Raises:
        RuntimeError: If no deserializer matching `location` is found in the registry or if
           all matching ones return `None`.
    """
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError(
        "don't know how to restore data location of "
        + torch.typename(storage)
        + " (tagged with "
        + location
        + ")"
    )


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace("Storage", "Tensor"))


def _is_path(name_or_buffer: object) -> TypeIs[Union[str, os.PathLike]]:
    return isinstance(name_or_buffer, (str, os.PathLike))


T = TypeVar("T")


class _opener(Generic[T]):
    def __init__(self, file_like: T) -> None:
        self.file_like: T = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_file(_opener[IO[bytes]]):
    def __init__(self, name: Union[str, os.PathLike[str]], mode: str) -> None:
        super().__init__(open(name, mode))

    def __exit__(self, *args):
        self.file_like.close()


class _open_buffer_reader(_opener[IO[bytes]]):
    def __init__(self, buffer: IO[bytes]) -> None:
        super().__init__(buffer)
        _check_seekable(buffer)


class _open_buffer_writer(_opener[IO[bytes]]):
    def __exit__(self, *args):
        self.file_like.flush()


def _open_file_like(name_or_buffer: FileLike, mode: str) -> _opener[IO[bytes]]:
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    else:
        if "w" in mode:
            return _open_buffer_writer(name_or_buffer)
        elif "r" in mode:
            return _open_buffer_reader(name_or_buffer)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


class _open_zipfile_reader(_opener[torch._C.PyTorchFileReader]):
    def __init__(self, name_or_buffer: Union[str, IO[bytes]]) -> None:
        super().__init__(torch._C.PyTorchFileReader(name_or_buffer))


class _open_zipfile_writer_file(_opener[torch._C.PyTorchFileWriter]):
    def __init__(self, name: str) -> None:
        self.file_stream = None
        self.name = name
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # PyTorchFileWriter only supports ascii filename.
            # For filenames with non-ascii characters, we rely on Python
            # for writing out the file.
            # pyrefly: ignore [bad-assignment]
            self.file_stream = io.FileIO(self.name, mode="w")
            super().__init__(
                torch._C.PyTorchFileWriter(  # pyrefly: ignore  # no-matching-overload
                    self.file_stream, get_crc32_options(), _get_storage_alignment()
                )
            )
        else:
            super().__init__(
                torch._C.PyTorchFileWriter(
                    self.name, get_crc32_options(), _get_storage_alignment()
                )
            )

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener[torch._C.PyTorchFileWriter]):
    def __init__(self, buffer: IO[bytes]) -> None:
        if not callable(getattr(buffer, "write", None)):
            msg = f"Buffer of {str(type(buffer)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(
            torch._C.PyTorchFileWriter(
                buffer, get_crc32_options(), _get_storage_alignment()
            )
        )

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        self.buffer.flush()


def _open_zipfile_writer(name_or_buffer: Union[str, IO[bytes]]) -> _opener:
    container: type[_opener]
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)  # type: ignore[arg-type]


def _is_compressed_file(f) -> bool:
    compress_modules = ["gzip"]
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False


def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False


def _check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only torch.load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False


def _check_dill_version(pickle_module) -> None:
    """Checks if using dill as the pickle module, and if so, checks if it is the correct version.
    If dill version is lower than 0.3.1, a ValueError is raised.

    Args:
        pickle_module: module used for pickling metadata and objects

    """
    if pickle_module is not None and pickle_module.__name__ == "dill":
        required_dill_version = (0, 3, 1)
        if not check_module_version_greater_or_equal(
            pickle_module, required_dill_version, False
        ):
            raise ValueError(
                (
                    "'torch' supports dill >= {}, but you have dill {}."
                    " Please upgrade dill or switch to 'pickle'"
                ).format(
                    ".".join([str(num) for num in required_dill_version]),
                    pickle_module.__version__,
                )
            )


def _check_save_filelike(f):
    if not _is_path(f) and not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute"
        )


def save(
    obj: object,
    f: FileLike,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    See :ref:`layout-control` for more advanced tools to manipulate a checkpoint.

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # xdoctest: +SKIP("makes cwd dirty")
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, "tensor.pt")
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    torch._C._log_api_usage_once("torch.save")
    _check_dill_version(pickle_module)
    _check_save_filelike(f)

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    if _use_new_zipfile_serialization:
        with _open_zipfile_writer(f) as opened_zipfile:
            _save(
                obj,
                opened_zipfile,
                pickle_module,
                pickle_protocol,
                _disable_byteorder_record,
            )
            return
    else:
        global _serialization_tls
        if _serialization_tls.skip_data:
            raise RuntimeError(
                "Cannot use skip_data=True with _use_new_zipfile_serialization=False"
            )
        with _open_file_like(f, "wb") as opened_file:
            _legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    import torch.nn as nn

    serialized_container_types = {}
    serialized_storages: dict[str, tuple[torch.UntypedStorage, torch.dtype]] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: dict[int, torch.dtype] = {}

    def persistent_id(obj: Any) -> Optional[tuple]:
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = "".join(source_lines)
            except (
                Exception
            ):  # saving the source is optional, so we can ignore any errors
                warnings.warn(
                    "Couldn't retrieve source code for container of "
                    "type " + obj.__name__ + ". It won't be checked "
                    "for correctness upon loading.",
                    stacklevel=2,
                )
            return ("module", obj, source_file, source)

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            storage: torch.UntypedStorage

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj._size()

            elif isinstance(obj, torch.UntypedStorage):
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                dtype = torch.uint8
                storage_numel = storage.nbytes()
            else:
                raise TypeError(f"type not recognized: {type(obj)}")

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            view_metadata: Optional[tuple[str, int, int]]

            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            storage_key = str(storage._cdata)
            location = location_tag(storage)

            # TODO: There's an issue here with FC. It might be impossible to
            # solve, but it's worth noting. Imagine we save a list `[storage,
            # tensor]`, where `tensor.storage()` is the same as `storage`, and
            # `tensor.element_size() > 1`. Let's say that `tensor.dtype ==
            # torch.float`.  The storage will be serialized with element size
            # of 1, since we're choosing to serialize the first occurrence of
            # a duplicate storage. Since this legacy serialization format saves
            # the numel of the storage, rather than nbytes directly, we'll be
            # effectively saving nbytes in this case.  We'll be able to load it
            # and the tensor back up with no problems in _this_ and future
            # versions of pytorch, but in older versions, here's the problem:
            # the storage will be loaded up as a UntypedStorage, and then the
            # FloatTensor will loaded and the UntypedStorage will be assigned to
            # it. Since the storage dtype does not match the tensor dtype, this
            # will cause an error.  If we reverse the list, like `[tensor,
            # storage]`, then we will save the `tensor.storage()` as a faked
            # `FloatStorage`, and the saved size will be the correct
            # dtype-specific numel count that old versions expect. `tensor`
            # will be able to load up properly in old versions, pointing to
            # a FloatStorage. However, `storage` is still being translated to
            # a UntypedStorage, and it will try to resolve to the same
            # FloatStorage that `tensor` contains. This will also cause an
            # error. It doesn't seem like there's any way around this.
            # Probably, we just cannot maintain FC for the legacy format if the
            # saved list contains both a tensor and a storage that point to the
            # same data.  We should still be able to maintain FC for lists of
            # just tensors, as long as all views share the same dtype as the
            # tensor they are viewing.

            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (storage, dtype)
            is_view = storage._cdata != storage._cdata
            if is_view:
                view_metadata = (str(storage._cdata), offset, storage.nbytes())
            else:
                view_metadata = None

            res = (
                "storage",
                storage_type,
                storage_key,
                location,
                storage_numel,
                view_metadata,
            )
            return res
        return None

    sys_info = {
        "protocol_version": PROTOCOL_VERSION,
        "little_endian": sys.byteorder == "little",
        "type_sizes": {
            "short": SHORT_SIZE,
            "int": INT_SIZE,
            "long": LONG_SIZE,
        },
    }

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)

    class PyTorchLegacyPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)  # noqa: F821

    pickler = PyTorchLegacyPickler(f, protocol=pickle_protocol)
    pickler.dump(obj)

    # The class def keeps the persistent_id closure alive, leaking memory.
    del persistent_id

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        storage, dtype = serialized_storages[key]
        storage._write_file(
            f, _should_read_directly(f), True, torch._utils._element_size(dtype)
        )


def _save(
    obj,
    zip_file,
    pickle_module,
    pickle_protocol,
    _disable_byteorder_record,
):
    serialized_storages: dict[str, torch.storage.UntypedStorage] = {}
    id_map: dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: dict[int, torch.dtype] = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(obj, "_fake_device") and obj._fake_device is not None:
                location = str(obj._fake_device)
            else:
                location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()

    class PyTorchPickler(pickle_module.Pickler):  # type: ignore[name-defined]
        def persistent_id(self, obj):
            return persistent_id(obj)  # noqa: F821

    pickler = PyTorchPickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)

    # The class def keeps the persistent_id closure alive, leaking memory.
    del persistent_id

    data_value = data_buf.getvalue()
    zip_file.write_record("data.pkl", data_value, len(data_value))
    # .format_version is used to track
    #     1. version 1 represents the order of storages being changed from
    #        lexicographical based on keys to numerically ordered based on keys
    #     2. version 2 represents including storage_alignment as a record
    #        within the zipfile
    zip_file.write_record(".format_version", "1", len("1"))
    storage_alignment = str(_get_storage_alignment())
    zip_file.write_record(
        ".storage_alignment", storage_alignment, len(storage_alignment)
    )

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in serialized_storages:
        name = f"data/{key}"
        storage = serialized_storages[key]
        num_bytes = storage.nbytes()
        global _serialization_tls
        if _serialization_tls.skip_data:
            zip_file.write_record_metadata(name, num_bytes)
        else:
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != "cpu":
                from torch.utils.serialization import config

                if (
                    config.save.use_pinned_memory_for_d2h
                    and (
                        acc := torch.accelerator.current_accelerator(
                            check_available=True
                        )
                    )
                    is not None
                    and acc.type == storage.device.type
                ):
                    new_storage = torch.empty(
                        num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                    ).untyped_storage()
                    new_storage.copy_(storage)
                    torch.accelerator.current_stream(storage.device.index).synchronize()
                    storage = new_storage
                else:
                    storage = storage.cpu()
            # Now that it is on the CPU we can directly copy it into the zip file
            zip_file.write_record(name, storage, num_bytes)


def load(
    f: FileLike,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

    .. warning::
        :func:`torch.load()` uses an unpickler under the hood. **Never load data from an untrusted source.**

        See :ref:`weights-only-security` for more details.

    :func:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``'cpu'``
    for CPU tensors and ``'cuda:device_id'`` (e.g. ``'cuda:2'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn't specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`serialization.py_kw.md_docs.md`](./serialization.py_kw.md_docs.md)
- [`library.py_kw.md_docs.md`](./library.py_kw.md_docs.md)
- [`overrides.py_docs.md_docs.md`](./overrides.py_docs.md_docs.md)
- [`script.h_kw.md_docs.md`](./script.h_kw.md_docs.md)
- [`_sources.py_kw.md_docs.md`](./_sources.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`_torch_docs.py_docs.md_docs.md`](./_torch_docs.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `serialization.py_docs.md_docs.md`
- **Keyword Index**: `serialization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
