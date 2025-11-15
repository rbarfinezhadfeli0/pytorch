# Documentation: `torch/_inductor/cache.py`

## File Metadata

- **Path**: `torch/_inductor/cache.py`
- **Size**: 14,823 bytes (14.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from ast import literal_eval
from functools import cached_property
from hashlib import sha256
from os import getenv
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, override, Self

from torch.utils._filelock import FileLock


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


# TypeVars can't be recursive, so generic types that fall within
# Key or Value can't be bound properly; for example, Key should
# only take tuples of other Key types: tuple[Key, ...]. this is
# a known shortcoming of torch's typing
Key = TypeVar("Key", str, int, tuple[Any, ...])
Value = TypeVar("Value", str, int, tuple[Any, ...], bytes, dict[Any, Any], list[Any])


class CacheError(ValueError):
    """
    Exception raised for errors encountered during cache operations.
    """


class Cache(ABC, Generic[Key, Value]):
    """
    Abstract base class for cache implementations.
    Provides the interface for cache operations.
    """

    @abstractmethod
    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """

    @abstractmethod
    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """


class InMemoryCache(Cache[Key, Value]):
    """
    In-memory cache implementation using a dictionary and thread lock.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an empty in-memory cache.
        """
        self._cache: dict[Key, Value] = {}
        self._lock: Lock = Lock()

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """
        with self._lock:
            if (value := self._cache.get(key)) is not None:
                return value
            return None

    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """
        with self._lock:
            if key in self._cache:
                # no overwrites for insert!
                return False
            self._cache[key] = value
            return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        """
        Create an in-memory cache from an environment variable.
        Args:
            env_var (str): Name of the environment variable containing cache data.
        Returns:
            InMemoryCache: An instance populated from the environment variable.
        Raises:
            CacheError: If the environment variable is malformed or contains invalid data.
        """
        cache = cls()

        if (env_val := getenv(env_var)) is None:
            # env_var doesn't exist = empty cache
            return cache

        for kv_pair in env_val.split(";"):
            # ignore whitespace prefix/suffix
            kv_pair = kv_pair.strip()

            if not kv_pair:
                # kv_pair could be '' if env_val is '' or has ; suffix
                continue

            try:
                # keys and values should be comma separated
                key_bytes_repr, value_bytes_repr = kv_pair.split(",", 1)
            except ValueError as err:
                raise CacheError(
                    f"Malformed kv_pair {kv_pair!r} from env_var {env_var!r}, likely missing comma separator."
                ) from err

            # ignore whitespace prefix/suffix, again
            key_bytes_repr, value_bytes_repr = (
                key_bytes_repr.strip(),
                value_bytes_repr.strip(),
            )

            try:
                # check that key_bytes_str is an actual, legitimate encoding
                key_bytes = literal_eval(key_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err
            try:
                # check that value_bytes_str is an actual, legitimate encoding
                value_bytes = literal_eval(value_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err

            try:
                key = pickle.loads(key_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err
            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err

            # true duplicates, i.e. multiple occurrences of the same key => value
            # mapping are ok and treated as a no-op; key duplicates with differing
            # values, i.e. key => value_1 and key => value_2 where value_1 != value_2,
            # are not okay since we don't allow overwriting cached values (it's bad regardless)
            if (not cache.insert(key, value)) and (cache.get(key) != value):
                raise CacheError(
                    f"Multiple values for key {key!r} found, got {cache.get(key)!r} and {value!r}."
                )

        return cache

    @classmethod
    def from_file_path(cls, fpath: Path) -> Self:
        """
        Create an in-memory cache from a file path.
        Args:
            fpath (Path): Path to the file containing pickled cache data.
        Returns:
            InMemoryCache: An instance populated from the file.
        Raises:
            CacheError: If the file is not a valid pickled dictionary.
        """
        cache = cls()

        if not fpath.is_file():
            # fpath doesn't exit = empty cache
            return cache

        try:
            with open(fpath, "rb") as fp:
                cache._cache = pickle.load(fp)
        except pickle.UnpicklingError as err:
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents are un-pickle-able."
            ) from err

        if not isinstance(cache._cache, dict):
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents not pickled dict[Key, Value]."
            )

        return cache


class AsyncCache(Cache[Key, Value]):
    """
    Asynchronous cache implementation using ThreadPoolExecutor.
    """

    def get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        """
        Retrieve a value from the cache asynchronously.
        Args:
            key (Key): The key to look up.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[Value | None]: Future for the cached value or None.
        """
        return executor.submit(self.get, key)

    def insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        """
        Insert a value into the cache asynchronously.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[bool]: Future for the result of insertion.
        """
        return executor.submit(self.insert, key, value)


class OnDiskCache(AsyncCache[Key, Value]):
    """
    On-disk cache implementation using files and file locks.
    Stores cache data in files on disk, with atomic operations and versioning.
    Supports custom cache directory names.
    Attributes:
        version (int): The version used for cache versioning.
        name (str): The name of the cache directory.
    """

    version: int = 0

    def __init__(self: Self, name: str | None = None) -> None:
        """
        Initialize an on-disk cache instance.
        Args:
            name (str | None, optional): The name of the cache directory. If None,
                defaults to "on_disk_cache".
        """
        self.name = name or "on_disk_cache"

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the cache.
        Returns:
            Path: The base directory path for storing cache files.
        """
        return Path(gettempdir()) / "cache" / self.name

    def _fpath_from_key(self: Self, key: Key) -> Path:
        """
        Get the file path for a given key.
        Args:
            key (Key): The key to convert to a file path.
        Returns:
            Path: The file path for the key.
        Raises:
            CacheError: If the key is not pickle-able.
        """
        try:
            return self.base_dir / sha256(pickle.dumps(key)).hexdigest()[:32]
        except (AttributeError, pickle.PicklingError) as err:
            raise CacheError(
                f"Failed to get fpath for key {key!r}, key is not pickle-able."
            ) from err
        # pyrefly: ignore [bad-argument-type]
        assert_never(key)

    def _flock_from_fpath(self: Self, fpath: Path) -> FileLock:
        """
        Get a file lock for a given file path.
        Args:
            fpath (Path): The file path.
        Returns:
            FileLock: The file lock for the path.
        """
        # fpath.name is a hex digest, meaning there are 16^4 potential values
        # for fpath.name[:4]; this is more than enough unique locks to not
        # cause additional overhead from shared locks and it also saves our
        # cache dir from becoming 50 percent locks
        # pyrefly: ignore [bad-return]
        return FileLock(str(fpath.parent / "locks" / fpath.name[:4]) + ".lock")

    @property
    def version_prefix(self: Self) -> bytes:
        """
        Get the version prefix for the cache.
        Returns:
            bytes: The version prefix as bytes, derived from the cache version string.
        """
        return sha256(str(OnDiskCache.version).encode()).digest()[:4]

    @override
    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present and version matches, else None.
        Raises:
            CacheError: If the value is corrupted or cannot be unpickled.
        Side Effects:
            Removes stale cache files if the version prefix does not match.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)

        with flock:
            if not fpath.is_file():
                return None

            value_bytes = None
            prefix_length = len(self.version_prefix)
            with open(fpath, "rb") as fp:
                if fp.read(prefix_length) == self.version_prefix:
                    value_bytes = fp.read()

            if value_bytes is None:
                # version_prefix did not match, so we can't read the stale
                # cached value; we should also remove the stale cached value,
                # so that key can be re-cached by the newer version
                fpath.unlink()
                return None

            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Failed to get key {key!r}, value is potentially corrupted (value is not un-pickle-able)."
                ) from err

            return value

    @override
    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        Raises:
            CacheError: If the value is not pickle-able.
        Side Effects:
            Creates the cache directory if it does not exist.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite); use
            # flock for added atomicity guarantee and to prevent partial writes
            with flock as _, open(fpath, "xb") as fp:
                fp.write(self.version_prefix)
                pickle.dump(value, fp)
        except pickle.PicklingError as err:
            raise CacheError(
                f"Failed to insert key {key!r} with value {value!r}, value is not pickle-able."
            ) from err
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache[Key, Value]):
    """
    Inductor-specific on-disk cache implementation.
    Uses a custom base directory for Inductor cache files.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an inductor on-disk cache instance.
        Sets the cache directory name to "inductor_on_disk_cache".
        """
        super().__init__("inductor_on_disk_cache")

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the Inductor cache.
        Returns:
            Path: The base directory path for Inductor cache files.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache", self.name)

```



## High-Level Overview

"""    Exception raised for errors encountered during cache operations.

This Python file contains 7 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CacheError`, `Cache`, `InMemoryCache`, `AsyncCache`, `OnDiskCache`, `InductorOnDiskCache`

**Functions defined**: `get`, `insert`, `__init__`, `get`, `insert`, `from_env_var`, `from_file_path`, `get_async`, `insert_async`, `__init__`, `base_dir`, `_fpath_from_key`, `_flock_from_fpath`, `version_prefix`, `get`, `insert`, `__init__`, `base_dir`

**Key imports**: annotations, pickle, ABC, abstractmethod, literal_eval, cached_property, sha256, getenv, Path, gettempdir, Lock


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `pickle`
- `abc`: ABC, abstractmethod
- `ast`: literal_eval
- `functools`: cached_property
- `hashlib`: sha256
- `os`: getenv
- `pathlib`: Path
- `tempfile`: gettempdir
- `threading`: Lock
- `typing`: Any, Generic, TYPE_CHECKING, TypeVar
- `typing_extensions`: assert_never, override, Self
- `torch.utils._filelock`: FileLock
- `concurrent.futures`: Future, ThreadPoolExecutor
- `torch._inductor.runtime.runtime_utils`: default_cache_dir


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `cache.py_docs.md`
- **Keyword Index**: `cache.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
