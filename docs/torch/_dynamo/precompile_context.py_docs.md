# Documentation: `torch/_dynamo/precompile_context.py`

## File Metadata

- **Path**: `torch/_dynamo/precompile_context.py`
- **Size**: 7,420 bytes (7.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import copy
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

    key: str
    content: Any

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    def after_deserialization(self) -> Any:
        return self.content


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext:
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[_BackendId, BackendCacheArtifact[Any]] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact[Any],
    ) -> None:
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        cls._backend_artifacts_by_key[_BackendId(artifact.key)] = copy.deepcopy(
            artifact
        )

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        assert key in cls._backend_artifacts_by_key, f"Key {key} not found in artifacts"
        artifact = cls._backend_artifacts_by_key[_BackendId(key)]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[BackendCacheArtifact[Any]]:
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(_BackendId(key), None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[_BackendId, BackendCacheArtifact[Any]],
    ) -> dict[str, Any]:
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[str, list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                result = PrecompileCacheEntry.from_cache_entry(
                    cache_entry, backend_artifacts
                )
                if result is not None:
                    precompile_cache_entries[key] = result
            except Exception as e:
                logger.warning("Failed to create cache entry %s", key, exc_info=True)

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue
        return precompile_cache_entries, debug_info

```



## High-Level Overview

"""Classes and implementations related to precompile

This Python file contains 6 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BackendCacheArtifact`, `MyPrecompileCacheArtifact`, `EagerCacheArtifact`, `BypassDynamoCacheEntry`, `PrecompileContext`

**Functions defined**: `after_deserialization`, `after_deserialization`, `edit_contents`, `after_deserialization`, `clear`, `record_artifact`, `record_dynamo_cache_entry`, `edit_artifact`, `serialize_artifact_by_key`, `dump_debug_info`, `save_to_dynamo_cache`, `create_cache_entries`

**Key imports**: copy, json, logging, abstractmethod, defaultdict, Callable, dataclass, Any, Generic, Optional, TypeVar, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `json`
- `logging`
- `abc`: abstractmethod
- `collections`: defaultdict
- `collections.abc`: Callable
- `dataclasses`: dataclass
- `typing`: Any, Generic, Optional, TypeVar
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `precompile_context.py_docs.md`
- **Keyword Index**: `precompile_context.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
