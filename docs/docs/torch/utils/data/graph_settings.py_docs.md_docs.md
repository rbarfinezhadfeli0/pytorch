# Documentation: `docs/torch/utils/data/graph_settings.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/graph_settings.py_docs.md`
- **Size**: 8,145 bytes (7.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/graph_settings.py`

## File Metadata

- **Path**: `torch/utils/data/graph_settings.py`
- **Size**: 5,556 bytes (5.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import inspect
import warnings
from typing import Any
from typing_extensions import deprecated

import torch
from torch.utils.data.datapipes.iter.sharding import (
    _ShardingIterDataPipe,
    SHARDING_PRIORITIES,
)
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps


__all__ = [
    "apply_random_seed",
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]


def get_all_graph_pipes(graph: DataPipeGraph) -> list[DataPipe]:
    return _get_all_graph_pipes_helper(graph, set())


def _get_all_graph_pipes_helper(
    graph: DataPipeGraph, id_cache: set[int]
) -> list[DataPipe]:
    results: list[DataPipe] = []
    for dp_id, (datapipe, sub_graph) in graph.items():
        if dp_id in id_cache:
            continue
        id_cache.add(dp_id)
        results.append(datapipe)
        results.extend(_get_all_graph_pipes_helper(sub_graph, id_cache))
    return results


def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    return isinstance(datapipe, _ShardingIterDataPipe) or (
        hasattr(datapipe, "apply_sharding")
        and inspect.ismethod(datapipe.apply_sharding)
    )


def apply_sharding(
    datapipe: DataPipe,
    num_of_instances: int,
    instance_id: int,
    sharding_group=SHARDING_PRIORITIES.DEFAULT,
) -> DataPipe:
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.

    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None) -> None:
        for dp, sub_graph in graph.values():
            applied = None
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError(
                        "Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                        f"Sharding already applied to {prev_applied} while trying to apply to {dp}"
                    )
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(
                        num_of_instances, instance_id, sharding_group=sharding_group
                    )
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe


def _is_shuffle_datapipe(datapipe: DataPipe) -> bool:
    return (
        hasattr(datapipe, "set_shuffle")
        and hasattr(datapipe, "set_seed")
        and inspect.ismethod(datapipe.set_shuffle)
        and inspect.ismethod(datapipe.set_seed)
    )


def apply_shuffle_settings(datapipe: DataPipe, shuffle: bool | None = None) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    if shuffle is None:
        return datapipe

    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task.",
            stacklevel=2,
        )
        datapipe = datapipe.shuffle()
        shufflers = [
            datapipe,
        ]

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe


@deprecated(
    "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases. "
    "Please use `apply_random_seed` instead.",
    category=FutureWarning,
)
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    return apply_random_seed(datapipe, rng)


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    return hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed)


def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    # Using a set to track id of DataPipe to prevent setting randomness per DataPipe more than once.
    # And, `id` is used in case of unhashable DataPipe
    cache = set()
    random_datapipes = []
    for pipe in all_pipes:
        if id(pipe) in cache:
            continue
        if _is_random_datapipe(pipe):
            random_datapipes.append(pipe)
            cache.add(id(pipe))

    for pipe in random_datapipes:
        random_seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=rng).item()
        )
        pipe.set_seed(random_seed)

    return datapipe

```



## High-Level Overview


This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_all_graph_pipes`, `_get_all_graph_pipes_helper`, `_is_sharding_datapipe`, `apply_sharding`, `_helper`, `_is_shuffle_datapipe`, `apply_shuffle_settings`, `apply_shuffle_seed`, `_is_random_datapipe`, `apply_random_seed`

**Key imports**: inspect, warnings, Any, deprecated, torch, DataPipe, DataPipeGraph, traverse_dps


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `warnings`
- `typing`: Any
- `typing_extensions`: deprecated
- `torch`
- `torch.utils.data.graph`: DataPipe, DataPipeGraph, traverse_dps


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/utils/data`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`dataset.py_docs.md`](./dataset.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`backward_compatibility.py_docs.md`](./backward_compatibility.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`sampler.py_docs.md`](./sampler.py_docs.md)
- [`dataloader.py_docs.md`](./dataloader.py_docs.md)


## Cross-References

- **File Documentation**: `graph_settings.py_docs.md`
- **Keyword Index**: `graph_settings.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/utils/data`):

- [`backward_compatibility.py_kw.md_docs.md`](./backward_compatibility.py_kw.md_docs.md)
- [`graph_settings.py_kw.md_docs.md`](./graph_settings.py_kw.md_docs.md)
- [`graph.py_kw.md_docs.md`](./graph.py_kw.md_docs.md)
- [`sampler.py_kw.md_docs.md`](./sampler.py_kw.md_docs.md)
- [`dataloader.py_kw.md_docs.md`](./dataloader.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`dataset.py_kw.md_docs.md`](./dataset.py_kw.md_docs.md)
- [`distributed.py_docs.md_docs.md`](./distributed.py_docs.md_docs.md)
- [`sampler.py_docs.md_docs.md`](./sampler.py_docs.md_docs.md)
- [`dataset.py_docs.md_docs.md`](./dataset.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `graph_settings.py_docs.md_docs.md`
- **Keyword Index**: `graph_settings.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
