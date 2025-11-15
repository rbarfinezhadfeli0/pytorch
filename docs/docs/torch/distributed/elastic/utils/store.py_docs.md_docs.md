# Documentation: `docs/torch/distributed/elastic/utils/store.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/utils/store.py_docs.md`
- **Size**: 9,784 bytes (9.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/utils/store.py`

## File Metadata

- **Path**: `torch/distributed/elastic/utils/store.py`
- **Size**: 7,278 bytes (7.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import timedelta
from typing import Optional

import torch


DistStoreError = torch._C._DistStoreError

_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"
_TRACE = "/TRACE"
_TRACING_GATE = "/TRACING_GATE"
_MAX_TRACE_MISSING_RANKS = 16


__all__ = ["store_timeout", "get_all", "synchronize", "barrier"]


@contextmanager
def store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """

    old_timeout = store.timeout
    store.set_timeout(timedelta(seconds=timeout))
    yield
    store.set_timeout(old_timeout)


def get_all(store, rank: int, prefix: str, world_size: int):
    r"""
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    The Rank0 process waits at the end to make sure all other processes
    finished the procedure before exiting.

    Usage

    ::

     values = get_all(store, "torchelastic/data", 3)
     value1 = values[0]  # retrieves the data for key torchelastic/data0
     value2 = values[1]  # retrieves the data for key torchelastic/data1
     value3 = values[2]  # retrieves the data for key torchelastic/data2

    """
    data_arr = store.multi_get([f"{prefix}{idx}" for idx in range(world_size)])

    barrier_key = _barrier_nonblocking(
        store=store,
        world_size=world_size,
        key_prefix=f"{prefix}/finished",
    )
    if rank == 0:
        # Rank0 runs the TCPStore daemon, as a result it needs to exit last.
        # Otherwise, the barrier may timeout if rank0 process finished the work
        # before other processes finished `get_all` method
        store.wait([barrier_key])

    return data_arr


def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    timeout: float = 300,
) -> list[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.

    Time complexity: O(N) per worker, O(N^2) globally.
    """
    with store_timeout(store, timeout):
        store.set(f"{key_prefix}{rank}", data)
        agent_data = get_all(store, rank, key_prefix, world_size)
        return agent_data


def _try_detecting_missing_ranks(
    store,
    world_size: int,
    key_prefix: str,
    rank: int,
    rank_decoder: Callable[[int], str],
    trace_timeout: float,
) -> Optional[Iterable[str]]:
    store.set(f"{key_prefix}{rank}{_TRACE}", "<val_ignored>")

    def _find_missing_ranks():
        missing_rank_info = set()
        ranks_missing = 0
        for i in range(1, world_size):
            # reduce noise, assuming in general 8 ranks per node
            # It is valuable to know that 1 or >1 nodes have timed-out.
            if ranks_missing >= _MAX_TRACE_MISSING_RANKS:
                break
            try:
                if ranks_missing == 0:
                    store.wait(
                        [f"{key_prefix}{i}{_TRACE}"], timedelta(seconds=trace_timeout)
                    )
                else:
                    # use a shortest timeout, some ranks have failed to check-in
                    store.wait([f"{key_prefix}{i}{_TRACE}"], timedelta(milliseconds=1))
            except DistStoreError:
                ranks_missing += 1
                missing_rank_info.add(rank_decoder(i))
        return missing_rank_info

    def _checkin():
        try:
            store.wait([f"{key_prefix}{_TRACING_GATE}"])
            return [f"[<check rank 0 ({rank_decoder(0)}) for missing rank info>]"]
        except DistStoreError:
            # in case rank0 is the source of the timeout, original exception will be raised
            return None

    if rank == 0:
        missing_rank_info = _find_missing_ranks()
        store.set(f"{key_prefix}{_TRACING_GATE}", "<val_ignored>")
        return missing_rank_info
    else:
        return _checkin()


def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns the final key
    that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN

    idx = store.add(num_members_key, 1)
    if idx == world_size:
        store.set(last_member_key, "<val_ignored>")

    return last_member_key


def barrier(
    store,
    world_size: int,
    key_prefix: str,
    barrier_timeout: float = 300,
    rank: Optional[int] = None,
    rank_tracing_decoder: Optional[Callable[[int], str]] = None,
    trace_timeout: float = 10,
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Optionally, passing rank will enable tracing of missing ranks on timeouts.
    `rank_tracing_decoder` lambda arg can be used to convert rank data
    into a more meaningful information at an app level (e.g. hostname).

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    if rank is None:
        assert rank_tracing_decoder is None, "Tracing requires rank information"

    with store_timeout(store, barrier_timeout):
        last_member_key = _barrier_nonblocking(
            store=store, world_size=world_size, key_prefix=key_prefix
        )
        try:
            store.wait([last_member_key])
        except DistStoreError as e:
            if rank is None:
                raise e
            else:
                missing_ranks = _try_detecting_missing_ranks(
                    store,
                    world_size,
                    key_prefix,
                    rank,
                    rank_tracing_decoder or (lambda x: str(x)),
                    trace_timeout,
                )
                if missing_ranks is not None:
                    raise DistStoreError(
                        "Timed out waiting on barrier on "
                        "rank {}, for key prefix: {} (world_size={}, missing_ranks={}, timeout={})".format(
                            rank,
                            key_prefix,
                            world_size,
                            f"[{', '.join(missing_ranks)}]",
                            barrier_timeout,
                        )
                    ) from None
                else:
                    raise e

```



## High-Level Overview

"""    This sets the timeout and then restores the old timeout when the context    manager exits.    Args:        store: the store to set the timeout on        timeout: the timeout to set

This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `store_timeout`, `get_all`, `synchronize`, `_try_detecting_missing_ranks`, `_find_missing_ranks`, `_checkin`, `_barrier_nonblocking`, `barrier`

**Key imports**: Callable, Iterable, contextmanager, timedelta, Optional, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable, Iterable
- `contextlib`: contextmanager
- `datetime`: timedelta
- `typing`: Optional
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/distributed/elastic/utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`log_level.py_docs.md`](./log_level.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`logging.py_docs.md`](./logging.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)


## Cross-References

- **File Documentation**: `store.py_docs.md`
- **Keyword Index**: `store.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/elastic/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/elastic/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/distributed/elastic/utils`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`log_level.py_docs.md_docs.md`](./log_level.py_docs.md_docs.md)
- [`log_level.py_kw.md_docs.md`](./log_level.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`store.py_kw.md_docs.md`](./store.py_kw.md_docs.md)
- [`distributed.py_docs.md_docs.md`](./distributed.py_docs.md_docs.md)
- [`distributed.py_kw.md_docs.md`](./distributed.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`logging.py_docs.md_docs.md`](./logging.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `store.py_docs.md_docs.md`
- **Keyword Index**: `store.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
