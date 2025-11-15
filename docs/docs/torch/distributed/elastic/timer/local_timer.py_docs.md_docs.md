# Documentation: `docs/torch/distributed/elastic/timer/local_timer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/timer/local_timer.py_docs.md`
- **Size**: 7,255 bytes (7.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/timer/local_timer.py`

## File Metadata

- **Path**: `torch/distributed/elastic/timer/local_timer.py`
- **Size**: 4,282 bytes (4.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


class LocalTimerClient(TimerClient):
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """

    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)


class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> list[TimerRequest]:
        requests = []
        wait = timeout
        for _ in range(size):
            start = time.time()

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break

            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break

        return requests


class LocalTimerServer(TimerServer):
    """
    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    """

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._timers: dict[tuple[Any, str], TimerRequest] = {}

    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                self._timers[(pid, scope_id)] = request

    def clear_timers(self, worker_ids: set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_ids:
                self._timers.pop((pid, scope_id))

    def get_expired_timers(self, deadline: float) -> dict[Any, list[TimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers: dict[Any, list[TimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_id: int) -> bool:
        try:
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_id)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_id)
        return False

```



## High-Level Overview

"""    Client side of ``LocalTimerServer``. This client is meant to be used    on the same host that the ``LocalTimerServer`` is running on and uses    pid to uniquely identify a worker. This is particularly useful in situations    where one spawns a subprocess (trainer) per GPU on a host with multiple    GPU devices.

This Python file contains 3 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LocalTimerClient`, `MultiprocessingRequestQueue`, `LocalTimerServer`

**Functions defined**: `__init__`, `acquire`, `release`, `__init__`, `size`, `get`, `__init__`, `register_timers`, `clear_timers`, `get_expired_timers`, `_reap_worker`

**Key imports**: logging, multiprocessing as mp, os, signal, time, Empty, Any, RequestQueue, TimerClient, TimerRequest, TimerServer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/timer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `multiprocessing as mp`
- `os`
- `signal`
- `time`
- `queue`: Empty
- `typing`: Any
- `.api`: RequestQueue, TimerClient, TimerRequest, TimerServer


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/elastic/timer`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`debug_info_logging.py_docs.md`](./debug_info_logging.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`file_based_local_timer.py_docs.md`](./file_based_local_timer.py_docs.md)


## Cross-References

- **File Documentation**: `local_timer.py_docs.md`
- **Keyword Index**: `local_timer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/elastic/timer`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/elastic/timer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/elastic/timer`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`file_based_local_timer.py_kw.md_docs.md`](./file_based_local_timer.py_kw.md_docs.md)
- [`file_based_local_timer.py_docs.md_docs.md`](./file_based_local_timer.py_docs.md_docs.md)
- [`debug_info_logging.py_docs.md_docs.md`](./debug_info_logging.py_docs.md_docs.md)
- [`debug_info_logging.py_kw.md_docs.md`](./debug_info_logging.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`local_timer.py_kw.md_docs.md`](./local_timer.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`api.py_docs.md_docs.md`](./api.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `local_timer.py_docs.md_docs.md`
- **Keyword Index**: `local_timer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
