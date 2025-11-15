# Documentation: `docs/torch/_inductor/compile_worker/tracked_process_pool.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/compile_worker/tracked_process_pool.py_docs.md`
- **Size**: 6,503 bytes (6.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/compile_worker/tracked_process_pool.py`

## File Metadata

- **Path**: `torch/_inductor/compile_worker/tracked_process_pool.py`
- **Size**: 3,693 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import atexit
import concurrent
import dataclasses
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from time import time
from typing import Any, Optional, TypeVar
from typing_extensions import ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401


_P = ParamSpec("_P")
_R = TypeVar("_R")


log = logging.getLogger(__name__)


@dataclass
class _QueueStats:
    # Mapping from id(future) -> start time
    pending: dict[int, float] = dataclasses.field(default_factory=dict)
    timing: list[float] = dataclasses.field(default_factory=list)
    enqueue_count: int = 0
    dequeue_count: int = 0
    max_queue_depth: int = 0
    pool_count: int = 0


# The queue statistics tracked by TrackedProcessPoolExecutor. Always grab
# _queue_stats_lock before touching.
_queue_stats = _QueueStats()
_queue_stats_lock = threading.Lock()


class TrackedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        max_workers: Optional[int] = None,
        mp_context: Optional[BaseContext] = None,
        initializer: Optional[Callable[[], object]] = None,
    ) -> None:
        with _queue_stats_lock:
            _queue_stats.pool_count += 1
        super().__init__(max_workers, mp_context, initializer)

    def _record_dequeue(self, f: Future[Any]) -> None:
        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            if (start_time := stats.pending.pop(id(f), None)) is None:
                return
            stats.dequeue_count += 1
            duration = now - start_time
            stats.timing.append(duration)

    def _record_enqueue(self, f: Future[Any]) -> None:
        # Monkeypatch the set_running_or_notify_cancel so we can track when the Future moves out of PENDING.
        saved_running_or_notify_cancel = f.set_running_or_notify_cancel

        def set_running_or_notify_cancel() -> Any:
            self._record_dequeue(f)
            return saved_running_or_notify_cancel()

        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            stats.pending[id(f)] = now
            stats.enqueue_count += 1
            stats.max_queue_depth = max(stats.max_queue_depth, len(stats.pending))
            f.set_running_or_notify_cancel = set_running_or_notify_cancel  # type: ignore[method-assign]

        if f._state != concurrent.futures._base.PENDING:
            self._record_dequeue(f)

    def submit(
        self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_R]:
        # pyrefly: ignore [bad-argument-type]
        f = super().submit(fn, *args, **kwargs)
        self._record_enqueue(f)
        return f


@atexit.register
def _queue_stats_report() -> None:
    stats = _queue_stats
    if stats.pool_count == 0:
        return

    timing = stats.timing
    timing.sort()

    log.info("AsyncCompile Metrics:")
    log.info("  Pools %s", stats.pool_count)
    log.info(
        "  Items %d enqueued / %d dequeued", stats.enqueue_count, stats.dequeue_count
    )
    log.info("  Max Queue Depth: %d", stats.max_queue_depth)
    n = len(timing)
    if n > 0:
        log.info("  Longest queue time: %0.2fs", timing[-1])
        log.info("  P50: %0.2fs", timing[n // 2])
        if n >= 20:
            log.info("  P95: %0.2fs", timing[n * 95 // 100])

```



## High-Level Overview


This Python file contains 3 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_QueueStats`, `TrackedProcessPoolExecutor`

**Functions defined**: `__init__`, `_record_dequeue`, `_record_enqueue`, `set_running_or_notify_cancel`, `submit`, `_queue_stats_report`

**Key imports**: atexit, concurrent, dataclasses, logging, threading, Callable, Future, ProcessPoolExecutor, dataclass, BaseContext, time


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/compile_worker`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `atexit`
- `concurrent`
- `dataclasses`
- `logging`
- `threading`
- `collections.abc`: Callable
- `concurrent.futures`: Future, ProcessPoolExecutor
- `multiprocessing.context`: BaseContext
- `time`: time
- `typing`: Any, Optional, TypeVar
- `typing_extensions`: ParamSpec
- `installs`
- `torch._thread_safe_fork  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_inductor/compile_worker`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`timer.py_docs.md`](./timer.py_docs.md)
- [`subproc_pool.py_docs.md`](./subproc_pool.py_docs.md)
- [`__main__.py_docs.md`](./__main__.py_docs.md)


## Cross-References

- **File Documentation**: `tracked_process_pool.py_docs.md`
- **Keyword Index**: `tracked_process_pool.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/compile_worker`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/compile_worker`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_inductor/compile_worker`):

- [`subproc_pool.py_kw.md_docs.md`](./subproc_pool.py_kw.md_docs.md)
- [`subproc_pool.py_docs.md_docs.md`](./subproc_pool.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`timer.py_kw.md_docs.md`](./timer.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`timer.py_docs.md_docs.md`](./timer.py_docs.md_docs.md)
- [`tracked_process_pool.py_kw.md_docs.md`](./tracked_process_pool.py_kw.md_docs.md)
- [`__main__.py_docs.md_docs.md`](./__main__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tracked_process_pool.py_docs.md_docs.md`
- **Keyword Index**: `tracked_process_pool.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
