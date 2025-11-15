# Documentation: `torch/utils/data/_utils/signal_handling.py`

## File Metadata

- **Path**: `torch/utils/data/_utils/signal_handling.py`
- **Size**: 3,261 bytes (3.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""Signal handling for multiprocessing data loading.

NOTE [ Signal handling in multiprocessing data loading ]

In cases like DataLoader, if a worker process dies due to bus error/segfault
or just hang, the main process will hang waiting for data. This is difficult
to avoid on PyTorch side as it can be caused by limited shm, or other
libraries users call in the workers. In this file and `DataLoader.cpp`, we make
our best effort to provide some error message to users when such unfortunate
events happen.

When a _BaseDataLoaderIter starts worker processes, their pids are registered in a
defined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]
via `_set_worker_pids`.

When an error happens in a worker process, the main process received a SIGCHLD,
and Python will eventually call the handler registered below
(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`
call checks all registered worker pids and raise proper error message to
prevent main process from hanging waiting for data from worker.

Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,
`_set_worker_signal_handlers` is called to register critical signal handlers
(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an error
message to stderr before triggering the default handler. So a message will also
be printed from the worker process when it is killed by such signals.

See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning of
this signal handling design and other mechanism we implement to make our
multiprocessing data loading robust to errors.
"""

import signal
import threading

# Some of the following imported functions are not used in this file, but are to
# be used `_utils.signal_handling.XXXXX`.
from torch._C import (  # noqa: F401
    _error_if_any_worker_fails,
    _remove_worker_pids,
    _set_worker_pids,
    _set_worker_signal_handlers,
)

from . import IS_WINDOWS


_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler() -> None:
    # Windows doesn't support SIGCHLD handler
    if IS_WINDOWS:
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):  # type: ignore[attr-defined]
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        # This doesn't catch default handler, but SIGCHLD default handler is a
        # no-op.
        previous_handler = None

    def handler(signum, frame) -> None:
        # This following call uses `waitid` with WNOHANG from C side. Therefore,
        # Python can still get and update the process status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            if not callable(previous_handler):
                raise AssertionError("previous_handler is not callable")
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True

```



## High-Level Overview

r"""Signal handling for multiprocessing data loading.NOTE [ Signal handling in multiprocessing data loading ]In cases like DataLoader, if a worker process dies due to bus error/segfaultor just hang, the main process will hang waiting for data. This is difficultto avoid on PyTorch side as it can be caused by limited shm, or otherlibraries users call in the workers. In this file and `DataLoader.cpp`, we makeour best effort to provide some error message to users when such unfortunateevents happen.When a _BaseDataLoaderIter starts worker processes, their pids are registered in adefined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]via `_set_worker_pids`.When an error happens in a worker process, the main process received a SIGCHLD,and Python will eventually call the handler registered below(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`call checks all registered worker pids and raise proper error message toprevent main process from hanging waiting for data from worker.Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,`_set_worker_signal_handlers` is called to register critical signal handlers(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an errormessage to stderr before triggering the default handler. So a message will alsobe printed from the worker process when it is killed by such signals.See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning ofthis signal handling design and other mechanism we implement to make ourmultiprocessing data loading robust to errors.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_set_SIGCHLD_handler`, `handler`

**Key imports**: signal, threading, IS_WINDOWS


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/_utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `signal`
- `threading`
- `.`: IS_WINDOWS


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/utils/data/_utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`worker.py_docs.md`](./worker.py_docs.md)
- [`fetch.py_docs.md`](./fetch.py_docs.md)
- [`pin_memory.py_docs.md`](./pin_memory.py_docs.md)
- [`collate.py_docs.md`](./collate.py_docs.md)


## Cross-References

- **File Documentation**: `signal_handling.py_docs.md`
- **Keyword Index**: `signal_handling.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
