# Documentation: `docs/torch/multiprocessing/spawn.py_docs.md`

## File Metadata

- **Path**: `docs/torch/multiprocessing/spawn.py_docs.md`
- **Size**: 15,875 bytes (15.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/multiprocessing/spawn.py`

## File Metadata

- **Path**: `torch/multiprocessing/spawn.py`
- **Size**: 12,935 bytes (12.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import sys
import tempfile
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Optional

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]


ENV_VAR_PARALLEL_START = "TORCH_MP_PARALLEL_START"

log = logging.getLogger(__name__)

__all__ = [
    "ProcessContext",
    "ProcessException",
    "ProcessExitedException",
    "ProcessRaisedException",
    "spawn",
    "SpawnContext",
    "start_processes",
]


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return type(self), (self.msg, self.error_index, self.pid)


class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
    ):
        super().__init__(msg, error_index, error_pid)


class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""

    __slots__ = ["exit_code"]

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: Optional[str] = None,
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (
            type(self),
            (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name),
        )


def _wrap(fn, i, args, error_file):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback

        with open(error_file, "wb") as fh:
            pickle.dump(traceback.format_exc(), fh)
        sys.exit(1)


class ProcessContext:
    def __init__(self, processes, error_files):
        self.error_files = error_files
        self.processes = processes
        self.sentinels = {
            process.sentinel: index for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def _join_procs_with_timeout(self, timeout: float):
        """Attempt to join all processes with a shared timeout."""
        end = time.monotonic() + timeout
        for process in self.processes:
            # pyrefly: ignore [no-matching-overload]
            time_to_wait = max(0, end - time.monotonic())
            process.join(time_to_wait)

    def join(
        self, timeout: Optional[float] = None, grace_period: Optional[float] = None
    ):
        r"""Join one or more processes within spawn context.

        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes (optionally with a grace period)
        and raises an exception with the cause of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long (in seconds) before giving up on waiting.
            grace_period (float): When any processes fail, wait this long (in seconds)
                for others to shutdown gracefully before terminating them. If they
                still don't exit, wait another grace period before killing them.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
        # An error occurred. Clean-up all processes before returning.
        # First, allow a grace period for processes to shutdown themselves.
        if grace_period is not None:
            self._join_procs_with_timeout(grace_period)
        # Then, terminate processes that are still alive. Try SIGTERM first.
        for process in self.processes:
            if process.is_alive():
                log.warning("Terminating process %s via signal SIGTERM", process.pid)
                process.terminate()

        # Try SIGKILL if the process isn't going down after another grace_period.
        # The reason is related to python signal handling is limited
        # to main thread and if that is in c/c++ land and stuck it won't
        # to handle it. We have seen processes getting stuck not handling
        # SIGTERM for the above reason.
        self._join_procs_with_timeout(30 if grace_period is None else grace_period)
        for process in self.processes:
            if process.is_alive():
                log.warning(
                    "Unable to shutdown process %s via SIGTERM , forcefully exiting via SIGKILL",
                    process.pid,
                )
                process.kill()
            process.join()

        # The file will only be created if the process crashed.
        failed_process = self.processes[error_index]
        if not os.access(self.error_files[error_index], os.R_OK):
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                try:
                    name = signal.Signals(-exitcode).name
                except ValueError:
                    name = f"<Unknown signal {-exitcode}>"
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with signal {name}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name,
                )
            else:
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with exit code {exitcode:d}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                )

        with open(self.error_files[error_index], "rb") as fh:
            original_trace = pickle.load(fh)
        msg = f"\n\n-- Process {error_index:d} terminated with the following error:\n"
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_files):
        warnings.warn(
            "SpawnContext is renamed to ProcessContext since 1.4 release.", stacklevel=2
        )
        super().__init__(processes, error_files)


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(
    fn,
    args=(),
    nprocs=1,
    join=True,
    daemon=False,
    start_method="spawn",
):
    # To speed up performance in certain cases (see https://github.com/pytorch/pytorch/issues/133010),
    # this func will start processes in parallel if start_method is 'forkserver'.
    # Please opt in to this perf optimization by setting env var (TORCH_MP_PARALLEL_START) to 1.
    # todo: investigate why spawn does not work with threadpool and raises SIGINT
    if (
        start_method == "forkserver"
        and os.environ.get(ENV_VAR_PARALLEL_START, "0") == "1"
    ):
        log.info("Starting processes in parallel.")
        start_parallel = True
    else:
        # Set env var TORCH_MP_PARALLEL_START to 0 to disable parallel start
        start_parallel = False

    mp = multiprocessing.get_context(start_method)
    error_files = [None] * nprocs
    processes = [None] * nprocs

    def start_process(i):
        # Each process is assigned a file to write tracebacks to.  We
        # use the file being non-empty to indicate an exception
        # occurred (vs an expected shutdown).  Note: this previously
        # used a multiprocessing.Queue but that can be prone to
        # deadlocks, so we went with a simpler solution for a one-shot
        # message between processes.
        tf = tempfile.NamedTemporaryFile(
            prefix="pytorch-errorfile-", suffix=".pickle", delete=False
        )
        tf.close()
        os.unlink(tf.name)

        process = mp.Process(  # pyrefly: ignore  # missing-attribute
            target=_wrap,
            args=(fn, i, args, tf.name),
            daemon=daemon,
        )

        process.start()
        return i, process, tf.name

    if not start_parallel:
        for i in range(nprocs):
            idx, process, tf_name = start_process(i)
            error_files[idx] = tf_name
            processes[idx] = process
    else:
        with ThreadPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(start_process, i) for i in range(nprocs)]
            for fut in as_completed(futures):
                idx, process, tf_name = fut.result()
                # idx and process rank needs to be the same.
                error_files[idx] = tf_name
                processes[idx] = process
    context = ProcessContext(processes, error_files)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != "spawn":
        msg = (
            f"This method only supports start_method=spawn (got: {start_method}).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)"
        )
        warnings.warn(msg, FutureWarning, stacklevel=2)
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")

```



## High-Level Overview

"""Exception raised when a process failed due to an exception raised by the code."""    def __init__(        self,

This Python file contains 5 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ProcessException`, `ProcessRaisedException`, `ProcessExitedException`, `ProcessContext`, `SpawnContext`

**Functions defined**: `__init__`, `__reduce__`, `__init__`, `__init__`, `__reduce__`, `_wrap`, `__init__`, `pids`, `_join_procs_with_timeout`, `join`, `__init__`, `start_processes`, `start_process`, `spawn`

**Key imports**: logging, multiprocessing, multiprocessing.connection, os, pickle, signal, sys, tempfile, time, warnings


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/multiprocessing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `multiprocessing`
- `multiprocessing.connection`
- `os`
- `pickle`
- `signal`
- `sys`
- `tempfile`
- `time`
- `warnings`
- `concurrent.futures`: as_completed, ThreadPoolExecutor
- `typing`: Optional
- `.`: _prctl_pr_set_pdeathsig  
- `traceback`


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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/multiprocessing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`pool.py_docs.md`](./pool.py_docs.md)
- [`_atfork.py_docs.md`](./_atfork.py_docs.md)
- [`cuda_multiprocessing.md_docs.md`](./cuda_multiprocessing.md_docs.md)
- [`reductions.py_docs.md`](./reductions.py_docs.md)
- [`queue.py_docs.md`](./queue.py_docs.md)


## Cross-References

- **File Documentation**: `spawn.py_docs.md`
- **Keyword Index**: `spawn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/multiprocessing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/multiprocessing`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/multiprocessing`):

- [`queue.py_docs.md_docs.md`](./queue.py_docs.md_docs.md)
- [`cuda_multiprocessing.md_docs.md_docs.md`](./cuda_multiprocessing.md_docs.md_docs.md)
- [`reductions.py_kw.md_docs.md`](./reductions.py_kw.md_docs.md)
- [`_atfork.py_docs.md_docs.md`](./_atfork.py_docs.md_docs.md)
- [`queue.py_kw.md_docs.md`](./queue.py_kw.md_docs.md)
- [`pool.py_kw.md_docs.md`](./pool.py_kw.md_docs.md)
- [`pool.py_docs.md_docs.md`](./pool.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_atfork.py_kw.md_docs.md`](./_atfork.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `spawn.py_docs.md_docs.md`
- **Keyword Index**: `spawn.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
