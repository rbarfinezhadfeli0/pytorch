# Documentation: `torch/_lazy/closure.py`

## File Metadata

- **Path**: `torch/_lazy/closure.py`
- **Size**: 5,677 bytes (5.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import os
import threading
from queue import Empty as EmptyQueue, Queue

from torch._lazy.device_context import get_device_context


class ClosureHandler:
    def __init__(self) -> None:
        pass

    def run(self, closure):
        """Run closure function

        Args:
        closure: callable function to run
        """
        closure()

    def __call__(self, closures):
        for closure in closures:
            self.run(closure)


class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size=100):
        super().__init__()
        self._closure_queue: Queue = Queue(
            int(os.environ.get("LTC_MAX_ASYNC_QUEUE", max_queue_size))
        )
        self._closure_exception: Queue = Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop = None

    def start_event_loop(self):
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:

            def event_loop():
                # Run loop until closure event is set and closure queue is empty
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return

            self._closure_event_loop = threading.Thread(
                target=event_loop
            )  # pyrefly: ignore [bad-assignment]
            self._closure_event_loop.start()  # pyrefly: ignore [missing-attribute]

    def run(self, closure):
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if (
                self._closure_event_loop is None
                or not self._closure_event_loop.is_alive()
            ):
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError(
                        "Cannot run asynchronous closure due to previously raised exception"
                    ) from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()


def add_step_closure(closure, args=(), run_async=False):
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    devctx = get_device_context()
    closures_type = "async_step_closures" if run_async else "step_closures"
    step_closures = getattr(devctx, closures_type, None)
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))


def run_step_closures():
    devctx = get_device_context()
    async_step_closures = getattr(devctx, "async_step_closures", None)
    if async_step_closures is not None:
        devctx.async_step_closures = []  # type: ignore[attr-defined]
        async_closure_handler = getattr(devctx, "async_closure_handler", None)
        if async_closure_handler is None:
            async_closure_handler = AsyncClosureHandler()
            devctx.async_closure_handler = async_closure_handler  # type: ignore[attr-defined]
        async_closure_handler(async_step_closures)

    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is not None:
        devctx.step_closures = []  # type: ignore[attr-defined]
        closure_handler = getattr(devctx, "closure_handler", None)
        if closure_handler is None:
            closure_handler = ClosureHandler()
            devctx.closure_handler = closure_handler  # type: ignore[attr-defined]
        closure_handler(step_closures)
    return devctx

```



## High-Level Overview

"""Run closure function        Args:        closure: callable function to run

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ClosureHandler`, `AsyncClosureHandler`

**Functions defined**: `__init__`, `run`, `__call__`, `__init__`, `start_event_loop`, `event_loop`, `run`, `add_step_closure`, `run_step_closures`

**Key imports**: os, threading, Empty as EmptyQueue, Queue, get_device_context


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_lazy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `threading`
- `queue`: Empty as EmptyQueue, Queue
- `torch._lazy.device_context`: get_device_context


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_lazy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`computation.py_docs.md`](./computation.py_docs.md)
- [`metrics.py_docs.md`](./metrics.py_docs.md)
- [`tensor_factory_functions.py_docs.md`](./tensor_factory_functions.py_docs.md)
- [`ir_cache.py_docs.md`](./ir_cache.py_docs.md)
- [`ts_backend.py_docs.md`](./ts_backend.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`extract_compiled_graph.py_docs.md`](./extract_compiled_graph.py_docs.md)


## Cross-References

- **File Documentation**: `closure.py_docs.md`
- **Keyword Index**: `closure.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
