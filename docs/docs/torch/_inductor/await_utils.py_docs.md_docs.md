# Documentation: `docs/torch/_inductor/await_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/await_utils.py_docs.md`
- **Size**: 8,868 bytes (8.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/await_utils.py`

## File Metadata

- **Path**: `torch/_inductor/await_utils.py`
- **Size**: 5,841 bytes (5.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import asyncio
import sys
import weakref
from asyncio import AbstractEventLoop, Future
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterator
from contextlib import contextmanager, ExitStack
from contextvars import Context
from typing import Any, Optional, Protocol, TypeVar

from torch.utils._ordered_set import OrderedSet


T = TypeVar("T")
TCoro = Generator[Any, None, T]

if sys.version_info >= (3, 11):

    class TaskFactory(Protocol):
        def __call__(
            self,
            __loop: AbstractEventLoop,
            __factory: Coroutine[None, None, object] | Generator[None, None, object],
            __context: Context | None = None,
            /,
        ) -> asyncio.futures.Future[object]: ...

    TaskFactoryType = TaskFactory
else:
    TaskFactoryType = Callable[[AbstractEventLoop, Generator[TCoro, None, T]], Future]  # type: ignore[valid-type]


def await_sync(awaitable: Awaitable[T]) -> T:
    with get_loop() as loop:
        return loop.run_until_complete(awaitable)


@contextmanager
def get_loop(
    always_create_new_loop: bool = False,
) -> Iterator[AbstractEventLoop]:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as re:
        if "There is no current event loop in thread" in str(re):
            with _new_loop() as loop:
                yield loop
            return
        else:
            raise

    @contextmanager
    def _restore_loop(
        loop: asyncio.AbstractEventLoop,
    ) -> Iterator[None]:
        try:
            yield
        finally:
            asyncio.set_event_loop(loop)

    @contextmanager
    def _restore_running_loop() -> Iterator[None]:
        loop_from_events = asyncio.events._get_running_loop()
        asyncio.events._set_running_loop(None)
        try:
            yield
        finally:
            asyncio.events._set_running_loop(loop_from_events)

    with ExitStack() as stack:
        if loop.is_running():
            stack.enter_context(_restore_running_loop())
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop(loop.get_task_factory()))  # type: ignore[arg-type]
        elif loop.is_closed():
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        elif always_create_new_loop:
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        yield loop


@contextmanager
def _new_loop(
    task_factory: Optional[TaskFactoryType] = None,
) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    tasks = _patch_loop(loop)

    if task_factory:
        # pyre-ignore[6]
        loop.set_task_factory(task_factory)  # type: ignore[arg-type]

    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            _cancel_all_tasks(loop, tasks)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_tasks(
    loop: AbstractEventLoop,
    tasks: OrderedSet[Future],  # type: ignore[type-arg]
) -> None:
    to_cancel = [task for task in tasks if not task.done()]

    if not to_cancel:
        return

    # pyre-fixme[1001]: Awaitable assigned to `task` is never awaited.
    for task in to_cancel:
        task.cancel()

    # pyrefly: ignore [bad-argument-type]
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _patch_loop(loop: AbstractEventLoop) -> OrderedSet[Future]:  # type: ignore[type-arg]
    tasks: weakref.WeakSet[Future] = weakref.WeakSet()  # type: ignore[type-arg]

    task_factories: list[Optional[TaskFactoryType]] = [None]

    def _set_task_factory(factory: Optional[TaskFactoryType]) -> None:
        task_factories[0] = factory

    def _get_task_factory() -> Optional[TaskFactoryType]:
        return task_factories[0]

    def _safe_task_factory(
        loop: AbstractEventLoop,
        coro: TCoro,  # type: ignore[type-arg]
        *,
        context: Context | None = None,
    ) -> asyncio.Future:  # type: ignore[valid-type, type-arg]
        task_factory = task_factories[0]
        if task_factory is None:
            if sys.version_info >= (3, 11):
                # pyrefly: ignore [bad-argument-type]
                task = asyncio.Task(coro, loop=loop, context=context)
            else:
                task = asyncio.Task(coro, loop=loop)
            # pyre-ignore[16]: `Task` has no attribute `_source_traceback`.
            if task._source_traceback:  # type: ignore[attr-defined]
                del task._source_traceback[  # type: ignore[attr-defined]
                    -1
                ]  # pragma: no cover  # type: ignore[attr-defined]
        else:
            if sys.version_info >= (3, 11):
                task = task_factory(loop, coro, context=context)  # type: ignore[arg-type, call-arg, assignment]
            else:
                task = task_factory(loop, coro)  # type: ignore[arg-type]
        #  `Union[Task[Any], Future[Any]]`.
        tasks.add(task)
        return task

    # pyre-ignore[6]
    loop.set_task_factory(_safe_task_factory)  # type: ignore[method-assign, arg-type]
    # pyre-ignore[8]
    loop.set_task_factory = _set_task_factory  # type: ignore[method-assign, assignment]
    # pyre-ignore[8]
    loop.get_task_factory = _get_task_factory  # type: ignore[method-assign, assignment]

    return tasks  # type: ignore[return-value]

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TaskFactory`

**Functions defined**: `__call__`, `await_sync`, `get_loop`, `_restore_loop`, `_restore_running_loop`, `_new_loop`, `_cancel_all_tasks`, `_patch_loop`, `_set_task_factory`, `_get_task_factory`, `_safe_task_factory`

**Key imports**: asyncio, sys, weakref, AbstractEventLoop, Future, Awaitable, Callable, Coroutine, Generator, Iterator, contextmanager, ExitStack, Context, Any, Optional, Protocol, TypeVar, OrderedSet


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `asyncio`
- `sys`
- `weakref`
- `collections.abc`: Awaitable, Callable, Coroutine, Generator, Iterator
- `contextlib`: contextmanager, ExitStack
- `contextvars`: Context
- `typing`: Any, Optional, Protocol, TypeVar
- `torch.utils._ordered_set`: OrderedSet


## Code Patterns & Idioms

### Common Patterns

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

- **File Documentation**: `await_utils.py_docs.md`
- **Keyword Index**: `await_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `await_utils.py_docs.md_docs.md`
- **Keyword Index**: `await_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
