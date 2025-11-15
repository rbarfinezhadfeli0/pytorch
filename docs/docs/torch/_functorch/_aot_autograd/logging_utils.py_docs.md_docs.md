# Documentation: `docs/torch/_functorch/_aot_autograd/logging_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_functorch/_aot_autograd/logging_utils.py_docs.md`
- **Size**: 7,602 bytes (7.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_functorch/_aot_autograd/logging_utils.py`

## File Metadata

- **Path**: `torch/_functorch/_aot_autograd/logging_utils.py`
- **Size**: 4,578 bytes (4.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""
Contains utils for logging in AOTAutograd, including managing the names of the graphs under
compilation, capturing user-friendly tracebacks, and debug messages.
"""

import collections
from contextlib import contextmanager

import torch
import torch.fx.traceback as fx_traceback


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: list[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> tuple[list[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(aot_config, graph_name):
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    old_name = None
    if tracing_context := torch._guards.TracingContext.try_get():
        old_name = tracing_context.aot_graph_name
        tracing_context.aot_graph_name = graph_being_compiled
        has_tracing_context = True
    else:
        has_tracing_context = False
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []
        if has_tracing_context:
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.aot_graph_name = old_name


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: list):
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()  # type: ignore[var-annotated]
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_, seq_nr):
        def prehook(grad_output):
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(  # type: ignore[attr-defined]
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)

        return prehook

    def get_posthook(special_stack_, seq_nr):
        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()

        return posthook

    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))

        special_stack = forward_node_stack.copy()
        special_stack.append(fx_traceback.GRADIENT_ACC_SPECIAL_STACK)
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))


def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"


def format_guard_bug_msg(aot_config, expected):
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )

```



## High-Level Overview

"""Contains utils for logging in AOTAutograd, including managing the names of the graphs undercompilation, capturing user-friendly tracebacks, and debug messages.

This Python file contains 0 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `set_model_name`, `get_aot_compilation_context`, `get_aot_graph_name`, `track_graph_compiling`, `setup_stacktrace_preservation_hooks`, `iter_graph`, `get_callback`, `callback`, `get_prehook`, `prehook`, `get_posthook`, `posthook`, `describe_input`, `format_guard_bug_msg`

**Key imports**: collections, contextmanager, torch, torch.fx.traceback as fx_traceback


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `contextlib`: contextmanager
- `torch`
- `torch.fx.traceback as fx_traceback`


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_functorch/_aot_autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_cache.py_docs.md`](./autograd_cache.py_docs.md)
- [`functional_utils.py_docs.md`](./functional_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`descriptors.py_docs.md`](./descriptors.py_docs.md)
- [`collect_metadata_analysis.py_docs.md`](./collect_metadata_analysis.py_docs.md)
- [`frontend_utils.py_docs.md`](./frontend_utils.py_docs.md)
- [`subclass_parametrization.py_docs.md`](./subclass_parametrization.py_docs.md)
- [`runtime_wrappers.py_docs.md`](./runtime_wrappers.py_docs.md)


## Cross-References

- **File Documentation**: `logging_utils.py_docs.md`
- **Keyword Index**: `logging_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_functorch/_aot_autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_functorch/_aot_autograd`):

- [`graph_compile.py_kw.md_docs.md`](./graph_compile.py_kw.md_docs.md)
- [`frontend_utils.py_kw.md_docs.md`](./frontend_utils.py_kw.md_docs.md)
- [`autograd_cache.py_docs.md_docs.md`](./autograd_cache.py_docs.md_docs.md)
- [`input_output_analysis.py_kw.md_docs.md`](./input_output_analysis.py_kw.md_docs.md)
- [`schemas.py_docs.md_docs.md`](./schemas.py_docs.md_docs.md)
- [`collect_metadata_analysis.py_docs.md_docs.md`](./collect_metadata_analysis.py_docs.md_docs.md)
- [`functional_utils.py_docs.md_docs.md`](./functional_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`graph_capture.py_kw.md_docs.md`](./graph_capture.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `logging_utils.py_docs.md_docs.md`
- **Keyword Index**: `logging_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
