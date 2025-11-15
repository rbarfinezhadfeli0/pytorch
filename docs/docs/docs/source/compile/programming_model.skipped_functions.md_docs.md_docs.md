# Documentation: `docs/docs/source/compile/programming_model.skipped_functions.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/programming_model.skipped_functions.md_docs.md`
- **Size**: 8,032 bytes (7.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/programming_model.skipped_functions.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.skipped_functions.md`
- **Size**: 5,038 bytes (4.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch

import header_code
import logging
torch._logging.set_logs(dynamo=logging.DEBUG)
```

# Skipped Functions

**Summary:**
- Sometimes, `torch.compile` completely gives up compiling a function and runs it eagerly instead,
  resulting in potentially lost optimization opportunities.
- There are ways to work around skipped functions in order to re-enable tracing around the problematic code.

Sometimes, `torch.compile` with `fullgraph=False` is unable to resume tracing when encountering a graph break
or other compiler error. In many of these cases, `torch.compile` will skip compiling the function entirely and run it eagerly.

Note that the skip is only applied to the current function and NOT any nested function calls.
`torch.compile` will still attempt to compile nested calls.

<!-- TODO: fix logging for skipped functions. -->

```{code-cell}
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    torch._dynamo.skip_frame()
    x = inner2(x)
fn(torch.randn(3))
```

In the above example, `torch.compile` will trace `fn` (including `inner1`) up until the `skip_frame`.
Then `fn` is skipped and run eagerly - `inner1` and `inner2` are compiled when they are called.

Skipping functions may result in lost optimization opportunities,
so it is important to check if code you want compiled is being skipped, and if so, to work around the skip.

## Graph Break in a Loop

`torch.compile` cannot resume tracing if a graph break occurs in a loop:

```{code-cell}
@torch.compile
def fn(x):
    for i in range(5):
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    return x
fn(torch.randn(3))
```

In this example, we can avoid skipping by unrolling the loop:

```{code-cell}
@torch.compile
def fn(x):
    def inner(i):
        nonlocal x
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    inner(0)
    inner(1)
    inner(2)
    inner(3)
    inner(4)
    return x
fn(torch.randn(3))
```

In general, resolving the graph break causing the skip will also resolve the skip.

## Graph Break in a Context Manager

Another common example of an unresumable graph break is a graph break in most context managers:

```{code-cell}
class CustomCtxManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
fn(torch.randn(3))
```

We can avoid skipping by moving the graph break outside of the context manager:

```{code-cell}
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
    torch._dynamo.graph_break()
    with CustomCtxManager():
        return x + 1
fn(torch.randn(3))
```

There are some context managers where Dynamo can resume after a graph break.
Some of these can be found in `supported_ctx_manager_classes` in `torch/_dynamo/variables/torch.py`.
In general, any context manager represented by a `ContextWrappingVariable` subclass in
`torch/_dynamo/variables/ctx_manager.py` support resuming after a graph break. For example:

```{code-cell}
import contextlib
@torch.compile
def fn(x):
    with contextlib.nullcontext():
        with torch.no_grad():
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1
fn(torch.randn(3))
```

## Graph Break in a Try Block

A graph break in a try block cannot be resumed:

```{code-cell}
@torch.compile
def fn(x):
    try:
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

We can avoid skipping by moving the graph break outside of the try block:

```{code-cell}
@torch.compile
def fn(x):
    try:
        x = x + 1
    except Exception as e:
        pass
    torch._dynamo.graph_break()
    try:
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

## Hitting a Recompilation Limit
See [Changing the Cache Size Limit.](programming_model.recompilation.changing_cache_size_limit)

## Compiler Errors
Some compiler errors will result in skipped functions.
Other compiler errors will result in a hard error rather than a skipped function.

## Dealing with Skipped Functions
In general, you can resolve a skipped function by fixing the underlying graph break or error that
is causing the function to be skipped.

If the graph break/error causing the skipped function is difficult to fix,
then consider isolating the graph break/error in its own function so that minimal things are skipped.

```{code-cell}
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    def problematic_code():
        torch._dynamo.skip_frame()
    problematic_code()
    x = inner2(x)
fn(torch.randn(3))
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source/compile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source/compile`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/source/compile`):

- [`dynamic_shapes_troubleshooting_guardon_errors.md_docs.md`](./dynamic_shapes_troubleshooting_guardon_errors.md_docs.md)
- [`dynamic_shapes_core_concepts.md_docs.md`](./dynamic_shapes_core_concepts.md_docs.md)
- [`programming_model.error_on_graph_break.md_docs.md`](./programming_model.error_on_graph_break.md_docs.md)
- [`dynamic_shapes_troubleshooting.md_docs.md`](./dynamic_shapes_troubleshooting.md_docs.md)
- [`programming_model.recompilation.md_docs.md`](./programming_model.recompilation.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md)
- [`dynamic_shapes_beyond_the_basics.md_docs.md`](./dynamic_shapes_beyond_the_basics.md_docs.md)
- [`programming_model.graph_breaks_index.md_docs.md`](./programming_model.graph_breaks_index.md_docs.md)
- [`programming_model.md_docs.md`](./programming_model.md_docs.md)
- [`dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md`](./dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.skipped_functions.md_docs.md`
- **Keyword Index**: `programming_model.skipped_functions.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source/compile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source/compile`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/docs/source/compile`):

- [`dynamic_shapes_advanced_control_options.md_docs.md_docs.md`](./dynamic_shapes_advanced_control_options.md_docs.md_docs.md)
- [`programming_model.recompilation.md_docs.md_docs.md`](./programming_model.recompilation.md_docs.md_docs.md)
- [`programming_model.dynamo_core_concepts.md_docs.md_docs.md`](./programming_model.dynamo_core_concepts.md_docs.md_docs.md)
- [`programming_model.nested_graph_breaks.md_kw.md_docs.md`](./programming_model.nested_graph_breaks.md_kw.md_docs.md)
- [`programming_model.where_to_apply_compile.md_docs.md_docs.md`](./programming_model.where_to_apply_compile.md_docs.md_docs.md)
- [`programming_model.graph_breaks_index.md_docs.md_docs.md`](./programming_model.graph_breaks_index.md_docs.md_docs.md)
- [`programming_model.nested_graph_breaks.md_docs.md_docs.md`](./programming_model.nested_graph_breaks.md_docs.md_docs.md)
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.skipped_functions.md_docs.md_docs.md`
- **Keyword Index**: `programming_model.skipped_functions.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
