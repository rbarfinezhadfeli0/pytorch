# Documentation: `docs/docs/source/compile/programming_model.nested_graph_breaks.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/programming_model.nested_graph_breaks.md_docs.md`
- **Size**: 8,364 bytes (8.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/programming_model.nested_graph_breaks.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.nested_graph_breaks.md`
- **Size**: 5,469 bytes (5.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Nested Graph Breaks

Summary:
- Graph breaks in nested functions can result in hard-to-understand compiler behavior, which we document below
- A nested graph break results in {math}`\mathcal O(N)` duplicate graph break behavior

Recall that when `torch.compile` is applied to a function, any nested function calls are also traced.
A **nested graph break** refers to any graph break that happens in a nested function call.

```python
def inner(x):
    ...
    torch._dynamo.graph_break()  # nested graph break
    ...

@torch.compile
def outer(x):
    ...
    y = inner(x)
    ...
```

The resumption semantics around nested graph breaks can be confusing, so we describe the behavior here.

Recall that in `fullgraph=False`, [graph breaks are handled](programming_model.dynamo_core_concepts.graph_breaks) by compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph.
Resuming a function is actually a fairly complicated technical feat, so resuming tracing is only supported on top-level functions.

We can therefore resume tracing after a nested graph break with this restriction in the following way:

First, consider the below example where `torch.compile` traces from `f` and traces all the way until the
graph break in `inner1` is encountered.

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

def inner2(x):
    x = x + 4
    x = inner1(x)
    x = x + 8

@torch.compile
def f(x):
    # start tracing from here
    x = x + 16
    x = inner2(x)
    x = x + 32

f(torch.randn(3))
```

Since we can only resume from top-level functions, we graph break on the `inner2` call in `f`.
```python
# The semantics of torch.compile(f)(x) is roughly this:
def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

`inner2` is then automatically compiled as a top-level function.
We trace all the way until the graph break in `inner1` is encountered again.

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

# this torch.compile is automatically applied
@torch.compile
def inner2(x):
    # start tracing from here
    x = x + 4
    x = inner1(x)
    x = x + 8

def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

Then we graph break on the `inner1` call in `inner2`.
```python
def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8
```

`inner1` is then automatically compiled as a top-level function.
The graph break is from `inner1`, so we handle the graph break normally.
```python
# this torch.compile is automatically applied
@torch.compile
def inner1(x):
    # start tracing from here
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

compiled_f_semantics(torch.randn(3))
```

`inner1` is handled normally:

```python
def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2
```

So the initial code is semantically equivalent to
```python
def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = compiled_inner1_semantics(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2

compiled_f_semantics(torch.randn(3))
```

Note in particular that we traced 3 top-level functions, and that we traced the same graph break 3 times.
**This explains why you may encounter duplicate graph breaks when using `torch.compile`.**

In summary, nested graph breaks are handled by:
- Tracing from the top-level function all the way to the nested graph break
- Graph breaking on the top-level function at the call to the second-level function
- Compiling the PyTorch ops tracked so far and running the compiled graph
- Calling the second-level function, which gets automatically compiled as a top-level function
- Resuming tracing after the second-level function call

Note that the runtime of handling this graph break is {math}`\mathcal O(NK)`, where {math}`N` is the nesting depth,
and {math}`K` is the number of instructions from the top-level function to the graph break.
We end up tracing {math}`\mathcal O(N^2)` frames, and we trace the same graph break {math}`\mathcal O(N)` times.

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

*No specific patterns automatically detected.*


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

- **File Documentation**: `programming_model.nested_graph_breaks.md_docs.md`
- **Keyword Index**: `programming_model.nested_graph_breaks.md_kw.md`
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.nested_graph_breaks.md_docs.md_docs.md`
- **Keyword Index**: `programming_model.nested_graph_breaks.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
