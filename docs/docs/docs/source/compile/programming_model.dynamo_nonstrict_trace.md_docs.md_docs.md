# Documentation: `docs/docs/source/compile/programming_model.dynamo_nonstrict_trace.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/programming_model.dynamo_nonstrict_trace.md_docs.md`
- **Size**: 6,776 bytes (6.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/programming_model.dynamo_nonstrict_trace.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.dynamo_nonstrict_trace.md`
- **Size**: 3,866 bytes (3.78 KB)
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
```

# Use `torch._dynamo.nonstrict_trace`

**Summary:**
- Use `nonstrict_trace` to trace a function with non-strict tracing inside of a `torch.compile`'d region.
  You may wish to do this because the Dynamo graph breaks on something inside of the function
  and you are sure that the function is non-strict traceable.

Consider the following scenario:

```{code-cell}
def get_magic_num():
    # This explicit graph break call is meant to emulate any kind of Dynamo
    # graph break, e.g., the function is implemented in C, or uses some python
    # language feature Dynamo doesn't yet support.
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
try:
    func(torch.rand(10))
except Exception as e:
    print(e)
```

If we run the code above, we'll get an error from Dynamo, because it sees a graph break while the user specified `fullgraph=True`.

In these situations, if a user still wants to keep `fullgraph=True`, they typically have several options:

1. The graph break is due to a language feature Dynamo doesn't yet support.
   In this case, the user either rewrites their code, or files an issue on GitHub.
2. The graph break is due to a call to a function implemented in C.
   In this case, the user can try to use a custom op.
   The user could also try providing a polyfill (a reference implementation in Python)
   so that Dynamo can trace through it.
3. Worst case scenario -- an internal compiler error. In this case, the user likely has to file an issue on GitHub.

In addition to all these options, PyTorch does provide an alternative `torch._dynamo.nonstrict_trace`, if the function call that induced the graph break satisfies certain requirements:

- The requirements of [general non-strict tracing](programming_model.non_strict_tracing_model).
- The inputs and outputs must contain either basic types (e.g., `int`, `float`, `list`, `dict`, `torch.Tensor`),
  or user-defined types that are registered to `torch.utils._pytree`.
- The function must be defined outside the `torch.compile`'d region.
- Any non-input values read by the function will be treated as a constant
  (e.g., a global tensor), and will not be guarded on.

When tracing through a call to a `torch._dynamo.nonstrict_trace`'d function, `torch.compile` switches to [non-strict tracing](programming_model.non_strict_tracing_model),
and the FX graph will eventually contain all the relevant tensor operations which happened inside that function.

For the example above, we can use `torch._dynamo.nonstrict_trace to eliminate` the graph break:

```{code-cell}
@torch._dynamo.nonstrict_trace
def get_magic_num():
    # This explicit graph break call is meant to emulate any kind of Dynamo
    # graph break, e.g., the function is implemented in C, or uses some python
    # language feature Dynamo doesn't yet support.
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
print(func(torch.rand(10)))
# No graph break and no error.
```

Note that one can use it inside a `torch.compile`'d region as well:

```{code-cell}
def get_magic_num():
    # This explicit graph break call is meant to emulate any kind of Dynamo
    # graph break, e.g., the function is implemented in C, or uses some python
    # language feature Dynamo doesn't yet support.
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = torch._dynamo.nonstrict_trace(get_magic_num)()
    return x + n
print(func(torch.rand(10)))
# No graph break and no error.
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

- **Error Handling**: Includes exception handling


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

- **File Documentation**: `programming_model.dynamo_nonstrict_trace.md_docs.md`
- **Keyword Index**: `programming_model.dynamo_nonstrict_trace.md_kw.md`
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

- **Error Handling**: Includes exception handling


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
- [`programming_model.nested_graph_breaks.md_docs.md_docs.md`](./programming_model.nested_graph_breaks.md_docs.md_docs.md)
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.dynamo_nonstrict_trace.md_docs.md_docs.md`
- **Keyword Index**: `programming_model.dynamo_nonstrict_trace.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
