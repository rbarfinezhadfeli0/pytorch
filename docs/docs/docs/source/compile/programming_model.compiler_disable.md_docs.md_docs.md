# Documentation: `docs/docs/source/compile/programming_model.compiler_disable.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/programming_model.compiler_disable.md_docs.md`
- **Size**: 5,230 bytes (5.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/programming_model.compiler_disable.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.compiler_disable.md`
- **Size**: 2,347 bytes (2.29 KB)
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

torch._logging.set_logs(graph_breaks=True, graph_code=True)
```

# Disabling and Suppressing Errors
For some model architectures, there are portions of the model which are particularly difficult to compile -
either there are many graph breaks, or there are crashes.
You may want to explicitly disable these portions of the model which are problematic so that you can apply
`torch.compile` to the parts that work. You can do this by using the `@torch.compiler.disable` decorator.
When `torch.compile` attempts to call a disabled function, it breaks the graph and skips tracing the disabled function,
resuming tracing after the call. By default, all recursive calls made from a disabled function are also disabled.
Use the `recursive=False` option to allow compilation for recursive calls.

```{code-cell}
def inner1(x):
    torch._dynamo.graph_break()  # not traced
    return x + 1  # not traced

@torch.compiler.disable
def outer1(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner1(x)

@torch.compile
def f(x):
    x = outer1(x)
    return x + 4  # traced

print(f(torch.ones(3)))
```

```{code-cell}
def inner2(x):
    torch._dynamo.graph_break()  # traced
    return x + 1  # traced

@torch.compiler.disable(recursive=False)
def outer2(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner2(x)

@torch.compile
def g(x):
    x = outer2(x)
    return x + 4  # traced

print(g(torch.ones(3)))
```

For example, one can use `torch.compiler.disable` to disable `torch.compile` on sparse architecture in
recommendation models, as the sparse arch is difficult to compile.
Preprocessing and logging functions are other examples of functions that typically cause
a lot of graph breaks and do not get value from being compiled.

If you are experiencing compiler crashes and you want to continue regardless,
you can set `torch._dynamo.config.suppress_errors = True`.
When the compiler crashes, we will just skip tracing the function and try again later.
**This is not best practice** - it is better to eventually manually add `disable` annotations as necessary.

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

- **File Documentation**: `programming_model.compiler_disable.md_docs.md`
- **Keyword Index**: `programming_model.compiler_disable.md_kw.md`
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
- [`programming_model.nested_graph_breaks.md_docs.md_docs.md`](./programming_model.nested_graph_breaks.md_docs.md_docs.md)
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.compiler_disable.md_docs.md_docs.md`
- **Keyword Index**: `programming_model.compiler_disable.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
