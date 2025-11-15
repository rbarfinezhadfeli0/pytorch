# Documentation: `docs/docs/source/compile/programming_model.dynamo_core_concepts.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/programming_model.dynamo_core_concepts.md_docs.md`
- **Size**: 8,871 bytes (8.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/programming_model.dynamo_core_concepts.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.dynamo_core_concepts.md`
- **Size**: 5,972 bytes (5.83 KB)
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

# Dynamo Core Concepts

**Summary:**

- Dynamo, `torch.compile`'s frontend, performs **tracing** to capture the semantics of a Python function
  (and its nested function calls) into a linear sequence of operations (the "(FX) graph"),
  residual bytecode, and "guards" (a list of conditions under which the graph and bytecode are valid).
- Unsupported Python features lead to **graph breaks**, where Dynamo compiles a partial graph acquired from tracing,
  then runs the unsupported code, then resumes tracing.
- Graph breaks may lead to slowness in torch.compile and prevent backend optimization opportunities.
  If you're not seeing the performance you expect, then check for graph breaks.

## Dynamo Tracing
`torch.compile`'s frontend (Dynamo) is a custom Python bytecode interpreter designed to allow graph compilation
in PyTorch programs while retaining the full flexibility of Python. Given a function to be compiled, Dynamo
interprets Python bytecode to extract sequences of PyTorch operations into 1 or more FX graphs that may be further optimized by a backend.

![Summary diagram of Dynamo](_static/dynamo_summary_diagram.png)

For example, for the function `f` in the above diagram, Dynamo produces:
- a single **FX graph** that takes in the original input plus some additional inputs required by the function.
- **Python bytecode** that can be used as a drop-in replacement for `f`. In our example, the bytecode retrieves
  the additional inputs and passes it to the graph and also contains unoptimizable Python side effects (the list append)
- **guards** that specify the conditions under which the graph and bytecode are valid. Unless otherwise specified,
  the graph produced by Dynamo specializes on the shapes of input Tensors.

(programming_model.dynamo_core_concepts.graph_breaks)=

## Graph Breaks
Dynamo traces your code and attempts to capture your PyTorch code into a single computation graph of PyTorch
operators (FX graph). However, this is not always possible. When encountering code that can't be traced, a "**graph break**" occurs.
In the default `torch.compile` settings, a graph break involves compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph.

Graph breaks are a feature that allows Dynamo to run over arbitrary Python code and carve out functional subgraphs that can each be individually optimized.

However, it is possible for graph breaks to lead to unexpected slowness in `torch.compile`.
If you're not getting the speedups you expect, we recommend checking for graph breaks and removing them.

Graph breaks may occur on things like:

- Data-dependent if-statements
- Many Python built-in functions
- C functions

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(graph_breaks=True)
```

Below is an example of a graph break due to calling an unsupported operation `torch.save`:

```{code-cell}
@torch.compile
def f(x):
   y = x ** 2  / 2
   torch.save(y, "foo.pt")  # torch.save is an unsupported operation
   z = y ** 3 / 6
   return z

x = torch.randn(3)
print(f(x))
```

```{code-cell}
:tags: [remove-cell]
import os
os.remove("foo.pt")
```

The semantics of `torch.compile(f)(x)` are roughly this:

```python
def compiled_f_semantics(x):
   y = torch.compile(g, fullgraph=True)(x)
   torch.save(y, "foo.pt")
   z = torch.compile(h, fullgraph=True)(x)
   return z

def g(x):
    return x ** 2  / 2

def h(x):
    return y ** 3 / 6
```

## Guards

`torch.compile` makes some assumptions about runtime values as we trace through code. During tracing, we generate "guards",
which are runtime checks for these assumptions. Guards are run in future calls to the compiled function to determine if we
can reuse previously compiled code. Examples of runtime checks are constant values, types, and object IDs.

Below is an example of generated guards. The `TENSOR_MATCH` guard checks for the input's type, device, dtype, shape, etc.

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(guards=True)
```

```{code-cell}
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
```

## Recompilations
If the guards fail for every instance of previously compiled code, then `torch.compile` must "recompile" the function,
requiring the original code to be traced again. In the example below, recompilation is necessary because the guard checking the tensor argument's shape failed.

```{code-cell}
:tags: [remove-cell]
torch._logging.set_logs(recompiles=True)
```

```{code-cell}
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))
```

## Dynamic Shapes

`torch.compile` initially assumes tensor shapes are static/constant and guards based on these assumptions. By using "dynamic shapes,"
we can get `torch.compile` to produce compiled code that can accept tensor inputs with different shapes - we avoid recompiling every time shapes differ.
By default, automatic dynamic shapes are enabled in `torch.compile(dynamic=None)` - if compilation fails due to shape mismatch,
recompilation is attempted with dynamic shapes. Dynamic shapes can also be fully enabled (`dynamic=True`) or disabled (`dynamic=False`).

Below, we enable dynamic shapes and note that we no longer need to recompile.

```{code-cell}
:tags: [remove-cell]
import logging
torch._logging.set_logs(dynamic=logging.DEBUG, recompiles=True)
```

```{code-cell}
@torch.compile(dynamic=True)
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))
```

For more information on dynamic shapes, see [The dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng).

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

- **File Documentation**: `programming_model.dynamo_core_concepts.md_docs.md`
- **Keyword Index**: `programming_model.dynamo_core_concepts.md_kw.md`
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
- [`programming_model.nested_graph_breaks.md_kw.md_docs.md`](./programming_model.nested_graph_breaks.md_kw.md_docs.md)
- [`programming_model.where_to_apply_compile.md_docs.md_docs.md`](./programming_model.where_to_apply_compile.md_docs.md_docs.md)
- [`programming_model.graph_breaks_index.md_docs.md_docs.md`](./programming_model.graph_breaks_index.md_docs.md_docs.md)
- [`programming_model.nested_graph_breaks.md_docs.md_docs.md`](./programming_model.nested_graph_breaks.md_docs.md_docs.md)
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `programming_model.dynamo_core_concepts.md_docs.md_docs.md`
- **Keyword Index**: `programming_model.dynamo_core_concepts.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
