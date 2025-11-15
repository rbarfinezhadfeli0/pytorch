# Documentation: `docs/source/compile/programming_model.common_graph_breaks.md`

## File Metadata

- **Path**: `docs/source/compile/programming_model.common_graph_breaks.md`
- **Size**: 3,368 bytes (3.29 KB)
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

torch._logging.set_logs(graph_breaks=True)
```

# Common Graph Breaks

Below are some common graph breaks and some workarounds.

## Incorrect Code
Your code might contain errors (meaning it doesn't execute even without `torch.compile`). In the example below, there's a typo in the `torch.sin` call due to an extra argument. **Always disable `torch.compile` to check if the code runs correctly.**


```{code-cell}
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    fn(torch.ones(3, 3))
except Exception as e:
    pass
```

Dynamo makes a best-effort attempt to hint if a graph break is caused by your code.
But it can still sometimes be difficult to tell from the logs if the graph break is caused by an error in your code,
is a more complicated graph break, or is a `torch.compile` bug. In order to differentiate, we recommend trying to run your code without `torch.compile` to see if you still get the error reported by the graph break.

## Data-dependent operations

`torch.compile` graph breaks on data-dependent operations such as data-dependent control flow (if-statements, loops with tensors) and direct tensor data accesses (`.item`, `.data_ptr`).

```{code-cell}
@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

print(fn(torch.ones(3, 3)))
```

The general workaround for these graph breaks is to avoid doing data-dependent operations. Some specific workarounds are:

- If your control flow doesn't actually depend on data values, consider modifying your code to perform control flow on constants.


```{code-cell}
# old
x = torch.randn(3, 3)
@torch.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

```{code-cell}
# new
x = torch.randn(3, 3)
cond = (x.sum() > 0).item()
@torch.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

- Use higher-order ops like {ref}`cond` in place of data-dependent control flow


```{code-cell}
# old
@torch.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

print(fn(torch.ones(3, 3)))
```

```{code-cell}
# new
@torch.compile
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )

print(fn(torch.ones(3, 3)))
```

- If you have a `.item()` call, try `torch._dynamo.config.capture_scalar_outputs = True`
or `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`.
- Wrap problematic parts of the function in a custom operator

## Printing and logging

Printing/logging/issuing warnings will result in a graph break.
You can try working around this by using `torch._dynamo.config.reorderable_logging_functions`.
This config is used to reorder logging functions so that they are called at the end of the
traced function, thus avoiding a graph break.
However, the logged contents may differ if, for example, a mutation occurs.


```{code-cell}
torch._dynamo.config.reorderable_logging_functions.add(print)

@torch.compile
def fn(x):
    x += 1
    print("log!")
    return torch.sin(x)

print(fn(torch.ones(3, 3)))
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

- **File Documentation**: `programming_model.common_graph_breaks.md_docs.md`
- **Keyword Index**: `programming_model.common_graph_breaks.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
