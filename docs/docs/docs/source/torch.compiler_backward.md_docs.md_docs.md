# Documentation: `docs/docs/source/torch.compiler_backward.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/torch.compiler_backward.md_docs.md`
- **Size**: 5,523 bytes (5.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/torch.compiler_backward.md`

## File Metadata

- **Path**: `docs/source/torch.compiler_backward.md`
- **Size**: 2,931 bytes (2.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
``torch.compile`` has different autograd semantics
==================================================

When you apply ``torch.compile`` to a function in your model's forward pass,
it will automatically generate a backward pass for the compiled function.
During compilation, it will trace out a graph for the backward pass that
is used whenever autograd is invoked. We refer to the component inside
``torch.compile`` that is responsible for this as ``AOTDispatcher``
(sometimes known as ``AOTAutograd``).

As so, ``torch.compile`` bakes in details of the computation into the
traced-out backward graph during compilation of the function
in the forward pass.
However, in eager-mode PyTorch, the backward computation is dynamic:
outside of the forward pass, you can wrap the call to
``tensor.backward()`` or ``torch.autograd.grad(...)``
in a context manager that may change its behavior.

This page documents how ``torch.compile``'s autograd semantics differ from
eager-mode PyTorch and how to work around it.

``Autocast`` behavior
---------------------

``torch.compile`` bakes in an assumption on if the backward pass will be
run under an ambient autocast context manager. By default,
Use ``torch._functorch.config.backward_pass_autocast``
to control that assumption; an incorrect assumption may lead to silent
incorrectness.

The options are either:
- `"same_as_forward"` (default).
  We assume that the backward of the ``torch.compile``'ed region
  will be run under the same autocast context manager that the region was run
  under (if any). Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
      # backward pass run under the same autocast context as the compiled region
      z.backward()
  ```
- `"off"`. We assume that the backward of the torch.compile'd region will
  not be run under any autocast context managers.
  Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
  # Backward pass runs under no autocast.
  z.backward()
  ```
- There is a third option. If you set ``torch._functorch.config.backward_pass_autocast``
  to a list of kwargs, we will assume the backward pass runs under an autocast context
  constructed by those kwargs.

  For example, if your code looks like the following:
  ```py
  y = torch.compile(region)(x)
  ...
  # Backward pass runs under special context manager
  with torch.amp.autocast(**kwargs):
      z.backward()
  ```
  then set ``torch._functorch.config.backward_pass_autocast = kwargs``.

Use ``patch`` to apply the option to a specific ``torch.compile`` call:
```py
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # backward pass run under the same autocast context as the compiled region
    z.backward()
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `torch.compiler_backward.md_docs.md`
- **Keyword Index**: `torch.compiler_backward.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`docs/docs/source`):

- [`distributions.md_docs.md_docs.md`](./distributions.md_docs.md_docs.md)
- [`distributed.optim.md_docs.md_docs.md`](./distributed.optim.md_docs.md_docs.md)
- [`torch.compiler_dynamic_shapes.md_kw.md_docs.md`](./torch.compiler_dynamic_shapes.md_kw.md_docs.md)
- [`tensor_attributes.rst_docs.md_docs.md`](./tensor_attributes.rst_docs.md_docs.md)
- [`tensor_attributes.rst_kw.md_docs.md`](./tensor_attributes.rst_kw.md_docs.md)
- [`torch.compiler_dynamo_overview.md_docs.md_docs.md`](./torch.compiler_dynamo_overview.md_docs.md_docs.md)
- [`mtia.memory.md_kw.md_docs.md`](./mtia.memory.md_kw.md_docs.md)
- [`nn.attention.varlen.md_kw.md_docs.md`](./nn.attention.varlen.md_kw.md_docs.md)
- [`cpu.rst_kw.md_docs.md`](./cpu.rst_kw.md_docs.md)
- [`torch.compiler_faq.md_docs.md_docs.md`](./torch.compiler_faq.md_docs.md_docs.md)


## Cross-References

- **File Documentation**: `torch.compiler_backward.md_docs.md_docs.md`
- **Keyword Index**: `torch.compiler_backward.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
