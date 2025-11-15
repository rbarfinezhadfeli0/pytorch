# Documentation: `docs/docs/source/library.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/library.md_docs.md`
- **Size**: 5,260 bytes (5.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/library.md`

## File Metadata

- **Path**: `docs/source/library.md`
- **Size**: 2,796 bytes (2.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(torch-library-docs)=

# torch.library

```{eval-rst}
.. py:module:: torch.library
.. currentmodule:: torch.library
```

torch.library is a collection of APIs for extending PyTorch's core library
of operators. It contains utilities for testing custom operators, creating new
custom operators, and extending operators defined with PyTorch's C++ operator
registration APIs (e.g. aten operators).

For a detailed guide on effectively using these APIs, please see
[PyTorch Custom Operators Landing Page](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
for more details on how to effectively use these APIs.

## Testing custom ops

Use {func}`torch.library.opcheck` to test custom ops for incorrect usage of the
Python torch.library and/or C++ TORCH_LIBRARY APIs. Also, if your operator supports
training, use {func}`torch.autograd.gradcheck` to test that the gradients are
mathematically correct.

```{eval-rst}
.. autofunction:: opcheck
```

## Creating new custom ops in Python

Use {func}`torch.library.custom_op` to create new custom ops.

```{eval-rst}
.. autofunction:: custom_op
.. autofunction:: triton_op
.. autofunction:: wrap_triton
```

## Extending custom ops (created from Python or C++)

Use the `register.*` methods, such as {func}`torch.library.register_kernel` and
{func}`torch.library.register_fake`, to add implementations
for any operators (they may have been created using {func}`torch.library.custom_op` or
via PyTorch's C++ operator registration APIs).

```{eval-rst}
.. autofunction:: register_kernel
.. autofunction:: register_autocast
.. autofunction:: register_autograd
.. autofunction:: register_fake
.. autofunction:: register_vmap
.. autofunction:: impl_abstract
.. autofunction:: get_ctx
.. autofunction:: register_torch_dispatch
.. autofunction:: infer_schema
.. autoclass:: torch._library.custom_ops.CustomOpDef
   :members: set_kernel_enabled
.. autofunction:: get_kernel
```

## Low-level APIs

The following APIs are direct bindings to PyTorch's C++ low-level
operator registration APIs.

```{eval-rst}
.. warning:: The low-level operator registration APIs and the PyTorch Dispatcher are a complicated PyTorch concept. We recommend you use the higher level APIs above (that do not require a torch.library.Library object) when possible. `This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ is a good starting point to learn about the PyTorch Dispatcher.
```

A tutorial that walks you through some examples on how to use this API is available on [Google Colab](https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing).

```{eval-rst}
.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define

.. autofunction:: impl
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

- **File Documentation**: `library.md_docs.md`
- **Keyword Index**: `library.md_kw.md`
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

- **File Documentation**: `library.md_docs.md_docs.md`
- **Keyword Index**: `library.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
