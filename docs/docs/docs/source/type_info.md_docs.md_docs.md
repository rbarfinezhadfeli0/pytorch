# Documentation: `docs/docs/source/type_info.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/type_info.md_docs.md`
- **Size**: 5,325 bytes (5.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/type_info.md`

## File Metadata

- **Path**: `docs/source/type_info.md`
- **Size**: 2,878 bytes (2.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
```{eval-rst}
.. currentmodule:: torch
```

(type-info-doc)=
# Type Info

The numerical properties of a {class}`torch.dtype` can be accessed through either the {class}`torch.finfo` or the {class}`torch.iinfo`.

(finfo-doc)=
## torch.finfo

```{eval-rst}
.. class:: torch.finfo
```

A {class}`torch.finfo` is an object that represents the numerical properties of a floating point
{class}`torch.dtype`, (i.e. ``torch.float32``, ``torch.float64``, ``torch.float16``, and ``torch.bfloat16``).
This is similar to [numpy.finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).

A {class}`torch.finfo` provides the following attributes:

| Name            | Type  | Description                                                                                 |
| :-------------- | :---- | :------------------------------------------------------------------------------------------ |
| bits            | int   | The number of bits occupied by the type.                                                    |
| eps             | float | The difference between 1.0 and the next smallest representable float larger than 1.0.       |
| max             | float | The largest representable number.                                                           |
| min             | float | The smallest representable number (typically ``-max``).                                     |
| tiny            | float | The smallest positive normal number. Equivalent to ``smallest_normal``.                     |
| smallest_normal | float | The smallest positive normal number. See notes.                                             |
| resolution      | float | The approximate decimal resolution of this type, i.e., ``10**-precision``.                  |

```{note}
  The constructor of {class}`torch.finfo` can be called without argument,
  in which case the class is created for the pytorch default dtype (as returned by {func}`torch.get_default_dtype`).
```

```{note}
  `smallest_normal` returns the smallest *normal* number, but there are smaller
  subnormal numbers. See https://en.wikipedia.org/wiki/Denormal_number
  for more information.
```

(iinfo-doc)=
## torch.iinfo

```{eval-rst}
.. class:: torch.iinfo
```

A {class}`torch.iinfo` is an object that represents the numerical properties of a integer
{class}`torch.dtype` (i.e. ``torch.uint8``, ``torch.int8``, ``torch.int16``, ``torch.int32``, and ``torch.int64``).
This is similar to [numpy.iinfo](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html).

A {class}`torch.iinfo` provides the following attributes:

| Name | Type | Description                              |
| :--- | :--- | :--------------------------------------- |
| bits | int  | The number of bits occupied by the type. |
| max  | int  | The largest representable number.        |
| min  | int  | The smallest representable number.       |

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

*No specific patterns automatically detected.*


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

- **File Documentation**: `type_info.md_docs.md`
- **Keyword Index**: `type_info.md_kw.md`
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

- **File Documentation**: `type_info.md_docs.md_docs.md`
- **Keyword Index**: `type_info.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
