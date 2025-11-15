# Documentation: `docs/docs/source/distributed.tensor.parallel.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/distributed.tensor.parallel.md_docs.md`
- **Size**: 5,506 bytes (5.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/distributed.tensor.parallel.md`

## File Metadata

- **Path**: `docs/source/distributed.tensor.parallel.md`
- **Size**: 2,915 bytes (2.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
:::{role} hidden
    :class: hidden-section
:::

# Tensor Parallelism - torch.distributed.tensor.parallel

Tensor Parallelism(TP) is built on top of the PyTorch DistributedTensor
([DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md))
and provides different parallelism styles: Colwise, Rowwise, and Sequence Parallelism.

:::{warning}
Tensor Parallelism APIs are experimental and subject to change.
:::

The entrypoint to parallelize your `nn.Module` using Tensor Parallelism is:

```{eval-rst}
.. automodule:: torch.distributed.tensor.parallel
```

```{eval-rst}
.. currentmodule:: torch.distributed.tensor.parallel
```

```{eval-rst}
.. autofunction::  parallelize_module
```

Tensor Parallelism supports the following parallel styles:

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.ColwiseParallel
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.RowwiseParallel
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.SequenceParallel
  :members:
  :undoc-members:
```

To simply configure the nn.Module's inputs and outputs with DTensor layouts
and perform necessary layout redistributions, without distribute the module
parameters to DTensors, the following `ParallelStyle` s can be used in
the `parallelize_plan` when calling `parallelize_module`:


```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInput
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleOutput
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInputOutput
  :members:
  :undoc-members:
```

:::{note}
when using the `Shard(dim)` as the input/output layouts for the above
`ParallelStyle` s, we assume the input/output activation tensors are evenly sharded on
the tensor dimension `dim` on the `DeviceMesh` that TP operates on. For instance,
since `RowwiseParallel` accepts input that is sharded on the last dimension, it assumes
the input tensor has already been evenly sharded on the last dimension. For the case of uneven sharded activation tensors, one could pass in DTensor directly to the partitioned modules, and use `use_local_output=False` to return DTensor after each `ParallelStyle`, where DTensor could track the uneven sharding information.
:::

For models like Transformer, we recommend users to use `ColwiseParallel`
and `RowwiseParallel` together in the parallelize_plan for achieve the desired
sharding for the entire model (i.e. Attention and MLP).

Parallelized cross-entropy loss computation (loss parallelism), is supported via the following context manager:

```{eval-rst}
.. autofunction:: torch.distributed.tensor.parallel.loss_parallel
```
:::{warning}
    The loss_parallel API is experimental and subject to change.
:::

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

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `distributed.tensor.parallel.md_docs.md`
- **Keyword Index**: `distributed.tensor.parallel.md_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `distributed.tensor.parallel.md_docs.md_docs.md`
- **Keyword Index**: `distributed.tensor.parallel.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
