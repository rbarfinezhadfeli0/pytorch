# Documentation: `docs/docs/source/checkpoint.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/checkpoint.md_docs.md`
- **Size**: 4,780 bytes (4.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/checkpoint.md`

## File Metadata

- **Path**: `docs/source/checkpoint.md`
- **Size**: 2,257 bytes (2.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# torch.utils.checkpoint

```{note}
Checkpointing is implemented by rerunning a forward-pass segment for
each checkpointed segment during backward propagation.  This can cause persistent
states like the RNG state to be more advanced than they would without
checkpointing.  By default, checkpointing includes logic to juggle
the RNG state such that checkpointed passes making use of RNG
(through dropout for example) have deterministic output as
compared to non-checkpointed passes.  The logic to stash and restore
RNG states can incur a moderate performance hit depending on the runtime
of checkpointed operations.  If deterministic output compared to
non-checkpointed passes is not required, supply `preserve_rng_state=False`
to `checkpoint` or `checkpoint_sequential` to omit stashing and
restoring the RNG state during each checkpoint.

The stashing logic saves and restores the RNG state for CPU and another
device type (infer the device type from Tensor arguments excluding CPU
tensors by `_infer_device_type`) to the `run_fn`. If there are multiple
devices, device state will only be saved for devices of a single device type,
and the remaining devices will be ignored. Consequently, if any checkpointed
functions involve randomness, this may result in incorrect gradients. (Note
that if CUDA devices are among the devices detected, it will be prioritized;
otherwise, the first device encountered will be selected.) If there are no
CPU-tensors, the default device type state (default value is `cuda`, and it
could be set to other device by `DefaultDeviceType`) will be saved and restored.
However, the logic has no way to anticipate if the user will move
Tensors to a new device within the `run_fn` itself.  Therefore, if you move
Tensors to a new device ("new" meaning not belonging to the set of
[current device + devices of Tensor arguments]) within `run_fn`, deterministic
output compared to non-checkpointed passes is never guaranteed.
```

```{eval-rst}
.. currentmodule:: torch.utils.checkpoint
.. autofunction:: checkpoint
.. autofunction:: checkpoint_sequential
.. autofunction:: set_checkpoint_debug_enabled
.. autoclass:: CheckpointPolicy
.. autoclass:: SelectiveCheckpointContext
.. autofunction:: create_selective_checkpoint_contexts
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

- **File Documentation**: `checkpoint.md_docs.md`
- **Keyword Index**: `checkpoint.md_kw.md`
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

- **File Documentation**: `checkpoint.md_docs.md_docs.md`
- **Keyword Index**: `checkpoint.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
