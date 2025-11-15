# Documentation: `docs/source/torch.compiler_nn_module.md`

## File Metadata

- **Path**: `docs/source/torch.compiler_nn_module.md`
- **Size**: 3,424 bytes (3.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# PyTorch 2.0 NNModule Support

**Author**: [Will Constable](https://github.com/wconstab)

`torch.compile` has special handling for torch.nn.Module objects, tracing them differently than it traces
arbitrary python classes, with the intent of producing faster code by making assumptions about the structure.

This doc describes some of the tradeoffs or edge cases that come up due to this specialization.

## NNModule Hooks Support

Previously, `torch.compile` had no support for hooks on nn.Modules, and if hooks were registered
they would simply be ignored in the compiled program. Indeed many users do not
use nn.Module hooks at all, or only use them for debug workflows, but there are valid use cases
for composing nn.Module hooks with `torch.compile`.

Hooks that are orchestrated via nn.Module.__call__ implementation include `_forward_pre_hooks`,
`forward_hooks`, `_backward_pre_hooks`, and `_backward_hooks`, and will be referred to as 'call hooks'.
These hooks are partially supported by `torch.compile` with limitations described below.

Another category of hooks includes `_state_dict_hooks` and its `pre` and `load_` variants, and are still
unsupported by `torch.compile`.

## `nn.Module.__call__` Hooks Usage and limitations

By default, `torch.compile` will trace the contents of `nn.Module.__call__` which means it will encounter
and run forward/pre-forward hooks.  If you install hooks before calling `torch.compile` and then do not remove
or alter the hooks later, your use case should be supported by default.

Backward/Pre-backward hooks are generally also supported, with similar caveats: currently graph-breaks in dynamo
occur when accessing backward_hooks dicts, which is probably avoiable with some work.  Graph-breaks also impact the
timing of firing backward hooks, since graph-segments are run as autograd-functions which produce all their grads at
the same time.  Assuming it were possible for dynamo to not graph-break on the presence of backward-hooks, we would
still expect the backward hooks for a series of modules to all fire together after the whole compiled graph's backward
ran.

**hooks on 'allowed modules'**
`torch.compile` treats common modules such as torch.conv, as well as modules that are difficult to trace, specially
by allowing them to be called opaquely in the dynamo graph instead of traced into by dynamo.  For such modules, hooks
currently trigger a graph-break so that the affected modules run outside of dynamo.  Depending on the model, this could
introduce a significant performance regression, and additional work is required to improve this support.

**skip_nnmodule_hook_guards**
By default, `torch._dynamo.config.skip_nnmodule_hook_guards` is set to True, meaning no guards will be installed
on each nn.Module hook dictionary, improving runtime by reducing guard execution time, at the cost of not noticing
if any hook dict is changed after compilation.

If you want to be able to remove or modify hooks after compilation and have `torch.compile` react appropriately
(by recompiling), then you need to set `skip_nnmodule_hook_guards=False` and expect a runtime penalty for the added
guards.

TODO: confirm if backward/pre_backward hooks are working or not and document accordingly

## state_dict Hooks

State dict hooks have not yet been supported in `torch.compile`.


TODO: warn_once if graph-breaking on hooks.  warn_once to point to this doc if hooks are present.
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

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `torch.compiler_nn_module.md_docs.md`
- **Keyword Index**: `torch.compiler_nn_module.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
