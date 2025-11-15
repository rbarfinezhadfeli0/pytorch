# Documentation: `docs/torch/distributed/_tools/fsdp2_mem_tracker.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/_tools/fsdp2_mem_tracker.py_kw.md`
- **Size**: 6,341 bytes (6.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/_tools/fsdp2_mem_tracker.py`

## File Information

- **Original File**: [torch/distributed/_tools/fsdp2_mem_tracker.py](../../../../torch/distributed/_tools/fsdp2_mem_tracker.py)
- **Documentation**: [`fsdp2_mem_tracker.py_docs.md`](./fsdp2_mem_tracker.py_docs.md)
- **Folder**: `torch/distributed/_tools`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FSDPMemTracker`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_FSDPModMemStats`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_FSDPModState`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_FSDPRefType`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_FSDPState`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_SavedFSDPMethods`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`to`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)

### Functions

- **`__enter__`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`__exit__`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`__init__`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`__torch_dispatch__`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_deregister_module_and_optimizer_hooks`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_fsdp_param_group_post_backward`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_fsdp_param_group_pre_backward`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_fsdp_state_post_forward`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_fsdp_state_pre_forward`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_instrument_fsdp_module`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_instrument_fsdp_sharded_params_grads`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_instrument_optimizer`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_opt_step_post_hook`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_opt_step_pre_hook`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_register_module_and_optimizer_hooks`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_track_inputs`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`inner`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`track_external`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`track_inputs`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)

### Imports

- **`Any`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`Callable`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`DTensor`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`FSDPModule`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`FSDPParamGroup`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`ParamSpec`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`RemovableHandle`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`TorchDispatchMode`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`WeakIdKeyDictionary`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`_RefType`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`active_fake_mode`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`auto`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`collections.abc`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`copy`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`deepcopy`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`enum`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`functools`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`nn`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`partial`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch._guards`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.distributed._tools.fake_collectives`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.distributed._tools.mem_tracker`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.distributed.fsdp`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_param_group`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.distributed.tensor`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.utils._python_dispatch`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.utils._pytree`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.utils.hooks`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`torch.utils.weak`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`tree_map_only`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`typing`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)
- **`typing_extensions`**: [fsdp2_mem_tracker.py_docs.md](./fsdp2_mem_tracker.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_tools`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_tools`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/_tools`):

- [`fsdp2_mem_tracker.py_docs.md_docs.md`](./fsdp2_mem_tracker.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`runtime_estimator.py_kw.md_docs.md`](./runtime_estimator.py_kw.md_docs.md)
- [`mod_tracker.py_kw.md_docs.md`](./mod_tracker.py_kw.md_docs.md)
- [`sac_estimator.py_docs.md_docs.md`](./sac_estimator.py_docs.md_docs.md)
- [`ilp_utils.py_kw.md_docs.md`](./ilp_utils.py_kw.md_docs.md)
- [`sac_estimator.py_kw.md_docs.md`](./sac_estimator.py_kw.md_docs.md)
- [`fake_collectives.py_docs.md_docs.md`](./fake_collectives.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`runtime_estimator.py_docs.md_docs.md`](./runtime_estimator.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fsdp2_mem_tracker.py_kw.md_docs.md`
- **Keyword Index**: `fsdp2_mem_tracker.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
