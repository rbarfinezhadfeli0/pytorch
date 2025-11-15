# Documentation: `docs/torch/distributed/_tools/mem_tracker.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/_tools/mem_tracker.py_kw.md`
- **Size**: 7,626 bytes (7.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/_tools/mem_tracker.py`

## File Information

- **Original File**: [torch/distributed/_tools/mem_tracker.py](../../../../torch/distributed/_tools/mem_tracker.py)
- **Documentation**: [`mem_tracker.py_docs.md`](./mem_tracker.py_docs.md)
- **Folder**: `torch/distributed/_tools`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MemTracker`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_MemRefType`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_ModMemStats`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_ModState`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_RefType`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_State`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_UpdateType`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_WeakRefInfo`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`self`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`to`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)

### Functions

- **`__enter__`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`__exit__`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`__init__`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`__torch_dispatch__`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_calculate_mem_consumed`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_delete_callback`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_deregister_param_and_optimizer_hooks`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_get_mem_divisor`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_grad_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_opt_step_post_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_opt_step_pre_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_post_bw_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_post_fw_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_pre_bw_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_pre_fw_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_print_snapshot`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_print_snapshot_tabular`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_print_state_snapshots`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_print_state_snapshots_tabular`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_register_global_optimizer_hook`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_restore_resize`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_rounding_fn`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_track`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_track_inputs_or_outputs`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_track_module_params_and_buffers`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_track_optimizer_states`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_track_resize`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_update_and_maybe_create_winfos`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_update_peak_stats`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`_update_snap`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`add_inps_or_outs`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`create_winfo`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`display_modulewise_snapshots`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`display_snapshot`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`get_tracker_snapshot`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`natural_sort_key`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`reset_mod_stats`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`resize_`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`track_external`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`update_mem_consumed`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)

### Imports

- **`Any`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`Callable`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`DTensor`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`ModTracker`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`RemovableHandle`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`Self`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`TorchDispatchMode`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`WeakIdKeyDictionary`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`active_fake_mode`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`auto`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`collections.abc`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`copy`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`deepcopy`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`enum`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`functools`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`get_untyped_storages`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`math`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`nn`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`os`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`partial`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`re`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`tabulate`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch._guards`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.distributed._tools.common_utils`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.distributed._tools.fake_collectives`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.distributed._tools.mod_tracker`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.distributed.tensor`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.optim.optimizer`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.utils._python_dispatch`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.utils._pytree`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.utils.hooks`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`torch.utils.weak`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`tree_flatten`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`typing`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`typing_extensions`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)
- **`warnings`**: [mem_tracker.py_docs.md](./mem_tracker.py_docs.md)


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

- **File Documentation**: `mem_tracker.py_kw.md_docs.md`
- **Keyword Index**: `mem_tracker.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
