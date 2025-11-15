# Documentation: `docs/torch/_functorch/_aot_autograd/graph_capture_wrappers.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_functorch/_aot_autograd/graph_capture_wrappers.py_kw.md`
- **Size**: 9,594 bytes (9.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_functorch/_aot_autograd/graph_capture_wrappers.py`

## File Information

- **Original File**: [torch/_functorch/_aot_autograd/graph_capture_wrappers.py](../../../../torch/_functorch/_aot_autograd/graph_capture_wrappers.py)
- **Documentation**: [`graph_capture_wrappers.py_docs.md`](./graph_capture_wrappers.py_docs.md)
- **Folder**: `torch/_functorch/_aot_autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Metadata`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`class`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`from`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`graph`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`guards`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`info`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`metadata`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`outputs`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)

### Functions

- **`_functionalized_f_helper`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`_get_inductor_storage_resized_counter`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`_get_mutation_counter`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`_get_mutation_counters`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`_get_storage_changed_counter`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`_post_forward`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`aot_dispatch_subclass`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`append_rng_offsets`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`apply_in_graph_mutations`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`create_functional_call`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`create_functionalized_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`create_functionalized_rng_ops_wrapper`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`create_joint`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`disable_autocast`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`f`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`fn_input_mutations_to_outputs`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`fn_prepped_for_autograd`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`functional_call`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`fw_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`handle_effect_tokens_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`inner_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`inner_fn_with_anomaly`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`inner_fw_only`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`joint_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`joint_helper`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`metadata_fn`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`override_get_rng_state`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`override_set_rng_state`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`sc_visit`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`set_partitioner_tag`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`set_partitioner_tag_is_backward`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`set_partitioner_tag_must_be_in_backward`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`set_partitioner_tag_must_be_in_forward`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`traced_forward`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`traced_joint`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`visit`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)

### Imports

- **`..`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.collect_metadata_analysis`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.descriptors`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.functional_utils`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.logging_utils`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.schemas`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.subclass_utils`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`.utils`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`AbstractContextManager`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`Any`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`CUDARngStateHelper`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`Callable`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`FunctionalTensor`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`PhiloxStateTracker`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`Tensor`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`TreeSpec`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`collections.abc`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`config`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`contextlib`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`dataclass`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`dataclasses`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`detect_fake_mode`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`is_traceable_wrapper_subclass`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`patch`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`run_functionalized_fw_and_collect_metadata`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`setup_stacktrace_preservation_hooks`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`stateless`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch._decomp.decompositions_for_rng`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch._guards`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch._prims_common`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.fx.traceback`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.nn.utils`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.utils._python_dispatch`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`torch.utils._pytree`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`typing`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`unittest.mock`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)
- **`warnings`**: [graph_capture_wrappers.py_docs.md](./graph_capture_wrappers.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_functorch/_aot_autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_functorch/_aot_autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/_functorch/_aot_autograd`):

- [`graph_compile.py_kw.md_docs.md`](./graph_compile.py_kw.md_docs.md)
- [`frontend_utils.py_kw.md_docs.md`](./frontend_utils.py_kw.md_docs.md)
- [`autograd_cache.py_docs.md_docs.md`](./autograd_cache.py_docs.md_docs.md)
- [`input_output_analysis.py_kw.md_docs.md`](./input_output_analysis.py_kw.md_docs.md)
- [`schemas.py_docs.md_docs.md`](./schemas.py_docs.md_docs.md)
- [`collect_metadata_analysis.py_docs.md_docs.md`](./collect_metadata_analysis.py_docs.md_docs.md)
- [`functional_utils.py_docs.md_docs.md`](./functional_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`logging_utils.py_docs.md_docs.md`](./logging_utils.py_docs.md_docs.md)
- [`graph_capture.py_kw.md_docs.md`](./graph_capture.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `graph_capture_wrappers.py_kw.md_docs.md`
- **Keyword Index**: `graph_capture_wrappers.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
