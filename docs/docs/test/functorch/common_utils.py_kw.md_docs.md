# Documentation: `docs/test/functorch/common_utils.py_kw.md`

## File Metadata

- **Path**: `docs/test/functorch/common_utils.py_kw.md`
- **Size**: 6,009 bytes (5.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `test/functorch/common_utils.py`

## File Information

- **Original File**: [test/functorch/common_utils.py](../../../test/functorch/common_utils.py)
- **Documentation**: [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- **Folder**: `test/functorch`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DisableVmapFallback`**: [common_utils.py_docs.md](./common_utils.py_docs.md)

### Functions

- **`__enter__`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`__exit__`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`_compute_quantities_for_vmap_test`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`add_batch_dim`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`add_bdim_if_tensor`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`check_vmap_fallback`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`clone_if_tensor`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`compute_quantities_for_vmap_test`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`construct_in_dims`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`decorate`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`decorateForModules`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`decorator`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`expectedFailureIf`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`f`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`generate_vmap_inputs`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`get_batched_arg`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`get_bdim_choices`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`get_bdim_choices_batch_norm`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`get_fallback_and_vmap_exhaustive`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`is_batch_norm_training`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`is_valid_inplace_sample_input`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`loop`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`loop2`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`make_batched`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`maybe_clone_inputs`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`memoize`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`opinfo_in_dict`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`opsToleranceOverride`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`saved_tensors_hooks_to_gm`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`set_manual_hash`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`skip`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`skipOps`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`tol1`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`tol2`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`wrapped`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`xfail`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`xfailIf`**: [common_utils.py_docs.md](./common_utils.py_docs.md)

### Imports

- **`DecorateInfo`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`additional_op_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`autograd_function_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`collections`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`custom_op_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`functorch`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`functorch_additional_op_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`itertools`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`make_fx`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`module_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`namedtuple`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`os`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`sample_skips_and_xfails`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`toleranceOverride`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.functorch`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.autograd_function_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.common_methods_invocations`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.common_modules`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.custom_op_db`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.testing._internal.opinfo.core`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`torch.utils._pytree`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`unittest`**: [common_utils.py_docs.md](./common_utils.py_docs.md)
- **`vmap`**: [common_utils.py_docs.md](./common_utils.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/test/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/functorch`, which is part of the **core PyTorch library**.



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

This is a test file. Run it with:

```bash
python docs/test/functorch/common_utils.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/functorch`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_aot_joint_with_descriptors.py_kw.md_docs.md`](./test_aot_joint_with_descriptors.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_eager_transforms.py_docs.md_docs.md`](./test_eager_transforms.py_docs.md_docs.md)
- [`functorch_additional_op_db.py_kw.md_docs.md`](./functorch_additional_op_db.py_kw.md_docs.md)
- [`test_ac_knapsack.py_docs.md_docs.md`](./test_ac_knapsack.py_docs.md_docs.md)
- [`test_logging.py_kw.md_docs.md`](./test_logging.py_kw.md_docs.md)
- [`test_rearrange.py_kw.md_docs.md`](./test_rearrange.py_kw.md_docs.md)
- [`test_dims.py_kw.md_docs.md`](./test_dims.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `common_utils.py_kw.md_docs.md`
- **Keyword Index**: `common_utils.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
