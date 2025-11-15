# Documentation: `docs/torch/autograd/gradcheck.py_kw.md`

## File Metadata

- **Path**: `docs/torch/autograd/gradcheck.py_kw.md`
- **Size**: 8,643 bytes (8.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/autograd/gradcheck.py`

## File Information

- **Original File**: [torch/autograd/gradcheck.py](../../../torch/autograd/gradcheck.py)
- **Documentation**: [`gradcheck.py_docs.md`](./gradcheck.py_docs.md)
- **Folder**: `torch/autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GradcheckError`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)

### Functions

- **`_adjusted_atol`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_allclose_with_type_promotion`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_allocate_jacobians_with_inputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_allocate_jacobians_with_outputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_as_tuple`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_analytical_jacobian_attributes`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_analytical_numerical_equal`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_inputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_jacobians_equal`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_no_differentiable_outputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_no_differentiable_outputs_fast`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_outputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_check_outputs_same_dtype_and_shape`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_combine_jacobian_cols`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_compute_analytical_jacobian_rows`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_compute_numerical_gradient`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_compute_numerical_jvps_wrt_specific_input`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_densify`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_differentiable_outputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_dot_with_type_promotion`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_fast_gradcheck`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_analytical_jacobian`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_analytical_jacobian_forward_ad`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_analytical_vJu_backward_mode`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_analytical_vjps_wrt_specific_output`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_failed_batched_grad_test_msg`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_inp_tensors`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_input_to_perturb`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_notallclose_msg`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_numerical_jacobian`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_numerical_jvp_fn`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_numerical_jvp_wrt_specific_input`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_get_numerical_vJu`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_gradcheck_helper`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_gradcheck_real_imag`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_is_float_or_complex_tensor`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_is_sparse_any_tensor`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_is_sparse_compressed_tensor`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_iter_tensor`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_iter_tensors`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_make_vectors`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_mul_tensor_or_tuple`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_prepare_input`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_real_and_imag_input`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_real_and_imag_output`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_reshape_tensor_or_tuple`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_run_slow_mode_and_get_error`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_slow_gradcheck`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_stack_and_check_tensors`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_test_backward_mul_by_grad_output`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_test_batched_grad`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_test_batched_grad_forward_ad`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_test_undefined_backward_mode`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_test_undefined_forward_mode`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_to_flat_dense_if_sparse`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_to_real_dtype`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_transpose`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_vec_from_tensor`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_vec_from_tensor_cpu`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_with_prepare_inputs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`apply_to_c_inps`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`apply_to_c_outs`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`check_undefined_grad_support`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`compute`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`fn_pack_inps`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`get_analytical_jacobian`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`get_numerical_jacobian`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`get_numerical_jacobian_wrt_specific_input`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`get_stride`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`gradcheck`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`gradgradcheck`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`jvp`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`jvp_fn`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`new_fn`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`new_func`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`one`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`vjp`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`vjp_fn`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`warn_bc_breaking`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`wrapped_fn`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)

### Imports

- **`Callable`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`Optional`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_TensorOrTensors`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`_vmap`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`collections`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`collections.abc`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`deprecated`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`functools`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`is_tensor_like`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`issues`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`itertools`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`product`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`torch`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`torch._vmap_internals`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`torch.overrides`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`torch.testing`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`torch.types`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`typing`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`typing_extensions`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)
- **`warnings`**: [gradcheck.py_docs.md](./gradcheck.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/autograd`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/autograd`):

- [`profiler.py_docs.md_docs.md`](./profiler.py_docs.md_docs.md)
- [`profiler_util.py_kw.md_docs.md`](./profiler_util.py_kw.md_docs.md)
- [`profiler_util.py_docs.md_docs.md`](./profiler_util.py_docs.md_docs.md)
- [`variable.py_docs.md_docs.md`](./variable.py_docs.md_docs.md)
- [`forward_ad.py_kw.md_docs.md`](./forward_ad.py_kw.md_docs.md)
- [`profiler_legacy.py_docs.md_docs.md`](./profiler_legacy.py_docs.md_docs.md)
- [`graph.py_kw.md_docs.md`](./graph.py_kw.md_docs.md)
- [`forward_ad.py_docs.md_docs.md`](./forward_ad.py_docs.md_docs.md)
- [`functional.py_kw.md_docs.md`](./functional.py_kw.md_docs.md)
- [`profiler.py_kw.md_docs.md`](./profiler.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gradcheck.py_kw.md_docs.md`
- **Keyword Index**: `gradcheck.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
