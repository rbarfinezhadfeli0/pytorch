# Keyword Index: `torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py](../../../../torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py)
- **Documentation**: [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppWrapperCpuArrayRef`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`is`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)

### Functions

- **`__init__`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`_assert_safe_to_use_borrow_arrayref_tensor_as_tensor`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`_generate_index_put_fallback`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`_generate_kernel_call_helper`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`_generate_scatter_fallback`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`can_stack_allocate_buffer`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`codegen_device_copy`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`codegen_input_numel_asserts`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`codegen_reinterpret_view`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`codegen_tensor_item`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`create`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`create_new_tensor_handle`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`create_reinterpret_call`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_c_shim_extern_kernel_call`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_extern_kernel_alloc`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_extern_kernel_out`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_fallback_kernel`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_fallback_kernel_with_runtime_lookup`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_index_put_fallback`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_return`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`generate_scatter_fallback`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`get_device_include_path`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`get_input_cpp_type`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`is_safe_to_use_borrow_arrayref_tensor_as_tensor`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`make_allocation`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`make_buffer_allocation`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`make_buffer_free`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`make_buffer_reuse`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`memory_plan`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`memory_plan_reuse`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`use_thread_local_cached_output_tensor`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`val_to_arg_str`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`write_wrapper_decl`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)

### Imports

- **`..`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`..graph`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`..utils`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`..virtualized`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`.cpp_utils`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`.cpp_wrapper_cpu`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`.memory_planning`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`.wrapper`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`Any`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`Callable`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`CppWrapperCpu`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`DTYPE_TO_CPP`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`MemoryPlanner`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`V`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`collections.abc`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`config`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`may_get_constant_buffer_dtype`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`sympy`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`sympy_product`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`torch`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`torch._inductor.async_compile`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`torch._ops`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)
- **`typing`**: [cpp_wrapper_cpu_array_ref.py_docs.md](./cpp_wrapper_cpu_array_ref.py_docs.md)


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
