# Keyword Index: `torch/_inductor/codegen/cpp_wrapper_gpu.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_wrapper_gpu.py](../../../../torch/_inductor/codegen/cpp_wrapper_gpu.py)
- **Documentation**: [`cpp_wrapper_gpu.py_docs.md`](./cpp_wrapper_gpu.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppWrapperGpu`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`class`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)

### Functions

- **`__init__`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`_define_kernel_helper`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`_generate_experimental_tma_descriptor`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`_generate_kernel_call_helper`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`_generate_stable_tma_descriptor`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`codegen_inputs`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`cpp_string_literal`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`create`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`fill_array`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`finalize_prefix`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate_args_decl`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate_grid`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate_launch_kernel`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate_load_kernel`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`generate_tma_descriptor`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`get_autotuning_input_name`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`make_zero_buffer`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`prepare_triton_wrapper_args`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`process_args`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`process_args_for_input_shape`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`process_tma_stable_arg`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`signature_is_tma_desc`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`write_dummy_scalar_ivalue`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`write_get_raw_stream`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`write_header`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`write_tma_descriptor_helpers_once`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)

### Imports

- **`..`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`..codecache`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`..ir`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`..runtime.triton_heuristics`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`..utils`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`..virtualized`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.aoti_hipify_utils`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.common`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.cpp_utils`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.cpp_wrapper_cpu`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.multi_kernel`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.triton_utils`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`.wrapper`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`Any`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`CppWrapperCpu`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`CudaKernelParamCache`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`GridExpr`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`MultiKernelCall`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`PythonWrapperCodegen`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`Self`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`V`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`__future__`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`annotations`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`cache_on_self`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`cexpr`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`config`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`count`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`dataclasses`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`dtype`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`dynamo_timed`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`get_cpp_wrapper_cubin_path_name`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`get_device_op_overrides`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`itertools`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`maybe_hipify_code_wrapper`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`re`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`should_unwrap_unspec_arg`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`sympy`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`sys`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`torch`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`torch._inductor.codecache`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`typing`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)
- **`typing_extensions`**: [cpp_wrapper_gpu.py_docs.md](./cpp_wrapper_gpu.py_docs.md)


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
