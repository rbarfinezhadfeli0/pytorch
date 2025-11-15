# Keyword Index: `torch/_inductor/codegen/cpp_micro_gemm.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_micro_gemm.py](../../../../torch/_inductor/codegen/cpp_micro_gemm.py)
- **Documentation**: [`cpp_micro_gemm.py_docs.md`](./cpp_micro_gemm.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppMicroBrgemm`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemm`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemmAMX`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemmFP32Vec`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemmRef`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemmWoQInt4Amx`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppMicroGemmWoQInt4Avx512`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`LayoutType`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`class`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`generates`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`that`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`with`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)

### Functions

- **`__init__`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_amx_extra`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_amx_fp16_extra`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_brgemm_extra`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_int8_bf16_amx_extra`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_int8_woq_small_m_dim`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`check_woq_int4_extra`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`codegen_allocate_weight_buffer`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`codegen_call`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`codegen_define`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`codegen_finalize`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`codegen_init`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`create_from_config`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`create_micro_gemm`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`do_not_use_with_small_m_for_int8_woq`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`generate_gemm_config`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_b_layout`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_common_options`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_kernel_declaration`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_kernel_extra_args`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_kernel_extra_args_declare`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`get_restrict_keyword`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`inner`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`is_int8_woq_gemm_small_m_dim_corner_case`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`is_woq_int4`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`register_micro_gemm`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`skip_amx_kernel_for_woq`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`use_local_vnni_blocking`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)

### Imports

- **`..`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`..cpu_vec_isa`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`..utils`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`..virtualized`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`.common`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`.cpp_template_kernel`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`.cpp_utils`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`Callable`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`CppTemplateKernel`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`DTYPE_TO_CPP`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`Enum`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`IndentedBuffer`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`KernelTemplate`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`Optional`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`V`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`collections.abc`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`cpp_builder`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`dataclasses`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`enum`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`has_free_symbols`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`operator`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`sys`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`torch`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)
- **`typing`**: [cpp_micro_gemm.py_docs.md](./cpp_micro_gemm.py_docs.md)


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
