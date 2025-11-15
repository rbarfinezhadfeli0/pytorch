# Keyword Index: `torch/_inductor/codegen/rocm/ck_universal_gemm_template.py`

## File Information

- **Original File**: [torch/_inductor/codegen/rocm/ck_universal_gemm_template.py](../../../../../torch/_inductor/codegen/rocm/ck_universal_gemm_template.py)
- **Documentation**: [`ck_universal_gemm_template.py_docs.md`](./ck_universal_gemm_template.py_docs.md)
- **Folder**: `torch/_inductor/codegen/rocm`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CKGemmTemplate`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)

### Functions

- **`GENERATE_CK_STANDALONE_RUNNER`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`__init__`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`_check_num_k_loops`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`_get_kBatch`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`_has_padding`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`_is_rcr_f16`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`_prefetch_stages`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`add_ck_gemm_choices`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`emit_ck_instance`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`filter_op`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`gen_ops`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`globals`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`header`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`inline_utils`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`is_static_int`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`render`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`size_args`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch_layout_to_ck_layout`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)

### Imports

- **`...utils`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`Buffer`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`CKTemplate`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`DTYPE_TO_CPP`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`IndentedBuffer`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`Optional`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`ROCmTemplateKernel`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`ck4inductor.batched_universal_gemm.gen_instances`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`ck4inductor.universal_gemm.gen_instances`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`collections`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`config`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`copy`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`logging`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`math`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`namedtuple`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`next_power_of_2`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`random`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`rocm_compile_command`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`sympy`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.cpp_utils`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.ck_template`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.compile_command`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.rocm_kernel`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.ir`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)
- **`typing`**: [ck_universal_gemm_template.py_docs.md](./ck_universal_gemm_template.py_docs.md)


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
