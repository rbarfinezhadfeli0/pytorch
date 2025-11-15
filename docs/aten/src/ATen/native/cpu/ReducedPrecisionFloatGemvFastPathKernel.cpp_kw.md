# Keyword Index: `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp](../../../../../../aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp)
- **Documentation**: [`ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md`](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ForcedUnrollTargetBFloat16`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)

### Functions

- **`IntegerLog2`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`bf16_dot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`bf16_dot_with_fp32_arith`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`bf16_gemv_trans`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`bf16_gemv_trans_fp32_arith_by_dot_products`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_main_inner_loop_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_main_inner_loop_no_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_main_loop_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_main_loop_no_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_no_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_vectorized_tail_inner_loop_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_dot`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_dot_with_fp16_arith`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_dot_with_fp32_arith`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_gemv_trans`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_gemv_trans_fp16_arith_by_dot_products`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`fp16_gemv_trans_fp32_arith_by_dot_products`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`if`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`reduce`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`ATen/Dispatch.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`ATen/Parallel.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`ATen/cpu/vec/functional.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`ATen/cpu/vec/vec.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`arm_neon.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`c10/macros/Macros.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`c10/util/Exception.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`c10/util/Half.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`c10/util/Unroll.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`c10/util/irange.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`cpuinfo.h`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)

### Namespaces

- **`CPU_CAPABILITY`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`at`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)
- **`template`**: [ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md](./ReducedPrecisionFloatGemvFastPathKernel.cpp_docs.md)


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
