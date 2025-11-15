# Index: `torch/_inductor/codegen/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/_inductor/codegen/`

## Subfolders

- [`aoti_runtime/`](./aoti_runtime/index.md) - aoti_runtime module
- [`cuda/`](./cuda/index.md) - cuda module
- [`cutedsl/`](./cutedsl/index.md) - cutedsl module
- [`mtia/`](./mtia/index.md) - mtia module
- [`rocm/`](./rocm/index.md) - rocm module
- [`xpu/`](./xpu/index.md) - xpu module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`__init__.py`](../../../../torch/_inductor/codegen/__init__.py) | Package initialization | [docs](./__init__.py_docs.md) | [keywords](./__init__.py_kw.md) |
| [`aoti_hipify_utils.py`](../../../../torch/_inductor/codegen/aoti_hipify_utils.py) | Source code | [docs](./aoti_hipify_utils.py_docs.md) | [keywords](./aoti_hipify_utils.py_kw.md) |
| [`block_analysis.py`](../../../../torch/_inductor/codegen/block_analysis.py) | Source code | [docs](./block_analysis.py_docs.md) | [keywords](./block_analysis.py_kw.md) |
| [`common.py`](../../../../torch/_inductor/codegen/common.py) | Source code | [docs](./common.py_docs.md) | [keywords](./common.py_kw.md) |
| [`cpp.py`](../../../../torch/_inductor/codegen/cpp.py) | Source code | [docs](./cpp.py_docs.md) | [keywords](./cpp.py_kw.md) |
| [`cpp_bmm_template.py`](../../../../torch/_inductor/codegen/cpp_bmm_template.py) | Source code | [docs](./cpp_bmm_template.py_docs.md) | [keywords](./cpp_bmm_template.py_kw.md) |
| [`cpp_flex_attention_template.py`](../../../../torch/_inductor/codegen/cpp_flex_attention_template.py) | Source code | [docs](./cpp_flex_attention_template.py_docs.md) | [keywords](./cpp_flex_attention_template.py_kw.md) |
| [`cpp_gemm_template.py`](../../../../torch/_inductor/codegen/cpp_gemm_template.py) | Source code | [docs](./cpp_gemm_template.py_docs.md) | [keywords](./cpp_gemm_template.py_kw.md) |
| [`cpp_grouped_gemm_template.py`](../../../../torch/_inductor/codegen/cpp_grouped_gemm_template.py) | Source code | [docs](./cpp_grouped_gemm_template.py_docs.md) | [keywords](./cpp_grouped_gemm_template.py_kw.md) |
| [`cpp_micro_gemm.py`](../../../../torch/_inductor/codegen/cpp_micro_gemm.py) | Source code | [docs](./cpp_micro_gemm.py_docs.md) | [keywords](./cpp_micro_gemm.py_kw.md) |
| [`cpp_template.py`](../../../../torch/_inductor/codegen/cpp_template.py) | Source code | [docs](./cpp_template.py_docs.md) | [keywords](./cpp_template.py_kw.md) |
| [`cpp_template_kernel.py`](../../../../torch/_inductor/codegen/cpp_template_kernel.py) | Source code | [docs](./cpp_template_kernel.py_docs.md) | [keywords](./cpp_template_kernel.py_kw.md) |
| [`cpp_utils.py`](../../../../torch/_inductor/codegen/cpp_utils.py) | Source code | [docs](./cpp_utils.py_docs.md) | [keywords](./cpp_utils.py_kw.md) |
| [`cpp_wrapper_cpu.py`](../../../../torch/_inductor/codegen/cpp_wrapper_cpu.py) | Source code | [docs](./cpp_wrapper_cpu.py_docs.md) | [keywords](./cpp_wrapper_cpu.py_kw.md) |
| [`cpp_wrapper_cpu_array_ref.py`](../../../../torch/_inductor/codegen/cpp_wrapper_cpu_array_ref.py) | Source code | [docs](./cpp_wrapper_cpu_array_ref.py_docs.md) | [keywords](./cpp_wrapper_cpu_array_ref.py_kw.md) |
| [`cpp_wrapper_gpu.py`](../../../../torch/_inductor/codegen/cpp_wrapper_gpu.py) | Source code | [docs](./cpp_wrapper_gpu.py_docs.md) | [keywords](./cpp_wrapper_gpu.py_kw.md) |
| [`cpp_wrapper_mps.py`](../../../../torch/_inductor/codegen/cpp_wrapper_mps.py) | Source code | [docs](./cpp_wrapper_mps.py_docs.md) | [keywords](./cpp_wrapper_mps.py_kw.md) |
| [`cpu_device_op_overrides.py`](../../../../torch/_inductor/codegen/cpu_device_op_overrides.py) | Source code | [docs](./cpu_device_op_overrides.py_docs.md) | [keywords](./cpu_device_op_overrides.py_kw.md) |
| [`cuda_combined_scheduling.py`](../../../../torch/_inductor/codegen/cuda_combined_scheduling.py) | Source code | [docs](./cuda_combined_scheduling.py_docs.md) | [keywords](./cuda_combined_scheduling.py_kw.md) |
| [`debug_utils.py`](../../../../torch/_inductor/codegen/debug_utils.py) | Source code | [docs](./debug_utils.py_docs.md) | [keywords](./debug_utils.py_kw.md) |
| [`halide.py`](../../../../torch/_inductor/codegen/halide.py) | Source code | [docs](./halide.py_docs.md) | [keywords](./halide.py_kw.md) |
| [`memory_planning.py`](../../../../torch/_inductor/codegen/memory_planning.py) | Source code | [docs](./memory_planning.py_docs.md) | [keywords](./memory_planning.py_kw.md) |
| [`mps.py`](../../../../torch/_inductor/codegen/mps.py) | Source code | [docs](./mps.py_docs.md) | [keywords](./mps.py_kw.md) |
| [`mps_device_op_overrides.py`](../../../../torch/_inductor/codegen/mps_device_op_overrides.py) | Source code | [docs](./mps_device_op_overrides.py_docs.md) | [keywords](./mps_device_op_overrides.py_kw.md) |
| [`multi_kernel.py`](../../../../torch/_inductor/codegen/multi_kernel.py) | Source code | [docs](./multi_kernel.py_docs.md) | [keywords](./multi_kernel.py_kw.md) |
| [`pallas.py`](../../../../torch/_inductor/codegen/pallas.py) | Source code | [docs](./pallas.py_docs.md) | [keywords](./pallas.py_kw.md) |
| [`python_wrapper_mtia.py`](../../../../torch/_inductor/codegen/python_wrapper_mtia.py) | Source code | [docs](./python_wrapper_mtia.py_docs.md) | [keywords](./python_wrapper_mtia.py_kw.md) |
| [`segmented_tree.py`](../../../../torch/_inductor/codegen/segmented_tree.py) | Source code | [docs](./segmented_tree.py_docs.md) | [keywords](./segmented_tree.py_kw.md) |
| [`simd.py`](../../../../torch/_inductor/codegen/simd.py) | Source code | [docs](./simd.py_docs.md) | [keywords](./simd.py_kw.md) |
| [`simd_kernel_features.py`](../../../../torch/_inductor/codegen/simd_kernel_features.py) | Source code | [docs](./simd_kernel_features.py_docs.md) | [keywords](./simd_kernel_features.py_kw.md) |
| [`subgraph.py`](../../../../torch/_inductor/codegen/subgraph.py) | Source code | [docs](./subgraph.py_docs.md) | [keywords](./subgraph.py_kw.md) |
| [`triton.py`](../../../../torch/_inductor/codegen/triton.py) | Source code | [docs](./triton.py_docs.md) | [keywords](./triton.py_kw.md) |
| [`triton_combo_kernel.py`](../../../../torch/_inductor/codegen/triton_combo_kernel.py) | Source code | [docs](./triton_combo_kernel.py_docs.md) | [keywords](./triton_combo_kernel.py_kw.md) |
| [`triton_split_scan.py`](../../../../torch/_inductor/codegen/triton_split_scan.py) | Source code | [docs](./triton_split_scan.py_docs.md) | [keywords](./triton_split_scan.py_kw.md) |
| [`triton_utils.py`](../../../../torch/_inductor/codegen/triton_utils.py) | Source code | [docs](./triton_utils.py_docs.md) | [keywords](./triton_utils.py_kw.md) |
| [`wrapper.py`](../../../../torch/_inductor/codegen/wrapper.py) | Source code | [docs](./wrapper.py_docs.md) | [keywords](./wrapper.py_kw.md) |
| [`wrapper_fxir.py`](../../../../torch/_inductor/codegen/wrapper_fxir.py) | Source code | [docs](./wrapper_fxir.py_docs.md) | [keywords](./wrapper_fxir.py_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
