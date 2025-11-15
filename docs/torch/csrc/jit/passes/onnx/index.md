# Index: `torch/csrc/jit/passes/onnx/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/csrc/jit/passes/onnx/`

## Subfolders

- [`pattern_conversion/`](./pattern_conversion/index.md) - pattern_conversion module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`README.md`](../../../../../../torch/csrc/jit/passes/onnx/README.md) | Documentation | [docs](./README.md_docs.md) | [keywords](./README.md_kw.md) |
| [`cast_all_constant_to_floating.cpp`](../../../../../../torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp) | Source code | [docs](./cast_all_constant_to_floating.cpp_docs.md) | [keywords](./cast_all_constant_to_floating.cpp_kw.md) |
| [`cast_all_constant_to_floating.h`](../../../../../../torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h) | Source code | [docs](./cast_all_constant_to_floating.h_docs.md) | [keywords](./cast_all_constant_to_floating.h_kw.md) |
| [`constant_fold.cpp`](../../../../../../torch/csrc/jit/passes/onnx/constant_fold.cpp) | Source code | [docs](./constant_fold.cpp_docs.md) | [keywords](./constant_fold.cpp_kw.md) |
| [`constant_fold.h`](../../../../../../torch/csrc/jit/passes/onnx/constant_fold.h) | Source code | [docs](./constant_fold.h_docs.md) | [keywords](./constant_fold.h_kw.md) |
| [`constant_map.cpp`](../../../../../../torch/csrc/jit/passes/onnx/constant_map.cpp) | Source code | [docs](./constant_map.cpp_docs.md) | [keywords](./constant_map.cpp_kw.md) |
| [`constant_map.h`](../../../../../../torch/csrc/jit/passes/onnx/constant_map.h) | Source code | [docs](./constant_map.h_docs.md) | [keywords](./constant_map.h_kw.md) |
| [`deduplicate_initializers.cpp`](../../../../../../torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp) | Source code | [docs](./deduplicate_initializers.cpp_docs.md) | [keywords](./deduplicate_initializers.cpp_kw.md) |
| [`deduplicate_initializers.h`](../../../../../../torch/csrc/jit/passes/onnx/deduplicate_initializers.h) | Source code | [docs](./deduplicate_initializers.h_docs.md) | [keywords](./deduplicate_initializers.h_kw.md) |
| [`eliminate_unused_items.cpp`](../../../../../../torch/csrc/jit/passes/onnx/eliminate_unused_items.cpp) | Source code | [docs](./eliminate_unused_items.cpp_docs.md) | [keywords](./eliminate_unused_items.cpp_kw.md) |
| [`eliminate_unused_items.h`](../../../../../../torch/csrc/jit/passes/onnx/eliminate_unused_items.h) | Source code | [docs](./eliminate_unused_items.h_docs.md) | [keywords](./eliminate_unused_items.h_kw.md) |
| [`eval_peephole.cpp`](../../../../../../torch/csrc/jit/passes/onnx/eval_peephole.cpp) | Source code | [docs](./eval_peephole.cpp_docs.md) | [keywords](./eval_peephole.cpp_kw.md) |
| [`eval_peephole.h`](../../../../../../torch/csrc/jit/passes/onnx/eval_peephole.h) | Source code | [docs](./eval_peephole.h_docs.md) | [keywords](./eval_peephole.h_kw.md) |
| [`fixup_onnx_controlflow.cpp`](../../../../../../torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.cpp) | Source code | [docs](./fixup_onnx_controlflow.cpp_docs.md) | [keywords](./fixup_onnx_controlflow.cpp_kw.md) |
| [`fixup_onnx_controlflow.h`](../../../../../../torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h) | Source code | [docs](./fixup_onnx_controlflow.h_docs.md) | [keywords](./fixup_onnx_controlflow.h_kw.md) |
| [`function_extraction.cpp`](../../../../../../torch/csrc/jit/passes/onnx/function_extraction.cpp) | Source code | [docs](./function_extraction.cpp_docs.md) | [keywords](./function_extraction.cpp_kw.md) |
| [`function_extraction.h`](../../../../../../torch/csrc/jit/passes/onnx/function_extraction.h) | Source code | [docs](./function_extraction.h_docs.md) | [keywords](./function_extraction.h_kw.md) |
| [`function_substitution.cpp`](../../../../../../torch/csrc/jit/passes/onnx/function_substitution.cpp) | Source code | [docs](./function_substitution.cpp_docs.md) | [keywords](./function_substitution.cpp_kw.md) |
| [`function_substitution.h`](../../../../../../torch/csrc/jit/passes/onnx/function_substitution.h) | Source code | [docs](./function_substitution.h_docs.md) | [keywords](./function_substitution.h_kw.md) |
| [`helper.cpp`](../../../../../../torch/csrc/jit/passes/onnx/helper.cpp) | Source code | [docs](./helper.cpp_docs.md) | [keywords](./helper.cpp_kw.md) |
| [`helper.h`](../../../../../../torch/csrc/jit/passes/onnx/helper.h) | Source code | [docs](./helper.h_docs.md) | [keywords](./helper.h_kw.md) |
| [`list_model_parameters.cpp`](../../../../../../torch/csrc/jit/passes/onnx/list_model_parameters.cpp) | Source code | [docs](./list_model_parameters.cpp_docs.md) | [keywords](./list_model_parameters.cpp_kw.md) |
| [`list_model_parameters.h`](../../../../../../torch/csrc/jit/passes/onnx/list_model_parameters.h) | Source code | [docs](./list_model_parameters.h_docs.md) | [keywords](./list_model_parameters.h_kw.md) |
| [`naming.cpp`](../../../../../../torch/csrc/jit/passes/onnx/naming.cpp) | Source code | [docs](./naming.cpp_docs.md) | [keywords](./naming.cpp_kw.md) |
| [`naming.h`](../../../../../../torch/csrc/jit/passes/onnx/naming.h) | Source code | [docs](./naming.h_docs.md) | [keywords](./naming.h_kw.md) |
| [`onnx_log.cpp`](../../../../../../torch/csrc/jit/passes/onnx/onnx_log.cpp) | Source code | [docs](./onnx_log.cpp_docs.md) | [keywords](./onnx_log.cpp_kw.md) |
| [`onnx_log.h`](../../../../../../torch/csrc/jit/passes/onnx/onnx_log.h) | Source code | [docs](./onnx_log.h_docs.md) | [keywords](./onnx_log.h_kw.md) |
| [`peephole.cpp`](../../../../../../torch/csrc/jit/passes/onnx/peephole.cpp) | Source code | [docs](./peephole.cpp_docs.md) | [keywords](./peephole.cpp_kw.md) |
| [`peephole.h`](../../../../../../torch/csrc/jit/passes/onnx/peephole.h) | Source code | [docs](./peephole.h_docs.md) | [keywords](./peephole.h_kw.md) |
| [`prepare_division_for_onnx.cpp`](../../../../../../torch/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp) | Source code | [docs](./prepare_division_for_onnx.cpp_docs.md) | [keywords](./prepare_division_for_onnx.cpp_kw.md) |
| [`prepare_division_for_onnx.h`](../../../../../../torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h) | Source code | [docs](./prepare_division_for_onnx.h_docs.md) | [keywords](./prepare_division_for_onnx.h_kw.md) |
| [`preprocess_for_onnx.cpp`](../../../../../../torch/csrc/jit/passes/onnx/preprocess_for_onnx.cpp) | Source code | [docs](./preprocess_for_onnx.cpp_docs.md) | [keywords](./preprocess_for_onnx.cpp_kw.md) |
| [`preprocess_for_onnx.h`](../../../../../../torch/csrc/jit/passes/onnx/preprocess_for_onnx.h) | Source code | [docs](./preprocess_for_onnx.h_docs.md) | [keywords](./preprocess_for_onnx.h_kw.md) |
| [`remove_inplace_ops_for_onnx.cpp`](../../../../../../torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.cpp) | Source code | [docs](./remove_inplace_ops_for_onnx.cpp_docs.md) | [keywords](./remove_inplace_ops_for_onnx.cpp_kw.md) |
| [`remove_inplace_ops_for_onnx.h`](../../../../../../torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h) | Source code | [docs](./remove_inplace_ops_for_onnx.h_docs.md) | [keywords](./remove_inplace_ops_for_onnx.h_kw.md) |
| [`scalar_type_analysis.cpp`](../../../../../../torch/csrc/jit/passes/onnx/scalar_type_analysis.cpp) | Source code | [docs](./scalar_type_analysis.cpp_docs.md) | [keywords](./scalar_type_analysis.cpp_kw.md) |
| [`scalar_type_analysis.h`](../../../../../../torch/csrc/jit/passes/onnx/scalar_type_analysis.h) | Source code | [docs](./scalar_type_analysis.h_docs.md) | [keywords](./scalar_type_analysis.h_kw.md) |
| [`shape_type_inference.cpp`](../../../../../../torch/csrc/jit/passes/onnx/shape_type_inference.cpp) | Source code | [docs](./shape_type_inference.cpp_docs.md) | [keywords](./shape_type_inference.cpp_kw.md) |
| [`shape_type_inference.h`](../../../../../../torch/csrc/jit/passes/onnx/shape_type_inference.h) | Source code | [docs](./shape_type_inference.h_docs.md) | [keywords](./shape_type_inference.h_kw.md) |
| [`unpack_quantized_weights.cpp`](../../../../../../torch/csrc/jit/passes/onnx/unpack_quantized_weights.cpp) | Source code | [docs](./unpack_quantized_weights.cpp_docs.md) | [keywords](./unpack_quantized_weights.cpp_kw.md) |
| [`unpack_quantized_weights.h`](../../../../../../torch/csrc/jit/passes/onnx/unpack_quantized_weights.h) | Source code | [docs](./unpack_quantized_weights.h_docs.md) | [keywords](./unpack_quantized_weights.h_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
