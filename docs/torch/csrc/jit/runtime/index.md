# Index: `torch/csrc/jit/runtime/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/csrc/jit/runtime/`

## Subfolders

- [`interpreter/`](./interpreter/index.md) - interpreter module
- [`static/`](./static/index.md) - static module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`argument_spec.cpp`](../../../../../torch/csrc/jit/runtime/argument_spec.cpp) | Source code | [docs](./argument_spec.cpp_docs.md) | [keywords](./argument_spec.cpp_kw.md) |
| [`argument_spec.h`](../../../../../torch/csrc/jit/runtime/argument_spec.h) | Source code | [docs](./argument_spec.h_docs.md) | [keywords](./argument_spec.h_kw.md) |
| [`autodiff.cpp`](../../../../../torch/csrc/jit/runtime/autodiff.cpp) | Source code | [docs](./autodiff.cpp_docs.md) | [keywords](./autodiff.cpp_kw.md) |
| [`autodiff.h`](../../../../../torch/csrc/jit/runtime/autodiff.h) | Source code | [docs](./autodiff.h_docs.md) | [keywords](./autodiff.h_kw.md) |
| [`calculate_necessary_args.h`](../../../../../torch/csrc/jit/runtime/calculate_necessary_args.h) | Source code | [docs](./calculate_necessary_args.h_docs.md) | [keywords](./calculate_necessary_args.h_kw.md) |
| [`custom_operator.h`](../../../../../torch/csrc/jit/runtime/custom_operator.h) | Source code | [docs](./custom_operator.h_docs.md) | [keywords](./custom_operator.h_kw.md) |
| [`decomposition_registry.cpp`](../../../../../torch/csrc/jit/runtime/decomposition_registry.cpp) | Source code | [docs](./decomposition_registry.cpp_docs.md) | [keywords](./decomposition_registry.cpp_kw.md) |
| [`decomposition_registry.h`](../../../../../torch/csrc/jit/runtime/decomposition_registry.h) | Source code | [docs](./decomposition_registry.h_docs.md) | [keywords](./decomposition_registry.h_kw.md) |
| [`decomposition_registry_util.cpp`](../../../../../torch/csrc/jit/runtime/decomposition_registry_util.cpp) | Source code | [docs](./decomposition_registry_util.cpp_docs.md) | [keywords](./decomposition_registry_util.cpp_kw.md) |
| [`decomposition_registry_util.h`](../../../../../torch/csrc/jit/runtime/decomposition_registry_util.h) | Source code | [docs](./decomposition_registry_util.h_docs.md) | [keywords](./decomposition_registry_util.h_kw.md) |
| [`exception_message.h`](../../../../../torch/csrc/jit/runtime/exception_message.h) | Source code | [docs](./exception_message.h_docs.md) | [keywords](./exception_message.h_kw.md) |
| [`graph_executor.cpp`](../../../../../torch/csrc/jit/runtime/graph_executor.cpp) | Source code | [docs](./graph_executor.cpp_docs.md) | [keywords](./graph_executor.cpp_kw.md) |
| [`graph_executor.h`](../../../../../torch/csrc/jit/runtime/graph_executor.h) | Source code | [docs](./graph_executor.h_docs.md) | [keywords](./graph_executor.h_kw.md) |
| [`graph_executor_impl.h`](../../../../../torch/csrc/jit/runtime/graph_executor_impl.h) | Source code | [docs](./graph_executor_impl.h_docs.md) | [keywords](./graph_executor_impl.h_kw.md) |
| [`graph_iterator.h`](../../../../../torch/csrc/jit/runtime/graph_iterator.h) | Source code | [docs](./graph_iterator.h_docs.md) | [keywords](./graph_iterator.h_kw.md) |
| [`instruction.cpp`](../../../../../torch/csrc/jit/runtime/instruction.cpp) | Source code | [docs](./instruction.cpp_docs.md) | [keywords](./instruction.cpp_kw.md) |
| [`instruction.h`](../../../../../torch/csrc/jit/runtime/instruction.h) | Source code | [docs](./instruction.h_docs.md) | [keywords](./instruction.h_kw.md) |
| [`interpreter.cpp`](../../../../../torch/csrc/jit/runtime/interpreter.cpp) | Source code | [docs](./interpreter.cpp_docs.md) | [keywords](./interpreter.cpp_kw.md) |
| [`interpreter.h`](../../../../../torch/csrc/jit/runtime/interpreter.h) | Source code | [docs](./interpreter.h_docs.md) | [keywords](./interpreter.h_kw.md) |
| [`jit_exception.cpp`](../../../../../torch/csrc/jit/runtime/jit_exception.cpp) | Source code | [docs](./jit_exception.cpp_docs.md) | [keywords](./jit_exception.cpp_kw.md) |
| [`jit_exception.h`](../../../../../torch/csrc/jit/runtime/jit_exception.h) | Source code | [docs](./jit_exception.h_docs.md) | [keywords](./jit_exception.h_kw.md) |
| [`jit_trace.cpp`](../../../../../torch/csrc/jit/runtime/jit_trace.cpp) | Source code | [docs](./jit_trace.cpp_docs.md) | [keywords](./jit_trace.cpp_kw.md) |
| [`jit_trace.h`](../../../../../torch/csrc/jit/runtime/jit_trace.h) | Source code | [docs](./jit_trace.h_docs.md) | [keywords](./jit_trace.h_kw.md) |
| [`logging.cpp`](../../../../../torch/csrc/jit/runtime/logging.cpp) | Source code | [docs](./logging.cpp_docs.md) | [keywords](./logging.cpp_kw.md) |
| [`logging.h`](../../../../../torch/csrc/jit/runtime/logging.h) | Source code | [docs](./logging.h_docs.md) | [keywords](./logging.h_kw.md) |
| [`operator.cpp`](../../../../../torch/csrc/jit/runtime/operator.cpp) | Source code | [docs](./operator.cpp_docs.md) | [keywords](./operator.cpp_kw.md) |
| [`operator.h`](../../../../../torch/csrc/jit/runtime/operator.h) | Source code | [docs](./operator.h_docs.md) | [keywords](./operator.h_kw.md) |
| [`operator_options.h`](../../../../../torch/csrc/jit/runtime/operator_options.h) | Source code | [docs](./operator_options.h_docs.md) | [keywords](./operator_options.h_kw.md) |
| [`print_handler.cpp`](../../../../../torch/csrc/jit/runtime/print_handler.cpp) | Source code | [docs](./print_handler.cpp_docs.md) | [keywords](./print_handler.cpp_kw.md) |
| [`print_handler.h`](../../../../../torch/csrc/jit/runtime/print_handler.h) | Source code | [docs](./print_handler.h_docs.md) | [keywords](./print_handler.h_kw.md) |
| [`profiling_graph_executor_impl.cpp`](../../../../../torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp) | Source code | [docs](./profiling_graph_executor_impl.cpp_docs.md) | [keywords](./profiling_graph_executor_impl.cpp_kw.md) |
| [`profiling_graph_executor_impl.h`](../../../../../torch/csrc/jit/runtime/profiling_graph_executor_impl.h) | Source code | [docs](./profiling_graph_executor_impl.h_docs.md) | [keywords](./profiling_graph_executor_impl.h_kw.md) |
| [`profiling_record.cpp`](../../../../../torch/csrc/jit/runtime/profiling_record.cpp) | Source code | [docs](./profiling_record.cpp_docs.md) | [keywords](./profiling_record.cpp_kw.md) |
| [`profiling_record.h`](../../../../../torch/csrc/jit/runtime/profiling_record.h) | Source code | [docs](./profiling_record.h_docs.md) | [keywords](./profiling_record.h_kw.md) |
| [`register_c10_ops.cpp`](../../../../../torch/csrc/jit/runtime/register_c10_ops.cpp) | Source code | [docs](./register_c10_ops.cpp_docs.md) | [keywords](./register_c10_ops.cpp_kw.md) |
| [`register_cuda_ops.cpp`](../../../../../torch/csrc/jit/runtime/register_cuda_ops.cpp) | Source code | [docs](./register_cuda_ops.cpp_docs.md) | [keywords](./register_cuda_ops.cpp_kw.md) |
| [`register_distributed_ops.cpp`](../../../../../torch/csrc/jit/runtime/register_distributed_ops.cpp) | Source code | [docs](./register_distributed_ops.cpp_docs.md) | [keywords](./register_distributed_ops.cpp_kw.md) |
| [`register_ops_utils.cpp`](../../../../../torch/csrc/jit/runtime/register_ops_utils.cpp) | Source code | [docs](./register_ops_utils.cpp_docs.md) | [keywords](./register_ops_utils.cpp_kw.md) |
| [`register_ops_utils.h`](../../../../../torch/csrc/jit/runtime/register_ops_utils.h) | Source code | [docs](./register_ops_utils.h_docs.md) | [keywords](./register_ops_utils.h_kw.md) |
| [`register_prim_ops.cpp`](../../../../../torch/csrc/jit/runtime/register_prim_ops.cpp) | Source code | [docs](./register_prim_ops.cpp_docs.md) | [keywords](./register_prim_ops.cpp_kw.md) |
| [`register_prim_ops_fulljit.cpp`](../../../../../torch/csrc/jit/runtime/register_prim_ops_fulljit.cpp) | Source code | [docs](./register_prim_ops_fulljit.cpp_docs.md) | [keywords](./register_prim_ops_fulljit.cpp_kw.md) |
| [`register_special_ops.cpp`](../../../../../torch/csrc/jit/runtime/register_special_ops.cpp) | Source code | [docs](./register_special_ops.cpp_docs.md) | [keywords](./register_special_ops.cpp_kw.md) |
| [`script_profile.cpp`](../../../../../torch/csrc/jit/runtime/script_profile.cpp) | Source code | [docs](./script_profile.cpp_docs.md) | [keywords](./script_profile.cpp_kw.md) |
| [`script_profile.h`](../../../../../torch/csrc/jit/runtime/script_profile.h) | Source code | [docs](./script_profile.h_docs.md) | [keywords](./script_profile.h_kw.md) |
| [`serialized_shape_function_registry.cpp`](../../../../../torch/csrc/jit/runtime/serialized_shape_function_registry.cpp) | Source code | [docs](./serialized_shape_function_registry.cpp_docs.md) | [keywords](./serialized_shape_function_registry.cpp_kw.md) |
| [`serialized_shape_function_registry.h`](../../../../../torch/csrc/jit/runtime/serialized_shape_function_registry.h) | Source code | [docs](./serialized_shape_function_registry.h_docs.md) | [keywords](./serialized_shape_function_registry.h_kw.md) |
| [`shape_function_registry.h`](../../../../../torch/csrc/jit/runtime/shape_function_registry.h) | Source code | [docs](./shape_function_registry.h_docs.md) | [keywords](./shape_function_registry.h_kw.md) |
| [`simple_graph_executor_impl.cpp`](../../../../../torch/csrc/jit/runtime/simple_graph_executor_impl.cpp) | Source code | [docs](./simple_graph_executor_impl.cpp_docs.md) | [keywords](./simple_graph_executor_impl.cpp_kw.md) |
| [`simple_graph_executor_impl.h`](../../../../../torch/csrc/jit/runtime/simple_graph_executor_impl.h) | Source code | [docs](./simple_graph_executor_impl.h_docs.md) | [keywords](./simple_graph_executor_impl.h_kw.md) |
| [`slice_indices_adjust.cpp`](../../../../../torch/csrc/jit/runtime/slice_indices_adjust.cpp) | Source code | [docs](./slice_indices_adjust.cpp_docs.md) | [keywords](./slice_indices_adjust.cpp_kw.md) |
| [`slice_indices_adjust.h`](../../../../../torch/csrc/jit/runtime/slice_indices_adjust.h) | Source code | [docs](./slice_indices_adjust.h_docs.md) | [keywords](./slice_indices_adjust.h_kw.md) |
| [`symbolic_script.cpp`](../../../../../torch/csrc/jit/runtime/symbolic_script.cpp) | Source code | [docs](./symbolic_script.cpp_docs.md) | [keywords](./symbolic_script.cpp_kw.md) |
| [`symbolic_script.h`](../../../../../torch/csrc/jit/runtime/symbolic_script.h) | Source code | [docs](./symbolic_script.h_docs.md) | [keywords](./symbolic_script.h_kw.md) |
| [`symbolic_shape_registry.cpp`](../../../../../torch/csrc/jit/runtime/symbolic_shape_registry.cpp) | Source code | [docs](./symbolic_shape_registry.cpp_docs.md) | [keywords](./symbolic_shape_registry.cpp_kw.md) |
| [`symbolic_shape_registry.h`](../../../../../torch/csrc/jit/runtime/symbolic_shape_registry.h) | Source code | [docs](./symbolic_shape_registry.h_docs.md) | [keywords](./symbolic_shape_registry.h_kw.md) |
| [`symbolic_shape_registry_util.cpp`](../../../../../torch/csrc/jit/runtime/symbolic_shape_registry_util.cpp) | Source code | [docs](./symbolic_shape_registry_util.cpp_docs.md) | [keywords](./symbolic_shape_registry_util.cpp_kw.md) |
| [`symbolic_shape_registry_util.h`](../../../../../torch/csrc/jit/runtime/symbolic_shape_registry_util.h) | Source code | [docs](./symbolic_shape_registry_util.h_docs.md) | [keywords](./symbolic_shape_registry_util.h_kw.md) |
| [`vararg_functions.cpp`](../../../../../torch/csrc/jit/runtime/vararg_functions.cpp) | Source code | [docs](./vararg_functions.cpp_docs.md) | [keywords](./vararg_functions.cpp_kw.md) |
| [`vararg_functions.h`](../../../../../torch/csrc/jit/runtime/vararg_functions.h) | Source code | [docs](./vararg_functions.h_docs.md) | [keywords](./vararg_functions.h_kw.md) |
| [`variable_tensor_list.h`](../../../../../torch/csrc/jit/runtime/variable_tensor_list.h) | Source code | [docs](./variable_tensor_list.h_docs.md) | [keywords](./variable_tensor_list.h_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
