# Documentation: `test/cpp/nativert/CMakeLists.txt`

## File Metadata

- **Path**: `test/cpp/nativert/CMakeLists.txt`
- **Size**: 4,240 bytes (4.14 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```
set(NATIVERT_TEST_ROOT ${TORCH_ROOT}/test/cpp/nativert)

file(GLOB_RECURSE NATIVERT_ALL_TEST_FILES "${NATIVERT_TEST_ROOT}/test_*.cpp")

# Build the cpp gtest binary containing the cpp-only tests.
set(NATIVERT_TEST_SRCS
  ${NATIVERT_ALL_TEST_FILES}
  ${TORCH_ROOT}/torch/nativert/ModelRunner.cpp
  ${TORCH_ROOT}/torch/nativert/graph/TensorMeta.cpp
  ${TORCH_ROOT}/torch/nativert/graph/Graph.cpp
  ${TORCH_ROOT}/torch/nativert/graph/GraphPasses.cpp
  ${TORCH_ROOT}/torch/nativert/graph/GraphSignature.cpp
  ${TORCH_ROOT}/torch/nativert/graph/GraphUtils.cpp
  ${TORCH_ROOT}/torch/nativert/graph/Serialization.cpp
  ${TORCH_ROOT}/torch/nativert/executor/OpKernel.cpp
  ${TORCH_ROOT}/torch/nativert/executor/PlacementUtils.cpp
  ${TORCH_ROOT}/torch/nativert/executor/Weights.cpp
  ${TORCH_ROOT}/torch/nativert/common/FileUtil.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/FunctionSchema.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ExecutionPlanner.cpp
  ${TORCH_ROOT}/torch/nativert/detail/ITree.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ExecutionFrame.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/C10Kernel.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/GreedyBySize.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/Bump.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/DisjointStorageGroups.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/LayoutPlanner.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/LayoutManager.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/AliasAnalyzer.cpp
  ${TORCH_ROOT}/torch/nativert/executor/Executor.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/KernelFactory.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ConstantFolder.cpp
  ${TORCH_ROOT}/torch/nativert/executor/GraphExecutorBase.cpp
  ${TORCH_ROOT}/torch/nativert/executor/SerialGraphExecutor.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ParallelGraphExecutor.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/AutoFunctionalizeKernel.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/CallTorchBindKernel.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/HigherOrderKernel.cpp
  ${TORCH_ROOT}/torch/nativert/graph/passes/SubgraphRewriter.cpp
  ${TORCH_ROOT}/torch/nativert/graph/passes/pass_manager/GraphPasses.cpp
  ${TORCH_ROOT}/torch/nativert/graph/passes/pass_manager/PassManager.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/KernelHandlerRegistry.cpp
  ${TORCH_ROOT}/torch/nativert/executor/triton/CpuTritonKernelManager.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/TritonKernel.cpp
  ${TORCH_ROOT}/torch/nativert/executor/DelegateExecutor.cpp
  ${TORCH_ROOT}/torch/nativert/executor/AOTInductorDelegateExecutor.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/ETCallDelegateKernel.cpp
  ${TORCH_ROOT}/torch/csrc/inductor/aoti_torch/oss_proxy_executor.cpp
)

if(USE_CUDA OR USE_ROCM)
  list(APPEND NATIVERT_TEST_SRCS ${TORCH_ROOT}/torch/nativert/executor/triton/CudaTritonKernelManager.cpp)
  list(APPEND NATIVERT_TEST_SRCS ${TORCH_ROOT}/torch/nativert/executor/AOTInductorModelContainerCudaShim.cpp)
endif()

add_executable(test_nativert
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${NATIVERT_TEST_SRCS}
)

if(MSVC)
  target_compile_definitions(test_nativert PRIVATE NATIVERT_MSVC_TEST)
endif()

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_nativert PRIVATE USE_GTEST)

set(NATIVERT_TEST_DEPENDENCIES torch gtest_main)

target_link_libraries(test_nativert PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(test_nativert PRIVATE ${NATIVERT_TEST_DEPENDENCIES})
target_link_libraries(test_nativert PRIVATE fmt::fmt-header-only)
target_include_directories(test_nativert PRIVATE ${ATen_CPU_INCLUDE})

if(USE_CUDA)
  target_compile_definitions(test_nativert PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_nativert PRIVATE
    hiprtc::hiprtc
    hip::amdhip64
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_nativert PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  set_target_properties(test_nativert PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_nativert DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_nativert> DESTINATION bin OPTIONAL)
  endif()
endif()

```



## High-Level Overview

This file is part of the PyTorch framework located at `test/cpp/nativert`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/nativert/CMakeLists.txt
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/nativert`):

- [`test_alias_analyzer.cpp_docs.md`](./test_alias_analyzer.cpp_docs.md)
- [`test_placement.cpp_docs.md`](./test_placement.cpp_docs.md)
- [`test_static_kernel_ops.cpp_docs.md`](./test_static_kernel_ops.cpp_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_docs.md`](./test_static_dispatch_kernel_registration.cpp_docs.md)
- [`test_graph.cpp_docs.md`](./test_graph.cpp_docs.md)
- [`test_c10_kernel.cpp_docs.md`](./test_c10_kernel.cpp_docs.md)
- [`test_function_schema.cpp_docs.md`](./test_function_schema.cpp_docs.md)
- [`test_execution_frame.cpp_docs.md`](./test_execution_frame.cpp_docs.md)
- [`test_triton_kernel_manager_registration.cpp_docs.md`](./test_triton_kernel_manager_registration.cpp_docs.md)


## Cross-References

- **File Documentation**: `CMakeLists.txt_docs.md`
- **Keyword Index**: `CMakeLists.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
