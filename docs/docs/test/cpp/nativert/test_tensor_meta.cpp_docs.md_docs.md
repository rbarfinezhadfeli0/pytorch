# Documentation: `docs/test/cpp/nativert/test_tensor_meta.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_tensor_meta.cpp_docs.md`
- **Size**: 4,875 bytes (4.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_tensor_meta.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_tensor_meta.cpp`
- **Size**: 2,196 bytes (2.14 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <torch/nativert/graph/TensorMeta.h>

namespace torch::nativert {
TEST(TensorMetaTest, ScalarTypeConversion) {
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::FLOAT),
      c10::ScalarType::Float);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::INT),
      c10::ScalarType::Int);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::HALF),
      c10::ScalarType::Half);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::COMPLEXHALF),
      c10::ScalarType::ComplexHalf);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::BFLOAT16),
      c10::ScalarType::BFloat16);
  EXPECT_THROW(
      convertJsonScalarType(static_cast<torch::_export::ScalarType>(100)),
      c10::Error);
}
TEST(TensorMetaTest, MemoryFormatConversion) {
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::ContiguousFormat),
      c10::MemoryFormat::Contiguous);
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::ChannelsLast),
      c10::MemoryFormat::ChannelsLast);
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::PreserveFormat),
      c10::MemoryFormat::Preserve);
  EXPECT_THROW(
      convertJsonMemoryFormat(static_cast<torch::_export::MemoryFormat>(100)),
      c10::Error);
}

TEST(TensorMetaTest, LayoutConversion) {
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::Strided), c10::Layout::Strided);
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::SparseCsr),
      c10::Layout::SparseCsr);
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::_mkldnn), c10::Layout::Mkldnn);
  EXPECT_THROW(
      convertJsonLayout(static_cast<torch::_export::Layout>(100)), c10::Error);
}
TEST(TensorMetaTest, DeviceConversion) {
  torch::_export::Device cpu_device;
  cpu_device.set_type("cpu");
  EXPECT_EQ(convertJsonDevice(cpu_device), c10::Device(c10::DeviceType::CPU));
  torch::_export::Device cuda_device;
  cuda_device.set_type("cuda");
  cuda_device.set_index(0);
  EXPECT_EQ(
      convertJsonDevice(cuda_device), c10::Device(c10::DeviceType::CUDA, 0));
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/nativert/graph/TensorMeta.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/nativert/test_tensor_meta.cpp
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
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `test_tensor_meta.cpp_docs.md`
- **Keyword Index**: `test_tensor_meta.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/nativert`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/nativert/test_tensor_meta.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/nativert`):

- [`test_execution_frame.cpp_kw.md_docs.md`](./test_execution_frame.cpp_kw.md_docs.md)
- [`test_tensor_meta.cpp_kw.md_docs.md`](./test_tensor_meta.cpp_kw.md_docs.md)
- [`test_graph_signature.cpp_kw.md_docs.md`](./test_graph_signature.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_static_kernel_ops.cpp_kw.md_docs.md`](./test_static_kernel_ops.cpp_kw.md_docs.md)
- [`test_layout_planner_algorithm.cpp_docs.md_docs.md`](./test_layout_planner_algorithm.cpp_docs.md_docs.md)
- [`test_pass_manager.cpp_docs.md_docs.md`](./test_pass_manager.cpp_docs.md_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_kw.md_docs.md`](./test_static_dispatch_kernel_registration.cpp_kw.md_docs.md)
- [`test_placement.cpp_kw.md_docs.md`](./test_placement.cpp_kw.md_docs.md)
- [`test_static_kernel_ops.cpp_docs.md_docs.md`](./test_static_kernel_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_tensor_meta.cpp_docs.md_docs.md`
- **Keyword Index**: `test_tensor_meta.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
