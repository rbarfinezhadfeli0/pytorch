# Documentation: `docs/test/cpp/nativert/test_placement.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_placement.cpp_docs.md`
- **Size**: 5,903 bytes (5.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_placement.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_placement.cpp`
- **Size**: 3,255 bytes (3.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp

#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <unordered_map>

#include <torch/nativert/executor/Placement.h>

using namespace ::testing;

namespace torch::nativert {

TEST(PlacementTest, IsSameDevice) {
  c10::Device cpuDevice = c10::Device(c10::DeviceType::CPU);
  c10::Device cpuDevice1 = c10::Device(c10::DeviceType::CPU);
  cpuDevice1.set_index(1);

  EXPECT_TRUE(isSameDevice(cpuDevice, cpuDevice));
  EXPECT_TRUE(isSameDevice(cpuDevice, cpuDevice1));

  c10::Device cudaDevice = c10::Device(c10::DeviceType::CUDA);
  c10::Device cudaDevice0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cudaDevice1 = c10::Device(c10::DeviceType::CUDA, 1);
  EXPECT_TRUE(isSameDevice(cudaDevice, cudaDevice0));
  EXPECT_FALSE(isSameDevice(cudaDevice0, cudaDevice1));

  EXPECT_FALSE(isSameDevice(cudaDevice0, cpuDevice));
}

TEST(PlacementTest, PlacementDefaultOnly) {
  Placement placement(c10::Device(c10::DeviceType::CUDA, 0));

  std::ostringstream os;
  os << placement;
  EXPECT_EQ(os.str(), "|cuda:0");

  c10::Device cuda0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cuda1 = c10::Device(c10::DeviceType::CUDA, 1);
  c10::Device cuda2 = c10::Device(c10::DeviceType::CUDA, 2);

  EXPECT_EQ(placement.getMappedDevice(cuda0), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda1), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda2), cuda0);
}

TEST(PlacementTest, PlacementBasic) {
  Placement placement(
      {{c10::Device(c10::DeviceType::CPU), c10::Device(c10::DeviceType::CPU)},
       {c10::Device(c10::DeviceType::CUDA, 0),
        c10::Device(c10::DeviceType::CUDA, 1)},
       {c10::Device(c10::DeviceType::CUDA, 1),
        c10::Device(c10::DeviceType::CUDA, 2)}},
      c10::Device(c10::DeviceType::CUDA, 0));

  std::ostringstream os;
  os << placement;
  EXPECT_EQ(os.str(), "cpu|cpu,cuda:0|cuda:1,cuda:1|cuda:2,|cuda:0");

  c10::Device cpu = c10::Device(c10::DeviceType::CPU);
  c10::Device cuda0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cuda1 = c10::Device(c10::DeviceType::CUDA, 1);
  c10::Device cuda2 = c10::Device(c10::DeviceType::CUDA, 2);
  c10::Device cuda3 = c10::Device(c10::DeviceType::CUDA, 3);

  EXPECT_EQ(placement.getMappedDevice(cpu), cpu);
  EXPECT_EQ(placement.getMappedDevice(cuda0), cuda1);
  EXPECT_EQ(placement.getMappedDevice(cuda1), cuda2);
  EXPECT_EQ(placement.getMappedDevice(cuda2), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda3), cuda0);
}

TEST(PlacementTest, Placement) {
  std::unordered_map<c10::Device, c10::Device> deviceMap1 = {
      {c10::Device("cuda:0"), c10::Device("cuda:1")}};
  Placement p1(deviceMap1);
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cpu")), c10::Device("cpu"));
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cuda")), c10::Device("cuda"));
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cuda:0")), c10::Device("cuda:1"));

  std::unordered_map<c10::Device, c10::Device> deviceMap2 = {
      {c10::Device("cpu"), c10::Device("cuda:0")}};
  Placement p2(deviceMap2);
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cpu")), c10::Device("cuda:0"));
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cuda:0")), c10::Device("cuda:0"));
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cuda:1")), c10::Device("cuda:1"));
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `c10/core/Device.h`
- `gtest/gtest.h`
- `unordered_map`
- `torch/nativert/executor/Placement.h`


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
python test/cpp/nativert/test_placement.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/nativert`):

- [`test_alias_analyzer.cpp_docs.md`](./test_alias_analyzer.cpp_docs.md)
- [`test_static_kernel_ops.cpp_docs.md`](./test_static_kernel_ops.cpp_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_docs.md`](./test_static_dispatch_kernel_registration.cpp_docs.md)
- [`test_graph.cpp_docs.md`](./test_graph.cpp_docs.md)
- [`test_c10_kernel.cpp_docs.md`](./test_c10_kernel.cpp_docs.md)
- [`test_function_schema.cpp_docs.md`](./test_function_schema.cpp_docs.md)
- [`test_execution_frame.cpp_docs.md`](./test_execution_frame.cpp_docs.md)
- [`test_triton_kernel_manager_registration.cpp_docs.md`](./test_triton_kernel_manager_registration.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `test_placement.cpp_docs.md`
- **Keyword Index**: `test_placement.cpp_kw.md`
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
python docs/test/cpp/nativert/test_placement.cpp_docs.md
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

- **File Documentation**: `test_placement.cpp_docs.md_docs.md`
- **Keyword Index**: `test_placement.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
