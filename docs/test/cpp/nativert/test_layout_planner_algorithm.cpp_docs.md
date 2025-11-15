# Documentation: `test/cpp/nativert/test_layout_planner_algorithm.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_layout_planner_algorithm.cpp`
- **Size**: 2,483 bytes (2.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/Enumerate.h>
#include <gtest/gtest.h>

#include <torch/nativert/executor/memory/Bump.h>
#include <torch/nativert/executor/memory/DisjointStorageGroups.h>
#include <torch/nativert/executor/memory/GreedyBySize.h>

using namespace ::testing;
using namespace torch::nativert;

std::vector<AllocationSpec> create_test_allocation_specs() {
  std::vector<AllocationSpec> specs;

  const std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
      {0, 1, 32},
      {1, 4, 28},
      {2, 5, 36},
      {3, 5, 16},
      {4, 5, 8},
      {5, 7, 64},
      {6, 8, 10},
      {7, 8, 40},
  };

  specs.reserve(test_cases.size());
  for (const auto& [l_start, l_end, size] : test_cases) {
    specs.push_back(AllocationSpec{AllocationLifetime(l_start, l_end), size});
  };

  return specs;
}

// figure 6 -- https://arxiv.org/pdf/2001.03288
TEST(LayoutPlannerAlgorithmTests, TestGreedyBySize) {
  auto result = GreedyBySizeAllocationPlanner(create_test_allocation_specs());

  EXPECT_EQ(result.total_size, 124);

  auto& allocations = result.allocations;

  EXPECT_EQ(allocations[0].offset, 0);
  EXPECT_EQ(allocations[1].offset, 32);
  EXPECT_EQ(allocations[2].offset, 64);
  EXPECT_EQ(allocations[3].offset, 100);
  EXPECT_EQ(allocations[4].offset, 116);
  EXPECT_EQ(allocations[5].offset, 0);
  EXPECT_EQ(allocations[6].offset, 104);
  EXPECT_EQ(allocations[7].offset, 64);
}

TEST(LayoutPlannerAlgorithmTests, TestBump) {
  auto specs = create_test_allocation_specs();
  auto result = BumpAllocationPlanner(create_test_allocation_specs());

  auto& allocations = result.allocations;

  size_t offset = 0;
  for (auto&& [i, spec] : c10::enumerate(specs)) {
    EXPECT_EQ(allocations[i].offset, offset);
    offset += spec.size;
  }

  EXPECT_EQ(result.total_size, offset);
}

TEST(LayoutPlannerAlgorithmTests, TestStorageGroup) {
  auto specs = create_test_allocation_specs();
  auto result = DisjointStorageGroupsPlanner(create_test_allocation_specs());

  auto& allocations = result.allocations;

  EXPECT_EQ(allocations[0].offset, 0);
  EXPECT_EQ(allocations[1].offset, 36);
  EXPECT_EQ(allocations[2].offset, 0);
  EXPECT_EQ(allocations[3].offset, 100);
  EXPECT_EQ(allocations[4].offset, 140);
  EXPECT_EQ(allocations[5].offset, 36);
  EXPECT_EQ(allocations[6].offset, 140);
  EXPECT_EQ(allocations[7].offset, 100);

  for (auto&& [i, spec] : c10::enumerate(specs)) {
    EXPECT_EQ(allocations[i].size, spec.size);
  }

  EXPECT_EQ(result.total_size, 150);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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

- `c10/util/Enumerate.h`
- `gtest/gtest.h`
- `torch/nativert/executor/memory/Bump.h`
- `torch/nativert/executor/memory/DisjointStorageGroups.h`
- `torch/nativert/executor/memory/GreedyBySize.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/nativert/test_layout_planner_algorithm.cpp
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

- **File Documentation**: `test_layout_planner_algorithm.cpp_docs.md`
- **Keyword Index**: `test_layout_planner_algorithm.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
