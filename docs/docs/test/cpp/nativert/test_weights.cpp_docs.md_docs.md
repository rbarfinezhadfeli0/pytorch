# Documentation: `docs/test/cpp/nativert/test_weights.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_weights.cpp_docs.md`
- **Size**: 5,936 bytes (5.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_weights.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_weights.cpp`
- **Size**: 3,054 bytes (2.98 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/custom_class.h>
#include <torch/torch.h>
#include <memory>

#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {
class WeightsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static constexpr std::string_view source =
        R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
    graph = stringToGraph(source);
    placement = std::make_unique<Placement>(c10::Device(c10::DeviceType::CPU));
  }
  std::shared_ptr<Graph> graph;
  std::unique_ptr<Placement> placement;
};
TEST_F(WeightsTest, ConstructEmptyStateDict) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  // Check that weights are initialized correctly
  EXPECT_TRUE(weights.parameters().empty());
  EXPECT_TRUE(weights.buffers().empty());
  EXPECT_FALSE(weights.contains("non_existent_weight"));
}
TEST_F(WeightsTest, SetAndGetValue) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  at::Tensor tensor = at::ones({2, 2});
  weights.setValue("added_weight", tensor);
  EXPECT_TRUE(weights.contains("added_weight"));
  EXPECT_EQ(weights.at("added_weight").sizes(), tensor.sizes());
}

} // namespace torch::nativert

using namespace ::testing;
struct ContainsTensorDict : torch::CustomClassHolder {
  explicit ContainsTensorDict(at::Tensor t) : t_(t) {}

  explicit ContainsTensorDict(c10::Dict<std::string, at::Tensor> dict) {
    t_ = dict.at(std::string("init_tensor"));
  }

  c10::Dict<std::string, at::Tensor> serialize() const {
    c10::Dict<std::string, at::Tensor> dict;
    dict.insert(std::string("init_tensor"), t_);
    return dict;
  }

  at::Tensor t_;
};

static auto reg =
    torch::class_<ContainsTensorDict>("testing", "ContainsTensorDict")
        .def(torch::init<at::Tensor>())
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<ContainsTensorDict>& self)
                -> c10::Dict<std::string, at::Tensor> {
              return self->serialize();
            },
            // __setstate__
            [](c10::Dict<std::string, at::Tensor> data)
                -> c10::intrusive_ptr<ContainsTensorDict> {
              return c10::make_intrusive<ContainsTensorDict>(std::move(data));
            });

TEST(CustomWeightsTest, TestCustomObjWithContainedTensor) {
  // Save
  auto customObj =
      c10::make_intrusive<ContainsTensorDict>(torch::tensor({1, 2, 3}));
  const auto bytes = torch::jit::pickle_save(c10::IValue(std::move(customObj)));

  // Load
  const auto loadedCustomObj =
      torch::jit::pickle_load_obj(std::string{bytes.begin(), bytes.end()});
  EXPECT_TRUE(loadedCustomObj.isObject());
  EXPECT_EQ(
      loadedCustomObj.to<c10::intrusive_ptr<ContainsTensorDict>>()
          ->t_[0]
          .item<int>(),
      1);
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `WeightsTest`, `ContainsTensorDict`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/csrc/jit/serialization/pickle.h`
- `torch/custom_class.h`
- `torch/torch.h`
- `memory`
- `torch/nativert/executor/Placement.h`
- `torch/nativert/executor/Weights.h`
- `torch/nativert/graph/Graph.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/nativert/test_weights.cpp
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

- **File Documentation**: `test_weights.cpp_docs.md`
- **Keyword Index**: `test_weights.cpp_kw.md`
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

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/nativert/test_weights.cpp_docs.md
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

- **File Documentation**: `test_weights.cpp_docs.md_docs.md`
- **Keyword Index**: `test_weights.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
