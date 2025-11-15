# Documentation: `docs/test/cpp/nativert/test_execution_frame.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_execution_frame.cpp_docs.md`
- **Size**: 5,399 bytes (5.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_execution_frame.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_execution_frame.cpp`
- **Size**: 2,818 bytes (2.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ops/tensor.h>
#include <torch/nativert/executor/ExecutionFrame.h>

namespace torch::nativert {

TEST(ExecutionFrameTest, CreateFrame) {
  auto graph = stringToGraph(R"(
    graph(%x, %y):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  auto frame = ExecutionFrame(*graph);

  for (auto* v : graph->values()) {
    frame.setIValue(v->id(), c10::IValue(at::tensor({v->id()}, at::kInt)));
    auto& frame_v = frame.getIValue(v->id());
    EXPECT_EQ(frame_v.tagKind(), "Tensor");
  }

  auto outputs = frame.tryMoveUserOutputs();

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].tagKind(), "Tensor");
  EXPECT_EQ(outputs[0].toTensor().item().toInt(), graph->getValue("c")->id());
}

TEST(ExecutionFrameTest, TestSetBorrowedValue) {
  auto graph = stringToGraph(R"(
    graph(%x, %y):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  auto x = c10::IValue(at::tensor({1}, at::kInt));
  auto y = c10::IValue(at::tensor({2}, at::kInt));

  {
    auto frame = ExecutionFrame(*graph);

    frame.setBorrowedIValue(
        graph->getValue("x")->id(),
        c10::MaybeOwnedTraits<c10::IValue>::createBorrow(x));
    frame.setBorrowedIValue(
        graph->getValue("y")->id(),
        c10::MaybeOwnedTraits<c10::IValue>::createBorrow(y));

    [[maybe_unused]] auto& w = frame.getIValue(graph->getValue("x")->id());
    [[maybe_unused]] auto& z = frame.getIValue(graph->getValue("y")->id());

    EXPECT_EQ(x.use_count(), 1);
    EXPECT_EQ(y.use_count(), 1);

    EXPECT_TRUE(c10::MaybeOwnedTraits<c10::IValue>{}.debugBorrowIsValid(
        frame.getIValue(graph->getValue("x")->id())));
    EXPECT_TRUE(c10::MaybeOwnedTraits<c10::IValue>{}.debugBorrowIsValid(
        frame.getIValue(graph->getValue("y")->id())));
  }

  EXPECT_EQ(x.use_count(), 1);
  EXPECT_EQ(y.use_count(), 1);
}

TEST(ExecutionFrameTest, TestPersistentValue) {
  auto graph = stringToGraph(R"(
    graph(%x, %y, %my_weight):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  Weights weights(graph.get());
  weights.setValue("my_weight", at::tensor({1}, at::kInt));

  auto new_sig = graph->signature();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const_cast<std::vector<std::pair<std::string, std::string>>&>(
      new_sig.inputsToWeights())
      .emplace_back("my_weight", "my_weight");
  graph->setSignature(new_sig);

  auto frame = ExecutionFrame(*graph, weights);

  EXPECT_EQ(frame.weightVersion(), 0);
  auto wid = graph->getValue("my_weight")->id();

  EXPECT_NO_THROW(frame.getTensor(wid));
  // can't release persistent value
  frame.releaseValueIfNeeded(wid);
  EXPECT_FALSE(frame.getIValue(wid).isNone());
}

} // namespace torch::nativert

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

- `gtest/gtest.h`
- `ATen/ops/tensor.h`
- `torch/nativert/executor/ExecutionFrame.h`


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
python test/cpp/nativert/test_execution_frame.cpp
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
- [`test_triton_kernel_manager_registration.cpp_docs.md`](./test_triton_kernel_manager_registration.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `test_execution_frame.cpp_docs.md`
- **Keyword Index**: `test_execution_frame.cpp_kw.md`
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
python docs/test/cpp/nativert/test_execution_frame.cpp_docs.md
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

- **File Documentation**: `test_execution_frame.cpp_docs.md_docs.md`
- **Keyword Index**: `test_execution_frame.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
