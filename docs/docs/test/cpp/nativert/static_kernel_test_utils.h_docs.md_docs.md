# Documentation: `docs/test/cpp/nativert/static_kernel_test_utils.h_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/static_kernel_test_utils.h_docs.md`
- **Size**: 7,951 bytes (7.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/static_kernel_test_utils.h`

## File Metadata

- **Path**: `test/cpp/nativert/static_kernel_test_utils.h`
- **Size**: 5,050 bytes (4.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```c
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/torch.h>

#include <torch/nativert/kernels/KernelHandlerRegistry.h>

namespace torch::nativert {

/*
 * This is a lightweight version of ModelRunner that executes a model in
 * interpreter mode given a string graph with no weights/attributes
 */
class SimpleTestModelRunner {
 public:
  SimpleTestModelRunner(
      const std::string_view source,
      const ExecutorConfig& config) {
    register_kernel_handlers();
    graph_ = stringToGraph(source);
    weights_ = std::make_shared<Weights>(graph_.get());

    executor_ = std::make_unique<Executor>(config, graph_, weights_);
  }

  std::vector<c10::IValue> run(const std::vector<c10::IValue>& inputs) const {
    return executor_->execute(inputs);
  }

  ProfileMetrics benchmarkIndividualNodes(
      const std::vector<c10::IValue>& inputs) const {
    return executor_->benchmarkIndividualNodes({inputs}, 10, 10);
  }

 private:
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<Executor> executor_;
  std::shared_ptr<Weights> weights_;
};

inline void compareIValue(
    const c10::IValue& expected,
    const c10::IValue& actual,
    bool native = false) {
  if (expected.isTensor()) {
    EXPECT_TRUE(actual.isTensor());
    EXPECT_TRUE(torch::allclose(
        expected.toTensor(),
        actual.toTensor(),
        1e-5,
        1e-8,
        /*equal_nan*/ true));
    if (!native) {
      EXPECT_TRUE(expected.toTensor().strides() == actual.toTensor().strides());
    }
  } else if (expected.isTuple()) {
    EXPECT_TRUE(actual.isTuple());
    auto expected_tuple = expected.toTupleRef().elements();
    auto actual_tuple = actual.toTupleRef().elements();
    ASSERT_TRUE(expected_tuple.size() == actual_tuple.size());
    for (size_t i = 0; i < expected_tuple.size(); i++) {
      compareIValue(expected_tuple[i], actual_tuple[i], native);
    }
  } else if (expected.isList()) {
    EXPECT_TRUE(actual.isList());
    auto expected_list = expected.toList();
    auto actual_list = actual.toList();
    ASSERT_TRUE(expected_list.size() == actual_list.size());
    for (size_t i = 0; i < expected_list.size(); i++) {
      compareIValue(expected_list[i], actual_list[i], native);
    }
  } else if (expected.isGenericDict()) {
    EXPECT_TRUE(actual.isGenericDict());
    auto expected_dict = expected.toGenericDict();
    auto actual_dict = actual.toGenericDict();
    EXPECT_TRUE(expected_dict.size() == actual_dict.size());
    for (auto& expected_kv : expected_dict) {
      auto actual_kv = actual_dict.find(expected_kv.key());
      ASSERT_FALSE(actual_kv == actual_dict.end());
      compareIValue(expected_kv.value(), actual_kv->value(), native);
    }
  } else {
    // Fall back to default comparison from IValue
    EXPECT_TRUE(expected == actual);
  }
}

void compareIValues(
    std::vector<c10::IValue> expected,
    std::vector<c10::IValue> actual,
    bool native = false) {
  ASSERT_TRUE(expected.size() == actual.size());
  for (size_t i = 0; i < expected.size(); i++) {
    compareIValue(expected[i], actual[i], native);
  }
}

inline void testStaticKernelEqualityInternal(
    const SimpleTestModelRunner& modelRunner,
    const SimpleTestModelRunner& staticModelRunner,
    const std::vector<c10::IValue>& args,
    bool native = false) {
  auto expected = modelRunner.run(args);

  auto output = staticModelRunner.run(args);
  compareIValues(expected, output, native);

  // Run again to test the static kernel when outputs IValue are cached in the
  // execution frame
  auto output2 = staticModelRunner.run(args);
  compareIValues(expected, output2, native);
}

void testStaticKernelEquality(
    const std::string_view source,
    const std::vector<c10::IValue>& args,
    bool native = false) {
  ExecutorConfig config;
  config.enableStaticCPUKernels = false;
  SimpleTestModelRunner model(source, config);

  config.enableStaticCPUKernels = true;
  SimpleTestModelRunner staticKernelModel(source, config);

  testStaticKernelEqualityInternal(model, staticKernelModel, args, native);
}

inline void testGraphABEquality(
    const std::string_view graph_a,
    const std::string_view graph_b,
    const std::vector<c10::IValue>& args,
    const ExecutorConfig& config = {},
    bool native = false) {
  SimpleTestModelRunner model_a(graph_a, config);
  auto expected = model_a.run(args);

  SimpleTestModelRunner model_b(graph_b, config);
  auto output = model_b.run(args);

  compareIValues(expected, output, native);
}

inline void testGraphABPerf(
    const std::string_view graph_a,
    const std::string_view graph_b,
    const std::vector<c10::IValue>& args,
    const ExecutorConfig& config = {}) {
  SimpleTestModelRunner model_a(graph_a, config);
  auto resultA = model_a.benchmarkIndividualNodes(args);

  SimpleTestModelRunner model_b(graph_b, config);
  auto resultB = model_b.benchmarkIndividualNodes(args);
  ASSERT_TRUE(resultA.totalTime > resultB.totalTime);
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SimpleTestModelRunner`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `torch/nativert/executor/Executor.h`
- `torch/nativert/graph/Graph.h`
- `torch/torch.h`
- `torch/nativert/kernels/KernelHandlerRegistry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python test/cpp/nativert/static_kernel_test_utils.h
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

- **File Documentation**: `static_kernel_test_utils.h_docs.md`
- **Keyword Index**: `static_kernel_test_utils.h_kw.md`
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

- Implements or uses **caching** mechanisms.
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
python docs/test/cpp/nativert/static_kernel_test_utils.h_docs.md
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

- **File Documentation**: `static_kernel_test_utils.h_docs.md_docs.md`
- **Keyword Index**: `static_kernel_test_utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
