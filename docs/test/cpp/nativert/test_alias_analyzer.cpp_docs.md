# Documentation: `test/cpp/nativert/test_alias_analyzer.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_alias_analyzer.cpp`
- **Size**: 5,714 bytes (5.58 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <fmt/format.h>

#include <torch/nativert/executor/memory/AliasAnalyzer.h>
#include <torch/nativert/graph/Graph.h>

#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/kernels/KernelFactory.h>

using namespace ::testing;
using namespace torch::nativert;

using AliasTestCase = std::tuple<
    std::string /* value */,
    AllocationLifetime,
    bool /* is_alias */,
    bool /* is_storage_associated_with_output */,
    c10::FastSet<std::string> /* source(s) */>;

class AliasAnalyzerTests : public testing::Test {
  void SetUp() override {}

  void TearDown() override {
    test_cases.clear();
    model.clear();
  }

 public:
  void setTestCases(std::vector<AliasTestCase> cases) {
    test_cases = std::move(cases);
  }

  void setModel(std::string m) {
    model = std::move(m);
  }

  void run() {
    EXPECT_FALSE(test_cases.empty());
    EXPECT_FALSE(model.empty());

    ExecutorConfig cfg;
    cfg.enableStaticCPUKernels = true;

    auto graph = stringToGraph(model);
    auto kernels =
        KernelFactory().initializeNodeKernels(*graph, nullptr, cfg, nullptr);
    auto kernelSchemas = Executor::getKernelSchemas(kernels.nodeKernels);

    AliasAnalyzer analyzer(*graph, kernelSchemas);

    for (
        auto& [value, lifetime, is_alias, is_storage_associated_with_output, srcs] :
        test_cases) {
      LOG(INFO) << fmt::format(
          "running test: value={}, lifetime=({}, {}), is_alias={}, is_storage_associated_with_output={}, src={}",
          value,
          lifetime.start,
          lifetime.end,
          is_alias,
          is_storage_associated_with_output,
          srcs.empty() ? "{}"
                       : std::accumulate(
                             srcs.begin(),
                             srcs.end(),
                             std::string{},
                             [](std::string cur, const std::string& src) {
                               cur.append(",");
                               cur.append(src);
                               return cur;
                             }));
      auto* v = graph->getValue(value);
      EXPECT_EQ(analyzer.lifetime(v), lifetime);
      EXPECT_EQ(analyzer.is_alias(v), is_alias);
      EXPECT_EQ(
          analyzer.is_storage_associated_with_output(v),
          is_storage_associated_with_output);
      const auto* resolved_srcs = analyzer.get_sources_of_alias(v);
      if (resolved_srcs /* ensure set equality between *resolved_srcs and srcs */) {
        EXPECT_FALSE(srcs.empty());
        EXPECT_EQ(resolved_srcs->size(), srcs.size());
        for (const auto& resolved_src : *resolved_srcs) {
          EXPECT_TRUE(srcs.erase(std::string(resolved_src->name())) == 1);
        }
        EXPECT_TRUE(srcs.empty());
      } else {
        EXPECT_TRUE(srcs.empty());
      }
    }
  }

 private:
  std::string model;
  std::vector<AliasTestCase> test_cases;
};

TEST_F(AliasAnalyzerTests, TestNoAlias) {
  setModel(R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.clone.default(self=%out_t, memory_format=None)
  return (%res))");

  setTestCases({
      {"out_t", AllocationLifetime(1, 2), false, false, {}},
      {"res", AllocationLifetime(2, 3), false, true, {}},
  });

  run();
}

TEST_F(AliasAnalyzerTests, TestSimpleAlias) {
  setModel(R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %res = torch.ops.aten.slice.Tensor(self=%out_t, dim=1, start=0, end=0, step=1)
  return (%res))");

  setTestCases({
      {"out_t", AllocationLifetime(1, 3), false, true, {}},
      {"res", AllocationLifetime(2, 3), true, false, {"out_t"}},
  });

  run();
}

TEST_F(AliasAnalyzerTests, TestDeepAlias) {
  setModel(R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %a1 = torch.ops.aten.slice.Tensor(self=%out_t, dim=1, start=0, end=0, step=1)
      %res = torch.ops.aten.slice.Tensor(self=%a1, dim=1, start=0, end=0, step=1)
  return (%res))");

  setTestCases({
      {"out_t", AllocationLifetime(1, 4), false, true, {}},
      {"a1", AllocationLifetime(2, 4), true, false, {"out_t"}},
      {"res", AllocationLifetime(3, 4), true, false, {"out_t"}},
  });

  run();
}

TEST_F(AliasAnalyzerTests, TestPackedListUnpack) {
  setModel(R"(
    graph(%a, %b, %c, %d):
  %input_list[] = prim.ListPack(l0=%a, l1=%b, l2=%c, l3=%d)
  %x0, %x1, %x2, %x3 = prim.ListUnpack(input=%input_list)
  return (%x1, %x3))");

  setTestCases({
      {"a", AllocationLifetime(0, 2), false, false, {}},
      {"x0", AllocationLifetime(2, 2), true, false, {"a"}},
      {"b", AllocationLifetime(0, 3), false, true, {}},
      {"x1", AllocationLifetime(2, 3), true, false, {"b"}},
      {"c", AllocationLifetime(0, 2), false, false, {}},
      {"x2", AllocationLifetime(2, 2), true, false, {"c"}},
      {"d", AllocationLifetime(0, 3), false, true, {}},
      {"x3", AllocationLifetime(2, 3), true, false, {"d"}},
  });

  run();
}

TEST_F(AliasAnalyzerTests, TestAmbiguousSourceOfAlias) {
  setModel(R"(
    graph(%y0, %y1):
      %out_t = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %out_t2 = torch.ops.aten.matmul.default(self=%y0, other=%y1)
      %a1 = prim.VarStack(l0=%out_t, l1=%out_t2)
      %res = torch.ops.aten.slice.Tensor(self=%a1, dim=1, start=0, end=0, step=1)
  return (%res))");

  setTestCases({
      {"out_t", AllocationLifetime(1, 5), false, true, {}},
      {"out_t2", AllocationLifetime(2, 5), false, true, {}},
      {"a1", AllocationLifetime(3, 5), true, false, {"out_t", "out_t2"}},
      {"res", AllocationLifetime(4, 5), true, false, {"out_t", "out_t2"}},
  });

  run();
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `AliasAnalyzerTests`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `fmt/format.h`
- `torch/nativert/executor/memory/AliasAnalyzer.h`
- `torch/nativert/graph/Graph.h`
- `torch/nativert/executor/Executor.h`
- `torch/nativert/kernels/KernelFactory.h`


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
python test/cpp/nativert/test_alias_analyzer.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/nativert`):

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

- **File Documentation**: `test_alias_analyzer.cpp_docs.md`
- **Keyword Index**: `test_alias_analyzer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
