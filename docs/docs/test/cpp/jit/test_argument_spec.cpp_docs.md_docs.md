# Documentation: `docs/test/cpp/jit/test_argument_spec.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_argument_spec.cpp_docs.md`
- **Size**: 9,358 bytes (9.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_argument_spec.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_argument_spec.cpp`
- **Size**: 6,566 bytes (6.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/jit.h>

#include "test/cpp/jit/test_utils.h"

namespace torch {
namespace jit {

namespace {

at::Device device(const autograd::Variable& v) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return v.device();
}

bool isEqual(at::IntArrayRef lhs, at::IntArrayRef rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const CompleteArgumentInfo& ti, const autograd::Variable& v) {
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && isEqual(ti.sizes(), v.sizes()) &&
      isEqual(ti.strides(), v.strides());
}

bool isEqual(const ArgumentInfo& ti, const autograd::Variable& v) {
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && ti.dim() == v.dim();
}

autograd::Variable var(
    at::TensorOptions t,
    at::IntArrayRef sizes,
    bool requires_grad) {
  return autograd::make_variable(at::rand(sizes, t), requires_grad);
}
autograd::Variable undef() {
  return autograd::Variable();
}
} // namespace

TEST(ArgumentSpecTest, CompleteArgumentSpec_CUDA) {
  auto const CF = at::CPU(at::kFloat);
  auto const CD = at::CPU(at::kDouble);
  auto const GF = at::CUDA(at::kFloat);
  auto const GD = at::CUDA(at::kDouble);

  auto list = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});
  list2[1].toTensor().transpose_(0, 1);

  CompleteArgumentSpec a(true, list);
  CompleteArgumentSpec b(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  CompleteArgumentSpec d(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    ASSERT_TRUE(isEqual(a.at(i), list[i].toTensor()));
  }
  CompleteArgumentSpec no_grad(/*with_grad=*/false, list);
  ASSERT_TRUE(no_grad != a);

  std::unordered_set<CompleteArgumentSpec> spec;
  spec.insert(a); // we use a below, so no move
  ASSERT_TRUE(spec.count(b) > 0);
  ASSERT_EQ(spec.count(no_grad), 0);
  spec.insert(std::move(no_grad));
  ASSERT_EQ(spec.count(CompleteArgumentSpec(true, list)), 1);

  list2[1].toTensor().transpose_(0, 1);
  CompleteArgumentSpec c(true, list2); // same as list, except for one stride
  ASSERT_FALSE(c == a);
  ASSERT_EQ(spec.count(c), 0);

  Stack stack = {var(CF, {1, 2}, true), 3, var(CF, {1, 2}, true)};
  CompleteArgumentSpec with_const(true, stack);
  ASSERT_EQ(with_const.at(2).sizes().size(), 2);
}

// TODO: this test was disabled for unknown reasons and doesn't run.
// static size_t hashCode(const TensorTypePtr& ptr) {
//   return std::hash<TensorType>()(*ptr.get());
// }

// TEST(ArgumentSpecTest, VaryingShape) {
//   c10::VaryingShape<int64_t> vs(std::optional<size_t>{});
//   auto ptt_empty1 = TensorType::create({}, {}, vs, vs, false);
//   auto ptt_empty2 = TensorType::create({}, {}, vs, vs, false);
//   ASSERT_EQ(hashCode(ptt_empty1), hashCode(ptt_empty2));

//   c10::VaryingShape<int64_t> vs22(std::vector<int64_t>{2, 2});
//   auto ptt_vs22_vs22_1 = TensorType::create({}, {}, vs22, vs22, false);
//   auto ptt_vs22_vs22_2 = TensorType::create({}, {}, vs22, vs22, false);
//   ASSERT_EQ(hashCode(ptt_vs22_vs22_1), hashCode(ptt_vs22_vs22_2));

//   c10::VaryingShape<int64_t> vs23(std::vector<int64_t>{2, 3});
//   auto ptt_vs22_vs23_2 = TensorType::create({}, {}, vs22, vs23, false);
//   ASSERT_NE(hashCode(ptt_vs22_vs22_1), hashCode(ptt_vs22_vs23_2));

//   auto ptt_vs22_vs22_1_true = TensorType::create({}, {}, vs22, vs22, true);
//   auto ptt_vs22_vs22_2_true = TensorType::create({}, {}, vs22, vs22, true);
//   ASSERT_EQ(hashCode(ptt_vs22_vs22_1_true), hashCode(ptt_vs22_vs22_2_true));

//   auto ptt_vs22_vs22_1_false = TensorType::create({}, {}, vs22, vs22, false);
//   ASSERT_NE(hashCode(ptt_vs22_vs22_1_true), hashCode(ptt_vs22_vs22_1_false));
// }

TEST(ArgumentSpecTest, Basic_CUDA) {
  auto& CF = at::CPU(at::kFloat);
  auto& CD = at::CPU(at::kDouble);
  auto& GF = at::CUDA(at::kFloat);
  auto& GD = at::CUDA(at::kDouble);

  auto graph = toGraphFunction(jit::compile(R"JIT(
   def fn(a, b, c, d, e):
      return a, b, c, d, e
   )JIT")
                                   ->get_function("fn"))
                   .graph();

  ArgumentSpecCreator arg_spec_creator(*graph);

  auto list = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack(
      {var(CF, {1}, true),
       var(CD, {1, 2}, false),
       var(GF, {}, true),
       var(GD, {4, 5, 6}, false),
       undef()});
  list2[1].toTensor().transpose_(0, 1);

  ArgumentSpec a = arg_spec_creator.create(true, list);
  ArgumentSpec b = arg_spec_creator.create(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  ArgumentSpec d = arg_spec_creator.create(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    ASSERT_TRUE(isEqual(a.tensorAt(i), list[i].toTensor()));
  }
  ArgumentSpec no_grad = arg_spec_creator.create(/*with_grad=*/false, list);
  ASSERT_TRUE(no_grad != a);

  std::unordered_set<ArgumentSpec> spec;
  spec.insert(a); // we still need a for the test below
  ASSERT_TRUE(spec.count(b) > 0);
  ASSERT_EQ(spec.count(no_grad), 0);
  spec.insert(std::move(no_grad));
  ASSERT_EQ(spec.count(arg_spec_creator.create(true, list)), 1);

  list2[1].toTensor().transpose_(0, 1);
  ArgumentSpec c = arg_spec_creator.create(
      true, list2); // same as list, except for one stride, used to be
                    // different, now the same
  ASSERT_TRUE(c == a);
  ASSERT_EQ(spec.count(c), 1);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `TEST`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/runtime/argument_spec.h`
- `torch/jit.h`
- `test/cpp/jit/test_utils.h`


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
python test/cpp/jit/test_argument_spec.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_lite_interpreter_direct.cpp_docs.md`](./test_lite_interpreter_direct.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_argument_spec.cpp_docs.md`
- **Keyword Index**: `test_argument_spec.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/jit`, which is part of the **testing infrastructure**.



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
python docs/test/cpp/jit/test_argument_spec.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/jit`):

- [`test_graph_iterator.cpp_kw.md_docs.md`](./test_graph_iterator.cpp_kw.md_docs.md)
- [`test_qualified_name.cpp_docs.md_docs.md`](./test_qualified_name.cpp_docs.md_docs.md)
- [`test_fuser.cpp_kw.md_docs.md`](./test_fuser.cpp_kw.md_docs.md)
- [`test_utils.cpp_docs.md_docs.md`](./test_utils.cpp_docs.md_docs.md)
- [`test_custom_class_registrations.h_docs.md_docs.md`](./test_custom_class_registrations.h_docs.md_docs.md)
- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_cs_debug_info_serialization.cpp_docs.md_docs.md`](./test_cs_debug_info_serialization.cpp_docs.md_docs.md)
- [`torch_python_test.cpp_docs.md_docs.md`](./torch_python_test.cpp_docs.md_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_argument_spec.cpp_docs.md_docs.md`
- **Keyword Index**: `test_argument_spec.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
