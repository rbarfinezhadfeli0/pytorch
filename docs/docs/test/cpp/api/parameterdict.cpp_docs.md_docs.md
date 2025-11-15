# Documentation: `docs/test/cpp/api/parameterdict.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/parameterdict.cpp_docs.md`
- **Size**: 7,561 bytes (7.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/parameterdict.cpp`

## File Metadata

- **Path**: `test/cpp/api/parameterdict.cpp`
- **Size**: 5,215 bytes (5.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ParameterDictTest : torch::test::SeedingFixture {};

TEST_F(ParameterDictTest, ConstructFromTensor) {
  ParameterDict dict;
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(torch::all(torch::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(torch::all(torch::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

TEST_F(ParameterDictTest, ConstructFromOrderedDict) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"A", ta}, {"B", tb}, {"C", tc}};
  auto dict = torch::nn::ParameterDict(params);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(torch::all(torch::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(torch::all(torch::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

TEST_F(ParameterDictTest, InsertAndContains) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("A"));
  ASSERT_FALSE(dict->contains("C"));
}

TEST_F(ParameterDictTest, InsertAndClear) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  dict->clear();
  ASSERT_EQ(dict->size(), 0);
}

TEST_F(ParameterDictTest, InsertAndPop) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_THROWS_WITH(dict->pop("B"), "Parameter 'B' is not defined");
  torch::Tensor p = dict->pop("A");
  ASSERT_EQ(dict->size(), 0);
  ASSERT_TRUE(torch::eq(p, torch::tensor({1.0})).item<bool>());
}

TEST_F(ParameterDictTest, SimpleUpdate) {
  ParameterDict dict;
  ParameterDict wrongDict;
  ParameterDict rightDict;
  dict->insert("A", torch::tensor({1.0}));
  dict->insert("B", torch::tensor({2.0}));
  dict->insert("C", torch::tensor({3.0}));
  wrongDict->insert("A", torch::tensor({5.0}));
  wrongDict->insert("D", torch::tensor({5.0}));
  ASSERT_THROWS_WITH(dict->update(*wrongDict), "Parameter 'D' is not defined");
  rightDict->insert("A", torch::tensor({5.0}));
  dict->update(*rightDict);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(torch::eq(dict["A"], torch::tensor({5.0})).item<bool>());
}

TEST_F(ParameterDictTest, Keys) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0})},
      {"c", torch::tensor({1.0, 2.0})}};
  auto dict = torch::nn::ParameterDict(params);
  std::vector<std::string> keys = dict->keys();
  std::vector<std::string> true_keys{"a", "b", "c"};
  ASSERT_EQ(keys, true_keys);
}

TEST_F(ParameterDictTest, Values) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", ta}, {"b", tb}, {"c", tc}};
  auto dict = torch::nn::ParameterDict(params);
  std::vector<torch::Tensor> values = dict->values();
  std::vector<torch::Tensor> true_values{ta, tb, tc};
  for (auto i = 0U; i < values.size(); i += 1) {
    ASSERT_TRUE(torch::all(torch::eq(values[i], true_values[i])).item<bool>());
  }
}

TEST_F(ParameterDictTest, Get) {
  ParameterDict dict;
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(torch::all(torch::eq(dict->get("A"), ta)).item<bool>());
  ASSERT_TRUE(dict->get("A").requires_grad());
  ASSERT_TRUE(torch::all(torch::eq(dict->get("B"), tb)).item<bool>());
  ASSERT_FALSE(dict->get("B").requires_grad());
}

TEST_F(ParameterDictTest, PrettyPrintParameterDict) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0, 1.0})},
      {"c", torch::tensor({{3.0}, {2.1}})},
      {"d", torch::tensor({{3.0, 1.3}, {1.2, 2.1}})}};
  auto dict = torch::nn::ParameterDict(params);
  ASSERT_EQ(
      c10::str(dict),
      "torch::nn::ParameterDict(\n"
      "(a): Parameter containing: [Float of size [1]]\n"
      "(b): Parameter containing: [Float of size [2]]\n"
      "(c): Parameter containing: [Float of size [2, 1]]\n"
      "(d): Parameter containing: [Float of size [2, 2]]\n"
      ")");
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ParameterDictTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/torch.h`
- `algorithm`
- `memory`
- `vector`
- `test/cpp/api/support.h`


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
python test/cpp/api/parameterdict.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`any.cpp_docs.md`](./any.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `parameterdict.cpp_docs.md`
- **Keyword Index**: `parameterdict.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/api/parameterdict.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `parameterdict.cpp_docs.md_docs.md`
- **Keyword Index**: `parameterdict.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
