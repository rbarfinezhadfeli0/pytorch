# Documentation: `test/cpp/api/parameterlist.cpp`

## File Metadata

- **Path**: `test/cpp/api/parameterlist.cpp`
- **Size**: 5,885 bytes (5.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ParameterListTest : torch::test::SeedingFixture {};

TEST_F(ParameterListTest, ConstructsFromSharedPointer) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ParameterListTest, isEmpty) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  ParameterList list;
  ASSERT_TRUE(list->is_empty());
  list->append(ta);
  ASSERT_FALSE(list->is_empty());
  ASSERT_EQ(list->size(), 1);
}

TEST_F(ParameterListTest, PushBackAddsAnElement) {
  ParameterList list;
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  ASSERT_EQ(list->size(), 0);
  ASSERT_TRUE(list->is_empty());
  list->append(ta);
  ASSERT_EQ(list->size(), 1);
  list->append(tb);
  ASSERT_EQ(list->size(), 2);
  list->append(tc);
  ASSERT_EQ(list->size(), 3);
  list->append(td);
  ASSERT_EQ(list->size(), 4);
}
TEST_F(ParameterListTest, ForEachLoop) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  ParameterList list(ta, tb, tc, td);
  std::vector<torch::Tensor> params = {ta, tb, tc, td};
  ASSERT_EQ(list->size(), 4);
  int idx = 0;
  for (const auto& pair : *list) {
    ASSERT_TRUE(
        torch::all(torch::eq(pair.value(), params[idx++])).item<bool>());
  }
}

TEST_F(ParameterListTest, AccessWithAt) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  std::vector<torch::Tensor> params = {ta, tb, tc, td};

  ParameterList list;
  for (auto& param : params) {
    list->append(param);
  }
  ASSERT_EQ(list->size(), 4);

  // returns the correct module for a given index
  for (const auto i : c10::irange(params.size())) {
    ASSERT_TRUE(torch::all(torch::eq(list->at(i), params[i])).item<bool>());
  }

  for (const auto i : c10::irange(params.size())) {
    ASSERT_TRUE(torch::all(torch::eq(list[i], params[i])).item<bool>());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at(params.size() + 100), "Index out of range");
  ASSERT_THROWS_WITH(list->at(params.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(list[params.size() + 1], "Index out of range");
}

TEST_F(ParameterListTest, ExtendPushesParametersFromOtherParameterList) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  torch::Tensor te = torch::randn({1, 2});
  torch::Tensor tf = torch::randn({1, 2, 3});
  ParameterList a(ta, tb);
  ParameterList b(tc, td);
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(torch::all(torch::eq(a[0], ta)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[1], tb)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[2], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[3], td)).item<bool>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());

  std::vector<torch::Tensor> c = {te, tf};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[2], te)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[3], tf)).item<bool>());
}

TEST_F(ParameterListTest, PrettyPrintParameterList) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ParameterList(\n"
      "(0): Parameter containing: [Float of size [1, 2]]\n"
      "(1): Parameter containing: [Float of size [1, 2]]\n"
      "(2): Parameter containing: [Float of size [1, 2]]\n"
      ")");
}

TEST_F(ParameterListTest, IncrementAdd) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  torch::Tensor te = torch::randn({1, 2});
  torch::Tensor tf = torch::randn({1, 2, 3});
  ParameterList listA(ta, tb, tc);
  ParameterList listB(td, te, tf);
  std::vector<torch::Tensor> tensors{ta, tb, tc, td, te, tf};
  int idx = 0;
  *listA += *listB;
  ASSERT_TRUE(torch::all(torch::eq(listA[0], ta)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[1], tb)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[2], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[3], td)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[4], te)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[5], tf)).item<bool>());
  for (const auto& P : listA->named_parameters(false))
    ASSERT_TRUE(torch::all(torch::eq(P.value(), tensors[idx++])).item<bool>());

  ASSERT_EQ(idx, 6);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ParameterListTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
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
python test/cpp/api/parameterlist.cpp
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

- **File Documentation**: `parameterlist.cpp_docs.md`
- **Keyword Index**: `parameterlist.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
