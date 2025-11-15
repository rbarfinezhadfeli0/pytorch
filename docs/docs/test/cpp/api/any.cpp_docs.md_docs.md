# Documentation: `docs/test/cpp/api/any.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/any.cpp_docs.md`
- **Size**: 16,375 bytes (15.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/any.cpp`

## File Metadata

- **Path**: `test/cpp/api/any.cpp`
- **Size**: 13,890 bytes (13.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <algorithm>
#include <string>

using namespace torch::nn;

struct AnyModuleTest : torch::test::SeedingFixture {};

TEST_F(AnyModuleTest, SimpleReturnType) {
  struct M : torch::nn::Module {
    int forward() {
      return 123;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(), 123);
}

TEST_F(AnyModuleTest, SimpleReturnTypeAndSingleArgument) {
  struct M : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(5), 5);
}

TEST_F(AnyModuleTest, StringLiteralReturnTypeAndArgument) {
  struct M : torch::nn::Module {
    const char* forward(const char* x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<const char*>("hello"), std::string("hello"));
}

TEST_F(AnyModuleTest, StringReturnTypeWithConstArgument) {
  struct M : torch::nn::Module {
    std::string forward(int x, const double f) {
      return std::to_string(static_cast<int>(x + f));
    }
  };
  AnyModule any(M{});
  int x = 4;
  ASSERT_EQ(any.forward<std::string>(x, 3.14), std::string("7"));
}

TEST_F(
    AnyModuleTest,
    TensorReturnTypeAndStringArgumentsWithFunkyQualifications) {
  struct M : torch::nn::Module {
    torch::Tensor forward(
        std::string a,
        const std::string& b,
        std::string&& c) {
      const auto s = a + b + c;
      return torch::ones({static_cast<int64_t>(s.size())});
    }
  };
  AnyModule any(M{});
  ASSERT_TRUE(
      any.forward(std::string("a"), std::string("ab"), std::string("abc"))
          .sum()
          .item<int32_t>() == 6);
}

TEST_F(AnyModuleTest, WrongArgumentType) {
  struct M : torch::nn::Module {
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_THROWS_WITH(
      any.forward(5.0),
      "Expected argument #0 to be of type float, "
      "but received value of type double");
}

struct M_test_wrong_number_of_arguments : torch::nn::Module {
  int forward(int a, int b) {
    return a + b;
  }
};

TEST_F(AnyModuleTest, WrongNumberOfArguments) {
  AnyModule any(M_test_wrong_number_of_arguments{});
#if defined(_MSC_VER)
  std::string module_name = "struct M_test_wrong_number_of_arguments";
#else
  std::string module_name = "M_test_wrong_number_of_arguments";
#endif
  ASSERT_THROWS_WITH(
      any.forward(),
      module_name +
          "'s forward() method expects 2 argument(s), but received 0. "
          "If " +
          module_name +
          "'s forward() method has default arguments, "
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
  ASSERT_THROWS_WITH(
      any.forward(5),
      module_name +
          "'s forward() method expects 2 argument(s), but received 1. "
          "If " +
          module_name +
          "'s forward() method has default arguments, "
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
  ASSERT_THROWS_WITH(
      any.forward(1, 2, 3),
      module_name +
          "'s forward() method expects 2 argument(s), but received 3.");
}

struct M_default_arg_with_macro : torch::nn::Module {
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {1, torch::nn::AnyValue(2)},
      {2, torch::nn::AnyValue(3.0)})
};

struct M_default_arg_without_macro : torch::nn::Module {
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }
};

TEST_F(
    AnyModuleTest,
    PassingArgumentsToModuleWithDefaultArgumentsInForwardMethod) {
  {
    AnyModule any(M_default_arg_with_macro{});

    ASSERT_EQ(any.forward<double>(1), 6.0);
    ASSERT_EQ(any.forward<double>(1, 3), 7.0);
    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

    ASSERT_THROWS_WITH(
        any.forward(),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 0.");
    ASSERT_THROWS_WITH(
        any.forward(1, 2, 3.0, 4),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 4.");
  }
  {
    AnyModule any(M_default_arg_without_macro{});

    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

#if defined(_MSC_VER)
    std::string module_name = "struct M_default_arg_without_macro";
#else
    std::string module_name = "M_default_arg_without_macro";
#endif

    ASSERT_THROWS_WITH(
        any.forward(),
        module_name +
            "'s forward() method expects 3 argument(s), but received 0. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward<double>(1),
        module_name +
            "'s forward() method expects 3 argument(s), but received 1. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward<double>(1, 3),
        module_name +
            "'s forward() method expects 3 argument(s), but received 2. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward(1, 2, 3.0, 4),
        module_name +
            "'s forward() method expects 3 argument(s), but received 4.");
  }
}

struct M : torch::nn::Module {
  explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
  int value;
  int forward(float x) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return x;
  }
};

TEST_F(AnyModuleTest, GetWithCorrectTypeSucceeds) {
  AnyModule any(M{5});
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, GetWithIncorrectTypeThrows) {
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.get<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, PtrWithBaseClassSucceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->name(), "M");
}

TEST_F(AnyModuleTest, PtrWithGoodDowncastSuccceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr<M>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->value, 5);
}

TEST_F(AnyModuleTest, PtrWithBadDowncastThrows) {
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.ptr<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, DefaultStateIsEmpty) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>(5);
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, AllMethodsThrowForEmptyAnyModule) {
  struct M : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  ASSERT_THROWS_WITH(any.get<M>(), "Cannot call get() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr<M>(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.type_info(), "Cannot call type_info() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.forward<int>(5), "Cannot call forward() on an empty AnyModule");
}

TEST_F(AnyModuleTest, CanMoveAssignDifferentModules) {
  struct M : torch::nn::Module {
    std::string forward(int x) {
      return std::to_string(x);
    }
  };
  struct N : torch::nn::Module {
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return 3 + x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<std::string>(5), "5");
  any = std::make_shared<N>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<int>(5.0f), 8);
}

TEST_F(AnyModuleTest, ConstructsFromModuleHolder) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : torch::nn::Module("M"), value(value_) {}
    int value;
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };

  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  AnyModule any(M{5});
  ASSERT_EQ(any.get<MImpl>().value, 5);
  ASSERT_EQ(any.get<M>()->value, 5);

  AnyModule module(Linear(3, 4));
  std::shared_ptr<Module> ptr = module.ptr();
  Linear linear(module.get<Linear>());
}

TEST_F(AnyModuleTest, ConvertsVariableToTensorCorrectly) {
  struct M : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };

  // When you have an autograd::Variable, it should be converted to a
  // torch::Tensor before being passed to the function (to avoid a type
  // mismatch).
  AnyModule any(M{});
  ASSERT_TRUE(
      any.forward(torch::autograd::Variable(torch::ones(5)))
          .sum()
          .item<float>() == 5);
  // at::Tensors that are not variables work too.
  ASSERT_EQ(any.forward(at::ones(5)).sum().item<float>(), 5);
}

namespace torch {
namespace nn {
struct TestAnyValue {
  template <typename T>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit TestAnyValue(T&& value) : value_(std::forward<T>(value)) {}
  AnyValue operator()() {
    return std::move(value_);
  }
  AnyValue value_;
};
template <typename T>
AnyValue make_value(T&& value) {
  return TestAnyValue(std::forward<T>(value))();
}
} // namespace nn
} // namespace torch

struct AnyValueTest : torch::test::SeedingFixture {};

TEST_F(AnyValueTest, CorrectlyAccessesIntWhenCorrectType) {
  auto value = make_value<int>(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  // const and non-const types have the same typeid(),
  // but casting Holder<int> to Holder<const int> is undefined
  // behavior according to UBSAN:
  // https://github.com/pytorch/pytorch/issues/26964
  // ASSERT_NE(value.try_get<const int>(), nullptr);
  ASSERT_EQ(value.get<int>(), 5);
}
// This test does not work at all, because it looks like make_value
// decays const int into int.
// TEST_F(AnyValueTest, CorrectlyAccessesConstIntWhenCorrectType) {
//  auto value = make_value<const int>(5);
//  ASSERT_NE(value.try_get<const int>(), nullptr);
//  // ASSERT_NE(value.try_get<int>(), nullptr);
//  ASSERT_EQ(value.get<const int>(), 5);
//}
TEST_F(AnyValueTest, CorrectlyAccessesStringLiteralWhenCorrectType) {
  auto value = make_value("hello");
  ASSERT_NE(value.try_get<const char*>(), nullptr);
  ASSERT_EQ(value.get<const char*>(), std::string("hello"));
}
TEST_F(AnyValueTest, CorrectlyAccessesStringWhenCorrectType) {
  auto value = make_value(std::string("hello"));
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesPointersWhenCorrectType) {
  std::string s("hello");
  std::string* p = &s;
  auto value = make_value(p);
  ASSERT_NE(value.try_get<std::string*>(), nullptr);
  ASSERT_EQ(*value.get<std::string*>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesReferencesWhenCorrectType) {
  std::string s("hello");
  const std::string& t = s;
  auto value = make_value(t);
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}

TEST_F(AnyValueTest, TryGetReturnsNullptrForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_EQ(value.try_get<float>(), nullptr);
  ASSERT_EQ(value.try_get<long>(), nullptr);
  ASSERT_EQ(value.try_get<std::string>(), nullptr);
}

TEST_F(AnyValueTest, GetThrowsForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_THROWS_WITH(
      value.get<float>(),
      "Attempted to cast AnyValue to float, "
      "but its actual type is int");
  ASSERT_THROWS_WITH(
      value.get<long>(),
      "Attempted to cast AnyValue to long, "
      "but its actual type is int");
}

TEST_F(AnyValueTest, MoveConstructionIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(std::move(value));
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, MoveAssignmentIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(10);
  copy = std::move(value);
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForInt) {
  auto value = make_value(5);
  ASSERT_EQ(value.type_info().hash_code(), typeid(int).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForStringLiteral) {
  auto value = make_value("hello");
  ASSERT_EQ(value.type_info().hash_code(), typeid(const char*).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForString) {
  auto value = make_value(std::string("hello"));
  ASSERT_EQ(value.type_info().hash_code(), typeid(std::string).hash_code());
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 85 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `nn`

**Classes/Structs**: `AnyModuleTest`, `M`, `M`, `M`, `M`, `M`, `M`, `M_test_wrong_number_of_arguments`, `M_test_wrong_number_of_arguments`, `M_default_arg_with_macro`, `M_default_arg_without_macro`, `M_default_arg_without_macro`, `M`, `N`, `N`, `M`, `M`, `M`, `N`, `MImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/torch.h`
- `test/cpp/api/support.h`
- `algorithm`
- `string`


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
python test/cpp/api/any.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `any.cpp_docs.md`
- **Keyword Index**: `any.cpp_kw.md`
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
python docs/test/cpp/api/any.cpp_docs.md
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

- **File Documentation**: `any.cpp_docs.md_docs.md`
- **Keyword Index**: `any.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
