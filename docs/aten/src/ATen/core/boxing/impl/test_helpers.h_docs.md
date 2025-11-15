# Documentation: `aten/src/ATen/core/boxing/impl/test_helpers.h`

## File Metadata

- **Path**: `aten/src/ATen/core/boxing/impl/test_helpers.h`
- **Size**: 4,401 bytes (4.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This appears to be a **test file**.

## Original Source

```c
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/irange.h>

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

inline at::Tensor dummyTensor(
    c10::DispatchKeySet ks,
    bool requires_grad = false) {
  auto* allocator = c10::GetCPUAllocator();
  int64_t nelements = 1;
  auto dtype = caffe2::TypeMeta::Make<float>();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  at::Tensor t =
      at::detail::make_tensor<c10::TensorImpl>(storage_impl, ks, dtype);
  // TODO: We add this to simulate the ideal case where we only have Autograd
  // backend keys
  //       on Tensor when it requires grad. But currently Autograd keys are
  //       added in TensorImpl constructor by default.
  if (!requires_grad) {
    t.unsafeGetTensorImpl()->remove_autograd_key();
  }
  return t;
}

inline at::Tensor dummyTensor(
    c10::DispatchKey dispatch_key,
    bool requires_grad = false) {
  return dummyTensor(c10::DispatchKeySet(dispatch_key), requires_grad);
}

template <class... Args>
inline std::vector<c10::IValue> callOp(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  op.callBoxed(&stack);
  return stack;
}

template <class Result, class... Args>
inline Result callOpUnboxed(const c10::OperatorHandle& op, Args... args) {
  return op.typed<Result(Args...)>().call(std::forward<Args>(args)...);
}

template <class Result, class... Args>
inline Result callOpUnboxedWithDispatchKey(
    const c10::OperatorHandle& op,
    c10::DispatchKey dispatchKey,
    Args... args) {
  return op.typed<Result(Args...)>().callWithDispatchKey(
      dispatchKey, std::forward<Args>(args)...);
}

template <class Result, class... Args>
inline Result callOpUnboxedWithPrecomputedDispatchKeySet(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet ks,
    Args... args) {
  return op.typed<Result(Args...)>().redispatch(
      ks, std::forward<Args>(args)...);
}

inline void expectDoesntFindKernel(
    const char* op_name,
    c10::DispatchKey dispatch_key) {
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
  EXPECT_ANY_THROW(callOp(*op, dummyTensor(dispatch_key), 5););
}

inline void expectDoesntFindOperator(const char* op_name) {
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
  EXPECT_FALSE(op.has_value());
}

template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception containing \""
                << expectMessageContains << "\" but didn't throw";
}

template <class T, size_t N>
void expectListEquals(c10::ArrayRef<T> expected, std::array<T, N> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

template <class T>
void expectListEquals(c10::ArrayRef<T> expected, c10::ArrayRef<T> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

template <class T>
void expectListEquals(c10::ArrayRef<T> expected, c10::List<T> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual.get(i));
  }
}

template <class T>
void expectListEquals(c10::ArrayRef<T> expected, std::vector<T> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (const auto i : c10::irange(expected.size())) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

// NB: This is not really sound, but all of the type sets constructed here
// are singletons so it's fine
static inline c10::DispatchKey extractDispatchKey(const at::Tensor& t) {
  return legacyExtractDispatchKey(t.key_set());
}

```



## High-Level Overview


This C++ file contains approximately 9 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `Result`, `Result`, `Result`, `Exception`, `Functor`, `T`, `T`, `T`, `T`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core/boxing/impl`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gmock/gmock.h`
- `gtest/gtest.h`
- `ATen/core/Tensor.h`
- `ATen/core/dispatch/Dispatcher.h`
- `ATen/core/ivalue.h`
- `c10/core/CPUAllocator.h`
- `c10/util/irange.h`


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
python aten/src/ATen/core/boxing/impl/test_helpers.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/core/boxing/impl`):

- [`kernel_stackbased_test.cpp_docs.md`](./kernel_stackbased_test.cpp_docs.md)
- [`kernel_function_test.cpp_docs.md`](./kernel_function_test.cpp_docs.md)
- [`WrapFunctionIntoRuntimeFunctor.h_docs.md`](./WrapFunctionIntoRuntimeFunctor.h_docs.md)
- [`kernel_lambda_legacy_test.cpp_docs.md`](./kernel_lambda_legacy_test.cpp_docs.md)
- [`make_boxed_from_unboxed_functor_test.cpp_docs.md`](./make_boxed_from_unboxed_functor_test.cpp_docs.md)
- [`kernel_function_legacy_test.cpp_docs.md`](./kernel_function_legacy_test.cpp_docs.md)
- [`kernel_lambda_test.cpp_docs.md`](./kernel_lambda_test.cpp_docs.md)
- [`boxing.h_docs.md`](./boxing.h_docs.md)
- [`make_boxed_from_unboxed_functor.h_docs.md`](./make_boxed_from_unboxed_functor.h_docs.md)


## Cross-References

- **File Documentation**: `test_helpers.h_docs.md`
- **Keyword Index**: `test_helpers.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
