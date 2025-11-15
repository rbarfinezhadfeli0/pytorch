# Documentation: `aten/src/ATen/test/extension_backend_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/extension_backend_test.cpp`
- **Size**: 2,347 bytes (2.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <torch/csrc/jit/runtime/operator.h>

// NB. These tests use the MAIA dispatch key to test backend dispatching
// machinery, but these tests are not specific to MAIA at all. The MAIA
// backend is fully out-of-tree, so it's safe to use this key for
// in-tree tests.

using namespace at;

static int test_int;

Tensor empty_override(SymIntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout,
                      std::optional<Device> device, std::optional<bool> pin_memory, std::optional<MemoryFormat> optional_memory_format) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),
          0,
          at::DataPtr(nullptr, Device(DeviceType::MAIA, 1)),
          nullptr,
          false),
      DispatchKey::MAIA,
      caffe2::TypeMeta::Make<float>());
  return Tensor(std::move(tensor_impl));
}

Tensor add_override(const Tensor & a, const Tensor & b , const Scalar& c) {
  auto out = empty({5, 5}, at::kMAIA);  // Don't return self as-is
  test_int = 2;
  return out;
}

Tensor empty_strided_override(
  IntArrayRef size,
  IntArrayRef stride,
  std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout,
  std::optional<c10::Device> device,
  std::optional<bool> pin_memory) {

  return empty_override(fromIntArrayRefSlow(size), dtype, layout, device, pin_memory, std::nullopt);
}

TORCH_LIBRARY_IMPL(aten, MAIA, m) {
  m.impl("aten::empty.memory_format",  empty_override);
  m.impl("aten::empty_strided",        empty_strided_override);
  m.impl("aten::add.Tensor",           add_override);
}

TEST(BackendExtensionTest, TestRegisterOp) {
  Tensor a = empty({5, 5}, at::kMAIA);
  ASSERT_EQ(a.device().type(), at::kMAIA);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  Tensor b = empty_like(a, at::kMAIA);
  ASSERT_EQ(b.device().type(), at::kMAIA);
  ASSERT_EQ(b.device().index(), 1);
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  add(a, b);
  ASSERT_EQ(test_int, 2);

  // Ensure that non-MAIA operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/NativeFunctions.h`
- `torch/library.h`
- `torch/csrc/jit/runtime/operator.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python aten/src/ATen/test/extension_backend_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `extension_backend_test.cpp_docs.md`
- **Keyword Index**: `extension_backend_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
