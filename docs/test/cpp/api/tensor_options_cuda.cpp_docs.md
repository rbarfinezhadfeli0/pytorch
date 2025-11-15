# Documentation: `test/cpp/api/tensor_options_cuda.cpp`

## File Metadata

- **Path**: `test/cpp/api/tensor_options_cuda.cpp`
- **Size**: 2,964 bytes (2.89 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/cuda.h>

// NB: This file is compiled even in CPU build (for some reason), so
// make sure you don't include any CUDA only headers.

using namespace at;

// TODO: This might be generally helpful aliases elsewhere.
at::Device CPUDevice() {
  return at::Device(at::kCPU);
}
at::Device CUDADevice(DeviceIndex index) {
  return at::Device(at::kCUDA, index);
}

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                  \
  ASSERT_EQ(options.device().type(), Device((device_), (index_)).type()); \
  ASSERT_TRUE(                                                            \
      options.device().index() == Device((device_), (index_)).index());   \
  ASSERT_EQ(typeMetaToScalarType(options.dtype()), (type_));              \
  ASSERT_TRUE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_EQ(tensor.device().type(), Device((device_), (index_)).type());   \
  ASSERT_EQ(tensor.device().index(), Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.scalar_type(), (type_));                                \
  ASSERT_TRUE(tensor.options().layout() == (layout_))

TEST(TensorOptionsTest, ConstructsWellFromCUDATypes_CUDA) {
  auto options = CUDA(kFloat).options();
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kStrided);

  options = CUDA(kInt).options();
  REQUIRE_OPTIONS(kCUDA, -1, kInt, kStrided);

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat).options();
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kSparse);

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kByte).options();
  REQUIRE_OPTIONS(kCUDA, -1, kByte, kSparse);

  // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
  options = CUDA(kFloat).options(/*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kStrided);

  options =
      // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
      getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat)
          .options(/*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kSparse);
}

TEST(TensorOptionsTest, ConstructsWellFromCUDATensors_MultiCUDA) {
  auto options = empty(5, device(kCUDA).dtype(kDouble)).options();
  REQUIRE_OPTIONS(kCUDA, 0, kDouble, kStrided);

  options = empty(5, getDeprecatedTypeProperties(Backend::SparseCUDA, kByte))
                .options();
  REQUIRE_OPTIONS(kCUDA, 0, kByte, kSparse);

  if (torch::cuda::device_count() > 1) {
    Tensor tensor;
    {
      DeviceGuard guard(CUDADevice(1));
      tensor = empty(5, device(kCUDA));
    }
    options = tensor.options();
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

    {
      DeviceGuard guard(CUDADevice(1));
      tensor = empty(5, device(kCUDA).layout(kSparse));
    }
    options = tensor.options();
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kSparse);
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/torch.h`
- `torch/cuda.h`


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
python test/cpp/api/tensor_options_cuda.cpp
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

- **File Documentation**: `tensor_options_cuda.cpp_docs.md`
- **Keyword Index**: `tensor_options_cuda.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
