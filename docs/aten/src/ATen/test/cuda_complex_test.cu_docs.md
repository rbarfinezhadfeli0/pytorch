# Documentation: `aten/src/ATen/test/cuda_complex_test.cu`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_complex_test.cu`
- **Size**: 3,516 bytes (3.43 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cuda
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAException.h>
#include <c10/test/util/complex_test_common.h>

__global__ void test_thrust_kernel() {
  // thrust conversion
  {
  [[maybe_unused]] constexpr float num1 = float(1.23);
  [[maybe_unused]] constexpr float num2 = float(4.56);
  assert(c10::complex<float>(thrust::complex<float>(num1, num2)).real() == num1);
  assert(c10::complex<float>(thrust::complex<float>(num1, num2)).imag() == num2);
  }
  {
  [[maybe_unused]] constexpr double num1 = double(1.23);
  [[maybe_unused]] constexpr double num2 = double(4.56);
  assert(c10::complex<double>(thrust::complex<double>(num1, num2)).real() == num1);
  assert(c10::complex<double>(thrust::complex<double>(num1, num2)).imag() == num2);
  }
  // thrust assignment
  auto tup = assignment::one_two_thrust();
  assert(std::get<c10::complex<double>>(tup).real() == double(1));
  assert(std::get<c10::complex<double>>(tup).imag() == double(2));
  assert(std::get<c10::complex<float>>(tup).real() == float(1));
  assert(std::get<c10::complex<float>>(tup).imag() == float(2));
}

__global__ void test_std_functions_kernel() {
  assert(std::abs(c10::complex<float>(3, 4)) == float(5));
  assert(std::abs(c10::complex<double>(3, 4)) == double(5));

  assert(std::abs(std::arg(c10::complex<float>(0, 1)) - PI / 2) < 1e-6);
  assert(std::abs(std::arg(c10::complex<double>(0, 1)) - PI / 2) < 1e-6);

  assert(std::abs(c10::polar(float(1), float(PI / 2)) - c10::complex<float>(0, 1)) < 1e-6);
  assert(std::abs(c10::polar(double(1), double(PI / 2)) - c10::complex<double>(0, 1)) < 1e-6);
}

__global__ void test_reinterpret_cast() {
  std::complex<float> z(1, 2);
  c10::complex<float> zz = *reinterpret_cast<c10::complex<float>*>(&z);
  assert(zz.real() == float(1));
  assert(zz.imag() == float(2));

  std::complex<double> zzz(1, 2);
  c10::complex<double> zzzz = *reinterpret_cast<c10::complex<double>*>(&zzz);
  assert(zzzz.real() == double(1));
  assert(zzzz.imag() == double(2));

  [[maybe_unused]] cuComplex cuComplex_zz = *reinterpret_cast<cuComplex*>(&zz);
  assert(cuComplex_zz.x == float(1));
  assert(cuComplex_zz.y == float(2));

  [[maybe_unused]] cuDoubleComplex cuDoubleComplex_zzzz = *reinterpret_cast<cuDoubleComplex*>(&zzzz);
  assert(cuDoubleComplex_zzzz.x == double(1));
  assert(cuDoubleComplex_zzzz.y == double(2));
}

int safeDeviceCount() {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver || err == cudaErrorNoDevice) {
    return 0;
  }
  return count;
}

#define SKIP_IF_NO_GPU()                    \
  do {                                      \
    if (safeDeviceCount() == 0) {           \
      return;                               \
    }                                       \
  } while(0)

TEST(DeviceTests, ThrustConversion) {
  SKIP_IF_NO_GPU();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  test_thrust_kernel<<<1, 1>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(DeviceTests, StdFunctions) {
  SKIP_IF_NO_GPU();
  cudaDeviceSynchronize();
  test_std_functions_kernel<<<1, 1>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(DeviceTests, ReinterpretCast) {
  SKIP_IF_NO_GPU();
  cudaDeviceSynchronize();
  test_reinterpret_cast<<<1, 1>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/CUDABlas.h`
- `c10/cuda/CUDAException.h`
- `c10/test/util/complex_test_common.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/test/cuda_complex_test.cu
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

- **File Documentation**: `cuda_complex_test.cu_docs.md`
- **Keyword Index**: `cuda_complex_test.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
