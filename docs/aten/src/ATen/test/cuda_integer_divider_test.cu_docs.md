# Documentation: `aten/src/ATen/test/cuda_integer_divider_test.cu`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_integer_divider_test.cu`
- **Size**: 5,535 bytes (5.41 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cuda
#include <gtest/gtest.h>

// Test IntegerDivider: this tests *all* 32-bit pairs (a, b) where a % b is 0 or
// (b-1), so it takes a few minutes to run.

#include <assert.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IntegerDivider.cuh>

using std::vector;
using at::cuda::detail::IntDivider;
using at::cuda::detail::DivMod;

template<typename Value>
struct TestCase {
  Value dividend;
  int divisor_idx;
  int steps;

  TestCase(Value dividend, int divisor_idx, int steps)
      : dividend(dividend), divisor_idx(divisor_idx), steps(steps) {}
};

template <typename Value>
__global__ void testIntDivider(
    const IntDivider<Value>* dividers,
    const TestCase<Value>* testCases,
    int numCases) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numCases; i += stride) {
    const TestCase<Value>& tc = testCases[i];
    Value dividend = tc.dividend;
    const IntDivider<Value>& divider = dividers[tc.divisor_idx];
    Value divisor = divider.divisor;

    for (int j = 0; j < tc.steps; j++) {
      if (sizeof(Value) == 4 && dividend > INT32_MAX)
        return;

      DivMod<Value> qr = divider.divmod(dividend);
      assert(qr.div == dividend / divisor && qr.mod == dividend % divisor);
      dividend += divisor;
    }
  }
}

enum {
  // Number of test cases per each kernel invocation.
  NUM_CASES = 1000000,

  // Maximum number of steps per each test case.
  MAX_STEPS = 10000,
};

// Test the magic division algorithm.
template<typename Value>
class IntDividerTester {
 public:
  IntDividerTester() {
    cudaError_t err;

    err = cudaMalloc(&dividersBuf_, NUM_CASES * sizeof(IntDivider<Value>));
    bool isEQ = err == cudaSuccess;
    EXPECT_TRUE(isEQ);
    err = cudaMalloc(&testCasesBuf_, NUM_CASES * sizeof(TestCase<Value>));
    isEQ = err == cudaSuccess;
    EXPECT_TRUE(isEQ);
  }

  ~IntDividerTester() {
    cudaError_t err;

    err = cudaFree(dividersBuf_);
    bool isEQ = err == cudaSuccess;
    EXPECT_TRUE(isEQ);
    err = cudaFree(testCasesBuf_);
    isEQ = err == cudaSuccess;
    EXPECT_TRUE(isEQ);
  }

  void addTestCase(Value dividend, Value divisor, int steps) {
    // Append a new IntDivider using 'divisor' if necessary.
    if (dividers_.empty() || dividers_.back().divisor != divisor)
      dividers_.emplace_back(divisor);

    // Append the test case.
    testCases_.emplace_back(dividend, dividers_.size() - 1, steps);

    // Launch the test kernel if the buffer is full.
    if (testCases_.size() == NUM_CASES)
      flush();
  }

  void flush() {
    cudaError_t err;
    bool isTrue;
    if (testCases_.empty())
      return;

    ASSERT_FALSE(dividers_.empty());

    isTrue = dividers_.size() <= NUM_CASES;
    ASSERT_TRUE(isTrue);
    isTrue = testCases_.size() <= NUM_CASES;
    ASSERT_TRUE(isTrue);
    err = cudaMemcpy(
        dividersBuf_,
        dividers_.data(),
        dividers_.size() * sizeof(IntDivider<Value>),
        cudaMemcpyHostToDevice);
    isTrue = err == cudaSuccess;
    ASSERT_TRUE(isTrue);
    err = cudaMemcpy(
        testCasesBuf_,
        testCases_.data(),
        testCases_.size() * sizeof(TestCase<Value>),
        cudaMemcpyHostToDevice);
    isTrue = err == cudaSuccess;
    ASSERT_TRUE(isTrue);

    int numCases = testCases_.size();
    testIntDivider<Value><<<512, 512>>>(dividersBuf_, testCasesBuf_, numCases);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dividers_.clear();
    testCases_.clear();
  }

 private:
  vector<IntDivider<Value>> dividers_;
  vector<TestCase<Value>> testCases_;

  IntDivider<Value>* dividersBuf_;
  TestCase<Value>* testCasesBuf_;
};

static void testUint32Divider()
{
  fprintf(stderr, "Testing 32-bit integer division ...");

  IntDividerTester<uint32_t> tester;

  for (uint64_t divisor = 1; divisor <= INT32_MAX; divisor++) {
    if (divisor < 1000000 && divisor % 10000 == 0)
      fprintf(stderr, ".");
    if (divisor % 10000000 == 0)
      fprintf(stderr, "-");

    // In order to save time, we only test when the remainder is zero or
    // (divisor - 1).
    uint64_t dividend = 0;
    while (dividend <= INT32_MAX) {
      uint64_t steps = (INT32_MAX - dividend) / divisor + 1;
      if (steps > MAX_STEPS)
        steps = MAX_STEPS;

      tester.addTestCase(dividend, divisor, steps);
      tester.addTestCase(dividend + divisor - 1, divisor, steps);

      dividend += divisor * steps;
    }

    // Check the boundary cases.
    tester.addTestCase(1, divisor, 1);
    tester.addTestCase(INT32_MAX, divisor, 1);
  }

  tester.flush();

  fprintf(stderr, " Done!\n");
}

// uint64_t divider uses plain division, so we just check a few random cases.
static void testUint64Divider()
{
  IntDividerTester<uint64_t> tester;

  uint64_t dividend = 0x123456789ULL;
  uint64_t divisor = 0x54321ULL;

  for (int i = 0; i < 1000; i++) {
    if (divisor != 0) {
      tester.addTestCase(dividend, divisor, 100);

      // Test small divisor.
      tester.addTestCase(dividend, divisor % 65536, 100);

      // Create pseudorandom numbers.
      dividend *= 0x100000001b3ULL;
      dividend ^= 0x1234567890abcdefULL;
      divisor *= 0x100000001b3ULL;
      divisor ^= 0x1234567890abcdefULL;
    }
  }

  tester.flush();
}

TEST(TestCUDAIntegerDivider, IntegerDivider) {
  if (!at::cuda::is_available()) return;
  testUint64Divider();
  testUint32Divider();

  cudaError_t err = cudaDeviceSynchronize();
  bool isTrue = err == cudaSuccess;
  ASSERT_TRUE(isTrue);
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/test`.

## Detailed Analysis

### Code Structure

**Classes/Structs**: `TestCase`, `IntDividerTester`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `assert.h`
- `stdint.h`
- `memory`
- `vector`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/detail/IntegerDivider.cuh`


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
python aten/src/ATen/test/cuda_integer_divider_test.cu
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

- **File Documentation**: `cuda_integer_divider_test.cu_docs.md`
- **Keyword Index**: `cuda_integer_divider_test.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
