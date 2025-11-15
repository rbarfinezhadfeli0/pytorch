# Documentation: `test/cpp/api/fft.cpp`

## File Metadata

- **Path**: `test/cpp/api/fft.cpp`
- **Size**: 4,380 bytes (4.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>

// Naive DFT of a 1 dimensional tensor
torch::Tensor naive_dft(torch::Tensor x, bool forward = true) {
  TORCH_INTERNAL_ASSERT(x.dim() == 1);
  x = x.contiguous();
  auto out_tensor = torch::zeros_like(x);
  const int64_t len = x.size(0);

  // Roots of unity, exp(-2*pi*j*n/N) for n in [0, N), reversed for inverse
  // transform
  std::vector<c10::complex<double>> roots(len);
  const auto angle_base = (forward ? -2.0 : 2.0) * M_PI / len;
  for (const auto i : c10::irange(len)) {
    auto angle = i * angle_base;
    roots[i] = c10::complex<double>(std::cos(angle), std::sin(angle));
  }

  const auto in = x.data_ptr<c10::complex<double>>();
  const auto out = out_tensor.data_ptr<c10::complex<double>>();
  for (const auto i : c10::irange(len)) {
    for (const auto j : c10::irange(len)) {
      out[i] += roots[(j * i) % len] * in[j];
    }
  }
  return out_tensor;
}

// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

TEST(FFTTest, fft) {
  auto t = torch::randn(128, torch::kComplexDouble);
  auto actual = torch::fft::fft(t);
  auto expect = naive_dft(t);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, fft_real) {
  auto t = torch::randn(128, torch::kDouble);
  auto actual = torch::fft::fft(t);
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble));
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, fft_pad) {
  auto t = torch::randn(128, torch::kComplexDouble);
  auto actual = torch::fft::fft(t, 200);
  auto expect = torch::fft::fft(torch::constant_pad_nd(t, {0, 72}));
  ASSERT_TRUE(torch::allclose(actual, expect));

  actual = torch::fft::fft(t, 64);
  expect = torch::fft::fft(torch::constant_pad_nd(t, {0, -64}));
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, fft_norm) {
  auto t = torch::randn(128, torch::kComplexDouble);
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto unnorm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/{});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"forward");
  ASSERT_TRUE(torch::allclose(unnorm / 128, norm));

  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto ortho_norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"ortho");
  ASSERT_TRUE(torch::allclose(unnorm / std::sqrt(128), ortho_norm));
}

TEST(FFTTest, ifft) {
  auto T = torch::randn(128, torch::kComplexDouble);
  auto actual = torch::fft::ifft(T);
  auto expect = naive_dft(T, /*forward=*/false) / 128;
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, fft_ifft) {
  auto t = torch::randn(77, torch::kComplexDouble);
  auto T = torch::fft::fft(t);
  ASSERT_EQ(T.size(0), 77);
  ASSERT_EQ(T.scalar_type(), torch::kComplexDouble);

  auto t_round_trip = torch::fft::ifft(T);
  ASSERT_EQ(t_round_trip.size(0), 77);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kComplexDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

TEST(FFTTest, rfft) {
  auto t = torch::randn(129, torch::kDouble);
  auto actual = torch::fft::rfft(t);
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, rfft_irfft) {
  auto t = torch::randn(128, torch::kDouble);
  auto T = torch::fft::rfft(t);
  ASSERT_EQ(T.size(0), 65);
  ASSERT_EQ(T.scalar_type(), torch::kComplexDouble);

  auto t_round_trip = torch::fft::irfft(T);
  ASSERT_EQ(t_round_trip.size(0), 128);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

TEST(FFTTest, ihfft) {
  auto T = torch::randn(129, torch::kDouble);
  auto actual = torch::fft::ihfft(T);
  auto expect = torch::fft::ifft(T.to(torch::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

TEST(FFTTest, hfft_ihfft) {
  auto t = torch::randn(64, torch::kComplexDouble);
  t[0] = .5; // Must be purely real to satisfy hermitian symmetry
  auto T = torch::fft::hfft(t, 127);
  ASSERT_EQ(T.size(0), 127);
  ASSERT_EQ(T.scalar_type(), torch::kDouble);

  auto t_round_trip = torch::fft::ihfft(T);
  ASSERT_EQ(t_round_trip.size(0), 64);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kComplexDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
- `test/cpp/api/support.h`
- `torch/torch.h`


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
python test/cpp/api/fft.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

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

- **File Documentation**: `fft.cpp_docs.md`
- **Keyword Index**: `fft.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
