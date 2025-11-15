# Documentation: `docs/aten/src/ATen/test/cpu_rng_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/cpu_rng_test.cpp_docs.md`
- **Size**: 21,964 bytes (21.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/cpu_rng_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/cpu_rng_test.cpp`
- **Size**: 19,173 bytes (18.72 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <ATen/test/rng_test.h>
#include <ATen/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <torch/library.h>
#include <optional>
#include <torch/all.h>
#include <stdexcept>

using namespace at;

#ifndef ATEN_CPU_STATIC_DISPATCH
namespace {

constexpr auto kCustomRNG = DispatchKey::CustomRNGKeyId;

struct TestCPUGenerator : public c10::GeneratorImpl {
  TestCPUGenerator(uint64_t value) : GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(kCustomRNG)}, value_(value) { }
  ~TestCPUGenerator() override = default;
  uint32_t random() { return value_; }
  uint64_t random64() { return value_; }
  std::optional<float> next_float_normal_sample() { return next_float_normal_sample_; }
  std::optional<double> next_double_normal_sample() { return next_double_normal_sample_; }
  void set_next_float_normal_sample(std::optional<float> randn) { next_float_normal_sample_ = randn; }
  void set_next_double_normal_sample(std::optional<double> randn) { next_double_normal_sample_ = randn; }
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  void set_offset(uint64_t offset) override { throw std::runtime_error("not implemented"); }
  uint64_t get_offset() const override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static DeviceType device_type() { return DeviceType::CPU; }

  uint64_t value_;
  std::optional<float> next_float_normal_sample_;
  std::optional<double> next_double_normal_sample_;
};

// ==================================================== Random ========================================================

Tensor& random_(Tensor& self, std::optional<Generator> generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<Generator> generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, std::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Normal ========================================================

Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<native::templates::cpu::NormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

Tensor& normal_Tensor_float_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_float_Tensor_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor& normal_Tensor_Tensor_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(output, mean, std, gen);
}

Tensor normal_Tensor_float(const Tensor& mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_float_Tensor(double mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<native::templates::cpu::NormalKernel, TestCPUGenerator>(mean, std, gen);
}

// ==================================================== Uniform =======================================================

Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> generator) {
  return at::native::templates::uniform_impl_<native::templates::cpu::UniformKernel, TestCPUGenerator>(self, from, to, generator);
}

// ==================================================== Cauchy ========================================================

Tensor& cauchy_(Tensor& self, double median, double sigma, std::optional<Generator> generator) {
  return at::native::templates::cauchy_impl_<native::templates::cpu::CauchyKernel, TestCPUGenerator>(self, median, sigma, generator);
}

// ================================================== LogNormal =======================================================

Tensor& log_normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::log_normal_impl_<native::templates::cpu::LogNormalKernel, TestCPUGenerator>(self, mean, std, gen);
}

// ================================================== Geometric =======================================================

Tensor& geometric_(Tensor& self, double p, std::optional<Generator> gen) {
  return at::native::templates::geometric_impl_<native::templates::cpu::GeometricKernel, TestCPUGenerator>(self, p, gen);
}

// ================================================== Exponential =====================================================

Tensor& exponential_(Tensor& self, double lambda, std::optional<Generator> gen) {
  return at::native::templates::exponential_impl_<native::templates::cpu::ExponentialKernel, TestCPUGenerator>(self, lambda, gen);
}

// ================================================== Bernoulli =======================================================

Tensor& bernoulli_Tensor(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p_, gen);
}

Tensor& bernoulli_float(Tensor& self, double p, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(self, p, gen);
}

Tensor& bernoulli_out(const Tensor& self, std::optional<Generator> gen, Tensor& result) {
  return at::native::templates::bernoulli_out_impl<native::templates::cpu::BernoulliKernel, TestCPUGenerator>(result, self, gen);
}

TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // Random
  m.impl("random_.from",             random_from_to);
  m.impl("random_.to",               random_to);
  m.impl("random_",                  random_);
  // Normal
  m.impl("normal_",                  normal_);
  m.impl("normal.Tensor_float_out",  normal_Tensor_float_out);
  m.impl("normal.float_Tensor_out",  normal_float_Tensor_out);
  m.impl("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);
  m.impl("normal.Tensor_float",      normal_Tensor_float);
  m.impl("normal.float_Tensor",      normal_float_Tensor);
  m.impl("normal.Tensor_Tensor",     normal_Tensor_Tensor);
  m.impl("uniform_",                 uniform_);
  // Cauchy
  m.impl("cauchy_",                  cauchy_);
  // LogNormal
  m.impl("log_normal_",              log_normal_);
  // Geometric
  m.impl("geometric_",               geometric_);
  // Exponential
  m.impl("exponential_",             exponential_);
  // Bernoulli
  m.impl("bernoulli.out",            bernoulli_out);
  m.impl("bernoulli_.Tensor",        bernoulli_Tensor);
  m.impl("bernoulli_.float",         bernoulli_float);
}

class RNGTest : public ::testing::Test {
};

static constexpr auto MAGIC_NUMBER = 424242424242424242ULL;

// ==================================================== Random ========================================================

TEST_F(RNGTest, RandomFromTo) {
  const at::Device device("cpu");
  test_random_from_to<TestCPUGenerator, torch::kBool, bool>(device);
  test_random_from_to<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random_from_to<TestCPUGenerator, torch::kFloat64, double>(device);
}

TEST_F(RNGTest, Random) {
  const at::Device device("cpu");
  test_random<TestCPUGenerator, torch::kBool, bool>(device);
  test_random<TestCPUGenerator, torch::kUInt8, uint8_t>(device);
  test_random<TestCPUGenerator, torch::kInt8, int8_t>(device);
  test_random<TestCPUGenerator, torch::kInt16, int16_t>(device);
  test_random<TestCPUGenerator, torch::kInt32, int32_t>(device);
  test_random<TestCPUGenerator, torch::kInt64, int64_t>(device);
  test_random<TestCPUGenerator, torch::kFloat32, float>(device);
  test_random<TestCPUGenerator, torch::kFloat64, double>(device);
}

// This test proves that Tensor.random_() distribution is able to generate unsigned 64 bit max value(64 ones)
// https://github.com/pytorch/pytorch/issues/33299
TEST_F(RNGTest, Random64bits) {
  auto gen = at::make_generator<TestCPUGenerator>(std::numeric_limits<uint64_t>::max());
  auto actual = torch::empty({1}, torch::kInt64);
  actual.random_(std::numeric_limits<int64_t>::min(), std::nullopt, gen);
  ASSERT_EQ(static_cast<uint64_t>(actual[0].item<int64_t>()), std::numeric_limits<uint64_t>::max());
}

// ==================================================== Normal ========================================================

TEST_F(RNGTest, Normal) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  actual.normal_(mean, std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, mean, torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, torch::full({10}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor_out) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  at::normal_out(actual, torch::full({10}, mean), torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_float_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(mean, torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_float) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(torch::full({10}, mean), std, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Normal_Tensor_Tensor) {
  const auto mean = 123.45;
  const auto std = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::normal(torch::full({10}, mean), torch::full({10}, std), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::normal_kernel(expected, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Uniform =======================================================

TEST_F(RNGTest, Uniform) {
  const auto from = -24.24;
  const auto to = 42.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.uniform_(from, to, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::uniform_kernel(iter, from, to, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Cauchy ========================================================

TEST_F(RNGTest, Cauchy) {
  const auto median = 123.45;
  const auto sigma = 67.89;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.cauchy_(median, sigma, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::cauchy_kernel(iter, median, sigma, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== LogNormal =======================================================

TEST_F(RNGTest, LogNormal) {
  const auto mean = 12.345;
  const auto std = 6.789;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({10});
  actual.log_normal_(mean, std, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::log_normal_kernel(iter, mean, std, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Geometric =======================================================

TEST_F(RNGTest, Geometric) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.geometric_(p, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::geometric_kernel(iter, p, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ================================================== Exponential =====================================================

TEST_F(RNGTest, Exponential) {
  const auto lambda = 42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.exponential_(lambda, gen);

  auto expected = torch::empty_like(actual);
  auto iter = TensorIterator::nullary_op(expected);
  native::templates::cpu::exponential_kernel(iter, lambda, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

// ==================================================== Bernoulli =====================================================

TEST_F(RNGTest, Bernoulli_Tensor) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.bernoulli_(torch::full({3,3}, p), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_scalar) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  actual.bernoulli_(p, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::bernoulli(torch::full({3,3}, p), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_2) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::full({3,3}, p).bernoulli(gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_p) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = at::bernoulli(torch::empty({3, 3}), p, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_p_2) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3}).bernoulli(p, gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, p, check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST_F(RNGTest, Bernoulli_out) {
  const auto p = 0.42;
  auto gen = at::make_generator<TestCPUGenerator>(MAGIC_NUMBER);

  auto actual = torch::empty({3, 3});
  at::bernoulli_out(actual, torch::full({3,3}, p), gen);

  auto expected = torch::empty_like(actual);
  native::templates::cpu::bernoulli_kernel(expected, torch::full({3,3}, p), check_generator<TestCPUGenerator>(gen));

  ASSERT_TRUE(torch::allclose(actual, expected));
}
}
#endif // ATEN_CPU_STATIC_DISPATCH

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TestCPUGenerator`, `RNGTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/test/rng_test.h`
- `ATen/Generator.h`
- `c10/core/GeneratorImpl.h`
- `ATen/Tensor.h`
- `ATen/native/DistributionTemplates.h`
- `ATen/native/cpu/DistributionTemplates.h`
- `torch/library.h`
- `optional`
- `torch/all.h`
- `stdexcept`


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
python aten/src/ATen/test/cpu_rng_test.cpp
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

- **File Documentation**: `cpu_rng_test.cpp_docs.md`
- **Keyword Index**: `cpu_rng_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/aten/src/ATen/test/cpu_rng_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cpu_rng_test.cpp_docs.md_docs.md`
- **Keyword Index**: `cpu_rng_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
