# Documentation: `aten/src/ATen/test/vulkan_quantized_api_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/vulkan_quantized_api_test.cpp`
- **Size**: 137,793 bytes (134.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <random>

#include <cstdio>

using namespace at::native::vulkan::api::utils;

/*
 * TODO: rename this file to something like vulkan_experimental_test and move
 * this under caffe2/fb/vulkan. This file should be used to test experimental
 * features of the Vulkan backend. vulkan_api_test cannot serve this purpose
 * because it cannot link against symbols in the ATen/native/vulkan folder.
 */

namespace {

using namespace at::native::vulkan;

#ifdef USE_VULKAN_FP16_INFERENCE
constexpr float kTolerance = 1e-2;
#else
constexpr float kTolerance = 1e-5;
#endif

bool checkRtol(
    const at::Tensor& diff,
    const std::vector<at::Tensor>& inputs,
    const float tolerated_error = 0) {
  double maxValue = 0.0;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<double>(), maxValue);
  }

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  return diff.abs().max().item<double>() <=
      (tolerance * maxValue + tolerated_error);
}

bool almostEqual(
    const at::Tensor& a,
    const at::Tensor& b,
    const float tolerated_error = 0) {
  return checkRtol(a - b, {a, b}, tolerated_error);
}

/* Unused function
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}
*/

void showRtol(
    const at::Tensor& a,
    const at::Tensor& b,
    long* xpos = nullptr,
    long* ypos = nullptr) {
  const auto diff = (a - b).abs();

  double maxValue = a.abs().max().item<double>();
  maxValue = fmax(b.abs().max().item<double>(), maxValue);

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  const double maxDiff = maxValue * tolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  std::cout << "Max Diff found is: " << diff.max().item<double>() << std::endl;
  if (diff.sizes().size() == 2) {
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      for (const auto x : c10::irange(diff.sizes()[1])) {
        double diff_xy = diff[y][x].item<double>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
          if (diff.max().item<double>() == diff_xy) {
            std::cout << " : " << diff_xy;
            if (xpos && ypos) {
              *xpos = x;
              *ypos = y;
              return;
            }
          }
        } else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

template <class... Args>
inline std::vector<c10::IValue> callOpByHandle(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  return stack;
}

template <class... Args>
inline std::vector<c10::IValue> callOpByName(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  const std::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  assert(op_handle.has_value());
  return callOpByHandle(op_handle.value(), std::forward<Args>(args)...);
}

using namespace at::native::vulkan;
using at::native::vulkan::api::utils::ivec3;
using at::native::vulkan::api::utils::ivec4;
using at::native::vulkan::api::utils::vec4;

std::ostream& operator<<(std::ostream& os, const vec4& v) {
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ", "
     << v.data[3u] << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ivec3& v) {
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ivec4& v) {
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ", "
     << v.data[3u] << ")";
  return os;
}

} // namespace

namespace {

double rand_double(const double min, const double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (std::fabs(max - min) < std::numeric_limits<double>::epsilon()) {
    return min;
  }
  return std::uniform_real_distribution<double>(min, max)(gen);
}

int rand_int(const int min, const int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  return std::uniform_int_distribution<int>(min, max)(gen);
}

int rand_pos_int(const int max_val) {
  TORCH_CHECK(max_val > 0, "max value must be positive");
  return 1 + rand_int(0, max_val);
}

at::Tensor produce_random_tensor(
    const at::IntArrayRef tensor_shape,
    const double s_min = 1.0,
    const double s_max = 100.0,
    const double shift = 0.45) {
  // tensor is randomly generated with values in the range
  // [-shift * s, (1-shift) * s), where s is randomly generated in the range
  // [s_min, s_max]
  // with these default values, s is randomly generated in the range [1, 100]
  // this means that the range of the tensor values could be as narrow as
  // [-0.45, 0.55) or as wide as [-45.0, 55.0)
  TORCH_CHECK(s_min > 0, "scalar lower bound must be positive");
  TORCH_CHECK(s_min <= s_max, "scalar lower bound must be <= upper bound");
  const auto scalar = rand_double(s_min, s_max);
  return scalar *
      (at::rand(tensor_shape, at::device(at::kCPU).dtype(at::kFloat)) - shift);
}

double produce_random_scale(
    const double scale_min = 0.001,
    const double scale_max = 2.0) {
  TORCH_CHECK(scale_min <= scale_max, "scale min must be <= scale max");
  // scale is randomly generated in the range [scale_min, scale_max)
  return rand_double(scale_min, scale_max);
  ;
}

int produce_random_zero_point(const c10::ScalarType dtype) {
  int zero_point = 0;
  switch (dtype) {
    case c10::ScalarType::QUInt8:
      zero_point = rand_int(0, 255);
      break;
    case c10::ScalarType::QInt8:
      zero_point = rand_int(-128, 127);
      break;
    case c10::ScalarType::QInt32:
      zero_point = rand_int(-100000, 100000);
      break;
    default:
      TORCH_CHECK(
          false,
          "Vulkan quantization currently not supported for dtype ",
          dtype);
  }
  return zero_point;
}

std::tuple<double, int> compute_quant_params(
    const at::Tensor& tensor,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  int zero_point_min = 0;
  int zero_point_max = 255;
  if (dtype == c10::ScalarType::QUInt8) {
    zero_point_min = 0;
    zero_point_max = 255;
  } else if (dtype == c10::ScalarType::QInt8) {
    zero_point_min = -128;
    zero_point_max = 127;
  } else {
    TORCH_CHECK(
        false,
        "Computation of quant params only available for dtypes",
        "QUInt8 and QInt8");
  }
  const auto tensor_max = tensor.max().item<double>();
  const auto tensor_min = tensor.min().item<double>();
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/safe_downcast<float>(tensor_min),
      /*max=*/safe_downcast<float>(tensor_max),
      /*qmin=*/zero_point_min,
      /*qmax=*/zero_point_max,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/false);
  return std::tuple<double, int>(q_params.scale, q_params.zero_point);
}

} // namespace

namespace {

class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";
    }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    at::native::vulkan::api::context()->reset_querypool();
#endif
  }

  void TearDown() override {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    try {
      at::native::vulkan::api::context()->querypool().extract_results();
      at::native::vulkan::api::context()->querypool().print_results();
    } catch (const std::exception& e) {
      std::cout << "Could not get querypool results!"
                << " Reason: " << e.what() << std::endl;
    }
#endif
  }
};

at::Tensor cpu_to_vulkan(at::Tensor in_cpu) {
  auto options = in_cpu.options();
  if (options.dtype().toScalarType() == c10::ScalarType::QUInt8 ||
      options.dtype().toScalarType() == c10::ScalarType::QInt8 ||
      options.dtype().toScalarType() == c10::ScalarType::QInt32) {
    auto ret = at::native::vulkan::ops::_empty_affine_quantized(
        in_cpu.sizes(),
        options.dtype().toScalarType(),
        options.layout(),
        options.device(),
        options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point(),
        c10::MemoryFormat::Contiguous);
    at::native::vulkan::ops::copy_(ret, in_cpu);
    return ret;
  } else {
    auto ret = at::empty(in_cpu.sizes(), options);
    at::native::vulkan::ops::copy_(ret, in_cpu);
    return ret;
  }
}

at::Tensor vulkan_to_cpu(at::Tensor vulkan, at::Tensor in_cpu) {
  auto q_options = in_cpu.options();
  if (q_options.dtype().toScalarType() == c10::ScalarType::QUInt8 ||
      q_options.dtype().toScalarType() == c10::ScalarType::QInt8 ||
      q_options.dtype().toScalarType() == c10::ScalarType::QInt32) {
    auto output = at::native::empty_affine_quantized(
        in_cpu.sizes(),
        q_options.dtype().toScalarType(),
        q_options.layout(),
        q_options.device(),
        q_options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point());
    at::native::vulkan::ops::copy_(output, vulkan);
    return output;
  } else {
    auto output = at::empty(in_cpu.sizes(), q_options);
    at::native::vulkan::ops::copy_(output, vulkan);
    return output;
  }
}

TEST_F(VulkanAPITest, uniform_buffer_copy) {
  using namespace at::native::vulkan;

  struct TestStruct {
    int a;
    int b;
    int c;
  };

  TestStruct test_struct{4, 9, 10};

  api::UniformParamsBuffer params(api::context(), test_struct);
  api::UniformParamsBuffer params_copy = params;

  api::MemoryMap copy_mapping(
      params_copy.buffer(), api::MemoryAccessType::READ);

  TestStruct* test_copy_p = copy_mapping.template data<TestStruct>();

  ASSERT_TRUE(test_copy_p->a == test_struct.a);
  ASSERT_TRUE(test_copy_p->b == test_struct.b);
  ASSERT_TRUE(test_copy_p->c == test_struct.c);
}

TEST_F(VulkanAPITest, copy_to_buffer) {
  using namespace at::native::vulkan;

  std::array<at::Tensor, 4> test_tensors = {
      // 4D
      at::rand(
          {7, 17, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 3D
      at::rand({67, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 2D
      at::rand({229, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
      // 1D
      at::rand({1902}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  for (auto in_cpu : test_tensors) {
    vTensor in_vk_copied = ops::to_vulkan(in_cpu, api::StorageType::BUFFER);
    at::Tensor out_copied = ops::from_vulkan(in_vk_copied);

    const auto check_copy = almostEqual(out_copied, in_cpu);

    if (!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes() << "with dtype"
                << in_cpu.dtype() << std::endl;
    }

    ASSERT_TRUE(check_copy);
  }
}

TEST_F(VulkanAPITest, copy_to_buffer_channels_last) {
  using namespace at::native::vulkan;

  at::TensorOptions options(at::kCPU);
  options = options.dtype(at::kFloat);

  std::array<at::Tensor, 1> test_tensors = {
      // 4D
      at::rand({7, 17, 134, 213}, options).to(at::MemoryFormat::ChannelsLast),
  };

  for (auto in_cpu : test_tensors) {
    vTensor in_vk_copied = ops::to_vulkan(in_cpu, api::StorageType::BUFFER);
    at::Tensor out_copied = ops::from_vulkan(in_vk_copied);

    const auto check_copy = almostEqual(out_copied, in_cpu);

    if (!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes() << "with dtype"
                << in_cpu.dtype() << std::endl;
    }

    ASSERT_TRUE(check_copy);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_support_vulkan) {
  const double scale = 0.1;
  const int zero_point = 10;

  auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 12 -
      6;
  auto in_cpu_quantized = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);

  auto in_vulkan_quantized = cpu_to_vulkan(in_cpu_quantized);
  at::native::vulkan::api::PipelineBarrier pipeline_barrier{};
  at::native::vulkan::vTensor& v_self =
      at::native::vulkan::ops::convert(in_vulkan_quantized);
  if (in_cpu.dtype() == c10::kQUInt8) {
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::READ);
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::WRITE);
  }
  auto output = vulkan_to_cpu(in_vulkan_quantized, in_cpu_quantized);
  const auto check = almostEqual(
      at::native::int_repr_quantized_cpu(in_cpu_quantized),
      at::native::int_repr_quantized_cpu(output));

  if (!check) {
    showRtol(
        at::native::int_repr_quantized_cpu(in_cpu_quantized),
        at::native::int_repr_quantized_cpu(output));
  }

  ASSERT_TRUE(check);
}

void test_cpu_to_vulkan_and_vulkan_to_cpu(
    const at::IntArrayRef input_shape,
    const double scale,
    const int zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  // produce random quantized cpu tensor
  auto in_cpu = produce_random_tensor(input_shape);
  auto in_q_cpu = at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // copy quantized cpu tensor to vulkan
  auto in_q_cpu_vk = cpu_to_vulkan(in_q_cpu);

  // copy quantized vulkan tensor to cpu
  auto out_q_cpu = vulkan_to_cpu(in_q_cpu_vk, in_q_cpu);

  // check that the copy equals the original
  const auto diff = at::native::int_repr_quantized_cpu(in_q_cpu) -
      at::native::int_repr_quantized_cpu(out_q_cpu);

  const int error = diff.abs().max().item<int>();

  const auto check = (error == 0);

  if (!check) {
    std::cout << "Copy to vulkan and back to cpu failed with input shape: "
              << input_shape << " scale: " << scale
              << " and zero point: " << zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }

  ASSERT_TRUE(check);
}

void test_cpu_to_vulkan_and_vulkan_to_cpu_random(const c10::ScalarType dtype) {
  const double scale = produce_random_scale();
  const int zero_point = produce_random_zero_point(dtype);
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  test_cpu_to_vulkan_and_vulkan_to_cpu(tensor_shape, scale, zero_point, dtype);
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_quint8) {
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, 21, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, 120, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, 15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, 43, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_qint8) {
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, -120, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, -15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_cpu_to_vulkan_and_vulkan_to_cpu_qint32) {
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 4, 1}, 0.228, -12023, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 11, 17}, 0.073, -15, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1572, 102, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 12, 17}, 0.147, -156, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 17, 12}, 0.129, 10448, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({2, 4, 17, 12}, 0.137, -10, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43267, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1345, 196, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({4, 4, 25, 29}, 0.129, -19489, dtype);
  test_cpu_to_vulkan_and_vulkan_to_cpu({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_vulkan_to_cpu_random(dtype);
  }
}

void test_cpu_to_vulkan_and_dequantize(
    const at::IntArrayRef input_shape,
    const double scale,
    const int zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8) {
  // produce random quantized cpu tensor
  auto in_cpu = produce_random_tensor(input_shape);
  auto in_q_cpu = at::quantize_per_tensor(in_cpu, scale, zero_point, dtype);

  // copy quantized cpu tensor to vulkan
  auto in_q_cpu_vk = cpu_to_vulkan(in_q_cpu);

  // dequantize tensors
  const auto out_cpu_deq = at::dequantize(in_q_cpu);
  const auto out_vk_deq = at::dequantize(in_q_cpu_vk);
  const auto out_vk_deq_cpu = out_vk_deq.cpu();

  // check dequantized tensors are equal
  const auto check = almostEqual(out_cpu_deq, out_vk_deq_cpu);

  if (!check) {
    const auto error =
        at::abs(out_vk_deq_cpu - out_cpu_deq).max().item<float>();
    std::cout << "Copy cpu to vulkan and dequantize failed with input shape: "
              << input_shape << " scale: " << scale
              << " and zero point: " << zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }
  ASSERT_TRUE(check);
}

void test_cpu_to_vulkan_and_dequantize_random(const c10::ScalarType dtype) {
  const double scale = produce_random_scale();
  const int zero_point = produce_random_zero_point(dtype);
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  test_cpu_to_vulkan_and_dequantize(tensor_shape, scale, zero_point, dtype);
}

TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_quint8) {
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.07, 15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, 43, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}

TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_qint8) {
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.07, -15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.1, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, -43, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.1, -19, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}

TEST_F(VulkanAPITest, cpu_to_vulkan_and_dequantize_qint32) {
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 4, 1}, 0.228, -12023, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 11, 17}, 0.073, -15, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 12, 17}, 0.1572, 102, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 12, 17}, 0.147, -156, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 17, 12}, 0.129, 10448, dtype);
  test_cpu_to_vulkan_and_dequantize({2, 4, 17, 12}, 0.137, -10, dtype);
  test_cpu_to_vulkan_and_dequantize({1, 1, 10, 14}, 0.0001, 101, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 14}, 0.009, -43267, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_cpu_to_vulkan_and_dequantize({3, 5, 25, 29}, 0.1345, 196, dtype);
  test_cpu_to_vulkan_and_dequantize({4, 4, 25, 29}, 0.129, -19489, dtype);
  test_cpu_to_vulkan_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_cpu_to_vulkan_and_dequantize_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  auto output_for_quantized_vulkan = vulkan_to_cpu(out_vulkan, out_cpu);

  int rtol = 1;
  const auto check = at::allclose(
      at::native::int_repr_quantized_cpu(out_cpu),
      at::native::int_repr_quantized_cpu(output_for_quantized_vulkan),
      rtol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

void test_quantize_per_tensor_and_vulkan_to_cpu(
    const at::IntArrayRef input_shape,
    const double input_scale,
    const int input_zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8,
    const int tolerance = 1) {
  // tolerance = 1, to allow for precision differences after dividing by random
  // scale which could result on a difference of 1 unit in the quantized result

  at::Tensor input = produce_random_tensor(input_shape);

  // quantize tensor
  at::Tensor out_q_cpu =
      at::quantize_per_tensor(input, input_scale, input_zero_point, dtype);

  at::Tensor out_q_vk = at::quantize_per_tensor(
      input.vulkan(), input_scale, input_zero_point, dtype);

  // copy vulkan tensor to cpu
  at::Tensor out_q_vk_cpu = vulkan_to_cpu(out_q_vk, out_q_cpu);

  const auto diff = at::native::int_repr_quantized_cpu(out_q_vk_cpu) -
      at::native::int_repr_quantized_cpu(out_q_cpu);

  const int error = diff.abs().max().item<int>();

  const auto check = (error <= tolerance);

  if (!check) {
    std::cout << "Quantize and copy to cpu failed with input shape: "
              << input_shape << " scale: " << input_scale
              << " and zero point: " << input_zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }

  ASSERT_TRUE(check);
}

void test_quantize_per_tensor_and_vulkan_to_cpu_random(
    const c10::ScalarType dtype) {
  const double scale = produce_random_scale();
  const int zero_point = produce_random_zero_point(dtype);
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  test_quantize_per_tensor_and_vulkan_to_cpu(
      tensor_shape, scale, zero_point, dtype);
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_quint8) {
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, 21, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, 120, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, 15, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 10, 14}, 0.0001, 101, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, 43, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {11, 17, 25, 29}, 0.027, 89, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 16, 77, 54}, 0.204173, 229, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_qint8) {
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 4, 1}, 0.2, -120, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.04, 97, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.07, -15, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 12, 17}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 10, 14}, 0.0001, 101, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 14}, 0.009, -43, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 9, 17}, 0.1, -19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({4, 4, 25, 29}, 0.1, -19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {11, 17, 25, 29}, 0.027, 89, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 16, 77, 54}, 0.204173, 229, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

// TODO: Fix vulkan to cpu on Android
TEST_F(VulkanAPITest, DISABLED_quantize_per_tensor_and_vulkan_to_cpu_qint32) {
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 4, 1}, 0.228, -12023, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({1, 1, 11, 17}, 0.073, -15, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 12, 17}, 0.1572, 102, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 12, 17}, 0.147, -156, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 17, 12}, 0.129, 10448, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({2, 4, 17, 12}, 0.137, -10, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {1, 1, 10, 14}, 0.0001, 101, dtype, 1);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 10, 14}, 0.009, -43267, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 5, 25, 29}, 0.1345, 196, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {4, 4, 25, 29}, 0.129, -19489, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {11, 17, 25, 29}, 0.027, 89, dtype);
  test_quantize_per_tensor_and_vulkan_to_cpu(
      {3, 16, 77, 54}, 0.204173, 229, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_vulkan_to_cpu_random(dtype);
  }
}

TEST_F(VulkanAPITest, quantize_dequantize) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;
  // quantize tensors
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // dequantize tensors
  const auto out_cpu_deq = at::dequantize(out_cpu);
  const auto out_vulkan_deq = at::native::vulkan::ops::dequantize(out_vulkan);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu);

  float rtol = 1;
  float atol = 0.5;
  const auto check =
      at::allclose(in_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);

  const auto check_two =
      at::allclose(out_cpu_deq, output_for_dequantized_vulkan, rtol, atol);

  if (!check_two) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check_two);
}

void test_quantize_per_tensor_and_dequantize(
    const at::IntArrayRef input_shape,
    const double input_scale,
    const int input_zero_point,
    const c10::ScalarType dtype = c10::ScalarType::QUInt8,
    bool use_qparams = false) {
  at::Tensor input = produce_random_tensor(input_shape);

  at::Tensor input_scale_qparam = at::empty({1});
  input_scale_qparam[0] = input_scale;
  at::Tensor input_zero_point_qparam = at::empty({1});
  input_zero_point_qparam[0] = input_zero_point;

  // quantize tensors
  at::Tensor out_q_cpu = use_qparams
      ? at::quantize_per_tensor(
            input, input_scale_qparam, input_zero_point_qparam, dtype)
      : at::quantize_per_tensor(input, input_scale, input_zero_point, dtype);
  at::Tensor out_q_vk = use_qparams
      ? at::quantize_per_tensor(
            input.vulkan(), input_scale_qparam, input_zero_point_qparam, dtype)
      : at::quantize_per_tensor(
            input.vulkan(), input_scale, input_zero_point, dtype);

  // dequantize tensors
  const auto out_cpu_deq = at::dequantize(out_q_cpu);
  const auto out_vk_deq = at::dequantize(out_q_vk);
  const auto out_vk_deq_cpu = out_vk_deq.cpu();

  // check dequantized tensor are equal
  const float tolerance = safe_downcast<float>(input_scale);
  // tolerated error = scale, to allow for precision differences after dividing
  // by random scale, which could result on a difference of 1 unit in the
  // quantized result.
  const auto check = almostEqual(out_cpu_deq, out_vk_deq_cpu, tolerance);

  if (!check) {
    const auto error =
        at::abs(out_vk_deq_cpu - out_cpu_deq).max().item<float>();
    std::cout << "Quantize and Dequantize failed with input shape: "
              << input_shape << " scale: " << input_scale
              << " and zero point: " << input_zero_point << std::endl;
    std::cout << "Error: " << error << std::endl;
  }
  ASSERT_TRUE(check);
}

void test_quantize_per_tensor_and_dequantize_random(
    const c10::ScalarType dtype,
    bool use_qparams = false) {
  const double scale = produce_random_scale();
  const int zero_point = produce_random_zero_point(dtype);
  const at::IntArrayRef tensor_shape = {
      rand_pos_int(30), rand_pos_int(30), rand_pos_int(100), rand_pos_int(100)};
  test_quantize_per_tensor_and_dequantize(
      tensor_shape, scale, zero_point, dtype, use_qparams);
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_quint8) {
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, 15, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, 43, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype);
  }
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_quint8_qparams) {
  const c10::ScalarType dtype = c10::ScalarType::QUInt8;
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, 21, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, 120, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 11, 17}, 0.07, 15, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 10, 14}, 0.001, 101, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 14}, 0.009, 43, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {11, 17, 25, 29}, 0.027, 89, dtype, true);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint8) {
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.07, -15, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype);
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.1, -10, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, -43, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.1, -19, dtype);
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype);
  }
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint8_qparams) {
  const c10::ScalarType dtype = c10::ScalarType::QInt8;
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.2, -120, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.3, 87, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.04, 97, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 11, 17}, 0.07, -15, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 12, 17}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.1, 10, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {2, 4, 17, 12}, 0.1, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 10, 14}, 0.001, 101, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 14}, 0.009, -43, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1, -19, dtype, true);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {4, 4, 25, 29}, 0.1, -19, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {11, 17, 25, 29}, 0.027, 89, dtype, true);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint32) {
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 1}, 0.13, -21123, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 1, 4}, 0.339, 8734, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 4, 1}, 0.228, -12023, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 7, 7}, 0.338, 8723, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 8, 8}, 0.193, -1023, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 8, 8}, 0.0449, 972, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 11, 17}, 0.073, -15, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 12, 17}, 0.1572, 102, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 12, 17}, 0.147, -156, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 17, 12}, 0.129, 10448, dtype);
  test_quantize_per_tensor_and_dequantize({2, 4, 17, 12}, 0.137, -10, dtype);
  test_quantize_per_tensor_and_dequantize({1, 1, 10, 14}, 0.001, 101, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 14}, 0.009, -43267, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 10, 15}, 0.1243, 19, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 9, 17}, 0.1889, -19784, dtype);
  test_quantize_per_tensor_and_dequantize({3, 5, 25, 29}, 0.1345, 196, dtype);
  test_quantize_per_tensor_and_dequantize({4, 4, 25, 29}, 0.129, -19489, dtype);
  test_quantize_per_tensor_and_dequantize({11, 17, 25, 29}, 0.027, 89, dtype);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype);
  }
}

TEST_F(VulkanAPITest, quantize_per_tensor_and_dequantize_qint32_qparams) {
  const c10::ScalarType dtype = c10::ScalarType::QInt32;
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 1, 1}, 0.13, -21123, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 1, 4}, 0.339, 8734, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 4, 1}, 0.228, -12023, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 7, 7}, 0.338, 8723, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 8, 8}, 0.193, -1023, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 8, 8}, 0.0449, 972, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 11, 17}, 0.073, -15, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 12, 17}, 0.1572, 102, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 12, 17}, 0.147, -156, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 17, 12}, 0.129, 10448, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {2, 4, 17, 12}, 0.137, -10, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {1, 1, 10, 14}, 0.001, 101, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 14}, 0.009, -43267, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 10, 15}, 0.1243, 19, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {4, 4, 9, 17}, 0.1889, -19784, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {3, 5, 25, 29}, 0.1345, 196, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {4, 4, 25, 29}, 0.129, -19489, dtype, true);
  test_quantize_per_tensor_and_dequantize(
      {11, 17, 25, 29}, 0.027, 89, dtype, true);

  for (int i = 0; i < 20; i += 1) {
    test_quantize_per_tensor_and_dequantize_random(dtype, true);
  }
}

TEST_F(VulkanAPITest, quantized_add) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add", "", out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast) {
  const auto in_cpu =
      at::rand({2, 13, 1, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 98 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `at`

**Classes/Structs**: `VulkanAPITest`, `TestStruct`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/dispatch/Dispatcher.h`
- `ATen/native/quantized/PackedParams.h`
- `ATen/native/quantized/cpu/QuantUtils.h`
- `ATen/native/vulkan/api/Utils.h`
- `ATen/native/vulkan/api/api.h`
- `ATen/native/vulkan/impl/Packing.h`
- `ATen/native/vulkan/ops/Common.h`
- `ATen/native/vulkan/ops/Convert.h`
- `ATen/native/vulkan/ops/Copy.h`
- `ATen/native/vulkan/ops/Factory.h`
- `ATen/native/vulkan/ops/Mm.h`
- `ATen/native/vulkan/ops/QuantizedFunctions.h`
- `c10/util/irange.h`
- `gtest/gtest.h`
- `math.h`
- `cstring`
- `iostream`
- `random`
- `cstdio`


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
python aten/src/ATen/test/vulkan_quantized_api_test.cpp
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

- **File Documentation**: `vulkan_quantized_api_test.cpp_docs.md`
- **Keyword Index**: `vulkan_quantized_api_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
