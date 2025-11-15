# Documentation: `aten/src/ATen/test/vulkan_api_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/vulkan_api_test.cpp`
- **Size**: 266,303 bytes (260.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#ifdef USE_VULKAN_API

// @lint-ignore-every CLANGTIDY

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/api.h>
#include <c10/util/irange.h>
#include <c10/util/ArrayRef.h>

// TODO: These functions should move to a common place.

namespace {

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float kTolerance = 1e-2;
#else
  constexpr float kTolerance = 1e-5;
#endif

bool checkRtol(const at::Tensor& diff, float maxTolerance) {
  if (diff.numel() == 0) {
    return true;
  }
  return diff.abs().max().item<float>() <= maxTolerance;
}

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  if (diff.numel() == 0) {
    return true;
  }
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return checkRtol(diff, kTolerance * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool checkHardShrink(
    const at::Tensor& ref, const at::Tensor& out, const float clamp_thresh) {
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  float abs_clamp_thresh = std::abs(clamp_thresh);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);

    // For values near the clamp threshold, results may be ambiguous.
    float distance_from_thresh = std::abs(std::abs(ref_val) - abs_clamp_thresh);
    if (distance_from_thresh < kTolerance * abs_clamp_thresh) {
      if (out_val != 0.0f) {
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    else if (std::abs(ref_val) < std::abs(abs_clamp_thresh)) {
      if (out_val != 0.0f) {
        return false;
      }
    }
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  return true;
}

bool checkThreshold(
    const at::Tensor& ref,
    const at::Tensor& out,
    const float clamp_thresh,
    const float value) {
  float* ref_ptr = ref.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  float ref_max = ref.abs().max().item<float>();
  float out_max = out.abs().max().item<float>();
  float max_val = std::fmax(ref_max, out_max);

  for (int i = 0; i < ref.numel(); ++i) {
    float ref_val = ref_ptr[i];
    float out_val = out_ptr[i];

    float abs_diff = std::abs(ref_val - out_val);
    float val_diff = std::abs(out_val - value);

    // For values near the clamp threshold, results may be ambiguous.
    float distance_from_thresh = std::abs(std::abs(ref_val) - clamp_thresh);
    if (distance_from_thresh < kTolerance * clamp_thresh) {
      if (val_diff >= kTolerance * value) {
        if (abs_diff >= kTolerance * max_val) {
          return false;
        }
      }
    }
    else if (std::abs(ref_val) < std::abs(clamp_thresh)) {
      if (val_diff >= kTolerance * value) {
        return false;
      }
    }
    else if (abs_diff >= kTolerance * max_val) {
      return false;
    }
  }
  return true;
}

void showRtol(const at::Tensor& a, const at::Tensor& b) {
  const auto diff = (a - b).abs();

  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

  const float maxDiff = maxValue * kTolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  if (diff.sizes().size() == 2) {
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      for (const auto x : c10::irange(diff.sizes()[1])) {
        float diff_xy = diff[y][x].item<float>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        }
        else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}


static void gen_allpermutations(std::vector<std::vector<int64_t>>& out, std::vector<int64_t> in, unsigned i) {
  // generate all permutations of a given dims
  if (i == in.size()) {
    out.push_back(in);
  }
  else {
    for (const auto j : c10::irange(i, in.size())) {
      std::swap(in[i], in[j]);
      gen_allpermutations(out, in, i + 1);
    }
  }
}

static void gen_all_subsets(
    std::vector<std::vector<int64_t>>& out,
    int64_t n,
    unsigned i,
    std::vector<int64_t> curr) {
  // generate all subsets of set {0,...,n - 1} through backtracking
  if (i == n) {
    out.push_back(curr);
  } else {
    curr.push_back(i);
    gen_all_subsets(out, n, i + 1, curr);
    curr.pop_back();
    gen_all_subsets(out, n, i + 1, curr);
  }
}

static void slice_test(
    const std::vector<int64_t>& size,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  // Arrange
  const auto in_cpu = at::rand(size, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  // Act
  const auto out_cpu = at::slice(in_cpu, dim, start, end, step);
  const auto out_vulkan = at::slice(in_vulkan, dim, start, end, step);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

static void slice_tests(const std::unordered_map<int64_t, std::vector<int64_t>>& dim2sizes) {
  for (const auto& dim2size : dim2sizes) {
    slice_test(dim2size.second, dim2size.first, 10, 30, 1);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:30:1]
    slice_test(dim2size.second, dim2size.first, 10, 30, 7);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:30:7]
    slice_test(dim2size.second, dim2size.first, 10, 50, 2);         // i.e., 4D tensor's equivalent indexing = [:,:,:,10:50:2] with end=out of range
    slice_test(dim2size.second, dim2size.first, -60, 60, 2);        // i.e., 4D tensor's equivalent indexing = [:,:,:,-60:60:2] with start/end=out of range
    slice_test(dim2size.second, dim2size.first, -30, -10, 1);       // i.e., 4D tensor's equivalent indexing = [:,:,:,-30:-10:1] with negative start/end
    slice_test(dim2size.second, dim2size.first, 0, INT64_MAX, 1);   // i.e., 4D 's equivalent indexing = [:,:,:,0:9223372036854775807:1] with end=INT64_MAX
    slice_test(dim2size.second, dim2size.first, -10, INT64_MAX, 1); // i.e., 4D 's equivalent indexing = [:,:,:,-10:9223372036854775807:1] with negative start and end=INT64_MAX
    // This triggers a SymInt assert since [-2^63, -2^62-1] range is reserved for packed symints
    //slice_test(dim2size.second, dim2size.first, INT64_MIN, INT64_MAX, 1); // i.e., 4D 's equivalent indexing = [:,:,:,-9223372036854775808:9223372036854775807:1] with start=INT64_MIN and end=INT64_MAX
    slice_test(dim2size.second, dim2size.first, {}, {}, 1);         // i.e., 4D 's equivalent indexing = [:,:,:,::1] with empty start/end
  }
}

static void clone_test(const std::vector<int64_t>& size, std::optional<at::MemoryFormat> optional_memory_format) {
  // Arrange
  const auto in_cpu = at::rand(size, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  // Act
  const auto out_cpu = at::clone(in_cpu, optional_memory_format);
  const auto out_vulkan = at::clone(in_vulkan, optional_memory_format);

  // Assert
  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
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

} // namespace

namespace {

class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() {
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";
    }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      at::native::vulkan::api::context()->reset_querypool();
    }
#endif
  }

  void TearDown() {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    if (at::native::vulkan::api::context()->op_profiling_enabled()) {
      try {
        at::native::vulkan::api::context()->querypool().extract_results();
        at::native::vulkan::api::context()->querypool().print_results();
      } catch (const std::exception& e) {
        std::cout << "Could not get querypool results!"
                  << " Reason: " << e.what() << std::endl;
      }
    }
#endif
  }
};

TEST_F(VulkanAPITest, zero_size_tensor) {
  auto cpu = at::rand({1, 0, 0}, at::device(at::kCPU).dtype(at::kFloat));
  auto vk = cpu.vulkan();
  auto out_vk = vk.cpu();
  ASSERT_TRUE(at::equal(out_vk, cpu));
}

TEST_F(VulkanAPITest, zero_size_tensor_numel) {
  auto vk = at::rand({18, 0, 5}, at::device(at::kVulkan).dtype(at::kFloat));
  ASSERT_TRUE(vk.numel() == 0);
}

TEST_F(VulkanAPITest, zero_dim_tensor_1) {
  auto cpu = at::rand({}, at::device(at::kCPU).dtype(at::kFloat));
  auto vv = cpu.item<float>();

  auto vk = cpu.vulkan();
  auto out_vk = vk.cpu();
  ASSERT_TRUE(almostEqual(cpu, out_vk));

  auto vk_vv = out_vk.item<float>();
  EXPECT_NEAR(vv, vk_vv, kTolerance);
}

TEST_F(VulkanAPITest, zero_dim_tensor_2) {
  float v = 3.14f;
  auto cpu = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat)) + v;
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;

  ASSERT_TRUE(almostEqual(cpu, vk.cpu()));
}

TEST_F(VulkanAPITest, zero_dim_tensor_3) {
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat));

  ASSERT_TRUE(vk.cpu().item<float>() == 0.0f);
}

TEST_F(VulkanAPITest, local_scalar_dense) {
  float v = 8.31f;
  // Force the zero-dim tensor to a non-zero constant v.
  auto vk = at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)) + v;
  c10::Scalar scalar = at::_local_scalar_dense(vk);
  EXPECT_NEAR(v, scalar.toFloat(), kTolerance);
}

TEST_F(VulkanAPITest, copy_to_texture) {
  using namespace at::native::vulkan;
  at::Tensor test_tensors[] = {
    // 4D
    at::rand({7, 17, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 3D
    at::rand({67, 134, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 2D
    at::rand({229, 213}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
    // 1D
    at::rand({1902}, at::TensorOptions(at::kCPU).dtype(at::kFloat)),
  };

  for (auto in_cpu : test_tensors) {
    at::Tensor in_vk_copied = in_cpu.vulkan();
    at::Tensor out_copied = in_vk_copied.cpu();

    const auto check_copy = almostEqual(out_copied, in_cpu);

    if(!check_copy) {
      std::cout << "Copy failed on size " << in_cpu.sizes()
                << "with dtype" << in_cpu.dtype() << std::endl;
    }

    ASSERT_TRUE(check_copy);
  }
}

void test_copy_to_texture_bool(const at::IntArrayRef input_shape) {
  using namespace at::native::vulkan;
  auto cpu = at::randint(0, 2, input_shape, at::TensorOptions(at::kCPU).dtype(at::kBool));
  auto in_vulkan = cpu.vulkan();

  auto out_vulkan = in_vulkan.cpu();
  auto check = at::equal(cpu, out_vulkan.cpu());

  if (!check) {
    std::cout << "Copy texture to bool failed on input_shape " << input_shape << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_hw) {
  // Uses the shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 != 0,  ch < 4
  test_copy_to_texture_bool({5, 1, 2, 2});
  test_copy_to_texture_bool({17, 2, 4, 2});
  test_copy_to_texture_bool({9, 3, 3, 8});

  // ch % 4 != 0, ch > 5
  test_copy_to_texture_bool({7, 17, 4, 8});
  test_copy_to_texture_bool({8, 6, 2, 4});
  test_copy_to_texture_bool({13, 31, 4, 57});

  // 3d, 2d, 1d
  test_copy_to_texture_bool({17, 31, 4});
  test_copy_to_texture_bool({64, 16});
  test_copy_to_texture_bool({8});
}

TEST_F(VulkanAPITest, copy_to_texture_bool_mul4_chw) {
  // Uses the shader: image_to_nchw_quantized_mul4 ((H * W) % 4 == 0)
  // ch % 4 == 0
  test_copy_to_texture_bool({5, 16, 2, 16});
  test_copy_to_texture_bool({8, 8, 2, 2});
  test_copy_to_texture_bool({16, 31, 4});
}

TEST_F(VulkanAPITest, copy_to_texture_bool) {
  // Uses the shader: image_to_nchw_uint ((H * W) % 4 != 0)
  test_copy_to_texture_bool({13, 1, 3, 5});
  test_copy_to_texture_bool({13, 7, 1, 5});
  test_copy_to_texture_bool({13, 8, 2, 5});
  test_copy_to_texture_bool({13, 31, 2, 57});

  test_copy_to_texture_bool({67, 19, 7});
  test_copy_to_texture_bool({229, 213});
  test_copy_to_texture_bool({1902});
}

TEST_F(VulkanAPITest, adaptive_avg_pool2d) {
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_add(const at::IntArrayRef input_shape, const at::IntArrayRef other_shape, float alpha) {
  const auto in_cpu = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu = at::rand(other_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto in_vulkan = in_cpu.vulkan();
  const auto other_vulkan = other_cpu.vulkan();

  const auto out_cpu = at::add(in_cpu, other_cpu, alpha);
  const auto out_vulkan = at::add(in_vulkan, other_vulkan, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_invalid_inputs) {
  // Incompatible dimensions for broadcasting for binary elementwise op
  auto in_cpu = at::rand({2, 3, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));
  auto other_cpu = at::rand({2, 4, 4, 5}, at::device(at::kCPU).dtype(at::kFloat));

  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);
}

TEST_F(VulkanAPITest, add) {
  test_add({2, 3}, {2, 3}, 1.0f);
  test_add({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, add_broadcast0) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, add_broadcast1) {
  test_add({3, 5, 179, 221}, {3, 5, 1, 221}, 1.8f);
}

TEST_F(VulkanAPITest, add_broadcast2) {
  test_add({3, 4, 179, 221}, {4, 1, 1}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast3) {
  test_add({3, 4, 41, 53}, {1, 1, 41, 53}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast4) {
  test_add({3, 4, 41, 1}, {1, 41, 53}, 2.5f);
}

TEST_F(VulkanAPITest, add_broadcast5) {
  test_add({2, 1, 7, 1}, {1, 5, 1, 4}, 1.2f);
}

TEST_F(VulkanAPITest, add_broadcast6) {
  test_add({1, 15, 5, 4}, {21, 1, 5, 4}, 1.8f);
}

TEST_F(VulkanAPITest, add_zero_dim) {
 test_add({2, 6, 5, 6}, {}, 1.5f);
}

void test_add_other_cpu_int(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef other_shape,
    float alpha) {
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu =
      (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100)
          .to(at::kInt);

  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::add(in_cpu, other_cpu, alpha);
  const auto out_vulkan = at::add(in_vulkan, other_cpu, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_other_cpu_int) {
  test_add_other_cpu_int({2, 3}, {2, 3}, 1.0f);
  test_add_other_cpu_int({11, 7, 139, 109}, {11, 7, 139, 109}, 2.1f);
}

TEST_F(VulkanAPITest, add_broadcast0_other_cpu_int) {
  test_add_other_cpu_int({3, 5, 179, 221}, {3, 5, 1, 1}, 1.8f);
}

TEST_F(VulkanAPITest, add_other_cpu_unsupported_type_should_fail) {
  const auto in_cpu = at::rand({2,2,2}, at::device(at::kCPU).dtype(at::kFloat));

  const auto other_cpu =
    at::zeros({2, 2, 2}, at::device(at::kCPU).dtype(at::kComplexFloat));

  EXPECT_THROW(at::add(in_cpu.vulkan(), other_cpu.vulkan(), 1.0f), ::std::exception);
}

TEST_F(VulkanAPITest, add_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_broadcast0_) {
  auto a_cpu = at::rand({16, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({16, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_other_cpu_int_) {
  std::vector<int64_t> input_shape{12, 17, 29, 33};
  const auto in_cpu =
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto other_cpu =
      (at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat)) * 100)
          .to(at::kInt);

  const auto in_vulkan = in_cpu.vulkan();

  float alpha = -8.31f;
  in_cpu.add(other_cpu, alpha);
  in_vulkan.add(other_cpu, alpha);

  const auto check = almostEqual(in_cpu, in_vulkan.cpu());
  if (!check) {
    showRtol(in_cpu, in_vulkan.cpu());
  }
}

TEST_F(VulkanAPITest, add_broadcast1_) {
  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar) {
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_) {
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_scalar_wrapped_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_scalar = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, add_to_scalar_wrapped) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({1, 1280}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({1280, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_expand2) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, addmm_error_bias) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // mismatched bias size (should be 1-dim or {17, 9})
  const auto bias_cpu = at::rand({5, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({17, 6}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({6, 9}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, avg_pool2d) {
  const auto in_cpu = at::rand({3, 19, 43, 79}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);
  const auto out_vulkan = at::avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_batch_norm_invalid_inputs) {
  c10::InferenceMode mode;

  // Act: Vulkan batchnorm only supports evaluation mode
  EXPECT_THROW({
    at::batch_norm(
      at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      true,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects 4-dim input
  EXPECT_THROW({
    at::batch_norm(
      at::rand({3, 8, 5}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects 4-dim input
  EXPECT_THROW({
    at::batch_norm(
      at::rand({2, 8, 3, 5, 7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: Vulkan batchnorm expects channel dim to be multiple of 4
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 7, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({7}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: weight tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: bias tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: running mean tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);

  // Act: running var tensor contains incorrect number of elements
  EXPECT_THROW({
    at::batch_norm(
      at::rand({4, 8, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({8}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      at::rand({12}, at::device(at::kCPU).dtype(at::kFloat)).vulkan(),
      false,
      0.1,
      1e-05,
      false);
  }, ::std::exception);
}

TEST_F(VulkanAPITest, batch_norm_small) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({1, 4, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({4}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, batch_norm_medium) {
  c10::InferenceMode mode;

  const auto input_cpu = at::rand({3, 8, 5, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({8}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, batch_norm_large) {
  c10::InferenceMode mode;


  const auto input_cpu = at::rand({11, 52, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto input_vulkan = input_cpu.vulkan();

  const auto weight_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  const auto running_mean_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_mean_vulkan = running_mean_cpu.vulkan();

  const auto running_var_cpu = at::rand({52}, at::device(at::kCPU).dtype(at::kFloat));
  const auto running_var_vulkan = running_var_cpu.vulkan();

  const auto output_cpu = at::batch_norm(input_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, false, 0.1, 1e-05, false);
  const auto output_vulkan = at::batch_norm(input_vulkan, weight_vulkan, bias_vulkan, running_mean_vulkan, running_var_vulkan, false, 0.1, 1e-05, false);

  const auto check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

void test_baddbmm(
    at::Tensor bias_cpu,
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    float beta,
    float alpha) {
  const auto out_cpu = at::baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan =
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu.vulkan(), beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, baddbmm) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  int batch = 9;
  int n = 10;
  int p = 41;
  int m = 13;

  const auto bias_cpu =
      at::rand({batch, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_small) {
  constexpr float alpha = -1.0f;
  constexpr float beta = 2.0f;
  int batch = 3;
  int n = 3;
  int p = 5;
  int m = 4;

  const auto bias_cpu_0 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu_1 =
      at::ones({1, n, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu_2 =
      at::rand({1, n, m}, at::device(at::kCPU).dtype(at::kFloat)) * -1;
  const auto bias_cpu = at::cat({bias_cpu_0, bias_cpu_1, bias_cpu_2}, 0);

  const auto m1_cpu =
      at::rand({batch, n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({batch, p, m}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_one) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));

  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bais_error) {
  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  // mismatched dimensions of batch sizes.
  const auto bias_cpu =
      at::rand({200, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(
      at::baddbmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha), ::std::exception);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({150, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_width) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({150, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_width) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_batch_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_one) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch1) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({179, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch2) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu =
      at::rand({1, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_batch_height) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu = at::rand({163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

TEST_F(VulkanAPITest, baddbmm_bias_boardcast_reduce_all) {
  constexpr float alpha = 1.5f;
  constexpr float beta = 2.0f;
  const auto bias_cpu = at::rand({1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu =
      at::rand({150, 179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({150, 67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  test_baddbmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);
}

void test_matmul(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  c10::InferenceMode mode;
  const auto out_cpu = at::matmul(m1_cpu, m2_cpu);
  auto out_vk =
      at::matmul(m1_cpu.vulkan(), m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  const auto check = almostEqual(out_cpu, out_vk.cpu());
  if (!check) {
    showRtol(out_cpu, out_vk.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_vulkan) {
  // This will call at::bmm. Will crash for unknown reason.
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, DISABLED_matmul_3d_weight_cpu) {
  // This will call at::bmm. Will crash for unknown reason.
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, matmul_2d_weight_vulkan) {
  // This will call at::mm
  const auto m1_cpu = at::rand({7, 42}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({42, 9}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, matmul_2d_weight_cpu) {
  // This will call at::mm
  const auto m1_cpu =
      at::rand({23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_matmul(m1_cpu, m2_cpu);
}

void test_bmm(
    at::Tensor m1_cpu,
    at::Tensor m2_cpu,
    bool m2_use_vulkan = false) {
  const auto out_cpu = m1_cpu.bmm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan =
      m1_vulkan.bmm(m2_use_vulkan ? m2_cpu.vulkan() : m2_cpu);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, bmm_vulkan_small) {
  const auto m1_cpu =
      at::rand({5, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({5, 3, 4}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_vulkan_small_width) {
  const auto m1_cpu =
      at::rand({9, 32, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({9, 5, 13}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_vulkan_large_width) {
  const auto m1_cpu =
      at::rand({9, 7, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({9, 45, 6}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu, true);
}

TEST_F(VulkanAPITest, bmm_cpu) {
  const auto m1_cpu =
      at::rand({13, 23, 45}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({13, 45, 26}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_small) {
  const auto m1_cpu =
      at::rand({2, 6, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({2, 5, 3}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_one) {
  const auto m1_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  test_bmm(m1_cpu, m2_cpu);
}

TEST_F(VulkanAPITest, bmm_error) {
  // mismatched dimensions of batch sizes.
  const auto m1_cpu =
      at::rand({100, 235, 546}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu =
      at::rand({200, 546, 267}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_vulkan = m1_cpu.vulkan();
  EXPECT_THROW(m1_vulkan.bmm(m2_cpu), ::std::exception);
}

TEST_F(VulkanAPITest, clamp) {
  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  const auto out_cpu = at::clamp(in_cpu, min_value, max_value);
  const auto out_vulkan = at::clamp(in_vulkan, min_value, max_value);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, clamp_) {
  const auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  cpu.clamp_(min_value, max_value);
  vulkan.clamp_(min_value, max_value);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv1d_simple) {
  // This is a simple case using arange for input, ones for weights, and arange
  // for bias. This makes debugging easiser.
  int64_t kernel_size = 3;
  int64_t channels = 5;
  int64_t lengths = 9;

  c10::InferenceMode mode;

  const auto input_cpu = at::arange(lengths * channels, at::kFloat).reshape({1, channels, lengths});
  const auto weights_cpu = at::ones({channels, 1, kernel_size}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::arange(channels, at::kFloat);

  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  int64_t stride = 1;
  int64_t padding = 0;
  int64_t dilation = 1;

  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, channels);

  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, channels);
  const auto output_vk_cpu = output_vk.cpu();

  const bool check = almostEqual(output_cpu, output_vk_cpu);
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  ASSERT_TRUE(check);
}

void test_conv1d(
    int64_t kernel_size,
    int64_t groups,
    int64_t lengths,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t in_group_size = 1,
    int64_t out_group_size = 1,
    int64_t batch_size = 1) {
  c10::InferenceMode mode;

  int64_t in_channels = in_group_size * groups;
  int64_t out_channels = out_group_size * groups;

  const auto input_cpu = at::rand({batch_size, in_channels, lengths}, at::kFloat);
  const auto weights_cpu = at::rand({out_channels, in_group_size, kernel_size}, at::kFloat);
  const auto bias_cpu = at::rand({out_channels,}, at::kFloat);

  const auto input_vk = input_cpu.vulkan();
  const auto weights_vk = weights_cpu.vulkan();
  const auto bias_vk = bias_cpu.vulkan();

  const auto output_cpu = at::conv1d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const auto output_vk = at::conv1d(
      input_vk, weights_vk, bias_vk, stride, padding, dilation, groups);
  const auto output_vk_cpu = output_vk.cpu();

  const bool check = almostEqual(output_cpu, output_vk_cpu);
  if (!check) {
    showRtol(output_cpu, output_vk_cpu);
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv1d) {
  test_conv1d(3, 5, 8);
  test_conv1d(9, 5, 9);
  test_conv1d(1, 12, 3);
  test_conv1d(1, 12, 1);
  test_conv1d(10, 12, 20);
  test_conv1d(3, 5, 9, 2, 0, 1);
  test_conv1d(3, 5, 9, 2, 1, 1);
  test_conv1d(3, 5, 9, 2, 1, 2);
  test_conv1d(3, 5, 9, 1, 4, 2);
  test_conv1d(6, 22, 30, 5, 5, 3);
  test_conv1d(6, 5, 30, 5, 5, 3, 3, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 2);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 5);
  test_conv1d(6, 5, 30, 5, 5, 3, 4, 2, 9);
}



void test_conv2d_context(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
  c10::InferenceMode mode;

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // vulkan
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::create_conv2d_context",
      "",
      weight, bias, stride, padding, dilation, groups, std::nullopt, std::nullopt);

  const auto vulkan_output = callOpByName(
      "vulkan_prepack::run_conv2d_context",
      "",
      input.vulkan(), prepack_vulkan[0]);

  const auto out_vulkan = vulkan_output[0].toTensor();
  const auto out_vk_cpu = out_vulkan.cpu();

  // check
  const bool check = almostEqual(out_cpu, out_vk_cpu);
  if (!check) {
    showRtol(out_cpu, out_vk_cpu);
  }

  ASSERT_TRUE(check);
}

void test_backwards_compatible_conv2d_context(
    const at::IntArrayRef input_shape,
    const at::IntArrayRef weight_shape,
    const at::IntArrayRef bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
  c10::InferenceMode mode;

  at::Tensor input = at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight = at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor bias = at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  // cpu
  const auto out_cpu = at::conv2d(
    input, weight, bias, stride, padding, dilation, groups);

  // vulkan
  const auto prepack_vulkan = callOpByName(
      "vulkan_prepack::conv2d_clamp_prepack",
      "",
      weight, bias, stride, padding, dilation, groups, std::nullopt, std::nullopt);

  
```



## High-Level Overview


This C++ file contains approximately 9 class(es)/struct(s) and 315 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `at`

**Classes/Structs**: `VulkanAPITest`, `OpType`, `BaseOp`, `Addmm`, `Conv2d`, `Hardtanh_`, `Mean`, `OpsList`, `MobileNetV2`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/core/dispatch/Dispatcher.h`
- `ATen/native/vulkan/api/api.h`
- `c10/util/irange.h`
- `c10/util/ArrayRef.h`


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
python aten/src/ATen/test/vulkan_api_test.cpp
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

- **File Documentation**: `vulkan_api_test.cpp_docs.md`
- **Keyword Index**: `vulkan_api_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
