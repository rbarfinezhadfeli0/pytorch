# Documentation: test_lazy_ops.cpp

## File Metadata
- **Path**: `test/cpp/lazy/test_lazy_ops.cpp`
- **Size**: 420262 bytes
- **Lines**: 11587
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the torchscript backend, without which lazy device won't work.
// FIXME: This registers the backend for the whole test binary. We should
// probably do it and undo it in the test fixture below.
static bool inline init_backend() {
  torch::lazy::InitTorchScriptBackend();
  return true;
}
static const bool backend_initialized = init_backend();

} // namespace

class LazyTsTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup() {}

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ExpectCounterChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ResetCounters() {}

 private:
  void MakeEndSnapshot() {}
};

class LazyOpsTestBase : public LazyTsTest {
 protected:
  static void SetUpTestCase() {}
};

void LazyTsTest::SetUp() {
  (void)backend_initialized; // avoid unused parameter warning
  at::manual_seed(42);
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(
      torch::lazy::BackendDevice(), 42);
}

void LazyTsTest::TearDown() {}

namespace {
using torch::lazy::DebugUtil;

class LazyOpsTest : public LazyOpsTestBase {};

static inline bool IsCuda() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType() == at::kCUDA;
}

static inline at::DeviceType DefaultDevice() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType();
}

} // namespace

TEST_F(LazyOpsTest, TestScalarTensor) {
  torch::Tensor scalar_tensor = torch::scalar_tensor(
      1., torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_scalar_tensor = torch::scalar_tensor(
        1., torch::TensorOptions(torch::kFloat).device(torch::kLazy));
    AllClose(scalar_tensor, lazy_scalar_tensor);
  });
}

TEST_F(LazyOpsTest, TestClone) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = lazy_a.clone();
    AllClose(a, lazy_b);
    lazy_a.add_(1.0);
    AllClose(a, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTo) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestIsFloatingPoint) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    bool is_float = torch::is_floating_point(a);
    bool lazy_is_float = torch::is_floating_point(lazy_a);
    EXPECT_EQ(is_float, lazy_is_float);
  });
}

TEST_F(LazyOpsTest, TestIsSigned) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    bool is_signed = torch::is_signed(a);
    bool lazy_is_signed = torch::is_signed(lazy_a);
    EXPECT_EQ(is_signed, lazy_is_signed);
  });
}

TEST_F(LazyOpsTest, TestCastByte) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Byte(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Byte(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastChar) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Char(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Char(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastShort) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Short(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Short(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastInt) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Int(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Int(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastLong) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Long(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Long(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastFloat) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Float(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Float(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRetainType) {
  torch::Tensor lazy_a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor lazy_b = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor lazy_c = lazy_a + lazy_b;
  EXPECT_EQ(lazy_c.scalar_type(), torch::ScalarType::Byte);
}

TEST_F(LazyOpsTest, TestLogicalTypeWithInterop) {
  torch::Tensor query = torch::rand(
      {2, 12, 20, 64},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor key = torch::rand(
      {2, 12, 64, 20},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor scores =
      torch::matmul(query, key) /
      torch::scalar_tensor(
          8, torch::TensorOptions(torch::kDouble).device(torch::kLazy));
  torch::Tensor p_attn = torch::softmax(scores, /*dim=*/-1);
  EXPECT_EQ(p_attn.scalar_type(), torch::ScalarType::Float);
}

TEST_F(LazyOpsTest, TestAdd) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddHalf) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddMixedPrecision) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor lazy_c = lazy_a.add_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(1);
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::add(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor lazy_c = lazy_a.add_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddZeroSizeDim) {
  torch::Tensor a = torch::rand(
      {0, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {1, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSub) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::sub(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor lazy_c = lazy_a.sub_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(1);
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::sub(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor lazy_c = lazy_a.sub_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMul) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::mul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor lazy_c = lazy_a.mul_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(3);
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::mul(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalarInPlace) {
  torch::Scalar b(3);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor lazy_c = lazy_a.mul_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDiv) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = torch::div(lazy_a, lazy_b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      int lower_bound = (scalar_type1 == torch::kByte) ? 0 : -100;
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          : torch::randint(
                lower_bound, 50, {3, 4}, torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 :
           {torch::kFloat,
            torch::kByte,
            torch::kChar,
            torch::kShort,
            torch::kInt,
            torch::kLong}) {
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            : torch::randint(
                  51, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        torch::Tensor c = torch::div(a, b, rounding_mode);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = torch::div(lazy_a, lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlace) {
  for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = lazy_a.div_(lazy_b);
        ;
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlaceWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          : torch::randint(
                -100, 100, {3, 4}, torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            : torch::randint(
                  1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor c = a.div_(b, rounding_mode);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = lazy_a.div_(lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalar) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_c = torch::div(lazy_a, b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalarInPlace) {
  for (torch::ScalarType scalar_type : {torch::kFloat}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor lazy_c = lazy_a.div_(b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivOut) {
  for (torch::ScalarType scalar_type : {torch::kFloat, torch::kDouble}) {
    torch::Tensor a = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor c = torch::empty(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::div_out(c, a, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = torch::empty({3, 4}, lazy_b.options());
      torch::div_out(lazy_c, lazy_a, lazy_b);
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestRsubScalar) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(1.5);
  torch::Scalar alpha(2.5);
  torch::Tensor result = torch::rsub(input, other, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::rsub(lazy_input, other, alpha);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestNe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::ne(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::ne(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestNeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor a_copy = a.clone();
  torch::Tensor b = a.clone();
  b[0] += 1;
  a.ne_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ne_(lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestEq) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::eq(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::eq(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEqInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  torch::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.eq_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::ge(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::ge(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ge_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::le(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::le(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.le_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGt) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::gt(b, a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::gt(lazy_b, lazy_a);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGtInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.gt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLt) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::lt(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::lt(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLtInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.lt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestNeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0));
  torch::Tensor result = torch::ne(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::ne(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEqScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::eq(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::eq(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::ge(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::ge(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.ge_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::le(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::le(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLeScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.le_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestGtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0.5));
  torch::Tensor result = torch::gt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::gt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGtScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.gt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1.5));
  torch::Tensor result = torch::lt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::lt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLtScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.lt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestIntegerAdd) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Scalar one =
          isIntegralType(type, false) ? torch::Scalar(1) : torch::Scalar(1.0);
      torch::Tensor c = torch::add(b, one);

      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = torch::add(lazy_b, one);

      AllEqual(c, lazy_c);
    }
  });
}

TEST_F(LazyOpsTest, TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = torch::svd(lazy_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        torch::Tensor diag = std::get<1>(b);
        torch::Tensor lazy_diag = std::get<1>(lazy_b);
        ASSERT_EQ(diag.sizes(), lazy_diag.sizes());
        AllClose(
            diag,
            lazy_diag,
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<2>(b).abs(),
            std::get<2>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::qr(a);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = torch::qr(lazy_a);
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<1>(b).abs(),
            std::get<1>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      torch::Tensor a = torch::rand(
          {3, m, m},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor pd_a =
          torch::matmul(a, torch::transpose(a, 1, 2)) +
          torch::eye(
              m, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::cholesky(pd_a, upper);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(pd_a, device);
        auto lazy_b = torch::cholesky(lazy_a, upper);
        AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    torch::Tensor a = torch::rand(
        {3, m, m}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
        torch::eye(m,
                   torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor b = torch::logdet(pd_a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(pd_a, device);
      torch::Tensor lazy_b = torch::logdet(lazy_a);
      AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestTriangularSolve) {
  static const int dims[] = {4, 7};
  for (bool batched_a : {true, false}) {
    for (bool batched_b : {true, false}) {
      for (auto m : dims) {
        for (auto n : dims) {
          for (bool upper : {true, false}) {
            for (bool transpose : {true, false}) {
              for (bool unitriangular : {true, false}) {
                torch::Tensor a = torch::randn(
                    {m, m},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                torch::Tensor b = torch::randn(
                    {m, n},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = torch::triangular_solve(
                    b,
                    a,
                    /*upper=*/upper,
                    /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const torch::Device& device) {
                  torch::Tensor lazy_a = CopyToDevice(a, device);
                  torch::Tensor lazy_b = CopyToDevice(b, device);
                  auto lazy_result = torch::triangular_solve(
                      lazy_b,
                      lazy_a,
                      /*upper=*/upper,
                      /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  AllClose(
                      std::get<0>(result),
                      std::get<0>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                  AllClose(
                      std::get<1>(result),
                      std::get<1>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKthValue) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = torch::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::kthvalue(lazy_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTopK) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = torch::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::topk(lazy_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSort) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = torch::sort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::sort(lazy_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSortDescWithMinValue) {
  std::vector<int8_t> values{-128, 100};
  torch::Tensor input =
      torch::tensor(values, torch::TensorOptions(torch::kChar));
  auto output = torch::sort(input, /*dim=*/0, /*descending=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    auto lazy_output = torch::sort(lazy_input, /*dim=*/0, /*descending=*/true);
    AllEqual(std::get<0>(output), std::get<0>(lazy_output));
    AllEqual(std::get<1>(output), std::get<1>(lazy_output));
  });
}

TEST_F(LazyOpsTest, TestArgSort) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        torch::Tensor b = torch::argsort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::argsort(lazy_a, dim, descending);
          AllEqual(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::min(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::min(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMax) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::max(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::max(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestUnaryMin) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::min(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::min(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestUnaryMax) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::max(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::max(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAll) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::all(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDim) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDimKeep) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAmax) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amax(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_values =
            torch::amax(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_values =
              torch::amax(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amax", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAmin) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amin(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_values =
            torch::amin(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_values =
              torch::amin(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amin", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAny) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::any(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDim) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDimKeep) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMean) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::mean(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::mean(lazy_a);
    ASSERT_EQ(b.sizes(), lazy_b.sizes());
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::mean(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::mean(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::mean(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimsKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims, true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, dims, true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimOut) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::empty(
        {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::mean_out(b, a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::empty({4, 4}, lazy_a.options());
      torch::mean_out(lazy_b, lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStd) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto unbiased : {true, false}) {
    torch::Tensor b = torch::std(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::std(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStdInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (auto unbiased : {true, false}) {
    for (auto keepdim : {true, false}) {
      for (int dim = -rank; dim < rank; ++dim) {
        torch::Tensor b = torch::std(a, {dim}, unbiased, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::std(lazy_a, {dim}, unbiased, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        torch::Tensor b = torch::std(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::std(lazy_a, dim, correction, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdMeanWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        auto b = torch::std_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::std_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSum) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumU8) {
  torch::Tensor a = torch::ones(
      {256}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::sum(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::sum(lazy_a, dims, /*keepdim=*/true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVar) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (bool unbiased : {true, false}) {
    torch::Tensor b = torch::var(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::var(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVarWithDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (bool unbiased : {true, false}) {
        torch::Tensor b = torch::var(a, dims, unbiased, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dims, unbiased, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestVarWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (const auto& correction : corrections) {
        torch::Tensor b = torch::var(a, dim, correction, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dim, correction, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::var", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestVarMeanWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (const auto& correction : corrections) {
      for (auto keepdim : {true, false}) {
        auto b = torch::var_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::var_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxInDim) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            torch::max(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMinInDim) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            torch::min(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNorm) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDimsKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormalTwoTensor) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleMean) {
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(0, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleStd) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, 1);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalInPlace) {
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.normal_(/*mean=*/0, /*std=*/1);
    double res_mean = lazy_a.mean().item().toDouble();
    double res_std = lazy_a.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestUniformInPlace) {
  const double eps = 1e-3;
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.uniform_(/*from=*/0, /*to=*/1);
    at::Tensor cpu_a = ToCpuTensor(lazy_a);
    double res_min = cpu_a.min().item().toDouble();
    double res_max = cpu_a.max().item().toDouble();
    EXPECT_GT(res_min, 0.0 - eps);
    EXPECT_LT(res_max, 1.0 + eps);
  });
}

TEST_F(LazyOpsTest, TestRandomInPlace) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    const double eps = 0.2;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*from=*/0, /*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefaultFrom) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    const double eps = 0.2;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefault) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    auto input = torch::zeros({10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      auto lazyInput = CopyToDevice(input, device);
      lazyInput.random_();
      auto output = ToCpuTensor(lazyInput);
      EXPECT_TRUE(torch::all(output.ne(input)).item<bool>());
    });
  }
}

TEST_F(LazyOpsTest, TestNormGeneral) {
  torch::Tensor a = torch::randn(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a, 3.5);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a, 3.5);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormNuclear) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNorm) {
  int num_channels = 6;
  torch::Tensor input = torch::rand(
      {20, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    torch::Tensor output = torch::group_norm(
        input,
        num_groups,
        weight,
        bias,
        eps,
        /*cudnn_enabled=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_weight = CopyToDevice(weight, device);
      torch::Tensor lazy_bias = CopyToDevice(bias, device);
      torch::Tensor lazy_output = torch::group_norm(
          lazy_input,
          num_groups,
          lazy_weight,
          lazy_bias,
          eps,
          /*cudnn_enabled=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNormBackward) {
  int num_channels = 6;
  torch::Tensor input = torch::rand(
      {2, num_channels, 5, 5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int num_groups : {3, 6, 1}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::group_norm(
            /*input=*/inputs[0],
            num_groups,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-3,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  torch::Tensor input = torch::rand(
      {batch, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_mean = torch::zeros(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_var = torch::ones(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 1e-05;
  torch::Tensor output = torch::instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*use_input_stats=*/true,
      momentum,
      eps,
      /*cudnn_enabled=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
    torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
    torch::Tensor lazy_output = torch::instance_norm(
        lazy_input,
        lazy_weight,
        lazy_bias,
        lazy_running_mean,
        lazy_running_var,
        /*use_input_stats=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/false);
    AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLayerNorm) {
  torch::Tensor input = torch::rand(
      {20, 10, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  torch::Tensor undef;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor output = torch::layer_norm(
          input,
          normalized_shape,
          undef_weight ? undef : weight,
          undef_weight ? undef : bias,
          eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        torch::Tensor lazy_output = torch::layer_norm(
            lazy_input,
            normalized_shape,
            lazy_weight,
            lazy_bias,
            eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLayerNormBackward) {
  torch::Tensor input = torch::rand(
      {2, 3, 3, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::layer_norm(
            /*input=*/inputs[0],
            normalized_shape,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-4,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNuclearNorm) {
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::nuclear_norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::nuclear_norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestPairwiseDistance) {
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::pairwise_distance(x1, x2, p, eps, keepdim);
        torch::Tensor lazy_x1 = CopyToDevice(x1, device);
        torch::Tensor lazy_x2 = CopyToDevice(x2, device);
        torch::Tensor lazy_output =
            torch::pairwise_distance(lazy_x1, lazy_x2, p, eps, keepdim);
        AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCosineSimilarity) {
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::cosine_similarity(x1, x2, dim, eps);
      torch::Tensor lazy_x1 = CopyToDevice(x1, device);
      torch::Tensor lazy_x2 = CopyToDevice(x2, device);
      torch::Tensor lazy_output =
          torch::cosine_similarity(lazy_x1, lazy_x2, dim, eps);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestCosineEmbeddingLoss) {
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::cosine_embedding_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestHingeEmbeddingLoss) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::hinge_embedding_loss(input, target, margin, reduction);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::hinge_embedding_loss(
            lazy_input, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestTripletMarginLoss) {
  torch::Tensor anchor = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor positive = torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor negative = torch::neg(torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum}) {
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor output = torch::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            torch::Tensor lazy_anchor = CopyToDevice(anchor, device);
            torch::Tensor lazy_positive = CopyToDevice(positive, device);
            torch::Tensor lazy_negative = CopyToDevice(negative, device);
            torch::Tensor lazy_output = torch::triplet_margin_loss(
                lazy_anchor,
                lazy_positive,
                lazy_negative,
                margin,
                p,
                eps,
                swap,
                reduction);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean,
        torch::Reduction::Sum,
        torch::Reduction::None}) {
    for (bool undef_weight : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_output = torch::binary_cross_entropy(
            lazy_input, lazy_target, lazy_weight, reduction);
        AllClose(output, lazy_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMarginRankingLoss) {
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::margin_ranking_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor pos_weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor output = torch::binary_cross_entropy_with_logits(
              input,
              target,
              undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight,
              reduction);
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_target = CopyToDevice(target, device);
          torch::Tensor lazy_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          torch::Tensor lazy_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          torch::Tensor lazy_output = torch::binary_cross_entropy_with_logits(
              lazy_input, lazy_target, lazy_weight, lazy_pos_weight, reduction);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKlDiv) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (bool log_target : {true, false}) {
    for (torch::Reduction::Reduction reduction :
         {torch::Reduction::Mean, torch::Reduction::Sum}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::kl_div(input, target, reduction, log_target);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output =
            torch::kl_div(lazy_input, lazy_target, reduction, log_target);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestProd) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::prod(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::prod(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::prod(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::prod(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::prod(lazy_a, dim);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::prod(lazy_a, dim, /*keepdim=*/true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::prod(lazy_a, dim, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSum) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCast) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result =
          torch::cumsum(lazy_input, dim, torch::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumLong) {
  torch::Tensor input = torch::randint(
      1000,
      {4, 3, 4},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCastLong) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProd) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumprod(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCast) {
  torch::Tensor input = torch::mul(
      torch::rand(
          {4, 3, 4},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice())),
      10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result =
          torch::cumprod(lazy_input, dim, torch::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdLong) {
  torch::Tensor input = torch::randint(
      7, {2, 3}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCastLong) {
  torch::Tensor input =
      torch::rand(
          {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      7;
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMin) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmin(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmin(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinDim) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinSameValue) {
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::argmin(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinWrapper) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMax) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxDim) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxSameValue) {
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxWrapper) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAsin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::asin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::asin(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): LazyTsTest, LazyOpsTestBase, LazyOpsTest


## Key Components

The file contains 33393 words across 11587 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 420262 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
