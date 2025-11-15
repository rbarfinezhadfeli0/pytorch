# Documentation: `benchmarks/static_runtime/test_generated_ops.cc`

## File Metadata

- **Path**: `benchmarks/static_runtime/test_generated_ops.cc`
- **Size**: 206,056 bytes (201.23 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**.

## Original Source

```cpp
// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

TEST(StaticRuntime, autogen_absolute) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::absolute(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_angle) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::angle(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sgn) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sgn(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_acos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::acos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arccos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arccos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__add_relu_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::_add_relu(%self, %other, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  auto alpha0 = 2;
  std::vector<IValue> args{self0, other0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, other1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addmv) {
  const std::string script = R"IR(
    graph(%self: Tensor, %mat: Tensor, %vec: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::addmv(%self, %mat, %vec, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2});
  auto mat0 = at::rand({2, 2});
  auto vec0 = at::rand({2});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, mat0, vec0, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({35});
  auto mat1 = at::rand({35, 35});
  auto vec1 = at::rand({35});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, mat1, vec1, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addr) {
  const std::string script = R"IR(
    graph(%self: Tensor, %vec1: Tensor, %vec2: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::addr(%self, %vec1, %vec2, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto vec10 = at::rand({6});
  auto vec20 = at::rand({6});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, vec10, vec20, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22});
  auto vec11 = at::rand({22});
  auto vec21 = at::rand({22});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, vec11, vec21, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__test_functorch_fallback) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::_test_functorch_fallback(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_argmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int?, %keepdim: bool):
        %bias: None = prim::Constant()
        %ret = aten::argmax(%self, %dim, %keepdim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto keepdim0 = false;
  std::vector<IValue> args{self0, dim0, keepdim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto keepdim1 = false;
  std::vector<IValue> args2{self1, dim1, keepdim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_acosh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::acosh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2, 2, 2}) + at::ones({2, 2, 2});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({5, 5, 5}) + at::ones({5, 5, 5});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_asinh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::asinh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arcsinh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arcsinh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_atanh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::atanh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arctanh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arctanh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_asin) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::asin(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arcsin) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arcsin(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_atan) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::atan(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arctan) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arctan(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_baddbmm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %batch1: Tensor, %batch2: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::baddbmm(%self, %batch1, %batch2, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto batch10 = at::rand({6, 6, 6});
  auto batch20 = at::rand({6, 6, 6});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, batch10, batch20, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto batch11 = at::rand({22, 22, 22});
  auto batch21 = at::rand({22, 22, 22});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, batch11, batch21, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_not) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_not(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_copysign_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::copysign(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_not) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_not(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_xor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_xor(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_and) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_and(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_or) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_or(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ceil) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ceil(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clamp_max) {
  const std::string script = R"IR(
    graph(%self: Tensor, %max: int):
        %bias: None = prim::Constant()
        %ret = aten::clamp_max(%self, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto max0 = 2;
  std::vector<IValue> args{self0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto max1 = 2;
  std::vector<IValue> args2{self1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clamp_max_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %max: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::clamp_max(%self, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto max0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto max1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clip) {
  const std::string script = R"IR(
    graph(%self: Tensor, %min: int?, %max: int?):
        %bias: None = prim::Constant()
        %ret = aten::clip(%self, %min, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto min0 = 2;
  auto max0 = 2;
  std::vector<IValue> args{self0, min0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto min1 = 2;
  auto max1 = 2;
  std::vector<IValue> args2{self1, min1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_complex) {
  const std::string script = R"IR(
    graph(%real: Tensor, %imag: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::complex(%real, %imag)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto real0 = at::rand({6, 6, 6});
  auto imag0 = at::rand({6, 6, 6});
  std::vector<IValue> args{real0, imag0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto real1 = at::rand({22, 22, 22});
  auto imag1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{real1, imag1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_polar) {
  const std::string script = R"IR(
    graph(%abs: Tensor, %angle: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::polar(%abs, %angle)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto abs0 = at::rand({6, 6, 6});
  auto angle0 = at::rand({6, 6, 6});
  std::vector<IValue> args{abs0, angle0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto abs1 = at::rand({22, 22, 22});
  auto angle1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{abs1, angle1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::cos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cosh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::cosh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cumprod) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %dtype: int?):
        %bias: None = prim::Constant()
        %ret = aten::cumprod(%self, %dim, %dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{self0, dim0, dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{self1, dim1, dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_diff) {
  const std::string script = R"IR(
    graph(%self: Tensor, %n: int, %dim: int, %prepend: Tensor?, %append: Tensor?):
        %bias: None = prim::Constant()
        %ret = aten::diff(%self, %n, %dim, %prepend, %append)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto n0 = 1;
  auto dim0 = 1;
  auto prepend0 = at::rand({6, 6, 6});
  auto append0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, n0, dim0, prepend0, append0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto n1 = 1;
  auto dim1 = 1;
  auto prepend1 = at::rand({22, 22, 22});
  auto append1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, n1, dim1, prepend1, append1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_divide_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::divide(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_true_divide_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::true_divide(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_dot) {
  const std::string script = R"IR(
    graph(%self: Tensor, %tensor: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::dot(%self, %tensor)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({16});
  auto tensor0 = at::rand({16});
  std::vector<IValue> args{self0, tensor0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({64});
  auto tensor1 = at::rand({64});
  std::vector<IValue> args2{self1, tensor1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_vdot) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::vdot(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({16});
  auto other0 = at::rand({16});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({64});
  auto other1 = at::rand({64});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_erf) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::erf(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_erfc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::erfc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_exp) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::exp(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_exp2) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::exp2(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_expm1) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::expm1(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_floor) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::floor(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_frac) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::frac(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gcd) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::gcd(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lcm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::lcm(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_index_copy) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %source: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::index_copy(%self, %dim, %index, %source)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2});
  auto dim0 = 0;
  auto index0 = at::randint(0, 1, {2}, at::kLong);
  auto source0 = at::rand({2});
  std::vector<IValue> args{self0, dim0, index0, source0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({32});
  auto dim1 = 0;
  auto index1 = at::randint(0, 10, {32}, at::kLong);
  auto source1 = at::rand({32});
  std::vector<IValue> args2{self1, dim1, index1, source1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Tensor_Tensor) {
  const std::string script = R"IR(
    graph(%elements: Tensor, %test_elements: Tensor, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%elements, %test_elements, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto elements0 = at::rand({6, 6, 6});
  auto test_elements0 = at::rand({6, 6, 6});
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{elements0, test_elements0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto elements1 = at::rand({22, 22, 22});
  auto test_elements1 = at::rand({22, 22, 22});
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{elements1, test_elements1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Tensor_Scalar) {
  const std::string script = R"IR(
    graph(%elements: Tensor, %test_element: int, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%elements, %test_element, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto elements0 = at::rand({6, 6, 6});
  auto test_element0 = 2;
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{elements0, test_element0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto elements1 = at::rand({22, 22, 22});
  auto test_element1 = 2;
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{elements1, test_element1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Scalar_Tensor) {
  const std::string script = R"IR(
    graph(%element: int, %test_elements: Tensor, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%element, %test_elements, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto element0 = 2;
  auto test_elements0 = at::rand({6, 6, 6});
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{element0, test_elements0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto element1 = 2;
  auto test_elements1 = at::rand({22, 22, 22});
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{element1, test_elements1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_kron) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::kron(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ldexp_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ldexp(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log10) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log10(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log1p) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log1p(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log2) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log2(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logaddexp) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logaddexp(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logaddexp2) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logaddexp2(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_xlogy_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::xlogy(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__log_softmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %half_to_float: bool):
        %bias: None = prim::Constant()
        %ret = aten::_log_softmax(%self, %dim, %half_to_float)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto half_to_float0 = false;
  std::vector<IValue> args{self0, dim0, half_to_float0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto half_to_float1 = false;
  std::vector<IValue> args2{self1, dim1, half_to_float1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__log_softmax_backward_data) {
  const std::string script = R"IR(
    graph(%grad_output: Tenso
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/static_runtime`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/csrc/jit/runtime/static/impl.h`
- `torch/torch.h`
- `test_utils.h`


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
python benchmarks/static_runtime/test_generated_ops.cc
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/static_runtime`):

- [`test_cpu_fusion.cc_docs.md`](./test_cpu_fusion.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_utils.cc_docs.md`](./test_utils.cc_docs.md)
- [`test_static_runtime.cc_docs.md`](./test_static_runtime.cc_docs.md)
- [`test_utils.h_docs.md`](./test_utils.h_docs.md)
- [`deep_wide_pt.cc_docs.md`](./deep_wide_pt.cc_docs.md)
- [`deep_wide_pt.h_docs.md`](./deep_wide_pt.h_docs.md)
- [`test_static_module.cc_docs.md`](./test_static_module.cc_docs.md)
- [`deep_wide_pt_bench.cc_docs.md`](./deep_wide_pt_bench.cc_docs.md)


## Cross-References

- **File Documentation**: `test_generated_ops.cc_docs.md`
- **Keyword Index**: `test_generated_ops.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
