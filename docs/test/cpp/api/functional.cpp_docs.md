# Documentation: `test/cpp/api/functional.cpp`

## File Metadata

- **Path**: `test/cpp/api/functional.cpp`
- **Size**: 120,419 bytes (117.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

namespace F = torch::nn::functional;

using namespace torch::nn;

struct FunctionalTest : torch::test::SeedingFixture {};

TEST_F(FunctionalTest, Conv1d) {
  auto x = torch::arange(30, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({2, 3, 5});
  auto weight =
      torch::arange(18, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({2, 3, 3});
  auto y = F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
  auto expected = torch::tensor(
      {{{312., 348., 384.}, {798., 915., 1032.}},

       {{852., 888., 924.}, {2553., 2670., 2787.}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv1d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, Conv2dEven) {
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5});
  auto weight =
      torch::arange(54, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({2, 3, 3, 3});
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
  auto expected = torch::tensor(
      {{{{15219., 15570., 15921.},
         {16974., 17325., 17676.},
         {18729., 19080., 19431.}},

        {{37818., 38898., 39978.},
         {43218., 44298., 45378.},
         {48618., 49698., 50778.}}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, Conv2dUneven) {
  auto x = torch::arange(60, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 4});
  auto weight =
      torch::arange(36, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({2, 3, 3, 2});
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
  auto expected = torch::tensor(
      {{{{5289., 5442., 5595.}, {5901., 6054., 6207.}, {6513., 6666., 6819.}},

        {{13227., 13704., 14181.},
         {15135., 15612., 16089.},
         {17043., 17520., 17997.}}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, Conv3d) {
  auto x = torch::arange(375, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5, 5});
  auto weight =
      torch::arange(162, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({2, 3, 3, 3, 3});
  auto y = F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
  auto expected = torch::tensor(
      {{{{{700704., 703944., 707184.},
          {716904., 720144., 723384.},
          {733104., 736344., 739584.}},

         {{781704., 784944., 788184.},
          {797904., 801144., 804384.},
          {814104., 817344., 820584.}},

         {{862704., 865944., 869184.},
          {878904., 882144., 885384.},
          {895104., 898344., 901584.}}},

        {{{1724220., 1734021., 1743822.},
          {1773225., 1783026., 1792827.},
          {1822230., 1832031., 1841832.}},

         {{1969245., 1979046., 1988847.},
          {2018250., 2028051., 2037852.},
          {2067255., 2077056., 2086857.}},

         {{2214270., 2224071., 2233872.},
          {2263275., 2273076., 2282877.},
          {2312280., 2322081., 2331882.}}}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv3d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

TEST_F(FunctionalTest, MaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(FunctionalTest, MaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(FunctionalTest, MaxPool2dBackward) {
  auto input = torch::rand(
      {1, 2, 4, 4}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = F::max_pool2d(input, F::MaxPool2dFuncOptions(2));
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, MaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::max_pool3d(x, F::MaxPool3dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(FunctionalTest, AvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, FractionalMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::fractional_max_pool2d(
      x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));

  auto y_with_indices = F::fractional_max_pool2d_with_indices(
      x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y_with_indices),
      torch::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}})));
  ASSERT_EQ(
      std::get<1>(y_with_indices).sizes(), std::vector<int64_t>({2, 2, 2}));

  auto x1 = torch::ones({2, 2, 5, 5});
  auto y1 = F::fractional_max_pool2d(
      x1, F::FractionalMaxPool2dFuncOptions(3).output_size(2));

  ASSERT_EQ(y1.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y1, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y1.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  auto y1_with_indices = F::fractional_max_pool2d_with_indices(
      x1, F::FractionalMaxPool2dFuncOptions(3).output_size(2));
  ASSERT_TRUE(torch::equal(y1, std::get<0>(y1_with_indices)));
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y1_with_indices),
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}},
           {{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}})));
  ASSERT_EQ(
      std::get<1>(y1_with_indices).sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, FractionalMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::fractional_max_pool3d(
      x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  auto y_with_indices = F::fractional_max_pool3d_with_indices(
      x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  ASSERT_TRUE(torch::allclose(
      std::get<1>(y_with_indices),
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}})));
  ASSERT_EQ(
      std::get<1>(y_with_indices).sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, LPPool1d) {
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  auto x = torch::ones({1, 1, 5});
  auto y = F::lp_pool1d(
      x, F::LPPool1dFuncOptions(norm_type, kernel_size).stride(stride));
  auto expected =
      (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) *
       kernel_size)
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, LPPool2d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  auto x = torch::ones({1, 1, 2, 5});
  auto y = F::lp_pool2d(
      x, F::LPPool2dFuncOptions(norm_type, kernel_size).stride(stride));
  auto expected =
      (torch::pow(torch::tensor({{{{1, 1}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 2}));
}

TEST_F(FunctionalTest, LPPool3d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({1, 2, 3});

  auto x = torch::ones({1, 1, 1, 2, 5});
  auto y = F::lp_pool3d(
      x, F::LPPool3dFuncOptions(norm_type, kernel_size).stride(stride));
  auto expected =
      (torch::pow(torch::tensor({{{{{1, 1}}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1] * kernel_size[2]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 5);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 1, 2}));
}

TEST_F(FunctionalTest, CosineSimilarity) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output = F::cosine_similarity(
      input1, input2, F::CosineSimilarityFuncOptions().dim(1));
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, SmoothL1LossDefaultOptions) {
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output = F::smooth_l1_loss(input, target);
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SmoothL1LossBeta) {
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,bugprone-argument-comment)
      F::smooth_l1_loss(
          input, target, /*reduction=*/torch::kMean, /*beta=*/0.5);
  auto expected = torch::tensor(1.67, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SmoothL1LossBetaOptions) {
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      F::smooth_l1_loss(
          input,
          target,
          F::SmoothL1LossFuncOptions().reduction(torch::kMean).beta(0.5));
  auto expected = torch::tensor(1.67, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SmoothL1LossNoReduction) {
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      // NOLINTNEXTLINE(bugprone-argument-comment)
      F::smooth_l1_loss(input, target, /*reduction=*/torch::kNone);
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, HuberLossDefaultOptions) {
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output = F::huber_loss(input, target);
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, HuberLossDelta) {
  auto input = torch::tensor(
      {0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto options = F::HuberLossFuncOptions().reduction(torch::kMean).delta(0.5);
  auto output = F::huber_loss(input, target, options);
  auto expected = torch::tensor(1.67 * 0.5, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, HuberLossNoReduction) {
  auto input = torch::tensor(
      {0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto options = F::HuberLossFuncOptions().reduction(torch::kNone);
  auto output = F::huber_loss(input, target, options);
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

TEST_F(FunctionalTest, SoftMarginLossDefaultOptions) {
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output = F::soft_margin_loss(input, target);
  auto expected = torch::tensor({1.3767317}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelSoftMarginLossDefaultOptions) {
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto output = F::multilabel_soft_margin_loss(input, target);
  auto expected = torch::tensor({0.7608436}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, SoftMarginLossNoReduction) {
  auto input = torch::tensor(
      {2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output = F::soft_margin_loss(input, target, torch::kNone);
  auto expected = torch::tensor(
      {2.1269281, 0.01814993, 0.3132617, 3.0485873}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  auto input = torch::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto target =
      torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto weight = torch::tensor({0.1, 0.6, 0.4, 0.8}, torch::kFloat);
  auto options = F::MultilabelSoftMarginLossFuncOptions()
                     .reduction(torch::kNone)
                     .weight(weight);
  auto output = F::multilabel_soft_margin_loss(input, target, options);
  auto expected = torch::tensor({0.4876902, 0.3321295}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, PairwiseDistance) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output = F::pairwise_distance(
      input1, input2, F::PairwiseDistanceFuncOptions().p(1));
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, PDist) {
  {
    auto input = torch::tensor({{-1.0, -5.0, -1.0}, {2.0, 4.0, 6.0}});
    auto output = F::pdist(input);
    auto expected = torch::tensor({11.7898});
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    auto input = torch::tensor({{1.0, -1.0}, {1.0, 3.0}, {3.0, 3.0}});
    auto output = F::pdist(input, 1.5);
    auto expected = torch::tensor({4.0, 4.8945, 2.0});
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(FunctionalTest, AdaptiveMaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_max_pool1d(x, F::AdaptiveMaxPool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_avg_pool3d(x, F::AdaptiveAvgPool3dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, L1Loss) {
  auto input = torch::randn({5, 6}, torch::requires_grad());
  auto target = torch::empty({5, 6}).random_(2);
  auto output = F::l1_loss(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MSELoss) {
  auto input = torch::randn({5, 6}, torch::requires_grad());
  auto target = torch::empty({5, 6}).random_(2);
  auto output = F::mse_loss(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, BCELoss) {
  auto input = torch::randn({5, 6}, torch::requires_grad());
  auto target = torch::empty({5, 6}).random_(2);
  auto output = F::binary_cross_entropy(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, KLDivLoss) {
  KLDivLoss loss;
  auto input = torch::randn({5, 6}, torch::requires_grad());
  auto target = torch::empty({5, 6}).random_(2);
  auto output = F::kl_div(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, HingeEmbeddingLoss) {
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::kFloat);
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = F::hinge_embedding_loss(
      input, target, F::HingeEmbeddingLossFuncOptions().margin(2));
  auto expected = torch::tensor({10}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, GridSample) {
  auto input =
      torch::arange(9, torch::kFloat).view(std::vector<int64_t>({1, 1, 3, 3}));
  auto grid = torch::tensor(
      {{{{-2., -1.}, {-1., -1.}, {0., -1.}},
        {{-1., 0.}, {0., 0.}, {1., 0.}},
        {{0., 1.}, {1., 1.}, {2., 1.}}}},
      torch::kFloat);

  // bilinear, zeros, true
  auto options = F::GridSampleFuncOptions()
                     .mode(torch::kBilinear)
                     .padding_mode(torch::kZeros)
                     .align_corners(true);
  auto output = F::grid_sample(input, grid, options);
  auto expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, zeros, false
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor(
      {{{{0., 0., 0.5}, {1.5, 4., 2.5}, {3.5, 2., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // default options (bilinear, zeros, false) same result as above
  output = F::grid_sample(input, grid);

  ASSERT_TRUE(output.allclose(expected));

  // nearest, zeros, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kNearest)
                .padding_mode(torch::kZeros)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bicubic, zeros, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kBicubic)
                .padding_mode(torch::kZeros)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, border, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor(
      {{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 8.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, reflection, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kReflection)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor(
      {{{{1., 0., 1.}, {3., 4., 5.}, {7., 8., 7.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, AffineGrid) {
  {
    // 2D affine.
    auto theta = torch::arange(1., 13).view(std::vector<int64_t>({2, 2, 3}));
    auto size = std::vector<int64_t>({2, 3, 2, 2});
    auto align_corners = true;
    auto output = F::affine_grid(theta, size, !align_corners);
    auto expected = torch::tensor(
        {{{{1.50, 1.50}, {2.50, 5.50}}, {{3.50, 6.50}, {4.50, 10.50}}},
         {{{1.50, 1.50}, {8.50, 11.50}}, {{9.50, 12.50}, {16.50, 22.50}}}});
    auto output_aligned = F::affine_grid(theta, size, align_corners);
    auto expected_aligned = torch::tensor(
        {{{{0.0, -3.0}, {2.0, 5.0}}, {{4.0, 7.0}, {6.0, 15.0}}},
         {{{-6.0, -9.0}, {8.0, 11.0}}, {{10.0, 13.0}, {24.0, 33.0}}}});

    ASSERT_TRUE(output.allclose(expected));
    ASSERT_TRUE(output_aligned.allclose(expected_aligned));
  }
  {
    // 3D affine.
    auto theta = torch::arange(1., 13).view(std::vector<int64_t>({1, 3, 4}));
    auto size = std::vector<int64_t>({1, 1, 3, 2, 2});
    auto align_corners = true;
    auto output = F::affine_grid(theta, size, !align_corners);
    auto expected = torch::tensor(
        {{{{{0.5000, -2.1667, -4.8333}, {1.5000, 2.8333, 4.1667}},
           {{2.5000, 3.8333, 5.1667}, {3.5000, 8.8333, 14.1667}}},
          {{{2.5000, 2.5000, 2.5000}, {3.5000, 7.5000, 11.5000}},
           {{4.5000, 8.5000, 12.5000}, {5.5000, 13.5000, 21.5000}}},
          {{{4.5000, 7.1667, 9.8333}, {5.5000, 12.1667, 18.8333}},
           {{6.5000, 13.1667, 19.8333}, {7.5000, 18.1667, 28.8333}}}}});
    auto output_aligned = F::affine_grid(theta, size, align_corners);
    auto expected_aligned = torch::tensor(
        {{{{{-2.0, -10.0, -18.0}, {0.0, 0.0, 0.0}},
           {{2.0, 2.0, 2.0}, {4.0, 12.0, 20.0}}},
          {{{1.0, -3.0, -7.0}, {3.0, 7.0, 11.0}},
           {{5.0, 9.0, 13.0}, {7.0, 19.0, 31.0}}},
          {{{4.0, 4.0, 4.0}, {6.0, 14.0, 22.0}},
           {{8.0, 16.0, 24.0}, {10.0, 26.0, 42.0}}}}});

    ASSERT_TRUE(output.allclose(expected, 1e-2));
    ASSERT_TRUE(output_aligned.allclose(expected_aligned));
  }
  {
    auto theta = torch::empty({1, 2, 3}, torch::kDouble);
    auto size = std::vector<int64_t>({1, 1, 2, 2});
    ASSERT_THROWS_WITH(
        F::affine_grid(torch::empty({2, 2, 3}), {-1, 1, 2, 2}),
        "Expected non-zero, positive output size. Got [-1, 1, 2, 2]");
    ASSERT_THROWS_WITH(
        F::affine_grid(torch::empty({2, 2, 3}, torch::kInt), size),
        "Expected theta to have floating point type, but got int");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta[0], size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [2, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.unsqueeze(0), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 1, 2, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 2, 1}), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 4, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 1, 2}), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 2, 6].");
  }
  {
    auto theta = torch::empty({1, 3, 4}, torch::kDouble);
    auto size = std::vector<int64_t>({1, 1, 2, 2, 3});
    ASSERT_THROWS_WITH(
        F::affine_grid(theta[0], size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [3, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.unsqueeze(0), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 1, 3, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 2, 1}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 6, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 1, 2}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 3, 8].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1, 1, 2, 2, 3}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1, 1, 2, 2, 3]");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1]");
  }
}

TEST_F(FunctionalTest, MultiMarginLoss) {
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  auto input = torch::tensor(
      {{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  auto output = F::multi_margin_loss(
      input, target, F::MultiMarginLossFuncOptions().margin(2).weight(weight));
  auto expected = torch::tensor({0.305556}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, CosineEmbeddingLoss) {
  auto input1 = torch::tensor({{2, 3, 4}, {6, 2, 4}});
  auto input2 = torch::tensor({{2, 3, 5}, {9, 12, 0}});
  auto target = torch::tensor({1, -1});
  auto output = F::cosine_embedding_loss(
      input1, input2, target, F::CosineEmbeddingLossFuncOptions().margin(0.5));
  auto expected = torch::tensor({0.1004}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-4));
}

TEST_F(FunctionalTest, MultiLabelMarginLossDefaultOptions) {
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = F::multilabel_margin_loss(input, target);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, MultiLabelMarginLossNoReduction) {
  auto input = torch::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = F::multilabel_margin_loss(input, target, torch::kNone);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(FunctionalTest, TripletMarginLoss) {
  auto anchor = torch::tensor({{3., 3.}}, torch::kFloat);
  auto positive = torch::tensor({{2., 2.}}, torch::kFloat);
  auto negative = torch::tensor({{0., 0.}}, torch::kFloat);
  auto output = F::triplet_margin_loss(
      anchor,
      positive,
      negative,
      F::TripletMarginLossFuncOptions().margin(1.0));
  auto expected = torch::tensor({0.}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, TripletMarginWithDistanceLossDefaultParity) {
  // Check that if we use torch::pairwise_distance with the default
  // TripletMarginLoss options as our distance function, the outputs
  // are equal (i.e., equal under defaults).

  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      torch::kSum, torch::kMean, torch::kNone};
  std::vector<float> margins = {0.5, 1.0, 1.5};
  std::vector<bool> swaps = {true, false};

  for (auto& reduction : reductions) {
    for (auto& margin : margins) {
      for (const auto& swap : swaps) {
        auto anchor = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        auto positive = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        auto negative = torch::randn(
            {100, 128}, torch::dtype(torch::kFloat).requires_grad(true));

        auto basicOptions = F::TripletMarginLossFuncOptions()
                                .reduction(reduction)
                                .margin(margin)
                                .swap(swap);
        auto distanceOptions = F::TripletMarginWithDistanceLossFuncOptions()
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);
        TripletMarginLoss basicLoss(basicOptions);
        TripletMarginWithDistanceLoss distanceLoss(distanceOptions);

        auto basicOutput =
            F::triplet_margin_loss(anchor, positive, negative, basicOptions);
        auto distanceOutput = F::triplet_margin_with_distance_loss(
            anchor, positive, negative, distanceOptions);

        ASSERT_TRUE(distanceOutput.allclose(basicOutput, 1e-6, 1e-6));

        // handle for torch::kNone reduction
        auto sum = distanceOutput.sum();
        sum.backward();
        ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
        ASSERT_EQ(positive.sizes(), positive.grad().sizes());
        ASSERT_EQ(negative.sizes(), negative.grad().sizes());
      }
    }
  }
}

TEST_F(FunctionalTest, NLLLoss) {
  auto input = torch::tensor(
      {{-0.1315, -3.1315, -2.5315},
       {-3.7038, -0.1038, -2.6038},
       {-2.3422, -1.3422, -0.4422}},
      torch::kFloat);
  auto target = torch::tensor({1, 0, 2}, torch::kLong);
  auto output = F::nll_loss(
      input,
      target,
      F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
  auto expected = torch::tensor(2.4258, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_TRUE(F::nll_loss(input, target).allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, CrossEntropy) {
  auto input = torch::tensor({{3., 3.}, {2., 2.}}, torch::kFloat);
  auto target = torch::tensor({0, 1}, torch::kLong);
  auto output = F::cross_entropy(
      input,
      target,
      F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
  auto expected = torch::tensor(0.6931, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_TRUE(F::cross_entropy(input, target).allclose(expected, 1e-04));

  // label smoothing with class indices
  input = torch::tensor({{3., 1.}, {1., 2.}}, torch::kFloat);
  output = F::cross_entropy(
      input,
      target,
      F::CrossEntropyFuncOptions().label_smoothing(0.15).reduction(
          torch::kMean));
  expected = torch::tensor(0.3326, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));

  // label smoothing with target probabilities
  target = torch::tensor({{0.8, 0.2}, {0.1, 0.9}}, torch::kFloat);
  output = F::cross_entropy(
      input,
      target,
      F::CrossEntropyFuncOptions().label_smoothing(0.2).reduction(
          torch::kMean));
  expected = torch::tensor(0.5701, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, MaxUnpool1d) {
  auto x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  x = torch::tensor(
      {{2, 4, 5}}, torch::dtype(torch::kFloat).requires_grad(true));
  indices = torch::tensor({{1, 3, 4}}, torch::kLong);
  y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{0, 2, 0, 4, 5, 0, 0, 0, 0}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 9}));

  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(
      x,
      indices,
      F::MaxUnpool1dFuncOptions(3).output_size(
          std::vector<int64_t>({1, 1, 9})));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(
      x, indices, F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

TEST_F(FunctionalTest, MaxUnpool2d) {
  auto indices = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
      torch::kLong);
  auto x = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto y = F::max_unpool2d(
      x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{0, 0, 0, 0, 0},
             {0, 6, 0, 8, 9},
             {0, 0, 0, 0, 0},
             {0, 16, 0, 18, 19},
             {0, 21, 0, 23, 24}}},
           {{{0, 0, 0, 0, 0},
             {0, 31, 0, 33, 34},
             {0, 0, 0, 0, 0},
             {0, 41, 0, 43, 44},
             {0, 46, 0, 48, 49}}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 1, 5, 5}));

  indices = torch::tensor(
      {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
       {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
      torch::kLong);
  x = torch::tensor(
      {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
       {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  y = F::max_unpool2d(
      x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{0, 0, 0, 0, 0},
            {0, 6, 0, 8, 9},
            {0, 0, 0, 0, 0},
            {0, 16, 0, 18, 19},
            {0, 21, 0, 23, 24}},
           {{0, 0, 0, 0, 0},
            {0, 31, 0, 33, 34},
            {0, 0, 0, 0, 0},
            {0, 41, 0, 43, 44},
            {0, 46, 0, 48, 49}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 5, 5}));
}

TEST_F(FunctionalTest, MaxUnpool3d) {
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  auto x = torch::tensor(
      {{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));

  indices = torch::tensor({{{{26}}}}, torch::kLong);
  x = torch::tensor(
      {{{{26}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  y = F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(FunctionalTest, ELU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto x_bf16 =
          torch::linspace(-10.0, 10.0, size * size * size).to(torch::kBFloat16);
      x_bf16.resize_({size, size, size});

      auto y_exp = torch::max(torch::zeros_like(x), x) +
          torch::min(torch::zeros_like(x), alpha * (torch::expm1(x)));
      auto y = F::elu(x, F::ELUFuncOptions().alpha(alpha).inplace(inplace));
      auto y_bf16 =
          F::elu(x_bf16, F::ELUFuncOptions().alpha(alpha).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      ASSERT_TRUE(torch::allclose(y_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
        ASSERT_TRUE(torch::allclose(x_bf16.to(torch::kFloat), y, 1e-2, 1e-2));
      }
    }
  }
  ASSERT_TRUE(F::elu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, SELU) {
  {
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    for (const auto inplace : {false, true}) {
      auto input = torch::randn({5, 5});
      auto input_bf16 = input.clone().to(torch::kBFloat16);
      auto expected = scale *
          (torch::max(torch::zeros_like(input), input) +
           torch::min(torch::zeros_like(input), alpha * (torch::expm1(input))));
      auto output = F::selu(input, inplace);
      auto output_bf16 = F::selu(input_bf16, inplace);

      ASSERT_TRUE(output.allclose(expected));
      ASSERT_TRUE(output_bf16.to(torch::kFloat).allclose(output, 1e-2, 1e-2));
      if (inplace) {
        ASSERT_TRUE(input.allclose(expected));
        ASSERT_TRUE(input_bf16.to(torch::kFloat).allclose(output, 1e-2, 1e-2));
      }
    }
  }
  {
    auto input = torch::arange(0, 9, torch::kDouble).view({3, 3});
    auto output = F::selu(input);
    auto expected = F::selu(input, false);
    ASSERT_TRUE(output.allclose(expected));
  }
  ASSERT_TRUE(F::selu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, GLU) {
  int64_t dim = 1;
  auto input = torch::randn({4, 2}, torch::requires_grad());
  auto output = F::glu(input, dim);
  auto input_size = input.sizes()[dim] / 2;
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  auto expected = first_half * torch::sigmoid(second_half);

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(F::glu(input).allclose(expected));
}

TEST_F(FunctionalTest, GELU) {
  const auto x = torch::linspace(-3.0, 3.0, 100);
  const auto y_exp = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
  const auto y = F::gelu(x, F::GELUFuncOptions().approximate("none"));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(FunctionalTest, TanhGELU) {
  const auto x = torch::linspace(-3.0, 3.0, 100);
  const auto inner = std::sqrt(2 / M_PI) * (x + 0.044715 * x.pow(3.0));
  const auto y_exp = 0.5 * x * (1.0 + inner.tanh());
  const auto y = F::gelu(x, F::GELUFuncOptions().approximate("tanh"));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(FunctionalTest, Hardshrink) {
  const auto size = 3;
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = F::hardshrink(x, F::HardshrinkFuncOptions().lambda(lambda));
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x.abs() > lambda) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(FunctionalTest, OneHot) {
  { // Test #1
    auto x = torch::arange(0, 5, torch::kLong);
    auto y = F::one_hot(x % 3);
    auto expected = torch::tensor(
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, torch::kLong);

    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 3}));
  }

  { // Test #2
    auto x = torch::arange(0, 5, torch::kLong);
    auto y = F::one_hot(x % 3, 5);
    auto expected = torch::tensor(
        {{1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0}},
        torch::kLong);

    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 5}));
  }

  { // Test #3
    auto x = torch::arange(0, 6, torch::kLong);
    auto y = F::one_hot(x.view(std::vector<int64_t>({3, 2})) % 3);
    auto expected = torch::tensor(
        {{{1, 0, 0}, {0, 1, 0}},
         {{0, 0, 1}, {1, 0, 0}},
         {{0, 1, 0}, {0, 0, 1}}},
        torch::kLong);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({3, 2, 3}));
  }
}

TEST_F(FunctionalTest, Hardtanh) {
  const auto size = 3;
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    for (const auto max_val : {0.0, 0.42, 1.0, 4.2}) {
      for (const auto inplace : {false, true}) {
        auto x = torch::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        auto y_exp = (x < min_val) * min_val +
            ((x >= min_val) * (x <= max_val)) * x + (x > max_val) * max_val;
        auto y = F::hardtanh(
            x,
            F::HardtanhFuncOptions().min_val(min_val).max_val(max_val).inplace(
                inplace));

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  ASSERT_TRUE(F::hardtanh(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, LeakyReLU) {
  const auto size = 3;
  for (const auto negative_slope : {0.0, 0.42, 1.0}) {
    for (const auto inplace : {false, true}) {
      for (const auto type : {torch::kFloat, torch::kBFloat16}) {
        auto x = torch::linspace(-10.0, 10.0, size * size * size).to(type);
        x.resize_({size, size, size});
        auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
        auto y = F::leaky_relu(
            x,
            F::LeakyReLUFuncOptions()
                .negative_slope(negative_slope)
                .inplace(inplace));

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  ASSERT_TRUE(F::leaky_relu(torch::tensor(1.)).defined());
}

TEST_F(FunctionalTest, LogSigmoid) {
  const auto size = 3;
  LogSigmoid model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y = F::logsigmoid(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = torch::log(
      torch::ones_like(x) / (torch::ones_like(x) + torch::exp(torch::neg(x))));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(FunctionalTest, GumbelSoftmax) {
  // Test 1: No-options
  {
    auto logits = torch::randn({5});
    int expected_count = 1;
    auto y_draw = F::gumbel_softmax(logits);

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  // Test 2: 1D shape, 0 and -1 dim
  for (const auto dim : {0, -1}) {
    auto logits = torch::randn({5});
    int expected_count = 1;
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dim));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  { // Test 3: 2D shape, 1 dim
    auto logits = torch::randn({5, 4});
    int expected_count = 5;
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(1));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  // Test 4: 3D shape, 1 and -1 dim
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int dims[] = {1, -1};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  int expected[] = {5 * 3, 5 * 4};
  for (const auto i : c10::irange(2)) {
    auto logits = torch::randn({5, 4, 3});
    int expected_count = expected[i];
    auto y_draw = F::gumbel_softmax(
        logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dims[i]));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(
        y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  { // Test 5: Straight through
    int num_draws = 100;
    auto logits = torch::tensor({{0.2, 0.8, 0.1}});
    logits = logits.reshape({1, 3});
    logits.requires_grad();
    auto probs = logits.softmax(-1);

    auto counts = torch::zeros_like(logits);
    torch::Tensor y_draw;
    for ([[maybe_unused]] const auto i : c10::irange(num_draws)) {
      y_draw =
          F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true));
      counts += y_draw;
    }

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Each experiment should result in 1 draw
    ASSERT_EQ(counts.sum().item<int>(), num_draws);

    // Check results are asymptotically as expected
    auto expected = probs * num_draws;
    // ~z is approximately N(0,1) for unbiased count
    auto z = (counts - expected) / (expected * (1 - probs)).sqrt();
    // A (lazy) approximate 99% two-sided test:
    // occurs with prob alpha~>=0.01 if unbiased
    ASSERT_LT(z.abs().max().item<float>(), 2.58);
  }
}

TEST_F(FunctionalTest, Softmax) {
  auto input = t
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 36 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `F`

**Classes/Structs**: `FunctionalTest`, `indices`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
- `torch/torch.h`
- `test/cpp/api/support.h`


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
python test/cpp/api/functional.cpp
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

- **File Documentation**: `functional.cpp_docs.md`
- **Keyword Index**: `functional.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
