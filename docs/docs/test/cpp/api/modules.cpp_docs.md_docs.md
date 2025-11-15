# Documentation: `docs/test/cpp/api/modules.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/modules.cpp_docs.md`
- **Size**: 52,595 bytes (51.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/modules.cpp`

## File Metadata

- **Path**: `test/cpp/api/modules.cpp`
- **Size**: 194,559 bytes (190.00 KB)
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

#include <torch/expanding_array.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/options/activation.h>
#include <limits>
#include <random>

using namespace torch::nn;
using namespace torch::test;

class TestModel : public torch::nn::Module {
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))),
        l2(register_module("l2", Linear(3, 5))),
        l3(register_module("l3", Linear(5, 100))) {}

  Linear l1, l2, l3;
};

class NestedModel : public torch::nn::Module {
 public:
  NestedModel()
      : param_(register_parameter("param", torch::empty({3, 2, 21}))),
        l1(register_module("l1", Linear(5, 20))),
        t(register_module("test", std::make_shared<TestModel>())) {}

  torch::Tensor param_;
  Linear l1;
  std::shared_ptr<TestModel> t;
};

struct ModulesTest : torch::test::SeedingFixture {};

TEST_F(ModulesTest, Conv1d) {
  Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      torch::arange(18, torch::dtype(torch::kFloat)).reshape({2, 3, 3}));
  auto x = torch::arange(30, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({2, 3, 5});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{312., 348., 384.}, {798., 915., 1032.}},

       {{852., 888., 924.}, {2553., 2670., 2787.}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}

TEST_F(ModulesTest, Conv1dSameStrided) {
  auto options = Conv1dOptions(3, 2, 3);
  options.stride(1).padding(torch::kSame);
  Conv1d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv1d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, Conv1dIvalidArg) {
  auto options = Conv1dOptions(3, 2, 3).groups(-1);
  ASSERT_THROWS_WITH(
      Conv1d(options), "in_channels, groups and out_channels must");
}

TEST_F(ModulesTest, Conv2dEven) {
  Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      torch::arange(54, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 3}));
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{{15219., 15570., 15921.},
         {16974., 17325., 17676.},
         {18729., 19080., 19431.}},

        {{37818., 38898., 39978.},
         {43218., 44298., 45378.},
         {48618., 49698., 50778.}}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}

TEST_F(ModulesTest, Conv2dUneven) {
  Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  model->weight.set_data(
      torch::arange(36, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 2}));
  auto x = torch::arange(60, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 4});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{{5289., 5442., 5595.}, {5901., 6054., 6207.}, {6513., 6666., 6819.}},

        {{13227., 13704., 14181.},
         {15135., 15612., 16089.},
         {17043., 17520., 17997.}}}},
      torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, Conv2dSameStrided) {
  auto options = Conv2dOptions(3, 2, {3, 4});
  options.stride(1).padding(torch::kSame);
  Conv2d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv2d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  ASSERT_THROWS_WITH(
      [&] { Conv2d model_invalid(options.stride({1, 2})); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, Conv3d) {
  Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      torch::arange(162, torch::dtype(torch::kFloat)).reshape({2, 3, 3, 3, 3}));
  auto x = torch::arange(375, torch::dtype(torch::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5, 5});
  auto y = model(x);
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

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(model->weight.grad().numel() == 3 * 2 * 3 * 3 * 3);
}

TEST_F(ModulesTest, Conv3dSameStrided) {
  auto options = Conv3dOptions(3, 2, {3, 4, 5});
  options.stride(1).padding(torch::kSame);
  Conv3d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv3d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  ASSERT_THROWS_WITH(
      [&] { Conv3d model_invalid(options.stride({1, 2, 1})); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, ConvTranspose1d) {
  ConvTranspose1d model(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(torch::arange(18.).view({2, 3, 3}));
  auto x = torch::arange(20.).reshape({2, 2, 5});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{45., 104., 179., 212., 245., 188., 107.},
        {60., 140., 242., 293., 344., 260., 146.},
        {75., 176., 305., 374., 443., 332., 185.}},
       {{135., 304., 509., 542., 575., 428., 237.},
        {210., 460., 752., 803., 854., 620., 336.},
        {285., 616., 995., 1064., 1133., 812., 435.}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}

TEST_F(ModulesTest, ConvTranspose2dEven) {
  ConvTranspose2d model(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(torch::arange(54.).view({2, 3, 3, 3}));
  auto x = torch::arange(50.).view({1, 2, 5, 5});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{{675., 1402., 2183., 2270., 2357., 1634., 849.},
         {1560., 3240., 5044., 5236., 5428., 3760., 1952.},
         {2685., 5574., 8673., 8988., 9303., 6438., 3339.},
         {3180., 6594., 10248., 10563., 10878., 7518., 3894.},
         {3675., 7614., 11823., 12138., 12453., 8598., 4449.},
         {2820., 5832., 9040., 9268., 9496., 6544., 3380.},
         {1605., 3314., 5129., 5252., 5375., 3698., 1907.}},
        {{900., 1870., 2912., 3053., 3194., 2210., 1146.},
         {2100., 4356., 6772., 7072., 7372., 5092., 2636.},
         {3630., 7518., 11670., 12147., 12624., 8706., 4500.},
         {4395., 9078., 14055., 14532., 15009., 10326., 5325.},
         {5160., 10638., 16440., 16917., 17394., 11946., 6150.},
         {3900., 8028., 12388., 12724., 13060., 8956., 4604.},
         {2190., 4502., 6938., 7115., 7292., 4994., 2564.}},
        {{1125., 2338., 3641., 3836., 4031., 2786., 1443.},
         {2640., 5472., 8500., 8908., 9316., 6424., 3320.},
         {4575., 9462., 14667., 15306., 15945., 10974., 5661.},
         {5610., 11562., 17862., 18501., 19140., 13134., 6756.},
         {6645., 13662., 21057., 21696., 22335., 15294., 7851.},
         {4980., 10224., 15736., 16180., 16624., 11368., 5828.},
         {2775., 5690., 8747., 8978., 9209., 6290., 3221.}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}

TEST_F(ModulesTest, ConvTranspose2dUneven) {
  ConvTranspose2d model(
      ConvTranspose2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  model->weight.set_data(torch::arange(36.).view({2, 3, 3, 2}));
  auto x = torch::arange(40.).view({1, 2, 5, 4});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{{360., 758., 796., 834., 440.},
         {832., 1752., 1836., 1920., 1012.},
         {1432., 3014., 3152., 3290., 1732.},
         {1696., 3566., 3704., 3842., 2020.},
         {1960., 4118., 4256., 4394., 2308.},
         {1504., 3152., 3252., 3352., 1756.},
         {856., 1790., 1844., 1898., 992.}},
        {{480., 1010., 1072., 1134., 596.},
         {1120., 2352., 2484., 2616., 1372.},
         {1936., 4058., 4268., 4478., 2344.},
         {2344., 4898., 5108., 5318., 2776.},
         {2752., 5738., 5948., 6158., 3208.},
         {2080., 4328., 4476., 4624., 2404.},
         {1168., 2426., 2504., 2582., 1340.}},
        {{600., 1262., 1348., 1434., 752.},
         {1408., 2952., 3132., 3312., 1732.},
         {2440., 5102., 5384., 5666., 2956.},
         {2992., 6230., 6512., 6794., 3532.},
         {3544., 7358., 7640., 7922., 4108.},
         {2656., 5504., 5700., 5896., 3052.},
         {1480., 3062., 3164., 3266., 1688.}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, ConvTranspose3d) {
  ConvTranspose3d model(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false));
  model->weight.set_data(torch::arange(32.).reshape({2, 2, 2, 2, 2}));
  auto x = torch::arange(16.).reshape({1, 2, 2, 2, 2});
  auto y = model(x);
  auto expected = torch::tensor(
      {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
         {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
         {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
        {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
         {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
         {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  torch::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(model->weight.grad().numel() == 2 * 2 * 2 * 2 * 2);
}

TEST_F(ModulesTest, MaxPool1d) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool1dReturnIndices) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));

  ASSERT_TRUE(
      torch::allclose(indices, torch::tensor({{{0, 2}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool2dEven) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dUneven) {
  MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dReturnIndices) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3d) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3dReturnIndices) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}},
          torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool1d) {
  AvgPool1d model(AvgPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, AvgPool2dEven) {
  AvgPool2d model(AvgPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool2dUneven) {
  AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool3d) {
  AvgPool3d model(AvgPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool2d) {
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool2dReturnIndices) {
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  ASSERT_TRUE(torch::allclose(
      indices, torch::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}})));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool3d) {
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool3dReturnIndices) {
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}})));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, LPPool1d) {
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  LPPool1d model(LPPool1dOptions(norm_type, kernel_size).stride(stride));
  auto x = torch::ones({1, 1, 5});
  auto y = model(x);
  auto expected =
      (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) *
       kernel_size)
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(ModulesTest, LPPool2d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  LPPool2d model(LPPool2dOptions(norm_type, kernel_size).stride(stride));
  auto x = torch::ones({1, 1, 2, 5});
  auto y = model(x);
  auto expected =
      (torch::pow(torch::tensor({{{{1, 1}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 2}));
}

TEST_F(ModulesTest, LPPool3d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({1, 2, 3});

  LPPool3d model(LPPool3dOptions(norm_type, kernel_size).stride(stride));
  auto x = torch::ones({1, 1, 1, 2, 5});
  auto y = model(x);
  auto expected =
      (torch::pow(torch::tensor({{{{{1, 1}}}}}, torch::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1] * kernel_size[2]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 5);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 1, 1, 2}));
}

TEST_F(ModulesTest, Identity) {
  Identity identity;
  auto input = torch::tensor(
      {{1, 3, 4}, {2, 3, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = identity->forward(input);
  auto expected = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, Flatten) {
  Flatten flatten;
  auto input = torch::tensor(
      {{1, 3, 4}, {2, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = flatten->forward(input);
  auto expected = torch::tensor({{1, 3, 4}, {2, 5, 6}}, torch::kFloat);
  auto s = output.sum();

  s.backward();
  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));

  // Testing with optional arguments start_dim and end_dim
  Flatten flatten_optional_dims(FlattenOptions().start_dim(2).end_dim(3));
  input = torch::tensor(
      {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
       {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}},
      torch::dtype(torch::kFloat)
          .requires_grad(true)); // Tensor with sizes (2, 2, 2, 2)

  output = flatten_optional_dims->forward(input);
  expected = torch::tensor(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}},
      torch::kFloat); // Tensor with sizes (2, 2, 4)

  s = output.sum();
  s.backward();
  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, Unflatten) {
  // Non-named tensor
  Unflatten unflatten(UnflattenOptions(0, {2, 2}));
  auto output = unflatten->forward(torch::tensor({1, 2, 3, 4}));
  auto expected = torch::tensor({{1, 2}, {3, 4}});
  ASSERT_TRUE(torch::equal(output, expected));

  // Named tensor
  auto make_dimnames = [](std::vector<std::string> names) {
    std::vector<torch::Dimname> dimnames;
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto name : names) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      dimnames.push_back(
          torch::Dimname::fromSymbol(torch::Symbol::dimname(name)));
    }
    return dimnames;
  };

  unflatten = Unflatten(UnflattenOptions(
      "B",
      {std::pair<std::string, int64_t>{"B1", 2},
       std::pair<std::string, int64_t>{"B2", 2}}));
  output = unflatten->forward(
      torch::tensor({{1, 2, 3, 4}}).refine_names(make_dimnames({"A", "B"})));
  expected = torch::tensor({{{1, 2}, {3, 4}}})
                 .refine_names(make_dimnames({"A", "B1", "B2"}));
  ASSERT_TRUE(torch::equal(output, expected));
}

TEST_F(ModulesTest, AdaptiveMaxPool1d) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool1dReturnIndices) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
  ASSERT_TRUE(
      torch::allclose(indices, torch::tensor({{{1, 3, 4}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dEven) {
  AdaptiveMaxPool2d model(3);
  auto x = torch::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = torch::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesEven) {
  AdaptiveMaxPool2d model(3);
  auto x = torch::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
          },
          torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = torch::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{5, 7}, {13, 15}, {17, 19}},
          },
          torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3d) {
  AdaptiveMaxPool3d model(3);
  auto x = torch::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3dReturnIndices) {
  AdaptiveMaxPool3d model(3);
  auto x = torch::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      indices,
      torch::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool1d) {
  AdaptiveAvgPool1d model(3);
  auto x = torch::tensor(
      {{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{1.5, 3.0, 4.5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dEven) {
  AdaptiveAvgPool2d model(3);
  auto x = torch::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{3.0, 4.5, 6.0}, {10.5, 12.0, 13.5}, {18.0, 19.5, 21.0}},
              {{28.0, 29.5, 31.0}, {35.5, 37.0, 38.5}, {43.0, 44.5, 46.0}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dUneven) {
  AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
  auto x = torch::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{2.5, 4.5}, {8.5, 10.5}, {14.5, 16.5}},
              {{22.5, 24.5}, {28.5, 30.5}, {34.5, 36.5}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveAvgPool3d) {
  AdaptiveAvgPool3d model(3);
  auto x = torch::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {
              {{10.5, 11.5, 12.5}, {14.5, 15.5, 16.5}, {18.5, 19.5, 20.5}},
              {{26.5, 27.5, 28.5}, {30.5, 31.5, 32.5}, {34.5, 35.5, 36.5}},
              {{42.5, 43.5, 44.5}, {46.5, 47.5, 48.5}, {50.5, 51.5, 52.5}},
          },
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool1d) {
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool1d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  x = torch::tensor(
      {{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  model = MaxUnpool1d{MaxUnpool1dOptions(3).stride(2).padding(1)};
  y = model->forward(x, indices, std::vector<int64_t>({1, 1, 5}));

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

TEST_F(ModulesTest, MaxPool1d_MaxUnpool1d) {
  MaxPool1d pool{MaxPool1dOptions(2).stride(2)};
  MaxUnpool1d unpool{MaxUnpool1dOptions(2).stride(2)};
  auto input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8}}}, torch::kFloat);
  auto [output, indices] = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, torch::kFloat)));

  // Example showcasing the use of output_size
  input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8, 9}}}, torch::kFloat);
  std::tie(output, indices) = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices, input.sizes().vec()),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8, 0}}}, torch::kFloat)));
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, torch::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool2d) {
  auto indices = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
      torch::kLong);
  auto x = torch::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool2d{MaxUnpool2dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices);

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
}

TEST_F(ModulesTest, MaxPool2d_MaxUnpool2d) {
  MaxPool2d pool{MaxPool2dOptions(2).stride(2)};
  MaxUnpool2d unpool{MaxUnpool2dOptions(2).stride(2)};
  auto input = torch::tensor(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}},
      torch::kFloat);
  auto [output, indices] = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
      unpool(output, indices),
      torch::tensor(
          {{{{0, 0, 0, 0}, {0, 6, 0, 8}, {0, 0, 0, 0}, {0, 14, 0, 16}}}},
          torch::kFloat)));

  ASSERT_TRUE(torch::allclose(
      unpool(output, indices, std::vector<int64_t>{1, 1, 5, 5}),
      torch::tensor(
          {{{{0, 0, 0, 0, 0},
             {6, 0, 8, 0, 0},
             {0, 0, 0, 14, 0},
             {16, 0, 0, 0, 0},
             {0, 0, 0, 0, 0}}}},
          torch::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool3d) {
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  auto x = torch::tensor(
      {{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool3dOutputSize) {
  auto indices = torch::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}}, torch::kLong);
  auto x = torch::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}},
      torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{MaxUnpool3dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices, std::vector<int64_t>({1, 1, 4, 4, 4}));

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(torch::allclose(
      y,
      torch::tensor(
          {{{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 21, 0, 23}, {0, 0, 0, 0}, {0, 29, 0, 31}},
             {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 53, 0, 55}, {0, 0, 0, 0}, {0, 61, 0, 63}}}}},
          torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 4, 4, 4}));
}

TEST_F(ModulesTest, MaxPool3d_MaxUnpool3d) {
  MaxPool3d pool{MaxPool3dOptions(3).stride(2)};
  MaxUnpool3d unpool{MaxUnpool3dOptions(3).stride(2)};
  auto input = torch::randn({20, 16, 51, 33, 15});
  auto [output, indices] = pool->forward_with_indices(input);
  auto unpooled_output = unpool(output, indices);
  ASSERT_EQ(
      unpooled_output.sizes(), std::vector<int64_t>({20, 16, 51, 33, 15}));
}

TEST_F(ModulesTest, Linear) {
  {
    Linear model(5, 2);
    auto x = torch::randn({10, 5}, torch::requires_grad());
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = torch::addmm(model->bias, x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    Linear model(LinearOptions(5, 2).bias(false));
    auto x = torch::randn({10, 5}, torch::requires_grad());
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = torch::mm(x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, LocalResponseNorm) {
  {
    LocalResponseNorm model(LocalResponseNormOptions(2));
    const auto x =
        torch::arange(100., 136, torch::requires_grad()).reshape({2, 3, 3, 2});
    auto y = model(x);
    const auto y_exp = torch::tensor(
        {{{{73.7788, 74.1462}, {74.5031, 74.8572}, {75.2010, 75.5420}},

          {{61.6057, 61.7227}, {61.8347, 61.9418}, {62.0441, 62.1418}},

          {{62.2349, 62.3235}, {62.4077, 62.4877}, {62.5635, 62.6353}}},

         {{{79.3915, 79.6491}, {79.8978, 80.1446}, {80.3827, 80.6190}},

          {{63.0317, 63.0742}, {63.1135, 63.1496}, {63.1826, 63.2126}},

          {{63.2396, 63.2637}, {63.2850, 63.3036}, {63.3195, 63.3328}}}},
        torch::kFloat);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 4);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.sizes(), x.sizes());
    ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
  }
}

TEST_F(ModulesTest, LayerNorm) {
  LayerNorm model(LayerNormOptions({2, 2}).eps(2e-5));
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = model(x);
  auto y_exp = torch::layer_norm(x, {2, 2}, model->weight, model->bias, 2e-5);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 2 * 2);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, GroupNorm) {
  GroupNorm model(GroupNormOptions(2, 2).eps(2e-5));
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = model(x);
  auto y_exp = torch::group_norm(x, 2, model->weight, model->bias, 2e-5);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 2);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Bilinear) {
  Bilinear model(5, 3, 2);
  auto x1 = torch::randn({10, 5}, torch::requires_grad());
  auto x2 = torch::randn({10, 3}, torch::requires_grad());
  auto y = model(x1, x2);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5 * 3);
}

TEST_F(ModulesTest, Fold) {
  {
    Fold model(FoldOptions({3, 2}, {2, 2}));
    auto input = torch::ones({1, 3 * 2 * 2, 2}, torch::requires_grad());
    auto output = model(input);
    auto expected = torch::tensor(
        {{{{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}}}},
        torch::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 3, 2}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Fold model(FoldOptions({8, 8}, {3, 3}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 3, 16, 16})),
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported (got 4D)");
  }
}

TEST_F(ModulesTest, Unfold) {
  {
    Unfold model(UnfoldOptions({2, 2}).padding(1).stride(2));
    auto input =
        torch::arange(2., 14, torch::requires_grad()).view({1, 2, 2, 3});
    auto output = model(input);
    auto expected = torch::tensor(
        {{{0.0, 0.0, 0.0, 6.0},
          {0.0, 0.0, 5.0, 7.0},
          {0.0, 3.0, 0.0, 0.0},
          {2.0, 4.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 12.0},
          {0.0, 0.0, 11.0, 13.0},
          {0.0, 9.0, 0.0, 0.0},
          {8.0, 10.0, 0.0, 0.0}}},
        torch::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Unfold model(UnfoldOptions({2, 4}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 5, 2})),
        "Input Error: Only 4D input Tensors are supported (got 3D)");
  }
  {
    // calculated output shape is too small
    Unfold model(UnfoldOptions({2, 3}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 2, 2, 2})),
        "Given input with spatial size (2, 2), kernel_size=(2, 3), "
        "dilation=(1, 1), padding=(0, 0), calculated shape of the array of "
        "sliding blocks as (1, 0), but its components must be at least one.");
  }
}

TEST_F(ModulesTest, SimpleContainer) {
  auto model = std::make_shared<SimpleContainer>();
  auto l1 = model->add(Linear(10, 3), "l1");
  auto l2 = model->add(Linear(3, 5), "l2");
  auto l3 = model->add(Linear(5, 100), "l3");

  auto x = torch::randn({1000, 10}, torch::requires_grad());
  x = l1(x).clamp_min(0);
  x = l2(x).clamp_min(0);
  x = l3(x).clamp_min(0);

  x.backward(torch::ones_like(x));
  ASSERT_EQ(x.ndimension(), 2);
  ASSERT_EQ(x.size(0), 1000);
  ASSERT_EQ(x.size(1), 100);
  ASSERT_EQ(x.min().item<float>(), 0);
}

TEST_F(ModulesTest, EmbeddingBasic) {
  const int64_t dict_size = 10;
  Embedding model(dict_size, 2);
  ASSERT_TRUE(model->named_parameters().contains("weight"));
  ASSERT_EQ(model->weight.ndimension(), 2);
  ASSERT_EQ(model->weight.size(0), dict_size);
  ASSERT_EQ(model->weight.size(1), 2);

  // Cannot get gradients to change indices (input) - only for embedding
  // params
  auto x = torch::full({10}, dict_size - 1, torch::kInt64);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * dict_size);
}

TEST_F(ModulesTest, EmbeddingList) {
  Embedding model(6, 4);
  auto x = torch::full({2, 3}, 5, torch::kInt64);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.size(0), 2);
  ASSERT_EQ(y.size(1), 3);
  ASSERT_EQ(y.size(2), 4);
}

TEST_F(ModulesTest, EmbeddingFromPretrained) {
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  Embedding embedding = torch::nn::Embedding::from_pretrained(weight);
  auto input = torch::tensor({1}, torch::kLong);
  ASSERT_TRUE(torch::allclose(
      embedding(input), torch::tensor({4.0000, 5.1000, 6.3000})));
}

TEST_F(ModulesTest, EmbeddingBagFromPretrained) {
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  EmbeddingBag embeddingbag = torch::nn::EmbeddingBag::from_pretrained(weight);
  auto input = torch::zeros({{1, 2}}, torch::kLong);
  input[0] = torch::tensor({1, 0});
  ASSERT_TRUE(torch::allclose(
      embeddingbag(input), torch::tensor({2.5000, 3.7000, 4.6500})));
}

TEST_F(ModulesTest, AlphaDropout) {
  AlphaDropout alpha_dropout(0.5);
  torch::Tensor x = torch::ones(100, torch::requires_grad());
  torch::Tensor y = alpha_dropout(x);

  y.backward(torch::ones_like(y));

  ASSERT_EQ(y.ndimension(), 1);
  ASSERT_EQ(y.size(0), 100);
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  ASSERT_GT(y.sum().item<float>(), 40); // Probably

  alpha_dropout->eval();
  y = alpha_dropout(x);

  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, FeatureAlphaDropout) {
  FeatureAlphaDropout feature_alpha_dropout(0.5);
  torch::Tensor x = torch::ones({10, 10}, torch::requires_grad());
  torch::Tensor y = feature_alpha_dropout(x);

  y.backward(torch::ones_like(y));

  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 10);
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  ASSERT_GT(y.sum().item<float>(), 40); // Probably

  feature_alpha_dropout->eval();
  y = feature_alpha_dropout(x);

  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, Dropout) {
  for (const auto inplace : {false, true}) {
    Dropout dropout(DropoutOptions(0.5).inplace(inplace));
    torch::Tensor x = torch::ones(100);
    if (!inplace) {
      x.requires_grad_(true);
    }
    torch::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 1);
    ASSERT_EQ(y.size(0), 100);
    ASSERT_LT(y.sum().item<float>(), 130); // Probably
    ASSERT_GT(y.sum().item<float>(), 70); // Probably
    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(torch::ones_like(y));
    }

    dropout->eval();
    y = dropout(torch::ones(100));
    ASSERT_EQ(y.sum().item<float>(), 100);
  }
}

TEST_F(ModulesTest, Dropout2d) {
  auto p = 0.5;
  for (const auto inplace : {false, true}) {
    Dropout2d dropout(Dropout2dOptions(p).inplace(inplace));
    torch::Tensor x = torch::empty({50, 50, 2, 2}).fill_(1 - p);
    if (!inplace) {
      x.requires_grad_(true);
    }
    torch::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 4);
    ASSERT_EQ(y.size(0), 50);
    ASSERT_EQ(y.size(1), 50);
    ASSERT_EQ(y.size(2), 2);
    ASSERT_EQ(y.size(3), 2);
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(torch::ones_like(y));
    }

    dropout->eval();
    y = dropout(torch::ones({2, 2, 10, 10}));
    ASSERT_EQ(y.sum().item<float>(), 400);
  }
}

TEST_F(ModulesTest, Dropout3d) {
  for (const auto inplace : {false, true}) {
    auto p = 0.5;
    Dropout3d dropout(Dropout3dOptions(p).inplace(inplace));
    torch::Tensor x = torch::empty({50, 50, 2, 2, 2}).fill_(1 - p);
    if (!inplace) {
      x.requires_grad_(true);
    }
    torch::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 5);
    ASSERT_EQ(y.size(0), 50);
    ASSERT_EQ(y.size(1), 50);
    ASSERT_EQ(y.size(2), 2);
    ASSERT_EQ(y.size(3), 2);
    ASSERT_EQ(y.size(4), 2);
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(torch::ones_like(y));
    }

    dropout->eval();
    y = dropout(torch::ones({4, 4, 5, 5}));
    ASSERT_EQ(y.sum().item<float>(), 400);
  }
}

TEST_F(ModulesTest, Parameters) {
  auto model = std::make_shared<NestedModel>();
  auto parameters = model->named_parameters();
  ASSERT_EQ(parameters["param"].size(0), 3);
  ASSERT_EQ(parameters["param"].size(1), 2);
  ASSERT_EQ(parameters["param"].size(2), 21);
  ASSERT_EQ(parameters["l1.bias"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(1), 5);
  ASSERT_EQ(parameters["test.l1.bias"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(1), 10);
  ASSERT_EQ(parameters["test.l2.bias"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(1), 3);
  ASSERT_EQ(parameters["test.l3.bias"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(1), 5);
}

TEST_F(ModulesTest, FunctionalCallsSuppliedFunction) {
  bool was_called = false;
  auto functional = Functional([&was_called](torch::Tensor input) {
    was_called = true;
    return input;
  });
  auto output = functional(torch::ones(5, torch::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));

  was_called = false;
  // Use the call operator overload here.
  output = functional(torch::ones(5, torch::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));
}

TEST_F(ModulesTest, FunctionalWithTorchFunction) {
  auto functional = Functional(torch::relu);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(torch::ones({}) * -1).item<float>(), 0);
}

TEST_F(ModulesTest, FunctionalArgumentBinding) {
  auto functional =
      Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 0);
}

TEST_F(ModulesTest, BatchNorm1dStateful) {
  BatchNorm1d bn(5);

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm1dStateless) {
  BatchNorm1d bn(
      BatchNorm1dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/api/modules.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `modules.cpp_docs.md_docs.md`
- **Keyword Index**: `modules.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
