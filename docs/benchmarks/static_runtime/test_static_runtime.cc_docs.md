# Documentation: `benchmarks/static_runtime/test_static_runtime.cc`

## File Metadata

- **Path**: `benchmarks/static_runtime/test_static_runtime.cc`
- **Size**: 114,580 bytes (111.89 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**.

## Original Source

```cpp
#include <ATen/core/dispatch/OperatorOptions.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <stdexcept>

#include "deep_wide_pt.h"
#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

/*
 When adding a test for an operator implemented in static runtime, there are
 several things that you need to pay attention to:

 1) if the op is an out variant, in the test script of the op,
 instead of:
    def forward(self, input):
      return myop(input)

  do:
    def forward(self, input):
      return myop(input).clone()

 This makes sure that the output of myop is managed by the memory planner and
 exercise the code path in the op impl that otherwise doesn't get exercised. The
 output of the model is not managed by the memory planner, because it needs to
 be returned to the client.

 2) The memory planner rounds up the size of each Tensor's storage to multiples
 of 64 bytes (alignment requirement on AVX512). Make sure the sizes of the input
 tensors in args2 are big enough to trigger resizing.

 3) for view ops such as aten::reshape or aten::to, if you want it to be
 replaced by the copy version with the ReplaceWithCopy pass in passes.h, you
 also want to make sure its output is not returned as the model output. The
 reason is that ReplaceWithCopy only replaces the op whose output is not an
 alias of the model output.
*/

C10_DECLARE_bool(static_runtime_enable_fast_math);

TEST(StaticRuntime, UnaryOps) {
  const auto aten_sum = R"JIT(
    def forward(self, input):
        return torch.sum(input).clone()
  )JIT";

  const auto aten_sum_0 = R"JIT(
    def forward(self, input):
        return torch.sum(input, 0).clone()
  )JIT";

  const auto aten_sum_1 = R"JIT(
    def forward(self, input):
        return torch.sum(input, 1).clone()
  )JIT";

  const auto aten_sum_0_true = R"JIT(
    def forward(self, input):
        return torch.sum(input, 0, True).clone()
  )JIT";

  const auto aten_sum_1_true = R"JIT(
    def forward(self, input):
        return torch.sum(input, 1, True).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({3, 3, 6});

  std::vector<IValue> args{a}, args2{b};

  // sum
  testStaticRuntime(aten_sum, args);
  testStaticRuntime(aten_sum_0, args);
  testStaticRuntime(aten_sum_1, args);
  testStaticRuntime(aten_sum_0_true, args);
  testStaticRuntime(aten_sum_1_true, args);

  testStaticRuntime(aten_sum, args, args2, false, false, false);
  testStaticRuntime(aten_sum_0, args, args2);
  testStaticRuntime(aten_sum_1, args, args2);
  testStaticRuntime(aten_sum_0_true, args, args2);
  testStaticRuntime(aten_sum_1_true, args, args2);
}

TEST(StaticRuntime, Max) {
  auto src_max_reduce = R"JIT(
    def forward(self, input):
        return torch.max(input).clone()
  )JIT";

  auto src_max_dim = R"JIT(
    def forward(self, input, dim: int):
        values, indices = torch.max(input, dim)
        return values.clone(), indices.clone()
  )JIT";

  auto src_max_dim_keepdim = R"JIT(
    def forward(self, input, dim: int):
        values, indices = torch.max(input, dim, keepdim=True)
        return values.clone(), indices.clone()
  )JIT";

  auto src_max_pointwise = R"JIT(
    def forward(self, input, other):
        return torch.max(input, other).clone()
  )JIT";

  auto input = at::randn({2, 3, 2});
  auto input_other = at::randn({2, 3, 2});
  auto large_input = at::randn({8, 9, 10});
  auto large_input_other = at::randn({8, 9, 10});

  testStaticRuntime(src_max_reduce, {input});
  testStaticRuntime(src_max_dim, {input, 1});
  testStaticRuntime(src_max_dim, {input, 1}, {large_input, 0});
  testStaticRuntime(src_max_dim_keepdim, {input, 0});
  testStaticRuntime(src_max_dim_keepdim, {input, 0}, {large_input, 2});
  testStaticRuntime(src_max_pointwise, {input, input_other});
  testStaticRuntime(src_max_pointwise, {input, input_other}, {large_input, large_input_other});
}

TEST(StaticRuntime, Mean) {
  const auto src_default = R"JIT(
    def forward(self, input):
        return torch.mean(input).clone()
  )JIT";
  const auto src_dtype = R"JIT(
    def forward(self, input, dtype: int):
        return torch.mean(input, dtype=dtype).clone()
  )JIT";
  const auto src_dim = R"JIT(
    def forward(self, input, dim: List[int]):
        return torch.mean(input, dim).clone()
  )JIT";
  const auto src_dim_keepdim = R"JIT(
    def forward(self, input, dim: List[int]):
        return torch.mean(input, dim, keepdim=True).clone()
  )JIT";
  const auto src_dim_dtype = R"JIT(
    def forward(self, input, dim: List[int], dtype: int):
        return torch.mean(input, dim, dtype=dtype).clone()
  )JIT";

  auto input = at::randn({2, 3, 2});
  auto large_input = at::randn({8, 7, 6, 8});

  std::vector<IValue> args_default = {input};
  std::vector<IValue> args_dtype = {input, torch::kFloat};
  std::vector<IValue> args_dim = {input, c10::List<int64_t>{0, 2}};
  std::vector<IValue> args_dim_keepdim = {input, c10::List<int64_t>{1, 2}};
  std::vector<IValue> args_dim_dtype = {input, c10::List<int64_t>{0, 1}, torch::kBFloat16};

  testStaticRuntime(src_default, args_default);
  testStaticRuntime(src_dtype, args_dtype);
  testStaticRuntime(src_dim, args_dim);
  testStaticRuntime(src_dim_keepdim, args_dim_keepdim);
  testStaticRuntime(src_dim_dtype, args_dim_dtype);

  std::vector<IValue> large_args_dim = {large_input, c10::List<int64_t>{0, 3}};
  std::vector<IValue> large_args_dim_keepdim = {large_input, c10::List<int64_t>{1, 2}};
  std::vector<IValue> large_args_dim_dtype = {large_input, c10::List<int64_t>{1, 3}, torch::kBFloat16};

  testStaticRuntime(src_dim, args_dim, large_args_dim);
  testStaticRuntime(src_dim_keepdim, args_dim_keepdim, large_args_dim_keepdim);
  testStaticRuntime(src_dim_dtype, args_dim_dtype, large_args_dim_dtype);
}

TEST(StaticRuntime, Sigmoid) {
  const auto sigmoid_script = R"JIT(
    def forward(self, inp: Tensor):
        b = torch.sigmoid(inp).clone()
        return (b)
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a}, args2{b};

  testStaticRuntime(sigmoid_script, args, /*args2=*/{}, /*use_allclose=*/true);
  testStaticRuntime(sigmoid_script, args, {args2}, /*use_allclose=*/true);

  FLAGS_static_runtime_enable_fast_math = false;
  testStaticRuntime(sigmoid_script, args, /*args2=*/{}, /*use_allclose=*/true);
  testStaticRuntime(sigmoid_script, args, {args2}, /*use_allclose=*/true);
  FLAGS_static_runtime_enable_fast_math = true;
}

TEST(StaticRuntime, Clone) {
  /*
  Clone called two times to trigger memory planner for output of first clone.
  The output of last op(second clone) is not managed by memory planner since it
  needs to be returned to the client and cannot be reused by planner.
  */
  const auto clone_script_0 = R"JIT(
    def forward(self, input):
        a = torch.clone(input).clone()
        return (a * a)
  )JIT";

  // Case: clone with different set of memory_formats
  const auto clone_script_1 = R"JIT(
    def forward(self, input: Tensor, memory_format: int):
        a = torch.clone(input, memory_format=memory_format).clone()
        return (a * a)
  )JIT";

  /*
  Case: input stride set to 0 (due to expand op)
  calls native clone instead of out variant
  */
  const auto clone_script_2 = R"JIT(
    def forward(self, input: Tensor, other:Tensor):
        a = input.expand_as(other)
        return a.clone().clone()
  )JIT";

  /*
  Case: testing the case of sliced tensor for
  testing non-contiguous tensor storage
  */
  const auto clone_script_3 = R"JIT(
    def forward(self, input: Tensor):
        a = input[:, 0:10:2]
        return a.clone().clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({3, 2}).as_strided({3, 2}, {1, 3});
  auto b_larger = at::randn({30, 20}).as_strided({30, 20}, {1, 3});
  auto c = at::randn({1, 20, 13, 8});
  auto d = at::randn({1, 0, 3, 4});
  auto e = at::randn({2, 1});
  auto f = at::randn({2, 10});
  auto g = at::randn({3, 20});
  std::vector<IValue> args_0{b, c10::MemoryFormat::Contiguous};
  std::vector<IValue> args_1{b_larger, c10::MemoryFormat::Preserve};
  std::vector<IValue> args_2{c, c10::MemoryFormat::ChannelsLast};
  std::vector<IValue> args_3{d, c10::MemoryFormat::ChannelsLast};
  std::vector<IValue> args_4{e,a};
  std::vector<IValue> args_5{e,f};

  testStaticRuntime(clone_script_0, {a});
  testStaticRuntime(clone_script_0, {a}, {b_larger});

  testStaticRuntime(clone_script_1, args_0);
  testStaticRuntime(clone_script_1, args_1);
  testStaticRuntime(clone_script_1, args_2);
  testStaticRuntime(clone_script_1, args_3);
  testStaticRuntime(clone_script_1, args_0, args_1);
  testStaticRuntime(clone_script_1, args_3, args_2);

  testStaticRuntime(clone_script_2, args_4);
  testStaticRuntime(clone_script_2, args_4, args_5);

  testStaticRuntime(clone_script_3, {f});
  testStaticRuntime(clone_script_3, {f}, {g});
}

TEST(StaticRuntime, Clamp) {
  const auto clamp_script_1 = R"JIT(
    def forward(self, inp: Tensor, min: int, max: int):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";

  const auto clamp_script_2 = R"JIT(
    def forward(self, inp: Tensor, min: Tensor, max: Tensor):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";
  auto a = at::randn({2, 3});
  auto max_t = at::full_like(a, 1);
  auto min_t = at::full_like(a, -1);

  auto b = at::randn({4, 3, 2});
  auto max_t1 = at::full_like(b, 1);
  auto min_t1 = at::full_like(b, -1);

  testStaticRuntime(clamp_script_1, {a, -1, 1});
  testStaticRuntime(clamp_script_2, {a, min_t, max_t});

  testStaticRuntime(clamp_script_1, {a, -1, 1}, {b, -1, 1});
  testStaticRuntime(clamp_script_2, {a, min_t, max_t}, {b, max_t1, min_t1});
}

TEST(StaticRuntime, ClampMinOnly) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, min: float):
        a = torch.clamp(inp, min, None).clone()
        return (a)
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});
  testStaticRuntime(src, {a, 0.5});
  testStaticRuntime(src, {a, 0.5}, {b, 0.25});
}

TEST(StaticRuntime, ClampMaxOnly) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, max: float):
        a = torch.clamp(inp, None, max).clone()
        return (a)
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});
  testStaticRuntime(src, {a, 0.5});
  testStaticRuntime(src, {a, 0.5}, {b, 0.25});
}

TEST(StaticRuntime, ClampIntTensor) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, min: float, max: float):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";
  auto a = at::randint(0, 20, {2, 3}, at::kFloat);
  auto b = at::randint(0, 20, {4, 3, 2}, at::kFloat);
  auto min = 5.0f;
  auto max = 5.0f;
  testStaticRuntime(src, {a, min, max});
  testStaticRuntime(src, {a, min, max}, {b, min, max});
}

TEST(StaticRuntime, LenWithTuple) {
  const auto src = R"IR(
    graph(%input : int[]):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {c10::List<int64_t>(4)});
}

TEST(StaticRuntime, LenWithTensor) {
  const auto src = R"IR(
    graph(%input : Tensor):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {at::randn({2, 2, 2})});
}

TEST(StaticRuntime, LenWithStr) {
  const auto src = R"IR(
    graph(%input : str):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {"static_runtime"});
}

TEST(StaticRuntime, LenWithDict_str) {
  const auto script = R"JIT(
    def forward(self, input: Dict[str, str]):
        return len(input)
  )JIT";

  c10::Dict<std::string, std::string> dict;
  dict.insert("abc", "123");
  dict.insert("def", "456");
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_int) {
  const auto script = R"JIT(
    def forward(self, input: Dict[int, int]):
        return len(input)
  )JIT";

  c10::Dict<int64_t, int64_t> dict;
  dict.insert(0, 1);
  dict.insert(2, 3);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_bool) {
  const auto script = R"JIT(
    def forward(self, input: Dict[bool, bool]):
        return len(input)
  )JIT";

  c10::Dict<bool, bool> dict;
  dict.insert(true, false);
  dict.insert(false, true);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_float) {
  const auto script = R"JIT(
    def forward(self, input: Dict[float, float]):
        return len(input)
  )JIT";

  c10::Dict<double, double> dict;
  dict.insert(0.1, 0.9);
  dict.insert(0.8, 0.18);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_complex) {
  const auto script = R"JIT(
    def forward(self, input: Dict[complex, complex]):
        return len(input)
  )JIT";

  c10::Dict<c10::complex<double>, c10::complex<double>> dict;
  dict.insert(0.1, 0.4);
  dict.insert(0.9, 0.45);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_Tensor) {
  const auto script = R"JIT(
    def forward(self, input: Dict[Tensor, Tensor]):
        return len(input)
  )JIT";

  c10::Dict<at::Tensor, at::Tensor> dict;
  dict.insert(at::randn({1, 2}), at::randn({1, 2}));
  dict.insert(at::randn({1, 2}), at::randn({1, 2}));
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, Logit) {
  // no nnc
  const auto logit_script_1 = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.logit(inp).clone()
        return (a)
  )JIT";

  // with nnc
  const auto logit_script_2 = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.logit(inp, 1e-6).clone()
        return (a)
  )JIT";

  // no nnc
  const auto logit_script_3 = R"JIT(
    def forward(self, inp: Tensor, eps: float):
        a = torch.logit(inp, eps).clone()
        return (a)
  )JIT";
  auto a = at::ones({2, 3});
  double b = 1e-6;
  std::vector<IValue> args_1{a};
  std::vector<IValue> args_2({a, b});

  auto c = at::ones({4, 3, 2});

  // logit
  testStaticRuntime(logit_script_1, args_1);
  testStaticRuntime(logit_script_2, args_1);
  testStaticRuntime(logit_script_3, args_2);

  testStaticRuntime(logit_script_1, args_1, {c});
  testStaticRuntime(logit_script_2, args_1, {c});
  testStaticRuntime(logit_script_3, args_2, {c, b});
}

TEST(StaticRuntime, EmbeddingBag) {
  const std::string embedding_bag_default = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_mean = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 1)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_max = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 2)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_sum_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 0, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_mean_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 1, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_max_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 2, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  at::Tensor weight = torch::randn({3, 11}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2, 4});
  std::vector<IValue> args{weight, input, offset};
  testStaticRuntime(embedding_bag_default, args);
  testStaticRuntime(embedding_bag_mean, args);
  testStaticRuntime(embedding_bag_max, args);
  testStaticRuntime(embedding_bag_sum_last_offset, args);
  testStaticRuntime(embedding_bag_mean_last_offset, args);
  testStaticRuntime(embedding_bag_max_last_offset, args);

  at::Tensor weight2 = torch::randn({10, 11}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 1});
  at::Tensor offset2 = torch::tensor({0, 1, 2, 3, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};
  testStaticRuntime(embedding_bag_default, args, args2);
  testStaticRuntime(embedding_bag_mean, args, args2);
  testStaticRuntime(embedding_bag_max, args, args2);
  testStaticRuntime(embedding_bag_sum_last_offset, args, args2);
  testStaticRuntime(embedding_bag_mean_last_offset, args, args2);
  testStaticRuntime(embedding_bag_max_last_offset, args, args2);
}

TEST(StaticRuntime, EmbeddingBagWithManagedOutput) {
  const std::string embedding_bag_managed_output = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        # The outputs of embedding_bag become an intermediate tensors
        # since they are not directly returned from the graph.
        x, y, z, _ = torch.embedding_bag(a, b, c)
        return x + x
  )JIT";

  at::Tensor weight = torch::randn({3, 8}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2});
  std::vector<IValue> args{weight, input, offset};

  at::Tensor weight2 = torch::randn({6, 8}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 3, 4});
  at::Tensor offset2 = torch::tensor({0, 2, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};

  testStaticRuntime(embedding_bag_managed_output, args);
  testStaticRuntime(embedding_bag_managed_output, args, args2);
}

TEST(StaticRuntime, EmbeddingBagWithExtraneousOutput) {
  const std::string embedding_bag_default_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=0]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  auto graph = getGraphFromIR(embedding_bag_default_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_mean_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=1]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  graph = getGraphFromIR(embedding_bag_mean_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_max_last_offset_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=2]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=1]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  graph = getGraphFromIR(embedding_bag_max_last_offset_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_normal_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=0]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res0 : Tensor = aten::clone(%y0, %none)
        %res1 : Tensor = aten::clone(%y1, %none)
        %res2 : Tensor = aten::clone(%y2, %none)
        %res3 : Tensor = aten::clone(%y3, %none)
        return (%res0, %res1, %res2, %res3)
  )IR";
  graph = getGraphFromIR(embedding_bag_normal_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check_not("static_runtime::embedding_bag")
      ->run(*graph);

  at::Tensor weight = torch::randn({3, 11}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2, 4});
  std::vector<IValue> args{weight, input, offset};
  testStaticRuntime(embedding_bag_default_ir, args);
  testStaticRuntime(embedding_bag_mean_ir, args);
  testStaticRuntime(embedding_bag_max_last_offset_ir, args);

  at::Tensor weight2 = torch::randn({10, 11}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 1});
  at::Tensor offset2 = torch::tensor({0, 1, 2, 3, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};
  testStaticRuntime(embedding_bag_default_ir, args, args2);
  testStaticRuntime(embedding_bag_mean_ir, args, args2);
  testStaticRuntime(embedding_bag_max_last_offset_ir, args, args2);
}

TEST(StaticRuntime, EmbeddingBagWithMixedInt32Int64Input) {
  const std::string embedding_bag_default = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";
  auto weight = torch::randn({3, 11}, at::ScalarType::Float);
  auto input = torch::tensor({0, 1, 0, 2}, at::ScalarType::Long);
  auto offset = torch::tensor({0, 2, 4}, at::ScalarType::Int);
  std::vector<IValue> args{weight, input, offset};
  testStaticRuntime(embedding_bag_default, args);
}

TEST(StaticRuntime, LayerNorm) {
  const std::string layer_norm_with_weights = R"JIT(
    def forward(self, input: Tensor, normalized_shape: List[int], weight: Tensor, bias: Tensor):
        return torch.layer_norm(input, normalized_shape, weight, bias, 1e-05, False).clone()
  )JIT";

  const std::string layer_norm_without_weights = R"JIT(
    def forward(self, input: Tensor, normalized_shape: List[int]):
        return torch.layer_norm(input, normalized_shape, None, None, 1e-05, False).clone()
  )JIT";

  const std::string layer_norm_with_noncontiguous_input = R"JIT(
    def forward(self, input: Tensor, normalized_shape: List[int], weight: Tensor, bias: Tensor):
        input = torch.transpose(input, 1, 2)
        return torch.layer_norm(input, normalized_shape, weight, bias, 1e-05, False).clone()
  )JIT";

  const auto a = torch::rand({1, 2, 2, 2});
  const auto b = torch::rand({3, 2, 2, 2});
  for (int normalized_size : {2, 3}) {
    std::vector<int64_t> normalized_shape(normalized_size, 2);
    const auto weight = torch::rand(normalized_shape);
    const auto bias = torch::rand(normalized_shape);

    std::vector<IValue> args{a, normalized_shape, weight, bias};
    std::vector<IValue> args1{b, normalized_shape, weight, bias};
    testStaticRuntime(layer_norm_with_weights, args);
    testStaticRuntime(layer_norm_with_weights, args, args1);
    testStaticRuntime(layer_norm_with_noncontiguous_input, args);

    args = {a, normalized_shape};
    testStaticRuntime(layer_norm_without_weights, args);
    testStaticRuntime(layer_norm_without_weights, args, {b, normalized_shape});
  }
}

TEST(StaticRuntime, Bmm) {
  const auto bmm_script = R"JIT(
    def forward(self, inp: Tensor, mat2: Tensor):
      return torch.bmm(inp, mat2).clone()
  )JIT";

  auto a = at::randn({10, 4, 5});
  auto b = at::randn({10, 5, 6});

  auto c = at::randn({12, 5, 6});
  auto d = at::randn({12, 6, 7});

  std::vector<IValue> args{a, b};
  std::vector<IValue> args1{c, d};
  testStaticRuntime(bmm_script, args);
  testStaticRuntime(bmm_script, args1);
  testStaticRuntime(bmm_script, args, args1);
}

TEST(StaticRuntime, Addmm) {
  const auto addmm_script = R"JIT(
    def forward(self, inp: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float):
      return torch.addmm(inp, mat1, mat2, alpha=alpha, beta=beta).clone()
  )JIT";
  auto inp1 = at::randn({5});
  auto mat1 = at::randn({3, 4});
  auto mat2 = at::randn({4, 5});

  auto inp2 = at::randn({3, 7});
  auto mat3 = at::randn({3, 6});
  auto mat4 = at::randn({6, 7});

  std::vector<IValue> args{inp1, mat1, mat2, 1.0, 2.0};
  std::vector<IValue> args1{inp2, mat3, mat4, 2.0, 1.0};
  testStaticRuntime(addmm_script, args);
  testStaticRuntime(addmm_script, args1);
  testStaticRuntime(addmm_script, args, args1);
}

TEST(StaticRuntime, Abs) {
  const auto abs_script = R"JIT(
    def forward(self, a):
      return a.abs().clone()
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 2, 3});
  std::vector<IValue> args{a};
  std::vector<IValue> args2{b};
  testStaticRuntime(abs_script, args);
  testStaticRuntime(abs_script, args, args2);
}

TEST(StaticRuntime, Binary) {
  const auto add_script = R"JIT(
    def forward(self, a, b):
        c = a + b
        return (c.clone())
  )JIT";

  const auto add_script_ints = R"JIT(
    def forward(self, a: int, b: int):
        c = a + b
        d = c + 1
        return d
  )JIT";

  const auto add_list_script = R"JIT(
    def forward(self, a: List[int], b: List[int]):
        c = a + b
        return c[::]
  )JIT";

  const auto list_construct_script = R"JIT(
    def forward(self, a, b):
      return [a, b]
  )JIT";

  const auto list_construct_script_2 = R"JIT(
    def forward(self, a, b):
      c = a + a
      return [c, c]
  )JIT";

  const auto list_construct_script_3 = R"JIT(
    def forward(self, a, b):
      c = a + a
      return [c, c.flatten()]
  )JIT";

  const auto list_unpack_script = R"JIT(
    def forward(self, a, b):
      c = [a, b]
      x, y = c
      z = x + y
      return z.clone()
  )JIT";

  const auto list_unpack_script_2 = R"JIT(
    def forward(self, a, b):
      c = [a, b]
      x, y = c
      z = (x, y)
      return z
  )JIT";

  const auto tuple_construct_script = R"JIT(
    def forward(self, a, b):
      return (a, b)
  )JIT";

  const auto tuple_construct_script_2 = R"JIT(
    def forward(self, a, b):
      return (a.flatten(), b)
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::ones({2, 3});

  auto c = at::randn({4, 2, 3});
  auto d = at::ones({4, 2, 3});

  std::vector<IValue> args{a, b};

  testStaticRuntime(add_script, args);
  testStaticRuntime(add_script_ints, {1, 2});
  testStaticRuntime(add_script, args, {c, d});
  testStaticRuntime(list_construct_script, args);
  testStaticRuntime(list_construct_script_2, args);
  testStaticRuntime(list_construct_script_3, args);
  testStaticRuntime(list_unpack_script, args);
  testStaticRuntime(list_unpack_script_2, args);
  testStaticRuntime(tuple_construct_script, args);
  testStaticRuntime(tuple_construct_script_2, args);

  std::vector<IValue> list_args{
      c10::List<int64_t>{1, 2, 3}, c10::List<int64_t>{4, 5, 6}};
  testStaticRuntime(add_list_script, list_args);
}

TEST(StaticRuntime, MatMul) {
  const auto aten_matmul = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.matmul(a, b).clone()
  )JIT";

  // 1-D, 1-D
  std::vector<IValue> args{at::randn({3}), at::randn({3})};
  testStaticRuntime(aten_matmul, args);
  // 2-D, 2-D
  std::vector<IValue> args1 = {at::randn({3, 2}), at::randn({2, 3})};
  testStaticRuntime(aten_matmul, args1);
  // 1-D, 2-D
  std::vector<IValue> args2 = {at::randn({3}), at::randn({3, 5})};
  testStaticRuntime(aten_matmul, args2);
  // 2-D, 1-D
  std::vector<IValue> args3 = {at::randn({3, 5}), at::randn({5})};
  testStaticRuntime(aten_matmul, args3);
  // > 2-D , > 2-D
  std::vector<IValue> args4 = {at::randn({3, 1, 4, 5}), at::randn({2, 5, 6})};
  testStaticRuntime(aten_matmul, args4);

  testStaticRuntime(aten_matmul, args3, args4);
}

TEST(StaticRuntime, Sign) {
  const auto sign_tensor = R"JIT(
    def forward(self, input: Tensor):
        return torch.sign(input).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a};
  testStaticRuntime(sign_tensor, args);
  testStaticRuntime(sign_tensor, args, {b});
}

TEST(StaticRuntime, Div) {
  const auto div_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.div(a, b).clone()
  )JIT";

  const auto div_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.div(a, b).clone()
  )JIT";

  const auto div_tensor_mode = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: str):
        return torch.div(a, b, rounding_mode=c).clone()
  )JIT";

  const auto div_scalar_mode = R"JIT(
    def forward(self, a: Tensor, b: float, c: str):
        return torch.div(a, b, rounding_mode=c).clone()
  )JIT";

  const auto div_strided = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        a_strided = torch.transpose(a, 0, 1)
        b_strided = torch.transpose(b, 0, 1)
        return torch.div(a_strided, b_strided).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto bs = at::randn({3, 2}).transpose(0, 1);
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});
  auto ds = at::randn({3, 4, 2}).transpose(0, 1);

  std::vector<IValue> args0{a, b};
  testStaticRuntime(div_tensor, args0);
  testStaticRuntime(div_tensor, args0, {c, d});

  testStaticRuntime(div_strided, args0);
  testStaticRuntime(div_strided, args0, {c, d});

  testStaticRuntime(div_tensor, {a, bs});
  testStaticRuntime(div_tensor, {a, bs}, {c, ds});

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(div_scalar, args1);
  testStaticRuntime(div_scalar, args1, {c, 4});

  std::vector<IValue> args2{a, b, "floor"};
  testStaticRuntime(div_tensor_mode, args2);
  testStaticRuntime(div_tensor_mode, args2, {c, d, "floor"});

  std::vector<IValue> args3{a, 2.3, "trunc"};
  testStaticRuntime(div_scalar_mode, args3);
  testStaticRuntime(div_scalar_mode, args3, {c, 1.5, "trunc"});
}

TEST(StaticRuntime, Mul) {
  const auto mul_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.mul(a, b).clone()
  )JIT";

  const auto mul_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.mul(a, b).clone()
  )JIT";

  const auto mul_list = R"JIT(
    def forward(self, a: List[int], n: int):
        b = a * n
        return b[::]
  )JIT";

  auto a = at::randn({3, 3});
  auto b = at::randn({3, 3});
  auto c = at::randn({3, 3, 3});
  auto d = at::randn({3, 3, 3});

  std::vector<IValue> tensor_args1{a, b};
  std::vector<IValue> tensor_args2{c, d};

  testStaticRuntime(mul_tensor, tensor_args1);
  testStaticRuntime(mul_tensor, tensor_args1, tensor_args2);

  std::vector<IValue> scalar_args1{a, 42};
  std::vector<IValue> scalar_args2{c, 42};

  testStaticRuntime(mul_scalar, scalar_args1);
  testStaticRuntime(mul_scalar, scalar_args1, scalar_args2);

  std::vector<IValue> list_args{c10::List<int64_t>{1, 2}, 3};
  testStaticRuntime(mul_list, list_args);
}

TEST(StaticRuntime, Log) {
  const auto log_tensor = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.log(inp).clone()
        return (a)
  )JIT";

  // Ensure that the input values are valid.
  auto a = at::abs(at::randn({2, 3}));
  auto b = at::abs(at::randn({4, 3, 2}));

  std::vector<IValue> args{a};
  testStaticRuntime(log_tensor, args);
  testStaticRuntime(log_tensor, args, {b});
}

TEST(StaticRuntime, Sub) {
  const auto sub_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.sub(a, b).clone()
  )JIT";

  const auto sub_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.sub(a, b).clone()
  )JIT";

  const auto sub_tensor_alpha = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: float):
        return torch.sub(a, b, alpha=c).clone()
  )JIT";

  const auto sub_scalar_alpha = R"JIT(
    def forward(self, a: Tensor, b: float, c: int):
        return torch.sub(a, b, alpha=c).clone()
  )JIT";

  const auto sub_two_scalars = R"JIT(
    def forward(self, a: int, b: int):
        return (a - b - b)
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});

  std::vector<IValue> args0{a, b};
  testStaticRuntime(sub_tensor, args0);
  testStaticRuntime(sub_tensor, args0, {c, d});

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(sub_scalar, args1);
  testStaticRuntime(sub_scalar, args1, {c, 4});

  std::vector<IValue> args2{a, b, 2.3};
  testStaticRuntime(sub_tensor_alpha, args2);
  testStaticRuntime(sub_tensor_alpha, {c, d, 3.1});

  std::vector<IValue> args3{a, 2.3, 4};
  testStaticRuntime(sub_scalar_alpha, args3);
  testStaticRuntime(sub_scalar_alpha, {c, 1.3, 2});

  std::vector<IValue> args4{1, 2};
  testStaticRuntime(sub_two_scalars, args4);
}

TEST(StaticRuntime, NanToNum) {
  const auto nan_to_num_script = R"JIT(
    def forward(self, a: Tensor, nan: float, posinf: float, neginf: float):
        return torch.nan_to_num(a, nan, posinf, neginf).clone()
  )JIT";

  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();

  auto a = torch::tensor({{1.0, nan}, {-inf, inf}});
  auto b = at::randn({3, 6});
  float* b_data = b.data_ptr<float>();
  b_data[0] = nan;
  b_data[4] = -inf;
  b_data[11] = inf;
  b_data[13] = nan;

  std::vector<IValue> args1{a, 1.0, 2.0, -2.0};
  std::vector<IValue> args2{b, 1.0, 2.0, -2.0};

  testStaticRuntime(
      nan_to_num_script,
      args1,
      /*args2*/ {},
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
  testStaticRuntime(
      nan_to_num_script,
      args1,
      args2,
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
}

TEST(StaticRuntime, Stack) {
  const auto stack_dim = R"JIT(
    def forward(self, a: Tensor, b: Tensor, dim: int):
        inputs = [a]
        inputs.append(b) # mutation to avoid using VarStack
        return torch.stack(inputs, dim = dim).clone()
  )JIT";

  const auto stack_three = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        inputs = [a, b]
        inputs.append(c) # mutation to avoid using VarStack
        return torch.stack(inputs).clone()
  )JIT";

  auto a = at::randn({2, 2});
  auto b = at::randn({2, 2});
  auto c = at::randn({2, 2});

  auto d = at::randn({3, 3, 3});
  auto e = at::randn({3, 3, 3});
  auto f = at::randn({3, 3, 3});

  std::vector<IValue> args1_dim{a, b, 0};
  std::vector<IValue> args2_dim{d, e, 1};
  std::vector<IValue> args_dim_negative{d, e, -1};

  std::vector<IValue> args1_three_tensors{a, b, c};
  std::vector<IValue> args2_three_tensors{d, e, f};

  testStaticRuntime(stack_dim, args1_dim);
  testStaticRuntime(stack_dim, args1_dim, args2_dim);

  testStaticRuntime(stack_dim, args_dim_negative);

  testStaticRuntime(stack_three, args1_three_tensors);
  testStaticRuntime(stack_three, args1_three_tensors, args2_three_tensors);
}

TEST(StaticRuntime, ReLU) {
  const auto relu_script = R"JIT(
    def forward(self, a: Tensor):
        return torch.relu(a).clone()
  )JIT";
  auto a = at::randint(-10, 10, {2, 4});
  auto b = at::randint(-10, 10, {3, 6});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(relu_script, args1);
  testStaticRuntime(relu_script, args1, args2);
}

TEST(StaticRuntime, Tanh) {
  const auto tanh_script = R"JIT(
    def forward(self, a):
        return torch.tanh(a).clone()
  )JIT";
  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(tanh_script, args1, /*args2*/ {}, /*use_allclose*/ true);
  testStaticRuntime(tanh_script, args1, args2, /*use_allclose*/ true);
}

TEST(StaticRuntime, Norm) {
  const auto norm_2arg = R"JIT(
    def forward(self, a: Tensor, p: int):
        return torch.norm(a, p).clone()
  )JIT";

  const auto norm_3arg = R"JIT(
    def forward(self, a: Tensor, p: int, dtype: int):
        return torch.norm(a, p, dtype=dtype).clone()
  )JIT";

  const auto norm_4arg = R"JIT(
    def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool):
        return torch.norm(a, p, dim, keepdim).clone()
  )JIT";

  const auto norm_5arg = R"JIT(
    def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool, dtype: int):
        return torch.norm(a, p, dim, keepdim, dtype=dtype).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 5});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args2{a, 2};
  testStaticRuntime(norm_2arg, args2);
  testStaticRuntime(norm_2arg, args2, {b, 2}, false, false, false);

  std::vector<IValue> args3{a, 2, dtype};
  testStaticRuntime(norm_3arg, args3);
  testStaticRuntime(norm_3arg, args3, {b, 2, dtype}, false, false, false);

  std::vector<IValue> args4{a, 3, dim, false};
  testStaticRuntime(norm_4arg, args4);
  testStaticRuntime(norm_4arg, args4, {b, 3, dim, false});

  std::vector<IValue> args5{a, 4, dim, true, dtype};
  testStaticRuntime(norm_5arg, args5);
  testStaticRuntime(norm_5arg, args5, {b, 4, dim, true, dtype});
}

TEST(StaticRuntime, Reshape) {
  const auto reshape_script_1 = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.reshape(shape)
        return b + b
  )JIT";

  const auto reshape_script_2 = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.transpose(0, 1)
        return b.reshape(shape)
  )JIT";

  const auto reshape_script_3 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = a.reshape(shape)
        d = c + c
        e = d + d
        f = e * e
        g = f * f
        return b.reshape(shape), g
  )JIT";

  // exercise reshape_copy and flatten_copy
  const auto reshape_script_4 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        k = inp + inp
        a = k + k
        b = a.reshape(shape)
        c = a.flatten().reshape(shape)
        return b + c
  )JIT";

  // exercise reshape_copy
  const auto reshape_script_5 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = a.reshape(shape).relu()
        d = c + c
        e = d + d
        f = e * e
        g = f * f
        return g
  )JIT";

  const auto reshape_inplace_script = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = b.sigmoid_()
        d = c + c
        e = a + a
        f = b + b
        return (d, e, f)
  )JIT";

  // b is in_contiguous
  const auto reshape_incontiguous_script = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.transpose(0, 1)
        c = b.reshape(shape)
        c = c.relu()
        return (c)
  )JIT";

  auto a = at::randn({2, 3});
  auto b = std::vector<int64_t>({3, 2});
  std::vector<IValue> args{a, b};

  auto c = at::randn({4, 5});
  auto d = std::vector<int64_t>({5, 1, 2, 2});
  std::vector<IValue> args1{c, d};

  testStaticRuntime(reshape_script_1, args);
  testStaticRuntime(reshape_script_2, args);
  testStaticRuntime(reshape_script_3, args);
  testStaticRuntime(reshape_script_4, args);
  testStaticRuntime(reshape_script_5, args);
  testStaticRuntime(reshape_inplace_script, args);
  testStaticRuntime(reshape_incontiguous_script, args);

  testStaticRuntime(reshape_script_1, args, args1);
  testStaticRuntime(reshape_script_2, args, args1);
  testStaticRuntime(reshape_script_3, args, args1);
  testStaticRuntime(reshape_script_4, args, args1);
  testStaticRuntime(reshape_script_5, args, args1);
  testStaticRuntime(reshape_inplace_script, args, args1);
  testStaticRuntime(reshape_incontiguous_script, args, args1);
}

TEST(StaticRuntime, Repeat) {
  const std::string repeat = R"JIT(
    def forward(self, a: Tensor, repeats: List[int]):
        return torch.repeat(a, repeats).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3});
  auto c = std::vector<int64_t>({1, 2});
  auto d = std::vector<int64_t>({2, 3});
  std::vector<IValue> args1{a, c};
  std::vector<IValue> args2{b, d};

  testStaticRuntime(repeat, args1);
  testStaticRuntime(repeat, args2);
  testStaticRuntime(repeat, args1, args2);
}

TEST(StaticRuntime, Flatten) {
  // exercise flatten_copy
  const auto flatten_script_1 = R"JIT(
    def forward(self, a: Tensor, start_dim: int, end_dim: int):
        b = a * a
        c = torch.flatten(b, start_dim, end_dim)
        d = torch.relu(c)
        return d
  )JIT";

  const auto flatten_script_2 = R"JIT(
    def forward(self, a: Tensor, start_dim: int, end_dim: int):
        b = a.transpose(0, 1)
        return torch.flatten(b, start_dim, end_dim).clone()
  )JIT";

  auto test_flatten =
      [&](std::vector<int64_t> shape, int64_t start_dim, int64_t end_dim) {
        std::vector<int64_t> shape1(shape);
        if (shape1.size() > 0) {
          shape1[0] *= 6;
        }
        auto a = at::randn(shape);
        auto b = at::randn(shape1);
        std::vector<IValue> args{a, start_dim, end_dim};
        bool check_resize = shape1.size() > 0;
        testStaticRuntime(flatten_script_1, args);
        testStaticRuntime(
            flatten_script_1,
            args,
            {b, start_dim, end_dim},
            false, /* use_allclose */
            false, /* use_equalnan */
            check_resize);
        if (shape.size() > 2) {
          testStaticRuntime(flatten_script_2, args);
          testStaticRuntime(flatten_script_2, args, {b, start_dim, end_dim});
        }
      };

  test_flatten({2, 3}, 0, 1);
  test_flatten({2, 1, 3}, 1, 2);
  test_flatten({0, 1, 3, 0}, 1, 2);
  test_flatten({2, 3}, 1, 1);
  test_flatten({}, 0, 0);
}

TEST(StaticRuntime, pow) {
  const auto pow_script_ten_sca = R"JIT(
    def forward(self, input : Tensor, exponent : int):
        return torch.pow(input, exponent).clone()
  )JIT";

  const auto pow_script_ten_ten = R"JIT(
    def forward(self, input : Tensor, exponent : Tensor):
        return torch.pow(input, exponent).clone()
  )JIT";

  const auto pow_script_sca_ten = R"JIT(
    def forward(self, input : int, exponent : Tensor):
        return torch.pow(input, exponent).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});

  std::vector<IValue> args0{a, 4};
  testStaticRuntime(pow_script_ten_sca, args0);
  testStaticRuntime(pow_script_ten_sca, args0, {c, 4});

  std::vector<IValue> args1{at::abs(a), b};
  testStaticRuntime(pow_script_ten_ten, args1);
  testStaticRuntime(pow_script_ten_ten, args1, {at::abs(c), d});

  std::vector<IValue> args2{5, b};
  testStaticRuntime(pow_script_sca_ten, args2);
  testStaticRuntime(pow_script_sca_ten, args2, {3, d});
}

TEST(StaticRuntime, to) {
  const auto to_script_dtype = R"JIT(
    def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
        a = input + input
        return torch.to(a, dtype, non_blocking, copy, memory_format).clone()
  )JIT";

  const auto to_script_dtype_strided = R"JIT(
    def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
        b = input.permute(0, 2, 3, 1)
        return torch.to(b, dtype, non_blocking, copy, memory_format).clone()
  )JIT";

  const auto to_script_prim_dtype = R"JIT(
    def forward(self, input:Tensor, dtype: Optional[int], non_blocking: bool, copy: bool):
        a = input + input
        return torch.to(a, dtype, non_blocking, copy).clone()
  )JIT";

  const auto to_script_other = R"JIT(
    def forward(self, input:Tensor, other: Tensor, non_blocking: bool, copy: bool, memory_format: int):
        a = input + input
        return torch.to(a, other, non_blocking, copy, memory_format).clone()
  )JIT";

  // if input is float tensor, b could be alias of a
  const auto to_script_alias = R"JIT(
    def forward(self, input:Tensor):
        a = input + input
        b = a.float()
        c = b * b
        return (c)
  )JIT";

  const auto to_script_fails_managed_output_check = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float()
        return e
  )JIT";

  const auto to_script_select_tensor_output_into_tuple = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float()
        return (d, e)
  )JIT";

  const auto to_script_memory_planning_fail = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float().relu()
        return e
  )JIT";

  auto test_to = [&](at::ScalarType b, bool c, bool d, c10::MemoryFormat e) {
    auto a = at::randn({4, 3, 1, 2});
    auto other = at::randn({4, 3, 1, 2}).to(b);
    auto a2 = at::randn({3, 2, 2, 4});
    auto a2_other = at::randn({3, 2, 2, 4}).to(b);

    std::vector<IValue> args0{a, b, c, d, e};
    std::vector<IValue> args1{a, b, c, d};
    std::vector<IValue> args2{a, other, c, d, e};
    std::vector<IValue> args2WithDifferentOtherType{
        a, at::randn({4, 3, 1, 2}, ScalarType::Double), c, d, e};
    std::vector<IValue> args3{a, std::nullopt, c, d};

    std::vector<IValue> args0WithInt{a, ScalarType::Int, c, d, e};
    testStaticRuntime(
        to_script_dtype,
        args0,
        args0WithInt,
        /* default for use_allclose */ false,
        /* default for use_equalnan */ false,
        /* check_resize */ false);
    testStaticRuntime(to_script_dtype_strided, args0);
    testStaticRuntime(to_script_prim_dtype, args1);
    if (!d) {
      testStaticRuntime(to_script_prim_dtype, args3);
    }
    // Second set of args tests case where the `other` tensor's dtype
    // changes between iterations.
    testStaticRuntime(
        to_script_other,
        args2,
        args2WithDifferentOtherType,
        /* default for use_allclose */ false,
        /* default for use_equalnan */ false,
        /* check_resize */ false);
    testStaticRuntime(to_script_alias, {a});

    testStaticRuntime(to_script_memory_planning_fail, {a, a});
    testStaticRuntime(to_script_fails_managed_output_check, {a, a});
    testStaticRuntime(to_script_select_tensor_output_into_tuple, {a, a});

    // dynamic shapes
    testStaticRuntime(to_script_dtype, args0, {a2, b, c, d, e});
    testStaticRuntime(to_script_dtype_strided, args0, {a2, b, c, d, e});
    testStaticRuntime(to_script_prim_dtype, args1, {a2, b, c, d});
    if (!d) {
      testStaticRuntime(to_script_prim_dtype, args3, {a2, std::nullopt, c, d});
    }
    testStaticRuntime(to_script_other, args2, {a2, a2_other, c, d, e});
    testStaticRuntime(to_script_alias, {a}, {a2});
  };
  for (const bool non_blocking : {false, true}) {
    for (const bool copy : {false, true}) {
      // float->float, NCHW->NHWC
      test_to(
          at::ScalarType::Float,
          non_blocking,
          copy,
          c10::MemoryFormat::ChannelsLast);
      // float->half
      test_to(
          at::ScalarType::Half,
          non_blocking,
          copy,
          c10::MemoryFormat::Preserve);
      // float->float
      test_to(
          at::ScalarType::Float,
          non_blocking,
          copy,
          c10::MemoryFormat::Contiguous);
      test_to(
          at::ScalarType::Bool,
          non_blocking,
          copy,
          c10::MemoryFormat::Contiguous);
      // TODO: check if fbgemm is enabled properly in this case
      // half->float, NCHW->NHWC
      test_to(
          at::ScalarType::Half,
          non_blocking,
          copy,
          c10::MemoryFormat::ChannelsLast);
    }
  }
}

TEST(StaticRuntime, ExpandAs) {
  const auto expand_as_script = R"JIT(
    def forward(self, input: Tensor, other:Tensor):
        a = input.expand_as(other)
        return a.clone()
  )JIT";

  auto a = at::randn({3, 1});
  auto b = at::randn({3, 2});
  auto c = at::randn({4, 1});
  auto d = at::randn({4, 2});
  std::vector<IValue> args{a, b};
  std::vector<IValue> args2{c, d};
  testStaticRuntime(expand_as_script, args);
  testStaticRuntime(expand_as_script, args, args2);
}

TEST(StaticRuntime, Full) {
  const auto full_script = R"JIT(
    def forward(self,
                size: List[int],
                fill_value: int,
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool]):
        a = torch.full(size
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 270 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `torch`, `TEST`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/static_runtime`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/dispatch/OperatorOptions.h`
- `c10/core/ScalarType.h`
- `gtest/gtest.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/runtime/static/ProcessedNodeInputs.h`
- `torch/csrc/jit/runtime/static/impl.h`
- `torch/csrc/jit/runtime/static/passes.h`
- `torch/csrc/jit/testing/file_check.h`
- `stdexcept`
- `deep_wide_pt.h`
- `test_utils.h`


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python benchmarks/static_runtime/test_static_runtime.cc
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/static_runtime`):

- [`test_cpu_fusion.cc_docs.md`](./test_cpu_fusion.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_utils.cc_docs.md`](./test_utils.cc_docs.md)
- [`test_generated_ops.cc_docs.md`](./test_generated_ops.cc_docs.md)
- [`test_utils.h_docs.md`](./test_utils.h_docs.md)
- [`deep_wide_pt.cc_docs.md`](./deep_wide_pt.cc_docs.md)
- [`deep_wide_pt.h_docs.md`](./deep_wide_pt.h_docs.md)
- [`test_static_module.cc_docs.md`](./test_static_module.cc_docs.md)
- [`deep_wide_pt_bench.cc_docs.md`](./deep_wide_pt_bench.cc_docs.md)


## Cross-References

- **File Documentation**: `test_static_runtime.cc_docs.md`
- **Keyword Index**: `test_static_runtime.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
