# Documentation: `test/cpp/jit/test_misc.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_misc.cpp`
- **Size**: 101,146 bytes (98.78 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <c10/macros/Macros.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include <torch/script.h>

#include <onnx/onnx_pb.h>

#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& list) {
  size_t i = 0;
  out << "{";
  for (auto&& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}

TEST(InternedStringsTest, Basic) {
  ASSERT_EQ(prim::Param, Symbol::prim("Param"));
  ASSERT_EQ(prim::Return, Symbol::prim("Return"));
  ASSERT_EQ(prim::Return.toUnqualString(), std::string("Return"));
  ASSERT_EQ(prim::Return.toQualString(), std::string("prim::Return"));
  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  size_t symstart = newsym;
  ASSERT_EQ(newsym.toQualString(), std::string("aten::__NEW_SYMBOL"));
  // TODO: This test is a bit too close to the implementation details.
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol(symstart + 2).toUnqualString(), std::string("What2"));
}

TEST(FromQualStringTest, Basic) {
  ASSERT_EQ(Symbol::fromQualString("prim::Param"), Symbol::prim("Param"));
  ASSERT_EQ(Symbol::fromQualString("aten::mm"), Symbol::aten("mm"));
  ASSERT_EQ(Symbol::fromQualString("onnx::LSTM"), Symbol::onnx("LSTM"));
  ASSERT_EQ(Symbol::fromQualString("attr::value"), Symbol::attr("value"));
  ASSERT_EQ(Symbol::fromQualString("scope::"), Symbol::scope(""));
  ASSERT_EQ(Symbol::fromQualString("::").toUnqualString(), std::string(""));
  ASSERT_EQ(
      Symbol::fromQualString("::").ns().toQualString(),
      std::string("namespaces::"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").toUnqualString(),
      std::string("param"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns().toUnqualString(),
      std::string("new_ns"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns(),
      Symbol::fromQualString("namespaces::new_ns"));

  auto bad_inputs = {"scope", ":", ""};
  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      ASSERT_TRUE(0);
    } catch (const std::exception& c) {
    }
  }
}

TEST(THNNConvTest, Basic) {
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  std::vector<int64_t> kernel_size = {3, 5};
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {2, 1};
  constexpr int out_channels = 5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn(
      {out_channels, input_size[1], kernel_size[0], kernel_size[1]});
  at::Tensor bias = torch::randn({out_channels});

  // run forward eagerly
  at::Tensor output = at::_slow_conv2d_forward(
      input, weight, kernel_size, bias, stride, padding);

  // make grad_outputs
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);

  // run backward eagerly
  auto [grad_input, grad_weight, grad_bias] = at::_slow_conv2d_backward(
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto ksz_val = graph->insertConstant(kernel_size);
  auto kst_val = graph->insertConstant(stride);
  auto pad_val = graph->insertConstant(padding);

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");

  Value* conv = graph->insert(
      aten::_slow_conv2d_forward,
      {inputg, weightg, ksz_val, biasg, kst_val, pad_val});
  auto outputs = conv->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);

  // Get outputs from the interpreter
  auto [tensors_out, tensor_grads_out] =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

TEST(ATenNativeBatchNormTest, Basic) {
  // aten::native_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor
  // running_mean, Tensor running_var, bool training, float momentum, float eps)
  // -> (Tensor, Tensor, Tensor)
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  bool training = true;
  float momentum = 0.9;
  float eps = 1e-5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn({input_size[1]});
  at::Tensor bias = torch::randn({input_size[1]});
  at::Tensor running_mean = torch::randn({input_size[1]});
  at::Tensor running_var = torch::randn({input_size[1]});

  // running_mean and running_var are changed in-place, so clone and send them
  at::Tensor running_mean_eager = running_mean.clone();
  at::Tensor running_var_eager = running_var.clone();
  at::Tensor running_mean_jit = running_mean.clone();
  at::Tensor running_var_jit = running_var.clone();

  // run forward eagerly
  auto [output, savemean, saveinvstd] = at::native_batch_norm(
      input,
      weight,
      bias,
      running_mean_eager,
      running_var_eager,
      training,
      momentum,
      eps);

  // make grad_outputs
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);
  at::Tensor grad_savemean =
      torch::zeros_like(savemean, at::MemoryFormat::Preserve);
  at::Tensor grad_saveinvstd =
      torch::zeros_like(saveinvstd, at::MemoryFormat::Preserve);

  // run backward eagerly
  // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor
  // weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor
  // save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor,
  // Tensor, Tensor)
  auto [grad_input, grad_weight, grad_bias] = at::native_batch_norm_backward(
      grad_output,
      input,
      weight,
      running_mean_eager,
      running_var_eager,
      savemean,
      saveinvstd,
      training,
      eps,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto training_val = graph->insertConstant(IValue(training));
  auto momentum_val = graph->insertConstant(IValue(momentum));
  auto eps_val = graph->insertConstant(IValue(eps));

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");
  auto running_meang = graph->addInput("running_mean");
  auto running_varg = graph->addInput("running_var");

  Value* bn = graph->insert(
      aten::native_batch_norm,
      {inputg,
       weightg,
       biasg,
       running_meang,
       running_varg,
       training_val,
       momentum_val,
       eps_val});
  auto outputs = bn->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);
  tensors_in.push_back(running_mean_jit);
  tensors_in.push_back(running_var_jit);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);
  tensor_grads_in.push_back(grad_savemean);
  tensor_grads_in.push_back(grad_saveinvstd);

  // Get outputs from the interpreter
  auto [tensors_out, tensor_grads_out] =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensors_out.push_back(savemean);
  expected_tensors_out.push_back(saveinvstd);
  expected_tensors_out.push_back(running_mean_eager);
  expected_tensors_out.push_back(running_var_eager);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  tensors_out.push_back(running_mean_jit);
  tensors_out.push_back(running_var_jit);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

TEST(CustomFusionTest, Basic) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  auto graph_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  torch::jit::overrideCanFuseOnCPU(true);
  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() != prim::Param; },
      Symbol::fromQualString("prim::FusionGroup"));
  torch::jit::overrideCanFuseOnCPU(false);

  const auto& nodes = g->nodes();
  auto fusion_group =
      std::find_if(nodes.begin(), nodes.end(), [](const Node* node) {
        return node->kind() == Symbol::fromQualString("prim::FusionGroup");
      });
  AT_ASSERT(fusion_group != nodes.end());

  auto subgraph = fusion_group->g(attr::Subgraph);
  auto hits = 0;
  // two multiplications
  for (const auto& n : subgraph->nodes()) {
    (void)n;
    hits++;
  }
  AT_ASSERT(hits == 2);
}

TEST(CustomFusionTest, NestedBlocks) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  auto graph_string = R"IR(
  graph(%0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(2, 3, 4)):
    %3 : int = prim::Constant[value=1]()
    %4 : Tensor = prim::If(%2)
      block0():
        %5 : Tensor = aten::mul(%0, %2)
        %6 : Tensor = aten::mul(%5, %1)
        -> (%6)
      block1():
        %7 : Tensor = aten::add(%0, %2, %3)
        %8 : Tensor = aten::add(%7, %1, %3)
        -> (%8)
    %9 : Tensor = aten::add(%4, %2, %3)
    return (%4))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() == aten::mul; },
      Symbol::fromQualString("prim::FusionGroup"));

  // Could be done in more efficient ways, but this is only a test.
  std::function<bool(const Block*, Symbol)> dfs = [&](const Block* b,
                                                      Symbol s) {
    for (auto node : b->nodes()) {
      if (node->kind() == s)
        return true;
      for (auto nested_b : node->blocks())
        if (dfs(nested_b, s))
          return true;
    }
    return false;
  };

  AT_ASSERT(dfs(g->block(), Symbol::fromQualString("prim::FusionGroup")));
}

static const auto cf_examples = R"JIT(
  def if_test(a, b):
      # FIXME: use 0 instead of a.
      # c = 0
      c = a
      if bool(a < b):
        c = b
      else:
        c = a
      return c
  def if_one(a, b):
    c = b
    if bool(a < b):
      c = a
    return c
  def while_test(a, i):
    while bool(i < 3):
      a *= a
      i += 1
    return a
)JIT";

TEST(ControlFlowTest, Basic) {
  auto cu = compile(cf_examples);

  auto run = [&](const std::string& name, std::vector<IValue> stack) {
    auto graph = toGraphFunction(cu->get_function(name)).graph();
    Code code(graph, "");
    InterpreterState interp(code);
    interp.run(stack);
    return stack;
  };

  auto L = [](int64_t l) { return IValue(scalar_to_tensor(at::Scalar(l))); };
  auto V = [](IValue t) { return std::move(t).toTensor().item<int64_t>(); };
  auto run_binary = [&](const std::string& name, int64_t a, int64_t b) {
    return V(run(name, {L(a), L(b)})[0]);
  };
  ASSERT_EQ(2, run_binary("if_test", 1, 2));
  ASSERT_EQ(3, run_binary("if_test", 3, 2));
  ASSERT_EQ(2, run_binary("if_one", 2, 3));
  ASSERT_EQ(2, run_binary("if_one", 3, 2));
  ASSERT_EQ(256, run_binary("while_test", 2, 0));
}

#if !(C10_ASAN_ENABLED || C10_UBSAN_ENABLED)
// This test fails vptr UBSAN checks

TEST(ProtoTest, Basic) {
  ::ONNX_NAMESPACE::ModelProto proto;
  proto.set_producer_name("foo");
}
#endif

// test a few features that are not directly used in schemas yet
TEST(SchemaParserTest, NestedArrays) {
  // nested arrays
  auto s = parseSchema("at::what(int[][4] foo) -> ()");
  ASSERT_TRUE(s.arguments().at(0).N() == 4);
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
  auto s2 = parseSchema("at::what(int[][] foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s2.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
}

TEST(SchemaParserTest, OutVariant) {
  auto schema_with_out = parseSchema(
      "at::foo(Tensor self, *, Tensor(a!) f, Tensor(b!) l) -> (Tensor(a!) f, Tensor(b!) l)");
  ASSERT_TRUE(schema_with_out.arguments().at(1).is_out());
  ASSERT_TRUE(schema_with_out.arguments().at(2).is_out());

  auto schema_without_out =
      parseSchema("at::foo(Tensor self, *, int scalar) -> (int)");

  for (const auto& arg : schema_without_out.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }

  auto schema_with_is_write = parseSchema(
      "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))");

  for (const auto& arg : schema_with_is_write.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SchemaParserTest, NamedReturns) {
  // named returns
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  auto s3 =
      parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_TRUE(s3.returns().at(0).name() == "the_return");
  ASSERT_TRUE(s3.returns().at(1).name() == "the_return2");
}

TEST(SchemaParserTest, Futures) {
  // futures
  auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(
      *s4.arguments().at(0).type()->expectRef<FutureType>().getElementType()));
}

TEST(SchemaParserTest, AnnotatedAliasSets) {
  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");
}

TEST(SchemaParserTest, TensorListAnnotatedAliasSets) {
  const auto s = parseSchema(
      "at::foo(Tensor(a!) self, Tensor(b!)[] out)"
      " -> ()");
  const AliasInfo* selfAliasInfo = s.arguments().at(0).alias_info();
  const AliasInfo* outAliasInfo = s.arguments().at(1).alias_info();
  ASSERT_TRUE(
      selfAliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(selfAliasInfo->isWrite());

  ASSERT_TRUE(outAliasInfo->isWrite());
  ASSERT_TRUE(outAliasInfo->beforeSets().empty());
  ASSERT_EQ(outAliasInfo->containedTypes().size(), 1);

  auto containedType = outAliasInfo->containedTypes()[0];

  ASSERT_TRUE(containedType.isWrite());
  ASSERT_TRUE(
      containedType.beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::b")});
}

TEST(SchemaParserTest, AnnotatedAliasWithoutBeforeSet) {
  EXPECT_THAT(
      []() { parseSchema("at::foo(Tensor(!) self) -> Tensor"); },
      ::testing::Throws<std::runtime_error>(::testing::Property(
          &std::runtime_error::what,
          ::testing::HasSubstr("expected ident but found '!' here"))));
}

TEST(SchemaParserTest, BeforeAfterSets) {
  const auto s = parseSchema(
      "at::what(Tensor(b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // The list itself is annotated with `a`
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  ASSERT_NE(aliasInfo, nullptr);
  ASSERT_TRUE(
      aliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(aliasInfo->isWrite());

  // Check the contained types
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  const auto expected = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
      Symbol::fromQualString("alias::c"),
  };
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expected);
  ASSERT_TRUE(containedAliasInfo.afterSets() == expected);
  ASSERT_FALSE(containedAliasInfo.isWrite());
}

TEST(SchemaParserTest, BeforeAfterSets2) {
  const auto s = parseSchema(
      "at::what(Tensor(b -> b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // The list itself is annotated with `a`
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  ASSERT_NE(aliasInfo, nullptr);
  ASSERT_EQ(
      aliasInfo->beforeSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_EQ(
      aliasInfo->afterSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(aliasInfo->isWrite());
  ASSERT_EQ(aliasInfo->containedTypes().size(), 1);

  // Check the contained types
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  const auto expectedBefore = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
  };
  const auto expectedAfter = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"), Symbol::fromQualString("alias::c")};
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expectedBefore);
  ASSERT_TRUE(containedAliasInfo.afterSets() == expectedAfter);
  ASSERT_FALSE(containedAliasInfo.isWrite());
}

TEST(TopologicalIndexTest, Basic) {
  Graph graph;
  auto node1 = graph.create(prim::AutogradZero);
  auto node2 = graph.create(prim::AutogradZero);
  auto node3 = graph.create(prim::AutogradZero);
  auto node4 = graph.create(prim::AutogradZero);

  graph.appendNode(node4);
  graph.prependNode(node1);
  node2->insertAfter(node1);
  node3->insertBefore(node4);

  // nodes should be in numerical order
  ASSERT_TRUE(node1->isBefore(node2));
  ASSERT_TRUE(node1->isBefore(node3));
  ASSERT_TRUE(node1->isBefore(node4));
  ASSERT_TRUE(node2->isAfter(node1));
  ASSERT_TRUE(node2->isBefore(node3));
  ASSERT_TRUE(node2->isBefore(node4));
  ASSERT_FALSE(node3->isBefore(node1));
  ASSERT_FALSE(node3->isBefore(node2));
  ASSERT_FALSE(node3->isAfter(node4));

  // Built up a block structure
  //  node3
  //   /\        ...
  //  A  B     block1
  //      \      ...
  //      C    block2
  auto block1 = node3->addBlock();
  auto A = graph.create(prim::AutogradZero);
  block1->appendNode(A);
  auto B = graph.create(prim::AutogradZero);
  block1->appendNode(B);
  auto block2 = B->addBlock();
  auto C = graph.create(prim::AutogradZero);
  block2->appendNode(C);

  // Check isAfter on different block levels
  ASSERT_TRUE(node1->isBefore(A));
  ASSERT_TRUE(A->isBefore(B));
  ASSERT_TRUE(A->isBefore(C));

  // make sure things don't blow up on deletions
  node2->destroy();
  auto node2p = graph.create(prim::AutogradZero);
  node2p->insertAfter(node1);
  ASSERT_TRUE(node1->isBefore(node2p));
  ASSERT_TRUE(node2p->isBefore(node3));
}

TEST(TopologicalIndexTest, Reindex) {
  // Induce reindexing to test that path
  Graph graph;
  std::map<size_t, Node*> nodes;

  auto anchor = graph.create(prim::AutogradZero);
  graph.appendNode(anchor);
  // Inserting to the same place a lot will trigger reindexing
  for (auto i = 0; i < 100; ++i) {
    auto n = graph.create(prim::AutogradZero);
    n->insertAfter(anchor);
    nodes[i] = n;
  }

  // Nodes should be in reverse order
  for (auto i = 0; i < 100; ++i) {
    for (auto j = i + 1; j < 100; ++j) {
      ASSERT_TRUE(nodes[i]->isAfter(nodes[j]));
    }
  }
}

at::Tensor invokeTestRecordFunction(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto t2 = t.pow(2);
  return t2;
}

static const auto invokeTestRecordFunction_JIT = R"JIT(
  def foo(self, t):
    t2 = t.pow(2)
    return t2

  def forward(self, t):
    return self.foo(t)
)JIT";

at::Tensor invokeTestRecordFunctionJIT(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto module = std::make_shared<script::Module>(
      "RecordFunctionTestModule", std::make_shared<script::CompilationUnit>());
  module->define(invokeTestRecordFunction_JIT);
  return module->forward({t}).toTensor();
}

using TracedTestValues =
    std::vector<std::tuple<std::string, std::vector<std::vector<int64_t>>>>;

void checkTracedInputs(const TracedTestValues& inputs) {
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  for (const auto& input : inputs) {
    const auto& fn = std::get<0>(input);
    const auto& sizes = std::get<1>(input);

    if (fn == "test") {
      found_test = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 2);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
      TORCH_CHECK(sizes[1].empty());
    } else if (fn == "aten::mul") {
      found_mul = true;
      TORCH_CHECK(sizes.size() > 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

void checkTracedOutputs(const TracedTestValues& outputs) {
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  for (const auto& output : outputs) {
    const auto& fn = std::get<0>(output);
    const auto& sizes = std::get<1>(output);

    if (fn == "test") {
      found_test = true;
      TORCH_CHECK(sizes.empty());
    } else if (fn == "aten::pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::mul") {
      found_mul = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

static bool bad_scope = false;
template <RecordScope scope, size_t* cnt>
std::unique_ptr<at::ObserverContext> checkScopeCallback(
    const at::RecordFunction& fn) {
  if (fn.scope() == scope) {
    ++(*cnt);
  } else {
    bad_scope = true;
  }
  return nullptr;
}

template <RecordScope scope, size_t* cnt>
void pushScopedCallback() {
  at::addGlobalCallback(
      at::RecordFunctionCallback(checkScopeCallback<scope, cnt>)
          .scopes({scope}));
}

// These cannot be function-local because that would prohibit them
// from being used as template arguments prior to C++17.
static size_t fun_cnt;
static size_t ts_fun_cnt;
static size_t user_scope_cnt;

void checkScopeCallbacks() {
  static bool found_function_scope;
  static bool found_method_scope;
  static bool found_user_scope;
  found_function_scope = false;
  found_method_scope = false;
  found_user_scope = false;
  at::addGlobalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        if (fn.scope() == at::RecordScope::FUNCTION &&
            std::string(fn.name()) == "test_function") {
          found_function_scope = true;
        }
        if (fn.scope() == at::RecordScope::TORCHSCRIPT_FUNCTION &&
            std::string(fn.name()) == "test_method") {
          found_method_scope = true;
        }
        if (fn.scope() == at::RecordScope::USER_SCOPE &&
            std::string(fn.name()) == "test_user_scope") {
          found_user_scope = true;
        }
        return nullptr;
      }));

  bad_scope = false;
  fun_cnt = 0;
  pushScopedCallback<at::RecordScope::FUNCTION, &fun_cnt>();
  ts_fun_cnt = 0;
  pushScopedCallback<at::RecordScope::TORCHSCRIPT_FUNCTION, &ts_fun_cnt>();
  user_scope_cnt = 0;
  pushScopedCallback<at::RecordScope::USER_SCOPE, &user_scope_cnt>();

  TORCH_CHECK(at::hasCallbacks());

  {
    RECORD_TORCHSCRIPT_FUNCTION("test_method", {});
    {
      RECORD_FUNCTION("test_function", {});
    }
    {
      RECORD_USER_SCOPE("test_user_scope");
    }
  }

  TORCH_CHECK(!bad_scope);
  TORCH_CHECK(fun_cnt == 1);
  TORCH_CHECK(ts_fun_cnt == 1);
  TORCH_CHECK(user_scope_cnt == 1);

  TORCH_CHECK(found_function_scope);
  TORCH_CHECK(found_method_scope);
  TORCH_CHECK(found_user_scope);
}

static TracedTestValues traced_inputs;
static TracedTestValues traced_outputs;
static std::unordered_set<std::string> ts_input_names;
static std::unordered_set<std::string> ts_output_names;

std::unique_ptr<at::ObserverContext> tracedInputsCallback(
    const RecordFunction& fn) {
  if (fn.scope() == RecordScope::FUNCTION) {
    auto inputs = fn.inputs();
    std::vector<std::vector<int64_t>> sizes;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        sizes.push_back(input.toTensor().sizes().vec());
      } else if (input.isScalar()) {
        // NOLINTNEXTLINE(modernize-use-emplace)
        sizes.push_back(std::vector<int64_t>());
      }
    }
    traced_inputs.push_back(std::make_tuple(fn.name(), sizes));
  } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
    ts_input_names.insert(fn.name());
  }
  return nullptr;
}

void tracedOutputsCallback(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  if (fn.scope() == RecordScope::FUNCTION) {
    auto outputs = fn.outputs();
    std::vector<std::vector<int64_t>> sizes;
    for (const auto& output : outputs) {
      if (output.isTensor()) {
        sizes.push_back(output.toTensor().sizes().vec());
      } else if (output.isScalar()) {
        sizes.emplace_back();
      }
    }
    traced_outputs.push_back(std::make_tuple(fn.name(), sizes));
  } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
    ts_output_names.insert(fn.name());
  }
}

TEST(RecordFunctionTest, TracedTestInputsOutputs) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  // [(fn, [[sizes], [sizes], ...]), ...]
  addGlobalCallback(
      RecordFunctionCallback(tracedInputsCallback, tracedOutputsCallback)
          .needsInputs(true)
          .needsOutputs(true));

  TracedTestValues eager_inputs, eager_outputs, jit_inputs, jit_outputs;
  {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = invokeTestRecordFunction(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    eager_inputs = traced_inputs;
    eager_outputs = traced_outputs;
    traced_inputs.clear();
    traced_outputs.clear();

    TORCH_CHECK(ts_input_names.empty());
    TORCH_CHECK(ts_output_names.empty());

    t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    t2 = invokeTestRecordFunctionJIT(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    jit_inputs = traced_inputs;
    jit_outputs = traced_outputs;
    traced_inputs.clear();
    traced_outputs.clear();
  }

  TORCH_CHECK(ts_input_names.find("forward") != ts_input_names.end());
  TORCH_CHECK(ts_input_names.find("foo") != ts_input_names.end());
  TORCH_CHECK(ts_output_names.find("forward") != ts_output_names.end());
  TORCH_CHECK(ts_output_names.find("foo") != ts_output_names.end());

  checkTracedInputs(eager_inputs);
  checkTracedOutputs(eager_outputs);
  checkTracedInputs(jit_inputs);
  checkTracedOutputs(jit_outputs);
  at::clearCallbacks();
}

static int sampled_cb_ctr = 0;
std::unique_ptr<ObserverContext> sampledCallback(const RecordFunction& fn) {
  if (std::string(fn.name()) == "test") {
    ++sampled_cb_ctr;
  }
  return nullptr;
}

static int non_sampled_cb_ctr = 0;
std::unique_ptr<ObserverContext> nonSampledCallback(const RecordFunction& fn) {
  if (std::string(fn.name()) == "test") {
    ++non_sampled_cb_ctr;
  }
  return nullptr;
}

TEST(RecordFunctionTest, SampledCallbacks) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  // test sampled callbacks
  sampled_cb_ctr = 0;
  auto setup_sampled_callback = [](double sampling_prob) {
    return addGlobalCallback(
        RecordFunctionCallback(sampledCallback).samplingProb(sampling_prob));
  };

  addGlobalCallback(RecordFunctionCallback(nonSampledCallback));

  auto handle = setup_sampled_callback(0.5);

  auto run_test_function = []() {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    for (auto k = 0; k < 1000; k++) {
      invokeTestRecordFunction(t);
    }
  };

  run_test_function();
  TORCH_CHECK(non_sampled_cb_ctr == 1000);
  TORCH_CHECK(sampled_cb_ctr > 0 && sampled_cb_ctr < 1000);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  handle = setup_sampled_callback(0.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 2000);
  TORCH_CHECK(sampled_cb_ctr == 0);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  handle = setup_sampled_callback(1.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 3000);
  TORCH_CHECK(sampled_cb_ctr == 1000);
  clearCallbacks();

  // test the scope of the callbacks
  checkScopeCallbacks();
  clearCallbacks();
}

TEST(RecordFunctionTest, RecordFunctionGuard) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static std::vector<std::string> fn_names;
  static std::mutex guard_mtx;

  // check record function guard
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        std::lock_guard<std::mutex> lock(guard_mtx);
        // NOLINTNEXTLINE(modernize-use-emplace)
        fn_names.push_back(fn.name());
        return nullptr;
      }));
  {
    RecordFunctionGuard g1(false);
    {
      RECORD_USER_SCOPE("A");
      {
        RecordFunctionGuard g2(true);
        RECORD_USER_SCOPE("B");
        {
          DisableRecordFunctionGuard g3;
          RECORD_USER_SCOPE("C");
        }
      }
      {
        RECORD_USER_SCOPE("D");
      }
    }
  }
  TORCH_CHECK(fn_names.size() == 1);
  TORCH_CHECK(fn_names[0] == "B");
  clearCallbacks();
}

static std::vector<size_t> ids;

template <size_t id>
auto add_remove_test_add_cb() {
  return addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        ids.push_back(id);
        return nullptr;
      }));
}

TEST(RecordFunctionTest, Callbacks) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  auto h1 = add_remove_test_add_cb<1>();
  add_remove_test_add_cb<2>();
  auto h3 = add_remove_test_add_cb<3>();

  {
    RECORD_USER_SCOPE("test");
  }

  TORCH_CHECK(ids.size() == 3);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h1);

  {
    RECORD_USER_SCOPE("test");
  }

  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h3);

  {
    RECORD_USER_SCOPE("test");
  }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());

  clearCallbacks();

  // thread local / global callbacks

  ids.clear();
  add_remove_test_add_cb<1>();

  {
    RECORD_USER_SCOPE("test");
  }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  auto th = std::thread([]() {
    addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          ids.push_back(2);
          return nullptr;
        }));

    {
      RECORD_USER_SCOPE("test_thread");
    }
  });
  th.join();
  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  ids.clear();

  {
    RECORD_USER_SCOPE("test");
  }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  clearCallbacks();

  // START: thread local / global context check callbacks
  struct TestContext : public ObserverContext {
    int a{0};
    std::string b;
  };
  ids.clear();
  { // START: global test
    addGlobalCallback(RecordFunctionCallback(
        [](const RecordFunction&
           /* unused */) -> std::unique_ptr<at::ObserverContext> {
          auto ctx = std::make_unique<TestContext>();
          ctx->a = 123;
          ctx->b = "test_str";
          ids.push_back(1);
          return ctx;
        },
        [](const RecordFunction& /* unused */, ObserverContext* ctx_ptr) {
          auto ctx = dynamic_cast<TestContext*>(ctx_ptr);
          TORCH_CHECK(ctx != nullptr);
          TORCH_CHECK(ctx->a == 123);
          TORCH_CHECK(ctx->b == "test_str");
        }));

    {
      RECORD_USER_SCOPE("test");
    }

    TORCH_CHECK(ids.size() == 1);
    TORCH_CHECK(ids[0] == 1);
    ids.clear();
  } // END: global test
  { // START: thread local test
    auto ctx_th = std::thread([]() {
      const std::string test_str = "test thread str";
      addThreadLocalCallback(RecordFunctionCallback(
          [](const RecordFunction&
             /* unused */) -> std::unique_ptr<at::ObserverContext> {
            auto ctx = std::make_unique<TestContext>();
            ctx->a = 234;
            ctx->b = "test_thread_str";
            ids.push_back(2);
            return ctx;
          },
          [](const RecordFunction& /* unused */, ObserverContext* ctx_ptr) {
            auto ctx = dynamic_cast<TestContext*>(ctx_ptr);
            TORCH_CHECK(ctx_ptr != nullptr);
            TORCH_CHECK(ctx->a == 234);
            TORCH_CHECK(ctx->b == "test_thread_str");
          }));

      // Will call both global and thread local callbacks.
      {
        RECORD_USER_SCOPE("test_thread");
      }
    });
    ctx_th.join();
    TORCH_CHECK(ids.size() == 2);
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
    ids.clear();
  } // END: thread local test

  clearCallbacks();
}

TEST(RecordFunctionTest, ShouldRun) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static bool ran = false;
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        ran = true;
        return nullptr;
      }));

  {
    RECORD_USER_SCOPE("test");
  }

  EXPECT_TRUE(ran) << "first run didn't happen";
  ran = false;

  disableCallback(handle);

  {
    RECORD_USER_SCOPE("test");
  }

  EXPECT_FALSE(ran) << "second run happened but shouldn't have";
  ran = false;

  reenableCallback(handle);

  {
    RECORD_USER_SCOPE("test");
  }

  EXPECT_TRUE(ran) << "run after re-enable didn't happen";
  ran = false;

  clearCallbacks();
}

TEST(RecordFunctionTest, Basic) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static std::string recorded_op;
  static bool has_ids = false;

  // test propagation of TLS callbacks
  std::thread t([]() {
    RecordFunctionGuard enable_rec_fn;
    auto handle = addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          recorded_op = fn.name();
          return nullptr;
        }));
    ThreadLocalState state;
    std::thread t_child([state]() {
      ThreadLocalStateGuard g_tls(state);
      RECORD_USER_SCOPE("test_in_thread");
    });
    t_child.join();
    EXPECT_EQ(recorded_op, "test_in_thread");
    removeCallback(handle);
  });
  t.join();
  clearCallbacks();

  // test set ids
  addGlobalCallback(
      RecordFunctionCallback(
          [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            has_ids = fn.handle() > 0;
            return nullptr;
          })
          .needsIds(true));
  {
    RECORD_USER_SCOPE("test");
  }
  TORCH_CHECK(has_ids);
  clearCallbacks();
  has_ids = false;
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        has_ids = fn.handle() > 0;
        return nullptr;
      }));
  {
    RECORD_USER_SCOPE("test");
  }
  TORCH_CHECK(!has_ids);
  clearCallbacks();
}

TEST(RecordFunctionTest, OperatorNameOverload) {
  static std::set<std::string> operator_names;
  at::addGlobalCallback(at::RecordFunctionCallback(
                            [](const at::RecordFunction& fn)
                                -> std::unique_ptr<at::ObserverContext> {
                              std::optional<c10::OperatorName> op_name =
                                  fn.operator_name();
                              if (op_name.has_value()) {
                                operator_names.insert(c10::toString(*op_name));
                              } else {
                                operator_names.insert("No Operator Name");
                              }
                              return nullptr;
                            })
                            .scopes({at::RecordScope::FUNCTION}));
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  t.set_requires_grad(false);
  auto t2 = t.pow(2);

  at::clearCallbacks();
  EXPECT_TRUE(operator_names.count("No Operator Name") == 0)
      << "Expected that all traced operators had an associated OperatorName object";
  EXPECT_TRUE(operator_names.count("aten::randn") == 1)
      << "Expected aten::randn to have been called and recorded, but it was not";
  EXPECT_TRUE(operator_names.count("aten::pow.Tensor_Scalar") == 1)
      << "Expected aten::pow.Tensor_Scalar to have been called and recorded, but it was not";
}

class TestThreadLocalDebugInfo : public c10::DebugInfoBase {
 public:
  int getModelId() const {
    return model_id_;
  }

  void setModelId(int model_id) {
    model_id_ = model_id;
  }

  // NOLINTNEXTLINE(modernize-use-equals-default)
  virtual ~TestThreadLocalDebugInfo() override {}

 private:
  int model_id_ = 0;
};

void checkDebugInfo(c10::DebugInfoKind kind, int model_id) {
  auto* debug_info = c10::ThreadLocalDebugInfo::get(kind);
  TORCH_CHECK(debug_info != nullptr);
  auto* test_debug_info = dynamic_cast<TestThreadLocalDebugInfo*>(debug_info);
  TORCH_CHECK(test_debug_info != nullptr);
  TORCH_CHECK(test_debug_info->getModelId() == model_id);
}

TEST(ThreadLocalDebugInfoTest, Basic) {
  static std::atomic<bool> done{false};

  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
  debug_info->setModelId(42);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
  }

  // check that thread local debug info is propagated through fork calls
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    at::launch([]() {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      done = true;
    });
  }
  while (!done) {
  }

  // check that thread local debug info is propagated through backward pass
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  done = false;
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
        checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
        done = true;
        return nullptr;
      }));
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = t.pow(2);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
  }
  removeCallback(handle);
  TORCH_CHECK(done);

  // check nested debug info
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      {
        auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
        debug_info->setModelId(314);
        c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO_2, debug_info);
        {
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
          done = false;
          at::launch([]() {
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
            done = true;
          });
          while (!done) {
          }
        }
      }
    }
  }
}

TEST(TestSymIntArrayRef, BasicConversion) {
  const size_t X = 2, Y = 4, Z = 5;
  std::vector<int64_t> tgt_size_v{2, 4, 5};
  std::vector<c10::SymInt> tgt_size({SymInt(X), SymInt(Y), SymInt(Z)});
  auto a = at::randn({1, 4, 1}, at::kCPU);
  auto b = a.expand_symint(tgt_size);
  auto c = a.expand(tgt_size_v);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, NarrowCopyWithSymbolicInt) {
  static const size_t LENGTH = 5;
  auto a = at::randn({10}, at::kCPU);
  c10::SymInt si(LENGTH);
  auto b = a.narrow_copy_symint(0, 0, si);
  auto c = a.narrow(0, 0, LENGTH);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, NarrowCopy) {
  static const size_t LENGTH = 5;
  auto a = at::randn({10}, at::kCPU);
  auto b = a.narrow_copy(0, 0, LENGTH);
  auto c = a.narrow(0, 0, LENGTH);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, AddSymbolicInt) {
  c10::SymInt a(5);
  c10::SymInt b(3);
  ASSERT_TRUE((a + b).expect_int() == 8);
}

TEST(FallbackGraphsTest, Basic) {
  auto x = at::randn({1}, at::kCPU);
  auto y = at::randn({1}, at::kCPU);
  auto stack = createStack({x.clone(), y.clone()});

  auto graph_string = R"IR(
    graph(%0 : Float(1),
          %1 : Float(1)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  {
    Code code(graph, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
  }
  at::Tensor et;
  pop(stack, et);
  float ef = et.item<float>();
  {
    EnableProfilingGuard epg;
    GraphFunction f("fallbackGraphs", graph, nullptr);
    for (size_t i = 0; i < getNumProfiledRuns() + 1; i++) {
      stack.emplace_back(x.clone());
      stack.emplace_back(y.clone());
      if (i == getNumProfiledRuns()) {
        // we will be modifying a profiled graph
        // before ProfilingGraphExecutor
        // will optimize it in the next iteration
        auto opt_graph = lastExecutedOptimizedGraph();
        // this is safe to do since we are done profiling
        ProfilingRecord::removeProfileCounter(opt_graph->block());
        replaceBlockWithFallbackGraph(opt_graph->block(), opt_graph->inputs());
        auto it = opt_graph->block()->nodes().begin();
        ASSERT_EQ(it->kind(), prim::FallbackGraph);
        auto fallback = *it++;
        ASSERT_EQ(it, opt_graph->block()->nodes().end());
        ASSERT_TRUE(fallback->hasAttribute(attr::Subgraph));
        testing::FileCheck()
            .check("Tensor = aten::mul")
            ->check("Tensor = aten::mul")
            ->run(*fallback->g(attr::Subgraph));
      }
      f.run(stack);
      at::Tensor at;
      pop(stack, at);
      float af = at.item<float>();
      ASSERT_EQ(af, ef);
    }

    auto opt_graph = lastExecutedOptimizedGraph();
    testing::FileCheck()
        .check("(Tensor) = prim::CallFunction")
        ->run(*opt_graph);
  }
}

// TODO this test wasn't running and is broken.
// TEST(AutogradProfilerTest, Basic) {
//   constexpr int batch_size = 4;
//   constexpr int input_size = 256;
//   constexpr int seq_len = 32;

//   int hidden_size = 2 * input_size;
//   auto input = torch::randn({seq_len, batch_size, input_size}, at::kCPU);
//   auto hx = torch::randn({batch_size, hidden_size}, at::kCPU);
//   auto cx = torch::randn({batch_size, hidden_size}, at::kCPU);
//   auto w_ih = t_def(torch::randn({4 * hidden_size, input_size}, at::kCPU));
//   auto w_hh = t_def(torch::randn({4 * hidden_size, hidden_size}, at::kCPU));

//   std::stringstream ss;
//   {
//     RecordProfile guard(ss);
//     for (size_t i = 0; i < 100; ++i) {
//       std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
//     }
//   }

//   std::string result = ss.str();
//   size_t count = 0;
//   for (size_t pos = 0; (pos = result.find("tanh", pos)) != std::string::npos;
//        count++, pos++) {
//   }
//   ASSERT_EQ((count, 200);
// }

TEST(NoneSchemaMatchTest, Basic) {
  RegisterOperators reg({
      Operator(
          "prim::test_none() -> int?",
          [](Stack& stack) { push(stack, IValue()); },
          aliasAnalysisFromSchema()),
      Operator(
          "prim::is_none(int? a) -> bool",
          [](Stack& stack) {
            IValue a = pop(stack);
            if (a.isNone()) {
              push(stack, true);
            } else {
              push(stack, false);
            }
          },
          aliasAnalysisFromSchema()),
  });

  // Constant propagation will run test_none and produce a None,
  // testing that its type is set appropriately and schema matching  doesn't
  // fail when running is_none

  auto r = std::make_shared<Graph>();
  auto& g = *r;
  auto opt_int = g.ins
```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 179 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`

**Classes/Structs**: `TestContext`, `TestThreadLocalDebugInfo`, `Composed`, `WithCPUFuser`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gmock/gmock.h`
- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/Parallel.h`
- `ATen/core/interned_strings.h`
- `ATen/core/ivalue.h`
- `ATen/core/jit_type_base.h`
- `c10/macros/Macros.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/passes/remove_mutation.h`
- `torch/csrc/jit/passes/tensorexpr_fuser.h`
- `torch/csrc/jit/tensorexpr/kernel.h`
- `torch/csrc/autograd/engine.h`
- `torch/csrc/autograd/generated/variable_factories.h`
- `torch/csrc/autograd/profiler.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/codegen/fuser/interface.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/frontend/tracer.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/attributes.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/ir/scope.h`
- `torch/csrc/jit/ir/type_hashing.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/bailout_graph.h`
- `torch/csrc/jit/passes/canonicalize.h`
- `torch/csrc/jit/passes/common_subexpression_elimination.h`


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
python test/cpp/jit/test_misc.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_lite_interpreter_direct.cpp_docs.md`](./test_lite_interpreter_direct.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_misc.cpp_docs.md`
- **Keyword Index**: `test_misc.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
