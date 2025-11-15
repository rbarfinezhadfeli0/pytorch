# Documentation: `torch/csrc/jit/tensorexpr/lowerings.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/lowerings.cpp`
- **Size**: 80,200 bytes (78.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

#include <ATen/native/Activation.h>
#include <ATen/native/mkldnn/Common.h>

namespace torch::jit::tensorexpr {

FunctionSchemaMap<NNCLoweringFunction>& getNNCLoweringRegistry() {
  static FunctionSchemaMap<NNCLoweringFunction> lowering_registry_;
  return lowering_registry_;
}

RegisterNNCLoweringsFunction::RegisterNNCLoweringsFunction(
    const std::vector<std::string>& schemas,
    const NNCLoweringFunction& fn) {
  for (const auto& schema_str : schemas) {
    getNNCLoweringRegistry().insert(parseSchema(schema_str), fn);
  }
}

namespace {
int nnc_lowerings_lazy_registration() {
  RegisterNNCLoweringsFunction aten_dropout(
      {"aten::dropout(Tensor input, float p, bool train) -> (Tensor)"},
      computeNoop);
  RegisterNNCLoweringsFunction aten_contiguous(
      {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> (Tensor(a))"},
      computeNoop);

#ifdef USE_XNNPACK
  // TODO: add a test
  RegisterNNCLoweringsFunction prepacked_conv2d_clamp_run(
      {"prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> (Tensor Y)"},
      computePrepackedConv2dClampRun);

  // TODO: add a test
  RegisterNNCLoweringsFunction prepacked_linear_clamp_run(
      {"prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> (Tensor Y)"},
      computePrepackedLinearClampRun);
#endif

#if AT_MKLDNN_ENABLED()
  RegisterNNCLoweringsFunction mkldnn_prepacked_conv2d_run(
      {"mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> (Tensor Y)"},
      computeMkldnnPrepackedConvRun);
#endif // AT_MKLDNN_ENABLED()

  RegisterNNCLoweringsFunction aten_sub(
      {"aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
       "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
          // NB: sub isn't supported on boolean, no need to promote to integer.
          return lhs - rhs;
        };
        TORCH_INTERNAL_ASSERT(
            inputs.size() == 2 || inputs.size() == 3,
            buildErrorMessage("Invalid number of input operands"));
        return (inputs.size() > 2) ? computeTwoOperandWithAlpha(
                                         "aten_sub",
                                         inputs,
                                         outputShape,
                                         outputStrides,
                                         outputType,
                                         sub_lambda)
                                   : computeTwoOperand(
                                         "aten_sub",
                                         inputs,
                                         outputShape,
                                         outputStrides,
                                         outputType,
                                         sub_lambda);
      });

  RegisterNNCLoweringsFunction aten_mul(
      {"aten::mul.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::mul.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_mul",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return boolToInteger(lhs) * boolToInteger(rhs);
            });
      });

#define DEFINE_BINARY_SCALAR_OP_LOWERING(op_name, op)                     \
  RegisterNNCLoweringsFunction aten_##op_name##_scalar(                   \
      {"aten::" #op_name ".int(int a, int b) -> (int)",                   \
       "aten::" #op_name ".int_float(int a, float b) -> (float)",         \
       "aten::" #op_name ".float_int(float a, int b) -> (float)",         \
       "aten::" #op_name ".float(float a, float b) -> (float)"},          \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });
  DEFINE_BINARY_SCALAR_OP_LOWERING(mul, a * b)
  DEFINE_BINARY_SCALAR_OP_LOWERING(add, a + b)
  DEFINE_BINARY_SCALAR_OP_LOWERING(sub, a - b)
#undef DEFINE_BINARY_SCALAR_OP_LOWERING
  RegisterNNCLoweringsFunction aten_div_scalar(
      {"aten::div(Scalar a, Scalar b) -> (float)",
       "aten::div.int(int a, int b) -> (float)",
       "aten::div.int_float(int a, float b) -> (float)",
       "aten::div.float_int(float a, int b) -> (float)",
       "aten::div.float(float a, float b) -> (float)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeScalar(
            "aten_div",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a, const ExprHandle& b) {
              return promoteIntegerToDefaultType(a) /
                  promoteIntegerToDefaultType(b);
            });
      });

#define DEFINE_COMPARISON_SCALAR_OP_LOWERING(op_name, op)                 \
  RegisterNNCLoweringsFunction aten_##op_name##_scalar(                   \
      {"aten::" #op_name ".bool(bool a, bool b) -> (bool)",               \
       "aten::" #op_name ".int(int a, int b) -> (bool)",                  \
       "aten::" #op_name ".int_float(int a, float b) -> (bool)",          \
       "aten::" #op_name ".float_int(float a, int b) -> (bool)",          \
       "aten::" #op_name ".float(float a, float b) -> (bool)"},           \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(lt, cast<bool>(a < b))
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(le, cast<bool>(a <= b))
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(eq, cast<bool>(a == b))
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(ne, cast<bool>(a != b))
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(gt, cast<bool>(a > b))
  DEFINE_COMPARISON_SCALAR_OP_LOWERING(ge, cast<bool>(a >= b))
#undef DEFINE_COMPARISON_SCALAR_OP_LOWERING

#define DEFINE_BITWISE_SCALAR_OP_LOWERING(op_name, op)                    \
  RegisterNNCLoweringsFunction aten_##op_name##_int_scalar(               \
      {"aten::" #op_name ".int(int a, int b) -> (int)"},                  \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });
  DEFINE_BITWISE_SCALAR_OP_LOWERING(
      __and__, boolToInteger(a) & boolToInteger(b))
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__or__, boolToInteger(a) | boolToInteger(b))
  DEFINE_BITWISE_SCALAR_OP_LOWERING(
      __xor__, boolToInteger(a) ^ boolToInteger(b))
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__lshift__, a << b)
  DEFINE_BITWISE_SCALAR_OP_LOWERING(__rshift__, a >> b)
#undef DEFINE_BITWISE_SCALAR_OP_LOWERING

#define DEFINE_LOGICAL_SCALAR_OP_LOWERING(op_name, op)                    \
  RegisterNNCLoweringsFunction aten_##op_name##_bool_scalar(              \
      {"aten::" #op_name ".bool(bool a, bool b) -> (bool)"},              \
      [](const std::vector<ArgValue>& inputs,                             \
         const std::vector<ExprHandle>& outputShape,                      \
         const std::vector<ExprHandle>& outputStrides,                    \
         const std::optional<ScalarType>& outputType,                     \
         at::Device device) {                                             \
        return computeScalar(                                             \
            "aten_#op_name",                                              \
            inputs,                                                       \
            outputShape,                                                  \
            outputStrides,                                                \
            outputType,                                                   \
            [](const ExprHandle& a, const ExprHandle& b) { return op; }); \
      });
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__and__, a && b)
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__or__, a || b)
  DEFINE_LOGICAL_SCALAR_OP_LOWERING(__xor__, a != b)
#undef DEFINE_LOGICAL_SCALAR_OP_LOWERING

  RegisterNNCLoweringsFunction aten_div(
      {"aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::div.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_div",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return promoteIntegerToDefaultType(lhs) /
                  promoteIntegerToDefaultType(rhs);
            });
      });

  RegisterNNCLoweringsFunction aten___and__(
      {"aten::__and__.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::__and__.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_and",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return boolToInteger(lhs) & boolToInteger(rhs);
            });
      });

  RegisterNNCLoweringsFunction aten___or__(
      {"aten::__or__.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::__or__.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_or",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return boolToInteger(lhs) | boolToInteger(rhs);
            });
      });

  RegisterNNCLoweringsFunction aten___xor__(
      {"aten::__xor__.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::__xor__.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_xor",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return boolToInteger(lhs) ^ boolToInteger(rhs);
            });
      });

  RegisterNNCLoweringsFunction aten___lshift__(
      {"aten::__lshift__.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::__lshift__.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_lshift",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return lhs << rhs;
            });
      });

  RegisterNNCLoweringsFunction aten___rshift__(
      {"aten::__rshift__.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::__rshift__.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_rshift",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return lhs >> rhs;
            });
      });

  RegisterNNCLoweringsFunction aten_eq(
      {"aten::eq.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::eq.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_eq",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs == rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_ne(
      {"aten::ne.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::ne.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_ne",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs != rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_ge(
      {"aten::ge.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::ge.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_ge",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs >= rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_gt(
      {"aten::gt.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::gt.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_gt",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs > rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_le(
      {"aten::le.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::le.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_le",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs <= rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_lt(
      {"aten::lt.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::lt.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_lt",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return cast<bool>(lhs < rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_min_pointwise(
      {"aten::min.other(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_min",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return Min::make(boolToInteger(lhs), boolToInteger(rhs), false);
            });
      });

  RegisterNNCLoweringsFunction aten_max_pointwise(
      {"aten::max.other(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_max",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return Max::make(boolToInteger(lhs), boolToInteger(rhs), false);
            });
      });

  RegisterNNCLoweringsFunction aten_masked_fill(
      {"aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> (Tensor)",
       "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeThreeOperand(
            "aten_masked_fill",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& input,
               const ExprHandle& mask,
               const ExprHandle& value) {
              // value needs to promote to input, not vice versa
              auto val = promoteToDtype(value, input.dtype().scalar_type());
              return ifThenElse(mask, val, input);
            },
            /*promote_inputs*/ false);
      });
  RegisterNNCLoweringsFunction aten_clamp(
      {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> (Tensor)",
       "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        bool noMin = false;
        bool noMax = false;
        if (std::get_if<ArgNone>(&inputs[1])) {
          noMin = true;
        }

        if (std::get_if<ArgNone>(&inputs[2])) {
          noMax = true;
        }

        return computeThreeOperand(
            "aten_clamp",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [noMin, noMax](
                const ExprHandle& in,
                const ExprHandle& min,
                const ExprHandle& max) {
              auto cast = [&](const ExprHandle& e) {
                return Cast::make(in.dtype(), e);
              };

              if (noMin && noMax) {
                return in;
              } else if (noMin) {
                auto cmax = cast(max);
                return CompareSelect::make(in, cmax, cmax, in, kGT);
              } else if (noMax) {
                auto cmin = cast(min);
                return CompareSelect::make(in, cmin, cmin, in, kLT);
              } else {
                auto cmax = cast(max);
                auto cmin = cast(min);
                return clamp(cmin, cmax, in);
              }
            },
            false /* promote_inputs */);
      });

  RegisterNNCLoweringsFunction aten_addcmul(
      {"aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeFourOperand(
            "aten_addcmul",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a0,
               const ExprHandle& a1,
               const ExprHandle& a2,
               const ExprHandle& a3) { return a0 + a3 * a1 * a2; });
      });

  RegisterNNCLoweringsFunction aten_sigmoid(
      {"aten::sigmoid(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        // check if the activation is quantized
        const BufHandle& x = std::get<BufHandle>(inputs[0]);
        if (x.node()->qscale()) {
          return computeQuantizedSigmoidExternalCall(
              inputs, outputShape, outputStrides, outputType, device);
        }
        return computeOneOperand(
            "aten_sigmoid",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return sigmoid(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_silu(
      {"aten::silu(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_silu",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) { return a * sigmoid(a); });
      });

  RegisterNNCLoweringsFunction aten_reciprocal(
      {"aten::reciprocal(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_reciprocal",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) { return ExprHandle(1.0f) / a; });
      });

  RegisterNNCLoweringsFunction aten_neg(
      {"aten::neg(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_neg",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) { return ExprHandle(-0) - a; });
      });

  RegisterNNCLoweringsFunction aten_isnan(
      {"aten::isnan(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_isnan",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              if (!a.dtype().is_floating_point()) {
                return IntImm::make(0);
              }
              return isnan(a);
            });
      });

  RegisterNNCLoweringsFunction aten_relu(
      {"aten::relu(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        auto A = std::get<BufHandle>(inputs[0]);
        if (A.node()->qscale()) {
          return computeQuantizedRelu(
              inputs, outputShape, outputStrides, outputType, device);
        }
        return computeOneOperand(
            "aten_relu",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              auto zero = Cast::make(a.dtype(), 0);
              return CompareSelect::make(a, zero, zero, a, kLT);
            });
      });

  RegisterNNCLoweringsFunction aten_leaky_relu(
      {"aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_leaky_relu",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a, const ExprHandle& negative_slope) {
              auto neg_slope = Cast::make(a.dtype(), negative_slope);
              auto zero = Cast::make(a.dtype(), 0);
              auto one = Cast::make(a.dtype(), 1);
              auto cs = CompareSelect::make(a, zero, one, neg_slope, kGT);
              return a * cs;
            });
      });

  RegisterNNCLoweringsFunction aten_relu6(
      {"aten::relu6(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_relu6",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              auto zero = Cast::make(a.dtype(), 0);
              auto six = Cast::make(a.dtype(), 6.);
              return clamp(zero, six, a);
            });
      });

  RegisterNNCLoweringsFunction aten_gelu(
      {"aten::gelu(Tensor self, *, str approximate='none') -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        const auto& kApproximate = std::get<std::string>(inputs[1]);
        std::vector<ArgValue> operands = {inputs.front()};
        if (at::native::get_gelutype_enum(kApproximate) ==
            at::native::GeluType::Tanh) {
          // approximate == 'tanh'
          return computeOneOperand(
              "aten_tanh_gelu",
              operands,
              outputShape,
              outputStrides,
              outputType,
              [](const ExprHandle& a) {
                auto one = Cast::make(a.dtype(), 1.);
                auto point_five = Cast::make(a.dtype(), .5);
                auto beta = Cast::make(a.dtype(), M_SQRT2 * M_2_SQRTPI * 0.5);
                auto kappa = Cast::make(a.dtype(), 0.044715);
                auto a_cube = a * a * a;
                auto inner = beta * (a + kappa * a_cube);
                return point_five * a * (one + tanh(inner));
              });
        } else {
          // approximate == 'none'
          return computeOneOperand(
              "aten_gelu",
              operands,
              outputShape,
              outputStrides,
              outputType,
              [](const ExprHandle& a) {
                auto m_sqrt1_2 = Cast::make(a.dtype(), M_SQRT1_2);
                auto one = Cast::make(a.dtype(), 1.);
                auto point_five = Cast::make(a.dtype(), .5);
                return a * point_five * (one + erf(a * m_sqrt1_2));
              });
        }
      });

  RegisterNNCLoweringsFunction aten_batch_norm(
      {"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor)"},
      computeBatchNorm);

  RegisterNNCLoweringsFunction aten_log(
      {"aten::log(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_log",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return log(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_log10(
      {"aten::log10(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_log10",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return log10(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_log1p(
      {"aten::log1p(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_log1p",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return log1p(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_log2(
      {"aten::log2(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_log2",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return log2(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_exp(
      {"aten::exp(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_exp",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return exp(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_expm1(
      {"aten::expm1(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_expm1",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return expm1(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_erf(
      {"aten::erf(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_erf",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return erf(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_erfc(
      {"aten::erfc(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_erfc",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return erfc(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_cos(
      {"aten::cos(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_cos",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return cos(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_sin(
      {"aten::sin(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_sin",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return sin(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_tan(
      {"aten::tan(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_tan",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return tan(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_type_as(
      {"aten::type_as(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        const BufHandle& rhs = std::get<BufHandle>(inputs[1]);
        auto dtype = rhs.dtype();
        return computeOneOperand(
            "aten_type_as",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [dtype](const ExprHandle& lhs) { return Cast::make(dtype, lhs); });
      });

  RegisterNNCLoweringsFunction aten_pow(
      {"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)",
       "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)",
       "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_pow",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              if (!rhs.node()->isConstant()) {
                return pow(lhs, rhs);
              }
              double val =
                  immediateAs<double>(IRSimplifier::simplify(rhs.node()));

              if (val == 1.0f) {
                return lhs;
              } else if (val == 2.0f) { // NOLINT
                return lhs * lhs;
              } else if (val == 3.0f) { // NOLINT
                return (lhs * lhs) * lhs;
              } else if (val == 4.0f) { // NOLINT
                ExprHandle tmp = lhs * lhs;
                return tmp * tmp;
              } else if (val == 0.5f) { // NOLINT
                return sqrt(lhs);
              } else if (val == 0.0f) {
                return ExprHandle(1.0f);
              } else if (val == -0.5f) { // NOLINT
                return rsqrt(lhs);
              } else if (val == -1.0f) {
                return ExprHandle(1.0f) / lhs;
              } else if (val == -2.0f) { // NOLINT
                return ExprHandle(1.0f) / (lhs * lhs);
              }
              return pow(lhs, rhs);
            });
      });

  RegisterNNCLoweringsFunction aten_fmod(
      {"aten::fmod.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::fmod.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_fmod",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return fmod(promoteHalfToFloat(lhs), promoteHalfToFloat(rhs));
            });
      });

  RegisterNNCLoweringsFunction aten_lerp(
      {"aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> (Tensor)",
       "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeThreeOperand(
            "aten_lerp",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a,
               const ExprHandle& end,
               const ExprHandle& weight) { return a + weight * (end - a); });
      });

  RegisterNNCLoweringsFunction aten_remainder(
      {"aten::remainder.Scalar(Tensor self, Scalar other) -> (Tensor)",
       "aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)",
       "aten::remainder.Tensor(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        auto imodImpl = [](const ExprHandle& lhs, const ExprHandle& rhs) {
          return Mod::make(lhs, rhs);
        };
        auto fmodImpl = [](const ExprHandle& lhs, const ExprHandle& rhs) {
          auto lhs_t = promoteHalfToFloat(lhs);
          auto rhs_t = promoteHalfToFloat(rhs);
          return fmod((rhs_t + fmod(lhs_t, rhs_t)), rhs_t);
        };
        {
          auto const& shape =
              broadcastShapes(valueShape(inputs[0]), valueShape(inputs[1]));
          return Compute(
              "aten_remainder", shape, [&](const std::vector<VarHandle>& axes) {
                std::vector<ExprHandle> indices(axes.begin(), axes.end());
                std::vector<ExprHandle> exprInputs = {
                    tensorOrConstant(inputs[0], indices),
                    tensorOrConstant(inputs[1], indices),
                };

                promoteInputs(exprInputs);
                bool allInt = true;
                for (auto& e : exprInputs) {
                  if (e.dtype().is_floating_point()) {
                    allInt = false;
                    break;
                  }
                }
                if (allInt) {
                  return demoteOutput(
                      imodImpl(exprInputs[0], exprInputs[1]), outputType);
                } else {
                  return demoteOutput(
                      fmodImpl(exprInputs[0], exprInputs[1]), outputType);
                }
              });
        }
      });

  RegisterNNCLoweringsFunction prim_ConstantChunk(
      {"prim::ConstantChunk(...) -> (...)"}, computeChunk);

  RegisterNNCLoweringsFunction aten_acos(
      {"aten::acos(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_acos",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return acos(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_asin(
      {"aten::asin(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_asin",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return asin(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_cosh(
      {"aten::cosh(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_cosh",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return cosh(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_sinh(
      {"aten::sinh(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_sinh",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return sinh(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_atan(
      {"aten::atan(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_atan",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return atan(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_atan2(
      {"aten::atan2(Tensor self, Tensor other) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeTwoOperand(
            "aten_atan2",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& lhs, const ExprHandle& rhs) {
              return atan2(
                  promoteIntegerToDefaultType(lhs),
                  promoteIntegerToDefaultType(rhs));
            });
      });

  RegisterNNCLoweringsFunction aten_tanh(
      {"aten::tanh(Tensor self) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
         const std::optional<ScalarType>& outputType,
         at::Device device) {
        return computeOneOperand(
            "aten_tanh",
            inputs,
            outputShape,
            outputStrides,
            outputType,
            [](const ExprHandle& a) {
              return tanh(promoteIntegerToDefaultType(a));
            });
      });

  RegisterNNCLoweringsFunction aten_hardtanh(
      {"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)"},
      [](const std::vector<ArgValue>& inputs,
         const std::vector<ExprHandle>& outputShape,
         const std::vector<ExprHandle>& outputStrides,
        
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 274 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `NNCLoweringFunction`

**Classes/Structs**: `an`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/function_schema_parser.h`
- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `torch/csrc/jit/tensorexpr/lowerings.h`
- `torch/csrc/jit/tensorexpr/operators/operators.h`
- `ATen/native/Activation.h`
- `ATen/native/mkldnn/Common.h`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `lowerings.cpp_docs.md`
- **Keyword Index**: `lowerings.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
