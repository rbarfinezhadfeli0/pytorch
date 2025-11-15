# Documentation: pointwise.cpp

## File Metadata
- **Path**: `torch/csrc/jit/tensorexpr/operators/pointwise.cpp`
- **Size**: 8052 bytes
- **Lines**: 224
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/pointwise.h>

namespace torch::jit::tensorexpr {

using namespace torch::jit::tensorexpr;

Tensor computeSign(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::optional<std::vector<ExprHandle>>& outputStrides) {
  return Compute(
      "aten_sign", outputShape, outputStrides, [&](ParameterList& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};
        auto inp = inputs[0];
        auto zero = ExprHandle(immLike(inp, 0.0f));
        auto res = (zero < inp) - (inp < zero);
        return promoteToDtype(res, inp.dtype().scalar_type());
      });
}

Tensor computeOneOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr,
    const int checkParamTypes) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr, checkParamTypes](
          const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};
        promoteInputs(inputs, checkParamTypes);
        ExprHandle compute = innerExpr(inputs[0]);
        return demoteOutput(compute, outputType);
      });
}

Tensor computeTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[1]);
        return demoteOutput(compute, outputType);
      });
}

Tensor computeTwoOperandWithAlpha(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[2] * inputs[1]);
        return demoteOutput(compute, outputType);
      });
}

Tensor computeConditionWithTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        promoteInputs(inputs);
        // First expr is the condition, which we don't promote
        inputs.emplace(
            inputs.begin(), tensorOrConstant(inputValues[0], indices));
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, outputType);
      });
}

Tensor computeThreeOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr,
    bool promote_inputs) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr, promote_inputs](
          const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        if (promote_inputs) {
          promoteInputs(inputs);
        }
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, outputType);
      });
}
Tensor computeFourOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr) {
  return Compute(
      name,
      outputShape,
      outputStrides,
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
            tensorOrConstant(inputValues[3], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute =
            innerExpr(inputs[0], inputs[1], inputs[2], inputs[3]);
        return demoteOutput(compute, outputType);
      });
}

Tensor computeNoop(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  return computeOneOperand(
      "copy",
      inputValues,
      outputShape,
      outputStrides,
      outputType,
      [](const ExprHandle& a) { return a; });
}

Tensor computeScalar(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  auto dt = Dtype(*outputType);
  VarPtr let_var = alloc<Var>(name + "_var", dt);
  std::vector<ExprHandle> inputs = {
      scalarOrConstant(inputValues[0]), scalarOrConstant(inputValues[1])};
  promoteInputs(inputs);
  ExprHandle compute = innerExpr(inputs[0], inputs[1]);
  StmtPtr let_stmt =
      Let::make(VarHandle(let_var), demoteOutput(compute, outputType));
  std::vector<ExprPtr> dims;
  BufPtr buf = alloc<Buf>(let_var, dims, dt);
  return Tensor(buf, let_stmt);
}

} // namespace torch::jit::tensorexpr

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 559 words across 224 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 8052 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
