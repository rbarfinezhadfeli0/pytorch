# Documentation: `docs/torch/csrc/jit/tensorexpr/operators/pointwise.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/operators/pointwise.cpp_docs.md`
- **Size**: 10,407 bytes (10.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/operators/pointwise.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/operators/pointwise.cpp`
- **Size**: 8,052 bytes (7.86 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

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


This C++ file contains approximately 0 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/operators/misc.h`
- `torch/csrc/jit/tensorexpr/operators/pointwise.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/tensorexpr/operators`):

- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`conv2d.h_docs.md`](./conv2d.h_docs.md)
- [`misc.cpp_docs.md`](./misc.cpp_docs.md)
- [`softmax.cpp_docs.md`](./softmax.cpp_docs.md)
- [`misc.h_docs.md`](./misc.h_docs.md)
- [`quantization.cpp_docs.md`](./quantization.cpp_docs.md)
- [`conv2d.cpp_docs.md`](./conv2d.cpp_docs.md)
- [`softmax.h_docs.md`](./softmax.h_docs.md)
- [`matmul.h_docs.md`](./matmul.h_docs.md)


## Cross-References

- **File Documentation**: `pointwise.cpp_docs.md`
- **Keyword Index**: `pointwise.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr/operators`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/csrc/jit/tensorexpr/operators`):

- [`matmul.h_docs.md_docs.md`](./matmul.h_docs.md_docs.md)
- [`matmul.h_kw.md_docs.md`](./matmul.h_kw.md_docs.md)
- [`misc.cpp_docs.md_docs.md`](./misc.cpp_docs.md_docs.md)
- [`quantization.h_docs.md_docs.md`](./quantization.h_docs.md_docs.md)
- [`quantization.cpp_kw.md_docs.md`](./quantization.cpp_kw.md_docs.md)
- [`quantization.cpp_docs.md_docs.md`](./quantization.cpp_docs.md_docs.md)
- [`pointwise.h_kw.md_docs.md`](./pointwise.h_kw.md_docs.md)
- [`norm.cpp_kw.md_docs.md`](./norm.cpp_kw.md_docs.md)
- [`reduction.h_kw.md_docs.md`](./reduction.h_kw.md_docs.md)
- [`operators.h_docs.md_docs.md`](./operators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pointwise.cpp_docs.md_docs.md`
- **Keyword Index**: `pointwise.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
