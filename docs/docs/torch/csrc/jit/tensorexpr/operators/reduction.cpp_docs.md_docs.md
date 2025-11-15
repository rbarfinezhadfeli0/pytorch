# Documentation: `docs/torch/csrc/jit/tensorexpr/operators/reduction.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/operators/reduction.cpp_docs.md`
- **Size**: 8,487 bytes (8.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/operators/reduction.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/operators/reduction.cpp`
- **Size**: 6,126 bytes (5.98 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/operators/reduction.h>

using namespace torch::jit::tensorexpr;

// Remove all indices from axes positions.
static std::vector<VarHandle> squeezeIndices(
    const ParameterList& indices,
    const std::vector<size_t>& axes) {
  std::vector<VarHandle> indices_squeezed;
  for (size_t dim = 0; dim < indices.size(); ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      indices_squeezed.push_back(indices[dim]);
    }
  }
  return indices_squeezed;
}

namespace torch::jit::tensorexpr {

Tensor computeSum(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  std::vector<size_t> axes;
  bool keepdim = false;
  // aten::sum takes the input tensor named self.
  auto sizes = valueShape(inputs[0]);

  size_t rank = sizes.size();
  if (inputs.size() > 2) {
    if (auto emptyAxes = std::get_if<BufList>(&inputs[1])) {
      // If dim-array is an empty list, it will appear as BufList instead of
      // IntList, and hence we need a special handling for it.
      // In that case, we need to sum over all axes.
      TORCH_INTERNAL_ASSERT(emptyAxes->empty());
      axes.resize(rank);
      std::iota(axes.begin(), axes.end(), 0);
    } else if (rank > 0) {
      auto const& nodeAxes = std::get<IntList>(inputs[1]);
      // Canonicalize axes: wrap around, sort and make unique.
      for (auto axis : nodeAxes) {
        axes.push_back(at::maybe_wrap_dim(axis, static_cast<int64_t>(rank)));
      }
      std::sort(axes.begin(), axes.end());
      axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    }
    keepdim = std::get<bool>(inputs[2]);
  } else {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }
  // Axes go into reduction dimensions.
  std::vector<ExprHandle> reductionDims;
  reductionDims.reserve(rank);
  for (size_t axis : axes) {
    reductionDims.emplace_back(sizes[axis]);
  }
  std::vector<ExprHandle> outputDims;
  // Output dimensions are the complement of axes. When keepdim is set, a
  // one-sized dimension is inserted for each axis.
  for (size_t dim = 0; dim < rank; ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      outputDims.emplace_back(sizes[dim]);
    } else if (keepdim) {
      outputDims.emplace_back(1);
    }
  }

  return Reduce(
      "sum",
      outputDims,
      outputStrides,
      Sum(),
      [&](ParameterList& indices) {
        // "Squeeze" out indices inserted when keepdim is set.
        auto indices_squeezed =
            keepdim ? squeezeIndices(indices, axes) : indices;
        TORCH_INTERNAL_ASSERT(axes.size() <= indices_squeezed.size());
        // Move innermost indices into axes positions:
        //   1. Fill the outermost indices first.
        //   2. Insert the innermost indices into the correct axis position,
        //   displacing the outermost indices as needed.
        std::vector<ExprHandle> indices_exprs;
        size_t i = 0;
        for (; i < indices_squeezed.size() - axes.size(); ++i) {
          indices_exprs.push_back(indices_squeezed[i]);
        }
        for (auto axis : axes) {
          indices_exprs.insert(
              indices_exprs.begin() + static_cast<std::ptrdiff_t>(axis),
              indices_squeezed[i]);
          ++i;
        }
        auto indexed = tensorOrConstant(inputs[0], indices_exprs);
        if (outputType) {
          return Cast::make(ToDtype(*outputType), indexed);
        } else {
          return indexed;
        }
      },
      reductionDims);
}

Tensor computeMean(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  bool keepdim = false;
  BufHandle ResultBuf("mean", outputShape, dtype);
  auto const& InputBuf = std::get<BufHandle>(inputs[0]);
  std::vector<ExprHandle> extra_args;
  if (inputs.size() > 2) {
    keepdim = std::get<bool>(inputs[2]);
  }

  if (auto mean_dims = std::get_if<IntList>(&inputs[1])) {
    extra_args = c10::fmap<ExprHandle>(*mean_dims);
  } else {
    // When dims argument is not specified, reduce over all dimensions
    for (int64_t idx = 0; idx < static_cast<int64_t>(InputBuf.ndim()); ++idx) {
      extra_args.emplace_back(idx);
    }
  }
  extra_args.push_back(LongImm::make(static_cast<int64_t>(keepdim)));
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_mean", {InputBuf}, extra_args));
}

Tensor computeMax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("max", outputShape, dtype);
  auto const& InputBuf = std::get<BufHandle>(inputs[0]);
  auto max_dim = std::get<int64_t>(inputs[1]);
  auto keep_dim = std::get<bool>(inputs[2]);
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_max_red",
          {InputBuf},
          {max_dim, (int64_t)keep_dim}));
}

Tensor computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("adaptive_avgpool2d", outputShape, dtype);
  auto const& out_size_param = std::get<IntList>(inputs[1]);
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_adaptive_avg_pool2d",
          {std::get<BufHandle>(inputs[0])},
          c10::fmap<ExprHandle>(out_size_param)));
}

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

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

- `torch/csrc/jit/tensorexpr/operators/reduction.h`


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
- [`pointwise.cpp_docs.md`](./pointwise.cpp_docs.md)
- [`matmul.h_docs.md`](./matmul.h_docs.md)


## Cross-References

- **File Documentation**: `reduction.cpp_docs.md`
- **Keyword Index**: `reduction.cpp_kw.md`
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

- **File Documentation**: `reduction.cpp_docs.md_docs.md`
- **Keyword Index**: `reduction.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
